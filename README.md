# ALICE-TRT

**GPU Ternary Inference Engine — Trojan Horse Edition**

> "Don't move data. Move the law."
> "The GPU thinks it's computing FP16. It's computing {-1, 0, +1}."

**Current release: v1.3.1** — semver-stable PGS solver bridge (gravity + floor + multi-distance constraints, byte-exact on Metal / Vulkan / DX12 WARP). See [Cross-Crate Bridges](#cross-crate-bridges).

ALICE-TRT is a GPU inference engine that exploits **memory bandwidth compression** for 1.58-bit ternary neural networks. Weights are stored as 2-bit bitplanes in VRAM, transported at 1/8 the bandwidth of FP16, and expanded on-the-fly in registers before hitting the compute units.

Part of the [Project A.L.I.C.E.](https://github.com/ext-sakamoro) ecosystem.

## The Trojan Horse Strategy

```
Traditional GPU Inference:
  VRAM ──[FP16 weights]──▶ L2 ──▶ L1 ──▶ Registers ──▶ FMA
  Bandwidth: 16 bits/weight (bottleneck!)

ALICE-TRT:
  VRAM ──[2-bit packed]──▶ L2 ──▶ L1 ──▶ Registers ──[expand]──▶ FMA
  Bandwidth: 2 bits/weight (8x reduction!)
  The expand step costs ~2 cycles. The bandwidth savings buy ~100 cycles.
```

On CPU (ALICE-ML), the trick was "eliminate multiplication, use addition only."
On GPU, multiplication is free (Tensor Cores do it at 125+ TFLOPS).
**The bottleneck is memory bandwidth.** So we compress the data instead.

The Tensor Core receives valid FP16 fragments `{-1.0, 0.0, +1.0}`.
It has no idea the original weights were 2-bit. This is the Trojan Horse.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  ALICE-TRT Trojan Horse Edition                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Backend 1: wgpu (Cross-Platform)                                │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────┐               │
│  │ GpuDevice │──│ TernaryCompute   │──│ GpuTensor│               │
│  │ (wgpu)   │  │ (4 WGSL shaders) │  │ (output) │               │
│  └──────────┘  └──────────────────┘  └──────────┘               │
│       Metal (macOS) / Vulkan (Linux) / DX12 (Windows)            │
│                                                                  │
│  Backend 2: CUDA (NVIDIA, --features cuda)                       │
│  ┌──────────────────┐  ┌─────────────────────────────────┐      │
│  │ CudaTernaryEngine │──│ bitnet_kernel.cu                │      │
│  │ (Rust safe API)   │  │ Kernel 1: wmma  (Tensor Core)  │      │
│  └──────────────────┘  │ Kernel 2: dp4a  (INT8 dot)     │      │
│                         │ Kernel 3: popc  (bit-parallel) │      │
│                         └─────────────────────────────────┘      │
│                                                                  │
│  Backend 3: TensorRT Plugin (optional)                           │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ AliceTernaryPlugin : IPluginV2DynamicExt             │        │
│  │ Replaces GEMM layers in existing TensorRT networks.  │        │
│  │ Serializes 2-bit weights, dispatches to CUDA kernels.│        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  Weight Storage (both backends):                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ GpuTernaryWeight / DeviceWeights                      │       │
│  │ plus_bits: u32[]  │  minus_bits: u32[]  │  scale: f32 │       │
│  │ 32 weights per u32 word. 8x smaller than FP16.        │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

## CUDA Kernel Dispatch Hierarchy

Auto-selected at runtime based on GPU compute capability:

| Kernel | Compute Cap | Strategy | Throughput |
|--------|-------------|----------|------------|
| **wmma** | >= 7.0 (Volta, Turing, Ampere, Hopper) | 2-bit → FP16 → `wmma::mma_sync` (Tensor Core) | 125+ TFLOPS |
| **dp4a** | >= 6.1 (Pascal GP106+) | 2-bit → INT8 → `__dp4a` (4-element dot product in 1 cycle) | 4x scalar |
| **popc** | any (Maxwell, Kepler, ...) | `__popc` + `__ffs` bit-parallel popcount, zero-weight skip | 32 weights/cycle |

### popc kernel optimizations

- `__popc(pw | mw)` counts non-zero weights per u32 word
- **Sparse path** (<=16/32 non-zero): `__ffs` iterates only set bits, skips zeros entirely
- **Dense path** (>16/32 non-zero): 8-wide unrolled branchless accumulation
- **Zero word skip**: `active == 0` → `continue` (typical ternary models have ~30-50% zero words)

## Benchmarks

Measured on Apple M2 Max (wgpu/Metal backend).
CPU baseline: ALICE-ML (branchless ternary matvec with SIMD).

### GPU matvec: Throughput by matrix size

| Size (N x N) | GPU (wgpu) | Notes |
|--------------|-----------|-------|
| 64 x 64 | 1.37 ms | Dispatch overhead dominates |
| 256 x 256 | 1.28 ms | Dispatch overhead dominates |
| 512 x 512 | 1.48 ms | |
| 1024 x 1024 | 1.57 ms | Tiled kernel activates (shared memory + parallel reduction) |
| 2048 x 2048 | 1.79 ms | |
| 4096 x 4096 | 2.61 ms | Compute-bound regime |

### CPU vs GPU: Crossover point

| Size | CPU (ALICE-ML) | GPU (wgpu) | Winner |
|------|---------------|-----------|--------|
| 256 | 60 us | 1.28 ms | CPU 21x faster |
| 1024 | 912 us | 1.34 ms | CPU 1.5x faster |
| **4096** | **21.3 ms** | **2.52 ms** | **GPU 8.4x faster** |

> **Crossover at ~2048.** Below that, CPU wins due to wgpu dispatch overhead (~1.3ms).
> Above that, GPU wins and the gap widens with matrix size.
> On NVIDIA CUDA (Tensor Core), the crossover is much earlier and the speedup is much larger.

### Batch matmul (512 x 512 weights)

| Batch Size | GPU (wgpu) | Throughput |
|-----------|-----------|------------|
| 1 | 1.45 ms | 0.69 batch/ms |
| 4 | 1.59 ms | 2.52 batch/ms |
| 16 | 1.38 ms | 11.6 batch/ms |
| 64 | 2.45 ms | 26.1 batch/ms |
| 256 | 4.03 ms | 63.5 batch/ms |

### Memory compression

| Format | Size (1024x1024) | Ratio |
|--------|-----------------|-------|
| FP32 | 4,194,304 bytes | 1x |
| FP16 | 2,097,152 bytes | 2x |
| INT8 | 1,048,576 bytes | 4x |
| **ALICE-TRT (2-bit)** | **262,144 bytes** | **16x** |

## Quick Start

### Choose Your Path

Pick the entry point that matches your role. All paths share the same `GpuTernaryWeight` / `GpuTensor` primitives, so you can start with wgpu and swap to CUDA (or add FFI later) without rewriting the model.

| You are… | Path | Best for | Section |
|----------|------|----------|---------|
| **First time here** | [Hello, first inference](#hello-first-inference-30-seconds) ↓ | Sanity-check the install with one matvec | ↓ |
| **Cross-platform Rust dev** | [wgpu backend](#wgpu-backend-cross-platform-default) | Metal (macOS) / Vulkan (Linux) / DX12 (Windows), no NVIDIA lock-in | ↓ |
| **NVIDIA CUDA user** | [CUDA backend](#cuda-backend-nvidia-gpu) | Tensor Core (`wmma` / `dp4a`), maximum throughput | ↓ |
| **Multi-layer inference** | [Multi-layer](#multi-layer-inference) | 3+ layer feed-forward, forward pass, activations | ↓ |
| **Coming from ALICE-ML (CPU)** | [Import from ALICE-ML](#import-from-alice-ml) | Reuse quantized CPU weights on GPU, drop-in migration | ↓ |
| **Physics offload (Fix128)** | [Physics Solver Bridge](#physics-solver-bridge-feature-physics-solver-stable-since-v100) | GPU PGS iteration for ALICE-Physics, byte-exact vs CPU | ↓ Cross-Crate |
| **Neural SDF (real-time)** | [SDF Bridge](#sdf-bridge-feature-sdf) | Fit ternary NN to CSG tree, GPU eval millions of points | ↓ Cross-Crate |
| **Unity / UE5 integrator** | [C-ABI FFI](#c-abi-ffi-unity--ue5) | 37 `extern "C"` functions, ABI-versioned | ↓ |
| **Benchmarking / research** | [Benchmarks](#benchmarks) + `cargo bench` | Throughput, crossover point, compression ratio | ↑ |

### Hello, First Inference (30 seconds)

The smallest working sample — one 2-element matvec on the cross-platform wgpu backend:

```rust
use alice_trt::prelude::*;

let device  = GpuDevice::new().unwrap();
let compute = TernaryCompute::new(&device);

// 2x2 ternary weight matrix { {1, -1}, {0, 1} }
let weights = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
let input   = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);

let output = compute.matvec(&device, &input, &weights);
let result = output.download(&device);
assert_eq!(result, vec![-1.0, 3.0]);
```

Once this runs, jump to the path that matches your role.

### wgpu backend (cross-platform, default)

```rust
use alice_trt::prelude::*;

// 1. Initialize GPU
let device = GpuDevice::new().unwrap();
let compute = TernaryCompute::new(&device);

// 2. Upload ternary weights (2-bit in VRAM)
let weights = GpuTernaryWeight::from_ternary(
    &device, &[1, -1, 0, 1], 2, 2
);

// 3. Upload input
let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);

// 4. GPU inference (stays in VRAM)
let output = compute.matvec(&device, &input, &weights);

// 5. Download result
let result = output.download(&device);
// result = [-1.0, 3.0]
```

### CUDA backend (NVIDIA GPU)

```rust
use alice_trt::cuda::CudaTernaryEngine;

// 1. Initialize CUDA
let engine = CudaTernaryEngine::init(0).unwrap();
println!("{engine}");
// "CudaTernaryEngine(NVIDIA RTX 4090, sm8.9, 128SM, 24.0GB, TensorCores)"

// 2. Upload weights
let weights = engine.upload_weights(
    &plus_bits, &minus_bits, out_features, in_features, scale
).unwrap();

// 3. Single-shot inference
let result = engine.infer_matvec(&input_data, &weights).unwrap();
```

### Multi-layer inference

```rust
use alice_trt::prelude::*;

let device = GpuDevice::new().unwrap();
let compute = TernaryCompute::new(&device);

// Build a 3-layer network: 784 → 256 → 128 → 10
let mut engine = GpuInferenceEngine::new();
engine.add_layer(w1, Activation::ReLU);   // 784 → 256
engine.add_layer(w2, Activation::ReLU);   // 256 → 128
engine.add_layer(w3, Activation::None);   // 128 → 10

// Forward pass (all computation stays in VRAM)
let input = GpuTensor::from_f32(&device, &pixel_data, &[784]);
let output = engine.forward(&device, &compute, &input);
let logits = output.download(&device);
```

### Import from ALICE-ML

```rust
// Direct import from ALICE-ML's quantized weights
let cpu_kernel = alice_ml::TernaryWeightKernel::from_ternary(&weights, out, inp);
let gpu_weights = GpuTernaryWeight::from_kernel(&device, &cpu_kernel);

// Or from packed format
let cpu_packed = alice_ml::TernaryWeight::from_ternary(&weights, out, inp);
let gpu_weights = GpuTernaryWeight::from_packed(&device, &cpu_packed);
```

### Common Recipes

Additional patterns that come up frequently — copy-paste starting points.

**1. Batch matmul (many inputs, one weight)**

```rust
use alice_trt::prelude::*;

let device  = GpuDevice::new().unwrap();
let compute = TernaryCompute::new(&device);

let weights = GpuTernaryWeight::from_ternary(&device, &packed_ternary, out_dim, in_dim);
let batch   = GpuTensor::from_f32(&device, &batched_input, &[batch_size, in_dim]);

let output = compute.matmul_batch(&device, &batch, &weights);   // stays in VRAM
let host   = output.download(&device);
```

**2. Fused ReLU (in-place, no round-trip)**

```rust
let mut hidden = compute.matvec(&device, &input, &w1);
compute.relu_inplace(&device, &mut hidden);              // no separate download
let logits = compute.matvec(&device, &hidden, &w2);
```

**3. Physics solver bridge (v1.3.1, byte-exact vs CPU)**

```rust
use alice_physics::gpu_bridge::{DiffFixture, GpuSolverBridge};
use alice_physics::math::Fix128;
use alice_trt::{GpuDevice, TrtSolverAdapter};

let device      = GpuDevice::new()?;
let mut adapter = TrtSolverAdapter::new(&device);

adapter.send_island(&positions, &velocities);
adapter.set_gravity(Some([Fix128::ZERO, Fix128::from_ratio(-98, 10), Fix128::ZERO]));
adapter.push_distance_constraint(0, 1, Fix128::from_int(2));
adapter.dispatch_iterations(10, Fix128::from_ratio(1, 60));
adapter.recv_island(&mut positions, &mut velocities);
```

Requires `--features physics-solver`; the ALICE-Physics side auto-enables its `gpu-solver-bridge` feature.

**4. Neural SDF fit (ternary NN approximates a CSG tree)**

```rust
use alice_trt::sdf_bridge::{GpuNeuralSdf, NeuralSdfConfig};

let neural = GpuNeuralSdf::fit(
    &device,
    &sdf_node,                    // any alice_sdf::SdfNode
    bounds_min,
    bounds_max,
    &NeuralSdfConfig { hidden_dim: 64, num_layers: 4, training_points: 100_000, ..Default::default() },
)?;
let dists = neural.eval_batch(&device, &compute, &query_points)?;
```

Requires `--features sdf`.

**5. FFI usage from Unity / UE5 (C ABI)**

```c
// Enable with cargo: --features ffi (produces libalice_trt.dylib / .so / .dll)
AliceTrtDevice*   dev = alice_trt_device_new();
AliceTrtWeight*   w   = alice_trt_weight_from_ternary(dev, plus_bits, minus_bits, out_dim, in_dim, scale);
AliceTrtTensor*   in  = alice_trt_tensor_from_f32(dev, input, in_dim);
AliceTrtTensor*   out = alice_trt_tensor_output(dev, out_dim);
AliceTrtCompute*  c   = alice_trt_compute_new(dev);
alice_trt_compute_matvec(c, in, w, out);
float* result = alice_trt_tensor_download(out);
```

Full 37-function surface documented in the [C-ABI FFI](#c-abi-ffi-unity--ue5) section.

### Where to go next

| I want to… | See |
|------------|-----|
| Understand the Trojan Horse compression | [The Trojan Horse Strategy](#the-trojan-horse-strategy) |
| Compare CUDA kernels (wmma / dp4a / popc) | [CUDA Kernel Dispatch Hierarchy](#cuda-kernel-dispatch-hierarchy) |
| See raw throughput numbers | [Benchmarks](#benchmarks) |
| Offload Fix128 physics to GPU | [Cross-Crate Bridges → Physics](#physics-solver-bridge-feature-physics-solver-stable-since-v100) |
| Fit neural SDF from CSG | [Cross-Crate Bridges → SDF](#sdf-bridge-feature-sdf) |
| Persist inference metrics | [Cross-Crate Bridges → DB](#alice-db-bridge-feature-db) |
| Wire into Unity / UE5 | [C-ABI FFI](#c-abi-ffi-unity--ue5) + `bindings/unity/` + `bindings/ue5/` |
| Prove byte-exact vs CPU | [Test Suite](#test-suite) (37 fix128 + 170 physics-solver, 3-platform CI) |

Companion crates that plug directly into ALICE-TRT:

- **[ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML)** — CPU ternary inference, weights migrate 1:1 via `GpuTernaryWeight::from_kernel`
- **[ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics)** — Fix128 physics with `gpu-solver-bridge` opt-in for GPU PGS offload
- **[ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF)** — CSG source for `sdf_bridge` neural-fit inference

Japanese version: [README.ja.md](README.ja.md).

## C-ABI FFI (Unity / UE5)

37 `extern "C"` functions for GPU ternary inference from game engines. Enable with `--features ffi`.

| Module | Functions | Description |
|--------|-----------|-------------|
| Device | 3 | new, free, info |
| Weight | 9 | from_ternary, from_ternary_scaled, free, out_features, in_features, scale, words_per_row, memory_bytes, compression_ratio |
| Tensor | 10 | from_f32, zeros, output, free, download, ndim, shape, len, is_empty, memory_bytes |
| Compute | 5 | new, free, matvec, matmul_batch, relu_inplace |
| Engine | 9 | new, free, add_layer, num_layers, total_weight_bytes, equivalent_fp32_bytes, compression_ratio, forward, forward_batch |
| Version | 1 | version |

- **Unity C#**: `bindings/unity/AliceTrt.cs` — 37 `[DllImport]` + 5 RAII `IDisposable` handles
- **UE5 C++**: `bindings/ue5/AliceTrt.h` — 37 `extern "C"` + 5 RAII `unique_ptr` handles

## Test Suite

```bash
cargo test                    # Core tests (54)
cargo test --features ffi     # + FFI (63)
```

| Config | Tests |
|--------|-------|
| default | 54 (52 unit + 2 doc) |
| `--features ffi` | 63 (52 unit + 9 FFI + 2 doc) |

## Build

```bash
# Default: wgpu cross-platform backend
cargo build --release

# With CUDA backend (requires NVIDIA CUDA Toolkit)
cargo build --release --features cuda

# With FFI bindings
cargo build --release --features ffi

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build TensorRT plugin (requires TensorRT SDK, Linux only)
cd csrc && mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT
make -j$(nproc)
```

### Requirements

| Backend | Requirements |
|---------|-------------|
| wgpu (default) | Any GPU with Metal/Vulkan/DX12 support |
| CUDA | NVIDIA GPU + CUDA Toolkit 11.0+ |
| TensorRT plugin | CUDA + TensorRT 8.0+ (Linux) |

## Project Structure

```
ALICE-TRT/
├── Cargo.toml                    # Rust package config
├── build.rs                      # CUDA compilation (cc crate)
├── LICENSE                       # AGPL-3.0
│
├── src/
│   ├── lib.rs                    # Module declarations, re-exports, tests
│   ├── device.rs                 # GpuDevice (wgpu Device+Queue wrapper)
│   ├── kernel.rs                 # 4 WGSL compute shaders (source strings)
│   ├── weights.rs                # GpuTernaryWeight (2-bit bitplanes in VRAM)
│   ├── tensor.rs                 # GpuTensor (f32 buffer on GPU)
│   ├── pipeline.rs               # TernaryCompute (4 pre-compiled pipelines)
│   ├── inference.rs              # GpuInferenceEngine (multi-layer forward)
│   ├── ffi.rs                    # C-ABI FFI (37 functions, --features ffi)
│   ├── python.rs                 # PyO3 Python bindings (5 classes, --features python)
│   └── cuda/                     # CUDA backend (--features cuda)
│       ├── mod.rs                # Module declarations
│       ├── ffi.rs                # extern "C" FFI bindings
│       └── engine.rs             # CudaTernaryEngine (safe RAII wrapper)
│
├── csrc/
│   ├── alice_trt_c_api.h         # C ABI header (Rust FFI bridge)
│   ├── bitnet_kernel.cu          # CUDA kernels: wmma + dp4a + popc + C API
│   ├── CMakeLists.txt            # CMake build (static lib + TRT plugin)
│   └── plugin/
│       ├── alice_ternary_plugin.h    # TensorRT IPluginV2DynamicExt
│       ├── alice_ternary_plugin.cpp  # Plugin registration & serialization
│       └── alice_ternary_plugin.cu   # enqueue() dispatch
│
└── benches/
    └── gpu_matmul.rs             # Criterion benchmarks
```

## How It Works

### Weight Encoding

Each ternary weight `w in {-1, 0, +1}` is encoded as 2 bits across two bitplanes:

| Weight | plus_bit | minus_bit |
|--------|----------|-----------|
| +1 | 1 | 0 |
| 0 | 0 | 0 |
| -1 | 0 | 1 |

32 weights are packed into a single `u32` word.
A 4096x4096 weight matrix occupies **256 KB** instead of 64 MB (FP32) or 32 MB (FP16).

### Dot Product Computation

```
Branchless ternary multiply:
  acc += input[i] * (float)(plus_bit[i] - minus_bit[i])

Where (plus_bit - minus_bit) is always {-1, 0, +1}.
No branch divergence on GPU. Full warp utilization.
```

### The wmma Trick (Tensor Core)

```
For each 16x16 tile:
  1. Load 2-bit weights from VRAM            (8x bandwidth savings)
  2. Unpack to FP16 {-1.0, 0.0, +1.0}       (in shared memory, ~2 cycles)
  3. wmma::load_matrix_sync(fragment)         (standard Tensor Core API)
  4. wmma::mma_sync(C, A, B, C)              (Tensor Core fires at 125 TFLOPS)

The Tensor Core has no idea it's processing ternary data.
It sees valid FP16. It computes. We win.
```

## Cross-Crate Bridges

ALICE-TRT connects to other ALICE ecosystem crates via feature-gated bridge modules:

| Bridge | Feature | Target Crate | Description |
|--------|---------|--------------|-------------|
| `physics_bridge::GpuPhysicsController` | `physics` | [ALICE-Physics](../ALICE-Physics) | GPU ternary inference for physics **control policies**, batched force computation |
| `physics_bridge::TrtSolverAdapter` | `physics-solver` | [ALICE-Physics v0.8.0+](../ALICE-Physics) | Compute-shader-based Fix128 **PGS solver offload** — gravity + floor + N-way distance constraints (Gauss-Seidel), byte-exact vs CPU on Metal / Vulkan / DX12 |
| `fix128::Fix128GpuKernel` | `fix128-arithmetic` | [ALICE-Physics v0.8.0+](../ALICE-Physics) | Fix128 GPU primitives — `add` / `sub` / `mul` / `dot` (2-stage reduce) + PGS `integrate` / `project_floor` / `project_distance` WGSL kernels |
| `sdf_bridge` | `sdf` | [ALICE-SDF](../ALICE-SDF) | GPU neural SDF approximation via ternary networks for real-time distance field queries |

### Physics Solver Bridge (feature: `physics-solver`, stable since v1.0.0)

Compute-shader-based Fix128 PGS solver offload for [ALICE-Physics v0.8.0+](../ALICE-Physics). `TrtSolverAdapter` implements the `alice_physics::gpu_bridge::GpuSolverBridge` trait so ALICE-Physics callers can inject the GPU path via dependency injection. The default CPU-native TGS pipeline remains untouched unless the adapter is explicitly wired in.

**Semver stability commitment (v1.0.0+):** the public `TrtSolverAdapter` surface (constructor / `send_island` / `dispatch_iterations` / `set_gravity` / `set_floor` / `push_distance_constraint` / `recv_island` / `assert_bit_exact_vs_cpu`) is byte-stable across the v1.x line. Additive helpers ship as minor bumps; breaking changes go to v2.

**Pair with `alice-physics/gpu-solver-bridge`.** The `physics-solver` feature automatically enables it on the ALICE-Physics side.

```toml
[dependencies]
alice-trt     = { path = "../ALICE-TRT",     features = ["physics-solver"] }
alice-physics = { path = "../ALICE-Physics", features = ["gpu-solver-bridge"] }
```

```rust
use alice_physics::gpu_bridge::{DiffFixture, GpuSolverBridge};
use alice_physics::math::Fix128;
use alice_trt::{GpuDevice, TrtSolverAdapter};

// 1. Construct an adapter bound to a wgpu device.
let device = GpuDevice::new()?;
let mut adapter = TrtSolverAdapter::new(&device);

// 2. Upload the current island state.
let positions:  Vec<[Fix128; 3]> = /* ... */;
let velocities: Vec<[Fix128; 3]> = /* ... */;
adapter.send_island(&positions, &velocities);

// 3. Install constraints (all optional, all additive).
adapter.set_gravity(Some([Fix128::ZERO, Fix128::from_ratio(-98, 10), Fix128::ZERO]));
adapter.set_floor(Some(Fix128::ZERO));                             // y >= 0
adapter.push_distance_constraint(0, 1, Fix128::from_int(2));       // |p0 - p1| = 2
adapter.push_distance_constraint(1, 2, Fix128::from_int(2));       // Gauss-Seidel ordered

// 4. Dispatch PGS iterations on the GPU (live shader path, not a stub).
adapter.dispatch_iterations(10, Fix128::from_ratio(1, 60));

// 5. Read back into caller-owned slices.
let mut out_positions  = vec![[Fix128::ZERO; 3]; positions.len()];
let mut out_velocities = vec![[Fix128::ZERO; 3]; velocities.len()];
adapter.recv_island(&mut out_positions, &mut out_velocities);

// 6. Certify byte-for-byte equivalence with the CPU-side solver.
adapter.assert_bit_exact_vs_cpu(&DiffFixture {
    description: "3body_triangle_10iter",
    tolerance: Fix128::ZERO, // strict byte-for-byte equality
})?;
```

**Feature timeline (v0.3.0 → v1.3.1):**

| Feature | Since | Notes |
|---------|-------|-------|
| Live PGS dispatch (integrate) | v0.4.0 | Semi-implicit Euler on GPU, byte-exact vs CPU |
| Gravity | v0.5.0 | `set_gravity(Option<[Fix128; 3]>)` |
| Floor constraint | v0.6.0 | `set_floor(Option<Fix128>)`, projects `y >= floor` |
| WARP crash fix (uniform-flow) | v0.8.1 | FXC X4026 discipline, 3-platform CI green |
| Distance constraint (single) | v1.2.0 | CPU-precompute scalar + GPU project |
| Multi-distance (Gauss-Seidel) | v1.3.0 | `push_distance_constraint(a, b, L)` × N |
| Accessors | v1.3.1 | `distance_constraint_count()` / `has_distance_constraints()` |

**CI matrix:** every release is validated on macOS (Metal) / Ubuntu (Vulkan lavapipe) / Windows (DX12 WARP) with byte-exact assertions against the CPU golden solver.

Companion releases: [ALICE-Physics v0.8.0+](https://github.com/ext-sakamoro/ALICE-Physics/releases).

### Fix128 GPU Primitive (feature: `fix128-arithmetic`, stable since v1.0.0)

Standalone `Fix128Gpu` (`#[repr(C)]` + `bytemuck::Pod`, layout-compatible with `alice_physics::Fix128`) plus the `Fix128GpuKernel` trait. Enable this feature without `physics-solver` to build the Fix128 GPU primitive without pulling in ALICE-Physics.

```toml
[dependencies]
alice-trt = { path = "../ALICE-TRT", features = ["fix128-arithmetic"] }
```

**Live GPU kernels (all WGSL, all byte-exact vs CPU reference):**

| Kernel | Since | WGSL constant |
|--------|-------|---------------|
| `add` / `sub` | v0.3.0 | `FIX128_ADD_WGSL` / `FIX128_SUB_WGSL` |
| `mul` | v0.3.0 | `FIX128_MUL_WGSL` |
| `dot` (2-stage reduce) | v0.7.1 / v0.8.1 | `FIX128_DOT_WGSL` + `FIX128_DOT_FINAL_WGSL` |
| PGS integrate (semi-implicit Euler) | v0.4.0 | `FIX128_PGS_INTEGRATE_WGSL` |
| PGS project (floor) | v0.6.0 | `FIX128_PGS_PROJECT_FLOOR_WGSL` |
| PGS project (distance, rigid rod) | v1.4.2 | `FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL` |
| PGS project (distance, batched) | v1.5.1 | `FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL` |
| Fix128 GPU `div` | v1.4.0 | `FIX128_DIV_WGSL` |
| Fix128 GPU `sqrt` | v1.4.1 | `FIX128_SQRT_WGSL` |
| Fix128 AABB helpers (Phase 3 §1) | v2.1.0 | `FIX128_AABB_HELPERS_WGSL` |
| Fix128 Morton code (Phase 3 §1) | v2.1.0 | `FIX128_MORTON_CODE_WGSL` |
| Fix128 Morton sort (Phase 3 §2) | v2.2.0 | `FIX128_MORTON_SORT_WGSL` + `dispatch_fix128_morton_sort` |
| Fix128 GPU BVH build (Phase 3 §3) | v2.3.0 | `FIX128_BVH_BUILD_WGSL` + `dispatch_fix128_bvh_build` |
| Fix128 GPU BVH find_pairs (Phase 3 §4) | v2.4.0 | `FIX128_BVH_FIND_PAIRS_WGSL` + `dispatch_fix128_bvh_find_pairs` |
| Fix128 GPU sphere-sphere contact (Phase 3 §5) | v2.5.0 | `FIX128_SPHERE_SPHERE_CONTACT_WGSL` + `dispatch_fix128_sphere_sphere_contact` |
| Fix128 GPU PGS contact solve (Phase 3 §6) | v2.6.0 | `FIX128_PGS_CONTACT_SOLVE_WGSL` + `dispatch_fix128_pgs_contact_solve` + `TrtSolverAdapter::dispatch_contact_solve_iteration` |

CPU-side `Fix128Gpu` also exposes `sqrt` (v1.0.1, delegates to `alice_physics::Fix128::sqrt`) and `div` (v1.0.6) for host precompute paths. The v1.1.0 single-constraint uniform variant (`FIX128_PGS_PROJECT_DISTANCE_WGSL`) was deprecated in v1.7.0 and removed in v2.0.0; external `Fix128GpuKernel` implementers should migrate to the rigid-rod or batched shader listed above.

**37 Fix128 unit tests + 170 physics-solver tests** run on all three platforms per release. The v2.3.0 BVH build kernel ships with a byte-exact CPU-GPU golden on three fixtures (pile 4×4×2, uniform 3×3×3 grid, degenerate all-colocated) — every node in the returned `Vec<BvhNodeGpu>` is required to match `alice_physics::bvh::LinearBvh::build(...).nodes` bit-for-bit on Metal / Vulkan lavapipe / DX12 WARP.

### SDF Bridge (feature: `sdf`)

GPU-accelerated neural SDF approximation via ternary networks. Fits a lightweight ternary neural network to approximate an SDF node's distance field, enabling real-time GPU evaluation of complex CSG trees.

```toml
[dependencies]
alice-trt = { path = "../ALICE-TRT", features = ["sdf"] }
```

```rust
use alice_trt::sdf_bridge::{GpuNeuralSdf, NeuralSdfConfig};

let config = NeuralSdfConfig {
    hidden_dim: 64,
    num_layers: 4,
    training_points: 100_000,
    ..Default::default()
};

// Fit ternary NN to SDF
let neural_sdf = GpuNeuralSdf::fit(&device, &sdf_node, bounds_min, bounds_max, &config)?;

// Batch evaluate on GPU
let distances = neural_sdf.eval_batch(&device, &compute, &query_points)?;
```

### ALICE-DB Bridge (feature: `db`)

Inference metrics persistence for model performance tracking.

- `InferenceRecord` — 34-byte binary serialization (model_hash, latency_us, batch_size, throughput_fps)
- `TrtDbStore` — Store/query inference metrics by model and time range
- `avg_latency_us()` — Compute average latency for a model

Enable: `alice-trt = { features = ["db"] }`

### ALICE-View Bridge (feature: `view`)

Neural upscaling for GPU-rendered content (DLSS-style).

- `NeuralUpscaler` — Quality tiers: Performance (4x) / Balanced (2.5x) / Quality (1.7x) / UltraQuality (1.3x)
- `render_scale()` / `internal_resolution()` — Compute internal render resolution
- `estimate_psnr()` — Quality estimation

Enable: `alice-trt = { features = ["view"] }`

### ALICE-Voice Bridge (feature: `voice`)

GPU-accelerated voice feature extraction.

- `GpuVoiceExtractor` — Mel-frequency, energy, ZCR, spectral centroid extraction
- `MelConfig` — Configurable mel filterbank (n_mels, fmin, fmax)
- `extract_features()` — Frame-by-frame voice feature extraction

Enable: `alice-trt = { features = ["voice"] }`

### Cargo Profile

Standardized `[profile.bench]` with `lto = "thin"`, `codegen-units = 1`, `debug = false` for consistent benchmarking across ALICE crates.

## ALICE Ecosystem

| Crate | Role |
|-------|------|
| [ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML) | CPU inference engine (branchless ternary, AVX2 SIMD) |
| **ALICE-TRT** | GPU inference engine (wgpu + CUDA Tensor Core) |
| [ALICE-Eco-System](https://github.com/ext-sakamoro/ALICE-Eco-System) | Integration demos & benchmarks |

## License

AGPL-3.0. See [LICENSE](LICENSE).

Copyright (c) Moroya Sakamoto
