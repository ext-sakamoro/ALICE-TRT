# ALICE-TRT

**GPU Ternary Inference Engine — Trojan Horse Edition**

> "Don't move data. Move the law."
> "The GPU thinks it's computing FP16. It's computing {-1, 0, +1}."

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

## Build

```bash
# Default: wgpu cross-platform backend
cargo build --release

# With CUDA backend (requires NVIDIA CUDA Toolkit)
cargo build --release --features cuda

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
| `physics_bridge` | `physics` | [ALICE-Physics](../ALICE-Physics) | GPU ternary inference for physics control policies, batched force computation |
| `sdf_bridge` | `sdf` | [ALICE-SDF](../ALICE-SDF) | GPU neural SDF approximation via ternary networks for real-time distance field queries |

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
