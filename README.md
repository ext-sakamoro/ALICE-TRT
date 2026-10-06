# ALICE-TRT

A GPU inference engine for 1.58-bit ternary neural networks, plus a set of
Fix128 (128-bit fixed-point) compute kernels that let ALICE-Physics run parts of
its solver on the GPU with byte-exact results.

English | [日本語](README_JP.md)

[![CI](https://github.com/ext-sakamoro/ALICE-TRT/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-TRT/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.87-blue)](#minimum-supported-rust-version)
[![License](https://img.shields.io/badge/license-AGPL--3.0%20OR%20Commercial-blue)](#license)

Ternary weights `{-1, 0, +1}` are stored in VRAM as two bitplanes (2 bits per
weight, 32 weights per `u32` word) and expanded to `{-1.0, 0.0, +1.0}` in
registers inside the compute shader. Moving 2 bits per weight instead of 16
(FP16) reduces the memory traffic that limits matrix-vector products; the
multiply-add units see ordinary floating-point data.

```text
FP16 weights:   VRAM ──[16 bits/weight]──▶ L2 ──▶ L1 ──▶ registers ──▶ FMA
ALICE-TRT:      VRAM ──[ 2 bits/weight]──▶ L2 ──▶ L1 ──▶ registers ──[expand]──▶ FMA
```

The default backend is [wgpu](https://wgpu.rs) (Metal, Vulkan, DX12). A CUDA
backend and a TensorRT plugin are included as source behind the `cuda` feature
and `csrc/`.

It is not a general-purpose inference runtime: there is no model format
loader, no operator set beyond ternary matvec / batched matmul / ReLU, and no
training. Below a few thousand rows the wgpu dispatch overhead (about 1.3 ms)
dominates and a CPU implementation such as ALICE-ML is faster (see
[Performance](#performance)).

## Contents

- [Installation](#installation)
- [Example](#example)
- [What is included](#what-is-included)
- [Cargo features](#cargo-features)
- [Validation and known defects](#validation-and-known-defects)
- [Bindings](#bindings)
- [Performance](#performance)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [Building and testing](#building-and-testing)
- [Related crates](#related-crates)
- [License](#license)

## Installation

ALICE-TRT is not published on crates.io. It depends on
[ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML) through a path dependency,
so check out both repositories side by side:

```sh
git clone https://github.com/ext-sakamoro/ALICE-ML.git
git clone https://github.com/ext-sakamoro/ALICE-TRT.git
```

```toml
[dependencies]
alice-trt = { path = "../ALICE-TRT" }
```

The `physics` / `physics-solver`, `sdf` and `db` features additionally need
`../ALICE-Physics`, `../ALICE-SDF` and `../ALICE-DB`.

## Example

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
// result ≈ [-1.0, 3.0]
```

A multi-layer network is built with `GpuInferenceEngine::add_layer(weights,
Activation::ReLU)` and run with `forward` / `forward_batch`; intermediate
activations stay in VRAM. Weights quantised on the CPU by ALICE-ML are uploaded
with `GpuTernaryWeight::from_kernel` / `from_packed`.

## What is included

| Area | Items | Feature |
|------|-------|---------|
| Ternary inference (wgpu) | `GpuDevice`, `GpuTernaryWeight` (2-bit bitplanes), `GpuTensor`, `TernaryCompute` (`matvec`, `matmul_batch`, `relu_inplace`; simple and tiled kernels selected by layer size), `GpuInferenceEngine` (multi-layer forward) | default |
| Profiling | `InferenceProfiler` (per-layer wall-clock time with a GPU sync between layers, FLOP estimates) | default |
| Constraint colouring | `constraint_graph::ConstraintGraph::greedy_color` (deterministic greedy colouring used by the batched contact solve) | default |
| Fix128 GPU arithmetic | `Fix128Gpu` (layout-compatible with `alice_physics::Fix128`), WGSL kernels for add / sub / mul / div / sqrt / dot, quaternion multiply and rotate | `fix128-arithmetic` |
| Fix128 GPU physics pipeline | AABB helpers, Morton code and sort, BVH build, BVH `find_pairs`, sphere-sphere contact, PGS integrate / floor / distance projection, PGS contact solve (sequential and colour-batched), ball-socket joint solve | `fix128-arithmetic` |
| Physics solver offload | `physics_bridge::TrtSolverAdapter`, an implementation of `alice_physics::gpu_bridge::GpuSolverBridge` (inject into `PhysicsWorld::step_with_bridge`) | `physics-solver` |
| Physics control policies | `physics_bridge::GpuPhysicsController` (batched ternary network that produces forces for rigid bodies) | `physics` |
| CUDA backend | `cuda::CudaTernaryEngine` over `csrc/bitnet_kernel.cu` (wmma / dp4a / popc kernels chosen by compute capability) | `cuda` |
| TensorRT plugin | `csrc/plugin/` (`IPluginV2DynamicExt`, built with CMake, not by Cargo) | none |
| C ABI | 37 `extern "C"` functions over device / weight / tensor / compute / engine | `ffi` |
| Python | PyO3 module with 5 classes | `python` |
| Neural SDF | `sdf_bridge::GpuNeuralSdf` (`eval_batch`; `fit` is not implemented, see below) | `sdf` |
| Inference metrics | `db_bridge::TrtDbStore` (34-byte `InferenceRecord` in ALICE-DB) | `db` |
| View / voice helpers | `view_bridge` (render-scale, internal resolution, timing and PSNR estimates; no pixel processing), `voice_bridge` (mel / energy / zero-crossing / centroid features computed on the CPU) | `view`, `voice` |

The CUDA backend picks one of three kernels by compute capability at run time:

| Kernel | Compute capability | Method |
|--------|--------------------|--------|
| `wmma` | 7.0 and later | 2-bit → FP16 `{-1, 0, +1}` tiles in shared memory → `wmma::mma_sync` (Tensor Cores) |
| `dp4a` | 6.1 and later | 2-bit → INT8 → `__dp4a` 4-element dot product |
| `popc` | any | `__popc` / `__ffs` over the bitplanes; words with no non-zero weight are skipped, sparse words iterate set bits only |

The full API is listed on the generated documentation (`cargo doc --open`).
Design notes for the physics pipeline are in
[docs/PHASE_3_DESIGN.md](docs/PHASE_3_DESIGN.md) and
[docs/PHASE_4_DESIGN.md](docs/PHASE_4_DESIGN.md); the v2 to v3 API change is in
[docs/MIGRATION_v3.md](docs/MIGRATION_v3.md).

## Cargo features

| Feature | Enables | Needs |
|---------|---------|-------|
| `ffi` | C ABI (`src/ffi.rs`) | |
| `python` | PyO3 bindings | Python 3.7 to 3.13 |
| `cuda` | CUDA backend | NVIDIA CUDA Toolkit (`nvcc`) |
| `physics` | `GpuPhysicsController` | `../ALICE-Physics` |
| `sdf` | Neural SDF bridge | `../ALICE-SDF` |
| `db` | Inference metrics in ALICE-DB | `../ALICE-DB` |
| `view` | View helpers | |
| `voice` | Voice feature helpers | |
| `fix128-arithmetic` | Fix128 GPU kernels | |
| `physics-solver` | `TrtSolverAdapter`; implies `physics`, `fix128-arithmetic` and `alice-physics/gpu-solver-bridge` | `../ALICE-Physics` |

No feature is enabled by default.

## Validation and known defects

What CI checks ([.github/workflows/ci.yml](.github/workflows/ci.yml)):

- **Byte-exact CPU-GPU goldens** for every Fix128 kernel and the physics
  pipeline stages (Morton sort, BVH build, `find_pairs`, sphere-sphere contact,
  PGS contact solve sequential and batched, `step_with_bridge`): each GPU result
  must equal the ALICE-Physics CPU result bit for bit. They run on three
  adapters: Metal (macOS), Vulkan lavapipe (Linux, software) and DX12 WARP
  (Windows, software).
  <!-- claim-test: wgpu_pgs_contact_solve_matches_cpu_golden -->
- **Closed-form oracles** in [tests/analytic_oracle.rs](tests/analytic_oracle.rs):
  ternary matvec on both kernels equals the integer dot product, the scaled
  kernel multiplies by the scale, batched rows equal single matvecs, a two-layer
  ReLU network matches an f64 closed form, Fix128 arithmetic matches dyadic
  closed forms, and the view / voice helpers match their formulas. A missing GPU
  adapter fails these tests instead of skipping them.
  <!-- claim-test: gpu_ternary_matvec_is_the_exact_integer_dot_product_on_both_kernels -->
- Device-free unit tests and doctests on Linux, macOS and Windows; clippy with
  `-D warnings`; rustdoc with `-D warnings`; the MSRV build; and, in
  [security-audit.yml](.github/workflows/security-audit.yml), `cargo audit`,
  `cargo deny` and `cargo machete`.

Not built in CI: the `cuda` feature and the TensorRT plugin (no NVIDIA runner),
and the `sdf` / `db` features (their sibling crates pull further path
dependencies that CI does not check out).

Known defects:

- **Windows DX12 WARP: the analytic oracle process crashes.** In the
  `Fix128 GPU (windows-latest)` job, `tests/analytic_oracle.rs` terminates with
  `STATUS_ACCESS_VIOLATION` inside
  `gpu_ternary_matvec_is_the_exact_integer_dot_product_on_both_kernels`. The
  Fix128 GPU unit tests pass on the same adapter. The job is red until this is
  fixed.
- **Windows DX12 WARP: the BVH build golden crashes.** In the
  `Fix128 GPU + physics-solver goldens (windows-latest)` job, the test process
  exits with code 2173 inside `fix128::tests::wgpu_bvh_build_matches_cpu_golden`
  (present since the BVH build kernel was added). The job is red until this is
  fixed; Metal and Vulkan lavapipe pass.
- **`GpuNeuralSdf::fit` is not implemented.** It panics with `todo!`. An earlier
  version returned a network unrelated to the SDF; the oracle
  `neural_sdf_fit_of_a_unit_sphere_is_within_a_quarter_radius` is `#[ignore]`d
  until fitting exists.
- **The `db` feature does not compile against the current ALICE-DB**
  (`src/db_bridge.rs` uses a `range` method and a `put` signature that ALICE-DB
  no longer has).
- **`TrtSolverAdapter::assert_bit_exact_vs_cpu` does not compare anything at run
  time** and returns `Ok(())`; the byte-exact property rests on the golden tests
  above.
- `TrtSolverAdapter::send_joints` supports `Joint::Ball` only and panics on any
  other joint type (see [docs/PHASE_4_DESIGN.md](docs/PHASE_4_DESIGN.md)).

## Bindings

| Language | File | Contents |
|----------|------|----------|
| C | [csrc/alice_trt_c_api.h](csrc/alice_trt_c_api.h) | C interface of the CUDA kernels (called from `src/cuda/` over FFI) |
| C# (Unity) | [bindings/unity/AliceTrt.cs](bindings/unity/AliceTrt.cs) | `[DllImport]` declarations for the `ffi` functions and `IDisposable` handles |
| C++ (UE5) | [bindings/ue5/AliceTrt.h](bindings/ue5/AliceTrt.h) | `extern "C"` declarations and RAII `unique_ptr` handles |
| Python | `src/python.rs` | PyO3 classes `GpuDevice`, `GpuTernaryWeight`, `GpuTensor`, `TernaryCompute`, `InferenceEngine` |

Build the shared library with `cargo build --release --features ffi`
(`libalice_trt.dylib` / `libalice_trt.so` / `alice_trt.dll`).

## Performance

Measured with `benches/gpu_matmul.rs` (Criterion) on an arm64 laptop GPU
through the wgpu Metal backend. The CPU column is ALICE-ML's branchless ternary
matvec on the same machine.
<!-- perf-measured: 2026-02-03 benches/gpu_matmul.rs -->

GPU matvec, N × N ternary weights:

| N | GPU (wgpu) | Notes |
|---|-----------|-------|
| 64 | 1.37 ms | dispatch overhead dominates |
| 256 | 1.28 ms | dispatch overhead dominates |
| 512 | 1.48 ms | |
| 1024 | 1.57 ms | tiled kernel (shared memory + parallel reduction) |
| 2048 | 1.79 ms | |
| 4096 | 2.61 ms | |

CPU (ALICE-ML) and GPU:

| N | CPU (ALICE-ML) | GPU (wgpu) | Faster |
|---|---------------|-----------|--------|
| 256 | 60 µs | 1.28 ms | CPU, 21× |
| 1024 | 912 µs | 1.34 ms | CPU, 1.5× |
| 4096 | 21.3 ms | 2.52 ms | GPU, 8.4× |

The crossover is near N = 2048. Batched matmul with 512 × 512 weights: batch 1
1.45 ms, 4 1.59 ms, 16 1.38 ms, 64 2.45 ms, 256 4.03 ms.

Weight memory for a 1024 × 1024 layer: FP32 4,194,304 bytes, FP16 2,097,152,
INT8 1,048,576, ALICE-TRT 262,144 (1/16 of FP32).

## Minimum supported Rust version

<!-- readme-sync: msrv -->
Rust 1.87 (`rust-version` in Cargo.toml). The floor comes from the ALICE-ML path
dependency. CI compiles the library with exactly this toolchain; the
development toolchain is pinned to 1.92.0 in `rust-toolchain.toml`.

## Building and testing

```sh
cargo build --release                       # wgpu backend
cargo build --release --features ffi        # + C ABI
cargo build --release --features cuda       # + CUDA backend (needs nvcc)
cargo test --features fix128-arithmetic,physics-solver -- --test-threads=1
cargo bench --bench gpu_matmul
scripts/preflight.sh                        # every CI gate that runs locally
```

GPU tests need an adapter; on Linux without a GPU, Mesa's lavapipe
(`mesa-vulkan-drivers`) provides one. The TensorRT plugin is built separately:

```sh
cd csrc && mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT
make -j"$(nproc)"
```

## Related crates

| Crate | Relation |
|-------|----------|
| [ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML) | CPU ternary inference; weights move to the GPU with `from_kernel` / `from_packed` |
| [ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics) | deterministic Fix128 physics; `TrtSolverAdapter` implements its `GpuSolverBridge` |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | SDF trees for the `sdf` bridge |
| [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) | storage for the `db` bridge |

## License

`AGPL-3.0 OR LicenseRef-Commercial`: dual-licensed, pick either.

| Option | Terms | Use it when |
|--------|-------|-------------|
| **AGPL-3.0** | [LICENSE-AGPL](LICENSE-AGPL), free, no reporting obligation | Your project is itself AGPL-compatible open source, or you only use it internally |
| **Commercial License** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md), paid, removes the copyleft | Closed-source product, proprietary SaaS, edge / firmware distribution, plugin redistribution, or a platform NDA that forbids source disclosure |

AGPL is a strong copyleft: a product, firmware image, or service that links
`alice-trt` and is distributed or served to users must be released under the
AGPL as well. That is intentional for the open ecosystem, and the Commercial
License exists for the cases where it is not something you are able to do.

Commercial licence enquiries: <contact@extoria.co.jp>
