# Changelog

All notable changes to ALICE-TRT will be documented in this file.

## [0.4.2] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU mul WGSL skeleton
  - `FIX128_MUL_WGSL` compute shader source with the schoolbook helpers reused by the full pipeline:
    - `umul_wide(a, b) -> vec2<u32>` — u32×u32 → u64 via 16-bit halving (bit-exact)
    - `u64_add(a, b) -> vec3<u32>` — 64-bit unsigned add returning (lo, hi, carry)
    - `u64_mul_wide(a, b) -> vec4<u32>` — u64×u64 → u128 via 4 u32×u32 partial products
  - `fix128_mul_unsigned_lo_main` @compute entry point emitting the low 128 bits of the unsigned 128×128→256 product (validation harness for the schoolbook helpers)
  - 2 shader-source tests + 1 real-GPU compile validity test (naga parser via `Device::create_shader_module`)

### Roadmap for v0.5.0

- Signed correction for the mixed `i64 × u64` partial products (`hl` / `lh` in the CPU reference)
- Middle-128-bit extraction (bits [192:64] of the 256-bit signed product) to complete `Fix128Gpu::mul` on the GPU
- `Fix128WgpuKernel::mul` end-to-end dispatch + `wgpu_mul_matches_cpu_golden` bit-exact test
- `FIX128_DOT_WGSL` — index-ordered accumulate with `workgroupBarrier` sync (no subgroup reduce, keeps determinism contract §1 経路 3)

### Backwards compatibility

- Fully backwards compatible with v0.4.1
- No new public API on the trait surface; the `FIX128_MUL_WGSL` constant is additive
- `Fix128WgpuKernel::mul` still returns `unimplemented!()` until the signed pipeline lands in v0.5.0

## [0.4.1] - 2026-07-03

### Added

- **`--features fix128-arithmetic`** — Fix128Gpu::mul CPU reference
  - Byte-for-byte mirror of `alice_physics::math::Fix128::mul` (I64F64 semantics, middle 128 bits of the 256-bit signed product via schoolbook)
  - 4 unit tests (identity / integer / negative / fractional-half scaling)

## [0.4.0] - 2026-07-03

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU sub kernel
  - `FIX128_SUB_WGSL` compute shader source with borrow-aware `u64_sub` helper
  - `Fix128WgpuKernel::sub` real-GPU dispatch
  - `Fix128GpuKernel` trait impl for `Fix128WgpuKernel<'_>` (add + sub live, mul + dot skeleton)
  - `wgpu_sub_matches_cpu_golden` real-GPU bit-exact test
  - `wgpu_trait_add_matches_inherent_add` trait-vs-inherent equivalence test

## [0.3.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU add kernel
  - `FIX128_ADD_WGSL` compute shader source with carry-aware `u64_add` helper
  - `Fix128WgpuKernel::new / add` real-GPU dispatch via wgpu ComputePipeline
  - `wgpu_add_matches_cpu_golden` real-GPU bit-exact test (skips when no adapter)

## [0.2.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU primitive skeleton
  - `Fix128Gpu` (`#[repr(C)]` + `bytemuck::Pod`, layout-compatible with `alice_physics::Fix128`)
  - `Fix128GpuKernel` trait (`add` / `sub` / `mul` / `dot` signatures)
  - Determinism contract documented in rustdoc (subgroup reduce ordering + traversal ordering)
  - 2 unit tests (constant + round-trip)
- **`--features physics-solver`** — Compute-shader-based Fix128 physics solver bridge
  - `TrtSolverAdapter` implementing `alice_physics::gpu_bridge::GpuSolverBridge`
  - Buffers island contents into `Fix128Gpu` storage for the WGSL kernels (kernels land in the follow-up)
  - 1 unit test (zero-iteration round-trip byte-for-byte)
  - Composite feature: enables `physics` + `fix128-arithmetic` + `alice-physics/gpu-solver-bridge`

### Companion release

Pair with [ALICE-Physics v0.8.0](https://github.com/ext-sakamoro/ALICE-Physics/releases/tag/v0.8.0) `--features gpu-solver-bridge` (auto-enabled by `physics-solver`).

### Backwards compatibility

- Fully backwards compatible with v0.1.x
- Existing `GpuPhysicsController` (ML control policy inference) is untouched
- New Fix128 GPU primitive + `TrtSolverAdapter` are strictly opt-in; default `cargo build` sees zero new API surface

## [0.1.1] - 2026-03-04

### Added
- `ffi` — C-ABI FFI 37 `extern "C"` functions (Device/Weight/Tensor/Compute/Engine/Version)
- `python` — PyO3 5 classes (GpuDevice, GpuTernaryWeight, GpuTensor, TernaryCompute, InferenceEngine)
- Unity C# bindings — 37 DllImport + 5 RAII IDisposable handles (`bindings/unity/AliceTrt.cs`)
- UE5 C++ bindings — 37 extern C + 5 RAII unique_ptr handles (`bindings/ue5/AliceTrt.h`)
- FFI prefix: `at_trt_*`
- 63 tests (52 core + 9 FFI + 2 doc-tests)

### Fixed
- `cargo fmt` 21箇所の末尾スペース修正

## [0.1.0] - 2026-02-23

### Added
- `device` — `GpuDevice` wgpu initialization (Metal/Vulkan/DX12)
- `kernel` — WGSL compute shaders: matvec, tiled matvec, batched matmul, ReLU
- `weights` — `GpuTernaryWeight` 2-bit bitplane storage (8x compression vs FP16)
- `tensor` — `GpuTensor` VRAM buffer with upload/download
- `pipeline` — `TernaryCompute` 4-pipeline dispatch with auto kernel selection
- `inference` — `GpuInferenceEngine` multi-layer forward pass (all data stays in VRAM)
- Feature flags: `cuda`, `physics`, `sdf`, `db`, `view`, `voice`
- ALICE-ML `TernaryWeightKernel` / `TernaryWeight` zero-copy import
- 52 unit tests + 2 doc-tests
