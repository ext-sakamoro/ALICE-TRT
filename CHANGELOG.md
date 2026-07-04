# Changelog

All notable changes to ALICE-TRT will be documented in this file.

## [0.7.0] - 2026-07-04

### Changed

- **`FIX128_DOT_WGSL::fix128_dot_main`** rewritten from single-thread serial (`@workgroup_size(1)`) to **64-thread blocked reduction** (`@workgroup_size(64)`):
  - **Phase 1 (parallel)**: thread `t ∈ [0, 64)` computes `partials[t] = Σ_{i=t·B}^{min((t+1)·B, N)} a[i]·b[i]` with `B = ⌈N/64⌉`, iterating **in-block index order**.
  - `workgroupBarrier()` synchronises the 64 threads.
  - **Phase 2 (serial)**: thread 0 folds `partials[0..64]` in block-index order.
  - Total order = canonical index 0..N → **byte-for-byte equal to the previous single-thread serial fold**. The change is purely a performance shift; the determinism contract §1 経路 3 is preserved.
  - Parallel speedup up to 64× for `N ≥ 64`.

### Added

- **`wgpu_dot_parallel_100_matches_cpu_golden`** — new determinism proof test on `N = 100` mixed-sign fixture with interleaved hi/lo bits designed to expose ordering-dependent wraparound. GPU (64-thread blocked) matches CPU single-thread golden byte-for-byte.
- **`wgpu_dot_zero_elements_returns_zero`** — empty-input contract.
- Additional shader-source symbol coverage in `wgsl_dot_shader_helpers_present`: `@workgroup_size(64)` / `var<workgroup> partials` / `workgroupBarrier`.

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 26/26 fix128 tests pass, 146/146 physics-solver tests pass (local)
- **Platform matrix CI** — pending (Metal / Vulkan lavapipe / DX12 WARP)

### Roadmap for v0.7.1+

- Multi-workgroup dot for `N ≫ 4096` (Phase 3: cross-workgroup index-ordered final serial via second dispatch)
- `criterion` benchmark for measured speedup on Apple M3 Metal
- Real hardware CI (self-hosted runner or GitHub-hosted GPU tier)

### Backwards compatibility

- Fully backwards compatible with v0.6.x (public API unchanged; internal algorithm shift only)
- Existing callers of `Fix128WgpuKernel::dot` / `Fix128GpuKernel::dot` see identical output values, just faster on `N ≥ 64`

## [0.6.1] - 2026-07-04

### Added

- **Platform matrix CI** — new `fix128-gpu-matrix` job in `.github/workflows/ci.yml`
  - Runs on `macos-latest` (Metal) / `ubuntu-latest` (Vulkan via lavapipe / Mesa) / `windows-latest` (DX12 WARP)
  - Three test tiers per platform:
    1. CPU reference (11 `fix128_gpu_*` fixture tests — always run, no GPU dependency)
    2. Shader source coverage (`wgsl_*` presence + naga compile checks)
    3. Full GPU dispatch (`wgpu_*` bit-exact golden — self-skip on no-adapter, exercise the full pipeline when an adapter is exposed)
  - Ubuntu step installs `mesa-vulkan-drivers` / `vulkan-tools` / `libvulkan1` so lavapipe can serve a software Vulkan adapter
  - Toolchain pinned to `1.92.0` (matches `rust-toolchain.toml`)
  - Cargo registry + `target/` cache keyed on `hashFiles('Cargo.toml')`

### Backwards compatibility

- Fully backwards compatible with v0.6.0
- No public API changes; CI-only patch release

## [0.6.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU dot product (index-ordered serial reduction)
  - `FIX128_DOT_WGSL::fix128_dot_main` @compute entry point (`@workgroup_size(1)`) — single-thread `for i in 0..N { acc = acc + a[i] * b[i] }`
  - Self-contained shader: schoolbook helpers (`umul_wide` / `u64_add` / `u64_mul_wide`) + inline `fix128_add_kernel` + `fix128_mul_kernel`
  - **Determinism contract**: two's-complement 128-bit addition is not associative under wraparound, so the reduction must preserve canonical index order. No `subgroup{Add,...}`, no `atomicAdd`, no parallel tree reduction — the accumulate is strictly serial per §1 経路 3.
  - `Fix128WgpuKernel::dot(&self, a, b) -> Fix128Gpu` — real-GPU dispatch, returns `ZERO` for empty inputs
  - `Fix128GpuKernel::dot` trait method now routes to the live pipeline (previously `unimplemented!()`)
  - 4 new tests:
    - `wgsl_dot_shader_helpers_present` — shader source symbol coverage
    - `wgsl_dot_shader_compiles` — real-GPU compile validity via naga
    - `wgpu_dot_matches_cpu_golden` — 3 fixtures (single-element / 4 positive integers Σ=100 / mixed-sign Σ=19) byte-for-byte equal to `for i { acc = acc.add(a[i].mul(b[i])) }`
    - `wgpu_trait_dot_matches_inherent_dot` — trait routing equivalence

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 24/24 fix128 tests pass, 144/144 physics-solver tests pass

### Roadmap for v0.6.1+

- Platform matrix CI (Metal / Vulkan / DX12 golden agreement across all four Fix128 GPU ops)
- High-throughput blocked dot with index-ordered final serial accumulate (profile-driven)

### Backwards compatibility

- Fully backwards compatible with v0.5.x
- No breaking API changes; `dot` was previously `unimplemented!()` on the GPU path
- With this release the `Fix128GpuKernel` trait has zero `unimplemented!()` methods — the full add / sub / mul / dot surface is live on the GPU

## [0.5.0] - 2026-07-04

### Added

- **`--features fix128-arithmetic`** — Fix128 GPU mul full pipeline
  - `FIX128_MUL_WGSL::fix128_mul_main` @compute entry point — signed 128×128 → middle 128 bits
    - 4 `u64 × u64 → u128` partial products (`ll` / `lh` / `hl` / `hh`) via the schoolbook helpers shipped in v0.4.2
    - Full carry propagation across positions 2..5 of the 256-bit intermediate
    - Two's-complement sign correction: subtracts `b_lo` from `(P[4], P[5])` when `a` is negative, `a_lo` when `b` is negative (matches `alice_physics::math::Fix128::mul`)
  - `Fix128WgpuKernel::mul` — real-GPU dispatch method paired with the shader
  - `Fix128GpuKernel::mul` trait method now routes to the live pipeline (previously `unimplemented!()`)
  - 2 new tests:
    - `wgpu_mul_matches_cpu_golden` — 4 fixtures (identity / integer / negative / fractional 0.5×0.5=0.25) byte-for-byte equal to `Fix128Gpu::mul`
    - `wgpu_trait_mul_matches_inherent_mul` — trait routing equivalence

### Verified on

- **macOS Apple Silicon (M3, Metal)** — 20/20 fix128 tests pass, 140/140 physics-solver tests pass

### Roadmap for v0.5.1+

- `FIX128_DOT_WGSL` — index-ordered accumulate with `workgroupBarrier` sync (no subgroup reduce, keeps determinism contract §1 経路 3)
- Platform matrix CI (Metal / Vulkan / DX12 golden agreement)

### Backwards compatibility

- Fully backwards compatible with v0.4.x
- No breaking API changes; `mul` was previously `unimplemented!()` on the GPU path

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
