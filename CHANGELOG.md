# Changelog

All notable changes to ALICE-TRT will be documented in this file.

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
