# Changelog

All notable changes to ALICE-TRT will be documented in this file.

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
