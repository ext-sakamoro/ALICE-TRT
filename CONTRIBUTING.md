# Contributing to ALICE-TRT

## Build

```bash
cargo build
cargo build --features cuda
```

## Test

```bash
cargo test
```

Note: Most tests require a GPU. They gracefully skip when no GPU is available.

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **Trojan Horse strategy**: GPU sees normal FP32 FMA instructions, but weights are {-1, 0, +1} expanded from 2-bit bitplanes in registers.
- **8x bandwidth reduction**: 2-bit bitplane vs FP16 weights, decompression costs ~2 cycles, bandwidth savings buy ~100 cycles.
- **Cross-platform GPU**: wgpu backend — Metal (macOS), Vulkan (Linux), DX12 (Windows).
- **Auto kernel selection**: tiled kernel for large layers (in_features >= 1024), simple for small.
- **Zero CPU roundtrips**: all intermediate data stays in VRAM during multi-layer inference.
- **ALICE-ML compatible**: direct import from `TernaryWeightKernel` and `TernaryWeight` (packed 2-bit).
