//! CUDA backend: Native NVIDIA GPU inference via BitNet kernels.
//!
//! This module provides Rust bindings to the CUDA/C++ ternary GEMM kernels
//! in `csrc/`. It is only available when compiled with `--features cuda`.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Rust: src/cuda/engine.rs                       │
//! │  CudaTernaryEngine (safe API, RAII)             │
//! ├─────────────────────────────────────────────────┤
//! │  Rust: src/cuda/ffi.rs                          │
//! │  extern "C" { alice_trt_* }                     │
//! ├─────────────────────────────────────────────────┤
//! │  C/CUDA: csrc/kernel/bitnet_kernel.cu           │
//! │  Kernel 1: wmma (Tensor Core, 2bit→FP16)        │
//! │  Kernel 2: cuda (CUDA Core, branchless FP32)    │
//! │  C API: alice_trt_init/gemm/relu/shutdown        │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! alice-trt = { path = "../ALICE-TRT", features = ["cuda"] }
//! ```
//!
//! ```no_run
//! use alice_trt::cuda::CudaTernaryEngine;
//!
//! let engine = CudaTernaryEngine::init(0).unwrap();
//! println!("{engine}");
//! ```

pub mod ffi;
pub mod engine;

pub use engine::{CudaTernaryEngine, CudaError, CudaResult, DeviceBuffer, DeviceWeights};
