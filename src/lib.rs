//! ALICE-TRT: GPU Ternary Inference Engine (Trojan Horse Edition)
//!
//! > "Don't move data. Move the law."
//! > "The GPU thinks it's computing FP32. It's computing {-1, 0, +1}."
//!
//! A GPU inference engine that exploits memory bandwidth compression:
//! - **Weights**: 1.58-bit ternary {-1, 0, +1} stored as 2-bit bitplanes in VRAM
//! - **Decompression**: On-the-fly in compute shader (register-level, zero latency)
//! - **Computation**: GPU's FMA units process `x * weight` at full throughput
//! - **The Trick**: weight is always {-1, 0, +1}, so FMA degenerates to add/sub/nop
//!
//! # The Trojan Horse Strategy
//!
//! ```text
//! Traditional GPU Inference:
//!   VRAM ──[FP16 weights]──▶ L2 ──▶ L1 ──▶ Registers ──▶ FMA
//!   Bandwidth: 16 bits/weight (bottleneck!)
//!
//! ALICE-TRT:
//!   VRAM ──[2-bit packed]──▶ L2 ──▶ L1 ──▶ Registers ──[expand]──▶ FMA
//!   Bandwidth: 2 bits/weight (8x reduction!)
//!   The expand step costs ~2 cycles. The bandwidth savings buy ~100 cycles.
//! ```
//!
//! # Features
//!
//! - **8x bandwidth reduction** vs FP16, 4x vs INT8
//! - **Cross-platform GPU**: Metal (macOS), Vulkan (Linux), DX12 (Windows)
//! - **Auto kernel selection**: Tiled for large layers, simple for small
//! - **Multi-layer inference**: All computation stays in VRAM
//! - **ALICE-ML compatible**: Direct import of ternary weights
//!
//! # Example
//!
//! ```no_run
//! use alice_trt::prelude::*;
//!
//! // 1. Initialize GPU
//! let device = GpuDevice::new().unwrap();
//! let compute = TernaryCompute::new(&device);
//!
//! // 2. Upload ternary weights (2-bit in VRAM)
//! let weights = GpuTernaryWeight::from_ternary(
//!     &device, &[1, -1, 0, 1], 2, 2
//! );
//!
//! // 3. Upload input
//! let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);
//!
//! // 4. GPU inference (stays in VRAM)
//! let output = compute.matvec(&device, &input, &weights);
//!
//! // 5. Download result
//! let result = output.download(&device);
//! // result ≈ [-1.0, 3.0]
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                  ALICE-TRT Trojan Horse Edition                     │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌──────────┐    ┌──────────────────┐    ┌──────────┐              │
//! │  │ GpuDevice │───▶│ TernaryCompute   │───▶│ GpuTensor│              │
//! │  │ (wgpu)   │    │ (4 pipelines)    │    │ (output) │              │
//! │  └──────────┘    └──────────────────┘    └──────────┘              │
//! │       │                   │                    │                    │
//! │       ▼                   ▼                    ▼                    │
//! │  ┌───────────────────────────────────────────────────────────┐     │
//! │  │           GpuTernaryWeight (2-bit bitplanes)               │     │
//! │  │  plus_bits: u32[]  │  minus_bits: u32[]  │  scale: f32    │     │
//! │  │  32 weights per u32 word. 8x smaller than FP16.           │     │
//! │  └───────────────────────────────────────────────────────────┘     │
//! │                          │                                         │
//! │                          ▼                                         │
//! │  ┌───────────────────────────────────────────────────────────┐     │
//! │  │              WGSL Compute Shaders (GPU-side)               │     │
//! │  │  1. Load 2-bit packed weights from VRAM                    │     │
//! │  │  2. Expand to {-1.0, 0.0, +1.0} in registers (2 cycles)   │     │
//! │  │  3. FMA: acc += input[col] * expanded_weight               │     │
//! │  │  4. GPU sees normal FP32 math. Trojan Horse complete.      │     │
//! │  └───────────────────────────────────────────────────────────┘     │
//! │                          │                                         │
//! │                          ▼                                         │
//! │  ┌───────────────────────────────────────────────────────────┐     │
//! │  │              GpuInferenceEngine (multi-layer)              │     │
//! │  │  Layer 1: TernaryMatVec → ReLU                             │     │
//! │  │  Layer 2: TernaryMatVec → ReLU                             │     │
//! │  │  Layer N: TernaryMatVec → None                             │     │
//! │  │  All intermediate data stays in VRAM. Zero CPU roundtrips. │     │
//! │  └───────────────────────────────────────────────────────────┘     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always
)]

pub mod device;
pub mod inference;
pub mod kernel;
pub mod pipeline;
pub mod profiler;
pub mod tensor;
pub mod weights;

#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "db")]
pub mod db_bridge;
#[cfg(feature = "physics")]
pub mod physics_bridge;
#[cfg(feature = "sdf")]
pub mod sdf_bridge;
#[cfg(feature = "view")]
pub mod view_bridge;
#[cfg(feature = "voice")]
pub mod voice_bridge;

// ============================================================================
// Core Re-exports
// ============================================================================

pub use device::GpuDevice;
pub use inference::{Activation, GpuInferenceEngine, GpuLayer};
pub use kernel::{GpuParams, ReluParams};
pub use pipeline::TernaryCompute;
pub use profiler::{InferenceProfile, InferenceProfiler, LayerProfile};
pub use tensor::GpuTensor;
pub use weights::GpuTernaryWeight;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// Prelude
// ============================================================================

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::Activation;
    pub use crate::GpuDevice;
    pub use crate::GpuInferenceEngine;
    pub use crate::GpuTensor;
    pub use crate::GpuTernaryWeight;
    pub use crate::TernaryCompute;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_gpu_init() {
        // GPU may not be available in all test environments
        if let Ok(device) = GpuDevice::new() {
            assert!(!device.info().is_empty());
        } else {
            // No GPU available, skip
        }
    }

    #[test]
    fn test_simple_matvec() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        // W = [[1, -1], [0, 1]]
        // x = [2.0, 3.0]
        // y[0] = 2 - 3 = -1
        // y[1] = 0 + 3 = 3
        let weights = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);

        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert_eq!(result.len(), 2);
        assert!((result[0] - (-1.0)).abs() < 1e-4, "got {}", result[0]);
        assert!((result[1] - 3.0).abs() < 1e-4, "got {}", result[1]);
    }

    #[test]
    fn test_all_plus() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        let weights = GpuTernaryWeight::from_ternary(&device, &[1, 1, 1, 1], 1, 4);
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);

        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_all_minus() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        let weights = GpuTernaryWeight::from_ternary(&device, &[-1, -1, -1, -1], 1, 4);
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);

        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0] - (-10.0)).abs() < 1e-4);
    }

    #[test]
    fn test_batch_matmul() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        // W = [[1, -1], [1, 1]]
        let weights = GpuTernaryWeight::from_ternary(&device, &[1, -1, 1, 1], 2, 2);

        // batch of 2: [[1, 2], [3, 4]]
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let output = compute.matmul_batch(&device, &input, &weights, 2);
        let result = output.download(&device);

        assert_eq!(result.len(), 4);
        // batch 0: [1-2, 1+2] = [-1, 3]
        assert!((result[0] - (-1.0)).abs() < 1e-4);
        assert!((result[1] - 3.0).abs() < 1e-4);
        // batch 1: [3-4, 3+4] = [-1, 7]
        assert!((result[2] - (-1.0)).abs() < 1e-4);
        assert!((result[3] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_relu() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        let tensor = GpuTensor::from_f32(&device, &[-1.0, 0.0, 1.0, -2.0, 3.0], &[5]);
        compute.relu_inplace(&device, &tensor);

        let result = tensor.download(&device);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
        assert!((result[3] - 0.0).abs() < 1e-6);
        assert!((result[4] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_inference_engine() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        // 2-layer network: 3 -> 2 -> 2
        let w1 = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1, -1, 1], 2, 3);
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1, 1, -1, 1], 2, 2);

        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w1, Activation::ReLU);
        engine.add_layer(w2, Activation::None);

        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0], &[3]);
        let output = engine.forward(&device, &compute, &input);
        let result = output.download(&device);

        assert_eq!(result.len(), 2);
        // Layer 1: [1,-1,0]·[1,2,3] = -1 → ReLU → 0
        //          [1,-1,1]·[1,2,3] = 2  → ReLU → 2
        // Layer 2: [1,1]·[0,2] = 2
        //          [-1,1]·[0,2] = 2
        assert!((result[0] - 2.0).abs() < 1e-4, "got {}", result[0]);
        assert!((result[1] - 2.0).abs() < 1e-4, "got {}", result[1]);
    }

    #[test]
    fn test_from_alice_ml_kernel() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        // Create via ALICE-ML's TernaryWeightKernel
        let cpu_kernel = alice_ml::TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let gpu_weights = GpuTernaryWeight::from_kernel(&device, &cpu_kernel);

        let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);
        let output = compute.matvec(&device, &input, &gpu_weights);
        let result = output.download(&device);

        assert!((result[0] - (-1.0)).abs() < 1e-4);
        assert!((result[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_from_alice_ml_packed() {
        let Ok(device) = GpuDevice::new() else { return };

        let compute = TernaryCompute::new(&device);

        // Create via ALICE-ML's TernaryWeight (packed 2-bit)
        let cpu_packed = alice_ml::TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        let gpu_weights = GpuTernaryWeight::from_packed(&device, &cpu_packed);

        let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);
        let output = compute.matvec(&device, &input, &gpu_weights);
        let result = output.download(&device);

        assert!((result[0] - (-1.0)).abs() < 1e-4);
        assert!((result[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_compression_ratio() {
        let Ok(device) = GpuDevice::new() else { return };

        let n = 1024;
        let values: Vec<i8> = (0..n * n).map(|i| (i % 3) as i8 - 1).collect();
        let weights = GpuTernaryWeight::from_ternary(&device, &values, n, n);

        let ratio = weights.compression_ratio();
        // Bitplane: 2 * (n * ceil(n/32) * 4) = 2 * 1024 * 32 * 4 = 262144
        // FP32: 1024 * 1024 * 4 = 4194304
        // Ratio: 4194304 / 262144 = 16x
        assert!(ratio > 15.0 && ratio <= 16.0, "ratio = {ratio}");
    }

    #[test]
    fn test_all_zeros_weights() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        // All-zero weight matrix: output should be all zeros
        let weights = GpuTernaryWeight::from_ternary(&device, &[0, 0, 0, 0], 2, 2);
        let input = GpuTensor::from_f32(&device, &[5.0, 7.0], &[2]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0]).abs() < 1e-4, "expected 0, got {}", result[0]);
        assert!((result[1]).abs() < 1e-4, "expected 0, got {}", result[1]);
    }

    #[test]
    fn test_identity_like_matvec() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        // "Identity-like" ternary: W = [[1,0,0],[0,1,0],[0,0,1]]
        let weights = GpuTernaryWeight::from_ternary(&device, &[1, 0, 0, 0, 1, 0, 0, 0, 1], 3, 3);
        let input = GpuTensor::from_f32(&device, &[3.0, 5.0, 7.0], &[3]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0] - 3.0).abs() < 1e-4);
        assert!((result[1] - 5.0).abs() < 1e-4);
        assert!((result[2] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_scaled_weights() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        // W = [[1, 1]] with scale = 2.0
        // y = (1*3 + 1*5) * 2.0 = 16.0
        let weights = GpuTernaryWeight::from_ternary_scaled(&device, &[1, 1], 1, 2, 2.0);
        let input = GpuTensor::from_f32(&device, &[3.0, 5.0], &[2]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0] - 16.0).abs() < 1e-3, "got {}", result[0]);
    }

    #[test]
    fn test_weight_accessors() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1; 12], 3, 4);
        assert_eq!(w.out_features(), 3);
        assert_eq!(w.in_features(), 4);
        assert!((w.scale() - 1.0).abs() < 1e-6);
        assert_eq!(w.words_per_row(), 1); // ceil(4/32) = 1
    }

    #[test]
    fn test_weight_display() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        let s = format!("{w}");
        assert!(s.contains("GpuTernaryWeight"));
        assert!(s.contains("2x2"));
        assert!(s.contains("compression"));
    }

    #[test]
    fn test_single_element_matvec() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        // 1x1 matrix: W = [[1]], input = [42.0], output = [42.0]
        let weights = GpuTernaryWeight::from_ternary(&device, &[1], 1, 1);
        let input = GpuTensor::from_f32(&device, &[42.0], &[1]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0] - 42.0).abs() < 1e-4);
    }

    #[test]
    fn test_negative_input_values() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        // W = [[1, -1]], input = [-3.0, -5.0]
        // y = 1*(-3) + (-1)*(-5) = -3 + 5 = 2
        let weights = GpuTernaryWeight::from_ternary(&device, &[1, -1], 1, 2);
        let input = GpuTensor::from_f32(&device, &[-3.0, -5.0], &[2]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);

        assert!((result[0] - 2.0).abs() < 1e-4, "got {}", result[0]);
    }

    #[test]
    fn test_relu_all_positive() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        let tensor = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 100.0], &[4]);
        compute.relu_inplace(&device, &tensor);
        let result = tensor.download(&device);

        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_all_negative() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        let tensor = GpuTensor::from_f32(&device, &[-1.0, -2.0, -100.0], &[3]);
        compute.relu_inplace(&device, &tensor);
        let result = tensor.download(&device);

        for (i, &v) in result.iter().enumerate() {
            assert!((v).abs() < 1e-6, "result[{i}] = {v} should be 0");
        }
    }

    #[test]
    fn test_prelude_exports() {
        // Verify prelude contains expected types (compile-time check)
        fn check_prelude() {
            use crate::prelude::*;
            let _ = std::any::type_name::<GpuDevice>();
            let _ = std::any::type_name::<GpuTernaryWeight>();
            let _ = std::any::type_name::<GpuTensor>();
            let _ = std::any::type_name::<TernaryCompute>();
            let _ = std::any::type_name::<GpuInferenceEngine>();
            let _ = std::any::type_name::<Activation>();
        }
        check_prelude();
    }

    #[test]
    fn test_device_display() {
        let Ok(device) = GpuDevice::new() else { return };
        let s = format!("{device}");
        assert!(s.contains("ALICE-TRT"));
    }

    #[test]
    fn test_compute_display() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        let s = format!("{compute}");
        assert!(s.contains("TernaryCompute"));
        assert!(s.contains("4 pipelines"));
    }

    // -----------------------------------------------------------------------
    // Tiled kernel path (in_features >= 1024 triggers matvec_tiled)
    // -----------------------------------------------------------------------

    #[test]
    fn test_tiled_kernel_1024_cols() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // 1 output row, 1024 inputs: all +1 weights, all 1.0 inputs => sum = 1024
        let values: Vec<i8> = vec![1; 1024];
        let input_data: Vec<f32> = vec![1.0; 1024];
        let weights = GpuTernaryWeight::from_ternary(&device, &values, 1, 1024);
        let input = GpuTensor::from_f32(&device, &input_data, &[1024]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1024.0).abs() < 1.0, "got {}", result[0]);
    }

    #[test]
    fn test_tiled_kernel_minus_weights() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // 1 output row, 1024 inputs: all -1 weights, all 1.0 inputs => sum = -1024
        let values: Vec<i8> = vec![-1; 1024];
        let input_data: Vec<f32> = vec![1.0; 1024];
        let weights = GpuTernaryWeight::from_ternary(&device, &values, 1, 1024);
        let input = GpuTensor::from_f32(&device, &input_data, &[1024]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);
        assert!((result[0] - (-1024.0)).abs() < 1.0, "got {}", result[0]);
    }

    #[test]
    fn test_matvec_4x4() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // W = identity-like 4x4
        #[rustfmt::skip]
        let w_vals: &[i8] = &[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ];
        let weights = GpuTernaryWeight::from_ternary(&device, w_vals, 4, 4);
        let input = GpuTensor::from_f32(&device, &[10.0, 20.0, 30.0, 40.0], &[4]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);
        assert!((result[0] - 10.0).abs() < 1e-4);
        assert!((result[1] - 20.0).abs() < 1e-4);
        assert!((result[2] - 30.0).abs() < 1e-4);
        assert!((result[3] - 40.0).abs() < 1e-4);
    }

    #[test]
    fn test_matvec_mixed_ternary_values() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // W = [[1, 0, -1]] (1 row, 3 cols)
        // x = [5.0, 9.0, 3.0]
        // y = 5 + 0 - 3 = 2
        let weights = GpuTernaryWeight::from_ternary(&device, &[1, 0, -1], 1, 3);
        let input = GpuTensor::from_f32(&device, &[5.0, 9.0, 3.0], &[3]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);
        assert!((result[0] - 2.0).abs() < 1e-4, "got {}", result[0]);
    }

    #[test]
    fn test_batch_size_1_equals_matvec() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // batch_size=1 matmul must equal matvec result
        let w_vals: &[i8] = &[1, -1, 0, 1];
        let weights = GpuTernaryWeight::from_ternary(&device, w_vals, 2, 2);
        let input_mv = GpuTensor::from_f32(&device, &[3.0, 7.0], &[2]);
        let input_mm = GpuTensor::from_f32(&device, &[3.0, 7.0], &[1, 2]);
        let out_mv = compute
            .matvec(&device, &input_mv, &weights)
            .download(&device);
        let out_mm = compute
            .matmul_batch(&device, &input_mm, &weights, 1)
            .download(&device);
        assert_eq!(out_mv.len(), out_mm.len());
        for (a, b) in out_mv.iter().zip(out_mm.iter()) {
            assert!((a - b).abs() < 1e-4, "matvec={a} matmul_batch={b}");
        }
    }

    #[test]
    fn test_scaled_weights_half() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // W = [[1, 1]] scale=0.5 => (3+5)*0.5 = 4.0
        let weights = GpuTernaryWeight::from_ternary_scaled(&device, &[1, 1], 1, 2, 0.5);
        let input = GpuTensor::from_f32(&device, &[3.0, 5.0], &[2]);
        let output = compute.matvec(&device, &input, &weights);
        let result = output.download(&device);
        assert!((result[0] - 4.0).abs() < 1e-3, "got {}", result[0]);
    }

    #[test]
    fn test_single_layer_none_activation_engine() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // Negative result must pass through unchanged with None activation
        let w = GpuTernaryWeight::from_ternary(&device, &[-1], 1, 1);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::None);
        let input = GpuTensor::from_f32(&device, &[7.0], &[1]);
        let output = engine.forward(&device, &compute, &input);
        let result = output.download(&device);
        assert!((result[0] - (-7.0)).abs() < 1e-4, "got {}", result[0]);
    }

    #[test]
    fn test_engine_display_shows_layer_count() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        for _ in 0..3 {
            let w = GpuTernaryWeight::from_ternary(&device, &[1, 0, -1, 1], 2, 2);
            engine.add_layer(w, Activation::None);
        }
        let s = format!("{engine}");
        assert!(s.contains("3 layers"), "display = {s}");
    }

    #[test]
    fn test_weight_memory_bytes_matches_formula() {
        let Ok(device) = GpuDevice::new() else { return };
        // 4 rows, 8 cols: words_per_row = ceil(8/32) = 1
        // memory = 4 * 1 * 4 * 2 = 32 bytes
        let w = GpuTernaryWeight::from_ternary(&device, &[1; 32], 4, 8);
        assert_eq!(w.memory_bytes(), 32);
        assert_eq!(w.words_per_row(), 1);
    }

    #[test]
    fn test_relu_single_element_positive() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        let tensor = GpuTensor::from_f32(&device, &[5.5], &[1]);
        compute.relu_inplace(&device, &tensor);
        let result = tensor.download(&device);
        assert!((result[0] - 5.5).abs() < 1e-6);
    }

    #[test]
    fn test_relu_single_element_zero() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        let tensor = GpuTensor::from_f32(&device, &[0.0], &[1]);
        compute.relu_inplace(&device, &tensor);
        let result = tensor.download(&device);
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_words_per_row_64_cols() {
        let Ok(device) = GpuDevice::new() else { return };
        // 64 cols => ceil(64/32) = 2 words
        let w = GpuTernaryWeight::from_ternary(&device, &[1; 64], 1, 64);
        assert_eq!(w.words_per_row(), 2);
    }

    #[test]
    fn test_compression_ratio_large_matrix() {
        let Ok(device) = GpuDevice::new() else { return };
        // 512 rows x 512 cols: words_per_row = 16
        // fp32  = 512 * 512 * 4 = 1_048_576 bytes
        // bits  = 512 * 16 * 4 * 2 = 65_536 bytes
        // ratio = 1_048_576 / 65_536 = 16.0
        let values: Vec<i8> = vec![1; 512 * 512];
        let w = GpuTernaryWeight::from_ternary(&device, &values, 512, 512);
        let ratio = w.compression_ratio();
        assert!((ratio - 16.0).abs() < 0.1, "ratio = {ratio}");
    }

    #[test]
    fn test_forward_batch_two_layer_engine() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // 2-layer: 2->2->1, batch=2
        // W1 = [[1,1],[-1,1]], ReLU
        // W2 = [[1,1]],        None
        let w1 = GpuTernaryWeight::from_ternary(&device, &[1, 1, -1, 1], 2, 2);
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1, 1], 1, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w1, Activation::ReLU);
        engine.add_layer(w2, Activation::None);
        // batch[0] = [1,2], batch[1] = [3,4]
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let output = engine.forward_batch(&device, &compute, &input, 2);
        let result = output.download(&device);
        assert_eq!(result.len(), 2);
        // batch[0]: L1=[1+2,-1+2]=[3,1]->ReLU->[3,1]; L2=3+1=4
        assert!((result[0] - 4.0).abs() < 1e-4, "batch0 = {}", result[0]);
        // batch[1]: L1=[3+4,-3+4]=[7,1]->ReLU->[7,1]; L2=7+1=8
        assert!((result[1] - 8.0).abs() < 1e-4, "batch1 = {}", result[1]);
    }
}
