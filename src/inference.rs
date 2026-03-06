//! Inference Engine: Multi-Layer GPU Pipeline
//!
//! Stack layers, run forward pass. All computation stays on GPU
//! until you call `download()` on the final output.

use crate::device::GpuDevice;
use crate::pipeline::TernaryCompute;
use crate::tensor::GpuTensor;
use crate::weights::GpuTernaryWeight;

/// Activation function applied after each layer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// No activation (identity)
    None,
    /// `ReLU`: max(0, x)
    ReLU,
}

/// A single inference layer (weights + optional activation)
pub struct GpuLayer {
    weights: GpuTernaryWeight,
    activation: Activation,
}

impl GpuLayer {
    /// Create layer with activation
    pub const fn new(weights: GpuTernaryWeight, activation: Activation) -> Self {
        Self {
            weights,
            activation,
        }
    }
}

/// Multi-layer GPU inference engine
///
/// All intermediate computations stay in VRAM.
/// Only the final output is downloaded to CPU.
///
/// # Example
///
/// ```no_run
/// use alice_trt::*;
///
/// let device = GpuDevice::new().unwrap();
/// let compute = TernaryCompute::new(&device);
/// let mut engine = GpuInferenceEngine::new();
///
/// // Add layers
/// let w1 = GpuTernaryWeight::from_ternary(&device, &[1,-1,0,1], 2, 2);
/// let w2 = GpuTernaryWeight::from_ternary(&device, &[1,1,-1,0], 2, 2);
/// engine.add_layer(w1, Activation::ReLU);
/// engine.add_layer(w2, Activation::None);
///
/// // Forward pass
/// let input = GpuTensor::from_f32(&device, &[1.0, 2.0], &[2]);
/// let output = engine.forward(&device, &compute, &input);
/// let result = output.download(&device);
/// ```
pub struct GpuInferenceEngine {
    layers: Vec<GpuLayer>,
}

impl GpuInferenceEngine {
    /// Create empty engine
    #[must_use]
    pub const fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer
    pub fn add_layer(&mut self, weights: GpuTernaryWeight, activation: Activation) {
        self.layers.push(GpuLayer::new(weights, activation));
    }

    /// Number of layers
    #[inline(always)]
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run forward pass through all layers
    ///
    /// Input shape: `[in_features]` (single vector)
    /// Output shape: `[last_layer_out_features]`
    ///
    /// # Panics
    ///
    /// Panics if the engine has no layers.
    pub fn forward(
        &self,
        device: &GpuDevice,
        compute: &TernaryCompute,
        input: &GpuTensor,
    ) -> GpuTensor {
        assert!(!self.layers.is_empty(), "Engine has no layers");

        let mut current = compute.matvec(device, input, &self.layers[0].weights);

        if self.layers[0].activation == Activation::ReLU {
            compute.relu_inplace(device, &current);
        }

        for layer in &self.layers[1..] {
            let next = compute.matvec(device, &current, &layer.weights);

            if layer.activation == Activation::ReLU {
                compute.relu_inplace(device, &next);
            }

            current = next;
        }

        // Ensure all GPU work completes
        device.poll_wait();

        current
    }

    /// Run batched forward pass
    ///
    /// Input shape: [`batch_size`, `in_features`]
    /// Output shape: [`batch_size`, `last_layer_out_features`]
    ///
    /// # Panics
    ///
    /// Panics if the engine has no layers.
    pub fn forward_batch(
        &self,
        device: &GpuDevice,
        compute: &TernaryCompute,
        input: &GpuTensor,
        batch_size: usize,
    ) -> GpuTensor {
        assert!(!self.layers.is_empty(), "Engine has no layers");

        let mut current = compute.matmul_batch(device, input, &self.layers[0].weights, batch_size);

        if self.layers[0].activation == Activation::ReLU {
            compute.relu_inplace(device, &current);
        }

        for layer in &self.layers[1..] {
            let next = compute.matmul_batch(device, &current, &layer.weights, batch_size);

            if layer.activation == Activation::ReLU {
                compute.relu_inplace(device, &next);
            }

            current = next;
        }

        device.poll_wait();

        current
    }

    /// レイヤーの重みと活性化関数の参照リストを返す (プロファイラ用)。
    #[must_use]
    pub fn layer_info(&self) -> Vec<(&GpuTernaryWeight, Activation)> {
        self.layers
            .iter()
            .map(|l| (&l.weights, l.activation))
            .collect()
    }

    /// Total VRAM usage for all layer weights
    #[must_use]
    pub fn total_weight_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.weights.memory_bytes()).sum()
    }

    /// Equivalent FP32 weight size (for comparison)
    #[must_use]
    pub fn equivalent_fp32_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.out_features() * l.weights.in_features() * 4)
            .sum()
    }

    /// Overall compression ratio
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let fp32 = self.equivalent_fp32_bytes();
        let actual = self.total_weight_bytes();
        if actual == 0 {
            0.0
        } else {
            let inv_actual = 1.0 / actual as f32;
            fp32 as f32 * inv_actual
        }
    }
}

impl Default for GpuInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for GpuInferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuInferenceEngine[{} layers, {} bytes VRAM, {:.1}x compression]",
            self.layers.len(),
            self.total_weight_bytes(),
            self.compression_ratio(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_default() {
        let engine = GpuInferenceEngine::default();
        assert_eq!(engine.num_layers(), 0);
        assert_eq!(engine.total_weight_bytes(), 0);
        assert_eq!(engine.equivalent_fp32_bytes(), 0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_engine_compression_ratio_empty() {
        let engine = GpuInferenceEngine::new();
        assert_eq!(engine.compression_ratio(), 0.0);
    }

    #[test]
    fn test_activation_debug_and_eq() {
        assert_eq!(Activation::ReLU, Activation::ReLU);
        assert_eq!(Activation::None, Activation::None);
        assert_ne!(Activation::ReLU, Activation::None);
        let dbg = format!("{:?}", Activation::ReLU);
        assert_eq!(dbg, "ReLU");
    }

    #[test]
    fn test_activation_copy() {
        let a = Activation::ReLU;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    #[test]
    fn test_engine_display_empty() {
        let engine = GpuInferenceEngine::new();
        let s = format!("{engine}");
        assert!(s.contains("0 layers"));
        assert!(s.contains("0 bytes"));
    }

    #[test]
    fn test_engine_add_layers_and_count() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        assert_eq!(engine.num_layers(), 0);

        let w1 = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        engine.add_layer(w1, Activation::ReLU);
        assert_eq!(engine.num_layers(), 1);

        let w2 = GpuTernaryWeight::from_ternary(&device, &[1, 0, -1, 1], 2, 2);
        engine.add_layer(w2, Activation::None);
        assert_eq!(engine.num_layers(), 2);
    }

    #[test]
    fn test_engine_weight_bytes() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        let w = GpuTernaryWeight::from_ternary(&device, &[1; 64], 8, 8);
        let single_bytes = w.memory_bytes();
        engine.add_layer(w, Activation::None);
        assert_eq!(engine.total_weight_bytes(), single_bytes);
        assert!(engine.compression_ratio() > 0.0);
    }

    #[test]
    fn test_engine_equivalent_fp32() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        // 4x4 = 16 weights, FP32 = 16 * 4 = 64 bytes
        let w = GpuTernaryWeight::from_ternary(&device, &[1; 16], 4, 4);
        engine.add_layer(w, Activation::None);
        assert_eq!(engine.equivalent_fp32_bytes(), 64);
    }

    #[test]
    #[should_panic(expected = "Engine has no layers")]
    fn test_engine_forward_empty_panics() {
        let Ok(device) = GpuDevice::new() else {
            panic!("Engine has no layers");
        };
        let compute = TernaryCompute::new(&device);
        let engine = GpuInferenceEngine::new();
        let input = GpuTensor::from_f32(&device, &[1.0], &[1]);
        let _out = engine.forward(&device, &compute, &input);
    }

    #[test]
    fn test_engine_display_with_layers() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        let w = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        engine.add_layer(w, Activation::ReLU);
        let s = format!("{engine}");
        assert!(s.contains("1 layers"));
        assert!(s.contains("compression"));
    }

    #[test]
    fn test_engine_forward_batch_single() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // W = [[1, -1], [0, 1]], batch_size = 1
        // input = [2.0, 3.0] => y = [-1.0, 3.0]
        let w = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::None);
        let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[1, 2]);
        let output = engine.forward_batch(&device, &compute, &input, 1);
        let result = output.download(&device);
        assert_eq!(result.len(), 2);
        assert!((result[0] - (-1.0)).abs() < 1e-4, "got {}", result[0]);
        assert!((result[1] - 3.0).abs() < 1e-4, "got {}", result[1]);
    }

    #[test]
    fn test_engine_forward_batch_two_items() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // W = [[1, 1]], 1 output, 2 inputs, batch = 2
        // batch[0] = [1,2] => 3.0
        // batch[1] = [3,4] => 7.0
        let w = GpuTernaryWeight::from_ternary(&device, &[1, 1], 1, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::None);
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let output = engine.forward_batch(&device, &compute, &input, 2);
        let result = output.download(&device);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-4, "got {}", result[0]);
        assert!((result[1] - 7.0).abs() < 1e-4, "got {}", result[1]);
    }

    #[test]
    fn test_engine_three_layers() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // 3-layer: 2->2->2->1
        // L1: [[1,1],[-1,1]], ReLU
        // L2: [[1,-1],[1,1]], ReLU
        // L3: [[1,1]],        None
        let w1 = GpuTernaryWeight::from_ternary(&device, &[1, 1, -1, 1], 2, 2);
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1, -1, 1, 1], 2, 2);
        let w3 = GpuTernaryWeight::from_ternary(&device, &[1, 1], 1, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w1, Activation::ReLU);
        engine.add_layer(w2, Activation::ReLU);
        engine.add_layer(w3, Activation::None);
        assert_eq!(engine.num_layers(), 3);
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0], &[2]);
        let output = engine.forward(&device, &compute, &input);
        let result = output.download(&device);
        assert_eq!(result.len(), 1);
        // L1: [1+2, -1+2] = [3, 1] -> ReLU -> [3, 1]
        // L2: [3-1, 3+1]  = [2, 4] -> ReLU -> [2, 4]
        // L3: 2+4 = 6
        assert!((result[0] - 6.0).abs() < 1e-4, "got {}", result[0]);
    }

    #[test]
    fn test_engine_all_none_activations() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // Two None layers: negative values must not be clamped
        // W1 = [[-1,-1]], W2 = [[1]]
        let w1 = GpuTernaryWeight::from_ternary(&device, &[-1, -1], 1, 2);
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1], 1, 1);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w1, Activation::None);
        engine.add_layer(w2, Activation::None);
        let input = GpuTensor::from_f32(&device, &[1.0, 2.0], &[2]);
        let output = engine.forward(&device, &compute, &input);
        let result = output.download(&device);
        // L1: -(1+2) = -3, no ReLU
        // L2: 1*(-3) = -3
        assert!((result[0] - (-3.0)).abs() < 1e-4, "got {}", result[0]);
    }

    #[test]
    fn test_engine_total_weight_bytes_two_layers() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        let w1 = GpuTernaryWeight::from_ternary(&device, &[1; 4], 2, 2);
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1; 9], 3, 3);
        let bytes1 = w1.memory_bytes();
        let bytes2 = w2.memory_bytes();
        engine.add_layer(w1, Activation::None);
        engine.add_layer(w2, Activation::None);
        assert_eq!(engine.total_weight_bytes(), bytes1 + bytes2);
    }

    #[test]
    fn test_engine_equivalent_fp32_two_layers() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        // Layer 1: 2x2 = 4 weights => 16 bytes FP32
        let w1 = GpuTernaryWeight::from_ternary(&device, &[1; 4], 2, 2);
        // Layer 2: 3x3 = 9 weights => 36 bytes FP32
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1; 9], 3, 3);
        engine.add_layer(w1, Activation::None);
        engine.add_layer(w2, Activation::None);
        assert_eq!(engine.equivalent_fp32_bytes(), 16 + 36);
    }

    #[test]
    fn test_engine_compression_ratio_non_empty() {
        let Ok(device) = GpuDevice::new() else { return };
        let mut engine = GpuInferenceEngine::new();
        // 1 row, 256 cols => high compression
        let values: Vec<i8> = vec![1; 256];
        let w = GpuTernaryWeight::from_ternary(&device, &values, 1, 256);
        engine.add_layer(w, Activation::None);
        let ratio = engine.compression_ratio();
        assert!(ratio > 1.0, "expected ratio > 1, got {ratio}");
    }

    #[test]
    fn test_gpu_layer_new_constructs() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1, 0, -1, 1], 2, 2);
        // GpuLayer::new is pub; ensure it constructs without panic
        let _layer = GpuLayer::new(w, Activation::ReLU);
    }

    #[test]
    fn test_engine_single_layer_relu_clamps_negatives() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);
        // W = [[-1,-1]], input = [2.0, 3.0] => -5 -> ReLU -> 0
        let w = GpuTernaryWeight::from_ternary(&device, &[-1, -1], 1, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::ReLU);
        let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);
        let output = engine.forward(&device, &compute, &input);
        let result = output.download(&device);
        assert!(
            (result[0]).abs() < 1e-4,
            "expected 0 after ReLU, got {}",
            result[0]
        );
    }

    #[test]
    #[should_panic(expected = "Engine has no layers")]
    fn test_engine_forward_batch_empty_panics() {
        let Ok(device) = GpuDevice::new() else {
            panic!("Engine has no layers");
        };
        let compute = TernaryCompute::new(&device);
        let engine = GpuInferenceEngine::new();
        let input = GpuTensor::from_f32(&device, &[1.0], &[1, 1]);
        let _out = engine.forward_batch(&device, &compute, &input, 1);
    }
}
