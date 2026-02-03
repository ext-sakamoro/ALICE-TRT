//! Inference Engine: Multi-Layer GPU Pipeline
//!
//! Stack layers, run forward pass. All computation stays on GPU
//! until you call download() on the final output.

use crate::device::GpuDevice;
use crate::pipeline::TernaryCompute;
use crate::tensor::GpuTensor;
use crate::weights::GpuTernaryWeight;

/// Activation function applied after each layer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// No activation (identity)
    None,
    /// ReLU: max(0, x)
    ReLU,
}

/// A single inference layer (weights + optional activation)
pub struct GpuLayer {
    weights: GpuTernaryWeight,
    activation: Activation,
}

impl GpuLayer {
    /// Create layer with activation
    pub fn new(weights: GpuTernaryWeight, activation: Activation) -> Self {
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
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer
    pub fn add_layer(&mut self, weights: GpuTernaryWeight, activation: Activation) {
        self.layers.push(GpuLayer::new(weights, activation));
    }

    /// Number of layers
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run forward pass through all layers
    ///
    /// Input shape: [in_features] (single vector)
    /// Output shape: [last_layer_out_features]
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
    /// Input shape: [batch_size, in_features]
    /// Output shape: [batch_size, last_layer_out_features]
    pub fn forward_batch(
        &self,
        device: &GpuDevice,
        compute: &TernaryCompute,
        input: &GpuTensor,
        batch_size: usize,
    ) -> GpuTensor {
        assert!(!self.layers.is_empty(), "Engine has no layers");

        let mut current =
            compute.matmul_batch(device, input, &self.layers[0].weights, batch_size);

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

    /// Total VRAM usage for all layer weights
    pub fn total_weight_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.weights.memory_bytes()).sum()
    }

    /// Equivalent FP32 weight size (for comparison)
    pub fn equivalent_fp32_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.out_features() * l.weights.in_features() * 4)
            .sum()
    }

    /// Overall compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let fp32 = self.equivalent_fp32_bytes();
        let actual = self.total_weight_bytes();
        if actual == 0 {
            0.0
        } else {
            fp32 as f32 / actual as f32
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
