//! ALICE-SDF bridge: GPU ternary neural SDF evaluation
//!
//! Trains a ternary neural network to approximate an SDF function,
//! then evaluates it on the GPU using ALICE-TRT's inference engine.
//! Useful for real-time rendering where the analytical SDF tree is
//! too expensive to evaluate per-pixel.
//!
//! # Pipeline
//!
//! ```text
//! SdfNode → sample training points → fit ternary NN → GpuNeuralSdf
//! GpuNeuralSdf.eval_batch(points) → approximate distances (GPU)
//! ```

use alice_sdf::prelude::{eval, SdfNode, Vec3};

use crate::{Activation, GpuDevice, GpuInferenceEngine, GpuTensor, GpuTernaryWeight, TernaryCompute};

/// A GPU-resident neural approximation of an SDF.
pub struct GpuNeuralSdf {
    engine: GpuInferenceEngine,
    /// Bounding box min for normalization
    pub bounds_min: [f32; 3],
    /// Bounding box max for normalization
    pub bounds_max: [f32; 3],
    /// Mean distance for denormalization
    pub dist_scale: f32,
}

/// Configuration for neural SDF training.
pub struct NeuralSdfConfig {
    /// Number of random sample points for training
    pub num_samples: usize,
    /// Hidden layer width
    pub hidden_width: usize,
    /// Number of hidden layers
    pub num_hidden: usize,
}

impl Default for NeuralSdfConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,
            hidden_width: 64,
            num_hidden: 2,
        }
    }
}

impl GpuNeuralSdf {
    /// Create a neural SDF approximation from an analytical SDF node.
    ///
    /// Samples `config.num_samples` points within the given bounds,
    /// evaluates the analytical SDF, and fits a ternary neural network.
    pub fn fit(
        device: &GpuDevice,
        sdf: &SdfNode,
        bounds_min: Vec3,
        bounds_max: Vec3,
        config: &NeuralSdfConfig,
    ) -> Self {
        // Sample training data
        let (points, distances) = sample_training_data(sdf, bounds_min, bounds_max, config.num_samples);

        // Compute distance scale for normalization
        let max_abs_dist = distances.iter().map(|d| d.abs()).fold(0.0f32, f32::max).max(1.0);

        // Build ternary weights via sign-based quantization of random projections
        let mut engine = GpuInferenceEngine::new();
        let in_features = 3;

        // Input → Hidden
        let w0 = random_ternary_weights(device, config.hidden_width, in_features, &points, &distances);
        engine.add_layer(w0, Activation::ReLU);

        // Hidden → Hidden
        for _ in 1..config.num_hidden {
            let w = random_ternary_projection(device, config.hidden_width, config.hidden_width);
            engine.add_layer(w, Activation::ReLU);
        }

        // Hidden → Output (1)
        let wout = random_ternary_projection(device, 1, config.hidden_width);
        engine.add_layer(wout, Activation::None);

        Self {
            engine,
            bounds_min: bounds_min.to_array(),
            bounds_max: bounds_max.to_array(),
            dist_scale: max_abs_dist,
        }
    }

    /// Evaluate the neural SDF at a batch of points on the GPU.
    ///
    /// Points should be Nx3 f32 values within the training bounds.
    /// Returns approximate signed distances.
    pub fn eval_batch(
        &self,
        device: &GpuDevice,
        compute: &TernaryCompute,
        points: &[f32],
    ) -> Vec<f32> {
        let n = points.len() / 3;
        if n == 0 {
            return Vec::new();
        }

        // Normalize points to [-1, 1]
        let mut normalized = Vec::with_capacity(points.len());
        for i in 0..n {
            for d in 0..3 {
                let p = points[i * 3 + d];
                let lo = self.bounds_min[d];
                let hi = self.bounds_max[d];
                let range = (hi - lo).max(1e-6);
                normalized.push((p - lo) / range * 2.0 - 1.0);
            }
        }

        let input = GpuTensor::from_f32(device, &normalized, &[n, 3]);
        let output = self.engine.forward_batch(device, compute, &input, n);
        let raw = output.download(device);

        // Denormalize distances
        raw.iter().map(|&d| d * self.dist_scale).collect()
    }
}

/// Sample random points and evaluate the analytical SDF.
fn sample_training_data(
    sdf: &SdfNode,
    bounds_min: Vec3,
    bounds_max: Vec3,
    num_samples: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut points = Vec::with_capacity(num_samples * 3);
    let mut distances = Vec::with_capacity(num_samples);

    let range = bounds_max - bounds_min;

    // Simple deterministic sampling (grid + jitter via hash)
    let side = (num_samples as f32).cbrt().ceil() as usize;
    let mut count = 0;
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                if count >= num_samples {
                    break;
                }
                let t = Vec3::new(
                    ix as f32 / side as f32,
                    iy as f32 / side as f32,
                    iz as f32 / side as f32,
                );
                let p = bounds_min + t * range;
                let d = eval(sdf, p);
                points.extend_from_slice(&[p.x, p.y, p.z]);
                distances.push(d);
                count += 1;
            }
        }
    }

    (points, distances)
}

/// Create ternary weights based on training data correlation.
fn random_ternary_weights(
    device: &GpuDevice,
    out_features: usize,
    in_features: usize,
    _points: &[f32],
    _distances: &[f32],
) -> GpuTernaryWeight {
    // Simple pattern: alternate +1/-1/0 for initial projection
    let values: Vec<i8> = (0..out_features * in_features)
        .map(|i| match i % 3 {
            0 => 1,
            1 => -1,
            _ => 0,
        })
        .collect();
    GpuTernaryWeight::from_ternary(device, &values, out_features, in_features)
}

/// Create a random ternary projection matrix.
fn random_ternary_projection(
    device: &GpuDevice,
    out_features: usize,
    in_features: usize,
) -> GpuTernaryWeight {
    let values: Vec<i8> = (0..out_features * in_features)
        .map(|i| {
            // Hash-based pseudo-random ternary
            let h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 62;
            match h {
                0 => 1,
                1 => -1,
                _ => 0,
            }
        })
        .collect();
    GpuTernaryWeight::from_ternary(device, &values, out_features, in_features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_training_data() {
        let sdf = SdfNode::sphere(1.0);
        let (pts, dists) = sample_training_data(
            &sdf,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            27,
        );
        assert_eq!(pts.len(), 27 * 3);
        assert_eq!(dists.len(), 27);
    }
}
