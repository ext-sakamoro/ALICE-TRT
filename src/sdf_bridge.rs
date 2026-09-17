//! ALICE-SDF bridge: GPU ternary neural SDF evaluation
//!
//! Evaluates a ternary neural network that approximates an SDF on the GPU
//! using ALICE-TRT's inference engine — for real-time rendering where the
//! analytical SDF tree is too expensive to evaluate per-pixel.
//!
//! # Status (2026-09-17)
//!
//! **Fitting is not implemented.** [`GpuNeuralSdf::fit`] fails fast with
//! `todo!`; until 2026-09-17 it silently built the network from a fixed
//! `+1/−1/0` pattern and a hash-based random projection while its doc
//! claimed to "fit a ternary neural network" — the returned field bore no
//! relation to the SDF (oracle `tests/analytic_oracle.rs`, sphere RMS error
//! ≈ radius; CLAUDE.md § 仮実装完了偽装の禁止).  The training path is
//! alice-train (STE on CPU) → [`crate::GpuTernaryWeight::from_kernel`]; Backlog
//! ALICE-TRT.  [`sample_training_data`] and [`GpuNeuralSdf::eval_batch`]
//! are real and stay.
//!
//! # Pipeline
//!
//! ```text
//! SdfNode → sample training points → (fit: TODO) → GpuNeuralSdf
//! GpuNeuralSdf.eval_batch(points) → approximate distances (GPU)
//! ```

use alice_sdf::prelude::{eval, SdfNode, Vec3};

use crate::{GpuDevice, GpuInferenceEngine, GpuTensor, TernaryCompute};

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
    /// Samples `config.num_samples` points within the given bounds and
    /// evaluates the analytical SDF (the training set), then must fit a
    /// ternary network to it.
    ///
    /// # Panics
    ///
    /// Always — the fit is not implemented (see the module docs).  The
    /// sampled training set is discarded after being computed so that a
    /// caller at least gets the SDF-evaluation cost profile.
    pub fn fit(
        device: &GpuDevice,
        sdf: &SdfNode,
        bounds_min: Vec3,
        bounds_max: Vec3,
        config: &NeuralSdfConfig,
    ) -> Self {
        let (_points, _distances) =
            sample_training_data(sdf, bounds_min, bounds_max, config.num_samples);
        let _ = device;
        todo!(
            "STUB: GpuNeuralSdf::fit — ternary network fitting (hidden {}×{}) is not implemented; \
             the previous body built a fixed +1/-1/0 pattern unrelated to the SDF (2026-09-17)",
            config.hidden_width,
            config.num_hidden
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_training_data() {
        let sdf = SdfNode::sphere(1.0);
        let (pts, dists) = sample_training_data(&sdf, Vec3::splat(-2.0), Vec3::splat(2.0), 27);
        assert_eq!(pts.len(), 27 * 3);
        assert_eq!(dists.len(), 27);
    }
}
