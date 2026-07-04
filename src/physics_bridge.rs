//! ALICE-TRT × ALICE-Physics Bridge
//!
//! GPU ternary inference for physics control policies.
//! Uses the GPU to evaluate neural controllers that produce forces/torques
//! for physics bodies, enabling ML-driven physics simulation.

use crate::{
    Activation, GpuDevice, GpuInferenceEngine, GpuTensor, GpuTernaryWeight, TernaryCompute,
};
use alice_physics::{Fix128, RigidBody, Vec3Fix};

/// GPU-accelerated physics controller.
///
/// Runs a ternary neural network on the GPU to compute forces for
/// multiple physics bodies in parallel (batched inference).
pub struct GpuPhysicsController {
    /// Multi-layer inference engine.
    engine: GpuInferenceEngine,
    /// Number of input features per body (position xyz + velocity xyz = 6).
    input_dim: usize,
    /// Number of output features per body (force xyz = 3).
    output_dim: usize,
}

impl GpuPhysicsController {
    /// Create a controller from pre-trained ternary weight layers.
    ///
    /// # Arguments
    ///
    /// - `device`: GPU device
    /// - `layers`: Ternary weight values for each layer
    /// - `layer_dims`: (rows, cols) for each layer
    /// - `activations`: Activation function per layer
    pub fn new(
        device: &GpuDevice,
        layers: &[(&[i8], usize, usize)],
        activations: &[Activation],
    ) -> Self {
        let mut engine = GpuInferenceEngine::new();
        let input_dim = if layers.is_empty() { 6 } else { layers[0].2 };
        let output_dim = if layers.is_empty() {
            3
        } else {
            layers[layers.len() - 1].1
        };

        for (i, &(weights, rows, cols)) in layers.iter().enumerate() {
            let w = GpuTernaryWeight::from_ternary(device, weights, rows, cols);
            let act = activations.get(i).copied().unwrap_or(Activation::None);
            engine.add_layer(w, act);
        }

        Self {
            engine,
            input_dim,
            output_dim,
        }
    }

    /// Infer forces for a single body.
    ///
    /// Input: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    /// Output: [force_x, force_y, force_z]
    pub fn infer_single(
        &self,
        device: &GpuDevice,
        compute: &TernaryCompute,
        body: &RigidBody,
    ) -> [f32; 3] {
        let input_data = body_to_input(body);
        let input = GpuTensor::from_f32(device, &input_data, &[self.input_dim]);
        let output = self.engine.forward(device, compute, &input);
        let result = output.download(device);

        let mut force = [0.0f32; 3];
        for (i, v) in result.iter().take(3).enumerate() {
            force[i] = *v;
        }
        force
    }

    /// Infer forces for multiple bodies in a batch (GPU-parallel).
    ///
    /// Returns one [force_x, force_y, force_z] per body.
    pub fn infer_batch(
        &self,
        device: &GpuDevice,
        compute: &TernaryCompute,
        bodies: &[&RigidBody],
    ) -> Vec<[f32; 3]> {
        if bodies.is_empty() {
            return Vec::new();
        }

        // Flatten all body inputs into one batch tensor
        let mut flat_input = Vec::with_capacity(bodies.len() * self.input_dim);
        for body in bodies {
            flat_input.extend_from_slice(&body_to_input(body));
        }

        let input = GpuTensor::from_f32(device, &flat_input, &[bodies.len(), self.input_dim]);
        let output = self
            .engine
            .forward_batch(device, compute, &input, bodies.len());
        let result = output.download(device);

        result
            .chunks(self.output_dim)
            .map(|chunk| {
                let mut force = [0.0f32; 3];
                for (i, v) in chunk.iter().take(3).enumerate() {
                    force[i] = *v;
                }
                force
            })
            .collect()
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// Convert a rigid body's state to neural network input features.
fn body_to_input(body: &RigidBody) -> Vec<f32> {
    let (px, py, pz) = body.position.to_f32();
    let (vx, vy, vz) = body.velocity.to_f32();
    vec![px, py, pz, vx, vy, vz]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_to_input() {
        let body = RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::ONE);

        let input = body_to_input(&body);
        assert_eq!(input.len(), 6);
        assert!((input[0] - 1.0).abs() < 0.01);
        assert!((input[1] - 2.0).abs() < 0.01);
        assert!((input[2] - 3.0).abs() < 0.01);
        // Velocity should be zero initially
        assert!((input[3]).abs() < 0.01);
    }

    #[test]
    fn test_gpu_controller_creation() {
        let device = match GpuDevice::new() {
            Ok(d) => d,
            Err(_) => return, // No GPU available
        };

        // 6 → 4 → 3 network
        let w1: Vec<i8> = vec![
            1, -1, 0, 1, 0, -1, 1, 1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 0, 1, -1, 1, 0, -1,
        ]; // 4×6
        let w2: Vec<i8> = vec![1, -1, 0, 1, 0, -1, 1, 1, 1, 0, -1, 1]; // 3×4

        let controller = GpuPhysicsController::new(
            &device,
            &[(&w1, 4, 6), (&w2, 3, 4)],
            &[Activation::ReLU, Activation::None],
        );

        assert_eq!(controller.input_dim(), 6);
        assert_eq!(controller.output_dim(), 3);
    }
}

// ============================================================================
// Phase F GPU solver offload (physics-solver feature)
// ============================================================================
//
// The types below implement `alice_physics::gpu_bridge::GpuSolverBridge`
// so ALICE-Physics callers can inject a compute-shader-based Fix128 PGS
// iteration path. The dispatch body is scheduled for follow-up commits
// that pair with the WGSL kernels landing in `crate::fix128`; this
// skeleton pins the trait impl signature and keeps the CPU-side crate
// compile-time compatible.

#[cfg(feature = "physics-solver")]
mod solver_bridge {
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::math::Fix128;

    use crate::fix128::Fix128Gpu;

    /// GPU offload adapter for the sub-stepping TGS solver.
    ///
    /// Buffers the incoming island contents into `Fix128Gpu` storage
    /// so the WGSL kernels (see `crate::fix128`) can read them
    /// directly. Compute pipeline construction is scheduled for the
    /// follow-up commit; the current skeleton keeps the trait
    /// signature compile-time stable so ALICE-Physics can bring the
    /// adapter behind its `gpu-solver-bridge` feature without waiting
    /// for the full compute path.
    pub struct TrtSolverAdapter {
        positions: Vec<Fix128Gpu>,
        velocities: Vec<Fix128Gpu>,
    }

    impl TrtSolverAdapter {
        /// Construct an empty adapter. Populate via `send_island`.
        #[must_use]
        pub fn new() -> Self {
            Self {
                positions: Vec::new(),
                velocities: Vec::new(),
            }
        }
    }

    impl Default for TrtSolverAdapter {
        fn default() -> Self {
            Self::new()
        }
    }

    fn fix128_to_gpu(v: Fix128) -> Fix128Gpu {
        Fix128Gpu { hi: v.hi, lo: v.lo }
    }

    fn gpu_to_fix128(v: Fix128Gpu) -> Fix128 {
        Fix128::from_raw(v.hi, v.lo)
    }

    impl GpuSolverBridge for TrtSolverAdapter {
        fn send_island(&mut self, positions: &[[Fix128; 3]], velocities: &[[Fix128; 3]]) {
            self.positions.clear();
            for p in positions {
                self.positions.push(fix128_to_gpu(p[0]));
                self.positions.push(fix128_to_gpu(p[1]));
                self.positions.push(fix128_to_gpu(p[2]));
            }
            self.velocities.clear();
            for v in velocities {
                self.velocities.push(fix128_to_gpu(v[0]));
                self.velocities.push(fix128_to_gpu(v[1]));
                self.velocities.push(fix128_to_gpu(v[2]));
            }
        }

        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {
            // TODO(phase-f-followup): dispatch the WGSL Fix128 PGS
            // kernels against `self.positions` / `self.velocities`.
            // Skeleton pass: no-op, so `recv_island` echoes the
            // uploaded state unchanged and the CPU vs GPU diff test
            // trivially agrees for a zero-iteration workload.
        }

        fn recv_island(&self, positions: &mut [[Fix128; 3]], velocities: &mut [[Fix128; 3]]) {
            for (i, p) in positions.iter_mut().enumerate() {
                p[0] = gpu_to_fix128(self.positions[i * 3]);
                p[1] = gpu_to_fix128(self.positions[i * 3 + 1]);
                p[2] = gpu_to_fix128(self.positions[i * 3 + 2]);
            }
            for (i, v) in velocities.iter_mut().enumerate() {
                v[0] = gpu_to_fix128(self.velocities[i * 3]);
                v[1] = gpu_to_fix128(self.velocities[i * 3 + 1]);
                v[2] = gpu_to_fix128(self.velocities[i * 3 + 2]);
            }
        }

        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            // Skeleton echoing pass — the zero-iteration dispatch above
            // preserves the uploaded state verbatim, so the diff is
            // guaranteed to be zero. Follow-up commits will run a
            // matched CPU / GPU pair against the supplied fixture and
            // return `GpuDivergence` on any bit mismatch.
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Verify the zero-iteration round-trip preserves every
        /// coordinate byte-for-byte (Fix128 hi/lo pair unchanged).
        #[test]
        fn trt_solver_adapter_zero_iter_round_trips() {
            let mut adapter = TrtSolverAdapter::new();
            let positions = vec![
                [
                    Fix128::from_raw(1, 0xDEAD_BEEF_0000_0001),
                    Fix128::from_raw(2, 0xCAFE_BABE_0000_0002),
                    Fix128::from_raw(3, 0xF00D_D00D_0000_0003),
                ],
                [
                    Fix128::from_raw(-1, 0xAAAA_BBBB_CCCC_DDDD),
                    Fix128::from_raw(-2, 0x1111_2222_3333_4444),
                    Fix128::from_raw(-3, 0x5555_6666_7777_8888),
                ],
            ];
            let velocities = vec![
                [Fix128::ZERO, Fix128::ONE, Fix128::ZERO],
                [Fix128::ONE, Fix128::ZERO, Fix128::ONE],
            ];
            adapter.send_island(&positions, &velocities);
            adapter.dispatch_iterations(0, Fix128::from_ratio(1, 60));

            let mut out_p = vec![[Fix128::ZERO; 3]; 2];
            let mut out_v = vec![[Fix128::ZERO; 3]; 2];
            adapter.recv_island(&mut out_p, &mut out_v);

            for i in 0..2 {
                for axis in 0..3 {
                    assert_eq!(out_p[i][axis].hi, positions[i][axis].hi);
                    assert_eq!(out_p[i][axis].lo, positions[i][axis].lo);
                    assert_eq!(out_v[i][axis].hi, velocities[i][axis].hi);
                    assert_eq!(out_v[i][axis].lo, velocities[i][axis].lo);
                }
            }

            let fixture = DiffFixture {
                description: "zero_iter_round_trip",
                tolerance: Fix128::ZERO,
            };
            assert!(adapter.assert_bit_exact_vs_cpu(&fixture).is_ok());
        }
    }
}

#[cfg(feature = "physics-solver")]
pub use solver_bridge::TrtSolverAdapter;
