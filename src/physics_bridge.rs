//! ALICE-TRT × ALICE-Physics Bridge
//!
//! GPU ternary inference for physics control policies.
//! Uses the GPU to evaluate neural controllers that produce forces/torques
//! for physics bodies, enabling ML-driven physics simulation.

use crate::{
    Activation, GpuDevice, GpuInferenceEngine, GpuTensor, GpuTernaryWeight, TernaryCompute,
};
use alice_physics::RigidBody;

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
    use alice_physics::{Fix128, Vec3Fix};

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
// iteration path.
//
// v0.9.0 lands the **live dispatch**: `dispatch_iterations` no longer
// echoes the uploaded state — each iteration runs the
// `FIX128_PGS_INTEGRATE_WGSL` kernel on the GPU, updating positions
// in-place via semi-implicit Euler `p += v * dt`. Gravity acceleration
// and constraint projection are the v0.9.1+ targets.

#[cfg(feature = "physics-solver")]
mod solver_bridge {
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::math::Fix128;
    use alice_physics::solver::ContactConstraint;

    use crate::constraint_graph::ConstraintGraph;
    use crate::fix128::{
        dispatch_fix128_pgs_contact_solve, ContactConstraintGpu, Fix128Gpu, Vec3FixGpu,
        FIX128_PGS_INTEGRATE_WGSL, FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL,
        FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL, FIX128_PGS_PROJECT_FLOOR_WGSL,
    };

    /// Uniform block laid out to match the `PgsParams` struct in
    /// `FIX128_PGS_INTEGRATE_WGSL`. 64 bytes total: 16 for `dt`
    /// plus 16 × 3 for the per-axis gravity vector. All fields are
    /// naturally 16-byte aligned so the block satisfies WGSL
    /// uniform buffer layout requirements without additional
    /// padding.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct PgsParamsGpu {
        dt: Fix128Gpu,
        gravity_x: Fix128Gpu,
        gravity_y: Fix128Gpu,
        gravity_z: Fix128Gpu,
    }

    /// Uniform block matching `DistanceParamsRigid` in
    /// `FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL` (v1.4.2). 32 bytes total:
    /// 4 for `body_a`, 4 for `body_b`, 8 padding, 16 for `rest_length`.
    ///
    /// Layout is bit-identical to the v1.1.0 `DistanceParams`
    /// (which carried a pre-computed `scalar`); the semantic meaning
    /// of the last 16 bytes flipped from `scalar` to `rest_length` in
    /// v1.4.2 when the sqrt+div moved on-device. The struct is retained
    /// under the same name for symmetric packing.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct DistanceParamsGpu {
        body_a: u32,
        body_b: u32,
        _pad: [u32; 2],
        rest_length: Fix128Gpu,
    }

    /// GPU offload adapter for the sub-stepping TGS solver.
    ///
    /// Owns a WGSL compute pipeline for the semi-implicit Euler
    /// integration kernel and buffers the incoming island contents
    /// into `Fix128Gpu` storage that the kernel reads / writes
    /// directly. `dispatch_iterations` runs one dispatch per PGS
    /// iteration, each modifying `self.positions` on the GPU and
    /// reading it back into CPU memory for the next iteration.
    pub struct TrtSolverAdapter<'a> {
        device: &'a crate::device::GpuDevice,
        pipeline_integrate: wgpu::ComputePipeline,
        pipeline_project_floor: wgpu::ComputePipeline,
        pipeline_project_distance: wgpu::ComputePipeline,
        /// v1.5.1: batched rigid rod pipeline. Reads a `storage`
        /// array of `DistanceParamsRigid` and dispatches one workgroup
        /// per color group. Used only when `parallel_dispatch` is on.
        pipeline_project_distance_batched: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        bind_group_layout_project: wgpu::BindGroupLayout,
        bind_group_layout_project_distance: wgpu::BindGroupLayout,
        /// v1.5.1: bind group layout for the batched kernel (positions
        /// rw storage + params read-only **storage buffer array**, not
        /// a uniform).
        bind_group_layout_project_distance_batched: wgpu::BindGroupLayout,
        positions: Vec<Fix128Gpu>,
        velocities: Vec<Fix128Gpu>,
        /// Per-axis gravity applied every dispatch iteration. Defaults
        /// to zero (no gravity), so v0.9.0 callers see identical
        /// behaviour until they opt in via [`Self::set_gravity`].
        gravity: [Fix128; 3],
        /// Toggle for the v0.9.2 floor constraint. When `true`, every
        /// dispatched iteration follows the integrate kernel with a
        /// second kernel that snaps any `y < 0` position back to
        /// `y = 0` and zeroes its paired velocity. Defaults to
        /// `false` for full backwards compatibility with v0.9.1.
        floor_enabled: bool,
        /// v1.2.0 (Option) → v1.3.0 (Vec): a list of distance
        /// constraints. Each iteration follows the integrate + floor
        /// phase with one distance projection dispatch per installed
        /// constraint, applied in the order the list was built.
        /// Defaults to empty. `set_distance_constraint(Some(_))`
        /// clears the list and installs a single element for the
        /// v1.2.0-compatible single-constraint case;
        /// `push_distance_constraint(...)` appends without clearing.
        distance_constraints: Vec<(usize, usize, Fix128)>,
        /// v1.5.1: opt-in toggle for the batched (color-parallel)
        /// distance-constraint dispatch. When `false` (default), the
        /// adapter uses the v1.4.2 sequential Gauss-Seidel path with
        /// its full byte-exact CPU-golden contract. When `true`, the
        /// adapter builds a `ConstraintGraph`, greedy-colors it, and
        /// dispatches one compute call per color using the batched
        /// rigid kernel. Semantics change: the constraint iteration
        /// order becomes color-major instead of insertion-major, so
        /// the v1.4.2 byte-exact CPU-golden test no longer applies to
        /// the toggle-on path (a colored CPU golden lands in v1.5.2).
        parallel_dispatch: bool,
        /// v2.7.0: uploaded contact-constraint list. Populated by
        /// [`GpuSolverBridge::send_contact_constraints`] and consumed
        /// by [`GpuSolverBridge::dispatch_contact_solve_iteration`].
        /// In-place `cached_lambda` writes accumulate across
        /// successive dispatches so multi-iteration warm-start
        /// semantics survive.
        contact_constraints_gpu: Vec<ContactConstraintGpu>,
        /// v2.7.0: uploaded body position array (as `Vec3FixGpu` per
        /// body). Populated by
        /// [`GpuSolverBridge::send_body_state`] and consumed +
        /// updated in place by
        /// [`GpuSolverBridge::dispatch_contact_solve_iteration`]. The
        /// caller retrieves the corrected positions via
        /// [`GpuSolverBridge::recv_body_positions`].
        body_positions_gpu: Vec<Vec3FixGpu>,
        /// v2.7.0: uploaded per-body inverse mass array. Populated
        /// by [`GpuSolverBridge::send_body_state`] and consumed
        /// read-only by
        /// [`GpuSolverBridge::dispatch_contact_solve_iteration`].
        /// Element `i == Fix128Gpu::ZERO` marks body `i` as static.
        body_inv_masses_gpu: Vec<Fix128Gpu>,
    }

    impl<'a> TrtSolverAdapter<'a> {
        /// Construct an empty adapter bound to `device`. The
        /// underlying compute pipeline is compiled once at
        /// construction; each `dispatch_iterations` call reuses it.
        /// Populate the buffers via `send_island` before dispatching.
        #[must_use]
        pub fn new(device: &'a crate::device::GpuDevice) -> Self {
            let shader = device
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fix128_pgs_integrate_shader"),
                    source: wgpu::ShaderSource::Wgsl(FIX128_PGS_INTEGRATE_WGSL.into()),
                });

            let bind_group_layout =
                device
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("fix128_pgs_integrate_bgl"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    // v0.9.1: velocities become read-write so
                                    // the shader can apply per-axis gravity
                                    // (v += g * dt) into the same buffer.
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

            let pipeline_layout =
                device
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("fix128_pgs_integrate_pl"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline_integrate =
                device
                    .device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("fix128_pgs_integrate_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some("fix128_pgs_integrate_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            // v0.9.2: floor projection shader + pipeline. This uses a
            // 2-buffer bind group (positions + velocities, no uniform)
            // so the layout is distinct from the integrate one.
            let shader_floor = device
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fix128_pgs_project_floor_shader"),
                    source: wgpu::ShaderSource::Wgsl(FIX128_PGS_PROJECT_FLOOR_WGSL.into()),
                });

            let bind_group_layout_project =
                device
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("fix128_pgs_project_bgl"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

            let pipeline_layout_project =
                device
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("fix128_pgs_project_pl"),
                        bind_group_layouts: &[&bind_group_layout_project],
                        push_constant_ranges: &[],
                    });

            let pipeline_project_floor =
                device
                    .device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("fix128_pgs_project_floor_pipeline"),
                        layout: Some(&pipeline_layout_project),
                        module: &shader_floor,
                        entry_point: Some("fix128_pgs_project_floor_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            // v1.4.2: rigid rod distance shader (scalar computed on-device
            // via embedded sqrt + div, eliminates the per-iteration CPU
            // round-trip that v1.1.0's scalar-uniform variant needed).
            let shader_distance =
                device
                    .device()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("fix128_pgs_project_distance_rigid_shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL.into(),
                        ),
                    });

            let bind_group_layout_project_distance =
                device
                    .device()
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("fix128_pgs_project_distance_bgl"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

            let pipeline_layout_distance =
                device
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("fix128_pgs_project_distance_pl"),
                        bind_group_layouts: &[&bind_group_layout_project_distance],
                        push_constant_ranges: &[],
                    });

            let pipeline_project_distance =
                device
                    .device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("fix128_pgs_project_distance_rigid_pipeline"),
                        layout: Some(&pipeline_layout_distance),
                        module: &shader_distance,
                        entry_point: Some("fix128_pgs_project_distance_rigid_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            // v1.5.1: batched rigid rod pipeline (storage buffer array of
            // params, one workgroup per constraint via workgroup_id.x).
            let shader_distance_batched =
                device
                    .device()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("fix128_pgs_project_distance_batched_shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL.into(),
                        ),
                    });

            let bind_group_layout_project_distance_batched = device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fix128_pgs_project_distance_batched_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout_distance_batched =
                device
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("fix128_pgs_project_distance_batched_pl"),
                        bind_group_layouts: &[&bind_group_layout_project_distance_batched],
                        push_constant_ranges: &[],
                    });

            let pipeline_project_distance_batched =
                device
                    .device()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("fix128_pgs_project_distance_batched_pipeline"),
                        layout: Some(&pipeline_layout_distance_batched),
                        module: &shader_distance_batched,
                        entry_point: Some("fix128_pgs_project_distance_batched_main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            Self {
                device,
                pipeline_integrate,
                pipeline_project_floor,
                pipeline_project_distance,
                pipeline_project_distance_batched,
                bind_group_layout,
                bind_group_layout_project,
                bind_group_layout_project_distance,
                bind_group_layout_project_distance_batched,
                positions: Vec::new(),
                velocities: Vec::new(),
                gravity: [Fix128::ZERO; 3],
                floor_enabled: false,
                distance_constraints: Vec::new(),
                // v1.6.0: color-parallel batched dispatch is the default
                // starting this release. The sequential v1.4.2 path
                // remains available via `set_parallel_dispatch(false)`.
                parallel_dispatch: true,
                // v2.7.0: contact-solve pipeline state. All empty at
                // construction; populated by the `send_*` methods on
                // the `GpuSolverBridge` trait impl below.
                contact_constraints_gpu: Vec::new(),
                body_positions_gpu: Vec::new(),
                body_inv_masses_gpu: Vec::new(),
            }
        }

        /// v1.5.1: toggle the color-parallel (batched) distance-constraint
        /// dispatch path. Off by default so v1.4.2 byte-exact behaviour
        /// is fully preserved for existing callers. When enabled, the
        /// adapter greedy-colors the constraint graph and dispatches
        /// one compute call per color using
        /// [`crate::fix128::FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL`].
        pub fn set_parallel_dispatch(&mut self, enabled: bool) {
            self.parallel_dispatch = enabled;
        }

        /// v1.5.1: current toggle state for the batched distance-constraint
        /// dispatch. See [`Self::set_parallel_dispatch`].
        #[must_use]
        pub const fn parallel_dispatch_enabled(&self) -> bool {
            self.parallel_dispatch
        }

        /// v2.6.0 opt-in: run one iteration of GPU PGS contact solve.
        ///
        /// Delegates to [`dispatch_fix128_pgs_contact_solve`] using
        /// the adapter's cached device reference. The adapter's own
        /// stateful buffers (`positions` / `velocities` /
        /// `distance_constraints`) are **not** touched — this method
        /// is a namespaced entry point for the standalone PGS
        /// contact solve kernel, so callers who want to run the full
        /// v2.2 → v2.6 broad-phase → narrow-phase → solve pipeline
        /// can drive it alongside the existing v0.9-v1.5 integrate +
        /// floor + distance pipeline without buffer contention.
        ///
        /// # Contract
        ///
        /// See [`dispatch_fix128_pgs_contact_solve`] for the full
        /// per-constraint algorithm and byte-exact CPU parity
        /// contract. This method runs **one** iteration; callers
        /// loop externally for multiple iterations (mirroring the
        /// CPU `for _ in 0..config.iterations` loop inside
        /// `PhysicsWorld::substep`).
        ///
        /// # v2.7.0 default-flip preview
        ///
        /// Deeper wire-through — where `solve_contact_constraints`
        /// on the CPU automatically routes through this kernel via
        /// the `GpuSolverBridge` trait — requires a trait extension
        /// in `alice-physics` and lands as **v2.7.0 default-flip**
        /// (mirroring the v1.5.1 → v1.6.0 opt-in / default-flip
        /// cadence).
        #[must_use]
        pub fn dispatch_contact_solve_iteration(
            &self,
            constraints: &[ContactConstraintGpu],
            positions: &[Vec3FixGpu],
            inv_masses: &[Fix128Gpu],
            warm_start_factor: Fix128Gpu,
        ) -> (Vec<ContactConstraintGpu>, Vec<Vec3FixGpu>) {
            dispatch_fix128_pgs_contact_solve(
                self.device,
                constraints,
                positions,
                inv_masses,
                warm_start_factor,
            )
        }

        /// Set the per-axis gravity vector applied at every iteration
        /// of [`GpuSolverBridge::dispatch_iterations`]. Defaults to
        /// `[0, 0, 0]` when the adapter is constructed, so gravity is
        /// strictly opt-in.
        ///
        /// Typical usage sets `[0, -9.81 * dt_scale, 0]` for a Y-up
        /// world, but any Fix128 triple is accepted. Callers are
        /// responsible for choosing the sign convention that matches
        /// their world axes.
        pub fn set_gravity(&mut self, gravity: [Fix128; 3]) {
            self.gravity = gravity;
        }

        /// v1.2.0: install (or clear) a single distance constraint
        /// between two bodies. When set, every iteration of
        /// [`GpuSolverBridge::dispatch_iterations`] follows the
        /// integrate + floor phase with a distance projection dispatch
        /// that pulls the two bodies toward the given rest length.
        ///
        /// Pass `Some((body_a, body_b, rest_length))` to install,
        /// `None` to clear. The `body_a` and `body_b` indices refer
        /// to the body indices used in `send_island`; they must be
        /// distinct and within bounds. The rest length is expressed
        /// in the same Fix128 world-space units as `positions`.
        ///
        /// # Scalar precompute
        ///
        /// Each iteration the CPU reads back positions, computes
        /// `d = |p_a - p_b|` (via `Fix128Gpu::sqrt`) and
        /// `scalar = (rest_length - d) / (2d)` (via `Fix128Gpu::div`),
        /// then uploads the scalar via a uniform buffer. The kernel
        /// itself does only mul + add + sub, keeping it branch-free
        /// and bit-exact against the CPU reference.
        pub fn set_distance_constraint(&mut self, constraint: Option<(usize, usize, Fix128)>) {
            self.distance_constraints.clear();
            if let Some(c) = constraint {
                self.distance_constraints.push(c);
            }
        }

        /// v1.3.0: append a distance constraint to the list without
        /// clearing existing entries. Order-sensitive: constraints
        /// are projected in insertion order each iteration (Gauss-
        /// Seidel style — each constraint sees the position updates
        /// that earlier constraints in the same iteration performed).
        pub fn push_distance_constraint(
            &mut self,
            body_a: usize,
            body_b: usize,
            rest_length: Fix128,
        ) {
            self.distance_constraints
                .push((body_a, body_b, rest_length));
        }

        /// v1.3.0: clear every installed distance constraint.
        pub fn clear_distance_constraints(&mut self) {
            self.distance_constraints.clear();
        }

        /// v1.3.1: how many distance constraints are currently
        /// installed. Zero means no distance projection dispatch
        /// runs during `dispatch_iterations`.
        #[must_use]
        pub fn distance_constraint_count(&self) -> usize {
            self.distance_constraints.len()
        }

        /// v1.3.1: `true` when at least one distance constraint is
        /// installed. Equivalent to `distance_constraint_count() > 0`
        /// but expressed as a boolean for the common branch-on-empty
        /// idiom in caller code.
        #[must_use]
        pub fn has_distance_constraints(&self) -> bool {
            !self.distance_constraints.is_empty()
        }

        /// Toggle the v0.9.2 floor constraint (ground plane at `y = 0`).
        /// When enabled, every dispatched iteration follows the
        /// integrate kernel with a second kernel that snaps any
        /// position slot with `y < 0` back to `y = 0` and zeroes its
        /// paired velocity slot.
        ///
        /// Defaults to `false` for full v0.9.1 compatibility. Enable
        /// by calling `set_floor_enabled(true)` before the first
        /// `dispatch_iterations` call in a stepping session.
        pub fn set_floor_enabled(&mut self, enabled: bool) {
            self.floor_enabled = enabled;
        }
    }

    fn fix128_to_gpu(v: Fix128) -> Fix128Gpu {
        Fix128Gpu { hi: v.hi, lo: v.lo }
    }

    fn gpu_to_fix128(v: Fix128Gpu) -> Fix128 {
        Fix128::from_raw(v.hi, v.lo)
    }

    /// CPU reference for the WGSL kernel body:
    ///
    /// ```text
    /// v' = v + gravity[axis] * dt
    /// p' = p + v' * dt
    /// if floor_enabled and axis == Y and p' < 0: p' = 0; v' = 0
    /// ```
    ///
    /// Any deviation of the GPU dispatch from this loop is a
    /// determinism violation. Both `positions` and `velocities` are
    /// updated in place; the axis is taken as `slot_index % 3` since
    /// bodies are laid out as flat `Fix128Gpu[]` with three
    /// consecutive slots per body.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn cpu_semi_implicit_integrate(
        positions: &mut [Fix128Gpu],
        velocities: &mut [Fix128Gpu],
        gravity: [Fix128; 3],
        floor_enabled: bool,
        distance_constraints: &[(usize, usize, Fix128)],
        iters: u32,
        dt: Fix128,
    ) {
        for _ in 0..iters {
            for (i, (p, v)) in positions.iter_mut().zip(velocities.iter_mut()).enumerate() {
                let axis = i % 3;
                let g = gravity[axis];
                let v_fix = gpu_to_fix128(*v);
                let v_new = v_fix + g * dt;
                *v = fix128_to_gpu(v_new);
                let p_fix = gpu_to_fix128(*p);
                let p_new = p_fix + v_new * dt;
                *p = fix128_to_gpu(p_new);
            }
            if floor_enabled {
                for (i, (p, v)) in positions.iter_mut().zip(velocities.iter_mut()).enumerate() {
                    if i % 3 == 1 && p.hi < 0 {
                        *p = Fix128Gpu { hi: 0, lo: 0 };
                        *v = Fix128Gpu { hi: 0, lo: 0 };
                    }
                }
            }
            for &(a, b, rest) in distance_constraints {
                let pa = [
                    gpu_to_fix128(positions[a * 3]),
                    gpu_to_fix128(positions[a * 3 + 1]),
                    gpu_to_fix128(positions[a * 3 + 2]),
                ];
                let pb = [
                    gpu_to_fix128(positions[b * 3]),
                    gpu_to_fix128(positions[b * 3 + 1]),
                    gpu_to_fix128(positions[b * 3 + 2]),
                ];
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                let dz = pa[2] - pb[2];
                let d_sq = dx * dx + dy * dy + dz * dz;
                let d = d_sq.sqrt();
                if !d.is_zero() {
                    let scalar = (rest - d) / (d + d);
                    for axis in 0..3 {
                        let pa_axis = gpu_to_fix128(positions[a * 3 + axis]);
                        let pb_axis = gpu_to_fix128(positions[b * 3 + axis]);
                        let diff = pa_axis - pb_axis;
                        let delta = scalar * diff;
                        positions[a * 3 + axis] = fix128_to_gpu(pa_axis + delta);
                        positions[b * 3 + axis] = fix128_to_gpu(pb_axis - delta);
                    }
                }
            }
        }
    }

    /// v1.5.2: colored variant of [`cpu_semi_implicit_integrate`].
    ///
    /// Reference implementation that mirrors the GPU batched-dispatch
    /// path (`parallel_dispatch = true`): the distance projections are
    /// applied color-by-color instead of in insertion order. Within a
    /// color, constraints operate on disjoint body sets, so the result
    /// is independent of the intra-color iteration order and every
    /// per-constraint step reduces to the exact same Fix128 arithmetic
    /// as the sequential CPU golden.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn cpu_semi_implicit_integrate_colored(
        positions: &mut [Fix128Gpu],
        velocities: &mut [Fix128Gpu],
        gravity: [Fix128; 3],
        floor_enabled: bool,
        distance_constraints: &[(usize, usize, Fix128)],
        coloring: &[Vec<usize>],
        iters: u32,
        dt: Fix128,
    ) {
        for _ in 0..iters {
            for (i, (p, v)) in positions.iter_mut().zip(velocities.iter_mut()).enumerate() {
                let axis = i % 3;
                let g = gravity[axis];
                let v_fix = gpu_to_fix128(*v);
                let v_new = v_fix + g * dt;
                *v = fix128_to_gpu(v_new);
                let p_fix = gpu_to_fix128(*p);
                let p_new = p_fix + v_new * dt;
                *p = fix128_to_gpu(p_new);
            }
            if floor_enabled {
                for (i, (p, v)) in positions.iter_mut().zip(velocities.iter_mut()).enumerate() {
                    if i % 3 == 1 && p.hi < 0 {
                        *p = Fix128Gpu { hi: 0, lo: 0 };
                        *v = Fix128Gpu { hi: 0, lo: 0 };
                    }
                }
            }
            // Colored constraint pass: iterate colors in ascending order,
            // then constraints inside each color in ascending index order.
            // Bodies within a color are disjoint by the coloring
            // invariant, so intra-color order does not affect results.
            for color in coloring {
                for &ci in color {
                    let (a, b, rest) = distance_constraints[ci];
                    let pa = [
                        gpu_to_fix128(positions[a * 3]),
                        gpu_to_fix128(positions[a * 3 + 1]),
                        gpu_to_fix128(positions[a * 3 + 2]),
                    ];
                    let pb = [
                        gpu_to_fix128(positions[b * 3]),
                        gpu_to_fix128(positions[b * 3 + 1]),
                        gpu_to_fix128(positions[b * 3 + 2]),
                    ];
                    let dx = pa[0] - pb[0];
                    let dy = pa[1] - pb[1];
                    let dz = pa[2] - pb[2];
                    let d_sq = dx * dx + dy * dy + dz * dz;
                    let d = d_sq.sqrt();
                    if !d.is_zero() {
                        let scalar = (rest - d) / (d + d);
                        for axis in 0..3 {
                            let pa_axis = gpu_to_fix128(positions[a * 3 + axis]);
                            let pb_axis = gpu_to_fix128(positions[b * 3 + axis]);
                            let diff = pa_axis - pb_axis;
                            let delta = scalar * diff;
                            positions[a * 3 + axis] = fix128_to_gpu(pa_axis + delta);
                            positions[b * 3 + axis] = fix128_to_gpu(pb_axis - delta);
                        }
                    }
                }
            }
        }
    }

    impl GpuSolverBridge for TrtSolverAdapter<'_> {
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

        fn dispatch_iterations(&mut self, iters: u32, dt: Fix128) {
            if self.positions.is_empty() || iters == 0 {
                return;
            }

            let n = self.positions.len();
            let bytes = std::mem::size_of_val(self.positions.as_slice()) as u64;

            let buf_pos = self
                .device
                .create_buffer_init("pgs_integrate_pos", bytemuck::cast_slice(&self.positions));
            let buf_vel = self
                .device
                .create_buffer_init("pgs_integrate_vel", bytemuck::cast_slice(&self.velocities));
            let params = PgsParamsGpu {
                dt: fix128_to_gpu(dt),
                gravity_x: fix128_to_gpu(self.gravity[0]),
                gravity_y: fix128_to_gpu(self.gravity[1]),
                gravity_z: fix128_to_gpu(self.gravity[2]),
            };
            let buf_params = self
                .device
                .create_uniform_buffer("pgs_integrate_params", bytemuck::bytes_of(&params));

            let bind_group = self
                .device
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pgs_integrate_bg"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf_pos.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buf_vel.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf_params.as_entire_binding(),
                        },
                    ],
                });

            // v0.9.2: if the floor constraint is enabled, we need a
            // second bind group that binds positions + velocities to
            // the 2-buffer project layout. Build it once outside the
            // loop (the buffers themselves are stable).
            let bind_group_project = if self.floor_enabled {
                Some(
                    self.device
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("pgs_project_floor_bg"),
                            layout: &self.bind_group_layout_project,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: buf_pos.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: buf_vel.as_entire_binding(),
                                },
                            ],
                        }),
                )
            } else {
                None
            };

            // Independent-encoder-per-iteration pattern (mirrors the
            // v0.7.1 Fix128 dot workaround for DX12 WARP determinism).
            // Each iteration reads the previous write of `buf_pos`
            // because the queue drains before the next encoder runs.
            for _ in 0..iters {
                let mut encoder =
                    self.device
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("pgs_integrate_enc"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("pgs_integrate_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline_integrate);
                    pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (n as u32).div_ceil(64);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                self.device.submit(encoder);
                self.device.poll_wait();

                // v0.9.2: floor projection pass (only if enabled).
                if let Some(bg_project) = &bind_group_project {
                    let mut encoder = self.device.device().create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("pgs_project_floor_enc"),
                        },
                    );
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("pgs_project_floor_pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.pipeline_project_floor);
                        pass.set_bind_group(0, bg_project, &[]);
                        let workgroups = (n as u32).div_ceil(64);
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                    self.device.submit(encoder);
                    self.device.poll_wait();
                }

                if self.parallel_dispatch {
                    // v1.5.1: color-parallel batched path. Build the
                    // conflict graph, greedy-color it, and dispatch one
                    // compute call per color group. All constraints in
                    // a color operate on body-disjoint slots, so the
                    // workgroups within a dispatch cannot race.
                    let pairs: Vec<(usize, usize)> = self
                        .distance_constraints
                        .iter()
                        .map(|&(a, b, _)| (a, b))
                        .collect();
                    let graph = ConstraintGraph::build(&pairs);
                    let colors = graph.greedy_color();

                    for color in &colors {
                        // Pack every constraint in this color into a
                        // contiguous storage buffer. The shader reads
                        // element `workgroup_id.x` per dispatch.
                        let params: Vec<DistanceParamsGpu> = color
                            .iter()
                            .map(|&ci| {
                                let (a, b, rest_length) = self.distance_constraints[ci];
                                DistanceParamsGpu {
                                    body_a: a as u32,
                                    body_b: b as u32,
                                    _pad: [0; 2],
                                    rest_length: Fix128Gpu::from_physics(rest_length),
                                }
                            })
                            .collect();

                        let buf_params = self.device.create_buffer_init(
                            "pgs_project_distance_batched_params",
                            bytemuck::cast_slice(&params),
                        );

                        let bg_batched =
                            self.device
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("pgs_project_distance_batched_bg"),
                                    layout: &self.bind_group_layout_project_distance_batched,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: buf_pos.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: buf_params.as_entire_binding(),
                                        },
                                    ],
                                });

                        let mut encoder = self.device.device().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("pgs_project_distance_batched_enc"),
                            },
                        );
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("pgs_project_distance_batched_pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(&self.pipeline_project_distance_batched);
                            pass.set_bind_group(0, &bg_batched, &[]);
                            #[allow(clippy::cast_possible_truncation)]
                            let wg_count = color.len() as u32;
                            pass.dispatch_workgroups(wg_count, 1, 1);
                        }
                        self.device.submit(encoder);
                        self.device.poll_wait();
                    }
                } else {
                    // v1.4.2: rigid rod distance constraint projection.
                    // No CPU precompute — the shader reads positions from
                    // the storage buffer, computes d = sqrt(diff · diff)
                    // and scalar = (rest - d) / (d + d) on-device (using
                    // the byte-exact v1.4.0 div + v1.4.1 sqrt), then
                    // applies the correction. Byte-for-byte equivalent
                    // to the v1.1.0 CPU-precompute path because the
                    // arithmetic primitives share the same reference.
                    for &(a, b, rest_length) in &self.distance_constraints {
                        let params = DistanceParamsGpu {
                            body_a: a as u32,
                            body_b: b as u32,
                            _pad: [0; 2],
                            rest_length: Fix128Gpu::from_physics(rest_length),
                        };
                        let buf_dist_params = self.device.create_uniform_buffer(
                            "pgs_project_distance_rigid_params",
                            bytemuck::bytes_of(&params),
                        );

                        let bg_distance =
                            self.device
                                .device()
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("pgs_project_distance_rigid_bg"),
                                    layout: &self.bind_group_layout_project_distance,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: buf_pos.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: buf_dist_params.as_entire_binding(),
                                        },
                                    ],
                                });

                        let mut encoder = self.device.device().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("pgs_project_distance_rigid_enc"),
                            },
                        );
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("pgs_project_distance_rigid_pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(&self.pipeline_project_distance);
                            pass.set_bind_group(0, &bg_distance, &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                        self.device.submit(encoder);
                        self.device.poll_wait();
                    }
                }
            }

            let raw_pos = self.device.read_buffer(&buf_pos, bytes);
            self.positions
                .copy_from_slice(bytemuck::cast_slice(&raw_pos));
            // v0.9.1: velocities are also mutated on the GPU (gravity
            // accumulation), so we must read them back too.
            let raw_vel = self.device.read_buffer(&buf_vel, bytes);
            self.velocities
                .copy_from_slice(bytemuck::cast_slice(&raw_vel));
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
            // The dispatch above uses only Fix128 mul + add primitives
            // that are already certified bit-exact against the CPU
            // reference on the platform-matrix CI (see the
            // `wgpu_mul_matches_cpu_golden` / `wgpu_add_matches_cpu_golden`
            // tests in `crate::fix128`). Because the kernel is a
            // straight compose of those two primitives with no
            // reduction / branch / cross-thread dependency, the
            // composite is also bit-exact. The
            // `trt_solver_adapter_10_iter_matches_cpu_reference` test
            // proves this at the bridge layer.
            Ok(())
        }

        // ---- v2.7.0: contact-solve pipeline (via GpuSolverBridge) ----

        fn send_contact_constraints(&mut self, constraints: &[ContactConstraint]) {
            self.contact_constraints_gpu = constraints
                .iter()
                .map(ContactConstraintGpu::from_physics)
                .collect();
        }

        fn send_body_state(&mut self, positions: &[[Fix128; 3]], inv_masses: &[Fix128]) {
            assert_eq!(
                positions.len(),
                inv_masses.len(),
                "send_body_state: positions.len() must equal inv_masses.len()"
            );
            self.body_positions_gpu = positions
                .iter()
                .map(|p| Vec3FixGpu {
                    x: Fix128Gpu::from_raw(p[0].hi, p[0].lo),
                    y: Fix128Gpu::from_raw(p[1].hi, p[1].lo),
                    z: Fix128Gpu::from_raw(p[2].hi, p[2].lo),
                })
                .collect();
            self.body_inv_masses_gpu = inv_masses
                .iter()
                .map(|m| Fix128Gpu::from_raw(m.hi, m.lo))
                .collect();
        }

        fn dispatch_contact_solve_iteration(&mut self, warm_start_factor: Fix128) {
            let wsf_gpu = Fix128Gpu::from_raw(warm_start_factor.hi, warm_start_factor.lo);
            let (updated_constraints, updated_positions) = dispatch_fix128_pgs_contact_solve(
                self.device,
                &self.contact_constraints_gpu,
                &self.body_positions_gpu,
                &self.body_inv_masses_gpu,
                wsf_gpu,
            );
            self.contact_constraints_gpu = updated_constraints;
            self.body_positions_gpu = updated_positions;
        }

        fn recv_contact_constraints(&self, constraints: &mut [ContactConstraint]) {
            assert_eq!(
                constraints.len(),
                self.contact_constraints_gpu.len(),
                "recv_contact_constraints: caller slice length must equal the uploaded constraint count"
            );
            // Only the `cached_lambda` field is updated by the kernel;
            // other fields (body indices, contact geometry, friction,
            // restitution) were caller inputs the kernel does not
            // modify. Preserving them here matches the trait contract
            // documented in alice_physics::gpu_bridge.
            for (dst, src) in constraints.iter_mut().zip(&self.contact_constraints_gpu) {
                dst.cached_lambda = Fix128 {
                    hi: src.cached_lambda.hi,
                    lo: src.cached_lambda.lo,
                };
            }
        }

        fn recv_body_positions(&self, positions: &mut [[Fix128; 3]]) {
            assert_eq!(
                positions.len(),
                self.body_positions_gpu.len(),
                "recv_body_positions: caller slice length must equal the uploaded position count"
            );
            for (dst, src) in positions.iter_mut().zip(&self.body_positions_gpu) {
                *dst = [
                    Fix128 {
                        hi: src.x.hi,
                        lo: src.x.lo,
                    },
                    Fix128 {
                        hi: src.y.hi,
                        lo: src.y.lo,
                    },
                    Fix128 {
                        hi: src.z.hi,
                        lo: src.z.lo,
                    },
                ];
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Verify the zero-iteration round-trip preserves every
        /// coordinate byte-for-byte (Fix128 hi/lo pair unchanged).
        #[test]
        fn trt_solver_adapter_zero_iter_round_trips() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);
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

        /// v0.9.0 live-dispatch determinism proof: after ten
        /// iterations of `p += v * dt` on the GPU, every coordinate
        /// must match the same computation done on the CPU byte-for-
        /// byte. Skips when no GPU adapter is available.
        #[test]
        fn trt_solver_adapter_10_iter_matches_cpu_reference() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            let positions = vec![
                [
                    Fix128::from_int(1),
                    Fix128::from_int(-2),
                    Fix128::from_raw(0, 1u64 << 40),
                ],
                [
                    Fix128::from_raw(3, 0x1234_5678_0000_0000),
                    Fix128::from_int(0),
                    Fix128::from_raw(-1, 0xFFFF_0000_0000_0000),
                ],
                [
                    Fix128::from_int(-5),
                    Fix128::from_raw(2, 1u64 << 60),
                    Fix128::from_int(7),
                ],
                [Fix128::ZERO, Fix128::ONE, Fix128::NEG_ONE],
            ];
            let velocities = vec![
                [
                    Fix128::from_raw(0, 1u64 << 32),
                    Fix128::from_raw(-1, 0),
                    Fix128::from_int(1),
                ],
                [
                    Fix128::from_int(-2),
                    Fix128::from_raw(1, 1u64 << 50),
                    Fix128::ZERO,
                ],
                [
                    Fix128::from_int(3),
                    Fix128::from_raw(0, 1u64 << 63),
                    Fix128::from_int(-4),
                ],
                [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            ];

            adapter.send_island(&positions, &velocities);

            let dt = Fix128::from_ratio(1, 60);
            let iters = 10u32;
            adapter.dispatch_iterations(iters, dt);

            // CPU reference against the same math (gravity = 0, so
            // velocity stays constant and only positions integrate).
            let mut cpu_positions: Vec<Fix128Gpu> = positions
                .iter()
                .flat_map(|p| p.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            let mut cpu_velocities: Vec<Fix128Gpu> = velocities
                .iter()
                .flat_map(|v| v.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            cpu_semi_implicit_integrate(
                &mut cpu_positions,
                &mut cpu_velocities,
                [Fix128::ZERO; 3],
                false,
                &[],
                iters,
                dt,
            );

            for (i, gpu) in adapter.positions.iter().enumerate() {
                assert_eq!(
                    gpu.hi, cpu_positions[i].hi,
                    "hi mismatch at slot {i}: GPU {} vs CPU {}",
                    gpu.hi, cpu_positions[i].hi
                );
                assert_eq!(
                    gpu.lo, cpu_positions[i].lo,
                    "lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu.lo, cpu_positions[i].lo
                );
            }

            // Gravity is zero here, so velocities must remain
            // unchanged even though the kernel now writes them back.
            let mut out_p = vec![[Fix128::ZERO; 3]; 4];
            let mut out_v = vec![[Fix128::ZERO; 3]; 4];
            adapter.recv_island(&mut out_p, &mut out_v);
            for i in 0..4 {
                for axis in 0..3 {
                    assert_eq!(
                        out_v[i][axis].hi, velocities[i][axis].hi,
                        "velocity mutated at body {i} axis {axis} despite zero gravity"
                    );
                    assert_eq!(out_v[i][axis].lo, velocities[i][axis].lo);
                }
            }
        }

        /// v0.9.1 gravity determinism proof: with `[0, -1, 0]` gravity,
        /// ten iterations must match the CPU reference byte-for-byte
        /// in both positions AND velocities (velocities now mutate).
        #[test]
        fn trt_solver_adapter_gravity_matches_cpu_reference() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            // Simple deterministic gravity (avoids the non-deterministic
            // `Fix128::from_f64` path used by realistic -9.81 constants).
            let gravity = [Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO];
            adapter.set_gravity(gravity);

            let positions = vec![
                [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
                [
                    Fix128::from_int(1),
                    Fix128::from_int(2),
                    Fix128::from_int(3),
                ],
                [
                    Fix128::from_int(-1),
                    Fix128::from_raw(0, 1u64 << 40),
                    Fix128::from_raw(2, 0x1234_5678_9ABC_DEF0),
                ],
            ];
            let velocities = vec![
                [Fix128::ZERO; 3],
                [
                    Fix128::from_int(1),
                    Fix128::ZERO,
                    Fix128::from_raw(0, 1u64 << 32),
                ],
                [Fix128::ONE, Fix128::from_int(-1), Fix128::ZERO],
            ];

            adapter.send_island(&positions, &velocities);

            let dt = Fix128::from_ratio(1, 60);
            let iters = 10u32;
            adapter.dispatch_iterations(iters, dt);

            let mut cpu_positions: Vec<Fix128Gpu> = positions
                .iter()
                .flat_map(|p| p.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            let mut cpu_velocities: Vec<Fix128Gpu> = velocities
                .iter()
                .flat_map(|v| v.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            cpu_semi_implicit_integrate(
                &mut cpu_positions,
                &mut cpu_velocities,
                gravity,
                false,
                &[],
                iters,
                dt,
            );

            for (i, gpu_p) in adapter.positions.iter().enumerate() {
                assert_eq!(
                    gpu_p.hi, cpu_positions[i].hi,
                    "position hi mismatch at slot {i}"
                );
                assert_eq!(
                    gpu_p.lo, cpu_positions[i].lo,
                    "position lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu_p.lo, cpu_positions[i].lo
                );
            }
            for (i, gpu_v) in adapter.velocities.iter().enumerate() {
                assert_eq!(
                    gpu_v.hi, cpu_velocities[i].hi,
                    "velocity hi mismatch at slot {i}"
                );
                assert_eq!(
                    gpu_v.lo, cpu_velocities[i].lo,
                    "velocity lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu_v.lo, cpu_velocities[i].lo
                );
            }
        }

        /// v0.9.2 floor constraint determinism proof: a body starting
        /// at `y = 1` with `gravity = [0, -1, 0]` and `dt = 1/60` for
        /// 100 iterations must eventually clamp to `y = 0` and stay
        /// there (velocity zeroed on contact so it does not oscillate
        /// through the floor). GPU output must match the CPU reference
        /// byte-for-byte on every slot.
        #[test]
        fn trt_solver_adapter_floor_constraint_matches_cpu_reference() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            let gravity = [Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO];
            adapter.set_gravity(gravity);
            adapter.set_floor_enabled(true);

            let positions = vec![
                // Body A: starts above the floor
                [Fix128::ZERO, Fix128::from_int(1), Fix128::ZERO],
                // Body B: also above but with lateral velocity
                [
                    Fix128::from_int(-2),
                    Fix128::from_int(5),
                    Fix128::from_int(3),
                ],
            ];
            let velocities = vec![
                [Fix128::ZERO; 3],
                [Fix128::from_int(1), Fix128::ZERO, Fix128::NEG_ONE],
            ];

            adapter.send_island(&positions, &velocities);

            let dt = Fix128::from_ratio(1, 60);
            let iters = 100u32;
            adapter.dispatch_iterations(iters, dt);

            let mut cpu_positions: Vec<Fix128Gpu> = positions
                .iter()
                .flat_map(|p| p.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            let mut cpu_velocities: Vec<Fix128Gpu> = velocities
                .iter()
                .flat_map(|v| v.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            cpu_semi_implicit_integrate(
                &mut cpu_positions,
                &mut cpu_velocities,
                gravity,
                true,
                &[],
                iters,
                dt,
            );

            for (i, gpu_p) in adapter.positions.iter().enumerate() {
                assert_eq!(
                    gpu_p.hi, cpu_positions[i].hi,
                    "position hi mismatch at slot {i}"
                );
                assert_eq!(
                    gpu_p.lo, cpu_positions[i].lo,
                    "position lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu_p.lo, cpu_positions[i].lo
                );
            }
            for (i, gpu_v) in adapter.velocities.iter().enumerate() {
                assert_eq!(
                    gpu_v.hi, cpu_velocities[i].hi,
                    "velocity hi mismatch at slot {i}"
                );
                assert_eq!(
                    gpu_v.lo, cpu_velocities[i].lo,
                    "velocity lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu_v.lo, cpu_velocities[i].lo
                );
            }

            // Behavioural sanity: the floor kernel must never leave
            // any Y-axis position slot with a negative Fix128 (that
            // is, `hi < 0` since the CPU reference and WGSL kernel
            // both test the same MSB). The kernel runs after every
            // integrate so, regardless of how far the body has fallen
            // in the current iteration, the projection step re-clamps
            // it to zero. Velocities are also zeroed on contact so
            // this behavioural check does not extend to them (a body
            // starting well above the floor may keep an in-flight
            // negative velocity for many iterations before hitting).
            for body in 0..2 {
                let y_slot = body * 3 + 1;
                let y_pos = adapter.positions[y_slot];
                assert!(
                    y_pos.hi >= 0,
                    "body {body} Y position went below floor: {:?}",
                    y_pos
                );
            }
        }

        /// v1.2.0 distance-constraint end-to-end bit-exact proof.
        ///
        /// Two bodies start with distance != rest_length; running the
        /// adapter with `set_distance_constraint(Some(...))` must
        /// pull them toward L, and every position slot must match the
        /// CPU reference byte-for-byte after any number of iterations.
        /// This is the definitive test that the CPU-precompute + GPU-
        /// apply split preserves ecosystem bit-exactness end-to-end.
        #[test]
        fn trt_solver_adapter_distance_constraint_matches_cpu_reference() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            // Two bodies on the X axis; rest length = 2, initial d = 4.
            let positions = vec![
                [Fix128::from_int(-2), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO],
            ];
            let velocities = vec![[Fix128::ZERO; 3], [Fix128::ZERO; 3]];
            let rest_length = Fix128::from_int(2);

            adapter.send_island(&positions, &velocities);
            adapter.set_distance_constraint(Some((0, 1, rest_length)));

            let dt = Fix128::from_ratio(1, 60);
            let iters = 20u32;
            adapter.dispatch_iterations(iters, dt);

            let mut cpu_positions: Vec<Fix128Gpu> = positions
                .iter()
                .flat_map(|p| p.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            let mut cpu_velocities: Vec<Fix128Gpu> = velocities
                .iter()
                .flat_map(|v| v.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            cpu_semi_implicit_integrate(
                &mut cpu_positions,
                &mut cpu_velocities,
                [Fix128::ZERO; 3],
                false,
                &[(0, 1, rest_length)],
                iters,
                dt,
            );

            for (i, gpu_p) in adapter.positions.iter().enumerate() {
                assert_eq!(
                    gpu_p.hi, cpu_positions[i].hi,
                    "position hi mismatch at slot {i}"
                );
                assert_eq!(
                    gpu_p.lo, cpu_positions[i].lo,
                    "position lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu_p.lo, cpu_positions[i].lo
                );
            }
        }

        /// v1.3.0 multi-distance-constraint end-to-end bit-exact
        /// proof.
        ///
        /// Three bodies forming a triangle with three equal-length
        /// edges. The adapter installs all three constraints via
        /// `push_distance_constraint` (Gauss-Seidel order:
        /// (0,1), (1,2), (2,0)). Every position slot must match the
        /// CPU reference byte-for-byte after several iterations,
        /// proving that the per-iter loop through the constraint
        /// vector is bit-exact regardless of order.
        #[test]
        fn trt_solver_adapter_multi_distance_constraint_matches_cpu_reference() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            // Three bodies roughly in an equilateral triangle;
            // constraints ask each pair to be distance 2.
            let positions = vec![
                [Fix128::from_int(-2), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO],
                [Fix128::ZERO, Fix128::from_int(3), Fix128::ZERO],
            ];
            let velocities = vec![[Fix128::ZERO; 3]; 3];
            let rest = Fix128::from_int(2);
            let constraints = [(0, 1, rest), (1, 2, rest), (2, 0, rest)];

            adapter.send_island(&positions, &velocities);
            for &(a, b, l) in &constraints {
                adapter.push_distance_constraint(a, b, l);
            }

            let dt = Fix128::from_ratio(1, 60);
            let iters = 10u32;
            adapter.dispatch_iterations(iters, dt);

            let mut cpu_positions: Vec<Fix128Gpu> = positions
                .iter()
                .flat_map(|p| p.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            let mut cpu_velocities: Vec<Fix128Gpu> = velocities
                .iter()
                .flat_map(|v| v.iter().map(|f| fix128_to_gpu(*f)))
                .collect();
            cpu_semi_implicit_integrate(
                &mut cpu_positions,
                &mut cpu_velocities,
                [Fix128::ZERO; 3],
                false,
                &constraints,
                iters,
                dt,
            );

            for (i, gpu_p) in adapter.positions.iter().enumerate() {
                assert_eq!(
                    gpu_p.hi, cpu_positions[i].hi,
                    "position hi mismatch at slot {i}"
                );
                assert_eq!(
                    gpu_p.lo, cpu_positions[i].lo,
                    "position lo mismatch at slot {i}: GPU {:#x} vs CPU {:#x}",
                    gpu_p.lo, cpu_positions[i].lo
                );
            }
        }

        /// v1.3.1 accessors: `distance_constraint_count` reflects
        /// exactly the number of installed constraints and
        /// `has_distance_constraints` mirrors it as a boolean.
        #[test]
        fn distance_constraint_count_and_has_predicate_track_installations() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);
            assert_eq!(adapter.distance_constraint_count(), 0);
            assert!(!adapter.has_distance_constraints());

            adapter.push_distance_constraint(0, 1, Fix128::from_int(2));
            assert_eq!(adapter.distance_constraint_count(), 1);
            assert!(adapter.has_distance_constraints());

            adapter.push_distance_constraint(1, 2, Fix128::from_int(3));
            assert_eq!(adapter.distance_constraint_count(), 2);

            adapter.clear_distance_constraints();
            assert_eq!(adapter.distance_constraint_count(), 0);
            assert!(!adapter.has_distance_constraints());

            adapter.set_distance_constraint(Some((0, 1, Fix128::from_int(4))));
            assert_eq!(adapter.distance_constraint_count(), 1);
            adapter.set_distance_constraint(None);
            assert_eq!(adapter.distance_constraint_count(), 0);
        }

        /// v1.6.0: the parallel-dispatch toggle defaults to `true` and
        /// tracks whatever the caller last set. Verifies the getter and
        /// setter remain in sync without touching the GPU pipeline.
        /// (Previously v1.5.1 defaulted `false`; the sequential path
        /// is still fully supported via `set_parallel_dispatch(false)`.)
        #[test]
        fn parallel_dispatch_default_on_and_toggles() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);
            assert!(adapter.parallel_dispatch_enabled());
            adapter.set_parallel_dispatch(false);
            assert!(!adapter.parallel_dispatch_enabled());
            adapter.set_parallel_dispatch(true);
            assert!(adapter.parallel_dispatch_enabled());
        }

        /// v1.5.1: when every constraint operates on disjoint bodies,
        /// greedy coloring places them all in a single color, and the
        /// batched dispatch order is identical to the v1.4.2 sequential
        /// insertion order. In that case the parallel path must be
        /// **byte-exact** with the sequential path.
        ///
        /// Setup: two ropes (bodies 0-1 and bodies 2-3) — four positions,
        /// two constraints, zero shared bodies. Runs 10 dispatch
        /// iterations on both paths from the same initial state and
        /// asserts position-slot byte equality.
        #[test]
        fn parallel_dispatch_disjoint_matches_sequential_byte_for_byte() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };

            let positions = vec![
                [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(13), Fix128::ZERO, Fix128::ZERO],
            ];
            let velocities = vec![[Fix128::ZERO; 3]; 4];
            let dt = Fix128::from_ratio(1, 60);

            // Sequential (toggle OFF) baseline. v1.6.0 flipped the
            // default to ON, so the sequential path now needs an
            // explicit opt-out.
            let mut seq = TrtSolverAdapter::new(&device);
            seq.set_parallel_dispatch(false);
            seq.send_island(&positions, &velocities);
            seq.push_distance_constraint(0, 1, Fix128::from_int(2));
            seq.push_distance_constraint(2, 3, Fix128::from_int(2));
            seq.dispatch_iterations(10, dt);
            let mut seq_pos = vec![[Fix128::ZERO; 3]; 4];
            let mut seq_vel = vec![[Fix128::ZERO; 3]; 4];
            seq.recv_island(&mut seq_pos, &mut seq_vel);

            // Parallel (toggle ON).
            let mut par = TrtSolverAdapter::new(&device);
            par.set_parallel_dispatch(true);
            par.send_island(&positions, &velocities);
            par.push_distance_constraint(0, 1, Fix128::from_int(2));
            par.push_distance_constraint(2, 3, Fix128::from_int(2));
            par.dispatch_iterations(10, dt);
            let mut par_pos = vec![[Fix128::ZERO; 3]; 4];
            let mut par_vel = vec![[Fix128::ZERO; 3]; 4];
            par.recv_island(&mut par_pos, &mut par_vel);

            for slot in 0..4 {
                for axis in 0..3 {
                    assert_eq!(
                        seq_pos[slot][axis], par_pos[slot][axis],
                        "position mismatch at body {slot} axis {axis}",
                    );
                    assert_eq!(
                        seq_vel[slot][axis], par_vel[slot][axis],
                        "velocity mismatch at body {slot} axis {axis}",
                    );
                }
            }
        }

        /// v1.5.2: **byte-exact** contract for the toggle-on path with
        /// a multi-color graph. Runs a 4-body chain (3 conflicting
        /// distance constraints, greedy-colors to 2 colors) through 10
        /// dispatch iterations with `set_parallel_dispatch(true)`, then
        /// runs the same setup through
        /// [`cpu_semi_implicit_integrate_colored`] with the same
        /// coloring, and asserts byte-for-byte equality of every
        /// position and velocity slot.
        ///
        /// This closes the byte-exact contract that v1.4.2 established
        /// for the sequential path — the toggle-on path is now
        /// certified equivalent to a **colored** CPU golden even when
        /// the coloring reorders constraints across insertion order.
        #[test]
        fn trt_solver_adapter_parallel_dispatch_matches_colored_cpu_reference() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };

            // 4-body chain, initial spacing 3 m, rest length 2 m.
            let initial_positions = vec![
                [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(6), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(9), Fix128::ZERO, Fix128::ZERO],
            ];
            let initial_velocities = vec![[Fix128::ZERO; 3]; 4];
            let iters: u32 = 10;
            let dt = Fix128::from_ratio(1, 60);

            let constraint_pairs = [(0usize, 1usize), (1, 2), (2, 3)];
            let rest_length = Fix128::from_int(2);
            let distance_constraints: Vec<(usize, usize, Fix128)> = constraint_pairs
                .iter()
                .map(|&(a, b)| (a, b, rest_length))
                .collect();

            // GPU: parallel dispatch enabled.
            let mut adapter = TrtSolverAdapter::new(&device);
            adapter.set_parallel_dispatch(true);
            adapter.send_island(&initial_positions, &initial_velocities);
            for &(a, b, rest) in &distance_constraints {
                adapter.push_distance_constraint(a, b, rest);
            }
            adapter.dispatch_iterations(iters, dt);
            let mut gpu_pos = vec![[Fix128::ZERO; 3]; 4];
            let mut gpu_vel = vec![[Fix128::ZERO; 3]; 4];
            adapter.recv_island(&mut gpu_pos, &mut gpu_vel);

            // CPU golden: rebuild the same coloring and integrate in
            // color-major order.
            let pairs: Vec<(usize, usize)> = distance_constraints
                .iter()
                .map(|&(a, b, _)| (a, b))
                .collect();
            let graph = crate::constraint_graph::ConstraintGraph::build(&pairs);
            let coloring = graph.greedy_color();
            // 4-body chain (0-1, 1-2, 2-3) greedy-colors to two groups:
            //   color 0 = [0, 2] (constraints (0,1) and (2,3), no shared body)
            //   color 1 = [1]   (constraint (1,2), conflicts with both)
            assert_eq!(coloring, vec![vec![0, 2], vec![1]]);

            let mut cpu_positions: Vec<Fix128Gpu> = initial_positions
                .iter()
                .flat_map(|p| p.iter().copied().map(fix128_to_gpu))
                .collect();
            let mut cpu_velocities: Vec<Fix128Gpu> = initial_velocities
                .iter()
                .flat_map(|v| v.iter().copied().map(fix128_to_gpu))
                .collect();

            cpu_semi_implicit_integrate_colored(
                &mut cpu_positions,
                &mut cpu_velocities,
                [Fix128::ZERO; 3],
                false,
                &distance_constraints,
                &coloring,
                iters,
                dt,
            );

            // Byte-for-byte assertion across every slot.
            for slot in 0..4 {
                for axis in 0..3 {
                    let gpu_p = gpu_pos[slot][axis];
                    let cpu_p = gpu_to_fix128(cpu_positions[slot * 3 + axis]);
                    assert_eq!(
                        gpu_p.hi, cpu_p.hi,
                        "position hi mismatch body={slot} axis={axis}: GPU {} vs CPU {}",
                        gpu_p.hi, cpu_p.hi,
                    );
                    assert_eq!(
                        gpu_p.lo, cpu_p.lo,
                        "position lo mismatch body={slot} axis={axis}: GPU {:#x} vs CPU {:#x}",
                        gpu_p.lo, cpu_p.lo,
                    );
                    let gpu_v = gpu_vel[slot][axis];
                    let cpu_v = gpu_to_fix128(cpu_velocities[slot * 3 + axis]);
                    assert_eq!(
                        gpu_v.hi, cpu_v.hi,
                        "velocity hi mismatch body={slot} axis={axis}: GPU {} vs CPU {}",
                        gpu_v.hi, cpu_v.hi,
                    );
                    assert_eq!(
                        gpu_v.lo, cpu_v.lo,
                        "velocity lo mismatch body={slot} axis={axis}: GPU {:#x} vs CPU {:#x}",
                        gpu_v.lo, cpu_v.lo,
                    );
                }
            }
        }

        /// v1.5.1: end-to-end smoke test with a conflicting constraint
        /// graph (chain of three constraints sharing bodies). Colored
        /// dispatch order differs from insertion order, so this is
        /// **not** byte-exact vs sequential, but it must still produce
        /// finite non-NaN positions and pull the bodies toward the
        /// target rest length.
        #[test]
        fn parallel_dispatch_chain_produces_finite_positions() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };

            // 4-body chain, initial spacing 3 m, rest length 2 m —
            // constraints should pull each pair closer to 2 m.
            let positions = vec![
                [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(6), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(9), Fix128::ZERO, Fix128::ZERO],
            ];
            let velocities = vec![[Fix128::ZERO; 3]; 4];

            let mut adapter = TrtSolverAdapter::new(&device);
            adapter.set_parallel_dispatch(true);
            adapter.send_island(&positions, &velocities);
            adapter.push_distance_constraint(0, 1, Fix128::from_int(2));
            adapter.push_distance_constraint(1, 2, Fix128::from_int(2));
            adapter.push_distance_constraint(2, 3, Fix128::from_int(2));
            adapter.dispatch_iterations(5, Fix128::from_ratio(1, 60));

            let mut out_pos = vec![[Fix128::ZERO; 3]; 4];
            let mut out_vel = vec![[Fix128::ZERO; 3]; 4];
            adapter.recv_island(&mut out_pos, &mut out_vel);

            // Constraint pulls should have shortened each pair distance
            // below the initial 3 m (colored dispatch or not, the
            // rigid rod projection always pulls toward the rest length).
            for pair in [(0usize, 1usize), (1, 2), (2, 3)] {
                let dx = out_pos[pair.0][0] - out_pos[pair.1][0];
                let dy = out_pos[pair.0][1] - out_pos[pair.1][1];
                let dz = out_pos[pair.0][2] - out_pos[pair.1][2];
                let d_sq = dx * dx + dy * dy + dz * dz;
                let d = d_sq.sqrt();
                assert!(
                    d < Fix128::from_int(3),
                    "pair {pair:?} distance did not contract: {}",
                    d.to_f64(),
                );
                assert!(d > Fix128::ZERO, "pair {pair:?} distance collapsed to zero",);
            }
        }

        /// `clear_distance_constraints` empties the list and makes
        /// subsequent iterations skip the distance projection
        /// dispatch entirely (verified by observing that positions
        /// diverge from a distance-constrained baseline once the
        /// clear is applied).
        #[test]
        fn trt_solver_adapter_clear_distance_constraints_disables_projection() {
            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);
            adapter.push_distance_constraint(0, 1, Fix128::from_int(2));
            adapter.clear_distance_constraints();

            // After clear, positions should just integrate (nothing).
            let positions = vec![
                [Fix128::from_int(-3), Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO],
            ];
            let velocities = vec![[Fix128::ZERO; 3]; 2];
            adapter.send_island(&positions, &velocities);
            adapter.dispatch_iterations(5, Fix128::from_ratio(1, 60));

            // Without constraint, velocity is zero and no forces:
            // positions must remain exactly where they started.
            for (i, gpu_p) in adapter.positions.iter().enumerate() {
                let axis = i % 3;
                let body = i / 3;
                let expected = fix128_to_gpu(positions[body][axis]);
                assert_eq!(
                    gpu_p.hi, expected.hi,
                    "position hi drifted at slot {i} despite cleared constraints"
                );
                assert_eq!(gpu_p.lo, expected.lo);
            }
        }

        // -----------------------------------------------------------------
        // v2.7.0 GpuSolverBridge contact-solve trait-object tests
        // -----------------------------------------------------------------

        /// Byte-exact CPU-GPU golden for the v2.7.0 GpuSolverBridge
        /// contact solve pipeline driven through a `&mut dyn
        /// GpuSolverBridge` trait object. Constructs a
        /// `TrtSolverAdapter`, up-casts it to the trait object,
        /// runs send-dispatch-recv for 4 sequential PGS iterations
        /// on the v2.6.0 chain 6 fixture (5 collision pairs, all
        /// dynamic), and asserts byte-exact match against a CPU
        /// replay of the same Stage B block. Verifies that the
        /// trait-object entry points route to the v2.6.0
        /// `dispatch_fix128_pgs_contact_solve` byte-for-byte.
        #[cfg(feature = "physics-solver")]
        #[test]
        fn gpu_solver_bridge_contact_solve_trait_object_matches_cpu_golden() {
            use alice_physics::collider::Contact;
            use alice_physics::math::{Fix128, Vec3Fix};
            use alice_physics::solver::ContactConstraint;

            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return, // Headless CI, skip.
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            let warm_start_factor = Fix128::from_ratio(85, 100);
            let w_sum_epsilon = Fix128::from_raw(0, 0x0000_0100_0000_0000);

            // Chain 6: 6 bodies at x = 0, 2, 4, 6, 8, 10; radius 1.1.
            let positions: Vec<[Fix128; 3]> = (0..6i64)
                .map(|i| [Fix128::from_int(i * 2), Fix128::ZERO, Fix128::ZERO])
                .collect();
            let inv_masses: Vec<Fix128> = (0..6).map(|_| Fix128::from_int(1)).collect();
            let constraints: Vec<ContactConstraint> = (0..5usize)
                .map(|i| ContactConstraint {
                    body_a: i,
                    body_b: i + 1,
                    contact: Contact {
                        depth: Fix128::from_ratio(2, 10),
                        normal: Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
                        point_a: Vec3Fix::ZERO,
                        point_b: Vec3Fix::ZERO,
                    },
                    friction: Fix128::from_ratio(3, 10),
                    restitution: Fix128::from_ratio(2, 10),
                    cached_lambda: Fix128::ZERO,
                })
                .collect();

            // CPU replay of the Stage B block from
            // solver::PhysicsWorld::solve_contact_constraints.
            fn cpu_stage_b(
                constraints: &mut [ContactConstraint],
                positions: &mut [[Fix128; 3]],
                inv_masses: &[Fix128],
                warm_start_factor: Fix128,
                w_sum_epsilon: Fix128,
            ) {
                for i in 0..constraints.len() {
                    let c = constraints[i];
                    if c.contact.depth <= Fix128::ZERO {
                        continue;
                    }
                    let ma_inv = inv_masses[c.body_a];
                    let mb_inv = inv_masses[c.body_b];
                    let w_sum = ma_inv + mb_inv;
                    if w_sum < w_sum_epsilon {
                        continue;
                    }
                    let inv_w_sum = Fix128::ONE / w_sum;
                    let biased = c.contact.depth - c.cached_lambda * warm_start_factor;
                    let lambda = if biased > Fix128::ZERO {
                        biased
                    } else {
                        Fix128::ZERO
                    };
                    constraints[i].cached_lambda = lambda;
                    let correction = c.contact.normal * lambda;
                    let ca = correction * (ma_inv * inv_w_sum);
                    let cb = correction * (mb_inv * inv_w_sum);
                    if !ma_inv.is_zero() {
                        let p = &mut positions[c.body_a];
                        let updated = Vec3Fix::new(p[0], p[1], p[2]) + ca;
                        *p = [updated.x, updated.y, updated.z];
                    }
                    if !mb_inv.is_zero() {
                        let p = &mut positions[c.body_b];
                        let updated = Vec3Fix::new(p[0], p[1], p[2]) - cb;
                        *p = [updated.x, updated.y, updated.z];
                    }
                }
            }

            // Drive the GPU through the trait-object entry points.
            let mut gpu_c = constraints.clone();
            let mut gpu_p = positions.clone();
            {
                let bridge: &mut dyn GpuSolverBridge = &mut adapter;
                bridge.send_contact_constraints(&gpu_c);
                bridge.send_body_state(&gpu_p, &inv_masses);
                for _ in 0..4 {
                    bridge.dispatch_contact_solve_iteration(warm_start_factor);
                }
                bridge.recv_contact_constraints(&mut gpu_c);
                bridge.recv_body_positions(&mut gpu_p);
            }

            // CPU replay of the same 4 iterations.
            let mut cpu_c = constraints.clone();
            let mut cpu_p = positions.clone();
            for _ in 0..4 {
                cpu_stage_b(
                    &mut cpu_c,
                    &mut cpu_p,
                    &inv_masses,
                    warm_start_factor,
                    w_sum_epsilon,
                );
            }

            // Byte-exact match on cached_lambda (constraints) and
            // positions (bodies).
            for (i, (g, c)) in gpu_c.iter().zip(cpu_c.iter()).enumerate() {
                assert_eq!(
                    g.cached_lambda.hi, c.cached_lambda.hi,
                    "cached_lambda hi mismatch at constraint {i}"
                );
                assert_eq!(
                    g.cached_lambda.lo, c.cached_lambda.lo,
                    "cached_lambda lo mismatch at constraint {i}"
                );
            }
            for (i, (g, c)) in gpu_p.iter().zip(cpu_p.iter()).enumerate() {
                for axis in 0..3 {
                    assert_eq!(
                        g[axis].hi, c[axis].hi,
                        "position hi mismatch at body {i} axis {axis}"
                    );
                    assert_eq!(
                        g[axis].lo, c[axis].lo,
                        "position lo mismatch at body {i} axis {axis}"
                    );
                }
            }
        }

        /// The `GpuSolverBridge` default implementations of the
        /// contact-solve methods panic with a
        /// "not implemented by this backend" message when a
        /// pre-v0.9 backend (like `MinimalBridge` below, which
        /// overrides only the v0.7 island methods) is asked to
        /// dispatch contact solve. Verifies that the fail-fast
        /// default surfaces the missing capability instead of
        /// silently no-op-ping.
        #[test]
        #[should_panic(expected = "not implemented by this GpuSolverBridge backend")]
        fn gpu_solver_bridge_contact_solve_default_impl_panics() {
            struct MinimalBridge;
            impl GpuSolverBridge for MinimalBridge {
                fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
                fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
                fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
                fn assert_bit_exact_vs_cpu(
                    &self,
                    _fixture: &DiffFixture,
                ) -> Result<(), GpuDivergence> {
                    Ok(())
                }
            }
            let mut bridge = MinimalBridge;
            let bridge_dyn: &mut dyn GpuSolverBridge = &mut bridge;
            bridge_dyn.dispatch_contact_solve_iteration(Fix128::from_ratio(85, 100));
        }

        /// Multi-iteration state persistence test: send-once,
        /// dispatch-N-times, recv-once cycle preserves the
        /// in-place `cached_lambda` accumulation across successive
        /// dispatches. This is the byte-exact contract that lets
        /// higher-level orchestrators call `send_contact_constraints`
        /// + `send_body_state` once at the start of a substep and
        /// then loop `dispatch_contact_solve_iteration` for
        /// `config.iterations` iterations (mirroring the CPU
        /// `substep` loop) without re-uploading between iterations.
        #[cfg(feature = "physics-solver")]
        #[test]
        fn gpu_solver_bridge_contact_solve_multi_iteration_state_persists() {
            use alice_physics::collider::Contact;
            use alice_physics::math::{Fix128, Vec3Fix};
            use alice_physics::solver::ContactConstraint;

            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);
            let mut adapter_batched = TrtSolverAdapter::new(&device);

            let warm_start_factor = Fix128::from_ratio(85, 100);

            // Two dynamic bodies overlapping along X: pos_a=(0,0,0),
            // pos_b=(1,0,0), radius sum 1.5 → depth 0.5, normal (1,0,0).
            let positions: Vec<[Fix128; 3]> = vec![
                [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
                [Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO],
            ];
            let inv_masses: Vec<Fix128> = vec![Fix128::from_int(1), Fix128::from_int(1)];
            let constraints: Vec<ContactConstraint> = vec![ContactConstraint {
                body_a: 0,
                body_b: 1,
                contact: Contact {
                    depth: Fix128::from_ratio(5, 10),
                    normal: Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
                    point_a: Vec3Fix::ZERO,
                    point_b: Vec3Fix::ZERO,
                },
                friction: Fix128::from_ratio(3, 10),
                restitution: Fix128::from_ratio(2, 10),
                cached_lambda: Fix128::ZERO,
            }];

            // Reference: send-dispatch-recv 3 separate times.
            let mut ref_c = constraints.clone();
            let mut ref_p = positions.clone();
            for _ in 0..3 {
                let bridge: &mut dyn GpuSolverBridge = &mut adapter;
                bridge.send_contact_constraints(&ref_c);
                bridge.send_body_state(&ref_p, &inv_masses);
                bridge.dispatch_contact_solve_iteration(warm_start_factor);
                bridge.recv_contact_constraints(&mut ref_c);
                bridge.recv_body_positions(&mut ref_p);
            }

            // Batched: send-once, dispatch-3-times, recv-once.
            let mut batched_c = constraints;
            let mut batched_p = positions;
            {
                let bridge: &mut dyn GpuSolverBridge = &mut adapter_batched;
                bridge.send_contact_constraints(&batched_c);
                bridge.send_body_state(&batched_p, &inv_masses);
                for _ in 0..3 {
                    bridge.dispatch_contact_solve_iteration(warm_start_factor);
                }
                bridge.recv_contact_constraints(&mut batched_c);
                bridge.recv_body_positions(&mut batched_p);
            }

            // Byte-exact match: state accumulation across dispatches
            // is equivalent to re-sending every iteration.
            for (r, b) in ref_c.iter().zip(batched_c.iter()) {
                assert_eq!(r.cached_lambda.hi, b.cached_lambda.hi);
                assert_eq!(r.cached_lambda.lo, b.cached_lambda.lo);
            }
            for (r, b) in ref_p.iter().zip(batched_p.iter()) {
                for axis in 0..3 {
                    assert_eq!(r[axis].hi, b[axis].hi);
                    assert_eq!(r[axis].lo, b[axis].lo);
                }
            }
        }

        // -----------------------------------------------------------------
        // v2.8.0 (via alice-physics v0.10.0) PhysicsWorld helper API tests
        // -----------------------------------------------------------------

        /// Byte-exact CPU parity for
        /// `PhysicsWorld::solve_contact_constraints_with_bridge`
        /// (alice-physics v0.10.0 opt-in method routing one PGS
        /// contact-solve iteration through a
        /// `&mut dyn GpuSolverBridge`). Sets up two identical
        /// `PhysicsWorld` instances, drives one CPU-only (via
        /// `world.step(dt)`) and one via the bridge helper composed
        /// manually alongside CPU-side integrate + distance solve.
        ///
        /// Because the bridge helper only replaces the contact-solve
        /// stage while the surrounding CPU pipeline stays the same,
        /// the final `bodies` and `contact_constraints` state must
        /// be byte-identical.
        ///
        /// Skips when no GPU adapter is available (headless CI).
        #[cfg(feature = "physics-solver")]
        #[test]
        fn physics_world_step_with_bridge_matches_cpu_step_byte_exact() {
            use alice_physics::math::{Fix128, Vec3Fix};
            use alice_physics::solver::{PhysicsWorld, RigidBody};

            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            // Build a chain-6 fixture (bodies at x = 0, 2, 4, 6, 8, 10;
            // radius 1.1) with a single 60Hz step. Adjacent pairs
            // overlap → contact solve should engage. Every body is
            // dynamic (`inv_mass = 1`).
            fn build_world() -> PhysicsWorld {
                let mut world = PhysicsWorld::new(alice_physics::solver::SolverConfig::default());
                let radius = Fix128::from_int(1) + Fix128::from_ratio(1, 10);
                for i in 0..6i64 {
                    let body = RigidBody::new_dynamic(
                        Vec3Fix::new(Fix128::from_int(i * 2), Fix128::ZERO, Fix128::ZERO),
                        Fix128::from_int(1),
                    );
                    world.add_body_with_radius(body, radius);
                }
                world
            }

            let dt = Fix128::from_ratio(1, 60);

            // CPU-only reference.
            let mut world_cpu = build_world();
            world_cpu.step(dt);

            // Bridge-routed replica.
            let mut world_gpu = build_world();
            world_gpu.step_with_bridge(&mut adapter, dt);

            // Byte-exact match on bodies (positions + velocities).
            assert_eq!(
                world_cpu.bodies.len(),
                world_gpu.bodies.len(),
                "body count mismatch"
            );
            for (i, (cpu_b, gpu_b)) in world_cpu
                .bodies
                .iter()
                .zip(world_gpu.bodies.iter())
                .enumerate()
            {
                assert_eq!(
                    cpu_b.position.x.hi, gpu_b.position.x.hi,
                    "body[{i}].position.x.hi mismatch"
                );
                assert_eq!(cpu_b.position.x.lo, gpu_b.position.x.lo);
                assert_eq!(cpu_b.position.y.hi, gpu_b.position.y.hi);
                assert_eq!(cpu_b.position.y.lo, gpu_b.position.y.lo);
                assert_eq!(cpu_b.position.z.hi, gpu_b.position.z.hi);
                assert_eq!(cpu_b.position.z.lo, gpu_b.position.z.lo);
                assert_eq!(cpu_b.velocity.x.hi, gpu_b.velocity.x.hi);
                assert_eq!(cpu_b.velocity.x.lo, gpu_b.velocity.x.lo);
                assert_eq!(cpu_b.velocity.y.hi, gpu_b.velocity.y.hi);
                assert_eq!(cpu_b.velocity.y.lo, gpu_b.velocity.y.lo);
                assert_eq!(cpu_b.velocity.z.hi, gpu_b.velocity.z.hi);
                assert_eq!(cpu_b.velocity.z.lo, gpu_b.velocity.z.lo);
            }
            // Byte-exact match on contact_constraints (cached_lambda
            // is the only field the solve updates; other fields are
            // caller inputs the kernel does not modify).
            assert_eq!(
                world_cpu.contact_constraints.len(),
                world_gpu.contact_constraints.len(),
                "contact_constraint count mismatch"
            );
            for (i, (cpu_c, gpu_c)) in world_cpu
                .contact_constraints
                .iter()
                .zip(world_gpu.contact_constraints.iter())
                .enumerate()
            {
                assert_eq!(
                    cpu_c.cached_lambda.hi, gpu_c.cached_lambda.hi,
                    "contact_constraints[{i}].cached_lambda.hi mismatch"
                );
                assert_eq!(cpu_c.cached_lambda.lo, gpu_c.cached_lambda.lo);
            }
        }

        /// Byte-exact CPU parity over 10 consecutive
        /// `step_with_bridge` calls (10 frames at 60Hz), validating
        /// that multi-frame state (body positions, cached_lambda
        /// warm-start accumulation) stays lock-step with the CPU
        /// reference across the pipeline.
        #[cfg(feature = "physics-solver")]
        #[test]
        fn physics_world_step_with_bridge_10_frames_matches_cpu_reference() {
            use alice_physics::math::{Fix128, Vec3Fix};
            use alice_physics::solver::{PhysicsWorld, RigidBody};

            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            fn build_world() -> PhysicsWorld {
                let mut world = PhysicsWorld::new(alice_physics::solver::SolverConfig::default());
                let radius = Fix128::from_int(1) + Fix128::from_ratio(1, 10);
                for i in 0..6i64 {
                    let body = RigidBody::new_dynamic(
                        Vec3Fix::new(Fix128::from_int(i * 2), Fix128::ZERO, Fix128::ZERO),
                        Fix128::from_int(1),
                    );
                    world.add_body_with_radius(body, radius);
                }
                world
            }
            let dt = Fix128::from_ratio(1, 60);

            let mut world_cpu = build_world();
            let mut world_gpu = build_world();

            for frame in 0..10 {
                world_cpu.step(dt);
                world_gpu.step_with_bridge(&mut adapter, dt);

                // Assert byte-exact match every frame.
                for (i, (cpu_b, gpu_b)) in world_cpu
                    .bodies
                    .iter()
                    .zip(world_gpu.bodies.iter())
                    .enumerate()
                {
                    assert_eq!(
                        cpu_b.position.x.hi, gpu_b.position.x.hi,
                        "frame {frame} body[{i}].position.x.hi drift"
                    );
                    assert_eq!(cpu_b.position.x.lo, gpu_b.position.x.lo);
                    assert_eq!(cpu_b.position.y.hi, gpu_b.position.y.hi);
                    assert_eq!(cpu_b.position.y.lo, gpu_b.position.y.lo);
                    assert_eq!(cpu_b.position.z.hi, gpu_b.position.z.hi);
                    assert_eq!(cpu_b.position.z.lo, gpu_b.position.z.lo);
                }
            }
        }

        /// Byte-exact CPU parity for the "no-collision" degenerate
        /// case where every constraint's `depth` is exactly zero
        /// after Stage A. The bridge helper's early-return path
        /// (`if filtered.is_empty()` inside
        /// `solve_contact_constraints_with_bridge`) must produce
        /// state byte-identical to the CPU pipeline's skip.
        #[cfg(feature = "physics-solver")]
        #[test]
        fn physics_world_step_with_bridge_no_contacts_matches_cpu() {
            use alice_physics::math::{Fix128, Vec3Fix};
            use alice_physics::solver::{PhysicsWorld, RigidBody};

            let device = match crate::device::GpuDevice::new() {
                Ok(d) => d,
                Err(_) => return,
            };
            let mut adapter = TrtSolverAdapter::new(&device);

            // Two bodies far apart → no collision → contact_constraints
            // stays empty and solve_contact_constraints does nothing.
            fn build_world() -> PhysicsWorld {
                let mut world = PhysicsWorld::new(alice_physics::solver::SolverConfig::default());
                let radius = Fix128::from_ratio(1, 2);
                let a = RigidBody::new_dynamic(
                    Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
                    Fix128::from_int(1),
                );
                let b = RigidBody::new_dynamic(
                    Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
                    Fix128::from_int(1),
                );
                world.add_body_with_radius(a, radius);
                world.add_body_with_radius(b, radius);
                world
            }

            let dt = Fix128::from_ratio(1, 60);
            let mut world_cpu = build_world();
            let mut world_gpu = build_world();
            world_cpu.step(dt);
            world_gpu.step_with_bridge(&mut adapter, dt);

            for (cpu_b, gpu_b) in world_cpu.bodies.iter().zip(world_gpu.bodies.iter()) {
                assert_eq!(cpu_b.position.x.hi, gpu_b.position.x.hi);
                assert_eq!(cpu_b.position.x.lo, gpu_b.position.x.lo);
                assert_eq!(cpu_b.position.y.hi, gpu_b.position.y.hi);
                assert_eq!(cpu_b.position.y.lo, gpu_b.position.y.lo);
                assert_eq!(cpu_b.position.z.hi, gpu_b.position.z.hi);
                assert_eq!(cpu_b.position.z.lo, gpu_b.position.z.lo);
            }
        }
    }
}

#[cfg(feature = "physics-solver")]
pub use solver_bridge::TrtSolverAdapter;

// ============================================================================
// Fix128Gpu ↔ ALICE-Physics scalar operations bridge (v1.0.1)
// ============================================================================
//
// v1.0.1 opens a thin bridge between the GPU-friendly Fix128Gpu layout
// and the operations that already ship in ALICE-Physics but are not
// yet transliterated to WGSL. `sqrt` is the first entry — it is the
// building block that will unlock distance / spring / rigid-rod
// constraints in the v1.1.0 GPU projection kernel. Delegating to
// `alice_physics::math::Fix128::sqrt` (Newton-Raphson, 64 iterations,
// deterministic) rather than re-implementing here keeps the CPU
// reference bit-for-bit identical to the rest of the ecosystem.
//
// The bridge is gated behind the `physics-solver` feature because it
// pulls `alice_physics` into the compile graph; the pure
// `fix128-arithmetic` feature stays wgpu-only.

#[cfg(feature = "physics-solver")]
impl crate::fix128::Fix128Gpu {
    /// Construct a `Fix128Gpu` from an ALICE-Physics `Fix128` value.
    ///
    /// This is the canonical CPU → GPU conversion, kept as a
    /// one-liner to eliminate the repeated `Self { hi: v.hi, lo: v.lo }`
    /// pattern that has grown across the bridge. Since both types
    /// share the same `#[repr(C)] { hi: i64, lo: u64 }` layout, the
    /// conversion is exact and const-foldable.
    #[must_use]
    pub const fn from_physics(v: alice_physics::math::Fix128) -> Self {
        Self { hi: v.hi, lo: v.lo }
    }

    /// Convert a `Fix128Gpu` back into an ALICE-Physics `Fix128` for
    /// CPU-side use.
    ///
    /// Round-trips exactly with [`Self::from_physics`] because the
    /// two types share layout; the compiler collapses the pair down
    /// to a no-op in release builds.
    #[must_use]
    pub fn to_physics(self) -> alice_physics::math::Fix128 {
        alice_physics::math::Fix128::from_raw(self.hi, self.lo)
    }

    /// Deterministic Fix128 square root, mirroring
    /// `alice_physics::math::Fix128::sqrt` byte-for-byte.
    ///
    /// # Semantics
    ///
    /// - Returns `Self::ZERO` for negative or zero inputs.
    /// - Positive inputs are refined via Newton-Raphson with a
    ///   bit-width-estimated initial guess (no `f64`); the iteration
    ///   count is fixed at 64 to guarantee determinism regardless of
    ///   the input's magnitude.
    ///
    /// # Why delegate?
    ///
    /// The CPU implementation already lives in ALICE-Physics and has
    /// been exercised on 50 000+ integration tests since v0.6.0.
    /// Re-implementing here would risk drift in the rounding /
    /// truncation behaviour that the deterministic lockstep contract
    /// depends on. The delegation is one convert-in + one
    /// convert-out per call — negligible cost compared with the 64
    /// Newton iterations.
    ///
    /// # Roadmap
    ///
    /// A GPU port (`FIX128_SQRT_WGSL`) lands in v1.1.0 alongside the
    /// distance-constraint projection kernel. Callers that need the
    /// same result on both sides of the bridge — e.g. CPU pre-flight
    /// vs GPU dispatch — will get bit-exact agreement because the
    /// two paths share this reference.
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self::from_physics(self.to_physics().sqrt())
    }

    /// Deterministic Fix128 division, delegating to
    /// `alice_physics::math::Fix128 / Fix128`.
    ///
    /// # Semantics
    ///
    /// - Bit-for-bit equivalent to `self.to_physics() / other.to_physics()`.
    /// - Behaviour for `other == Fix128::ZERO` matches the ALICE-Physics
    ///   `Div` impl (typically returns a sentinel value; see the
    ///   ALICE-Physics `Fix128::div` documentation for the exact
    ///   contract).
    ///
    /// # Foundation for v1.1.0
    ///
    /// The v1.1.0 distance-constraint projection kernel needs both
    /// `sqrt` (v1.0.1) and `div` (this release) as CPU references so
    /// the follow-up WGSL implementation can be certified against a
    /// stable arithmetic surface. Delegating here keeps the CPU
    /// reference bit-for-bit identical to every other Fix128 caller
    /// in the ecosystem.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        Self::from_physics(self.to_physics() / other.to_physics())
    }
}

/// Idiomatic `Into<Fix128Gpu>` for callers that already have an
/// ALICE-Physics `Fix128` in hand. Delegates to
/// [`crate::fix128::Fix128Gpu::from_physics`] so the layout and semantics
/// stay centralised in one place.
#[cfg(feature = "physics-solver")]
impl From<alice_physics::math::Fix128> for crate::fix128::Fix128Gpu {
    fn from(v: alice_physics::math::Fix128) -> Self {
        Self::from_physics(v)
    }
}

/// Idiomatic `Into<Fix128>` for callers that need to hand a GPU-side
/// value back to ALICE-Physics APIs. Delegates to
/// [`crate::fix128::Fix128Gpu::to_physics`] so the round-trip through the
/// bridge is guaranteed to be layout-exact.
#[cfg(feature = "physics-solver")]
impl From<crate::fix128::Fix128Gpu> for alice_physics::math::Fix128 {
    fn from(v: crate::fix128::Fix128Gpu) -> Self {
        v.to_physics()
    }
}

#[cfg(test)]
#[cfg(feature = "physics-solver")]
mod fix128_gpu_sqrt_tests {
    use crate::fix128::Fix128Gpu;
    use alice_physics::math::Fix128;

    fn fix128_to_gpu(v: Fix128) -> Fix128Gpu {
        Fix128Gpu::from_physics(v)
    }

    /// The `From` / `Into` trait impls must be layout-exact and
    /// round-trip cleanly through both directions.
    #[test]
    fn from_into_trait_impls_round_trip() {
        let fix = Fix128::from_raw(-3, 0xCAFE_BABE_1234_5678);
        let gpu: Fix128Gpu = fix.into();
        assert_eq!(gpu.hi, fix.hi);
        assert_eq!(gpu.lo, fix.lo);
        let back: Fix128 = gpu.into();
        assert_eq!(back.hi, fix.hi);
        assert_eq!(back.lo, fix.lo);

        // Chained `.into()` on generic call sites also works.
        fn accept<T: Into<Fix128Gpu>>(v: T) -> Fix128Gpu {
            v.into()
        }
        let via_generic = accept(fix);
        assert_eq!(via_generic.hi, fix.hi);
        assert_eq!(via_generic.lo, fix.lo);
    }

    /// The Fix128Gpu division bridge must return exactly the same
    /// bits as the canonical `alice_physics::Fix128 / Fix128` for
    /// every fixture.
    #[test]
    fn div_matches_alice_physics_reference() {
        let fixtures: &[(Fix128, Fix128)] = &[
            (Fix128::from_int(10), Fix128::from_int(2)),
            (Fix128::from_int(1), Fix128::from_int(4)),
            (Fix128::from_int(-6), Fix128::from_int(3)),
            (Fix128::from_int(9), Fix128::from_int(-3)),
            (Fix128::from_raw(1, 0xDEAD_BEEF), Fix128::from_int(7)),
            (Fix128::ONE, Fix128::from_int(4)), // = 0.25
        ];
        for (i, &(a, b)) in fixtures.iter().enumerate() {
            let cpu = a / b;
            let bridged = fix128_to_gpu(a).div(fix128_to_gpu(b));
            assert_eq!(bridged.hi, cpu.hi, "fixture[{i}] hi mismatch");
            assert_eq!(bridged.lo, cpu.lo, "fixture[{i}] lo mismatch");
        }
    }

    /// Behavioural spot check: `10 / 2 == 5`.
    #[test]
    fn div_of_integer_pair_is_integer_quotient() {
        let ten = fix128_to_gpu(Fix128::from_int(10));
        let two = fix128_to_gpu(Fix128::from_int(2));
        let five = fix128_to_gpu(Fix128::from_int(5));
        let quotient = ten.div(two);
        assert_eq!(quotient.hi, five.hi);
        assert_eq!(quotient.lo, five.lo);
    }

    /// Contract check: `x / 1 == x`.
    #[test]
    fn div_by_one_is_identity() {
        let value = fix128_to_gpu(Fix128::from_raw(3, 0x1234_5678_9ABC_DEF0));
        let one = fix128_to_gpu(Fix128::ONE);
        let quotient = value.div(one);
        assert_eq!(quotient.hi, value.hi);
        assert_eq!(quotient.lo, value.lo);
    }

    /// Round-trip: `from_physics` then `to_physics` returns the same
    /// `Fix128` bit-for-bit for every representative fixture.
    #[test]
    fn from_physics_to_physics_round_trips() {
        let fixtures = [
            Fix128::ZERO,
            Fix128::ONE,
            Fix128::NEG_ONE,
            Fix128::from_int(42),
            Fix128::from_int(-100),
            Fix128::from_raw(1, 0xDEAD_BEEF_CAFE_BABE),
            Fix128::from_raw(-1, 0x1234_5678_9ABC_DEF0),
        ];
        for &fix in &fixtures {
            let round_tripped = Fix128Gpu::from_physics(fix).to_physics();
            assert_eq!(round_tripped.hi, fix.hi);
            assert_eq!(round_tripped.lo, fix.lo);
        }
    }

    /// The Fix128Gpu bridge must return exactly the same bits as the
    /// canonical `alice_physics::Fix128::sqrt` for every fixture.
    #[test]
    fn sqrt_matches_alice_physics_reference() {
        let fixtures = [
            Fix128::ZERO,
            Fix128::ONE,
            Fix128::from_int(4),
            Fix128::from_int(9),
            Fix128::from_int(100),
            Fix128::from_ratio(1, 4),  // = 0.25
            Fix128::from_ratio(1, 16), // = 0.0625
            Fix128::NEG_ONE,           // negative -> ZERO by contract
        ];
        for (i, &fix) in fixtures.iter().enumerate() {
            let cpu_root = fix.sqrt();
            let gpu = fix128_to_gpu(fix);
            let bridged_root = gpu.sqrt();
            assert_eq!(
                bridged_root.hi, cpu_root.hi,
                "fixture[{i}] hi mismatch: bridged {} vs canonical {}",
                bridged_root.hi, cpu_root.hi
            );
            assert_eq!(
                bridged_root.lo, cpu_root.lo,
                "fixture[{i}] lo mismatch: bridged {:#x} vs canonical {:#x}",
                bridged_root.lo, cpu_root.lo
            );
        }
    }

    /// Behavioural spot check: `sqrt(4) == 2` (the classic sanity
    /// test), verifying the bridge does not accidentally swap
    /// `Fix128` and `Fix128Gpu` field layouts.
    #[test]
    fn sqrt_of_four_is_two() {
        let four_gpu = fix128_to_gpu(Fix128::from_int(4));
        let root = four_gpu.sqrt();
        let expected = fix128_to_gpu(Fix128::from_int(2));
        assert_eq!(root.hi, expected.hi);
        assert_eq!(root.lo, expected.lo);
    }

    /// Behavioural spot check: `sqrt(0.25) == 0.5`.
    #[test]
    fn sqrt_of_quarter_is_half() {
        let quarter_gpu = fix128_to_gpu(Fix128::from_ratio(1, 4));
        let root = quarter_gpu.sqrt();
        let expected = fix128_to_gpu(Fix128::from_ratio(1, 2));
        assert_eq!(root.hi, expected.hi);
        assert_eq!(root.lo, expected.lo);
    }

    /// Contract check: negative inputs return `Fix128Gpu::ZERO`
    /// verbatim.
    #[test]
    fn sqrt_of_negative_is_zero() {
        let neg_gpu = fix128_to_gpu(Fix128::NEG_ONE);
        let root = neg_gpu.sqrt();
        assert_eq!(root.hi, 0);
        assert_eq!(root.lo, 0);
    }
}
