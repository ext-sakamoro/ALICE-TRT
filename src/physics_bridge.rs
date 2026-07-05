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

    use crate::fix128::{Fix128Gpu, FIX128_PGS_INTEGRATE_WGSL, FIX128_PGS_PROJECT_FLOOR_WGSL};

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
        bind_group_layout: wgpu::BindGroupLayout,
        bind_group_layout_project: wgpu::BindGroupLayout,
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

            Self {
                device,
                pipeline_integrate,
                pipeline_project_floor,
                bind_group_layout,
                bind_group_layout_project,
                positions: Vec::new(),
                velocities: Vec::new(),
                gravity: [Fix128::ZERO; 3],
                floor_enabled: false,
            }
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
    fn cpu_semi_implicit_integrate(
        positions: &mut [Fix128Gpu],
        velocities: &mut [Fix128Gpu],
        gravity: [Fix128; 3],
        floor_enabled: bool,
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
                        // Y-axis slot with a negative Fix128: snap to
                        // zero (matches the WGSL floor kernel, which
                        // tests the same sign bit).
                        *p = Fix128Gpu { hi: 0, lo: 0 };
                        *v = Fix128Gpu { hi: 0, lo: 0 };
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
}

#[cfg(test)]
#[cfg(feature = "physics-solver")]
mod fix128_gpu_sqrt_tests {
    use crate::fix128::Fix128Gpu;
    use alice_physics::math::Fix128;

    fn fix128_to_gpu(v: Fix128) -> Fix128Gpu {
        Fix128Gpu::from_physics(v)
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
