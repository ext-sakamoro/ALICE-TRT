//! Compute Pipeline: Compile Once, Dispatch Many
//!
//! Pre-compiled WGSL shaders for ternary operations.
//! Bind groups connect GPU buffers to shader parameters.

use crate::device::GpuDevice;
use crate::kernel::{
    ReluParams, RELU_SHADER, TERNARY_MATMUL_BATCH_SHADER, TERNARY_MATVEC_SHADER,
    TERNARY_MATVEC_TILED_SHADER,
};
use crate::tensor::GpuTensor;
use crate::weights::GpuTernaryWeight;

/// Pre-compiled compute pipelines for all ternary operations
pub struct TernaryCompute {
    matvec_pipeline: wgpu::ComputePipeline,
    matvec_layout: wgpu::BindGroupLayout,
    matvec_tiled_pipeline: wgpu::ComputePipeline,
    matvec_tiled_layout: wgpu::BindGroupLayout,
    matmul_batch_pipeline: wgpu::ComputePipeline,
    matmul_batch_layout: wgpu::BindGroupLayout,
    relu_pipeline: wgpu::ComputePipeline,
    relu_layout: wgpu::BindGroupLayout,
}

impl TernaryCompute {
    /// Compile all shaders and create pipelines
    pub fn new(device: &GpuDevice) -> Self {
        let (matvec_pipeline, matvec_layout) =
            Self::create_ternary_pipeline(device, TERNARY_MATVEC_SHADER, "matvec");
        let (matvec_tiled_pipeline, matvec_tiled_layout) =
            Self::create_ternary_pipeline(device, TERNARY_MATVEC_TILED_SHADER, "matvec_tiled");
        let (matmul_batch_pipeline, matmul_batch_layout) =
            Self::create_ternary_pipeline(device, TERNARY_MATMUL_BATCH_SHADER, "matmul_batch");
        let (relu_pipeline, relu_layout) = Self::create_relu_pipeline(device);

        Self {
            matvec_pipeline,
            matvec_layout,
            matvec_tiled_pipeline,
            matvec_tiled_layout,
            matmul_batch_pipeline,
            matmul_batch_layout,
            relu_pipeline,
            relu_layout,
        }
    }

    /// Ternary matrix-vector multiplication (auto-selects kernel)
    ///
    /// Returns GPU tensor with shape [out_features].
    pub fn matvec(
        &self,
        device: &GpuDevice,
        input: &GpuTensor,
        weights: &GpuTernaryWeight,
    ) -> GpuTensor {
        assert_eq!(
            input.len(),
            weights.in_features(),
            "Input length must match weight in_features"
        );

        let output = GpuTensor::output(device, &[weights.out_features()]);

        // Auto-select kernel: tiled for large layers, simple for small
        let use_tiled = weights.in_features() >= 1024;

        let (pipeline, layout) = if use_tiled {
            (&self.matvec_tiled_pipeline, &self.matvec_tiled_layout)
        } else {
            (&self.matvec_pipeline, &self.matvec_layout)
        };

        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.plus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights.minus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weights.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matvec_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            if use_tiled {
                // Tiled: one workgroup per row
                pass.dispatch_workgroups(weights.out_features() as u32, 1, 1);
            } else {
                // Simple: one thread per row, groups of 256
                let groups = (weights.out_features() as u32 + 255) / 256;
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }

        device.submit(encoder);

        output
    }

    /// Batched ternary matrix multiplication
    ///
    /// Input shape: [batch_size, in_features]
    /// Output shape: [batch_size, out_features]
    pub fn matmul_batch(
        &self,
        device: &GpuDevice,
        input: &GpuTensor,
        weights: &GpuTernaryWeight,
        batch_size: usize,
    ) -> GpuTensor {
        assert_eq!(
            input.len(),
            batch_size * weights.in_features(),
            "Input length must be batch_size * in_features"
        );

        let output = GpuTensor::output(device, &[batch_size, weights.out_features()]);

        // Update params with batch size
        weights.update_params(device, batch_size as u32);

        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bg"),
            layout: &self.matmul_batch_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.plus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights.minus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weights.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let groups_x = (weights.out_features() as u32 + 255) / 256;
            pass.dispatch_workgroups(groups_x, batch_size as u32, 1);
        }

        device.submit(encoder);

        // Restore batch_size = 1 in params
        weights.update_params(device, 1);

        output
    }

    /// ReLU activation (in-place)
    pub fn relu_inplace(&self, device: &GpuDevice, tensor: &GpuTensor) {
        let params = ReluParams {
            len: tensor.len() as u32,
            _pad: [0; 3],
        };
        let params_buffer =
            device.create_uniform_buffer("relu_params", bytemuck::bytes_of(&params));

        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relu_bg"),
            layout: &self.relu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("relu_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relu_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.relu_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let groups = (tensor.len() as u32 + 255) / 256;
            pass.dispatch_workgroups(groups, 1, 1);
        }

        device.submit(encoder);
    }

    // ========================================================================
    // Internal pipeline creation
    // ========================================================================

    fn create_ternary_pipeline(
        device: &GpuDevice,
        shader_src: &str,
        label: &str,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{label}_layout")),
                    entries: &[
                        // binding 0: input (read-only storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 1: plus_bits (read-only storage)
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
                        // binding 2: minus_bits (read-only storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 3: output (read-write storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 4: params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                    label: Some(&format!("{label}_pipeline_layout")),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{label}_pipeline")),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        (pipeline, layout)
    }

    fn create_relu_pipeline(
        device: &GpuDevice,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("relu"),
                source: wgpu::ShaderSource::Wgsl(RELU_SHADER.into()),
            });

        let layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("relu_layout"),
                    entries: &[
                        // binding 0: data (read-write storage)
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
                        // binding 1: params (uniform)
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

        let pipeline_layout =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("relu_pipeline_layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("relu_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        (pipeline, layout)
    }
}

impl std::fmt::Display for TernaryCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TernaryCompute[4 pipelines: matvec, matvec_tiled, matmul_batch, relu]")
    }
}
