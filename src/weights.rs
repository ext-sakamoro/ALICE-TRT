//! GPU Ternary Weights: 2-bit Bitplane Storage in VRAM
//!
//! **The Memory Hack**: Weights are stored as two bitplanes (plus_bits, minus_bits)
//! in GPU storage buffers. Each u32 word encodes 32 ternary weights.
//!
//! VRAM usage: 2 bits per weight (vs 16 bits FP16, 8 bits INT8)
//! Bandwidth reduction: 8x vs FP16, 4x vs INT8

use crate::device::GpuDevice;
use crate::kernel::GpuParams;

/// Ternary weights on GPU (bitplane representation)
///
/// Two storage buffers:
/// - `plus_bits[row * words_per_row + word]`: bit i = 1 means weight[row][word*32+i] = +1
/// - `minus_bits[row * words_per_row + word]`: bit i = 1 means weight[row][word*32+i] = -1
/// - Both bits 0 means weight = 0 (free compute via sparsity)
pub struct GpuTernaryWeight {
    pub(crate) plus_buffer: wgpu::Buffer,
    pub(crate) minus_buffer: wgpu::Buffer,
    pub(crate) params_buffer: wgpu::Buffer,
    out_features: usize,
    in_features: usize,
    scale: f32,
    words_per_row: usize,
}

impl GpuTernaryWeight {
    /// Upload from ALICE-ML TernaryWeightKernel (bitplane format)
    ///
    /// Zero-copy compatible: TernaryWeightKernel already stores bitplanes.
    pub fn from_kernel(device: &GpuDevice, kernel: &alice_ml::TernaryWeightKernel) -> Self {
        let out_features = kernel.out_features();
        let in_features = kernel.in_features();
        let words_per_row = kernel.words_per_row();
        let scale = kernel.scale();

        let plus_buffer = device.create_buffer_init(
            "trt_plus_bits",
            bytemuck::cast_slice(kernel.plus_bits()),
        );
        let minus_buffer = device.create_buffer_init(
            "trt_minus_bits",
            bytemuck::cast_slice(kernel.minus_bits()),
        );

        let params = GpuParams {
            in_features: in_features as u32,
            out_features: out_features as u32,
            words_per_row: words_per_row as u32,
            scale,
            batch_size: 1,
            _pad: [0; 3],
        };
        let params_buffer =
            device.create_uniform_buffer("trt_params", bytemuck::bytes_of(&params));

        Self {
            plus_buffer,
            minus_buffer,
            params_buffer,
            out_features,
            in_features,
            scale,
            words_per_row,
        }
    }

    /// Upload from ALICE-ML TernaryWeight (packed 2-bit format)
    ///
    /// Converts to bitplane format internally.
    pub fn from_packed(device: &GpuDevice, weights: &alice_ml::TernaryWeight) -> Self {
        let kernel = alice_ml::TernaryWeightKernel::from_packed_weight(weights);
        Self::from_kernel(device, &kernel)
    }

    /// Create directly from ternary values (-1, 0, +1)
    pub fn from_ternary(
        device: &GpuDevice,
        values: &[i8],
        out_features: usize,
        in_features: usize,
    ) -> Self {
        let kernel =
            alice_ml::TernaryWeightKernel::from_ternary(values, out_features, in_features);
        Self::from_kernel(device, &kernel)
    }

    /// Create with custom scale
    pub fn from_ternary_scaled(
        device: &GpuDevice,
        values: &[i8],
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Self {
        let kernel = alice_ml::TernaryWeightKernel::from_ternary_scaled(
            values,
            out_features,
            in_features,
            scale,
        );
        Self::from_kernel(device, &kernel)
    }

    /// Update params buffer (e.g., for batch size changes)
    pub(crate) fn update_params(&self, device: &GpuDevice, batch_size: u32) {
        let params = GpuParams {
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            words_per_row: self.words_per_row as u32,
            scale: self.scale,
            batch_size,
            _pad: [0; 3],
        };
        device
            .queue()
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    #[inline]
    pub fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    /// VRAM usage in bytes (bitplanes only, excludes params)
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        let total_words = self.out_features * self.words_per_row;
        total_words * 4 * 2 // plus + minus bitplanes, 4 bytes per u32
    }

    /// Compression ratio vs FP32
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.out_features * self.in_features * 4;
        fp32_size as f32 / self.memory_bytes() as f32
    }
}

impl std::fmt::Display for GpuTernaryWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuTernaryWeight[{}x{}, {:.1}x compression, {} bytes VRAM]",
            self.out_features,
            self.in_features,
            self.compression_ratio(),
            self.memory_bytes(),
        )
    }
}
