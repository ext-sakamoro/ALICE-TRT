//! GPU Ternary Weights: 2-bit Bitplane Storage in VRAM
//!
//! **The Memory Hack**: Weights are stored as two bitplanes (`plus_bits`, `minus_bits`)
//! in GPU storage buffers. Each u32 word encodes 32 ternary weights.
//!
//! VRAM usage: 2 bits per weight (vs 16 bits FP16, 8 bits INT8)
//! Bandwidth reduction: 8x vs FP16, 4x vs INT8

use crate::device::GpuDevice;
use crate::kernel::GpuParams;

/// Ternary weights on GPU (bitplane representation)
///
/// Two storage buffers:
/// - `plus_bits[row * words_per_row + word]`: bit i = 1 means `weight[row][word*32+i] = +1`
/// - `minus_bits[row * words_per_row + word]`: bit i = 1 means `weight[row][word*32+i] = -1`
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
    /// Upload from ALICE-ML `TernaryWeightKernel` (bitplane format)
    ///
    /// Zero-copy compatible: `TernaryWeightKernel` already stores bitplanes.
    #[must_use]
    pub fn from_kernel(device: &GpuDevice, kernel: &alice_ml::TernaryWeightKernel) -> Self {
        let out_features = kernel.out_features();
        let in_features = kernel.in_features();
        let words_per_row = kernel.words_per_row();
        let scale = kernel.scale();

        let plus_buffer =
            device.create_buffer_init("trt_plus_bits", bytemuck::cast_slice(kernel.plus_bits()));
        let minus_buffer =
            device.create_buffer_init("trt_minus_bits", bytemuck::cast_slice(kernel.minus_bits()));

        let params = GpuParams {
            in_features: in_features as u32,
            out_features: out_features as u32,
            words_per_row: words_per_row as u32,
            scale,
            batch_size: 1,
            padding: [0; 3],
        };
        let params_buffer = device.create_uniform_buffer("trt_params", bytemuck::bytes_of(&params));

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

    /// Upload from ALICE-ML `TernaryWeight` (packed 2-bit format)
    ///
    /// Converts to bitplane format internally.
    #[must_use]
    pub fn from_packed(device: &GpuDevice, weights: &alice_ml::TernaryWeight) -> Self {
        let kernel = alice_ml::TernaryWeightKernel::from_packed_weight(weights);
        Self::from_kernel(device, &kernel)
    }

    /// Create directly from ternary values (-1, 0, +1)
    #[must_use]
    pub fn from_ternary(
        device: &GpuDevice,
        values: &[i8],
        out_features: usize,
        in_features: usize,
    ) -> Self {
        let kernel = alice_ml::TernaryWeightKernel::from_ternary(values, out_features, in_features);
        Self::from_kernel(device, &kernel)
    }

    /// Create with custom scale
    #[must_use]
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
            padding: [0; 3],
        };
        device
            .queue()
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    #[inline(always)]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    #[inline(always)]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    #[inline(always)]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    #[inline(always)]
    pub const fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    /// VRAM usage in bytes (bitplanes only, excludes params)
    #[inline(always)]
    pub const fn memory_bytes(&self) -> usize {
        let total_words = self.out_features * self.words_per_row;
        total_words * 4 * 2 // plus + minus bitplanes, 4 bytes per u32
    }

    /// Compression ratio vs FP32
    #[inline(always)]
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.out_features * self.in_features * 4;
        let mem = self.memory_bytes();
        if mem == 0 {
            return 0.0;
        }
        let inv_mem = 1.0 / mem as f32;
        fp32_size as f32 * inv_mem
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

#[cfg(test)]
mod tests {
    use super::*;

    // words_per_row = ceil(in_features / 32)
    // memory_bytes  = out_features * words_per_row * 4 * 2

    #[test]
    fn test_weight_memory_bytes_2x2() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        // words_per_row = ceil(2/32) = 1
        // memory_bytes  = 2 * 1 * 4 * 2 = 16
        assert_eq!(w.memory_bytes(), 16);
    }

    #[test]
    fn test_weight_words_per_row_exact_32() {
        let Ok(device) = GpuDevice::new() else { return };
        let values: Vec<i8> = vec![1; 32];
        let w = GpuTernaryWeight::from_ternary(&device, &values, 1, 32);
        // exactly 32 cols => 1 word
        assert_eq!(w.words_per_row(), 1);
        assert_eq!(w.in_features(), 32);
    }

    #[test]
    fn test_weight_words_per_row_33_cols() {
        let Ok(device) = GpuDevice::new() else { return };
        let values: Vec<i8> = vec![1; 33];
        let w = GpuTernaryWeight::from_ternary(&device, &values, 1, 33);
        // 33 cols => ceil(33/32) = 2 words
        assert_eq!(w.words_per_row(), 2);
    }

    #[test]
    fn test_weight_compression_ratio_1x1() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1], 1, 1);
        // fp32 = 1*1*4 = 4 bytes; bitplane = 1*1*4*2 = 8 bytes
        // ratio = 4/8 = 0.5
        let r = w.compression_ratio();
        assert!(r > 0.0 && r < 1.1, "ratio = {r}");
    }

    #[test]
    fn test_weight_compression_ratio_wide() {
        let Ok(device) = GpuDevice::new() else { return };
        // 1 row, 256 cols: words_per_row = 8
        let values: Vec<i8> = vec![1; 256];
        let w = GpuTernaryWeight::from_ternary(&device, &values, 1, 256);
        // fp32  = 1 * 256 * 4 = 1024 bytes
        // bits  = 1 * 8 * 4 * 2 = 64 bytes
        // ratio = 16.0
        let r = w.compression_ratio();
        assert!((r - 16.0).abs() < 0.01, "ratio = {r}");
    }

    #[test]
    fn test_weight_scale_custom() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary_scaled(&device, &[1, -1, 0, 1], 2, 2, 2.5);
        assert!((w.scale() - 2.5).abs() < 1e-5, "scale = {}", w.scale());
    }

    #[test]
    fn test_weight_display_contains_dimensions() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[0; 6], 2, 3);
        let s = format!("{w}");
        assert!(s.contains("2x3"), "display = {s}");
        assert!(s.contains("compression"), "display = {s}");
        assert!(s.contains("VRAM"), "display = {s}");
    }

    #[test]
    fn test_weight_display_contains_bytes() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1; 4], 2, 2);
        let s = format!("{w}");
        assert!(s.contains("bytes VRAM"), "display = {s}");
    }

    #[test]
    fn test_weight_scale_default_is_one() {
        let Ok(device) = GpuDevice::new() else { return };
        let w = GpuTernaryWeight::from_ternary(&device, &[1, 0, -1, 1], 2, 2);
        assert!((w.scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_out_in_features_non_square() {
        let Ok(device) = GpuDevice::new() else { return };
        let values: Vec<i8> = vec![1; 15];
        let w = GpuTernaryWeight::from_ternary(&device, &values, 3, 5);
        assert_eq!(w.out_features(), 3);
        assert_eq!(w.in_features(), 5);
    }
}
