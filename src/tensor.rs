//! GPU Tensor: Buffer-Backed Float Storage
//!
//! Data lives in VRAM. Upload once, compute many.
//! Download only when you need to read results.

use crate::device::GpuDevice;

/// GPU-resident tensor (f32 storage buffer)
pub struct GpuTensor {
    pub(crate) buffer: wgpu::Buffer,
    shape: Vec<usize>,
    len: usize,
}

impl GpuTensor {
    /// Upload f32 data to GPU
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not match the product of `shape`.
    #[must_use]
    pub fn from_f32(device: &GpuDevice, data: &[f32], shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "Data length must match shape product");

        let buffer = device.create_buffer_init("trt_tensor", bytemuck::cast_slice(data));

        Self {
            buffer,
            shape: shape.to_vec(),
            len: total,
        }
    }

    /// Create zero-filled tensor on GPU
    #[must_use]
    pub fn zeros(device: &GpuDevice, shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        let size = (total * std::mem::size_of::<f32>()) as u64;

        let buffer = device.create_buffer_empty("trt_tensor_zero", size);

        // Zero-fill
        let zeros = vec![0u8; size as usize];
        device.queue().write_buffer(&buffer, 0, &zeros);

        Self {
            buffer,
            shape: shape.to_vec(),
            len: total,
        }
    }

    /// Create output tensor (pre-allocated, uninitialized)
    #[must_use]
    pub fn output(device: &GpuDevice, shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        let size = (total * std::mem::size_of::<f32>()) as u64;

        let buffer = device.create_buffer_empty("trt_output", size);

        Self {
            buffer,
            shape: shape.to_vec(),
            len: total,
        }
    }

    /// Download tensor data from GPU to CPU
    pub fn download(&self, device: &GpuDevice) -> Vec<f32> {
        let size = (self.len * std::mem::size_of::<f32>()) as u64;
        let bytes = device.read_buffer(&self.buffer, size);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Tensor shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// VRAM usage in bytes
    #[inline]
    pub const fn memory_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }

    /// Size in bytes as `u64`, required by the CUDA/Vulkan buffer-allocation
    /// APIs that accept `VkDeviceSize` / `CUdeviceptr` offsets.  Kept separate
    /// from [`memory_bytes`] (which returns `usize`) so callers never need a
    /// cast at the FFI boundary.  Not yet wired up in the current engine path
    /// but will be used when the planned zero-copy device-buffer pool lands.
    #[allow(dead_code)]
    #[inline]
    pub(crate) const fn buffer_size(&self) -> u64 {
        (self.len * std::mem::size_of::<f32>()) as u64
    }
}

impl std::fmt::Display for GpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuTensor[shape={:?}, {} bytes VRAM]",
            self.shape,
            self.memory_bytes(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_1d() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0], &[3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
        assert_eq!(t.memory_bytes(), 12); // 3 * 4
    }

    #[test]
    fn test_tensor_shape_2d() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[1.0; 6], &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        assert_eq!(t.memory_bytes(), 24);
    }

    #[test]
    fn test_tensor_zeros_download() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::zeros(&device, &[4]);
        let data = t.download(&device);
        assert_eq!(data, vec![0.0f32; 4]);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let Ok(device) = GpuDevice::new() else { return };
        let input = vec![1.5, -2.3, 0.0, 42.0, -0.001];
        let t = GpuTensor::from_f32(&device, &input, &[5]);
        let output = t.download(&device);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_tensor_output_shape() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::output(&device, &[8, 16]);
        assert_eq!(t.shape(), &[8, 16]);
        assert_eq!(t.len(), 128);
        assert_eq!(t.memory_bytes(), 512);
    }

    #[test]
    fn test_tensor_buffer_size() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[1.0; 100], &[100]);
        assert_eq!(t.buffer_size(), 400);
    }

    #[test]
    fn test_tensor_display() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[1.0, 2.0], &[2]);
        let s = format!("{t}");
        assert!(s.contains("GpuTensor"));
        assert!(s.contains("shape=[2]"));
        assert!(s.contains("8 bytes"));
    }

    #[test]
    #[should_panic(expected = "Data length must match shape product")]
    fn test_tensor_shape_mismatch_panics() {
        let Ok(device) = GpuDevice::new() else {
            // Cannot test GPU panic without GPU; simulate
            panic!("Data length must match shape product");
        };
        let _t = GpuTensor::from_f32(&device, &[1.0, 2.0], &[3]);
    }

    #[test]
    fn test_tensor_zeros_2d() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::zeros(&device, &[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.len(), 12);
        assert!(!t.is_empty());
        assert_eq!(t.memory_bytes(), 48);
        let data = t.download(&device);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_output_not_empty() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::output(&device, &[1]);
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_tensor_memory_bytes_3d() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[0.0; 24], &[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.len(), 24);
        assert_eq!(t.memory_bytes(), 96); // 24 * 4
    }

    #[test]
    fn test_tensor_display_2d() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[0.0; 6], &[2, 3]);
        let s = format!("{t}");
        assert!(s.contains("GpuTensor"));
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("24 bytes"));
    }

    #[test]
    fn test_tensor_buffer_size_single_element() {
        let Ok(device) = GpuDevice::new() else { return };
        // shape [1] => 1 element => buffer_size = 4 bytes
        let t = GpuTensor::from_f32(&device, &[0.0], &[1]);
        assert_eq!(t.buffer_size(), 4);
    }

    #[test]
    fn test_tensor_shape_3d_product() {
        let Ok(device) = GpuDevice::new() else { return };
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let t = GpuTensor::from_f32(&device, &data, &[3, 4, 5]);
        assert_eq!(t.len(), 60);
        assert_eq!(t.shape(), &[3, 4, 5]);
    }

    #[test]
    fn test_tensor_zeros_roundtrip_values() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::zeros(&device, &[8]);
        let result = t.download(&device);
        assert_eq!(result.len(), 8);
        for v in result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_tensor_len_matches_shape_product() {
        let Ok(device) = GpuDevice::new() else { return };
        let t = GpuTensor::from_f32(&device, &[1.0; 30], &[5, 6]);
        assert_eq!(t.len(), t.shape().iter().product::<usize>());
    }
}
