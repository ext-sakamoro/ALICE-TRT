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
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// VRAM usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }

    /// Size in bytes (for buffer operations)
    #[inline]
    pub(crate) fn buffer_size(&self) -> u64 {
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
