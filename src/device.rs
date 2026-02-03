//! GPU Device Management
//!
//! Wraps wgpu initialization into a single struct.
//! Requests high-performance adapter by default.

use wgpu;

/// GPU device handle for all ALICE-TRT operations
///
/// Holds the wgpu Device and Queue. All GPU buffers and pipelines
/// are created through this handle.
pub struct GpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: String,
}

impl GpuDevice {
    /// Initialize GPU with high-performance adapter
    ///
    /// # Errors
    /// Returns error string if no GPU adapter is available.
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    /// Async initialization (for custom runtimes)
    pub async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No GPU adapter found".to_string())?;

        let adapter_info = adapter.get_info();
        let info = format!(
            "{} ({:?}, {:?})",
            adapter_info.name, adapter_info.backend, adapter_info.device_type
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ALICE-TRT"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {e}"))?;

        Ok(Self { device, queue, info })
    }

    /// Get wgpu device reference
    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get wgpu queue reference
    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// GPU adapter info string
    #[inline]
    pub fn info(&self) -> &str {
        &self.info
    }

    /// Create a storage buffer with initial data
    pub fn create_buffer_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create a uniform buffer with initial data
    pub fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create an empty storage buffer
    pub fn create_buffer_empty(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Submit a command encoder
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit([encoder.finish()]);
    }

    /// Poll device until all operations complete
    pub fn poll_wait(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Read buffer contents back to CPU
    pub fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_back"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.submit(encoder);
        self.poll_wait();

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.poll_wait();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging.unmap();

        result
    }
}

impl std::fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ALICE-TRT [{}]", self.info)
    }
}
