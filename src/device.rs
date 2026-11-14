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
    ///
    /// # Errors
    ///
    /// Returns an error string if GPU adapter or device request fails.
    pub async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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

        Ok(Self {
            device,
            queue,
            info,
        })
    }

    /// Get wgpu device reference
    #[inline]
    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get wgpu queue reference
    #[inline]
    #[must_use]
    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// GPU adapter info string
    #[inline]
    #[must_use]
    pub fn info(&self) -> &str {
        &self.info
    }

    /// Create a storage buffer with initial data
    #[must_use]
    pub fn create_buffer_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Create a uniform buffer with initial data
    #[must_use]
    pub fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Create an empty storage buffer
    #[must_use]
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
    ///
    /// # Panics
    ///
    /// Panics if the GPU readback channel is closed unexpectedly.
    pub fn poll_wait(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Read buffer contents back to CPU
    ///
    /// # Panics
    ///
    /// Panics if the GPU readback channel is closed or the map operation fails.
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
            // SAFETY: Channel receiver is alive — drop only after recv() below
            tx.send(result)
                .expect("GPU readback channel closed unexpectedly");
        });
        self.poll_wait();
        rx.recv()
            .expect("GPU map_async callback was never invoked — device may be lost")
            .expect("GPU buffer mapping failed — BufferAsyncError");

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

/// The GPU device for a unit test that needs one.
///
/// Without an adapter the test is skipped (returns `None`) and a
/// `[gpu] test skipped: no GPU adapter` line is printed. When the
/// environment variable `TRT_REQUIRE_GPU` is set (the GPU jobs of CI set it),
/// a missing adapter panics instead, so a runner that loses its adapter turns
/// red rather than passing every GPU test as a skip. Every acquired device
/// prints `[gpu] test device acquired`; CI counts these lines.
///
/// `TRT_TEST_FORCE_NO_ADAPTER` makes the adapter look missing; CI uses it to
/// check that the requirement actually fails the run.
///
/// The device holds a process-wide lock for as long as the test keeps it, so
/// tests that take one run one at a time whatever `--test-threads` says:
/// devices created and used from parallel threads hang the tests on NVIDIA's
/// proprietary Vulkan driver (A6000 and P5000) and crashed the test process on
/// the Windows runner's adapter (`STATUS_ACCESS_VIOLATION` in the WGSL compile
/// tests, which ran with the default thread count); Apple Metal shows neither.
/// `scripts/cargo_test_gpu.sh` still passes `--test-threads=1` as well.
/// The lock [`test_device`] holds for as long as a test keeps its device
#[cfg(test)]
static GPU_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn test_device() -> Option<TestDevice> {
    // a test that panicked while holding the device leaves the lock poisoned; the next test still runs
    let lock = GPU_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = if std::env::var_os("TRT_TEST_FORCE_NO_ADAPTER").is_some() {
        Err("adapter hidden by TRT_TEST_FORCE_NO_ADAPTER".to_string())
    } else {
        GpuDevice::new()
    };
    match result {
        Ok(device) => {
            eprintln!("[gpu] test device acquired");
            Some(TestDevice {
                device,
                _lock: lock,
            })
        }
        Err(e) => {
            assert!(
                std::env::var_os("TRT_REQUIRE_GPU").is_none(),
                "TRT_REQUIRE_GPU is set but no GPU adapter is available: {e}"
            );
            eprintln!("[gpu] test skipped: no GPU adapter ({e})");
            None
        }
    }
}

/// A test's GPU device and the lock that keeps other tests from using a device
/// at the same time (see [`test_device`]); the device is dropped before the
/// lock is released
#[cfg(test)]
pub(crate) struct TestDevice {
    device: GpuDevice,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(all(test, feature = "physics-solver"))]
impl TestDevice {
    /// The device in an `Arc` for code that takes shared ownership, still
    /// holding the lock; the `Arc` is dropped before the lock is released
    pub(crate) fn shared(self) -> SharedTestDevice<GpuDevice> {
        SharedTestDevice {
            device: std::sync::Arc::new(self.device),
            _lock: self._lock,
        }
    }
}

/// [`TestDevice::shared`]: the device in an `Arc` and the lock, in that field
/// order so the `Arc` is dropped first (a tuple bound as `let (device, lock)`
/// would release the lock first, the bindings drop in reverse order)
#[cfg(all(test, feature = "physics-solver"))]
pub(crate) struct SharedTestDevice<D> {
    device: std::sync::Arc<D>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(all(test, feature = "physics-solver"))]
impl<D> std::ops::Deref for SharedTestDevice<D> {
    type Target = std::sync::Arc<D>;

    fn deref(&self) -> &std::sync::Arc<D> {
        &self.device
    }
}

#[cfg(test)]
impl std::ops::Deref for TestDevice {
    type Target = GpuDevice;

    fn deref(&self) -> &GpuDevice {
        &self.device
    }
}

#[cfg(test)]
mod test_device_lock_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Tests that take a GPU device never hold one at the same time, whatever
    /// the harness's thread count: 4 threads each take a device, note how many
    /// are held at once, and keep it a moment; the most held at once is 1
    #[test]
    fn devices_are_held_one_test_at_a_time() {
        static HELD: AtomicUsize = AtomicUsize::new(0);
        static MOST: AtomicUsize = AtomicUsize::new(0);
        let acquired: usize = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        let Some(device) = super::test_device() else {
                            return 0;
                        };
                        let now = HELD.fetch_add(1, Ordering::SeqCst) + 1;
                        MOST.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        let _ = device.device();
                        HELD.fetch_sub(1, Ordering::SeqCst);
                        1
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("thread")).sum()
        });
        if acquired == 0 {
            eprintln!("[gpu] test skipped: no GPU adapter");
            return;
        }
        assert_eq!(acquired, 4);
        assert_eq!(
            MOST.load(Ordering::SeqCst),
            1,
            "devices were held at the same time"
        );
    }

    /// The shared form keeps the same guarantee: its `Arc` is dropped while
    /// the lock is still held. The device is a probe whose drop checks that the
    /// lock cannot be taken (a field order that released the lock first fails
    /// here every time, not only when another test happens to be waiting)
    #[cfg(feature = "physics-solver")]
    #[test]
    fn a_shared_device_is_dropped_before_the_lock_is_released() {
        struct Probe(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Probe {
            fn drop(&mut self) {
                let held = super::GPU_TESTS.try_lock().is_err();
                self.0.store(held, Ordering::SeqCst);
            }
        }
        let checked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let shared = super::SharedTestDevice {
            device: std::sync::Arc::new(Probe(std::sync::Arc::clone(&checked))),
            _lock: super::GPU_TESTS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        };
        let _ = &**shared;
        drop(shared);
        assert!(
            checked.load(Ordering::SeqCst),
            "the lock was released while the device was still alive"
        );
    }
}
