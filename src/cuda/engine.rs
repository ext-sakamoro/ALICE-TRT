//! Safe Rust wrapper for CUDA ternary inference.
//!
//! [`CudaTernaryEngine`] provides a safe, RAII-based interface to the
//! CUDA BitNet kernels. It manages device memory lifetime and provides
//! automatic kernel dispatch (Tensor Core vs CUDA Core).
//!
//! # Example
//!
//! ```no_run
//! use alice_trt::cuda::CudaTernaryEngine;
//!
//! let engine = CudaTernaryEngine::init(0).unwrap();
//! println!("{}", engine.device_name());
//! println!("Tensor Cores: {}", engine.has_tensor_cores());
//!
//! // Upload weights (2-bit bitplanes)
//! let plus_bits: Vec<u32> = vec![0b01; 4]; // example
//! let minus_bits: Vec<u32> = vec![0b00; 4];
//! let weights = engine.upload_weights(
//!     &plus_bits, &minus_bits,
//!     /*out_features=*/ 2, /*in_features=*/ 64, /*scale=*/ 1.0,
//! ).unwrap();
//!
//! // Upload input
//! let input_data = vec![1.0f32; 64];
//! let input = engine.upload_f32(&input_data).unwrap();
//!
//! // Allocate output
//! let output = engine.alloc_f32(2).unwrap();
//!
//! // Run GEMM
//! engine.gemm(&input, &weights, &output, 1, 2, 64).unwrap();
//!
//! // Download result
//! let result = engine.download_f32(&output, 2).unwrap();
//! ```

use super::ffi;
use std::ffi::{c_void, CStr};
use std::fmt;
use std::ptr;

// ============================================================================
// Error type
// ============================================================================

/// CUDA operation error.
#[derive(Debug, Clone)]
pub struct CudaError {
    pub code: i32,
    pub operation: &'static str,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {} in {}", self.code, self.operation)
    }
}

impl std::error::Error for CudaError {}

pub type CudaResult<T> = Result<T, CudaError>;

fn check(code: i32, op: &'static str) -> CudaResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError { code, operation: op })
    }
}

// ============================================================================
// Device buffer (RAII)
// ============================================================================

/// RAII wrapper for a device memory allocation.
///
/// Automatically frees device memory on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    /// Raw device pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::alice_trt_free(self.ptr); }
        }
    }
}

// ============================================================================
// Device weights (RAII)
// ============================================================================

/// RAII wrapper for ternary weight bitplanes on device.
///
/// Holds two device allocations (plus_bits, minus_bits) and the
/// weight descriptor needed by the CUDA kernel.
pub struct DeviceWeights {
    plus_buf: DeviceBuffer,
    minus_buf: DeviceBuffer,
    desc: ffi::AliceTrtWeights,
}

impl DeviceWeights {
    /// Get the weight descriptor (for FFI calls).
    pub fn descriptor(&self) -> &ffi::AliceTrtWeights {
        &self.desc
    }

    /// Output features (N).
    pub fn out_features(&self) -> i32 {
        self.desc.out_features
    }

    /// Input features (K).
    pub fn in_features(&self) -> i32 {
        self.desc.in_features
    }

    /// Scale factor.
    pub fn scale(&self) -> f32 {
        self.desc.scale
    }
}

// DeviceWeights Drop is handled by its DeviceBuffer fields

// ============================================================================
// CudaTernaryEngine
// ============================================================================

/// Safe CUDA ternary inference engine.
///
/// Wraps the C API with RAII memory management and safe dispatch.
/// Auto-dispatches: wmma (Tensor Core) → dp4a (INT8) → popc (bitwise).
/// Initialized once per GPU device, then reused for multiple inferences.
pub struct CudaTernaryEngine {
    device_info: ffi::AliceTrtDeviceInfo,
}

impl CudaTernaryEngine {
    /// Initialize the CUDA engine on the specified device.
    ///
    /// # Arguments
    /// * `device_id` — CUDA device index (0-indexed)
    ///
    /// # Errors
    /// Returns `CudaError` if the device doesn't exist or CUDA init fails.
    pub fn init(device_id: i32) -> CudaResult<Self> {
        unsafe {
            check(ffi::alice_trt_init(device_id), "alice_trt_init")?;
        }

        let mut info: ffi::AliceTrtDeviceInfo = unsafe { std::mem::zeroed() };
        unsafe {
            check(
                ffi::alice_trt_get_device_info(&mut info),
                "alice_trt_get_device_info",
            )?;
        }

        Ok(Self { device_info: info })
    }

    // ---- Device info ----

    /// GPU device name (e.g. "NVIDIA RTX 4090").
    pub fn device_name(&self) -> &str {
        let cstr = unsafe { CStr::from_ptr(self.device_info.name.as_ptr()) };
        cstr.to_str().unwrap_or("unknown")
    }

    /// CUDA compute capability major version.
    pub fn compute_major(&self) -> i32 {
        self.device_info.compute_major
    }

    /// CUDA compute capability minor version.
    pub fn compute_minor(&self) -> i32 {
        self.device_info.compute_minor
    }

    /// Number of streaming multiprocessors.
    pub fn sm_count(&self) -> i32 {
        self.device_info.sm_count
    }

    /// Total global memory in bytes.
    pub fn global_mem_bytes(&self) -> i64 {
        self.device_info.global_mem_bytes
    }

    /// Whether the GPU has Tensor Cores (compute >= 7.0).
    pub fn has_tensor_cores(&self) -> bool {
        self.device_info.has_tensor_cores != 0
    }

    /// Whether the GPU supports dp4a INT8 dot product (compute >= 6.1).
    pub fn has_dp4a(&self) -> bool {
        self.device_info.has_dp4a != 0
    }

    /// Selected kernel for auto-dispatch.
    pub fn dispatch_kernel(&self) -> &'static str {
        if self.has_tensor_cores() {
            "wmma (Tensor Core)"
        } else if self.has_dp4a() {
            "dp4a (INT8 dot product)"
        } else {
            "popc (bit-parallel popcount)"
        }
    }

    // ---- Memory management ----

    /// Allocate `count` f32 elements on the device.
    pub fn alloc_f32(&self, count: usize) -> CudaResult<DeviceBuffer> {
        let bytes = count * std::mem::size_of::<f32>();
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            check(
                ffi::alice_trt_alloc(&mut ptr, bytes),
                "alice_trt_alloc",
            )?;
        }
        Ok(DeviceBuffer { ptr, bytes })
    }

    /// Upload f32 data from host to device.
    pub fn upload_f32(&self, data: &[f32]) -> CudaResult<DeviceBuffer> {
        let buf = self.alloc_f32(data.len())?;
        unsafe {
            check(
                ffi::alice_trt_memcpy_h2d(
                    buf.ptr,
                    data.as_ptr() as *const c_void,
                    buf.bytes,
                ),
                "alice_trt_memcpy_h2d",
            )?;
        }
        Ok(buf)
    }

    /// Download f32 data from device to host.
    pub fn download_f32(&self, buf: &DeviceBuffer, count: usize) -> CudaResult<Vec<f32>> {
        let mut result = vec![0.0f32; count];
        let bytes = count * std::mem::size_of::<f32>();
        unsafe {
            check(
                ffi::alice_trt_memcpy_d2h(
                    result.as_mut_ptr() as *mut c_void,
                    buf.ptr as *const c_void,
                    bytes,
                ),
                "alice_trt_memcpy_d2h",
            )?;
        }
        Ok(result)
    }

    /// Upload ternary weight bitplanes to device.
    ///
    /// # Arguments
    /// * `plus_bits`    — Host u32 array: bit i=1 means weight[i] = +1
    /// * `minus_bits`   — Host u32 array: bit i=1 means weight[i] = -1
    /// * `out_features` — N (rows)
    /// * `in_features`  — K (columns)
    /// * `scale`        — Learned scaling factor
    pub fn upload_weights(
        &self,
        plus_bits: &[u32],
        minus_bits: &[u32],
        out_features: i32,
        in_features: i32,
        scale: f32,
    ) -> CudaResult<DeviceWeights> {
        let words_per_row = (in_features + 31) / 32;
        let total_words = out_features as usize * words_per_row as usize;
        assert_eq!(plus_bits.len(), total_words, "plus_bits length mismatch");
        assert_eq!(minus_bits.len(), total_words, "minus_bits length mismatch");

        let bytes = total_words * std::mem::size_of::<u32>();

        // Allocate + upload plus_bits
        let mut plus_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            check(ffi::alice_trt_alloc(&mut plus_ptr, bytes), "alloc plus_bits")?;
            check(
                ffi::alice_trt_memcpy_h2d(plus_ptr, plus_bits.as_ptr() as *const c_void, bytes),
                "upload plus_bits",
            )?;
        }
        let plus_buf = DeviceBuffer { ptr: plus_ptr, bytes };

        // Allocate + upload minus_bits
        let mut minus_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            check(ffi::alice_trt_alloc(&mut minus_ptr, bytes), "alloc minus_bits")?;
            check(
                ffi::alice_trt_memcpy_h2d(minus_ptr, minus_bits.as_ptr() as *const c_void, bytes),
                "upload minus_bits",
            )?;
        }
        let minus_buf = DeviceBuffer { ptr: minus_ptr, bytes };

        let desc = ffi::AliceTrtWeights {
            plus_bits: plus_buf.ptr,
            minus_bits: minus_buf.ptr,
            out_features,
            in_features,
            words_per_row,
            scale,
        };

        Ok(DeviceWeights {
            plus_buf,
            minus_buf,
            desc,
        })
    }

    /// Upload ternary weights from ALICE-ML's `TernaryWeightKernel`.
    ///
    /// Directly transfers bitplane representation without re-packing.
    pub fn upload_weights_from_kernel(
        &self,
        kernel: &alice_ml::TernaryWeightKernel,
        scale: f32,
    ) -> CudaResult<DeviceWeights> {
        self.upload_weights(
            kernel.plus_bits(),
            kernel.minus_bits(),
            kernel.out_features() as i32,
            kernel.in_features() as i32,
            scale,
        )
    }

    // ---- Compute ----

    /// Run ternary GEMM: output[M,N] = input[M,K] × W^T × scale.
    ///
    /// Auto-dispatches to Tensor Core or CUDA Core based on GPU capability.
    pub fn gemm(
        &self,
        input: &DeviceBuffer,
        weights: &DeviceWeights,
        output: &DeviceBuffer,
        m: i32,
        n: i32,
        k: i32,
    ) -> CudaResult<()> {
        unsafe {
            check(
                ffi::alice_trt_gemm(
                    input.ptr as *const f32,
                    &weights.desc,
                    output.ptr as *mut f32,
                    m, n, k,
                ),
                "alice_trt_gemm",
            )
        }
    }

    /// Force Tensor Core GEMM (requires compute >= 7.0).
    pub fn gemm_wmma(
        &self,
        input: &DeviceBuffer,
        weights: &DeviceWeights,
        output: &DeviceBuffer,
        m: i32,
        n: i32,
        k: i32,
    ) -> CudaResult<()> {
        unsafe {
            check(
                ffi::alice_trt_gemm_wmma(
                    input.ptr as *const f32,
                    &weights.desc,
                    output.ptr as *mut f32,
                    m, n, k,
                ),
                "alice_trt_gemm_wmma",
            )
        }
    }

    /// Force dp4a INT8 dot product GEMM (requires compute >= 6.1).
    pub fn gemm_dp4a(
        &self,
        input: &DeviceBuffer,
        weights: &DeviceWeights,
        output: &DeviceBuffer,
        m: i32,
        n: i32,
        k: i32,
    ) -> CudaResult<()> {
        unsafe {
            check(
                ffi::alice_trt_gemm_dp4a(
                    input.ptr as *const f32,
                    &weights.desc,
                    output.ptr as *mut f32,
                    m, n, k,
                ),
                "alice_trt_gemm_dp4a",
            )
        }
    }

    /// Force popcount bit-parallel GEMM (any GPU).
    pub fn gemm_popc(
        &self,
        input: &DeviceBuffer,
        weights: &DeviceWeights,
        output: &DeviceBuffer,
        m: i32,
        n: i32,
        k: i32,
    ) -> CudaResult<()> {
        unsafe {
            check(
                ffi::alice_trt_gemm_popc(
                    input.ptr as *const f32,
                    &weights.desc,
                    output.ptr as *mut f32,
                    m, n, k,
                ),
                "alice_trt_gemm_popc",
            )
        }
    }

    /// ReLU in-place on device buffer.
    pub fn relu_inplace(&self, data: &DeviceBuffer, len: i32) -> CudaResult<()> {
        unsafe {
            check(
                ffi::alice_trt_relu(data.ptr as *mut f32, len),
                "alice_trt_relu",
            )
        }
    }

    /// Convenience: single-vector inference (matvec).
    ///
    /// Uploads input, runs GEMM, downloads output.
    pub fn infer_matvec(
        &self,
        input: &[f32],
        weights: &DeviceWeights,
    ) -> CudaResult<Vec<f32>> {
        let k = weights.in_features();
        let n = weights.out_features();
        assert_eq!(input.len(), k as usize, "input length must equal in_features");

        let input_buf = self.upload_f32(input)?;
        let output_buf = self.alloc_f32(n as usize)?;

        self.gemm(&input_buf, weights, &output_buf, 1, n, k)?;

        self.download_f32(&output_buf, n as usize)
    }
}

impl Drop for CudaTernaryEngine {
    fn drop(&mut self) {
        unsafe { ffi::alice_trt_shutdown(); }
    }
}

impl fmt::Display for CudaTernaryEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CudaTernaryEngine({}, sm{}.{}, {}SM, {:.1}GB{})",
            self.device_name(),
            self.compute_major(),
            self.compute_minor(),
            self.sm_count(),
            self.global_mem_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            if self.has_tensor_cores() {
                ", TensorCores"
            } else if self.has_dp4a() {
                ", dp4a"
            } else {
                ", popc"
            },
        )
    }
}

// Safety: CudaTernaryEngine holds no thread-local state.
// The underlying CUDA context is process-global.
// CUDA API calls are internally synchronized.
unsafe impl Send for DeviceBuffer {}
unsafe impl Send for DeviceWeights {}
