//! Raw FFI bindings to `csrc/alice_trt_c_api.h`
//!
//! These are unsafe `extern "C"` declarations that match the C header exactly.
//! Use [`super::engine::CudaTernaryEngine`] for safe wrappers.

use std::ffi::c_void;
use std::os::raw::c_char;

/// Weight descriptor (bitplane representation on device).
///
/// Mirrors `AliceTrtWeights` from `alice_trt_c_api.h`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AliceTrtWeights {
    /// Device pointer: u32[] bitplane for +1 weights
    pub plus_bits: *mut c_void,
    /// Device pointer: u32[] bitplane for -1 weights
    pub minus_bits: *mut c_void,
    /// N: number of output neurons
    pub out_features: i32,
    /// K: number of input features
    pub in_features: i32,
    /// ceil(in_features / 32)
    pub words_per_row: i32,
    /// Learned scaling factor
    pub scale: f32,
}

/// Device information.
///
/// Mirrors `AliceTrtDeviceInfo` from `alice_trt_c_api.h`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AliceTrtDeviceInfo {
    pub name: [c_char; 256],
    pub compute_major: i32,
    pub compute_minor: i32,
    pub sm_count: i32,
    pub global_mem_bytes: i64,
    /// 1 if compute >= 7.0
    pub has_tensor_cores: i32,
    /// 1 if compute >= 6.1 (dp4a available)
    pub has_dp4a: i32,
}

extern "C" {
    // ---- Lifecycle ----

    /// Initialize CUDA context on specified device (0-indexed).
    /// Returns 0 on success.
    pub fn alice_trt_init(device_id: i32) -> i32;

    /// Query device info (call after init).
    /// Returns 0 on success.
    pub fn alice_trt_get_device_info(info: *mut AliceTrtDeviceInfo) -> i32;

    /// Shutdown and release CUDA context.
    pub fn alice_trt_shutdown();

    // ---- Memory management ----

    /// Allocate device memory. Returns device pointer in `*out_ptr`.
    /// Returns 0 on success.
    pub fn alice_trt_alloc(out_ptr: *mut *mut c_void, bytes: usize) -> i32;

    /// Free device memory.
    pub fn alice_trt_free(ptr: *mut c_void);

    /// Copy host → device. Returns 0 on success.
    pub fn alice_trt_memcpy_h2d(
        dst_device: *mut c_void,
        src_host: *const c_void,
        bytes: usize,
    ) -> i32;

    /// Copy device → host. Returns 0 on success.
    pub fn alice_trt_memcpy_d2h(
        dst_host: *mut c_void,
        src_device: *const c_void,
        bytes: usize,
    ) -> i32;

    // ---- Compute: Ternary GEMM ----

    /// Auto-dispatch GEMM: wmma → dp4a → popc.
    /// output[M,N] = input[M,K] × W^T[K,N] × scale.
    /// Returns 0 on success.
    pub fn alice_trt_gemm(
        input: *const f32,
        weights: *const AliceTrtWeights,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;

    /// Force Tensor Core path (requires compute >= 7.0).
    /// Returns 0 on success.
    pub fn alice_trt_gemm_wmma(
        input: *const f32,
        weights: *const AliceTrtWeights,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;

    /// Force dp4a INT8 dot product path (requires compute >= 6.1).
    /// Returns 0 on success.
    pub fn alice_trt_gemm_dp4a(
        input: *const f32,
        weights: *const AliceTrtWeights,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;

    /// Force popcount bit-parallel path (any GPU).
    /// Returns 0 on success.
    pub fn alice_trt_gemm_popc(
        input: *const f32,
        weights: *const AliceTrtWeights,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> i32;

    // ---- Compute: Activations ----

    /// ReLU in-place: data[i] = max(data[i], 0).
    /// Returns 0 on success.
    pub fn alice_trt_relu(data: *mut f32, len: i32) -> i32;
}
