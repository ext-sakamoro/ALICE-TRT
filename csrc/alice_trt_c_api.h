/*
 * ALICE-TRT C API: Rust FFI Bridge for CUDA Ternary Inference
 *
 * This header defines the C ABI that Rust calls via FFI.
 * All functions return 0 on success, non-zero on error.
 *
 * Memory model:
 *   - Device pointers are opaque (void*) to Rust
 *   - Host↔Device copies are explicit
 *   - Caller manages lifetime of device allocations
 *
 * Kernel dispatch hierarchy:
 *   1. wmma   (sm_70+) — Tensor Core, 2bit→FP16→wmma::mma_sync
 *   2. dp4a   (sm_61+) — INT8 dot product, 2bit→INT8→__dp4a
 *   3. popc   (any)    — Bit-parallel popcount, pure bitwise
 */

#ifndef ALICE_TRT_C_API_H
#define ALICE_TRT_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Weight descriptor (bitplane representation on device)
 * ================================================================ */

typedef struct {
    void*    plus_bits;      /* Device pointer: u32[] bitplane for +1 weights */
    void*    minus_bits;     /* Device pointer: u32[] bitplane for -1 weights */
    int32_t  out_features;   /* N: number of output neurons                  */
    int32_t  in_features;    /* K: number of input features                  */
    int32_t  words_per_row;  /* ceil(in_features / 32)                       */
    float    scale;          /* Learned scaling factor                        */
} AliceTrtWeights;

/* ================================================================
 * Device info
 * ================================================================ */

typedef struct {
    char     name[256];
    int32_t  compute_major;
    int32_t  compute_minor;
    int32_t  sm_count;
    int64_t  global_mem_bytes;
    int32_t  has_tensor_cores;  /* 1 if compute >= 7.0 */
    int32_t  has_dp4a;          /* 1 if compute >= 6.1 */
} AliceTrtDeviceInfo;

/* ================================================================
 * Lifecycle
 * ================================================================ */

/* Initialize CUDA context on specified device (0-indexed) */
int alice_trt_init(int32_t device_id);

/* Query device info (call after init) */
int alice_trt_get_device_info(AliceTrtDeviceInfo* info);

/* Shutdown and release CUDA context */
void alice_trt_shutdown(void);

/* ================================================================
 * Memory management
 * ================================================================ */

/* Allocate device memory. Returns device pointer in *out_ptr. */
int alice_trt_alloc(void** out_ptr, size_t bytes);

/* Free device memory */
void alice_trt_free(void* ptr);

/* Copy host → device */
int alice_trt_memcpy_h2d(void* dst_device, const void* src_host, size_t bytes);

/* Copy device → host */
int alice_trt_memcpy_d2h(void* dst_host, const void* src_device, size_t bytes);

/* ================================================================
 * Compute: Ternary GEMM
 *
 * Computes: output[M,N] = input[M,K] × W^T[K,N] × scale
 * where W is stored as 2-bit bitplanes {-1, 0, +1}
 *
 * alice_trt_gemm auto-dispatches: wmma → dp4a → popc
 * ================================================================ */

int alice_trt_gemm(
    const float*           input,    /* Device ptr: [M × K] row-major */
    const AliceTrtWeights* weights,  /* Weight descriptor              */
    float*                 output,   /* Device ptr: [M × N] row-major */
    int32_t M,                       /* Batch/rows dimension           */
    int32_t N,                       /* Output features                */
    int32_t K                        /* Input features                 */
);

/* Force Tensor Core path (requires compute >= 7.0) */
int alice_trt_gemm_wmma(
    const float*           input,
    const AliceTrtWeights* weights,
    float*                 output,
    int32_t M, int32_t N, int32_t K
);

/* Force dp4a INT8 dot product path (requires compute >= 6.1) */
int alice_trt_gemm_dp4a(
    const float*           input,
    const AliceTrtWeights* weights,
    float*                 output,
    int32_t M, int32_t N, int32_t K
);

/* Force popcount bit-parallel path (any GPU) */
int alice_trt_gemm_popc(
    const float*           input,
    const AliceTrtWeights* weights,
    float*                 output,
    int32_t M, int32_t N, int32_t K
);

/* ================================================================
 * Compute: Activations (in-place)
 * ================================================================ */

/* ReLU: data[i] = max(data[i], 0) */
int alice_trt_relu(float* data, int32_t len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ALICE_TRT_C_API_H */
