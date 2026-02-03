/*
 * AliceTernaryPlugin::enqueue() — CUDA Dispatch
 *
 * This is where the Trojan Horse fires.
 * TensorRT calls enqueue() during inference. We dispatch to our
 * BitNet kernel that reads 2-bit weights and feeds Tensor Cores.
 *
 * Dispatch hierarchy: wmma → dp4a → popc
 */

#include "alice_ternary_plugin.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

/* wmma tile dimensions — must match bitnet_kernel.cu */
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/* Forward declare kernels from bitnet_kernel.cu (linked via CMake) */
__global__ void bitnet_wmma_gemm_kernel(
    const float* __restrict__ input,
    const uint32_t* __restrict__ plus_bits,
    const uint32_t* __restrict__ minus_bits,
    float* __restrict__ output,
    int M, int N, int K, int words_per_row, float scale);

__global__ void bitnet_dp4a_gemm_kernel(
    const float* __restrict__ input,
    const uint32_t* __restrict__ plus_bits,
    const uint32_t* __restrict__ minus_bits,
    float* __restrict__ output,
    int M, int N, int K, int words_per_row, float scale);

__global__ void bitnet_popc_gemm_kernel(
    const float* __restrict__ input,
    const uint32_t* __restrict__ plus_bits,
    const uint32_t* __restrict__ minus_bits,
    float* __restrict__ output,
    int M, int N, int K, int words_per_row, float scale);

namespace alice {

int AliceTernaryPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    /*
     * Input tensor:  [batch_dims..., in_features]
     * Output tensor: [batch_dims..., out_features]
     *
     * We flatten all batch dims into M.
     * M = product of all dims except last
     * K = in_features (last dim of input)
     * N = out_features (last dim of output)
     */

    const auto& inDims = inputDesc[0].dims;
    int K = inDims.d[inDims.nbDims - 1];

    int M = 1;
    for (int i = 0; i < inDims.nbDims - 1; i++) {
        M *= inDims.d[i];
    }

    int N = mOutFeatures;

    const float* input  = static_cast<const float*>(inputs[0]);
    float*       output = static_cast<float*>(outputs[0]);

    /*
     * Runtime kernel dispatch based on compute capability.
     * Hierarchy: wmma (sm_70+) → dp4a (sm_61+) → popc (any)
     */
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major >= 7) {
        /* Tensor Core path: 2-bit → FP16 → wmma::mma_sync */
        dim3 grid((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
        dim3 block(32);

        bitnet_wmma_gemm_kernel<<<grid, block, 0, stream>>>(
            input, mDevPlusBits, mDevMinusBits, output,
            M, N, K, mWordsPerRow, mScale
        );
    } else if (prop.major > 6 || (prop.major == 6 && prop.minor >= 1)) {
        /* dp4a path: 2-bit → INT8 → __dp4a */
        dim3 block(16, 16);
        dim3 grid((M + 15) / 16, (N + 15) / 16);

        bitnet_dp4a_gemm_kernel<<<grid, block, 0, stream>>>(
            input, mDevPlusBits, mDevMinusBits, output,
            M, N, K, mWordsPerRow, mScale
        );
    } else {
        /* popc path: bit-parallel popcount */
        dim3 block(16, 16);
        dim3 grid((M + 15) / 16, (N + 15) / 16);

        bitnet_popc_gemm_kernel<<<grid, block, 0, stream>>>(
            input, mDevPlusBits, mDevMinusBits, output,
            M, N, K, mWordsPerRow, mScale
        );
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}  /* namespace alice */
