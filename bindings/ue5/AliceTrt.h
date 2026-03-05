// ALICE-TRT UE5 C++ Bindings
// 37 extern "C" + 5 RAII unique_ptr handles
//
// Author: Moroya Sakamoto

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Alice { namespace Trt {

// ============================================================================
// Opaque types
// ============================================================================

struct GpuDevice;
struct GpuTernaryWeight;
struct GpuTensor;
struct TernaryCompute;
struct GpuInferenceEngine;

// ============================================================================
// Raw C API (37 functions)
// ============================================================================

extern "C" {

// Version (1)
const char* at_trt_version();

// GpuDevice (3)
GpuDevice* at_trt_device_new();
void at_trt_device_free(GpuDevice* ptr);
uint32_t at_trt_device_info(const GpuDevice* ptr, char* buf, uint32_t buf_len);

// GpuTernaryWeight (9)
GpuTernaryWeight* at_trt_weight_from_ternary(const GpuDevice* device, const int8_t* values, uint32_t out_features, uint32_t in_features);
GpuTernaryWeight* at_trt_weight_from_ternary_scaled(const GpuDevice* device, const int8_t* values, uint32_t out_features, uint32_t in_features, float scale);
void at_trt_weight_free(GpuTernaryWeight* ptr);
uint32_t at_trt_weight_out_features(const GpuTernaryWeight* ptr);
uint32_t at_trt_weight_in_features(const GpuTernaryWeight* ptr);
float at_trt_weight_scale(const GpuTernaryWeight* ptr);
uint32_t at_trt_weight_words_per_row(const GpuTernaryWeight* ptr);
uint64_t at_trt_weight_memory_bytes(const GpuTernaryWeight* ptr);
float at_trt_weight_compression_ratio(const GpuTernaryWeight* ptr);

// GpuTensor (10)
GpuTensor* at_trt_tensor_from_f32(const GpuDevice* device, const float* data, uint32_t data_len, const uint32_t* shape, uint32_t ndim);
GpuTensor* at_trt_tensor_zeros(const GpuDevice* device, const uint32_t* shape, uint32_t ndim);
GpuTensor* at_trt_tensor_output(const GpuDevice* device, const uint32_t* shape, uint32_t ndim);
void at_trt_tensor_free(GpuTensor* ptr);
uint32_t at_trt_tensor_download(const GpuTensor* tensor, const GpuDevice* device, float* out, uint32_t max_len);
uint32_t at_trt_tensor_ndim(const GpuTensor* ptr);
uint32_t at_trt_tensor_shape(const GpuTensor* ptr, uint32_t dim);
uint32_t at_trt_tensor_len(const GpuTensor* ptr);
int32_t at_trt_tensor_is_empty(const GpuTensor* ptr);
uint64_t at_trt_tensor_memory_bytes(const GpuTensor* ptr);

// TernaryCompute (5)
TernaryCompute* at_trt_compute_new(const GpuDevice* device);
void at_trt_compute_free(TernaryCompute* ptr);
GpuTensor* at_trt_compute_matvec(const TernaryCompute* compute, const GpuDevice* device, const GpuTensor* input, const GpuTernaryWeight* weights);
GpuTensor* at_trt_compute_matmul_batch(const TernaryCompute* compute, const GpuDevice* device, const GpuTensor* input, const GpuTernaryWeight* weights, uint32_t batch_size);
void at_trt_compute_relu_inplace(const TernaryCompute* compute, const GpuDevice* device, const GpuTensor* tensor);

// GpuInferenceEngine (9)
GpuInferenceEngine* at_trt_engine_new();
void at_trt_engine_free(GpuInferenceEngine* ptr);
void at_trt_engine_add_layer(GpuInferenceEngine* engine, GpuTernaryWeight* weight, uint32_t activation);
uint32_t at_trt_engine_num_layers(const GpuInferenceEngine* ptr);
uint64_t at_trt_engine_total_weight_bytes(const GpuInferenceEngine* ptr);
uint64_t at_trt_engine_equivalent_fp32_bytes(const GpuInferenceEngine* ptr);
float at_trt_engine_compression_ratio(const GpuInferenceEngine* ptr);
GpuTensor* at_trt_engine_forward(const GpuInferenceEngine* engine, const GpuDevice* device, const TernaryCompute* compute, const GpuTensor* input);
GpuTensor* at_trt_engine_forward_batch(const GpuInferenceEngine* engine, const GpuDevice* device, const TernaryCompute* compute, const GpuTensor* input, uint32_t batch_size);

} // extern "C"

// ============================================================================
// RAII Handles (5)
// ============================================================================

struct GpuDeviceDeleter { void operator()(GpuDevice* p) const { at_trt_device_free(p); } };
struct GpuWeightDeleter { void operator()(GpuTernaryWeight* p) const { at_trt_weight_free(p); } };
struct GpuTensorDeleter { void operator()(GpuTensor* p) const { at_trt_tensor_free(p); } };
struct ComputeDeleter { void operator()(TernaryCompute* p) const { at_trt_compute_free(p); } };
struct EngineDeleter { void operator()(GpuInferenceEngine* p) const { at_trt_engine_free(p); } };

using GpuDevicePtr = std::unique_ptr<GpuDevice, GpuDeviceDeleter>;
using GpuWeightPtr = std::unique_ptr<GpuTernaryWeight, GpuWeightDeleter>;
using GpuTensorPtr = std::unique_ptr<GpuTensor, GpuTensorDeleter>;
using ComputePtr = std::unique_ptr<TernaryCompute, ComputeDeleter>;
using EnginePtr = std::unique_ptr<GpuInferenceEngine, EngineDeleter>;

// ============================================================================
// Helper: create RAII handles
// ============================================================================

inline GpuDevicePtr MakeDevice() {
    return GpuDevicePtr(at_trt_device_new());
}

inline std::string DeviceInfo(const GpuDevicePtr& dev) {
    char buf[512];
    at_trt_device_info(dev.get(), buf, 512);
    return std::string(buf);
}

inline GpuWeightPtr MakeWeight(const GpuDevicePtr& dev, const int8_t* values, uint32_t out, uint32_t in_f) {
    return GpuWeightPtr(at_trt_weight_from_ternary(dev.get(), values, out, in_f));
}

inline GpuWeightPtr MakeWeightScaled(const GpuDevicePtr& dev, const int8_t* values, uint32_t out, uint32_t in_f, float scale) {
    return GpuWeightPtr(at_trt_weight_from_ternary_scaled(dev.get(), values, out, in_f, scale));
}

inline GpuTensorPtr MakeTensor(const GpuDevicePtr& dev, const float* data, uint32_t len, const uint32_t* shape, uint32_t ndim) {
    return GpuTensorPtr(at_trt_tensor_from_f32(dev.get(), data, len, shape, ndim));
}

inline GpuTensorPtr MakeTensorZeros(const GpuDevicePtr& dev, const uint32_t* shape, uint32_t ndim) {
    return GpuTensorPtr(at_trt_tensor_zeros(dev.get(), shape, ndim));
}

inline std::vector<float> Download(const GpuTensorPtr& t, const GpuDevicePtr& dev) {
    uint32_t len = at_trt_tensor_len(t.get());
    std::vector<float> buf(len);
    at_trt_tensor_download(t.get(), dev.get(), buf.data(), len);
    return buf;
}

inline ComputePtr MakeCompute(const GpuDevicePtr& dev) {
    return ComputePtr(at_trt_compute_new(dev.get()));
}

inline GpuTensorPtr Matvec(const ComputePtr& c, const GpuDevicePtr& dev, const GpuTensorPtr& in, const GpuWeightPtr& w) {
    return GpuTensorPtr(at_trt_compute_matvec(c.get(), dev.get(), in.get(), w.get()));
}

inline EnginePtr MakeEngine() {
    return EnginePtr(at_trt_engine_new());
}

/// AddLayer consumes weight — do not use weight after this call
inline void AddLayer(const EnginePtr& eng, GpuWeightPtr weight, uint32_t activation) {
    at_trt_engine_add_layer(eng.get(), weight.release(), activation);
}

inline GpuTensorPtr Forward(const EnginePtr& eng, const GpuDevicePtr& dev, const ComputePtr& c, const GpuTensorPtr& in) {
    return GpuTensorPtr(at_trt_engine_forward(eng.get(), dev.get(), c.get(), in.get()));
}

inline GpuTensorPtr ForwardBatch(const EnginePtr& eng, const GpuDevicePtr& dev, const ComputePtr& c, const GpuTensorPtr& in, uint32_t batch) {
    return GpuTensorPtr(at_trt_engine_forward_batch(eng.get(), dev.get(), c.get(), in.get(), batch));
}

}} // namespace Alice::Trt
