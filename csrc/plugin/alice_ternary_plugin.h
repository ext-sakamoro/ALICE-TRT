/*
 * ALICE Ternary Plugin for TensorRT
 *
 * IPluginV2DynamicExt implementation that replaces standard GEMM layers
 * with ALICE's 2-bit ternary GEMM kernel.
 *
 * Usage: Load an ONNX model via TensorRT, and replace target MatMul/Linear
 * layers with this plugin. The plugin stores weights as 2-bit bitplanes
 * and dispatches to the BitNet CUDA kernel on enqueue().
 *
 * Requires: TensorRT >= 8.0, CUDA >= 11.0
 */

#ifndef ALICE_TERNARY_PLUGIN_H
#define ALICE_TERNARY_PLUGIN_H

#include <NvInfer.h>
#include <string>
#include <vector>
#include <cstdint>

namespace alice {

/* ================================================================
 * Plugin: AliceTernaryGemm
 *
 * Replaces a Dense/Linear/MatMul layer.
 * Input:  FP32 tensor [batch..., in_features]
 * Output: FP32 tensor [batch..., out_features]
 * ================================================================ */

class AliceTernaryPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    /* Construct from explicit weight data (for plugin creator) */
    AliceTernaryPlugin(
        const uint32_t* plus_bits,
        const uint32_t* minus_bits,
        int out_features,
        int in_features,
        float scale
    );

    /* Construct from serialized data (for deserialization) */
    AliceTernaryPlugin(const void* data, size_t length);

    ~AliceTernaryPlugin() override;

    /* ---- IPluginV2DynamicExt ---- */

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;

    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs
    ) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs
    ) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs
    ) const noexcept override;

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

    /* ---- IPluginV2Ext ---- */

    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs
    ) const noexcept override;

    /* ---- IPluginV2 ---- */

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

private:
    std::string mNamespace;

    /* Weight data (host copies for serialization) */
    std::vector<uint32_t> mPlusBits;
    std::vector<uint32_t> mMinusBits;
    int mOutFeatures;
    int mInFeatures;
    int mWordsPerRow;
    float mScale;

    /* Device copies (allocated in initialize, freed in terminate) */
    uint32_t* mDevPlusBits  = nullptr;
    uint32_t* mDevMinusBits = nullptr;
};


/* ================================================================
 * Plugin Creator
 * ================================================================ */

class AliceTernaryPluginCreator : public nvinfer1::IPluginCreator
{
public:
    AliceTernaryPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength
    ) noexcept override;

    void setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mFields;
};

}  /* namespace alice */

#endif /* ALICE_TERNARY_PLUGIN_H */
