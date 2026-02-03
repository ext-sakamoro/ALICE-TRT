/*
 * AliceTernaryPlugin — TensorRT Plugin Registration & Serialization
 *
 * This file implements all non-CUDA methods of the plugin:
 * - Construction, cloning, destruction
 * - Serialization / deserialization (2-bit weights as byte stream)
 * - Output dimension calculation
 * - Format support (FP32 only for now)
 * - Plugin creator and factory registration
 *
 * The enqueue() method (CUDA dispatch) is in alice_ternary_plugin.cu.
 */

#include "alice_ternary_plugin.h"
#include <cstring>
#include <cassert>

namespace alice {

static const char* PLUGIN_NAME    = "AliceTernaryGemm";
static const char* PLUGIN_VERSION = "1";

/* ================================================================
 * Helper: simple serialization into a byte buffer
 * ================================================================ */

template<typename T>
static void writeToBuffer(char*& buf, const T& val) {
    std::memcpy(buf, &val, sizeof(T));
    buf += sizeof(T);
}

template<typename T>
static T readFromBuffer(const char*& buf) {
    T val;
    std::memcpy(&val, buf, sizeof(T));
    buf += sizeof(T);
    return val;
}


/* ================================================================
 * Plugin: Construction
 * ================================================================ */

AliceTernaryPlugin::AliceTernaryPlugin(
    const uint32_t* plus_bits,
    const uint32_t* minus_bits,
    int out_features,
    int in_features,
    float scale)
    : mOutFeatures(out_features)
    , mInFeatures(in_features)
    , mScale(scale)
{
    mWordsPerRow = (in_features + 31) / 32;
    size_t total_words = (size_t)out_features * mWordsPerRow;

    mPlusBits.assign(plus_bits, plus_bits + total_words);
    mMinusBits.assign(minus_bits, minus_bits + total_words);
}

AliceTernaryPlugin::AliceTernaryPlugin(const void* data, size_t length)
{
    const char* ptr = static_cast<const char*>(data);

    mOutFeatures = readFromBuffer<int>(ptr);
    mInFeatures  = readFromBuffer<int>(ptr);
    mWordsPerRow = readFromBuffer<int>(ptr);
    mScale       = readFromBuffer<float>(ptr);

    size_t total_words = (size_t)mOutFeatures * mWordsPerRow;

    mPlusBits.resize(total_words);
    std::memcpy(mPlusBits.data(), ptr, total_words * sizeof(uint32_t));
    ptr += total_words * sizeof(uint32_t);

    mMinusBits.resize(total_words);
    std::memcpy(mMinusBits.data(), ptr, total_words * sizeof(uint32_t));
    ptr += total_words * sizeof(uint32_t);

    assert(ptr == static_cast<const char*>(data) + length);
}

AliceTernaryPlugin::~AliceTernaryPlugin()
{
    terminate();
}


/* ================================================================
 * Plugin: IPluginV2DynamicExt
 * ================================================================ */

nvinfer1::DimsExprs AliceTernaryPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    /* Input:  [..., in_features]
     * Output: [..., out_features]
     * Replace last dimension with out_features. */
    nvinfer1::DimsExprs output = inputs[0];
    output.d[output.nbDims - 1] = exprBuilder.constant(mOutFeatures);
    return output;
}

bool AliceTernaryPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) noexcept
{
    /* Support FP32 linear format only */
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

void AliceTernaryPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    /* Nothing to configure dynamically */
}

size_t AliceTernaryPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;  /* No workspace needed */
}


/* ================================================================
 * Plugin: IPluginV2Ext
 * ================================================================ */

nvinfer1::DataType AliceTernaryPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* inputTypes,
    int nbInputs) const noexcept
{
    return nvinfer1::DataType::kFLOAT;
}


/* ================================================================
 * Plugin: IPluginV2
 * ================================================================ */

const char* AliceTernaryPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* AliceTernaryPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int AliceTernaryPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int AliceTernaryPlugin::initialize() noexcept
{
    /* Upload weight bitplanes to device */
    size_t bytes = mPlusBits.size() * sizeof(uint32_t);

    if (cudaMalloc(&mDevPlusBits, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&mDevMinusBits, bytes) != cudaSuccess) return -1;

    cudaMemcpy(mDevPlusBits,  mPlusBits.data(),  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(mDevMinusBits, mMinusBits.data(), bytes, cudaMemcpyHostToDevice);

    return 0;
}

void AliceTernaryPlugin::terminate() noexcept
{
    if (mDevPlusBits)  { cudaFree(mDevPlusBits);  mDevPlusBits  = nullptr; }
    if (mDevMinusBits) { cudaFree(mDevMinusBits); mDevMinusBits = nullptr; }
}

size_t AliceTernaryPlugin::getSerializationSize() const noexcept
{
    size_t total_words = mPlusBits.size();
    return sizeof(int) * 3                       /* out_features, in_features, words_per_row */
         + sizeof(float)                         /* scale */
         + total_words * sizeof(uint32_t) * 2;   /* plus_bits + minus_bits */
}

void AliceTernaryPlugin::serialize(void* buffer) const noexcept
{
    char* ptr = static_cast<char*>(buffer);

    writeToBuffer(ptr, mOutFeatures);
    writeToBuffer(ptr, mInFeatures);
    writeToBuffer(ptr, mWordsPerRow);
    writeToBuffer(ptr, mScale);

    size_t bytes = mPlusBits.size() * sizeof(uint32_t);
    std::memcpy(ptr, mPlusBits.data(), bytes);
    ptr += bytes;
    std::memcpy(ptr, mMinusBits.data(), bytes);
}

void AliceTernaryPlugin::destroy() noexcept
{
    delete this;
}

void AliceTernaryPlugin::setPluginNamespace(const char* ns) noexcept
{
    mNamespace = ns;
}

const char* AliceTernaryPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::IPluginV2DynamicExt* AliceTernaryPlugin::clone() const noexcept
{
    auto* p = new AliceTernaryPlugin(
        mPlusBits.data(), mMinusBits.data(),
        mOutFeatures, mInFeatures, mScale
    );
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}


/* ================================================================
 * Plugin Creator
 * ================================================================ */

AliceTernaryPluginCreator::AliceTernaryPluginCreator()
{
    /* Define plugin fields (for ONNX parser integration) */
    mFields.push_back({"plus_bits",    nullptr, nvinfer1::PluginFieldType::kINT32, 0});
    mFields.push_back({"minus_bits",   nullptr, nvinfer1::PluginFieldType::kINT32, 0});
    mFields.push_back({"out_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFields.push_back({"in_features",  nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFields.push_back({"scale",        nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});

    mFC.nbFields = mFields.size();
    mFC.fields   = mFields.data();
}

const char* AliceTernaryPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* AliceTernaryPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* AliceTernaryPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* AliceTernaryPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept
{
    const uint32_t* plus_bits  = nullptr;
    const uint32_t* minus_bits = nullptr;
    int out_features = 0;
    int in_features  = 0;
    float scale      = 1.0f;

    for (int i = 0; i < fc->nbFields; i++) {
        const auto& f = fc->fields[i];
        if (std::string(f.name) == "plus_bits") {
            plus_bits = static_cast<const uint32_t*>(f.data);
        } else if (std::string(f.name) == "minus_bits") {
            minus_bits = static_cast<const uint32_t*>(f.data);
        } else if (std::string(f.name) == "out_features") {
            out_features = *static_cast<const int*>(f.data);
        } else if (std::string(f.name) == "in_features") {
            in_features = *static_cast<const int*>(f.data);
        } else if (std::string(f.name) == "scale") {
            scale = *static_cast<const float*>(f.data);
        }
    }

    return new AliceTernaryPlugin(plus_bits, minus_bits, out_features, in_features, scale);
}

nvinfer1::IPluginV2* AliceTernaryPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept
{
    return new AliceTernaryPlugin(serialData, serialLength);
}

void AliceTernaryPluginCreator::setPluginNamespace(const char* ns) noexcept
{
    mNamespace = ns;
}

const char* AliceTernaryPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


/* ================================================================
 * Register plugin with TensorRT
 * ================================================================ */

REGISTER_TENSORRT_PLUGIN(AliceTernaryPluginCreator);

}  /* namespace alice */
