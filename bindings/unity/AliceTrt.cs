// ALICE-TRT Unity C# Bindings
// 37 DllImport + 5 RAII IDisposable handles
//
// Author: Moroya Sakamoto

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Alice.Trt
{
    // ========================================================================
    // Raw P/Invoke (37 functions)
    // ========================================================================

    public static class AliceTrtNative
    {
        private const string Lib = "alice_trt";

        // Version (1)
        [DllImport(Lib)] public static extern IntPtr at_trt_version();

        // GpuDevice (3)
        [DllImport(Lib)] public static extern IntPtr at_trt_device_new();
        [DllImport(Lib)] public static extern void at_trt_device_free(IntPtr ptr);
        [DllImport(Lib)] public static extern uint at_trt_device_info(IntPtr ptr, StringBuilder buf, uint bufLen);

        // GpuTernaryWeight (9)
        [DllImport(Lib)] public static extern IntPtr at_trt_weight_from_ternary(IntPtr device, sbyte[] values, uint outFeatures, uint inFeatures);
        [DllImport(Lib)] public static extern IntPtr at_trt_weight_from_ternary_scaled(IntPtr device, sbyte[] values, uint outFeatures, uint inFeatures, float scale);
        [DllImport(Lib)] public static extern void at_trt_weight_free(IntPtr ptr);
        [DllImport(Lib)] public static extern uint at_trt_weight_out_features(IntPtr ptr);
        [DllImport(Lib)] public static extern uint at_trt_weight_in_features(IntPtr ptr);
        [DllImport(Lib)] public static extern float at_trt_weight_scale(IntPtr ptr);
        [DllImport(Lib)] public static extern uint at_trt_weight_words_per_row(IntPtr ptr);
        [DllImport(Lib)] public static extern ulong at_trt_weight_memory_bytes(IntPtr ptr);
        [DllImport(Lib)] public static extern float at_trt_weight_compression_ratio(IntPtr ptr);

        // GpuTensor (10)
        [DllImport(Lib)] public static extern IntPtr at_trt_tensor_from_f32(IntPtr device, float[] data, uint dataLen, uint[] shape, uint ndim);
        [DllImport(Lib)] public static extern IntPtr at_trt_tensor_zeros(IntPtr device, uint[] shape, uint ndim);
        [DllImport(Lib)] public static extern IntPtr at_trt_tensor_output(IntPtr device, uint[] shape, uint ndim);
        [DllImport(Lib)] public static extern void at_trt_tensor_free(IntPtr ptr);
        [DllImport(Lib)] public static extern uint at_trt_tensor_download(IntPtr tensor, IntPtr device, float[] output, uint maxLen);
        [DllImport(Lib)] public static extern uint at_trt_tensor_ndim(IntPtr ptr);
        [DllImport(Lib)] public static extern uint at_trt_tensor_shape(IntPtr ptr, uint dim);
        [DllImport(Lib)] public static extern uint at_trt_tensor_len(IntPtr ptr);
        [DllImport(Lib)] public static extern int at_trt_tensor_is_empty(IntPtr ptr);
        [DllImport(Lib)] public static extern ulong at_trt_tensor_memory_bytes(IntPtr ptr);

        // TernaryCompute (5)
        [DllImport(Lib)] public static extern IntPtr at_trt_compute_new(IntPtr device);
        [DllImport(Lib)] public static extern void at_trt_compute_free(IntPtr ptr);
        [DllImport(Lib)] public static extern IntPtr at_trt_compute_matvec(IntPtr compute, IntPtr device, IntPtr input, IntPtr weights);
        [DllImport(Lib)] public static extern IntPtr at_trt_compute_matmul_batch(IntPtr compute, IntPtr device, IntPtr input, IntPtr weights, uint batchSize);
        [DllImport(Lib)] public static extern void at_trt_compute_relu_inplace(IntPtr compute, IntPtr device, IntPtr tensor);

        // GpuInferenceEngine (9)
        [DllImport(Lib)] public static extern IntPtr at_trt_engine_new();
        [DllImport(Lib)] public static extern void at_trt_engine_free(IntPtr ptr);
        [DllImport(Lib)] public static extern void at_trt_engine_add_layer(IntPtr engine, IntPtr weight, uint activation);
        [DllImport(Lib)] public static extern uint at_trt_engine_num_layers(IntPtr ptr);
        [DllImport(Lib)] public static extern ulong at_trt_engine_total_weight_bytes(IntPtr ptr);
        [DllImport(Lib)] public static extern ulong at_trt_engine_equivalent_fp32_bytes(IntPtr ptr);
        [DllImport(Lib)] public static extern float at_trt_engine_compression_ratio(IntPtr ptr);
        [DllImport(Lib)] public static extern IntPtr at_trt_engine_forward(IntPtr engine, IntPtr device, IntPtr compute, IntPtr input);
        [DllImport(Lib)] public static extern IntPtr at_trt_engine_forward_batch(IntPtr engine, IntPtr device, IntPtr compute, IntPtr input, uint batchSize);
    }

    // ========================================================================
    // RAII Handles (5)
    // ========================================================================

    public sealed class GpuDeviceHandle : IDisposable
    {
        public IntPtr Ptr { get; private set; }
        public bool IsValid => Ptr != IntPtr.Zero;

        public GpuDeviceHandle()
        {
            Ptr = AliceTrtNative.at_trt_device_new();
        }

        public string Info
        {
            get
            {
                var sb = new StringBuilder(512);
                AliceTrtNative.at_trt_device_info(Ptr, sb, 512);
                return sb.ToString();
            }
        }

        public void Dispose()
        {
            if (Ptr != IntPtr.Zero)
            {
                AliceTrtNative.at_trt_device_free(Ptr);
                Ptr = IntPtr.Zero;
            }
        }
    }

    public sealed class GpuTernaryWeightHandle : IDisposable
    {
        public IntPtr Ptr { get; private set; }
        public bool IsValid => Ptr != IntPtr.Zero;

        public GpuTernaryWeightHandle(GpuDeviceHandle device, sbyte[] values, uint outFeatures, uint inFeatures)
        {
            Ptr = AliceTrtNative.at_trt_weight_from_ternary(device.Ptr, values, outFeatures, inFeatures);
        }

        public GpuTernaryWeightHandle(GpuDeviceHandle device, sbyte[] values, uint outFeatures, uint inFeatures, float scale)
        {
            Ptr = AliceTrtNative.at_trt_weight_from_ternary_scaled(device.Ptr, values, outFeatures, inFeatures, scale);
        }

        public uint OutFeatures => AliceTrtNative.at_trt_weight_out_features(Ptr);
        public uint InFeatures => AliceTrtNative.at_trt_weight_in_features(Ptr);
        public float Scale => AliceTrtNative.at_trt_weight_scale(Ptr);
        public ulong MemoryBytes => AliceTrtNative.at_trt_weight_memory_bytes(Ptr);
        public float CompressionRatio => AliceTrtNative.at_trt_weight_compression_ratio(Ptr);

        /// <summary>EngineにAddLayerした後はこのハンドルを使用しないこと</summary>
        public void MarkConsumed() { Ptr = IntPtr.Zero; }

        public void Dispose()
        {
            if (Ptr != IntPtr.Zero)
            {
                AliceTrtNative.at_trt_weight_free(Ptr);
                Ptr = IntPtr.Zero;
            }
        }
    }

    public sealed class GpuTensorHandle : IDisposable
    {
        public IntPtr Ptr { get; private set; }
        public bool IsValid => Ptr != IntPtr.Zero;

        public GpuTensorHandle(IntPtr raw) { Ptr = raw; }

        public static GpuTensorHandle FromF32(GpuDeviceHandle device, float[] data, uint[] shape)
        {
            var ptr = AliceTrtNative.at_trt_tensor_from_f32(device.Ptr, data, (uint)data.Length, shape, (uint)shape.Length);
            return new GpuTensorHandle(ptr);
        }

        public static GpuTensorHandle Zeros(GpuDeviceHandle device, uint[] shape)
        {
            var ptr = AliceTrtNative.at_trt_tensor_zeros(device.Ptr, shape, (uint)shape.Length);
            return new GpuTensorHandle(ptr);
        }

        public float[] Download(GpuDeviceHandle device)
        {
            uint len = AliceTrtNative.at_trt_tensor_len(Ptr);
            var buf = new float[len];
            AliceTrtNative.at_trt_tensor_download(Ptr, device.Ptr, buf, len);
            return buf;
        }

        public uint Len => AliceTrtNative.at_trt_tensor_len(Ptr);
        public uint Ndim => AliceTrtNative.at_trt_tensor_ndim(Ptr);
        public ulong MemoryBytes => AliceTrtNative.at_trt_tensor_memory_bytes(Ptr);

        public void Dispose()
        {
            if (Ptr != IntPtr.Zero)
            {
                AliceTrtNative.at_trt_tensor_free(Ptr);
                Ptr = IntPtr.Zero;
            }
        }
    }

    public sealed class TernaryComputeHandle : IDisposable
    {
        public IntPtr Ptr { get; private set; }
        public bool IsValid => Ptr != IntPtr.Zero;

        public TernaryComputeHandle(GpuDeviceHandle device)
        {
            Ptr = AliceTrtNative.at_trt_compute_new(device.Ptr);
        }

        public GpuTensorHandle Matvec(GpuDeviceHandle device, GpuTensorHandle input, GpuTernaryWeightHandle weights)
        {
            var ptr = AliceTrtNative.at_trt_compute_matvec(Ptr, device.Ptr, input.Ptr, weights.Ptr);
            return new GpuTensorHandle(ptr);
        }

        public GpuTensorHandle MatmulBatch(GpuDeviceHandle device, GpuTensorHandle input, GpuTernaryWeightHandle weights, uint batchSize)
        {
            var ptr = AliceTrtNative.at_trt_compute_matmul_batch(Ptr, device.Ptr, input.Ptr, weights.Ptr, batchSize);
            return new GpuTensorHandle(ptr);
        }

        public void ReluInplace(GpuDeviceHandle device, GpuTensorHandle tensor)
        {
            AliceTrtNative.at_trt_compute_relu_inplace(Ptr, device.Ptr, tensor.Ptr);
        }

        public void Dispose()
        {
            if (Ptr != IntPtr.Zero)
            {
                AliceTrtNative.at_trt_compute_free(Ptr);
                Ptr = IntPtr.Zero;
            }
        }
    }

    public sealed class InferenceEngineHandle : IDisposable
    {
        public IntPtr Ptr { get; private set; }
        public bool IsValid => Ptr != IntPtr.Zero;

        public InferenceEngineHandle()
        {
            Ptr = AliceTrtNative.at_trt_engine_new();
        }

        /// <summary>レイヤー追加。weightハンドルは消費される（Dispose不要）</summary>
        public void AddLayer(GpuTernaryWeightHandle weight, uint activation)
        {
            AliceTrtNative.at_trt_engine_add_layer(Ptr, weight.Ptr, activation);
            weight.MarkConsumed();
        }

        public GpuTensorHandle Forward(GpuDeviceHandle device, TernaryComputeHandle compute, GpuTensorHandle input)
        {
            var ptr = AliceTrtNative.at_trt_engine_forward(Ptr, device.Ptr, compute.Ptr, input.Ptr);
            return new GpuTensorHandle(ptr);
        }

        public GpuTensorHandle ForwardBatch(GpuDeviceHandle device, TernaryComputeHandle compute, GpuTensorHandle input, uint batchSize)
        {
            var ptr = AliceTrtNative.at_trt_engine_forward_batch(Ptr, device.Ptr, compute.Ptr, input.Ptr, batchSize);
            return new GpuTensorHandle(ptr);
        }

        public uint NumLayers => AliceTrtNative.at_trt_engine_num_layers(Ptr);
        public ulong TotalWeightBytes => AliceTrtNative.at_trt_engine_total_weight_bytes(Ptr);
        public float CompressionRatio => AliceTrtNative.at_trt_engine_compression_ratio(Ptr);

        public void Dispose()
        {
            if (Ptr != IntPtr.Zero)
            {
                AliceTrtNative.at_trt_engine_free(Ptr);
                Ptr = IntPtr.Zero;
            }
        }
    }
}
