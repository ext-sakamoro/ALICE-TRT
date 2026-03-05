//! C-ABI FFI for Unity / UE5 integration
//!
//! 37 `extern "C"` functions for GPU ternary inference.
//! Opaque pointer pattern: create → use → free.
//!
//! # 所有権ルール
//!
//! - `_new` / `_from_*` → 呼び出し元がポインタを所有、`_free`で解放
//! - `_free` → オブジェクトを解放。解放後のポインタ使用禁止
//! - `engine_add_layer` → weightポインタを**消費**（呼び出し後使用禁止）
//! - `compute_matvec` / `forward` → 新しいtensorポインタを返す（呼び出し元が解放）
//!
//! Author: Moroya Sakamoto

use std::ffi::{c_char, CString};
use std::ptr;
use std::slice;
use std::sync::OnceLock;

use crate::{
    Activation, GpuDevice, GpuInferenceEngine, GpuTensor, GpuTernaryWeight, TernaryCompute,
};

// ============================================================================
// Version (1)
// ============================================================================

/// ライブラリバージョン文字列（静的、解放不要）
#[no_mangle]
pub extern "C" fn at_trt_version() -> *const c_char {
    static C: OnceLock<CString> = OnceLock::new();
    C.get_or_init(|| CString::new(crate::VERSION).unwrap())
        .as_ptr()
}

// ============================================================================
// GpuDevice (3)
// ============================================================================

/// GPU初期化。GPU未検出時はnullを返す
#[no_mangle]
pub extern "C" fn at_trt_device_new() -> *mut GpuDevice {
    match GpuDevice::new() {
        Ok(d) => Box::into_raw(Box::new(d)),
        Err(_) => ptr::null_mut(),
    }
}

/// GPU解放
///
/// # Safety
/// ptrは`at_trt_device_new`で取得した有効なポインタ、またはnull
#[no_mangle]
pub unsafe extern "C" fn at_trt_device_free(ptr: *mut GpuDevice) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// デバイス情報をバッファにコピー。戻り値はバイト数（null終端除く）。
/// buf=nullまたはbuf_len=0の場合、必要なバイト数を返す
///
/// # Safety
/// ptrは有効なGpuDeviceポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_device_info(
    ptr: *const GpuDevice,
    buf: *mut c_char,
    buf_len: u32,
) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    let info = (*ptr).info();
    let needed = info.len();
    if buf.is_null() || buf_len == 0 {
        return needed as u32;
    }
    let copy_len = needed.min((buf_len - 1) as usize);
    ptr::copy_nonoverlapping(info.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0;
    copy_len as u32
}

// ============================================================================
// GpuTernaryWeight (9)
// ============================================================================

/// 三値重みを作成（値は {-1, 0, +1}、scale = 1.0）
///
/// # Safety
/// device, valuesは有効なポインタ。valuesの要素数は out_features × in_features
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_from_ternary(
    device: *const GpuDevice,
    values: *const i8,
    out_features: u32,
    in_features: u32,
) -> *mut GpuTernaryWeight {
    if device.is_null() || values.is_null() {
        return ptr::null_mut();
    }
    let n = (out_features as usize) * (in_features as usize);
    let vals = slice::from_raw_parts(values, n);
    let w =
        GpuTernaryWeight::from_ternary(&*device, vals, out_features as usize, in_features as usize);
    Box::into_raw(Box::new(w))
}

/// スケール付き三値重みを作成
///
/// # Safety
/// device, valuesは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_from_ternary_scaled(
    device: *const GpuDevice,
    values: *const i8,
    out_features: u32,
    in_features: u32,
    scale: f32,
) -> *mut GpuTernaryWeight {
    if device.is_null() || values.is_null() {
        return ptr::null_mut();
    }
    let n = (out_features as usize) * (in_features as usize);
    let vals = slice::from_raw_parts(values, n);
    let w = GpuTernaryWeight::from_ternary_scaled(
        &*device,
        vals,
        out_features as usize,
        in_features as usize,
        scale,
    );
    Box::into_raw(Box::new(w))
}

/// 重み解放
///
/// # Safety
/// ptrは`at_trt_weight_from_*`で取得したポインタ、またはnull
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_free(ptr: *mut GpuTernaryWeight) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// 出力特徴数
///
/// # Safety
/// ptrは有効なGpuTernaryWeightポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_out_features(ptr: *const GpuTernaryWeight) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).out_features() as u32
}

/// 入力特徴数
///
/// # Safety
/// ptrは有効なGpuTernaryWeightポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_in_features(ptr: *const GpuTernaryWeight) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).in_features() as u32
}

/// スケール係数
///
/// # Safety
/// ptrは有効なGpuTernaryWeightポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_scale(ptr: *const GpuTernaryWeight) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    (*ptr).scale()
}

/// 行あたりのu32ワード数（ビットプレーン）
///
/// # Safety
/// ptrは有効なGpuTernaryWeightポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_words_per_row(ptr: *const GpuTernaryWeight) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).words_per_row() as u32
}

/// VRAM使用量（バイト）
///
/// # Safety
/// ptrは有効なGpuTernaryWeightポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_memory_bytes(ptr: *const GpuTernaryWeight) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).memory_bytes() as u64
}

/// FP32比の圧縮率
///
/// # Safety
/// ptrは有効なGpuTernaryWeightポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_weight_compression_ratio(ptr: *const GpuTernaryWeight) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    (*ptr).compression_ratio()
}

// ============================================================================
// GpuTensor (10)
// ============================================================================

/// f32データからテンソルを作成
///
/// # Safety
/// device, data, shape_ptrは有効なポインタ。dataの要素数はdata_len。
/// shape_ptrの要素数はndim。
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_from_f32(
    device: *const GpuDevice,
    data: *const f32,
    data_len: u32,
    shape_ptr: *const u32,
    ndim: u32,
) -> *mut GpuTensor {
    if device.is_null() || data.is_null() || shape_ptr.is_null() {
        return ptr::null_mut();
    }
    let data_slice = slice::from_raw_parts(data, data_len as usize);
    let shape_raw = slice::from_raw_parts(shape_ptr, ndim as usize);
    let shape: Vec<usize> = shape_raw.iter().map(|&s| s as usize).collect();
    let t = GpuTensor::from_f32(&*device, data_slice, &shape);
    Box::into_raw(Box::new(t))
}

/// ゼロ初期化テンソルを作成
///
/// # Safety
/// device, shape_ptrは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_zeros(
    device: *const GpuDevice,
    shape_ptr: *const u32,
    ndim: u32,
) -> *mut GpuTensor {
    if device.is_null() || shape_ptr.is_null() {
        return ptr::null_mut();
    }
    let shape_raw = slice::from_raw_parts(shape_ptr, ndim as usize);
    let shape: Vec<usize> = shape_raw.iter().map(|&s| s as usize).collect();
    let t = GpuTensor::zeros(&*device, &shape);
    Box::into_raw(Box::new(t))
}

/// 出力用テンソルを作成（未初期化）
///
/// # Safety
/// device, shape_ptrは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_output(
    device: *const GpuDevice,
    shape_ptr: *const u32,
    ndim: u32,
) -> *mut GpuTensor {
    if device.is_null() || shape_ptr.is_null() {
        return ptr::null_mut();
    }
    let shape_raw = slice::from_raw_parts(shape_ptr, ndim as usize);
    let shape: Vec<usize> = shape_raw.iter().map(|&s| s as usize).collect();
    let t = GpuTensor::output(&*device, &shape);
    Box::into_raw(Box::new(t))
}

/// テンソル解放
///
/// # Safety
/// ptrは`at_trt_tensor_*`で取得したポインタ、またはnull
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_free(ptr: *mut GpuTensor) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// テンソルデータをCPUにダウンロード。戻り値はコピーした要素数
///
/// # Safety
/// tensor, device, outは有効なポインタ。outの容量はmax_len以上
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_download(
    tensor: *const GpuTensor,
    device: *const GpuDevice,
    out: *mut f32,
    max_len: u32,
) -> u32 {
    if tensor.is_null() || device.is_null() || out.is_null() {
        return 0;
    }
    let data = (*tensor).download(&*device);
    let copy_len = data.len().min(max_len as usize);
    ptr::copy_nonoverlapping(data.as_ptr(), out, copy_len);
    copy_len as u32
}

/// テンソルの次元数
///
/// # Safety
/// ptrは有効なGpuTensorポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_ndim(ptr: *const GpuTensor) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).shape().len() as u32
}

/// テンソルの指定次元のサイズ
///
/// # Safety
/// ptrは有効なGpuTensorポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_shape(ptr: *const GpuTensor, dim: u32) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    let shape = (*ptr).shape();
    if (dim as usize) < shape.len() {
        shape[dim as usize] as u32
    } else {
        0
    }
}

/// テンソルの総要素数
///
/// # Safety
/// ptrは有効なGpuTensorポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_len(ptr: *const GpuTensor) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).len() as u32
}

/// テンソルが空かどうか（0=false, 1=true）
///
/// # Safety
/// ptrは有効なGpuTensorポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_is_empty(ptr: *const GpuTensor) -> i32 {
    if ptr.is_null() {
        return 1;
    }
    i32::from((*ptr).is_empty())
}

/// テンソルのVRAM使用量（バイト）
///
/// # Safety
/// ptrは有効なGpuTensorポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_tensor_memory_bytes(ptr: *const GpuTensor) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).memory_bytes() as u64
}

// ============================================================================
// TernaryCompute (5)
// ============================================================================

/// コンピュートパイプライン作成（4シェーダーをコンパイル）
///
/// # Safety
/// deviceは有効なGpuDeviceポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_compute_new(device: *const GpuDevice) -> *mut TernaryCompute {
    if device.is_null() {
        return ptr::null_mut();
    }
    let c = TernaryCompute::new(&*device);
    Box::into_raw(Box::new(c))
}

/// コンピュートパイプライン解放
///
/// # Safety
/// ptrは`at_trt_compute_new`で取得したポインタ、またはnull
#[no_mangle]
pub unsafe extern "C" fn at_trt_compute_free(ptr: *mut TernaryCompute) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// 三値行列-ベクトル積（GPU実行）。新しいテンソルポインタを返す
///
/// # Safety
/// compute, device, input, weightsは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_compute_matvec(
    compute: *const TernaryCompute,
    device: *const GpuDevice,
    input: *const GpuTensor,
    weights: *const GpuTernaryWeight,
) -> *mut GpuTensor {
    if compute.is_null() || device.is_null() || input.is_null() || weights.is_null() {
        return ptr::null_mut();
    }
    let result = (*compute).matvec(&*device, &*input, &*weights);
    Box::into_raw(Box::new(result))
}

/// バッチ行列乗算（GPU実行）。新しいテンソルポインタを返す
///
/// # Safety
/// compute, device, input, weightsは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_compute_matmul_batch(
    compute: *const TernaryCompute,
    device: *const GpuDevice,
    input: *const GpuTensor,
    weights: *const GpuTernaryWeight,
    batch_size: u32,
) -> *mut GpuTensor {
    if compute.is_null() || device.is_null() || input.is_null() || weights.is_null() {
        return ptr::null_mut();
    }
    let result = (*compute).matmul_batch(&*device, &*input, &*weights, batch_size as usize);
    Box::into_raw(Box::new(result))
}

/// ReLU活性化（インプレース、GPU実行）
///
/// # Safety
/// compute, device, tensorは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_compute_relu_inplace(
    compute: *const TernaryCompute,
    device: *const GpuDevice,
    tensor: *const GpuTensor,
) {
    if compute.is_null() || device.is_null() || tensor.is_null() {
        return;
    }
    (*compute).relu_inplace(&*device, &*tensor);
}

// ============================================================================
// GpuInferenceEngine (9)
// ============================================================================

/// 推論エンジン作成（空、レイヤー0個）
#[no_mangle]
pub extern "C" fn at_trt_engine_new() -> *mut GpuInferenceEngine {
    Box::into_raw(Box::new(GpuInferenceEngine::new()))
}

/// 推論エンジン解放
///
/// # Safety
/// ptrは`at_trt_engine_new`で取得したポインタ、またはnull
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_free(ptr: *mut GpuInferenceEngine) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// レイヤー追加。**weightポインタは消費される**（呼び出し後使用禁止）。
/// activation: 0=None, 1=ReLU
///
/// # Safety
/// engine, weightは有効なポインタ。weightは本関数呼び出し後に使用・解放してはならない
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_add_layer(
    engine: *mut GpuInferenceEngine,
    weight: *mut GpuTernaryWeight,
    activation: u32,
) {
    if engine.is_null() || weight.is_null() {
        return;
    }
    let w = *Box::from_raw(weight);
    let act = if activation == 1 {
        Activation::ReLU
    } else {
        Activation::None
    };
    (*engine).add_layer(w, act);
}

/// レイヤー数
///
/// # Safety
/// ptrは有効なGpuInferenceEngineポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_num_layers(ptr: *const GpuInferenceEngine) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).num_layers() as u32
}

/// 全レイヤーの合計VRAM使用量（バイト）
///
/// # Safety
/// ptrは有効なGpuInferenceEngineポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_total_weight_bytes(ptr: *const GpuInferenceEngine) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).total_weight_bytes() as u64
}

/// FP32換算サイズ（バイト）
///
/// # Safety
/// ptrは有効なGpuInferenceEngineポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_equivalent_fp32_bytes(
    ptr: *const GpuInferenceEngine,
) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).equivalent_fp32_bytes() as u64
}

/// 全体の圧縮率
///
/// # Safety
/// ptrは有効なGpuInferenceEngineポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_compression_ratio(ptr: *const GpuInferenceEngine) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    (*ptr).compression_ratio()
}

/// フォワードパス（単一入力）。新しいテンソルポインタを返す
///
/// # Safety
/// engine, device, compute, inputは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_forward(
    engine: *const GpuInferenceEngine,
    device: *const GpuDevice,
    compute: *const TernaryCompute,
    input: *const GpuTensor,
) -> *mut GpuTensor {
    if engine.is_null() || device.is_null() || compute.is_null() || input.is_null() {
        return ptr::null_mut();
    }
    let result = (*engine).forward(&*device, &*compute, &*input);
    Box::into_raw(Box::new(result))
}

/// バッチフォワードパス。新しいテンソルポインタを返す
///
/// # Safety
/// engine, device, compute, inputは有効なポインタ
#[no_mangle]
pub unsafe extern "C" fn at_trt_engine_forward_batch(
    engine: *const GpuInferenceEngine,
    device: *const GpuDevice,
    compute: *const TernaryCompute,
    input: *const GpuTensor,
    batch_size: u32,
) -> *mut GpuTensor {
    if engine.is_null() || device.is_null() || compute.is_null() || input.is_null() {
        return ptr::null_mut();
    }
    let result = (*engine).forward_batch(&*device, &*compute, &*input, batch_size as usize);
    Box::into_raw(Box::new(result))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_version() {
        let v = at_trt_version();
        assert!(!v.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(v) };
        assert!(!s.to_str().unwrap().is_empty());
    }

    #[test]
    fn test_ffi_device_lifecycle() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return; // GPU未検出
        }

        let mut buf = [0i8; 256];
        let len = unsafe { at_trt_device_info(dev, buf.as_mut_ptr(), 256) };
        assert!(len > 0);

        unsafe { at_trt_device_free(dev) };
    }

    #[test]
    fn test_ffi_weight_lifecycle() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return;
        }

        let values: [i8; 4] = [1, -1, 0, 1];
        let w = unsafe { at_trt_weight_from_ternary(dev, values.as_ptr(), 2, 2) };
        assert!(!w.is_null());

        assert_eq!(unsafe { at_trt_weight_out_features(w) }, 2);
        assert_eq!(unsafe { at_trt_weight_in_features(w) }, 2);
        assert!((unsafe { at_trt_weight_scale(w) } - 1.0).abs() < 1e-6);
        assert_eq!(unsafe { at_trt_weight_words_per_row(w) }, 1);
        assert!(unsafe { at_trt_weight_memory_bytes(w) } > 0);
        assert!(unsafe { at_trt_weight_compression_ratio(w) } > 0.0);

        unsafe { at_trt_weight_free(w) };
        unsafe { at_trt_device_free(dev) };
    }

    #[test]
    fn test_ffi_weight_scaled() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return;
        }

        let values: [i8; 4] = [1, 1, -1, 0];
        let w = unsafe { at_trt_weight_from_ternary_scaled(dev, values.as_ptr(), 2, 2, 2.5) };
        assert!(!w.is_null());
        assert!((unsafe { at_trt_weight_scale(w) } - 2.5).abs() < 1e-6);

        unsafe { at_trt_weight_free(w) };
        unsafe { at_trt_device_free(dev) };
    }

    #[test]
    fn test_ffi_tensor_roundtrip() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return;
        }

        let data: [f32; 3] = [1.0, 2.0, 3.0];
        let shape: [u32; 1] = [3];
        let t = unsafe { at_trt_tensor_from_f32(dev, data.as_ptr(), 3, shape.as_ptr(), 1) };
        assert!(!t.is_null());

        assert_eq!(unsafe { at_trt_tensor_len(t) }, 3);
        assert_eq!(unsafe { at_trt_tensor_ndim(t) }, 1);
        assert_eq!(unsafe { at_trt_tensor_shape(t, 0) }, 3);
        assert_eq!(unsafe { at_trt_tensor_is_empty(t) }, 0);
        assert_eq!(unsafe { at_trt_tensor_memory_bytes(t) }, 12);

        let mut out = [0.0f32; 3];
        let n = unsafe { at_trt_tensor_download(t, dev, out.as_mut_ptr(), 3) };
        assert_eq!(n, 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);

        unsafe { at_trt_tensor_free(t) };
        unsafe { at_trt_device_free(dev) };
    }

    #[test]
    fn test_ffi_tensor_zeros() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return;
        }

        let shape: [u32; 1] = [4];
        let t = unsafe { at_trt_tensor_zeros(dev, shape.as_ptr(), 1) };
        assert!(!t.is_null());

        let mut out = [1.0f32; 4];
        unsafe { at_trt_tensor_download(t, dev, out.as_mut_ptr(), 4) };
        for v in &out {
            assert!(v.abs() < 1e-6);
        }

        unsafe { at_trt_tensor_free(t) };
        unsafe { at_trt_device_free(dev) };
    }

    #[test]
    fn test_ffi_compute_matvec() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return;
        }

        let compute = unsafe { at_trt_compute_new(dev) };
        assert!(!compute.is_null());

        // W = [[1, -1], [0, 1]], x = [2, 3] → y = [-1, 3]
        let values: [i8; 4] = [1, -1, 0, 1];
        let w = unsafe { at_trt_weight_from_ternary(dev, values.as_ptr(), 2, 2) };

        let data: [f32; 2] = [2.0, 3.0];
        let shape: [u32; 1] = [2];
        let input = unsafe { at_trt_tensor_from_f32(dev, data.as_ptr(), 2, shape.as_ptr(), 1) };

        let output = unsafe { at_trt_compute_matvec(compute, dev, input, w) };
        assert!(!output.is_null());

        let mut result = [0.0f32; 2];
        unsafe { at_trt_tensor_download(output, dev, result.as_mut_ptr(), 2) };
        assert!((result[0] - (-1.0)).abs() < 1e-4);
        assert!((result[1] - 3.0).abs() < 1e-4);

        unsafe {
            at_trt_tensor_free(output);
            at_trt_tensor_free(input);
            at_trt_weight_free(w);
            at_trt_compute_free(compute);
            at_trt_device_free(dev);
        }
    }

    #[test]
    fn test_ffi_engine_forward() {
        let dev = at_trt_device_new();
        if dev.is_null() {
            return;
        }

        let compute = unsafe { at_trt_compute_new(dev) };
        let engine = at_trt_engine_new();
        assert!(!engine.is_null());

        // 2-layer: 3→2→2
        let w1_vals: [i8; 6] = [1, -1, 0, 1, -1, 1];
        let w1 = unsafe { at_trt_weight_from_ternary(dev, w1_vals.as_ptr(), 2, 3) };
        unsafe { at_trt_engine_add_layer(engine, w1, 1) }; // ReLU（w1は消費済み）

        let w2_vals: [i8; 4] = [1, 1, -1, 1];
        let w2 = unsafe { at_trt_weight_from_ternary(dev, w2_vals.as_ptr(), 2, 2) };
        unsafe { at_trt_engine_add_layer(engine, w2, 0) }; // None（w2は消費済み）

        assert_eq!(unsafe { at_trt_engine_num_layers(engine) }, 2);
        assert!(unsafe { at_trt_engine_total_weight_bytes(engine) } > 0);
        assert!(unsafe { at_trt_engine_equivalent_fp32_bytes(engine) } > 0);
        assert!(unsafe { at_trt_engine_compression_ratio(engine) } > 0.0);

        let data: [f32; 3] = [1.0, 2.0, 3.0];
        let shape: [u32; 1] = [3];
        let input = unsafe { at_trt_tensor_from_f32(dev, data.as_ptr(), 3, shape.as_ptr(), 1) };

        let output = unsafe { at_trt_engine_forward(engine, dev, compute, input) };
        assert!(!output.is_null());

        let mut result = [0.0f32; 2];
        unsafe { at_trt_tensor_download(output, dev, result.as_mut_ptr(), 2) };
        assert!((result[0] - 2.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);

        unsafe {
            at_trt_tensor_free(output);
            at_trt_tensor_free(input);
            at_trt_engine_free(engine);
            at_trt_compute_free(compute);
            at_trt_device_free(dev);
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        // 全てのnullポインタ入力が安全にハンドルされることを確認
        unsafe {
            at_trt_device_free(ptr::null_mut());
            at_trt_weight_free(ptr::null_mut());
            at_trt_tensor_free(ptr::null_mut());
            at_trt_compute_free(ptr::null_mut());
            at_trt_engine_free(ptr::null_mut());

            assert_eq!(at_trt_device_info(ptr::null(), ptr::null_mut(), 0), 0);
            assert_eq!(at_trt_weight_out_features(ptr::null()), 0);
            assert_eq!(at_trt_weight_in_features(ptr::null()), 0);
            assert!(at_trt_weight_scale(ptr::null()).abs() < 1e-6);
            assert_eq!(at_trt_tensor_len(ptr::null()), 0);
            assert_eq!(at_trt_tensor_ndim(ptr::null()), 0);
            assert_eq!(at_trt_tensor_is_empty(ptr::null()), 1);
            assert_eq!(at_trt_engine_num_layers(ptr::null()), 0);
            assert!(at_trt_engine_compression_ratio(ptr::null()).abs() < 1e-6);

            assert!(at_trt_weight_from_ternary(ptr::null(), ptr::null(), 0, 0).is_null());
            assert!(at_trt_tensor_from_f32(ptr::null(), ptr::null(), 0, ptr::null(), 0).is_null());
            assert!(at_trt_compute_new(ptr::null()).is_null());
            assert!(
                at_trt_compute_matvec(ptr::null(), ptr::null(), ptr::null(), ptr::null()).is_null()
            );
            assert!(
                at_trt_engine_forward(ptr::null(), ptr::null(), ptr::null(), ptr::null()).is_null()
            );
        }
    }
}
