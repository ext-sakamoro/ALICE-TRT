//! PyO3 Python bindings for ALICE-TRT
//!
//! 5 classes: GpuDevice, GpuTernaryWeight, GpuTensor, TernaryCompute, InferenceEngine
//!
//! Author: Moroya Sakamoto

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::{
    Activation, GpuInferenceEngine, GpuTensor as RustTensor, GpuTernaryWeight as RustWeight,
};

/// GPUデバイスハンドル
#[pyclass(name = "GpuDevice")]
pub struct PyGpuDevice {
    pub(crate) inner: crate::GpuDevice,
}

#[pymethods]
impl PyGpuDevice {
    /// GPU初期化
    #[new]
    fn new() -> PyResult<Self> {
        crate::GpuDevice::new()
            .map(|d| Self { inner: d })
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// GPUアダプタ情報
    #[getter]
    fn info(&self) -> &str {
        self.inner.info()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// 三値重み（GPUビットプレーン格納）
#[pyclass(name = "GpuTernaryWeight")]
pub struct PyGpuTernaryWeight {
    pub(crate) inner: Option<RustWeight>,
}

#[pymethods]
impl PyGpuTernaryWeight {
    /// 三値重みを作成（値は {-1, 0, +1}）
    #[staticmethod]
    fn from_ternary(
        device: &PyGpuDevice,
        values: Vec<i8>,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        let w = RustWeight::from_ternary(&device.inner, &values, out_features, in_features);
        Self { inner: Some(w) }
    }

    /// スケール付き三値重みを作成
    #[staticmethod]
    fn from_ternary_scaled(
        device: &PyGpuDevice,
        values: Vec<i8>,
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Self {
        let w = RustWeight::from_ternary_scaled(
            &device.inner,
            &values,
            out_features,
            in_features,
            scale,
        );
        Self { inner: Some(w) }
    }

    /// 出力特徴数
    #[getter]
    fn out_features(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|w| w.out_features())
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))
    }

    /// 入力特徴数
    #[getter]
    fn in_features(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|w| w.in_features())
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))
    }

    /// スケール係数
    #[getter]
    fn scale(&self) -> PyResult<f32> {
        self.inner
            .as_ref()
            .map(|w| w.scale())
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))
    }

    /// VRAM使用量（バイト）
    #[getter]
    fn memory_bytes(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|w| w.memory_bytes())
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))
    }

    /// FP32比の圧縮率
    #[getter]
    fn compression_ratio(&self) -> PyResult<f32> {
        self.inner
            .as_ref()
            .map(|w| w.compression_ratio())
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(w) => format!("{w}"),
            None => "GpuTernaryWeight[consumed]".to_string(),
        }
    }
}

/// GPUテンソル（f32ストレージバッファ）
#[pyclass(name = "GpuTensor")]
pub struct PyGpuTensor {
    pub(crate) inner: RustTensor,
}

#[pymethods]
impl PyGpuTensor {
    /// f32データからテンソルを作成
    #[staticmethod]
    fn from_f32(device: &PyGpuDevice, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let t = RustTensor::from_f32(&device.inner, &data, &shape);
        Self { inner: t }
    }

    /// ゼロ初期化テンソル
    #[staticmethod]
    fn zeros(device: &PyGpuDevice, shape: Vec<usize>) -> Self {
        let t = RustTensor::zeros(&device.inner, &shape);
        Self { inner: t }
    }

    /// テンソルデータをCPUにダウンロード
    fn download(&self, device: &PyGpuDevice) -> Vec<f32> {
        self.inner.download(&device.inner)
    }

    /// テンソルの形状
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// 総要素数
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// VRAM使用量（バイト）
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// コンピュートパイプライン（4シェーダー）
#[pyclass(name = "TernaryCompute")]
pub struct PyTernaryCompute {
    pub(crate) inner: crate::TernaryCompute,
}

#[pymethods]
impl PyTernaryCompute {
    /// パイプラインをコンパイル
    #[new]
    fn new(device: &PyGpuDevice) -> Self {
        let c = crate::TernaryCompute::new(&device.inner);
        Self { inner: c }
    }

    /// 三値行列-ベクトル積
    fn matvec(
        &self,
        device: &PyGpuDevice,
        input: &PyGpuTensor,
        weights: &PyGpuTernaryWeight,
    ) -> PyResult<PyGpuTensor> {
        let w = weights
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))?;
        let result = self.inner.matvec(&device.inner, &input.inner, w);
        Ok(PyGpuTensor { inner: result })
    }

    /// バッチ行列乗算
    fn matmul_batch(
        &self,
        device: &PyGpuDevice,
        input: &PyGpuTensor,
        weights: &PyGpuTernaryWeight,
        batch_size: usize,
    ) -> PyResult<PyGpuTensor> {
        let w = weights
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))?;
        let result = self
            .inner
            .matmul_batch(&device.inner, &input.inner, w, batch_size);
        Ok(PyGpuTensor { inner: result })
    }

    /// ReLU活性化（インプレース）
    fn relu_inplace(&self, device: &PyGpuDevice, tensor: &PyGpuTensor) {
        self.inner.relu_inplace(&device.inner, &tensor.inner);
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// 多層推論エンジン
#[pyclass(name = "InferenceEngine")]
pub struct PyInferenceEngine {
    pub(crate) inner: GpuInferenceEngine,
}

#[pymethods]
impl PyInferenceEngine {
    /// 空のエンジンを作成
    #[new]
    fn new() -> Self {
        Self {
            inner: GpuInferenceEngine::new(),
        }
    }

    /// 三値レイヤーを直接追加（重みを内部で作成）
    /// activation: 0=None, 1=ReLU
    fn add_ternary_layer(
        &mut self,
        device: &PyGpuDevice,
        values: Vec<i8>,
        out_features: usize,
        in_features: usize,
        activation: u32,
    ) {
        let w = RustWeight::from_ternary(&device.inner, &values, out_features, in_features);
        let act = if activation == 1 {
            Activation::ReLU
        } else {
            Activation::None
        };
        self.inner.add_layer(w, act);
    }

    /// 重みオブジェクトからレイヤーを追加（重みは消費される）
    /// activation: 0=None, 1=ReLU
    fn add_layer(&mut self, weight: &mut PyGpuTernaryWeight, activation: u32) -> PyResult<()> {
        let w = weight
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Weight already consumed"))?;
        let act = if activation == 1 {
            Activation::ReLU
        } else {
            Activation::None
        };
        self.inner.add_layer(w, act);
        Ok(())
    }

    /// フォワードパス（単一入力）
    fn forward(
        &self,
        device: &PyGpuDevice,
        compute: &PyTernaryCompute,
        input: &PyGpuTensor,
    ) -> PyGpuTensor {
        let result = self
            .inner
            .forward(&device.inner, &compute.inner, &input.inner);
        PyGpuTensor { inner: result }
    }

    /// バッチフォワードパス
    fn forward_batch(
        &self,
        device: &PyGpuDevice,
        compute: &PyTernaryCompute,
        input: &PyGpuTensor,
        batch_size: usize,
    ) -> PyGpuTensor {
        let result =
            self.inner
                .forward_batch(&device.inner, &compute.inner, &input.inner, batch_size);
        PyGpuTensor { inner: result }
    }

    /// レイヤー数
    #[getter]
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// 全レイヤーの合計VRAM使用量（バイト）
    #[getter]
    fn total_weight_bytes(&self) -> usize {
        self.inner.total_weight_bytes()
    }

    /// 全体の圧縮率
    #[getter]
    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Pythonモジュール登録
#[pymodule]
fn alice_trt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuDevice>()?;
    m.add_class::<PyGpuTernaryWeight>()?;
    m.add_class::<PyGpuTensor>()?;
    m.add_class::<PyTernaryCompute>()?;
    m.add_class::<PyInferenceEngine>()?;
    Ok(())
}
