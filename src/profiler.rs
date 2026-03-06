//! 推論プロファイラモジュール
//!
//! レイヤー単位の実行時間計測、ボトルネック特定、
//! FLOPs 推定を提供する。
//!
//! CPU wall-clock (`std::time::Instant`) ベース。
//! GPU の `poll_wait()` をレイヤー間に挟むことで、
//! 各レイヤーの GPU 完了までの時間を計測する。
//!
//! Author: Moroya Sakamoto

use std::time::{Duration, Instant};

use crate::device::GpuDevice;
use crate::inference::{Activation, GpuInferenceEngine};
use crate::pipeline::TernaryCompute;
use crate::tensor::GpuTensor;

/// 単一レイヤーのプロファイル結果。
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// レイヤーインデックス (0-based)。
    pub index: usize,
    /// 実行時間。
    pub duration: Duration,
    /// 入力要素数。
    pub input_size: usize,
    /// 出力要素数。
    pub output_size: usize,
    /// 活性化関数。
    pub activation: Activation,
    /// 推定 FLOPs (乗算 + 加算)。
    ///
    /// Ternary は乗算を add/sub/nop に変換するため、
    /// 実際の演算は FLOPs の半分程度。
    pub estimated_flops: u64,
}

impl LayerProfile {
    /// 実行時間 (マイクロ秒)。
    #[must_use]
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000_000.0
    }

    /// スループット (GFLOP/s)。
    #[must_use]
    pub fn gflops(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.estimated_flops as f64 / secs / 1e9
    }
}

/// 推論全体のプロファイル結果。
#[derive(Debug, Clone)]
pub struct InferenceProfile {
    /// レイヤー別プロファイル。
    pub layers: Vec<LayerProfile>,
    /// 推論全体の実行時間。
    pub total_duration: Duration,
    /// バッチサイズ (1 = 単一推論)。
    pub batch_size: usize,
}

impl InferenceProfile {
    /// 最も遅いレイヤーのインデックス。
    #[must_use]
    pub fn bottleneck_index(&self) -> Option<usize> {
        self.layers
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.duration)
            .map(|(i, _)| i)
    }

    /// 最も遅いレイヤーのプロファイル。
    #[must_use]
    pub fn bottleneck(&self) -> Option<&LayerProfile> {
        self.bottleneck_index().and_then(|i| self.layers.get(i))
    }

    /// 推論全体のレイテンシ (マイクロ秒)。
    #[must_use]
    pub fn total_duration_us(&self) -> f64 {
        self.total_duration.as_secs_f64() * 1_000_000.0
    }

    /// 全レイヤーの FLOPs 合計。
    #[must_use]
    pub fn total_flops(&self) -> u64 {
        self.layers.iter().map(|l| l.estimated_flops).sum()
    }

    /// 推論全体のスループット (GFLOP/s)。
    #[must_use]
    pub fn total_gflops(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_flops() as f64 / secs / 1e9
    }

    /// レイヤー数。
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl std::fmt::Display for InferenceProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "InferenceProfile: {} layers, batch={}, total={:.1}us, {:.2} GFLOP/s",
            self.num_layers(),
            self.batch_size,
            self.total_duration_us(),
            self.total_gflops(),
        )?;
        for lp in &self.layers {
            writeln!(
                f,
                "  Layer {}: {:.1}us, {}x{}, {:?}, {:.2} GFLOP/s",
                lp.index,
                lp.duration_us(),
                lp.input_size,
                lp.output_size,
                lp.activation,
                lp.gflops(),
            )?;
        }
        if let Some(bn) = self.bottleneck() {
            write!(
                f,
                "  Bottleneck: Layer {} ({:.1}us)",
                bn.index,
                bn.duration_us()
            )?;
        }
        Ok(())
    }
}

/// 推論プロファイラ。
///
/// `GpuInferenceEngine` のレイヤーを1つずつ実行し、
/// 各レイヤーの実行時間を計測する。
pub struct InferenceProfiler;

impl InferenceProfiler {
    /// 単一入力の forward をプロファイルする。
    ///
    /// 各レイヤー実行後に `poll_wait()` で GPU 完了を待ち、
    /// CPU wall-clock でレイテンシを計測する。
    ///
    /// # Panics
    ///
    /// エンジンにレイヤーが無い場合パニック。
    pub fn profile_forward(
        engine: &GpuInferenceEngine,
        device: &GpuDevice,
        compute: &TernaryCompute,
        input: &GpuTensor,
    ) -> InferenceProfile {
        let layer_info = engine.layer_info();
        assert!(!layer_info.is_empty(), "Engine has no layers");

        let total_start = Instant::now();
        let mut profiles = Vec::new();

        // 最初のレイヤーは input を直接使用
        let (first_w, first_act) = layer_info[0];
        let in_size = input.len();
        let layer_start = Instant::now();
        let mut current = compute.matvec(device, input, first_w);
        if first_act == Activation::ReLU {
            compute.relu_inplace(device, &current);
        }
        device.poll_wait();
        profiles.push(LayerProfile {
            index: 0,
            duration: layer_start.elapsed(),
            input_size: in_size,
            output_size: first_w.out_features(),
            activation: first_act,
            estimated_flops: 2 * first_w.out_features() as u64 * first_w.in_features() as u64,
        });

        for (i, &(weights, activation)) in layer_info.iter().enumerate().skip(1) {
            let in_size = current.len();
            let layer_start = Instant::now();
            let output = compute.matvec(device, &current, weights);
            if activation == Activation::ReLU {
                compute.relu_inplace(device, &output);
            }
            device.poll_wait();
            profiles.push(LayerProfile {
                index: i,
                duration: layer_start.elapsed(),
                input_size: in_size,
                output_size: weights.out_features(),
                activation,
                estimated_flops: 2 * weights.out_features() as u64 * weights.in_features() as u64,
            });
            current = output;
        }

        InferenceProfile {
            layers: profiles,
            total_duration: total_start.elapsed(),
            batch_size: 1,
        }
    }

    /// バッチ forward をプロファイルする。
    ///
    /// # Panics
    ///
    /// エンジンにレイヤーが無い場合パニック。
    pub fn profile_forward_batch(
        engine: &GpuInferenceEngine,
        device: &GpuDevice,
        compute: &TernaryCompute,
        input: &GpuTensor,
        batch_size: usize,
    ) -> InferenceProfile {
        let layer_info = engine.layer_info();
        assert!(!layer_info.is_empty(), "Engine has no layers");

        let total_start = Instant::now();
        let mut profiles = Vec::new();

        let (first_w, first_act) = layer_info[0];
        let in_size = input.len();
        let layer_start = Instant::now();
        let mut current = compute.matmul_batch(device, input, first_w, batch_size);
        if first_act == Activation::ReLU {
            compute.relu_inplace(device, &current);
        }
        device.poll_wait();
        profiles.push(LayerProfile {
            index: 0,
            duration: layer_start.elapsed(),
            input_size: in_size,
            output_size: first_w.out_features() * batch_size,
            activation: first_act,
            estimated_flops: 2
                * batch_size as u64
                * first_w.out_features() as u64
                * first_w.in_features() as u64,
        });

        for (i, &(weights, activation)) in layer_info.iter().enumerate().skip(1) {
            let in_size = current.len();
            let layer_start = Instant::now();
            let output = compute.matmul_batch(device, &current, weights, batch_size);
            if activation == Activation::ReLU {
                compute.relu_inplace(device, &output);
            }
            device.poll_wait();
            profiles.push(LayerProfile {
                index: i,
                duration: layer_start.elapsed(),
                input_size: in_size,
                output_size: weights.out_features() * batch_size,
                activation,
                estimated_flops: 2
                    * batch_size as u64
                    * weights.out_features() as u64
                    * weights.in_features() as u64,
            });
            current = output;
        }

        InferenceProfile {
            layers: profiles,
            total_duration: total_start.elapsed(),
            batch_size,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::GpuTernaryWeight;

    // --- LayerProfile ---

    #[test]
    fn layer_profile_duration_us() {
        let lp = LayerProfile {
            index: 0,
            duration: Duration::from_micros(500),
            input_size: 128,
            output_size: 64,
            activation: Activation::ReLU,
            estimated_flops: 16384,
        };
        assert!((lp.duration_us() - 500.0).abs() < 1.0);
    }

    #[test]
    fn layer_profile_gflops() {
        let lp = LayerProfile {
            index: 0,
            duration: Duration::from_secs(1),
            input_size: 1024,
            output_size: 512,
            activation: Activation::None,
            estimated_flops: 2_000_000_000, // 2 GFLOP
        };
        assert!((lp.gflops() - 2.0).abs() < 0.01);
    }

    #[test]
    fn layer_profile_gflops_zero_duration() {
        let lp = LayerProfile {
            index: 0,
            duration: Duration::ZERO,
            input_size: 0,
            output_size: 0,
            activation: Activation::None,
            estimated_flops: 100,
        };
        assert!((lp.gflops()).abs() < 0.01);
    }

    // --- InferenceProfile ---

    #[test]
    fn inference_profile_empty() {
        let profile = InferenceProfile {
            layers: vec![],
            total_duration: Duration::ZERO,
            batch_size: 1,
        };
        assert_eq!(profile.num_layers(), 0);
        assert!(profile.bottleneck_index().is_none());
        assert!(profile.bottleneck().is_none());
        assert_eq!(profile.total_flops(), 0);
    }

    #[test]
    fn inference_profile_bottleneck() {
        let layers = vec![
            LayerProfile {
                index: 0,
                duration: Duration::from_micros(100),
                input_size: 64,
                output_size: 32,
                activation: Activation::ReLU,
                estimated_flops: 4096,
            },
            LayerProfile {
                index: 1,
                duration: Duration::from_micros(500), // 最も遅い
                input_size: 32,
                output_size: 16,
                activation: Activation::None,
                estimated_flops: 1024,
            },
            LayerProfile {
                index: 2,
                duration: Duration::from_micros(200),
                input_size: 16,
                output_size: 8,
                activation: Activation::None,
                estimated_flops: 256,
            },
        ];
        let profile = InferenceProfile {
            layers,
            total_duration: Duration::from_micros(800),
            batch_size: 1,
        };
        assert_eq!(profile.bottleneck_index(), Some(1));
        assert_eq!(profile.bottleneck().unwrap().index, 1);
    }

    #[test]
    fn inference_profile_total_flops() {
        let layers = vec![
            LayerProfile {
                index: 0,
                duration: Duration::from_micros(100),
                input_size: 64,
                output_size: 32,
                activation: Activation::None,
                estimated_flops: 1000,
            },
            LayerProfile {
                index: 1,
                duration: Duration::from_micros(200),
                input_size: 32,
                output_size: 16,
                activation: Activation::None,
                estimated_flops: 2000,
            },
        ];
        let profile = InferenceProfile {
            layers,
            total_duration: Duration::from_micros(300),
            batch_size: 1,
        };
        assert_eq!(profile.total_flops(), 3000);
        assert_eq!(profile.num_layers(), 2);
    }

    #[test]
    fn inference_profile_total_gflops() {
        let profile = InferenceProfile {
            layers: vec![LayerProfile {
                index: 0,
                duration: Duration::from_millis(1),
                input_size: 0,
                output_size: 0,
                activation: Activation::None,
                estimated_flops: 1_000_000, // 1 MFLOP
            }],
            total_duration: Duration::from_secs(1),
            batch_size: 1,
        };
        // 1 MFLOP / 1s = 0.001 GFLOP/s
        assert!((profile.total_gflops() - 0.001).abs() < 0.0001);
    }

    #[test]
    fn inference_profile_total_gflops_zero_duration() {
        let profile = InferenceProfile {
            layers: vec![],
            total_duration: Duration::ZERO,
            batch_size: 1,
        };
        assert!((profile.total_gflops()).abs() < 0.01);
    }

    #[test]
    fn inference_profile_total_duration_us() {
        let profile = InferenceProfile {
            layers: vec![],
            total_duration: Duration::from_micros(1234),
            batch_size: 1,
        };
        assert!((profile.total_duration_us() - 1234.0).abs() < 1.0);
    }

    #[test]
    fn inference_profile_display() {
        let profile = InferenceProfile {
            layers: vec![LayerProfile {
                index: 0,
                duration: Duration::from_micros(100),
                input_size: 64,
                output_size: 32,
                activation: Activation::ReLU,
                estimated_flops: 4096,
            }],
            total_duration: Duration::from_micros(100),
            batch_size: 1,
        };
        let s = format!("{profile}");
        assert!(s.contains("InferenceProfile"));
        assert!(s.contains("1 layers"));
        assert!(s.contains("batch=1"));
        assert!(s.contains("Layer 0"));
        assert!(s.contains("Bottleneck"));
    }

    // --- GPU テスト (GPUが使えない環境ではスキップ) ---

    #[test]
    fn profile_forward_single_layer() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        let w = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::ReLU);

        let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);
        let profile = InferenceProfiler::profile_forward(&engine, &device, &compute, &input);

        assert_eq!(profile.num_layers(), 1);
        assert_eq!(profile.batch_size, 1);
        assert!(profile.total_duration.as_nanos() > 0);
        assert_eq!(profile.layers[0].activation, Activation::ReLU);
        // FLOPs: 2 * 2 * 2 = 8
        assert_eq!(profile.layers[0].estimated_flops, 8);
    }

    #[test]
    fn profile_forward_two_layers() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        let w1 = GpuTernaryWeight::from_ternary(&device, &[1, 1, -1, 1], 2, 2);
        let w2 = GpuTernaryWeight::from_ternary(&device, &[1, 1], 1, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w1, Activation::ReLU);
        engine.add_layer(w2, Activation::None);

        let input = GpuTensor::from_f32(&device, &[1.0, 2.0], &[2]);
        let profile = InferenceProfiler::profile_forward(&engine, &device, &compute, &input);

        assert_eq!(profile.num_layers(), 2);
        assert!(profile.bottleneck_index().is_some());
        assert!(profile.total_flops() > 0);
    }

    #[test]
    fn profile_forward_batch() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        let w = GpuTernaryWeight::from_ternary(&device, &[1, 1], 1, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::None);

        let input = GpuTensor::from_f32(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let profile =
            InferenceProfiler::profile_forward_batch(&engine, &device, &compute, &input, 2);

        assert_eq!(profile.batch_size, 2);
        assert_eq!(profile.num_layers(), 1);
        // FLOPs: 2 * 2 * 1 * 2 = 8
        assert_eq!(profile.layers[0].estimated_flops, 8);
    }

    #[test]
    fn profile_bottleneck_single_layer() {
        let Ok(device) = GpuDevice::new() else { return };
        let compute = TernaryCompute::new(&device);

        let w = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
        let mut engine = GpuInferenceEngine::new();
        engine.add_layer(w, Activation::None);

        let input = GpuTensor::from_f32(&device, &[1.0, 2.0], &[2]);
        let profile = InferenceProfiler::profile_forward(&engine, &device, &compute, &input);

        // 1レイヤーならボトルネックは自身
        assert_eq!(profile.bottleneck_index(), Some(0));
    }

    #[test]
    fn layer_profile_clone() {
        let lp = LayerProfile {
            index: 0,
            duration: Duration::from_micros(100),
            input_size: 64,
            output_size: 32,
            activation: Activation::ReLU,
            estimated_flops: 4096,
        };
        let cloned = lp.clone();
        assert_eq!(cloned.index, lp.index);
        assert_eq!(cloned.estimated_flops, lp.estimated_flops);
    }

    #[test]
    fn inference_profile_clone() {
        let profile = InferenceProfile {
            layers: vec![],
            total_duration: Duration::from_micros(500),
            batch_size: 4,
        };
        let cloned = profile.clone();
        assert_eq!(cloned.batch_size, 4);
        assert_eq!(cloned.total_duration, Duration::from_micros(500));
    }
}
