//! Analytic oracles — closed-form checks for the GPU laws in ALICE-TRT
//! (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Every expected value is a closed form or an f64 reference computed in this
//! file; nothing is produced by the crate function under test.  The GPU is
//! required: a missing adapter is a failure, not a skip (the CI matrix runs
//! this on macOS / Windows native and Linux lavapipe, like the fix128 suite).
//!
//! Oracle sources:
//! - ternary matvec: y_j = Σ_i W_ji·x_i with W ∈ {−1,0,1} and integer x is an
//!   exact integer (|y| < 2²⁴ ⇒ exact in f32); scaled kernels multiply by γ
//! - batched matmul = per-row matvec (bit identical), ReLU = max(0, ·)
//! - two-layer network: f64 closed form of W₂·relu(W₁·x)
//! - Fix128 (I64F64): 0.5·0.25 = 0.125, 1/4 = 0.25, √0.25 = 0.5 exactly in the
//!   bit pattern; √2 to 1e-12; dot of small integer vectors exact
//! - voice features on a pure tone A·sin(2πft): RMS = A/√2, zero-crossing
//!   rate = 2f/sr, |sin| centroid = ½ frame; HTK mel m(f) = 2595·log10(1 + f/700)
//! - view bridge: internal = ⌊output × render_scale⌋

use alice_trt::prelude::*;
use alice_trt::TernaryCompute;

fn device() -> GpuDevice {
    GpuDevice::new().expect("a GPU adapter (native or lavapipe) is required for the TRT oracles")
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0
}

fn ternary_matrix(out: usize, inp: usize, seed: &mut u64) -> Vec<i8> {
    (0..out * inp)
        .map(|_| {
            let u = lcg(seed);
            if u < -0.33 {
                -1
            } else if u > 0.33 {
                1
            } else {
                0
            }
        })
        .collect()
}

// ───────────────────────── GPU ternary compute ────────────────────────────

#[test]
fn gpu_ternary_matvec_is_the_exact_integer_dot_product_on_both_kernels() {
    let dev = device();
    let compute = TernaryCompute::new(&dev);
    let mut seed = 3u64;
    // in_features ≥ 1024 selects the tiled kernel, below it the simple one
    for (out, inp) in [
        (4usize, 8usize),
        (37, 100),
        (16, 1024),
        (3, 2048),
        (129, 1100),
    ] {
        let w = ternary_matrix(out, inp, &mut seed);
        let x: Vec<f32> = (0..inp)
            .map(|_| (lcg(&mut seed) * 8.0).round() as f32)
            .collect();
        let weights = GpuTernaryWeight::from_ternary(&dev, &w, out, inp);
        let input = GpuTensor::from_f32(&dev, &x, &[inp]);
        let y = compute.matvec(&dev, &input, &weights);
        dev.poll_wait();
        let y = y.download(&dev);
        assert_eq!(y.len(), out);
        for j in 0..out {
            // oracle: exact integer sum
            let expected: f64 = (0..inp).map(|i| w[j * inp + i] as f64 * x[i] as f64).sum();
            assert_eq!(y[j] as f64, expected, "{out}x{inp} y[{j}]");
        }
        // scaled kernel: y = γ · W x
        let gamma = 0.0625f32; // power of two ⇒ still exact
        let weights_s = GpuTernaryWeight::from_ternary_scaled(&dev, &w, out, inp, gamma);
        let ys = compute.matvec(&dev, &input, &weights_s);
        dev.poll_wait();
        let ys = ys.download(&dev);
        for j in 0..out {
            assert_eq!(ys[j], y[j] * gamma, "{out}x{inp} scaled y[{j}]");
        }
        // batched matmul: every row is the matvec of that row, bit for bit
        let batch = 5;
        let xb: Vec<f32> = (0..batch * inp)
            .map(|_| (lcg(&mut seed) * 8.0).round() as f32)
            .collect();
        let input_b = GpuTensor::from_f32(&dev, &xb, &[batch, inp]);
        let yb = compute.matmul_batch(&dev, &input_b, &weights, batch);
        dev.poll_wait();
        let yb = yb.download(&dev);
        assert_eq!(yb.len(), batch * out);
        for b in 0..batch {
            let row = GpuTensor::from_f32(&dev, &xb[b * inp..(b + 1) * inp], &[inp]);
            let yr = compute.matvec(&dev, &row, &weights);
            dev.poll_wait();
            let yr = yr.download(&dev);
            for j in 0..out {
                let expected: f64 = (0..inp)
                    .map(|i| w[j * inp + i] as f64 * xb[b * inp + i] as f64)
                    .sum();
                assert_eq!(yb[b * out + j] as f64, expected, "batch {b} y[{j}]");
                assert_eq!(
                    yb[b * out + j].to_bits(),
                    yr[j].to_bits(),
                    "batch row {b} ≠ matvec"
                );
            }
        }
        // ReLU in place: max(0, ·), bit exact
        compute.relu_inplace(&dev, &input);
        dev.poll_wait();
        let r = input.download(&dev);
        for i in 0..inp {
            assert_eq!(r[i], x[i].max(0.0), "relu[{i}]");
        }
    }
    // tensor round trip and zeros
    let data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.37).collect();
    let t = GpuTensor::from_f32(&dev, &data, &[10, 100]);
    assert_eq!(t.download(&dev), data);
    assert_eq!(t.shape(), &[10, 100]);
    assert!(GpuTensor::zeros(&dev, &[64])
        .download(&dev)
        .iter()
        .all(|&v| v == 0.0));
}

#[test]
fn gpu_two_layer_relu_network_matches_the_f64_closed_form() {
    let dev = device();
    let compute = TernaryCompute::new(&dev);
    let mut seed = 17u64;
    let (n0, n1, n2) = (64usize, 48usize, 10usize);
    let w1 = ternary_matrix(n1, n0, &mut seed);
    let w2 = ternary_matrix(n2, n1, &mut seed);
    let x: Vec<f32> = (0..n0)
        .map(|_| (lcg(&mut seed) * 4.0).round() as f32)
        .collect();
    let mut engine = GpuInferenceEngine::new();
    engine.add_layer(
        GpuTernaryWeight::from_ternary(&dev, &w1, n1, n0),
        Activation::ReLU,
    );
    engine.add_layer(
        GpuTernaryWeight::from_ternary(&dev, &w2, n2, n1),
        Activation::None,
    );
    let y = engine
        .forward(&dev, &compute, &GpuTensor::from_f32(&dev, &x, &[n0]))
        .download(&dev);
    // oracle: h = relu(W₁x), y = W₂h — integers throughout, exact
    let h: Vec<f64> = (0..n1)
        .map(|j| {
            (0..n0)
                .map(|i| w1[j * n0 + i] as f64 * x[i] as f64)
                .sum::<f64>()
                .max(0.0)
        })
        .collect();
    for k in 0..n2 {
        let expected: f64 = (0..n1).map(|j| w2[k * n1 + j] as f64 * h[j]).sum();
        assert_eq!(y[k] as f64, expected, "y[{k}]");
    }
    // oracle: 2 bit-planes of u32 words per row ⇒ bytes = out·⌈in/32⌉·8, vs in·out·4
    let packed = |out: usize, inp: usize| out * inp.div_ceil(32) * 8;
    let fp32 = (n1 * n0 + n2 * n1) * 4;
    let expected = fp32 as f32 / (packed(n1, n0) + packed(n2, n1)) as f32;
    assert!(
        (engine.compression_ratio() - expected).abs() < 1e-5,
        "{} vs {expected}",
        engine.compression_ratio()
    );
    assert_eq!(engine.total_weight_bytes(), packed(n1, n0) + packed(n2, n1));
    assert_eq!(engine.equivalent_fp32_bytes(), fp32);
}

// ───────────────────────── Fix128 GPU arithmetic ──────────────────────────

#[cfg(feature = "fix128-arithmetic")]
mod fix128_oracle {
    use super::device;
    use alice_trt::fix128::{Fix128Gpu, Fix128WgpuKernel};

    const HALF: Fix128Gpu = Fix128Gpu::from_raw(0, 1 << 63);
    const QUARTER: Fix128Gpu = Fix128Gpu::from_raw(0, 1 << 62);
    const EIGHTH: Fix128Gpu = Fix128Gpu::from_raw(0, 1 << 61);

    fn same(a: Fix128Gpu, b: Fix128Gpu) -> bool {
        a.hi == b.hi && a.lo == b.lo
    }

    #[test]
    fn fix128_gpu_add_sub_mul_div_sqrt_dot_match_closed_forms() {
        let dev = device();
        let k = Fix128WgpuKernel::new(&dev);
        // exact dyadic rationals: bit patterns must match, not just to_f64
        let a = [
            HALF,
            Fix128Gpu::from_int(3),
            Fix128Gpu::from_int(-3),
            QUARTER,
            Fix128Gpu::from_int(7),
        ];
        let b = [
            QUARTER,
            Fix128Gpu::from_int(5),
            HALF,
            QUARTER,
            Fix128Gpu::from_int(-2),
        ];
        let mut out = [Fix128Gpu::from_int(0); 5];

        k.add(&a, &b, &mut out);
        let want = [
            Fix128Gpu::from_raw(0, (1u64 << 63) + (1u64 << 62)), // 0.75
            Fix128Gpu::from_int(8),
            Fix128Gpu::from_raw(-3, 1 << 63), // −2.5
            HALF,
            Fix128Gpu::from_int(5),
        ];
        for i in 0..5 {
            assert!(same(out[i], want[i]), "add[{i}] {} vs {}", out[i], want[i]);
        }
        k.sub(&a, &b, &mut out);
        let want = [
            QUARTER,
            Fix128Gpu::from_int(-2),
            Fix128Gpu::from_raw(-4, 1 << 63), // −3.5
            Fix128Gpu::from_int(0),
            Fix128Gpu::from_int(9),
        ];
        for i in 0..5 {
            assert!(same(out[i], want[i]), "sub[{i}] {} vs {}", out[i], want[i]);
        }
        k.mul(&a, &b, &mut out);
        let want = [
            EIGHTH,
            Fix128Gpu::from_int(15),
            Fix128Gpu::from_raw(-2, 1 << 63), // −1.5
            Fix128Gpu::from_raw(0, 1 << 60),  // 1/16
            Fix128Gpu::from_int(-14),
        ];
        for i in 0..5 {
            assert!(same(out[i], want[i]), "mul[{i}] {} vs {}", out[i], want[i]);
        }
        // division: exact dyadic results, then 1/3 to 1e-15
        let num = [
            Fix128Gpu::from_int(1),
            Fix128Gpu::from_int(-6),
            HALF,
            Fix128Gpu::from_int(1),
        ];
        let den = [
            Fix128Gpu::from_int(4),
            Fix128Gpu::from_int(4),
            QUARTER,
            Fix128Gpu::from_int(3),
        ];
        let mut q = [Fix128Gpu::from_int(0); 4];
        k.div(&num, &den, &mut q);
        assert!(same(q[0], QUARTER), "1/4 = {}", q[0]);
        assert!(
            same(q[1], Fix128Gpu::from_raw(-2, 1 << 63)),
            "−6/4 = {}",
            q[1]
        );
        assert!(same(q[2], Fix128Gpu::from_int(2)), "0.5/0.25 = {}", q[2]);
        assert!(
            (q[3].to_f64() - 1.0 / 3.0).abs() < 1e-15,
            "1/3 = {}",
            q[3].to_f64()
        );
        // sqrt: exact for perfect squares, 1e-12 for √2
        let s_in = [
            QUARTER,
            Fix128Gpu::from_int(9),
            Fix128Gpu::from_int(2),
            Fix128Gpu::from_int(0),
        ];
        let mut s = [Fix128Gpu::from_int(0); 4];
        k.sqrt(&s_in, &mut s);
        assert!(same(s[0], HALF), "√0.25 = {}", s[0]);
        assert!(same(s[1], Fix128Gpu::from_int(3)), "√9 = {}", s[1]);
        assert!(
            (s[2].to_f64() - std::f64::consts::SQRT_2).abs() < 1e-12,
            "√2 = {}",
            s[2].to_f64()
        );
        assert!(same(s[3], Fix128Gpu::from_int(0)));
        // dot: Σ i·(i+1) for i = 1..=64 = 64·65·66/3 = 91 520, plus a fractional case
        let v: Vec<Fix128Gpu> = (1..=64).map(Fix128Gpu::from_int).collect();
        let w: Vec<Fix128Gpu> = (2..=65).map(Fix128Gpu::from_int).collect();
        assert!(
            same(k.dot(&v, &w), Fix128Gpu::from_int(91_520)),
            "dot = {}",
            k.dot(&v, &w)
        );
        let halves = vec![HALF; 1000];
        let quarters = vec![QUARTER; 1000];
        assert!(
            same(k.dot(&halves, &quarters), Fix128Gpu::from_int(125)),
            "1000 · ⅛ = {}",
            k.dot(&halves, &quarters)
        );
        // CPU reference on the struct itself agrees with the GPU kernel
        assert!(same(HALF.mul(QUARTER), EIGHTH));
        assert!(same(
            Fix128Gpu::from_int(-3).add(HALF),
            Fix128Gpu::from_raw(-3, 1 << 63)
        ));
    }
}

// ───────────────────────── voice / view bridges (feature-gated, no deps) ──

#[cfg(feature = "voice")]
#[test]
fn voice_features_of_a_pure_tone_match_their_closed_forms() {
    use alice_trt::voice_bridge::{
        extract_features, frame_energy, mel_center_frequencies, MelConfig,
    };
    let cfg = MelConfig::default();
    assert_eq!(
        (cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.hop_length),
        (16_000, 512, 40, 160)
    );
    let sr = f64::from(cfg.sample_rate);
    let (f, amp) = (1000.0f64, 0.8f64);
    let n: usize = 16_000; // 1 s
    let tone: Vec<f32> = (0..n)
        .map(|i| (amp * (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin()) as f32)
        .collect();
    let feats = extract_features(&tone, &cfg);
    // frame count closed form
    let frames = (n - cfg.n_fft) / cfg.hop_length + 1;
    assert_eq!(feats.len(), frames * 3);
    for fr in 0..frames {
        let energy = f64::from(feats[fr * 3]);
        let zcr = f64::from(feats[fr * 3 + 1]);
        let centroid = f64::from(feats[fr * 3 + 2]);
        // 512 samples of a 1 kHz tone at 16 kHz = 32 whole cycles ⇒ RMS = A/√2
        assert!(
            (energy - amp / 2f64.sqrt()).abs() < 1e-3,
            "frame {fr}: energy {energy}"
        );
        // zero crossings: 2 per cycle ⇒ 64 per frame (±1 at the edges) / 512
        assert!(
            (zcr - 2.0 * f / sr).abs() <= 2.0 / cfg.n_fft as f64,
            "frame {fr}: zcr {zcr}"
        );
        // |sin| is symmetric over whole cycles ⇒ centroid at the frame middle
        assert!(
            (centroid - 0.5).abs() < 0.01,
            "frame {fr}: centroid {centroid}"
        );
    }
    assert!(
        extract_features(&tone[..cfg.n_fft - 1], &cfg).is_empty(),
        "shorter than one frame"
    );
    // RMS closed form on a known vector: √((1+4+9+16)/4) = √7.5
    assert!((f64::from(frame_energy(&[1.0, -2.0, 3.0, -4.0])) - 7.5f64.sqrt()).abs() < 1e-6);
    assert_eq!(frame_energy(&[]), 0.0);
    // HTK mel centres: m(f_i) = m(f_max) · (i + 1)/(n + 1), strictly increasing, below Nyquist
    let centres = mel_center_frequencies(cfg.n_mels, cfg.sample_rate);
    let mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let m_max = mel(sr / 2.0);
    for (i, &c) in centres.iter().enumerate() {
        let expected_mel = m_max * (i as f64 + 1.0) / (cfg.n_mels as f64 + 1.0);
        assert!(
            (mel(f64::from(c)) - expected_mel).abs() < 1e-3 * expected_mel,
            "mel centre {i}: {c} Hz"
        );
        assert!(f64::from(c) < sr / 2.0);
        if i > 0 {
            assert!(c > centres[i - 1]);
        }
    }
}

#[cfg(feature = "view")]
#[test]
fn view_bridge_internal_resolution_is_output_times_render_scale() {
    use alice_trt::view_bridge::{internal_resolution, render_scale, UpscaleQuality};
    for q in [
        UpscaleQuality::Performance,
        UpscaleQuality::Balanced,
        UpscaleQuality::Quality,
        UpscaleQuality::UltraQuality,
    ] {
        let s = render_scale(q);
        assert!(s > 0.0 && s < 1.0);
        for (w, h) in [(1920u32, 1080u32), (3840, 2160), (1, 1), (7, 3)] {
            let (iw, ih) = internal_resolution(w, h, q);
            assert_eq!(iw, ((w as f32 * s) as u32).max(1));
            assert_eq!(ih, ((h as f32 * s) as u32).max(1));
            assert!(iw <= w && ih <= h);
        }
    }
}

// ───────────────────────── SDF bridge (feature `sdf`, path dep) ───────────

#[cfg(feature = "sdf")]
#[test]
#[ignore = "GpuNeuralSdf::fit は未実装 (2026-09-17 まで固定 +1/−1/0 pattern を『fit』と称して返していた、今は todo! で fail fast) — Backlog ALICE-TRT の fit 実装 (alice-train STE) 後に ignore を外す"]
fn neural_sdf_fit_of_a_unit_sphere_is_within_a_quarter_radius() {
    use alice_sdf::prelude::{SdfNode, Vec3};
    use alice_trt::sdf_bridge::{GpuNeuralSdf, NeuralSdfConfig};
    let dev = device();
    let compute = TernaryCompute::new(&dev);
    let sphere = SdfNode::sphere(1.0);
    let field = GpuNeuralSdf::fit(
        &dev,
        &sphere,
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
        &NeuralSdfConfig::default(),
    );
    // oracle: |p| − 1 on a 9³ grid; a fit worth the name is within r/4 RMS
    let mut pts = Vec::new();
    let mut truth = Vec::new();
    for iz in 0..9 {
        for iy in 0..9 {
            for ix in 0..9 {
                let p = [
                    ix as f32 * 0.5 - 2.0,
                    iy as f32 * 0.5 - 2.0,
                    iz as f32 * 0.5 - 2.0,
                ];
                pts.extend_from_slice(&p);
                truth.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 1.0);
            }
        }
    }
    let got = field.eval_batch(&dev, &compute, &pts);
    let rms = (got
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / truth.len() as f32)
        .sqrt();
    assert!(rms <= 0.25, "sphere fit RMS error {rms} > r/4");
}
