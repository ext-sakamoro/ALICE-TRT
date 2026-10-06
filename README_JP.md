# ALICE-TRT

1.58-bit 三値ニューラルネットワーク向けの GPU 推論エンジンと、ALICE-Physics
のソルバの一部を GPU 上でバイト完全一致のまま実行するための Fix128 (128-bit
固定小数点) compute kernel 群

[English](README.md) | 日本語

[![CI](https://github.com/ext-sakamoro/ALICE-TRT/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-TRT/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.87-blue)](#最小サポート-rust-バージョン)
[![License](https://img.shields.io/badge/license-AGPL--3.0%20OR%20Commercial-blue)](#ライセンス)

三値重み `{-1, 0, +1}` は VRAM 上に 2 枚の bitplane (1 重み 2 bit、`u32` 1 word に
32 重み) として置かれ、compute shader 内のレジスタで `{-1.0, 0.0, +1.0}` に展開
される 1 重みあたり 16 bit (FP16) ではなく 2 bit を運ぶので、matrix-vector 積を
律速するメモリ転送量が減る 積和演算器から見えるのは通常の浮動小数点データ

```text
FP16 weights:   VRAM ──[16 bits/weight]──▶ L2 ──▶ L1 ──▶ registers ──▶ FMA
ALICE-TRT:      VRAM ──[ 2 bits/weight]──▶ L2 ──▶ L1 ──▶ registers ──[expand]──▶ FMA
```

既定の backend は [wgpu](https://wgpu.rs) (Metal / Vulkan / DX12) CUDA backend
と TensorRT plugin は `cuda` feature と `csrc/` にソースとして含まれる

汎用の推論 runtime ではない モデル形式の loader は無く、演算は三値 matvec /
batched matmul / ReLU のみで、学習機能も無い 数千行未満では wgpu の dispatch
overhead (約 1.3 ms) が支配的で、ALICE-ML のような CPU 実装の方が速い
([性能](#性能) 参照)

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [含まれるもの](#含まれるもの)
- [Cargo feature](#cargo-feature)
- [検証と既知の不具合](#検証と既知の不具合)
- [バインディング](#バインディング)
- [性能](#性能)
- [最小サポート Rust バージョン](#最小サポート-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [関連 crate](#関連-crate)
- [ライセンス](#ライセンス)

## インストール

ALICE-TRT は crates.io に公開していない
[ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML) に path 依存しているので、
2 つの repository を並べて checkout する

```sh
git clone https://github.com/ext-sakamoro/ALICE-ML.git
git clone https://github.com/ext-sakamoro/ALICE-TRT.git
```

```toml
[dependencies]
alice-trt = { path = "../ALICE-TRT" }
```

`physics` / `physics-solver`、`sdf`、`db` feature はさらに `../ALICE-Physics`、
`../ALICE-SDF`、`../ALICE-DB` を必要とする

## 使用例

```rust
use alice_trt::prelude::*;

// 1. Initialize GPU
let device = GpuDevice::new().unwrap();
let compute = TernaryCompute::new(&device);

// 2. Upload ternary weights (2-bit in VRAM)
let weights = GpuTernaryWeight::from_ternary(
    &device, &[1, -1, 0, 1], 2, 2
);

// 3. Upload input
let input = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);

// 4. GPU inference (stays in VRAM)
let output = compute.matvec(&device, &input, &weights);

// 5. Download result
let result = output.download(&device);
// result ≈ [-1.0, 3.0]
```

多層ネットワークは `GpuInferenceEngine::add_layer(weights, Activation::ReLU)`
で組み、`forward` / `forward_batch` で実行する 中間の活性値は VRAM に留まる
ALICE-ML が CPU で量子化した重みは `GpuTernaryWeight::from_kernel` /
`from_packed` で GPU に載せる

## 含まれるもの

| 分野 | 項目 | Feature |
|------|------|---------|
| 三値推論 (wgpu) | `GpuDevice`、`GpuTernaryWeight` (2-bit bitplane)、`GpuTensor`、`TernaryCompute` (`matvec` / `matmul_batch` / `relu_inplace`、layer の大きさで simple / tiled kernel を選択)、`GpuInferenceEngine` (多層 forward) | 既定 |
| プロファイル | `InferenceProfiler` (layer 間で GPU 同期を取った layer 単位の wall-clock 時間、FLOP 推定) | 既定 |
| 制約の彩色 | `constraint_graph::ConstraintGraph::greedy_color` (batched contact solve が使う決定論的な貪欲彩色) | 既定 |
| Fix128 GPU 演算 | `Fix128Gpu` (`alice_physics::Fix128` と同じ layout)、add / sub / mul / div / sqrt / dot、quaternion の積と回転の WGSL kernel | `fix128-arithmetic` |
| Fix128 GPU 物理 pipeline | AABB helper、Morton code と sort、BVH build、BVH `find_pairs`、球-球 contact、PGS integrate / floor / distance 射影、PGS contact solve (逐次と色分け batched)、ball-socket joint solve | `fix128-arithmetic` |
| 物理ソルバの offload | `physics_bridge::TrtSolverAdapter` (`alice_physics::gpu_bridge::GpuSolverBridge` の実装、`PhysicsWorld::step_with_bridge` に注入する) | `physics-solver` |
| 物理の制御 policy | `physics_bridge::GpuPhysicsController` (剛体に加える力を出す batched 三値ネットワーク) | `physics` |
| CUDA backend | `csrc/bitnet_kernel.cu` を使う `cuda::CudaTernaryEngine` (compute capability で wmma / dp4a / popc kernel を選択) | `cuda` |
| TensorRT plugin | `csrc/plugin/` (`IPluginV2DynamicExt`、Cargo でなく CMake で build) | なし |
| C ABI | device / weight / tensor / compute / engine の `extern "C"` 関数 37 個 | `ffi` |
| Python | 5 class の PyO3 module | `python` |
| Neural SDF | `sdf_bridge::GpuNeuralSdf` (`eval_batch`、`fit` は未実装、下記参照) | `sdf` |
| 推論 metrics | `db_bridge::TrtDbStore` (ALICE-DB に 34 byte の `InferenceRecord`) | `db` |
| View / voice helper | `view_bridge` (render scale、内部解像度、時間と PSNR の見積り、画素処理はしない)、`voice_bridge` (mel / energy / zero-crossing / centroid 特徴量を CPU で計算) | `view`、`voice` |

CUDA backend は実行時に compute capability で 3 つの kernel から 1 つを選ぶ

| Kernel | Compute capability | 方式 |
|--------|--------------------|------|
| `wmma` | 7.0 以降 | 2-bit → shared memory 上の FP16 `{-1, 0, +1}` tile → `wmma::mma_sync` (Tensor Core) |
| `dp4a` | 6.1 以降 | 2-bit → INT8 → `__dp4a` 4 要素内積 |
| `popc` | 任意 | bitplane に `__popc` / `__ffs`、非ゼロ重みの無い word は飛ばし、疎な word は立っている bit だけ走査 |

API の全体は生成ドキュメント (`cargo doc --open`) にある 物理 pipeline の設計
メモは [docs/PHASE_3_DESIGN.md](docs/PHASE_3_DESIGN.md) と
[docs/PHASE_4_DESIGN.md](docs/PHASE_4_DESIGN.md)、v2 から v3 への API 変更は
[docs/MIGRATION_v3.md](docs/MIGRATION_v3.md)

## Cargo feature

| Feature | 有効になるもの | 必要なもの |
|---------|----------------|------------|
| `ffi` | C ABI (`src/ffi.rs`) | |
| `python` | PyO3 binding | Python 3.7 から 3.13 |
| `cuda` | CUDA backend | NVIDIA CUDA Toolkit (`nvcc`) |
| `physics` | `GpuPhysicsController` | `../ALICE-Physics` |
| `sdf` | Neural SDF bridge | `../ALICE-SDF` |
| `db` | ALICE-DB への推論 metrics | `../ALICE-DB` |
| `view` | View helper | |
| `voice` | Voice 特徴量 helper | |
| `fix128-arithmetic` | Fix128 GPU kernel | |
| `physics-solver` | `TrtSolverAdapter`、`physics` / `fix128-arithmetic` / `alice-physics/gpu-solver-bridge` を含む | `../ALICE-Physics` |

既定で有効な feature は無い

## 検証と既知の不具合

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) が検査するもの

- **CPU-GPU のバイト完全一致 golden**: Fix128 の全 kernel と物理 pipeline の各段
  (Morton sort、BVH build、`find_pairs`、球-球 contact、PGS contact solve の逐次と
  batched、`step_with_bridge`) で、GPU の結果が ALICE-Physics の CPU の結果と
  bit 単位で一致すること Metal (macOS)、Vulkan lavapipe (Linux、software)、
  DX12 WARP (Windows、software) の 3 adapter で実行する
  <!-- claim-test: wgpu_pgs_contact_solve_matches_cpu_golden -->
- **閉形式 oracle** ([tests/analytic_oracle.rs](tests/analytic_oracle.rs)): 両
  kernel の三値 matvec が整数内積に一致、scaled kernel は scale 倍、batched の各行
  が単独 matvec と一致、2 層 ReLU network が f64 閉形式と一致、Fix128 演算が
  dyadic な閉形式と一致、view / voice helper が式と一致 GPU adapter が無い場合は
  skip でなく失敗にする
  <!-- claim-test: gpu_ternary_matvec_is_the_exact_integer_dot_product_on_both_kernels -->
- Linux / macOS / Windows での device を使わない unit test と doctest、
  `-D warnings` の clippy と rustdoc、MSRV の build、
  [security-audit.yml](.github/workflows/security-audit.yml) の `cargo audit` /
  `cargo deny` / `cargo machete`

CI で build しないもの: `cuda` feature と TensorRT plugin (NVIDIA の runner が無い)、
`sdf` / `db` feature (依存先の crate がさらに CI で checkout しない path 依存を持つ)

既知の不具合

- **Windows DX12 WARP で analytic oracle の process が落ちる**
  `Fix128 GPU (windows-latest)` job で `tests/analytic_oracle.rs` が
  `gpu_ternary_matvec_is_the_exact_integer_dot_product_on_both_kernels` の中で
  `STATUS_ACCESS_VIOLATION` により終了する 同じ adapter で Fix128 GPU の unit
  test は通る 修正まで job は red
- **Windows DX12 WARP で BVH build の golden が落ちる**
  `Fix128 GPU + physics-solver goldens (windows-latest)` job で test process が
  `fix128::tests::wgpu_bvh_build_matches_cpu_golden` の中で exit code 2173 で
  終了する (BVH build kernel の追加時から) 修正まで job は red、Metal と Vulkan
  lavapipe は通る
- **`GpuNeuralSdf::fit` は未実装** `todo!` で panic する 以前の版は SDF と無関係
  なネットワークを返していた oracle
  `neural_sdf_fit_of_a_unit_sphere_is_within_a_quarter_radius` は fit が実装される
  まで `#[ignore]`
- **`db` feature は現在の ALICE-DB に対して compile できない**
  (`src/db_bridge.rs` が ALICE-DB に既に無い `range` method と `put` の signature
  を使っている)
- **`TrtSolverAdapter::assert_bit_exact_vs_cpu` は実行時に何も比較せず**
  `Ok(())` を返す バイト完全一致の根拠は上記の golden test
- `TrtSolverAdapter::send_joints` は `Joint::Ball` のみ対応し、それ以外の joint
  では panic する ([docs/PHASE_4_DESIGN.md](docs/PHASE_4_DESIGN.md) 参照)

## バインディング

| 言語 | File | 内容 |
|------|------|------|
| C | [csrc/alice_trt_c_api.h](csrc/alice_trt_c_api.h) | CUDA kernel の C interface (`src/cuda/` から FFI で呼ぶ) |
| C# (Unity) | [bindings/unity/AliceTrt.cs](bindings/unity/AliceTrt.cs) | `ffi` 関数の `[DllImport]` 宣言と `IDisposable` handle |
| C++ (UE5) | [bindings/ue5/AliceTrt.h](bindings/ue5/AliceTrt.h) | `extern "C"` 宣言と RAII の `unique_ptr` handle |
| Python | `src/python.rs` | PyO3 class `GpuDevice` / `GpuTernaryWeight` / `GpuTensor` / `TernaryCompute` / `InferenceEngine` |

共有ライブラリは `cargo build --release --features ffi` で作る
(`libalice_trt.dylib` / `libalice_trt.so` / `alice_trt.dll`)

## 性能

`benches/gpu_matmul.rs` (Criterion) で、arm64 laptop GPU 上の wgpu Metal backend
で測定 CPU 列は同じ機械での ALICE-ML の branchless 三値 matvec
<!-- perf-measured: 2026-02-03 benches/gpu_matmul.rs -->

GPU matvec、N × N 三値重み

| N | GPU (wgpu) | 備考 |
|---|-----------|------|
| 64 | 1.37 ms | dispatch overhead が支配的 |
| 256 | 1.28 ms | dispatch overhead が支配的 |
| 512 | 1.48 ms | |
| 1024 | 1.57 ms | tiled kernel (shared memory + 並列 reduction) |
| 2048 | 1.79 ms | |
| 4096 | 2.61 ms | |

CPU (ALICE-ML) と GPU

| N | CPU (ALICE-ML) | GPU (wgpu) | 速い方 |
|---|---------------|-----------|--------|
| 256 | 60 µs | 1.28 ms | CPU、21 倍 |
| 1024 | 912 µs | 1.34 ms | CPU、1.5 倍 |
| 4096 | 21.3 ms | 2.52 ms | GPU、8.4 倍 |

交点は N = 2048 付近 512 × 512 重みの batched matmul: batch 1 で 1.45 ms、4 で
1.59 ms、16 で 1.38 ms、64 で 2.45 ms、256 で 4.03 ms

1024 × 1024 layer の重みメモリ: FP32 4,194,304 byte、FP16 2,097,152、INT8
1,048,576、ALICE-TRT 262,144 (FP32 の 1/16)

## 最小サポート Rust バージョン

<!-- readme-sync: msrv -->
Rust 1.87 (Cargo.toml の `rust-version`) 下限は ALICE-ML への path 依存で決まる
CI はちょうどこの toolchain で library を compile する 開発用 toolchain は
`rust-toolchain.toml` で 1.92.0 に固定している

## ビルドとテスト

```sh
cargo build --release                       # wgpu backend
cargo build --release --features ffi        # + C ABI
cargo build --release --features cuda       # + CUDA backend (needs nvcc)
cargo test --features fix128-arithmetic,physics-solver -- --test-threads=1
cargo bench --bench gpu_matmul
scripts/preflight.sh                        # every CI gate that runs locally
```

GPU test には adapter が要る GPU の無い Linux では Mesa の lavapipe
(`mesa-vulkan-drivers`) が adapter になる TensorRT plugin は別に build する

```sh
cd csrc && mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT
make -j"$(nproc)"
```

## 関連 crate

| Crate | 関係 |
|-------|------|
| [ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML) | CPU の三値推論、重みは `from_kernel` / `from_packed` で GPU に移る |
| [ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics) | 決定論的な Fix128 物理、`TrtSolverAdapter` がその `GpuSolverBridge` を実装する |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | `sdf` bridge の SDF tree |
| [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) | `db` bridge の保存先 |

## ライセンス

`AGPL-3.0 OR LicenseRef-Commercial`: デュアルライセンス、どちらかを選ぶ

| 選択肢 | 条件 | 使う場面 |
|--------|------|----------|
| **AGPL-3.0** | [LICENSE-AGPL](LICENSE-AGPL)、無償、報告義務なし | 自分のプロジェクトも AGPL 互換の OSS、または社内利用のみ |
| **商用ライセンス** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)、有償、copyleft を外す | クローズドソース製品、商用 SaaS、エッジ / ファームウェア配布、plugin の再配布、ソース開示を禁じるプラットフォーム NDA |

AGPL は強い copyleft で、`alice-trt` を link して配布またはサービス提供する製品、
ファームウェア、サービスは AGPL で公開する必要がある これはオープンな
エコシステムのための意図的な条件で、それができない場合のために商用ライセンスがある

商用ライセンスの問い合わせ: <contact@extoria.co.jp>
