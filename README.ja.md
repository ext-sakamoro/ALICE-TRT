# ALICE-TRT

**GPU 三値推論エンジン — トロイの木馬版**

> 「データを動かすな。法則を動かせ。」
> 「GPU は FP16 を計算していると思っている。実際は {-1, 0, +1} を計算している。」

**現在のリリース: v1.3.1** — semver 安定な PGS ソルバブリッジ (重力 + 床 + 複数距離制約、Metal / Vulkan / DX12 WARP でバイト完全一致)。詳細は [Cross-Crate Bridges](#クロスクレートブリッジ) を参照。

ALICE-TRT は 1.58-bit 三値ニューラルネットワーク向けに **メモリ帯域圧縮** を活かした GPU 推論エンジン。重みは VRAM 上に 2-bit ビットプレーンで格納され、FP16 の 1/8 の帯域で L1 まで運ばれ、compute unit 直前にレジスタ内で展開される。

[Project A.L.I.C.E.](https://github.com/ext-sakamoro) エコシステムの一部。

英語版: [README.md](README.md)

## トロイの木馬戦略

```
従来の GPU 推論:
  VRAM ──[FP16 weights]──▶ L2 ──▶ L1 ──▶ Registers ──▶ FMA
  帯域: 16 bit/重み (ボトルネック!)

ALICE-TRT:
  VRAM ──[2-bit packed]──▶ L2 ──▶ L1 ──▶ Registers ──[展開]──▶ FMA
  帯域: 2 bit/重み (8x 削減!)
  展開は ~2 cycle、帯域節約分は ~100 cycle 得。
```

CPU (ALICE-ML) 側の trick は「乗算を排除、加算のみで済ませる」。
GPU 側では乗算は無料 (Tensor Core が 125+ TFLOPS で回す)。
**ボトルネックはメモリ帯域**。だからデータそのものを圧縮する。

Tensor Core は有効な FP16 fragment `{-1.0, 0.0, +1.0}` を受け取る。
元が 2-bit だったなど知る由もない。これがトロイの木馬。

## アーキテクチャ

- **Backend 1**: wgpu (クロスプラットフォーム、Metal / Vulkan / DX12)
- **Backend 2**: CUDA (NVIDIA GPU、`--features cuda`)
- **Backend 3**: TensorRT plugin (オプション)
- **重み格納**: `plus_bits: u32[]` + `minus_bits: u32[]` + `scale: f32` (FP16 の 1/8)

構造図の詳細は [README.md](README.md#architecture) を参照。

## CUDA カーネルディスパッチ階層

実行時に GPU の compute capability に応じて自動選択:

| Kernel | Compute Cap | 戦略 | スループット |
|--------|-------------|------|-------------|
| **wmma** | >= 7.0 (Volta / Turing / Ampere / Hopper) | 2-bit → FP16 → `wmma::mma_sync` (Tensor Core) | 125+ TFLOPS |
| **dp4a** | >= 6.1 (Pascal GP106+) | 2-bit → INT8 → `__dp4a` (4-element dot、1 cycle) | 4x scalar |
| **popc** | 任意 (Maxwell、Kepler、…) | `__popc` + `__ffs` ビット並列 popcount、ゼロ重みスキップ | 32 weights/cycle |

## ベンチマーク

Apple M2 Max (wgpu/Metal backend) 実測。
CPU baseline: ALICE-ML (branchless 三値 matvec + SIMD)。

### GPU matvec: サイズ別スループット

| サイズ (N x N) | GPU (wgpu) | 備考 |
|--------------|-----------|-----|
| 64 x 64 | 1.37 ms | dispatch オーバヘッド支配 |
| 256 x 256 | 1.28 ms | dispatch オーバヘッド支配 |
| 1024 x 1024 | 1.57 ms | tiled kernel 発動 |
| 4096 x 4096 | 2.61 ms | compute-bound 領域 |

### CPU vs GPU: クロスオーバー点

| サイズ | CPU (ALICE-ML) | GPU (wgpu) | 勝者 |
|--------|---------------|-----------|-----|
| 256 | 60 us | 1.28 ms | CPU 21x 速い |
| 1024 | 912 us | 1.34 ms | CPU 1.5x 速い |
| **4096** | **21.3 ms** | **2.52 ms** | **GPU 8.4x 速い** |

> **~2048 でクロスオーバー**。それ以下は CPU 勝ち (wgpu dispatch ~1.3ms のため)、
> それ以上は GPU 勝ちでサイズが大きいほど差が広がる。
> NVIDIA CUDA (Tensor Core) ではクロスオーバーがもっと早く、差ももっと大きい。

### メモリ圧縮

| 形式 | サイズ (1024x1024) | 比 |
|-----|-------------------|-----|
| FP32 | 4,194,304 bytes | 1x |
| FP16 | 2,097,152 bytes | 2x |
| INT8 | 1,048,576 bytes | 4x |
| **ALICE-TRT (2-bit)** | **262,144 bytes** | **16x** |

## クイックスタート

### パスを選ぶ

役割ごとに最適な入り口を選んでください。全パスは同じ `GpuTernaryWeight` / `GpuTensor` プリミティブを共有するので、wgpu で書いて後から CUDA / FFI に載せ替えても書き直し不要。

| あなたは… | パス | 用途 | セクション |
|-----------|------|-----|-----------|
| **初めて触る** | [Hello, first inference](#hello-first-inference-30-秒) ↓ | インストール確認、1 回の matvec | ↓ |
| **クロスプラットフォーム Rust** | [wgpu backend](#wgpu-backend-クロスプラットフォームデフォルト) | Metal (macOS) / Vulkan (Linux) / DX12 (Windows)、NVIDIA 依存なし | ↓ |
| **NVIDIA CUDA ユーザー** | [CUDA backend](#cuda-backend-nvidia-gpu) | Tensor Core (`wmma` / `dp4a`)、最大スループット | ↓ |
| **マルチレイヤ推論** | Multi-layer (英語版参照) | 3+ 層 feed-forward、順伝播、活性化 | [README.md](README.md#multi-layer-inference) |
| **ALICE-ML (CPU) から移行** | Import from ALICE-ML | 量子化 CPU 重みを GPU で再利用、drop-in 移行 | [README.md](README.md#import-from-alice-ml) |
| **物理オフロード (Fix128)** | [物理ソルバブリッジ](#物理ソルバブリッジ-feature-physics-solverv100-以降-stable) | ALICE-Physics 向け GPU PGS、CPU と byte-exact | ↓ Cross-Crate |
| **Neural SDF (リアルタイム)** | [SDF ブリッジ](#sdf-ブリッジ-feature-sdf) | 三値 NN で CSG tree を fit、GPU 大量点評価 | ↓ Cross-Crate |
| **Unity / UE5 統合** | [C-ABI FFI](#c-abi-ffi-unity--ue5) | 37 個の `extern "C"` 関数、ABI version 付き | ↓ |
| **ベンチマーク / 研究** | [ベンチマーク](#ベンチマーク) + `cargo bench` | スループット、クロスオーバー、圧縮率 | ↑ |

### Hello, First Inference (30 秒)

最小の動くサンプル — wgpu クロスプラットフォーム backend で 2 要素 matvec 1 回:

```rust
use alice_trt::prelude::*;

let device  = GpuDevice::new().unwrap();
let compute = TernaryCompute::new(&device);

// 2x2 三値重み行列 { {1, -1}, {0, 1} }
let weights = GpuTernaryWeight::from_ternary(&device, &[1, -1, 0, 1], 2, 2);
let input   = GpuTensor::from_f32(&device, &[2.0, 3.0], &[2]);

let output = compute.matvec(&device, &input, &weights);
let result = output.download(&device);
assert_eq!(result, vec![-1.0, 3.0]);
```

これが動いたら表の中から自分の役割に合ったパスに進んでください。

### wgpu backend (クロスプラットフォーム、デフォルト)

詳細な Rust コード例は [README.md](README.md#quick-start) を参照。

```rust
use alice_trt::prelude::*;

let device = GpuDevice::new()?;
let compute = TernaryCompute::new(&device);
let weight = GpuTernaryWeight::from_ternary(&device, &plus_bits, &minus_bits, scale);
let input = GpuTensor::from_slice(&device, &input_data);
let output = compute.matvec(&device, &weight, &input);
let result = output.to_vec(&device);
```

### CUDA backend (NVIDIA GPU)

`--features cuda` を付けてビルド。CUDA Toolkit 12+ 必須。詳細は英語版参照。

### よく使うレシピ

繰り返し出てくるパターン。

**1. バッチ matmul (多入力、単一重み)**

```rust
let batch  = GpuTensor::from_f32(&device, &batched_input, &[batch_size, in_dim]);
let output = compute.matmul_batch(&device, &batch, &weights);
```

**2. ReLU 融合 (in-place、往復なし)**

```rust
let mut hidden = compute.matvec(&device, &input, &w1);
compute.relu_inplace(&device, &mut hidden);
let logits = compute.matvec(&device, &hidden, &w2);
```

**3. Physics ソルバブリッジ (v1.3.1、CPU と byte-exact)**

```rust
use alice_physics::gpu_bridge::{DiffFixture, GpuSolverBridge};
use alice_physics::math::Fix128;
use alice_trt::{GpuDevice, TrtSolverAdapter};

let device      = GpuDevice::new()?;
let mut adapter = TrtSolverAdapter::new(&device);
adapter.send_island(&positions, &velocities);
adapter.set_gravity(Some([Fix128::ZERO, Fix128::from_ratio(-98, 10), Fix128::ZERO]));
adapter.push_distance_constraint(0, 1, Fix128::from_int(2));
adapter.dispatch_iterations(10, Fix128::from_ratio(1, 60));
adapter.recv_island(&mut positions, &mut velocities);
```

`--features physics-solver` 必要。ALICE-Physics 側の `gpu-solver-bridge` は自動有効化。

**4. Neural SDF fit (三値 NN で CSG tree 近似)**

```rust
use alice_trt::sdf_bridge::{GpuNeuralSdf, NeuralSdfConfig};

let neural = GpuNeuralSdf::fit(
    &device, &sdf_node, bounds_min, bounds_max,
    &NeuralSdfConfig { hidden_dim: 64, num_layers: 4, training_points: 100_000, ..Default::default() },
)?;
let dists = neural.eval_batch(&device, &compute, &query_points)?;
```

`--features sdf` 必要。

### 次に見るべきドキュメント

| やりたいこと | 参照先 |
|------------|-------|
| トロイの木馬圧縮を理解 | [トロイの木馬戦略](#トロイの木馬戦略) |
| CUDA カーネル比較 (wmma / dp4a / popc) | [CUDA カーネルディスパッチ階層](#cuda-カーネルディスパッチ階層) |
| 生スループット数値 | [ベンチマーク](#ベンチマーク) |
| Fix128 物理を GPU オフロード | [クロスクレートブリッジ → 物理](#物理ソルバブリッジ-feature-physics-solverv100-以降-stable) |
| CSG から neural SDF fit | [クロスクレートブリッジ → SDF](#sdf-ブリッジ-feature-sdf) |
| 推論メトリクス永続化 | [ALICE-DB ブリッジ](#alice-db--alice-view--alice-voice-ブリッジ) |
| Unity / UE5 統合 | [C-ABI FFI](#c-abi-ffi-unity--ue5) + `bindings/unity/` + `bindings/ue5/` |
| CPU との byte-exact 保証 | [テストスイート](#テストスイート) (37 fix128 + 170 physics-solver、3 プラットフォーム CI) |

ALICE-TRT に直接プラグインできるコンパニオンクレート:

- **[ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML)** — CPU 三値推論、重みは `GpuTernaryWeight::from_kernel` で 1:1 移行
- **[ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics)** — Fix128 物理、`gpu-solver-bridge` で GPU PGS オフロード opt-in
- **[ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF)** — `sdf_bridge` neural-fit 推論の CSG ソース

## C-ABI FFI (Unity / UE5)

`--features ffi` で 37 関数の C-ABI ラッパー付き。Unity C# (P/Invoke) / UE5 C++ (extern "C") から呼べる。ABI 互換性は `version` エントリで検査。

## テストスイート

- 単体テスト: 総 2,600+ test
- Fix128 GPU: **37 test** (Metal / Vulkan / DX12 WARP 3 プラットフォーム CI)
- Physics solver bridge: **170 test** (バイト完全一致検証)
- 全リリースで 3 プラットフォーム CI green

## ビルド

```bash
# デフォルト: wgpu クロスプラットフォーム backend
cargo build --release

# CUDA backend (NVIDIA CUDA Toolkit 必須)
cargo build --release --features cuda

# FFI バインディング
cargo build --release --features ffi

# テスト
cargo test --release

# ベンチマーク
cargo bench

# TensorRT plugin (TensorRT SDK、Linux のみ)
cd tensorrt_plugin && make
```

### 要件

- Rust 1.75+
- wgpu backend: Metal (macOS) / Vulkan (Linux) / DX12 (Windows)
- CUDA backend: CUDA Toolkit 12+、NVIDIA GPU (compute cap 6.1+)
- TensorRT plugin: TensorRT 8.6+、Linux

## クロスクレートブリッジ

ALICE-TRT は feature-gated ブリッジモジュール経由で他の ALICE エコシステムクレートと連携:

| ブリッジ | Feature | 対象クレート | 説明 |
|---------|---------|------------|------|
| `physics_bridge::GpuPhysicsController` | `physics` | [ALICE-Physics](../ALICE-Physics) | 物理**制御ポリシー**の GPU 三値推論、バッチ力計算 |
| `physics_bridge::TrtSolverAdapter` | `physics-solver` | [ALICE-Physics v0.8.0+](../ALICE-Physics) | Compute-shader ベース Fix128 **PGS ソルバオフロード** — 重力 + 床 + N 個の距離制約 (Gauss-Seidel)、Metal / Vulkan / DX12 でバイト完全一致 |
| `fix128::Fix128GpuKernel` | `fix128-arithmetic` | [ALICE-Physics v0.8.0+](../ALICE-Physics) | Fix128 GPU プリミティブ — `add` / `sub` / `mul` / `dot` (2 段 reduce) + PGS `integrate` / `project_floor` / `project_distance` WGSL カーネル |
| `sdf_bridge` | `sdf` | [ALICE-SDF](../ALICE-SDF) | 三値ネット経由の GPU neural SDF 近似 |

### 物理ソルバブリッジ (feature: `physics-solver`、v1.0.0 以降 stable)

[ALICE-Physics v0.8.0+](../ALICE-Physics) 向け compute-shader ベース Fix128 PGS ソルバオフロード。`TrtSolverAdapter` は `alice_physics::gpu_bridge::GpuSolverBridge` trait を実装するので、ALICE-Physics 側は依存注入で GPU 経路を差し込める。既定の CPU-native TGS パイプラインには一切触れない。

**Semver 安定性コミット (v1.0.0+)**: `TrtSolverAdapter` の公開表面 (コンストラクタ / `send_island` / `dispatch_iterations` / `set_gravity` / `set_floor` / `push_distance_constraint` / `recv_island` / `assert_bit_exact_vs_cpu`) は v1.x で bytes-stable。追加ヘルパは minor bump、破壊的変更は v2 送り。

**`alice-physics/gpu-solver-bridge` とペア。** `physics-solver` feature が ALICE-Physics 側の gpu-solver-bridge を自動有効化する。

```toml
[dependencies]
alice-trt     = { path = "../ALICE-TRT",     features = ["physics-solver"] }
alice-physics = { path = "../ALICE-Physics", features = ["gpu-solver-bridge"] }
```

```rust
use alice_physics::gpu_bridge::{DiffFixture, GpuSolverBridge};
use alice_physics::math::Fix128;
use alice_trt::{GpuDevice, TrtSolverAdapter};

// 1. wgpu device に紐付けたアダプタを構築
let device = GpuDevice::new()?;
let mut adapter = TrtSolverAdapter::new(&device);

// 2. 現在の island 状態をアップロード
adapter.send_island(&positions, &velocities);

// 3. 制約をインストール (全て optional、全て additive)
adapter.set_gravity(Some([Fix128::ZERO, Fix128::from_ratio(-98, 10), Fix128::ZERO]));
adapter.set_floor(Some(Fix128::ZERO));                             // y >= 0
adapter.push_distance_constraint(0, 1, Fix128::from_int(2));       // |p0 - p1| = 2
adapter.push_distance_constraint(1, 2, Fix128::from_int(2));       // Gauss-Seidel 順

// 4. GPU 上で PGS 反復ディスパッチ (live shader、no-op ではない)
adapter.dispatch_iterations(10, Fix128::from_ratio(1, 60));

// 5. 呼び出し側 slice に読み戻し
adapter.recv_island(&mut out_positions, &mut out_velocities);

// 6. CPU 側ソルバとバイト完全一致を検証
adapter.assert_bit_exact_vs_cpu(&DiffFixture {
    description: "3body_triangle_10iter",
    tolerance: Fix128::ZERO,
})?;
```

**機能タイムライン (v0.3.0 → v1.3.1):**

| 機能 | 追加 | 備考 |
|-----|-----|-----|
| Live PGS ディスパッチ (integrate) | v0.4.0 | GPU 上 semi-implicit Euler、CPU と byte-exact |
| 重力 | v0.5.0 | `set_gravity(Option<[Fix128; 3]>)` |
| 床制約 | v0.6.0 | `set_floor(Option<Fix128>)`、`y >= floor` に projection |
| WARP crash 修正 (uniform-flow) | v0.8.1 | FXC X4026 規律、3 プラットフォーム CI green |
| 距離制約 (単一) | v1.2.0 | CPU precompute scalar + GPU project |
| 複数距離制約 (Gauss-Seidel) | v1.3.0 | `push_distance_constraint(a, b, L)` × N |
| Accessor | v1.3.1 | `distance_constraint_count()` / `has_distance_constraints()` |

**CI マトリクス**: 全リリースを macOS (Metal) / Ubuntu (Vulkan lavapipe) / Windows (DX12 WARP) で CPU golden ソルバとの byte-exact assertion で検証。

Companion release: [ALICE-Physics v0.8.0+](https://github.com/ext-sakamoro/ALICE-Physics/releases)。

### Fix128 GPU プリミティブ (feature: `fix128-arithmetic`、v1.0.0 以降 stable)

Standalone `Fix128Gpu` (`#[repr(C)]` + `bytemuck::Pod`、`alice_physics::Fix128` と layout 互換) + `Fix128GpuKernel` trait。`physics-solver` なしで有効化すれば ALICE-Physics を引き込まずに Fix128 GPU プリミティブだけ使える。

```toml
[dependencies]
alice-trt = { path = "../ALICE-TRT", features = ["fix128-arithmetic"] }
```

**Live GPU カーネル (全て WGSL、全て CPU 参照実装と byte-exact):**

| Kernel | 追加 | WGSL 定数 |
|--------|-----|----------|
| `add` / `sub` | v0.3.0 | `FIX128_ADD_WGSL` / `FIX128_SUB_WGSL` |
| `mul` | v0.3.0 | `FIX128_MUL_WGSL` |
| `dot` (2 段 reduce) | v0.7.1 / v0.8.1 | `FIX128_DOT_WGSL` + `FIX128_DOT_FINAL_WGSL` |
| PGS integrate (semi-implicit Euler) | v0.4.0 | `FIX128_PGS_INTEGRATE_WGSL` |
| PGS project (floor) | v0.6.0 | `FIX128_PGS_PROJECT_FLOOR_WGSL` |
| PGS project (distance, rigid rod) | v1.4.2 | `FIX128_PGS_PROJECT_DISTANCE_RIGID_WGSL` |
| PGS project (distance, batched) | v1.5.1 | `FIX128_PGS_PROJECT_DISTANCE_BATCHED_WGSL` |
| Fix128 GPU `div` | v1.4.0 | `FIX128_DIV_WGSL` |
| Fix128 GPU `sqrt` | v1.4.1 | `FIX128_SQRT_WGSL` |

CPU 側 `Fix128Gpu` は `sqrt` (v1.0.1、`alice_physics::Fix128::sqrt` に delegate) と `div` (v1.0.6) も持つ (host precompute 用)。v1.1.0 の単一制約 uniform 版 (`FIX128_PGS_PROJECT_DISTANCE_WGSL`) は v1.7.0 で deprecate、v2.0.0 で削除済。外部 `Fix128GpuKernel` 実装者は上表の rigid rod / batched シェーダーへ移行を。

**37 Fix128 単体テスト + 170 physics-solver テスト** を毎リリース 3 プラットフォーム全通し。

### SDF ブリッジ (feature: `sdf`)

三値ネットで SDF ノードを近似する GPU 加速 neural SDF。複雑な CSG tree を軽量三値 NN に fit させて GPU で高速評価。詳細は英語版参照。

### ALICE-DB / ALICE-View / ALICE-Voice ブリッジ

- **db**: 推論メトリクス永続化 (34 byte binary serialization、model 別レイテンシ集計)
- **view**: neural upscaling (DLSS 風、Performance 4x / Balanced 2.5x / Quality 1.7x / UltraQuality 1.3x)
- **voice**: GPU 音声特徴抽出 (mel / energy / ZCR / spectral centroid)

## ALICE エコシステム

| クレート | 役割 |
|---------|-----|
| [ALICE-ML](https://github.com/ext-sakamoro/ALICE-ML) | CPU 推論エンジン (branchless 三値、AVX2 SIMD) |
| **ALICE-TRT** | GPU 推論エンジン (wgpu + CUDA Tensor Core) |
| [ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics) | Fix128 決定論物理エンジン (`physics-solver` の連携先) |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | SDF プリミティブ (`sdf` bridge の連携先) |
| [ALICE-Eco-System](https://github.com/ext-sakamoro/ALICE-Eco-System) | 統合デモ & ベンチマーク |

## ライセンス

AGPL-3.0. [LICENSE](LICENSE) 参照。

Copyright (c) Moroya Sakamoto
