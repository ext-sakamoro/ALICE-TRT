# ALICE-TRT — Claude Code 設定

## プロジェクト概要

GPU Ternary Inference Engine - Don't move data Move the law

| 項目 | 値 |
|------|-----|
| クレート名 | `alice-trt` |
| バージョン | 0.1.1 |
| ライセンス | AGPL-3.0 |
| リポジトリ | `ext-sakamoro/ALICE-TRT` |
| Eco-Systemブリッジ | bridge_trt.rs |

## コーディングルール

メインCLAUDE.md「Git Commit設定」参照。日本語コミット・コメント、署名禁止、作成者 `Moroya Sakamoto`。

## ALICE 品質基準

ALICE-KARIKARI.md「100/100品質基準」参照。clippy基準: `pedantic+nursery`

| 指標 | 値 |
|------|-----|
| clippy (pedantic+nursery) | 0 warnings |
| テスト数 | 117 |
| fmt | clean |

## Eco-System パイプライン

本クレートはALICE-Eco-Systemの以下のパスで使用:
- Path G (AI Inference→TRT→SDF/Physics/View)

## 情報更新ルール

- バージョンアップ時: このCLAUDE.mdのバージョンを更新
- APIの破壊的変更時: ALICE-Eco-Systemブリッジへの影響をメモ
- テスト数/品質の変化時: 品質基準セクションを更新
- 新feature追加時: プロジェクト概要テーブルを更新
