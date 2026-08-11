# 学習トピック一覧 / Topics Index

LLM / VLM の理論学習ノートブック一覧です。
This is the index of theory-learning notebooks for LLM / VLM.

理論はカテゴリごとに整理していますが、**表の番号順に学習する**ことを推奨します。
新しいトピックを追加する際は、対象カテゴリにノートブックを追加し、必ずこの表も更新してください。

## 推奨学習順序 / Recommended Order

| # | トピック / Topic | カテゴリ / Category | 前提知識 / Prerequisites | ノートブック |
|---|---|---|---|---|
| 001 | Attention Mechanism(注意機構) | 01_foundations | なし / None | [001_attention_mechanism.ipynb](./01_foundations/001_attention_mechanism.ipynb) |
| 002 | Transformer Block | 01_foundations | 001 | (未作成 / TBD) |
| 003 | RoPE(回転位置エンコーディング) | 01_foundations | 002 | (未作成 / TBD) |
| ... | ... | ... | ... | ... |

> 上記はサンプル行です。実際のトピック追加時に書き換えてください。
> The rows above are placeholders — replace them as topics are actually added.

## カテゴリ概要 / Categories

### 01_foundations
Transformer の基本構造。以降すべての理論の前提となる。
Core Transformer architecture — the prerequisite for all other topics.

### 02_pretraining
言語モデルの事前学習・スケーリングに関わる理論(トークナイザ、スケーリング則など)。
Theory related to language model pretraining and scaling (tokenizers, scaling laws, etc.).

### 03_efficient_training
限られた計算資源(Colab 無料枠)で学習・推論するための効率化技術(LoRA、量子化、Flash Attention など)。
Techniques for training/inference under limited compute (LoRA, quantization, Flash Attention, etc.).

### 04_alignment
モデルを人間の意図に沿わせるための手法(SFT、DPO、RLHF など)。
Methods for aligning models with human intent (SFT, DPO, RLHF, etc.).

### 05_vision_language
画像とテキストを統合的に扱う VLM の理論(対照学習、Vision-Language 融合など)。
Theory for Vision-Language Models — contrastive learning, vision-language fusion, etc.

### 06_architectures
Transformer 以外も含む、モデルアーキテクチャの発展形(MoE、State Space Model など)。
Architectural developments beyond the vanilla Transformer (MoE, State Space Models, etc.).

> カテゴリは今後の新理論の追加に応じて自由に拡張してよい。
> Categories may be freely extended as new theories emerge.

## 各ノートブックの構成 / Notebook Structure

各トピックのノートブックは以下の構成で統一する(詳細は `CLAUDE.md` を参照)。

1. タイトル(日本語 / 英語併記)
2. 概要
3. 参考論文
4. 理論(動機・課題 / 数式・導出 / アルゴリズム)
5. 実装方針
6. 実装
7. 実験(Google Colab 無料枠 GPU)
8. 結果・考察
