# 学習トピック一覧 / Topics Index

LLM / VLM の理論学習ノートブック一覧です。
This is the index of theory-learning notebooks for LLM / VLM.

新しいトピックを追加する際は、対象カテゴリにノートブックを追加し、必ずこの表も更新してください。

## カテゴリ概要 / Categories

### 01_foundations

Transformer の基本構造。以降すべての理論の前提となる。
Core Transformer architecture — the prerequisite for all other topics.

### 02_pretraining

言語モデルの事前学習・生成・スケーリングに関わる理論(トークナイザ、デコーディング戦略、スケーリング則など)。
Theory related to language model pretraining, generation, and scaling (tokenizers, decoding strategies, scaling laws, etc.).

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

### 07_retrieval

テキスト埋め込みと検索(Retrieval)の理論。`apps/`の RAG アプリの理論的基盤となる。
Theory of text embeddings and retrieval — the theoretical foundation for the RAG app in `apps/`.

> カテゴリは今後の新理論の追加に応じて自由に拡張してよい。
> Categories may be freely extended as new theories emerge.

## 推奨学習順序 / Recommended Order

理論はカテゴリごとに整理していますが、**表の番号順に学習する** ことを推奨します。
The theories are organized by category, but I recommend **studying them in the numbered order shown in the table**.

| #   | トピック / Topic              | カテゴリ / Category   | 前提知識 / Prerequisites | 扱う内容 / Contents                                                                                                                                                                                                        | ノートブック                                                                    |
| --- | ----------------------------- | --------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 001 | 注意機構(Attention Mechanism) | 01_foundations        | なし / None              | Query / Key / Value、Scaled Dot-Product Attention とスケーリング係数 $\sqrt{d_k}$ の導出、因果マスク、多頭注意機構(Multi-Head Attention)。実装は`src/layers/attention.py`にスクラッチ実装し、重みの可視化と copy task の学習で検証する | [001_attention_mechanism.ipynb](./01_foundations/001_attention_mechanism.ipynb) |
| 002 | Transformer Block             | 01_foundations        | 001                      | 残差接続(Residual Connection)・層正規化(Layer Normalization)・順伝播ネットワーク(Feed-Forward Network)をスクラッチ実装し、正規化前置(Pre-Layer Normalization)と正規化後置(Post-Layer Normalization)の違いを扱う。001 で実装した多頭注意機構(Multi-Head Attention)を組み込んだ Encoder Block と、交差注意(cross-attention)を含む Decoder Block の両方を実装する。系列の順序情報を与えるため、正弦波(sinusoidal)方式の位置エンコーディング(Positional Encoding)を暫定的に導入する(各方式の比較は 003) | [002_transformer_block.ipynb](./01_foundations/002_transformer_block.ipynb)     |
| 003 | 位置エンコーディング / RoPE   | 01_foundations        | 002                      | 002 で暫定導入した正弦波(sinusoidal)方式に加え、学習可能な絶対位置埋め込み(Learned Absolute Positional Embedding)・相対位置エンコーディング(Relative Positional Encoding)を比較し、回転位置エンコーディング(RoPE: Rotary Position Embedding)の数学的導出と実装を扱う                                                     | (未作成 / TBD)                                                                  |
| 004 | 正規化と活性化の系譜          | 01_foundations        | 002                      | 002 でスクラッチ実装した層正規化(Layer Normalization)を起点に RMSNorm への変遷、GELU から SwiGLU への活性化関数の変遷を辿り、現代 LLM がこれらの構成を採用する理由を考察する                                                                                              | (未作成 / TBD)                                                                  |
| 005 | トークナイザ                  | 02_pretraining        | なし / None              | BPE(Byte Pair Encoding)の学習アルゴリズムと SentencePiece の仕組みを扱い、語彙サイズ(vocabulary size)と系列長のトレードオフを検討する                                                                                      | (未作成 / TBD)                                                                  |
| 006 | 小型 GPT の事前学習           | 02_pretraining        | 003, 004, 005            | 自己回帰言語モデリング(Autoregressive Language Modeling)の学習ループを実装し、loss 曲線と perplexity で学習の進行を評価する                                                                                                | (未作成 / TBD)                                                                  |
| 007 | 学習の安定化                  | 02_pretraining        | 006                      | 002 で観測した正規化後置の勾配の不均衡を踏まえ、AdamW、warmup + cosine スケジュール、gradient clipping、mixed precision など、大規模言語モデルの学習を安定化させる技術を扱う                                                | (未作成 / TBD)                                                                  |
| 008 | デコーディング戦略            | 02_pretraining        | 006                      | greedy / temperature / top-k / top-p / beam search など複数のデコーディング手法(Decoding Strategies)を実装し、生成品質を比較する                                                                                           | (未作成 / TBD)                                                                  |
| 009 | スケーリング則                | 02_pretraining        | 007                      | Kaplan らおよび Chinchilla のスケーリング則(Scaling Laws)を扱い、計算量最適(compute-optimal)なモデルサイズとデータ量の関係を導く                                                                                           | (未作成 / TBD)                                                                  |
| 010 | KV キャッシュと推論の計算量   | 03_efficient_training | 008                      | KV キャッシュ(KV Cache)のメモリ量、MQA(Multi-Query Attention)/ GQA(Grouped-Query Attention)、prefill と decode フェーズの違いを扱う                                                                                        | (未作成 / TBD)                                                                  |
| 011 | LoRA                          | 03_efficient_training | 006                      | 低ランク分解(Low-Rank Decomposition)による差分学習(LoRA: Low-Rank Adaptation)を rank・alpha のパラメータとともにスクラッチ実装する                                                                                         | (未作成 / TBD)                                                                  |
| 012 | 量子化の基礎                  | 03_efficient_training | 011                      | INT8 / NF4 などの量子化手法と量子化誤差(Quantization Error)を扱い、QLoRA の位置づけを整理する                                                                                                                              | (未作成 / TBD)                                                                  |
| 013 | Flash Attention               | 03_efficient_training | 010                      | タイリング(Tiling)と online softmax によるメモリ帯域律速の解消を扱い、Flash Attention の計算手順を追う                                                                                                                     | (未作成 / TBD)                                                                  |
| 014 | 長文脈拡張                    | 03_efficient_training | 003, 013                 | RoPE のスケーリング手法、NTK-aware スケーリング、YaRN など、長文脈(Long Context)への拡張技術を扱う                                                                                                                         | (未作成 / TBD)                                                                  |
| 015 | SFT(指示チューニング)         | 04_alignment          | 011                      | 指示データ(Instruction Data)の形式と損失マスク(Loss Masking)を扱い、LoRA を用いた SFT(Supervised Fine-Tuning)を実装する                                                                                                    | (未作成 / TBD)                                                                  |
| 016 | 報酬モデルと RLHF             | 04_alignment          | 015                      | 選好データ(Preference Data)と Bradley-Terry モデルによる報酬モデル(Reward Model)の学習、および PPO の枠組みを理論中心に扱う                                                                                                | (未作成 / TBD)                                                                  |
| 017 | DPO                           | 04_alignment          | 016                      | RLHF の閉形式解(Closed-Form Solution)から DPO(Direct Preference Optimization)損失を導出し、スクラッチ実装する                                                                                                              | (未作成 / TBD)                                                                  |
| 018 | ViT と画像パッチ埋め込み      | 05_vision_language    | 002                      | 画像のパッチ分割(Patch Embedding)と [CLS] トークンによる、Transformer の画像への適用(ViT: Vision Transformer)を扱う                                                                                                        | (未作成 / TBD)                                                                  |
| 019 | CLIP と対照学習               | 05_vision_language    | 018                      | InfoNCE 損失による対照学習(Contrastive Learning)を用いて画像 encoder とテキスト encoder を共同学習する CLIP の zero-shot 分類を扱う                                                                                        | (未作成 / TBD)                                                                  |
| 020 | LLaVA 型 Vision-Language 連結 | 05_vision_language    | 015, 019                 | projection 層による視覚特徴の LLM 埋め込み空間への写像と、2 段階学習(事前学習 + 指示チューニング)による Vision-Language モデルの構築を扱う                                                                                 | (未作成 / TBD)                                                                  |
| 021 | Mixture of Experts (MoE)      | 06_architectures      | 002                      | ルーティング(Routing)、top-k gating、負荷分散損失(Load Balancing Loss)など、MoE(Mixture of Experts)アーキテクチャの仕組みを扱う                                                                                            | (未作成 / TBD)                                                                  |
| 022 | State Space Model / Mamba     | 06_architectures      | 002                      | 状態空間モデル(State Space Model)の離散化と選択的 SSM(Selective SSM)の仕組みを扱い、線形時間での系列モデリングを検証する                                                                                                   | (未作成 / TBD)                                                                  |
| 023 | テキスト埋め込みと retriever  | 07_retrieval          | 019                      | 文埋め込み(Sentence Embedding)と対照学習による dense retriever を扱い、dual encoder 構造を実装する                                                                                                                         | (未作成 / TBD)                                                                  |
| 024 | ANN 検索とリランキング        | 07_retrieval          | 023                      | 近似最近傍探索(ANN: Approximate Nearest Neighbor、例: HNSW)と cross-encoder によるリランキング(Reranking)を扱う                                                                                                            | (未作成 / TBD)                                                                  |

> 「未作成 / TBD」の行は今後追加予定のトピックです。追加のたびにこの表を更新します。
> Rows marked "未作成 / TBD" are planned topics; this table is updated whenever a topic is added.

### 実装済みの共通モジュール / Implemented Shared Modules

| モジュール                           | 内容                                                                                                      | 初出トピック |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------ |
| `src/layers/attention.py`            | `scaled_dot_product_attention()`、`MultiHeadAttention`、`create_causal_mask()`、`create_padding_mask()`   | 001          |
| `src/utils/visualization.py`         | `plot_attention_heatmap()`、`plot_multi_head_attention()`、`plot_learning_curves()`                       | 001          |
| `src/layers/normalization.py`        | `LayerNormalization`                                                                                       | 002          |
| `src/layers/feedforward.py`          | `FeedForwardNetwork`                                                                                       | 002          |
| `src/layers/positional_encoding.py`  | `SinusoidalPositionalEncoding`                                                                             | 002          |
| `src/layers/transformer_block.py`    | `EncoderBlock`、`DecoderBlock`(いずれも`norm_first`で正規化前置・正規化後置を切り替え)                    | 002          |

## 各ノートブックの構成 / Notebook Structure

各トピックのノートブックは以下の構成で統一する(詳細は`CLAUDE.md`を参照)。

1. タイトル(日本語 / 英語併記)
2. 概要
3. 参考論文
4. 理論(動機・課題 / 数式・導出 / アルゴリズム)
5. 実装方針
6. 実装
7. 実験(Google Colab 無料枠 GPU)
8. 結果・考察
