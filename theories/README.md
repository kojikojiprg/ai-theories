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
| 003 | 位置エンコーディング / RoPE   | 01_foundations        | 002                      | 002 で暫定導入した正弦波(sinusoidal)方式に加え、学習可能な絶対位置埋め込み(Learned Absolute Positional Embedding)・相対位置エンコーディング(Shaw et al. 方式・T5 の相対位置バイアス)・ALiBi(Attention with Linear Biases)を比較し、回転位置エンコーディング(RoPE: Rotary Position Embedding)の数学的導出と実装を扱う。実装は`src/layers/positional_encoding.py`にスクラッチ実装し、`MultiHeadAttention`に注入する形で組み込む。可変長 copy task による 7 条件の比較(学習長内の精度と学習長を超える外挿性能)および Attention 重みの定量分析で検証する | [003_positional_encoding_rope.ipynb](./01_foundations/003_positional_encoding_rope.ipynb) |
| 004 | 正規化と活性化の系譜          | 01_foundations        | 002                      | 002 でスクラッチ実装した層正規化(Layer Normalization)を起点に RMSNorm(Root Mean Square Normalization)への変遷、ReLU から GELU、GLU(Gated Linear Unit)を経て SwiGLU に至る活性化関数の変遷を扱う。平均減算の有無・分散除算の有無による除去実験、常に負のユニット(always-negative unit)の測定、**乗法的相互作用の合成タスクとそれを含まない陰性対照(negative control)タスクの比較** によって各機構の寄与を検証する。学習には条件比較のための最小限の文字レベル言語モデリングを用い、条件間比較の評価ノイズを抑えるため固定した評価用バッチ集合を使う(本格的な自己回帰言語モデリングの事前学習は 006 で扱う) | [004_normalization_and_activation.ipynb](./01_foundations/004_normalization_and_activation.ipynb) |
| 005 | トークナイザと部分語分割      | 02_pretraining        | 001                       | BPE(Byte Pair Encoding)の学習・符号化(タイブレーク規則を固定、空白をチャンク先頭に保持する可逆な事前分割)とバイトレベル BPE(byte-level BPE)をスクラッチ実装し、WordPiece のスコア関数を理論として位置づける。Unigram 言語モデル(Unigram Language Model)の Viterbi 最尤分割をスクラッチ実装し(語彙学習は sentencepiece に委譲)、SentencePiece が「アルゴリズムではなく実装である」ことを明確にする。英語・日本語(日本語版 Wikipedia、CC BY-SA 4.0、記事とリビジョン ID を固定して取得)・コードの 3 ドメインで語彙サイズ別の fertility を比較し、空白による事前分割(pre-tokenization)が日本語で機能しないことを英語・コードを陰性対照(negative control)として検証する。初期語彙方式(バイトレベル・文字レベル)と語彙サイズの相互作用を語彙サイズの掃引で検証し、語彙サイズと系列長・計算量のトレードオフを理論計算で示す | [005_tokenizer.ipynb](./02_pretraining/005_tokenizer.ipynb)                    |
| 006 | 小型 GPT の事前学習           | 02_pretraining        | 003, 004, 005            | 001〜005 の部品(Transformer Block、RoPE、RMSNorm、SwiGLU、トークナイザ)を統合し、decoder-only な自己回帰言語モデル(`GPTLanguageModel`)の事前学習を初めて最後まで実行する。重み共有(weight tying)、perplexity と bits-per-byte の使い分け(トークナイザ間比較には bits-per-byte、言語間の絶対値比較はしない)を理論として扱う。日本語版・英語版 Wikipedia コーパス(段階 1 で取得、各 20 MB 以上)を用い、文字レベル・バイトレベル BPE(公比 2 の等比数列で 4 語彙サイズ)・Unigram 言語モデル(byte fallback により未知語による情報損失を排除、BPE の 1 条件と語彙サイズを厳密に一致させる)の計 5 トークナイザ条件を比較する(実験 A〜H)。ノイズ床を日英それぞれ 5 シードで測定し(標本標準偏差の過小バイアスを抑えるため)、対比量の誤差伝播に基づく判定閾値(実験 D・E・F)で支持 / 反証 / 判定不能の 3 値判定を行う。非埋め込みパラメータ数を条件間で揃えることで語彙サイズの影響を分離し、実験 G は model サイズを固定してステップ数のみを増やす。学習ステップ数は全言語・全トークナイザ条件の訓練トークン数を実測し、エポック上限(実験 C〜F は 1/2 エポック、実験 G は 1 エポック)から計算式で決定する。**本番実行の結果、部分語分割(BPE・Unigram)が文字レベルより優れるかは言語に依存することが判明した**(日本語は文字レベルが最良、英語は文字レベルが最悪で bits-per-byte の順序が言語間で反転する) | [006_pretraining_small_gpt.ipynb](./02_pretraining/006_pretraining_small_gpt.ipynb) |
| 007 | 学習の安定化                  | 02_pretraining        | 006                      | 002 で観測した正規化後置(Post-Layer Normalization)の勾配の不均衡、および 004 で観測した正規化を欠いた条件のシード間のばらつきの増大を踏まえ、正規化前置(Pre-Layer Normalization)/ 正規化後置(Post-Layer Normalization)と学習率を独立変数として不安定性を意図的に誘発し、AdamW(Decoupled Weight Decay Regularization)・warmup + cosine スケジュール・gradient clipping(勾配クリッピング)が学習を安定化させる効果を検証する。較正を本番と同じステップ数で行い、学習率の水準・gradient clipping の閾値(勾配ノルムの分位点方式)・シード数を決定する(006 のスケーリング外挿の手法を踏襲)。学習が実際に進んでいること・gradient clipping が実際に発動していることを、検証したい仮説とは独立な前提条件(pre-condition)として本番実行の前に宣言し、前提条件が不成立の主張は支持 / 反証と判定せず「前提不成立」として記録する。勾配ノルムのピーク / 平均比率・最大単一ステップ損失上昇幅・固定ステップ数終了時点の損失値を対比量として、対比量の標準偏差に基づく判定閾値で支持 / 反証 / 判定不能 / 前提不成立の 4 値判定を行う。AdamW については、二次モーメント推定によるスケーリングから重み減衰を独立させる効果を、勾配の二次モーメントが異なるパラメータ群間の実効的な減衰強度の乖離として直接検証する合成タスクの実験を別途設ける(混合精度学習は 03_efficient_training の該当トピックに切り出す)。**本番実行の結果、学習率の上昇は正規化方式によらず不安定性を増す一方、正規化後置と正規化前置自体の不安定性の差は判定不能だった。対比量を安定化技術の直接の作用点(勾配ノルムの分布統計)に近づけ、かつ極値統計を避けることで warmup + cosine の効果は支持されたが、gradient clipping は同様の工夫をしても判定不能に留まった** | [007_training_stabilization.ipynb](./02_pretraining/007_training_stabilization.ipynb) |
| 008 | デコーディング戦略            | 02_pretraining        | 006, 007                 | 006・007 の部品(小型 GPT・AdamW・warmup + cosine・gradient clipping)で英語の標準モデルを新たに学習し、top-k サンプリング・top-p サンプリング(nucleus sampling)・長さペナルティ付きビームサーチ(GNMT スタイル)をスクラッチ実装する。貪欲法・ビームサーチと temperature・top-p サンプリングの n-gram 重複率を比較して退化現象(Degeneration)を検証し、top-p の候補集合サイズが分布のエントロピーに応じて動的に変化する(top-k は常に固定サイズ)適応性の違いを相関係数で検証する。temperature による多様性(distinct-n)と一貫性のトレードオフ、beam size と生成確率・多様性のトレードオフは定性的な観察として扱う。学習済みモデルは Hugging Face Hub(`kojikojiprg/ai-theories-small-gpt-en`)に公開し、後続トピックの標準モデルとする | [008_decoding_strategies.ipynb](./02_pretraining/008_decoding_strategies.ipynb) |
| 009 | スケーリング則                | 02_pretraining        | 007                      | Kaplan らおよび Chinchilla のスケーリング則(Scaling Laws)を扱い、計算量最適(compute-optimal)なモデルサイズとデータ量の関係を導く                                                                                           | (未作成 / TBD)                                                                  |
| 010 | KV キャッシュと推論の計算量   | 03_efficient_training | 003, 008                 | KV キャッシュ(KV Cache)のメモリ量、MQA(Multi-Query Attention)/ GQA(Grouped-Query Attention)、prefill と decode フェーズの違いを扱う。逐次生成時の位置インデックス(003 で導入した`positions`引数)の扱いも扱う                                                                                        | (未作成 / TBD)                                                                  |
| 011 | 混合精度学習(Mixed Precision Training) | 03_efficient_training | 006                      | FP16 / BF16 による数値精度の低減と、勾配の underflow を防ぐための loss scaling を扱う。勾配のスケールが小さい層で発生するアンダーフローの現象と、gradient scaler による対処を検証する | (未作成 / TBD)                                                                  |
| 012 | LoRA                          | 03_efficient_training | 006                      | 低ランク分解(Low-Rank Decomposition)による差分学習(LoRA: Low-Rank Adaptation)を rank・alpha のパラメータとともにスクラッチ実装する                                                                                         | (未作成 / TBD)                                                                  |
| 013 | 量子化の基礎                  | 03_efficient_training | 012                      | INT8 / NF4 などの量子化手法と量子化誤差(Quantization Error)を扱い、QLoRA の位置づけを整理する                                                                                                                              | (未作成 / TBD)                                                                  |
| 014 | Flash Attention               | 03_efficient_training | 010                      | タイリング(Tiling)と online softmax によるメモリ帯域律速の解消を扱い、Flash Attention の計算手順を追う                                                                                                                     | (未作成 / TBD)                                                                  |
| 015 | 長文脈拡張                    | 03_efficient_training | 003, 014                 | RoPE のスケーリング手法(位置補間、NTK-aware スケーリング、YaRN など)による長文脈(Long Context)への拡張技術を扱う                                                                                                          | (未作成 / TBD)                                                                  |
| 016 | SFT(指示チューニング)         | 04_alignment          | 012                      | 指示データ(Instruction Data)の形式と損失マスク(Loss Masking)を扱い、LoRA を用いた SFT(Supervised Fine-Tuning)を実装する                                                                                                    | (未作成 / TBD)                                                                  |
| 017 | 報酬モデルと RLHF             | 04_alignment          | 016                      | 選好データ(Preference Data)と Bradley-Terry モデルによる報酬モデル(Reward Model)の学習、および PPO の枠組みを理論中心に扱う                                                                                                | (未作成 / TBD)                                                                  |
| 018 | DPO                           | 04_alignment          | 017                      | RLHF の閉形式解(Closed-Form Solution)から DPO(Direct Preference Optimization)損失を導出し、スクラッチ実装する                                                                                                              | (未作成 / TBD)                                                                  |
| 019 | ViT と画像パッチ埋め込み      | 05_vision_language    | 002                      | 画像のパッチ分割(Patch Embedding)と [CLS] トークンによる、Transformer の画像への適用(ViT: Vision Transformer)を扱う                                                                                                        | (未作成 / TBD)                                                                  |
| 020 | CLIP と対照学習               | 05_vision_language    | 019                      | InfoNCE 損失による対照学習(Contrastive Learning)を用いて画像 encoder とテキスト encoder を共同学習する CLIP の zero-shot 分類を扱う                                                                                        | (未作成 / TBD)                                                                  |
| 021 | LLaVA 型 Vision-Language 連結 | 05_vision_language    | 016, 020                 | projection 層による視覚特徴の LLM 埋め込み空間への写像と、2 段階学習(事前学習 + 指示チューニング)による Vision-Language モデルの構築を扱う                                                                                 | (未作成 / TBD)                                                                  |
| 022 | Mixture of Experts (MoE)      | 06_architectures      | 002                      | ルーティング(Routing)、top-k gating、負荷分散損失(Load Balancing Loss)など、MoE(Mixture of Experts)アーキテクチャの仕組みを扱う                                                                                            | (未作成 / TBD)                                                                  |
| 023 | State Space Model / Mamba     | 06_architectures      | 002                      | 状態空間モデル(State Space Model)の離散化と選択的 SSM(Selective SSM)の仕組みを扱い、線形時間での系列モデリングを検証する                                                                                                   | (未作成 / TBD)                                                                  |
| 024 | テキスト埋め込みと retriever  | 07_retrieval          | 020                      | 文埋め込み(Sentence Embedding)と対照学習による dense retriever を扱い、dual encoder 構造を実装する                                                                                                                         | (未作成 / TBD)                                                                  |
| 025 | ANN 検索とリランキング        | 07_retrieval          | 024                      | 近似最近傍探索(ANN: Approximate Nearest Neighbor、例: HNSW)と cross-encoder によるリランキング(Reranking)を扱う                                                                                                            | (未作成 / TBD)                                                                  |

> 「未作成 / TBD」の行は今後追加予定のトピックです。追加のたびにこの表を更新します。
> Rows marked "未作成 / TBD" are planned topics; this table is updated whenever a topic is added.

### 実装済みの共通モジュール / Implemented Shared Modules

| モジュール                           | 内容                                                                                                      | 初出トピック |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------ |
| `src/layers/attention.py`            | `scaled_dot_product_attention()`(003 で`bias`引数に対応)、`MultiHeadAttention`(003 で位置エンコーディングの注入(`positional_transform`・`attention_score_bias`・`positions`引数)に対応)、`create_causal_mask()`、`create_padding_mask()`   | 001, 003     |
| `src/utils/visualization.py`         | `plot_attention_heatmap()`、`plot_multi_head_attention()`、`plot_learning_curves()`(001)。004 で`plot_seed_scatter()`(複数シードの散布図)・`plot_bar_by_layer()`(層ごとの棒グラフ)・`plot_function_curves()`(関数曲線の比較)・`plot_learning_curves_multi_seed()`(条件ごとに複数シードの学習曲線を重ね描き)を追加。005 で`plot_grouped_bar()`(カテゴリ×系列のグループ棒グラフ)・`plot_dual_axis_curves()`(左右で尺度の異なる 2 系列を共通 x 軸に重ね描き)を追加。006 で`plot_grouped_bar()`に`noise_band`引数(基準条件の値を中心にノイズ床の帯を描画)を追加。007 で`plot_gradient_norm_trace()`(勾配ノルムの時系列に gradient clipping 閾値を水平線で重畳表示)を追加 | 001, 004, 005, 006, 007 |
| `src/layers/normalization.py`        | `LayerNormalization`(002)。004 で`center`・`scale`引数(平均減算・分散除算を個別に無効化)を追加し、`RMSNorm`を新規追加 | 002, 004     |
| `src/layers/feedforward.py`          | `FeedForwardNetwork`(002)。004 で活性化関数の指定方法を`activation_fn`(Callable)引数 1 つに一本化し(既定値`None`で ReLU、文字列引数は廃止)、`SwiGLUFeedForwardNetwork`を新規追加 | 002, 004     |
| `src/layers/positional_encoding.py`  | `SinusoidalPositionalEncoding`(002)。003 で`LearnedAbsolutePositionalEmbedding`・`QueryKeyPositionalTransform`・`RotaryPositionEmbedding`・`AttentionScoreBias`・`ShawRelativePositionBias`(`bias()`ではなく`relative_vectors()`を提供する特別な実装)・`T5RelativePositionBias`・`ALiBiPositionBias`を追加 | 002, 003     |
| `src/layers/transformer_block.py`    | `EncoderBlock`、`DecoderBlock`(いずれも`norm_first`で正規化前置・正規化後置を切り替え、002)。004 で`normalization_factory`・`feed_forward_factory`・`activation_fn`引数(正規化層・順伝播ネットワークの実装の注入)を追加。006 で`DecoderBlock`に`use_cross_attention`引数(False で交差注意の副層を生成しない decoder-only 構成)・`positions`引数(自己注意への位置インデックスの透過)を追加(002〜004 との後方互換性を検証済み) | 002, 004, 006 |
| `src/layers/activation.py`           | `gelu_exact()`、`gelu_tanh_approximation()`、`swish()`                                                    | 004          |
| `src/models/gpt.py`                  | `GPTLanguageModel`(decoder-only な自己回帰言語モデル。`DecoderBlock(use_cross_attention=False)`を積層し、`positional_transform`・`normalization_factory`・`feed_forward_factory`で RoPE・RMSNorm・SwiGLU を注入。`tie_embeddings`で埋め込み行列と出力層の重み共有を切り替え、`generate()`で貪欲法・temperature sampling による生成に対応)。007 で`norm_first`引数(正規化前置・正規化後置の切り替えを`DecoderBlock`へ透過、既定値`True`で 006 と同一の挙動)を追加、正規化後置でも非埋め込みパラメータ数が正規化前置と完全一致することを検証済み | 006, 007     |
| `src/training/trainer.py`            | `train_language_model()`(Adam・固定学習率・fp32 の学習ループ。ステップごとの訓練損失・勾配ノルム、`eval_interval`ごとの検証 bits-per-byte を記録)、`evaluate_bits_per_byte()`(非重複窓での逐次評価、ランダムバッチによる評価はしない。パディングマスクで最終窓の余剰部分を損失計算から除外する)。007 で`optimizer`(AdamW 等のスクラッチ実装または`torch.optim.Optimizer`を交換可能に注入)・`learning_rate_schedule`(ステップごとの学習率を返す callable)・`gradient_clip_threshold`(グローバルノルムでの gradient clipping)引数を追加し、history に`loss_step_delta`(直前ステップとの損失差)・`learning_rate`(実際に使われた学習率)・`gradient_clip_triggered`(ステップごとに clipping が実際に発動したかの bool、前提条件 P2 の検証に使う)を追加。3 引数を渡さない場合は 006 と完全に同一の挙動になることを検証済み(後方互換性) | 006, 007     |
| `src/training/optimizer.py`          | `AdamW`(Decoupled Weight Decay Regularization のスクラッチ実装、`torch.optim.AdamW`と数値的に一致することを検証済み)、`AdamWithL2Regularization`(重み減衰を勾配に混ぜる比較対象の実装)。いずれも`step()`・`zero_grad()`・`set_learning_rate()`を持ち、`train_language_model()`の`optimizer`引数に交換可能に渡せる | 007          |
| `src/training/schedule.py`           | `compute_warmup_cosine_learning_rate()`(線形 warmup + cosine decay による学習率スケジュール) | 007          |
| `src/generation/decoding.py`         | `top_k_filter()`(上位 k 個以外の logits を `-inf` にする)、`top_p_filter()`(Holtzman et al. 2020 の定義に厳密に従う nucleus sampling のフィルタリング)、`beam_search()`(Wu et al. 2016 スタイルの長さペナルティ付きビームサーチ。`beam_size=1` で `GPTLanguageModel.generate(temperature=0.0)` の貪欲法出力と完全一致することを検証済み) | 008          |
| `src/data/text.py`                   | `CharacterLevelTokenizer`、`load_tiny_shakespeare()`、`split_train_val()`、`get_random_batch()`(004)。005 で`load_japanese_corpus()`(日本語版 Wikipedia の記事本文を取得)・`load_code_corpus()`(本リポジトリ自身の`src/`を連結)を追加。006 で`load_wikipedia_corpus()`(言語・記事マニフェストを一般化、`load_japanese_corpus()`はこの薄いラッパーに変更、既存の返り値は不変)・`split_train_val_text()`(文字列の段階での訓練・検証分割)・`encode_corpus()`・`make_evaluation_windows()`(非重複の評価窓を作成。末尾の不完全な窓は切り捨てずパディングし、パディング位置を示すマスクを返す。分母の UTF-8 バイト数はトークナイザに依存せず検証テキスト全体から呼び出し側が直接計算するため、この関数自体はもう `tokenizer` を引数に取らない)を追加 | 004, 005, 006 |
| `src/data/tokenizer.py`              | `learn_bpe()`・`BPETokenizer`(BPE の学習・符号化・`decode()`、`byte_level`によるバイトレベル BPE 切り替え)、`pretokenize()`(事前分割、`chunk_split_mode`。`whitespace` は空白をチャンク先頭に保持し可逆)、`viterbi_segment()`(Unigram 言語モデルの Viterbi 最尤分割)、`UnigramTokenizer`・`train_unigram_model()`(sentencepiece への語彙学習の委譲)、`try_decode_byte_level_symbol()`(バイトレベル語彙の 1 シンボルが単一文字を表すかを判定)。006 で`BPETokenizer.decode()`を`errors="replace"`に変更(系列の途中で切り出した部分列がマルチバイト文字の途中で切れても例外を送出しないようにする、005 のテキスト全体の可逆性検証には影響しない)。`train_unigram_model()`に`byte_fallback`・`character_coverage`引数を追加(既定値は sentencepiece 自身の既定と同一、005 の呼び出し結果は不変。006 では未知語による情報損失を避けるため`byte_fallback=True`・`character_coverage=1.0`を明示的に指定する)。006 で`pretokenize()`・`learn_bpe()`・`BPETokenizer`に`max_chunk_bytes`引数を追加(既定値`None`は 005 の挙動を一切変えない。指定時は空白による事前分割後になお`max_chunk_bytes`バイトを超えるチャンクを UTF-8 の文字境界を壊さない位置でさらに分割し、日本語で空白による事前分割が機能しないこと(005)に起因する BPE の学習・符号化の計算量の急増を抑える)。あわせて`BPETokenizer.encode()`にチャンク単位のメモ化(`_chunk_cache`、上限件数に達すると以降の新規チャンクはキャッシュされないが正しさには影響しない)を追加 | 005, 006     |
| `src/utils/statistics.py`            | `compute_mean_to_rms_ratio()`、`compute_always_negative_unit_ratio()`、`compute_gradient_norm_by_unit_group()`、`compute_gradient_norm_per_layer()`(004)。005 で`compute_fertility()`・`compute_unknown_rate()`・`compute_chunk_length_statistics()`(先頭の連続空白を除いた統計も返す)・`compute_exact_match_rate()`・`compute_segmentation_agreement_rate()`(トークン境界集合の Jaccard 係数)・`compute_character_coverage()`(語彙による文字被覆率)を追加。006 で`compute_bits_per_byte()`・`compute_perplexity()`・`count_non_embedding_parameters()`(語彙サイズが条件間で異なる場合の公平な比較用)を追加。007 で`compute_gradient_norm_peak_to_mean_ratio()`(勾配ノルムのピーク / 平均比率)・`compute_max_single_step_loss_increase()`(最大単一ステップ損失上昇幅)・`compute_effective_decay_divergence()`(パラメータ群間の実効的な重み減衰強度の乖離)・`compute_loss_step_delta_std()`(単一ステップ損失差分の標準偏差)を追加。008 で`compute_ngram_repetition_rate()`(生成系列内の n-gram 重複率、Welleck et al. 2020 の seq-rep-n)・`compute_distinct_n()`(distinct-n、Li et al. 2016 の多様性指標)を追加 | 004, 005, 006, 007, 008 |

## 各ノートブックの構成 / Notebook Structure

各トピックのノートブックは以下の構成で統一する。

1. タイトル(日本語 / 英語併記)
2. 概要
3. 参考論文
4. 理論(動機・課題 / 数式・導出 / アルゴリズム)
5. 実装方針
6. 実装
7. 実験(Google Colab 無料枠 GPU)
8. 結果・考察
