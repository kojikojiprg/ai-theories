# アプリ / Apps

`theories/`で学んだ理論を活かして構築する、実際に動くシステムです。番号つきの別々のアプリではなく、**1 つの Web アプリ「AI Theories Lab」([`ai_theories_lab/`](./ai_theories_lab/))に、機能を段階的に足していきます。** 公開する場合は、Hugging Face Spaces の 1 つの Space(ZeroGPU 上の Gradio)にします。**現在は未公開で、ローカルで実行しています**(理由は「公開方針」の節)。
This is a single web app, "AI Theories Lab", to which features are added in stages, built by applying the theory learned in `theories/`. It is not published yet and currently runs locally.

**理論に対応する機能は、対応する理論のトピックが`theories/`にできてから実装します。** システムの仕組み(API の層・常駐サーバ・画面・計測・試験など)は、理論トピックなしで実装してかまいません。理論トピックが要るのは、機能が使う理論(アルゴリズム・手法)の部分で、それを載せる入れ物には要らないからです。仕組みが理論の実装(`src/`)を使うときは、使った理論への参照を明記します。

## 機能の一覧 / Features

| 機能 / Feature | 段階 / Stage | 使用した理論 / Based on Topics | 状態 / Status |
|---|---|---|---|
| 文章の生成(デコーディング戦略・KV キャッシュ・Grouped-Query Attention の切り替え、逐次表示、速さの表示、2 つの方法の並べた比較) | 1 | [008 デコーディング戦略](../theories/02_pretraining/008_decoding_strategies.ipynb)、[010 KV キャッシュと推論の計算量](../theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb) | 実装済み・未公開(ローカルで実行) / Implemented, not published (runs locally) |
| 指示への応答(016 の合成課題。見たことのあるテンプレートと未見のテンプレート) | 1 | [016 SFT(指示チューニング)](../theories/04_alignment/016_supervised_fine_tuning.ipynb)、[017 報酬モデルと RLHF](../theories/04_alignment/017_reward_model_and_rlhf.ipynb) | 実装済み・未公開(ローカルで実行) / Implemented, not published (runs locally) |
| モデルの中を見る(トークン分割と BPE の結合の過程・注意の重み・トークンごとの確率とエントロピー) | 1 | [001 注意機構](../theories/01_foundations/001_attention_mechanism.ipynb)、[005 トークナイザと部分語分割](../theories/02_pretraining/005_tokenizer.ipynb)、[008 デコーディング戦略](../theories/02_pretraining/008_decoding_strategies.ipynb) | 実装済み・未公開(ローカルで実行) / Implemented, not published (runs locally) |
| バッチ推論(画面のタブ。同じ合成の負荷を static batching と continuous batching で実行して比べる。KV キャッシュの管理方式・ブロックの大きさ・接頭辞の共有・退避、要求ごとの時間の帯と KV キャッシュの使用量の図、最大バッチサイズを振る計測(027 の観察 D)。機器によらない量(反復の回数・無駄な行の割合・待ちの反復数など)を、実時間の指標(同じ負荷を 3 回実行した中央値と範囲)より先に示し、1 反復の時間の内訳と固定費の割合も示す) | 2 | [027 推論サーバ: continuous batching と PagedAttention](../theories/03_efficient_training/027_inference_serving.ipynb)(実装は`src/serving/`)、[010 KV キャッシュと推論の計算量](../theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb)、[008 デコーディング戦略](../theories/02_pretraining/008_decoding_strategies.ipynb) | 実装済み・未公開(ローカルで実行。1 回の`@spaces.GPU`の関数で完結するので ZeroGPU と互換) / Implemented, not published (ZeroGPU-compatible) |
| 常駐サーバと HTTP の API(手元専用。実行中に届く要求を受け付ける常駐のエンジン、Server-Sent Events による逐次表示、待ち行列の上限・時間の上限・切断による取り消し・停止の手順、負荷を掛ける道具) | 2 | 使った理論: [027](../theories/03_efficient_training/027_inference_serving.ipynb)(`src/serving/`の部品)・[010](../theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb)・[008](../theories/02_pretraining/008_decoding_strategies.ipynb)。API の層そのものは、システムの仕組みなので理論トピックはない | 実装済み・手元専用(`serve.py`。Spaces では動かない) / Implemented, local only (does not run on Spaces) |

「状態」の「実装済み」は、代替のアーティファクト(乱数の重み)を使う試験が通ることを指します。本物の重みを使う試験(`tests/hub/`)の結果は含みません。「未公開」は、Hugging Face Spaces に Space を作成していないことを指し(理由は「公開方針」の節)、手元で`app.py`を起動して使います(起動方法は[アプリの README](./ai_theories_lab/README.md)の 5.1)。ZeroGPU 上の動作は確かめていません。「手元専用」は、常駐サーバ(`serve.py`)を指し、ZeroGPU には載せません(「公開方針」の例外)。

## 段階の計画 / Roadmap

ここでいう段階は、`theories/`のノートブックの「第 1 段階・第 2 段階」(Colab での本番実行の運用)とは別のものです。

| 段階 / Stage | 機能 | 対応する理論のトピック | 着手の条件 |
|---|---|---|---|
| 1 | 生成・KV キャッシュ・GQA・指示への応答・モデルの中を見る | 001・005・008・010・016・017 | 実装済み |
| 2 | **段階 2-1**: バッチ推論と待ち行列の自作、API の層。**段階 2-2**: 量子化、長い文脈 | バッチ推論と待ち行列の自作は 027(continuous batching と PagedAttention。実装は`src/serving/`)。量子化は 013、長い文脈は 015(と 014)。**API の層は、システムの仕組み(常駐サーバ・HTTP の端点・負荷を掛ける道具)なので、理論トピックは要らない** | **段階 2-1: 実装済み**(バッチ推論のタブと、常駐サーバ・HTTP の API。027 の判定は A・B1・B2・B3・C 支持)。最大バッチサイズの既定値は 16、混雑に備えて 32 まで(027 の 7.10 節)。大きなモデルでは成り立たないので、アプリで使うモデルで 027 の観察 D を測り直して決める。**測り直しは、タブの「最大バッチサイズを振って測る」から実行できる。既定値を変えるかは、手元で計測してから判断する(未実施)。** **段階 2-2(量子化・長い文脈): 未着手。** 段階 2-1 の API に載る機能として足す。着手の前に、対応するトピック(013・015)の範囲を確かめる |
| 3 | RAG(Retrieval-Augmented Generation、検索拡張生成) | 024・025・026(`07_retrieval`) | トピックはある。着手の前に、必要な範囲を確かめる |
| 4 | チャットボット(複数ターンの会話、思考の表示) | `theories/`にまだない | **`theories/`にトピックを足してから着手する** |
| 5 | エージェント | `theories/`にまだない | **`theories/`にトピックを足してから着手する** |

## 公開方針 / Deployment Policy

- **学習・実験(モデル学習、GPU 処理)**: Google Colab 無料枠(T4 GPU など)。
- **現在の状態**: **未公開で、ローカルで実行しています。** Hugging Face Spaces で Gradio の Space を作成しようとしましたが、無料のアカウントでは作成画面で Gradio が選べず、作成できませんでした(原因は確かめていません)。公開は、段階 3(RAG)の後に改めて判断します。
- **ZeroGPU 上での公開を想定して作る**: PRO に入るなどして公開できるようになったときに、そのまま使えるよう、次の ZeroGPU 向けの作り(`@spaces.GPU`、プロセス境界を越える値の規則、入力の検査)は残しています。ZeroGPU の外(手元の MPS(Metal Performance Shaders)や CPU)では、`@spaces.GPU`は何もせず、同じ`app.py`がそのまま動きます。
  - GPU を使う処理は`@spaces.GPU`を付けた関数の中で行い、モデルの読み込みはモジュールの読み込み時に行う。
  - 引数と戻り値は Python の値と numpy 配列だけにし、入力の検査は`@spaces.GPU`の関数に入る前に行う(詳細はアプリの README)。
- **デモを公開する場合**: Hugging Face Spaces の **ZeroGPU 上の Gradio** で、**1 つの Space**(名前は`ai-theories-lab`を想定)にします。
  - **無料枠の条件(公式ドキュメントの記載)**: メールアドレスが確認済みで、アカウントの作成から 30 日以上が経っていること。ZeroGPU の Space は 2 つまで(PRO は 10 まで)。上の「現在の状態」のとおり、無料のアカウントでは Gradio の Space を作成できなかったので、この記載どおりに作れるとは限らない。
  - **閲覧者ごとの 1 日の GPU の利用枠**: 未ログイン 2 分・無料アカウント 5 分・PRO 40 分。最初の GPU の利用から 24 時間後にリセットされる。
  - 上の数値は、ZeroGPU の公式ドキュメント(https://huggingface.co/docs/hub/spaces-zerogpu)の 2026 年 10 月時点の記載で、変わりうる。公開を判断するときに、最新の記載を確かめる。
  - ハードウェア(ZeroGPU)は README の YAML では指定できず、Space の設定(Settings → Space hardware)で選ぶ。
- **例外: 常駐サーバは手元専用で、ZeroGPU には載せない。** continuous batching は、常駐するエンジンが複数の要求を同時に抱える仕組みで、要求ごとに別プロセスで GPU を割り当てる ZeroGPU とは合いません。そのため、常駐サーバと HTTP の API(`serve.py`、`lab/serving/`)は手元で`uv run python serve.py`として起動して使い、Spaces は`app.py`だけを起動するので、公開しても常駐サーバは動きません。ZeroGPU に載せるのは、1 回の`@spaces.GPU`の関数の中で完結する機能(バッチ推論のタブなど)だけです。
- **処理と画面の分離**: 機能ごとに、処理(`service.py`)と画面(`ui.py`)を分ける。`service.py`は Gradio を import しない。画面は当面 Gradio だが、段階 2 以降で API を足すときや、画面を差し替えるときに、処理の側をそのまま残せる。
- アプリのディレクトリ(`apps/ai_theories_lab/`)は、公開する場合に備えて、**そのまま Hugging Face Spaces のリポジトリルート** として扱えるようにしている(`app.py` / `pyproject.toml` / `uv.lock` / `requirements.txt` / Spaces 用 README を直下に配置)。
- **アプリは独立した uv プロジェクト**(そのディレクトリの`pyproject.toml` + `uv.lock`)。`ai-theories`の`src/`は、GitHub の **コミットを固定した git 依存** で持ち込む。`requirements.txt`は uv からのエクスポート成果物で、手で編集しない。
- **試験**は pytest。`tests/unit/`は常に走らせ(Hub を使わない)、本物の重みを使う`tests/hub/`は`hub`の印をつけて既定では走らせない。
- 公開する場合、ai-theories 本体と Spaces 側リポジトリは別リポジトリとして管理する。push 手順(公開する場合の手順)と、固定するコミットの更新手順は、アプリの README.md に記載する。

## アプリ README の構成 / App README Structure

アプリのディレクトリの`README.md`(Spaces の README を兼ねる)は以下の構成で統一する。

1. 概要 — 何を作るシステムか。機能の一覧
2. 使用した理論 — `theories/`内の該当ノートブックへのリンクと説明
3. システム構成図
4. 使用技術・構成(バックエンド / フロントエンド / フレームワーク、ディレクトリの構成)
5. デプロイ手順(手元での起動と試験の実行を先に、公開する場合の Hugging Face Spaces へのデプロイ手順を後に書く。Colab での実行方法は、学習・評価を伴う機能がある場合に限る)
