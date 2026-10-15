# アプリ一覧 / Apps Index

`theories/`で学んだ理論を活かして構築する、実際に動くシステム(Web アプリ)の一覧です。
This is the index of working systems (web apps) built by applying the theory learned in `theories/`.

基礎的なもの(自作モデルの推論 API 化など)から応用的なもの(RAG、マルチモーダル対話など)まで、
**難易度順の連番ディレクトリ** で段階的に追加します。

## アプリ一覧 / App List

| # | アプリ名 / App | 難易度 / Level | 使用した理論 / Based on Topics | デプロイ先 / Deploy | ステータス |
|---|---|---|---|---|---|
| 001 | [Text Generation Playground](./001_text_generation_playground/) | 基礎 / Basic | [008](../theories/02_pretraining/008_decoding_strategies.ipynb)・[010](../theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb)・[016](../theories/04_alignment/016_supervised_fine_tuning.ipynb)・[017](../theories/04_alignment/017_reward_model_and_rlhf.ipynb) | Hugging Face Spaces(ZeroGPU) | 実装中 / In progress |
| 002 | RAG App | 応用 / Applied | (未作成 / TBD) | Hugging Face Spaces | 未着手 / Not started |

- **001**: 008 の小型 GPT で、デコーディング戦略(貪欲法・temperature・top-k・top-p・ビームサーチ)と KV キャッシュの有無を切り替えて文章を生成し(逐次表示、速さの表示、2 つの方法の並べた比較)、017 が上げた SFT のモデルで 016 の合成課題の指示に答える(見たことのあるテンプレートと未見のテンプレート)。

## デプロイ方針 / Deployment Policy

- **学習・実験(モデル学習、GPU 処理)**: Google Colab 無料枠(T4 GPU など)
- **デモの公開**: Hugging Face Spaces の **ZeroGPU 上の Gradio**。Hugging Face Spaces では、Gradio・Docker の Space の作成に有料プランが必要になっており、無料の個人アカウントが置けるのは ZeroGPU で動く Gradio の Space(2 つまで)だけであるため。
  - **無料枠の条件**: メールアドレスが確認済みで、アカウントの作成から 30 日以上が経っていること。ZeroGPU の Space は 2 つまで(PRO は 10 まで)。
  - **閲覧者ごとの 1 日の GPU の利用枠**: 未ログイン 2 分・無料アカウント 5 分・PRO 40 分。最初の GPU の利用から 24 時間後にリセットされる。
  - 上の数値は、ZeroGPU の公式ドキュメント(https://huggingface.co/docs/hub/spaces-zerogpu)の 2026 年 10 月時点の記載で、変わりうる。
  - ハードウェア(ZeroGPU)は README の YAML では指定できず、Space の設定(Settings → Space hardware)で選ぶ。
- **ZeroGPU 向けの作り方**: GPU を使う処理は`@spaces.GPU`を付けた関数の中で行い、モデルの読み込みはモジュールの読み込み時に行う。ZeroGPU の外(手元の MPS や CPU)では、`@spaces.GPU`は何もせず、同じ`app.py`がそのまま動く。
- 各アプリディレクトリ(例: `apps/001_text_generation_playground/`)は、**そのまま Hugging Face Spaces のリポジトリルート** として扱う(`app.py` / `pyproject.toml` / `uv.lock` / `requirements.txt` / Spaces 用 README を直下に配置)。
- **各アプリは独立した uv プロジェクト**(そのディレクトリの`pyproject.toml` + `uv.lock`)。`ai-theories`の`src/`は、GitHub の **コミットを固定した git 依存** で持ち込む。`requirements.txt`は uv からのエクスポート成果物で、手で編集しない。
- ai-theories 本体と Spaces 側リポジトリは別リポジトリとして管理する。push 手順と、固定するコミットの更新手順は各アプリの README.md に記載する。

## 各アプリ README の構成 / App README Structure

各アプリディレクトリの`README.md`は以下の構成で統一する。

1. 概要 — 何を作るシステムか
2. 使用した理論 — `theories/`内の該当ノートブックへのリンクと説明
3. システム構成図
4. 使用技術・構成(バックエンド / フロントエンド / フレームワーク)
5. デプロイ手順(Colab での実行方法、Hugging Face Spaces へのデプロイ手順)
