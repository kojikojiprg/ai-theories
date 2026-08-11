# CLAUDE.md

このリポジトリで作業する際のガイドライン。

## プロジェクトの目的

最新の LLM / VLM 理論を学習し、システムの作り方を理解し、最終的に自分で AI を作れるようになることを目的とした個人プロジェクト。

- 対象は **LLM / VLM に関する理論に限定**する。古典的な機械学習理論(SVM、決定木、古典的統計学習理論など)は対象外。
- 「理論学習(`theories/`)」と「学んだ理論を活かしたシステム構築の実践(`apps/`)」の両輪で進める。

## リポジトリ構成

```
ai-theories/
├── CLAUDE.md
├── README.md               # プロジェクト全体の概要と theories/・apps/ へのリンクのみ
├── pyproject.toml          # uv / ruff の設定
├── theories/               # 理論学習(Jupyter Notebook)
│   ├── README.md           # 全トピック一覧・推奨学習順序・前提知識・カテゴリ概要
│   ├── 01_foundations/
│   ├── 02_pretraining/
│   ├── 03_efficient_training/
│   ├── 04_alignment/
│   ├── 05_vision_language/
│   └── 06_architectures/
├── apps/                   # システム構築の実践
│   ├── README.md           # アプリ一覧・難易度・デプロイ方法の概要
│   ├── 001_simple_chat_app/
│   └── 002_rag_app/
└── src/                    # theories/・apps/ 双方から再利用する共通モジュール
    ├── layers/
    ├── models/
    ├── training/
    ├── data/
    └── utils/
```

### theories/(理論学習)

- カテゴリごとのディレクトリ(例: `01_foundations`, `02_pretraining`, ...)の中に Jupyter Notebook(`.ipynb`)を配置する。
- ノートブックのファイル名は **`リポジトリ全体を通した3桁連番_スラッグ.ipynb`**(例: `001_attention_mechanism.ipynb`)。番号はカテゴリを跨いだ通し番号で、**学習順序**を表す。
- `theories/README.md` に、全トピックの一覧・推奨学習順序・各トピックの前提知識・カテゴリ概要を**表形式**でまとめる。トピック追加時はこの README も必ず更新する。
- カテゴリは今後の新理論の追加に応じて拡張してよい。**発展トピック用の固定カテゴリは作らず**、随時カテゴリまたは連番を追加する。

### apps/(システム構築の実践)

- 学んだ理論を使って実際に動くシステム(簡単な Web アプリ等)を構築する。
- 基礎的なもの(自作モデルを推論 API 化したチャット UI など)から応用的なもの(RAG、マルチモーダル対話アプリなど)まで、**難易度順の連番ディレクトリ**で段階的に追加する(例: `001_simple_chat_app`, `002_rag_app`)。
- `apps/README.md` に、アプリ一覧・難易度・デプロイ方法の概要をまとめる。アプリ追加時はこの README も必ず更新する。
- 各アプリのディレクトリに `README.md` を配置する(構成は後述)。
- `apps/` は `theories/` とは独立したペースで進めてよいが、**使用した理論トピックへの参照は必ず明記**する。
- 各アプリディレクトリ(例: `apps/001_simple_chat_app/`)は、**そのまま Hugging Face Spaces のリポジトリルートとして扱う**。すなわち、アプリディレクトリ直下に `app.py`・`requirements.txt`・Spaces 用の README(先頭に Spaces 用の YAML メタデータを含む)を配置する。
- デプロイは、`apps/001_simple_chat_app/` ディレクトリの内容を Hugging Face Spaces のリポジトリに push する運用とする(例: Spaces リポジトリを別途 remote として追加し `git subtree push` する、または当該ディレクトリの内容を Spaces 側リポジトリに同期する)。具体的な push 手順は各アプリの `README.md` に記載する。
- `ai-theories` リポジトリ本体と Spaces 側リポジトリは**別リポジトリ**として扱う。`ai-theories` 側は「開発・学習理論との紐付けを管理する場所」、Spaces 側は「公開用の実行環境」という役割分担とする。

### src/(共通モジュール)

- `layers/`, `models/`, `training/`, `data/`, `utils/` を配置し、`theories/`・`apps/` 双方から再利用可能なコードを切り出す。
- **ファイル名に `001` などの番号を含めない**。通常の Python モジュール命名規則(スネークケース)に従う。
- `src/` は import される前提のモジュール。`.py` 単体の直接実行は基本的に想定しない。

### ルート

- ルートの `README.md` はプロジェクト全体の概要と `theories/`・`apps/` へのリンクのみ。詳細は各ディレクトリの README.md に委ねる。

## ノートブックの内容構成(theories/)

1トピック = 1ノートブックに以下をすべて含める(Markdown セル + コードセル)。

1. **タイトル** — 日本語見出し、英語併記
2. **概要** — 2〜3文で理論を要約
3. **参考論文** — 著者, タイトル, 会議/年, URL を明記。本文中で理論に言及する際も対応する論文を明記する
4. **理論**
   - 動機・課題
   - 数式・導出(LaTeX 記法。**記号の定義を明記**する)
   - アルゴリズム(擬似コードレベル)
5. **実装方針** — 何を `src/` に切り出し、何をノートブック内に直接書くかを明記
6. **実装** — `src/` から import する、または直接スクラッチ実装する
7. **実験** — Google Colab 無料枠の GPU(T4 など)で実行可能な規模で構成する
8. **結果・考察** — 学習曲線・出力例などをグラフや可視化で分かりやすく示し、考察を Markdown セルで記述する

## アプリ README の内容構成(apps/)

1. **概要** — 何を作るシステムか
2. **使用した理論** — `theories/` 内の該当ノートブックへのリンクと簡単な説明
3. **システム構成図** — 簡単なテキストまたは図
4. **使用技術・構成** — バックエンド、フロントエンド、フレームワーク等
5. **デプロイ手順** — Colab での実行方法、Hugging Face Spaces へのデプロイ手順

## 言語ルール

- **日本語をメインの説明言語**とし、専門用語や重要なキーワードには英語を併記する(例:「注意機構(Attention Mechanism)」)。
- コード内のコメント・docstring も日本語ベース。必要に応じて英語を併記する。
- `apps/` の README.md も日本語メイン・英語併記。

## 実行環境

- `theories/` のモデル学習・実験は **Google Colab 無料枠(T4 GPU など)で完結する規模**のモデル・データセットを使用する。
- `apps/` のデモ公開は **Hugging Face Spaces** を基本のデプロイ先とし、**Gradio** を標準 UI フレームワークとする。
- 依存関係の管理は **uv が唯一の正**(`pyproject.toml` + `uv.lock`)。`requirements.txt` は uv からの**エクスポート成果物**であり、手で編集しない(詳細は「開発環境・ツール」)。
- `theories/` の学習・推論はノートブック上で行う。`.py` ファイル単体の直接実行は想定しない(`src/` は import される前提)。
- `theories/` 配下のノートブックは、Colab セットアップセルで `%cd ai-theories` した後、**リポジトリルートをカレントディレクトリとして実行される前提**とする。
- `src/` からの import は `from src.layers import ...` のように**リポジトリルートからの絶対パスで統一**する。ノートブックの配置場所(`theories/xx_category/` 配下)に関わらずこの前提が崩れないよう、セットアップセルの `%cd` を必ず先頭に置く。

## コーディング規約

- フレームワークは **PyTorch** を使用する。
- **理論的に本質的な部分は可能な限りスクラッチ実装**する(Attention 計算、正規化、LoRA の低ランク分解、DPO 損失関数など)。
- 理論理解に直結しない部分は既存ライブラリを活用してよい(トークナイザの BPE 学習に `tokenizers`、画像前処理に `torchvision` など)。
- `apps/` では実装効率を優先し、Gradio 等のライブラリを積極的に活用してよい(スクラッチ実装のこだわりは `theories/` 優先)。

## 開発環境・ツール

### 依存関係管理(uv)

- **uv が依存関係管理の唯一の正**。リポジトリ全体の依存関係は `pyproject.toml` + `uv.lock` で管理する。
- ローカルでの操作は uv のコマンドのみを使う(`pip install` を直接実行しない):

```bash
uv add torch transformers   # 依存追加(pyproject.toml と uv.lock を更新)
uv sync                     # ロックファイルから環境を再現
uv run python -c "import torch"   # 仮想環境内で実行
```

- `requirements.txt` は**手で編集しない**。uv からのエクスポート成果物としてのみ生成・更新する。

### Google Colab での実行(theories/)

Colab の kernel は `uv sync` が作る `.venv` を参照しないため、**`uv export` で生成した `requirements.txt` を Colab の system Python に入れる方式に統一する**(`uv sync` 方式は使わない)。

- リポジトリルートに、uv から生成した `requirements.txt` をコミットしておく:

```bash
uv export --format requirements-txt --no-hash --no-emit-project > requirements.txt
```

- 各ノートブックの**冒頭セル**に、以下に相当するセットアップセルを置く:

```python
# 環境セットアップ(Google Colab)
!git clone https://github.com/<user>/ai-theories.git
%cd ai-theories
!pip install uv -q
!uv pip install --system -r requirements.txt
```

- 依存関係を追加したら `uv add` → `uv export` で `requirements.txt` を再生成し、両方をコミットする。

### Hugging Face Spaces へのデプロイ(apps/)

Spaces のビルドは uv を前提としないため、**各アプリディレクトリに `requirements.txt` を配置**してビルドに使わせる。

```bash
uv export --format requirements-txt --no-hash --no-emit-project > apps/001_simple_chat_app/requirements.txt
```

- この `requirements.txt` も uv からのエクスポート成果物。アプリ固有の依存を追加した場合は `uv add` 後に再エクスポートする。
- アプリの依存が全体と大きく異なる場合は、アプリディレクトリを独立した uv プロジェクト(そのディレクトリの `pyproject.toml` + `uv.lock`)にしてから同様にエクスポートしてよい。

### フォーマッタ・リンタ(ruff)

- コードフォーマッタ・リンタには **ruff** を使用する。設定は `pyproject.toml` で管理する。
- `src/` 配下のコードは ruff のフォーマット・リント規則に従う。ノートブック(`theories/`, `apps/`)内のコードセルも可能な範囲で同様のスタイルに合わせる。

```bash
uv run ruff format .
uv run ruff check --fix .
```

## 実装依頼時のルール(Claude Code 向け)

- `theories/` の各ノートブックの「理論」セクションの**数式・アルゴリズムを一次情報として**実装を行う。
- **新しい理論トピックを追加するとき**は、以下を**両方**行う:
  1. `theories/README.md` の学習順序表を更新
  2. 対象カテゴリディレクトリへノートブックを追加(全体通し連番を採番)
- **新しいアプリを追加するとき**は、以下を**両方**行う:
  1. `apps/README.md` のアプリ一覧を更新
  2. 対象ディレクトリ(連番付き)を追加し、`README.md` を配置
- 実装後は、**結果や考察を該当セクションに追記**する(`theories/` はノートブックの「結果・考察」、`apps/` は README)。
