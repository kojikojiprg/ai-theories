# アプリ一覧 / Apps Index

`theories/`で学んだ理論を活かして構築する、実際に動くシステム(Web アプリ)の一覧です。
This is the index of working systems (web apps) built by applying the theory learned in `theories/`.

基礎的なもの(自作モデルの推論 API 化など)から応用的なもの(RAG、マルチモーダル対話など)まで、
**難易度順の連番ディレクトリ** で段階的に追加します。

## アプリ一覧 / App List

| # | アプリ名 / App | 難易度 / Level | 使用した理論 / Based on Topics | デプロイ先 / Deploy | ステータス |
|---|---|---|---|---|---|
| 001 | Simple Chat App | 基礎 / Basic | (未作成 / TBD) | Hugging Face Spaces | 未着手 / Not started |
| 002 | RAG App | 応用 / Applied | (未作成 / TBD) | Hugging Face Spaces | 未着手 / Not started |

> 上記はサンプル行です。実際のアプリ追加時に書き換えてください。
> The rows above are placeholders — replace them as apps are actually added.

## デプロイ方針 / Deployment Policy

- **学習・実験(モデル学習、GPU 処理)**: Google Colab 無料枠(T4 GPU など)
- **デモの公開**: Hugging Face Spaces(標準 UI フレームワークは Gradio)
- 各アプリディレクトリ(例: `apps/001_simple_chat_app/`)は、**そのまま Hugging Face Spaces のリポジトリルート** として扱う(`app.py` / `requirements.txt` / Spaces 用 README を直下に配置)。
- ai-theories 本体と Spaces 側リポジトリは別リポジトリとして管理する。push 手順は各アプリの README.md に記載する。

## 各アプリ README の構成 / App README Structure

各アプリディレクトリの`README.md`は以下の構成で統一する。

1. 概要 — 何を作るシステムか
2. 使用した理論 — `theories/`内の該当ノートブックへのリンクと説明
3. システム構成図
4. 使用技術・構成(バックエンド / フロントエンド / フレームワーク)
5. デプロイ手順(Colab での実行方法、Hugging Face Spaces へのデプロイ手順)
