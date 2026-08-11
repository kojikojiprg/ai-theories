# AI Theories

LLM / VLM の理論を学習し、最終的に自分で AI システムを作れるようになることを目的とした個人プロジェクトです。
A personal project to learn LLM / VLM theory, with the ultimate goal of being able to build AI systems from scratch.

古典的な機械学習理論(SVM、決定木など)は対象外とし、**LLM / VLM に関する理論に限定**しています。
Classical machine learning theory (e.g., SVM, decision trees) is out of scope — this project focuses exclusively on **LLM / VLM theory**.

## 構成 / Structure

- [`theories/`](./theories/) — LLM/VLM 理論の学習ノートブック(基礎 → 応用の順に整理)
  Notebooks for learning LLM/VLM theory, organized from foundational to advanced topics.
- [`apps/`](./apps/) — 学んだ理論を活かしたシステム構築の実践(Web アプリ等)
  Hands-on system building (web apps, etc.) that applies the theory learned above.
- `src/` — `theories/`・`apps/` から利用する共通モジュール
  Shared modules reused by both `theories/` and `apps/`.

詳細な学習順序・トピック一覧は [`theories/README.md`](./theories/README.md)、
アプリ一覧・デプロイ方法は [`apps/README.md`](./apps/README.md) を参照してください。
See [`theories/README.md`](./theories/README.md) for the recommended learning order and topic list,
and [`apps/README.md`](./apps/README.md) for the app list and deployment details.

## 開発環境 / Development Environment

- 依存関係管理 / Dependency management: [uv](https://docs.astral.sh/uv/)
- フォーマッタ / リンタ / Formatter & linter: [ruff](https://docs.astral.sh/ruff/)
- 学習・実験 / Training & experiments: Google Colab 無料枠(T4 GPU など)/ Google Colab free tier (T4 GPU, etc.)
- デモアプリの公開 / Demo hosting: Hugging Face Spaces(Gradio)

## License

MIT
