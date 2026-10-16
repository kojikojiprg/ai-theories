"""機能どうしで共通の画面部品(文言と表示用の整形)。"""

from __future__ import annotations

import torch

from lab.core.artifacts import LoadedModel
from lab.ui.links import REPOSITORY_URL
from lab.ui.markdown import items, paragraphs

MODEL_SUMMARY = (
    "このアプリのモデルは、スクラッチ実装した小型 GPT(英語の Wikipedia だけで事前学習)です。"
    "小さなモデルなので、文章はすぐに崩れ、事実の正しさも保証しません。"
)


def format_parameter_count(num_parameters: int) -> str:
    """パラメータ数を ``520 万`` のような日本語の単位で表す。"""
    return f"{num_parameters / 10_000:,.0f} 万"


def intro_markdown(main: LoadedModel) -> str:
    """アプリ全体の説明(画面の先頭)。"""
    return paragraphs(
        "# AI Theories Lab",
        f"[ai-theories]({REPOSITORY_URL})の`theories/`で学んだ理論を、ブラウザで手を動かして"
        "確かめる 1 つの Web アプリです。スクラッチ実装した小型 GPT"
        "(Generative Pre-trained Transformer、英語、約 "
        f"{format_parameter_count(main.num_parameters)}パラメータ)を動かして、機能ごとにタブで試します。機能は、対応する理論のトピックが"
        "できてから、段階的に足していきます。",
        items(
            "研究・教育目的のモデルで、品質の保証はありません。事実の正しさも保証しません。",
            "英語で学習したモデルです。入力は英語にしてください。",
            "ChatGPT のように会話できるモデルではありません。",
        ),
    )


def about_markdown() -> str:
    """画面の末尾の出典。"""
    return paragraphs(
        f"**ソースコードと理論**: [kojikojiprg/ai-theories]({REPOSITORY_URL})。"
        "モデルとトークナイザは Hugging Face Hub の "
        "[ai-theories-small-gpt-en](https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en)"
        "(ブランチ`main`・`gqa`・`sft-synthetic`)と "
        "[ai-theories-tokenizer-en](https://huggingface.co/kojikojiprg/ai-theories-tokenizer-en)"
        " です。",
    )


def describe_models(models: dict[str, LoadedModel], device: torch.device) -> str:
    """読み込んだモデルの一覧(Markdown の表)。3 ブランチを同時に載せたときの合計も示す。"""
    rows = [
        "| ブランチ | Query ヘッド数 | Key / Value ヘッド数 | パラメータ数 | 重みのメモリ |"
        " KV キャッシュ(1 トークンあたり) |",
        "|---|---|---|---|---|---|",
    ]
    total_bytes = 0
    for key, loaded in models.items():
        config = loaded.config
        key_value_heads = config.get("num_key_value_heads") or config["num_heads"]
        total_bytes += loaded.parameter_bytes
        rows.append(
            f"| `{key}` | {config['num_heads']} | {key_value_heads} |"
            f" {loaded.num_parameters:,} | {loaded.parameter_bytes / 2**20:.1f} MiB |"
            f" {loaded.key_value_cache_bytes_per_token:,} バイト |"
        )
    rows.append("")
    rows.append(
        f"3 ブランチを同時に`{device.type}`へ載せています"
        f"(重みの合計 {total_bytes / 2**20:.1f} MiB)。"
    )
    return "\n".join(rows)
