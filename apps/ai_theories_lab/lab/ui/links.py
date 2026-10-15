"""理論のノートブックへのリンク。

画面の各機能は、使った理論のノートブックへのリンクを置く。番号と名前は ``theories/README.md`` の
学習順序表と同じ。
"""

from __future__ import annotations

REPOSITORY_URL = "https://github.com/kojikojiprg/ai-theories"
_NOTEBOOKS = f"{REPOSITORY_URL}/blob/main/theories"

# 番号 -> (名前, theories/ からの相対パス)
THEORY_TOPICS: dict[str, tuple[str, str]] = {
    "001": ("注意機構", "01_foundations/001_attention_mechanism.ipynb"),
    "003": ("位置エンコーディングと RoPE", "01_foundations/003_positional_encoding_rope.ipynb"),
    "005": ("トークナイザと部分語分割", "02_pretraining/005_tokenizer.ipynb"),
    "006": ("小型 GPT の事前学習", "02_pretraining/006_pretraining_small_gpt.ipynb"),
    "008": ("デコーディング戦略", "02_pretraining/008_decoding_strategies.ipynb"),
    "010": (
        "KV キャッシュと推論の計算量",
        "03_efficient_training/010_kv_cache_and_inference_compute.ipynb",
    ),
    "016": ("SFT(指示チューニング)", "04_alignment/016_supervised_fine_tuning.ipynb"),
    "017": ("報酬モデルと RLHF", "04_alignment/017_reward_model_and_rlhf.ipynb"),
}


def notebook_url(number: str) -> str:
    """理論の番号から、ノートブックの URL を返す。"""
    return f"{_NOTEBOOKS}/{THEORY_TOPICS[number][1]}"


def notebook_link(number: str) -> str:
    """Markdown のリンク ``[008 デコーディング戦略](URL)`` を返す。"""
    return f"[{number} {THEORY_TOPICS[number][0]}]({notebook_url(number)})"
