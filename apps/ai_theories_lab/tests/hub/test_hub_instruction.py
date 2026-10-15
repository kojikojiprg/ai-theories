"""本物の ``sft-synthetic`` のモデルが、016 の合成課題に答えられること。"""

from __future__ import annotations

import numpy as np
import pytest

from lab.core.artifacts import Resources
from lab.features.instruction.service import (
    MAX_WORDS,
    MIN_WORDS,
    instruction_choices,
    respond,
)
from src.data.instruction import TASK_NAMES, WORD_VOCABULARY, sample_word_sequences


@pytest.fixture(scope="module")
def exact_match_rates(hub_resources: Resources) -> dict[str, float]:
    """見たことのあるテンプレートと未見のテンプレートの、完全一致率(50 入力 x 4 課題)。"""
    rng = np.random.default_rng(2026)
    inputs = sample_word_sequences(rng, 50, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS)
    sft = hub_resources.models["sft-synthetic"]
    rates = {}
    for kind in ("seen", "unseen"):
        exact = total = 0
        for index, words in enumerate(inputs):
            for task in TASK_NAMES:
                choices = instruction_choices(task, kind)
                instruction = choices[index % len(choices)]
                outcome = respond(
                    sft, hub_resources.tokenizer, hub_resources.device, task, instruction, words
                )
                exact += outcome.exact_match
                total += 1
        rates[kind] = exact / total
    return rates


def test_seen_templates_exact_match_rate_is_high(exact_match_rates: dict[str, float]) -> None:
    """016 の条件 1 の完全一致率の平均は 0.804。"""
    assert exact_match_rates["seen"] >= 0.5


def test_unseen_templates_stay_below_the_baseline(exact_match_rates: dict[str, float]) -> None:
    """016 では 0.022〜0.078 で、基準値 0.25(指示を無視する戦略の上界)を下回った。"""
    assert exact_match_rates["unseen"] < 0.25
