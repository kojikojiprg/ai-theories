"""本物の ``main`` のモデルで、008 の退化(繰り返し)が見えること。"""

from __future__ import annotations

import numpy as np

from lab.core.artifacts import Resources
from lab.features.generation.service import METHOD_GREEDY, METHOD_TOP_P, DecodingSettings
from lab.features.generation.ui import DEGENERATION_EXAMPLE_PROMPT
from tests.helpers import generate_ids, repeated_ngram_rate


def test_greedy_repeats_more_than_top_p_on_the_degeneration_example(
    hub_resources: Resources,
) -> None:
    """008 の実験 A の 4-gram 重複率は、貪欲法・ビームサーチで 0.52、top-p などで 0.0002。"""
    prompt = DEGENERATION_EXAMPLE_PROMPT
    greedy = generate_ids(hub_resources, "main", prompt, DecodingSettings(METHOD_GREEDY, 64))
    top_p_rates = [
        repeated_ngram_rate(
            generate_ids(
                hub_resources,
                "main",
                prompt,
                DecodingSettings(METHOD_TOP_P, 64, top_p=0.9, seed=seed),
            )
        )
        for seed in range(5)
    ]
    greedy_rate = repeated_ngram_rate(greedy)
    print(
        f"{prompt!r}: 4-gram 重複率 貪欲法 {greedy_rate:.3f} / "
        f"top-p 平均 {np.mean(top_p_rates):.3f}"
    )
    assert greedy_rate > float(np.mean(top_p_rates))
