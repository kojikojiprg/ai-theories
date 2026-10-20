"""本物の重みで、常駐のエンジンとバッチ推論が 1 要求ずつの生成と一致すること(小さな負荷)。

単体の試験(``tests/unit/test_serving_engine.py``)と同じ確認を、本物の ``main``・``gqa`` で行う。
許容誤差は、計算する型の丸めの単位から導く(``logits_tolerance``)。027 の本番(T4)の確認は、
同じ導出で、logits の最大の差が許容誤差の 0.0044 倍以下だった。
"""

from __future__ import annotations

import numpy as np
import pytest

from lab.core.artifacts import Resources
from lab.features.serving.service import BatchSettings, run_batching_comparison
from lab.serving.runtime import EngineConfig
from src.serving import logits_tolerance, reference_generate
from tests.serving_helpers import Collector, check_against_reference, drive, greedy, make_engine

PROMPTS = [
    "The history of the city began when",
    "Albert Einstein was born in",
    "The film was released in the spring, and critics",
    "In 1998, the",
]


@pytest.mark.parametrize("model_key", ["main", "gqa"])
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(EngineConfig(max_batch_size=3, block_size=16), id="continuous-paged"),
        pytest.param(EngineConfig(max_batch_size=3, block_size=4, num_blocks=11), id="eviction"),
        pytest.param(EngineConfig(scheduler="static", max_batch_size=3), id="static"),
    ],
)
def test_the_resident_engine_matches_one_request_at_a_time(
    hub_resources: Resources, model_key: str, config: EngineConfig
) -> None:
    engine = make_engine(hub_resources, config, model_key=model_key, record_logits=True)
    prompts = [hub_resources.tokenizer.encode(text) for text in PROMPTS]
    lengths = [20, 9, 14, 6]
    sinks = [Collector() for _ in prompts]
    for prompt, length, sink in zip(prompts, lengths, sinks, strict=True):
        engine.submit(prompt, greedy(length), sink)
    drive(engine)
    if config.num_blocks is not None:
        assert engine.stats()["preemptions"] > 0, "退避の経路が通っていない"
    for request_id, (prompt, sink) in enumerate(zip(prompts, sinks, strict=True)):
        check_against_reference(
            hub_resources.models[model_key], prompt, sink.tokens, engine.recorded_logits[request_id]
        )
    # 許容誤差は固定の値ではなく、丸めの単位から導いた値(本物の重みでの余裕を印字する)
    loaded = hub_resources.models[model_key]
    _, reference = reference_generate(loaded.model, prompts[0], lengths[0])
    difference = float(np.abs(np.stack(engine.recorded_logits[0]) - reference).max())
    tolerance = logits_tolerance(loaded.model, reference)
    print(f"{model_key} {config}: 最大の差 {difference:.2e} / 許容誤差 {tolerance:.2e}")
    assert difference <= tolerance


def test_the_batch_inference_tab_runs_with_the_real_weights(hub_resources: Resources) -> None:
    settings = BatchSettings(num_requests=12, output_mean=10, max_batch_size=4, kv_budget_slots=4)
    outcome = run_batching_comparison(
        hub_resources.models["main"], hub_resources.tokenizer, hub_resources.device, settings
    )
    assert outcome.static.active_requests.sum() == outcome.continuous.active_requests.sum()
    assert outcome.continuous.iterations <= outcome.static.iterations
    print(
        f"static {outcome.static.throughput:.0f} / continuous {outcome.continuous.throughput:.0f} "
        f"トークン/秒({outcome.device})"
    )
