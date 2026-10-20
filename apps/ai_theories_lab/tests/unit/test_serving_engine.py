"""常駐のエンジン(``lab/serving/engine.py``)の試験: 正しさ・``ServingEngine`` との一致・
サンプリング・取り消し・時間の上限・待ち行列の上限・メモリの整合。

重みが乱数の代替アーティファクトで走るので、生成の中身に意味はない。確かめるのは、エンジンを通した
生成が 1 要求ずつの生成(010 の KV キャッシュ)と一致すること(許容誤差は ``logits_tolerance`` で、
計算する型の丸めの単位から導く)と、``src/serving/`` の実行ループ(``ServingEngine.run()``)と
同じ反復の列になることである。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from lab.core.artifacts import Resources
from lab.features.generation.service import DecodingSettings, select_next_token
from lab.serving.engine import (
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_TIMEOUT,
    EngineClosed,
    QueueFull,
    RequestTooLarge,
)
from lab.serving.runtime import EngineConfig, build_scheduler, build_store
from src.serving import ModelRunner, Request, ServingEngine, ServingModel, WallClock
from tests.serving_helpers import (
    Collector,
    check_against_reference,
    drive,
    greedy,
    make_engine,
    random_prompt,
)


@dataclass(frozen=True)
class Spec:
    """1 件の要求の条件(書き出しの長さ、出力長、何反復目に届くか)。"""

    prompt_length: int
    output_length: int
    arrives_at: int = 0


def make_specs(count: int, seed: int, arrival_gap: int = 0) -> list[Spec]:
    rng = np.random.default_rng(seed)
    return [
        Spec(int(rng.integers(3, 20)), int(rng.integers(1, 14)), i * arrival_gap)
        for i in range(count)
    ]


def run_specs(engine, specs, prompts, settings=None) -> list[Collector]:
    """届く反復に合わせて ``submit()`` しながら、エンジンを進める(届く反復は 0 から数える)。"""
    sinks = [Collector() for _ in specs]
    order = sorted(range(len(specs)), key=lambda i: (specs[i].arrives_at, i))
    pending = list(order)
    iteration = 0
    while pending or engine.has_work():
        while pending and specs[pending[0]].arrives_at <= iteration:
            i = pending.pop(0)
            setting = greedy(specs[i].output_length) if settings is None else settings[i]
            engine.submit(prompts[i], setting, sinks[i])
        engine.step()
        iteration += 1
        assert iteration < 50_000, "エンジンが終わらない"
    return sinks


def build_prompts(resources: Resources, specs: list[Spec], seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    vocabulary = resources.tokenizer.vocab_size
    return [random_prompt(rng, spec.prompt_length, vocabulary) for spec in specs]


# ---------------------------------------------------------------------------
# 正しさ: 同時に処理した結果が、1 要求ずつの生成と一致する
# ---------------------------------------------------------------------------
CORRECTNESS_CASES = [
    pytest.param(EngineConfig(max_batch_size=3, block_size=4), 0, id="continuous-paged"),
    pytest.param(
        EngineConfig(max_batch_size=3, kv_manager="reserved"), 0, id="continuous-reserved"
    ),
    pytest.param(EngineConfig(scheduler="static", max_batch_size=3, block_size=4), 0, id="static"),
    pytest.param(EngineConfig(max_batch_size=4, block_size=4), 2, id="staggered-arrivals"),
    pytest.param(EngineConfig(max_batch_size=2, block_size=1), 1, id="block-size-1"),
    pytest.param(EngineConfig(max_batch_size=3, block_size=64), 0, id="block-size-64"),
]


@pytest.mark.parametrize(("config", "arrival_gap"), CORRECTNESS_CASES)
def test_concurrent_greedy_generation_matches_one_request_at_a_time(
    resources: Resources, config: EngineConfig, arrival_gap: int
) -> None:
    specs = make_specs(7, seed=11, arrival_gap=arrival_gap)
    prompts = build_prompts(resources, specs, seed=12)
    engine = make_engine(resources, config, record_logits=True)
    sinks = run_specs(engine, specs, prompts)
    loaded = resources.models["main"]
    for request_id, (spec, prompt, sink) in enumerate(zip(specs, prompts, sinks, strict=True)):
        assert sink.outcome is not None and sink.outcome.finish_reason == FINISH_LENGTH
        assert len(sink.tokens) == spec.output_length == sink.outcome.completion_tokens
        check_against_reference(loaded, prompt, sink.tokens, engine.recorded_logits[request_id])
    assert engine.stats()["running"] == 0 and engine.stats()["waiting"] == 0
    assert engine.store.usage().units_in_use == 0


def test_requests_that_need_eviction_still_match_the_reference(resources: Resources) -> None:
    """ブロックが足りず、実行中の要求を退避して再計算する経路(027 の P0 と同じ設計)。"""
    prompt_length, output_length, block = 12, 8, 4
    per_request = -(-(prompt_length + output_length - 1) // block)
    config = EngineConfig(max_batch_size=6, block_size=block, num_blocks=2 * per_request + 2)
    specs = [Spec(prompt_length, output_length) for _ in range(6)]
    prompts = build_prompts(resources, specs, seed=3)
    engine = make_engine(resources, config, record_logits=True)
    sinks = run_specs(engine, specs, prompts)
    assert engine.stats()["preemptions"] > 0, "退避の経路が通っていない"
    assert any(sink.outcome.num_preemptions > 0 for sink in sinks)
    for request_id, (prompt, sink) in enumerate(zip(prompts, sinks, strict=True)):
        assert sink.outcome.finish_reason == FINISH_LENGTH
        check_against_reference(
            resources.models["main"], prompt, sink.tokens, engine.recorded_logits[request_id]
        )
    engine.store.check_consistency()
    assert engine.store.num_free_blocks == engine.store.num_blocks


def test_a_shared_prefix_is_reused_and_still_matches_the_reference(resources: Resources) -> None:
    rng = np.random.default_rng(5)
    vocabulary = resources.tokenizer.vocab_size
    shared = random_prompt(rng, 8, vocabulary)  # ブロック 4 つの 2 つ分
    prompts = [[*shared, *random_prompt(rng, 3 + i, vocabulary)] for i in range(4)]
    specs = [Spec(len(p), 5, arrives_at=i * 2) for i, p in enumerate(prompts)]
    engine = make_engine(
        resources, EngineConfig(max_batch_size=2, block_size=4), record_logits=True
    )
    sinks = run_specs(engine, specs, prompts)
    assert engine.store.num_prefix_hit_blocks > 0, "接頭辞の共有が通っていない"
    for request_id, (prompt, sink) in enumerate(zip(prompts, sinks, strict=True)):
        check_against_reference(
            resources.models["main"], prompt, sink.tokens, engine.recorded_logits[request_id]
        )


def test_a_finished_row_in_a_static_batch_does_not_run_past_the_position_limit(
    resources: Resources,
) -> None:
    """static batching は終わった行も計算に残す。短い要求の行は、長い要求が終わるまでに、
    モデルの最大の位置(256)を超えて伸びうる。その行は計算から外れ、他の行の結果は変わらない。
    """
    long_prompt, short_output = 200, 1
    prompts = [
        random_prompt(np.random.default_rng(1), long_prompt, resources.tokenizer.vocab_size),
        random_prompt(np.random.default_rng(2), 1, resources.tokenizer.vocab_size),
    ]
    specs = [Spec(long_prompt, short_output), Spec(1, 250)]
    engine = make_engine(
        resources, EngineConfig(scheduler="static", max_batch_size=2), record_logits=True
    )
    sinks = run_specs(engine, specs, prompts)
    assert [len(s.tokens) for s in sinks] == [short_output, 250]
    for request_id, (prompt, sink) in enumerate(zip(prompts, sinks, strict=True)):
        check_against_reference(
            resources.models["main"], prompt, sink.tokens, engine.recorded_logits[request_id]
        )
    assert engine.store.usage().units_in_use == 0


# ---------------------------------------------------------------------------
# ServingEngine.run() と同じ反復の列になる(アプリ側の実行ループが、src/ の側と食い違わない)
# ---------------------------------------------------------------------------
ITERATION_FIELDS = (
    "prefill_rows",
    "decode_rows",
    "resumed_rows",
    "prefill_tokens",
    "query_tokens",
    "active_requests",
    "waiting",
    "tokens_used",
    "tokens_allocated",
    "units_in_use",
    "preemptions",
)

EQUIVALENCE_CASES = [
    pytest.param(EngineConfig(max_batch_size=3, block_size=4), id="continuous-paged"),
    pytest.param(
        EngineConfig(max_batch_size=3, block_size=4, num_blocks=14), id="continuous-eviction"
    ),
    pytest.param(EngineConfig(max_batch_size=3, block_size=4), id="prefix-sharing"),
    pytest.param(EngineConfig(max_batch_size=3, kv_manager="reserved"), id="continuous-reserved"),
    pytest.param(EngineConfig(scheduler="static", max_batch_size=3, block_size=4), id="static"),
]


@pytest.mark.parametrize("config", EQUIVALENCE_CASES)
def test_the_iterations_match_the_serving_engine_of_src(
    resources: Resources, config: EngineConfig
) -> None:
    """全要求が先に届いている負荷で、``ServingEngine.run()`` と、生成するトークンも反復ごとの記録
    (行数・退避の回数・メモリの使用量)も一致する。
    """
    specs = make_specs(9, seed=21)
    rng = np.random.default_rng(22)
    vocabulary = resources.tokenizer.vocab_size
    shared = random_prompt(rng, 8, vocabulary) if config.kv_manager == "paged" else []
    prompts = [[*shared, *random_prompt(rng, s.prompt_length, vocabulary)] for s in specs]

    loaded = resources.models["main"]
    reference_engine = ServingEngine(
        build_scheduler(config),
        build_store(ServingModel(loaded.model), config),
        ModelRunner(ServingModel(loaded.model)),
        WallClock(resources.device),
    )
    expected = reference_engine.run(
        [
            Request(i, tuple(p), s.output_length)
            for i, (p, s) in enumerate(zip(prompts, specs, strict=True))
        ]
    )

    engine = make_engine(resources, config, record_iterations=10_000)
    sinks = [Collector() for _ in specs]
    for prompt, spec, sink in zip(prompts, specs, sinks, strict=True):
        engine.submit(prompt, greedy(spec.output_length), sink)
    drive(engine)

    if config.num_blocks is not None:
        assert expected.num_preemptions > 0, "退避の経路が通っていない"
    assert [s.tokens for s in sinks] == [expected.generated[i] for i in range(len(specs))]
    actual = [tuple(getattr(r, f) for f in ITERATION_FIELDS) for r in engine.records]
    wanted = [tuple(getattr(r, f) for f in ITERATION_FIELDS) for r in expected.iterations]
    assert actual == wanted
    assert engine.stats()["preemptions"] == expected.num_preemptions


# ---------------------------------------------------------------------------
# サンプリング: 行ごとの設定とシード
# ---------------------------------------------------------------------------
SAMPLING_SETTINGS = [
    pytest.param(("temperature", 0.8, 40, 0.9), id="temperature"),
    pytest.param(("top_k", 1.0, 5, 0.9), id="top_k"),
    pytest.param(("top_p", 1.0, 40, 0.5), id="top_p"),
]


@pytest.mark.parametrize("sampling", SAMPLING_SETTINGS)
def test_sampling_rows_follow_their_own_settings_and_seeds(
    resources: Resources, sampling: tuple[str, float, int, float]
) -> None:
    """貪欲法の行とサンプリングの行を混ぜる。サンプリングの行のトークンは、記録した logits に、
    要求のシードで作った乱数生成器で同じ手順(``select_next_token``)を当てた結果と一致する。
    """
    method, temperature, top_k, top_p = sampling
    specs = make_specs(5, seed=31)
    prompts = build_prompts(resources, specs, seed=32)
    settings = [
        DecodingSettings(
            method=method if i % 2 == 0 else "greedy",
            max_new_tokens=specs[i].output_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=100 + i,
        )
        for i in range(len(specs))
    ]
    engine = make_engine(
        resources, EngineConfig(max_batch_size=3, block_size=4), record_logits=True
    )
    sinks = run_specs(engine, specs, prompts, settings)
    device = resources.device
    for request_id, (setting, sink) in enumerate(zip(settings, sinks, strict=True)):
        recorded = engine.recorded_logits[request_id]
        generator = torch.Generator(device=device)
        generator.manual_seed(setting.seed)
        replay = []
        for row in recorded:
            row_tensor = torch.from_numpy(np.asarray(row)).to(device)[None]
            replay.append(int(select_next_token(row_tensor, setting, generator).item()))
        assert sink.tokens == replay, f"要求 {request_id}({setting.method}): 再現できない"
    greedy_ids = [i for i, s in enumerate(settings) if s.method == "greedy"]
    for request_id in greedy_ids:
        assert sinks[request_id].tokens == [
            int(np.argmax(r)) for r in engine.recorded_logits[request_id]
        ]


def test_the_same_seed_gives_the_same_tokens_and_a_different_seed_does_not(
    resources: Resources,
) -> None:
    prompt = build_prompts(resources, [Spec(6, 12)], seed=1)[0]
    outputs = []
    for seed in (7, 7, 8):
        engine = make_engine(resources, EngineConfig(max_batch_size=2, block_size=4))
        sink = Collector()
        engine.submit(
            prompt,
            DecodingSettings(method="top_p", max_new_tokens=12, top_p=0.95, seed=seed),
            sink,
        )
        drive(engine)
        outputs.append(sink.tokens)
    assert outputs[0] == outputs[1]
    assert outputs[0] != outputs[2]


# ---------------------------------------------------------------------------
# 取り消し・時間の上限・待ち行列の上限・誤った要求
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(EngineConfig(max_batch_size=2, block_size=4), id="continuous-paged"),
        pytest.param(
            EngineConfig(max_batch_size=2, kv_manager="reserved"), id="continuous-reserved"
        ),
        pytest.param(EngineConfig(scheduler="static", max_batch_size=2, block_size=4), id="static"),
    ],
)
def test_cancelling_returns_the_blocks_and_does_not_disturb_the_others(
    resources: Resources, config: EngineConfig
) -> None:
    specs = [Spec(6, 30), Spec(5, 30), Spec(7, 10), Spec(4, 10)]
    prompts = build_prompts(resources, specs, seed=41)
    engine = make_engine(resources, config, record_logits=True)
    sinks = [Collector() for _ in specs]
    for prompt, spec, sink in zip(prompts, specs, sinks, strict=True):
        engine.submit(prompt, greedy(spec.output_length), sink)
    for _ in range(6):
        engine.step()
    engine.cancel(0)  # 実行中の要求
    engine.cancel(3)  # 待っている要求(最大バッチサイズ 2 なので、まだ実行に入っていない)
    drive(engine)

    assert sinks[0].outcome.finish_reason == FINISH_CANCELLED
    assert 0 < len(sinks[0].tokens) < specs[0].output_length
    assert sinks[3].outcome.finish_reason == FINISH_CANCELLED and sinks[3].tokens == []
    for request_id in (1, 2):
        assert sinks[request_id].outcome.finish_reason == FINISH_LENGTH
        check_against_reference(
            resources.models["main"],
            prompts[request_id],
            sinks[request_id].tokens,
            engine.recorded_logits[request_id],
        )
    assert engine.store.usage().units_in_use == 0, "取り消した要求のブロックが戻っていない"
    if hasattr(engine.store, "check_consistency"):
        engine.store.check_consistency()
        assert engine.store.num_free_blocks == engine.store.num_blocks
    stats = engine.stats()
    assert stats["requests"]["cancelled"] == 2 and stats["requests"]["completed"] == 2
    assert stats["not_started"] == 0


def test_cancelling_an_unknown_or_finished_request_does_nothing(resources: Resources) -> None:
    engine = make_engine(resources, EngineConfig(max_batch_size=2, block_size=4))
    sink = Collector()
    request_id = engine.submit([1, 2, 3], greedy(2), sink)
    drive(engine)
    engine.cancel(request_id)
    engine.cancel(12345)
    drive(engine)
    assert sink.outcome.finish_reason == FINISH_LENGTH


class FakeClock:
    """試験用の時計(進めた分だけ時間が経つ)。"""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_a_request_over_the_time_limit_is_cancelled_and_its_blocks_returned(
    resources: Resources,
) -> None:
    clock = FakeClock()
    config = EngineConfig(max_batch_size=2, block_size=4, request_timeout_seconds=5.0)
    engine = make_engine(resources, config, clock=clock)
    running, queued, fresh = Collector(), Collector(), Collector()
    engine.submit([1, 2, 3, 4], greedy(40), running)
    engine.submit([5, 6, 7], greedy(40), queued)
    for _ in range(3):
        engine.step()
    clock.now += 4.0
    engine.submit([8, 9], greedy(3), fresh)  # あとから届いた要求は、その時刻から数える
    clock.now += 2.0  # 最初の 2 件は 6 秒、3 件目は 2 秒
    drive(engine)
    assert running.outcome.finish_reason == FINISH_TIMEOUT
    assert queued.outcome.finish_reason == FINISH_TIMEOUT
    assert fresh.outcome.finish_reason == FINISH_LENGTH
    assert engine.store.usage().units_in_use == 0
    assert engine.stats()["requests"]["timed_out"] == 2


def test_the_queue_limit_counts_only_requests_that_have_not_started(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=1, block_size=4, max_queue_length=2)
    engine = make_engine(resources, config)
    sinks = [Collector() for _ in range(3)]
    engine.submit([1, 2, 3], greedy(20), sinks[0])
    engine.submit([1, 2, 3], greedy(20), sinks[1])
    with pytest.raises(QueueFull) as caught:  # 3 件目: まだ一度も実行に入っていない要求が 2 件ある
        engine.submit([1, 2, 3], greedy(20), sinks[2])
    assert caught.value.retry_after_seconds >= 1
    assert engine.stats()["requests"]["rejected"] == 1
    engine.step()  # 1 件目が実行に入る。待ち行列の数は 1 に減る
    assert engine.stats()["not_started"] == 1
    engine.submit([1, 2, 3], greedy(20), sinks[2])
    drive(engine)
    assert all(s.outcome.finish_reason == FINISH_LENGTH for s in sinks)
    assert engine.stats()["not_started"] == 0


def test_invalid_requests_are_refused_before_they_are_queued(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=2, block_size=4, num_blocks=8)  # 1 件の最大は 32 トークン
    engine = make_engine(resources, config)
    sink = Collector()
    max_len = resources.models["main"].max_sequence_length
    with pytest.raises(ValueError, match=str(max_len)):
        engine.submit([1] * 200, greedy(60), sink)  # 書き出し + 生成が最大の文脈長を超える
    with pytest.raises(ValueError, match="書き出しが空"):
        engine.submit([], greedy(3), sink)
    with pytest.raises(ValueError, match="ビームサーチ"):
        engine.submit([1], DecodingSettings(method="beam", max_new_tokens=3), sink)
    with pytest.raises(RequestTooLarge):  # モデルの上限には収まるが、KV キャッシュの予算を超える
        engine.submit([1] * 20, greedy(20), sink)
    assert engine.stats()["not_started"] == 0
    assert not engine.has_work()


def test_draining_refuses_new_requests_and_cancels_the_ones_not_started(
    resources: Resources,
) -> None:
    engine = make_engine(resources, EngineConfig(max_batch_size=1, block_size=4))
    first, second = Collector(), Collector()
    engine.submit([1, 2, 3], greedy(6), first)
    engine.submit([4, 5, 6], greedy(6), second)
    engine.step()  # 1 件目だけが実行に入る
    engine.begin_drain()
    with pytest.raises(EngineClosed):
        engine.submit([7], greedy(2), Collector())
    drive(engine)
    assert first.outcome.finish_reason == FINISH_LENGTH, "実行中の要求は終わらせる"
    assert second.outcome.finish_reason == "shutdown"
    assert engine.store.usage().units_in_use == 0


def test_abort_all_finishes_every_request_and_returns_the_blocks(resources: Resources) -> None:
    for scheduler in ("continuous", "static"):
        engine = make_engine(
            resources, EngineConfig(scheduler=scheduler, max_batch_size=2, block_size=4)
        )
        sinks = [Collector() for _ in range(4)]
        for sink in sinks:
            engine.submit([1, 2, 3, 4, 5], greedy(30), sink)
        for _ in range(4):
            engine.step()
        engine.abort_all("shutdown", "停止")
        assert all(s.outcome is not None and s.outcome.finish_reason == "shutdown" for s in sinks)
        assert engine.store.usage().units_in_use == 0
        assert not engine.has_work()
        engine.store.check_consistency()


def test_a_tight_budget_finishes_every_request_by_evicting(resources: Resources) -> None:
    """1 件ぶんの予算しかなくても、2 件同時に伸びたときは退避して順に終わる。"""
    # 書き出し 4 + 生成 12 = 16 トークン(ブロック 4 の 4 つ分)は予算 4 ブロックに載る。
    config = EngineConfig(max_batch_size=2, block_size=4, num_blocks=4)
    engine = make_engine(resources, config, record_logits=True)
    sinks = [Collector(), Collector()]
    for sink in sinks:
        engine.submit([3, 4, 5, 6], greedy(12), sink)
    drive(engine)
    assert all(s.outcome.finish_reason == FINISH_LENGTH for s in sinks)
    assert engine.stats()["preemptions"] > 0
    assert engine.store.usage().units_in_use == 0
