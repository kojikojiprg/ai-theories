"""バッチ推論(``lab/features/serving/``)の処理と画面の整形の試験。

重みが乱数の代替アーティファクトで走るので、スループットなどの数値に意味はない。確かめるのは、
負荷が決定的なこと、実行の記録の不変条件(時刻の順序・生成したトークン数・KV キャッシュの使用量)、
見積もり(``DryRunner``)が実行と合うこと、入力の検査、画面の整形と図が作れること。
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from lab.core.artifacts import Resources
from lab.features.serving.figures import memory_figure, sweep_figure, timeline_figure
from lab.features.serving.service import (
    ARRIVAL_OFFLINE,
    ARRIVAL_POISSON,
    ESTIMATE_LIMIT_SECONDS,
    ESTIMATE_SECONDS_PER_ITERATION,
    KV_PAGED,
    KV_RESERVED,
    REPEATS,
    STRUCTURE_FIELDS,
    SWEEP_BATCH_SIZES,
    BatchSettings,
    SweepSettings,
    _store_for,
    build_scheduler,
    check_batch_request,
    check_settings,
    check_sweep_request,
    dry_run,
    estimate_batch_iterations,
    iteration_metrics,
    make_requests,
    output_distribution,
    prompt_material,
    run_batch_size_sweep,
    run_batching_comparison,
    summarize_run,
)
from lab.features.serving.ui import TIMING_NOTE, format_summary, format_sweep
from src.serving import (
    IterationCostModel,
    ServingModel,
    maximum_coefficient_of_variation,
)

SMALL = BatchSettings(num_requests=10, output_mean=8, max_batch_size=4, kv_budget_slots=4)


def comparison(resources: Resources, settings: BatchSettings):
    return run_batching_comparison(
        resources.models[settings.model_key], resources.tokenizer, resources.device, settings
    )


# ---------------------------------------------------------------------------
# 負荷
# ---------------------------------------------------------------------------
def test_the_load_is_deterministic_and_the_arrival_rate_does_not_change_its_content(
    resources: Resources,
) -> None:
    kwargs = {
        "num_requests": 16,
        "output_mean": 12,
        "output_cv": 0.5,
        "seed": 3,
        "shared_prefix": False,
    }
    offline = make_requests(resources.tokenizer, arrival_rate=None, **kwargs)
    again = make_requests(resources.tokenizer, arrival_rate=None, **kwargs)
    online = make_requests(resources.tokenizer, arrival_rate=25.0, **kwargs)
    assert offline == again
    assert {r.arrival_time for r in offline} == {0.0}
    assert [r.arrival_time for r in online] == sorted(r.arrival_time for r in online)
    assert online[-1].arrival_time > 0
    # 到着の乱数は別の系列なので、同じシードの負荷は、到着率が違っても出力長・書き出しが同じ
    key = lambda r: (r.prompt_token_ids, r.output_length)  # noqa: E731
    assert sorted(map(key, offline)) == sorted(map(key, online))
    other_seed = make_requests(resources.tokenizer, arrival_rate=None, **{**kwargs, "seed": 4})
    assert offline != other_seed


def test_every_request_fits_in_the_context_length(resources: Resources) -> None:
    max_length = resources.models["main"].max_sequence_length
    for shared in (False, True):
        requests = make_requests(
            resources.tokenizer,
            num_requests=64,
            output_mean=48,
            output_cv=1.0,
            seed=0,
            shared_prefix=shared,
            arrival_rate=None,
        )
        assert all(r.prompt_length + r.output_length <= max_length for r in requests)
        if shared:
            _, system = prompt_material(resources.tokenizer)
            assert len(system) == 64
            assert all(r.prompt_token_ids[:64] == system for r in requests)


def test_the_output_distribution_matches_the_requested_mean_and_variation() -> None:
    for mean, cv in ((24, 0.5), (48, 1.0), (8, 0.25)):
        distribution = output_distribution(mean, cv, 192)
        assert distribution.mean == pytest.approx(mean, abs=1e-6)
        assert distribution.coefficient_of_variation == pytest.approx(cv, abs=1e-6)


# ---------------------------------------------------------------------------
# 実行の記録の不変条件
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(SMALL, id="offline-paged"),
        pytest.param(
            BatchSettings(**{**SMALL.__dict__, "arrival": ARRIVAL_POISSON, "arrival_rate": 50.0}),
            id="poisson-paged",
        ),
        pytest.param(BatchSettings(**{**SMALL.__dict__, "kv_manager": KV_RESERVED}), id="reserved"),
        pytest.param(
            BatchSettings(**{**SMALL.__dict__, "shared_prefix": True, "model_key": "gqa"}),
            id="shared-prefix-gqa",
        ),
    ],
)
def test_the_two_runs_process_the_same_load_and_their_records_are_consistent(
    resources: Resources, settings: BatchSettings
) -> None:
    outcome = comparison(resources, settings)
    static, continuous = outcome.static, outcome.continuous
    assert (static.scheduler, continuous.scheduler) == ("static", "continuous")
    assert np.array_equal(static.output_lengths, continuous.output_lengths)
    assert np.array_equal(static.prompt_lengths, continuous.prompt_lengths)
    assert np.array_equal(static.arrival, continuous.arrival)
    for run in (static, continuous):
        assert (run.arrival <= run.admitted + 1e-9).all()
        assert (run.admitted <= run.first_token + 1e-9).all()
        assert (run.first_token <= run.finish + 1e-9).all()
        # 1 反復で 1 トークンずつ生成するので、実行中の要求数の総和 = 生成したトークンの総数
        assert run.active_requests.sum() == run.output_lengths.sum()
        assert (run.computed_rows >= run.active_requests).all()
        assert run.iterations == len(run.active_requests) == len(run.units_in_use)
        assert run.throughput > 0 and run.makespan > 0
        assert 0.0 <= run.wasted_row_fraction < 1.0
        assert (run.tokens_used <= run.tokens_allocated).all()
    # static は終わった行を計算に残すので、無駄な行がある。continuous には無駄な行がない
    assert continuous.wasted_row_fraction == 0.0
    assert continuous.iterations <= static.iterations
    # 予算(continuous の KV キャッシュ)を超えて使わない。static は常に予算の制限なし
    assert continuous.units_in_use.max() <= continuous.kv_capacity_units
    assert static.kv_capacity_units == settings.max_batch_size * (
        1 if settings.kv_manager == KV_RESERVED else -(-256 // settings.block_size)
    )
    assert outcome.device == resources.device.type


def test_in_an_offline_load_the_estimated_iterations_equal_the_executed_ones(
    resources: Resources,
) -> None:
    """全要求が最初に届く負荷では、スケジューラとメモリ管理はトークンの値に依存しない。
    ``DryRunner`` の見積もりの反復の回数は、実行の反復の回数と一致する。
    """
    for settings in (
        SMALL,
        BatchSettings(**{**SMALL.__dict__, "kv_budget_slots": 2, "output_mean": 14}),
    ):
        outcome = comparison(resources, settings)
        assert outcome.iterations_estimated == (
            outcome.static.iterations,
            outcome.continuous.iterations,
        )


def test_a_budget_smaller_than_the_batch_makes_paged_run_more_requests_than_reserved(
    resources: Resources,
) -> None:
    """027 の実験 C と同じ設計: 予算(最大長の要求 3 件分)を揃えると、PagedAttention は
    最大長の分を確保する方式より多くの要求を同時に実行できる。
    """
    base = {
        **SMALL.__dict__,
        "num_requests": 12,
        "output_mean": 8,
        "max_batch_size": 8,
        "kv_budget_slots": 3,
    }
    paged = comparison(resources, BatchSettings(**{**base, "kv_manager": KV_PAGED})).continuous
    reserved = comparison(
        resources, BatchSettings(**{**base, "kv_manager": KV_RESERVED})
    ).continuous
    assert reserved.active_requests.max() <= 3
    assert paged.active_requests.max() > reserved.active_requests.max()
    assert paged.mean_active_requests > reserved.mean_active_requests


def test_a_tight_paged_budget_triggers_eviction_and_still_finishes(resources: Resources) -> None:
    """予算が最大長の要求 1 件分しかなく、同時に伸びた要求を退避して再計算する(退避の経路)。"""
    settings = BatchSettings(
        num_requests=6, output_mean=40, output_cv=0.3, max_batch_size=6, kv_budget_slots=1
    )
    run = comparison(resources, settings).continuous
    assert run.preemptions > 0
    assert run.request_preemptions.sum() >= run.preemptions
    assert run.active_requests.sum() == run.output_lengths.sum()


# ---------------------------------------------------------------------------
# 観察 D
# ---------------------------------------------------------------------------
def test_the_batch_size_sweep_measures_every_level(resources: Resources) -> None:
    outcome = run_batch_size_sweep(
        resources.models["main"],
        resources.tokenizer,
        resources.device,
        SweepSettings(num_requests=10, output_mean=5, arrival_ratio=0.8),
    )
    assert tuple(outcome.batch_sizes) == SWEEP_BATCH_SIZES
    for values in (
        outcome.throughput,
        outcome.time_to_first_token_mean,
        outcome.time_to_first_token_p95,
        outcome.time_per_output_token_mean,
        outcome.mean_iteration_seconds,
        outcome.mean_active_requests,
    ):
        assert values.shape == (len(SWEEP_BATCH_SIZES),) and np.isfinite(values).all()
    assert outcome.reference_rate == pytest.approx(outcome.reference_throughput / 5)
    assert (outcome.time_to_first_token_p95 >= outcome.time_to_first_token_mean - 1e-9).all()
    # 最大バッチサイズが 1 のとき、同時に実行する要求は高々 1 件
    assert outcome.mean_active_requests[0] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 入力の検査(GPU を使わない)
# ---------------------------------------------------------------------------
def invalid(**overrides) -> BatchSettings:
    return BatchSettings(**{**SMALL.__dict__, **overrides})


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (invalid(model_key="sft-synthetic"), "モデル"),
        (invalid(num_requests=0), "要求の数"),
        (invalid(num_requests=129), "要求の数"),
        (invalid(output_mean=1), "出力長の平均"),
        (invalid(output_mean=97), "出力長の平均"),
        (invalid(output_cv=0.0), "変動係数"),
        (invalid(output_mean=96, output_cv=1.0), "変動係数"),
        (invalid(arrival="burst"), "到着"),
        (invalid(arrival=ARRIVAL_POISSON, arrival_rate=0.0), "到着率"),
        (invalid(arrival=ARRIVAL_POISSON, arrival_rate=1e6), "到着率"),
        (invalid(max_batch_size=0), "最大バッチサイズ"),
        (invalid(max_batch_size=64), "最大バッチサイズ"),
        (invalid(kv_manager="mystery"), "管理方式"),
        (invalid(block_size=7), "ブロックの大きさ"),
        (invalid(kv_budget_slots=0), "予算"),
        (invalid(seed=-1), "シード"),
    ],
)
def test_invalid_settings_are_refused_with_a_japanese_message(
    resources: Resources, settings: BatchSettings, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_settings(256, settings)
    with pytest.raises(ValueError, match=message):
        check_batch_request(resources.models["main"], resources.tokenizer, settings)


def test_the_maximum_variation_bounds_the_check(resources: Resources) -> None:
    limit = maximum_coefficient_of_variation(48.0, 1, 192)
    ok = invalid(output_mean=48, output_cv=round(limit - 0.1, 2))
    check_settings(256, ok)
    with pytest.raises(ValueError, match="変動係数"):
        check_settings(256, invalid(output_mean=48, output_cv=round(limit + 0.05, 2)))


def test_a_load_over_the_time_limit_is_refused_before_the_gpu_is_used(
    resources: Resources,
) -> None:
    heavy = invalid(num_requests=64, output_mean=64, output_cv=0.5, max_batch_size=1)
    with pytest.raises(ValueError, match="負荷が大きすぎます") as caught:
        check_batch_request(resources.models["main"], resources.tokenizer, heavy)
    assert f"{ESTIMATE_LIMIT_SECONDS:.0f}" in str(caught.value)
    # 同じ出力長でも、要求の数を減らし最大バッチサイズを増やせば収まる
    wide = invalid(num_requests=32, output_mean=48, output_cv=0.5, max_batch_size=32)
    check_batch_request(resources.models["main"], resources.tokenizer, wide)
    check_batch_request(resources.models["main"], resources.tokenizer, BatchSettings())


def test_the_time_estimate_includes_the_repeats(resources: Resources) -> None:
    """実時間の指標のために同じ負荷を ``REPEATS`` 回ずつ実行するので、1 回の見積もりが上限に収まって
    いても、繰り返しを含めると上限を超える負荷は、断られる。
    """
    loaded, tokenizer = resources.models["main"], resources.tokenizer
    found = False
    for count in range(8, 129, 4):
        settings = invalid(num_requests=count, output_mean=64, output_cv=0.5, max_batch_size=8)
        single = sum(estimate_batch_iterations(loaded, tokenizer, settings))
        once = single * ESTIMATE_SECONDS_PER_ITERATION
        if once <= ESTIMATE_LIMIT_SECONDS < once * REPEATS:
            found = True
            with pytest.raises(ValueError, match="繰り返し") as caught:
                check_batch_request(loaded, tokenizer, settings)
            assert f"{REPEATS} 回" in str(caught.value)
            break
        if once <= ESTIMATE_LIMIT_SECONDS:
            check_batch_request(loaded, tokenizer, settings)  # 繰り返しを含めても収まる
    assert found, "1 回では収まり、繰り返しを含めると収まらない負荷が見つからない"


def test_the_estimate_grows_with_the_load_and_shrinks_with_the_batch_size(
    resources: Resources,
) -> None:
    loaded, tokenizer = resources.models["main"], resources.tokenizer
    small = check_batch_request(loaded, tokenizer, invalid(num_requests=8))
    large = check_batch_request(loaded, tokenizer, invalid(num_requests=32))
    wide = check_batch_request(loaded, tokenizer, invalid(num_requests=32, max_batch_size=16))
    assert small[1] < large[1]
    assert wide[1] < large[1]


def test_sweep_settings_are_checked(resources: Resources) -> None:
    loaded, tokenizer = resources.models["main"], resources.tokenizer
    assert check_sweep_request(loaded, tokenizer, SweepSettings(num_requests=8, output_mean=5)) > 0
    with pytest.raises(ValueError, match="到着の速さ"):
        check_sweep_request(loaded, tokenizer, SweepSettings(arrival_ratio=2.0))
    with pytest.raises(ValueError, match="要求の数"):
        check_sweep_request(loaded, tokenizer, SweepSettings(num_requests=0))
    with pytest.raises(ValueError, match="負荷が大きすぎます"):
        check_sweep_request(
            loaded, tokenizer, SweepSettings(num_requests=128, output_mean=96, output_cv=0.5)
        )


# ---------------------------------------------------------------------------
# 画面の整形と図
# ---------------------------------------------------------------------------
def test_the_summary_and_the_figures_are_built_from_an_outcome(resources: Resources) -> None:
    outcome = comparison(
        resources, invalid(arrival=ARRIVAL_POISSON, arrival_rate=50.0, shared_prefix=True)
    )
    summary = format_summary(outcome)
    for expected in (
        "static",
        "continuous",
        "スループット",
        "無駄な行の割合",
        "最初のトークンまでの時間",
        "出力トークンあたりの時間",
        "退避の回数",
        "接頭辞の共有",
        "理論との比較",
    ):
        assert expected in summary
    timeline = timeline_figure(outcome)
    memory = memory_figure(outcome)
    assert len(timeline.axes) == 2 and len(memory.axes) == 2
    # 要求ごとの帯の数 = 要求の数
    assert len(timeline.axes[0].patches) == 3 * outcome.settings.num_requests
    reserved = comparison(resources, invalid(kv_manager=KV_RESERVED))
    assert "スロット" in format_summary(reserved)
    assert "接頭辞の共有" not in format_summary(reserved)


def test_the_sweep_summary_and_figure(resources: Resources) -> None:
    outcome = run_batch_size_sweep(
        resources.models["main"],
        resources.tokenizer,
        resources.device,
        SweepSettings(num_requests=8, output_mean=5),
    )
    text = format_sweep(outcome)
    assert "基準の到着率" in text and "出力トークンあたり" in text
    assert text.count("\n| ") >= len(SWEEP_BATCH_SIZES)
    assert len(sweep_figure(outcome).axes) == 3


# ---------------------------------------------------------------------------
# 機器によらない量と、実時間の指標の繰り返し
# ---------------------------------------------------------------------------
LOAD = invalid(num_requests=14, output_mean=10, max_batch_size=4, kv_budget_slots=4)
"""反復が 8 回以上ある一括の負荷(式の当てはめができる)。"""


def structure(run) -> dict:
    return {f: getattr(run, f) for f in STRUCTURE_FIELDS} | {
        "iterations": run.iterations,
        "wasted": run.wasted_row_fraction,
        "preemptions": run.preemptions,
        "per_token": run.iterations_per_output_token,
    }


def assert_same_structure(a, b) -> None:
    left, right = structure(a), structure(b)
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_the_machine_independent_quantities_do_not_change_between_executions(
    resources: Resources,
) -> None:
    """同じ一括の負荷・同じ設定なら、反復の回数・無駄な行の割合・待ちの反復数・出力トークンあたりの
    反復数・KV キャッシュの使用量は、実行ごとに(2 回の呼び出しの間でも、1 回の呼び出しの中の
    ``REPEATS`` 回の間でも)完全に一致する。実時間の指標は一致しない。
    """
    first = comparison(resources, LOAD)
    second = comparison(resources, LOAD)
    for one, two in ((first.static, second.static), (first.continuous, second.continuous)):
        assert_same_structure(one, two)
        assert one.structure_identical and two.structure_identical
        assert one.repeats == two.repeats == REPEATS
    assert first.iterations_estimated == (first.static.iterations, first.continuous.iterations)


def test_the_machine_independent_quantities_equal_those_of_a_run_without_the_model(
    resources: Resources,
) -> None:
    """モデルを実行しない ``DryRunner``(機器もデバイスも使わない)の実行から、同じ量が得られる。
    1 反復の時間は仮の費用モデルなので、実時間とは違うが、一括の負荷では反復の列は同じになる。
    """
    loaded, tokenizer = resources.models["main"], resources.tokenizer
    outcome = comparison(resources, LOAD)
    serving_model = ServingModel(loaded.model)
    cpu = resources.device.__class__("cpu")
    requests = make_requests(
        tokenizer,
        num_requests=LOAD.num_requests,
        output_mean=LOAD.output_mean,
        output_cv=LOAD.output_cv,
        seed=LOAD.seed,
        shared_prefix=False,
        arrival_rate=None,
    )
    for kind, run in (("static", outcome.static), ("continuous", outcome.continuous)):
        for cost in (
            IterationCostModel(0.020, 0.0005, 0.0001),
            IterationCostModel(0.002, 0.0001, 0.00002),
        ):
            store = _store_for(serving_model, LOAD, kind, cpu, materialize=False)
            result = dry_run(requests, build_scheduler(kind, LOAD.max_batch_size), store, cost)
            dry = summarize_run(kind, result, store, requests)
            assert_same_structure(run, dry)
            wait, per_token = iteration_metrics(result)
            np.testing.assert_array_equal(wait, run.queue_wait_iterations)
            np.testing.assert_array_equal(per_token, run.iterations_per_output_token)


def test_the_iteration_based_waits_have_the_expected_shape(resources: Resources) -> None:
    outcome = comparison(resources, LOAD)
    static, continuous = outcome.static, outcome.continuous
    for run in (static, continuous):
        assert run.queue_wait_iterations.dtype == np.int64
        assert (run.queue_wait_iterations >= 0).all()
        # 退避がなければ、1 つの要求は 1 反復に 1 トークンを生成するので、1 反復 / トークン
        assert run.preemptions == 0
        finite = run.iterations_per_output_token[~np.isnan(run.iterations_per_output_token)]
        np.testing.assert_allclose(finite, 1.0)
    # 最初に実行に入れる要求(最大バッチサイズ分)は、1 反復も待たない
    assert (static.queue_wait_iterations[: LOAD.max_batch_size] == 0).all()
    assert (continuous.queue_wait_iterations[: LOAD.max_batch_size] == 0).all()
    # static は組が終わるまで次の組を入れないので、continuous より長く待つ要求がある
    assert static.queue_wait_iterations.max() >= continuous.queue_wait_iterations.max()


def test_a_poisson_load_reports_whether_its_iterations_matched_between_repeats(
    resources: Resources,
) -> None:
    """Poisson 過程の到着は秒で決まるので、機器によらない量になるとは限らない。一致したかを返す。"""
    outcome = comparison(
        resources, invalid(**{**LOAD.__dict__, "arrival": ARRIVAL_POISSON, "arrival_rate": 80.0})
    )
    assert isinstance(outcome.static.structure_identical, bool)
    assert isinstance(outcome.continuous.structure_identical, bool)
    assert outcome.settings.arrival != ARRIVAL_OFFLINE
    assert "機器によらない量にならず" in format_summary(outcome)


def test_the_timing_metrics_are_repeated_and_reported_as_a_median_and_a_range(
    resources: Resources,
) -> None:
    outcome = comparison(resources, LOAD)
    for run in (outcome.static, outcome.continuous):
        assert run.repeats == REPEATS
        assert set(run.timing) == {
            "throughput",
            "makespan",
            "iteration_seconds",
            "time_to_first_token_mean",
            "time_to_first_token_p95",
            "time_per_output_token_mean",
            "queue_wait_mean",
        }
        for median, low, high in run.timing.values():
            assert low <= median <= high
        # 代表の実行は、全体の時間が中央値の実行
        assert run.makespan == run.timing["makespan"][0]
        assert run.throughput > 0


def test_the_cost_fit_shows_how_much_of_an_iteration_is_fixed(resources: Resources) -> None:
    outcome = comparison(resources, LOAD)
    for run in (outcome.static, outcome.continuous):
        assert run.iterations >= 8
        fit = run.cost_fit
        assert set(fit) == {
            "fixed",
            "per_row",
            "per_extra_token",
            "r2",
            "fixed_share",
            "mean_forward_seconds",
        }
        assert all(np.isfinite(v) for v in fit.values())
        assert fit["fixed"] >= 0 and fit["per_row"] >= 0 and fit["per_extra_token"] >= 0
        assert 0.0 <= fit["fixed_share"] <= 1.0
        breakdown = run.time_breakdown
        assert breakdown["forward"] + breakdown["other"] + breakdown["between"] + breakdown[
            "idle"
        ] == pytest.approx(breakdown["makespan"], rel=1e-6)
        assert breakdown["layout"] <= breakdown["forward"] + 1e-9
    tiny = comparison(resources, invalid(num_requests=4, output_mean=2, max_batch_size=4))
    assert tiny.continuous.iterations < 8 and tiny.continuous.cost_fit == {}
    assert "-" in format_summary(tiny)


def test_the_summary_puts_the_machine_independent_quantities_before_the_timing(
    resources: Resources,
) -> None:
    summary = format_summary(comparison(resources, LOAD))
    independent = summary.index("機器によらない量(スケジューラとメモリ管理だけで決まる)")
    timing = summary.index("実時間の指標(機器に依存")
    breakdown = summary.index("1 反復の時間の内訳")
    assert independent < timing < breakdown
    for label in (
        "待ちの反復数",
        "出力トークンあたりの反復数",
        "無駄な行の割合",
        "反復の回数",
        "固定の項の割合",
        "中央値(最小〜最大)",
    ):
        assert label in summary
    assert str(REPEATS) in summary


def test_the_note_on_the_page_explains_that_timing_depends_on_the_device() -> None:
    """画面の説明文: 実時間の結果は機器に依存し、手元の機器では固定費が支配的で
    continuous batching が速く見えないことがある。027 の判定は T4 での結果。
    測定値(ms・%・秒・トークン/秒)は書かない。
    """
    for expected in (
        "機器に依存",
        "固定費",
        "continuous batching が速く見えない",
        "Google Colab の T4",
    ):
        assert expected in TIMING_NOTE
    assert not re.search(r"\d+(?:\.\d+)?\s*(?:ms|%|秒|トークン)", TIMING_NOTE)
