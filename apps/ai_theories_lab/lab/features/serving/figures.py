"""バッチ推論の図(matplotlib)。

図の文字は英語だけにする(Spaces の環境には日本語のフォントがなく、日本語は表示されないため)。
説明は、図の外の画面の文言(日本語)で行う。pyplot は使わず ``Figure`` を直接作る
(pyplot はグローバルな状態を持ち、複数のリクエストが同時に図を作ると壊れるため)。
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from lab.features.serving.service import BatchOutcome, SchedulerRun, SweepOutcome

WAIT_COLOR = "#b8b8b8"
PREFILL_COLOR = "#e69f00"
DECODE_COLOR = "#0072b2"
STATIC_COLOR = "#d55e00"
CONTINUOUS_COLOR = "#0072b2"


def _timeline(axes, run: SchedulerRun, title: str) -> None:
    index = np.arange(len(run.arrival))
    axes.barh(index, run.admitted - run.arrival, left=run.arrival, color=WAIT_COLOR, height=0.85)
    axes.barh(
        index, run.first_token - run.admitted, left=run.admitted, color=PREFILL_COLOR, height=0.85
    )
    axes.barh(
        index, run.finish - run.first_token, left=run.first_token, color=DECODE_COLOR, height=0.85
    )
    axes.set_title(title, fontsize=10)
    axes.set_ylabel("request (by arrival)", fontsize=9)
    axes.invert_yaxis()
    axes.margins(y=0.01)


def timeline_figure(outcome: BatchOutcome) -> Figure:
    """要求ごとの時間の帯(待ち・prefill・decode)。上が static、下が continuous。

    横軸は同じ(秒)で、同じ負荷が、どちらのスケジューラで早く終わるかを比べられる。
    """
    runs = [
        (outcome.static, f"Static batching  ({outcome.static.makespan:.2f} s)"),
        (outcome.continuous, f"Continuous batching  ({outcome.continuous.makespan:.2f} s)"),
    ]
    count = max(len(outcome.static.arrival), 1)
    figure = Figure(figsize=(9.0, min(9.0, 2.0 + 0.06 * count) + 1.6), dpi=100)
    axes = figure.subplots(2, 1, sharex=True)
    for axis, (run, title) in zip(axes, runs, strict=True):
        _timeline(axis, run, title)
    axes[1].set_xlabel("time since the start [s]", fontsize=9)
    figure.legend(
        handles=[
            Patch(color=WAIT_COLOR, label="waiting in the queue"),
            Patch(color=PREFILL_COLOR, label="until the first token (prefill)"),
            Patch(color=DECODE_COLOR, label="decode"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def memory_figure(outcome: BatchOutcome) -> Figure:
    """反復ごとの KV キャッシュの使用量。

    左: 使用中のブロック(または枠)の数。破線は予算(continuous batching の実行)。
    右: continuous batching の実行で、割り当てたトークン枠と、実際に使っているトークン枠
    (差が内部の断片化)。
    """
    static, continuous = outcome.static, outcome.continuous
    figure = Figure(figsize=(10.0, 3.8), dpi=100)
    left, right = figure.subplots(1, 2)
    unit = f"{continuous.kv_unit}s"
    left.plot(np.arange(static.iterations), static.units_in_use, color=STATIC_COLOR, label="static")
    left.plot(
        np.arange(continuous.iterations),
        continuous.units_in_use,
        color=CONTINUOUS_COLOR,
        label="continuous",
    )
    left.axhline(
        continuous.kv_capacity_units,
        color=CONTINUOUS_COLOR,
        linestyle="--",
        linewidth=1,
        label=f"continuous budget ({continuous.kv_capacity_units} {unit})",
    )
    left.set_xlabel("iteration", fontsize=9)
    left.set_ylabel(f"{unit} in use", fontsize=9)
    left.set_title(f"KV cache {unit} in use per iteration", fontsize=10)
    left.legend(fontsize=8)
    steps = np.arange(continuous.iterations)
    right.fill_between(
        steps, continuous.tokens_allocated, color="#cccccc", label="token slots allocated"
    )
    right.plot(steps, continuous.tokens_used, color=CONTINUOUS_COLOR, label="token slots used")
    right.set_xlabel("iteration", fontsize=9)
    right.set_ylabel("token slots", fontsize=9)
    right.set_title("Continuous: allocated vs used (gap = internal fragmentation)", fontsize=10)
    right.legend(fontsize=8)
    figure.tight_layout()
    return figure


def sweep_figure(outcome: SweepOutcome) -> Figure:
    """最大バッチサイズ(横軸、等比)に対する、スループット・最初のトークンまでの時間・
    出力トークンあたりの時間。
    """
    sizes = outcome.batch_sizes
    figure = Figure(figsize=(11.0, 3.6), dpi=100)
    axes = figure.subplots(1, 3)
    panels = [
        (outcome.throughput, "throughput [tokens/s]", "Throughput", False),
        (
            outcome.time_to_first_token_p95 * 1000.0,
            "95th percentile time to first token [ms]",
            "Time to first token (p95)",
            True,
        ),
        (
            outcome.time_per_output_token_mean * 1000.0,
            "mean time per output token [ms]",
            "Time per output token",
            False,
        ),
    ]
    for axis, (values, label, title, log) in zip(axes, panels, strict=True):
        axis.plot(sizes, values, "o-", color=CONTINUOUS_COLOR)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, [str(int(b)) for b in sizes])
        if log:
            axis.set_yscale("log")
        axis.set_xlabel("max batch size", fontsize=9)
        axis.set_ylabel(label, fontsize=9)
        axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure
