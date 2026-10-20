"""バッチ推論の処理(027 推論サーバ: continuous batching と PagedAttention。Gradio に依存しない)。

1 回の ``@spaces.GPU`` の関数の中で完結する。合成の負荷(``src/serving/workload.py``)を作り、
**同じ負荷** を ``ServingEngine``(``src/serving/engine.py``)に、static batching と continuous
batching で順に流して比べる。常駐するサーバ(``lab/serving/``)とは別で、ZeroGPU と互換である。

- ``run_batching_comparison``: 利用者が決めた負荷・最大バッチサイズ・KV キャッシュの管理方式で、
  2 つのスケジューラを実行する。
- ``run_batch_size_sweep``: 027 の観察 D。最大バッチサイズを等比の水準で振って、スループットと
  待ち時間の関係を測る(027 の 7.10 節は、最大バッチサイズを、アプリで使うモデルでこの観察を
  測り直して決めるとしている)。
- ``check_*``: GPU を使わずに判定できる入力の検査。``DryRunner``(モデルを実行せず、スケジューラと
  メモリ管理だけを進める)で、実行する反復の回数を見積もり、``@spaces.GPU`` の関数の時間(60 秒)に
  収まらない負荷を、GPU を割り当てる前に断る。

**static batching の KV キャッシュの予算は、常に制限なし** にする(最大バッチサイズ × 最大長)。
static batching は終わった行も計算に残すので、予算が足りなくなっても退避できず、実行が止まる
(027 の実験 A・B も、予算に制約がかからない設定で比べた)。予算の制限(退避が起きる)は、
continuous batching の実行にだけ掛ける(027 の実験 C と同じ)。

**機器によらない量と、実時間の指標を分けて返す。**

- 機器によらない量(反復の回数・無駄な行の割合・要求ごとの待ちの反復数・出力トークンあたりの反復数・
  KV キャッシュの使用量・退避の回数)は、スケジューラとメモリ管理だけで決まる。全要求が最初に届く
  (一括の)負荷では、同じ負荷・同じ設定なら、実行ごと・機器ごとに変わらない(``DryRunner`` で、
  モデルを実行せずに同じ値が得られる)。Poisson 過程の到着は、到着の時刻が秒で決まるので、1 反復の
  時間に依存し、機器によらない量にならない。
- 実時間の指標(スループット・待ち時間・1 反復の時間)は、機器に依存する。ウォームアップの後に、
  同じ負荷を ``REPEATS`` 回ずつ実行し、中央値と範囲を返す。手元の機器(MPS)では、1 反復の固定費
  (Python とカーネルの起動)が支配的で、実行ごとのばらつきも大きい。027 の判定は、Google Colab の
  T4 での結果である。

時間は ``WallClock``(デバイスの同期を取ってから実時間を読む。何も実行する要求がなく次の到着まで
待つ時間は、実際には待たずに時刻を進める)で測る。

導入: 段階 2-1(使った理論は 027・010・008)。
"""

from __future__ import annotations

import dataclasses
import functools
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from lab.core.artifacts import LoadedModel
from lab.core.passages import PASSAGE_TEXT, SYSTEM_PROMPT_TEXT
from src.data.tokenizer import BPEIDTokenizer
from src.serving import (
    ContinuousBatchScheduler,
    DryRunner,
    IterationCostModel,
    KeyValueStore,
    ModelClock,
    ModelRunner,
    PagedKeyValueStore,
    Request,
    ReservedKeyValueStore,
    RunResult,
    Scheduler,
    ServingEngine,
    ServingModel,
    StaticBatchScheduler,
    WallClock,
    build_requests,
    fit_cost_model,
    fit_output_length_distribution,
    latency_statistics,
    maximum_coefficient_of_variation,
    mean_active_requests,
    mean_computed_rows,
    memory_statistics,
    poisson_arrival_times,
    predicted_static_to_continuous_ratio,
    sample_output_lengths,
    sample_prompt_lengths,
    throughput,
    wasted_row_fraction,
)

SCHEDULER_STATIC = "static"
SCHEDULER_CONTINUOUS = "continuous"
KV_RESERVED = "reserved"
KV_PAGED = "paged"
KV_MANAGERS = (KV_RESERVED, KV_PAGED)
ARRIVAL_OFFLINE = "offline"
ARRIVAL_POISSON = "poisson"
ARRIVALS = (ARRIVAL_OFFLINE, ARRIVAL_POISSON)
BATCH_MODEL_KEYS = ("main", "gqa")
BLOCK_SIZES = (4, 8, 16, 32, 64)
"""画面で選べるブロックの大きさ(027 の観察 E と同じ等比の水準)。"""
SWEEP_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
"""観察 D の最大バッチサイズの水準(等比。027 の観察 D と同じ)。"""
SWEEP_ARRIVAL_RATIOS = (0.4, 0.8, 1.2)
"""観察 D の到着の速さ(オフラインの処理能力に対する比。027 の観察 D と同じ)。"""
REFERENCE_BATCH_SIZE = 16
"""観察 D の基準の到着率を測る、オフラインの実行の最大バッチサイズ(027 の既定)。"""

# 負荷の形(027 の設計。書き出し + 出力が、最大の文脈長 256 に収まる)
PROMPT_RANGE = (16, 48)
OUTPUT_MAX = 192
USER_PROMPT_RANGE = (8, 24)
SYSTEM_PROMPT_LENGTH = 64
OUTPUT_MAX_SHARED = 160
PROMPT_WINDOW_STRIDE = 5

# 画面で変えられる範囲
MAX_REQUESTS = 128
MAX_OUTPUT_MEAN = 96
MAX_BATCH_SIZE = 32
MAX_ARRIVAL_RATE = 500.0
MAX_BUDGET_SLOTS = 32

# @spaces.GPU の関数の時間(秒)に収める見積もり
GPU_SECONDS = 60
ESTIMATE_LIMIT_SECONDS = 40.0
"""見積もった計算時間の上限(``GPU_SECONDS`` から、準備と画面の更新の余裕を引いた値)。"""
ESTIMATE_COST_MODELS = (
    IterationCostModel(fixed=0.020, per_row=0.0005, per_extra_token=0.0001),
    IterationCostModel(fixed=0.002, per_row=0.0001, per_extra_token=0.00002),
)
"""見積もり用の保守的な 1 反復の時間(遅い環境と速い環境の 2 通り)。実測の値ではない
(027 の T4 の実測は、固定の項が約 6 ms)。到着が重なる負荷では、1 反復の時間によって
反復の回数が変わるので、2 通りの大きい方で見積もる。"""
ESTIMATE_SECONDS_PER_ITERATION = ESTIMATE_COST_MODELS[0].fixed
"""見積もりの反復の回数を、時間に直すときの 1 反復あたりの秒(遅い環境の固定の項)。"""

WARMUP_OUTPUT = 3
REPEATS = 3
"""実時間の指標を測るために、同じ負荷を同じスケジューラで実行する回数(ウォームアップの後)。
``@spaces.GPU`` の関数の時間の見積もりは、この回数を含める。"""
MIN_ITERATIONS_FOR_COST_FIT = 8
"""1 反復の時間の式(固定の項と行数への依存)を当てはめるのに必要な、反復の最小の回数。"""


@dataclass(frozen=True)
class BatchSettings:
    """「バッチ推論」の設定。

    Attributes:
        model_key: モデル(``main`` または ``gqa``)。
        num_requests: 要求の数。
        output_mean, output_cv: 出力長の平均と変動係数(切り詰めたガンマ分布の形)。
        arrival: ``offline``(全要求が最初に届く)または ``poisson``(Poisson 過程の到着)。
        arrival_rate: Poisson 過程の到着率 [要求 / 秒]。
        max_batch_size: 同時に実行する要求の数の上限。
        kv_manager: ``reserved``(最大長の分を確保)または ``paged``(PagedAttention)。
        block_size: ブロックの大きさ(トークン)。
        kv_budget_slots: KV キャッシュの予算(最大長 256 トークンの要求の何件分か)。
            continuous batching の実行にだけ掛ける。
        shared_prefix: 全要求が同じ system prompt(64 トークン)を持つ。接頭辞の共有は paged のとき。
        seed: 負荷を決めるシード。
    """

    model_key: str = "main"
    num_requests: int = 32
    output_mean: int = 24
    output_cv: float = 0.5
    arrival: str = ARRIVAL_OFFLINE
    arrival_rate: float = 30.0
    max_batch_size: int = 16
    kv_manager: str = KV_PAGED
    block_size: int = 16
    kv_budget_slots: int = 16
    shared_prefix: bool = False
    seed: int = 0


@dataclass(frozen=True)
class SweepSettings:
    """観察 D(最大バッチサイズを振る)の設定。KV キャッシュは paged で、予算に制約をかけない。"""

    model_key: str = "main"
    num_requests: int = 32
    output_mean: int = 24
    output_cv: float = 0.5
    arrival_ratio: float = 1.2
    block_size: int = 16
    seed: int = 0


@dataclass(frozen=True)
class SchedulerRun:
    """1 つのスケジューラでの実行の結果(時間は実行の開始を 0 とする秒)。

    要求ごとの配列は ``request_id`` の昇順(到着の昇順)。反復ごとの配列は反復の順。
    配列と ``time_breakdown``・``cost_fit`` は、``REPEATS`` 回の実行のうち、全体の時間が中央値の
    実行のもの(代表の実行)。

    Attributes:
        queue_wait_iterations: 要求ごとの、到着から最初に実行に入るまでに進んだ反復の数
            (機器によらない待ち時間)。
        iterations_per_output_token: 要求ごとの、最初のトークンの後の、出力トークン 1 つあたりの
            反復の数(退避がなければ 1。出力が 1 トークンの要求は NaN)。
        forward_seconds, query_tokens: 反復ごとの forward の時間と、処理した問い合わせのトークン数。
        cost_fit: 1 反復の時間の式 ``固定の項 + 行数の項 + 追加のトークンの項`` を、反復ごとの
            (行数, 問い合わせのトークン数, forward の時間)に当てはめた係数と診断量(``fixed``・
            ``per_row``・``per_extra_token`` [秒]、``r2``、``fixed_share`` = 固定の項 ÷ forward の
            時間の平均)。反復が少ないときは空。
        timing: 実時間の指標ごとの ``(中央値, 最小, 最大)``(``REPEATS`` 回の実行。時間は秒)。
        repeats: 実行した回数。
        structure_identical: ``REPEATS`` 回の実行で、反復の列(行数・待ち・KV キャッシュの使用量・
            待ちの反復数)が完全に一致したか。
    """

    scheduler: str
    arrival: np.ndarray
    admitted: np.ndarray
    first_token: np.ndarray
    finish: np.ndarray
    prompt_lengths: np.ndarray
    output_lengths: np.ndarray
    request_preemptions: np.ndarray
    queue_wait_iterations: np.ndarray
    iterations_per_output_token: np.ndarray
    iteration_end: np.ndarray
    forward_seconds: np.ndarray
    query_tokens: np.ndarray
    active_requests: np.ndarray
    computed_rows: np.ndarray
    waiting: np.ndarray
    units_in_use: np.ndarray
    tokens_used: np.ndarray
    tokens_allocated: np.ndarray
    throughput: float
    makespan: float
    iterations: int
    wasted_row_fraction: float
    mean_active_requests: float
    mean_computed_rows: float
    preemptions: int
    prefix_hit_blocks: int
    copy_on_write: int
    kv_unit: str
    kv_capacity_units: int
    latency: dict[str, float]
    memory: dict[str, float]
    time_breakdown: dict[str, float]
    cost_fit: dict[str, float]
    timing: dict[str, tuple[float, float, float]]
    repeats: int
    structure_identical: bool


@dataclass(frozen=True)
class BatchOutcome:
    """2 つのスケジューラの比較の結果。

    Attributes:
        settings: 実行した設定。
        device: 実行したデバイスの種類。
        static, continuous: それぞれの実行の結果。
        theory_ratio: 反復の回数の比(static / continuous)の理論値 ``E[max_B O] / E[O]``
            (027 の 3 章。continuous が行を隙間なく詰める理想の上限として読む)。
        output_mean_realized, output_cv_realized: 標本抽出した出力長の平均と変動係数。
        iterations_estimated: 実行前の見積もり(static, continuous)。
        compute_seconds: 関数全体の時間(準備を含む)。
    """

    settings: BatchSettings
    device: str
    static: SchedulerRun
    continuous: SchedulerRun
    theory_ratio: float
    output_mean_realized: float
    output_cv_realized: float
    iterations_estimated: tuple[int, int]
    compute_seconds: float


@dataclass(frozen=True)
class SweepOutcome:
    """観察 D の結果。配列は ``batch_sizes`` の順。"""

    settings: SweepSettings
    device: str
    reference_rate: float
    reference_throughput: float
    batch_sizes: np.ndarray
    throughput: np.ndarray
    time_to_first_token_mean: np.ndarray
    time_to_first_token_p95: np.ndarray
    time_per_output_token_mean: np.ndarray
    queue_wait_mean: np.ndarray
    mean_iteration_seconds: np.ndarray
    mean_active_requests: np.ndarray
    iterations_estimated: int
    compute_seconds: float


# ---------------------------------------------------------------------------
# 入力の検査(GPU を使わない)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def output_distribution(mean: int, coefficient_of_variation: float, maximum: int):
    """出力長の分布(平均と変動係数が指定の値に一致する、切り詰めたガンマ分布の形)。

    解くのに 1 秒近くかかることがあるので、同じ引数ではキャッシュを返す。
    """
    return fit_output_length_distribution(float(mean), coefficient_of_variation, 1, maximum)


@functools.lru_cache(maxsize=4)
def prompt_material(
    tokenizer: BPEIDTokenizer,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """書き出しの候補(窓)と、共通の system prompt のトークン列。"""
    tokens = tokenizer.encode(PASSAGE_TEXT)
    window = PROMPT_RANGE[1]
    pool = tuple(
        tuple(tokens[start : start + window])
        for start in range(0, len(tokens) - window + 1, PROMPT_WINDOW_STRIDE)
    )
    if len(pool) < 20:
        raise ValueError("書き出しの候補が足りない(英文のトークン数が少なすぎる)")
    system = tokenizer.encode(SYSTEM_PROMPT_TEXT * 8)
    return pool, tuple(system[:SYSTEM_PROMPT_LENGTH])


def output_maximum(shared_prefix: bool) -> int:
    return OUTPUT_MAX_SHARED if shared_prefix else OUTPUT_MAX


def _check_load(
    max_context: int,
    num_requests: int,
    output_mean: int,
    output_cv: float,
    shared_prefix: bool,
) -> None:
    if not 1 <= num_requests <= MAX_REQUESTS:
        raise ValueError(f"要求の数は 1 以上 {MAX_REQUESTS} 以下にしてください。")
    maximum = output_maximum(shared_prefix)
    prompt_max = USER_PROMPT_RANGE[1] + SYSTEM_PROMPT_LENGTH if shared_prefix else PROMPT_RANGE[1]
    if prompt_max + maximum > max_context:
        raise ValueError("書き出しと出力の上限の和が、モデルの最大の文脈長を超えます。")
    if not 2 <= output_mean <= min(MAX_OUTPUT_MEAN, maximum - 1):
        raise ValueError(
            f"出力長の平均は 2 以上 {min(MAX_OUTPUT_MEAN, maximum - 1)} 以下にしてください。"
        )
    if not output_cv > 0:
        raise ValueError("出力長の変動係数は正にしてください。")
    limit = maximum_coefficient_of_variation(float(output_mean), 1, maximum)
    if output_cv >= limit:
        raise ValueError(
            f"出力長の平均 {output_mean} では、変動係数は {limit:.2f} 未満にしてください"
            f"(範囲 1〜{maximum} で取りうる上限)。"
        )


def check_settings(max_context: int, settings: BatchSettings) -> None:
    """設定の範囲と、組み合わせが作れる負荷かを確かめる。

    Raises:
        ValueError: 画面にそのまま表示できる日本語のメッセージ。
    """
    if settings.model_key not in BATCH_MODEL_KEYS:
        raise ValueError(f"モデルは {', '.join(BATCH_MODEL_KEYS)} のどちらかにしてください。")
    if settings.arrival not in ARRIVALS:
        raise ValueError("到着は、一斉か Poisson 過程のどちらかにしてください。")
    if settings.kv_manager not in KV_MANAGERS:
        raise ValueError(
            "KV キャッシュの管理方式は、最大長の確保か PagedAttention にしてください。"
        )
    _check_load(
        max_context,
        settings.num_requests,
        settings.output_mean,
        settings.output_cv,
        settings.shared_prefix,
    )
    if not 1 <= settings.max_batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"最大バッチサイズは 1 以上 {MAX_BATCH_SIZE} 以下にしてください。")
    if settings.block_size not in BLOCK_SIZES:
        raise ValueError(f"ブロックの大きさは {', '.join(map(str, BLOCK_SIZES))} のどれかです。")
    if not 1 <= settings.kv_budget_slots <= MAX_BUDGET_SLOTS:
        raise ValueError(
            f"KV キャッシュの予算は 1 以上 {MAX_BUDGET_SLOTS} 以下(最大長の要求の件数)にします。"
        )
    if settings.arrival == ARRIVAL_POISSON and not 0 < settings.arrival_rate <= MAX_ARRIVAL_RATE:
        raise ValueError(f"到着率は 0 より大きく {MAX_ARRIVAL_RATE:g} 以下にしてください。")
    if not 0 <= settings.seed < 2**32:
        raise ValueError("乱数のシードは 0 以上 2^32 未満にしてください。")


def check_sweep_settings(max_context: int, settings: SweepSettings) -> None:
    if settings.model_key not in BATCH_MODEL_KEYS:
        raise ValueError(f"モデルは {', '.join(BATCH_MODEL_KEYS)} のどちらかにしてください。")
    _check_load(max_context, settings.num_requests, settings.output_mean, settings.output_cv, False)
    if settings.arrival_ratio not in SWEEP_ARRIVAL_RATIOS:
        ratios = "・".join(f"{r:g}" for r in SWEEP_ARRIVAL_RATIOS)
        raise ValueError(f"到着の速さ(比)は {ratios} のどれかです。")
    if settings.block_size not in BLOCK_SIZES:
        raise ValueError(f"ブロックの大きさは {', '.join(map(str, BLOCK_SIZES))} のどれかです。")
    if not 0 <= settings.seed < 2**32:
        raise ValueError("乱数のシードは 0 以上 2^32 未満にしてください。")


# ---------------------------------------------------------------------------
# 負荷・スケジューラ・KV キャッシュ
# ---------------------------------------------------------------------------
def make_requests(
    tokenizer: BPEIDTokenizer,
    *,
    num_requests: int,
    output_mean: int,
    output_cv: float,
    seed: int,
    shared_prefix: bool,
    arrival_rate: float | None,
) -> list[Request]:
    """シードから決定的な負荷を作る(027 の ``make_requests`` と同じ手順)。

    到着の乱数は別の系列から引くので、同じシードの負荷は到着率が違っても中身が同じ。
    """
    pool, system = prompt_material(tokenizer)
    distribution = output_distribution(output_mean, output_cv, output_maximum(shared_prefix))
    rng = np.random.default_rng([seed, int(round(output_cv * 100)), int(shared_prefix)])
    outputs = sample_output_lengths(distribution, num_requests, rng)
    prompts = sample_prompt_lengths(
        *(USER_PROMPT_RANGE if shared_prefix else PROMPT_RANGE), num_requests, rng
    )
    if arrival_rate is None:
        arrivals = np.zeros(num_requests)
    else:
        arrival_rng = np.random.default_rng([seed, 7, int(shared_prefix)])
        arrivals = poisson_arrival_times(arrival_rate, num_requests, arrival_rng)
    return build_requests(
        pool, prompts, outputs, arrivals, rng, shared_prefix=system if shared_prefix else ()
    )


def build_scheduler(kind: str, max_batch_size: int) -> Scheduler:
    cls = StaticBatchScheduler if kind == SCHEDULER_STATIC else ContinuousBatchScheduler
    return cls(max_batch_size)


def build_store(
    serving_model: ServingModel,
    *,
    kv_manager: str,
    block_size: int,
    budget_slots: int,
    prefix_caching: bool,
    headroom: int,
    device: torch.device,
    materialize: bool,
) -> KeyValueStore:
    """``budget_slots`` は、最大長(モデルの最大の文脈長)の要求の何件分の予算か。"""
    common = {
        "num_layers": serving_model.num_layers,
        "num_key_value_heads": serving_model.num_key_value_heads,
        "head_dim": serving_model.head_dim,
        "device": device,
        "dtype": serving_model.dtype,
        "materialize": materialize,
    }
    max_length = serving_model.max_sequence_length
    if kv_manager == KV_RESERVED:
        return ReservedKeyValueStore(num_slots=budget_slots, slot_length=max_length, **common)
    num_blocks = budget_slots * -(-max_length // block_size)
    return PagedKeyValueStore(
        num_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_caching=prefix_caching,
        headroom_blocks_per_running=headroom,
        **common,
    )


def _store_for(
    serving_model: ServingModel,
    settings: BatchSettings,
    scheduler: str,
    device: torch.device,
    materialize: bool,
) -> KeyValueStore:
    limited = (
        scheduler == SCHEDULER_CONTINUOUS and settings.kv_budget_slots < settings.max_batch_size
    )
    budget = (
        settings.kv_budget_slots if scheduler == SCHEDULER_CONTINUOUS else settings.max_batch_size
    )
    return build_store(
        serving_model,
        kv_manager=settings.kv_manager,
        block_size=settings.block_size,
        budget_slots=budget,
        prefix_caching=settings.shared_prefix and settings.kv_manager == KV_PAGED,
        headroom=1 if limited else 0,
        device=device,
        materialize=materialize,
    )


def _iteration_index(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``values``(要求の記録の時刻)が、反復の記録の時刻のどれと一致するかの添字。

    要求の最初に実行に入った時刻・最初のトークンの時刻・終了の時刻は、反復の記録の同じ変数の値なので、
    完全に一致する。
    """
    index = np.searchsorted(times, values, side="left")
    if not (index < len(times)).all() or not np.array_equal(
        times[np.minimum(index, len(times) - 1)], values
    ):
        raise RuntimeError("要求の時刻が、どの反復の時刻とも一致しない")
    return index


def iteration_metrics(result: RunResult) -> tuple[np.ndarray, np.ndarray]:
    """機器によらない、要求ごとの量 ``(待ちの反復数, 出力トークンあたりの反復数)``。

    - 待ちの反復数: 到着の後に始まった最初の反復から、最初に実行に入った反復までに進んだ反復の数。
    - 出力トークンあたりの反復数: 最初のトークンを生成した反復から最後のトークンを生成した反復までの
      反復の数 ÷ (出力長 - 1)。退避がなければ 1(1 つの要求は 1 反復に 1 トークン)。

    どちらも、反復の列(スケジューラとメモリ管理が決める)だけから決まる。一括の負荷では、機器によらず
    実行ごとに変わらない。
    """
    its = result.iterations
    starts = np.array([it.start_time for it in its], dtype=np.float64)
    forward_ends = np.array([it.forward_end_time for it in its], dtype=np.float64)
    records = result.requests
    arrival = np.array([r.arrival_time for r in records], dtype=np.float64)
    admitted = _iteration_index(starts, np.array([r.admitted_time for r in records]))
    first = _iteration_index(forward_ends, np.array([r.first_token_time for r in records]))
    last = _iteration_index(forward_ends, np.array([r.finish_time for r in records]))
    arrived = np.searchsorted(starts, arrival, side="left")
    outputs = np.array([r.output_length for r in records], dtype=np.float64)
    per_token = np.where(outputs >= 2, (last - first) / np.maximum(outputs - 1, 1), np.nan)
    return (admitted - arrived).astype(np.int64), per_token


def fit_iteration_cost(result: RunResult) -> dict[str, float]:
    """1 反復の時間の式を、反復ごとの forward の時間に当てはめる(固定の項の割合を見るため)。

    反復が ``MIN_ITERATIONS_FOR_COST_FIT`` 回に満たないときは空の辞書。``fixed_share`` は、
    固定の項を forward の時間の平均で割った値で、1 に近いほど、行数にほとんど依存せず固定費が
    支配的。
    """
    its = result.iterations
    if len(its) < MIN_ITERATIONS_FOR_COST_FIT:
        return {}
    samples = [
        (it.prefill_rows + it.decode_rows, it.query_tokens, it.forward_seconds) for it in its
    ]
    model, diagnostics = fit_cost_model(samples)
    mean_forward = float(np.mean([sample[2] for sample in samples]))
    return {
        "fixed": float(model.fixed),
        "per_row": float(model.per_row),
        "per_extra_token": float(model.per_extra_token),
        "r2": float(diagnostics["r2"]),
        "fixed_share": min(float(model.fixed) / mean_forward, 1.0) if mean_forward > 0 else 0.0,
        "mean_forward_seconds": mean_forward,
    }


def summarize_run(
    scheduler: str, result: RunResult, store: KeyValueStore, requests: list[Request]
) -> SchedulerRun:
    """``RunResult`` を、プロセスの境界を越えられる値(配列と数値)にまとめる。"""
    its = result.iterations
    records = result.requests
    unit = "slot" if isinstance(store, ReservedKeyValueStore) else "block"
    capacity_units = store.num_slots if unit == "slot" else store.num_blocks
    wait_iterations, per_token = iteration_metrics(result)
    return SchedulerRun(
        scheduler=scheduler,
        arrival=np.array([r.arrival_time for r in records], dtype=np.float64),
        admitted=np.array([r.admitted_time for r in records], dtype=np.float64),
        first_token=np.array([r.first_token_time for r in records], dtype=np.float64),
        finish=np.array([r.finish_time for r in records], dtype=np.float64),
        prompt_lengths=np.array([r.prompt_length for r in records], dtype=np.int64),
        output_lengths=np.array([r.output_length for r in records], dtype=np.int64),
        request_preemptions=np.array([r.num_preemptions for r in records], dtype=np.int64),
        queue_wait_iterations=wait_iterations,
        iterations_per_output_token=per_token,
        iteration_end=np.array([it.end_time for it in its], dtype=np.float64),
        forward_seconds=np.array([it.forward_seconds for it in its], dtype=np.float64),
        query_tokens=np.array([it.query_tokens for it in its], dtype=np.int64),
        active_requests=np.array([it.active_requests for it in its], dtype=np.int64),
        computed_rows=np.array([it.prefill_rows + it.decode_rows for it in its], dtype=np.int64),
        waiting=np.array([it.waiting for it in its], dtype=np.int64),
        units_in_use=np.array([it.units_in_use for it in its], dtype=np.int64),
        tokens_used=np.array([it.tokens_used for it in its], dtype=np.int64),
        tokens_allocated=np.array([it.tokens_allocated for it in its], dtype=np.int64),
        throughput=float(throughput(result)),
        makespan=float(result.makespan),
        iterations=len(its),
        wasted_row_fraction=float(wasted_row_fraction(result)),
        mean_active_requests=float(mean_active_requests(result)),
        mean_computed_rows=float(mean_computed_rows(result)),
        preemptions=int(result.num_preemptions),
        prefix_hit_blocks=int(getattr(store, "num_prefix_hit_blocks", 0)),
        copy_on_write=int(getattr(store, "num_copy_on_write", 0)),
        kv_unit=unit,
        kv_capacity_units=int(capacity_units),
        latency={k: float(v) for k, v in latency_statistics(result).items()},
        memory={k: float(v) for k, v in memory_statistics(result).items()},
        time_breakdown={k: float(v) for k, v in result.time_breakdown().items()},
        cost_fit=fit_iteration_cost(result),
        timing={},
        repeats=1,
        structure_identical=True,
    )


# 反復の列が実行ごとに変わらないかを見る、機器によらない量(配列)
STRUCTURE_FIELDS = (
    "active_requests",
    "computed_rows",
    "waiting",
    "units_in_use",
    "tokens_used",
    "tokens_allocated",
    "queue_wait_iterations",
)


def combine_repeats(runs: list[SchedulerRun]) -> SchedulerRun:
    """同じ負荷の ``REPEATS`` 回の実行を 1 つにまとめる。

    全体の時間が中央値の実行を代表にして、実時間の指標の ``(中央値, 最小, 最大)`` と、
    反復の列が一致したかを加える。
    """
    ordered = sorted(runs, key=lambda run: run.makespan)
    representative = ordered[len(ordered) // 2]

    def spread(values: list[float]) -> tuple[float, float, float]:
        return float(np.median(values)), float(np.min(values)), float(np.max(values))

    timing = {
        "throughput": spread([run.throughput for run in runs]),
        "makespan": spread([run.makespan for run in runs]),
        "iteration_seconds": spread([run.makespan / run.iterations for run in runs]),
        "time_to_first_token_mean": spread(
            [run.latency["time_to_first_token_mean"] for run in runs]
        ),
        "time_to_first_token_p95": spread([run.latency["time_to_first_token_p95"] for run in runs]),
        "time_per_output_token_mean": spread(
            [run.latency["time_per_output_token_mean"] for run in runs]
        ),
        "queue_wait_mean": spread([run.latency["queue_wait_mean"] for run in runs]),
    }
    first = runs[0]
    identical = all(
        run.iterations == first.iterations
        and all(np.array_equal(getattr(run, f), getattr(first, f)) for f in STRUCTURE_FIELDS)
        for run in runs[1:]
    )
    return dataclasses.replace(
        representative, timing=timing, repeats=len(runs), structure_identical=identical
    )


# ---------------------------------------------------------------------------
# 実行の見積もり(GPU を使わない)
# ---------------------------------------------------------------------------
def dry_run(
    requests: list[Request],
    scheduler: Scheduler,
    store: KeyValueStore,
    cost_model: IterationCostModel,
) -> RunResult:
    """モデルを実行せずに、スケジューラとメモリ管理だけを進める(出力長は割り当てで決まる)。"""
    return ServingEngine(scheduler, store, DryRunner(), ModelClock(cost_model)).run(requests)


def _check_budget(iterations: int, repeats: int = 1) -> None:
    """反復の回数(1 回の実行の合計)× 繰り返しの回数 の見積もりが、時間の上限に収まるか。"""
    seconds = iterations * repeats * ESTIMATE_SECONDS_PER_ITERATION
    if seconds > ESTIMATE_LIMIT_SECONDS:
        times = f" × 繰り返し {repeats} 回" if repeats > 1 else ""
        raise ValueError(
            f"負荷が大きすぎます(反復 {iterations} 回{times}、見積もり {seconds:.0f} 秒 > "
            f"上限 {ESTIMATE_LIMIT_SECONDS:.0f} 秒。GPU の割り当ての時間 {GPU_SECONDS} 秒に収める"
            "ためです)。要求の数・出力長の平均を減らすか、最大バッチサイズを増やしてください。"
        )


def estimate_batch_iterations(
    loaded: LoadedModel, tokenizer: BPEIDTokenizer, settings: BatchSettings
) -> tuple[int, int]:
    """2 つのスケジューラの、1 回の実行の反復の回数を ``DryRunner`` で見積もる(GPU を使わない)。

    2 通りの 1 反復の時間(``ESTIMATE_COST_MODELS``)のうち大きい方を返す。一括の負荷では、反復の
    回数は 1 反復の時間によらず、実行の反復の回数と一致する。上限の検査は ``check_batch_request``。
    """
    check_settings(loaded.max_sequence_length, settings)
    serving_model = ServingModel(loaded.model)
    cpu = torch.device("cpu")
    rate = settings.arrival_rate if settings.arrival == ARRIVAL_POISSON else None
    requests = make_requests(
        tokenizer,
        num_requests=settings.num_requests,
        output_mean=settings.output_mean,
        output_cv=settings.output_cv,
        seed=settings.seed,
        shared_prefix=settings.shared_prefix,
        arrival_rate=rate,
    )
    counts = []
    for kind in (SCHEDULER_STATIC, SCHEDULER_CONTINUOUS):
        worst = 0
        for cost in ESTIMATE_COST_MODELS:
            store = _store_for(serving_model, settings, kind, cpu, materialize=False)
            result = dry_run(requests, build_scheduler(kind, settings.max_batch_size), store, cost)
            worst = max(worst, len(result.iterations))
        counts.append(worst)
    return counts[0], counts[1]


def check_batch_request(
    loaded: LoadedModel, tokenizer: BPEIDTokenizer, settings: BatchSettings
) -> tuple[int, int]:
    """設定を検査し、2 つの実行の反復の回数を見積もって ``(static, continuous)`` を返す。

    GPU を使わない。実行は ``REPEATS`` 回ずつ繰り返すので、見積もりの時間は、反復の回数の合計に
    ``REPEATS`` をかけた値で、時間の上限(``ESTIMATE_LIMIT_SECONDS``)を超えるときも ``ValueError``。

    Raises:
        ValueError: 画面にそのまま表示できる日本語のメッセージ。
    """
    counts = estimate_batch_iterations(loaded, tokenizer, settings)
    _check_budget(sum(counts), REPEATS)
    return counts


def check_sweep_request(
    loaded: LoadedModel, tokenizer: BPEIDTokenizer, settings: SweepSettings
) -> int:
    """観察 D の設定を検査し、全部の実行の反復の回数を見積もって返す(GPU を使わない)。"""
    check_sweep_settings(loaded.max_sequence_length, settings)
    serving_model = ServingModel(loaded.model)
    cpu = torch.device("cpu")
    worst_per_cost = []
    for cost in ESTIMATE_COST_MODELS:
        offline = make_requests(
            tokenizer,
            num_requests=settings.num_requests,
            output_mean=settings.output_mean,
            output_cv=settings.output_cv,
            seed=settings.seed,
            shared_prefix=False,
            arrival_rate=None,
        )
        reference = dry_run(
            offline,
            ContinuousBatchScheduler(REFERENCE_BATCH_SIZE),
            _sweep_store(serving_model, settings, REFERENCE_BATCH_SIZE, cpu, False),
            cost,
        )
        rate = settings.arrival_ratio * _reference_rate(reference, settings.output_mean)
        online = make_requests(
            tokenizer,
            num_requests=settings.num_requests,
            output_mean=settings.output_mean,
            output_cv=settings.output_cv,
            seed=settings.seed,
            shared_prefix=False,
            arrival_rate=rate,
        )
        counts = [len(reference.iterations)]
        for batch_size in SWEEP_BATCH_SIZES:
            result = dry_run(
                online,
                ContinuousBatchScheduler(batch_size),
                _sweep_store(serving_model, settings, batch_size, cpu, False),
                cost,
            )
            counts.append(len(result.iterations))
        worst_per_cost.append(counts)
    # 2 通りの 1 反復の時間のうち、実行ごとに大きい方の反復の回数を足す
    total = int(np.max(np.array(worst_per_cost), axis=0).sum())
    _check_budget(total)
    return total


# ---------------------------------------------------------------------------
# 実行(@spaces.GPU の関数の中から呼ぶ)
# ---------------------------------------------------------------------------
def warm_up(
    serving_model: ServingModel, device: torch.device, requests: list[Request], rows: int
) -> None:
    """測る前に、カーネルの選択やメモリの確保を済ませる(最初の実行だけが遅くならないように)。

    実行する負荷の最初の ``rows`` 件の書き出し(短い出力)を、最大バッチサイズと同じ行数で 1 回流す。
    これで、実際の最初の prefill と同じ形の forward を、測る前に 1 度実行しておく。
    """
    sample = [
        Request(i, r.prompt_token_ids, min(WARMUP_OUTPUT, r.output_length), 0.0)
        for i, r in enumerate(requests[:rows])
    ]
    store = build_store(
        serving_model,
        kv_manager=KV_PAGED,
        block_size=16,
        budget_slots=max(len(sample), 1),
        prefix_caching=False,
        headroom=0,
        device=device,
        materialize=True,
    )
    ServingEngine(
        ContinuousBatchScheduler(max(len(sample), 1)),
        store,
        ModelRunner(serving_model),
        WallClock(device),
    ).run(sample)


def run_batching_comparison(
    loaded: LoadedModel, tokenizer: BPEIDTokenizer, device: torch.device, settings: BatchSettings
) -> BatchOutcome:
    """同じ負荷を static batching と continuous batching で実行して比べる。

    ``@spaces.GPU`` を付けた関数の中から呼ぶ。

    Raises:
        ValueError: 設定が誤っている、または負荷が時間の上限を超える。
    """
    started = time.perf_counter()
    estimated = check_batch_request(loaded, tokenizer, settings)
    serving_model = ServingModel(loaded.model)
    rate = settings.arrival_rate if settings.arrival == ARRIVAL_POISSON else None
    requests = make_requests(
        tokenizer,
        num_requests=settings.num_requests,
        output_mean=settings.output_mean,
        output_cv=settings.output_cv,
        seed=settings.seed,
        shared_prefix=settings.shared_prefix,
        arrival_rate=rate,
    )
    runner = ModelRunner(serving_model)
    warm_up(serving_model, device, requests, settings.max_batch_size)
    repeated: dict[str, list[SchedulerRun]] = {SCHEDULER_STATIC: [], SCHEDULER_CONTINUOUS: []}
    for _ in range(REPEATS):  # 方式を交互に実行する(機器の状態の変化が、片方に偏らないように)
        for kind in repeated:
            store = _store_for(serving_model, settings, kind, device, materialize=True)
            engine = ServingEngine(
                build_scheduler(kind, settings.max_batch_size), store, runner, WallClock(device)
            )
            result = engine.run(requests)
            repeated[kind].append(summarize_run(kind, result, store, requests))
    runs = [
        combine_repeats(repeated[SCHEDULER_STATIC]),
        combine_repeats(repeated[SCHEDULER_CONTINUOUS]),
    ]
    outputs = np.array([r.output_length for r in requests], dtype=np.float64)
    distribution = output_distribution(
        settings.output_mean, settings.output_cv, output_maximum(settings.shared_prefix)
    )
    return BatchOutcome(
        settings=settings,
        device=device.type,
        static=runs[0],
        continuous=runs[1],
        theory_ratio=float(
            predicted_static_to_continuous_ratio(distribution, settings.max_batch_size)
        ),
        output_mean_realized=float(outputs.mean()),
        output_cv_realized=float(outputs.std() / outputs.mean()),
        iterations_estimated=estimated,
        compute_seconds=time.perf_counter() - started,
    )


def _sweep_store(
    serving_model: ServingModel,
    settings: SweepSettings,
    batch_size: int,
    device: torch.device,
    materialize: bool,
) -> KeyValueStore:
    return build_store(
        serving_model,
        kv_manager=KV_PAGED,
        block_size=settings.block_size,
        budget_slots=batch_size,
        prefix_caching=False,
        headroom=0,
        device=device,
        materialize=materialize,
    )


def _reference_rate(reference: RunResult, output_mean: int) -> float:
    """オフラインの処理能力(トークン / 秒)を出力長の平均で割った、基準の到着率 [要求 / 秒]。"""
    return float(throughput(reference)) / output_mean


def run_batch_size_sweep(
    loaded: LoadedModel, tokenizer: BPEIDTokenizer, device: torch.device, settings: SweepSettings
) -> SweepOutcome:
    """観察 D: 最大バッチサイズを振って、スループットと待ち時間の関係を測る。

    手順は 027 の観察 D と同じ。まず、同じ負荷を一括(オフライン)で continuous batching に流し、
    その処理能力から基準の到着率を決める。次に、基準の到着率の ``arrival_ratio`` 倍の Poisson 過程の
    到着で、最大バッチサイズ 1・2・4・8・16・32 の continuous batching を実行する。
    各条件 1 回だけの測定で、繰り返しはしていない(ばらつきは見ていない)。
    """
    started = time.perf_counter()
    estimated = check_sweep_request(loaded, tokenizer, settings)
    serving_model = ServingModel(loaded.model)
    runner = ModelRunner(serving_model)
    offline = make_requests(
        tokenizer,
        num_requests=settings.num_requests,
        output_mean=settings.output_mean,
        output_cv=settings.output_cv,
        seed=settings.seed,
        shared_prefix=False,
        arrival_rate=None,
    )
    warm_up(serving_model, device, offline, REFERENCE_BATCH_SIZE)
    reference = ServingEngine(
        ContinuousBatchScheduler(REFERENCE_BATCH_SIZE),
        _sweep_store(serving_model, settings, REFERENCE_BATCH_SIZE, device, True),
        runner,
        WallClock(device),
    ).run(offline)
    reference_rate = _reference_rate(reference, settings.output_mean)
    online = make_requests(
        tokenizer,
        num_requests=settings.num_requests,
        output_mean=settings.output_mean,
        output_cv=settings.output_cv,
        seed=settings.seed,
        shared_prefix=False,
        arrival_rate=settings.arrival_ratio * reference_rate,
    )
    columns: dict[str, list[float]] = {
        key: []
        for key in (
            "throughput",
            "ttft_mean",
            "ttft_p95",
            "tpot_mean",
            "wait_mean",
            "iteration_seconds",
            "active",
        )
    }
    for batch_size in SWEEP_BATCH_SIZES:
        result = ServingEngine(
            ContinuousBatchScheduler(batch_size),
            _sweep_store(serving_model, settings, batch_size, device, True),
            runner,
            WallClock(device),
        ).run(online)
        latency = latency_statistics(result)
        compute = sum(it.end_time - it.start_time for it in result.iterations)
        columns["throughput"].append(throughput(result))
        columns["ttft_mean"].append(latency["time_to_first_token_mean"])
        columns["ttft_p95"].append(latency["time_to_first_token_p95"])
        columns["tpot_mean"].append(latency["time_per_output_token_mean"])
        columns["wait_mean"].append(latency["queue_wait_mean"])
        columns["iteration_seconds"].append(compute / len(result.iterations))
        columns["active"].append(mean_active_requests(result))
    as_array = {key: np.array(values, dtype=np.float64) for key, values in columns.items()}
    return SweepOutcome(
        settings=settings,
        device=device.type,
        reference_rate=float(reference_rate),
        reference_throughput=float(throughput(reference)),
        batch_sizes=np.array(SWEEP_BATCH_SIZES, dtype=np.int64),
        throughput=as_array["throughput"],
        time_to_first_token_mean=as_array["ttft_mean"],
        time_to_first_token_p95=as_array["ttft_p95"],
        time_per_output_token_mean=as_array["tpot_mean"],
        queue_wait_mean=as_array["wait_mean"],
        mean_iteration_seconds=as_array["iteration_seconds"],
        mean_active_requests=as_array["active"],
        iterations_estimated=estimated,
        compute_seconds=time.perf_counter() - started,
    )


def ratio_to_theory(realized: float, theory: float) -> float:
    """実測の比を理論の比で割った値(1 に近いほど理論どおり)。"""
    return realized / theory if theory > 0 and math.isfinite(theory) else float("nan")
