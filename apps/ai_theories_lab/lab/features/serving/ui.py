"""バッチ推論の画面(027 推論サーバ: continuous batching と PagedAttention)。

処理は ``service.py`` にあり、ここは画面の組み立てと、結果の表示用の整形だけを行う。
GPU を使う処理は ``@gpu`` を付けた関数の中で、1 回の呼び出しで完結する(ZeroGPU と互換)。
常駐して要求を受け付けるサーバは、別の入口(``serve.py``、``lab/serving/``)で、手元専用。
"""

from __future__ import annotations

import gradio as gr
from matplotlib.figure import Figure

from lab.core.artifacts import Resources
from lab.core.device import gpu
from lab.features.serving.figures import memory_figure, sweep_figure, timeline_figure
from lab.features.serving.service import (
    ARRIVAL_OFFLINE,
    ARRIVAL_POISSON,
    BLOCK_SIZES,
    ESTIMATE_LIMIT_SECONDS,
    GPU_SECONDS,
    KV_PAGED,
    KV_RESERVED,
    MAX_ARRIVAL_RATE,
    MAX_BATCH_SIZE,
    MAX_BUDGET_SLOTS,
    MAX_OUTPUT_MEAN,
    MAX_REQUESTS,
    REPEATS,
    SWEEP_ARRIVAL_RATIOS,
    SWEEP_BATCH_SIZES,
    BatchOutcome,
    BatchSettings,
    SchedulerRun,
    SweepOutcome,
    SweepSettings,
    check_batch_request,
    check_sweep_request,
    output_distribution,
    output_maximum,
    ratio_to_theory,
    run_batch_size_sweep,
    run_batching_comparison,
)
from lab.ui.layout import Page
from lab.ui.links import notebook_link
from lab.ui.markdown import items, paragraphs

PAGE_TITLE = "バッチ推論 / Batch Inference"

# @spaces.GPU の duration(秒)。見積もりの上限(ESTIMATE_LIMIT_SECONDS)は、これに収まる値。
BATCH_GPU_SECONDS = GPU_SECONDS
SWEEP_GPU_SECONDS = GPU_SECONDS

MODEL_CHOICES = [
    ("main — 標準の多頭注意機構 / Multi-Head Attention (008)", "main"),
    ("gqa — Grouped-Query Attention (010)", "gqa"),
]
ARRIVAL_CHOICES = [
    ("一斉(全要求が最初に届く) / Offline", ARRIVAL_OFFLINE),
    ("Poisson 過程の到着 / Poisson arrivals", ARRIVAL_POISSON),
]
KV_CHOICES = [
    ("最大長の分を前もって確保 / Reserved slots", KV_RESERVED),
    ("PagedAttention(ブロック管理) / Paged blocks", KV_PAGED),
]

BATCH_MARKDOWN = paragraphs(
    "合成の負荷(出力長の平均と変動係数、到着の仕方を指定)を、 "
    "**同じ条件で 2 つのスケジューラに流して** "
    f"比べます。理論は {notebook_link('027')} です(実装は`src/serving/`)。",
    items(
        "**static batching**: 要求の組を固定し、バッチの中で最も出力長が長い要求が終わるまで、"
        "組を入れ替えません。短い要求が先に終わっても、その行は(捨てる出力のために)計算され続けます。",
        "**continuous batching**(Orca の iteration-level scheduling): 反復ごとに、"
        "終わった要求を抜き、待っている要求を入れます。",
        "**KV キャッシュの管理**: 最大長(256 トークン)の分を要求ごとに前もって確保する方式と、"
        "固定長のブロックに分けて要求が伸びるたびに割り当てる PagedAttention(接頭辞の共有・"
        "メモリ不足のときの退避つき)。",
    ),
    "出力長は、終端記号がないモデルなので、要求ごとに割り当てた長さで止めます。"
    "出力長のばらつき(変動係数)が大きいほど、static batching は短い要求の待ちが増えて"
    "不利になります。",
)

LIMIT_MARKDOWN = (
    f"**GPU の割り当ての時間({GPU_SECONDS} 秒)に収めるため**、実行の前に、モデルを動かさずに"
    "反復の回数を見積もり(`DryRunner`)、"
    f"見積もりが {ESTIMATE_LIMIT_SECONDS:.0f} 秒を超える負荷は、GPU を使う前に断ります。"
    f"見積もりは、実時間の指標のための繰り返し({REPEATS} 回)を含めます。"
    "**static batching の KV キャッシュの予算は常に制限なし**(最大バッチサイズ × 最大長)です。"
    "終わった行も計算に残すので、予算が足りなくなっても退避できず、実行が止まってしまうためです。"
    "予算の制限と退避は、continuous batching の実行にだけ掛かります(027 の実験 C と同じ)。"
)

TIMING_NOTE = (
    "**実時間の結果は機器に依存します。** 手元の機器"
    "(Apple Silicon の MPS(Metal Performance Shaders)など)では、"
    "1 反復の固定費(Python とカーネルの起動)が支配的で、"
    "continuous batching が速く見えないことがあります。"
    "027 の判定は、Google Colab の T4 での結果です。"
    "**機器によらない量**(反復の回数・無駄な行の割合・待ちの反復数など)は、スケジューラと"
    "メモリ管理だけで決まるので、一括の負荷では、機器が変わっても実行ごとに同じ値になります。"
    "実時間の指標は、ウォームアップの後に、同じ負荷を"
    f" {REPEATS} 回ずつ実行した、中央値と(最小〜最大)です。"
    "デバイスの同期を取ってから読み、何も実行する要求がなく次の到着まで待つ時間は、"
    "実際には待たずに時刻を進めています。"
)

SWEEP_MARKDOWN = paragraphs(
    "027 の **観察 D** を、このアプリで使うモデルで測り直します。027 の 7.10 節は、"
    "最大バッチサイズの既定値(16)を、アプリで使うモデルで観察 D を測り直して決めるとしています。",
    items(
        "まず、同じ負荷を一括(オフライン)で continuous batching(最大バッチサイズ 16)に流し、"
        "その処理能力から **基準の到着率** を決めます(スループット ÷ 出力長の平均)。",
        "次に、基準の到着率の「到着の速さ(比)」倍の Poisson 過程の到着で、最大バッチサイズ "
        f"{'・'.join(map(str, SWEEP_BATCH_SIZES))} の continuous batching を実行します"
        "(KV キャッシュは PagedAttention で、予算に制約をかけません)。",
        "最大バッチサイズを増やすと、スループットは上がりますが、1 反復の時間"
        "(出力トークンあたりの時間)が伸びるモデルでは、待ち時間との交換が生じます。"
        "このモデルで起きるかを見ます。",
    ),
    "**測るのは 1 回だけで、既定値を変えるかは、測った結果を見て別に判断します。**"
    "1 台のデバイス・1 つのモデル・合成の負荷での観察で、判定を設けていません。",
)


def format_milliseconds(seconds: float) -> str:
    return f"{seconds * 1000:,.1f}"


def _ratio(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.2f}" if denominator > 0 else "-"


def format_range(value: tuple[float, float, float], scale: float = 1.0, digits: int = 1) -> str:
    """``(中央値, 最小, 最大)`` を ``中央値(最小〜最大)`` の文字列にする。"""
    median, low, high = value
    return f"{median * scale:,.{digits}f}({low * scale:,.{digits}f}〜{high * scale:,.{digits}f})"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _share(parts: dict[str, float], key: str) -> str:
    return f"{parts[key] / parts['makespan']:.1%}"


def format_summary(outcome: BatchOutcome) -> str:
    """2 つのスケジューラの結果。機器によらない量の表、実時間の指標の表、1 反復の時間の内訳の順。"""
    s, c = outcome.static, outcome.continuous
    settings = outcome.settings
    unit_ja = "スロット" if c.kv_unit == "slot" else "ブロック"
    ratio = _ratio

    independent_rows = [
        (
            "反復の回数(1 反復 = 1 回の forward)",
            str(s.iterations),
            str(c.iterations),
            ratio(c.iterations, s.iterations),
        ),
        (
            "反復あたりの実行中の要求数(平均)",
            f"{s.mean_active_requests:.2f}",
            f"{c.mean_active_requests:.2f}",
            "",
        ),
        (
            "反復あたりの計算した行数(平均)",
            f"{s.mean_computed_rows:.2f}",
            f"{c.mean_computed_rows:.2f}",
            "",
        ),
        (
            "無駄な行の割合(終わった行を計算した分)",
            f"{s.wasted_row_fraction:.1%}",
            f"{c.wasted_row_fraction:.1%}",
            "",
        ),
        (
            "待ちの反復数(到着から最初に実行に入るまで。平均 / 最大)",
            f"{s.queue_wait_iterations.mean():.1f} / {int(s.queue_wait_iterations.max())}",
            f"{c.queue_wait_iterations.mean():.1f} / {int(c.queue_wait_iterations.max())}",
            ratio(c.queue_wait_iterations.mean(), s.queue_wait_iterations.mean()),
        ),
        (
            "出力トークンあたりの反復数(平均。1 は退避なし)",
            _mean_or_dash(s.iterations_per_output_token),
            _mean_or_dash(c.iterations_per_output_token),
            "",
        ),
        ("退避の回数(メモリ不足で取り下げて再計算)", str(s.preemptions), str(c.preemptions), ""),
        (
            f"使用中の{unit_ja}数(平均 / 最大)",
            f"{s.units_in_use.mean():.1f} / {int(s.units_in_use.max())}",
            f"{c.units_in_use.mean():.1f} / {int(c.units_in_use.max())}",
            "",
        ),
        (
            "KV キャッシュの使用率(使っている枠 / 割り当てた枠)",
            f"{s.memory['utilization']:.1%}",
            f"{c.memory['utilization']:.1%}",
            "",
        ),
    ]
    if c.kv_unit == "block":
        independent_rows.append(
            (
                "接頭辞の共有で再利用したブロック",
                str(s.prefix_hit_blocks),
                str(c.prefix_hit_blocks),
                "",
            )
        )
        independent_rows.append(
            ("書き込み時の複製(copy-on-write)", str(s.copy_on_write), str(c.copy_on_write), "")
        )

    st, ct = s.timing, c.timing
    timing_rows = [
        (
            "スループット(トークン/秒)",
            format_range(st["throughput"]),
            format_range(ct["throughput"]),
            ratio(ct["throughput"][0], st["throughput"][0]),
        ),
        (
            "全体の時間(秒)",
            format_range(st["makespan"], digits=3),
            format_range(ct["makespan"], digits=3),
            ratio(ct["makespan"][0], st["makespan"][0]),
        ),
        (
            "1 反復の時間(全体の時間 ÷ 反復の回数、ms)",
            format_range(st["iteration_seconds"], 1000, 2),
            format_range(ct["iteration_seconds"], 1000, 2),
            ratio(ct["iteration_seconds"][0], st["iteration_seconds"][0]),
        ),
        (
            "最初のトークンまでの時間(平均、ms)",
            format_range(st["time_to_first_token_mean"], 1000),
            format_range(ct["time_to_first_token_mean"], 1000),
            ratio(ct["time_to_first_token_mean"][0], st["time_to_first_token_mean"][0]),
        ),
        (
            "同(95 パーセンタイル、ms)",
            format_range(st["time_to_first_token_p95"], 1000),
            format_range(ct["time_to_first_token_p95"], 1000),
            ratio(ct["time_to_first_token_p95"][0], st["time_to_first_token_p95"][0]),
        ),
        (
            "出力トークンあたりの時間(平均、ms)",
            format_range(st["time_per_output_token_mean"], 1000),
            format_range(ct["time_per_output_token_mean"], 1000),
            ratio(ct["time_per_output_token_mean"][0], st["time_per_output_token_mean"][0]),
        ),
        (
            "待ち行列の待ち時間(平均、ms)",
            format_range(st["queue_wait_mean"], 1000),
            format_range(ct["queue_wait_mean"], 1000),
            "",
        ),
    ]

    sb, cb = s.time_breakdown, c.time_breakdown
    breakdown_rows = [
        ("forward(入力の配置とモデルの計算)", _share(sb, "forward"), _share(cb, "forward")),
        (
            "  うち入力の配置(添字の組み立てとデバイスへの転送)",
            _share(sb, "layout"),
            _share(cb, "layout"),
        ),
        (
            "反復の中の、forward 以外(スケジューラの判断・置き場所の確保と解放・記録)",
            _share(sb, "other"),
            _share(cb, "other"),
        ),
        ("反復と反復の間", _share(sb, "between"), _share(cb, "between")),
        (
            "待ち(何も実行する要求がなく、次の到着まで時計を進めた分)",
            _share(sb, "idle"),
            _share(cb, "idle"),
        ),
    ]
    fit_rows = _cost_fit_rows(s, c)

    iteration_ratio = s.iterations / c.iterations
    time_ratio = (st["makespan"][0] / s.iterations) / (ct["makespan"][0] / c.iterations)
    throughput_ratio = st["makespan"][0] / ct["makespan"][0]
    theory = outcome.theory_ratio
    identical = s.structure_identical and c.structure_identical
    load = (
        f"**負荷**: 要求 {settings.num_requests} 件、"
        f"出力長の平均 {outcome.output_mean_realized:.1f}・"
        f"変動係数 {outcome.output_cv_realized:.2f}(標本。指定は {settings.output_mean}・"
        f"{settings.output_cv:g})、最大バッチサイズ {settings.max_batch_size}、"
        f"KV キャッシュ {'PagedAttention' if settings.kv_manager == KV_PAGED else '最大長の確保'}"
        f"(ブロック {settings.block_size})、continuous の予算 {c.kv_capacity_units} {unit_ja}、"
        f"到着 {'Poisson 過程' if settings.arrival == ARRIVAL_POISSON else '一括'}。"
        f"実行したデバイス: {outcome.device}、関数全体の時間 {outcome.compute_seconds:.1f} 秒。"
    )
    independent_note = (
        f"{s.repeats} 回の実行で、反復の列(行数・待ち・KV キャッシュの使用量・待ちの反復数)は"
        + ("完全に一致しました。" if identical else "一致しませんでした。")
        + (
            ""
            if settings.arrival == ARRIVAL_OFFLINE
            else "Poisson 過程の到着は、到着の時刻が秒で決まり、1 反復の時間に依存するので、"
            "この表の値は機器によらない量にならず、実行ごとに変わりえます。"
        )
    )
    comparison = (
        f"**理論との比較**: 反復の回数の比(static / continuous)は {iteration_ratio:.2f}、"
        "出力長の分布から求めた理論値 E[max O] / E[O]"
        f"(最大バッチサイズ {settings.max_batch_size})は "
        f"{theory:.2f}(実測 ÷ 理論 = {ratio_to_theory(iteration_ratio, theory):.2f})。"
        "理論値は、continuous が行を隙間なく詰め続ける理想の上限なので、有限の負荷の実測はこれを"
        "下回りやすくなります(027 の 7.9 節)。"
    )
    decomposition = (
        f"**スループットの比の分解**(中央値の全体の時間から): continuous / static = "
        f"{throughput_ratio:.2f} = 反復の回数の比の逆数 {1 / iteration_ratio:.2f} × "
        f"1 反復の時間の比の逆数 {time_ratio:.2f}"
        "(反復の回数が実行ごとに同じとき、この式は厳密に成り立ちます)。"
        "1 反復の時間が 2 つの方式で違うと、反復の回数が減っても、スループットは上がりません。"
        "1 反復の時間は、機器やモデルの大きさで決まる量です。"
    )
    return paragraphs(
        load,
        "### 機器によらない量(スケジューラとメモリ管理だけで決まる)",
        independent_note,
        _table(("指標", "static", "continuous", "比(continuous / static)"), independent_rows),
        f"### 実時間の指標(機器に依存。同じ負荷を {s.repeats} 回実行した、中央値(最小〜最大))",
        _table(("指標", "static", "continuous", "比(中央値どうし)"), timing_rows),
        comparison,
        decomposition,
        "### 1 反復の時間の内訳(全体の時間が中央値の実行)",
        _table(("全体の時間に占める割合", "static", "continuous"), breakdown_rows),
        "1 反復の時間が、バッチの行数にどれだけ依存するかを見るために、反復ごとの"
        "(行数、処理した問い合わせのトークン数、forward の時間)に、"
        "「固定の項 + 1 行あたりの項 × 行数 + 追加の 1 トークンあたりの項 × "
        "(問い合わせのトークン数 - 行数)」の式を当てはめます(027 の費用モデル)。"
        "固定の項の割合が 1 に近いほど、行数にほとんど依存せず、"
        "固定費(Python とカーネルの起動)が支配的です。",
        _table(("式の当てはめ", "static", "continuous"), fit_rows),
    )


def _mean_or_dash(values) -> str:
    finite = [v for v in values if v == v]
    return f"{sum(finite) / len(finite):.2f}" if finite else "-"


def _cost_fit_rows(s: SchedulerRun, c: SchedulerRun) -> list[tuple[str, str, str]]:
    def cell(run: SchedulerRun, key: str, scale: float, fmt: str) -> str:
        return "-" if not run.cost_fit else format(run.cost_fit[key] * scale, fmt)

    def rows_range(run: SchedulerRun) -> str:
        return f"{int(run.computed_rows.min())}〜{int(run.computed_rows.max())}"

    return [
        ("固定の項(ms)", cell(s, "fixed", 1000, ".2f"), cell(c, "fixed", 1000, ".2f")),
        (
            "1 行あたりの項(ms / 行)",
            cell(s, "per_row", 1000, ".3f"),
            cell(c, "per_row", 1000, ".3f"),
        ),
        (
            "追加の 1 トークンあたりの項(ms)",
            cell(s, "per_extra_token", 1000, ".4f"),
            cell(c, "per_extra_token", 1000, ".4f"),
        ),
        (
            "forward の時間の平均(ms)",
            cell(s, "mean_forward_seconds", 1000, ".2f"),
            cell(c, "mean_forward_seconds", 1000, ".2f"),
        ),
        (
            "固定の項の割合(固定の項 ÷ forward の時間の平均)",
            cell(s, "fixed_share", 1, ".1%"),
            cell(c, "fixed_share", 1, ".1%"),
        ),
        ("決定係数", cell(s, "r2", 1, ".3f"), cell(c, "r2", 1, ".3f")),
        ("計算した行数の範囲(反復ごと)", rows_range(s), rows_range(c)),
    ]


def format_sweep(outcome: SweepOutcome) -> str:
    """観察 D の結果の表と読み方。"""
    header = [
        "| 最大バッチサイズ | スループット(トークン/秒) | 最初のトークンまで 平均(ms) |"
        " 同 95%(ms) |"
        " 出力トークンあたり(ms) | 待ち行列の待ち 平均(ms) | 実行中の要求数(平均) | 1 反復(ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, size in enumerate(outcome.batch_sizes):
        header.append(
            f"| {int(size)} | {outcome.throughput[i]:,.1f} |"
            f" {format_milliseconds(outcome.time_to_first_token_mean[i])} |"
            f" {format_milliseconds(outcome.time_to_first_token_p95[i])} |"
            f" {format_milliseconds(outcome.time_per_output_token_mean[i])} |"
            f" {format_milliseconds(outcome.queue_wait_mean[i])} |"
            f" {outcome.mean_active_requests[i]:.2f} |"
            f" {format_milliseconds(outcome.mean_iteration_seconds[i])} |"
        )
    settings = outcome.settings
    top, bottom = outcome.throughput.argmax(), outcome.throughput.argmin()
    tpot = outcome.time_per_output_token_mean
    growth = tpot[-1] / tpot[0] if tpot[0] > 0 else float("nan")
    reading = (
        f"**基準の到着率**: オフラインの処理能力(最大バッチサイズ 16)は "
        f"{outcome.reference_throughput:,.1f} トークン/秒で、出力長の平均 {settings.output_mean} で"
        f"割った {outcome.reference_rate:.2f} 要求/秒。到着の速さ(比){settings.arrival_ratio:g} は "
        f"{settings.arrival_ratio * outcome.reference_rate:.2f} 要求/秒。"
        f"実行したデバイス: {outcome.device}、関数全体の時間 {outcome.compute_seconds:.1f} 秒。"
    )
    observation = (
        f"**読み方**: スループットが最大なのは最大バッチサイズ {int(outcome.batch_sizes[top])}"
        f"({outcome.throughput[top]:,.1f} トークン/秒)、最小は {int(outcome.batch_sizes[bottom])}"
        f"({outcome.throughput[bottom]:,.1f})。出力トークンあたりの時間は、最大バッチサイズ "
        f"{int(outcome.batch_sizes[0])} の {format_milliseconds(tpot[0])} ms から "
        f"{int(outcome.batch_sizes[-1])} の {format_milliseconds(tpot[-1])} ms へ {growth:.2f} 倍。"
        "倍率が 1 に近いほど、最大バッチサイズを増やしても出力トークンあたりの時間が伸びず、"
        "スループットと待ち時間の交換が起きていません(027 の T4 の観察 D では 1.05〜1.09 倍)。"
    )
    return paragraphs(reading, "\n".join(header), observation)


def build_batch_settings(
    model_key: str,
    num_requests: float,
    output_mean: float,
    output_cv: float,
    arrival: str,
    arrival_rate: float,
    max_batch_size: float,
    kv_manager: str,
    block_size: float,
    kv_budget_slots: float,
    shared_prefix: bool,
    seed: float | None,
) -> BatchSettings:
    """画面の値から設定を作る(スライダーは float で届くので、整数の設定は int にそろえる)。"""
    return BatchSettings(
        model_key=model_key,
        num_requests=int(num_requests),
        output_mean=int(output_mean),
        output_cv=float(output_cv),
        arrival=arrival,
        arrival_rate=float(arrival_rate),
        max_batch_size=int(max_batch_size),
        kv_manager=kv_manager,
        block_size=int(block_size),
        kv_budget_slots=int(kv_budget_slots),
        shared_prefix=bool(shared_prefix),
        seed=0 if seed is None else int(seed),
    )


def build_sweep_settings(
    model_key: str,
    num_requests: float,
    output_mean: float,
    output_cv: float,
    arrival_ratio: float,
    seed: float | None,
) -> SweepSettings:
    return SweepSettings(
        model_key=model_key,
        num_requests=int(num_requests),
        output_mean=int(output_mean),
        output_cv=float(output_cv),
        arrival_ratio=float(arrival_ratio),
        seed=0 if seed is None else int(seed),
    )


def describe_load(max_context: int) -> str:
    """出力長の分布の説明(画面の補足)。"""
    distribution = output_distribution(24, 0.5, output_maximum(False))
    return (
        f"出力長は、範囲 1〜{output_maximum(False)}(共通の system prompt を付けると "
        f"1〜{output_maximum(True)})に切り詰めたガンマ分布の形で、平均と変動係数が指定の値に"
        "一致するように形状を解いています"
        f"(例: 平均 24・変動係数 0.5 は形状 {distribution.shape:.2f})。"
        f"書き出しは 16〜48 トークン(system prompt あり: 64 + 8〜24)。書き出し + 出力は、"
        f"モデルの最大の文脈長 {max_context} に収まります。"
    )


def build_page(resources: Resources) -> None:
    """バッチ推論のページを、``gr.Tab`` の中に組み立てる。

    画面のイベントの関数(``run_batch_inference``・``measure_batch_sizes``)は、GPU を使わずに
    判定できる入力の検査(反復の回数の見積もりを含む)を先に行い、誤りなら ``gr.Error`` にする。
    検査を通った入力だけを、``@gpu`` を付けた関数(``*_on_gpu``。計算だけを行う)に渡す。
    """
    main = resources.models["main"]

    @gpu(duration=BATCH_GPU_SECONDS)
    def run_batch_inference_on_gpu(settings: BatchSettings) -> BatchOutcome:
        """検査済みの設定で、同じ負荷を static と continuous の両方で実行する。"""
        return run_batching_comparison(
            resources.models[settings.model_key], resources.tokenizer, resources.device, settings
        )

    @gpu(duration=SWEEP_GPU_SECONDS)
    def measure_batch_sizes_on_gpu(settings: SweepSettings) -> SweepOutcome:
        """検査済みの設定で、最大バッチサイズを振って測る(観察 D)。"""
        return run_batch_size_sweep(
            resources.models[settings.model_key], resources.tokenizer, resources.device, settings
        )

    def run_batch_inference(
        model_key: str,
        num_requests: float,
        output_mean: float,
        output_cv: float,
        arrival: str,
        arrival_rate: float,
        max_batch_size: float,
        kv_manager: str,
        block_size: float,
        kv_budget_slots: float,
        shared_prefix: bool,
        seed: float | None,
    ) -> tuple[str, Figure, Figure]:
        """同じ負荷を static batching と continuous batching で実行して、結果を表示する。"""
        settings = build_batch_settings(
            model_key,
            num_requests,
            output_mean,
            output_cv,
            arrival,
            arrival_rate,
            max_batch_size,
            kv_manager,
            block_size,
            kv_budget_slots,
            shared_prefix,
            seed,
        )
        try:
            check_batch_request(resources.models[settings.model_key], resources.tokenizer, settings)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        outcome = run_batch_inference_on_gpu(settings)
        return format_summary(outcome), timeline_figure(outcome), memory_figure(outcome)

    def measure_batch_sizes(
        model_key: str,
        num_requests: float,
        output_mean: float,
        output_cv: float,
        arrival_ratio: float,
        seed: float | None,
    ) -> tuple[str, Figure]:
        """最大バッチサイズを振って、スループットと待ち時間の関係を表示する(観察 D)。"""
        settings = build_sweep_settings(
            model_key, num_requests, output_mean, output_cv, arrival_ratio, seed
        )
        try:
            check_sweep_request(resources.models[settings.model_key], resources.tokenizer, settings)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        outcome = measure_batch_sizes_on_gpu(settings)
        return format_sweep(outcome), sweep_figure(outcome)

    gr.Markdown(BATCH_MARKDOWN)
    with gr.Row():
        with gr.Column(scale=2):
            model_key = gr.Radio(
                choices=MODEL_CHOICES, value="main", label="モデル / Model (Hub のブランチ)"
            )
            num_requests = gr.Slider(
                4, MAX_REQUESTS, value=32, step=1, label="要求の数 / Number of requests"
            )
            output_mean = gr.Slider(
                4, MAX_OUTPUT_MEAN, value=24, step=1, label="出力長の平均 / Mean output length"
            )
            output_cv = gr.Slider(
                0.1,
                1.0,
                value=0.5,
                step=0.05,
                label="出力長の変動係数 / CV of output length",
                info="大きいほど static batching が不利になります。",
            )
            arrival = gr.Radio(
                choices=ARRIVAL_CHOICES, value=ARRIVAL_OFFLINE, label="到着 / Arrival"
            )
            arrival_rate = gr.Slider(
                1,
                MAX_ARRIVAL_RATE,
                value=30,
                step=1,
                label="到着率 [要求/秒](Poisson 過程のとき)",
            )
            max_batch_size = gr.Slider(
                1,
                MAX_BATCH_SIZE,
                value=16,
                step=1,
                label="最大バッチサイズ / Max batch size",
                info="既定の 16 は 027 の 7.10 節による。",
            )
            kv_manager = gr.Radio(
                choices=KV_CHOICES, value=KV_PAGED, label="KV キャッシュの管理 / KV cache manager"
            )
            block_size = gr.Dropdown(
                choices=list(BLOCK_SIZES), value=16, label="ブロックの大きさ(PagedAttention のとき)"
            )
            kv_budget_slots = gr.Slider(
                1,
                MAX_BUDGET_SLOTS,
                value=16,
                step=1,
                label="KV キャッシュの予算 / KV budget(最大長 256 トークンの要求の何件分か)",
                info="continuous batching の実行にだけ掛かります。最大バッチサイズより小さいと、"
                "メモリ不足(退避)が起きます。",
            )
            shared_prefix = gr.Checkbox(
                value=False,
                label="全要求が同じ system prompt(64 トークン)を持つ",
                info="PagedAttention では、接頭辞の共有でブロックが減ります。",
            )
            seed = gr.Number(value=0, precision=0, minimum=0, label="負荷のシード / Seed")
        with gr.Column(scale=3):
            run_button = gr.Button("実行する / Run both schedulers", variant="primary")
            gr.Markdown(TIMING_NOTE)
            summary = gr.Markdown()
            timeline_plot = gr.Plot(label="要求ごとの時間の帯 / Per-request timeline")
            memory_plot = gr.Plot(label="KV キャッシュの使用量 / KV cache usage")
            gr.Markdown(describe_load(main.max_sequence_length))
            gr.Markdown(LIMIT_MARKDOWN)
    run_button.click(
        run_batch_inference,
        inputs=[
            model_key,
            num_requests,
            output_mean,
            output_cv,
            arrival,
            arrival_rate,
            max_batch_size,
            kv_manager,
            block_size,
            kv_budget_slots,
            shared_prefix,
            seed,
        ],
        outputs=[summary, timeline_plot, memory_plot],
        api_name="run_batch_inference",
    )

    with gr.Accordion("最大バッチサイズを振って測る(観察 D) / Max Batch Size Sweep", open=False):
        gr.Markdown(SWEEP_MARKDOWN)
        with gr.Row():
            sweep_model = gr.Radio(choices=MODEL_CHOICES, value="main", label="モデル / Model")
            sweep_requests = gr.Slider(8, MAX_REQUESTS, value=32, step=1, label="要求の数")
        with gr.Row():
            sweep_mean = gr.Slider(4, MAX_OUTPUT_MEAN, value=24, step=1, label="出力長の平均")
            sweep_cv = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="出力長の変動係数")
        with gr.Row():
            sweep_ratio = gr.Radio(
                choices=[(f"{r:g}", r) for r in SWEEP_ARRIVAL_RATIOS],
                value=SWEEP_ARRIVAL_RATIOS[-1],
                label="到着の速さ(基準の到着率に対する比)",
            )
            sweep_seed = gr.Number(value=0, precision=0, minimum=0, label="負荷のシード")
        sweep_button = gr.Button("測る / Measure", variant="primary")
        sweep_summary = gr.Markdown()
        sweep_plot = gr.Plot(label="最大バッチサイズと、スループット・待ち時間 / Sweep")
        sweep_button.click(
            measure_batch_sizes,
            inputs=[sweep_model, sweep_requests, sweep_mean, sweep_cv, sweep_ratio, sweep_seed],
            outputs=[sweep_summary, sweep_plot],
            api_name="measure_batch_sizes",
        )


PAGE = Page(PAGE_TITLE, build_page)
