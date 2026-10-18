"""合成の負荷(workload)の生成。

負荷は、要求の出力長・書き出しの長さ・到着時刻を、乱数のシードから決定的に作る。

**出力長の分布**: 平均と変動係数(coefficient of variation)を指定でき、整数の範囲
``[minimum, maximum]`` に切り詰めたガンマ分布の形(``p(x) ∝ x^(k-1) exp(-x/s)``、整数 x)を
使う。形状 ``k`` と尺度 ``s`` は、切り詰めと離散化の後の確率質量関数が指定の平均と変動係数に
一致するように数値で解く(切り詰めると平均・変動係数が元の分布からずれるため、ずれを直さずに
指定することはしない)。出力長の上限は、モデルの最大の文脈長から書き出しの最大の長さを引いた値以下に
する(呼び出し側で保証する)。

**到着過程**: 一括処理(offline)は全要求が時刻 0 に到着する。オンラインは到着率 λ の Poisson
過程(到着の間隔が平均 1/λ の指数分布)。

導入: 027。

記号 / Notation:
    O     : 出力長
    m     : 出力長の平均 E[O]
    CV    : 出力長の変動係数(標準偏差 / 平均)
    B     : 最大バッチサイズ
    λ     : 到着率(1 秒あたりの要求の数)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.serving.requests import Request


@dataclass(frozen=True)
class OutputLengthDistribution:
    """整数の範囲に切り詰めた出力長の分布(確率質量関数)。

    Attributes:
        support: 取りうる出力長(昇順の整数)。
        probabilities: ``support`` の各値の確率(総和 1)。
        shape: 解いた形状 k。
        scale: 解いた尺度 s。
    """

    support: np.ndarray
    probabilities: np.ndarray
    shape: float
    scale: float

    @property
    def mean(self) -> float:
        return float((self.support * self.probabilities).sum())

    @property
    def standard_deviation(self) -> float:
        mean = self.mean
        return math.sqrt(float((((self.support - mean) ** 2) * self.probabilities).sum()))

    @property
    def coefficient_of_variation(self) -> float:
        return self.standard_deviation / self.mean

    @property
    def minimum(self) -> int:
        return int(self.support[0])

    @property
    def maximum(self) -> int:
        return int(self.support[-1])


def _kernel_probabilities(support: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """``x^(shape-1) exp(-x/scale)`` を整数の台で正規化する(対数の領域で計算する)。"""
    log_weights = (shape - 1.0) * np.log(support) - support / scale
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


def _moments(support: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    mean = float((support * probabilities).sum())
    variance = float((((support - mean) ** 2) * probabilities).sum())
    return mean, math.sqrt(variance) / mean


def _solve_scale_for_mean(support: np.ndarray, shape: float, target_mean: float) -> float:
    """形状を固定したとき、平均が ``target_mean`` になる尺度を二分法で求める(尺度について単調)。"""
    low, high = math.log(1e-3), math.log(1e7)
    for _ in range(200):
        middle = 0.5 * (low + high)
        mean, _ = _moments(support, _kernel_probabilities(support, shape, math.exp(middle)))
        if mean < target_mean:
            low = middle
        else:
            high = middle
    return math.exp(0.5 * (low + high))


def maximum_coefficient_of_variation(mean: float, minimum: int, maximum: int) -> float:
    """範囲 ``[minimum, maximum]`` と平均 ``mean`` で取りうる変動係数の上限。

    二点分布(``minimum`` と ``maximum`` だけに確率を置く)のとき標準偏差が最大になり、その値は
    ``sqrt((mean - minimum) * (maximum - mean))``。
    """
    return math.sqrt((mean - minimum) * (maximum - mean)) / mean


def fit_output_length_distribution(
    mean: float, coefficient_of_variation: float, minimum: int, maximum: int
) -> OutputLengthDistribution:
    """平均と変動係数が指定の値に一致する、切り詰めたガンマ分布の形の出力長の分布を作る。

    Args:
        mean: 出力長の平均 m。``minimum < mean < maximum``。
        coefficient_of_variation: 出力長の変動係数 CV(正)。
        minimum: 出力長の下限(1 以上)。
        maximum: 出力長の上限。

    Returns:
        平均と変動係数が(数値の精度で)指定の値に一致する分布。

    Raises:
        ValueError: 範囲と平均から取りうる変動係数の上限を超える場合、または解けない場合。
    """
    if not (1 <= minimum < mean < maximum):
        raise ValueError(
            f"1 <= minimum < mean < maximum である必要がある: {minimum}, {mean}, {maximum}"
        )
    if coefficient_of_variation <= 0:
        raise ValueError(f"変動係数は正である必要がある: {coefficient_of_variation}")
    limit = maximum_coefficient_of_variation(mean, minimum, maximum)
    if coefficient_of_variation >= limit:
        raise ValueError(
            f"変動係数 {coefficient_of_variation} が、この範囲・平均で取りうる上限 {limit:.4f} 以上"
        )

    support = np.arange(minimum, maximum + 1, dtype=np.int64)
    support_float = support.astype(np.float64)

    def coefficient_at(shape: float) -> tuple[float, float]:
        scale = _solve_scale_for_mean(support_float, shape, mean)
        _, variation = _moments(support_float, _kernel_probabilities(support_float, shape, scale))
        return variation, scale

    # 平均を固定すると、変動係数は形状について単調に減少する。
    low, high = math.log(0.02), math.log(4000.0)
    variation_at_low, _ = coefficient_at(math.exp(low))
    variation_at_high, _ = coefficient_at(math.exp(high))
    if not (variation_at_high <= coefficient_of_variation <= variation_at_low):
        raise ValueError(
            f"変動係数 {coefficient_of_variation} は、"
            f"探索範囲 [{variation_at_high:.4f}, {variation_at_low:.4f}] の外"
        )
    for _ in range(100):
        middle = 0.5 * (low + high)
        variation, _ = coefficient_at(math.exp(middle))
        if variation > coefficient_of_variation:
            low = middle
        else:
            high = middle
    shape = math.exp(0.5 * (low + high))
    _, scale = coefficient_at(shape)
    probabilities = _kernel_probabilities(support_float, shape, scale)
    return OutputLengthDistribution(support, probabilities, shape, scale)


def expected_maximum(distribution: OutputLengthDistribution, count: int) -> float:
    """分布から独立に ``count`` 個を取ったときの最大値の期待値 E[max]。

    ``E[max] = Σ_x x (F(x)^count - F(x-1)^count)``(F は累積分布関数)。
    """
    cdf = np.cumsum(distribution.probabilities)
    previous = np.concatenate([[0.0], cdf[:-1]])
    return float((distribution.support * (cdf**count - previous**count)).sum())


def predicted_static_to_continuous_ratio(
    distribution: OutputLengthDistribution, batch_size: int
) -> float:
    """static batching と continuous batching の反復の回数の比の理論値 ``E[max_B O] / E[O]``。

    static batching は、バッチの中の最も長い要求が終わるまで組を入れ替えないので、バッチあたりの
    反復は max(O) で、continuous batching が(入れ替えで隙間なく詰められる理想的な場合に)使う
    反復は 1 要求あたり平均 O / B。両者の反復数の比が E[max_B O] / E[O] になる。
    """
    return expected_maximum(distribution, batch_size) / distribution.mean


def sample_output_lengths(
    distribution: OutputLengthDistribution, count: int, rng: np.random.Generator
) -> np.ndarray:
    """出力長を ``count`` 個、独立に標本抽出する。"""
    return rng.choice(distribution.support, size=count, p=distribution.probabilities)


def sample_prompt_lengths(low: int, high: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """書き出しの長さを ``[low, high]`` の一様な整数から ``count`` 個標本抽出する。"""
    return rng.integers(low, high + 1, size=count)


def poisson_arrival_times(rate: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """到着率 ``rate`` の Poisson 過程の到着時刻(昇順)を ``count`` 個作る。"""
    if rate <= 0:
        raise ValueError(f"到着率は正である必要がある: {rate}")
    return np.cumsum(rng.exponential(1.0 / rate, size=count))


def moment_sampling_distribution(
    distribution: OutputLengthDistribution,
    count: int,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``count`` 個の標本の平均と変動係数が、標本抽出でどれだけばらつくかを乱数で見積もる。

    負荷の実現(標本の平均と変動係数が設計どおりか)の許容幅を、分布から導くために使う。

    Returns:
        ``(標本の平均, 標本の変動係数)``。それぞれ長さ ``repetitions`` の配列。
    """
    samples = rng.choice(
        distribution.support, size=(repetitions, count), p=distribution.probabilities
    ).astype(np.float64)
    means = samples.mean(axis=1)
    variations = samples.std(axis=1) / means
    return means, variations


def build_requests(
    prompt_pool: Sequence[Sequence[int]],
    prompt_lengths: Sequence[int],
    output_lengths: Sequence[int],
    arrival_times: Sequence[float],
    rng: np.random.Generator,
    shared_prefix: Sequence[int] = (),
) -> list[Request]:
    """書き出しの候補(``prompt_pool``)から要求を組み立てる。

    各要求の書き出しは、``shared_prefix``(全要求に共通の接頭辞。system prompt を想定)の後ろに、
    ``prompt_pool`` から選んだ 1 本の先頭 ``prompt_lengths[i]`` トークンをつなげたもの。
    ``prompt_pool`` の各要素は、``max(prompt_lengths)`` 以上の長さを持つこと。

    Args:
        prompt_pool: 書き出しの元になるトークン列の候補(コーパスの記事の先頭など)。
        prompt_lengths: 要求ごとの書き出しの長さ(接頭辞を除く部分)。
        output_lengths: 要求ごとの出力長。
        arrival_times: 要求ごとの到着時刻。
        rng: 候補の選択に使う乱数生成器。
        shared_prefix: 全要求に共通の接頭辞。

    Returns:
        ``request_id`` が 0 から連番の要求のリスト(到着時刻の昇順に並べる)。
    """
    if not (len(prompt_lengths) == len(output_lengths) == len(arrival_times)):
        raise ValueError("書き出しの長さ・出力長・到着時刻の数が一致しない")
    need = max(prompt_lengths) if len(prompt_lengths) else 0
    short = [i for i, tokens in enumerate(prompt_pool) if len(tokens) < need]
    if short:
        raise ValueError(f"書き出しの候補 {short[:3]} などが、必要な長さ {need} に足りない")
    picks = rng.integers(0, len(prompt_pool), size=len(prompt_lengths))
    order = np.argsort(np.asarray(arrival_times), kind="stable")
    requests = []
    for new_id, index in enumerate(order):
        tokens = tuple(int(t) for t in shared_prefix) + tuple(
            int(t) for t in prompt_pool[int(picks[index])][: int(prompt_lengths[index])]
        )
        requests.append(
            Request(
                request_id=new_id,
                prompt_token_ids=tokens,
                output_length=int(output_lengths[index]),
                arrival_time=float(arrival_times[index]),
            )
        )
    return requests
