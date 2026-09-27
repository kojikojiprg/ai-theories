"""best-of-n サンプリング(Best-of-n Sampling)の期待値の推定量と KL ダイバージェンスの上界(017)。

best-of-n サンプリングは、参照方策 pi_ref から応答を n 個独立に抽選し、代理報酬(報酬モデルの
スコア)が最大のものを選ぶ方策である。n を増やすほど代理報酬に対する最適化の圧力が強くなる。

**期待値の推定量**(Gao, Schulman & Hilton, ICML 2023 と同じ不偏推定量):
プロンプトごとに pi_ref から M 個の応答のプールを 1 度だけ抽選し、代理報酬の昇順に並べる
(i 番目の応答の代理報酬が小さい方から i 番目)。プールから n 個の部分集合を一様に選んだとき、
i 番目の応答が部分集合の最大になるのは、残りの n - 1 個をそれより下位の i - 1 個から選ぶ場合なので、
その確率は

.. math::

    w_i = \\binom{i - 1}{n - 1} \\Big/ \\binom{M}{n}

である(``best_of_n_order_weights()``)。任意の量 g(真の報酬など)の、選ばれた応答での期待値を
``sum_i w_i g_(i)`` で求める(``best_of_n_expected_value()``)。これは「M 個から n 個を選ぶ
すべての部分集合で最大の応答の g を平均した値」と等しく、M 個が pi_ref からの独立な抽選なら、
n 個を独立に抽選する best-of-n の期待値の不偏推定量(U 統計量)になる。部分集合を乱数で重複抽選して
平均するモンテカルロ推定と違い、抽選による誤差を持ち込まない。

**代理報酬の同点**: 同じ代理報酬の応答は、プール内の添字の小さい方を下位として並べる
(``np.argsort(kind="stable")``)。同じ応答の重複は代理報酬も真の報酬も等しいので、並べ方は結果に
影響しない。

**KL ダイバージェンス** ``best_of_n_kl_upper_bound()``: best-of-n の方策 pi_n と pi_ref の
KL ダイバージェンス(Kullback-Leibler divergence)について、よく使われる式
``log n - (n - 1) / n`` は、応答の空間が連続で同点(同じ応答の重複)が起きない場合の値であり、
一般には **上界** である(Beirami et al., arXiv 2024)。言語モデルの応答は離散で、同じ応答が
何度も抽選されうるので、実際の KL ダイバージェンスはこれより小さい。

記号 / Notation:
    M : プールの大きさ(プロンプトごとの応答の数)
    n : best-of-n の n(1 <= n <= M)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def best_of_n_order_weights(pool_size: int, n: int) -> np.ndarray:
    """昇順で i 番目(1-indexed)の応答が n 個の部分集合の最大になる確率 ``w_i``(長さ M)。

    ``w_i = C(i - 1, n - 1) / C(M, n)``。対数ガンマ関数で計算する(M が大きくても桁があふれない)。
    ``i < n`` では 0。和は 1 になる。
    """
    if not 1 <= n <= pool_size:
        raise ValueError(f"1 <= n <= M である必要がある: n={n}, M={pool_size}")
    ranks = np.arange(1, pool_size + 1)
    weights = np.zeros(pool_size, dtype=np.float64)
    valid = ranks >= n
    log_numerator = np.array(
        [math.lgamma(i) - math.lgamma(n) - math.lgamma(i - n + 1) if i >= n else 0.0 for i in ranks]
    )
    log_denominator = (
        math.lgamma(pool_size + 1) - math.lgamma(n + 1) - math.lgamma(pool_size - n + 1)
    )
    weights[valid] = np.exp(log_numerator[valid] - log_denominator)
    return weights


def best_of_n_expected_value(
    proxy_scores: Sequence[float], values: Sequence[float], n: int
) -> float:
    """代理報酬 ``proxy_scores`` で n 個の最大を選んだときの、``values`` の期待値(1 プロンプト分)。

    Args:
        proxy_scores: プールの応答ごとの代理報酬(長さ M)。
        values: 同じ応答ごとの、期待値をとる量(真の報酬など、長さ M)。
        n: best-of-n の n。
    """
    proxy_scores = np.asarray(proxy_scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if proxy_scores.shape != values.shape or proxy_scores.ndim != 1:
        raise ValueError("proxy_scores と values は同じ長さの 1 次元配列である必要がある")
    order = np.argsort(proxy_scores, kind="stable")
    return float(best_of_n_order_weights(len(values), n) @ values[order])


def best_of_n_expected_values(
    proxy_scores: np.ndarray, values: np.ndarray, ns: Sequence[int]
) -> np.ndarray:
    """複数のプロンプト・複数の n について ``best_of_n_expected_value()`` をまとめて計算する。

    Args:
        proxy_scores: 形状 ``(P, M)``(P はプロンプトの数)。
        values: 形状 ``(P, M)``。
        ns: n の水準。

    Returns:
        形状 ``(P, len(ns))``。
    """
    proxy_scores = np.asarray(proxy_scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if proxy_scores.shape != values.shape or proxy_scores.ndim != 2:
        raise ValueError("proxy_scores と values は同じ形状 (P, M) である必要がある")
    order = np.argsort(proxy_scores, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)  # (P, M) 昇順
    weights = np.stack([best_of_n_order_weights(values.shape[1], n) for n in ns])  # (L, M)
    return sorted_values @ weights.T


def best_of_n_kl_upper_bound(n: int) -> float:
    """best-of-n の KL ダイバージェンスの上界 ``log n - (n - 1) / n``(nats)。"""
    if n < 1:
        raise ValueError(f"n は 1 以上である必要がある: {n}")
    return math.log(n) - (n - 1) / n
