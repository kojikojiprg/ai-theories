"""recall と費用の曲線(025)。

近似最近傍探索の索引は、探索の幅(HNSW の ``ef``)や探索するリストの数(転置ファイルの
``p``)を増やすと、recall が上がり、1 query あたりの距離計算の回数(費用)も増える。
索引どうしを公平に比べるには、**同じ recall を達成するときの費用** で比べる(ANN-Benchmarks,
Aumüller et al., Information Systems 2020 の考え方)。

**目標の recall での費用**: 宣言した格子(探索の幅または探索するリストの数を小さい順に並べたもの)を
掃引し、各格子点での recall の平均 ``r_j`` と費用の平均 ``c_j``(いずれも query
について平均)を求める。``r_j`` が初めて目標 ``r*`` 以上になる格子点を ``j``(``j >= 1``、
すなわち最小の格子点では``r_0 < r*``)として、**費用の対数を recall について隣り合う
2 点の間で線形に補間する**。

.. math::

    \\log c^* = \\log c_{j-1} + \\frac{r^* - r_{j-1}}{r_j - r_{j-1}} \\left( \\log c_j - \\log
    c_{j-1} \\right)

格子の最小点で ``r_0 >= r*`` なら補間でなく外挿になるので、費用は定義しない(``NaN``)。
格子の最大点でも``r* `` に届かなければ、同様に定義しない。recall は ``r_j``
が必ずしも単調でないため、「初めて``r*`` 以上になる点」を使う(掃引は ``r_j`` が停止の閾値
``r_stop`` 以上になった格子点で止めてよい。止めた点より後の格子点は、目標の recall
での費用に影響しない)。

**ブートストラップ**: query を記事ごとのまとまりとして復元抽出するとき、各抽出の query
の重み(抽出回数)で``r_j``・``c_j`` を重み付き平均し、
補間をやり直す(``interpolate_log_cost_batch()``)。

記号 / Notation:
    r* : 目標の recall、r_j・c_j : 格子点 j での recall と費用の平均、Q : query の数、G : 格子点の数
"""

from __future__ import annotations

import numpy as np


def interpolate_log_cost(recalls: np.ndarray, costs: np.ndarray, target: float) -> float:
    """目標の recall での費用の対数(定義できなければ ``NaN``)。

    Args:
        recalls: 形状 ``(G,)``。格子の順(探索の幅などが小さい順)の recall の平均。
        costs: 形状 ``(G,)``。同じ順の費用の平均(正)。
        target: 目標の recall。
    """
    return float(interpolate_log_cost_batch(recalls[None, :], costs[None, :], target)[0])


def interpolate_log_cost_batch(recalls: np.ndarray, costs: np.ndarray, target: float) -> np.ndarray:
    """``interpolate_log_cost()`` を、形状 ``(R, G)``
    の複数の曲線(ブートストラップの再標本など)にまとめて行う。

        Returns:
            形状 ``(R,)``。補間できない曲線(最小の格子点で既に ``target`` 以上、
            または最大の格子点でも届かない)は ``NaN``。
    """
    recalls = np.asarray(recalls, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    reached = recalls >= target
    has_crossing = reached.any(axis=1)
    first = np.where(has_crossing, reached.argmax(axis=1), 0)
    valid = has_crossing & (first >= 1)
    rows = np.arange(recalls.shape[0])
    lower = np.maximum(first - 1, 0)
    r0, r1 = recalls[rows, lower], recalls[rows, first]
    log_c0, log_c1 = np.log(costs[rows, lower]), np.log(costs[rows, first])
    weight = (target - r0) / np.where(r1 > r0, r1 - r0, 1.0)
    result = log_c0 + weight * (log_c1 - log_c0)
    return np.where(valid, result, np.nan)


def weighted_curve(
    per_query_recall: np.ndarray, per_query_cost: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """query の重み(ブートストラップの抽出回数)で、格子点ごとの recall と費用を重み付き平均する。

    Args:
        per_query_recall: 形状 ``(G, Q)``。
        per_query_cost: 形状 ``(G, Q)``。
        weights: 形状 ``(R, Q)``。再標本ごとの query の重み(合計は正)。

    Returns:
        ``(recall の平均 (R, G)、費用の平均 (R, G))``。
    """
    total = weights.sum(axis=1, keepdims=True)
    recalls = (weights @ per_query_recall.T) / total
    costs = (weights @ per_query_cost.T) / total
    return recalls, costs


def fit_log_log_slope(x: np.ndarray, log_y: np.ndarray) -> np.ndarray:
    """``log y = log a + b log x`` の最小二乗の傾き ``b``(``x`` は共通、``log_y`` は ``(..., L)``)。

    ``x`` は ``L`` 個の水準(正)。複数の曲線(シード・再標本)の傾きを一度に求める。
    """
    log_x = np.log(np.asarray(x, dtype=np.float64))
    centered = log_x - log_x.mean()
    return (log_y - log_y.mean(axis=-1, keepdims=True)) @ centered / float((centered**2).sum())


def positions_in_subset(subset: np.ndarray, num_total: int) -> np.ndarray:
    """要素の番号 -> ``subset`` の中での位置(含まれなければ ``-1``)の対応 ``(num_total,)``。"""
    positions = np.full(num_total, -1, dtype=np.int64)
    positions[np.asarray(subset)] = np.arange(len(subset))
    return positions


def geometric_grid(start: int, ratio: float, count: int) -> list[int]:
    """``start`` から公比 ``ratio`` の等比の整数の格子(昇順、重複なし。候補は ``count`` 個)。"""
    values = sorted({int(round(start * ratio**j)) for j in range(count)})
    return values


def sweep_hnsw_search_width(
    index,
    queries: np.ndarray,
    excluded: np.ndarray,
    ground_truth: np.ndarray,
    grid: list[int],
    k: int = 10,
    stop_recall: float = 0.95,
    use_hierarchy: bool = True,
    entry_elements: np.ndarray | None = None,
) -> dict:
    """HNSW の探索の幅 ``ef`` の格子を小さい順に掃引し、query ごとの recall と費用を記録する。

    各格子点で ``k + 1`` 件を探索し、query の元の
    passage(``excluded``)が含まれていれば取り除いて先頭 ``k`` 件を返す(全探索の ``excluded``
    と同じ候補の集合)。recall の平均が ``stop_recall``以上になった格子点で掃引を止める。

    Args:
        index: ``HierarchicalNavigableSmallWorld``(掃引する部分集合を挿入済み)。
        queries: 形状 ``(Q, d)``。
        excluded: 形状 ``(Q,)``。候補から除く要素の番号(索引の部分集合に含まれないものは ``-1``)。
        ground_truth: 形状 ``(Q, k)``。全探索の上位 ``k`` 件の要素の番号。
        grid: 探索の幅の格子(昇順。最小は ``k + 1`` 以上)。
        k: 返す件数。
        stop_recall: 掃引を止める recall の平均。
        use_hierarchy, entry_elements: ``index.search_batch()`` に渡す(階層の診断用)。

    Returns:
        ``{"grid": 掃引した格子点, "recall": (G_used, Q), "cost": (G_used, Q)}``。
    """
    from src.retrieval.exact_search import recall_against_ground_truth, remove_excluded_and_truncate

    used, recalls, costs = [], [], []
    for width in grid:
        ids, cost = index.search_batch(
            queries, k + 1, width, use_hierarchy=use_hierarchy, entry_elements=entry_elements
        )
        returned = remove_excluded_and_truncate(ids, excluded, k)
        recall = recall_against_ground_truth(returned, ground_truth)
        used.append(width)
        recalls.append(recall)
        costs.append(cost.astype(np.float64))
        if recall.mean() >= stop_recall:
            break
    return {"grid": used, "recall": np.stack(recalls), "cost": np.stack(costs)}


def sweep_inverted_file_probes(
    index,
    queries,
    local_excluded: np.ndarray,
    subset: np.ndarray,
    ground_truth: np.ndarray,
    grid: list[int],
    k: int = 10,
    stop_recall: float = 0.95,
) -> dict:
    """転置ファイルの探索するリストの数 ``p`` の格子を掃引し、query ごとの recall と費用を記録する。

    全探索のときと同じく、query の元の passage は候補から除く(``local_excluded``)。``index``
    は``subset`` のベクトルだけを持つので、返る添字(``subset`` の中の位置)を ``subset``
    で要素の番号に戻して、正解と比べる。recall の平均が ``stop_recall``
    以上になった格子点より後ろは記録に含めない(``p`` が ``K``を超える格子点も除く)。

    Returns:
        ``{"grid": 記録した格子点, "recall": (G_used, Q), "cost": (G_used, Q)}``。
    """
    from src.retrieval.exact_search import recall_against_ground_truth

    probe_counts = [p for p in grid if p <= index.num_lists]
    if index.num_lists not in probe_counts:
        probe_counts.append(index.num_lists)
    results = index.search_batch_by_probe_counts(queries, k, probe_counts, excluded=local_excluded)
    used, recalls, costs = [], [], []
    for p in probe_counts:
        local_ids, cost = results[p]
        returned = np.where(local_ids >= 0, np.asarray(subset)[np.maximum(local_ids, 0)], -1)
        recall = recall_against_ground_truth(returned, ground_truth)
        used.append(p)
        recalls.append(recall)
        costs.append(cost.astype(np.float64))
        if recall.mean() >= stop_recall:
            break
    return {"grid": used, "recall": np.stack(recalls), "cost": np.stack(costs)}


def cluster_bootstrap_weights(
    clusters, num_resamples: int, seed: int, chunk_size: int = 500
) -> np.ndarray:
    """クラスタ(記事など)を復元抽出するブートストラップの、単位ごとの重み ``(R, n)``。

    反復ごとにクラスタを ``C`` 個(``C`` はクラスタの数)復元抽出し、各単位の重みを、
    その単位が属するクラスタの抽出回数とする(``paired_cluster_bootstrap_ratio_of_sums()``
    と同じ再標本)。同じ``seed`` で作った重みを、比べる全条件・全シードに使えば、
    対応付きの再標本になる。

    Args:
        clusters: 長さ ``n`` の、各単位が属するクラスタの識別子(ハッシュ可能な任意の値)。
        num_resamples: 反復回数 ``R``。
        seed: 乱数シード。
    """
    labels = {c: i for i, c in enumerate(dict.fromkeys(list(clusters)))}
    cluster_index = np.array([labels[c] for c in clusters])
    num_clusters = len(labels)
    rng = np.random.default_rng(seed)
    probabilities = np.full(num_clusters, 1.0 / num_clusters)
    weights = np.empty((num_resamples, len(cluster_index)), dtype=np.float64)
    for start in range(0, num_resamples, chunk_size):
        size = min(chunk_size, num_resamples - start)
        counts = rng.multinomial(num_clusters, probabilities, size=size).astype(np.float64)
        weights[start : start + size] = counts[:, cluster_index]
    return weights
