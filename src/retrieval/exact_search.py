"""全探索(exhaustive search)と、探索の費用の数え方(025)。

**全探索の計算量**: ``N`` 個のベクトル(次元 ``d``)から query に近いものを探すとき、全探索は query
ごとに ``N`` 回の距離計算(内積)を行うので、計算量は ``O(N d)`` である。

**単位ベクトルでの 3 つの尺度の一致**: ``x`` と ``y`` が単位ベクトル(``||x|| = ||y|| = 1``)のとき

.. math::

    \\lVert x - y \\rVert^2 = \\lVert x \\rVert^2 - 2 x^\\top y + \\lVert y \\rVert^2 = 2 - 2
    x^\\top y

なので、コサイン類似度 ``x^T y``・内積・二乗ユークリッド距離は、
どれも順位が一致する(類似度が大きいほど距離が小さい)。このモジュールの探索は、
内積の大きい順を「近い順」とする。

**費用の数え方**(``CostCounter``): 索引ごとに、距離計算の回数を **種類別に** 数える。
1 回の距離計算とは、query と 1 本のベクトル(全次元)の内積(または距離)を 1 つ求めることである。
壁時計時間はハードウェアに依存するので、費用の判定には使わない。

- 全探索: ``N``。
- 階層的な近傍グラフ(HNSW): 探索中に距離を求めたベクトルの数(``src/retrieval/hnsw.py``)。
- 転置ファイル: 重心との比較の回数と、走査したベクトルの数(``src/retrieval/inverted_file.py``)。
- Product Quantization: 参照表の作成(部分ベクトルと符号語の距離)と表引きの回
  数(``src/retrieval/product_quantization.py``)。

記号 / Notation:
    N : 索引のベクトルの数
    d : ベクトルの次元
    Q : query の数
    k : 返す近傍の数
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor


class CostCounter:
    """距離計算の回数を、名前をつけて数えるカウンタ。

    索引は種類の異なる計算(重心との比較・走査したベクトル・参照表の作成・表引きなど)を行うので、
    名前ごとに別々に数える。``snapshot()`` で現在の値の辞書を返し、``reset()`` で 0 に戻す。
    """

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._counts: dict[str, int] = {name: 0 for name in names}

    def add(self, name: str, amount: int) -> None:
        self._counts[name] = self._counts.get(name, 0) + int(amount)

    def get(self, name: str) -> int:
        return self._counts.get(name, 0)

    def reset(self) -> None:
        for name in self._counts:
            self._counts[name] = 0

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)


def exact_search(
    queries: Tensor,
    vectors: Tensor,
    k: int,
    excluded: np.ndarray | None = None,
    chunk_size: int = 512,
    counter: CostCounter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """内積による全探索で、各 query の上位 ``k`` 件(内積の大きい順)を返す。

    Args:
        queries: 形状 ``(Q, d)``(単位ベクトル)。
        vectors: 形状 ``(N, d)``(単位ベクトル)。
        k: 返す件数(``N`` 以下)。
        excluded: 形状 ``(Q,)`` の整数。各 query の候補から除くベクトルの添字(query の元になった
            passage)。負の値は「除かない」。``None`` なら除かない。
        chunk_size: 一度に処理する query の数(結果に影響しない)。
        counter: 与えると、``"exact"`` に距離計算の回数(``Q x N``。除いたベクトルも内積は計算する
            ので数える)を加える。

    Returns:
        ``(上位 k 件の添字 (Q, k)、その内積 (Q, k))``。
    """
    num_queries, num_vectors = queries.size(0), vectors.size(0)
    if not 1 <= k <= num_vectors:
        raise ValueError(f"k は 1 以上 N 以下: k={k}, N={num_vectors}")
    ids = np.empty((num_queries, k), dtype=np.int64)
    scores = np.empty((num_queries, k), dtype=np.float32)
    transposed = vectors.t().contiguous()
    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        similarity = queries[start:end] @ transposed
        if excluded is not None:
            rows = np.nonzero(excluded[start:end] >= 0)[0]
            if len(rows):
                columns = torch.as_tensor(excluded[start:end][rows], device=similarity.device)
                similarity[torch.as_tensor(rows, device=similarity.device), columns] = float("-inf")
        top = similarity.topk(k, dim=1)
        ids[start:end] = top.indices.cpu().numpy()
        scores[start:end] = top.values.cpu().numpy()
    if counter is not None:
        counter.add("exact", num_queries * num_vectors)
    return ids, scores


def squared_distance_from_similarity(similarity: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """単位ベクトルの内積から二乗ユークリッド距離 ``2 - 2 x^T y`` を求める。"""
    return 2.0 - 2.0 * similarity


def recall_against_ground_truth(returned: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """各 query の、返した近傍と正解(全探索の上位 ``k`` 件)の一致率(``|返した ∩ 正解| / k``)。

    Args:
        returned: 形状 ``(Q, k)``。返した近傍の添字(``-1`` は「返せなかった」)。
        ground_truth: 形状 ``(Q, k)``。全探索の上位 ``k`` 件の添字。

    Returns:
        形状 ``(Q,)`` の float64。各 query の recall(0 以上 1 以下)。
    """
    if returned.shape != ground_truth.shape:
        raise ValueError("returned と ground_truth の形状が一致しない")
    k = ground_truth.shape[1]
    hits = (returned[:, :, None] == ground_truth[:, None, :]).any(axis=2) & (returned >= 0)
    return hits.sum(axis=1).astype(np.float64) / k


def remove_excluded_and_truncate(
    candidates: np.ndarray, excluded: np.ndarray | None, k: int
) -> np.ndarray:
    """近似探索の結果(``k + 1`` 件以上)から、除くベクトルを取り除いて先頭 ``k`` 件にする。

    近似探索は query の元になった passage を候補から除けないので、``k + 1`` 件を取得して、その
    passage が含まれていれば取り除く(含まれていなければ、先頭の ``k`` 件をそのまま使う)。
    全探索の``exact_search(..., excluded=...)`` と同じ候補の集合に揃えるための処理である。

    Args:
        candidates: 形状 ``(Q, k + 1)``(近い順。足りない場合は ``-1`` で埋める)。
        excluded: 形状 ``(Q,)``。除くベクトルの添字(負の値は「除かない」)。``None`` なら除かない。
        k: 返す件数。
    """
    result = np.full((candidates.shape[0], k), -1, dtype=np.int64)
    for row in range(candidates.shape[0]):
        values = candidates[row]
        if excluded is not None and excluded[row] >= 0:
            values = values[values != excluded[row]]
        result[row, : min(k, len(values))] = values[:k]
    return result
