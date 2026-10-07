"""転置ファイル(Inverted File Index)のスクラッチ実装(025)。

**考え方**: 全ベクトルを k-means(``kmeans.py``)で ``K`` 個のクラスタ(Voronoi 領域)に分け、
クラスタごとに属するベクトルの一覧(転置リスト)を作る。query が来たら、重心との距離を ``K``
個すべて求め、近い順に``p`` 個のリスト(``p`` を探索するリストの数と呼ぶ)だけを走査して、
その中の上位を返す。query に近いベクトルは query に近い重心のリストに入っている可能性が高い、
という仮定に基づく近似である。リストを増やすほど recall が上がり、費用も増える。

**費用の式**(1 query あたりの距離計算の回数。ベクトルは ``N`` 個、リストは ``K`` 個、
各リストの大きさが平均``N / K``、探索するリストの数が ``p``):

.. math::

    C(K, p) = K + p \\cdot \\frac{N}{K}

第 1 項が重心との比較、第 2 項が走査するベクトルの数である(リストの大きさが等しいときの値。
実際の走査数はリストの大きさに依って変わるので、``search()`` は実際に数える)。

**リスト数を √N に比例させると費用が O(√N) になること**: ``p`` を固定して ``K`` で最小化する。

.. math::

    \\frac{\\partial C}{\\partial K} = 1 - \\frac{p N}{K^2} = 0 \\;\\Rightarrow\\; K = \\sqrt{p
    N}, \\qquad
    C = 2 \\sqrt{p N} = O(\\sqrt{N})

``K = c \\sqrt{N}``(``c`` は定数)とおけば ``C = (c + p / c) \\sqrt{N}`` で、``N`` に対して
``\\sqrt{N}`` で増える。
重心との比較(第 1 項)と走査(第 2 項)が同じ大きさになる ``K = \\sqrt{p N}`` が最小である。
この導出は、目標の
recall に必要な ``p`` が ``N`` によらず一定という仮定を置いている。実際には、``K``
が増えると各リストが小さく
なり、近いベクトルが別のリストに入る割合が増えるので、同じ recall に必要な ``p`` は ``N``
とデータに依存する
(必要な ``p`` が ``N`` とともに増えれば、費用は ``\\sqrt{N}`` より速く増える)。025 の実験 B は、
``K`` を ``\\sqrt{N}`` の 1/2 倍から 8 倍までの 5 点(公比 2)で試し、
目標の recall での費用が最小のものを採用する。

**単体テスト用の性質**: すべてのリストを探索する(``p = K``)と、
全探索と一致する(重心との比較の分の費用が余計にかかるだけ)。

**費用の数え方**(``CostCounter``): ``"centroid_comparisons"``(重心との比較)と
``"scanned_vectors"``(走査したベクトル)を別々に数える。

**実装の注意**: ``search()`` は、重心との距離を求め、選んだリストのベクトルだけを走査する。
評価を速くするために、``search_batch_by_probe_counts()`` は、
全ベクトルとの内積を一括で計算してから、探索するリストに入っていないベクトルを除く(結果と、
数える距離計算の回数は ``search()`` と一致する。025 の単体テストで確かめる)。
壁時計時間は判定に使わない。

記号 / Notation:
    N : ベクトルの数、K : リストの数(``num_lists``)、p : 探索するリストの数(``num_probed_lists``)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from src.retrieval.exact_search import CostCounter
from src.retrieval.kmeans import assign_to_nearest, fit_kmeans


class InvertedFileIndex:
    """転置ファイルの索引。

    Args:
        vectors: 形状 ``(N, d)``。単位ベクトル(CPU の float32)。
        num_lists: ``K``。
        num_iterations: k-means の反復回数。
        seed: k-means の初期化のシード。
    """

    def __init__(
        self, vectors: Tensor, num_lists: int, num_iterations: int = 20, seed: int = 0
    ) -> None:
        self.num_vectors, self.dimension = vectors.shape
        self.num_lists = num_lists
        result = fit_kmeans(vectors, num_lists, num_iterations, seed)
        self.centroids = result.centroids
        self.kmeans_inertia_history = result.inertia_history
        self.num_empty_cluster_events = result.num_empty_cluster_events
        assignments, _ = assign_to_nearest(vectors, self.centroids)
        self.assignments = assignments
        order = torch.argsort(assignments, stable=True)  # リストごとにベクトルを連続して並べる
        self.sorted_ids = order
        self.sorted_vectors = vectors[order].contiguous()
        self.list_sizes = torch.bincount(assignments, minlength=num_lists)
        self.offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(self.list_sizes, dim=0)]
        )
        self.counter = CostCounter(("centroid_comparisons", "scanned_vectors"))

    def _nearest_lists(self, query: Tensor) -> Tensor:
        """query に近い順(二乗距離の昇順。同点は番号の小さい順)のリストの番号 ``(K,)``。"""
        squared = (self.centroids - query[None, :]).pow(2).sum(dim=1)
        return torch.argsort(squared, stable=True)

    def search(
        self,
        query: Tensor,
        k: int,
        num_probed_lists: int,
        excluded: int = -1,
        count_cost: bool = True,
    ) -> tuple[np.ndarray, int, int]:
        """query に近い上位 ``k`` 件を、近い順の ``num_probed_lists`` 個のリストだけを走査して探す。

        Args:
            query: 形状 ``(d,)``(単位ベクトル)。
            k: 返す件数。
            num_probed_lists: ``p``(``K`` 以下)。
            excluded: 候補から除くベクトルの添字(負の値は除かない)。
            count_cost: 真なら ``counter`` に加える。

        Returns:
            上位 ``k`` 件のベクトルの添字(足りない場合は ``-1``)、重心との比較の回数、走査数。
        """
        if not 1 <= num_probed_lists <= self.num_lists:
            raise ValueError("num_probed_lists は 1 以上 K 以下")
        lists = self._nearest_lists(query)[:num_probed_lists]
        bounds = [(int(self.offsets[j]), int(self.offsets[j + 1])) for j in lists.tolist()]
        pieces = [self.sorted_vectors[first:last] for first, last in bounds]
        piece_ids = [self.sorted_ids[first:last] for first, last in bounds]
        scanned_vectors = torch.cat(pieces)
        scanned_ids = torch.cat(piece_ids)
        scores = scanned_vectors @ query
        if excluded >= 0:
            scores[scanned_ids == excluded] = float("-inf")
        top = scores.topk(min(k, len(scores)))
        ids = np.full(k, -1, dtype=np.int64)
        valid = torch.isfinite(top.values)
        chosen = scanned_ids[top.indices][valid].numpy()
        ids[: len(chosen)] = chosen
        if count_cost:
            self.counter.add("centroid_comparisons", self.num_lists)
            self.counter.add("scanned_vectors", len(scanned_ids))
        return ids, self.num_lists, int(len(scanned_ids))

    def search_batch_by_probe_counts(
        self,
        queries: Tensor,
        k: int,
        probe_counts: Sequence[int],
        excluded: np.ndarray | None = None,
        chunk_size: int = 256,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """複数の query を複数の ``p`` で探索する(評価用の一括計算。結果は ``search()``と一致する)。

        全ベクトルとの内積を一括で計算し、各 query の
        ``p``個の近いリストに入っていないベクトルを除いて上位 ``k`` 件を求める。数える費用は
        ``search()`` と同じ(重心との比較 ``K``、走査したベクトルの数 =選んだリストの大きさの和)。

        Returns:
            ``{p: (上位 k 件の添字 (Q, k)、query ごとの距離計算の回数 K + 走査数 (Q,))}``。
        """
        probe_counts = sorted(set(int(p) for p in probe_counts))
        if probe_counts[0] < 1 or probe_counts[-1] > self.num_lists:
            raise ValueError("probe_counts は 1 以上 K 以下")
        num_queries = queries.size(0)
        ids = {p: np.full((num_queries, k), -1, dtype=np.int64) for p in probe_counts}
        costs = {p: np.empty(num_queries, dtype=np.int64) for p in probe_counts}
        sorted_ids = self.sorted_ids
        # ベクトル -> そのベクトルが属する位置(sorted_ids の並びでの位置)のリストの番号
        list_of_position = torch.repeat_interleave(torch.arange(self.num_lists), self.list_sizes)
        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            block = queries[start:end]
            squared = (
                block.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * block @ self.centroids.t()
                + self.centroids.pow(2).sum(dim=1)[None, :]
            )
            order = torch.argsort(squared, dim=1, stable=True)  # 近い順のリストの番号
            rank_of_list = torch.empty_like(order)
            rank_of_list.scatter_(
                1, order, torch.arange(self.num_lists).expand_as(order).contiguous()
            )
            position_rank = rank_of_list[
                :, list_of_position
            ]  # (chunk, N): 各ベクトルのリストの順位
            scores = block @ self.sorted_vectors.t()  # (chunk, N): sorted_ids の並び
            if excluded is not None:
                excluded_block = torch.as_tensor(excluded[start:end])
                scores = scores.masked_fill(
                    sorted_ids[None, :] == excluded_block[:, None], float("-inf")
                )
            cumulative = torch.cumsum(self.list_sizes[order], dim=1)  # 近い順にリストを走査した数
            for p in probe_counts:
                masked = scores.masked_fill(position_rank >= p, float("-inf"))
                top = masked.topk(min(k, masked.size(1)), dim=1)
                chosen = sorted_ids[top.indices]
                chosen = torch.where(
                    torch.isfinite(top.values), chosen, torch.full_like(chosen, -1)
                )
                ids[p][start:end, : chosen.size(1)] = chosen.numpy()
                costs[p][start:end] = (self.num_lists + cumulative[:, p - 1]).numpy()
        return {p: (ids[p], costs[p]) for p in probe_counts}

    def expected_cost(self, num_probed_lists: int) -> float:
        """リストの大きさが等しいときの費用の式 ``K + p N / K``(理論値)。"""
        return self.num_lists + num_probed_lists * self.num_vectors / self.num_lists
