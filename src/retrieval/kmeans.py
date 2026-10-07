"""k-means(Lloyd のアルゴリズム)のスクラッチ実装(025)。

転置ファイル(``inverted_file.py``)の Voronoi 分割と、Product
Quantization(``product_quantization.py``)の符号帳の学習で共用する。

**目的関数**: 点の集合 ``{x_i}`` を ``K`` 個の重心 ``{c_j}`` で表し、
各点を最も近い重心に割り当てたときの二乗誤差の和(inertia)を小さくする。

.. math::

    J = \\sum_{i=1}^{n} \\min_{j} \\lVert x_i - c_j \\rVert^2

**Lloyd のアルゴリズム**: 次の 2 つの手順を交互に行う。どちらも ``J`` を増やさないので、``J``
は反復ごとに単調に減少する(``J`` が下に有界なので収束する。ただし局所最適)。

1. 割り当て: 各点を、二乗距離が最小の重心に割り当てる(同点は番号の小さい重心)。
2. 更新: 各重心を、割り当てられた点の平均にする。

**固定する設定**(再現のために宣言する):

- 初期化: 点の集合から、シードで固定した乱数で ``K``
  個を重複なく選んで重心の初期値にする(k-means++ は使わない)。
- 反復回数: 引数 ``num_iterations`` の回数で固定する(収束の判定による早期終了はしない)。
- 空のクラスタ: 割り当てが 0 個になった重心は、その反復の割り当ての後で、
  割り当てられた重心までの距離が最も大きい点(空のクラスタの数だけ、距離の大きい順)に置き直す。
  置き直しは ``J`` を減らす向きの更新なので、``J`` の単調減少を壊さない。

**距離の計算**: ``||x - c||^2 = ||x||^2 - 2 x^T c + ||c||^2``
を行列積で一括して計算する(点を``chunk_size`` 個ずつ処理する)。

記号 / Notation:
    n : 点の数
    d : 点の次元
    K : 重心の数(クラスタの数)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class KMeansResult:
    """k-means の結果。

    Attributes:
        centroids: 形状 ``(K, d)``。最後の更新の後の重心。
        assignments: 形状 ``(n,)``。
        最後の更新の後の重心への割り当て(最後の更新の後に割り当て直した値)。
        inertia_history: 反復ごとの、更新の前の割り当てでの二乗誤差の和 ``J``。長さ
        ``num_iterations``。
        num_empty_cluster_events: 空のクラスタを置き直した回数の合計。
    """

    centroids: Tensor
    assignments: Tensor
    inertia_history: list[float] = field(default_factory=list)
    num_empty_cluster_events: int = 0


def assign_to_nearest(
    points: Tensor, centroids: Tensor, chunk_size: int = 8192
) -> tuple[Tensor, Tensor]:
    """各点を最も近い重心(二乗ユークリッド距離)に割り当てる。

    Returns:
        ``(割り当て (n,)、割り当てた重心までの二乗距離 (n,))``。同点は番号の小さい重心
        (``torch.argmin`` は最初の最小を返す)。
    """
    centroid_norms = centroids.pow(2).sum(dim=1)
    assignments = torch.empty(points.size(0), dtype=torch.long, device=points.device)
    distances = torch.empty(points.size(0), dtype=points.dtype, device=points.device)
    for start in range(0, points.size(0), chunk_size):
        block = points[start : start + chunk_size]
        squared = block.pow(2).sum(dim=1, keepdim=True) - 2.0 * block @ centroids.t()
        squared = squared + centroid_norms[None, :]
        best = squared.argmin(dim=1)
        assignments[start : start + chunk_size] = best
        distances[start : start + chunk_size] = (
            squared.gather(1, best[:, None]).squeeze(1).clamp_min(0.0)
        )
    return assignments, distances


def fit_kmeans(
    points: Tensor,
    num_clusters: int,
    num_iterations: int,
    seed: int,
    chunk_size: int = 8192,
) -> KMeansResult:
    """Lloyd のアルゴリズムで k-means を学習する(初期化・反復回数・空のクラスタの扱いは冒頭の説明)。

    Args:
        points: 形状 ``(n, d)`` の浮動小数点のテンソル(``n`` は ``num_clusters`` 以上)。
        num_clusters: ``K``。
        num_iterations: 反復回数(1 回 = 割り当て + 更新)。
        seed: 初期化の乱数シード。
        chunk_size: 距離を一度に計算する点の数(結果には影響しない)。
    """
    n = points.size(0)
    if not 1 <= num_clusters <= n:
        raise ValueError(f"num_clusters は 1 以上 n 以下: {num_clusters}, n={n}")
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(n, generator=generator)[:num_clusters].to(points.device)
    centroids = points[initial].clone()
    inertia_history: list[float] = []
    empty_events = 0
    for _ in range(num_iterations):
        assignments, distances = assign_to_nearest(points, centroids, chunk_size)
        inertia_history.append(float(distances.double().sum()))
        counts = torch.bincount(assignments, minlength=num_clusters)
        sums = torch.zeros_like(centroids).index_add_(0, assignments, points)
        updated = sums / counts.clamp_min(1).to(points.dtype)[:, None]
        empty = (counts == 0).nonzero().squeeze(1)
        if len(empty):
            farthest = distances.topk(len(empty)).indices  # 割り当てた重心までの距離が最も大きい点
            updated[empty] = points[farthest]
            empty_events += len(empty)
        centroids = updated
    assignments, _ = assign_to_nearest(points, centroids, chunk_size)
    return KMeansResult(centroids, assignments, inertia_history, empty_events)


def inertia(points: Tensor, centroids: Tensor, chunk_size: int = 8192) -> float:
    """点を最も近い重心に割り当てたときの二乗誤差の和 ``J``(FP64 で和をとる)。"""
    _, distances = assign_to_nearest(points, centroids, chunk_size)
    return float(distances.double().sum())


def list_sizes(assignments: Tensor, num_clusters: int) -> np.ndarray:
    """クラスタごとの点の数(形状 ``(K,)``)。"""
    return torch.bincount(assignments, minlength=num_clusters).cpu().numpy()
