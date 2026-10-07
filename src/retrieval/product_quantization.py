"""Product Quantization(直積量子化)のスクラッチ実装(025)。

Jégou, Douze & Schmid, "Product Quantization for Nearest Neighbor Search", IEEE TPAMI
2011 に従う。

**符号化**: ``d`` 次元のベクトル ``y`` を ``m`` 個の部分ベクトル ``y^1, ..., y^m``(各
``d / m``次元)に分け、部分空間ごとに独立な k-means(``kmeans.py``)で ``k*``
個の符号語(centroid)の符号帳``C^1, ..., C^m`` を学習する。
各部分ベクトルを最も近い符号語の番号(``log_2 k*`` ビット)で表す。

.. math::

    q(y) = \\left( q^1(y^1), \\dots, q^m(y^m) \\right), \\qquad q^j(y^j) = \\arg\\min_{c \\in
    C^j} \\lVert y^j - c \\rVert^2

符号長は ``m \\log_2 k*`` ビット(``k* = 256`` なら ``m`` バイト)。**実効的な符号帳の大きさ** は、
部分空間の
符号語の直積 ``C = C^1 \\times \\dots \\times C^m`` の要素数 ``(k*)^m`` で、``m``
個の符号帳の合計 ``m k*`` 個の
符号語を持つだけで、全体では ``(k*)^m`` 個の再構成のベクトルを表せる(``k* = 256``、``m = 32`` なら
``256^{32}``)。通常の k-means で同じ数の符号語を持つことは、メモリと学習の面で不可能である。

**非対称距離計算**(Asymmetric Distance Computation): query ``x`` は量子化せず、
データベースのベクトル ``y`` だけを量子化して、二乗距離を次の和で近似する。

.. math::

    \\hat{d}_{\\mathrm{asymmetric}}(x, y)^2 = \\lVert x - q(y) \\rVert^2 = \\sum_{j=1}^{m}
    \\lVert x^j - q^j(y^j) \\rVert^2

query ごとに、部分空間 ``j`` と符号語 ``c`` のすべての組について ``\\lVert x^j - c \\rVert^2``
を求めて参照表
(``m x k*`` 個の値)を作り、各ベクトルの距離は ``m`` 回の表引きの和で求まる。

**対称距離計算**(Symmetric Distance Computation): query も量子化して、
符号語どうしの距離で近似する。

.. math::

    \\hat{d}_{\\mathrm{symmetric}}(x, y)^2 = \\lVert q(x) - q(y) \\rVert^2 = \\sum_{j=1}^{m}
    \\lVert q^j(x^j) - q^j(y^j) \\rVert^2

符号語どうしの距離の表(``m x k* x k*``)は索引の構築時に 1 回だけ作る。query
の量子化(最も近い符号語を探す)に ``m k*`` 回の部分距離の計算が要る。

**距離の誤差と量子化の平均二乗誤差の関係**: 符号語は k-means の重心(領域に属するベクトルの
平均)なので、``e_y = y - q(y)``(領域の中での位置のずれ。量子化の誤差)は、領域を決めると
平均 0 である。``D = \\mathbb{E} \\lVert e_y \\rVert^2`` を量子化の平均二乗誤差とし、
query ``x`` は、``y`` の領域の中での位置と独立とする。``x - y = (x - q(y)) - e_y`` を展開すると、

.. math::

    \\lVert x - y \\rVert^2 = \\lVert x - q(y) \\rVert^2
    - 2 \\langle x - q(y), e_y \\rangle + \\lVert e_y \\rVert^2

右辺の第 2 項は、``e_y`` の平均が 0 で ``x - q(y)`` と独立なので期待値が 0、第 3 項の期待値は
``D`` である。したがって非対称距離計算の推定の二乗距離は、真の値に対して
**平均として ``D`` だけ小さい**(偏りは ``-D``。``y`` を重心に置き換えると、``y`` の領域の中の
ばらつきのぶんだけ、``x`` に近づいて見える)。対称距離計算では ``x`` も量子化するので、
``e_x = x - q(x)`` を使い、

.. math::

    \\lVert x - y \\rVert^2 = \\lVert q(x) - q(y) \\rVert^2
    + 2 \\langle q(x) - q(y), e_x - e_y \\rangle + \\lVert e_x - e_y \\rVert^2

となり、第 2 項の期待値は 0、第 3 項の期待値は
``\\mathbb{E}\\lVert e_x \\rVert^2 + \\mathbb{E}\\lVert e_y \\rVert^2``(``e_x`` と ``e_y`` が独立で
平均 0 のとき)なので、偏りは
``-(\\mathbb{E}\\lVert e_x \\rVert^2 + \\mathbb{E}\\lVert e_y \\rVert^2)`` で、
``x`` と ``y`` が同じ分布なら ``-2D`` である。誤差の分散も、``e_x`` の項が加わるぶん大きい。
偏りが全ベクトルで一様なら順位は変わらないが、``\\lVert e_y \\rVert^2`` の大きさがベクトルごとに
異なるので、順位は乱れる。対称距離計算は ``x`` の誤差も加わるぶん、非対称距離計算より誤差が
大きい(Jégou et al. は、解析と実験で、非対称距離計算の方が検索の精度が高いことを示した)。
query が埋め込みの分布から外れている(passage と長さが違う)場合は、
``\\lVert e_x \\rVert^2`` がさらに大きくなりうる。

**スカラー量子化との対応**(013): 013 の一様量子化は、1 つの値を
``2^b``個の等間隔の水準に丸める(1 次元の
量子化。誤差 ``\\Delta^2 / 12``)。Product Quantization は、これを **部分ベクトル(``d / m`` 次元)
を単位とする
ベクトル量子化** に拡張したもので、水準が等間隔である必要がなく、
データの分布に合わせて学習した符号語を
使う。``m = d``(部分ベクトルが 1 次元)のときがスカラー量子化に対応する。

**再採点**(re-scoring): 圧縮した符号による距離で上位 ``R`` 件の候補に絞り、その
``R``件だけ元のベクトルで距離を計算し直して順位をつける。距離計算の回数は ``R`` 回(全次元)増える。

**費用の数え方**(``CostCounter``):

- ``"table_entries"``: 参照表の作成での部分ベクトルと符号語の距離の計算の回数。非対称距離計算では
  query ごとに``m k*``、対称距離計算では query の量子化で query ごとに
  ``m k*``(符号語どうしの表は構築時に 1 回、``"construction_table_entries"`` に ``m (k*)^2``)。
- ``"table_lookups"``: 表引き(部分空間ごとの加算)の回数。全件を走査するので query ごとに ``N m``。
- ``"rescored_vectors"``: 再採点した、元のベクトルとの距離計算(全次元)の回数。

**実装の注意**: 全ベクトルの距離を、部分空間ごとの表引きを足して一括で計算す
る(``torch.index_select``)。

記号 / Notation:
    d : 次元、m : 部分ベクトルの数(``num_subvectors``)、k* : 1
    つの符号帳の符号語の数(``num_codewords``)、
    N : ベクトルの数
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.retrieval.exact_search import CostCounter
from src.retrieval.kmeans import assign_to_nearest, fit_kmeans


class ProductQuantizer:
    """部分空間ごとの符号帳による量子化器。

    Args:
        dimension: ``d``(``num_subvectors`` で割り切れること)。
        num_subvectors: ``m``。
        num_codewords: ``k*``(256 以下で、符号を 1 バイトで表せること)。
    """

    def __init__(self, dimension: int, num_subvectors: int, num_codewords: int = 256) -> None:
        if dimension % num_subvectors != 0:
            raise ValueError("dimension は num_subvectors で割り切れる必要がある")
        if not 1 <= num_codewords <= 256:
            raise ValueError("num_codewords は 1 以上 256 以下(符号を 1 バイトで表す)")
        self.dimension = dimension
        self.num_subvectors = num_subvectors
        self.num_codewords = num_codewords
        self.subvector_dimension = dimension // num_subvectors
        self.codebooks: Tensor | None = None  # (m, k*, d / m)
        self.symmetric_table: Tensor | None = None  # (m, k*, k*)
        self.kmeans_inertia: list[float] = []

    @property
    def code_bytes(self) -> int:
        """1 ベクトルあたりの符号の大きさ(バイト)。"""
        return self.num_subvectors

    def split(self, vectors: Tensor) -> Tensor:
        """``(n, d)`` を部分ベクトルに分ける ``(n, m, d / m)``。"""
        return vectors.reshape(vectors.size(0), self.num_subvectors, self.subvector_dimension)

    def fit(self, vectors: Tensor, num_iterations: int, seed: int) -> None:
        """部分空間ごとに k-means で符号帳を学習し、符号語どうしの距離の表を作る。"""
        parts = self.split(vectors)
        books, inertia = [], []
        for j in range(self.num_subvectors):
            result = fit_kmeans(
                parts[:, j, :].contiguous(), self.num_codewords, num_iterations, seed + 1000 * j
            )
            books.append(result.centroids)
            inertia.append(result.inertia_history[-1])
        self.codebooks = torch.stack(books)
        self.kmeans_inertia = inertia
        # 符号語どうしの二乗距離 ||c_a - c_b||^2: (m, k*, k*)
        norms = self.codebooks.pow(2).sum(dim=2)
        self.symmetric_table = (
            norms[:, :, None]
            - 2.0 * self.codebooks @ self.codebooks.transpose(1, 2)
            + norms[:, None, :]
        ).clamp_min(0.0)

    def encode(self, vectors: Tensor) -> Tensor:
        """各部分ベクトルを最も近い符号語の番号にする ``(n, m)``(uint8)。"""
        if self.codebooks is None:
            raise RuntimeError("fit() を先に呼ぶ")
        parts = self.split(vectors)
        codes = torch.empty(vectors.size(0), self.num_subvectors, dtype=torch.uint8)
        for j in range(self.num_subvectors):
            assignments, _ = assign_to_nearest(parts[:, j, :].contiguous(), self.codebooks[j])
            codes[:, j] = assignments.to(torch.uint8)
        return codes

    def decode(self, codes: Tensor) -> Tensor:
        """符号から再構成のベクトル ``q(y)`` を作る ``(n, d)``。"""
        if self.codebooks is None:
            raise RuntimeError("fit() を先に呼ぶ")
        parts = [self.codebooks[j][codes[:, j].long()] for j in range(self.num_subvectors)]
        return torch.cat(parts, dim=1)

    def quantization_mean_squared_error(self, vectors: Tensor) -> float:
        """量子化の平均二乗誤差 ``D = E ||y - q(y)||^2``(FP64 で平均)。"""
        reconstructed = self.decode(self.encode(vectors))
        return float((vectors - reconstructed).double().pow(2).sum(dim=1).mean())

    def asymmetric_table(self, queries: Tensor) -> Tensor:
        """query ごとの参照表 ``T[q, j, c] = ||x^j - C^j_c||^2``(形状 ``(Q, m, k*)``)。"""
        if self.codebooks is None:
            raise RuntimeError("fit() を先に呼ぶ")
        parts = self.split(queries)  # (Q, m, d/m)
        part_norms = parts.pow(2).sum(dim=2)  # (Q, m)
        cross = torch.einsum("qjd,jcd->qjc", parts, self.codebooks)
        book_norms = self.codebooks.pow(2).sum(dim=2)  # (m, k*)
        return (part_norms[:, :, None] - 2.0 * cross + book_norms[None, :, :]).clamp_min(0.0)


class ProductQuantizationIndex:
    """Product Quantization の索引(全件を走査する。転置ファイルとの組み合わせは使わない)。

    Args:
        quantizer: 学習済みの ``ProductQuantizer``。
        vectors: 形状 ``(N, d)``。符号化して保持する(再採点のために元のベクトルも参照する)。
    """

    def __init__(self, quantizer: ProductQuantizer, vectors: Tensor) -> None:
        self.quantizer = quantizer
        self.vectors = vectors
        self.num_vectors = vectors.size(0)
        self.codes = quantizer.encode(vectors)
        self.counter = CostCounter(
            ("construction_table_entries", "table_entries", "table_lookups", "rescored_vectors")
        )
        m, k_star = quantizer.num_subvectors, quantizer.num_codewords
        self.counter.add("construction_table_entries", m * k_star * k_star)

    def _estimated_distances(self, tables: Tensor) -> Tensor:
        """参照表 ``tables (Q, m, k*)`` から、全ベクトルとの推定の二乗距離 ``(Q, N)`` を作る。"""
        total = None
        for j in range(self.quantizer.num_subvectors):
            gathered = tables[:, j, :].index_select(1, self.codes[:, j].long())
            total = gathered if total is None else total + gathered
        return total

    def _search(
        self,
        tables_for_block,
        queries: Tensor,
        k: int,
        excluded: np.ndarray | None,
        num_rescored: int,
        chunk_size: int,
        table_entries_per_query: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        m = self.quantizer.num_subvectors
        num_queries = queries.size(0)
        ids = np.full((num_queries, k), -1, dtype=np.int64)
        candidates = max(k, num_rescored)
        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            distances = self._estimated_distances(tables_for_block(queries[start:end]))
            if excluded is not None:
                rows = np.nonzero(excluded[start:end] >= 0)[0]
                if len(rows):
                    distances[torch.as_tensor(rows), torch.as_tensor(excluded[start:end][rows])] = (
                        float("inf")
                    )
            top = distances.topk(candidates, dim=1, largest=False).indices  # 推定の距離が小さい順
            if num_rescored:
                block = queries[start:end]
                gathered = self.vectors[top]  # (chunk, R, d)
                exact = torch.einsum("qrd,qd->qr", gathered, block)
                order = exact.topk(k, dim=1).indices
                top = top.gather(1, order)
            ids[start:end] = top[:, :k].numpy()
        costs = {
            "table_entries": np.full(num_queries, table_entries_per_query, dtype=np.int64),
            "table_lookups": np.full(num_queries, self.num_vectors * m, dtype=np.int64),
            "rescored_vectors": np.full(num_queries, num_rescored, dtype=np.int64),
        }
        for name, values in costs.items():
            self.counter.add(name, int(values.sum()))
        return ids, costs

    def search_asymmetric(
        self,
        queries: Tensor,
        k: int,
        excluded: np.ndarray | None = None,
        num_rescored: int = 0,
        chunk_size: int = 16,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """非対称距離計算で上位 ``k`` 件を探す(``num_rescored > 0`` なら上位
        ``R``件を元のベクトルで再採点する)。

                Returns:
                    ``(上位 k 件の添字 (Q, k)、費用の種類ごとの query ごとの回数)``。
        """
        m, k_star = self.quantizer.num_subvectors, self.quantizer.num_codewords
        return self._search(
            self.quantizer.asymmetric_table,
            queries,
            k,
            excluded,
            num_rescored,
            chunk_size,
            m * k_star,
        )

    def search_symmetric(
        self,
        queries: Tensor,
        k: int,
        excluded: np.ndarray | None = None,
        num_rescored: int = 0,
        chunk_size: int = 16,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """対称距離計算(query も量子化し、符号語どうしの距離の表を使う)で上位 ``k`` 件を探す。"""
        m, k_star = self.quantizer.num_subvectors, self.quantizer.num_codewords
        quantizer = self.quantizer

        def tables(block: Tensor) -> Tensor:
            query_codes = quantizer.encode(block).long()  # (Q, m)
            rows = [
                quantizer.symmetric_table[j][query_codes[:, j]] for j in range(m)
            ]  # (Q, k*) ずつ
            return torch.stack(rows, dim=1)

        return self._search(tables, queries, k, excluded, num_rescored, chunk_size, m * k_star)

    def reconstructed_distances(self, queries: Tensor, asymmetric: bool) -> Tensor:
        """単体テスト・診断用: 復元したベクトルとの二乗距離を直接計算した値 ``(Q, N)``。

        非対称なら ``||x - q(y)||^2``、対称なら ``||q(x) - q(y)||^2``。
        """
        reconstructed = self.quantizer.decode(self.codes)
        left = queries if asymmetric else self.quantizer.decode(self.quantizer.encode(queries))
        return (left[:, None, :] - reconstructed[None, :, :]).pow(2).sum(dim=2)
