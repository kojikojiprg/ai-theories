"""HNSW(Hierarchical Navigable Small World)のスクラッチ実装(025)。

Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs", IEEE TPAMI 2020(arXiv:1603.09320)の Algorithm 1〜5 に従う。NumPy
だけで書く(faiss・hnswlib は使わない)。

**構造**: 要素(ベクトル)を頂点とする近傍グラフを、層(layer)ごとに持つ。最下層(層
0)がすべての要素を含み、上の層ほど要素が少ない(上の層の頂点の集合は下の層の部分集合)。
要素の層の番号 ``l`` は、挿入のときに指数分布から引く(原論文の式)。

.. math::

    l = \\lfloor -\\ln(u) \\cdot m_L \\rfloor, \\qquad u \\sim \\mathrm{Uniform}(0, 1), \\qquad
    m_L = 1 / \\ln M

``M`` は 1 つの要素が張る双方向の接続の数の目安、
``m_L``は層の番号の分布を決める正規化の係数である。``P(l >= j) = M^{-j}`` なので、層 ``j``
の頂点の数は ``N M^{-j}`` 個程度で、層の数は ``log_M N``程度になる。

**パラメータ**(原論文の記号。コードの引数名は略さない):

- ``M``(``max_connections``): 要素が挿入時に張る接続の数。
- ``M_max``: 層 1 以上での 1 つの頂点の接続の数の上限(``M`` に等しい)。``M_max0``: 層
  0 での上限(``2 M``)。
- ``efConstruction``(``construction_search_width``): 挿入のときの探索の幅。
- ``ef``(``search_width``): 探索の幅(候補の集合 ``W`` の大きさの上限)。``ef`` を大きくすると
  recall が上がり、費用も増える。

**アルゴリズム**: ``insert()`` が Algorithm 1(挿入)、``_search_layer()`` が Algorithm
2(層の中の探索)、``_select_neighbors_simple()`` が Algorithm 3、``_select_neighbors_heuristic()``
が Algorithm 4(近傍の選択のヒューリスティック)、``search()`` が Algorithm 5(k 近傍探索)である。

- Algorithm 4(ヒューリスティック): 候補を query に近い順に調べ、すでに選んだどの要素よりも query
  に近い候補だけを選ぶ(``d(e, q) < d(e, r)`` がすべての選んだ ``r`` で成り立つとき)。
  方向の異なる近傍を選ぶので、クラスタの間をつなぐ接続が残りやすい。
  ``extendCandidates``(``extend_candidates``)は、原論文の既定どおり偽。
  ``keepPrunedConnections``(``keep_pruned_connections``)は、捨てた候補から接続の数が ``M``
  になるまで補うオプションで、原論文は既定値を指定していない。本実装の既定は真。

**距離**: 単位ベクトルを前提とし、``distance = 1 - x^T y``(コサイン距離。内積の大きい順
=距離の小さい順)を使う。1 つの頂点の近傍への距離は、行列積で 1 回にまとめて計算する。

**費用の数え方**: 距離計算の回数を数える(``CostCounter`` の名前 ``"search"``
と``"construction"``)。

- 探索(層の中の探索): 訪問済みでない近傍ごとに 1 回(Algorithm 2 は、
  近傍が候補に入るかの判定のために query との距離を求める)。入口の要素との距離も
  1 回として数える。
- 構築: 層の中の探索の距離計算に加えて、Algorithm 4 が実際に比べた距離(``d(e, r)``)と、
  接続の数が上限を超えた近傍の再選択で求めた距離を数える。

**挿入の順序と入れ子の索引**: 挿入の順序と各要素の層の番号は、シードで固定した乱数で決める。
``add()`` は挿入の順序の続きから要素を追加するので、途中で探索すれば、先頭
``n``個を挿入した時点の索引を得られる(挿入の順序の先頭 ``n`` 個の部分集合は、``n``
を増やすと入れ子になる)。

記号 / Notation:
    N : 要素の総数、d : 次元、M : 接続の数、k : 返す近傍の数
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from src.retrieval.exact_search import CostCounter


def draw_levels_and_insertion_order(
    num_elements: int, max_connections: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """要素ごとの層の番号 ``l = floor(-ln(u) m_L)`` と、挿入の順序を、シードで固定した乱数で引く。

    ``HierarchicalNavigableSmallWorld`` が使うものと同じで、索引を作らずに、
    同じ挿入の順序の先頭``n`` 個(入れ子の部分集合)を、他の索引(転置ファイル・Product
    Quantization)に使うために切り出してある。

    Returns:
        ``(層の番号 (N,)、挿入の順序 (N,))``。
    """
    level_multiplier = 1.0 / math.log(max_connections)  # m_L
    rng = np.random.default_rng(seed)
    uniform = 1.0 - rng.random(num_elements)  # (0, 1]。log(0) を避ける
    levels = np.floor(-np.log(uniform) * level_multiplier).astype(np.int64)
    return levels, rng.permutation(num_elements).astype(np.int64)


class HierarchicalNavigableSmallWorld:
    """HNSW の索引。

    Args:
        vectors: 形状 ``(N, d)``。単位ベクトル(``dtype`` で保持する)。挿入するのは ``add()`` で
            指定した数だけで、挿入していない要素は探索の対象にならない。
        max_connections: ``M``。
        construction_search_width: ``efConstruction``。
        seed: 層の番号と挿入の順序を決める乱数シード。
        extend_candidates: Algorithm 4 の ``extendCandidates``。
        keep_pruned_connections: Algorithm 4 の ``keepPrunedConnections``。
        use_heuristic: 近傍の選択に Algorithm 4(ヒューリスティック)を使うか。偽なら Algorithm
        3(単純な
            上位 ``M`` 件)を使う。
        dtype: ベクトルの型(既定は ``np.float32``。単体テストでは ``np.float64``)。
    """

    def __init__(
        self,
        vectors: np.ndarray,
        max_connections: int = 16,
        construction_search_width: int = 100,
        seed: int = 0,
        extend_candidates: bool = False,
        keep_pruned_connections: bool = True,
        use_heuristic: bool = True,
        dtype: type = np.float32,
    ) -> None:
        if max_connections < 2:
            raise ValueError("max_connections は 2 以上")
        self.vectors = np.ascontiguousarray(vectors, dtype=dtype)
        self.dtype = dtype
        self.num_elements, self.dimension = self.vectors.shape
        self.max_connections = max_connections
        self.max_connections_layer0 = 2 * max_connections
        self.construction_search_width = construction_search_width
        self.extend_candidates = extend_candidates
        self.keep_pruned_connections = keep_pruned_connections
        self.use_heuristic = use_heuristic
        self.seed = seed
        self.level_multiplier = 1.0 / math.log(max_connections)  # m_L
        self.levels, self.insertion_order = draw_levels_and_insertion_order(
            self.num_elements, max_connections, seed
        )
        self.max_level = int(self.levels.max()) if self.num_elements else 0
        # 層ごとの接続: links[l] は (その層の頂点数, 上限) の配列、degree[l] は各頂点の接続の数、
        # slot[l] は要素の番号 -> links[l] の行(層 0 は要素の番号そのもの)
        self.links: list[np.ndarray] = []
        self.degree: list[np.ndarray] = []
        self.slot: list[np.ndarray | None] = []
        for layer in range(self.max_level + 1):
            members = np.nonzero(self.levels >= layer)[0]
            capacity = self.max_connections_layer0 if layer == 0 else self.max_connections
            self.links.append(np.full((len(members), capacity), -1, dtype=np.int32))
            self.degree.append(np.zeros(len(members), dtype=np.int32))
            if layer == 0:
                self.slot.append(None)
            else:
                slot = np.full(self.num_elements, -1, dtype=np.int64)
                slot[members] = np.arange(len(members))
                self.slot.append(slot)
        self.num_inserted = 0
        self.entry_point = -1
        self.top_level = -1
        self._visited = np.zeros(self.num_elements, dtype=np.int64)
        self._stamp = 0
        self.counter = CostCounter(("search", "construction"))

    # ---- 接続の読み書き ----

    def _row(self, layer: int, node: int) -> int:
        return node if layer == 0 else int(self.slot[layer][node])

    def neighbors(self, layer: int, node: int) -> np.ndarray:
        """層 ``layer`` での要素 ``node`` の近傍(接続先)の番号。"""
        row = self._row(layer, node)
        return self.links[layer][row, : self.degree[layer][row]]

    def _set_neighbors(self, layer: int, node: int, neighbor_ids: np.ndarray) -> None:
        row = self._row(layer, node)
        count = len(neighbor_ids)
        self.links[layer][row, :count] = neighbor_ids
        self.links[layer][row, count:] = -1
        self.degree[layer][row] = count

    def capacity(self, layer: int) -> int:
        return self.max_connections_layer0 if layer == 0 else self.max_connections

    # ---- Algorithm 2: SEARCH-LAYER ----

    def _search_layer(
        self,
        query: np.ndarray,
        entry_ids: list[int],
        entry_distances: list[float],
        search_width: int,
        layer: int,
    ) -> tuple[list[tuple[float, int]], int]:
        """層 ``layer`` で、``query`` に近い要素を最大 ``search_width`` 個探す(Algorithm 2)。

        Args:
            query: 形状 ``(d,)``。
            entry_ids: 入口の要素(``ep``)。
            entry_distances: 入口の要素と query の距離(求めてあるものを渡す。ここでは数えない)。
            search_width: ``ef``。
            layer: 層の番号。

        Returns:
            距離の昇順の ``(距離, 要素の番号)`` のリスト ``W`` と、この呼び出しの距離計算の回数。
        """
        self._stamp += 1
        stamp, visited, vectors = self._stamp, self._visited, self.vectors
        links, degree = self.links[layer], self.degree[layer]
        slot = self.slot[layer]
        candidates: list[tuple[float, int]] = []  # 距離の小さい順に取り出す(最小ヒープ)
        results: list[
            tuple[float, int]
        ] = []  # 距離の大きい順に取り出す(最大ヒープ: 距離の符号を反転)
        for entry, distance in zip(entry_ids, entry_distances, strict=True):
            visited[entry] = stamp
            heapq.heappush(candidates, (distance, entry))
            heapq.heappush(results, (-distance, entry))
            if len(results) > search_width:
                heapq.heappop(results)
        computed = 0
        while candidates:
            distance_c, c = heapq.heappop(candidates)
            if distance_c > -results[0][0]:  # 最も近い候補が、W の最も遠い要素より遠い
                break
            row = c if slot is None else slot[c]
            neighbor_ids = links[row, : degree[row]]
            unvisited = neighbor_ids[visited[neighbor_ids] != stamp]
            if len(unvisited) == 0:
                continue
            visited[unvisited] = stamp
            distances = (
                1.0 - vectors[unvisited] @ query
            )  # 近傍への距離を 1 回の演算でまとめて計算する
            computed += len(unvisited)
            if (
                len(results) >= search_width
            ):  # W が満杯のとき、最も遠い要素より遠い近傍は採用されない
                keep = distances < -results[0][0]
                if not keep.any():
                    continue
                unvisited, distances = unvisited[keep], distances[keep]
            for distance_e, e in zip(distances.tolist(), unvisited.tolist(), strict=True):
                if len(results) < search_width or distance_e < -results[0][0]:
                    heapq.heappush(candidates, (distance_e, e))
                    if len(results) < search_width:
                        heapq.heappush(results, (-distance_e, e))
                    else:
                        heapq.heapreplace(results, (-distance_e, e))
        return sorted((-negative, e) for negative, e in results), computed

    # ---- Algorithm 3 / 4: SELECT-NEIGHBORS ----

    def _select_neighbors_simple(
        self, candidate_ids: np.ndarray, candidate_distances: np.ndarray, count: int
    ) -> np.ndarray:
        """Algorithm 3: 候補のうち、基準の点に近い順に ``count`` 個を選ぶ(候補は距離の昇順)。"""
        return candidate_ids[:count]

    def _select_neighbors_heuristic(
        self,
        base_vector: np.ndarray,
        candidate_ids: np.ndarray,
        candidate_distances: np.ndarray,
        count: int,
        layer: int,
    ) -> tuple[np.ndarray, int]:
        """Algorithm 4: ヒューリスティックで ``count`` 個を選ぶ(候補は基準の点への距離の昇順)。

        基準の点 ``q`` に近い順に候補 ``e`` を調べ、選んだ要素 ``r``
        のすべてについて``d(e, q) < d(e, r)`` なら ``e``
        を選ぶ(そうでなければ捨てた候補の列に回す)。選んだ数が``count`` に達したら終わる。
        ``extend_candidates`` が真なら、調べる前に、
        候補の近傍(層``layer``)も候補に加える(基準の点との距離を新たに求める)。

        Args:
            base_vector: 形状 ``(d,)``。基準の点 ``q`` のベクトル。
            candidate_ids: 候補の要素の番号(基準の点への距離の昇順)。
            candidate_distances: 候補の、基準の点への距離。
            count: 選ぶ数(``M`` または ``M_max``)。
            layer: 層の番号(``extend_candidates`` の近傍の取得に使う)。

        Returns:
            ``(選んだ要素の番号 (選んだ順)、距離を求めた回数)``。距離を求めた回数は、``d(e, r)``
            の比較と、
            ``extend_candidates`` で加えた候補の ``d(e, q)`` の合計。
        """
        ids, distances = candidate_ids, candidate_distances
        computed = 0
        if self.extend_candidates:
            known = set(candidate_ids.tolist())
            added = sorted(
                {int(n) for e in candidate_ids for n in self.neighbors(layer, int(e))} - known
            )
            if added:
                added_ids = np.array(added, dtype=np.int64)
                added_distances = 1.0 - self.vectors[added_ids] @ base_vector
                computed += len(added_ids)
                ids = np.concatenate([candidate_ids, added_ids])
                distances = np.concatenate(
                    [candidate_distances, added_distances.astype(np.float64)]
                )
                order = np.argsort(distances, kind="stable")
                ids, distances = ids[order], distances[order]
        selected: list[int] = []
        selected_vectors = np.empty((count, self.dimension), dtype=self.dtype)
        discarded: list[int] = []
        for index in range(len(ids)):
            if len(selected) >= count:
                break
            e = int(ids[index])
            if selected:
                distances_to_selected = 1.0 - selected_vectors[: len(selected)] @ self.vectors[e]
                computed += len(selected)
                accept = bool((distances_to_selected > distances[index]).all())
            else:
                accept = True
            if accept:
                selected_vectors[len(selected)] = self.vectors[e]
                selected.append(e)
            else:
                discarded.append(e)
        if self.keep_pruned_connections:
            for e in discarded:
                if len(selected) >= count:
                    break
                selected.append(e)
        return np.array(selected, dtype=np.int64), computed

    def _select(
        self,
        base_vector: np.ndarray,
        ids: np.ndarray,
        distances: np.ndarray,
        count: int,
        layer: int,
    ) -> tuple[np.ndarray, int]:
        if self.use_heuristic:
            return self._select_neighbors_heuristic(base_vector, ids, distances, count, layer)
        return self._select_neighbors_simple(ids, distances, count), 0

    # ---- Algorithm 1: INSERT ----

    def insert(self, element: int) -> None:
        """要素 ``element`` を挿入する(Algorithm 1)。"""
        query = self.vectors[element]
        level = int(self.levels[element])
        construction = 0
        if self.entry_point < 0:  # 最初の要素
            self.entry_point, self.top_level = element, level
            self.num_inserted += 1
            return
        entry_ids = [self.entry_point]
        entry_distances = [float(1.0 - self.vectors[self.entry_point] @ query)]
        construction += 1
        for layer in range(self.top_level, level, -1):  # 層 L ... l + 1: ef = 1 の貪欲探索
            found, computed = self._search_layer(query, entry_ids, entry_distances, 1, layer)
            construction += computed
            entry_ids, entry_distances = [found[0][1]], [found[0][0]]
        for layer in range(min(self.top_level, level), -1, -1):  # 層 min(L, l) ... 0
            found, computed = self._search_layer(
                query, entry_ids, entry_distances, self.construction_search_width, layer
            )
            construction += computed
            found_ids = np.array([e for _, e in found], dtype=np.int64)
            found_distances = np.array([d for d, _ in found], dtype=np.float64)
            neighbors, compared = self._select(
                query, found_ids, found_distances, self.max_connections, layer
            )
            construction += compared
            self._set_neighbors(layer, element, neighbors)
            capacity = self.capacity(layer)
            for e in neighbors.tolist():  # 双方向の接続。上限を超えた近傍は再選択する
                row = self._row(layer, e)
                current = self.degree[layer][row]
                if current < capacity:
                    self.links[layer][row, current] = element
                    self.degree[layer][row] = current + 1
                else:
                    pool = np.append(self.links[layer][row, :current], element).astype(np.int64)
                    pool_distances = 1.0 - self.vectors[pool] @ self.vectors[e]
                    construction += len(pool)
                    order = np.argsort(pool_distances, kind="stable")
                    kept, compared = self._select(
                        self.vectors[e], pool[order], pool_distances[order], capacity, layer
                    )
                    construction += compared
                    self._set_neighbors(layer, e, kept)
            entry_ids = [e for _, e in found]  # ep <- W
            entry_distances = [d for d, _ in found]
        if level > self.top_level:
            self.entry_point, self.top_level = element, level
        self.num_inserted += 1
        self.counter.add("construction", construction)

    def add(self, count: int) -> None:
        """挿入の順序の続きから ``count`` 個の要素を挿入する。"""
        end = min(self.num_inserted + count, self.num_elements)
        for position in range(self.num_inserted, end):
            self.insert(int(self.insertion_order[position]))

    def inserted_elements(self) -> np.ndarray:
        """これまでに挿入した要素の番号(挿入の順序の先頭 ``num_inserted`` 個)。"""
        return self.insertion_order[: self.num_inserted]

    # ---- Algorithm 5: K-NN-SEARCH ----

    def search(
        self,
        query: np.ndarray,
        k: int,
        search_width: int,
        use_hierarchy: bool = True,
        entry_element: int | None = None,
        count_cost: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """query に近い上位 ``k`` 件を探す(Algorithm 5)。

        Args:
            query: 形状 ``(d,)``。
            k: 返す件数(``search_width`` 以下)。
            search_width: ``ef``。
            use_hierarchy: 偽なら、上の層を使わず、層 0 だけを ``entry_element`` から探索する(階層の
                効果を見るための診断)。
            entry_element: ``use_hierarchy`` が偽のときの入口の要素(挿入済みの要素)。
            count_cost: 真なら距離計算の回数を ``counter`` の ``"search"`` に加える。

        Returns:
            ``(上位 k 件の要素の番号、その距離、この query の距離計算の回数)``。足りない場合は
            ``-1`` で
            埋める。
        """
        if k > search_width:
            raise ValueError("k は search_width 以下")
        if self.entry_point < 0:
            raise ValueError("索引が空")
        query = np.asarray(query, dtype=self.dtype)
        computed = 1
        if use_hierarchy:
            entry = self.entry_point
            entry_ids = [entry]
            entry_distances = [float(1.0 - self.vectors[entry] @ query)]
            for layer in range(self.top_level, 0, -1):
                found, count = self._search_layer(query, entry_ids, entry_distances, 1, layer)
                computed += count
                entry_ids, entry_distances = [found[0][1]], [found[0][0]]
        else:
            if entry_element is None:
                raise ValueError("use_hierarchy が偽のときは entry_element が必要")
            entry_ids = [int(entry_element)]
            entry_distances = [float(1.0 - self.vectors[entry_element] @ query)]
        found, count = self._search_layer(query, entry_ids, entry_distances, search_width, 0)
        computed += count
        ids = np.full(k, -1, dtype=np.int64)
        distances = np.full(k, np.inf, dtype=np.float64)
        for position, (distance, e) in enumerate(found[:k]):
            ids[position], distances[position] = e, distance
        if count_cost:
            self.counter.add("search", computed)
        return ids, distances, computed

    def search_batch(
        self,
        queries: np.ndarray,
        k: int,
        search_width: int,
        use_hierarchy: bool = True,
        entry_elements: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """複数の query を 1 つずつ探索する。

        Returns:
            ``(上位 k 件の要素の番号 (Q, k)、query ごとの距離計算の回数 (Q,))``。
        """
        ids = np.empty((len(queries), k), dtype=np.int64)
        costs = np.empty(len(queries), dtype=np.int64)
        for row, query in enumerate(queries):
            entry = None if entry_elements is None else int(entry_elements[row])
            ids[row], _, costs[row] = self.search(
                query, k, search_width, use_hierarchy=use_hierarchy, entry_element=entry
            )
        return ids, costs

    def layer_sizes(self) -> list[int]:
        """挿入済みの要素のうち、各層に属する要素の数(層 0 が全要素)。"""
        inserted_levels = self.levels[self.inserted_elements()]
        return [int((inserted_levels >= layer).sum()) for layer in range(self.top_level + 1)]
