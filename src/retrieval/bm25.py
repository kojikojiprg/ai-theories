"""BM25 のスクラッチ実装(024)。

BM25(Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", Foundations and
Trends in Information Retrieval, 2009)は、語の一致に基づく検索のスコア関数である。query ``q`` の語
``t`` の集合について、文書 ``D`` のスコアを次の和で与える。

.. math::

    \\mathrm{BM25}(D, q) = \\sum_{t \\in q} \\mathrm{IDF}(t) \\cdot
    \\frac{f(t, D) \\, (k_1 + 1)}{f(t, D) + k_1 \\left(1 - b + b \\, \\lvert D \\rvert /
    \\mathrm{avgdl}\\right)}

.. math::

    \\mathrm{IDF}(t) = \\ln\\left(1 + \\frac{N - n(t) + 0.5}{n(t) + 0.5}\\right)

- ``f(t, D)``: 語 ``t`` の文書 ``D`` での出現回数(term frequency)。``|D|``: 文書の長さ(トークン数)。
  ``avgdl``: 全文書の平均の長さ。``N``: 文書の数。``n(t)``: 語 ``t`` を含む文書の数(document
  frequency)。``k_1``・``b``: 定数(既定値 1.2・0.75)。
- 逆文書頻度は、Robertson–Spärck Jones の式に 1 を加えて常に正にした形(Lucene の実装と同じ)。
- query の語は **重複を除いた集合** として和をとる(query の中の出現回数は使わない)。

**入力はトークン ID の列**(語彙は BPE のトークン)。文書は長さの異なる列でよい。本トピックの
passage は固定長なので、``|D| = avgdl`` となり、長さの正規化の項は定数 1 になる。したがって
``b`` は結果に影響しない。

**実装**: 転置索引(inverted index)を ``(語, 文書)`` の組を並べた配列として持ち、各組の寄与
``(語 t の逆文書頻度) f (k_1 + 1) / (f + k_1 (1 - b + b |D| / avgdl))`` を **索引の構築時に**
計算しておく。
query のスコアは、query の各語の転置リストの寄与を文書ごとに足すだけである。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class BM25Index:
    """BM25 の転置索引。

    Args:
        documents: 文書ごとのトークン ID の列(長さが異なってもよい)。
        vocabulary_size: 語彙サイズ(トークン ID は ``[0, vocabulary_size)``)。
        k1: 語の出現回数の飽和の強さ ``k_1``。
        b: 文書の長さの正規化の強さ。
    """

    def __init__(
        self,
        documents: Sequence[Sequence[int]] | np.ndarray,
        vocabulary_size: int,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.num_documents = len(documents)
        self.vocabulary_size = vocabulary_size
        self.k1 = k1
        self.b = b
        lengths = np.array([len(d) for d in documents], dtype=np.float64)
        self.document_lengths = lengths
        self.average_length = float(lengths.mean())

        # (語, 文書) の組と出現回数を、語の昇順・文書の昇順に並べる
        keys, frequencies = np.unique(
            np.concatenate(
                [
                    np.asarray(d, dtype=np.int64) * self.num_documents + i
                    for i, d in enumerate(documents)
                ]
            ),
            return_counts=True,
        )
        posting_terms = keys // self.num_documents
        self.posting_documents = (keys % self.num_documents).astype(np.int64)
        self.document_frequency = np.bincount(posting_terms, minlength=vocabulary_size)
        # 語 t の転置リストは [pointer[t], pointer[t + 1])
        self.pointer = np.concatenate([[0], np.cumsum(self.document_frequency)]).astype(np.int64)
        self.inverse_document_frequency = np.log(
            1.0
            + (self.num_documents - self.document_frequency + 0.5) / (self.document_frequency + 0.5)
        )
        normalizer = k1 * (1.0 - b + b * lengths[self.posting_documents] / self.average_length)
        self.posting_contributions = (
            self.inverse_document_frequency[posting_terms]
            * frequencies
            * (k1 + 1.0)
            / (frequencies + normalizer)
        )

    def score(self, query: Sequence[int]) -> np.ndarray:
        """1 つの query の、全文書のスコア(形状 ``(num_documents,)``、float64)。"""
        scores = np.zeros(self.num_documents, dtype=np.float64)
        for term in np.unique(np.asarray(query, dtype=np.int64)):
            begin, end = self.pointer[term], self.pointer[term + 1]
            # 1 つの語の転置リストでは文書が重複しないので、添字付きの加算でよい
            scores[self.posting_documents[begin:end]] += self.posting_contributions[begin:end]
        return scores

    def score_batch(self, queries: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        """複数の query のスコア(形状 ``(len(queries), num_documents)``、float32)。"""
        return np.stack([self.score(q) for q in queries]).astype(np.float32)
