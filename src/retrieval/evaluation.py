"""検索の評価指標と、埋め込みの幾何の診断量(024)。

**評価指標**(query ``i`` について、正解の集合を ``G_i``、候補を内積(または BM25 のスコア)の降順
に並べたときの最も上位の正解の順位を ``rank_i`` とする。**query を含む passage は候補から除く
**):

- ``Recall@k``: 上位 ``k`` 件に正解が 1 つ以上あれば 1、なければ 0(query について平均する)。
  ``1[rank_i <= k]``。

- ``MRR`` (Mean Reciprocal Rank): ``1 / rank_i`` の query についての平均。

- ``nDCG@k`` (normalized Discounted Cumulative Gain): 二値の関連度 ``rel_j`` について
  ``DCG@k = sum_{j=1}^{k} rel_j / log2(j + 1)``、
  ``IDCG@k = sum_{j=1}^{min(|G_i|, k)} 1 / log2(j + 1)``、``nDCG@k = DCG@k / IDCG@k``。

**同点の扱い**: 順位は、最も上位の正解と **同じ以上のスコアを持つ正解でない候補** の数に 1 を加
えた値とする(同点は正解に不利に数える)。埋め込みの内積は実数なので、同点は同一の passage が複数
ある場合などに限られる。

**探索は全件の内積による厳密な探索** である(近似最近傍探索は使わない)。

**埋め込みの幾何**(Wang & Isola, "Understanding Contrastive Representation Learning through
Alignment and Uniformity on the Hypersphere", ICML 2020):

- ``alignment`` = ``E[ || f(x) - f(y) ||^alpha ]``(``(x, y)`` は正例の組、既定 ``alpha = 2``)。
  小さいほど正例が近い。

- ``uniformity`` = ``log E[ exp(-t || f(x) - f(y) ||^2) ]``(``x, y`` は独立な標本、既定
  ``t = 2``)。小さい(より負)ほど、単位球面上に一様に広がっている。

- 異方性(anisotropy): ランダムな組のコサイン類似度の平均(Ethayarajh, "How Contextual are
  Contextualized Word Representations?", EMNLP 2019)。1 に近いほど、埋め込みが狭い錐に偏っている。

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.data.retrieval import RetrievalSet


@dataclass
class RankingResult:
    """query ごとの順位と指標。

    Attributes:
        rank: 形状 ``(Q,)``。最も上位の正解の順位(1 始まり)。
        ndcg_at_10: 形状 ``(Q,)``。nDCG@10。
        num_positives: 形状 ``(Q,)``。正解の数。
        num_candidates: 候補の数(query の元の passage を除いた索引の大きさ)。
    """

    rank: np.ndarray
    ndcg_at_10: np.ndarray
    num_positives: np.ndarray
    num_candidates: int

    def hits(self, k: int = 10) -> np.ndarray:
        """query ごとの Recall@k (0 または 1、bool)。"""
        return self.rank <= k

    def recall_at(self, k: int = 10) -> float:
        return float(self.hits(k).mean())

    def mean_reciprocal_rank(self) -> float:
        return float((1.0 / self.rank).mean())

    def mean_ndcg_at_10(self) -> float:
        return float(self.ndcg_at_10.mean())


@torch.no_grad()
def rank_candidates(
    scores: Tensor,
    query_articles: np.ndarray,
    passage_articles: np.ndarray,
    query_sources: np.ndarray,
    chunk_size: int = 512,
) -> RankingResult:
    """スコア行列から query ごとの順位と nDCG@10 を求める。

    Args:
        scores: 形状 ``(Q, P)`` のスコア(内積または BM25。大きいほど上位)。
        query_articles: 形状 ``(Q,)``。query の元の記事。
        passage_articles: 形状 ``(P,)``。passage の記事。
        query_sources: 形状 ``(Q,)``。query を含む passage の添字(候補から除く)。
    """
    scores = torch.as_tensor(scores)
    device = scores.device
    passage_articles_t = torch.as_tensor(passage_articles, device=device)
    num_queries, num_passages = scores.shape
    log_discount = np.log2(np.arange(2, 12, dtype=np.float64))  # 位置 1〜10 の割引 log2(j + 1)
    ideal_cumulative = np.cumsum(1.0 / log_discount)  # 先頭 j 個が正解のときの DCG
    ranks, ndcgs, positives = [], [], []
    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        s = scores[start:end].float().clone()
        rows = torch.arange(end - start, device=device)
        sources = torch.as_tensor(query_sources[start:end], device=device)
        articles = torch.as_tensor(query_articles[start:end], device=device)
        is_positive = passage_articles_t[None, :] == articles[:, None]
        is_positive[rows, sources] = False
        s[rows, sources] = float("-inf")
        num_positive = is_positive.sum(dim=1)
        if bool((num_positive == 0).any()):
            raise ValueError("正解が 1 つもない query がある")
        best_positive = s.masked_fill(~is_positive, float("-inf")).max(dim=1).values
        ahead = ((s >= best_positive[:, None]) & ~is_positive).sum(dim=1)
        ranks.append((1 + ahead).cpu())
        top = s.topk(min(10, num_passages), dim=1).indices
        relevance = np.zeros((end - start, 10))  # 候補が 10 件に満たない場合は残りを 0 で埋める
        found = is_positive.gather(1, top).cpu().numpy()
        relevance[:, : found.shape[1]] = found
        num_positive = num_positive.cpu()
        dcg = (relevance / log_discount[None, :]).sum(
            axis=1
        )  # MPS は FP64 を扱えないので CPU で計算する
        ideal = ideal_cumulative[np.minimum(num_positive.numpy(), 10) - 1]
        ndcgs.append(dcg / ideal)
        positives.append(num_positive)
    return RankingResult(
        rank=torch.cat(ranks).numpy().astype(np.int64),
        ndcg_at_10=np.concatenate(ndcgs),
        num_positives=torch.cat(positives).numpy().astype(np.int64),
        num_candidates=num_passages - 1,
    )


@torch.no_grad()
def evaluate_embeddings(
    query_embeddings: Tensor, passage_embeddings: Tensor, retrieval_set: RetrievalSet
) -> RankingResult:
    """正規化済みの埋め込みの内積(全件の厳密な探索)で、評価用の索引と query を評価する。"""
    scores = query_embeddings.float() @ passage_embeddings.float().t()
    return rank_candidates(
        scores,
        retrieval_set.query_articles,
        retrieval_set.passage_articles,
        retrieval_set.query_sources,
    )


def expected_random_recall(num_positives: np.ndarray, num_candidates: int, k: int = 10) -> float:
    """候補をランダムに並べたときの Recall@k の期待値(query について平均)。

    正解が ``g`` 個、候補が ``M`` 個のとき、上位 ``k`` 件に正解が 1 つもない確率は
    ``C(M - g, k) / C(M, k) = prod_{i=0}^{k-1} (M - g - i) / (M - i)`` なので、Recall@k の期待値は
    ``1 - prod_{i=0}^{k-1} (M - g - i) / (M - i)`` である。``M - g < k`` のとき積は 0 になる
    (因子 ``M - g - i`` が 0 を通る)。積の形で計算するので、丸め誤差は項数 ``k`` x
    丸めの単位程度である。
    """
    index = np.arange(k, dtype=np.float64)
    values = []
    for g in np.asarray(num_positives, dtype=np.int64):
        if num_candidates - g < k:
            values.append(1.0)
        else:
            values.append(
                1.0 - float(np.prod((num_candidates - g - index) / (num_candidates - index)))
            )
    return float(np.mean(values))


def mean_pairwise_cosine(embeddings: Tensor) -> float:
    """埋め込みの、異なる 2 つの組すべてのコサイン類似度の平均(異方性)。

    L2 正規化済みの埋め込みを渡す。
    ``sum_{i != j} e_i . e_j = || sum_i e_i ||^2 - sum_i || e_i ||^2`` を使って、全
    ``n (n - 1)`` 組の平均を厳密に求める。ノルムは 1 とみなさず、実際の値 ``sum_i || e_i ||^2`` を
    引く(FP32 で正規化した埋め込みのノルムは 1 から約 ``1e-7`` ずれる)。FP64 で計算する。
    """
    e = embeddings.detach().cpu().double()
    n = e.size(0)
    total = e.sum(dim=0)
    return float((total.dot(total) - e.pow(2).sum()) / (n * (n - 1)))


def alignment(embeddings_x: Tensor, embeddings_y: Tensor, alpha: float = 2.0) -> float:
    """正例の組 ``(x_i, y_i)`` の ``E[ || f(x) - f(y) ||^alpha ]`` (Wang & Isola)。FP64。"""
    difference = embeddings_x.detach().cpu().double() - embeddings_y.detach().cpu().double()
    return float(difference.norm(dim=1).pow(alpha).mean())


def uniformity(embeddings: Tensor, t: float = 2.0) -> float:
    """``log E[ exp(-t || f(x) - f(y) ||^2) ]`` (Wang & Isola)。異なる 2
    つの組すべてについて平均する。FP64。"""
    squared = torch.pdist(embeddings.detach().cpu().double(), p=2).pow(2)
    return float(squared.mul(-t).exp().mean().log())
