"""並べ替え(reranking)の推論と評価(025)。

**2 段階の検索(retrieve-then-rerank)**: 第 1 段が全ベクトルから上位 ``K`` 件の候補を絞り込み、第
2 段がその ``K`` 件だけをスコアの高い順に並べ替える。

**費用のモデル**(1 query あたり):

.. math::

    C_{\\mathrm{total}} = C_{\\mathrm{first}} + K \\cdot F_{\\mathrm{pair}}

``C_first`` は第 1 段の距離計算の回数(全探索なら ``N``)、``K`` は並べ替える候補の数、``F_pair``は
1 組あたりの順伝播のコストである。cross-encoder は組ごとに順伝播が要るので、第
2 段の順伝播の回数は ``K``回である。dual encoder で並べ替える場合は、passage
の埋め込みを事前に計算してあれば、query の順伝播 1 回と``K`` 回の内積になる(``F_pair`` を内積の
1 回とみなせる)。cross-encoder の ``F_pair`` は 1 回の内積よりはるかに大きい(列の長さ
``L_q + L_p + 2`` のトークンぶんの順伝播)ので、``K`` を小さくできるなら第 1 段の``C_first`` より第
2 段の費用が支配的になりうる。

**指標**: 024 と同じ関連性の定義(同じ記事の、元の passage 以外の passage が正例)と指標(Recall@10、
Mean Reciprocal Rank、normalized Discounted Cumulative Gain の上位 10 件の値)を使う。
並べ替えの対象は第 1 段の上位 ``K`` 件だけなので、候補に正例が 1 つもない query の順位は
``K + 1``(候補の外)とし、平均逆順位では逆数を 0 とみなす(024 の索引全体の順位と異なり、
候補の外の順位は定義されない)。同点のスコアは、第
1 段の順位(候補の並び)が先の候補を上位とする(正例に有利・不利のどちらにも偏らない)。

記号 / Notation:
    Q : query の数、K : 並べ替える候補の数、L_q・L_p : query・passage の長さ、G_i : query i
    の正解の集合
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.models.cross_encoder import CrossEncoder
from src.models.text_embedding import TextEmbeddingModel, truncate_and_normalize


@dataclass
class RerankingResult:
    """並べ替えの後の query ごとの順位と指標。

    Attributes:
        rank: 形状 ``(Q,)``。候補を並べ替えた後の、最も上位の正解の順位(1 始まり。
        候補に正解がなければ ``K + 1``)。
        normalized_discounted_cumulative_gain_at_10: 形状 ``(Q,)``。normalized Discounted
        Cumulative Gain の上位 10 件の値。
        num_candidates: 並べ替えた候補の数 ``K``。
    """

    rank: np.ndarray
    normalized_discounted_cumulative_gain_at_10: np.ndarray
    num_candidates: int

    def hits(self, k: int = 10) -> np.ndarray:
        """query ごとの Recall@k(0 または 1、bool)。"""
        return self.rank <= k

    def recall_at(self, k: int = 10) -> float:
        return float(self.hits(k).mean())

    def coverage(self) -> float:
        """候補に正解が 1 つ以上入っている query の割合(並べ替えで到達できる Recall の上限)。"""
        return float((self.rank <= self.num_candidates).mean())

    def mean_reciprocal_rank(self) -> float:
        reciprocal = np.where(self.rank <= self.num_candidates, 1.0 / self.rank, 0.0)
        return float(reciprocal.mean())

    def mean_normalized_discounted_cumulative_gain_at_10(self) -> float:
        return float(self.normalized_discounted_cumulative_gain_at_10.mean())


def rank_reranked_candidates(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    query_articles: np.ndarray,
    passage_articles: np.ndarray,
    num_positives: np.ndarray,
) -> RerankingResult:
    """候補のスコアで並べ替えた後の順位と指標を求める。

    Args:
        scores: 形状 ``(Q, K)``。候補のスコア(大きいほど上位)。
        candidate_ids: 形状 ``(Q, K)``。候補のコーパスの passage の添字(第 1 段の順位の順)。
        query_articles: 形状 ``(Q,)``。query の記事。
        passage_articles: コーパスの passage ごとの記事。
        num_positives: 形状 ``(Q,)``。query ごとの正解の数(コーパス全体での、同じ記事の他の
        passage の数)。
    """
    num_queries, k = candidate_ids.shape
    is_positive = passage_articles[candidate_ids] == query_articles[:, None]
    order = np.argsort(-scores.astype(np.float64), axis=1, kind="stable")
    reordered = np.take_along_axis(is_positive, order, axis=1)
    any_positive = reordered.any(axis=1)
    rank = np.where(any_positive, reordered.argmax(axis=1) + 1, k + 1).astype(np.int64)
    log_discount = np.log2(np.arange(2, 12, dtype=np.float64))
    top = np.zeros((num_queries, 10))
    width = min(10, k)
    top[:, :width] = reordered[:, :width]
    dcg = (top / log_discount[None, :]).sum(axis=1)
    ideal_cumulative = np.cumsum(1.0 / log_discount)
    ideal = ideal_cumulative[np.minimum(num_positives, 10) - 1]
    return RerankingResult(
        rank=rank, normalized_discounted_cumulative_gain_at_10=dcg / ideal, num_candidates=k
    )


@torch.no_grad()
def score_candidates_cross_encoder(
    model: CrossEncoder,
    query_tokens: Tensor,
    passage_tokens: Tensor,
    candidate_ids: np.ndarray,
    batch_size: int,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """cross-encoder で、各 query の候補のスコア ``(Q, K)`` を求める(組ごとに 1 回の順伝播)。

    Args:
        model: ``CrossEncoder``(呼び出しの間 ``eval()`` にし、終わったら元のモードに戻す)。
        query_tokens: 形状 ``(Q, L_q)``。
        passage_tokens: 形状 ``(P, L_p)``(コーパスの passage。``model`` と同じデバイスまたは CPU)。
        candidate_ids: 形状 ``(Q, K)``。
        batch_size: 1 回の順伝播の組の数(結果に影響しない)。
        use_fp16_autocast: 真なら本体を FP16 の autocast で計算する(CUDA のみ。ヘッドは FP32)。
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    num_queries, k = candidate_ids.shape
    flat_queries = np.repeat(np.arange(num_queries), k)
    flat_passages = candidate_ids.reshape(-1)
    scores = np.empty(len(flat_queries), dtype=np.float32)
    try:
        for start in range(0, len(flat_queries), batch_size):
            end = min(start + batch_size, len(flat_queries))
            queries = query_tokens[torch.as_tensor(flat_queries[start:end])].to(device)
            passages = passage_tokens[torch.as_tensor(flat_passages[start:end])].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
            ):
                values = model(queries, passages)
            scores[start:end] = values.float().cpu().numpy()
    finally:
        model.train(was_training)
    return scores.reshape(num_queries, k)


@torch.no_grad()
def score_candidates_dual_encoder(
    model: TextEmbeddingModel,
    query_tokens: Tensor,
    passage_tokens: Tensor,
    candidate_ids: np.ndarray,
    batch_size: int,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """dual encoder で、各 query の候補のスコア(埋め込みの内積)``(Q, K)`` を求める。

    候補に現れる passage を 1 回ずつ符号化する(passage の埋め込みは事前に計算できる)。
    埋め込みの正規化と内積は FP32 で計算する。
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    unique, inverse = np.unique(candidate_ids, return_inverse=True)
    inverse = inverse.reshape(candidate_ids.shape)
    try:
        passage_embeddings = []
        for start in range(0, len(unique), batch_size):
            batch = passage_tokens[torch.as_tensor(unique[start : start + batch_size])].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
            ):
                pooled = model.pooled(batch)
            passage_embeddings.append(truncate_and_normalize(pooled.float()))
        query_embeddings = []
        for start in range(0, query_tokens.size(0), batch_size):
            batch = query_tokens[start : start + batch_size].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
            ):
                pooled = model.pooled(batch)
            query_embeddings.append(truncate_and_normalize(pooled.float()))
    finally:
        model.train(was_training)
    passage_matrix = torch.cat(passage_embeddings)
    query_matrix = torch.cat(query_embeddings)
    gathered = passage_matrix[torch.as_tensor(inverse, device=device)]  # (Q, K, d)
    return torch.einsum("qkd,qd->qk", gathered, query_matrix).float().cpu().numpy()


def score_candidates(
    model,
    kind: str,
    query_tokens: Tensor,
    passage_tokens: Tensor,
    candidate_ids: np.ndarray,
    batch_size: int,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """``kind``(cross_encoder か dual_encoder)で、候補のスコアを求める関数を切り替える。"""
    if kind == "cross_encoder":
        return score_candidates_cross_encoder(
            model, query_tokens, passage_tokens, candidate_ids, batch_size, use_fp16_autocast
        )
    if kind == "dual_encoder":
        return score_candidates_dual_encoder(
            model, query_tokens, passage_tokens, candidate_ids, batch_size, use_fp16_autocast
        )
    raise ValueError(f"kind は 'cross_encoder' か 'dual_encoder': {kind!r}")
