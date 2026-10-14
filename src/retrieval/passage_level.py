"""passage 単位の検索の評価と、2 段階の検索の部品(026)。

**passage 単位の関連性**: query ごとに、正例は **ちょうど 1 つの passage**(query の文を抜いた元の
passage)である。同じ記事の他の passage は正例ではない。

**同点は正例に不利に数える**: 正例の順位は ``1 + (スコアが正例以上の、正例でない候補の数)`` で、
同点では正例でない候補を上位とする(025 と同じ。定数を出力するモデルの Recall@10 が偶然の水準以下
になる)。並べ替えの前の順位(第 1 段の順位)は、同点の解決に使わない。

このモジュールは、単一の正例の全件探索の順位(``rank_single_positive()``)、可変長の
cross-encoder による
候補のスコア(``score_candidates_variable()``)、並べ替えの順位(``rerank_single_positive()``。025 の
``rank_reranked_candidates()`` を、passage
の識別子そのものを「記事」として使って再利用する)を提供する。

記号 / Notation:
    Q : query の数、P : 索引の passage の数、K : 並べ替える候補の数
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.retrieval.evaluation import RankingResult, rank_candidates
from src.retrieval.reranking import RerankingResult, rank_reranked_candidates


@torch.no_grad()
def rank_single_positive(
    query_embeddings: Tensor,
    passage_embeddings: Tensor,
    positive: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """全件の内積で、各 query の正例(1 つの passage)の順位を求める(同点は正例に不利)。

    導入: 026。

    Args:
        query_embeddings: 形状 ``(Q, d)``(単位ベクトル)。
        passage_embeddings: 形状 ``(P, d)``。
        positive: 形状 ``(Q,)``。正例の passage の添字。
        chunk_size: 1 回に処理する query の数(結果には影響しない)。
    Returns:
        形状 ``(Q,)`` の順位(1 始まり)。
    """
    ranks = []
    for start in range(0, query_embeddings.size(0), chunk_size):
        end = min(start + chunk_size, query_embeddings.size(0))
        scores = query_embeddings[start:end].float() @ passage_embeddings.float().t()
        index = torch.as_tensor(positive[start:end], device=scores.device)
        target = scores.gather(1, index[:, None])
        ahead = (scores >= target).sum(dim=1) - 1  # 正例自身を除く
        ranks.append((1 + ahead).cpu())
    return torch.cat(ranks).numpy().astype(np.int64)


def rerank_single_positive(
    scores: np.ndarray, candidate_ids: np.ndarray, positive: np.ndarray, num_passages: int
) -> RerankingResult:
    """候補のスコアで並べ替えた後の順位と指標(正例は 1 つの passage。同点は正例に不利)。

    導入: 026。

    Args:
        scores: 形状 ``(Q, K)``。候補のスコア(大きいほど上位)。
        candidate_ids: 形状 ``(Q, K)``。候補の passage の添字。
        positive: 形状 ``(Q,)``。正例の passage の添字。
        num_passages: 索引の passage の数。

    Returns:
        並べ替えた後の順位と指標(``RerankingResult``)。
    """
    return rank_reranked_candidates(
        scores,
        candidate_ids,
        np.asarray(positive),
        np.arange(num_passages),
        np.ones(len(positive), dtype=np.int64),
    )


@torch.no_grad()
def score_candidates_variable(
    model,
    query_tokens: Tensor,
    query_lengths: np.ndarray,
    passage_tokens: Tensor,
    passage_lengths: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    batch_size: int,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """可変長の cross-encoder で、各 query の候補のスコア ``(Q, K)`` を求める(組ごとに 1
    回の順伝播)。

    組を、列の長さの昇順に並べ替えてバッチにし(パディングの計算を減らす)、結果を元の並びに戻す。
    呼び出しの前後でモデルのモードを変えない。

    導入: 026。

    Args:
        model: ``VariableLengthCrossEncoder``。
        query_tokens, query_lengths: 形状 ``(Qn, L_q)`` と ``(Qn,)``(query の表。``query_ids``
        で引く)。
        passage_tokens, passage_lengths: 形状 ``(P, L_p)`` と ``(P,)``。
        query_ids: 形状 ``(Q,)``。
        candidate_ids: 形状 ``(Q, K)``。
        batch_size: 1 回の順伝播の組の数(結果に影響しない)。
        use_fp16_autocast: 真なら本体を FP16 の autocast で計算する(CUDA のみ)。

    Returns:
        形状 ``(Q, K)`` のスコア(float32)。
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    num_queries, k = candidate_ids.shape
    flat_queries = np.repeat(np.asarray(query_ids), k)
    flat_passages = candidate_ids.reshape(-1)
    total = query_lengths[flat_queries] + passage_lengths[flat_passages]
    order = np.argsort(total, kind="stable")
    scores = np.empty(len(order), dtype=np.float32)
    try:
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            ql = query_lengths[flat_queries[index]]
            pl = passage_lengths[flat_passages[index]]
            q = query_tokens[torch.as_tensor(flat_queries[index])][:, : int(ql.max())].to(device)
            p = passage_tokens[torch.as_tensor(flat_passages[index])][:, : int(pl.max())].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
            ):
                out = model(
                    q, p, torch.as_tensor(ql, device=device), torch.as_tensor(pl, device=device)
                )
            scores[index] = out.float().cpu().numpy()
    finally:
        model.train(was_training)
    return scores.reshape(num_queries, k)


@torch.no_grad()
def rank_article_label(
    query_embeddings: Tensor,
    passage_embeddings: Tensor,
    query_positions: np.ndarray,
    passage_articles: np.ndarray,
    chunk_size: int = 512,
) -> RankingResult:
    """同じ索引・同じ query で、関連のラベルを記事単位にした順位(024 と同じ定義)。

    正解は、query と同じ記事の、query の元の passage 以外の passage。query の元の passage 自身は
    候補から除く(正解でも不正解でもない)。順位は、最もスコアの高い正解の順位(同点は正解に不利)で、
    Recall@k は上位 k 件に正解が 1 つ以上ある query の割合。

    導入: 026。

    Args:
        query_embeddings: 形状 ``(Q, d)``。
        passage_embeddings: 形状 ``(P, d)``。
        query_positions: 形状 ``(Q,)``。各 query の元の passage の、索引の中の位置。
        passage_articles: 形状 ``(P,)``。索引の passage ごとの記事。
        chunk_size: 1 回に処理する query の数(結果には影響しない)。

    Returns:
        query ごとの順位と nDCG@10(``RankingResult``)。
    """
    positions = np.asarray(query_positions)
    articles = np.asarray(passage_articles)
    pieces = []
    for start in range(0, query_embeddings.size(0), chunk_size):
        end = min(start + chunk_size, query_embeddings.size(0))
        scores = query_embeddings[start:end].float() @ passage_embeddings.float().t()
        pieces.append(
            rank_candidates(
                scores, articles[positions[start:end]], articles, positions[start:end], chunk_size
            )
        )
    return RankingResult(
        rank=np.concatenate([p.rank for p in pieces]),
        ndcg_at_10=np.concatenate([p.ndcg_at_10 for p in pieces]),
        num_positives=np.concatenate([p.num_positives for p in pieces]),
        num_candidates=pieces[0].num_candidates,
    )
