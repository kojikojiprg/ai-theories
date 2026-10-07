"""並べ替え(reranking)の学習データと候補の構成(025)。

024 の評価用の索引と query(``src/data/retrieval.py`` の ``build_retrieval_set()``)を、
使う記事のすべて(9,470 記事)に対して構成した集合を、検索の対象(コーパス)とする。passage
は記事ごとに最大 12 個で、query はコーパスの passage の一部(長さ ``L_q``)、**関連性の定義は
024 と同じ**(query と同じ記事の、元の passage 以外の passage が正例)である。

**並べ替えの学習の組**: 並べ替えモデルの学習データは **学習用の記事のみ** から作る。
1 つの組は``(query、正例、負例 n 個)`` で、

- query: 学習用の記事の query(コーパスの query のうち、学習用の記事から切り出したもの)。
- 正例: その query の記事の、元の passage 以外の passage から一様に 1 つ。
- 困難な負例(hard negative): 第 1 段の検索器(024 のモデルによる全探索)で、**学習用の記事の passage
  だけ**を対象にして上位 ``K_mine`` 件を求め、そのうち **query と同じ記事の passage
  を除いた**残りから、一様に``n`` 個を選ぶ(偽の負例の抑制。同じ記事の passage
  は正例なので負例にしない)。推論時に並べ替える候補は、第 1 段の上位 ``K`` 件なので、
  ``K_mine = K`` にして、学習時の負例の分布を推論時の入力の分布に揃える。
- ランダムな負例: 学習用の記事の passage から、query と同じ記事のものを除いて、一様に ``n`` 個。

同じシードでは、query と正例の系列は **負例の種類によらず同じ**である(困難な負例とランダムな負例
で、乱数の生成器を分ける)。したがって、困難な負例の条件とランダムな負例の条件は、同じ query
と正例を同じ順序で見る。

記号 / Notation:
    L_q : query の長さ、L_p : passage の長さ、n : 負例の数、K : 並べ替える候補の数、
    K_mine : 困難な負例を採掘する上位の件数、T : 学習ステップ数、B : 1 ステップの query の数
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.retrieval.exact_search import exact_search


@dataclass
class RerankingSchedule:
    """並べ替えの学習の組(学習の前に全ステップぶんを作る)。

    Attributes:
        query: 形状 ``(T, B)``。コーパスの query の添字。
        positive: 形状 ``(T, B)``。正例のコーパスの passage の添字。
        hard_negatives: 形状 ``(T, B, n)``。困難な負例のコーパスの passage の添字。
        random_negatives: 形状 ``(T, B, n)``。ランダムな負例のコーパスの passage の添字。
        num_same_article_removed: 採掘の上位 ``K_mine`` 件から、同じ記事の passage
        として除いた件数の合計。
        num_mined_considered: 採掘の上位 ``K_mine`` 件の、調べた件数の合計(``T * B * K_mine``)。
        num_fallback_queries: 同じ記事を除いた残りが ``n`` 個に満たず、
        ランダムな負例で補った組の数。
    """

    query: np.ndarray
    positive: np.ndarray
    hard_negatives: np.ndarray
    random_negatives: np.ndarray
    num_same_article_removed: int
    num_mined_considered: int
    num_fallback_queries: int

    def digest(self) -> str:
        """query と正例の系列の SHA-256(負例の種類によらない。同じシードで組が揃うことの確認)。"""
        h = hashlib.sha256()
        for array in (self.query, self.positive):
            h.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
        return h.hexdigest()

    def negatives(self, kind: str) -> np.ndarray:
        if kind == "hard":
            return self.hard_negatives
        if kind == "random":
            return self.random_negatives
        raise ValueError(f"負例の種類は 'hard' か 'random': {kind!r}")


def article_passage_ranges(passage_articles: np.ndarray) -> dict[int, tuple[int, int]]:
    """記事ごとの passage の添字の範囲 ``{記事: (先頭, 末尾 + 1)}``。

    ``build_retrieval_set()`` は記事の昇順に passage を並べるので、同じ記事の passage
    は連続している。
    """
    ranges: dict[int, tuple[int, int]] = {}
    start = 0
    for position in range(1, len(passage_articles) + 1):
        if (
            position == len(passage_articles)
            or passage_articles[position] != passage_articles[start]
        ):
            ranges[int(passage_articles[start])] = (start, position)
            start = position
    return ranges


@torch.no_grad()
def mine_candidate_lists(
    query_embeddings: Tensor,
    passage_embeddings: Tensor,
    pool_passages: np.ndarray,
    query_sources: np.ndarray,
    num_candidates: int,
    chunk_size: int = 2048,
) -> np.ndarray:
    """各 query について、``pool_passages`` だけを対象に全探索し、上位 ``num_candidates``件を返す。

    Args:
        query_embeddings: 形状 ``(Q, d)``(単位ベクトル。デバイスは問わない)。
        passage_embeddings: 形状 ``(P, d)``(コーパス全体の passage の埋め込み)。
        pool_passages: 探索の対象にする passage の添字(昇順)。
        query_sources: 形状 ``(Q,)``。各 query の元の passage の添字(候補から除く)。
        num_candidates: 件数。
        chunk_size: 一度に処理する query の数(結果に影響しない)。

    Returns:
        形状 ``(Q, num_candidates)``。コーパスの passage の添字(内積の大きい順)。
    """
    pool = torch.as_tensor(pool_passages, device=query_embeddings.device)
    pool_vectors = passage_embeddings[pool]
    position_in_pool = {int(g): i for i, g in enumerate(pool_passages.tolist())}
    excluded = np.array([position_in_pool.get(int(s), -1) for s in query_sources])
    local, _ = exact_search(query_embeddings, pool_vectors, num_candidates, excluded, chunk_size)
    return np.asarray(pool_passages)[local]


def sample_reranking_schedule(
    train_queries: np.ndarray,
    mined_candidates: np.ndarray,
    query_articles: np.ndarray,
    query_sources: np.ndarray,
    passage_articles: np.ndarray,
    pool_passages: np.ndarray,
    num_steps: int,
    batch_queries: int,
    num_negatives: int,
    seed: int,
) -> RerankingSchedule:
    """学習全体の組を、学習の前に作る(決定的)。

    Args:
        train_queries: 学習用の記事の query の、コーパスの query の添字(昇順)。
        mined_candidates: 形状 ``(len(train_queries), K_mine)``。各学習用の query の、第 1 段の上位
            ``K_mine`` 件(``mine_candidate_lists()``。``train_queries`` と同じ並び)。
        query_articles: コーパスの query ごとの記事の番号。
        query_sources: コーパスの query ごとの元の passage の添字。
        passage_articles: コーパスの passage ごとの記事の番号(記事の昇順に並んでいる)。
        pool_passages: ランダムな負例・補いの対象にする passage の添字(学習用の記事の passage)。
        num_steps: ``T``。
        batch_queries: ``B``。
        num_negatives: ``n``。
        seed: 乱数シード。

    Returns:
        ``RerankingSchedule``。query と正例は乱数の生成器 ``(seed, 0)``、困難な負例は
        ``(seed, 1)``、
        ランダムな負例は ``(seed, 2)`` で引く(負例の種類によらず query と正例が同じになる)。
    """
    ranges = article_passage_ranges(passage_articles)
    rng_query = np.random.default_rng([seed, 0])
    rng_hard = np.random.default_rng([seed, 1])
    rng_random = np.random.default_rng([seed, 2])
    pool = np.asarray(pool_passages)
    query_out = np.empty((num_steps, batch_queries), dtype=np.int64)
    positive_out = np.empty_like(query_out)
    hard_out = np.empty((num_steps, batch_queries, num_negatives), dtype=np.int64)
    random_out = np.empty_like(hard_out)
    removed = considered = fallback = 0
    for step in range(num_steps):
        positions = rng_query.choice(len(train_queries), size=batch_queries, replace=False)
        for b, position in enumerate(positions.tolist()):
            query = int(train_queries[position])
            article, source = int(query_articles[query]), int(query_sources[query])
            first, last = ranges[article]
            members = np.arange(first, last)
            members = members[members != source]
            query_out[step, b] = query
            positive_out[step, b] = members[rng_query.integers(0, len(members))]
            mined = mined_candidates[position]
            kept = mined[passage_articles[mined] != article]
            removed += len(mined) - len(kept)
            considered += len(mined)
            if len(kept) >= num_negatives:
                hard_out[step, b] = kept[
                    rng_hard.choice(len(kept), size=num_negatives, replace=False)
                ]
            else:  # 同じ記事を除くと足りない(まれ): 残りをランダムな負例で補う
                fallback += 1
                extra = _draw_random_negatives(
                    rng_hard, pool, passage_articles, article, num_negatives - len(kept)
                )
                hard_out[step, b] = np.concatenate([kept, extra])
            random_out[step, b] = _draw_random_negatives(
                rng_random, pool, passage_articles, article, num_negatives
            )
    return RerankingSchedule(
        query_out, positive_out, hard_out, random_out, removed, considered, fallback
    )


def _draw_random_negatives(
    rng: np.random.Generator,
    pool: np.ndarray,
    passage_articles: np.ndarray,
    article: int,
    count: int,
) -> np.ndarray:
    """``pool`` から、記事が ``article`` でない passage を重複なく ``count`` 個引く。"""
    chosen: list[int] = []
    while len(chosen) < count:
        draws = pool[rng.integers(0, len(pool), size=2 * count)]
        for g in draws.tolist():
            if passage_articles[g] != article and g not in chosen:
                chosen.append(g)
                if len(chosen) == count:
                    break
    return np.array(chosen, dtype=np.int64)


def encode_articles_with_cache(
    tokenizer,
    corpus_text: str,
    spans,
    article_ids,
    cache_path,
    num_verified: int = 24,
) -> tuple[list[np.ndarray | None], bool]:
    """``encode_articles()`` の結果を``cache_path``にキャッシュする(決定的に再生成できる)。

    キャッシュのキーは、コーパスの長さ・記事の境界・符号化する記事の番号・トークナイザの指紋
    (固定した文の符号化と語彙サイズ)のハッシュで、キーが一致しないキャッシュは読まず、作り直す。
    キャッシュを読んだ場合は、等間隔に選んだ ``num_verified`` 記事を符号化し直して、キャッシュと
    完全に一致することを確かめる。

    Returns:
        ``(記事ごとのトークン ID の配列のリスト(使わない記事は None)、キャッシュを読んだか)``。
    """
    from pathlib import Path

    from src.data.retrieval import encode_articles

    cache_path = Path(cache_path)
    article_ids = sorted(int(a) for a in article_ids)
    fingerprint_text = "The quick brown fox jumps over the lazy dog.\n\nÀ la carte 日本語 123."
    h = hashlib.sha256()
    h.update(repr((len(corpus_text), len(spans), tokenizer.vocab_size)).encode())
    h.update(repr(tokenizer.encode(fingerprint_text)).encode())
    h.update(np.asarray([(s["start"], s["end"]) for s in spans], dtype=np.int64).tobytes())
    h.update(np.asarray(article_ids, dtype=np.int64).tobytes())
    key = h.hexdigest()[:32]

    def assemble(flat: np.ndarray, counts: np.ndarray) -> list[np.ndarray | None]:
        tokens: list[np.ndarray | None] = [None] * len(spans)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for article, start, count in zip(article_ids, starts, counts, strict=True):
            tokens[article] = flat[start : start + count]
        return tokens

    if cache_path.exists():
        with np.load(cache_path) as stored:
            if str(stored["key"]) == key:
                tokens = assemble(stored["flat"], stored["counts"])
                for position in np.linspace(0, len(article_ids) - 1, num_verified).astype(int):
                    article = article_ids[int(position)]
                    span = spans[article]
                    fresh = np.asarray(
                        tokenizer.encode(corpus_text[span["start"] : span["end"]]), dtype=np.int32
                    )
                    if not np.array_equal(fresh, tokens[article]):
                        raise RuntimeError(
                            f"キャッシュが符号化し直した結果と一致しない(記事 {article})"
                        )
                return tokens, True
    tokens = encode_articles(tokenizer, corpus_text, spans, article_ids)
    counts = np.array([len(tokens[a]) for a in article_ids], dtype=np.int64)
    flat = np.concatenate([tokens[a] for a in article_ids]).astype(np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, key=np.array(key), flat=flat, counts=counts)
    return assemble(flat, counts), False
