"""テキスト検索(retrieval)の学習データ・評価データの構成(024)。

英語 Wikipedia のコーパス(``kojikojiprg/ai-theories-corpus-en``、9,826 記事)のうち、使う記事を
**記事を単位として** 学習用・検証用・評価用に分け、次の 2 つを作る。

1. **学習の組**: 同じ記事から切り出した、**重ならない** 2 つの区間(長さ ``L_q`` の query 側と長
   さ ``L_p`` の passage 側)。Contriever(Izacard et al., TMLR 2022)の independent cropping に相
   当するが、Contriever は 2 つの区間が重なることを許す。重なりがあると語の一致だけで正例を当て
   られるので、本実装は重ならないようにする。1 つのバッチには同じ記事の組を 2 つ以上入れない(同
   じ記事の別の組は偽の負例(false negative)になるため)。

2. **評価用の索引と query**: 記事を重ならない長さ ``L_p`` の passage に分けて索引にする。query
   は passage の一部(長さ ``L_q``)から作り、**その query を含む passage は、その query の検索で
   は候補から除く**。正解は同じ記事の他の passage である。

分割と索引の構成は、シードと引数だけで決まる決定的な関数にしてある。025 などの後続のトピックが同
じ評価用の passage の集合を再構成できる。

**記事ごとの passage 数の上限**: 記事の長さは極端に偏る(中央値 約 1.4 万トークン、最長 約 12.5 万ト
ークン)。上限がないと、最長の記事 1 本が索引と query の多くを占め、正解(同じ記事の passage)が大
量にあるので検索が易しくなる。そこで、索引に入れる passage は記事ごとに最大
``max_passages_per_article`` 個とし、記事の全体から等間隔に選ぶ。

記号 / Notation:
    L_q : query の長さ(トークン数)
    L_p : passage の長さ(トークン数)
    N   : バッチサイズ(1 つのバッチの組の数)
    T   : 学習ステップ数
    P   : 索引の passage の数
    Q   : query の数

"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArticleSplit:
    """記事の学習用・検証用・評価用への分割。

    記事の番号は、コーパスの中の並び順の 0 始まりの番号である。
    """

    train: np.ndarray
    validation: np.ndarray
    evaluation: np.ndarray

    def digest(self) -> str:
        """分割の SHA-256(先頭 16 文字)。分割が再現されたことの確認に使う。"""
        h = hashlib.sha256()
        for part in (self.train, self.validation, self.evaluation):
            h.update(np.asarray(part, dtype=np.int64).tobytes())
            h.update(b"|")
        return h.hexdigest()[:16]


def find_articles_starting_at_or_after(spans: Sequence[dict], character_position: int) -> list[int]:
    """文字位置 ``character_position`` 以降に **始まる** 記事の番号(コーパスの中の並び順)を返す。

    008 の事前学習と英語のトークナイザの学習は、コーパスの先頭から一定の位置の手前までだけを使う
    ので、その位置以降に始まる記事は、どちらも一度も見ていない記事である(位置をまたぐ記事は、一部を
    見ているので含めない)。

    Args:
        spans: ``locate_wikipedia_article_spans()`` が返す、``{"start", "end", ...}`` の辞書のリスト
            (コーパスの中の順)。
        character_position: 文字位置。
    """
    return [i for i, span in enumerate(spans) if span["start"] >= character_position]


def split_articles(
    article_token_counts: Sequence[int],
    min_tokens_for_retrieval: int,
    candidate_articles: Sequence[int],
    num_validation: int,
    num_evaluation: int,
    seed: int,
) -> ArticleSplit:
    """候補の記事を学習用・検証用・評価用に分ける(決定的)。

    ``candidate_articles`` に含まれない記事は、3 つのどれにも入れない(使わない)。検証用・評価用の
    記事は、候補のうち検索に使える記事(トークン数が ``min_tokens_for_retrieval`` 以上、すなわち
    passage を 2 個以上持つ記事)の中から、シードで固定した乱数による並べ替えの先頭から選ぶ。
    先頭の ``num_evaluation`` 個を評価用、次の ``num_validation`` 個を検証用にする。候補の残り(検索
    に使えない短い記事を含む)はすべて学習用に入れる(短くて組が作れない記事は、学習でも使われない)。

    Args:
        article_token_counts: 記事ごとのトークン数(コーパスの中の順。候補でない記事の値は使わない)。
        min_tokens_for_retrieval: 検証用・評価用の記事に要求する最小のトークン数。
        candidate_articles: 使ってよい記事の番号。
        num_validation: 検証用の記事の数。
        num_evaluation: 評価用の記事の数。
        seed: 並べ替えの乱数シード。

    Returns:
        昇順に並べた記事の番号の 3 つ組(互いに素で、和集合が ``candidate_articles``)。
    """
    counts = np.asarray(article_token_counts, dtype=np.int64)
    candidates = np.array(sorted({int(a) for a in candidate_articles}), dtype=np.int64)
    pool = candidates[counts[candidates] >= min_tokens_for_retrieval]
    if num_evaluation + num_validation > len(pool):
        raise ValueError("検索に使える記事が足りない")
    order = np.random.default_rng(seed).permutation(pool)
    evaluation = np.sort(order[:num_evaluation]).astype(np.int64)
    validation = np.sort(order[num_evaluation : num_evaluation + num_validation]).astype(np.int64)
    held_out = np.concatenate([evaluation, validation])
    train = candidates[~np.isin(candidates, held_out)]
    return ArticleSplit(train=train, validation=validation, evaluation=evaluation)


def encode_articles(
    tokenizer,
    corpus_text: str,
    spans: Sequence[dict],
    article_ids: Sequence[int] | None = None,
) -> list[np.ndarray | None]:
    """記事ごとに別々に符号化する(記事の境界をまたぐトークンを作らない)。

    Args:
        tokenizer: ``encode()`` を持つトークナイザ。
        corpus_text: コーパス全文。
        spans: 記事ごとの ``{"start", "end"}``(コーパスの中の順)。
        article_ids: 符号化する記事の番号。``None`` なら全記事。指定しない記事の要素は ``None`` に
            する(使わない記事を符号化して、時間とメモリを使わないため)。

    Returns:
        記事ごとのトークン ID の配列(int32。語彙サイズが 2^31 未満の範囲で値は変わらない)の
        リスト(コーパスの中の順、``len(spans)`` 個)。
    """
    selected = set(range(len(spans))) if article_ids is None else {int(a) for a in article_ids}
    return [
        np.asarray(tokenizer.encode(corpus_text[span["start"] : span["end"]]), dtype=np.int32)
        if i in selected
        else None
        for i, span in enumerate(spans)
    ]


def select_evenly(num_available: int, num_selected: int) -> np.ndarray:
    """``0, ..., num_available - 1`` から、等間隔に ``min(num_available, num_selected)`` 個を選ぶ。

    選んだ添字は重複せず昇順である。``num_available <= num_selected`` なら全部を返す。
    """
    if num_available <= num_selected:
        return np.arange(num_available)
    # 隣り合う値の間隔が 1 以上なので、四捨五入(0.5 は切り上げ)しても重複しない
    chosen = np.floor(np.linspace(0, num_available - 1, num_selected) + 0.5).astype(np.int64)
    assert len(np.unique(chosen)) == num_selected
    return chosen


@dataclass
class RetrievalSet:
    """評価用の索引と query。

    Attributes:
        passages: 形状 ``(P, L_p)``。索引の passage のトークン ID(重ならない窓)。
        passage_articles: 形状 ``(P,)``。passage が属する記事の番号(コーパスの中の並び順)。
        passage_windows: 形状 ``(P,)``。記事の中での窓の番号(``L_p`` トークンごとの窓を 0
        から数える)。
        queries: 形状 ``(Q, L_q)``。query のトークン ID。
        query_articles: 形状 ``(Q,)``。query の元の記事の番号。
        query_sources: 形状 ``(Q,)``。query を切り出した passage の ``passages`` での添字
            (その query の検索では、この passage を候補から除く)。
        query_offsets: 形状 ``(Q,)``。元の passage の中での query の開始位置。
    """

    passages: np.ndarray
    passage_articles: np.ndarray
    passage_windows: np.ndarray
    queries: np.ndarray
    query_articles: np.ndarray
    query_sources: np.ndarray
    query_offsets: np.ndarray

    @property
    def num_passages(self) -> int:
        return int(self.passages.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.queries.shape[0])

    def num_positives(self) -> np.ndarray:
        """query ごとの正解の数(同じ記事の、元の passage 以外の passage の数)。"""
        per_article = np.bincount(self.passage_articles)
        return per_article[self.query_articles] - 1

    def digest(self) -> str:
        """索引と query の SHA-256(先頭 16 文字)。
        後続のトピックが同じ集合を再構成できたかの確認に使う。"""
        h = hashlib.sha256()
        for array in (
            self.passages,
            self.passage_articles,
            self.queries,
            self.query_articles,
            self.query_sources,
            self.query_offsets,
        ):
            h.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
            h.update(b"|")
        return h.hexdigest()[:16]


def build_retrieval_set(
    article_tokens: Sequence[np.ndarray],
    article_ids: Sequence[int],
    passage_length: int,
    query_length: int,
    max_passages_per_article: int,
    max_queries_per_article: int,
    seed: int,
) -> RetrievalSet:
    """記事の集合から、評価用の索引と query を決定的に構成する。

    1. 記事 ``a`` を長さ ``passage_length`` の重ならない窓に分ける(端数は捨てる)。窓が
       ``max_passages_per_article`` 個を超える記事は、``select_evenly()`` で等間隔に選ぶ。
    2. passage を 2 個以上持つ記事のそれぞれから、索引の passage を等間隔に
       ``max_queries_per_article`` 個まで選び、元の passage にする。query は、元の passage の中の
       シード付きの乱数で決めた位置から ``query_length`` トークン切り出す。

    Args:
        article_tokens: 記事のトークン ID(コーパスの中の順、``encode_articles()``)。
            ``article_ids`` の記事の要素だけを参照する。
        article_ids: 使う記事の番号。
        passage_length: ``L_p``。
        query_length: ``L_q``(``L_p`` 以下)。
        max_passages_per_article: 記事ごとの索引の passage 数の上限。
        max_queries_per_article: 記事ごとの query 数の上限。
        seed: query の切り出し位置の乱数シード。
    """
    if query_length > passage_length:
        raise ValueError("query_length は passage_length 以下である必要がある")
    passages, passage_articles, passage_windows = [], [], []
    queries, query_articles, query_sources, query_offsets = [], [], [], []
    for article in sorted(int(a) for a in article_ids):
        tokens = article_tokens[article]
        num_windows = len(tokens) // passage_length
        if num_windows == 0:
            continue
        windows = select_evenly(num_windows, max_passages_per_article)
        base = len(passage_articles)
        for window in windows:
            passages.append(tokens[window * passage_length : (window + 1) * passage_length])
            passage_articles.append(article)
            passage_windows.append(int(window))
        if len(windows) < 2:
            continue
        for slot in select_evenly(len(windows), max_queries_per_article):
            rng = np.random.default_rng([seed, article, int(slot)])
            offset = int(rng.integers(0, passage_length - query_length + 1))
            source = base + int(slot)
            queries.append(passages[source][offset : offset + query_length])
            query_articles.append(article)
            query_sources.append(source)
            query_offsets.append(offset)
    return RetrievalSet(
        passages=np.stack(passages).astype(np.int64),
        passage_articles=np.array(passage_articles, dtype=np.int64),
        passage_windows=np.array(passage_windows, dtype=np.int64),
        queries=np.stack(queries).astype(np.int64),
        query_articles=np.array(query_articles, dtype=np.int64),
        query_sources=np.array(query_sources, dtype=np.int64),
        query_offsets=np.array(query_offsets, dtype=np.int64),
    )


def compute_article_sampling_weights(
    article_token_counts: Sequence[int],
    query_length: int,
    passage_length: int,
    max_passages_per_article: int,
) -> np.ndarray:
    """学習で記事を選ぶ重み。

    組を作れる記事(トークン数が ``query_length + passage_length`` 以上)の重みは、その記事が持つ
    長さ ``passage_length`` の窓の数を ``max_passages_per_article`` で頭打ちにした値
    ``min(floor(n / L_p), max_passages_per_article)``(ただし 1 以上)にする。
    評価の索引と同じ上限なので、
    最長の記事に偏らない。組を作れない記事の重みは 0。
    """
    counts = np.asarray(article_token_counts, dtype=np.int64)
    windows = np.minimum(counts // passage_length, max_passages_per_article)
    weights = np.maximum(windows, 1).astype(np.float64)
    weights[counts < query_length + passage_length] = 0.0
    return weights


def sample_pair_schedule(
    article_token_counts: Sequence[int],
    weights: np.ndarray,
    query_length: int,
    passage_length: int,
    num_steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """学習全体のバッチを、学習の前に作る(決定的)。

    各ステップで、重みに比例する確率で ``batch_size`` 個の記事を **非復元抽出** する(1 つのバッチに
    同じ記事の組が 2 つ入らない)。各記事から、長さ ``L_q`` の query 側の区間と長さ ``L_p`` の
    passage 側の区間を、**重ならないように** 切り出す。記事の長さを ``n``、余白を
    ``s = n - L_q - L_p`` として、``[0, s]`` の整数を 2 つ一様に引いて ``x <= y`` に並べ、先に置く
    区間の開始位置を ``x``、後に置く区間の開始位置を ``y + (先に置いた区間の長さ)`` とする。query 側
    と passage 側のどちらを先に置くかは、等確率で決める。

    Args:
        article_token_counts: 学習用の記事ごとのトークン数。
        weights: 記事を選ぶ重み(``compute_article_sampling_weights()``)。正の重みの記事が
            ``batch_size`` 個以上必要。
        query_length: ``L_q``。
        passage_length: ``L_p``。
        num_steps: ``T``。
        batch_size: ``N``。
        seed: 乱数シード。

    Returns:
        ``"article"``(形状 ``(T, N)``、学習用の記事の添字)、``"query_start"``・``"passage_start"``
        (形状 ``(T, N)``、記事の先頭からの開始位置)。
    """
    counts = np.asarray(article_token_counts, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if int((weights > 0).sum()) < batch_size:
        raise ValueError("組を作れる記事の数がバッチサイズより少ない")
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    articles = np.stack(
        [
            rng.choice(len(counts), size=batch_size, replace=False, p=probabilities)
            for _ in range(num_steps)
        ]
    )
    slack = counts[articles] - query_length - passage_length
    assert (slack >= 0).all()
    first = rng.integers(0, slack + 1)
    second = rng.integers(0, slack + 1)
    low, high = np.minimum(first, second), np.maximum(first, second)
    query_first = rng.random(size=slack.shape) < 0.5
    query_start = np.where(query_first, low, high + passage_length)
    passage_start = np.where(query_first, high + query_length, low)
    return {
        "article": articles.astype(np.int64),
        "query_start": query_start.astype(np.int64),
        "passage_start": passage_start.astype(np.int64),
    }


def concatenate_articles(
    article_tokens: Sequence[np.ndarray], article_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """学習用の記事のトークンを 1 本の配列に連結する。

    Returns:
        ``(連結したトークン ID, 各記事の先頭の位置)``。``article_ids`` の順に並べる。
    """
    parts = [article_tokens[int(a)] for a in article_ids]
    starts = np.concatenate([[0], np.cumsum([len(p) for p in parts])[:-1]]).astype(np.int64)
    return np.concatenate(parts, dtype=np.int64), starts
