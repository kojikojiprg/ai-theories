"""Inverse Cloze Task の学習データと、passage 単位の評価の索引の構成(026)。

**Inverse Cloze Task**(Lee, Chang, Toutanova, ACL 2019): passage から 1 文を抜いて query にし、
残りの文脈(その文を除いた passage)を正例にする自己教師ありの検索の学習。原論文は、語の一致も
学ばせるために、**一定の割合(10%)で抜いた文を文脈に残す**。

**文の分割**(トークン列の上の規則。テキストにデコードしない): 008 のトークナイザ(バイトレベルの
BPE)のトークンは、空白・改行を直後の語の先頭に付ける(例: ``"ĠIt"``)ので、「次のトークンが空白
または改行で始まる位置」が語の境目になる。位置 ``t`` の直後で文が終わるのは、次のいずれか。

1. 次のトークンが **改行** で始まる(行の終わり。箇条書き・見出し・段落の区切り)。
2. 次のトークンが **空白** で始まり、かつ位置 ``t`` のトークンのテキストが ``.`` ``!`` ``?``
で終わる
   (直後に閉じ引用符・閉じ括弧が付いていてもよい)。位置 ``t`` のトークンが閉じ引用符・閉じ括弧
   だけで、その直前のトークンが ``.`` ``!`` ``?`` で終わる場合も含める(例: ``Yes."``)。

窓(passage)の端は文の端として数える。**既知の誤り**: ``U.S.`` ``Dr.`` のような略語・イニシャルの後
の空白で分割してしまい、``3.14`` のように空白を伴わないものは分割しない。略語の辞書は使わない。

**Inverse Cloze Task の対象**: 窓の文が 2 つ以上あり、長さが ``[min_query_tokens,
max_removed_tokens]`` の文が 1 つ
以上ある窓。抜く文は、その範囲の文から一様に選ぶ(query の長さの上限以内に収まる文があれば、
その中から選び、収まる文がない窓だけ長い文を選んで先頭を切り詰める)。query はその文の先頭
``query_length`` トークン
(抜く範囲は文の全体)。文脈は窓から文を抜いた残り(長さは窓の長さ - 文の長さで、可変)。可変長の
入力は、**右側をパディングして長さの配列を添える**(因果マスクのもとでは、右側のパディングは
手前の位置の出力に影響しない)。

**評価の索引**(``build_cloze_index()``): 索引のすべての passage について、抜く文を決定的に 1 つ
選び、その文を除いた形を索引に入れる(すべての passage が「1 文を除いた形」なので、除いたかどうか
が手がかりにならない)。抜いた文がその passage の query で、正例はその passage だけである。

記号 / Notation:
    L_p : 窓(passage)の長さ、L_q : query の長さの上限、M : 窓の数、T : 学習ステップ数、
    N : バッチサイズ
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_END_PATTERN = re.compile(r"[.!?][\"')\]’”]*$")
_CLOSER_PATTERN = re.compile(r"^[\"')\]’”]+$")


@dataclass(frozen=True)
class SentenceTables:
    """トークン ID ごとの真偽の表(長さは語彙サイズ)。

    Attributes:
        starts_with_whitespace: テキストが空白または改行で始まる。
        starts_with_newline: テキストが改行で始まる。
        ends_sentence: テキストが ``.`` ``!`` ``?``(と、続く閉じ引用符・閉じ括弧)で終わる。
        closer_only: テキストが閉じ引用符・閉じ括弧だけ。
    """

    starts_with_whitespace: np.ndarray
    starts_with_newline: np.ndarray
    ends_sentence: np.ndarray
    closer_only: np.ndarray


def build_sentence_tables(token_texts: Sequence[str | None]) -> SentenceTables:
    """トークンごとのテキスト(UTF-8 として不正なトークンは ``None``)から表を作る。"""
    size = len(token_texts)
    whitespace = np.zeros(size, dtype=bool)
    newline = np.zeros(size, dtype=bool)
    ends = np.zeros(size, dtype=bool)
    closer = np.zeros(size, dtype=bool)
    for i, text in enumerate(token_texts):
        if not text:
            continue
        whitespace[i] = text[0] in " \n\t\r"
        newline[i] = text[0] == "\n"
        ends[i] = _END_PATTERN.search(text) is not None
        closer[i] = _CLOSER_PATTERN.match(text) is not None
    return SentenceTables(whitespace, newline, ends, closer)


def sentence_boundaries(windows: np.ndarray, tables: SentenceTables) -> np.ndarray:
    """形状 ``(M, L)`` の窓について、位置 ``t`` の直後で文が終わるかの真偽 ``(M, L)``。

    最後の位置は常に真(窓の端は文の端)。
    """
    windows = np.asarray(windows)
    rows, length = windows.shape
    boundary = np.ones((rows, length), dtype=bool)
    if length < 2:
        return boundary
    current = windows[:, :-1]
    following = windows[:, 1:]
    previous_ends = np.zeros((rows, length - 1), dtype=bool)
    if length > 2:
        previous_ends[:, 1:] = tables.ends_sentence[windows[:, : length - 2]]
    ends_here = tables.ends_sentence[current] | (tables.closer_only[current] & previous_ends)
    boundary[:, :-1] = tables.starts_with_newline[following] | (
        tables.starts_with_whitespace[following] & ends_here
    )
    return boundary


def sentence_structure(
    windows: np.ndarray, tables: SentenceTables
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """窓ごとの文の構造。

    Returns:
        ``(is_start, sentence_length, num_sentences, boundary)``。``is_start`` は形状 ``(M,
        L)`` で、
        位置が文の先頭か。``sentence_length`` は形状 ``(M, L)`` で、その位置が属する文の長さ。
        ``num_sentences`` は形状 ``(M,)``。
    """
    boundary = sentence_boundaries(windows, tables)
    rows, length = boundary.shape
    is_start = np.zeros((rows, length), dtype=bool)
    is_start[:, 0] = True
    is_start[:, 1:] = boundary[:, :-1]
    sentence_id = np.cumsum(is_start, axis=1) - 1
    flat = (sentence_id + np.arange(rows)[:, None] * length).ravel()
    counts = np.bincount(flat, minlength=rows * length)
    sentence_length = counts[flat].reshape(rows, length)
    num_sentences = sentence_id[:, -1] + 1
    return is_start, sentence_length, num_sentences, boundary


def choose_cloze_sentence(
    windows: np.ndarray,
    tables: SentenceTables,
    uniform: np.ndarray,
    min_query_tokens: int,
    max_removed_tokens: int,
    preferred_max_tokens: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """窓ごとに、抜く文を一様に選ぶ。

    Args:
        windows: 形状 ``(M, L)``。
        tables: 文の分割の表。
        uniform: 形状 ``(M,)``。``[0, 1)`` の乱数(選ぶ文の決定に使う)。
        min_query_tokens: 抜く文の最小の長さ。
        max_removed_tokens: 抜く文の最大の長さ(文脈が短くなりすぎないように)。
        preferred_max_tokens: 指定すると、この長さ以下の文(query の長さの上限以内に収まる文)が
            窓にあれば、その中から選ぶ。収まる文がない窓だけ、より長い文(``max_removed_tokens``
            以下)
            から選ぶ(その query は先頭を切り詰める)。``None`` なら優先しない。

    Returns:
        ``(valid, start, length)``。``valid`` は形状 ``(M,)``(対象か)。``start``・``length`` は
        抜く文の先頭の位置と長さ(``valid`` が偽の窓では 0)。
    """
    is_start, sentence_length, num_sentences, _ = sentence_structure(windows, tables)
    eligible_start = (
        is_start & (sentence_length >= min_query_tokens) & (sentence_length <= max_removed_tokens)
    )
    eligible_start &= (num_sentences >= 2)[:, None]
    valid = eligible_start.any(axis=1)
    if preferred_max_tokens is not None:
        preferred = eligible_start & (sentence_length <= preferred_max_tokens)
        eligible_start = np.where(preferred.any(axis=1)[:, None], preferred, eligible_start)
    count = eligible_start.sum(axis=1)
    pick = np.minimum((np.asarray(uniform) * count).astype(np.int64), np.maximum(count - 1, 0))
    cumulative = np.cumsum(eligible_start, axis=1)
    chosen = eligible_start & (cumulative == (pick + 1)[:, None])
    start = np.where(valid, chosen.argmax(axis=1), 0)
    length = np.where(valid, sentence_length[np.arange(len(start)), start], 0)
    return valid, start.astype(np.int64), length.astype(np.int64)


def remove_span(
    windows: np.ndarray, start: np.ndarray, length: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """窓から ``[start, start + length)`` を抜き、右側を 0 でパディングする。

    Returns:
        ``(文脈 (M, L)、文脈の長さ (M,))``。
    """
    rows, width = windows.shape
    position = np.arange(width)[None, :]
    source = np.where(position < start[:, None], position, position + length[:, None])
    context_length = width - length
    gathered = np.take_along_axis(windows, np.minimum(source, width - 1), axis=1)
    return np.where(position < context_length[:, None], gathered, 0), context_length.astype(
        np.int64
    )


def extract_span(
    windows: np.ndarray, start: np.ndarray, length: np.ndarray, query_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """窓から文の先頭 ``query_length`` トークンを切り出し、右側を 0 でパディングする。

    Returns:
        ``(query (M, query_length)、query の長さ (M,))``。
    """
    width = windows.shape[1]
    kept = np.minimum(length, query_length)
    position = np.arange(query_length)[None, :]
    source = np.minimum(start[:, None] + position, width - 1)
    gathered = np.take_along_axis(windows, source, axis=1)
    return np.where(position < kept[:, None], gathered, 0), kept.astype(np.int64)


def splitmix64_uniform(seed: int, *keys: np.ndarray) -> np.ndarray:
    """整数のキーから ``[0, 1)`` の乱数を決定的に作る(キーの並びに依らず、要素ごとに決まる)。"""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    initial = (int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    state = np.full(np.broadcast(*keys).shape, initial, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for key in keys:
            state = (
                state
                + np.asarray(key, dtype=np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
                + np.uint64(0x9E3779B97F4A7C15)
            ) & mask
            z = state
            z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
            z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
            state = z ^ (z >> np.uint64(31))
    return (state >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _eligible_window_positions(
    stream: np.ndarray,
    article_start: int,
    article_tokens: int,
    passage_length: int,
    tables: SentenceTables,
    min_query_tokens: int,
    max_removed_tokens: int,
    chunk: int = 4096,
) -> np.ndarray:
    """記事のすべての窓の開始位置のうち、Inverse Cloze Task の対象の窓の位置。"""
    found = []
    offsets = np.arange(passage_length)[None, :]
    total = article_tokens - passage_length + 1
    for first in range(0, total, chunk):
        positions = np.arange(first, min(first + chunk, total))
        candidate = stream[article_start + positions[:, None] + offsets].astype(np.int64)
        valid, _, _ = choose_cloze_sentence(
            candidate, tables, np.zeros(len(positions)), min_query_tokens, max_removed_tokens
        )
        found.append(positions[valid])
    return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)


def pairable_articles(
    stream: np.ndarray,
    article_starts: np.ndarray,
    article_token_counts: np.ndarray,
    candidate: np.ndarray,
    passage_length: int,
    tables: SentenceTables,
    min_query_tokens: int,
    max_removed_tokens: int,
    chunk: int = 256,
) -> np.ndarray:
    """記事ごとに、重ならない対象の窓の組(2 個)が取れるかを厳密に判定する。

    対象の窓の開始位置の最小値と最大値は、それ自身が組をなす中で最も離れている。したがって、
    組が取れる必要十分条件は ``最大値 - 最小値 >= passage_length`` である。最小値は記事の先頭から、
    最大値は末尾から、``chunk`` 個ずつの窓を見て最初に見つかった対象の窓で決める(多くの記事は
    最初の区間で見つかる)。

    Args:
        candidate: 形状 ``(記事の数,)`` の真偽。判定する記事(偽の記事は偽を返す)。

    Returns:
        形状 ``(記事の数,)`` の真偽。
    """
    result = np.zeros(len(article_token_counts), dtype=bool)
    offsets = np.arange(passage_length)[None, :]

    def first_eligible(article: int, from_left: bool) -> int:
        total = int(article_token_counts[article]) - passage_length + 1
        base = int(article_starts[article])
        for first in range(0, total, chunk):
            block = np.arange(first, min(first + chunk, total))
            if not from_left:
                block = total - 1 - block  # 末尾から降順
            candidate_windows = stream[base + block[:, None] + offsets].astype(np.int64)
            valid, _, _ = choose_cloze_sentence(
                candidate_windows,
                tables,
                np.zeros(len(block)),
                min_query_tokens,
                max_removed_tokens,
            )
            if valid.any():
                return int(block[valid.argmax()])
        return -1

    for article in np.nonzero(np.asarray(candidate))[0].tolist():
        if int(article_token_counts[article]) < 2 * passage_length:
            continue
        low = first_eligible(article, True)
        if low < 0:
            continue
        high = first_eligible(article, False)
        result[article] = high - low >= passage_length
    return result


@dataclass
class ClozePairBatches:
    """Inverse Cloze Task の学習の組(学習の前に全ステップぶんを作る)。

    1 つのバッチは ``articles_per_batch`` 個の記事 x ``pairs_per_article`` 組で、同じ記事の組は
    バッチの中で隣り合う(位置 ``pairs_per_article * k + j``)。同じ記事の組の窓は互いに重ならない。

    Attributes:
        query_tokens: 形状 ``(T, N, L_q)``(右側を 0 でパディング)。
        query_lengths: 形状 ``(T, N)``。
        context_tokens: 形状 ``(T, N, L_p)``(文脈。右側を 0 でパディング)。
        context_lengths: 形状 ``(T, N)``。
        articles: 形状 ``(T, N)``。組の記事の添字。
        kept: 形状 ``(T, N)``。文を文脈に残した組か。
        truncated: 形状 ``(T, N)``。抜いた文が query の長さの上限を超え、先頭で切った組か。
        window_starts: 形状 ``(T, N)``。窓の、記事の先頭からの開始位置。
        num_attempts: 対象の窓の組が見つかるまでに引いた窓の組の数の合計。
        num_exhaustive: 引き直しの上限に達し、記事のすべての窓から選んだ組の数。
    """

    query_tokens: np.ndarray
    query_lengths: np.ndarray
    context_tokens: np.ndarray
    context_lengths: np.ndarray
    articles: np.ndarray
    kept: np.ndarray
    truncated: np.ndarray
    window_starts: np.ndarray
    num_attempts: int
    num_exhaustive: int = 0


def sample_cloze_pair_batches(
    stream: np.ndarray,
    article_starts: np.ndarray,
    article_token_counts: np.ndarray,
    article_weights: np.ndarray,
    tables: SentenceTables,
    num_steps: int,
    articles_per_batch: int,
    pairs_per_article: int,
    passage_length: int,
    query_length: int,
    min_query_tokens: int,
    max_removed_tokens: int,
    keep_probability: float,
    seed: int,
    max_attempts: int = 64,
    eligible_cache: dict[int, np.ndarray] | None = None,
) -> ClozePairBatches:
    """学習の組を作る(決定的)。

    各ステップで、重みに比例する確率で ``articles_per_batch`` 個の記事を非復元抽出する。各記事から、
    **互いに重ならない** ``pairs_per_article`` 個の窓(長さ ``passage_length``)を一様な位置で引く。
    窓は Inverse Cloze Task の対象でなければ引き直す(``max_attempts`` 回まで。
    尽きたら記事のすべての窓から選ぶ)。
    対象の窓から抜く文を選び(``query_length`` 以内に収まる文を優先する)、確率
    ``keep_probability`` で
    その文を文脈に残す(文脈は窓そのもの)。

    Args:
        stream: 学習用の記事を連結したトークン ID(1 次元)。
        article_starts: 各記事の ``stream`` での先頭の位置。
        article_token_counts: 各記事のトークン数。
        article_weights: 記事を選ぶ重み(重みが 0 の記事は選ばない。選ばれる記事は
            ``pairs_per_article * passage_length`` トークン以上を要する)。
        tables: 文の分割の表。
        num_steps: ``T``。
        articles_per_batch: 1 バッチの記事の数。
        pairs_per_article: 1 つの記事あたりの組の数(同じバッチに入る)。
        passage_length, query_length, min_query_tokens, max_removed_tokens: 5 節のとおり。
        keep_probability: 抜いた文を文脈に残す確率(原論文は 0.1)。
        seed: 乱数シード。
        max_attempts: 窓の組を引き直す回数の上限(尽きた記事は記事のすべての窓から選ぶ)。
        eligible_cache: 記事の添字から対象の窓の開始位置への辞書。渡すと、記事のすべての窓から選ぶ
            処理の走査の結果を呼び出しをまたいで使い回す(結果は変わらない。時間だけが変わる)。
    """
    weights = np.asarray(article_weights, dtype=np.float64)
    counts = np.asarray(article_token_counts, dtype=np.int64)
    if (counts[weights > 0] < pairs_per_article * passage_length).any():
        raise ValueError("重みが正の記事に、窓を重ならずに取れない短い記事が含まれている")
    if int((weights > 0).sum()) < articles_per_batch:
        raise ValueError("組を作れる記事の数がバッチの記事の数より少ない")
    rng = np.random.default_rng(seed)
    probabilities = weights / weights.sum()
    articles = np.stack(
        [
            rng.choice(len(counts), size=articles_per_batch, replace=False, p=probabilities)
            for _ in range(num_steps)
        ]
    )
    flat_articles = articles.reshape(-1)
    count = flat_articles.size
    pair_count = pairs_per_article
    window_start = np.zeros((count, pair_count), dtype=np.int64)
    windows = np.zeros((count, pair_count, passage_length), dtype=np.int64)
    start = np.zeros((count, pair_count), dtype=np.int64)
    length = np.zeros((count, pair_count), dtype=np.int64)
    offsets = np.arange(passage_length)[None, :]

    def draw_valid(article_index: np.ndarray, positions: np.ndarray):
        # positions: (len(article_index), pair_count)。窓が対象か、選んだ文
        flat_positions = positions.reshape(-1)
        flat_base = np.repeat(article_starts[flat_articles[article_index]], pair_count)
        candidate = stream[(flat_base + flat_positions)[:, None] + offsets].astype(np.int64)
        valid, s, n = choose_cloze_sentence(
            candidate,
            tables,
            rng.random(len(candidate)),
            min_query_tokens,
            max_removed_tokens,
            query_length,
        )
        shape = (len(article_index), pair_count)
        return (
            candidate.reshape(*shape, -1),
            valid.reshape(shape),
            s.reshape(shape),
            n.reshape(shape),
        )

    pending = np.arange(count)
    attempts = 0
    for _ in range(max_attempts):
        if len(pending) == 0:
            break
        attempts += len(pending)
        slack = counts[flat_articles[pending]] - passage_length
        positions = (rng.random((len(pending), pair_count)) * (slack + 1)[:, None]).astype(np.int64)
        ordered = np.sort(positions, axis=1)
        separated = (np.diff(ordered, axis=1) >= passage_length).all(axis=1)
        candidate, valid, s, n = draw_valid(pending, positions)
        accepted = separated & valid.all(axis=1)
        index = pending[accepted]
        windows[index], window_start[index] = candidate[accepted], positions[accepted]
        start[index], length[index] = s[accepted], n[accepted]
        pending = pending[~accepted]
    exhaustive = 0
    if len(pending):  # 対象の窓の組がまれな記事: 記事のすべての窓から、重ならない対象の窓の組を選ぶ
        eligible_positions = eligible_cache if eligible_cache is not None else {}
        for index in pending.tolist():
            article = int(flat_articles[index])
            if article not in eligible_positions:
                eligible_positions[article] = _eligible_window_positions(
                    stream,
                    int(article_starts[article]),
                    int(article_token_counts[article]),
                    passage_length,
                    tables,
                    min_query_tokens,
                    max_removed_tokens,
                )
            eligible = eligible_positions[article]
            if pair_count != 2:
                raise NotImplementedError(
                    "記事のすべての窓から選ぶ処理は、1 つの記事あたり 2 組のときだけ"
                )
            # 重ならない相手がいる窓(最小・最大の窓との距離で判定できる)の中から最初の窓を一様に選ぶ
            has_partner = (
                (eligible - eligible[:1] >= passage_length)
                | (eligible[-1:] - eligible >= passage_length)
                if len(eligible)
                else np.zeros(0, dtype=bool)
            )
            firsts = eligible[has_partner]
            if len(firsts) == 0:
                raise RuntimeError(
                    f"記事 {article} から、重ならない対象の窓を {pair_count} 個取れない"
                    "(pairable_articles() で重みを 0 にしておくこと)"
                )
            first = int(firsts[int(rng.integers(0, len(firsts)))])
            partners = eligible[np.abs(eligible - first) >= passage_length]
            chosen = [first, int(partners[int(rng.integers(0, len(partners)))])]
            candidate, valid, s, n = draw_valid(np.array([index]), np.array([chosen]))
            assert bool(valid.all())
            windows[index], window_start[index], start[index], length[index] = (
                candidate[0],
                chosen,
                s[0],
                n[0],
            )
            exhaustive += 1
    flat_windows = windows.reshape(-1, passage_length)
    flat_start, flat_length = start.reshape(-1), length.reshape(-1)
    kept = rng.random(flat_windows.shape[0]) < keep_probability
    context, context_length = remove_span(flat_windows, flat_start, flat_length)
    context = np.where(kept[:, None], flat_windows, context)
    context_length = np.where(kept, passage_length, context_length)
    query, query_len = extract_span(flat_windows, flat_start, flat_length, query_length)
    batch = articles_per_batch * pair_count
    return ClozePairBatches(
        query_tokens=query.reshape(num_steps, batch, query_length),
        query_lengths=query_len.reshape(num_steps, batch),
        context_tokens=context.reshape(num_steps, batch, passage_length),
        context_lengths=context_length.reshape(num_steps, batch),
        articles=np.repeat(articles, pair_count, axis=1),
        kept=kept.reshape(num_steps, batch),
        truncated=(flat_length > query_length).reshape(num_steps, batch),
        window_starts=window_start.reshape(num_steps, batch),
        num_attempts=attempts,
        num_exhaustive=exhaustive,
    )


@dataclass
class ClozeIndex:
    """passage 単位の評価の索引。

    ``passage_ids`` は元の passage の添字(対象の passage だけ。昇順)。``context_tokens``・
    ``context_lengths`` は、抜く文を除いた形(索引に入れる形)。
    ``query_tokens``・``query_lengths`` は、
    抜いた文(その passage の query)。正例はその passage だけ(索引の中の位置は ``positive``)。
    """

    passage_ids: np.ndarray
    context_tokens: np.ndarray
    context_lengths: np.ndarray
    query_tokens: np.ndarray
    query_lengths: np.ndarray
    removed_start: np.ndarray
    removed_length: np.ndarray
    num_excluded: int
    num_single_sentence: int

    @property
    def num_passages(self) -> int:
        return int(len(self.passage_ids))


def build_cloze_index(
    passages: np.ndarray,
    passage_articles: np.ndarray,
    passage_windows: np.ndarray,
    tables: SentenceTables,
    query_length: int,
    min_query_tokens: int,
    max_removed_tokens: int,
    seed: int,
) -> ClozeIndex:
    """索引の passage(固定長の窓)から、抜く文を決定的に選んで「1 文を除いた形」の索引を作る。

    抜く文を選ぶ乱数は ``(seed, 記事, 窓の番号)`` だけで決まる(passage の並びに依らない)。
    Inverse Cloze Task の
    対象でない passage(文が 2 つ未満、または範囲の文がない)は索引に入れない(除いた形にできない
    passage を、除いていない形で索引に入れると、除いたかどうかが手がかりになる)。
    """
    uniform = splitmix64_uniform(seed, passage_articles, passage_windows)
    valid, start, length = choose_cloze_sentence(
        passages, tables, uniform, min_query_tokens, max_removed_tokens, query_length
    )
    _, _, num_sentences, _ = sentence_structure(passages, tables)
    index = np.nonzero(valid)[0]
    context, context_length = remove_span(passages[index], start[index], length[index])
    query, query_len = extract_span(passages[index], start[index], length[index], query_length)
    return ClozeIndex(
        passage_ids=index.astype(np.int64),
        context_tokens=context,
        context_lengths=context_length,
        query_tokens=query,
        query_lengths=query_len,
        removed_start=start[index],
        removed_length=length[index],
        num_excluded=int((~valid).sum()),
        num_single_sentence=int((num_sentences < 2).sum()),
    )
