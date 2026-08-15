"""部分語分割(Subword Tokenization)のスクラッチ実装。

Byte Pair Encoding(BPE)の学習・符号化と、Unigram 言語モデル(Unigram Language
Model)の Viterbi 最尤分割をスクラッチ実装する。Unigram 言語モデルの語彙学習
(候補語彙からの EM ベースの反復的な縮小)は sentencepiece に委ねる
(``theories/02_pretraining/005_tokenizer.ipynb`` の実装方針を参照)。

記号 / Notation:
    V : 語彙サイズ(vocabulary size)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import sentencepiece as spm

ChunkSplitMode = Literal["whitespace", "none"]

# BPETokenizer._chunk_cache(006、本番実行前の修正 22)がキャッシュするチャンクの
# 最大件数。メモリ量が際限なく増えないよう上限を設ける。
_MAX_CHUNK_CACHE_ENTRIES = 200_000


def _build_byte_to_unicode() -> dict[int, str]:
    """UTF-8 バイト値(0〜255)を印字可能な Unicode 1 文字に単射で対応付ける。

    GPT-2(Radford et al., 2019)と同じ構成: 印字可能な ASCII / Latin-1 のバイト値は
    それ自身の文字にマップし、残りの(制御文字などの)バイト値は U+0100 以降の
    未使用領域に順にマップする。バイト列を常に「見える」文字列として扱えるように
    するための工夫であり、この単射関係はバイト列としての UTF-8 デコードとは独立である。
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_to_code = dict(zip(printable, printable, strict=True))
    next_code = 2**8
    for b in range(2**8):
        if b not in byte_to_code:
            byte_to_code[b] = next_code
            next_code += 1
    return {b: chr(code) for b, code in byte_to_code.items()}


_BYTE_TO_UNICODE: dict[int, str] = _build_byte_to_unicode()
_UNICODE_TO_BYTE: dict[str, int] = {ch: b for b, ch in _BYTE_TO_UNICODE.items()}


def try_decode_byte_level_symbol(symbol: str) -> str | None:
    """バイトレベル BPE の 1 シンボル(語彙の要素)を UTF-8 文字列としてデコードを試み、
    成功すればその文字列を、失敗すれば ``None`` を返す。

    シンボルの各文字を ``_UNICODE_TO_BYTE`` で対応するバイト値に戻し、その
    バイト列を UTF-8 としてデコードする。以下のいずれかの場合に失敗する
    (``None`` を返す)。

    - シンボルの文字が ``_UNICODE_TO_BYTE`` の定義域(バイト値 0〜255 に対応する
      256 個の Unicode 文字)に含まれない場合(バイトレベル表現として解釈できない、
      例えば文字レベル語彙のシンボルを渡した場合)。
    - バイト列が正しい UTF-8 として不正な場合(BPE のマージが 1 つの Unicode 文字を
      再構成しきっていない、すなわちマルチバイト文字の一部のバイトだけが
      マージされた状態)。

    005 の ``compute_character_coverage``(``src/utils/statistics.py``)で、
    バイトレベル語彙のある要素が特定の 1 文字を単一トークンとして表現できているかを
    判定するために使う。
    """
    try:
        byte_values = bytes(_UNICODE_TO_BYTE[ch] for ch in symbol)
        return byte_values.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


_WHITESPACE_CHUNK_RE = re.compile(r"\s*\S+|\s+")


def _split_chunk_by_max_bytes(chunk: str, max_chunk_bytes: int) -> list[str]:
    """チャンクの UTF-8 バイト長が ``max_chunk_bytes`` を超える場合、文字境界
    (Unicode コードポイントの境界)を壊さない位置で分割する(006、本番実行前の
    修正 21)。

    Python の文字列は 1 文字ずつ反復してもマルチバイト文字が分断されないため、
    文字単位でバイト長を積算し、上限を超える直前で区切ることで、UTF-8 として
    不正な断片を作らない。1 文字だけで ``max_chunk_bytes`` を超える場合(絵文字など)
    は、その 1 文字のみからなる断片を許容する(それ以上分割しようがないため)。

    Args:
        chunk: 分割対象のチャンク(``pretokenize`` の内部でのみ呼ばれる)。
        max_chunk_bytes: 断片ごとの UTF-8 バイト長の上限。

    Returns:
        分割後の断片のリスト。すべて連結すると ``chunk`` に戻る(可逆性は保たれる)。
    """
    if len(chunk.encode("utf-8")) <= max_chunk_bytes:
        return [chunk]

    pieces: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for ch in chunk:
        ch_bytes = len(ch.encode("utf-8"))
        if current and current_bytes + ch_bytes > max_chunk_bytes:
            pieces.append("".join(current))
            current = []
            current_bytes = 0
        current.append(ch)
        current_bytes += ch_bytes
    if current:
        pieces.append("".join(current))
    return pieces


def pretokenize(
    text: str,
    chunk_split_mode: ChunkSplitMode,
    max_chunk_bytes: int | None = None,
) -> list[str]:
    """事前分割(pre-tokenization)。BPE のマージはチャンクの内部でのみ行われ、
    チャンクをまたいでは行われない。

    Args:
        text: 分割対象のテキスト。
        chunk_split_mode:
            ``"whitespace"``: 空白文字(スペース・タブ・改行)の直前で分割し、
                空白をその直後のチャンクの先頭に含める(GPT-2 のバイトレベル BPE
                と同じ方式)。例えば ``"a  b"`` は ``["a", "  b"]`` になる
                (``"b"`` の前の 2 個の空白がチャンク先頭に付く)。連続する空白・
                タブ・改行はすべて 1 つの塊としてまとめて次のチャンクの先頭に付く。
                テキスト末尾が空白で終わる場合、後続の非空白文字が無いため、
                その空白の塊だけが独立したチャンクになる。チャンクをすべて連結すると
                元のテキストが過不足なく復元できる(``BPETokenizer.encode()`` の
                可逆性の根拠。5 節の可逆性の検証を参照)。
            ``"none"``: 事前分割を行わない。ただし改行のみは区切りとして扱う
                (実装上の理由。改行をまたぐマージを許すとチャンクが際限なく
                大きくなり実装が煩雑になるため)。改行以外の空白(スペース・タブ)は
                チャンク内にそのまま残る。この改行の扱いにより、複数行にわたる
                テキストでは ``chunk_split_mode="whitespace"`` と異なり、
                改行が失われるため可逆ではない(空白の有無のみを実験の対照条件と
                するための単純化であり、005 の実装方針で明記する)。
        max_chunk_bytes: 指定すると、``chunk_split_mode`` による分割の後、
            なお UTF-8 バイト長がこの値を超えるチャンクをさらに分割する
            (``_split_chunk_by_max_bytes`` 参照、006、本番実行前の修正 21)。
            ``chunk_split_mode="whitespace"`` が機能しない言語(日本語など、005 の
            議論を参照)では 1 チャンクが 1 行・1 段落まるごとになり、チャンク長に
            対して超線形な BPE の学習・符号化コストが爆発しうる。この上限は
            そのコストの上限を抑えるための計算量対策であり、既定値 ``None``
            (上限なし)では 005 と完全に同一の結果を返す。
    """
    if chunk_split_mode == "whitespace":
        chunks = _WHITESPACE_CHUNK_RE.findall(text)
    elif chunk_split_mode == "none":
        chunks = [line for line in text.split("\n") if line]
    else:
        raise ValueError(f"未知の chunk_split_mode: {chunk_split_mode!r}")

    if max_chunk_bytes is None:
        return chunks
    return [
        piece for chunk in chunks for piece in _split_chunk_by_max_bytes(chunk, max_chunk_bytes)
    ]


def _count_pairs(symbols: Sequence[str], freq: int) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for i in range(len(symbols) - 1):
        counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _merge_pair_in_word(
    symbols: Sequence[str], pair: tuple[str, str], new_symbol: str
) -> list[str]:
    """シンボル列中の ``pair`` に一致する隣接ペアを、左から順に(重複せず)すべて
    ``new_symbol`` に置き換える。"""
    merged: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(new_symbol)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


@dataclass
class BPETokenizer:
    """学習済み BPE(Byte Pair Encoding)トークナイザ。

    符号化(encoding)は学習で得たマージ規則を **学習順に** 適用する。学習
    (``learn_bpe``)は頻度に基づき次のマージ規則を毎回選び直すのに対し、符号化は
    既に確定した規則列を順番に適用するだけであり、学習と符号化は非対称な処理である。

    Attributes:
        merges: 学習順のマージ規則のリスト(タプル ``(s1, s2)`` の列)。
        vocab: 語彙(初期シンボルとマージで生成されたシンボルの集合)。
        byte_level: True の場合、初期語彙は UTF-8 の 256 バイト
            (未知語(Out-of-Vocabulary)が原理的に発生しない)。False の場合は
            学習コーパスに出現した Unicode 文字が初期語彙になる(未知語が発生しうる)。
        chunk_split_mode: 事前分割(pre-tokenization)の方式(``pretokenize`` 参照)。
        unk_token: 未知語(``byte_level=False`` のときのみ発生しうる)を表す特殊トークン。
        max_chunk_bytes: チャンクの UTF-8 バイト長の上限(``pretokenize`` 参照、006、
            本番実行前の修正 21)。**学習(``learn_bpe``)時と符号化(``encode``)時で
            必ず同一の値を使うこと**(異なる値では学習時と異なる分割になり、
            符号化結果が学習した語彙と整合しなくなる)。既定値 ``None``(上限なし)
            では 005 と完全に同一の結果になる。
    """

    merges: list[tuple[str, str]]
    vocab: set[str]
    byte_level: bool
    chunk_split_mode: ChunkSplitMode
    unk_token: str = "<unk>"
    max_chunk_bytes: int | None = None

    @cached_property
    def merge_ranks(self) -> dict[tuple[str, str], int]:
        """マージ規則の学習順の順位(rank)。符号化時に「最も早く学習された
        マージ規則」を優先して適用するために使う(rank が小さいほど優先)。"""
        return {pair: rank for rank, pair in enumerate(self.merges)}

    @cached_property
    def _chunk_cache(self) -> dict[str, list[str]]:
        """チャンク文字列 -> ``encode_chunk`` の結果のメモ化キャッシュ(006、
        本番実行前の修正 22)。``max_chunk_bytes`` によるチャンク分割で同一チャンクの
        反復出現が増えるため、キャッシュの効果が大きい。

        件数が ``_MAX_CHUNK_CACHE_ENTRIES`` に達すると、それ以降の新規チャンクは
        キャッシュに追加しない(``encode`` 参照)。追加しないだけで、その後も
        ``encode_chunk`` で計算はでき符号化結果自体は変わらない。速度上の恩恵が
        なくなるだけであり、正しさには影響しない。
        """
        return {}

    def _initial_symbols(self, chunk: str) -> list[str]:
        if self.byte_level:
            return [_BYTE_TO_UNICODE[b] for b in chunk.encode("utf-8")]
        return [ch if ch in self.vocab else self.unk_token for ch in chunk]

    def encode_chunk(self, chunk: str) -> list[str]:
        """1 チャンク(事前分割で得た 1 単位)を部分語シンボル列に変換する。"""
        symbols = self._initial_symbols(chunk)
        while len(symbols) > 1:
            ranked = [
                (self.merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:], strict=False)
                if pair in self.merge_ranks
            ]
            if not ranked:
                break
            _, best_pair = min(ranked)
            symbols = _merge_pair_in_word(symbols, best_pair, best_pair[0] + best_pair[1])
        return symbols

    def encode(self, text: str) -> list[str]:
        """テキスト全体を部分語シンボル列に変換する(チャンクをまたぐマージは行わない)。

        チャンク単位でメモ化する(``_chunk_cache``、006、本番実行前の修正 22)。
        """
        cache = self._chunk_cache
        tokens: list[str] = []
        for chunk in pretokenize(text, self.chunk_split_mode, self.max_chunk_bytes):
            symbols = cache.get(chunk)
            if symbols is None:
                symbols = self.encode_chunk(chunk)
                if len(cache) < _MAX_CHUNK_CACHE_ENTRIES:
                    cache[chunk] = symbols
            tokens.extend(symbols)
        return tokens

    def decode(self, tokens: Sequence[str]) -> str:
        """符号化されたトークン列を元のテキストに復元する。

        ``chunk_split_mode="whitespace"`` は空白をチャンク先頭に保持するため、
        トークンを単純に連結するだけで元のテキストが復元できる
        (``chunk_split_mode="none"`` は改行を保持しないため、複数行にわたる
        テキストでは復元できない、``pretokenize`` の docstring を参照)。
        ``byte_level=True`` の場合、各トークンは UTF-8 バイト値を表す Unicode
        1 文字の列(``_BYTE_TO_UNICODE`` によるマッピング)であるため、バイト列に
        戻してから UTF-8 としてデコードする。

        ``tokens`` がテキスト全体を符号化した完全な列であれば常に正しい UTF-8 の
        バイト列になるが、系列の途中で切り出した任意の部分列(例:
        ``make_evaluation_windows``、``src/data/text.py``、006 で評価窓を作る際に
        使う)は、マルチバイト文字の途中で切れることがある。そのため
        ``errors="replace"`` でデコードし、不正なバイト列は置換文字
        (U+FFFD)に変換する(例外を送出しない)。005 の可逆性検証はテキスト全体を
        符号化した完全な列を復号するため、この変更による影響を受けない。
        """
        joined = "".join(tokens)
        if not self.byte_level:
            return joined
        byte_values = bytes(_UNICODE_TO_BYTE[ch] for ch in joined)
        return byte_values.decode("utf-8", errors="replace")


def learn_bpe(
    text: str,
    vocab_size: int,
    byte_level: bool = False,
    chunk_split_mode: ChunkSplitMode = "whitespace",
    max_chunk_bytes: int | None = None,
) -> BPETokenizer:
    """BPE(Byte Pair Encoding、Gage 1994 / Sennrich et al. 2016)の語彙を学習する。

    頻度最大のシンボル対を、目標語彙サイズに達するまで反復的にマージする。

    **タイブレーク規則**: 最大頻度のシンボル対が複数存在する場合、``(s1, s2)`` の
    タプル比較(Python の既定の辞書式比較。まず ``s1`` を比較し、等しければ ``s2``
    を比較する)によって一意に決める。この規則を固定しない実装は、同一の入力から
    毎回異なるマージ順序を生成しうるため再現性(reproducibility)を失う。

    効率化のため、全シンボル対の頻度を毎回数え直すのではなく、直前のマージで
    変化したチャンクのみ差分更新する(``pair_index`` によって「どのチャンクが
    どのペアを含むか」を追跡する)。

    Args:
        text: 学習コーパス全体。
        vocab_size: 目標語彙サイズ V(初期語彙 + マージで生成されるシンボルの
            合計)。学習コーパスから作れるユニークなペアが尽きた場合、
            この値に達する前に学習が終了することがある。
        byte_level: True の場合、初期語彙を UTF-8 の 256 バイトにする
            (バイトレベル BPE、byte-level BPE)。False の場合は学習コーパスに
            出現した Unicode 文字を初期語彙にする。
        chunk_split_mode: 事前分割の方式(``pretokenize`` 参照)。
        max_chunk_bytes: チャンクの UTF-8 バイト長の上限(``pretokenize`` 参照、006、
            本番実行前の修正 21)。返す ``BPETokenizer`` にもこの値を設定するため、
            ``encode()`` が学習時と同じ分割を再現できる(学習と符号化で異なる値を
            使うと結果が壊れることに注意、``BPETokenizer.max_chunk_bytes`` の
            docstring を参照)。

    Returns:
        学習済みの ``BPETokenizer``。
    """
    chunks = pretokenize(text, chunk_split_mode, max_chunk_bytes)
    chunk_freq = Counter(chunks)

    words: list[list[str]]
    vocab: set[str]
    if byte_level:
        words = [[_BYTE_TO_UNICODE[b] for b in chunk.encode("utf-8")] for chunk in chunk_freq]
        vocab = set(_BYTE_TO_UNICODE.values())
    else:
        words = [list(chunk) for chunk in chunk_freq]
        vocab = {ch for chunk in chunk_freq for ch in chunk}

    freqs = list(chunk_freq.values())
    merges: list[tuple[str, str]] = []

    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_index: dict[tuple[str, str], set[int]] = defaultdict(set)
    for idx, (symbols, freq) in enumerate(zip(words, freqs, strict=True)):
        for pair, count in _count_pairs(symbols, freq).items():
            pair_counts[pair] += count
            pair_index[pair].add(idx)

    while len(vocab) < vocab_size and pair_counts:
        best_pair = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        new_symbol = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab.add(new_symbol)

        affected_indices = pair_index.pop(best_pair, set())
        for idx in affected_indices:
            old_symbols = words[idx]
            freq = freqs[idx]
            old_pairs = _count_pairs(old_symbols, freq)
            new_symbols = _merge_pair_in_word(old_symbols, best_pair, new_symbol)
            new_pairs = _count_pairs(new_symbols, freq)

            for pair, count in old_pairs.items():
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_index[pair].discard(idx)
                if not pair_index[pair]:
                    del pair_index[pair]
            for pair, count in new_pairs.items():
                pair_counts[pair] += count
                pair_index[pair].add(idx)

            words[idx] = new_symbols

    return BPETokenizer(
        merges=merges,
        vocab=vocab,
        byte_level=byte_level,
        chunk_split_mode=chunk_split_mode,
        max_chunk_bytes=max_chunk_bytes,
    )


def viterbi_segment(text: str, vocab: dict[str, float]) -> list[str]:
    """Unigram 言語モデル(Unigram Language Model)の下で文字列を最尤分割する
    (Viterbi アルゴリズム)。

    Unigram 言語モデルでは分割全体の対数尤度は各部分語の対数確率の和になるため
    (``P(x) = Π_i p(x_i)`` の対数)、最尤分割は「対数確率の和が最大になる分割」を
    求める問題になる。文字位置を頂点、語彙に含まれる部分語を辺とする有向非巡回
    グラフ(分割格子、lattice)の上で、動的計画法(Viterbi)により最良経路を求める。

    Args:
        text: 分割対象の文字列。SentencePiece の内部表現と揃えるため、呼び出し側で
            正規化(空白の ``▁`` への置換など、``UnigramTokenizer.normalize`` 参照)を
            済ませておく必要がある。
        vocab: ``{部分語 (piece): 対数確率 (log probability)}`` の辞書。

    Returns:
        最尤分割された部分語のリスト。

    Raises:
        ValueError: 語彙に含まれない文字があり、分割が完成しない場合。
    """
    n = len(text)
    neg_inf = float("-inf")
    best_score = [neg_inf] * (n + 1)
    best_score[0] = 0.0
    backpointer = [-1] * (n + 1)
    max_piece_len = max((len(piece) for piece in vocab), default=1)

    for end in range(1, n + 1):
        for start in range(max(0, end - max_piece_len), end):
            log_prob = vocab.get(text[start:end])
            if log_prob is None:
                continue
            score = best_score[start] + log_prob
            if score > best_score[end]:
                best_score[end] = score
                backpointer[end] = start

    if best_score[n] == neg_inf:
        raise ValueError(
            "分割できない文字が含まれています(語彙に単一文字の部分語が存在しない可能性があります)"
        )

    pieces: list[str] = []
    pos = n
    while pos > 0:
        start = backpointer[pos]
        pieces.append(text[start:pos])
        pos = start
    pieces.reverse()
    return pieces


@dataclass
class UnigramTokenizer:
    """sentencepiece で学習した Unigram 言語モデルの薄いラッパー。

    語彙学習(候補語彙からの EM ベースの反復的な縮小)は sentencepiece に委ねる。
    分割(最尤分割)のみを ``viterbi_segment`` としてスクラッチ実装し、
    sentencepiece 自身の分割結果と比較することで実装の正しさを検証する
    (005 の実装方針・実験5を参照)。

    Attributes:
        processor: 学習済み sentencepiece の ``SentencePieceProcessor``。
        vocab: ``{部分語: 対数確率}`` の辞書(特殊トークンを除く)。
    """

    processor: spm.SentencePieceProcessor
    vocab: dict[str, float]

    @staticmethod
    def normalize(text: str) -> str:
        """SentencePiece の既定の前処理(空白の ``▁``(U+2581)への置換と、
        文字列先頭への ``▁`` の付与)を再現する。

        ``train_unigram_model`` は ``normalization_rule_name="identity"`` かつ
        ``remove_extra_whitespaces=False`` で学習するため、sentencepiece 内部で
        行われる処理はこの空白置換と先頭付与のみになる。自作の Viterbi 分割
        (``viterbi_segment``)を sentencepiece の分割結果と公平に比較するには、
        入力テキストにこの正規化を明示的に適用してから渡す必要がある。
        """
        return "▁" + text.replace(" ", "▁")

    def encode_with_library(self, text: str) -> list[str]:
        """sentencepiece 自身の分割結果(内部的に最尤分割を行う)を返す。"""
        return self.processor.encode(text, out_type=str)

    def encode_with_viterbi(self, text: str) -> list[str]:
        """スクラッチ実装した ``viterbi_segment`` による分割結果を返す。"""
        return viterbi_segment(self.normalize(text), self.vocab)


def train_unigram_model(
    text: str,
    vocab_size: int,
    model_prefix: str | Path,
    byte_fallback: bool = False,
    character_coverage: float = 0.9995,
) -> UnigramTokenizer:
    """sentencepiece を用いて Unigram 言語モデルの語彙を学習する。

    語彙学習(候補語彙からの EM ベースの反復的な縮小、Kudo 2018)は sentencepiece の
    実装に委ねる。``normalization_rule_name="identity"`` を指定して NFKC 正規化などを
    無効化し、``UnigramTokenizer.normalize`` による手動の正規化と厳密に一致させる
    (実験5で自作の Viterbi 分割と sentencepiece 自身の分割を公平に比較するため)。

    Args:
        text: 学習コーパス。
        vocab_size: 目標語彙サイズ。
        model_prefix: 学習済みモデルの出力先パスの接頭辞
            (``{model_prefix}.model`` / ``{model_prefix}.vocab`` が生成される)。
        byte_fallback: True の場合、語彙に含まれない文字を UTF-8 バイト列(256 個の
            バイトトークン)へ退避させる(sentencepiece の byte fallback)。False
            (既定値、sentencepiece 自身の既定と同一)の場合、語彙に含まれない文字は
            すべて単一の未知語トークン(``<unk>``)に潰れ、その文字が本来持っていた
            情報が失われる(006 の言語モデル評価で、この情報損失を避けるために
            ``byte_fallback=True`` を明示的に指定する。005 の既定値
            ``byte_fallback=False`` は変えていないため、005 の呼び出し結果は
            変わらない)。
        character_coverage: 語彙に含める文字の被覆率。既定値 0.9995(sentencepiece
            自身の既定と同一)は訓練コーパスの低頻度文字の一部を語彙から除外する。
            006 では ``byte_fallback=True`` と組み合わせて 1.0 を指定し、除外された
            文字も byte fallback 経由で表現できるようにする。

    Returns:
        学習済みの ``UnigramTokenizer``。
    """
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = model_prefix.with_suffix(".train.txt")
    corpus_path.write_text(text, encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
        byte_fallback=byte_fallback,
        character_coverage=character_coverage,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    processor = spm.SentencePieceProcessor(model_file=str(model_prefix.with_suffix(".model")))
    vocab = {
        processor.id_to_piece(i): processor.get_score(i)
        for i in range(processor.get_piece_size())
        if i != processor.unk_id()
    }
    return UnigramTokenizer(processor=processor, vocab=vocab)
