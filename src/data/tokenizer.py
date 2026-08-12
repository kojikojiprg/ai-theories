"""部分語分割(Subword Tokenization)のスクラッチ実装。

Byte Pair Encoding(BPE)の学習・符号化と、Unigram 言語モデル(Unigram Language
Model)の Viterbi 最尤分割をスクラッチ実装する。Unigram 言語モデルの語彙学習
(候補語彙からの EM ベースの反復的な縮小)は sentencepiece に委ねる
(``theories/02_pretraining/005_tokenizer.ipynb`` の実装方針を参照)。

記号 / Notation:
    V : 語彙サイズ(vocabulary size)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import sentencepiece as spm

ChunkSplitMode = Literal["whitespace", "none"]


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


def pretokenize(text: str, chunk_split_mode: ChunkSplitMode) -> list[str]:
    """事前分割(pre-tokenization)。BPE のマージはチャンクの内部でのみ行われ、
    チャンクをまたいでは行われない。

    Args:
        text: 分割対象のテキスト。
        chunk_split_mode:
            ``"whitespace"``: 空白文字(スペース・タブ・改行)で分割する
                (英語などスペース区切りの言語で一般的な単語分割)。
            ``"none"``: 事前分割を行わない。ただし改行のみは区切りとして扱う
                (実装上の理由。改行をまたぐマージを許すとチャンクが際限なく
                大きくなり実装が煩雑になるため。空白の有無のみを実験の対照条件と
                するための単純化であり、005 の実装方針で明記する)。
    """
    if chunk_split_mode == "whitespace":
        return text.split()
    if chunk_split_mode == "none":
        return [line for line in text.split("\n") if line]
    raise ValueError(f"未知の chunk_split_mode: {chunk_split_mode!r}")


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
    """

    merges: list[tuple[str, str]]
    vocab: set[str]
    byte_level: bool
    chunk_split_mode: ChunkSplitMode
    unk_token: str = "<unk>"

    @cached_property
    def merge_ranks(self) -> dict[tuple[str, str], int]:
        """マージ規則の学習順の順位(rank)。符号化時に「最も早く学習された
        マージ規則」を優先して適用するために使う(rank が小さいほど優先)。"""
        return {pair: rank for rank, pair in enumerate(self.merges)}

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
        """テキスト全体を部分語シンボル列に変換する(チャンクをまたぐマージは行わない)。"""
        tokens: list[str] = []
        for chunk in pretokenize(text, self.chunk_split_mode):
            tokens.extend(self.encode_chunk(chunk))
        return tokens


def learn_bpe(
    text: str,
    vocab_size: int,
    byte_level: bool = False,
    chunk_split_mode: ChunkSplitMode = "whitespace",
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

    Returns:
        学習済みの ``BPETokenizer``。
    """
    chunks = pretokenize(text, chunk_split_mode)
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


def train_unigram_model(text: str, vocab_size: int, model_prefix: str | Path) -> UnigramTokenizer:
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
