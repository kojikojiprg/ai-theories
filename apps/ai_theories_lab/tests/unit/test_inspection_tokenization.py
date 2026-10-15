"""モデルの中を見る: トークン分割と BPE の結合の過程(005)。"""

from __future__ import annotations

import random

import pytest

from lab.core.artifacts import Resources
from lab.features.inspection.service import (
    MAX_INSPECTION_TOKENS,
    merge_symbol_pair,
    readable_symbol,
    tokenize,
    trace_chunk,
)
from src.data.tokenizer import _merge_pair_in_word, pretokenize
from tests.stand_in import TRAINING_SENTENCES

TEXTS = [
    *TRAINING_SENTENCES,
    "The film was released in 1999.",
    "unbelievable   spaces\tand\nnewlines",
    "café naïve — “quotes” and 日本語 and 🙂",
    "a",
    "  leading and trailing  ",
    "x" * 100,  # 1 チャンクが max_chunk_bytes(64)を超えて分割される(トークン数は上限の 128 以内)
]


@pytest.mark.parametrize("text", TEXTS)
def test_the_recorded_process_ends_at_the_result_of_encode_chunk(
    resources: Resources, text: str
) -> None:
    tokenizer = resources.tokenizer
    bpe = tokenizer.bpe_tokenizer
    for chunk in pretokenize(text, bpe.chunk_split_mode, bpe.max_chunk_bytes):
        trace = trace_chunk(tokenizer, chunk)
        assert list(trace.final_symbols) == bpe.encode_chunk(chunk)
        assert trace.token_ids == tuple(tokenizer.symbol_to_id[s] for s in trace.final_symbols)


@pytest.mark.parametrize("text", TEXTS)
def test_the_token_ids_are_those_of_the_tokenizer(resources: Resources, text: str) -> None:
    tokenization = tokenize(resources.tokenizer, text)
    assert list(tokenization.token_ids) == resources.tokenizer.encode(text)
    assert [p.index for p in tokenization.pieces] == list(range(len(tokenization.pieces)))


@pytest.mark.parametrize("text", TEXTS)
def test_the_tokens_decode_back_to_the_text(resources: Resources, text: str) -> None:
    tokenization = tokenize(resources.tokenizer, text)
    assert resources.tokenizer.decode(list(tokenization.token_ids)) == text
    assert "".join(trace.chunk for trace in tokenization.chunks) == text


@pytest.mark.parametrize("text", TEXTS)
def test_the_merge_rank_increases_at_every_step(resources: Resources, text: str) -> None:
    """結合した規則の順位は、段階ごとに必ず大きくなる(結合で新しくできるペアは、そのシンボルを
    作った規則より後に学習されたものだけで、残りのペアはすべて、選ばれた順位より後のものだから)。"""
    for trace in tokenize(resources.tokenizer, text).chunks:
        ranks = [step.rank for step in trace.steps]
        assert ranks == sorted(set(ranks))


@pytest.mark.parametrize("text", TEXTS)
def test_every_step_keeps_the_bytes_and_creates_the_merged_symbol(
    resources: Resources, text: str
) -> None:
    for trace in tokenize(resources.tokenizer, text).chunks:
        joined = "".join(trace.initial_symbols)
        previous = trace.initial_symbols
        for step in trace.steps:
            assert "".join(step.symbols_after) == joined
            assert step.pair[0] + step.pair[1] in step.symbols_after
            assert len(step.symbols_after) < len(previous)
            previous = step.symbols_after


def test_initial_symbols_are_one_byte_each(resources: Resources) -> None:
    (trace,) = tokenize(resources.tokenizer, "café").chunks
    assert len(trace.initial_symbols) == len("café".encode())


def test_merge_symbol_pair_matches_the_merge_in_src() -> None:
    rng = random.Random(0)
    alphabet = ["a", "b", "c", "ab", "bc"]
    for _ in range(300):
        symbols = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
        pair = (rng.choice(alphabet), rng.choice(alphabet))
        assert list(merge_symbol_pair(symbols, pair, pair[0] + pair[1])) == _merge_pair_in_word(
            symbols, pair, pair[0] + pair[1]
        )


def test_readable_symbol_shows_whitespace_visibly(resources: Resources) -> None:
    bpe = resources.tokenizer.bpe_tokenizer
    (space_the,) = [s for s in bpe.encode(" the") if "t" in s][:1]
    assert readable_symbol(space_the).startswith("␣")
    newline_symbol = bpe.encode("\n")[0]
    assert readable_symbol(newline_symbol) == "⏎"


def test_an_empty_text_is_rejected(resources: Resources) -> None:
    for text in ("", "   ", "\n"):
        with pytest.raises(ValueError):
            tokenize(resources.tokenizer, text)


def test_a_text_with_too_many_tokens_is_rejected(resources: Resources) -> None:
    with pytest.raises(ValueError, match=str(MAX_INSPECTION_TOKENS)):
        tokenize(resources.tokenizer, "word " * (MAX_INSPECTION_TOKENS + 1))
