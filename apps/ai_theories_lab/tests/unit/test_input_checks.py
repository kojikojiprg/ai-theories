"""入力の検査の関数(``check_*``): GPU を使わずに判定できる検査を、``service.py`` に置く。

画面のイベントの関数が、``@spaces.GPU`` の関数に入る前にこれらを呼び、``ValueError`` を
``gr.Error`` に変える。計算の関数(``generate_events``・``respond``・``analyze``)も、同じ検査を最初に
行う(単独で呼ばれても、誤った入力を受け付けない)ので、**検査の関数と計算の関数が同じメッセージ**に
なることを確かめる。
"""

from __future__ import annotations

import dataclasses

import pytest

from lab.core.artifacts import Resources
from lab.features.generation.service import (
    METHOD_BEAM,
    METHOD_GREEDY,
    DecodingSettings,
    check_generation_request,
    generate_events,
)
from lab.features.inspection.service import (
    MAX_INSPECTION_TOKENS,
    analyze,
    check_analysis_request,
    check_token_count,
    tokenize,
)
from lab.features.instruction.service import (
    check_instruction_request,
    instruction_choices,
    respond,
)
from src.data.instruction import format_prompt
from tests.stand_in import TRAINING_SENTENCES

PROMPT = "The history of the city began when"


def _message(call) -> str:
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


# ---- 文章の生成 ----
def test_generation_check_returns_the_prompt_tokens(resources: Resources) -> None:
    settings = DecodingSettings(method=METHOD_GREEDY, max_new_tokens=8)
    ids = check_generation_request(resources.models["main"], resources.tokenizer, PROMPT, settings)
    assert ids == resources.tokenizer.encode(PROMPT)


@pytest.mark.parametrize(
    ("prompt", "settings"),
    [
        ("", DecodingSettings(method=METHOD_GREEDY, max_new_tokens=8)),
        ("word " * 400, DecodingSettings(method=METHOD_GREEDY, max_new_tokens=8)),
        (PROMPT, DecodingSettings(method=METHOD_GREEDY, max_new_tokens=0)),
        (PROMPT, DecodingSettings(method=METHOD_GREEDY, max_new_tokens=250)),
        (PROMPT, DecodingSettings(method="unknown", max_new_tokens=8)),
        (PROMPT, DecodingSettings(method=METHOD_BEAM, max_new_tokens=8, beam_size=0)),
    ],
    ids=["empty", "long-prompt", "zero-tokens", "cache-overflow", "unknown-method", "zero-beams"],
)
def test_generation_check_and_generate_events_give_the_same_message(
    resources: Resources, prompt: str, settings: DecodingSettings
) -> None:
    loaded, tokenizer = resources.models["main"], resources.tokenizer
    checked = _message(lambda: check_generation_request(loaded, tokenizer, prompt, settings))
    computed = _message(
        lambda: list(generate_events(loaded, tokenizer, resources.device, prompt, settings))
    )
    assert checked == computed
    assert checked


# ---- 指示への応答 ----
WORDS = ("cat", "dog", "house", "tree")


def test_instruction_check_returns_the_prompt_and_its_tokens(resources: Resources) -> None:
    instruction = instruction_choices("reverse", "seen")[0]
    prompt, ids = check_instruction_request(
        resources.models["sft-synthetic"], resources.tokenizer, "reverse", instruction, WORDS
    )
    assert prompt == format_prompt(instruction, WORDS)
    assert ids == resources.tokenizer.encode(prompt)


@pytest.mark.parametrize(
    ("task", "instruction", "words"),
    [
        ("reverse", "Do something with the words.", WORDS),
        ("unknown", "Reverse the order of the words.", WORDS),
        ("reverse", instruction_choices("reverse", "seen")[0], ("x" * 300,)),
    ],
    ids=["unlisted-instruction", "unknown-task", "too-long"],
)
def test_instruction_check_and_respond_give_the_same_message(
    resources: Resources, task: str, instruction: str, words: tuple[str, ...]
) -> None:
    loaded, tokenizer = resources.models["sft-synthetic"], resources.tokenizer
    checked = _message(
        lambda: check_instruction_request(loaded, tokenizer, task, instruction, words)
    )
    computed = _message(
        lambda: respond(loaded, tokenizer, resources.device, task, instruction, words)
    )
    assert checked == computed
    assert checked


# ---- モデルの中を見る ----
@pytest.mark.parametrize(
    ("num_tokens", "valid"),
    [(0, False), (1, True), (MAX_INSPECTION_TOKENS, True), (MAX_INSPECTION_TOKENS + 1, False)],
)
def test_token_count_check_boundaries(num_tokens: int, valid: bool) -> None:
    if valid:
        check_token_count(num_tokens)
    else:
        assert str(num_tokens) in _message(lambda: check_token_count(num_tokens))


def test_analysis_check_returns_the_same_tokenization(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TRAINING_SENTENCES[0])
    assert check_analysis_request(tokenization) is tokenization


def test_analysis_check_asks_for_the_tokenization_first() -> None:
    assert "先にトークン分割" in _message(lambda: check_analysis_request(None))


def test_analysis_check_and_analyze_give_the_same_message(resources: Resources) -> None:
    base = tokenize(resources.tokenizer, TRAINING_SENTENCES[0])
    too_long = dataclasses.replace(base, pieces=base.pieces * 20)
    checked = _message(lambda: check_analysis_request(too_long))
    computed = _message(
        lambda: analyze(resources.models["main"], resources.device, too_long.token_ids)
    )
    assert checked == computed
    assert str(len(too_long.token_ids)) in checked
