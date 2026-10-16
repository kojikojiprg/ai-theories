"""指示への応答の処理(016 の合成課題・017 の SFT モデル)。"""

from __future__ import annotations

import pytest

from lab.core.artifacts import Resources
from lab.features.instruction.service import (
    MAX_NEW_TOKENS,
    MAX_WORDS,
    MIN_WORDS,
    TEMPLATE_KINDS,
    InstructionOutcome,
    instruction_choices,
    parse_words,
    random_words,
    respond,
    template_kind_of,
)
from lab.features.instruction.ui import format_instruction_outcome
from src.data.instruction import TASK_NAMES, WORD_VOCABULARY, apply_task, format_prompt


@pytest.mark.parametrize("task", TASK_NAMES)
def test_seen_templates_are_four_per_task(task: str) -> None:
    assert len(instruction_choices(task, "seen")) == 4


@pytest.mark.parametrize("task", TASK_NAMES)
def test_unseen_templates_are_two_per_task(task: str) -> None:
    assert len(instruction_choices(task, "unseen")) == 2


@pytest.mark.parametrize("task", TASK_NAMES)
def test_template_kind_can_be_looked_up_from_the_instruction(task: str) -> None:
    assert all(template_kind_of(task, t) == "seen" for t in instruction_choices(task, "seen"))
    assert all(template_kind_of(task, t) == "unseen" for t in instruction_choices(task, "unseen"))


def test_template_kinds_are_seen_and_unseen() -> None:
    assert set(TEMPLATE_KINDS) == {"seen", "unseen"}


def test_an_instruction_outside_the_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        template_kind_of("reverse", "Do something else.")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("one two", id="too few words"),
        pytest.param(" ".join(["cat"] * (MAX_WORDS + 1)), id="too many words"),
        pytest.param("cat dog 3d", id="not alphabetic"),
    ],
)
def test_invalid_words_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_words(text)


def test_words_are_normalized() -> None:
    parsed = parse_words("Cat, ghost  house")
    assert parsed.words == ("cat", "ghost", "house")
    assert not parsed.warnings


def test_words_outside_the_training_vocabulary_get_a_warning() -> None:
    assert parse_words("cat dog zzzz").warnings


def test_duplicate_words_get_a_warning() -> None:
    assert parse_words("cat cat dog").warnings


def test_random_words_are_decided_by_the_seed_and_come_from_the_vocabulary() -> None:
    words = random_words(0).split()
    assert random_words(0) == random_words(0)
    assert MIN_WORDS <= len(words) <= MAX_WORDS
    assert set(words) <= set(WORD_VOCABULARY)
    assert len(set(words)) == len(words)


@pytest.mark.parametrize("task", TASK_NAMES)
def test_respond_builds_the_short_level_prompt_and_the_rule_answer(
    resources: Resources, task: str
) -> None:
    words = tuple(random_words(0).split())
    instruction = instruction_choices(task, "seen")[0]
    outcome = respond(
        resources.models["sft-synthetic"],
        resources.tokenizer,
        resources.device,
        task,
        instruction,
        words,
    )
    assert outcome.prompt == format_prompt(instruction, words)
    assert outcome.answer == apply_task(task, words)
    assert outcome.num_new_tokens <= MAX_NEW_TOKENS


@pytest.mark.parametrize(
    ("format_ok", "exact_match", "marker"),
    [(True, True, "✅"), (True, False, "⚠️"), (False, False, "❌")],
)
def test_outcome_is_displayed_for_each_verdict(
    format_ok: bool, exact_match: bool, marker: str
) -> None:
    outcome = InstructionOutcome(
        prompt="p",
        response_text=" a b\n### End",
        answer=("a", "b"),
        format_ok=format_ok,
        exact_match=exact_match,
        num_new_tokens=4,
    )
    report = format_instruction_outcome(
        "reverse", instruction_choices("reverse", "unseen")[0], outcome
    )
    assert marker in report
    assert "未見のテンプレート" in report
    assert "`a b`" in report
