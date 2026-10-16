"""タブ 2(指示への応答)の指示の組み立て・応答の生成・採点。

016 の合成の指示データ(``src/data/instruction.py``)の定義と、016 の評価と同じ生成・採点
(``greedy_generate_until_stop``・``score_generated_response``)をそのまま使う。

- 課題は ``TASK_NAMES`` の 4 種類で、入力(英単語の列)から応答が規則で決まる。
- 指示文は、学習に使った 4 個(見たことのあるテンプレート)と、使っていない 2 個
  (未見のテンプレート)が課題ごとにある。
- ``sft-synthetic`` ブランチのモデルは、016 の条件 1(損失マスクあり・短い水準)のレシピで学習した
  ものなので、指示の前に長い前置きは付けない(短い水準のプロンプトだけを使う)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from lab.core.artifacts import LoadedModel
from src.data.instruction import (
    END_MARKER,
    SEEN_INSTRUCTIONS,
    TASK_NAMES,
    UNSEEN_INSTRUCTIONS,
    WORD_VOCABULARY,
    apply_task,
    format_prompt,
    sample_word_sequences,
    score_generated_response,
)
from src.data.tokenizer import BPEIDTokenizer
from src.generation.stopping import greedy_generate_until_stop

MAX_NEW_TOKENS = 32
"""生成するトークン数の上限。016 の評価と同じ(応答部分の最大長 22 に余裕を持たせた値)。"""

MIN_WORDS, MAX_WORDS = 3, 8
"""入力の単語数の範囲。016 の学習・評価と同じ。"""

TEMPLATE_SEEN = "seen"
TEMPLATE_UNSEEN = "unseen"
TEMPLATE_KINDS = (TEMPLATE_SEEN, TEMPLATE_UNSEEN)

_WORD_PATTERN = re.compile(r"[a-z]+")
_VOCABULARY = frozenset(WORD_VOCABULARY)


def instruction_choices(task: str, kind: str) -> list[str]:
    """課題 ``task`` の指示文のうち、``kind``(見たことのある / 未見)のものを返す。"""
    if task not in TASK_NAMES:
        raise ValueError(f"未知の課題: {task!r}")
    if kind == TEMPLATE_SEEN:
        return list(SEEN_INSTRUCTIONS[task])
    if kind == TEMPLATE_UNSEEN:
        return list(UNSEEN_INSTRUCTIONS[task])
    raise ValueError(f"未知のテンプレートの種類: {kind!r}")


def template_kind_of(task: str, instruction: str) -> str:
    """指示文が、課題 ``task`` の見たことのあるテンプレートか未見のテンプレートかを返す。

    Raises:
        ValueError: どちらにも含まれない指示文の場合(このアプリは自由な指示文を受け付けない)。
    """
    if task not in TASK_NAMES:
        raise ValueError(f"未知の課題: {task!r}")
    if instruction in SEEN_INSTRUCTIONS[task]:
        return TEMPLATE_SEEN
    if instruction in UNSEEN_INSTRUCTIONS[task]:
        return TEMPLATE_UNSEEN
    raise ValueError(
        "選んだ指示文が、この課題のテンプレートの一覧にありません。選び直してください。"
    )


def random_words(seed: int) -> str:
    """016 と同じ語彙から、重複なしの単語を ``MIN_WORDS`` 〜 ``MAX_WORDS`` 個返す。

    単語の数と選び方はシードで決まる。
    """
    rng = np.random.default_rng(seed)
    (sequence,) = sample_word_sequences(rng, 1, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS)
    return " ".join(sequence)


@dataclass(frozen=True)
class ParsedWords:
    """入力欄の単語。``warnings`` は、実行はできるが学習の分布の外にあることの注意。"""

    words: tuple[str, ...]
    warnings: tuple[str, ...]


def parse_words(text: str) -> ParsedWords:
    """入力欄の文字列(空白かカンマ区切り)を単語の列にする。

    Raises:
        ValueError: 単語数が範囲外、または英小文字以外を含む場合。
    """
    words = tuple(word.lower() for word in re.split(r"[\s,]+", text.strip()) if word)
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        raise ValueError(
            f"単語は {MIN_WORDS} 個以上 {MAX_WORDS} 個以下にしてください(今は {len(words)} 個)。"
        )
    for word in words:
        if not _WORD_PATTERN.fullmatch(word):
            raise ValueError(f"単語は英字だけにしてください: {word!r}")

    warnings: list[str] = []
    outside = [w for w in words if w not in _VOCABULARY]
    if outside:
        warnings.append(
            f"学習で使った単語の一覧({len(WORD_VOCABULARY)} 語)にない単語があります: "
            f"{', '.join(outside)}。学習の分布の外なので、応答が崩れやすくなります。"
        )
    if len(set(words)) != len(words):
        warnings.append("同じ単語が重複しています。学習の入力は重複のない単語の列でした。")
    return ParsedWords(words, tuple(warnings))


@dataclass(frozen=True)
class InstructionOutcome:
    """指示への応答の結果。

    Attributes:
        prompt: モデルに入力した指示部分(プロンプト)。
        response_text: モデルが生成した応答部分の文字列(終端記号まで、または上限トークン数まで)。
        answer: 規則で決まる正解の単語の列。
        format_ok: 終端記号 ``### End`` で終わり、それより前に他の区切り記号がないか
            (016 の形式の遵守)。
        exact_match: 形式を守り、かつ単語の列が正解と一致したか(016 の完全一致)。
        num_new_tokens: 生成したトークン数。
    """

    prompt: str
    response_text: str
    answer: tuple[str, ...]
    format_ok: bool
    exact_match: bool
    num_new_tokens: int


def check_instruction_request(
    loaded: LoadedModel,
    tokenizer: BPEIDTokenizer,
    task: str,
    instruction: str,
    words: tuple[str, ...],
) -> tuple[str, list[int]]:
    """GPU を使わずに判定できる入力の検査を行い、プロンプトとそのトークン ID を返す。

    ``@spaces.GPU`` の関数に入る前に呼ぶ(誤った入力で ZeroGPU の利用枠を使わないため)。
    ``respond`` も、同じ検査を最初に行う。入力の単語の検査は ``parse_words`` が行う。

    Raises:
        ValueError: 指示文が一覧にない、またはプロンプトが長すぎる場合
            (メッセージは画面に表示できる日本語)。
    """
    template_kind_of(task, instruction)  # 一覧にない指示文は例外にする
    prompt = format_prompt(instruction, words)
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) + MAX_NEW_TOKENS > loaded.max_sequence_length:
        raise ValueError("指示が長すぎます。単語を減らしてください。")
    return prompt, prompt_ids


def respond(
    loaded: LoadedModel,
    tokenizer: BPEIDTokenizer,
    device: torch.device,
    task: str,
    instruction: str,
    words: tuple[str, ...],
) -> InstructionOutcome:
    """指示を組み立て、貪欲法で終端記号まで生成して、016 と同じ基準で採点する。"""
    prompt, prompt_ids = check_instruction_request(loaded, tokenizer, task, instruction, words)
    answer = apply_task(task, words)

    (generated,) = greedy_generate_until_stop(
        loaded.model,
        [prompt_ids],
        tokenizer.decode,
        END_MARKER,
        MAX_NEW_TOKENS,
        device,
        batch_size=1,
    )
    response_text = tokenizer.decode(generated)
    format_ok, exact_match = score_generated_response(response_text, answer)
    return InstructionOutcome(
        prompt=prompt,
        response_text=response_text,
        answer=answer,
        format_ok=format_ok,
        exact_match=exact_match,
        num_new_tokens=len(generated),
    )
