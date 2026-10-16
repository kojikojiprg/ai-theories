"""ZeroGPU の試験で共有する、画面のイベントの引数の表(正しい入力と、誤った入力)。

画面のイベントの関数(``generate_text``・``compare_methods``・``answer_instruction``・
``run_model``)は、入力の検査を先に行い、検査を通った入力だけを ``@gpu`` を付けた関数
(``*_on_gpu``)に渡す。引数の名前と並びは、Gradio が部品の値を渡すときのものと同じにしてある
(スライダー・数値欄は float で届く)。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from lab.core.artifacts import Resources
from lab.features.generation.service import (
    DECODING_METHODS,
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TOP_K,
    METHOD_TOP_P,
)
from lab.features.generation.ui import EXAMPLE_PROMPTS
from lab.features.inspection.service import tokenize
from lab.features.instruction.service import instruction_choices
from tests.stand_in import TRAINING_SENTENCES

PROMPT = EXAMPLE_PROMPTS[0]

GPU_FUNCTION_OF = {
    "generate_text": "generate_text_on_gpu",
    "compare_methods": "compare_methods_on_gpu",
    "answer_instruction": "answer_instruction_on_gpu",
    "run_model": "run_model_on_gpu",
}
"""画面のイベントの関数の名前から、その中で呼ぶ ``@gpu`` の関数の名前へ。"""
GENERATOR_HANDLERS = {"generate_text", "compare_methods"}
"""逐次表示のイベントの関数(生成器)。``@gpu`` の関数も生成器でなければならない。"""


def generation_arguments(**overrides: Any) -> dict[str, Any]:
    """``generate_text`` の引数。"""
    return {
        "prompt": PROMPT,
        "model_key": "main",
        "use_cache": True,
        "method": METHOD_GREEDY,
        "temperature": 1.0,
        "top_k": 40.0,
        "top_p": 0.9,
        "beam_size": 2.0,
        "length_penalty": 1.0,
        "max_new_tokens": 6.0,
        "seed": 0.0,
    } | overrides


def comparison_arguments(**overrides: Any) -> dict[str, Any]:
    """``compare_methods`` の引数。"""
    arguments = generation_arguments(method_a=METHOD_GREEDY, method_b=METHOD_TOP_P) | overrides
    arguments.pop("method")
    return arguments


def instruction_arguments(task: str, kind: str, words_text: str) -> dict[str, Any]:
    """``answer_instruction`` の引数。"""
    return {
        "task": task,
        "instruction": instruction_choices(task, kind)[0],
        "words_text": words_text,
    }


def inspection_arguments(resources: Resources) -> dict[str, Any]:
    """``run_model`` の引数。"""
    return {"tokenization": tokenize(resources.tokenizer, TRAINING_SENTENCES[0])}


def _too_many_tokens(resources: Resources) -> dict[str, Any]:
    base = tokenize(resources.tokenizer, TRAINING_SENTENCES[0])
    return {"tokenization": dataclasses.replace(base, pieces=base.pieces * 20)}


ArgumentFactory = Callable[[Resources], dict[str, Any]]

VALID_CASES: list[Any] = [
    *(
        pytest.param(
            "generate_text",
            lambda r, m=method: generation_arguments(method=m),
            id=f"generate_text-{method}",
        )
        for method in DECODING_METHODS
    ),
    pytest.param(
        "generate_text",
        lambda r: generation_arguments(model_key="gqa", use_cache=False),
        id="generate_text-gqa-no-cache",
    ),
    pytest.param("compare_methods", lambda r: comparison_arguments(), id="compare-greedy-top_p"),
    pytest.param(
        "compare_methods",
        lambda r: comparison_arguments(method_a=METHOD_BEAM, method_b=METHOD_TOP_K),
        id="compare-beam-top_k",
    ),
    pytest.param(
        "answer_instruction",
        lambda r: instruction_arguments("reverse", "seen", "cat dog house tree"),
        id="answer_instruction-seen",
    ),
    pytest.param(
        "answer_instruction",
        lambda r: instruction_arguments("rotate_left", "unseen", "cat, dog house"),
        id="answer_instruction-unseen",
    ),
    pytest.param("run_model", inspection_arguments, id="run_model"),
]
"""正しい入力。``(イベントの関数の名前, 引数を作る関数)``。"""

# 最初の 9 件は、入力の検査を ``@gpu`` の関数の外に出す前から表にあった誤り。最後の 1 件は、
# 外に出したことで生成の前に検査されるようになった誤り(比べる 2 つの方法のうち、2 つ目だけが誤り)。
INVALID_CASES: list[Any] = [
    pytest.param(
        "generate_text", lambda r: generation_arguments(prompt=""), "空です", id="empty-prompt"
    ),
    pytest.param(
        "generate_text",
        lambda r: generation_arguments(max_new_tokens=0.0),
        "1 以上",
        id="zero-tokens",
    ),
    pytest.param(
        "generate_text",
        lambda r: generation_arguments(prompt="word " * 400),
        "長すぎ",
        id="long-prompt",
    ),
    pytest.param(
        "compare_methods",
        lambda r: comparison_arguments(method_a=METHOD_TOP_P, method_b=METHOD_TOP_P),
        "別のもの",
        id="same-methods",
    ),
    pytest.param(
        "compare_methods", lambda r: comparison_arguments(prompt=""), "空です", id="empty-compare"
    ),
    pytest.param(
        "answer_instruction",
        lambda r: instruction_arguments("reverse", "seen", "cat dog"),
        "単語は",
        id="too-few-words",
    ),
    pytest.param(
        "answer_instruction",
        lambda r: (
            instruction_arguments("reverse", "seen", "cat dog house tree")
            | {"instruction": "Do something with the words."}
        ),
        "テンプレート",
        id="unlisted-instruction",
    ),
    pytest.param("run_model", lambda r: {"tokenization": None}, "先にトークン分割", id="no-state"),
    pytest.param("run_model", _too_many_tokens, "トークン数", id="too-many-tokens"),
    pytest.param(
        "compare_methods",
        # 方法 A(ビームサーチ)は生成できるが、方法 B(貪欲法)は KV キャッシュの上限を超える
        lambda r: (
            comparison_arguments(method_a=METHOD_BEAM, method_b=METHOD_GREEDY)
            | {"max_new_tokens": 250.0}
        ),
        "KV キャッシュ",
        id="second-method-is-invalid",
    ),
]
"""誤った入力。``(イベントの関数の名前, 引数を作る関数, 画面に出るメッセージに含まれる文字列)``。"""
