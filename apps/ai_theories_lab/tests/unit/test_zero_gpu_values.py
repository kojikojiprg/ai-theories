"""ZeroGPU のプロセス境界を越える値: 引数・戻り値・生成器の yield・例外。

ZeroGPU では、``@spaces.GPU`` を付けた関数が別のプロセスで動き、引数と戻り値(生成器なら yield した
値の 1 つずつ)が ``multiprocessing`` のキューで受け渡される。手元(MPS・CPU)では ``@spaces.GPU`` は
何もしないので、この境界は現れない。そのため、**境界を越える値の型を、試験の側で調べる。**
手元に GPU のテンソルは出てこないので、「CUDA のテンソルがない」だけでは何も検出できない。
許可リストで型を調べ、その検査が実際に違反を検出できること(検出力)も、この中で確かめる。

1. 検査器の検出力(違反する値を見つけること)。
2. ``service.py`` の関数のうち、``@gpu`` の関数から呼ぶものの戻り値(生成器は yield のすべて)。
3. ``@gpu`` を付けた関数そのものの、引数・戻り値・yield。
4. ``@gpu`` を付けた関数の引数の型に、モデルやトークナイザがないこと。
5. ``@gpu`` を付けた関数が、起動時に読み込んだモデルを関数の中で参照していること。
6. 例外: 画面に出すメッセージが境界を越えること(``gr.Error`` だけがメッセージごと届く)。
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import gradio as gr
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from torch import Tensor

from lab.core.artifacts import Resources
from lab.features.generation import ui as generation_ui
from lab.features.generation.service import (
    DECODING_METHODS,
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TOP_K,
    METHOD_TOP_P,
    DecodingSettings,
    GenerationProgress,
    generate_events,
    run_beam_search,
    stream_generation,
)
from lab.features.generation.ui import EXAMPLE_PROMPTS
from lab.features.inspection import ui as inspection_ui
from lab.features.inspection.service import ModelAnalysis, Tokenization, analyze, tokenize
from lab.features.instruction import ui as instruction_ui
from lab.features.instruction.service import TASK_NAMES, instruction_choices, respond
from src.data.tokenizer import BPEIDTokenizer
from src.models.gpt import GPTLanguageModel
from tests.stand_in import TRAINING_SENTENCES
from tests.zero_gpu import (
    GpuApp,
    argument_type_problems,
    assert_same_value,
    assert_transferable,
    cross_process_boundary,
    find_violations,
    gpu_function_names_in_source,
)

# ---------------------------------------------------------------------------
# 1. 検査器の検出力
# ---------------------------------------------------------------------------


@dataclass
class OutsideDataclass:
    """このアプリ(``lab.``)の外で定義された dataclass。"""

    value: int = 0


class IntSubclass(int):
    """int の派生クラス。"""


ACCEPTED: dict[str, Any] = {
    "None": None,
    "bool": True,
    "int": 3,
    "float": float("nan"),
    "str": "日本語",
    "bytes": b"x",
    "tuple": (1, "a", (2.0,)),
    "list": [1, [2]],
    "dict": {"a": 1, 2: (3,)},
    "set": {1, 2},
    "frozenset": frozenset({"a"}),
    "float32 array": np.zeros((2, 3), dtype=np.float32),
    "int32 array": np.arange(3, dtype=np.int32),
    "bool array": np.array([True, False]),
    "numpy scalar": np.float32(1.5),
    "CPU tensor": torch.zeros(2),
    "dataclass of the app": GenerationProgress((1, 2), 0.5, True),
}

REJECTED: dict[str, tuple[Any, str]] = {
    "torch.nn.Module": (torch.nn.Linear(2, 2), "Linear"),
    "tensor on a device other than the CPU": (torch.zeros(2, device="meta"), "CPU 以外のテンソル"),
    "tensor with a gradient": (torch.zeros(2, requires_grad=True), "勾配"),
    "Parameter": (torch.nn.Parameter(torch.zeros(2)), "派生クラス"),
    "torch.device": (torch.device("cpu"), "torch.device"),
    "object array": (np.array([object()], dtype=object), "数値でない numpy 配列"),
    "string array": (np.array(["a"]), "数値でない numpy 配列"),
    "lambda": (lambda: 0, "function"),
    "generator object": ((i for i in range(2)), "generator"),
    "matplotlib Figure": (Figure(), "Figure"),
    "Gradio component": (gr.Dropdown(), "Dropdown"),
    "class": (int, "type"),
    "subclass of int": (IntSubclass(1), "IntSubclass"),
    "dataclass outside the app": (OutsideDataclass(), "このアプリの外"),
    "dict with a tuple key": ({(1, 2): 3}, "キー"),
}


@pytest.mark.parametrize("name", list(ACCEPTED))
def test_the_checker_accepts_plain_values(name: str) -> None:
    value = ACCEPTED[name]
    assert find_violations(value) == []
    assert_transferable(value, name, allow_tensors=True)


@pytest.mark.parametrize("name", list(REJECTED))
def test_the_checker_rejects_values_that_cannot_cross(name: str) -> None:
    value, expected = REJECTED[name]
    problems = find_violations(value)
    assert problems, f"{name} を許可リストが見逃した"
    assert expected in "\n".join(problems)
    with pytest.raises(AssertionError, match="越えられない"):
        assert_transferable(value, name)


def test_the_app_policy_also_rejects_a_tensor_on_the_cpu() -> None:
    """MPS や CPU の上では、``.cpu()`` を付け忘れたテンソルが「CPU 以外」にならない。"""
    value = {"attention": torch.zeros(2)}
    assert find_violations(value) == []  # 汎用の許可リストは CPU のテンソルを認める
    (problem,) = find_violations(value, "return", allow_tensors=False)
    assert problem.startswith("return['attention']: テンソル")
    with pytest.raises(AssertionError, match="越えられない"):
        assert_transferable(value, "return")


def test_the_checker_reports_where_in_a_nested_value_the_problem_is() -> None:
    value = {"results": [("ok", torch.zeros(2, device="meta"))]}
    (problem,) = find_violations(value, "return")
    assert problem.startswith("return['results'][0][1]:")


def test_a_dataclass_is_checked_field_by_field() -> None:
    analysis = ModelAnalysis(
        token_ids=(1, 2),
        attention=np.zeros((1, 1, 2, 2), dtype=np.float32),
        entropy=torch.zeros(2, device="meta"),  # フィールドの 1 つだけが違反
        next_probability=np.zeros(2, dtype=np.float32),
        next_rank=np.zeros(2, dtype=np.int32),
        top_ids=np.zeros((2, 1), dtype=np.int32),
        top_probabilities=np.zeros((2, 1), dtype=np.float32),
        vocabulary_size=3,
    )
    (problem,) = find_violations(analysis, "return")
    assert problem.startswith("return.entropy:")


@pytest.mark.parametrize(
    "value",
    [torch.nn.Linear(2, 2), torch.zeros(2, device="meta")],
    ids=["torch.nn.Module", "tensor on a device other than the CPU"],
)
def test_pickling_alone_cannot_find_these_but_the_allow_list_can(value: Any) -> None:
    """ZeroGPU の受け渡しと同じ pickle は、どちらも例外にならない(検出できない)。"""
    pickle.dumps(value)
    cross_process_boundary(value)
    assert find_violations(value)


def test_the_round_trip_fails_for_a_value_that_cannot_be_pickled() -> None:
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        cross_process_boundary(lambda: 0)


def test_the_comparison_detects_a_difference_after_the_round_trip() -> None:
    base = np.array([1.0, np.nan], dtype=np.float32)
    assert_same_value(base, base.copy())  # NaN どうしは等しい
    different = [
        base.astype(np.float64),  # dtype
        base[:1],  # 形状
        np.array([1.0, 2.0], dtype=np.float32),  # 値
        [1.0, float("nan")],  # 型
    ]
    for other in different:
        with pytest.raises(AssertionError):
            assert_same_value(base, other)
    with pytest.raises(AssertionError):
        assert_same_value((1, 2), [1, 2])


# ---------------------------------------------------------------------------
# 2. service.py の関数の戻り値(@gpu の関数から呼ぶもの)
# ---------------------------------------------------------------------------
PROMPT = EXAMPLE_PROMPTS[0]


def _generation(**overrides: Any) -> dict[str, Any]:
    """``generate_text`` の引数(スライダー・数値欄は float で届くので、そのとおりにする)。"""
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


def _comparison(**overrides: Any) -> dict[str, Any]:
    arguments = _generation(method_a=METHOD_GREEDY, method_b=METHOD_TOP_P) | overrides
    arguments.pop("method")
    return arguments


@pytest.mark.parametrize(
    ("method", "use_cache"),
    [(method, True) for method in DECODING_METHODS] + [(METHOD_GREEDY, False)],
)
def test_generate_events_yields_values_that_cross_the_boundary(
    resources: Resources, method: str, use_cache: bool
) -> None:
    settings = DecodingSettings(method=method, max_new_tokens=5, beam_size=2, use_cache=use_cache)
    events = list(
        generate_events(
            resources.models["main"], resources.tokenizer, resources.device, PROMPT, settings
        )
    )
    assert len(events) >= 2  # 逐次表示(ビームサーチは、計算中と最後の 2 回)
    for index, event in enumerate(events):
        assert_transferable(event, f"event[{index}]")


def test_stream_generation_yields_values_that_cross_the_boundary(resources: Resources) -> None:
    settings = DecodingSettings(method=METHOD_TOP_K, max_new_tokens=4)
    prompt_ids = resources.tokenizer.encode(PROMPT)
    progress = list(
        stream_generation(resources.models["main"].model, prompt_ids, settings, resources.device)
    )
    assert len(progress) == 4
    for index, item in enumerate(progress):
        assert_transferable(item, f"progress[{index}]")


def test_run_beam_search_returns_a_value_that_crosses_the_boundary(resources: Resources) -> None:
    settings = DecodingSettings(method=METHOD_BEAM, max_new_tokens=4, beam_size=2)
    prompt_ids = resources.tokenizer.encode(PROMPT)
    outcome = run_beam_search(
        resources.models["main"].model, prompt_ids, settings, resources.device
    )
    assert_transferable(outcome, "outcome")


@pytest.mark.parametrize("task", TASK_NAMES)
@pytest.mark.parametrize("kind", ["seen", "unseen"])
def test_respond_returns_a_value_that_crosses_the_boundary(
    resources: Resources, task: str, kind: str
) -> None:
    outcome = respond(
        resources.models["sft-synthetic"],
        resources.tokenizer,
        resources.device,
        task,
        instruction_choices(task, kind)[0],
        ("cat", "dog", "house", "tree"),
    )
    assert_transferable(outcome, "outcome")


@pytest.mark.parametrize("text", TRAINING_SENTENCES)
def test_analyze_returns_a_value_that_crosses_the_boundary(resources: Resources, text: str) -> None:
    tokenization = tokenize(resources.tokenizer, text)
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    assert_transferable(analysis, "analysis")


def test_analyze_of_a_single_token_crosses_the_boundary(resources: Resources) -> None:
    """最後の位置の確率は NaN になる。NaN を含む配列も、往復の後に同じ値で届く。"""
    analysis = analyze(resources.models["main"], resources.device, [5])
    assert np.isnan(analysis.next_probability).all()
    assert_transferable(analysis, "analysis")


def test_the_analysis_holds_only_numpy_arrays_and_python_numbers(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TRAINING_SENTENCES[0])
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    for field in dataclasses.fields(analysis):
        value = getattr(analysis, field.name)
        assert not isinstance(value, torch.Tensor), field.name
    assert {type(token_id) for token_id in analysis.token_ids} == {int}
    assert type(analysis.vocabulary_size) is int


# ---------------------------------------------------------------------------
# 3. @gpu を付けた関数の引数・戻り値・yield
# ---------------------------------------------------------------------------
GENERATOR_FUNCTIONS = {"generate_text", "compare_methods"}


def _inspection_arguments(resources: Resources) -> dict[str, Any]:
    return {"tokenization": tokenize(resources.tokenizer, TRAINING_SENTENCES[0])}


def _instruction_arguments(task: str, kind: str, words_text: str) -> dict[str, Any]:
    return {
        "task": task,
        "instruction": instruction_choices(task, kind)[0],
        "words_text": words_text,
    }


SUCCESS_CASES: list[Any] = [
    *(
        pytest.param(
            "generate_text", lambda r, m=method: _generation(method=m), id=f"generate_text-{method}"
        )
        for method in DECODING_METHODS
    ),
    pytest.param(
        "generate_text",
        lambda r: _generation(model_key="gqa", use_cache=False),
        id="generate_text-gqa-no-cache",
    ),
    pytest.param("compare_methods", lambda r: _comparison(), id="compare_methods-greedy-top_p"),
    pytest.param(
        "compare_methods",
        lambda r: _comparison(method_a=METHOD_BEAM, method_b=METHOD_TOP_K),
        id="compare_methods-beam-top_k",
    ),
    pytest.param(
        "answer_instruction",
        lambda r: _instruction_arguments("reverse", "seen", "cat dog house tree"),
        id="answer_instruction-seen",
    ),
    pytest.param(
        "answer_instruction",
        lambda r: _instruction_arguments("rotate_left", "unseen", "cat, dog house"),
        id="answer_instruction-unseen",
    ),
    pytest.param("run_model", _inspection_arguments, id="run_model"),
]

ERROR_CASES: list[Any] = [
    pytest.param("generate_text", lambda r: _generation(prompt=""), "空です", id="empty-prompt"),
    pytest.param(
        "generate_text", lambda r: _generation(max_new_tokens=0.0), "1 以上", id="zero-tokens"
    ),
    pytest.param(
        "generate_text", lambda r: _generation(prompt="word " * 400), "長すぎ", id="long-prompt"
    ),
    pytest.param(
        "compare_methods",
        lambda r: _comparison(method_a=METHOD_TOP_P, method_b=METHOD_TOP_P),
        "別のもの",
        id="same-methods",
    ),
    pytest.param("compare_methods", lambda r: _comparison(prompt=""), "空です", id="empty-compare"),
    pytest.param(
        "answer_instruction",
        lambda r: _instruction_arguments("reverse", "seen", "cat dog"),
        "単語は",
        id="too-few-words",
    ),
    pytest.param(
        "answer_instruction",
        lambda r: (
            _instruction_arguments("reverse", "seen", "cat dog house tree")
            | {"instruction": "Do something with the words."}
        ),
        "テンプレート",
        id="unlisted-instruction",
    ),
    pytest.param("run_model", lambda r: {"tokenization": None}, "先にトークン分割", id="no-state"),
    pytest.param(
        "run_model",
        lambda r: {
            "tokenization": dataclasses.replace(
                tokenize(r.tokenizer, TRAINING_SENTENCES[0]),
                pieces=tokenize(r.tokenizer, TRAINING_SENTENCES[0]).pieces * 20,
            )
        },
        "トークン数",
        id="too-many-tokens",
    ),
]


def _consume(result: Any) -> list[Any]:
    """生成器なら yield のすべて、そうでなければ戻り値 1 つを、境界を越えさせながら集める。"""
    if inspect.isgenerator(result):
        return [assert_transferable(item, f"yield[{i}]") for i, item in enumerate(result)]
    return [assert_transferable(result, "return")]


def test_every_gpu_function_has_a_case_below(gpu_app: GpuApp) -> None:
    """``@gpu`` を足したら、この試験の表にも足す(足さないと、境界を越える値が調べられない)。"""
    in_source = gpu_function_names_in_source()
    assert set(gpu_app.gpu_functions) == in_source
    assert {param.values[0] for param in SUCCESS_CASES} == in_source
    assert {param.values[0] for param in ERROR_CASES} == in_source


def test_every_gpu_function_is_the_function_the_event_calls(gpu_app: GpuApp) -> None:
    for name, record in gpu_app.gpu_functions.items():
        assert gpu_app.handler(name) is record.function, name
        assert inspect.isgeneratorfunction(record.function) == (name in GENERATOR_FUNCTIONS), name


@pytest.mark.parametrize(("name", "make_arguments"), SUCCESS_CASES)
def test_gpu_function_arguments_and_outputs_cross_the_boundary(
    resources: Resources, gpu_app: GpuApp, name: str, make_arguments: Callable[..., Any]
) -> None:
    function = gpu_app.gpu_functions[name].function
    sent = assert_transferable(make_arguments(resources), f"{name} の引数")
    outputs = _consume(function(**sent))

    if name == "generate_text":
        assert len(outputs) >= 2
        assert all(len(o) == 3 and all(type(v) is str for v in o) for o in outputs)
        assert outputs[-1][0].startswith(PROMPT)
    elif name == "compare_methods":
        assert len(outputs) >= 2
        assert all(len(o) == 6 and all(type(v) is str for v in o) for o in outputs)
    elif name == "answer_instruction":
        ((prompt, report),) = outputs
        assert type(prompt) is str
        assert type(report) is str
        assert "正解" in report
    else:
        (analysis,) = outputs
        assert isinstance(analysis, ModelAnalysis)
        assert analysis.token_ids == sent["tokenization"].token_ids


@pytest.mark.parametrize(("name", "make_arguments", "message"), ERROR_CASES)
def test_gpu_function_errors_keep_their_message_across_the_boundary(
    resources: Resources,
    gpu_app: GpuApp,
    name: str,
    make_arguments: Callable[..., Any],
    message: str,
) -> None:
    """ZeroGPU は、``gr.Error`` だけをメッセージごと画面に返す(``wrappers.py`` の
    ``exception_result``)。それ以外の例外は、クラス名だけの「ZeroGPU worker error」になり、
    日本語の説明が失われる。``@gpu`` の関数の中で起こりうる入力の誤りは、すべて
    ``gr.Error`` にする。
    """
    function = gpu_app.gpu_functions[name].function
    sent = assert_transferable(make_arguments(resources), f"{name} の引数")
    with pytest.raises(gr.Error) as caught:
        _consume(function(**sent))
    assert message in caught.value.message
    received = cross_process_boundary(caught.value)
    assert isinstance(received, gr.Error)
    assert received.message == caught.value.message


# ---------------------------------------------------------------------------
# 4. 引数の型にモデル・トークナイザがない
# ---------------------------------------------------------------------------
ALLOWED_ARGUMENT_TYPES: tuple[type, ...] = (
    str,
    int,
    float,
    bool,
    type(None),
    Tokenization,
    ModelAnalysis,
)


@pytest.mark.parametrize("name", sorted(gpu_function_names_in_source()))
def test_gpu_function_signatures_pass_no_model_or_tokenizer(gpu_app: GpuApp, name: str) -> None:
    function = gpu_app.gpu_functions[name].function
    assert argument_type_problems(function, ALLOWED_ARGUMENT_TYPES) == []


def _takes_a_module(model: torch.nn.Module) -> str:
    return ""


def _takes_the_model(model: GPTLanguageModel) -> str:
    return ""


def _takes_the_tokenizer(tokenizer: BPEIDTokenizer) -> str:
    return ""


def _takes_a_tensor(values: Tensor | None) -> str:
    return ""


def _takes_the_resources(resources: Resources) -> str:
    return ""


def _takes_an_unlisted_type(values: Figure) -> str:
    return ""


def _takes_without_annotation(values) -> str:
    return ""


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (_takes_a_module, "Module は渡さない"),
        (_takes_the_model, "GPTLanguageModel は渡さない"),
        (_takes_the_tokenizer, "BPEIDTokenizer は渡さない"),
        (_takes_a_tensor, "Tensor は渡さない"),
        (_takes_the_resources, "Resources は渡さない"),
        (_takes_an_unlisted_type, "許可リストにない型"),
        (_takes_without_annotation, "型の注釈がない"),
    ],
)
def test_the_signature_check_finds_models_tokenizers_and_unlisted_types(
    function: Callable[..., Any], expected: str
) -> None:
    problems = argument_type_problems(function, ALLOWED_ARGUMENT_TYPES)
    assert any(expected in problem for problem in problems), problems


# ---------------------------------------------------------------------------
# 5. 起動時に読み込んだモデルを、関数の中で参照する
# ---------------------------------------------------------------------------
def _spy(monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str) -> list[tuple[Any, ...]]:
    """``module.name`` を、呼ばれたときの位置引数を記録して元の関数へ渡すものに差し替える。"""
    original = getattr(module, name)
    calls: list[tuple[Any, ...]] = []

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


def test_gpu_functions_use_the_models_loaded_at_startup(
    resources: Resources, gpu_app: GpuApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """モデル・トークナイザ・デバイスは、引数ではなく、起動時の ``Resources`` から引く。"""
    generate_calls = _spy(monkeypatch, generation_ui, "generate_events")
    respond_calls = _spy(monkeypatch, instruction_ui, "respond")
    analyze_calls = _spy(monkeypatch, inspection_ui, "analyze")

    _consume(gpu_app.gpu_functions["generate_text"].function(**_generation(model_key="gqa")))
    _consume(gpu_app.gpu_functions["compare_methods"].function(**_comparison()))
    _consume(
        gpu_app.gpu_functions["answer_instruction"].function(
            **_instruction_arguments("reverse", "seen", "cat dog house tree")
        )
    )
    _consume(gpu_app.gpu_functions["run_model"].function(**_inspection_arguments(resources)))

    assert len(generate_calls) == 3  # generate_text が 1 回、compare_methods が 2 回
    assert generate_calls[0][0] is resources.models["gqa"]
    assert all(call[0] is resources.models["main"] for call in generate_calls[1:])
    assert all(call[1] is resources.tokenizer for call in generate_calls)
    assert all(call[2] is resources.device for call in generate_calls)
    ((loaded, tokenizer, device, *_),) = respond_calls
    assert loaded is resources.models["sft-synthetic"]
    assert tokenizer is resources.tokenizer
    assert device is resources.device
    ((loaded, device, *_),) = analyze_calls
    assert loaded is resources.models["main"]
    assert device is resources.device
