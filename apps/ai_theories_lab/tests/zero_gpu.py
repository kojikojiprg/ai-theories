"""ZeroGPU のプロセス境界を越える値の検査(試験で共有する)。

**ZeroGPU の受け渡し。** ``@spaces.GPU`` を付けた関数は、fork した別のプロセスで実行される。
引数も戻り値も、生成器なら yield した値の 1 つずつも、``multiprocessing`` の ``SimpleQueue`` で
受け渡される(``spaces`` 0.51.3 の ``zero/wrappers.py`` と ``utils.py``)。``SimpleQueue.put`` は
``ForkingPickler.dumps``、``get`` は ``ForkingPickler.loads`` を呼ぶだけなので、境界を越えられるかは
pickle できるかで決まる。ZeroGPU の外(手元の MPS や CPU)では ``@spaces.GPU`` は何もしないので、
**この境界は手元では現れない。** そこで、この境界を越えられる値を、試験の側で確かめる。

**pickle できるだけでは足りない。** ``torch.nn.Module`` も、CPU 以外のテンソルも、pickle の呼び出し
自体は成功する(CPU 以外のテンソルは別の仕組みで共有され、受け取る側のプロセスには GPU がない)。
さらに、手元には GPU のテンソルが出てこないので、「CUDA のテンソルがない」ことだけでは何も
検出できない。そこで、値の型を **許可リスト**(Python の値・numpy 配列・CPU のテンソル・このアプリの
dataclass とその入れ物)で調べ、許可リストにある型の中のテンソルは CPU 上にあることを調べる。
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import math
import pickle
import typing
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import ModuleType
from typing import Any

import gradio as gr
import numpy as np
import torch

from lab.core.artifacts import LoadedModel, Resources
from lab.ui.layout import build_app
from src.data.tokenizer import BPEIDTokenizer, BPETokenizer

APP_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = APP_DIR / "lab" / "features"

# ---------------------------------------------------------------------------
# 値の検査
# ---------------------------------------------------------------------------
PLAIN_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)
SEQUENCE_TYPES = (tuple, list, set, frozenset)
NUMERIC_KINDS = "biufc"
"""numpy の dtype の種類(真偽・整数・浮動小数点・複素数)。object は含めない。"""
APP_PACKAGE = "lab."


def find_violations(value: Any, path: str = "value", *, allow_tensors: bool = True) -> list[str]:
    """ZeroGPU のプロセス境界を越えられない部分を、``path`` つきで列挙する(空なら問題なし)。

    許可リスト: ``None``・真偽・整数・浮動小数点数・文字列・バイト列(厳密にその型。
    派生クラスは不可)、数値の numpy 配列とスカラー、CPU 上で勾配を持たないテンソル、
    これらを入れた tuple・list・set・frozenset・dict(キーは文字列か整数)、および ``lab.`` 以下で
    定義された dataclass(フィールドを再帰的に調べる)。

    ``allow_tensors=False`` なら、CPU 上のテンソルも不合格にする(下の ``assert_transferable``)。
    """
    return _find_violations(value, path, set(), allow_tensors)


def _find_violations(value: Any, path: str, active: set[int], allow_tensors: bool) -> list[str]:
    kind = type(value)
    if kind in PLAIN_SCALAR_TYPES:
        return []
    if isinstance(value, np.generic):
        if value.dtype.kind in NUMERIC_KINDS:
            return []
        return [f"{path}: 数値でない numpy のスカラー({value.dtype})"]
    if kind is np.ndarray:
        if value.dtype.kind in NUMERIC_KINDS:
            return []
        return [f"{path}: 数値でない numpy 配列(dtype={value.dtype})"]
    if isinstance(value, torch.Tensor):
        if not allow_tensors:
            return [f"{path}: テンソル(device={value.device})。numpy 配列か Python の値にして返す"]
        problems = []
        if kind is not torch.Tensor:
            problems.append(f"{path}: テンソルの派生クラス {kind.__qualname__}")
        if value.device.type != "cpu":
            problems.append(f"{path}: CPU 以外のテンソル(device={value.device})")
        if value.requires_grad:
            problems.append(f"{path}: 勾配を持つテンソル")
        return problems

    if id(value) in active:  # 自分自身を含む構造(pickle はできる)。無限に降りないための印
        return []
    active = active | {id(value)}
    problems: list[str] = []
    if kind in SEQUENCE_TYPES:
        for index, item in enumerate(value):
            problems += _find_violations(item, f"{path}[{index}]", active, allow_tensors)
        return problems
    if kind is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                problems.append(f"{path}: 辞書のキーが {type(key).__qualname__}(文字列か整数のみ)")
            problems += _find_violations(item, f"{path}[{key!r}]", active, allow_tensors)
        return problems
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if not kind.__module__.startswith(APP_PACKAGE):
            outside = f"{kind.__module__}.{kind.__qualname__}"
            return [f"{path}: このアプリの外で定義された dataclass {outside}"]
        for field in dataclasses.fields(value):
            problems += _find_violations(
                getattr(value, field.name), f"{path}.{field.name}", active, allow_tensors
            )
        return problems

    hint = ""
    if isinstance(value, torch.nn.Module):
        hint = "(モデルは、モジュールの読み込み時に載せたものを関数の中で参照する)"
    return [f"{path}: 許可リストにない型 {kind.__module__}.{kind.__qualname__}{hint}"]


def cross_process_boundary(value: Any) -> Any:
    """値を ZeroGPU と同じ経路(``ForkingPickler``)で、別のプロセスへ送って受け取った状態にする。

    ``SimpleQueue.put`` は ``ForkingPickler.dumps``、``get`` は ``ForkingPickler.loads`` を呼ぶ
    (``multiprocessing/queues.py``)。pickle できない値は、ここで ``PicklingError`` などになる。
    """
    payload = ForkingPickler.dumps(value)
    return ForkingPickler.loads(payload)


def assert_same_value(expected: Any, actual: Any, path: str = "value") -> None:
    """2 つの値が、型も中身も同じであること(NaN どうしは等しいとみなす)。

    境界を越えて受け取った値が、送った値と同じであることを確かめる。
    """
    assert type(expected) is type(actual), f"{path}: 型が違う {type(expected)} と {type(actual)}"
    if isinstance(expected, np.ndarray):
        assert expected.dtype == actual.dtype, f"{path}: dtype が違う"
        assert expected.shape == actual.shape, f"{path}: 形状が違う"
        equal_nan = expected.dtype.kind in "fc"
        assert np.array_equal(expected, actual, equal_nan=equal_nan), f"{path}: 値が違う"
    elif isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype, f"{path}: dtype が違う"
        assert expected.device == actual.device, f"{path}: device が違う"
        assert torch.equal(expected, actual), f"{path}: 値が違う"
    elif isinstance(expected, float):
        both_nan = math.isnan(expected) and math.isnan(actual)
        assert both_nan or expected == actual, f"{path}: {expected!r} と {actual!r}"
    elif isinstance(expected, tuple | list):
        assert len(expected) == len(actual), f"{path}: 長さが違う"
        for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
            assert_same_value(a, b, f"{path}[{index}]")
    elif isinstance(expected, dict):
        assert list(expected) == list(actual), f"{path}: キーが違う"
        for key in expected:
            assert_same_value(expected[key], actual[key], f"{path}[{key!r}]")
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            assert_same_value(
                getattr(expected, field.name), getattr(actual, field.name), f"{path}.{field.name}"
            )
    else:
        assert expected == actual, f"{path}: {expected!r} と {actual!r}"


def assert_transferable(value: Any, what: str = "value", *, allow_tensors: bool = False) -> Any:
    """``value`` が境界を越えられることを確かめ、越えて受け取った値を返す。

    1. 許可リストに合う型だけでできている(CPU 以外のテンソルがあれば不合格)。
    2. ``pickle.dumps`` できる。
    3. ZeroGPU と同じ経路で送って受け取ると、送った値と同じ値になる。

    **既定ではテンソルを、CPU 上でも不合格にする。** このアプリの ``@gpu`` の関数は、Python の値と
    numpy 配列だけをやり取りする。試験は MPS や CPU で動くので、``.cpu()`` を付け忘れたテンソルも
    CPU 上(または MPS 上)にあり、「CPU 以外のテンソルがない」だけでは見つからない。
    """
    problems = find_violations(value, what, allow_tensors=allow_tensors)
    assert not problems, "ZeroGPU のプロセス境界を越えられない値:\n  " + "\n  ".join(problems)
    pickle.dumps(value)
    received = cross_process_boundary(value)
    assert_same_value(value, received, what)
    return received


# ---------------------------------------------------------------------------
# 関数の引数の型
# ---------------------------------------------------------------------------
FORBIDDEN_ARGUMENT_TYPES: tuple[type, ...] = (
    torch.nn.Module,
    torch.Tensor,
    torch.device,
    BPEIDTokenizer,
    BPETokenizer,
    Resources,
    LoadedModel,
)


def argument_type_problems(function: Callable[..., Any], allowed: tuple[type, ...]) -> list[str]:
    """関数の引数(と戻り値)の型の注釈が、モデル・トークナイザなどを含まず、``allowed`` に収まるか。

    注釈は実行時の型として解決する(``from __future__ import annotations`` の文字列も解決する)。
    """
    hints = typing.get_type_hints(function)
    problems = []
    for name in [*inspect.signature(function).parameters, "return"]:
        if name not in hints:
            problems.append(f"{function.__name__}({name}): 型の注釈がない")
            continue
        for leaf in _leaf_types(hints[name]):
            if isinstance(leaf, type) and issubclass(leaf, FORBIDDEN_ARGUMENT_TYPES):
                problems.append(f"{function.__name__}({name}): {leaf.__qualname__} は渡さない")
            elif leaf not in allowed:
                problems.append(f"{function.__name__}({name}): 許可リストにない型 {leaf!r}")
    return problems


def _leaf_types(hint: Any) -> list[Any]:
    """``X | None``・``tuple[str, ...]``・``Iterator[...]`` などを、中の型に分解する。"""
    arguments = typing.get_args(hint)
    if not arguments:
        return [type(None) if hint is None else hint]
    leaves: list[Any] = []
    for argument in arguments:
        if argument is Ellipsis:
            continue
        leaves += _leaf_types(argument)
    return leaves


# ---------------------------------------------------------------------------
# @gpu を付けた関数の取り出し(実行時)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuFunction:
    """``@gpu`` を付けられた関数 1 つ。"""

    name: str
    module: str
    function: Callable[..., Any]
    duration: float | None


@dataclass(frozen=True)
class GpuApp:
    """``@gpu`` の付いた関数を記録しながら組み立てたアプリ。"""

    demo: gr.Blocks
    gpu_functions: dict[str, GpuFunction]

    def handler(self, name: str) -> Callable[..., Any]:
        """画面のイベントに結び付けられた関数を、名前で取り出す。"""
        return find_handler(self.demo, name)

    def handlers_reading_state(self) -> set[str]:
        """入力に ``gr.State`` を含むイベントの関数の名前。"""
        return {
            bf.name
            for bf in self.demo.fns.values()
            if bf.fn and any(isinstance(c, gr.State) for c in bf.inputs)
        }


def find_handler(demo: gr.Blocks, name: str) -> Callable[..., Any]:
    """画面のイベントに結び付けられた関数を、名前で取り出す(同名の別の関数が 2 つあればエラー)。"""
    found = [bf.fn for bf in demo.fns.values() if bf.fn and bf.name == name]
    assert len(found) >= 1, f"イベントに結び付けられた関数 {name} がない"
    assert all(f is found[0] for f in found), f"{name} という名前の別々の関数が複数ある"
    return found[0]


def feature_ui_modules() -> list[ModuleType]:
    names = sorted(path.parent.name for path in FEATURES_DIR.glob("*/ui.py"))
    return [importlib.import_module(f"lab.features.{name}.ui") for name in names]


def _recording_gpu(module_name: str, records: dict[str, GpuFunction]) -> Callable[..., Any]:
    def gpu(task: Callable[..., Any] | None = None, *, duration: float | None = None, **_: Any):
        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            assert function.__name__ not in records, f"同じ名前の @gpu 関数: {function.__name__}"
            records[function.__name__] = GpuFunction(
                function.__name__, module_name, function, duration
            )
            return function

        return decorate(task) if callable(task) else decorate

    return gpu


def build_app_with_gpu(
    resources: Resources, make_gpu: Callable[[str], Callable[..., Any]]
) -> gr.Blocks:
    """各機能の画面の ``gpu`` を、``make_gpu(モジュール名)`` の装飾子に差し替えて組み立てる。

    組み立てた後は、差し替えを元に戻す(``@gpu`` は画面を組み立てるときに適用されるため)。
    """
    modules = feature_ui_modules()
    originals = {module: module.gpu for module in modules}
    try:
        for module in modules:
            module.gpu = make_gpu(module.__name__)
        return build_app(resources, [module.PAGE for module in modules])
    finally:
        for module, original in originals.items():
            module.gpu = original


def build_app_recording_gpu(resources: Resources) -> GpuApp:
    """``@gpu`` を付けられた関数を記録しながらアプリを組み立てる。

    記録する装飾子は関数をそのまま返す(ZeroGPU の外の ``@spaces.GPU`` と同じ)。
    """
    records: dict[str, GpuFunction] = {}
    demo = build_app_with_gpu(resources, lambda module_name: _recording_gpu(module_name, records))
    return GpuApp(demo, records)


SIMULATED_CALLS: list[str] = []
"""``build_app_simulating_zero_gpu`` のアプリで、GPU の関数が呼ばれた順の名前(試験が数える)。"""


def simulated_zero_gpu(
    task: Callable[..., Any] | None = None, *, duration: float | None = None, **_: Any
) -> Callable[..., Any]:
    """ZeroGPU の受け渡しを手元で再現する ``@gpu``。実行は同じプロセスで行う。

    ZeroGPU と同じく、引数を境界を越えて渡し(``wrappers.py`` の ``arg_queue.put``)、
    戻り値(生成器なら yield した値の 1 つずつ)を境界を越えて受け取る(``res_queue.put``)。
    例外は、``gr.Error`` だけがメッセージごと届き、それ以外はクラス名だけの ``gr.Error`` になる
    (``wrappers.py`` の ``exception_result`` と ``error("ZeroGPU worker error", ...)``)。
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        name = function.__name__

        def convert(error: Exception) -> Exception:
            if isinstance(error, gr.Error):
                return cross_process_boundary(error)
            return gr.Error(f"ZeroGPU worker error: {type(error).__name__}")

        if inspect.isgeneratorfunction(function):

            @functools.wraps(function)
            def generator_wrapper(*args: Any, **kwargs: Any):
                SIMULATED_CALLS.append(name)
                args, kwargs = assert_transferable((args, kwargs), f"{name} の引数")
                try:
                    for index, item in enumerate(function(*args, **kwargs)):
                        yield assert_transferable(item, f"{name} の yield[{index}]")
                except AssertionError:  # 受け渡しの検査の失敗は、試験の失敗として見せる
                    raise
                except Exception as error:
                    raise convert(error) from None

            return generator_wrapper

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any):
            SIMULATED_CALLS.append(name)
            args, kwargs = assert_transferable((args, kwargs), f"{name} の引数")
            try:
                return assert_transferable(function(*args, **kwargs), f"{name} の戻り値")
            except AssertionError:
                raise
            except Exception as error:
                raise convert(error) from None

        return wrapper

    return decorate(task) if callable(task) else decorate


def build_app_simulating_zero_gpu(resources: Resources) -> gr.Blocks:
    """すべての ``@gpu`` の関数が、ZeroGPU と同じ受け渡しを通るアプリを組み立てる。"""
    return build_app_with_gpu(resources, lambda module_name: simulated_zero_gpu)


# ---------------------------------------------------------------------------
# ソースの検査(AST)
# ---------------------------------------------------------------------------
PLAIN_ANNOTATION_NAMES = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "None",
        "list",
        "tuple",
        "dict",
        "Tokenization",
        "ModelAnalysis",
        "DecodingSettings",
    }
)
"""``@gpu`` を付けた関数の引数に使ってよい型の注釈の名前(Python の組み込みの型と、このアプリの
データクラス)。"""
FORBIDDEN_ANNOTATION_NAMES = frozenset(
    {
        "Module",
        "GPTLanguageModel",
        "LoadedModel",
        "Resources",
        "BPEIDTokenizer",
        "BPETokenizer",
        "Tokenizer",
        "Tensor",
        "device",
        "Generator",
    }
)
MODEL_RUNNER_NAMES = frozenset(
    {
        "analyze",
        "generate_events",
        "respond",
        "stream_generation",
        "run_beam_search",
        "greedy_generate_until_stop",
    }
)
MODEL_ATTRIBUTES = frozenset({"model"})
"""モデルの本体(``LoadedModel.model``)を直接使うことを示す属性。``resources.models`` は、
設定の読み取り(``max_sequence_length`` など)にも使うので、対象にしない
(モデルを動かす関数の呼び出しは、名前で見る)。"""


def is_gpu_decorator(node: ast.expr) -> bool:
    """``@gpu``・``@gpu(...)``・``@spaces.GPU``・``@spaces.GPU(...)`` のいずれか。"""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "gpu"
    if isinstance(target, ast.Attribute):
        return target.attr == "GPU" and isinstance(target.value, ast.Name)
    return False


def gpu_functions_in(tree: ast.AST) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(is_gpu_decorator(d) for d in node.decorator_list)
    ]


def gpu_function_names_in_source() -> set[str]:
    """``lab/features/*/ui.py`` のソースで ``@gpu`` を付けられている関数の名前。"""
    names: set[str] = set()
    for path in FEATURES_DIR.glob("*/ui.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names |= {node.name for node in gpu_functions_in(tree)}
    return names


def _annotation_names(annotation: ast.AST) -> list[str]:
    """型の注釈に現れる名前(``torch.nn.Module`` のような属性は、点でつないだ 1 つの名前)。"""
    if isinstance(annotation, ast.Name):
        return [annotation.id]
    if isinstance(annotation, ast.Attribute):
        return [ast.unparse(annotation)]
    if isinstance(annotation, ast.Constant):
        if isinstance(annotation.value, str):  # 引用符で囲んだ注釈
            return _annotation_names(ast.parse(annotation.value, mode="eval").body)
        if annotation.value is Ellipsis:  # tuple[str, ...] の省略記号
            return []
        return ["None"] if annotation.value is None else [repr(annotation.value)]
    names: list[str] = []
    for child in ast.iter_child_nodes(annotation):
        names += _annotation_names(child)
    return names


def unsafe_gpu_parameters(tree: ast.AST) -> list[str]:
    """``@gpu`` を付けた関数の引数のうち、注釈がない・モデルやトークナイザなどを渡す・
    許可リスト(``PLAIN_ANNOTATION_NAMES``)にない型のものを列挙する。
    """
    problems = []
    for function in gpu_functions_in(tree):
        arguments = function.args
        parameters = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *[a for a in (arguments.vararg, arguments.kwarg) if a is not None],
        ]
        for parameter in parameters:
            where = f"{function.name}({parameter.arg})"
            if parameter.annotation is None:
                problems.append(f"{where}: 型の注釈がない")
                continue
            for name in _annotation_names(parameter.annotation):
                if name.split(".")[-1] in FORBIDDEN_ANNOTATION_NAMES:
                    problems.append(f"{where}: {name} は引数で渡さない")
                elif name not in PLAIN_ANNOTATION_NAMES:
                    problems.append(f"{where}: 許可リストにない型 {name}")
    return problems


def model_users_outside_gpu(
    tree: ast.AST, ignore: frozenset[str] = frozenset({"build_page"})
) -> list[str]:
    """モデルを動かす(または ``resources.models`` を引く)関数のうち、``@gpu`` を付けておらず、
    ``@gpu`` を付けた関数の中からしか呼ばれていないとも言えないものを列挙する。

    イベントに結び付けられた(関数として値で渡されている)関数は、Gradio から直接呼ばれるので、
    自分に ``@gpu`` が要る。補助の関数は、呼び出し元がすべて ``@gpu`` の関数(の補助)であればよい。
    ``ignore``(画面を組み立てる関数。モデルの設定を読むが、イベントでは動かない)は対象外。
    """
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name not in ignore
    ]
    names = {f.name for f in functions}
    is_gpu = {f.name: any(is_gpu_decorator(d) for d in f.decorator_list) for f in functions}

    def runs_model_directly(function: ast.FunctionDef) -> bool:
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                called = node.func
                if isinstance(called, ast.Name) and called.id in MODEL_RUNNER_NAMES:
                    return True
            if isinstance(node, ast.Attribute) and node.attr in MODEL_ATTRIBUTES:
                return True
        return False

    call_func_ids = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    passed_as_value = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in call_func_ids
    }
    callers: dict[str, set[str]] = {name: set() for name in names}
    for function in functions:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in names
                and node.func.id != function.name
            ):
                callers[node.func.id].add(function.name)

    touches = {f.name for f in functions if runs_model_directly(f)}
    changed = True
    while changed:  # 「@gpu でない、モデルを動かす関数」を呼ぶ関数も、モデルを動かす
        changed = False  # (@gpu の関数を呼ぶのは構わない。それが正しい使い方)
        for name in names - touches:
            if any(name in callers[callee] for callee in touches if not is_gpu[callee]):
                touches.add(name)
                changed = True

    def covered(name: str, path: tuple[str, ...] = ()) -> bool:
        if is_gpu[name]:
            return True
        if name in passed_as_value or name in path or not callers[name]:
            return False
        return all(covered(caller, (*path, name)) for caller in callers[name])

    return sorted(name for name in touches if not covered(name))


CHECK_CALL_PREFIXES = ("check_", "parse_", "validate")
"""入力の検査の関数の名前の始まり(``check_generation_request``・``parse_words``・
``settings.validate`` など)。"""


def checks_inside_gpu_functions(tree: ast.AST) -> list[str]:
    """``@gpu`` を付けた関数の本体にある、入力の検査らしい文を列挙する。

    入力の検査は、``@gpu`` の関数に入る前に、画面のイベントの関数で行う(誤った入力で ZeroGPU の
    利用枠を使わないため)。``@gpu`` の関数は計算だけを行うので、本体に ``raise``・``try``・
    検査の関数の呼び出しを書かない。見つけた文は ``関数名:行番号: 内容`` の形で返す。
    """
    found = []
    for function in gpu_functions_in(tree):
        for node in ast.walk(function):
            if node is function:
                continue
            where = f"{function.name}:{getattr(node, 'lineno', function.lineno)}"
            if isinstance(node, ast.Raise):
                found.append(f"{where}: raise")
            elif isinstance(node, ast.Try):
                found.append(f"{where}: try")
            elif isinstance(node, ast.Call):
                called = node.func
                name = called.id if isinstance(called, ast.Name) else getattr(called, "attr", "")
                if name.startswith(CHECK_CALL_PREFIXES):
                    found.append(f"{where}: {name}()")
    return found
