"""構成の規約: 処理(``service.py``)と画面(``ui.py``)の分離、``spaces`` の import の順序、
ZeroGPU の関数の引数・モデルの読み込みの場所・GPU を使う関数の範囲・入力の検査の位置
(誤った入力では ``@spaces.GPU`` の関数が呼ばれない)。"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from lab.core.artifacts import Resources
from tests.zero_gpu import (
    SIMULATED_CALLS,
    checks_inside_gpu_functions,
    find_handler,
    gpu_function_names_in_source,
    model_users_outside_gpu,
    simulated_zero_gpu,
    unsafe_gpu_parameters,
)
from tests.zero_gpu_cases import GPU_FUNCTION_OF, INVALID_CASES, VALID_CASES

APP_DIR = Path(__file__).resolve().parents[2]
SERVICE_FILES = sorted((APP_DIR / "lab" / "features").glob("*/service.py"))
CORE_FILES = sorted((APP_DIR / "lab" / "core").glob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_the_services_exist() -> None:
    assert {p.parent.name for p in SERVICE_FILES} == {"generation", "inspection", "instruction"}


@pytest.mark.parametrize(
    "path", SERVICE_FILES + CORE_FILES, ids=lambda p: str(p.relative_to(APP_DIR))
)
def test_services_and_core_do_not_import_gradio_or_ui(path: Path) -> None:
    for module in _imported_modules(path):
        assert module.split(".")[0] != "gradio", f"{path.name} が {module} を import している"
        assert not module.startswith("lab.ui"), f"{path.name} が {module} を import している"
        assert not module.endswith(".ui"), f"{path.name} が {module} を import している"


@pytest.mark.parametrize("path", SERVICE_FILES, ids=lambda p: p.parent.name)
def test_services_do_not_import_spaces(path: Path) -> None:
    """GPU の割り当ては画面の側の関数で行う。処理は ZeroGPU を知らない。"""
    assert "spaces" not in {m.split(".")[0] for m in _imported_modules(path)}


def test_app_py_imports_spaces_before_anything_else() -> None:
    tree = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"))
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    first = imports[0]
    assert isinstance(first, ast.Import)
    assert first.names[0].name == "spaces"


def test_importing_lab_loads_spaces_before_torch() -> None:
    """``lab`` のどのモジュールから import しても、``spaces`` が torch より先に読み込まれる。"""
    script = (
        "import sys\n"
        "import lab.core.artifacts\n"
        "order = list(sys.modules)\n"
        "assert order.index('spaces') < order.index('torch'), 'spaces が torch より後'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=APP_DIR, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# ZeroGPU: GPU を使う関数の引数、モデルの読み込みの場所、GPU を使う関数の範囲
#
# 手元(MPS・CPU)では @spaces.GPU が何もしないので、ZeroGPU のプロセス境界に関する誤りは、実行しても
# 現れない。そこで、ソースの構造から確かめる。値そのものの検査は test_zero_gpu_values.py にある。
# ---------------------------------------------------------------------------
UI_FILES = sorted((APP_DIR / "lab" / "features").glob("*/ui.py"))
FEATURE_FILES = sorted((APP_DIR / "lab" / "features").glob("*/*.py"))
MODEL_LOADING_CALLS = {"load_resources", "load_model", "build_model", "hf_hub_download", "load"}
DEVICE_MOVES = {"to", "cuda"}  # .cpu() は、結果を CPU の値にするために使う


def _ui_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_the_ui_files_exist_and_use_the_gpu_decorator() -> None:
    assert {p.parent.name for p in UI_FILES} == {"generation", "inspection", "instruction"}
    assert gpu_function_names_in_source() == set(GPU_FUNCTION_OF.values())


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.parent.name)
def test_the_ui_files_get_the_decorator_from_core_not_from_spaces(path: Path) -> None:
    """``@gpu`` は ``lab.core.device`` のもの。``spaces`` を直接使うと、検査を素通りする。"""
    assert "spaces" not in {m.split(".")[0] for m in _imported_modules(path)}
    uses = [
        node
        for node in ast.walk(_ui_tree(path))
        if isinstance(node, ast.Attribute) and node.attr == "GPU"
    ]
    assert uses == []


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.parent.name)
def test_gpu_function_arguments_are_plain_values(path: Path) -> None:
    """引数にモデル・トークナイザ・テンソルなどを渡さない(境界は pickle で越える。モデルは起動時に
    読み込んだものを、関数の中から参照する)。注釈のない引数も不合格(検査できないため)。
    """
    assert unsafe_gpu_parameters(_ui_tree(path)) == []


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.parent.name)
def test_functions_that_run_the_model_are_gpu_functions(path: Path) -> None:
    """モデルを動かす関数は、自分が ``@gpu`` を付けているか、``@gpu`` の関数からだけ呼ばれる。

    手元では、``@gpu`` のない関数がモデルを動かしても動いてしまう(ZeroGPU では GPU のない
    プロセスで動き、失敗する)。選び直しなどの、``gr.State`` だけで済む処理は対象外。
    """
    assert model_users_outside_gpu(_ui_tree(path)) == []


def test_the_check_for_gpu_function_arguments_finds_the_mistakes() -> None:
    source = """
@gpu(duration=10)
def good(text: str, state: Tokenization | None, seed: float | None) -> ModelAnalysis:
    ...

@spaces.GPU
def bad_model(model: GPTLanguageModel, tokenizer: BPEIDTokenizer, module: torch.nn.Module):
    ...

@gpu
def bad_others(values: Tensor, resources: Resources, loaded: "LoadedModel", extra, *rest: int):
    ...

def not_gpu(model: GPTLanguageModel):
    ...
"""
    problems = unsafe_gpu_parameters(ast.parse(source))
    text = "\n".join(problems)
    assert "good" not in text
    assert "not_gpu" not in text
    for expected in (
        "bad_model(model): GPTLanguageModel は引数で渡さない",
        "bad_model(tokenizer): BPEIDTokenizer は引数で渡さない",
        "bad_model(module): torch.nn.Module は引数で渡さない",
        "bad_others(values): Tensor は引数で渡さない",
        "bad_others(resources): Resources は引数で渡さない",
        "bad_others(loaded): LoadedModel は引数で渡さない",
        "bad_others(extra): 型の注釈がない",
    ):
        assert expected in problems, problems
    assert len(problems) == 7  # 余計な指摘(torch.nn や torch の名前)を出さない
    assert not any(p.startswith("bad_others(rest)") for p in problems)  # int は許可リストにある


def test_the_check_for_functions_that_run_the_model_finds_the_mistakes() -> None:
    wired_without_gpu = """
def build_page(resources):
    models = resources.models  # 画面の組み立て(イベントでは動かない)

    def run_model(tokenization):
        return analyze(models["main"], resources.device, tokenization.token_ids)

    button.click(run_model, inputs=[state], outputs=[state])
"""
    assert model_users_outside_gpu(ast.parse(wired_without_gpu)) == ["run_model"]

    decorated = wired_without_gpu.replace(
        "    def run_model", "    @gpu(duration=10)\n    def run_model"
    )
    assert model_users_outside_gpu(ast.parse(decorated)) == []

    helper = """
def build_page(resources):
    def run(prompt):
        return generate_events(resources.models["main"], prompt)

    @gpu
    def generate_text(prompt):
        yield from run(prompt)

    def preview(prompt):  # GPU のない関数が、同じ補助の関数を呼ぶ
        return run(prompt)

    button.click(generate_text)
    button.click(preview)
"""
    assert model_users_outside_gpu(ast.parse(helper)) == ["preview", "run"]

    only_from_gpu = """
def build_page(resources):
    def run(prompt):
        return generate_events(resources.models["main"], prompt)

    @gpu
    def generate_text(prompt):
        yield from run(prompt)

    button.click(generate_text)
"""
    assert model_users_outside_gpu(ast.parse(only_from_gpu)) == []

    state_only = """
def build_page(resources):
    def render_attention(analysis, tokenization, layer, head):
        return select_attention(analysis, layer, head)

    button.change(render_attention)
"""
    assert model_users_outside_gpu(ast.parse(state_only)) == []


def test_app_py_loads_the_models_at_module_level_and_passes_them_to_the_pages() -> None:
    """モデルの読み込みは、モジュールの読み込み時(ZeroGPU の推奨)。関数の中では読み込まない。"""
    tree = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)]
    ]
    by_name = {node.targets[0].id: node.value for node in assignments}
    resources = by_name["RESOURCES"]
    assert isinstance(resources, ast.Call)
    assert ast.unparse(resources.func) == "load_resources"
    build = by_name["demo"]
    assert isinstance(build, ast.Call)
    assert ast.unparse(build.func) == "build_app"
    assert ast.unparse(build.args[0]) == "RESOURCES"
    order = [node.targets[0].id for node in assignments]
    assert order.index("RESOURCES") < order.index("demo")


@pytest.mark.parametrize("path", FEATURE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_feature_code_loads_a_model_or_moves_one_to_a_device(path: Path) -> None:
    """機能のコード(画面・処理)は、モデルを読み込まず、デバイスへ移さない。

    ZeroGPU は、起動時にデバイスへ載せたモデルを、GPU の割り当ての時に移す。関数の中で読み込む・
    移すと、割り当てのたびにその時間がかかり、利用枠を使う(公式ドキュメントも非推奨としている)。
    読み込みと移動は ``lab/core/artifacts.py`` だけが行う。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if isinstance(called, ast.Name):
            assert called.id not in MODEL_LOADING_CALLS, f"{path.name}: {called.id}() を呼んでいる"
        elif isinstance(called, ast.Attribute):
            assert called.attr not in DEVICE_MOVES, f"{path.name}: .{called.attr}() を呼んでいる"
            assert called.attr not in MODEL_LOADING_CALLS, (
                f"{path.name}: .{called.attr}() を呼んでいる"
            )


# ---------------------------------------------------------------------------
# ZeroGPU: 入力の検査は、@gpu の関数に入る前に行う(誤った入力で利用枠を使わない)
#
# 画面のイベントの関数(generate_text・compare_methods・answer_instruction・run_model)が、
# GPU を使わずに判定できる入力の検査を行い、誤りなら gr.Error にする。検査を通った入力だけを、
# @gpu を付けた関数(*_on_gpu。計算だけを行う)に渡す。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.parent.name)
def test_gpu_functions_only_compute_and_do_not_check_inputs(path: Path) -> None:
    """``@gpu`` の関数の本体に、``raise``・``try``・検査の関数の呼び出しを書かない。"""
    assert checks_inside_gpu_functions(_ui_tree(path)) == []


def test_the_check_for_checks_inside_gpu_functions_finds_the_mistakes() -> None:
    source = """
@gpu(duration=10)
def computes_only(tokenization):
    return analyze(model, device, tokenization.token_ids)

@gpu
def raises(text):
    if not text:
        raise gr.Error("空です")
    return text

@spaces.GPU
def catches(text):
    try:
        return respond(text)
    except ValueError as error:
        return str(error)

@gpu
def validates(settings, loaded, tokenizer, prompt):
    settings.validate(3, 256)
    check_generation_request(loaded, tokenizer, prompt, settings)
    return parse_words(prompt)

def not_a_gpu_function(text):
    if not text:
        raise gr.Error("空です")
"""
    found = checks_inside_gpu_functions(ast.parse(source))
    assert [item.split(":")[0] for item in found].count("computes_only") == 0
    assert [item.split(":")[0] for item in found].count("not_a_gpu_function") == 0
    kinds = {(item.split(":")[0], item.split(": ")[1]) for item in found}
    assert kinds == {
        ("raises", "raise"),
        ("catches", "try"),
        ("validates", "validate()"),
        ("validates", "check_generation_request()"),
        ("validates", "parse_words()"),
    }


def test_the_gpu_call_counter_counts_a_gpu_function_that_fails_at_once() -> None:
    """検査が ``@gpu`` の関数の中に残っていると、誤った入力でも関数が呼ばれた記録が残る。"""

    @simulated_zero_gpu
    def checks_inside(text: str) -> str:
        if not text:
            raise gr.Error("空です")
        return text

    SIMULATED_CALLS.clear()
    with pytest.raises(gr.Error, match="空です"):
        checks_inside("")
    assert SIMULATED_CALLS == ["checks_inside"]
    SIMULATED_CALLS.clear()


def test_every_event_function_has_a_gpu_function_and_both_exist(simulated_app: gr.Blocks) -> None:
    for handler_name in GPU_FUNCTION_OF:
        assert callable(find_handler(simulated_app, handler_name))


@pytest.mark.parametrize(("name", "make_arguments", "message"), INVALID_CASES)
def test_invalid_input_never_reaches_the_gpu_function(
    resources: Resources,
    simulated_app: gr.Blocks,
    name: str,
    make_arguments: Callable[..., Any],
    message: str,
) -> None:
    """誤った入力では、``gr.Error`` になるだけで、``@gpu`` の関数は呼ばれない(利用枠を使わない)。

    ZeroGPU の外では ``@spaces.GPU`` が何もしないので、ZeroGPU と同じ受け渡しを通す
    ``@gpu``(呼ばれた順を記録する)に差し替えたアプリで確かめる。誤りの表示(メッセージ)は、
    検査を外に出す前と同じ。
    """
    handler = find_handler(simulated_app, name)
    SIMULATED_CALLS.clear()
    with pytest.raises(gr.Error) as caught:
        for _ in _iterate(handler(**make_arguments(resources))):
            pass
    assert message in caught.value.message
    assert SIMULATED_CALLS == [], f"{name}: 誤った入力で {SIMULATED_CALLS} が呼ばれた"


@pytest.mark.parametrize(("name", "make_arguments"), VALID_CASES)
def test_valid_input_reaches_the_gpu_function_exactly_once(
    resources: Resources, simulated_app: gr.Blocks, name: str, make_arguments: Callable[..., Any]
) -> None:
    """上の試験の対照: 正しい入力では、対応する ``@gpu`` の関数が 1 回だけ呼ばれる。"""
    handler = find_handler(simulated_app, name)
    SIMULATED_CALLS.clear()
    for _ in _iterate(handler(**make_arguments(resources))):
        pass
    assert [GPU_FUNCTION_OF[name]] == SIMULATED_CALLS
    SIMULATED_CALLS.clear()


def _iterate(result: Any) -> Any:
    """生成器なら yield のすべて、そうでなければ戻り値 1 つ。"""
    return result if inspect.isgenerator(result) else [result]
