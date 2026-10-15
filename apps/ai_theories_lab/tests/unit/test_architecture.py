"""構成の規約: 処理(``service.py``)と画面(``ui.py``)の分離、``spaces`` の import の順序。"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

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
