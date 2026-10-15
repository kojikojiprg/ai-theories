"""試験全体の fixture。

``stand_in_hub`` は、乱数の重みの代替アーティファクトを一時ディレクトリに作り、
``huggingface_hub`` のキャッシュとオフラインの設定をそこへ向ける。``unit/`` の試験は、
これを経由する fixture(``resources``・``lab_app``)だけを使う。``hub/`` の試験は使わない
(本物の Hub を使う)。
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import huggingface_hub.constants as hf_constants
import pytest

from lab.core.artifacts import Resources, load_resources
from tests.stand_in import build_stand_in_artifacts


@pytest.fixture(scope="session")
def stand_in_hub(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """代替のアーティファクトを作り、Hub のキャッシュをそこへ向ける(オフライン)。"""
    directory = tmp_path_factory.mktemp("stand_in")
    cache_dir = build_stand_in_artifacts(directory)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        yield cache_dir
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="session")
def resources(stand_in_hub: Path) -> Resources:
    """代替のアーティファクトから読み込んだトークナイザと 3 ブランチのモデル(アプリと同じ経路)。"""
    return load_resources()


@pytest.fixture(scope="session")
def lab_app(stand_in_hub: Path) -> ModuleType:
    """``app.py``(モジュールの読み込み時にモデルを読み込む)。"""
    return importlib.import_module("app")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """``tests/hub/`` の試験すべてに ``hub`` の印を付ける(付け忘れて既定で走るのを防ぐ)。"""
    for item in items:
        if "/tests/hub/" in Path(str(item.fspath)).as_posix():
            item.add_marker(pytest.mark.hub)
