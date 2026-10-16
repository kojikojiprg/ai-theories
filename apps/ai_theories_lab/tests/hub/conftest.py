"""本物の Hugging Face Hub のアーティファクトを使う試験の fixture。

ネットワークと、本物の重み(約 59 MiB)の取得が要る。``tests/hub/`` の試験は ``hub`` の印が
自動で付き、既定では走らない。走らせるときは、アプリのディレクトリで
``uv run pytest tests/hub -m hub``。
"""

from __future__ import annotations

import pytest

from lab.core.artifacts import Resources, load_resources


@pytest.fixture(scope="session")
def hub_resources() -> Resources:
    """Hub の本物のトークナイザと 3 ブランチのモデル(アプリと同じ経路)。"""
    return load_resources()
