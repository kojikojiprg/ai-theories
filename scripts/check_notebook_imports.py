"""ノートブックのコードセルが ``src/`` から import する名前を、すべて import できるか確かめる。

``src/`` の整理(関数の移動・統合・削除)の後に、実行済みのノートブックのコードセルを
変えずに済むことを確かめる。コードセルの実行はせず、``from src... import ...`` と
``import src...`` の文だけを取り出して実行する。

使い方(リポジトリのルートで)::

    uv run python scripts/check_notebook_imports.py
    uv run python scripts/check_notebook_imports.py theories/07_retrieval/*.ipynb
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from pathlib import Path


def extract_src_imports(notebook_path: Path) -> list[str]:
    """ノートブックのコードセルから、``src`` を import する文のソースを取り出す。

    IPython の記法(``%cd``・``!pip`` など)の行は、構文解析の前に ``pass`` に置き換える。
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    statements = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in "".join(cell["source"]).split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                line = " " * (len(line) - len(stripped)) + "pass"
            lines.append(line)
        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_names = [node.module or ""]
            elif isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            else:
                continue
            if any(name.split(".")[0] == "src" for name in module_names):
                statements.append(ast.unparse(node))
    return statements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "notebooks", nargs="*", help="対象のノートブック(省略時は theories/*/*.ipynb)"
    )
    arguments = parser.parse_args()
    paths = [Path(p) for p in arguments.notebooks] or [
        Path(p) for p in sorted(glob.glob("theories/*/*.ipynb"))
    ]
    sys.path.insert(0, str(Path.cwd()))
    total, failed = 0, 0
    for path in paths:
        for statement in sorted(set(extract_src_imports(path))):
            total += 1
            try:
                exec(statement, {})
            except Exception as error:  # noqa: BLE001 (失敗した import を全件報告するため)
                failed += 1
                print(f"失敗 {path.name}: {statement.splitlines()[0]} ... {error!r}")
    print(f"import 文 {total} 件(ノートブック {len(paths)} 本、重複を除く)のうち、失敗 {failed} 件")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
