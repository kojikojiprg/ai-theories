"""複数の Colab セッションに分けて実行するノートブックの、パートごとの出力を検査する(読み取り専用)。

016 のように、ノートブックを 2 つのセッション(`RUN_PART = "AB"` / `"C"`)に分けて実行し、
各セッションの出力を 1 つのノートブックに残す運用では、別のセッションで誤ってセルを再実行すると、
先のセッションの出力が「スキップした」出力に置き換わってしまう。本スクリプトは、各パートのコードセルについて、

- そのセルが属するパート(セルのメタデータ ``run_part``、なければ直前の見出しから判定)と、
- 出力の先頭に印字された実行パートの識別子(``[RUN_PART=AB]`` など)

が一致することを検査し、不一致・識別子の欠落・未実行のセルを一覧で報告する。共通のセル(どのパートにも
属さないセル)は検査しない。識別子 ``ALL`` はスモークテスト専用で、どのパートとも一致するとみなす。

ノートブックは読むだけで、出力を含めて一切変更しない。

使い方 / Usage::

    uv run python scripts/check_notebook_run_parts.py <notebook>              # 両パート(既定)
    uv run python scripts/check_notebook_run_parts.py <notebook> --parts AB   # セッション 1 の後

問題が 1 件でもあれば終了コード 1、なければ 0 を返す。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IDENTIFIER_PATTERN = re.compile(r"^\[RUN_PART=([A-Z]+)\]")

# 見出しからパートを判定する規則(セルのメタデータ run_part がない場合のみ使う)。
# 見出し(# で始まる行)に次の文字列を含むと、次の見出しまでのコードセルがそのパートに属する。
HEADING_PART_MARKERS = (
    ("【セッション 1】", "AB"),
    ("【セッション 2】", "C"),
    ("A.1 セッション 1", "AB"),
    ("A.2 セッション 2", "C"),
)


def heading_part(markdown: str) -> str | None | bool:
    """Markdown セルの見出しから、以降のセルのパートを返す。

    見出しを含まないセルは ``False``(直前のパートを引き継ぐ)、パートの目印のない見出しは ``None``
    (共通のセル)、目印のある見出しはそのパートを返す。
    """
    headings = [line for line in markdown.splitlines() if line.startswith("#")]
    if not headings:
        return False
    for marker, part in HEADING_PART_MARKERS:
        if marker in headings[-1]:
            return part
    return None


def first_output_text(cell: dict) -> str:
    """セルの出力のうち、最初のテキスト(stream または text/plain)を返す。"""
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            return "".join(output.get("text", ""))
        if output.get("output_type") == "error":
            return ""
        data = output.get("data", {})
        if "text/plain" in data:
            return "".join(data["text/plain"])
    return ""


def check_notebook(path: Path, parts: set[str]) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    current_part: str | None = None
    checked = 0
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", ""))
        if cell["cell_type"] == "markdown":
            part = heading_part(source)
            if part is not False:
                current_part = part
            continue
        if cell["cell_type"] != "code":
            continue
        part = cell.get("metadata", {}).get("run_part", current_part)
        if part is None or part not in parts:
            continue
        checked += 1
        first_line = source.strip().splitlines()[0] if source.strip() else ""
        location = {"cell": index, "part": part, "source": first_line[:60]}
        if cell.get("execution_count") is None or not cell.get("outputs"):
            problems.append({**location, "problem": "未実行(出力がない)"})
            continue
        if any(o.get("output_type") == "error" for o in cell["outputs"]):
            problems.append({**location, "problem": "出力に例外がある"})
        match = IDENTIFIER_PATTERN.match(first_output_text(cell))
        if match is None:
            problems.append({**location, "problem": "出力の先頭に実行パートの識別子がない"})
            continue
        found = match.group(1)
        if found not in (part, "ALL"):
            problems.append({**location, "problem": f"識別子の不一致(出力は [RUN_PART={found}])"})
    print(f"検査したノートブック: {path}")
    print(f"検査したパート: {sorted(parts)}、検査したコードセル: {checked} 個")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--parts", default="AB,C", help="検査するパート(カンマ区切り、既定 AB,C)")
    args = parser.parse_args()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    problems = check_notebook(args.notebook, parts)
    if not problems:
        print("問題なし: 全てのパートのセルで、識別子がセルの属するパートと一致した。")
        return 0
    print(f"問題 {len(problems)} 件:")
    for p in problems:
        print(f"  セル {p['cell']}(パート {p['part']}、{p['source']!r}): {p['problem']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
