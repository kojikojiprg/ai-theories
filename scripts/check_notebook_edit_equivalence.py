"""実行済みのノートブックの編集が、コードセルの実行結果を変えていないことを確かめる(読み取り専用)。

コードセルの出力(``outputs``)は、そのコードを実行した結果でなければならない。コードセルを変えると
再実行が必要になるが、コメント・空行・書式だけの変更は実行の結果を変えない。本スクリプトは、
作業ツリーのノートブックを git の基準(既定は ``HEAD``)のノートブックと比べ、編集が
**この種類の変更にとどまる** ことを機械的に確かめる。確かめるのは次の 5 つで、
1 つでも成り立たなければ理由を印字して終了コード 1 を返す。

1. コードセルの数と順序が同じ。セルの識別子(``id``)が両方のノートブックのすべてのコードセルにあり、
   それぞれの中で重複しなければ識別子で対応させ、そうでなければ位置で対応させる。
   Markdown セルの数・内容・位置は問わない(自由に書き換え・移動してよい)。
2. 対応するコードセルの構文木(abstract syntax tree)が完全に一致する。比べる前に、IPython の記法
   (``%cd``、``!pip`` など)を IPython の変換で Python のコードに直す。docstring と文字列は
   構文木に含まれるので、変更は検出される。構文木を作れないセルは、ソースの文字列そのものが
   一致することを求める。
3. 対応するコードセルの ``outputs`` と ``execution_count`` が完全に一致する。
4. 対応するコードセルのメタデータの違いは、``tags`` への Zenn のタグの追加・削除だけ。
   Zenn のタグは ``zenn-hide``・``zenn-hide-input``・``zenn-images-only``・``zenn-details``・
   ``zenn-appendix`` の 5 つで、それぞれ付けてよいセルの種類が決まっている(``zenn-hide-input`` と
   ``zenn-images-only`` はコードセルだけ、``zenn-appendix`` は Markdown セルだけ)。
   接頭辞 ``zenn-`` の未知のタグ、種類の合わないタグ、基準にない未知のタグは認めない。
   ``zenn-appendix`` は 1 つまでで、付録の見出し(``#`` で始まる行)の Markdown セルに付ける。
5. 上記以外のセルの種類(``markdown``・``code``・``raw`` 以外)がない。

報告するのは、コードセルの対応づけの方法と数、ソースが変わったコードセルの内訳(コメントが
変わったもの、空行・書式だけが変わったもの)、変わった Markdown セルの数、**付録を除いた
Markdown の文字数(本文の分量。付録は ``zenn-appendix`` のセルから末尾まで)**、Zenn のタグを
付けたセルの数(タグごと)である。文字数は判定に使わない。判定には影響しない **注意** も印字する。
特に、次の場合である。

- 構文木が同じでもソース上の行の位置が動いたセルの出力に、行番号を含みうるもの(例外、stderr の
  警告など)がある。出力が再実行の結果と一致しなくなりうる。
- 画像の出力がないコードセルに ``zenn-images-only`` が付いている。Zenn ではセルごと除かれる。
- セルの番号への参照(「セル 15〜34」など)を含むセルがあり、かつコードセルの位置が基準から動いた。
  参照が指すセルが変わっていないかを確かめる。

ノートブックは読むだけで、一切変更しない。``scripts/`` の他のスクリプトと同じく、ローカルで実行し、
結果は標準出力に残す。

使い方 / Usage::

    uv run python scripts/check_notebook_edit_equivalence.py <notebook> [<notebook> ...]
    uv run python scripts/check_notebook_edit_equivalence.py <notebook> --base HEAD~3

終了コード: 0 = 通った、1 = 編集が上記の条件を満たさない、2 = 実行できなかった(git の失敗、
ノートブックを読めない、など)。

試験は ``uv run python scripts/test_check_notebook_edit_equivalence.py`` で行う。
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

try:
    from IPython.core.inputtransformer2 import TransformerManager
except ModuleNotFoundError:  # pragma: no cover - 実行環境の案内のみ
    sys.exit(
        "IPython が見つからない。`uv run python scripts/...` で(開発用の依存を含めて)実行すること。"
    )

CELL_TYPES = ("code", "markdown", "raw")

# Zenn での表示を制御するタグと、付けてよいセルの種類。
ZENN_TAG_PREFIX = "zenn-"
ZENN_TAG_TARGETS: dict[str, tuple[str, ...]] = {
    "zenn-hide": ("code", "markdown", "raw"),
    "zenn-hide-input": ("code",),
    "zenn-images-only": ("code",),
    "zenn-details": ("code", "markdown", "raw"),
    "zenn-appendix": ("markdown",),
}

# セルの番号への参照(「セル 15〜34」など)。コードの文字列や説明が、セルの位置に依存する。
CELL_REFERENCE_PATTERN = re.compile(r"セル\s*\d+")

# 出力に、実行したコードの行番号を含みうることを示す文字列(例外の追跡、stderr の警告など)。
LINE_NUMBER_PATTERN = re.compile(r"\.py:\d+|\bline \d+|ipython-input|Cell In")


class ScriptError(Exception):
    """実行できなかった場合(終了コード 2)。"""


class SyntaxForm(NamedTuple):
    """コードセルのソースを比べるための形。

    ``kind`` が ``"syntax_tree"`` なら ``structure`` は属性(行番号など)を除いた構文木の文字列表現、
    ``line_numbers`` は各ノードの行番号の列。``"source"`` なら構文木を作れなかったので、
    ``structure`` はソースの文字列そのもの、``line_numbers`` は ``None``。
    """

    kind: str
    structure: str
    line_numbers: tuple[int | None, ...] | None


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pairing_method: str = ""
    base_code_count: int = 0
    current_code_count: int = 0
    source_unchanged: int = 0
    tree_same_changed: int = 0
    comment_changed: int = 0
    format_only_changed: int = 0
    position_shifted: int = 0
    markdown_base: int = 0
    markdown_current: int = 0
    markdown_added_or_rewritten: int = 0
    markdown_removed_or_rewritten: int = 0
    # Markdown の文字数。(本文 = 付録を除く, 付録)
    base_characters: tuple[int, int] = (0, 0)
    current_characters: tuple[int, int] = (0, 0)
    # タグごとの個数。(基準のコードセル, 現在のコードセル, 基準の Markdown, 現在の Markdown)
    tag_counts: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.problems


_transformer = TransformerManager()


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def describe_cell(index: int, cell: dict) -> str:
    return f"セル {index}({cell.get('cell_type')}、id={cell.get('id', 'なし')})"


def tags_of(cell: dict) -> list[str]:
    tags = cell.get("metadata", {}).get("tags", [])
    return [tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else []


def syntax_form(source: str) -> SyntaxForm:
    """IPython の記法を Python のコードに直し、構文木を作って比べる形にする。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # 不正なエスケープなどの SyntaxWarning は比較に関係しない
            tree = ast.parse(_transformer.transform_cell(source))
    except (SyntaxError, ValueError):
        return SyntaxForm("source", source, None)
    line_numbers = tuple(getattr(node, "lineno", None) for node in ast.walk(tree))
    return SyntaxForm("syntax_tree", ast.dump(tree, include_attributes=False), line_numbers)


def comment_tokens(source: str) -> tuple[str, ...] | None:
    """ソースのコメントを順に返す。字句に分けられなければ ``None``。"""
    try:
        return tuple(
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        )
    except (tokenize.TokenError, SyntaxError):
        return None


def outputs_may_contain_line_numbers(cell: dict) -> bool:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return True
        is_stderr = output.get("output_type") == "stream" and output.get("name") == "stderr"
        if is_stderr and LINE_NUMBER_PATTERN.search("".join(output.get("text", ""))):
            return True
    return False


def markdown_characters(notebook: dict) -> tuple[int, int]:
    """Markdown の文字数を (本文, 付録) で返す。

    付録は ``zenn-appendix`` のタグを付けたセルから末尾まで(なければ 0)。本文は、それより前の
    すべての Markdown セル(Zenn から除くセルを含む)。
    """
    body = appendix = 0
    in_appendix = False
    for cell in notebook["cells"]:
        if "zenn-appendix" in tags_of(cell):
            in_appendix = True
        if cell["cell_type"] == "markdown":
            if in_appendix:
                appendix += len(source_text(cell))
            else:
                body += len(source_text(cell))
    return body, appendix


def has_image_output(cell: dict) -> bool:
    return any(
        key.startswith("image/")
        for output in cell.get("outputs", [])
        for key in output.get("data", {})
    )


def code_cells(notebook: dict) -> list[tuple[int, dict]]:
    return [
        (index, cell) for index, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"
    ]


def pair_code_cells(
    base_code: list[tuple[int, dict]], current_code: list[tuple[int, dict]]
) -> tuple[str, list[tuple[int, dict, dict]], list[str]]:
    """コードセルを対応させる。

    (方法、(現在のセル番号, 基準のセル, 現在のセル) の列、構造の問題)を返す。
    """
    problems: list[str] = []
    base_ids = [cell.get("id") for _, cell in base_code]
    current_ids = [cell.get("id") for _, cell in current_code]
    use_ids = (
        all(identifier is not None for identifier in base_ids + current_ids)
        and len(set(base_ids)) == len(base_ids)
        and len(set(current_ids)) == len(current_ids)
    )
    if use_ids:
        base_by_id = {cell["id"]: cell for _, cell in base_code}
        current_id_set = set(current_ids)
        removed = [identifier for identifier in base_ids if identifier not in current_id_set]
        added = [identifier for identifier in current_ids if identifier not in base_by_id]
        if removed:
            problems.append(f"基準にあって現在にないコードセル(削除): id={removed}")
        if added:
            problems.append(f"基準になく現在にあるコードセル(追加): id={added}")
        common_in_base_order = [i for i in base_ids if i in current_id_set]
        common_in_current_order = [i for i in current_ids if i in base_by_id]
        if common_in_base_order != common_in_current_order:
            problems.append("共通のコードセルの順序が基準と違う(並べ替え)")
        pairs = [
            (index, base_by_id[cell["id"]], cell)
            for index, cell in current_code
            if cell["id"] in base_by_id
        ]
        return "セルの識別子(id)", pairs, problems
    if len(base_code) != len(current_code):
        problems.append(
            f"コードセルの数が基準と違う(基準 {len(base_code)} 個、現在 {len(current_code)} 個)。"
            "位置で対応させているため、セルごとの比較は行わない"
        )
        return "位置", [], problems
    pairs = [
        (current_index, base_cell, current_cell)
        for (_, base_cell), (current_index, current_cell) in zip(
            base_code, current_code, strict=True
        )
    ]
    return "位置", pairs, problems


def check_code_cell_pair(report: Report, index: int, base_cell: dict, current_cell: dict) -> None:
    label = describe_cell(index, current_cell)
    base_source, current_source = source_text(base_cell), source_text(current_cell)
    if base_source == current_source:
        report.source_unchanged += 1
    else:
        base_form, current_form = syntax_form(base_source), syntax_form(current_source)
        if base_form.kind != current_form.kind:
            report.problems.append(f"{label}: 構文木を作れるかどうかが基準と違う")
        elif base_form.structure != current_form.structure:
            reason = (
                "構文木が基準と違う(変更がコメント・空行・書式にとどまらない)"
                if current_form.kind == "syntax_tree"
                else "構文木を作れず、ソースの文字列が基準と違う"
            )
            report.problems.append(f"{label}: {reason}")
        else:
            report.tree_same_changed += 1
            if comment_tokens(base_source) != comment_tokens(current_source):
                report.comment_changed += 1
            else:
                report.format_only_changed += 1
            if base_form.line_numbers != current_form.line_numbers:
                report.position_shifted += 1
                if outputs_may_contain_line_numbers(current_cell):
                    report.notes.append(
                        f"{label}: ソース上の行の位置が動いた。出力(例外または stderr)に"
                        "行番号を含みうるので、再実行した結果と一致しない可能性がある"
                    )
    if base_cell.get("outputs", []) != current_cell.get("outputs", []):
        report.problems.append(f"{label}: outputs が基準と違う")
    if base_cell.get("execution_count") != current_cell.get("execution_count"):
        report.problems.append(
            f"{label}: execution_count が基準と違う"
            f"(基準 {base_cell.get('execution_count')}、現在 {current_cell.get('execution_count')})"
        )
    base_metadata = {k: v for k, v in base_cell.get("metadata", {}).items() if k != "tags"}
    current_metadata = {k: v for k, v in current_cell.get("metadata", {}).items() if k != "tags"}
    if base_metadata != current_metadata:
        keys = sorted(
            key
            for key in set(base_metadata) | set(current_metadata)
            if base_metadata.get(key) != current_metadata.get(key)
        )
        report.problems.append(f"{label}: tags 以外のメタデータが基準と違う(キー: {keys})")

    def other_tags(cell: dict) -> list[str]:
        return [
            t
            for t in tags_of(cell)
            if t not in ZENN_TAG_TARGETS and not t.startswith(ZENN_TAG_PREFIX)
        ]

    if other_tags(base_cell) != other_tags(current_cell):
        report.problems.append(
            f"{label}: Zenn のタグ以外のタグが基準と違う"
            f"(基準 {other_tags(base_cell)}、現在 {other_tags(current_cell)})"
        )


def check_tag_usage(report: Report, base_notebook: dict, current_notebook: dict) -> None:
    """現在のノートブックのすべてのセルで、タグの使い方が Zenn のタグの定めに合うかを確かめる。"""
    preexisting = {
        (cell["cell_type"], tag) for cell in base_notebook["cells"] for tag in tags_of(cell)
    }
    appendix_cells: list[tuple[int, dict]] = []
    for index, cell in enumerate(current_notebook["cells"]):
        cell_type = cell["cell_type"]
        raw_tags = cell.get("metadata", {}).get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
            report.problems.append(f"{describe_cell(index, cell)}: tags が文字列のリストではない")
            continue
        for tag in raw_tags:
            if tag in ZENN_TAG_TARGETS:
                if cell_type not in ZENN_TAG_TARGETS[tag]:
                    report.problems.append(
                        f"{describe_cell(index, cell)}: タグ {tag} は {cell_type} セルには"
                        f"付けられない(付けてよい種類: {list(ZENN_TAG_TARGETS[tag])})"
                    )
                if tag == "zenn-appendix":
                    appendix_cells.append((index, cell))
            elif tag.startswith(ZENN_TAG_PREFIX):
                report.problems.append(f"{describe_cell(index, cell)}: 未知の Zenn のタグ {tag}")
            elif cell_type != "code" and (cell_type, tag) not in preexisting:
                # コードセルの未知のタグは、対応するセルとのタグの比較で検出する
                report.problems.append(f"{describe_cell(index, cell)}: 基準にない未知のタグ {tag}")
    if len(appendix_cells) > 1:
        report.problems.append(
            f"zenn-appendix が {len(appendix_cells)} 個のセルにある"
            f"(セル {[i for i, _ in appendix_cells]})。付録の開始位置が 1 つに決まらない"
        )
    for index, cell in appendix_cells:
        lines = [line for line in source_text(cell).splitlines() if line.strip()]
        if not lines or not lines[0].lstrip().startswith("#"):
            report.problems.append(
                f"{describe_cell(index, cell)}: zenn-appendix は見出し(# で始まる行)のセルに付ける"
            )


def compare_notebooks(base_notebook: dict, current_notebook: dict) -> Report:
    report = Report()
    unknown_types = {
        cell["cell_type"]
        for notebook in (base_notebook, current_notebook)
        for cell in notebook["cells"]
        if cell["cell_type"] not in CELL_TYPES
    }
    if unknown_types:
        report.problems.append(f"未知のセルの種類がある: {sorted(unknown_types)}")

    base_code, current_code = code_cells(base_notebook), code_cells(current_notebook)
    report.base_code_count, report.current_code_count = len(base_code), len(current_code)
    report.pairing_method, pairs, structure_problems = pair_code_cells(base_code, current_code)
    report.problems.extend(structure_problems)
    for index, base_cell, current_cell in pairs:
        check_code_cell_pair(report, index, base_cell, current_cell)
    check_tag_usage(report, base_notebook, current_notebook)

    report.base_characters = markdown_characters(base_notebook)
    report.current_characters = markdown_characters(current_notebook)
    for index, cell in code_cells(current_notebook):
        if "zenn-images-only" in tags_of(cell) and not has_image_output(cell):
            report.notes.append(
                f"{describe_cell(index, cell)}: zenn-images-only だが画像の出力がない。"
                "Zenn ではセルごと除かれる(zenn-hide と同じ)"
            )
    base_index_of = {id(cell): index for index, cell in base_code}
    moved = [index for index, base_cell, _ in pairs if base_index_of[id(base_cell)] != index]
    references = [
        index
        for index, cell in enumerate(current_notebook["cells"])
        if CELL_REFERENCE_PATTERN.search(source_text(cell))
    ]
    if moved and references:
        report.notes.append(
            f"セルの番号への参照を含むセルがある(セル {references[:10]}"
            f"{'…' if len(references) > 10 else ''})。コードセルの位置が基準から {len(moved)} 個"
            f"動いた(セル {moved[:10]}{'…' if len(moved) > 10 else ''})。"
            "参照が指すセルが変わっていないかを確かめる"
        )

    base_markdown = Counter(
        source_text(c) for c in base_notebook["cells"] if c["cell_type"] == "markdown"
    )
    current_markdown = Counter(
        source_text(c) for c in current_notebook["cells"] if c["cell_type"] == "markdown"
    )
    report.markdown_base, report.markdown_current = (
        sum(base_markdown.values()),
        sum(current_markdown.values()),
    )
    report.markdown_added_or_rewritten = sum((current_markdown - base_markdown).values())
    report.markdown_removed_or_rewritten = sum((base_markdown - current_markdown).values())

    for tag in ZENN_TAG_TARGETS:
        counts = []
        for notebook in (base_notebook, current_notebook):
            for cell_type in ("code", "markdown"):
                counts.append(
                    sum(
                        1
                        for c in notebook["cells"]
                        if c["cell_type"] == cell_type and tag in tags_of(c)
                    )
                )
        report.tag_counts[tag] = (counts[0], counts[2], counts[1], counts[3])

    if any(c["cell_type"] == "raw" for n in (base_notebook, current_notebook) for c in n["cells"]):
        report.notes.append(
            "raw セルがある。raw セルの内容は検査していない(タグの使い方のみ検査する)"
        )
    for key in ("metadata", "nbformat", "nbformat_minor"):
        if base_notebook.get(key) != current_notebook.get(key):
            report.notes.append(f"ノートブック全体の {key} が基準と違う(判定には影響しない)")
    return report


def run_git(arguments: list[str], directory: Path) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=directory, capture_output=True, check=False
        )
    except FileNotFoundError as error:
        raise ScriptError("git が見つからない") from error
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ScriptError(f"git {' '.join(arguments)} が失敗した: {message}")
    return result.stdout


def load_current_notebook(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScriptError(f"{path} を読めない: {error}") from error


def load_base_notebook(path: Path, base_reference: str) -> tuple[dict, str]:
    """git の基準からノートブックを取り出す。(ノートブック、基準の説明)を返す。"""
    absolute = path.resolve()
    root = Path(
        run_git(["rev-parse", "--show-toplevel"], absolute.parent).decode("utf-8").strip()
    ).resolve()
    relative = absolute.relative_to(root).as_posix()
    content = run_git(["show", f"{base_reference}:{relative}"], root)
    try:
        notebook = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ScriptError(
            f"{base_reference} の {relative} が JSON として読めない: {error}"
        ) from error
    try:
        short_hash = run_git(["rev-parse", "--short", base_reference], root).decode("utf-8").strip()
    except ScriptError:
        short_hash = "不明"
    return notebook, f"{base_reference}({short_hash})"


def print_report(path: Path, base_description: str, report: Report) -> None:
    print(f"検査したノートブック: {path}")
    print(f"基準: {base_description}")
    print(
        f"コードセル: 基準 {report.base_code_count} 個、現在 {report.current_code_count} 個"
        f"(対応づけ: {report.pairing_method})"
    )
    print(f"  ソースが同一: {report.source_unchanged} 個")
    print(
        f"  ソースが変わり、構文木は同一: {report.tree_same_changed} 個"
        f"(コメントの変更を含む {report.comment_changed} 個、"
        f"空行・書式のみ {report.format_only_changed} 個、"
        f"うちソース上の行の位置が動いたもの {report.position_shifted} 個)"
    )
    print(
        f"Markdown セル: 基準 {report.markdown_base} 個、現在 {report.markdown_current} 個"
        f"(基準にない内容 {report.markdown_added_or_rewritten} 個、現在にない内容 "
        f"{report.markdown_removed_or_rewritten} 個。書き換えは両方に数える)"
    )
    (base_body, base_appendix), (body, appendix) = report.base_characters, report.current_characters
    print(
        f"Markdown の文字数: 本文(付録を除く)基準 {base_body:,} → 現在 {body:,}、"
        f"付録 基準 {base_appendix:,} → 現在 {appendix:,}"
    )
    print("Zenn のタグを付けたセル(現在の個数、括弧内は基準):")
    for tag, (base_code_n, current_code_n, base_md_n, current_md_n) in report.tag_counts.items():
        print(
            f"  {tag}: コードセル {current_code_n} 個({base_code_n})、"
            f"Markdown セル {current_md_n} 個({base_md_n})"
        )
    for note in report.notes:
        print(f"注意: {note}")
    if report.passed:
        print(
            "判定: 通った(コードセルの構文木・outputs・execution_count が一致し、"
            "タグの使い方が定めに合う)"
        )
    else:
        print(f"判定: 止まった(問題 {len(report.problems)} 件)")
        for problem in report.problems:
            print(f"  - {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path, help="検査するノートブック")
    parser.add_argument("--base", default="HEAD", help="比べる基準の git の参照(既定 HEAD)")
    args = parser.parse_args(argv)

    exit_code = 0
    failed: list[Path] = []
    for path in args.notebooks:
        try:
            current_notebook = load_current_notebook(path)
            base_notebook, base_description = load_base_notebook(path, args.base)
        except ScriptError as error:
            print(f"検査したノートブック: {path}\n実行できなかった: {error}")
            exit_code = 2
            failed.append(path)
            continue
        report = compare_notebooks(base_notebook, current_notebook)
        print_report(path, base_description, report)
        if not report.passed:
            exit_code = max(exit_code, 1)
            failed.append(path)
    if len(args.notebooks) > 1:
        print(f"\n合計: {len(args.notebooks)} 個を検査し、通らなかったもの {len(failed)} 個")
        for path in failed:
            print(f"  - {path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
