"""``check_notebook_edit_equivalence.py`` の試験(試験用の小さなノートブックを使う)。

比較の関数(``compare_notebooks``)は辞書のノートブックで、git の基準の取り出しと終了コードは
一時ディレクトリに作った git リポジトリで確かめる。ノートブックの実行はしない。

使い方 / Usage::

    uv run python scripts/test_check_notebook_edit_equivalence.py
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import check_notebook_edit_equivalence as checker  # noqa: E402

SCRIPT_PATH = SCRIPT_DIRECTORY / "check_notebook_edit_equivalence.py"


def code_cell(
    source: str,
    cell_id: str,
    outputs: list | None = None,
    execution_count: int | None = 1,
    metadata: dict | None = None,
) -> dict:
    if outputs is None:
        outputs = [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": metadata if metadata is not None else {},
        "source": source.splitlines(keepends=True),
        "outputs": outputs,
        "execution_count": execution_count,
    }


def markdown_cell(source: str, cell_id: str, metadata: dict | None = None) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": metadata if metadata is not None else {},
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def base_notebook() -> dict:
    return notebook(
        [
            markdown_cell("# タイトル\n\n本文", "m1"),
            code_cell(
                "%cd /content/ai-theories\n!pip install uv -q\nimport os\nx = 1\nprint(x)\n", "c1"
            ),
            markdown_cell("## 実験\n\n説明", "m2"),
            code_cell(
                'def f(a):\n    """二倍にする。"""\n    return a * 2\n\nname = "alpha"\n'
                "print(f(2), name)\n",
                "c2",
            ),
            code_cell("y = 3  # 値\nprint(y)\n", "c3"),
        ]
    )


def edited(base: dict, cell_id: str, source: str) -> dict:
    current = copy.deepcopy(base)
    for cell in current["cells"]:
        if cell["id"] == cell_id:
            cell["source"] = source.splitlines(keepends=True)
    return current


class ComparisonTests(unittest.TestCase):
    def assert_passes(self, current: dict, base: dict | None = None) -> checker.Report:
        report = checker.compare_notebooks(base or base_notebook(), current)
        self.assertTrue(report.passed, report.problems)
        return report

    def assert_stops(
        self, current: dict, expected: str, base: dict | None = None
    ) -> checker.Report:
        report = checker.compare_notebooks(base or base_notebook(), current)
        self.assertFalse(report.passed, "止まるはずの編集が通った")
        self.assertTrue(
            any(expected in problem for problem in report.problems),
            f"期待した理由 {expected!r} が見つからない: {report.problems}",
        )
        return report

    # --- 通る場合 ---

    def test_unchanged_notebook_passes(self):
        report = self.assert_passes(base_notebook())
        self.assertEqual(report.source_unchanged, 3)
        self.assertEqual(report.tree_same_changed, 0)

    def test_comment_change_passes(self):
        current = edited(
            base_notebook(), "c3", "y = 3  # 書き換えた値の説明\n# 新しい行\nprint(y)\n"
        )
        report = self.assert_passes(current)
        self.assertEqual((report.tree_same_changed, report.comment_changed), (1, 1))
        self.assertEqual(report.format_only_changed, 0)

    def test_comment_deletion_passes(self):
        report = self.assert_passes(edited(base_notebook(), "c3", "y = 3\nprint(y)\n"))
        self.assertEqual(report.comment_changed, 1)

    def test_blank_line_and_spacing_pass_as_format_only(self):
        current = edited(base_notebook(), "c3", "y   =   3  # 値\n\n\nprint( y )\n")
        report = self.assert_passes(current)
        self.assertEqual((report.comment_changed, report.format_only_changed), (0, 1))

    def test_markdown_free_rewrite_and_move_pass(self):
        current = base_notebook()
        current["cells"][0]["source"] = ["# 書き換えたタイトル\n"]
        current["cells"].append(current["cells"].pop(2))  # Markdown セルをコードセルの後ろへ移動
        current["cells"].append(markdown_cell("## 付録\n", "m3"))
        report = self.assert_passes(current)
        self.assertEqual(report.markdown_current, 3)
        self.assertGreaterEqual(report.markdown_added_or_rewritten, 2)

    def test_known_tags_pass_and_are_counted(self):
        current = base_notebook()
        current["cells"][1]["metadata"]["tags"] = ["zenn-hide"]
        current["cells"][3]["metadata"]["tags"] = ["zenn-hide-input", "zenn-details"]
        current["cells"][2]["metadata"]["tags"] = ["zenn-details"]
        current["cells"][0]["metadata"]["tags"] = ["zenn-appendix"]
        report = self.assert_passes(current)
        self.assertEqual(report.tag_counts["zenn-hide"], (0, 1, 0, 0))
        self.assertEqual(report.tag_counts["zenn-hide-input"], (0, 1, 0, 0))
        self.assertEqual(report.tag_counts["zenn-details"], (0, 1, 0, 1))
        self.assertEqual(report.tag_counts["zenn-appendix"], (0, 0, 0, 1))

    def test_tag_removal_passes(self):
        base = base_notebook()
        base["cells"][1]["metadata"]["tags"] = ["zenn-hide"]
        current = base_notebook()  # タグなし(キーごとない)
        self.assert_passes(current, base)
        current["cells"][1]["metadata"]["tags"] = []  # 空のリストも同じ扱い
        self.assert_passes(current, base)

    def test_preexisting_unknown_tag_unchanged_passes(self):
        base = base_notebook()
        base["cells"][1]["metadata"]["tags"] = ["parameters"]
        base["cells"][0]["metadata"]["tags"] = ["note"]
        current = copy.deepcopy(base)
        current["cells"][1]["metadata"]["tags"] = ["parameters", "zenn-hide"]
        self.assert_passes(current, base)

    # --- IPython の記法 ---

    def test_ipython_syntax_comment_change_passes(self):
        current = edited(
            base_notebook(),
            "c1",
            "# 環境セットアップ\n%cd /content/ai-theories\n!pip install uv -q\n"
            "import os\nx = 1\nprint(x)\n",
        )
        report = self.assert_passes(current)
        self.assertEqual(report.comment_changed, 1)

    def test_ipython_magic_argument_change_stops(self):
        current = edited(
            base_notebook(),
            "c1",
            "%cd /content/other\n!pip install uv -q\nimport os\nx = 1\nprint(x)\n",
        )
        self.assert_stops(current, "構文木が基準と違う")

    def test_ipython_shell_command_change_stops(self):
        current = edited(
            base_notebook(),
            "c1",
            "%cd /content/ai-theories\n!pip install uv\nimport os\nx = 1\nprint(x)\n",
        )
        self.assert_stops(current, "構文木が基準と違う")

    # --- 止まる場合 ---

    def test_string_change_stops(self):
        source = (
            'def f(a):\n    """二倍にする。"""\n    return a * 2\n\nname = "beta"\n'
            "print(f(2), name)\n"
        )
        self.assert_stops(edited(base_notebook(), "c2", source), "構文木が基準と違う")

    def test_print_message_change_stops(self):
        self.assert_stops(
            edited(base_notebook(), "c3", 'y = 3  # 値\nprint(y, "追加")\n'), "構文木が基準と違う"
        )

    def test_docstring_change_stops(self):
        source = (
            'def f(a):\n    """2 倍にする。"""\n    return a * 2\n\nname = "alpha"\n'
            "print(f(2), name)\n"
        )
        self.assert_stops(edited(base_notebook(), "c2", source), "構文木が基準と違う")

    def test_variable_rename_stops(self):
        self.assert_stops(
            edited(base_notebook(), "c3", "z = 3  # 値\nprint(z)\n"), "構文木が基準と違う"
        )

    def test_line_addition_stops(self):
        self.assert_stops(
            edited(base_notebook(), "c3", "y = 3  # 値\ny += 1\nprint(y)\n"), "構文木が基準と違う"
        )

    def test_number_change_stops(self):
        self.assert_stops(
            edited(base_notebook(), "c3", "y = 4  # 値\nprint(y)\n"), "構文木が基準と違う"
        )

    def test_integer_to_float_stops(self):
        self.assert_stops(
            edited(base_notebook(), "c3", "y = 3.0  # 値\nprint(y)\n"), "構文木が基準と違う"
        )

    def test_cell_deletion_stops(self):
        current = base_notebook()
        del current["cells"][4]
        self.assert_stops(current, "コードセル(削除)")

    def test_cell_addition_stops(self):
        current = base_notebook()
        current["cells"].append(code_cell("print(1)\n", "c4"))
        self.assert_stops(current, "コードセル(追加)")

    def test_code_cell_reorder_stops(self):
        current = base_notebook()
        current["cells"][3], current["cells"][4] = current["cells"][4], current["cells"][3]
        self.assert_stops(current, "並べ替え")

    def test_reorder_without_ids_stops(self):
        base, current = base_notebook(), base_notebook()
        for cell in base["cells"] + current["cells"]:
            del cell["id"]
        current["cells"][3], current["cells"][4] = current["cells"][4], current["cells"][3]
        report = self.assert_stops(current, "構文木が基準と違う", base)
        self.assertEqual(report.pairing_method, "位置")

    def test_cell_deletion_without_ids_stops(self):
        base, current = base_notebook(), base_notebook()
        for cell in base["cells"] + current["cells"]:
            del cell["id"]
        del current["cells"][4]
        self.assert_stops(current, "コードセルの数が基準と違う", base)

    def test_output_change_stops(self):
        current = base_notebook()
        current["cells"][4]["outputs"][0]["text"] = ["2\n"]
        self.assert_stops(current, "outputs が基準と違う")

    def test_output_removal_stops(self):
        current = base_notebook()
        current["cells"][4]["outputs"] = []
        self.assert_stops(current, "outputs が基準と違う")

    def test_execution_count_change_stops(self):
        current = base_notebook()
        current["cells"][4]["execution_count"] = 7
        self.assert_stops(current, "execution_count が基準と違う")

    def test_unknown_tag_on_code_cell_stops(self):
        current = base_notebook()
        current["cells"][1]["metadata"]["tags"] = ["hide-input"]
        self.assert_stops(current, "Zenn のタグ以外のタグが基準と違う")

    def test_unknown_zenn_tag_stops(self):
        current = base_notebook()
        current["cells"][1]["metadata"]["tags"] = ["zenn-hidden"]
        self.assert_stops(current, "未知の Zenn のタグ")

    def test_unknown_tag_on_markdown_cell_stops(self):
        current = base_notebook()
        current["cells"][0]["metadata"]["tags"] = ["note"]
        self.assert_stops(current, "基準にない未知のタグ")

    def test_tag_on_wrong_cell_type_stops(self):
        current = base_notebook()
        current["cells"][1]["metadata"]["tags"] = ["zenn-appendix"]
        self.assert_stops(current, "付けられない")
        current = base_notebook()
        current["cells"][0]["metadata"]["tags"] = ["zenn-hide-input"]
        self.assert_stops(current, "付けられない")

    def test_appendix_on_non_heading_and_duplicates_stop(self):
        current = base_notebook()
        current["cells"][0]["metadata"]["tags"] = ["zenn-appendix"]
        current["cells"][2]["metadata"]["tags"] = ["zenn-appendix"]
        self.assert_stops(current, "付録の開始位置が 1 つに決まらない")
        current = base_notebook()
        current["cells"][0]["source"] = ["見出しでない本文\n"]
        current["cells"][0]["metadata"]["tags"] = ["zenn-appendix"]
        self.assert_stops(current, "見出し")

    def test_other_metadata_change_stops(self):
        base = base_notebook()
        base["cells"][1]["metadata"]["run_part"] = "AB"
        current = copy.deepcopy(base)
        current["cells"][1]["metadata"]["run_part"] = "C"
        self.assert_stops(current, "tags 以外のメタデータが基準と違う", base)

    def test_code_cell_turned_into_markdown_stops(self):
        current = base_notebook()
        current["cells"][4] = markdown_cell("y = 3\n", "c3")
        self.assert_stops(current, "コードセル(削除)")

    # --- 構文木を作れないセル、注意 ---

    def test_unparseable_cell_requires_identical_source(self):
        base = notebook([code_cell("def f(:\n    pass\n", "u1")])
        self.assert_passes(copy.deepcopy(base), base)
        self.assert_stops(edited(base, "u1", "# 注\ndef f(:\n    pass\n"), "構文木を作れず", base)

    def test_syntax_validity_change_stops(self):
        base = notebook([code_cell("x = 1\n", "u1")])
        self.assert_stops(edited(base, "u1", "x = =\n"), "構文木を作れるかどうか", base)

    def test_line_shift_with_traceback_output_gives_note_but_passes(self):
        stderr = {
            "output_type": "stream",
            "name": "stderr",
            "text": ["/tmp/ipykernel_1/123.py:2: UserWarning: x\n"],
        }
        base = notebook(
            [code_cell("import warnings\nwarnings.warn('x')\n", "w1", outputs=[stderr])]
        )
        current = edited(base, "w1", "import warnings\n# 説明\nwarnings.warn('x')\n")
        report = self.assert_passes(current, base)
        self.assertEqual(report.position_shifted, 1)
        self.assertTrue(any("行の位置が動いた" in note for note in report.notes), report.notes)

    def test_line_shift_without_line_numbers_gives_no_note(self):
        current = edited(base_notebook(), "c3", "# 先頭の説明\ny = 3  # 値\nprint(y)\n")
        report = self.assert_passes(current)
        self.assertEqual(report.position_shifted, 1)
        self.assertEqual(report.notes, [])

    def test_same_line_comment_rewrite_does_not_shift_positions(self):
        report = self.assert_passes(edited(base_notebook(), "c3", "y = 3  # 別の説明\nprint(y)\n"))
        self.assertEqual(report.position_shifted, 0)


class GitIntegrationTests(unittest.TestCase):
    """git の基準の取り出し、サブディレクトリ、別のカレントディレクトリ、終了コードを確かめる。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.notebook_path = self.root / "theories" / "01_test" / "001_test.ipynb"
        self.notebook_path.parent.mkdir(parents=True)
        self.write(base_notebook())
        self.git("init", "-q")
        self.commit("base")

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *arguments: str):
        environment = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
        subprocess.run(
            ["git", *arguments], cwd=self.root, check=True, env=environment, capture_output=True
        )

    def commit(self, message: str):
        self.git("add", ".")
        self.git(
            *("-c", "user.name=test", "-c", "user.email=test@example.com"),
            *("-c", "commit.gpgsign=false", "commit", "-q", "-m", message),
        )

    def write(self, content: dict):
        self.notebook_path.write_text(
            json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def run_script(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *arguments],
            cwd=tempfile.gettempdir(),  # リポジトリの外のカレントディレクトリから実行する
            capture_output=True,
            text=True,
            check=False,
        )

    def test_unchanged_passes_with_exit_code_zero(self):
        result = self.run_script(str(self.notebook_path))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("判定: 通った", result.stdout)

    def test_comment_change_passes_and_is_reported(self):
        self.write(edited(base_notebook(), "c3", "y = 3  # 書き換えた\nprint(y)\n"))
        result = self.run_script(str(self.notebook_path))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("コメントの変更を含む 1 個", result.stdout)

    def test_code_change_stops_with_exit_code_one(self):
        self.write(edited(base_notebook(), "c3", "y = 4  # 値\nprint(y)\n"))
        result = self.run_script(str(self.notebook_path))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("判定: 止まった", result.stdout)

    def test_base_reference_selects_the_baseline(self):
        self.write(edited(base_notebook(), "c3", "y = 4  # 値\nprint(y)\n"))
        self.commit("code change")
        self.assertEqual(self.run_script(str(self.notebook_path)).returncode, 0)  # HEAD と同じ
        self.assertEqual(self.run_script(str(self.notebook_path), "--base", "HEAD~1").returncode, 1)

    def test_missing_base_file_exits_with_code_two(self):
        other = self.notebook_path.with_name("002_new.ipynb")
        other.write_text(json.dumps(base_notebook()), encoding="utf-8")
        result = self.run_script(str(other))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("実行できなかった", result.stdout)

    def test_invalid_json_exits_with_code_two(self):
        self.notebook_path.write_text("{ not json", encoding="utf-8")
        result = self.run_script(str(self.notebook_path))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_multiple_notebooks_report_each(self):
        other = self.notebook_path.with_name("002_new.ipynb")
        other.write_text(json.dumps(base_notebook()), encoding="utf-8")
        result = self.run_script(str(self.notebook_path), str(other))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("合計: 2 個を検査し、通らなかったもの 1 個", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
