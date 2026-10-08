"""Colab で実行したノートブックから、6.17 節のコードセルの出力だけを、
リポジトリのノートブックに移す。

025 の 6.17 節(実験 D・E の再実行)は、初回の本番の出力が残っているノートブックに、
別のセッションの出力を加える。Colab で全体を実行したノートブックをそのままコミットすると、
6.16 節までの初回の出力が上書きされる。そこで、**6.17 節のコードセルの出力と実行回数だけ** を
移し、それ以外のセルは 1 バイトも変えない。

次のどれかが成り立たなければ、**何も書き換えずに** 終了コード 1 で停止する。

1. 2 つのノートブックのセルの数と、すべてのセルの種類・ソースが一致する。許す違いは、
   6.17 節の実行のフラグ ``RUN_DE_RERUN``・``UPLOAD_RERUN_ARTIFACT`` の値だけ
   (Colab で書き換えるのはこの 2 行だけ)。
2. 移す側の 6.17 節のコードセルがすべて実行済みで、エラーの出力がなく、実行のフラグが
   True で実行されている(スモークテストの出力は、``--allow-smoke`` を付けない限り拒否する)。
3. 6.17 節以外のセル(6.16 節までと 7 章)の種類・ソース・出力・実行回数・メタデータと、
   ノートブック全体のメタデータが、書き込みの前後で一致する(書き込みの後に読み直して
   確かめ、一致しなければ元に戻す)。

使い方 / Usage::

    uv run python scripts/transfer_rerun_outputs.py <Colab で実行したノートブック>
    uv run python scripts/transfer_rerun_outputs.py <Colab で実行したノートブック> --dry-run

``--dry-run`` は検査だけを行う。``scripts/`` の他のスクリプトと同じく、ローカルで実行し、
結果は標準出力に残す。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

DEFAULT_REPOSITORY_NOTEBOOK = (
    Path(__file__).resolve().parent.parent
    / "theories"
    / "07_retrieval"
    / "025_ann_search_and_reranking.ipynb"
)
SECTION_START_PREFIX = "### 6.17"
SECTION_END_PREFIX = "## 7."
FLAG_LINE_PATTERN = re.compile(
    r"^(RUN_DE_RERUN|UPLOAD_RERUN_ARTIFACT) = (True|False)", flags=re.MULTILINE
)
SKIPPED_MESSAGE = "RUN_DE_RERUN が False なので"
SMOKE_MARKER = "[スモークテスト]"


class TransferError(Exception):
    pass


def source_of(cell: dict) -> str:
    return "".join(cell["source"])


def section_range(notebook: dict) -> tuple[int, int]:
    cells = notebook["cells"]
    starts = [i for i, c in enumerate(cells) if source_of(c).startswith(SECTION_START_PREFIX)]
    ends = [i for i, c in enumerate(cells) if source_of(c).startswith(SECTION_END_PREFIX)]
    if len(starts) != 1 or len(ends) != 1 or not starts[0] < ends[0]:
        raise TransferError(
            f"6.17 節の始まりと 7 章の始まりを 1 つずつ見つけられない: {starts}、{ends}"
        )
    return starts[0], ends[0]


def normalized_source(cell: dict, in_section: bool) -> str:
    text = source_of(cell)
    if in_section and cell["cell_type"] == "code":
        text = FLAG_LINE_PATTERN.sub(r"\1 = <値>", text)
    return text


def output_texts(cell: dict) -> list[str]:
    texts = []
    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            texts.append("".join(output["text"]))
        elif output["output_type"] in ("display_data", "execute_result"):
            texts.append("".join(output.get("data", {}).get("text/plain", [])))
    return texts


def fingerprint(cell: dict) -> str:
    return json.dumps(cell, ensure_ascii=False, sort_keys=True)


def check(repository: dict, executed: dict, allow_smoke: bool) -> tuple[int, int]:
    start, end = section_range(repository)
    if section_range(executed) != (start, end) or len(repository["cells"]) != len(
        executed["cells"]
    ):
        raise TransferError(
            "セルの数・6.17 節の位置が一致しない"
            f"(リポジトリ側 {len(repository['cells'])} 個・{(start, end)}、"
            f"実行した側 {len(executed['cells'])} 個・{section_range(executed)})"
        )
    for i, (a, b) in enumerate(zip(repository["cells"], executed["cells"], strict=True)):
        in_section = start <= i < end
        if a["cell_type"] != b["cell_type"] or normalized_source(
            a, in_section
        ) != normalized_source(b, in_section):
            raise TransferError(
                f"セル {i} の種類またはソースが一致しない"
                "(6.17 節の実行のフラグの値の違いだけは許す)"
            )
    for i in range(start, end):
        cell = executed["cells"][i]
        if cell["cell_type"] != "code":
            continue
        label = f"6.17 節のセル {i}(先頭: {source_of(cell).splitlines()[0][:40]!r})"
        if cell.get("execution_count") is None:
            raise TransferError(f"{label} が実行されていない")
        if not cell.get("outputs"):
            raise TransferError(f"{label} に出力がない")
        if any(o["output_type"] == "error" for o in cell["outputs"]):
            raise TransferError(f"{label} にエラーの出力がある")
        texts = output_texts(cell)
        if any(SKIPPED_MESSAGE in t for t in texts):
            raise TransferError(
                f"{label} は、実行のフラグ RUN_DE_RERUN が False のまま実行されている"
            )
        if not allow_smoke and any(SMOKE_MARKER in t for t in texts):
            raise TransferError(
                f"{label} の出力にスモークテストの印がある(--allow-smoke を付けない限り移さない)"
            )
    return start, end


def transfer(repository: dict, executed: dict, start: int, end: int) -> dict:
    result = copy.deepcopy(repository)
    for i in range(start, end):
        if repository["cells"][i]["cell_type"] != "code":
            continue
        result["cells"][i]["outputs"] = copy.deepcopy(executed["cells"][i]["outputs"])
        result["cells"][i]["execution_count"] = executed["cells"][i]["execution_count"]
    return result


def verify_untouched(before: dict, after: dict, start: int, end: int) -> None:
    if (
        before["metadata"] != after["metadata"]
        or before["nbformat"] != after["nbformat"]
        or before["nbformat_minor"] != after["nbformat_minor"]
    ):
        raise TransferError("ノートブック全体のメタデータが変わった")
    if len(before["cells"]) != len(after["cells"]):
        raise TransferError("セルの数が変わった")
    for i, (a, b) in enumerate(zip(before["cells"], after["cells"], strict=True)):
        if start <= i < end and a["cell_type"] == "code":
            if source_of(a) != source_of(b) or a["metadata"] != b["metadata"]:
                raise TransferError(f"6.17 節のセル {i} のソースまたはメタデータが変わった")
        elif fingerprint(a) != fingerprint(b):
            raise TransferError(f"6.17 節以外のセル {i} が変わった(6.16 節までと 7 章は変えない)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "executed_notebook",
        type=Path,
        help="Colab で実行して、リポジトリとは別のファイルとしてダウンロードしたノートブック",
    )
    parser.add_argument("--repository-notebook", type=Path, default=DEFAULT_REPOSITORY_NOTEBOOK)
    parser.add_argument("--dry-run", action="store_true", help="検査だけを行い、書き込まない")
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="スモークテストの出力を許す(スクリプトの動作確認用)",
    )
    args = parser.parse_args()
    try:
        original_text = args.repository_notebook.read_text(encoding="utf-8")
        repository = json.loads(original_text)
        executed = json.loads(args.executed_notebook.read_text(encoding="utf-8"))
        start, end = check(repository, executed, args.allow_smoke)
        result = transfer(repository, executed, start, end)
        verify_untouched(repository, result, start, end)
        code_cells = [i for i in range(start, end) if repository["cells"][i]["cell_type"] == "code"]
        print(
            f"検査: 通過(セル {len(repository['cells'])} 個、"
            f"6.17 節はセル {start}〜{end - 1}、コードセル {len(code_cells)} 個)"
        )
        if args.dry_run:
            print("--dry-run のため、書き込まない")
            return 0
        args.repository_notebook.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # 書き込みの後に読み直して、6.17 節以外が変わっていないことを確かめる(変わっていれば戻す)
        try:
            verify_untouched(
                repository,
                json.loads(args.repository_notebook.read_text(encoding="utf-8")),
                start,
                end,
            )
        except TransferError:
            args.repository_notebook.write_text(original_text, encoding="utf-8")
            raise
        print(
            f"移した: 6.17 節のコードセル {code_cells} の出力と実行回数。"
            "6.17 節以外のセルの出力は、書き込みの前後で一致した"
        )
        return 0
    except TransferError as error:
        print(f"停止: {error}(何も書き換えていない)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
