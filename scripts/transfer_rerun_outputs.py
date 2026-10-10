"""Colab で実行したノートブックから、6.17 節のコードセルの出力だけを、
リポジトリのノートブックに移す。

025 の 6.17 節(実験 D・E の再実行)は、初回の本番の出力が残っているノートブックに、
別のセッションの出力を加える。Colab で全体を実行したノートブックをそのままコミットすると、
6.16 節までの初回の出力が上書きされる。そこで、**6.17 節のコードセルの出力と実行回数だけ** を
移し、それ以外のセルは 1 バイトも変えない。停止用のセル(6.3 節の直後。実行すると常に例外で
止まる)の出力は移さない。**停止用のセルがない構成(現在の 025)では使わない。** 停止用の
セルを削除した後のノートブックは、上から順に実行できるので、出力の取り込みは行わない。

次のどれかが成り立たなければ、**何も書き換えずに** 終了コード 1 で停止する。

1. 2 つのノートブックのセルの数と、すべてのセルの種類・ソースが一致する(許す違いはない。
   Colab ではセルを書き換えない)。
2. 6.4〜6.16 節のコードセル(停止用のセルの次から 6.17 節の前まで)が、Colab 側で実行し
   直されていない。許す状態は、次のいずれか(「すべてのセルを実行」が停止用のセルで
   中断されたとき、Colab が後ろのセルの出力や実行回数を消す場合があるため)。
   (a) 出力が空で、実行回数がない。
   (b) 出力と実行回数が、リポジトリ側と完全に一致する。
   (c) 出力がリポジトリ側と一致し、実行回数だけがない(実行回数の消去)。
   停止用のセルは、出力が停止の例外(と、その直前の印字)だけであることを確かめる(移さない)。
3. 移す側の 6.17 節のコードセルがすべて実行済みで、エラーの出力がない(スモークテストの
   出力は、``--allow-smoke`` を付けない限り拒否する)。
4. 6.17 節以外のセル(6.16 節までと 7 章)の種類・ソース・出力・実行回数・メタデータと、
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
STOP_CELL_PREFIX = "# 停止用のセル"
SKIPPED_MESSAGE = "が False なので"  # 以前のフラグつきの構成で「実行しない」と出力された印
SMOKE_MARKER = "[スモークテスト]"


class TransferError(Exception):
    pass


def source_of(cell: dict) -> str:
    return "".join(cell["source"])


def find_unique(notebook: dict, prefix: str, what: str) -> int:
    found = [i for i, c in enumerate(notebook["cells"]) if source_of(c).startswith(prefix)]
    if len(found) != 1:
        raise TransferError(f"{what}を 1 つだけ見つけられない: {found}")
    return found[0]


def locate(notebook: dict) -> tuple[int, int, int]:
    """停止用のセル・6.17 節の最初のセル・7 章の最初のセルの位置。"""
    stop = find_unique(notebook, STOP_CELL_PREFIX, "停止用のセル")
    start = find_unique(notebook, SECTION_START_PREFIX, "6.17 節の見出し")
    end = find_unique(notebook, SECTION_END_PREFIX, "7 章の見出し")
    if not stop < start < end:
        raise TransferError(f"停止用のセル・6.17 節・7 章の順になっていない: {(stop, start, end)}")
    return stop, start, end


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


def normalized_outputs(cell: dict) -> str:
    # Colab が保存し直したときの、文字列の分割や末尾の空白の違いを無視して比べる
    def normalize(value):
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return "".join(value).rstrip()
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value.rstrip() if isinstance(value, str) else value

    return json.dumps(normalize(cell.get("outputs", [])), ensure_ascii=False, sort_keys=True)


def check_stop_cell(cell: dict) -> None:
    """停止用のセル: 出力は、停止の例外とその直前の印字だけ(出力が空なら、実行されていない)。"""
    outputs = cell.get("outputs", [])
    if not outputs:
        return
    errors = [o for o in outputs if o["output_type"] == "error"]
    others = [o for o in outputs if o["output_type"] not in ("error", "stream")]
    if others or len(errors) != 1:
        raise TransferError("停止用のセルの出力が、停止の例外とその直前の印字だけではない")
    error = errors[0]
    if error.get("ename") != "RuntimeError" or "停止用のセル" not in error.get("evalue", ""):
        raise TransferError("停止用のセルの例外が、停止用のセルのものではない")


def not_re_executed(repository_cell: dict, executed_cell: dict) -> bool:
    """6.4〜6.16 節のコードセルが、Colab 側で実行し直されていないか(許す状態 (a)(b)(c))。"""
    count_a, count_b = repository_cell.get("execution_count"), executed_cell.get("execution_count")
    same_outputs = normalized_outputs(repository_cell) == normalized_outputs(executed_cell)
    empty_and_uncounted = not executed_cell.get("outputs") and count_b is None  # (a)
    unchanged = same_outputs and count_b in (count_a, None)  # (b)(c)
    return empty_and_uncounted or unchanged


def check(repository: dict, executed: dict, allow_smoke: bool) -> tuple[int, int, int]:
    stop, start, end = locate(repository)
    if locate(executed) != (stop, start, end) or len(repository["cells"]) != len(executed["cells"]):
        raise TransferError(
            "セルの数・停止用のセルと 6.17 節の位置が一致しない"
            f"(リポジトリ側 {len(repository['cells'])} 個・{(stop, start, end)}、"
            f"実行した側 {len(executed['cells'])} 個・{locate(executed)})"
        )
    for i, (a, b) in enumerate(zip(repository["cells"], executed["cells"], strict=True)):
        if a["cell_type"] != b["cell_type"] or source_of(a) != source_of(b):
            raise TransferError(f"セル {i} の種類またはソースが一致しない(セルは書き換えない)")
    check_stop_cell(executed["cells"][stop])
    for i in range(stop + 1, start):  # 6.4〜6.16 節: 実行し直されていない
        a, b = repository["cells"][i], executed["cells"][i]
        if a["cell_type"] != "code":
            continue
        if not not_re_executed(a, b):
            raise TransferError(
                f"6.4〜6.16 節のセル {i} の実行回数か出力がリポジトリ側と違う"
                "(実行し直されている。停止用のセルで止めずに実行した可能性がある)"
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
            raise TransferError(f"{label} に「実行しない」の出力がある(実行されていない)")
        if not allow_smoke and any(SMOKE_MARKER in t for t in texts):
            raise TransferError(
                f"{label} の出力にスモークテストの印がある(--allow-smoke を付けない限り移さない)"
            )
    return stop, start, end


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
        _, start, end = check(repository, executed, args.allow_smoke)
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
