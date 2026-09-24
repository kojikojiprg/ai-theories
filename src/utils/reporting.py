"""判定の一次情報をセル出力に全件印字するための整形(013)。

``json.dumps(obj, indent=2)`` は数値の配列を 1 値 1 行に展開するため、長い配列(評価窓ごとの
負の対数尤度など)を含むと出力が縦に長くなりすぎ、GitHub のノートブックプレビューで読めなくなる。
``dumps_compact_json`` は次の規則で整形する。

- 辞書はインデントで展開する。
- スカラー(数値・文字列・bool・None)のみからなる配列は、1 行あたり ``width`` 文字以内で
  折り返し、1 行に複数の値を並べる(1 行に収まる場合は 1 行で出力する)。
- スカラーのみを値に持つ辞書は、1 行に収まる場合は 1 行で出力する。収まらない場合は展開する。
- 浮動小数点数は ``repr`` と同じ全桁で出力する(丸めない。``json.dumps`` と同じ表現)。

出力は有効な JSON であり、``json.loads(dumps_compact_json(x)) == x`` が成り立つ(``NaN``・
``Infinity`` を含まない入力の場合。タプルは配列として出力されるので、読み戻すとリストになる)。

使い方 / Usage::

    from src.utils.reporting import dumps_compact_json

    text = dumps_compact_json(primary_data)
    print(text)
    assert json.loads(text) == primary_data
"""

from __future__ import annotations

import json
from typing import Any

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _is_scalar(value: Any) -> bool:
    return isinstance(value, _SCALAR_TYPES)


def _scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _format(value: Any, level: int, column: int, indent: int, width: int) -> str:
    """``value`` を整形する。``column`` はこの値が書き始められる桁(直前のキーを含む)。

    行末に置かれうるカンマの 1 文字分を常に確保して ``width`` 以内に収める。
    """
    pad = " " * (indent * level)
    inner_pad = " " * (indent * (level + 1))

    if _is_scalar(value):
        return _scalar(value)

    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [(_scalar(str(k)), v) for k, v in value.items()]
        if all(_is_scalar(v) for _, v in items):
            inline = "{" + ", ".join(f"{k}: {_scalar(v)}" for k, v in items) + "}"
            if column + len(inline) + 1 <= width:
                return inline
        lines = []
        for k, v in items:
            head = f"{inner_pad}{k}: "
            lines.append(head + _format(v, level + 1, len(head), indent, width))
        return "{\n" + ",\n".join(lines) + "\n" + pad + "}"

    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(_is_scalar(v) for v in value):
            tokens = [_scalar(v) for v in value]
            inline = "[" + ", ".join(tokens) + "]"
            if column + len(inline) + 1 <= width:
                return inline
            lines, current = [], ""
            for token in tokens:
                candidate = token if not current else f"{current}, {token}"
                # 行末のカンマ 1 文字分を確保する
                if current and len(inner_pad) + len(candidate) + 1 > width:
                    lines.append(current)
                    current = token
                else:
                    current = candidate
            lines.append(current)
            return "[\n" + ",\n".join(inner_pad + line for line in lines) + "\n" + pad + "]"
        lines = [inner_pad + _format(v, level + 1, len(inner_pad), indent, width) for v in value]
        return "[\n" + ",\n".join(lines) + "\n" + pad + "]"

    raise TypeError(f"JSON に変換できない型: {type(value).__name__}")


def dumps_compact_json(obj: Any, indent: int = 2, width: int = 100) -> str:
    """スカラーの配列を折り返して並べる、縦に短い JSON の文字列を返す。

    整形の規則はモジュールの docstring を参照。

    Args:
        obj: 辞書・配列・スカラーからなるオブジェクト(``NaN``・``Infinity`` は不可)。
        indent: 1 段あたりのインデントの空白数。
        width: 1 行の最大文字数(単一の値や単一のキーだけで ``width`` を超える場合を除く)。

    Returns:
        有効な JSON の文字列。
    """
    return _format(obj, 0, 0, indent, width)
