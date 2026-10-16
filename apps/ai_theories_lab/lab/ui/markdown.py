"""Markdown の文章を組み立てる小さな関数。

Markdown は段落の中の改行を空白として描画する。日本語の文の途中に空白が入らないように、
ソースの上での折り返しは文字列リテラルの連結で行い、文字列としては改行しない。
"""

from __future__ import annotations


def paragraphs(*blocks: str) -> str:
    """段落(空行で区切る)を連結する。段落の中は改行せず、隣接する文字列リテラルで書く。"""
    return "\n\n".join(blocks)


def items(*entries: str) -> str:
    """箇条書き(各項目は改行しない)。"""
    return "\n".join(f"- {entry}" for entry in entries)
