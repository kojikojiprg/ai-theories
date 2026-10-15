"""画面のテーマ。"""

from __future__ import annotations

import gradio as gr

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
"""Gradio 6 では、テーマは ``Blocks`` ではなく ``launch()`` に渡す。"""


def launch_kwargs() -> dict:
    """``demo.launch()`` に渡す、画面の見た目に関する引数。"""
    return {"theme": THEME}
