"""ページの配置。各機能のページ(``features/*/ui.py``)を、タブとして組み立てる。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import gradio as gr

from lab.core.artifacts import Resources
from lab.ui.components import about_markdown, describe_models, intro_markdown

APP_TITLE = "AI Theories Lab"


@dataclass(frozen=True)
class Page:
    """タブ 1 つぶんのページ。

    Attributes:
        title: タブの表示名。
        build: ``gr.Tab`` の中で呼ばれ、そのページの部品とイベントを組み立てる関数。
    """

    title: str
    build: Callable[[Resources], None]


def build_app(resources: Resources, pages: Sequence[Page]) -> gr.Blocks:
    """画面全体を組み立てる(先頭の説明・読み込んだモデルの表・機能のタブ・末尾の出典)。"""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(intro_markdown(resources.models["main"]))
        with gr.Accordion("読み込んだモデル / Loaded Models", open=False):
            gr.Markdown(describe_models(resources.models, resources.device))
        with gr.Tabs():
            for page in pages:
                with gr.Tab(page.title):
                    page.build(resources)
        gr.Markdown(about_markdown())
    return demo
