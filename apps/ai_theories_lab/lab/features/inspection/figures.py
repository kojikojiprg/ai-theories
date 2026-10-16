"""注意の重みの図(matplotlib)。

図の文字は英語だけにする(Spaces の環境には日本語のフォントがなく、日本語は表示されないため)。
説明は、図の外の画面の文言(日本語)で行う。pyplot は使わず ``Figure`` を直接作る
(pyplot はグローバルな状態を持ち、複数のリクエストが同時に図を作ると壊れるため)。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure


def _tick_label(text: str) -> str:
    """図の目盛りの文字列。``$`` は matplotlib の数式の記法として解釈されるので、エスケープする。"""
    return text.replace("$", r"\$")


def attention_figure(matrix: np.ndarray, token_texts: Sequence[str], title: str) -> Figure:
    """注意の重み ``(S, S)`` のヒートマップ(行が Query の位置、列が Key の位置)を作る。

    因果マスクにより、対角より右上(未来のトークン)の重みは 0。その部分は白く塗って、
    「見られない」位置であることが分かるようにする。
    """
    num_tokens = matrix.shape[0]
    if matrix.shape != (num_tokens, num_tokens) or len(token_texts) != num_tokens:
        raise ValueError("注意の重みの形状とトークン数が合わない")
    masked = np.where(np.tril(np.ones_like(matrix, dtype=bool)), matrix, np.nan)

    side = min(12.0, max(4.5, 0.12 * num_tokens + 3.0))
    figure = Figure(figsize=(side, side * 0.92), dpi=100)
    axes = figure.subplots()
    image = axes.imshow(masked, cmap="viridis", vmin=0.0, vmax=float(np.nanmax(masked)))
    axes.set_title(title, fontsize=10)
    axes.set_xlabel("Key token (attended to)", fontsize=9)
    axes.set_ylabel("Query token (attending)", fontsize=9)
    labels = [_tick_label(t) for t in token_texts]
    font_size = 7 if num_tokens <= 48 else 5
    axes.set_xticks(range(num_tokens), labels, rotation=90, fontsize=font_size)
    axes.set_yticks(range(num_tokens), labels, fontsize=font_size)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.02, label="attention weight")
    figure.tight_layout()
    return figure
