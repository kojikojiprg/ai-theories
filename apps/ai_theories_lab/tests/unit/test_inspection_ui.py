"""モデルの中を見る: 画面の表示用の整形(HTML・表・図)。"""

from __future__ import annotations

import re

import numpy as np
import pytest
from matplotlib.figure import Figure

from lab.core.artifacts import Resources
from lab.features.inspection.figures import attention_figure
from lab.features.inspection.service import analyze, describe_position, tokenize
from lab.features.inspection.ui import (
    candidate_rows,
    chunk_choices,
    default_chunk,
    head_choices,
    layer_choices,
    merge_rows,
    position_choices,
    position_summary,
    probability_html,
    tokenization_summary,
    tokens_html,
)
from tests.stand_in import TRAINING_SENTENCES

TEXT = TRAINING_SENTENCES[0]
SELECTED_BORDER = "border:2px solid hsl(260,80%,55%)"


def test_tokens_html_escapes_the_input(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, '<script>alert(1)</script> & "q"')
    rendered = tokens_html(tokenization)
    assert "<script>" not in rendered
    assert "&lt;" in rendered
    assert "&amp;" in rendered


def test_tokens_html_has_one_chip_per_token(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    assert tokens_html(tokenization).count("<span title=") == len(tokenization.pieces)
    for piece in tokenization.pieces:
        assert f">{piece.token_id}</span>" in tokens_html(tokenization)


def test_neighboring_tokens_have_different_colors(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    hues = re.findall(r"background:hsla\((\d+),", tokens_html(tokenization))
    assert all(a != b for a, b in zip(hues, hues[1:], strict=False))


def test_summary_counts_characters_bytes_and_tokens(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, "café au lait")
    summary = tokenization_summary(tokenization)
    assert "12 文字(13 バイト)" in summary
    assert f"{len(tokenization.pieces)} トークン" in summary


def test_merge_rows_start_from_bytes_and_end_at_the_tokens(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, "The unbelievable results")
    for trace in tokenization.chunks:
        rows = merge_rows(trace)
        assert rows[0][0] == "0(開始)"
        assert len(rows) == len(trace.steps) + 1 or not trace.steps
        final_tokens = rows[-1][3].split(" | ")
        assert len(final_tokens) == len(trace.final_symbols)


def test_the_default_chunk_is_the_one_with_the_most_merge_steps(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, "a unbelievable the")
    chosen = tokenization.chunks[default_chunk(tokenization)]
    assert len(chosen.steps) == max(len(c.steps) for c in tokenization.chunks)
    values = [value for _, value in chunk_choices(tokenization)]
    assert values == list(range(len(tokenization.chunks)))


def test_probability_html_has_one_chip_per_token_and_marks_the_selected_one(
    resources: Resources,
) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    rendered = probability_html(analysis, tokenization, selected=3)
    assert rendered.count("<span title=") == len(tokenization.pieces)
    assert rendered.count(SELECTED_BORDER) == 1
    assert probability_html(analysis, tokenization, selected=None).count(SELECTED_BORDER) == 0
    assert "p=—" in rendered  # 最後の位置には、実際の次のトークンがない


def test_a_less_probable_next_token_gets_a_darker_background(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    rendered = probability_html(analysis, tokenization)
    alphas = [float(a) for a in re.findall(r"background:hsla\(25,90%,50%,([0-9.]+)\)", rendered)]
    probabilities = analysis.next_probability[:-1]
    order_by_probability = np.argsort(-probabilities)
    # 確率が最大の位置の背景は、最小の位置の背景より薄い
    assert alphas[int(order_by_probability[0])] < alphas[int(order_by_probability[-1])]


def test_position_choices_and_summary(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    values = [value for _, value in position_choices(tokenization)]
    assert values == list(range(analysis.num_tokens))
    middle = describe_position(analysis, resources.tokenizer, 2)
    assert "順位は" in position_summary(middle, analysis.max_entropy)
    last = describe_position(analysis, resources.tokenizer, analysis.num_tokens - 1)
    assert "実際の次のトークンはありません" in position_summary(last, analysis.max_entropy)
    assert len(candidate_rows(middle)) == len(middle.candidates)
    assert candidate_rows(middle)[0][0] == "1"


def test_layer_and_head_choices() -> None:
    assert [value for _, value in layer_choices(4)] == [0, 1, 2, 3]
    assert [value for _, value in head_choices(3)] == ["mean", "0", "1", "2"]


def test_attention_figure_draws_the_lower_triangle_of_the_matrix() -> None:
    matrix = np.tril(np.full((4, 4), 0.25, dtype=np.float32))
    figure = attention_figure(matrix, ["a", "b", "c", "$d"], "Layer 0, head 1")
    assert isinstance(figure, Figure)
    (axes,) = [ax for ax in figure.axes if ax.images]
    drawn = np.ma.filled(axes.images[0].get_array(), np.nan)
    np.testing.assert_array_equal(np.tril(drawn), np.tril(matrix))
    assert np.isnan(drawn[np.triu_indices(4, k=1)]).all()  # 未来のトークンは塗らない
    assert [t.get_text() for t in axes.get_xticklabels()][-1] == r"\$d"
    assert axes.get_title() == "Layer 0, head 1"


def test_attention_figure_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError):
        attention_figure(np.zeros((3, 3), dtype=np.float32), ["a", "b"], "t")
