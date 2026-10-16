"""「モデルの中を見る」の ``gr.State``: プロセス境界を越えた結果で、選び直しが動くこと。

ZeroGPU では、「解析する」の ``run_model`` が別のプロセスで動き、引数の ``Tokenization`` を
pickle で受け取り、結果の ``ModelAnalysis`` を pickle で返す。返った値が ``gr.State`` に入り、
層・ヘッド・位置・語を選び直すたびに、画面の側のプロセスがその値から表示を作り直す。
手元(ZeroGPU の外)では往復が起こらないので、往復を模擬して、往復の前と後で表示が同じであることを
確かめる。

選び直しは ``gr.State`` だけで完結し、モデルを動かさない(GPU の利用枠を使わない)。
モデルを動かすなら、その関数にも ``@gpu`` が要る。
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import gradio as gr
import pytest
from matplotlib.figure import Figure

from lab.core.artifacts import Resources
from lab.features.inspection.service import (
    ModelAnalysis,
    Tokenization,
    analyze,
    tokenize,
)
from tests.stand_in import TRAINING_SENTENCES
from tests.zero_gpu import GpuApp, assert_transferable

TEXT = TRAINING_SENTENCES[0]
RESELECTION_HANDLERS = {
    "render_attention",
    "render_position",
    "render_after_analysis",
    "render_merge_steps",
}


def display_of(value: Any) -> Any:
    """ハンドラの出力を、画面に出るものとして比べられる値にする(図は PNG の内容、部品は属性)。"""
    if isinstance(value, Figure):
        buffer = io.BytesIO()
        value.savefig(buffer, format="png")
        return ("figure", hashlib.sha256(buffer.getvalue()).hexdigest())
    if isinstance(value, gr.Dropdown):
        return ("dropdown", value.choices, value.value)
    if isinstance(value, tuple | list):
        return [display_of(item) for item in value]
    return value


@pytest.fixture(scope="module")
def states(resources: Resources, gpu_app: GpuApp) -> dict[str, Any]:
    """ZeroGPU と同じ順序の受け渡し: 引数を渡す → GPU の側で解析 → 戻り値を受け取る。

    ``before`` は往復していない値、``after`` は往復を経た値(``gr.State`` に入る値と、引数として
    届いた値)。
    """
    tokenization = tokenize(resources.tokenizer, TEXT)
    sent = assert_transferable(tokenization, "tokenization(引数)")
    produced = gpu_app.gpu_functions["run_model_on_gpu"].function(sent)
    received = assert_transferable(produced, "analysis(戻り値)")
    return {
        "before": (produced, tokenization),
        "after": (received, sent),
    }


def test_the_state_values_survive_the_boundary_unchanged(states: dict[str, Any]) -> None:
    (analysis_before, tokenization_before) = states["before"]
    (analysis_after, tokenization_after) = states["after"]
    assert analysis_after is not analysis_before  # 本当に別のオブジェクトに往復している
    assert tokenization_after is not tokenization_before
    assert isinstance(analysis_after, ModelAnalysis)
    assert isinstance(tokenization_after, Tokenization)


def test_the_analysis_state_holds_no_torch_tensor(states: dict[str, Any]) -> None:
    analysis, _ = states["after"]
    for name in (
        "attention",
        "entropy",
        "next_probability",
        "next_rank",
        "top_ids",
        "top_probabilities",
    ):
        assert type(getattr(analysis, name)).__module__ == "numpy", name


def test_reselecting_the_layer_and_head_shows_the_same_figure(
    gpu_app: GpuApp, states: dict[str, Any]
) -> None:
    render = gpu_app.handler("render_attention")
    before, after = states["before"], states["after"]
    shown = set()
    for layer in range(before[0].num_layers):
        for head in ("mean", "0", "5"):
            figure_before = display_of(render(*before, layer, head))
            figure_after = display_of(render(*after, layer, head))
            assert figure_before == figure_after, (layer, head)
            shown.add(figure_before[1])
    # 比べる値が、層・ヘッドの違いを区別できること(すべて同じ画像なら、比べる意味がない)
    assert len(shown) == before[0].num_layers * 3


def test_reselecting_the_position_shows_the_same_probabilities_and_candidates(
    gpu_app: GpuApp, states: dict[str, Any]
) -> None:
    render = gpu_app.handler("render_position")
    before, after = states["before"], states["after"]
    summaries = set()
    for position in range(before[0].num_tokens):
        shown_before = display_of(render(*before, position))
        shown_after = display_of(render(*after, position))
        assert shown_before == shown_after, position
        probability_html, rows, summary = shown_after
        assert probability_html
        assert rows
        summaries.add(summary)
    assert len(summaries) == before[0].num_tokens


def test_the_display_right_after_the_analysis_is_the_same(
    gpu_app: GpuApp, states: dict[str, Any]
) -> None:
    render = gpu_app.handler("render_after_analysis")
    before, after = states["before"], states["after"]
    for layer, head in [(0, "mean"), (1, "2"), (3, "7")]:
        assert display_of(render(*before, layer, head)) == display_of(render(*after, layer, head))


def test_reselecting_the_word_shows_the_same_merge_steps(
    gpu_app: GpuApp, states: dict[str, Any]
) -> None:
    render = gpu_app.handler("render_merge_steps")
    (_, tokenization_before) = states["before"]
    (_, tokenization_after) = states["after"]
    for chunk_index in range(len(tokenization_before.chunks)):
        shown = render(tokenization_before, chunk_index)
        assert shown == render(tokenization_after, chunk_index)
        assert shown[0][0] == "0(開始)"


def test_reselecting_without_a_state_shows_nothing(gpu_app: GpuApp) -> None:
    """新しい文を入れた直後は、前の解析が捨てられて ``None`` になる。選び直しは何も出さない。"""
    assert gpu_app.handler("render_attention")(None, None, 0, "mean") is None
    assert gpu_app.handler("render_position")(None, None, 0) == ("", [], "")
    assert gpu_app.handler("render_merge_steps")(None, 0) == []


def test_reselecting_never_runs_the_model(
    resources: Resources, gpu_app: GpuApp, states: dict[str, Any]
) -> None:
    """モデルの順伝播が呼ばれたらその場で失敗するようにして、選び直しをすべて呼ぶ。"""
    model = resources.models["main"].model

    def forbid(module: Any, args: Any) -> None:
        raise AssertionError("選び直しの処理でモデルが動いた")

    handle = model.register_forward_pre_hook(forbid)
    try:
        # この仕掛けが効くこと(解析はモデルを動かすので、失敗する)
        with pytest.raises(AssertionError, match="モデルが動いた"):
            analyze(resources.models["main"], resources.device, states["after"][1].token_ids)

        analysis, tokenization = states["after"]
        gpu_app.handler("render_attention")(analysis, tokenization, 1, "3")
        gpu_app.handler("render_position")(analysis, tokenization, 2)
        gpu_app.handler("render_after_analysis")(analysis, tokenization, 0, "mean")
        gpu_app.handler("render_merge_steps")(tokenization, 0)
    finally:
        handle.remove()


def test_no_reselection_handler_uses_the_gpu(gpu_app: GpuApp) -> None:
    assert RESELECTION_HANDLERS.isdisjoint(gpu_app.gpu_functions)


def test_every_non_gpu_handler_that_reads_a_state_is_covered_above(gpu_app: GpuApp) -> None:
    """``gr.State`` を入力に取るイベントの関数は、選び直しの 4 つと ``run_model`` だけ。

    ``run_model`` は入力の検査をして ``@gpu`` の関数(``run_model_on_gpu``)を呼ぶだけの関数で、
    モデルは動かさない。``@gpu`` の関数が ``gr.State`` を直接受け取るのは、``run_model_on_gpu``。
    新しい選び直しの処理を足したら、ここと上の試験に足す(足さないと失敗する)。
    """
    readers = gpu_app.handlers_reading_state()
    assert readers == RESELECTION_HANDLERS | {"run_model"}
    assert readers.isdisjoint(gpu_app.gpu_functions)  # イベントに結び付くのは、検査をする関数の側
