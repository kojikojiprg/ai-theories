"""ZeroGPU の受け渡しを模擬したアプリを、``gradio_client`` から呼ぶ(Gradio の実際の経路)。

すべての ``@gpu`` の関数について、引数と戻り値(生成器なら yield の 1 つずつ)を ZeroGPU と同じ
経路(``ForkingPickler``)で往復させ、例外は ``gr.Error`` だけがメッセージごと届くようにした
アプリを組み立てる(``tests.zero_gpu.build_app_simulating_zero_gpu``)。そのアプリを、境界を
模擬しないアプリ(ZeroGPU の外と同じ)と同じ手順で呼び、**決まった値になる出力が一致すること**を
確かめる。

速さの表示は計測値なので、比べない(含まれることだけ確かめる)。実際の ZeroGPU(別のプロセスと本物の
GPU)を使うものではないので、手元で確かめられるのは、受け渡す値の型と、Gradio の ``gr.State`` が
往復した値を扱えることまで。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from gradio_client import Client

from lab.core.artifacts import Resources
from lab.features.generation.service import METHOD_BEAM, METHOD_GREEDY, METHOD_TOP_P
from lab.features.generation.ui import EXAMPLE_PROMPTS
from lab.features.instruction.service import instruction_choices
from lab.ui.layout import build_app
from tests.helpers import free_port
from tests.stand_in import TRAINING_SENTENCES
from tests.zero_gpu import (
    SIMULATED_CALLS,
    build_app_simulating_zero_gpu,
    feature_ui_modules,
)

PROMPT = EXAMPLE_PROMPTS[0]
TEXT = TRAINING_SENTENCES[0]


def _launch(demo: Any) -> Client:
    _, url, _ = demo.launch(
        prevent_thread_lock=True, server_name="127.0.0.1", server_port=free_port(), quiet=True
    )
    return Client(url, verbose=False)


@pytest.fixture(scope="module")
def simulated(resources: Resources) -> Iterator[Client]:
    """すべての ``@gpu`` の関数が、境界を越える受け渡しを通るアプリ。"""
    demo = build_app_simulating_zero_gpu(resources)
    client = _launch(demo)
    try:
        yield client
    finally:
        demo.close()


@pytest.fixture(scope="module")
def direct(resources: Resources) -> Iterator[Client]:
    """境界を模擬しないアプリ(ZeroGPU の外で ``@spaces.GPU`` が何もしない状態と同じ)。"""
    demo = build_app(resources, [module.PAGE for module in feature_ui_modules()])
    client = _launch(demo)
    try:
        yield client
    finally:
        demo.close()


def _generate(client: Client, method: str, **overrides: float) -> tuple[tuple[str, str, str], int]:
    arguments = {
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "beam_size": 2,
        "length_penalty": 1.0,
        "max_new_tokens": 8,
        "seed": 3,
    } | overrides
    job = client.submit(
        PROMPT, "main", True, method, *arguments.values(), api_name="/generate_text"
    )
    return job.result(), len(job.outputs())


@pytest.mark.parametrize("method", [METHOD_GREEDY, METHOD_TOP_P, METHOD_BEAM])
def test_generation_streams_the_same_text_through_the_simulated_boundary(
    simulated: Client, direct: Client, method: str
) -> None:
    SIMULATED_CALLS.clear()
    (text, stats, details), updates = _generate(simulated, method)
    (expected_text, _, expected_details), _ = _generate(direct, method)
    assert SIMULATED_CALLS == ["generate_text"]
    assert text.startswith(PROMPT)
    assert "毎秒" in stats
    assert updates >= 2  # 逐次表示(ビームサーチは、計算中と最後の 2 回)
    # 貪欲法・ビームサーチは決まった結果。サンプリングも、シードが同じなら同じ結果
    assert text == expected_text
    assert details == expected_details


def test_compare_gives_the_same_texts_through_the_simulated_boundary(
    simulated: Client, direct: Client
) -> None:
    settings = (
        1.0,
        40,
        0.9,
        2,
        1.0,
        8,
        3,
    )  # temperature, k, p, ビーム数, 長さペナルティ, 長さ, シード

    def compare(client: Client) -> tuple[str, ...]:
        return client.predict(
            PROMPT,
            "main",
            True,
            METHOD_GREEDY,
            METHOD_TOP_P,
            *settings,
            api_name="/compare_methods",
        )

    SIMULATED_CALLS.clear()
    text_a, stats_a, _, text_b, stats_b, _ = compare(simulated)
    expected_a, _, _, expected_b, _, _ = compare(direct)
    assert SIMULATED_CALLS == ["compare_methods"]
    assert (text_a, text_b) == (expected_a, expected_b)
    assert "毎秒" in stats_a
    assert "毎秒" in stats_b


def test_the_instruction_answer_is_the_same_through_the_simulated_boundary(
    simulated: Client, direct: Client
) -> None:
    instruction = instruction_choices("reverse", "seen")[0]  # 画面の初期の選択肢

    def answer(client: Client) -> tuple[str, str]:
        return client.predict(
            "reverse", instruction, "cat dog house tree", api_name="/answer_instruction"
        )

    SIMULATED_CALLS.clear()
    assert answer(simulated) == answer(direct)
    assert SIMULATED_CALLS == ["answer_instruction"]


def _picture(plot: dict[str, str]) -> str:
    """``gr.Plot`` の matplotlib の図は、クライアントには画像のデータ(base64)として届く。"""
    assert plot["type"] == "matplotlib"
    return plot["plot"]


def _analyze_and_render(client: Client) -> tuple[Any, ...]:
    """「解析する」と同じ順序の 3 つの呼び出し(トークン分割 → モデル → 表示)。"""
    client.predict(TEXT, api_name="/inspect_tokenize")
    client.predict(api_name="/inspect_model")
    return client.predict(0, "mean", api_name="/inspect_render")


def test_inspection_state_survives_the_simulated_boundary(
    simulated: Client, direct: Client
) -> None:
    SIMULATED_CALLS.clear()
    plot, _, probability_html, rows, summary = _analyze_and_render(simulated)
    expected_plot, _, expected_html, expected_rows, expected_summary = _analyze_and_render(direct)
    assert SIMULATED_CALLS == ["run_model"]
    assert (probability_html, rows, summary) == (expected_html, expected_rows, expected_summary)
    assert _picture(plot) == _picture(expected_plot)


def test_reselecting_after_the_analysis_does_not_call_the_gpu_function(
    simulated: Client, direct: Client
) -> None:
    """解析は 1 回だけ。層・ヘッドを選び直しても、GPU の関数は呼ばれず、同じ State から表示する。"""
    _analyze_and_render(simulated)
    _analyze_and_render(direct)
    SIMULATED_CALLS.clear()
    shown = set()
    for layer, head in [(1, "3"), (2, "mean"), (3, "7")]:
        plot, _, probability_html, rows, summary = simulated.predict(
            layer, head, api_name="/inspect_render"
        )
        expected_plot, _, expected_html, expected_rows, expected_summary = direct.predict(
            layer, head, api_name="/inspect_render"
        )
        assert (probability_html, rows, summary) == (expected_html, expected_rows, expected_summary)
        assert _picture(plot) == _picture(expected_plot)
        shown.add(_picture(plot))
    assert SIMULATED_CALLS == []
    assert len(shown) == 3


def test_an_error_message_reaches_the_client_through_the_simulated_boundary(
    simulated: Client,
) -> None:
    """境界を越えて届くのは ``gr.Error`` のメッセージ。新しいセッションには、解析の結果がない。"""
    fresh = Client(simulated.src, verbose=False)
    with pytest.raises(Exception, match="先にトークン分割"):
        fresh.predict(api_name="/inspect_model")
    with pytest.raises(Exception, match="空です"):
        simulated.predict(
            "", "main", True, METHOD_GREEDY, 1.0, 40, 0.9, 2, 1.0, 8, 3, api_name="/generate_text"
        )
