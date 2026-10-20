"""``app.py`` を ``gradio_client`` から呼ぶ(画面の処理が、HTTP 越しに動くこと)。"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from types import ModuleType

import pytest
from gradio_client import Client

from lab.features.generation.service import (
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TEMPERATURE,
    METHOD_TOP_K,
    METHOD_TOP_P,
)
from lab.features.generation.ui import EXAMPLE_PROMPTS
from lab.features.instruction.service import instruction_choices
from src.data.instruction import format_prompt

PROMPT = EXAMPLE_PROMPTS[0]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def client(lab_app: ModuleType) -> Iterator[Client]:
    _, url, _ = lab_app.demo.launch(
        prevent_thread_lock=True, server_name="127.0.0.1", server_port=_free_port(), quiet=True
    )
    try:
        yield Client(url, verbose=False)
    finally:
        lab_app.demo.close()


def call_generate(
    client: Client, model_key: str, use_cache: bool, method: str, **overrides: float
) -> tuple[tuple[str, str, str], int]:
    """``/generate_text`` を呼び、最終結果と、途中経過が返った回数を返す。"""
    arguments = {
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "beam_size": 4,
        "length_penalty": 1.0,
        "max_new_tokens": 32,
        "seed": 0,
    } | overrides
    job = client.submit(
        PROMPT,
        model_key,
        use_cache,
        method,
        arguments["temperature"],
        arguments["top_k"],
        arguments["top_p"],
        arguments["beam_size"],
        arguments["length_penalty"],
        arguments["max_new_tokens"],
        arguments["seed"],
        api_name="/generate_text",
    )
    return job.result(), len(job.outputs())


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_the_continuation_follows_the_prompt(client: Client, model_key: str) -> None:
    (body, _, _), _ = call_generate(client, model_key, True, METHOD_GREEDY)
    assert body.startswith(PROMPT)
    assert len(body) > len(PROMPT)


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_greedy_returns_the_same_text_with_and_without_kv_cache(
    client: Client, model_key: str
) -> None:
    (cached, _, _), _ = call_generate(client, model_key, True, METHOD_GREEDY)
    (plain, _, _), _ = call_generate(client, model_key, False, METHOD_GREEDY)
    assert cached == plain


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_the_speed_display_is_returned(client: Client, model_key: str) -> None:
    (_, stats, _), _ = call_generate(client, model_key, True, METHOD_GREEDY)
    assert "毎秒" in stats
    assert "トークン" in stats


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_the_text_is_streamed_in_several_updates(client: Client, model_key: str) -> None:
    _, updates = call_generate(client, model_key, True, METHOD_GREEDY)
    assert updates > 1


@pytest.mark.parametrize("method", [METHOD_TEMPERATURE, METHOD_TOP_K, METHOD_TOP_P])
def test_sampling_methods_generate(client: Client, method: str) -> None:
    (body, stats, _), _ = call_generate(client, "main", True, method, seed=5)
    assert body.startswith(PROMPT)
    assert "毎秒" in stats


def test_beam_search_returns_the_candidates_and_the_speed(client: Client) -> None:
    (_, stats, details), _ = call_generate(client, "main", True, METHOD_BEAM, beam_size=3)
    assert "ビームサーチの候補" in details
    assert "毎秒" in stats


def _compare(client: Client) -> tuple[str, str, str, str, str, str]:
    return client.predict(
        PROMPT,
        "main",
        True,
        METHOD_GREEDY,
        METHOD_TOP_P,
        1.0,
        40,
        0.9,
        4,
        1.0,
        32,
        0,
        api_name="/compare_methods",
    )


def test_compare_shows_the_two_results_side_by_side(client: Client) -> None:
    text_a, _, _, text_b, _, _ = _compare(client)
    assert text_a.startswith(PROMPT)
    assert text_b.startswith(PROMPT)
    assert text_a != text_b


def test_compare_shows_the_speed_of_each(client: Client) -> None:
    _, stats_a, _, _, stats_b, _ = _compare(client)
    assert "毎秒" in stats_a
    assert "毎秒" in stats_b


def test_compare_method_a_matches_the_standalone_greedy_result(client: Client) -> None:
    text_a = _compare(client)[0]
    (standalone, _, _), _ = call_generate(client, "main", True, METHOD_GREEDY)
    assert standalone == text_a


def test_instruction_endpoint_shows_the_prompt_and_the_answer(client: Client) -> None:
    instruction = instruction_choices("reverse", "seen")[0]
    shown_prompt, report = client.predict(
        "reverse", instruction, "cat dog house tree", api_name="/answer_instruction"
    )
    assert shown_prompt == format_prompt(instruction, ("cat", "dog", "house", "tree"))
    assert "正解" in report
    assert "tree house dog cat" in report


# ---- モデルの中を見る(状態を持つ連続した呼び出し。同じ Client は同じセッション) ----


def test_inspection_runs_through_the_session_state(client: Client) -> None:
    text = "The history of the city began when"
    tokens_html, summary, *_ = client.predict(text, api_name="/inspect_tokenize")
    assert "トークン" in summary
    assert "hsla" in tokens_html
    client.predict(api_name="/inspect_model")  # 結果は State に入り、画面には返らない
    plot, _, probability_html, _, position_md = client.predict(api_name="/inspect_render")
    assert plot  # 図のファイルが返る
    assert "hsla(25,90%,50%" in probability_html
    assert "エントロピー" in position_md


# ---- バッチ推論 ----


def test_batch_inference_endpoint_shows_both_schedulers_and_the_figures(client: Client) -> None:
    summary, timeline, memory = client.predict(
        "main",
        8,
        6,
        0.5,
        "offline",
        30,
        4,
        "paged",
        16,
        4,
        False,
        0,
        api_name="/run_batch_inference",
    )
    for expected in ("static", "continuous", "スループット", "無駄な行の割合", "退避の回数"):
        assert expected in summary
    assert timeline and memory  # 図のファイルが返る


def test_batch_size_sweep_endpoint_returns_the_table_and_the_figure(client: Client) -> None:
    summary, plot = client.predict("main", 8, 5, 0.5, 0.8, 0, api_name="/measure_batch_sizes")
    assert "基準の到着率" in summary and "最大バッチサイズ" in summary
    assert plot
