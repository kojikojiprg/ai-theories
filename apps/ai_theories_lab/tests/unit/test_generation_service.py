"""文章の生成の処理(008 デコーディング戦略・010 KV キャッシュ)。"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from lab.core.artifacts import Resources
from lab.features.generation.service import (
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TEMPERATURE,
    METHOD_TOP_K,
    METHOD_TOP_P,
    DecodingSettings,
    run_beam_search,
    select_next_token,
    stream_generation,
)
from lab.features.generation.ui import (
    DEGENERATION_EXAMPLE_PROMPT,
    EXAMPLE_PROMPTS,
    format_generation_stats,
)
from src.generation import beam_search
from tests.helpers import generate_ids, reference_generate_with_filter

PROMPTS = [*EXAMPLE_PROMPTS, DEGENERATION_EXAMPLE_PROMPT]
MODEL_AND_PROMPT = [(model, prompt) for model in ("main", "gqa") for prompt in PROMPTS]


@pytest.mark.parametrize(("model_key", "prompt"), MODEL_AND_PROMPT)
def test_greedy_is_identical_with_and_without_kv_cache(
    resources: Resources, model_key: str, prompt: str
) -> None:
    with_cache = generate_ids(resources, model_key, prompt, DecodingSettings(METHOD_GREEDY, 48))
    without_cache = generate_ids(
        resources, model_key, prompt, DecodingSettings(METHOD_GREEDY, 48, use_cache=False)
    )
    assert with_cache == without_cache


@pytest.mark.parametrize(("model_key", "prompt"), MODEL_AND_PROMPT)
def test_greedy_matches_model_generate(resources: Resources, model_key: str, prompt: str) -> None:
    loaded = resources.models[model_key]
    prompt_ids = resources.tokenizer.encode(prompt)
    tensor = torch.tensor([prompt_ids], device=resources.device)
    reference_cache = loaded.model.generate(tensor, 48, temperature=0.0, use_cache=True)
    reference_plain = loaded.model.generate(tensor, 48, temperature=0.0, use_cache=False)
    with_cache = generate_ids(resources, model_key, prompt, DecodingSettings(METHOD_GREEDY, 48))
    without_cache = generate_ids(
        resources, model_key, prompt, DecodingSettings(METHOD_GREEDY, 48, use_cache=False)
    )
    assert with_cache == reference_cache[0, len(prompt_ids) :].tolist()
    assert without_cache == reference_plain[0, len(prompt_ids) :].tolist()


@pytest.mark.parametrize(("model_key", "prompt"), MODEL_AND_PROMPT)
def test_greedy_matches_beam_search_with_beam_size_one(
    resources: Resources, model_key: str, prompt: str
) -> None:
    greedy = generate_ids(resources, model_key, prompt, DecodingSettings(METHOD_GREEDY, 48))
    beam_one = generate_ids(
        resources,
        model_key,
        prompt,
        DecodingSettings(METHOD_BEAM, 48, beam_size=1, length_penalty=1.0),
    )
    assert greedy == beam_one


@pytest.mark.parametrize(
    ("method", "settings_kwargs", "reference_kwargs"),
    [
        (METHOD_TEMPERATURE, {"temperature": 0.8}, {"temperature": 0.8}),
        (METHOD_TOP_K, {"temperature": 1.0, "top_k": 40}, {"temperature": 1.0, "top_k": 40}),
        (METHOD_TOP_P, {"temperature": 1.0, "top_p": 0.9}, {"temperature": 1.0, "top_p": 0.9}),
    ],
)
def test_sampling_matches_008_generate_with_filter(
    resources: Resources, method: str, settings_kwargs: dict, reference_kwargs: dict
) -> None:
    """同じシードで、008 のノートブックの ``generate_with_filter()`` と同じトークンになる。"""
    prompt = EXAMPLE_PROMPTS[0]
    prompt_ids = resources.tokenizer.encode(prompt)
    tensor = torch.tensor([prompt_ids], device=resources.device)
    settings = DecodingSettings(method, 40, seed=7, use_cache=False, **settings_kwargs)
    mine = generate_ids(resources, "main", prompt, settings)
    reference = reference_generate_with_filter(
        resources.models["main"].model, tensor, 40, seed=7, **reference_kwargs
    )[0, len(prompt_ids) :].tolist()
    assert mine == reference
    # 参考: KV キャッシュの有無でトークンが一致した割合
    # (浮動小数点の丸めで食い違いうるので、判定には使わない)
    cached = generate_ids(
        resources,
        "main",
        prompt,
        DecodingSettings(method, 40, seed=7, use_cache=True, **settings_kwargs),
    )
    print(f"{method}: KV キャッシュの有無で一致した割合 {np.mean(np.array(mine) == cached):.3f}")


def test_same_seed_gives_the_same_text(resources: Resources) -> None:
    settings = DecodingSettings(METHOD_TOP_P, 32, temperature=1.5, top_p=0.95, seed=3)
    prompt = EXAMPLE_PROMPTS[0]
    assert generate_ids(resources, "main", prompt, settings) == generate_ids(
        resources, "main", prompt, settings
    )


def test_different_seed_gives_a_different_text(resources: Resources) -> None:
    prompt = EXAMPLE_PROMPTS[0]
    first = generate_ids(
        resources,
        "main",
        prompt,
        DecodingSettings(METHOD_TOP_P, 32, temperature=1.5, top_p=0.95, seed=3),
    )
    other = generate_ids(
        resources,
        "main",
        prompt,
        DecodingSettings(METHOD_TOP_P, 32, temperature=1.5, top_p=0.95, seed=4),
    )
    assert first != other


def test_top_k_of_one_is_greedy(resources: Resources) -> None:
    prompt = EXAMPLE_PROMPTS[0]
    greedy = generate_ids(resources, "main", prompt, DecodingSettings(METHOD_GREEDY, 32))
    top_k_one = generate_ids(
        resources, "main", prompt, DecodingSettings(METHOD_TOP_K, 32, top_k=1, seed=9)
    )
    assert top_k_one == greedy


def test_tiny_top_p_is_greedy(resources: Resources) -> None:
    prompt = EXAMPLE_PROMPTS[0]
    greedy = generate_ids(resources, "main", prompt, DecodingSettings(METHOD_GREEDY, 32))
    tiny_p = generate_ids(
        resources, "main", prompt, DecodingSettings(METHOD_TOP_P, 32, top_p=1e-6, seed=9)
    )
    assert tiny_p == greedy


def test_select_next_token_greedy_is_argmax(resources: Resources) -> None:
    logits = torch.tensor([[0.1, 3.0, 0.2, 2.9]], device=resources.device)
    chosen = {
        int(select_next_token(logits, DecodingSettings(METHOD_GREEDY, 1), None).item())
        for _ in range(5)
    }
    assert chosen == {1}


def test_beam_candidates_are_sorted_by_score(resources: Resources) -> None:
    loaded = resources.models["main"]
    prompt_ids = resources.tokenizer.encode(EXAMPLE_PROMPTS[0])
    outcome = run_beam_search(
        loaded.model,
        prompt_ids,
        DecodingSettings(METHOD_BEAM, 24, beam_size=3),
        resources.device,
    )
    scores = [candidate.score for candidate in outcome.candidates]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)


def test_best_beam_matches_src_beam_search(resources: Resources) -> None:
    loaded = resources.models["main"]
    prompt_ids = resources.tokenizer.encode(EXAMPLE_PROMPTS[0])
    outcome = run_beam_search(
        loaded.model,
        prompt_ids,
        DecodingSettings(METHOD_BEAM, 24, beam_size=3),
        resources.device,
    )
    reference = beam_search(
        loaded.model, torch.tensor([prompt_ids], device=resources.device), 3, 24, 1.0
    )
    assert (
        list(outcome.candidates[0].new_token_ids) == reference[0][0][0, len(prompt_ids) :].tolist()
    )


# ---- 速さの表示 ----


def test_progress_is_returned_one_token_at_a_time(resources: Resources) -> None:
    prompt_ids = resources.tokenizer.encode(EXAMPLE_PROMPTS[0])
    progress = list(
        stream_generation(
            resources.models["main"].model,
            prompt_ids,
            DecodingSettings(METHOD_GREEDY, 24),
            resources.device,
        )
    )
    assert [len(p.new_token_ids) for p in progress] == list(range(1, 25))
    assert [p.finished for p in progress] == [False] * 23 + [True]


def test_compute_time_excludes_the_wait_for_the_consumer(resources: Resources) -> None:
    """画面の更新に時間がかかっても、計算時間には含まれない。"""
    prompt_ids = resources.tokenizer.encode(EXAMPLE_PROMPTS[0])
    sleep_per_step = 0.02
    started = time.perf_counter()
    last = None
    count = 0
    for progress in stream_generation(
        resources.models["main"].model,
        prompt_ids,
        DecodingSettings(METHOD_GREEDY, 24),
        resources.device,
    ):
        time.sleep(sleep_per_step)
        last = progress
        count += 1
    wall = time.perf_counter() - started
    assert last is not None
    assert 0 < last.compute_seconds <= wall - 0.9 * sleep_per_step * count


def test_stats_line_shows_tokens_time_and_rate() -> None:
    line = format_generation_stats("生成結果", 24, 0.111, "cpu")
    assert "生成したトークン数 24" in line
    assert "時間" in line
    assert "毎秒" in line


# ---- 入力の検証 ----

LIMIT = 256


@pytest.mark.parametrize(
    ("settings", "prompt_length", "is_rejected"),
    [
        pytest.param(DecodingSettings(METHOD_GREEDY, 8), 0, True, id="empty prompt"),
        pytest.param(DecodingSettings(METHOD_GREEDY, 8), LIMIT + 1, True, id="prompt too long"),
        pytest.param(
            DecodingSettings(METHOD_GREEDY, 200, use_cache=True), 100, True, id="cache too long"
        ),
        pytest.param(
            DecodingSettings(METHOD_GREEDY, 200, use_cache=False), 100, False, id="no cache ok"
        ),
        pytest.param(
            DecodingSettings(METHOD_BEAM, 200, use_cache=True), 100, False, id="beam ignores cache"
        ),
        pytest.param(DecodingSettings(METHOD_TOP_P, 8, top_p=0.0), 4, True, id="top-p zero"),
        pytest.param(DecodingSettings(METHOD_TOP_K, 8, top_k=0), 4, True, id="top-k zero"),
        pytest.param(
            DecodingSettings(METHOD_TEMPERATURE, 8, temperature=0.0), 4, True, id="temperature zero"
        ),
        pytest.param(DecodingSettings(METHOD_TOP_P, 8, seed=-1), 4, True, id="negative seed"),
        pytest.param(DecodingSettings("unknown", 8), 4, True, id="unknown method"),
    ],
)
def test_settings_validation(
    settings: DecodingSettings, prompt_length: int, is_rejected: bool
) -> None:
    if is_rejected:
        with pytest.raises(ValueError):
            settings.validate(prompt_length, LIMIT)
    else:
        settings.validate(prompt_length, LIMIT)
