"""試験で共有する関数。"""

from __future__ import annotations

import torch

from lab.core.artifacts import Resources
from lab.features.generation.service import (
    METHOD_BEAM,
    DecodingSettings,
    run_beam_search,
    stream_generation,
)
from src.generation import top_k_filter, top_p_filter
from src.models.gpt import GPTLanguageModel


@torch.no_grad()
def reference_generate_with_filter(
    model: GPTLanguageModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """008 のノートブックの ``generate_with_filter()`` と同じ手順の参照実装(KV キャッシュなし)。"""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=prompt_ids.device)
        generator.manual_seed(seed)
    token_ids = prompt_ids
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.max_sequence_length :]
        next_logits = model(context)[:, -1, :]
        if top_k is not None:
            next_logits = top_k_filter(next_logits, top_k)
        if top_p is not None:
            next_logits = top_p_filter(next_logits, top_p)
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids


def generate_ids(
    resources: Resources, model_key: str, prompt: str, settings: DecodingSettings
) -> list[int]:
    """アプリと同じ関数で生成し、生成したトークン ID(書き出しを除く)を返す。"""
    loaded = resources.models[model_key]
    prompt_ids = resources.tokenizer.encode(prompt)
    if settings.method == METHOD_BEAM:
        outcome = run_beam_search(loaded.model, prompt_ids, settings, resources.device)
        return list(outcome.candidates[0].new_token_ids)
    progress = list(stream_generation(loaded.model, prompt_ids, settings, resources.device))
    return list(progress[-1].new_token_ids)


def repeated_ngram_rate(token_ids: list[int], n: int = 4) -> float:
    """n-gram のうち、同じ n-gram が既に出ているものの割合(008 の重複率と同じ考え方)。"""
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)
