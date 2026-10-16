"""本物のアーティファクトで、モデルの中を見る機能の一致の確認が成り立つこと。"""

from __future__ import annotations

import math

import pytest
import torch

from lab.core.artifacts import Resources
from lab.features.inspection.service import analyze, tokenize
from src.data.tokenizer import pretokenize
from tests.inspection_checks import (
    assert_hooks_do_not_change_logits,
    assert_weights_are_the_scaled_softmax_of_query_key,
    assert_weights_reproduce_layer_outputs,
)

SENTENCES = [
    "The history of the city began when the river town grew into a market and a port.",
    "Albert Einstein was born in Ulm, in the Kingdom of Württemberg.",
    "The unbelievable tokenization of uncommon words",
]


@pytest.mark.parametrize("text", SENTENCES)
def test_the_real_tokenizer_trace_matches_encode(hub_resources: Resources, text: str) -> None:
    tokenizer = hub_resources.tokenizer
    tokenization = tokenize(tokenizer, text)
    assert list(tokenization.token_ids) == tokenizer.encode(text)
    bpe = tokenizer.bpe_tokenizer
    assert [t.chunk for t in tokenization.chunks] == pretokenize(
        text, bpe.chunk_split_mode, bpe.max_chunk_bytes
    )


@pytest.mark.parametrize("model_key", ["main", "gqa", "sft-synthetic"])
@pytest.mark.parametrize("text", SENTENCES)
def test_the_extracted_weights_are_those_the_real_model_used(
    hub_resources: Resources, model_key: str, text: str
) -> None:
    model = hub_resources.models[model_key].model
    tokens = torch.tensor([hub_resources.tokenizer.encode(text)], device=hub_resources.device)
    assert_hooks_do_not_change_logits(model, tokens)
    assert_weights_reproduce_layer_outputs(model, tokens)
    assert_weights_are_the_scaled_softmax_of_query_key(model, tokens)


@pytest.mark.parametrize("text", SENTENCES)
def test_the_real_model_predicts_an_english_sentence_better_than_uniform(
    hub_resources: Resources, text: str
) -> None:
    """一様な予測の負の対数尤度は ln V。本物の事前学習済みモデルは、それより明確に低い
    (008 の学習では、1 トークンあたり 3.5〜4 nats 付近まで下がった)。"""
    loaded = hub_resources.models["main"]
    token_ids = hub_resources.tokenizer.encode(text)
    analysis = analyze(loaded, hub_resources.device, token_ids)
    mean_negative_log_likelihood = float(
        -torch.log(torch.tensor(analysis.next_probability[:-1])).mean()
    )
    assert mean_negative_log_likelihood < math.log(analysis.vocabulary_size) - 1.0
