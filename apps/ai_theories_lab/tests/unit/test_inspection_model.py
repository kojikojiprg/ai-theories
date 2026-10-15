"""モデルの中を見る: 注意の重み(001)とトークンごとの確率(008)。

注意の重みは、各層の ``self_attn`` に付けた forward hook で取り出す。その値が、モデルの中で実際に
使われているものであることを、次の 3 つで確かめる。

1. hook を付けても付けなくても、順伝播の結果(logits)が bit 単位で一致する。
2. 取り出した注意の重みと Value から出力を計算し直すと、その層が実際に返した出力と一致する。
3. 取り出した注意の重みは、Query・Key(位置エンコーディングの適用後)から softmax で計算し直した
   値と一致する。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lab.core.artifacts import Resources
from lab.features.inspection.service import (
    analyze,
    capture_attention,
    describe_position,
    select_attention,
    tokenize,
)
from tests.inspection_checks import (
    FLOAT32_EPS,
    assert_hooks_do_not_change_logits,
    assert_weights_are_the_scaled_softmax_of_query_key,
    assert_weights_reproduce_layer_outputs,
)
from tests.stand_in import TRAINING_SENTENCES

TEXT = TRAINING_SENTENCES[0]


def _token_tensor(resources: Resources, text: str = TEXT) -> torch.Tensor:
    ids = resources.tokenizer.encode(text)
    return torch.tensor([ids], device=resources.device)


@pytest.mark.parametrize("model_key", ["main", "gqa", "sft-synthetic"])
def test_the_hooks_do_not_change_the_logits(resources: Resources, model_key: str) -> None:
    assert_hooks_do_not_change_logits(resources.models[model_key].model, _token_tensor(resources))


def test_the_hooks_are_removed_after_use(resources: Resources) -> None:
    model = resources.models["main"].model
    with capture_attention(model):
        assert all(len(block.self_attn._forward_hooks) == 1 for block in model.blocks)
    assert all(len(block.self_attn._forward_hooks) == 0 for block in model.blocks)


def test_the_hooks_are_removed_even_if_the_forward_fails(resources: Resources) -> None:
    model = resources.models["main"].model
    too_long = torch.zeros(
        (1, model.max_sequence_length + 1), dtype=torch.long, device=resources.device
    )
    with pytest.raises(ValueError), capture_attention(model):
        model(too_long)
    assert all(len(block.self_attn._forward_hooks) == 0 for block in model.blocks)


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_the_attention_weights_are_causal_and_normalized(
    resources: Resources, model_key: str
) -> None:
    loaded = resources.models[model_key]
    token_ids = resources.tokenizer.encode(TEXT)
    analysis = analyze(loaded, resources.device, token_ids)
    num_tokens = len(token_ids)
    assert analysis.attention.shape == (
        loaded.model.num_layers,
        loaded.model.num_heads,
        num_tokens,
        num_tokens,
    )
    assert analysis.attention.dtype == np.float32
    assert (analysis.attention >= 0).all()
    upper = np.triu(np.ones((num_tokens, num_tokens), dtype=bool), k=1)
    assert (analysis.attention[..., upper] == 0).all()
    # 和が 1 になる許容は、要素数 S と float32 の丸めの単位から導く(S 個の確率の和の丸めの上限)
    np.testing.assert_allclose(
        analysis.attention.sum(axis=-1), 1.0, rtol=0, atol=num_tokens * FLOAT32_EPS
    )


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_the_extracted_weights_reproduce_the_output_of_every_layer(
    resources: Resources, model_key: str
) -> None:
    """取り出した注意の重みと Value から出力を計算し直すと、その層が実際に返した出力と一致する。

    ``gqa`` は Key / Value のヘッドを Query のヘッドへ複製する(Grouped-Query Attention)ので、
    その複製も含めて確かめる。
    """
    assert_weights_reproduce_layer_outputs(
        resources.models[model_key].model, _token_tensor(resources)
    )


@pytest.mark.parametrize("model_key", ["main", "gqa"])
def test_the_extracted_weights_are_the_softmax_of_the_scaled_query_key_scores(
    resources: Resources, model_key: str
) -> None:
    """Query・Key(RoPE の適用後)から、softmax(Q K^T / sqrt(d_k)) を計算し直した値と一致する。"""
    assert_weights_are_the_scaled_softmax_of_query_key(
        resources.models[model_key].model, _token_tensor(resources)
    )


def test_the_analysis_returns_the_weights_that_the_model_used(resources: Resources) -> None:
    loaded = resources.models["main"]
    token_ids = resources.tokenizer.encode(TEXT)
    analysis = analyze(loaded, resources.device, token_ids)
    with torch.no_grad(), capture_attention(loaded.model) as captured:
        loaded.model(torch.tensor([token_ids], device=resources.device))
    for layer in range(loaded.model.num_layers):
        np.testing.assert_array_equal(analysis.attention[layer], captured[layer][0].cpu().numpy())


def test_the_analysis_does_not_change_the_training_flag(resources: Resources) -> None:
    loaded = resources.models["main"]
    analyze(loaded, resources.device, resources.tokenizer.encode(TEXT))
    assert not loaded.model.training


def test_the_probabilities_and_the_entropy_match_a_direct_softmax(resources: Resources) -> None:
    loaded = resources.models["main"]
    token_ids = resources.tokenizer.encode(TEXT)
    analysis = analyze(loaded, resources.device, token_ids)
    with torch.no_grad():
        logits = loaded.model(torch.tensor([token_ids], device=resources.device))[0].double()
    probabilities = torch.softmax(logits, dim=-1)
    entropy = torch.distributions.Categorical(logits=logits).entropy()
    num_tokens = len(token_ids)

    # float32 の softmax と、float64 の参照の差の許容: float32 の丸めの単位 x 語彙サイズの対数程度
    np.testing.assert_allclose(
        analysis.entropy,
        entropy.cpu().numpy(),
        rtol=0,
        atol=FLOAT32_EPS * 64 * math.log(analysis.vocabulary_size),
    )
    for position in range(num_tokens - 1):
        actual = token_ids[position + 1]
        expected = float(probabilities[position, actual])
        assert analysis.next_probability[position] == pytest.approx(expected, abs=1e-6)
        better = int((probabilities[position] > probabilities[position, actual]).sum())
        assert (
            abs(int(analysis.next_rank[position]) - (better + 1)) <= 1
        )  # 丸めで同点の近傍が入れ替わりうる
    assert math.isnan(analysis.next_probability[-1])
    assert analysis.next_rank[-1] == 0


def test_the_top_candidates_are_sorted_and_form_a_partial_distribution(
    resources: Resources,
) -> None:
    analysis = analyze(resources.models["main"], resources.device, resources.tokenizer.encode(TEXT))
    assert (np.diff(analysis.top_probabilities, axis=1) <= 0).all()
    assert (analysis.top_probabilities.sum(axis=1) <= 1.0 + 1e-5).all()
    assert analysis.top_ids.shape == analysis.top_probabilities.shape


def test_the_entropy_is_between_zero_and_the_uniform_entropy(resources: Resources) -> None:
    analysis = analyze(resources.models["main"], resources.device, resources.tokenizer.encode(TEXT))
    assert (analysis.entropy >= 0).all()
    assert (analysis.entropy <= analysis.max_entropy + 1e-4).all()
    assert analysis.max_entropy == pytest.approx(math.log(analysis.vocabulary_size))


def test_describe_position_for_a_middle_position(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    view = describe_position(analysis, resources.tokenizer, 3)
    assert [c.rank for c in view.candidates] == list(range(1, len(view.candidates) + 1))
    assert [c.probability for c in view.candidates] == sorted(
        (c.probability for c in view.candidates), reverse=True
    )
    assert view.actual_text == tokenization.pieces[4].text
    assert view.actual_probability == pytest.approx(float(analysis.next_probability[3]))
    assert view.actual_rank == int(analysis.next_rank[3])
    assert view.context_text == tokenization.pieces[3].text


def test_describe_position_for_the_last_position_has_no_actual_token(resources: Resources) -> None:
    tokenization = tokenize(resources.tokenizer, TEXT)
    analysis = analyze(resources.models["main"], resources.device, tokenization.token_ids)
    view = describe_position(analysis, resources.tokenizer, analysis.num_tokens - 1)
    assert view.actual_text is None
    assert view.actual_probability is None
    assert view.actual_rank is None


def test_a_single_token_input_can_be_analyzed(resources: Resources) -> None:
    analysis = analyze(
        resources.models["main"], resources.device, resources.tokenizer.encode("The")
    )
    assert analysis.num_tokens == 1
    assert analysis.attention.shape[-2:] == (1, 1)
    assert (analysis.attention == 1.0).all()


def test_select_attention_returns_a_head_or_the_mean_of_all_heads(resources: Resources) -> None:
    analysis = analyze(resources.models["main"], resources.device, resources.tokenizer.encode(TEXT))
    np.testing.assert_array_equal(select_attention(analysis, 1, 2), analysis.attention[1, 2])
    np.testing.assert_allclose(
        select_attention(analysis, 1, None), analysis.attention[1].mean(axis=0), rtol=0, atol=1e-7
    )


@pytest.mark.parametrize(("layer", "head"), [(-1, 0), (99, 0), (0, -1), (0, 99)])
def test_select_attention_rejects_out_of_range_indices(
    resources: Resources, layer: int, head: int
) -> None:
    analysis = analyze(resources.models["main"], resources.device, resources.tokenizer.encode(TEXT))
    with pytest.raises(ValueError):
        select_attention(analysis, layer, head)


def test_describe_position_rejects_an_out_of_range_position(resources: Resources) -> None:
    analysis = analyze(resources.models["main"], resources.device, resources.tokenizer.encode(TEXT))
    with pytest.raises(ValueError):
        describe_position(analysis, resources.tokenizer, analysis.num_tokens)


@pytest.mark.parametrize("length", [0, 129])
def test_analyze_rejects_an_empty_or_too_long_input(resources: Resources, length: int) -> None:
    with pytest.raises(ValueError):
        analyze(resources.models["main"], resources.device, [0] * length)
