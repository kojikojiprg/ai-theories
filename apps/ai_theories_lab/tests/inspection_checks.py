"""注意の重みの取り出し(forward hook)が、モデルの中で実際に使われた値であることの確認。

代替のアーティファクト(``tests/unit/``)と本物のアーティファクト(``tests/hub/``)の両方で、
同じ確認を行うために、関数にまとめてある。確認は 3 つ。

1. hook を付けても付けなくても、順伝播の結果(logits)が bit 単位で一致する。
2. 取り出した注意の重みと Value から出力を計算し直すと、その層が実際に返した出力と一致する。
3. 取り出した注意の重みは、Query・Key(位置エンコーディングの適用後)から softmax で計算し直した
   値と一致する。

許容の誤差は、固定の値ではなく、float32 の丸めの単位と、足し合わせる要素数・値の大きさから導く。
"""

from __future__ import annotations

import math

import torch

from lab.features.inspection.service import capture_attention
from src.models.gpt import GPTLanguageModel

FLOAT32_EPS = float(torch.finfo(torch.float32).eps)


def assert_hooks_do_not_change_logits(model: GPTLanguageModel, tokens: torch.Tensor) -> None:
    with torch.no_grad():
        plain = model(tokens)
        with capture_attention(model):
            hooked = model(tokens)
    assert torch.equal(plain, hooked)


def _record_attention_inputs(
    model: GPTLanguageModel, tokens: torch.Tensor
) -> tuple[dict[int, tuple[torch.Tensor, ...]], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """各層の ``self_attn`` の入力(Query・Key・Value の入力)と、返り値を集める。

    返り値は ``(出力, 注意の重み)``。
    """
    inputs: dict[int, tuple[torch.Tensor, ...]] = {}
    outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def make_hook(layer: int):
        def hook(module, args, output) -> None:
            inputs[layer] = args[:3]
            outputs[layer] = output

        return hook

    handles = [
        block.self_attn.register_forward_hook(make_hook(i)) for i, block in enumerate(model.blocks)
    ]
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    assert len(outputs) == model.num_layers
    return inputs, outputs


def assert_weights_reproduce_layer_outputs(model: GPTLanguageModel, tokens: torch.Tensor) -> None:
    """注意の重みと Value から出力を計算し直して、その層が返した出力と比べる。

    Key / Value のヘッドを Query のヘッドへ複製する Grouped-Query Attention も含めて確かめる。
    """
    inputs, outputs = _record_attention_inputs(model, tokens)
    for layer, block in enumerate(model.blocks):
        attention = block.self_attn
        value_input = inputs[layer][2]  # 引数の 3 番目が Value の入力
        actual_output, weights = outputs[layer]
        batch, seq, _ = value_input.shape
        with torch.no_grad():
            value = attention.w_v(value_input).view(batch, seq, attention.num_key_value_heads, -1)
            value = value.transpose(1, 2).repeat_interleave(
                attention.num_heads // attention.num_key_value_heads, dim=1
            )  # (B, h, S, d_v)
            heads = torch.matmul(weights, value)
            recomputed = attention.w_o(heads.transpose(1, 2).reshape(batch, seq, -1))
        assert recomputed.dtype == torch.float32 and actual_output.dtype == torch.float32
        # 許容: float32 の丸めの単位 x 足し合わせる要素数(S と d_model)x 出力の大きさ
        tolerance = FLOAT32_EPS * (seq + attention.d_model) * float(actual_output.abs().max())
        torch.testing.assert_close(recomputed, actual_output, rtol=0, atol=tolerance)


def assert_weights_are_the_scaled_softmax_of_query_key(
    model: GPTLanguageModel, tokens: torch.Tensor
) -> None:
    """Query・Key(RoPE の適用後)から softmax(Q K^T / sqrt(d_k)) を計算し直して、重みと比べる。"""
    inputs, outputs = _record_attention_inputs(model, tokens)
    for layer, block in enumerate(model.blocks):
        attention = block.self_attn
        hidden = inputs[layer][0]
        weights = outputs[layer][1]
        batch, seq, _ = hidden.shape
        with torch.no_grad():
            q = attention.w_q(hidden).view(batch, seq, attention.num_heads, -1).transpose(1, 2)
            k = attention.w_k(hidden).view(batch, seq, attention.num_key_value_heads, -1)
            k = k.transpose(1, 2)
            q, k = attention.positional_transform.apply(q, k, None)
            k = k.repeat_interleave(attention.num_heads // attention.num_key_value_heads, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attention.d_k)
            causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=hidden.device))
            expected = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        # 許容: float32 の丸めの単位 x 足し合わせる要素数(d_k と S)(確率は 1 以下)
        tolerance = FLOAT32_EPS * (attention.d_k + seq)
        torch.testing.assert_close(weights, expected, rtol=0, atol=tolerance)
