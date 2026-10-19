"""1 要求ずつの生成(正しさの基準)と、数値の許容誤差。

``src/serving/engine.py`` を通した生成(PagedAttention・continuous batching・接頭辞の共有・退避)が、
010 の KV キャッシュによる 1 要求ずつの生成と一致することを確かめる基準を与える。

**許容誤差の導出**: 同じ計算でも、行列積の形(バッチの形)が変わると、浮動小数点の丸めの順序が変わり、
結果は丸めの単位の何倍かだけずれる。許容誤差を固定の絶対値にせず、**実際に計算する型の丸めの単位**
(機械イプシロン ε)と、**比べる値の大きさ**(logits の絶対値の最大)から導く。誤差は、
直列に並ぶ行列積の段数 D(層ごとに射影 4 + 注意 2 + 順伝播ネットワーク 3 + 正規化 2 で
約 12 段、層数 L をかける)と、各段の
内積の長さ K の平方根(丸め誤差が独立にランダムに積み上がるときの増え方)に比例して大きくなりうる。
したがって ``許容誤差 = (D * sqrt(K_max)) * ε * max(1, 最大|logit|)`` とする。

導入: 027。

記号 / Notation:
    ε     : 機械イプシロン(``torch.finfo(dtype).eps``)
    L     : 層数
    K_max : 内積の長さの最大(``max(d_model, d_ff, 最大の文脈長)``)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from src.generation.cache import KeyValueCache
from src.models.gpt import GPTLanguageModel

MATMUL_STAGES_PER_LAYER = 12


def logits_tolerance(model: GPTLanguageModel, reference_logits: np.ndarray) -> float:
    """logits の差の許容誤差(上の導出)。

    Args:
        model: 実行したモデル(層数・次元・計算する型を読む)。
        reference_logits: 比べる基準の logits(大きさの尺度に使う)。

    Raises:
        TypeError: 基準の logits の型が、モデルの計算する型と違うとき(違う型で比べると、
            許容誤差の導出が
            成り立たないため)。
    """
    parameter = next(model.parameters())
    dtype = parameter.dtype
    if reference_logits.dtype != torch.empty((), dtype=dtype).numpy().dtype:
        raise TypeError(
            f"基準の logits の型 {reference_logits.dtype} がモデルの計算する型 {dtype} と違う"
        )
    # 内積の長さの最大: 順伝播ネットワークの線形層の入力次元、モデルの次元、
    # 注意機構が足し合わせる Key の数
    linear_inputs = [
        module.in_features
        for block in model.blocks
        for module in block.feed_forward.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    reduction_max = max(
        model.blocks[0].self_attn.d_model, *linear_inputs, model.max_sequence_length
    )
    stages = MATMUL_STAGES_PER_LAYER * model.num_layers
    scale = max(1.0, float(np.abs(reference_logits).max()))
    return stages * math.sqrt(reduction_max) * float(torch.finfo(dtype).eps) * scale


@torch.no_grad()
def reference_generate(
    model: GPTLanguageModel, prompt_token_ids: Sequence[int], output_length: int
) -> tuple[list[int], np.ndarray]:
    """010 の KV キャッシュによる 1 要求ずつの貪欲法の生成。

    ``GPTLanguageModel.generate(use_cache=True, temperature=0.0)`` と同じ手順
    (prefill の後に decode)で、
    生成したトークンごとの logits も返す。

    Returns:
        ``(生成したトークン, logits)``。logits の形状は ``(output_length, V)``。
    """
    model.eval()
    device = next(model.parameters()).device
    cache = KeyValueCache(model.num_layers)
    ids = torch.tensor([list(prompt_token_ids)], device=device)
    logits = model(ids, positions=torch.arange(ids.size(1), device=device), kv_cache=cache)[:, -1]
    tokens = [int(logits.argmax(dim=-1))]
    all_logits = [logits[0].cpu().numpy()]
    for _ in range(output_length - 1):
        step = torch.tensor([cache.length], device=device)
        last = torch.tensor([[tokens[-1]]], device=device)
        logits = model(last, positions=step, kv_cache=cache)[:, -1]
        tokens.append(int(logits.argmax(dim=-1)))
        all_logits.append(logits[0].cpu().numpy())
    return tokens, np.stack(all_logits)
