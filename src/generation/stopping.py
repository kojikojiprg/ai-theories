"""終端記号の文字列で停止する貪欲法(greedy decoding)の生成(016)。

``GPTLanguageModel.generate()``(``src/models/gpt.py``)は指定したトークン数だけ生成し、
途中で止まらない。また 1 回の呼び出しで扱えるのは長さの揃った起点の文脈だけである。
016 の採点では、評価集合の全事例(長さの異なる指示部分)から、終端記号の文字列
(``### End``)が現れるまで、または上限トークン数に達するまで貪欲法で生成する。本モジュールは
そのための関数を提供し、既存の ``generate()`` は変更しない。

**バッチ化**: 起点の文脈(指示部分)を長さごとにまとめ、同じ長さのものだけを 1 つのバッチに
する。パディングを使わないので、位置(RoPE の ``positions``)と因果マスクは 1 事例ずつ生成
する場合と同じになる。KV キャッシュ(``src/generation/cache.py``)を使い、prefill の後は
新しいトークン 1 個ずつ順伝播する(``generate(use_cache=True)`` と同じ手順)。

**停止**: 各事例について、生成したトークン列を毎ステップ復号し、終端記号の文字列が初めて
現れたステップで、その事例の生成を止める(そのトークンまでを返す)。バッチ内の全事例が
止まるか、上限トークン数に達したら、バッチの生成を終える。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from src.generation.cache import KeyValueCache


@torch.no_grad()
def greedy_generate_until_stop(
    model: torch.nn.Module,
    prompts: Sequence[Sequence[int]],
    decode: Callable[[list[int]], str],
    stop_text: str,
    max_new_tokens: int,
    device: torch.device | str,
    batch_size: int = 64,
) -> list[list[int]]:
    """貪欲法で、終端記号の文字列が現れるまで(または上限トークン数まで)生成する。

    Args:
        model: ``GPTLanguageModel``(``forward(token_ids, positions, kv_cache)`` を持つ)。
        prompts: 事例ごとの起点の文脈(トークン ID の列)。
        decode: トークン ID の列を文字列に復号する関数。
        stop_text: 停止の条件とする文字列。
        max_new_tokens: 1 事例あたりの生成トークン数の上限。
        device: 生成に使うデバイス。
        batch_size: 同じ長さの文脈をまとめる 1 回の順伝播の事例数の上限。

    Returns:
        事例ごとの生成トークン列(起点の文脈を含まない)。``stop_text`` が現れた事例は、
        それが現れたトークンまでを含む。現れなかった事例は ``max_new_tokens`` 個。
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens は 1 以上である必要がある: {max_new_tokens}")
    longest = max(len(p) for p in prompts)
    if longest + max_new_tokens > model.max_sequence_length:
        raise ValueError(
            f"文脈の長さ({longest})+ 上限トークン数({max_new_tokens})が "
            f"max_sequence_length({model.max_sequence_length})を超える"
        )

    was_training = model.training
    model.eval()
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_length[len(prompt)].append(index)

    outputs: list[list[int] | None] = [None] * len(prompts)
    try:
        for length in sorted(by_length):
            members = by_length[length]
            for start in range(0, len(members), batch_size):
                chunk = members[start : start + batch_size]
                context = torch.tensor([list(prompts[i]) for i in chunk], dtype=torch.long)
                generated = _generate_same_length(
                    model, context.to(device), decode, stop_text, max_new_tokens
                )
                for index, tokens in zip(chunk, generated, strict=True):
                    outputs[index] = tokens
    finally:
        model.train(was_training)
    return [list(o) for o in outputs]


def _generate_same_length(
    model: torch.nn.Module,
    context: Tensor,
    decode: Callable[[list[int]], str],
    stop_text: str,
    max_new_tokens: int,
) -> list[list[int]]:
    """長さの揃った文脈のバッチ ``(b, S_0)`` から生成する(``greedy_generate_until_stop`` の本体)。

    本文は上の関数の docstring を参照。
    """
    batch, initial_length = context.shape
    kv_cache = KeyValueCache(model.num_layers)
    positions = torch.arange(initial_length, device=context.device)
    logits = model(context, positions=positions, kv_cache=kv_cache)
    next_tokens = logits[:, -1, :].argmax(dim=-1)

    generated: list[list[int]] = [[] for _ in range(batch)]
    finished = [False] * batch
    for step in range(max_new_tokens):
        values = next_tokens.tolist()
        for row in range(batch):
            if finished[row]:
                continue
            generated[row].append(values[row])
            if stop_text in decode(generated[row]):
                finished[row] = True
        if all(finished) or step == max_new_tokens - 1:
            break
        step_position = torch.tensor([kv_cache.length], device=context.device)
        logits = model(next_tokens[:, None], positions=step_position, kv_cache=kv_cache)
        next_tokens = logits[:, -1, :].argmax(dim=-1)
    return generated
