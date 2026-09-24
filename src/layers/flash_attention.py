"""Flash Attention のスクラッチ実装(014)。

タイリング(Tiling)と online softmax(Milakov & Gimelshein, 2018)による順伝播と、
softmax の出力 P を保存せずにブロックごとに再計算(recomputation)する逆伝播を実装する
(Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
NeurIPS 2022)。ループの順序は FlashAttention-2(Dao, ICLR 2024)に従い、順伝播は Query の
ブロックを外側のループ、逆伝播は Key・Value のブロックを外側のループとする。

- 入力の形状は ``(batch, heads, sequence, head_dim)``(001 の ``scaled_dot_product_attention()``
  と同じ並び)。先頭の次元(batch, heads)はまとめてベクトル化し、系列方向のみをブロックに分ける。
- 統計量(行ごとの最大値 m・正規化定数 ell・logsumexp L)と出力の累積は、入力が FP16・BF16 でも
  FP32 で行う(入力が FP64 の場合は FP64)。
- 系列長がブロックサイズで割り切れない場合、最後のブロックは短くなる。ブロックサイズが系列長以上の
  場合はブロックが 1 つになる。
- 因果マスク(``causal=True``)では、Query の位置 i は Key の位置 j <= i のみを参照する(001 の
  ``create_causal_mask()`` と同じ規約。Query と Key の系列長が等しいことを要求する)。マスクで
  完全に隠れるブロック(ブロック内の全ての j が全ての i より大きい)は計算を飛ばす。

Python のループで書いているため、速度は Flash Attention の本来の性能(1 つの融合カーネルで
SRAM 上に閉じて計算する)を反映しない。本実装の目的は、計算手順・数値的な性質・保存する
テンソルの大きさ(メモリ量の次数)を確かめることである。

使い方 / Usage::

    from src.layers.flash_attention import FlashAttentionFunction

    output = FlashAttentionFunction.apply(query, key, value, 64, 64, True, None)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """統計量と累積に使う dtype(FP16・BF16・FP32 は FP32、FP64 は FP64)。"""
    return torch.promote_types(dtype, torch.float32)


def _check_inputs(query: Tensor, key: Tensor, value: Tensor, causal: bool) -> None:
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query・key・value は (batch, heads, sequence, head_dim) の 4 階テンソル")
    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError(f"key と value の系列長が異なる: {key.shape} / {value.shape}")
    if query.shape[:2] != key.shape[:2] or query.size(-1) != key.size(-1):
        raise ValueError(f"query と key の形状が合わない: {query.shape} / {key.shape}")
    if causal and query.size(-2) != key.size(-2):
        raise ValueError("causal=True では Query と Key の系列長が等しい必要がある")


def _check_block_sizes(block_size_query: int, block_size_key: int) -> None:
    if block_size_query < 1 or block_size_key < 1:
        raise ValueError(
            f"ブロックサイズは 1 以上: block_size_query={block_size_query}, "
            f"block_size_key={block_size_key}"
        )


def key_block_end(query_block_end: int, sequence_length_key: int, causal: bool) -> int:
    """Query のブロック ``[i0, i1)`` が参照しうる Key の位置の上限(この位置を含まない)。

    因果マスクでは、ブロック内の最大の Query の位置 ``i1 - 1`` より後ろの Key は全て隠れるので、
    Key のブロックのループをここで打ち切る(完全に隠れるブロックを飛ばす)。
    """
    if causal:
        return min(query_block_end, sequence_length_key)
    return sequence_length_key


def query_block_start(key_block_begin: int, block_size_query: int, causal: bool) -> int:
    """Key のブロック ``[j0, j1)`` を参照しうる最初の Query のブロックの開始位置。

    因果マスクでは、Query のブロック ``[i0, i1)`` が Key の位置 ``j0`` を参照するのは
    ``i1 - 1 >= j0`` のときだけなので、それより前の Query のブロックを飛ばす。
    """
    if causal:
        return (key_block_begin // block_size_query) * block_size_query
    return 0


def count_block_pairs(
    sequence_length_query: int,
    sequence_length_key: int,
    block_size_query: int,
    block_size_key: int,
    causal: bool,
) -> tuple[int, int]:
    """計算するブロックの組 (i, j) の数と、全ブロックの組の数を返す。

    ``flash_attention_forward()`` と同じ規則(``key_block_end()``)で数える。因果マスクで
    飛ばしたブロックの割合は ``1 - processed / total`` になる。
    """
    _check_block_sizes(block_size_query, block_size_key)
    num_key_blocks = math.ceil(sequence_length_key / block_size_key)
    num_query_blocks = math.ceil(sequence_length_query / block_size_query)
    processed = 0
    for i0 in range(0, sequence_length_query, block_size_query):
        i1 = min(i0 + block_size_query, sequence_length_query)
        processed += math.ceil(key_block_end(i1, sequence_length_key, causal) / block_size_key)
    return processed, num_query_blocks * num_key_blocks


def _causal_block_mask(i0: int, i1: int, j0: int, j1: int, device: torch.device) -> Tensor | None:
    """ブロック ``[i0, i1) x [j0, j1)`` の因果マスク(True が参照する位置)。

    ブロック内の全ての組で ``j <= i`` が成り立つ(対角より下にある)場合は ``None`` を返す。
    """
    if j1 - 1 <= i0:
        return None
    rows = torch.arange(i0, i1, device=device)[:, None]
    cols = torch.arange(j0, j1, device=device)[None, :]
    return rows >= cols


def flash_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_size_query: int,
    block_size_key: int,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """タイリングと online softmax による Attention の順伝播。

    O = softmax(scale * Q K^T) V を、Query を ``block_size_query``(B_r)行、Key・Value を
    ``block_size_key``(B_c)行のブロックに分けて計算する。N x N のスコア行列 S と softmax の
    出力 P を実体化しない。Query のブロック i ごとに、行ごとの最大値 m_i・正規化定数 ell_i・
    正規化前の出力 O~_i を持ち、Key・Value のブロック j を 1 つ読むたびに

        S_ij = scale * Q_i K_j^T
        m_new = max(m_i, rowmax(S_ij))
        P~_ij = exp(S_ij - m_new)
        ell_i = exp(m_i - m_new) * ell_i + rowsum(P~_ij)
        O~_i = diag(exp(m_i - m_new)) O~_i + P~_ij V_j
        m_i = m_new

    と更新し、最後に O_i = diag(ell_i)^{-1} O~_i、L_i = m_i + log(ell_i) とする。

    Args:
        query: 形状 ``(batch, heads, N_q, d)``。
        key: 形状 ``(batch, heads, N_k, d)``。
        value: 形状 ``(batch, heads, N_k, d_v)``。
        block_size_query: Query のブロックの行数 B_r。
        block_size_key: Key・Value のブロックの行数 B_c。
        causal: 因果マスクを適用するか(``N_q == N_k`` が必要)。
        scale: スコアに掛ける係数。``None`` の場合 ``1 / sqrt(d)``(001 と同じ)。

    Returns:
        (output, logsumexp) のタプル。output は形状 ``(batch, heads, N_q, d_v)`` で入力と同じ
        dtype、logsumexp は形状 ``(batch, heads, N_q)`` で累積の dtype(FP32 または FP64)。
    """
    _check_inputs(query, key, value, causal)
    _check_block_sizes(block_size_query, block_size_key)
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    accumulation_dtype = _accumulation_dtype(query.dtype)
    sequence_length_query, sequence_length_key = query.size(-2), key.size(-2)
    leading_shape = query.shape[:-2]

    output = torch.empty(
        (*leading_shape, sequence_length_query, value.size(-1)),
        dtype=query.dtype,
        device=query.device,
    )
    logsumexp = torch.empty(
        (*leading_shape, sequence_length_query), dtype=accumulation_dtype, device=query.device
    )

    for i0 in range(0, sequence_length_query, block_size_query):
        i1 = min(i0 + block_size_query, sequence_length_query)
        # Q_i を「HBM から SRAM へ」読む(ブロック i の間は保持する)
        query_block = query[..., i0:i1, :].to(accumulation_dtype)
        row_max = torch.full(
            (*leading_shape, i1 - i0), float("-inf"), dtype=accumulation_dtype, device=query.device
        )
        row_sum = torch.zeros_like(row_max)
        output_block = torch.zeros(
            (*leading_shape, i1 - i0, value.size(-1)),
            dtype=accumulation_dtype,
            device=query.device,
        )

        for j0 in range(0, key_block_end(i1, sequence_length_key, causal), block_size_key):
            j1 = min(j0 + block_size_key, sequence_length_key)
            # K_j・V_j を読む
            key_block = key[..., j0:j1, :].to(accumulation_dtype)
            value_block = value[..., j0:j1, :].to(accumulation_dtype)

            scores = torch.matmul(query_block, key_block.transpose(-2, -1)) * scale
            if causal:
                mask = _causal_block_mask(i0, i1, j0, j1, query.device)
                if mask is not None:
                    scores = scores.masked_fill(~mask, float("-inf"))

            # 因果マスクでも、各行の最初に処理するブロック(j0 = 0)には j = 0 <= i が必ず
            # 含まれるので、new_max は最初のブロックで有限になり、exp(-inf - 有限) = 0 となる。
            new_max = torch.maximum(row_max, scores.amax(dim=-1))
            probabilities = torch.exp(scores - new_max[..., None])
            correction = torch.exp(row_max - new_max)
            row_sum = correction * row_sum + probabilities.sum(dim=-1)
            output_block = correction[..., None] * output_block + torch.matmul(
                probabilities, value_block
            )
            row_max = new_max

        # O_i と L_i を「HBM へ」書く
        output[..., i0:i1, :] = (output_block / row_sum[..., None]).to(query.dtype)
        logsumexp[..., i0:i1] = row_max + torch.log(row_sum)

    return output, logsumexp


def flash_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    logsumexp: Tensor,
    grad_output: Tensor,
    block_size_query: int,
    block_size_key: int,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """再計算(recomputation)による Attention の逆伝播。

    P を保存せず、順伝播が保存した logsumexp L から ``P_ij = exp(S_ij - L_i)`` をブロックごとに
    再計算する。D_i = rowsum(dO_i * O_i) を先に求めておき、Key・Value のブロック j を外側の
    ループ、Query のブロック i を内側のループとして

        dV_j += P_ij^T dO_i
        dP_ij = dO_i V_j^T
        dS_ij = P_ij * (dP_ij - D_i)
        dQ_i += scale * dS_ij K_j
        dK_j += scale * dS_ij^T Q_i

    を累積する(dK_j・dV_j はブロック j の間 SRAM に保持し、dQ_i は HBM 上で足し込む)。

    Args:
        query, key, value: 順伝播の入力。
        output: 順伝播の出力 O。
        logsumexp: 順伝播が返した行ごとの logsumexp L。
        grad_output: 出力に対する勾配 dO(``output`` と同じ形状)。
        block_size_query, block_size_key, causal, scale: 順伝播と同じ値。

    Returns:
        (grad_query, grad_key, grad_value) のタプル。いずれも対応する入力と同じ形状・dtype。
    """
    _check_inputs(query, key, value, causal)
    _check_block_sizes(block_size_query, block_size_key)
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    accumulation_dtype = _accumulation_dtype(query.dtype)
    sequence_length_query, sequence_length_key = query.size(-2), key.size(-2)

    # D_i = rowsum(dO_i * O_i)(= sum_j P_ij dP_ij)。形状 (batch, heads, N_q)
    delta = (grad_output.to(accumulation_dtype) * output.to(accumulation_dtype)).sum(dim=-1)
    grad_query = torch.zeros(query.shape, dtype=accumulation_dtype, device=query.device)
    grad_key = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    grad_value = torch.empty(value.shape, dtype=value.dtype, device=value.device)

    for j0 in range(0, sequence_length_key, block_size_key):
        j1 = min(j0 + block_size_key, sequence_length_key)
        key_block = key[..., j0:j1, :].to(accumulation_dtype)
        value_block = value[..., j0:j1, :].to(accumulation_dtype)
        grad_key_block = torch.zeros_like(key_block)
        grad_value_block = torch.zeros_like(value_block)

        for i0 in range(
            query_block_start(j0, block_size_query, causal), sequence_length_query, block_size_query
        ):
            i1 = min(i0 + block_size_query, sequence_length_query)
            query_block = query[..., i0:i1, :].to(accumulation_dtype)
            grad_output_block = grad_output[..., i0:i1, :].to(accumulation_dtype)

            scores = torch.matmul(query_block, key_block.transpose(-2, -1)) * scale
            if causal:
                mask = _causal_block_mask(i0, i1, j0, j1, query.device)
                if mask is not None:
                    scores = scores.masked_fill(~mask, float("-inf"))
            # P_ij の再計算(exp(-inf) = 0 なので、マスクした位置の確率は 0 になる)
            probabilities = torch.exp(scores - logsumexp[..., i0:i1, None])

            grad_value_block += torch.matmul(probabilities.transpose(-2, -1), grad_output_block)
            grad_probabilities = torch.matmul(grad_output_block, value_block.transpose(-2, -1))
            grad_scores = probabilities * (grad_probabilities - delta[..., i0:i1, None])
            grad_query[..., i0:i1, :] += scale * torch.matmul(grad_scores, key_block)
            grad_key_block += scale * torch.matmul(grad_scores.transpose(-2, -1), query_block)

        grad_key[..., j0:j1, :] = grad_key_block.to(key.dtype)
        grad_value[..., j0:j1, :] = grad_value_block.to(value.dtype)

    return grad_query.to(query.dtype), grad_key, grad_value


class FlashAttentionFunction(torch.autograd.Function):
    """``flash_attention_forward()``・``flash_attention_backward()``による autograd 関数。

    逆伝播のために保存するのは Q・K・V・O・L のみである(P や S は保存しない)。保存する量は
    系列長 N に対して Θ(N)(Q・K・V・O が 4 N d 要素、L が N 要素)で、標準の Attention を
    autograd で微分する場合に保存される N x N の P を持たない。

    使い方: ``FlashAttentionFunction.apply(query, key, value, block_size_query, block_size_key,
    causal, scale)``(``apply`` はキーワード引数を受け付けないので、位置引数で渡す)。
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        block_size_query: int,
        block_size_key: int,
        causal: bool = False,
        scale: float | None = None,
    ) -> Tensor:
        output, logsumexp = flash_attention_forward(
            query, key, value, block_size_query, block_size_key, causal, scale
        )
        ctx.save_for_backward(query, key, value, output, logsumexp)
        ctx.block_size_query = block_size_query
        ctx.block_size_key = block_size_key
        ctx.causal = causal
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value, output, logsumexp = ctx.saved_tensors
        grad_query, grad_key, grad_value = flash_attention_backward(
            query,
            key,
            value,
            output,
            logsumexp,
            grad_output.contiguous(),
            ctx.block_size_query,
            ctx.block_size_key,
            ctx.causal,
            ctx.scale,
        )
        return grad_query, grad_key, grad_value, None, None, None, None
