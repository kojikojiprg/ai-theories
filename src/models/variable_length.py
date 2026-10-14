"""可変長の入力に対応した埋め込みモデルと cross-encoder の部品(026)。

024 の ``TextEmbeddingModel`` と 025 の ``CrossEncoder`` は固定長の入力(
パディングなし)を前提にする。
026 の Inverse Cloze Task は、query(抜いた文)と passage(文を除いた文脈)の長さが組ごとに違う。
**右側をパディングして長さの配列を添える**。因果マスクのもとでは、位置 ``t`` の出力は位置 ``t``
までの
トークンにだけ依存するので、**右側のパディングは、パディングより手前の位置の出力に影響しない**
(パディングなしで符号化した出力と、丸めの範囲で一致する。026 のノートブックの単体テストで確かめる)。
変えるのは、出力を 1 本のベクトル(またはスコア)にまとめる段だけである。

- 平均プール: 長さ ``len`` の手前の位置だけを平均する。
- cross-encoder: 列を ``[query (長さ lq)] [区切り] [passage (長さ lp)] [終端]`` と左から詰めて並べ、
  終端の位置(``lq + lp + 1``)の出力を読む。

既存の ``src/models/text_embedding.py``・``src/models/cross_encoder.py`` は変更しない(固定長の入力
で長さの配列がすべて満長のときは、既存の経路と同じ計算になる)。

記号 / Notation:
    B : バッチサイズ、S : 系列長(バッチの中の最大の長さ)、d : 埋め込みの次元
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from src.models.cross_encoder import CrossEncoder
from src.models.text_embedding import (
    TextEmbeddingModel,
    compute_hidden_states,
    truncate_and_normalize,
)


def pooled_with_lengths(
    model: TextEmbeddingModel, token_ids: Tensor, lengths: Tensor, full_length: bool = False
) -> Tensor:
    """可変長の入力の、プーリングした **正規化前の** ベクトル ``(B, d)``。

    導入: 026。

    Args:
        model: 因果マスクの ``TextEmbeddingModel``。
        token_ids: 形状 ``(B, S)``。右側を任意のトークンでパディングしてよい。
        lengths: 形状 ``(B,)``。各行の有効な長さ(1 以上 ``S`` 以下)。
        full_length: すべての行が満長(``lengths == S``)であると呼び出し側が保証するとき真。
            ``model.pooled()`` と同じ計算(bit 単位で一致)になる。

    Returns:
        形状 ``(B, d)`` の、正規化前のプーリング済みベクトル。
    """
    if model.attention != "causal":
        raise ValueError("右側のパディングが出力に影響しないのは因果マスクのときだけ")
    if full_length:
        return model.pooled(token_ids)
    hidden = compute_hidden_states(model.backbone, token_ids, model.attention)
    batch, size = token_ids.shape
    if model.pooling == "last":
        return hidden[torch.arange(batch, device=hidden.device), lengths - 1, :]
    mask = (torch.arange(size, device=hidden.device)[None, :] < lengths[:, None]).to(hidden.dtype)
    summed = (hidden * mask[:, :, None]).sum(dim=1)
    return summed / lengths[:, None].to(hidden.dtype)


@torch.no_grad()
def encode_tokens_with_lengths(
    model: TextEmbeddingModel,
    token_ids: np.ndarray | Tensor,
    lengths: np.ndarray,
    batch_size: int,
    dimensions: Sequence[int | None] = (None,),
) -> dict[int, Tensor]:
    """可変長のトークン列の集合を FP32 で符号化する(評価用、勾配なし)。

    長さの昇順に並べ替えてバッチにし、バッチごとに最大の長さまで切り詰めて計算する(パディングの
    計算を減らす)。結果は元の並びに戻す。

    導入: 026。

    Args:
        model: 因果マスクの ``TextEmbeddingModel``。
        token_ids: 形状 ``(n, S)`` のトークン ID(右側をパディングしてよい)。
        lengths: 形状 ``(n,)``。各行の有効な長さ。
        batch_size: 1 回に符号化する行数。
        dimensions: 切り詰める次元の候補(``None`` は全次元)。

    Returns:
        ``{次元: 形状 (n, 次元) の埋め込み}``(モデルと同じデバイス。次元の ``None`` は全次元)。
    """
    device = next(model.parameters()).device
    tokens = torch.as_tensor(token_ids)
    lengths = np.asarray(lengths)
    order = np.argsort(lengths, kind="stable")
    was_training = model.training
    model.eval()
    full = model.embedding_dimension
    keys = [full if d is None else int(d) for d in dimensions]
    pieces: dict[int, list[Tensor]] = {k: [] for k in keys}
    try:
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            width = int(lengths[index].max())
            batch = tokens[torch.as_tensor(index)][:, :width].to(device)
            batch_lengths = torch.as_tensor(lengths[index], device=device)
            pooled = pooled_with_lengths(model, batch, batch_lengths).float()
            for key in keys:
                pieces[key].append(truncate_and_normalize(pooled, key))
    finally:
        model.train(was_training)
    inverse = torch.as_tensor(np.argsort(order, kind="stable"), device=device)
    return {k: torch.cat(v)[inverse] for k, v in pieces.items()}


def build_variable_cross_encoder_inputs(
    query_tokens: Tensor,
    query_lengths: Tensor,
    passage_tokens: Tensor,
    passage_lengths: Tensor,
    separator_id: int,
    terminal_id: int,
) -> tuple[Tensor, Tensor]:
    """``[query (lq)] [区切り] [passage (lp)] [終端]`` を左から詰めて並べ、右側を 0
    でパディングする。

    導入: 026。

    Args:
        query_tokens: 形状 ``(B, L_q)``(右側がパディング)。
        query_lengths: 形状 ``(B,)``。
        passage_tokens: 形状 ``(B, L_p)``(右側がパディング。組ごとに 1 対 1 に対応)。
        passage_lengths: 形状 ``(B,)``。
        separator_id: query と passage の間に置く区切りのトークン ID。
        terminal_id: 列の終端に置くトークン ID。

    Returns:
        ``(列 (B, W)、各行の有効な長さ (B,))``。``W`` は ``L_q + 1 + L_p + 1``。
    """
    batch, query_width = query_tokens.shape
    passage_width = passage_tokens.size(1)
    width = query_width + 1 + passage_width + 1
    device = query_tokens.device
    fixed = torch.cat(
        [
            query_tokens,
            torch.full((batch, 1), separator_id, dtype=query_tokens.dtype, device=device),
            passage_tokens.to(query_tokens.dtype),
            torch.full((batch, 1), terminal_id, dtype=query_tokens.dtype, device=device),
        ],
        dim=1,
    )
    position = torch.arange(width, device=device)[None, :]
    lq = query_lengths[:, None]
    lp = passage_lengths[:, None]
    # 詰めた列の位置 j が、固定の配置のどの位置から来るか
    source = torch.where(
        position < lq,
        position,
        torch.where(
            position == lq,
            torch.full_like(position, query_width),
            torch.where(
                position <= lq + lp,
                position - lq + query_width,
                torch.full_like(position, query_width + 1 + passage_width),
            ),
        ),
    )
    total = query_lengths + 1 + passage_lengths + 1
    gathered = torch.gather(fixed, 1, source.clamp(max=width - 1))
    return torch.where(position < total[:, None], gathered, torch.zeros_like(gathered)), total


class VariableLengthCrossEncoder(CrossEncoder):
    """可変長の query・passage に対応した cross-encoder(構成・ヘッドは ``CrossEncoder`` と同じ)。

    スコアは、列の終端の位置(有効な長さの最後)の、最終正規化層の後の隠れ状態に線形のヘッドを掛けた
    スカラーである。ヘッドは FP32 で計算する。

    導入: 026。

    Args:
        backbone: 本体の ``GPTLanguageModel``。
        separator_id: query と passage の間に置く区切りのトークン ID。
        terminal_id: 列の終端に置くトークン ID(この位置の隠れ状態がスコアになる)。
        head_init_std: 線形のヘッドの重みの初期化の標準偏差。
    """

    def forward(  # type: ignore[override]
        self,
        query_tokens: Tensor,
        passage_tokens: Tensor,
        query_lengths: Tensor | None = None,
        passage_lengths: Tensor | None = None,
    ) -> Tensor:
        batch = query_tokens.size(0)
        device = query_tokens.device
        if query_lengths is None:
            query_lengths = torch.full((batch,), query_tokens.size(1), device=device)
        if passage_lengths is None:
            passage_lengths = torch.full((batch,), passage_tokens.size(1), device=device)
        inputs, total = build_variable_cross_encoder_inputs(
            query_tokens,
            query_lengths,
            passage_tokens,
            passage_lengths,
            self.separator_id,
            self.terminal_id,
        )
        width = int(total.max())
        hidden = compute_hidden_states(self.backbone, inputs[:, :width], "causal")
        last = hidden[torch.arange(batch, device=device), total - 1, :]
        with torch.autocast(device_type=device.type, enabled=False):  # ヘッドは常に FP32
            return self.head(last.float()).squeeze(-1)
