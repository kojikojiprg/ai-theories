"""注意機構(Attention Mechanism)のスクラッチ実装。

Vaswani et al., "Attention Is All You Need", NeurIPS 2017 の
Scaled Dot-Product Attention と Multi-Head Attention を PyTorch で実装する。

記号 / Notation:
    B     : バッチサイズ(batch size)
    S_q   : Query 側の系列長(target sequence length)
    S_k   : Key / Value 側の系列長(source sequence length)
    d_model : モデルの隠れ次元(model dimension)
    h     : ヘッド数(number of heads)
    d_k   : 1 ヘッドあたりの Query / Key の次元(= d_model / h)
    d_v   : 1 ヘッドあたりの Value の次元(本実装では d_v = d_k)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Scaled Dot-Product Attention を計算する。

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        query: 形状 ``(..., S_q, d_k)`` の Query 行列 Q。
        key: 形状 ``(..., S_k, d_k)`` の Key 行列 K。
        value: 形状 ``(..., S_k, d_v)`` の Value 行列 V。
            先頭の ``...`` 次元(バッチやヘッド)は 3 つのテンソルで
            ブロードキャスト可能である必要がある。
        mask: 形状 ``(..., S_q, S_k)`` の bool テンソル(省略可)。
            **True の位置が「参加させる(attend する)」** を表し、
            False の位置のスコアは -inf に置き換えられる
            (PyTorch の ``F.scaled_dot_product_attention`` と同じ規約)。
        dropout: Attention 重みに適用する Dropout モジュール(省略可)。

    Returns:
        (output, attn_weights) のタプル。
        output は形状 ``(..., S_q, d_v)``、
        attn_weights は形状 ``(..., S_q, S_k)`` で最終次元の和が 1 になる。

    Note:
        ある行が全て False のマスクを与えると softmax が 0/0 となり NaN が出る。
        パディングマスクを作る際は、少なくとも 1 つは True を残すこと。
    """
    d_k = query.size(-1)

    # スコア行列(logits): (..., S_q, S_k)
    # sqrt(d_k) によるスケーリングが Scaled Dot-Product Attention の「Scaled」に対応する。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # False(= 参加させない)の位置を -inf にすることで softmax 後の重みを 0 にする。
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    # Key 方向(最終次元)に softmax をとり、各 Query の重み和を 1 にする。
    attn_weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 重み付き平均として Value を混合する: (..., S_q, S_k) @ (..., S_k, d_v)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention(多頭注意機構)。

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    実装上は、ヘッドごとの射影 ``W_i^Q ∈ R^{d_model × d_k}`` をヘッド方向に連結した
    1 つの ``nn.Linear(d_model, d_model)`` として持ち、射影後に
    ``(B, S, d_model) -> (B, h, S, d_k)`` へ reshape することで
    全ヘッドを 1 回の行列積でまとめて計算する(数学的には等価)。

    Args:
        d_model: モデルの隠れ次元。``num_heads`` で割り切れる必要がある。
        num_heads: ヘッド数 h。
        dropout: Attention 重みに適用する dropout 率。
        bias: 線形射影にバイアス項を持たせるか(原論文は bias なし)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) は num_heads ({num_heads}) で割り切れる必要がある"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 本実装では d_v = d_k

        # W^Q, W^K, W^V(全ヘッド分をまとめたもの)と出力射影 W^O
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier 一様分布で射影行列を初期化する。"""
        for module in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, S, d_model) -> (B, h, S, d_k) へ分割する。"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, h, S, d_k) -> (B, S, d_model) へ連結(Concat)する。"""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    @staticmethod
    def _expand_mask(mask: Tensor) -> Tensor:
        """マスクをヘッド次元を含む 4 階テンソル ``(B, h, S_q, S_k)`` にブロードキャストする。"""
        if mask.dim() == 2:  # (S_q, S_k) — 全バッチ・全ヘッド共通
            return mask[None, None, :, :]
        if mask.dim() == 3:  # (B, S_q, S_k) — 全ヘッド共通
            return mask[:, None, :, :]
        if mask.dim() == 4:  # (B, h, S_q, S_k)
            return mask
        raise ValueError(f"mask の次元数は 2, 3, 4 のいずれかである必要がある: {mask.dim()}")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Multi-Head Attention の順伝播。

        Args:
            query: 形状 ``(B, S_q, d_model)``。
            key: 形状 ``(B, S_k, d_model)``。
            value: 形状 ``(B, S_k, d_model)``。
                自己注意(self-attention)では query = key = value を渡す。
            mask: True が「参加させる」を表す bool マスク。
                形状は ``(S_q, S_k)`` / ``(B, S_q, S_k)`` / ``(B, h, S_q, S_k)``。

        Returns:
            (output, attn_weights) のタプル。
            output は ``(B, S_q, d_model)``、attn_weights は ``(B, h, S_q, S_k)``。
        """
        # 1. 線形射影(全ヘッド分をまとめて計算)
        q = self._split_heads(self.w_q(query))  # (B, h, S_q, d_k)
        k = self._split_heads(self.w_k(key))  # (B, h, S_k, d_k)
        v = self._split_heads(self.w_v(value))  # (B, h, S_k, d_v)

        if mask is not None:
            mask = self._expand_mask(mask)

        # 2. 各ヘッドで Scaled Dot-Product Attention
        head_outputs, attn_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 3. ヘッドを連結して出力射影 W^O を適用
        concatenated = self._merge_heads(head_outputs)  # (B, S_q, d_model)
        output = self.w_o(concatenated)
        return output, attn_weights


def create_causal_mask(
    seq_len: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """因果マスク(causal / look-ahead mask)を作る。

    位置 i の Query が位置 j <= i の Key のみを参照できるようにする下三角 bool 行列。
    デコーダの自己注意で未来の情報を見ないようにするために使う。

    Args:
        seq_len: 系列長 S。
        device: 生成先デバイス。

    Returns:
        形状 ``(S, S)`` の bool テンソル(True = 参加させる)。
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def create_padding_mask(pad_positions: Tensor) -> Tensor:
    """パディングマスクを作る。

    Args:
        pad_positions: 形状 ``(B, S_k)`` の bool テンソル。
            True がパディング位置(参照させたくない位置)。

    Returns:
        形状 ``(B, 1, S_k)`` の bool テンソル(True = 参加させる)。
        ``MultiHeadAttention.forward`` の ``mask`` にそのまま渡すと
        ``(B, 1, S_q, S_k)`` へブロードキャストされる。
    """
    return ~pad_positions.bool().unsqueeze(1)
