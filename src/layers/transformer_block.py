"""Transformer Block(Encoder Block / Decoder Block)のスクラッチ実装。

Vaswani et al., "Attention Is All You Need", NeurIPS 2017 の Encoder / Decoder 層に、
残差接続(Residual Connection、He et al., 2016)と層正規化(Layer Normalization、
Ba et al., 2016)を組み合わせたブロックを実装する。

正規化前置(Pre-Layer Normalization)と正規化後置(Post-Layer Normalization)の
どちらの構造を使うかは ``norm_first`` 引数で切り替える(Xiong et al., 2020 の議論に対応)。

記号 / Notation:
    d_model : ブロックの入出力次元
    h       : Multi-Head Attention のヘッド数
    d_ff    : Feed-Forward Network の中間層次元
"""

from __future__ import annotations

from torch import Tensor, nn

from src.layers.attention import MultiHeadAttention
from src.layers.feedforward import FeedForwardNetwork
from src.layers.normalization import LayerNormalization


class EncoderBlock(nn.Module):
    """Encoder Block(自己注意 + Feed-Forward Network の 2 サブレイヤー)。

    正規化後置(Post-Layer Normalization、``norm_first=False``)の場合:

        x = LayerNorm(x + SelfAttention(x, x, x))
        x = LayerNorm(x + FeedForwardNetwork(x))

    正規化前置(Pre-Layer Normalization、``norm_first=True``)の場合:

        x = x + SelfAttention(LayerNorm(x), LayerNorm(x), LayerNorm(x))
        x = x + FeedForwardNetwork(LayerNorm(x))

    どちらの場合も残差接続(Residual Connection)は必ず正規化前の ``x`` を経路に持つ
    (正規化前置では残差経路そのものが正規化を一切経由しない)。

    Args:
        d_model: 入出力次元。
        num_heads: Multi-Head Attention のヘッド数 h。
        d_ff: Feed-Forward Network の中間層次元。
        dropout: Attention 重み・各サブレイヤー出力に適用する dropout 率。
        activation: Feed-Forward Network の活性化関数("relu" または "gelu")。
        norm_first: True で正規化前置(Pre-Layer Normalization)、
            False で正規化後置(Post-Layer Normalization)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(
            d_model, d_ff, activation=activation, dropout=dropout
        )
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encoder Block の順伝播。

        Args:
            x: 形状 ``(B, S, d_model)`` の入力。
            mask: 自己注意に渡すマスク(省略可)。パディングマスクなどを想定する。

        Returns:
            (output, attn_weights) のタプル。
            output は ``(B, S, d_model)``、attn_weights は ``(B, h, S, S)``。
        """
        if self.norm_first:
            normed = self.norm1(x)
            attn_out, attn_weights = self.self_attn(normed, normed, normed, mask)
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            attn_out, attn_weights = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_out))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x, attn_weights


class DecoderBlock(nn.Module):
    """Decoder Block(masked self-attention + cross-attention + Feed-Forward Network の
    3 サブレイヤー)。

    Encoder Block に、Encoder の出力(memory)を参照する交差注意(cross-attention)を
    追加した構造。交差注意では Query が Decoder 側の表現から、Key / Value が
    Encoder の出力(memory)から作られる。

    正規化後置(``norm_first=False``)の場合:

        x = LayerNorm(x + SelfAttention(x, x, x, tgt_mask))
        x = LayerNorm(x + CrossAttention(x, memory, memory, memory_mask))
        x = LayerNorm(x + FeedForwardNetwork(x))

    正規化前置(``norm_first=True``)の場合:

        x = x + SelfAttention(LayerNorm(x), LayerNorm(x), LayerNorm(x), tgt_mask)
        x = x + CrossAttention(LayerNorm(x), memory, memory, memory_mask)
        x = x + FeedForwardNetwork(LayerNorm(x))

    Args:
        d_model: 入出力次元。
        num_heads: Multi-Head Attention のヘッド数 h。
        d_ff: Feed-Forward Network の中間層次元。
        dropout: Attention 重み・各サブレイヤー出力に適用する dropout 率。
        activation: Feed-Forward Network の活性化関数("relu" または "gelu")。
        norm_first: True で正規化前置、False で正規化後置。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(
            d_model, d_ff, activation=activation, dropout=dropout
        )
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decoder Block の順伝播。

        Args:
            x: 形状 ``(B, S_tgt, d_model)`` の Decoder 側入力。
            memory: 形状 ``(B, S_src, d_model)`` の Encoder 出力。
                交差注意の Key / Value になる。
            tgt_mask: 自己注意に渡すマスク。自己回帰生成では因果マスクを渡す。
            memory_mask: 交差注意に渡すマスク(Encoder 側のパディングマスクなど)。

        Returns:
            (output, self_attn_weights, cross_attn_weights) のタプル。
            output は ``(B, S_tgt, d_model)``、
            self_attn_weights は ``(B, h, S_tgt, S_tgt)``、
            cross_attn_weights は ``(B, h, S_tgt, S_src)``。
        """
        if self.norm_first:
            normed = self.norm1(x)
            self_out, self_attn_weights = self.self_attn(normed, normed, normed, tgt_mask)
            x = x + self.dropout1(self_out)

            normed = self.norm2(x)
            cross_out, cross_attn_weights = self.cross_attn(normed, memory, memory, memory_mask)
            x = x + self.dropout2(cross_out)

            x = x + self.dropout3(self.feed_forward(self.norm3(x)))
        else:
            self_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout1(self_out))

            cross_out, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
            x = self.norm2(x + self.dropout2(cross_out))

            x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x, self_attn_weights, cross_attn_weights
