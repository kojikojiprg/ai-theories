"""位置エンコーディング(Positional Encoding)のスクラッチ実装。

Transformer Block(Multi-Head Attention + Feed-Forward Network)は入力の並び替えに
対して置換同変(permutation equivariant)であり、単体では系列の順序を区別できない。
位置エンコーディングは、各位置に固有のベクトルを埋め込みに加算することで、
モデルに順序の情報を与えるための仕組みである。

本モジュールには、Vaswani et al. (2017) の正弦波(sinusoidal)方式のみを実装する。
学習可能な絶対位置埋め込み(Learned Absolute Positional Embedding)、
相対位置エンコーディング(Relative Positional Encoding)、
回転位置エンコーディング(RoPE: Rotary Position Embedding)などの他方式との比較は
トピック 003「位置エンコーディング / RoPE」で扱う(RoPE の実装もこのモジュールに
追加する予定)。クラス名を ``SinusoidalPositionalEncoding`` としているのはこのためである。

記号 / Notation:
    pos     : 系列中の位置(0-indexed)
    i       : 次元インデックス(0 <= i < d_model / 2)
    d_model : モデルの隠れ次元
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """正弦波(sinusoidal)位置エンコーディング。

    Vaswani et al., "Attention Is All You Need", NeurIPS 2017, 3.5 節の定義:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    各位置 pos に対して長さ d_model のベクトル PE(pos, :) を 1 つ定め、
    トークン埋め込みに加算して使う。位置ごとに異なる周波数の sin / cos を
    偶数・奇数次元に交互に割り当てることで、各位置が一意なパターンを持つ。

    Args:
        d_model: モデルの隠れ次元。
        max_len: 事前に計算しておく位置の最大数。
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        # 10000^(2i/d_model) を exp/log で計算(オーバーフロー回避のため対数空間で計算)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model / 2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 学習パラメータではないが device / dtype をモデルと合わせたいので buffer として登録する。
        # (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """入力の埋め込みに位置エンコーディングを加算する。

        Args:
            x: 形状 ``(B, S, d_model)`` のトークン埋め込み。``S <= max_len`` が必要。

        Returns:
            x と同じ形状の、位置エンコーディングを加算したテンソル。
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
