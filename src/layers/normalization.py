"""層正規化(Layer Normalization)のスクラッチ実装。

Ba, Kiros, Hinton, "Layer Normalization", arXiv:1607.06450, 2016 の定義に基づく。

記号 / Notation:
    d_model : 正規化を行う特徴次元(モデルの隠れ次元)
    γ (gamma) : 学習可能なスケールパラメータ、形状 (d_model,)
    β (beta)  : 学習可能なシフトパラメータ、形状 (d_model,)
    ε (eps)   : 分散がゼロに近いときの数値安定化のための微小定数
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNormalization(nn.Module):
    """層正規化(Layer Normalization)。

    各サンプル・各系列位置ごとに、最終次元(特徴次元 d_model)方向の
    平均・分散を使って正規化する(Batch Normalization のようにバッチ方向・
    系列方向をまたいで統計量を計算するのではない点に注意)。

    LayerNorm(x) = γ * (x - μ) / sqrt(σ^2 + ε) + β

    ここで μ, σ^2 は x の最終次元(d_model 方向)についての平均・分散である。
    バッチサイズや系列長に依存せず各位置ごとに独立して計算されるため、
    推論時にバッチ統計を保持する running mean / running variance を必要としない
    (これが Batch Normalization との主な違いである)。

    Args:
        d_model: 正規化を行う特徴次元。
        eps: 数値安定化のための微小定数 ε。
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """層正規化を適用する。

        Args:
            x: 形状 ``(..., d_model)`` の入力。

        Returns:
            x と同じ形状の正規化済みテンソル。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False: 分散は 1/d_model の標本分散を使う(nn.LayerNorm と同じ規約)。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
