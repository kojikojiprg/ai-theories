"""層正規化(Layer Normalization)・RMSNorm のスクラッチ実装。

Ba, Kiros, Hinton, "Layer Normalization", arXiv:1607.06450, 2016、および
Zhang, Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019 の定義に基づく。
2 つの手法の関係(平均減算(re-centering)の有無)は
``theories/01_foundations/004_normalization_and_activation.ipynb`` の理論セクションで詳しく扱う。

記号 / Notation:
    d_model : 正規化を行う特徴次元(モデルの隠れ次元)
    γ (gamma) : 学習可能なスケールパラメータ、形状 (d_model,)
    β (beta)  : 学習可能なシフトパラメータ、形状 (d_model,)。RMSNorm は持たない
    ε (eps)   : 分散(または二乗平均)がゼロに近いときの数値安定化のための微小定数
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

    ``center``・``scale`` 引数(004 で追加)により、平均減算(re-centering、
    ``center``)と分散除算(re-scaling、``scale``)を個別に無効化できる。
    4 通りの組み合わせは 004 の理論セクション・実験 C で比較する
    (いずれの組み合わせでも γ・β は保持するため、条件間でパラメータ数は変わらない)。

    Args:
        d_model: 正規化を行う特徴次元。
        eps: 数値安定化のための微小定数 ε。
        center: True の場合、平均 μ を減算する(re-centering)。
            False の場合は μ = 0 として扱う(平均減算を行わない)。
        scale: True の場合、標準偏差 sqrt(σ^2 + ε) で除算する(re-scaling)。
            False の場合は除算を行わない(σ = 1 として扱う)。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        center: bool = True,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.center = center
        self.scale = scale
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """層正規化を適用する。

        Args:
            x: 形状 ``(..., d_model)`` の入力。

        Returns:
            x と同じ形状の正規化済みテンソル。
        """
        if self.center and self.scale:
            # center=True, scale=True(既定値)は 002 と完全に同一の計算にする。
            mean = x.mean(dim=-1, keepdim=True)
            # unbiased=False: 分散は 1/d_model の標本分散を使う(nn.LayerNorm と同じ規約)。
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * x_hat + self.beta

        x_hat = x
        if self.center:
            x_hat = x_hat - x_hat.mean(dim=-1, keepdim=True)
        if self.scale:
            # center=False の場合、x_hat = x に対する 2 乗平均(平均減算前の分散)になる。
            mean_square = x_hat.pow(2).mean(dim=-1, keepdim=True)
            x_hat = x_hat / torch.sqrt(mean_square + self.eps)
        return self.gamma * x_hat + self.beta


class RMSNorm(nn.Module):
    """RMSNorm(Root Mean Square Normalization、二乗平均平方根正規化)。

    Zhang, Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019 の定義:

        RMS(a) = sqrt((1/d_model) * sum_i a_i^2)
        RMSNorm(a) = γ * a / sqrt(RMS(a)^2 + ε)

    層正規化(Layer Normalization)から平均減算(re-centering)を落とし、
    分散除算(re-scaling)のみを行う。学習パラメータは γ のみで β を持たない。
    この設計により、a の平行移動 a -> a + b・1(1 は全成分が 1 のベクトル)に対して
    不変ではなくなる(層正規化は平行移動不変性(shift invariance)を持つが、
    RMSNorm は持たない)。この不変性の違いの導出、および実際の隠れ状態で
    これが問題になるかどうかの検証は 004 の理論セクション・実験 B・実験 D で扱う。

    Args:
        d_model: 正規化を行う特徴次元。
        eps: 数値安定化のための微小定数 ε。
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """RMSNorm を適用する。

        Args:
            x: 形状 ``(..., d_model)`` の入力。

        Returns:
            x と同じ形状の正規化済みテンソル。
        """
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x_hat = x / torch.sqrt(mean_square + self.eps)
        return self.gamma * x_hat
