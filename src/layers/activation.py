"""活性化関数(Activation Function)のスクラッチ実装。

Hendrycks, Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415, 2016、
Ramachandran, Zoph, Le, "Searching for Activation Functions", arXiv:1710.05941, 2017 の
定義に基づく。CLAUDE.md のコーディング規約に従い ``torch.nn.functional`` の組み込み実装は
使わずスクラッチ実装するが、正しさの確認は
``theories/01_foundations/004_normalization_and_activation.ipynb`` で
対応する組み込み実装との数値一致を検証する。

記号 / Notation:
    x    : 入力テンソル
    Φ(x) : 標準正規分布(平均 0、分散 1)の累積分布関数(CDF)
    β    : Swish の形状パラメータ(スカラー)。β -> ∞ で ReLU に収束する
    σ(z) : シグモイド関数 1 / (1 + exp(-z))
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def gelu_exact(x: Tensor) -> Tensor:
    """GELU(Gaussian Error Linear Unit)の厳密形。

    GELU(x) = x * Φ(x)

    Φ は標準正規分布の累積分布関数(CDF)。誤差関数(error function)
    erf を用いた恒等式 Φ(x) = (1/2) * (1 + erf(x / sqrt(2))) で計算する。

    Args:
        x: 任意形状の入力テンソル。

    Returns:
        x と同じ形状の出力テンソル。
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh_approximation(x: Tensor) -> Tensor:
    """GELU(Gaussian Error Linear Unit)の tanh 近似形。

    Hendrycks, Gimpel (2016) 式 (2) の近似:

        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))

    誤差関数を使わずに済むため、``gelu_exact`` より計算コストが低い
    (原論文発表当時、erf の高速な実装が一部の環境になかったことが動機)。

    Args:
        x: 任意形状の入力テンソル。

    Returns:
        x と同じ形状の出力テンソル。
    """
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    """Swish(Ramachandran et al., 2017)。β = 1 のとき SiLU(Sigmoid Linear Unit)と一致する。

    Swish(x) = x * σ(β * x)

    β -> ∞ の極限で σ(β * x) がステップ関数に収束するため、Swish は ReLU に収束する
    (負領域で 0、正領域で x)。この極限の様子は 004 の実験 A で数値的に確認する。

    Args:
        x: 任意形状の入力テンソル。
        beta: 形状パラメータ β。既定値 1.0 のとき Swish / SiLU と呼ばれる形になる。

    Returns:
        x と同じ形状の出力テンソル。
    """
    return x * torch.sigmoid(beta * x)
