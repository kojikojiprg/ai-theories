"""画像のパッチ分割とパッチ埋め込み(Patch Embedding)、2 次元正弦波の位置埋め込みのスクラッチ実装。

Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
ICLR 2021 の式 1 に対応する。画像 x ∈ R^{H × W × C} を P × P のパッチ N = HW / P^2 個に分割し、
平坦化したパッチ x_p^i ∈ R^{P^2 C} を行列 E ∈ R^{(P^2 C) × D} で射影する。

この線形射影は、カーネル幅・stride がともに P の畳み込み(``nn.Conv2d(C, D, kernel_size=P,
stride=P)``)と等価である(パッチどうしが重ならないため、畳み込みの各出力位置が 1 つのパッチの
線形射影そのものになる)。``PatchEmbedding`` は畳み込みで実装し、``extract_patches()`` による
平坦化 + 線形射影と数値的に一致することを 019 のノートブックで確かめる。

2 次元正弦波の位置埋め込みは Beyer et al., "Better plain ViT baselines for ImageNet-1k",
arXiv:2205.01580, 2022 の実装(big_vision の ``posemb_sincos_2d``)に従う。

記号 / Notation:
    H, W : 画像の高さ・幅(画素)
    C    : 画像のチャネル数
    P    : パッチの一辺の画素数(パッチサイズ)
    N    : パッチの数 HW / P^2(Transformer に入力する系列長。[CLS] トークンを除く)
    D    : 埋め込みの次元(Transformer の d_model)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def extract_patches(images: Tensor, patch_size: int) -> Tensor:
    """画像を重ならない P × P のパッチに分割し、各パッチを平坦化する。

    パッチの並び順はラスタ順(パッチの格子の行を上から、各行の中を左から)である。
    各パッチの平坦化の順序は (C, P, P)(チャネル → 行 → 列)とし、``nn.Conv2d`` の重み
    ``(D, C, P, P)`` の並びに合わせる。原論文は (P, P, C) の順で平坦化するが、平坦化の順序の
    違いは E の行の並べ替えにすぎず、表現できる写像は同じである。

    Args:
        images: 形状 ``(B, C, H, W)`` の画像。H・W は P で割り切れる必要がある。
        patch_size: パッチの一辺 P。

    Returns:
        形状 ``(B, N, C * P * P)`` のテンソル(N = HW / P^2)。
    """
    batch_size, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"画像の大きさ ({height}, {width}) はパッチサイズ {patch_size} で割り切れる必要がある"
        )
    grid_h, grid_w = height // patch_size, width // patch_size
    x = images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
    # (B, C, grid_h, P, grid_w, P) -> (B, grid_h, grid_w, C, P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch_size, grid_h * grid_w, channels * patch_size * patch_size)


def permute_patches(images: Tensor, patch_size: int, permutation: Tensor) -> Tensor:
    """画像を P × P のパッチ単位で並べ替える(各パッチの中身は変えない)。

    出力のラスタ順で i 番目のパッチは、入力の ``permutation[i]`` 番目のパッチになる。
    位置埋め込みを持たない ViT の出力がこの並べ替えに対して不変であることの確認と、
    パッチを並べ替えた画像での正解率(位置情報への依存度の診断量)に使う。

    Args:
        images: 形状 ``(B, C, H, W)``。
        patch_size: パッチの一辺 P。
        permutation: 形状 ``(N,)`` の置換(0, ..., N-1 の並べ替え)。

    Returns:
        ``images`` と同じ形状のテンソル。
    """
    batch_size, channels, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    if permutation.numel() != grid_h * grid_w:
        raise ValueError(
            f"置換の長さ {permutation.numel()} がパッチの数 {grid_h * grid_w} と異なる"
        )
    x = images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, grid_h * grid_w, -1)
    x = x[:, permutation.to(images.device)]
    x = x.reshape(batch_size, grid_h, grid_w, channels, patch_size, patch_size)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)


class PatchEmbedding(nn.Module):
    """パッチ分割 + 線形射影(カーネル幅・stride が P の畳み込みとして実装する)。

    z_i = x_p^i E + b(i = 1, ..., N)

    Args:
        image_size: 入力画像の一辺 H = W(正方形を想定する)。
        patch_size: パッチの一辺 P。``image_size`` を割り切る必要がある。
        in_channels: 画像のチャネル数 C。
        d_model: 埋め込みの次元 D。
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) は patch_size ({patch_size}) で割り切れる必要がある"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def linear_projection_weight(self) -> tuple[Tensor, Tensor]:
        """畳み込みの重みを、平坦化したパッチに掛ける行列 E ∈ R^{(P^2 C) × D} と
        バイアス b ∈ R^{D} の形で返す(``extract_patches()`` の平坦化の順序に対応する)。"""
        weight = self.projection.weight.reshape(self.d_model, -1).t()
        return weight, self.projection.bias

    def forward(self, images: Tensor) -> Tensor:
        """パッチ埋め込みの系列を返す。

        Args:
            images: 形状 ``(B, C, H, W)`` の画像。

        Returns:
            形状 ``(B, N, D)`` のパッチ埋め込み(パッチはラスタ順)。
        """
        x = self.projection(images)  # (B, D, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


def build_2d_sinusoidal_position_embedding(
    grid_height: int,
    grid_width: int,
    d_model: int,
    temperature: float = 10_000.0,
) -> Tensor:
    """2 次元正弦波の位置埋め込み(Beyer et al., 2022、big_vision の ``posemb_sincos_2d``)。

    格子上の位置 (y, x) に対し、D/4 個の周波数 ω_k = temperature^{-k / (D/4 - 1)}
    (k = 0, ..., D/4 - 1)を使って、

        p(y, x) = [sin(x ω), cos(x ω), sin(y ω), cos(y ω)]

    を連結する(各ブロックが D/4 次元)。横方向の位置 x と縦方向の位置 y を別々の次元に
    符号化するので、2 つの位置の埋め込みの内積は、横方向の差と縦方向の差の関数の和になる。

    Args:
        grid_height: パッチの格子の行数。
        grid_width: パッチの格子の列数。
        d_model: 埋め込みの次元 D(4 で割り切れる必要がある)。
        temperature: 周波数の底(003 の正弦波の位置エンコーディングの 10000 と同じ)。

    Returns:
        形状 ``(grid_height * grid_width, d_model)`` の埋め込み(位置はラスタ順)。
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model ({d_model}) は 4 で割り切れる必要がある")
    quarter = d_model // 4
    y, x = torch.meshgrid(
        torch.arange(grid_height, dtype=torch.float64),
        torch.arange(grid_width, dtype=torch.float64),
        indexing="ij",
    )
    omega = torch.arange(quarter, dtype=torch.float64) / (quarter - 1)
    omega = 1.0 / (temperature**omega)
    y_angles = y.flatten()[:, None] * omega[None, :]
    x_angles = x.flatten()[:, None] * omega[None, :]
    embedding = torch.cat(
        [torch.sin(x_angles), torch.cos(x_angles), torch.sin(y_angles), torch.cos(y_angles)], dim=1
    )
    return embedding.to(torch.float32)
