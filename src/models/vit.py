"""Vision Transformer(ViT)のスクラッチ実装。

Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
ICLR 2021 の式 1〜4 に対応する。

    z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos          (式 1)
    z'_l = MSA(LN(z_{l-1})) + z_{l-1}                                  (式 2)
    z_l = MLP(LN(z'_l)) + z'_l                                         (式 3)
    y = LN(z_L^0)                                                      (式 4)

式 2・3 は 002 の ``EncoderBlock``(``norm_first=True``、正規化前置(Pre-Layer Normalization))
そのものであり、順伝播ネットワーク(Feed-Forward Network)の活性化は原論文どおり GELU
(004 の ``gelu_exact``)、正規化は層正規化(Layer Normalization、002 の ``LayerNormalization``)
とする。分類は [CLS] トークンの最終層の表現 y を、最終正規化の後に線形層(分類ヘッド)に通して行う。

位置埋め込み(Position Embedding)は次の 3 方式を引数で切り替える。

- ``"learned"``: 学習可能な 1 次元の位置埋め込み E_pos ∈ R^{(N+1) × D}(原論文の既定)。
  [CLS] トークンの位置も含む。初期値は標準偏差 0.02 の正規分布(原論文の実装と同じ)。
- ``"sinusoidal_2d"``: 2 次元正弦波の位置埋め込み(Beyer et al., 2022)。学習しない。
  [CLS] トークンには零ベクトルを加える(位置を持たない)。
- ``"none"``: 位置埋め込みを加えない。このとき出力はパッチの並べ替えに対して不変になる
  (019 の理論セクション 3.3 節で証明し、ノートブックで確かめる)。

記号 / Notation:
    P    : パッチサイズ
    N    : パッチの数(系列長。[CLS] を除く)
    D    : d_model(埋め込み・Transformer の隠れ次元)
    L    : Encoder Block の層数
    h    : 多頭注意機構(Multi-Head Attention)のヘッド数
    d_ff : 順伝播ネットワークの中間層の次元
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.layers.activation import gelu_exact
from src.layers.normalization import LayerNormalization
from src.layers.patch_embedding import PatchEmbedding, build_2d_sinusoidal_position_embedding
from src.layers.transformer_block import EncoderBlock

POSITION_EMBEDDING_TYPES = ("learned", "sinusoidal_2d", "none")

# 埋め込みとみなすパラメータの名前の接頭辞(非埋め込みパラメータ数の計算と重み減衰の除外に使う)
EMBEDDING_PARAMETER_PREFIXES = ("patch_embedding.", "class_token", "position_embedding")


class VisionTransformer(nn.Module):
    """Vision Transformer(分類用、[CLS] トークンによる読み出し)。

    Args:
        image_size: 入力画像の一辺(正方形を想定する)。
        patch_size: パッチサイズ P。
        in_channels: 画像のチャネル数 C。
        num_classes: クラス数。
        d_model: 埋め込みの次元 D。
        num_layers: Encoder Block の層数 L。
        num_heads: ヘッド数 h。
        d_ff: 順伝播ネットワークの中間層の次元。
        position_embedding: ``"learned"`` / ``"sinusoidal_2d"`` / ``"none"``。
        dropout: Encoder Block 内の dropout 率(019 の実験では 0)。
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 192,
        num_layers: int = 6,
        num_heads: int = 3,
        d_ff: int = 768,
        position_embedding: str = "learned",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if position_embedding not in POSITION_EMBEDDING_TYPES:
            raise ValueError(
                f"position_embedding は {POSITION_EMBEDDING_TYPES} のいずれか: "
                f"{position_embedding!r}"
            )
        self.position_embedding_type = position_embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches
        self.num_patches = num_patches

        # x_class: 原論文の実装と同じく零で初期化する
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if position_embedding == "learned":
            self.position_embedding = nn.Parameter(torch.empty(1, num_patches + 1, d_model))
            nn.init.normal_(self.position_embedding, std=0.02)
        elif position_embedding == "sinusoidal_2d":
            grid = self.patch_embedding.grid_size
            patch_positions = build_2d_sinusoidal_position_embedding(grid, grid, d_model)
            fixed = torch.cat([torch.zeros(1, d_model), patch_positions], dim=0)  # [CLS] は零
            self.register_buffer("position_embedding", fixed[None], persistent=True)
        else:
            self.position_embedding = None

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout=dropout,
                    norm_first=True,
                    activation_fn=gelu_exact,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = LayerNormalization(d_model)
        self.head = nn.Linear(d_model, num_classes)
        # 分類ヘッドは零で初期化する(原論文の fine-tuning、Beyer et al. と同じ)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def embed(self, images: Tensor) -> Tensor:
        """式 1: パッチ埋め込みの先頭に [CLS] を付け、位置埋め込みを加えた z_0 を返す。

        Args:
            images: 形状 ``(B, C, H, W)``。

        Returns:
            形状 ``(B, N + 1, D)``。位置 0 が [CLS] トークン。
        """
        patches = self.patch_embedding(images)  # (B, N, D)
        class_tokens = self.class_token.expand(patches.size(0), -1, -1)
        tokens = torch.cat([class_tokens, patches], dim=1)
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding
        return tokens

    def forward(
        self, images: Tensor, return_attention_weights: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """分類の logits を返す。

        Args:
            images: 形状 ``(B, C, H, W)``。
            return_attention_weights: True の場合、各層の注意の重み
                (形状 ``(B, h, N + 1, N + 1)`` のリスト、層の順)もあわせて返す。

        Returns:
            logits(形状 ``(B, num_classes)``)。``return_attention_weights=True`` の場合は
            ``(logits, attention_weights)``。
        """
        x = self.embed(images)
        attention_weights: list[Tensor] = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attention_weights:
                attention_weights.append(weights)
        class_representation = self.final_norm(x)[:, 0]  # 式 4: y = LN(z_L^0)
        logits = self.head(class_representation)
        if return_attention_weights:
            return logits, attention_weights
        return logits

    def no_weight_decay_parameter_names(self) -> set[str]:
        """重み減衰を掛けない行列形のパラメータの名前([CLS] トークンと学習可能な位置埋め込み)。"""
        names = {"class_token"}
        if self.position_embedding_type == "learned":
            names.add("position_embedding")
        return names


def count_vit_non_embedding_parameters(model: VisionTransformer) -> int:
    """埋め込み(パッチ埋め込み・[CLS] トークン・位置埋め込み)を除いたパラメータ数を数える。

    パッチサイズ P を変えるとパッチ埋め込み(P^2 C D + D)と学習可能な位置埋め込み
    ((N + 1) D)のパラメータ数が変わり、位置埋め込みの方式を変えると位置埋め込みの有無が
    変わる。それ以外(Encoder Block・最終正規化・分類ヘッド)は条件によらず同一になる。
    """
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith(EMBEDDING_PARAMETER_PREFIXES)
    )
