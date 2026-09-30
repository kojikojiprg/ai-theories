"""CLIP(Contrastive Language-Image Pre-training)の二重 encoder(dual encoder)のスクラッチ実装。

Radford et al., "Learning Transferable Visual Models From Natural Language Supervision",
ICML 2021 の構成に従い、画像 encoder とテキスト encoder の出力を共通の埋め込み空間へ線形射影し、
L2 正規化する(020)。

- **画像 encoder**: 019 の ``VisionTransformer`` の ``forward_features()``(最終正規化の後の
  [CLS] トークンの表現 y = LN(z_L^0))を使う。分類ヘッドは使わないので ``nn.Identity`` に置き換える
  (分類ヘッドは零で初期化されていて乱数を消費しないので、置き換えても他のパラメータの初期化は
  変わらない)。
- **テキスト encoder**(``CausalTextTransformer``): トークン埋め込みと学習可能な位置埋め込みの和に、
  002 のブロック(``DecoderBlock``、``use_cross_attention=False``、正規化前置、GELU)を
  因果マスクつきで積み、最終正規化の後、**終端トークン(``<eot>``)の位置の特徴** を取り出す。
  008 の小型 GPT(``GPTLanguageModel``)と同じく、因果マスクつきの decoder-only の Transformer
  であり、違いは(1)語彙への射影(次トークンの予測)の代わりに終端トークンの位置の特徴を
  埋め込みにすること、(2)位置の情報を RoPE ではなく CLIP の原論文と同じ学習可能な絶対位置埋め込みで
  与えることである。因果マスクがあるので、終端トークンより後ろのパディングは終端トークンの特徴に
  影響しない。
- **温度とバイアス**: ``logit_scale`` は CLIP の実装の log(1/τ)(softmax の損失、初期値
  log(1/0.07))または SigLIP(Zhai et al., ICCV 2023)の t'(sigmoid の損失、初期値 log 10)として
  使う。``logit_bias`` は SigLIP のバイアス b(初期値 -10)で、softmax の損失では使わない
  (softmax は行ごとの定数の加算に対して不変なので、バイアスは意味を持たない)。

記号 / Notation:
    d_i, d_t : 画像・テキスト encoder の出力次元(射影の前)
    d_e      : 共通の埋め込みの次元(CLIP の原論文の図 3 の擬似コードの記号)
    tau      : 温度(softmax の損失の logits は exp(logit_scale) * cos 類似度)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.layers.activation import gelu_exact
from src.layers.attention import create_causal_mask
from src.layers.normalization import LayerNormalization
from src.layers.transformer_block import DecoderBlock
from src.models.vit import VisionTransformer


class CausalTextTransformer(nn.Module):
    """因果マスクつきのテキスト encoder(CLIP のテキスト側)。

    Args:
        vocabulary_size: 語彙サイズ。
        context_length: 系列長の上限(学習可能な位置埋め込みの数)。
        d_model: 隠れ次元(出力次元 d_t)。
        num_layers: ブロックの層数。
        num_heads: ヘッド数。
        d_ff: 順伝播ネットワークの中間層の次元。
        end_token_id: 終端トークンの ID(この位置の特徴を取り出す)。
    """

    def __init__(
        self,
        vocabulary_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        end_token_id: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.end_token_id = end_token_id
        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, context_length, d_model))
        # CLIP の実装と同じ初期値(トークン埋め込みは標準偏差 0.02、位置埋め込みは 0.01)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    norm_first=True,
                    activation_fn=gelu_exact,
                    use_cross_attention=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = LayerNormalization(d_model)
        self.register_buffer("causal_mask", create_causal_mask(context_length), persistent=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        """終端トークンの位置の特徴(形状 ``(B, d_t)``)を返す。

        Args:
            token_ids: 形状 ``(B, S)`` の int64(``S <= context_length``)。
                各行に終端トークンがちょうど 1 つある。
        """
        seq_len = token_ids.size(1)
        x = self.token_embedding(token_ids) + self.position_embedding[:, :seq_len]
        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x, _, _ = block(x, None, tgt_mask=mask)
        x = self.final_norm(x)
        end_positions = (token_ids == self.end_token_id).int().argmax(dim=1)
        return x[torch.arange(x.size(0), device=x.device), end_positions]


class CLIPDualEncoder(nn.Module):
    """画像 encoder とテキスト encoder を共通の埋め込み空間に射影する二重 encoder。

    Args:
        vision: 画像 encoder(019 の ``VisionTransformer``。分類ヘッドは置き換える)。
        text: テキスト encoder(``CausalTextTransformer``)。
        embedding_dim: 共通の埋め込みの次元 d_e。
        init_logit_scale: ``logit_scale`` の初期値(softmax は log(1/0.07)、sigmoid は log 10)。
        init_logit_bias: ``logit_bias`` の初期値(sigmoid の損失でのみ使う。SigLIP は -10)。
    """

    def __init__(
        self,
        vision: VisionTransformer,
        text: CausalTextTransformer,
        embedding_dim: int,
        init_logit_scale: float = math.log(1 / 0.07),
        init_logit_bias: float = -10.0,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.vision.head = nn.Identity()  # 分類ヘッドは使わない(forward_features のみを使う)
        self.text = text
        vision_dim = vision.class_token.size(-1)
        text_dim = text.token_embedding.embedding_dim
        # 射影は CLIP と同じくバイアスなし。初期値は CLIP の実装と同じく標準偏差 width^-0.5
        self.image_projection = nn.Parameter(
            torch.randn(vision_dim, embedding_dim) * vision_dim**-0.5
        )
        self.text_projection = nn.Parameter(torch.randn(text_dim, embedding_dim) * text_dim**-0.5)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit_bias)))

    def encode_image(self, images: Tensor) -> Tensor:
        """L2 正規化した画像の埋め込み(形状 ``(B, d_e)``、float32)。"""
        features = self.vision.forward_features(images) @ self.image_projection
        return nn.functional.normalize(features.float(), dim=-1)

    def encode_text(self, token_ids: Tensor) -> Tensor:
        """L2 正規化したテキストの埋め込み(形状 ``(B, d_e)``、float32)。"""
        features = self.text(token_ids) @ self.text_projection
        return nn.functional.normalize(features.float(), dim=-1)

    def forward(self, images: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode_image(images), self.encode_text(token_ids)

    def no_weight_decay_parameter_names(self) -> set[str]:
        """重み減衰を掛けない行列形のパラメータ(位置埋め込みと [CLS] トークン)。"""
        names = {f"vision.{n}" for n in self.vision.no_weight_decay_parameter_names()}
        return names | {"text.position_embedding"}


def count_parameters_by_part(model: CLIPDualEncoder) -> dict[str, int]:
    """画像 encoder・テキスト encoder・射影・温度とバイアスのパラメータ数。"""
    parts = {"vision": 0, "text": 0, "projection": 0, "scale_and_bias": 0}
    for name, p in model.named_parameters():
        if name.startswith("vision."):
            parts["vision"] += p.numel()
        elif name.startswith("text."):
            parts["text"] += p.numel()
        elif name.endswith("_projection"):
            parts["projection"] += p.numel()
        else:
            parts["scale_and_bias"] += p.numel()
    return parts
