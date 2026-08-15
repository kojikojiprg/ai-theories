"""GPT(Generative Pre-trained Transformer)スタイルの decoder-only 自己回帰
言語モデル(Autoregressive Language Model)のスクラッチ実装。

Radford et al., "Improving Language Understanding by Generative Pre-Training", 2018
(GPT)、および Radford et al., "Language Models are Unsupervised Multitask Learners",
2019(GPT-2)の decoder-only アーキテクチャに基づく。トークン埋め込み → 因果マスク
(causal mask)付き Transformer Block(``DecoderBlock``、``use_cross_attention=False``)
の積層 → 最終正規化層 → 出力層(語彙への射影)という構成を、001〜004 でスクラッチ
実装した部品(``MultiHeadAttention``・``DecoderBlock``・``LayerNormalization``)を
再利用して組み立てる。

**正規化前置(Pre-Layer Normalization)を採用する**(``DecoderBlock`` の
``norm_first=True``、002 の知見に従う。深い層でも学習が安定するため)。

**位置エンコーディングは注入方式のみを使う**(``positional_transform``、例えば
003 の ``RotaryPositionEmbedding``)。002 のような正弦波位置エンコーディングや
学習可能な絶対位置埋め込みをモジュール内部に持たない。``positional_transform``
が ``None``(既定値)の場合、モデルは明示的な位置情報を一切持たない(因果マスクに
よる非対称性のみが位置の手がかりになる)。呼び出し側が明示的に位置エンコーディング
方式を選ぶ設計であり、003 の「``self_attn`` 属性を ``positional_transform`` を
指定した ``MultiHeadAttention`` に差し替える」手法をレイヤーごとに適用することで
実現する。

**重み共有(tie_embeddings)**: トークン埋め込み行列と出力層(``lm_head``)の重みを
共有できる(Press & Wolf, "Using the Output Embedding to Improve Language Models",
EACL 2017、Inan et al., "Tying Word Vectors and Word Classifiers", ICLR 2017)。
入力埋め込みと出力射影がいずれも「トークン ID <-> d_model 次元ベクトル」の対応を
学習する点で役割が近く、重みを共有することでパラメータ数を削減しつつ性能を
落とさないことが示されている。

記号 / Notation:
    V         : 語彙サイズ(vocabulary_size)
    d_model   : モデルの隠れ次元
    d_ff      : Feed-Forward Network の中間層次元
    h         : Multi-Head Attention のヘッド数(num_heads)
    L         : Transformer Block の層数(num_layers)
    B         : バッチサイズ
    S         : 系列長(sequence length)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from src.layers.attention import MultiHeadAttention, create_causal_mask
from src.layers.normalization import LayerNormalization
from src.layers.positional_encoding import QueryKeyPositionalTransform
from src.layers.transformer_block import DecoderBlock


class GPTLanguageModel(nn.Module):
    """decoder-only Transformer による自己回帰言語モデル(GPT スタイル)。

    構成: トークン埋め込み -> 因果マスク付き ``DecoderBlock``
    (``use_cross_attention=False``)を ``num_layers`` 段 -> 最終正規化層 ->
    出力層(語彙への射影)。

    Args:
        vocabulary_size: 語彙サイズ V。
        d_model: モデルの隠れ次元。
        num_layers: ``DecoderBlock`` の層数 L。
        num_heads: Multi-Head Attention のヘッド数 h。
        d_ff: Feed-Forward Network の中間層次元。
        max_sequence_length: モデルが扱う最大系列長。``forward`` に渡す系列長
            (``token_ids`` の第 2 次元)がこれを超える場合はエラーにする。
        positional_transform: 各層の ``self_attn`` に注入する位置変換
            (003 の ``QueryKeyPositionalTransform`` のサブクラス、例:
            ``RotaryPositionEmbedding``)。全層で同一インスタンスを共有する
            (RoPE は学習パラメータを持たないため共有して問題ない)。``None``
            (既定値)の場合、位置エンコーディングを一切注入しない。
        normalization_factory: ``DecoderBlock``・最終正規化層にそのまま渡す
            正規化層の factory(004 参照)。``None``(既定値)の場合は
            ``LayerNormalization`` を使う。
        feed_forward_factory: ``DecoderBlock`` にそのまま渡す順伝播ネットワークの
            factory(004 参照)。``None``(既定値)の場合は ReLU を使う
            標準の Feed-Forward Network になる。
        tie_embeddings: True(既定値)の場合、トークン埋め込み行列と出力層
            (``lm_head``)の重みを共有する(Press & Wolf 2017、Inan et al. 2017)。
        dropout: Attention 重み・各サブレイヤー出力に適用する dropout 率。
    """

    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_sequence_length: int,
        positional_transform: QueryKeyPositionalTransform | None = None,
        normalization_factory: Callable[[int], nn.Module] | None = None,
        feed_forward_factory: Callable[[], nn.Module] | None = None,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        # nn.Embedding の既定初期化(標準正規分布、標準偏差 1)は次元でスケールされて
        # おらず、そのままでは埋め込みベクトルのノルムが大きくなりすぎる
        # (norm ~ sqrt(d_model))。tie_embeddings=True の場合、この行列がそのまま
        # 出力層 lm_head の重みにもなるため、スケールの悪い初期化が「ランダム初期化
        # 直後の損失が ln(V) に一致するはず」という前提(実験 A)を崩す
        # (実測: 標準偏差 1 のままだと V=2045 で loss ≈ 54 対 ln(V) ≈ 7.6 という
        # 大きな乖離が生じた)。GPT-2(Radford et al., 2019)にならい標準偏差 0.02 の
        # 正規分布で初期化する。
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                self._build_block(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    normalization_factory,
                    feed_forward_factory,
                    positional_transform,
                )
                for _ in range(num_layers)
            ]
        )
        norm_fn = normalization_factory or LayerNormalization
        self.final_norm = norm_fn(d_model)
        self.lm_head = nn.Linear(d_model, vocabulary_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _build_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        normalization_factory: Callable[[int], nn.Module] | None,
        feed_forward_factory: Callable[[], nn.Module] | None,
        positional_transform: QueryKeyPositionalTransform | None,
    ) -> DecoderBlock:
        """交差注意を持たない ``DecoderBlock`` を 1 層分構築する。

        ``positional_transform`` が指定された場合、003 と同じ手法(``self_attn``
        属性を位置変換を注入した ``MultiHeadAttention`` に差し替える)で組み込む。
        ``DecoderBlock`` 自体は変更しない。
        """
        block = DecoderBlock(
            d_model,
            num_heads,
            d_ff,
            dropout=dropout,
            norm_first=True,
            normalization_factory=normalization_factory,
            feed_forward_factory=feed_forward_factory,
            use_cross_attention=False,
        )
        if positional_transform is not None:
            block.self_attn = MultiHeadAttention(
                d_model, num_heads, dropout=dropout, positional_transform=positional_transform
            )
        return block

    def forward(self, token_ids: Tensor, positions: Tensor | None = None) -> Tensor:
        """順伝播。

        Args:
            token_ids: 形状 ``(B, S)`` の LongTensor(トークン ID 列)。
                ``S`` は ``max_sequence_length`` 以下である必要がある。
            positions: 絶対位置インデックス(形状 ``(S,)``)。``positional_transform``
                に透過する(``None`` のときは 0 から S - 1 までの連番)。

        Returns:
            形状 ``(B, S, V)`` の logits。
        """
        _, seq_len = token_ids.shape
        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"系列長 ({seq_len}) が max_sequence_length "
                f"({self.max_sequence_length}) を超えている"
            )

        hidden = self.token_embedding(token_ids)
        causal_mask = create_causal_mask(seq_len, device=token_ids.device)
        for block in self.blocks:
            hidden, _, _ = block(hidden, None, tgt_mask=causal_mask, positions=positions)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> Tensor:
        """自己回帰的にトークンを生成する。

        ``temperature=0.0`` の場合は貪欲法(greedy decoding、常に最尤トークンを選ぶ)、
        それ以外は temperature sampling(``softmax(logits / temperature)`` から
        多項分布サンプリング)を行う。top-k・top-p(nucleus sampling)・beam search は
        実装しない(008 で扱う)。KV キャッシュを使わないため、生成ステップごとに
        直近 ``max_sequence_length`` トークンの文脈全体を再計算する(トピック 010 で
        KV キャッシュによる高速化を扱う)。

        Args:
            token_ids: 形状 ``(B, S_0)`` の LongTensor(生成の起点となる文脈)。
            max_new_tokens: 生成するトークン数。
            temperature: サンプリング温度。``0.0`` で貪欲法になる。
            seed: サンプリングに使う乱数シード(省略可、再現性のため)。
                ``temperature=0.0`` のときは使われない。

        Returns:
            形状 ``(B, S_0 + max_new_tokens)`` の LongTensor
            (``token_ids`` に生成トークンを連結したもの)。
        """
        if temperature < 0.0:
            raise ValueError(f"temperature は 0 以上である必要がある: {temperature}")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=token_ids.device)
            generator.manual_seed(seed)

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = token_ids[:, -self.max_sequence_length :]
                logits = self(context)
                next_token_logits = logits[:, -1, :]  # (B, V)
                if temperature == 0.0:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                token_ids = torch.cat([token_ids, next_token], dim=1)
        finally:
            self.train(was_training)
        return token_ids
