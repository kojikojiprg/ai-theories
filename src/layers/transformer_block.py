"""Transformer Block(Encoder Block / Decoder Block)のスクラッチ実装。

Vaswani et al., "Attention Is All You Need", NeurIPS 2017 の Encoder / Decoder 層に、
残差接続(Residual Connection、He et al., 2016)と層正規化(Layer Normalization、
Ba et al., 2016)を組み合わせたブロックを実装する。

正規化前置(Pre-Layer Normalization)と正規化後置(Post-Layer Normalization)の
どちらの構造を使うかは ``norm_first`` 引数で切り替える(Xiong et al., 2020 の議論に対応)。

004 で ``normalization_factory``・``feed_forward_factory`` 引数を追加し、
正規化層・順伝播ネットワークの実装(RMSNorm、SwiGLUFeedForwardNetwork など)を
ブロック内部の注入点として差し替え可能にした(003 で ``MultiHeadAttention`` に
``positional_transform`` 等を注入したのと同じ設計方針。ラッパークラスを増やすのではなく、
注入点がブロック内部にあるため optional 引数で拡張する)。

006 で ``DecoderBlock`` に ``use_cross_attention`` 引数を追加し、交差注意
(cross-attention)の副層とそれに対応する正規化層を生成しないモードを設けた
(decoder-only な自己回帰言語モデルの構築に ``DecoderBlock`` をそのまま使えるように
するため。004 までは Encoder-Decoder 構成を持たない decoder-only モデルには
``EncoderBlock`` に因果マスクを渡す方式を使っていたが、``DecoderBlock`` という
名前のクラスを decoder-only 構成に使えないのは紛らわしいため、006 でこの用途にも
``DecoderBlock`` を使えるようにした)。

記号 / Notation:
    d_model : ブロックの入出力次元
    h       : Multi-Head Attention のヘッド数
    d_ff    : Feed-Forward Network の中間層次元
"""

from __future__ import annotations

from collections.abc import Callable

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
        norm_first: True で正規化前置(Pre-Layer Normalization)、
            False で正規化後置(Post-Layer Normalization)。
        normalization_factory: 正規化層を生成する callable(004 で追加)。
            ``d_model`` を受け取り ``nn.Module`` を返す(例: ``RMSNorm`` を使いたい場合は
            ``lambda d: RMSNorm(d)``)。``None``(既定値)の場合は従来通り
            ``LayerNormalization`` を使う。
        feed_forward_factory: Feed-Forward Network を生成する引数なしの callable
            (004 で追加)。``nn.Module`` を返す(例: ``SwiGLUFeedForwardNetwork`` を
            使いたい場合は ``functools.partial`` や ``lambda`` で ``d_model``・``d_ff``
            を束縛して渡す)。``None``(既定値)の場合は従来通り ``activation_fn=None``
            の ``FeedForwardNetwork``(ReLU)を使う。
        activation_fn: ``feed_forward_factory`` が ``None`` のときに使う
            ``FeedForwardNetwork`` の活性化関数(004 で追加、``FeedForwardNetwork``
            にそのまま透過する)。``None``(既定値)の場合は ReLU
            (001・002・003 と同一の挙動)。``feed_forward_factory`` が指定された
            場合はこの引数は無視される。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_first: bool = True,
        normalization_factory: Callable[[int], nn.Module] | None = None,
        feed_forward_factory: Callable[[], nn.Module] | None = None,
        activation_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = (
            feed_forward_factory()
            if feed_forward_factory is not None
            else FeedForwardNetwork(d_model, d_ff, dropout=dropout, activation_fn=activation_fn)
        )
        norm_fn = normalization_factory or LayerNormalization
        self.norm1 = norm_fn(d_model)
        self.norm2 = norm_fn(d_model)
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
    Encoder の出力(memory)から作られる。``use_cross_attention=False``(006 で追加)
    を指定すると交差注意の副層を持たない decoder-only 構成になる(下記の数式は
    ``use_cross_attention=True`` の場合。False の場合は CrossAttention の行が
    そのまま省かれる)。

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
        norm_first: True で正規化前置、False で正規化後置。
        normalization_factory: 正規化層を生成する callable(004 で追加)。
            ``d_model`` を受け取り ``nn.Module`` を返す。``None``(既定値)の場合は
            従来通り ``LayerNormalization`` を使う(詳細は ``EncoderBlock`` の説明を参照)。
        feed_forward_factory: Feed-Forward Network を生成する引数なしの callable
            (004 で追加)。``None``(既定値)の場合は従来通り ``activation_fn=None``
            の ``FeedForwardNetwork``(ReLU)を使う(詳細は ``EncoderBlock`` の説明を参照)。
        activation_fn: ``feed_forward_factory`` が ``None`` のときに使う
            ``FeedForwardNetwork`` の活性化関数(004 で追加)。``None``(既定値)の
            場合は ReLU(001・002・003 と同一の挙動)。
        use_cross_attention: True(既定値)で従来通り交差注意の副層(``cross_attn``・
            ``norm2``)を持つ Encoder-Decoder 構成にする。False の場合、これらを
            一切生成しない(パラメータを持たない)decoder-only 構成になり、
            ``forward()`` の ``memory`` は ``None`` のみを受け付ける(006 で追加。
            自己回帰言語モデルの構築に交差注意が不要なため)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_first: bool = True,
        normalization_factory: Callable[[int], nn.Module] | None = None,
        feed_forward_factory: Callable[[], nn.Module] | None = None,
        activation_fn: Callable[[Tensor], Tensor] | None = None,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.use_cross_attention = use_cross_attention

        # 004 までと乱数消費の順序を完全に一致させるため(self_attn -> cross_attn ->
        # feed_forward -> 各正規化層 -> 各 dropout の順で構築する)、
        # use_cross_attention=True(既定値)のときに 002〜004 と bit-identical な
        # パラメータ初期化になるようにする(正規化層・dropout はパラメータを乱数
        # 初期化しないため実際に順序が影響するのは self_attn・cross_attn・
        # feed_forward の 3 つだが、念のため全体の構築順序も維持する)。
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        if use_cross_attention:
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        else:
            self.cross_attn = None
        self.feed_forward = (
            feed_forward_factory()
            if feed_forward_factory is not None
            else FeedForwardNetwork(d_model, d_ff, dropout=dropout, activation_fn=activation_fn)
        )
        norm_fn = normalization_factory or LayerNormalization
        self.norm1 = norm_fn(d_model)
        self.norm2 = norm_fn(d_model) if use_cross_attention else None
        self.norm3 = norm_fn(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) if use_cross_attention else None
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor | None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Decoder Block の順伝播。

        Args:
            x: 形状 ``(B, S_tgt, d_model)`` の Decoder 側入力。
            memory: 形状 ``(B, S_src, d_model)`` の Encoder 出力。交差注意の
                Key / Value になる。``use_cross_attention=False`` の場合は
                ``None`` のみを受け付ける(``None`` 以外が渡された場合はエラーにする)。
            tgt_mask: 自己注意に渡すマスク。自己回帰生成では因果マスクを渡す。
            memory_mask: 交差注意に渡すマスク(Encoder 側のパディングマスクなど)。
                ``use_cross_attention=False`` の場合は使われない。
            positions: 自己注意の Query 側絶対位置インデックス(006 で追加、
                ``MultiHeadAttention.forward`` にそのまま透過する)。``self_attn`` に
                ``positional_transform``(RoPE 等)が注入されている場合にのみ意味を持つ。
                ``None``(既定値)のときは 0 から S_tgt - 1 までの連番として扱われ、
                001〜004 と同一の挙動になる。KV キャッシュを用いた逐次推論
                (トピック 010)で、生成の各ステップにおける絶対位置を外部から
                指定できるようにするための引数。

        Returns:
            (output, self_attn_weights, cross_attn_weights) のタプル。
            output は ``(B, S_tgt, d_model)``、
            self_attn_weights は ``(B, h, S_tgt, S_tgt)``、
            cross_attn_weights は ``(B, h, S_tgt, S_src)``
            (``use_cross_attention=False`` の場合は ``None``)。
        """
        if not self.use_cross_attention and memory is not None:
            raise ValueError(
                "use_cross_attention=False の DecoderBlock は memory=None のみ受け付ける"
            )

        if self.norm_first:
            normed = self.norm1(x)
            self_out, self_attn_weights = self.self_attn(
                normed, normed, normed, tgt_mask, positions
            )
            x = x + self.dropout1(self_out)

            cross_attn_weights = None
            if self.use_cross_attention:
                normed = self.norm2(x)
                cross_out, cross_attn_weights = self.cross_attn(normed, memory, memory, memory_mask)
                x = x + self.dropout2(cross_out)

            x = x + self.dropout3(self.feed_forward(self.norm3(x)))
        else:
            self_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask, positions)
            x = self.norm1(x + self.dropout1(self_out))

            cross_attn_weights = None
            if self.use_cross_attention:
                cross_out, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
                x = self.norm2(x + self.dropout2(cross_out))

            x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x, self_attn_weights, cross_attn_weights
