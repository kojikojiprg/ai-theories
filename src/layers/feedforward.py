"""順伝播ネットワーク(Feed-Forward Network)のスクラッチ実装。

Vaswani et al., "Attention Is All You Need", NeurIPS 2017, 3.3 節の
Position-wise Feed-Forward Networks に対応する。``SwiGLUFeedForwardNetwork``
(004 で追加)は Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202, 2020 の
GLU(Gated Linear Unit)系の変種のうち SwiGLU に対応する
(理論的な導出は``theories/01_foundations/004_normalization_and_activation.ipynb``を参照)。

記号 / Notation:
    d_model : Transformer Block の入出力次元
    d_ff    : 中間層(隠れ層)の次元。原論文では d_ff = 4 * d_model
    W_1 ∈ R^{d_model × d_ff}, b_1 ∈ R^{d_ff} : 1 層目の線形変換
    W_2 ∈ R^{d_ff × d_model}, b_2 ∈ R^{d_model} : 2 層目の線形変換
    W, V ∈ R^{d_model × d_ff'}, W_2 ∈ R^{d_ff' × d_model} : SwiGLU の 3 つの線形変換
        (d_ff' は SwiGLU の中間次元。標準の Feed-Forward Network の d_ff と
        区別するためダッシュを付ける)
    ⊙ : 要素ごとの積(element-wise product)
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from src.layers.activation import swish


class FeedForwardNetwork(nn.Module):
    """位置ごとに独立に適用される 2 層の順伝播ネットワーク。

    FFN(x) = W_2 * activation(W_1 x + b_1) + b_2

    系列の各位置(トークン)に対して同じ重み ``W_1, W_2`` を独立に適用する
    (位置間の情報混合は Multi-Head Attention 側が担い、このモジュールは
    各位置内の非線形変換のみを行う)。

    Args:
        d_model: 入出力の次元。
        d_ff: 中間層の次元。原論文(Vaswani et al., 2017)では d_ff = 4 * d_model。
        dropout: 中間層の出力に適用する dropout 率。
        activation_fn: 中間層の活性化関数を関数として直接渡す。``None``(既定値)の
            場合は原論文(Vaswani et al., 2017)の設定である ``nn.ReLU()`` を使う
            (001・002・003 と同一の挙動)。004 で活性化関数を比較する際は、
            ``src.layers.activation`` のスクラッチ実装(``gelu_exact`` など)を
            ここに直接渡す。活性化関数の指定方法をこの引数 1 つに一本化しており
            (004 で、文字列引数との併存を解消)、``EncoderBlock`` / ``DecoderBlock``
            もこの引数をそのまま透過させる。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation: Callable[[Tensor], Tensor] = (
            activation_fn if activation_fn is not None else nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        """順伝播。

        Args:
            x: 形状 ``(..., d_model)`` の入力。

        Returns:
            x と同じ形状 ``(..., d_model)`` の出力。
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class SwiGLUFeedForwardNetwork(nn.Module):
    """SwiGLU 順伝播ネットワーク(Shazeer, 2020)。

    FFN_SwiGLU(x) = (Swish(x W) ⊙ x V) W_2

    標準の Feed-Forward Network が行列 2 つ(W_1, W_2)なのに対し、
    GLU(Gated Linear Unit、Dauphin et al., 2017)系の変種として行列 3 つ
    (W, V, W_2)を持つ。``Swish(x W)`` が「どれだけ通すか」を決める
    ゲート(gate)として働き、``x V`` を要素ごとに変調する。

    パラメータ数を標準の Feed-Forward Network(2 * d_model * d_ff)と揃えるには、
    中間次元を d_ff' = (2/3) * d_ff にする必要がある(3 * d_model * d_ff' が
    2 * d_model * d_ff と一致する条件から導出、詳細は 004 の理論セクション参照)。
    この丸め込みはノートブック側の責務とし、本クラスは ``d_ff`` を
    そのまま中間次元として使う(呼び出し側が既に丸めた値を渡す想定)。

    Args:
        d_model: 入出力の次元。
        d_ff: 中間次元 d_ff'。パラメータ数を揃えたい場合は呼び出し側で
            ``round((2 / 3) * d_ff_baseline)`` などを渡す。
        beta: Swish の形状パラメータ β。既定値 1.0(= SiLU)。
        bias: 線形変換にバイアス項を持たせるか(原論文は bias なし)。
        dropout: 出力(W_2 適用前)に適用する dropout 率。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        beta: float = 1.0,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.w = nn.Linear(d_model, d_ff, bias=bias)
        self.v = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """順伝播。

        Args:
            x: 形状 ``(..., d_model)`` の入力。

        Returns:
            x と同じ形状 ``(..., d_model)`` の出力。
        """
        gate = swish(self.w(x), self.beta)
        gated = gate * self.v(x)
        return self.w2(self.dropout(gated))
