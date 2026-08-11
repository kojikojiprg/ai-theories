"""順伝播ネットワーク(Feed-Forward Network)のスクラッチ実装。

Vaswani et al., "Attention Is All You Need", NeurIPS 2017, 3.3 節の
Position-wise Feed-Forward Networks に対応する。

記号 / Notation:
    d_model : Transformer Block の入出力次元
    d_ff    : 中間層(隠れ層)の次元。原論文では d_ff = 4 * d_model
    W_1 ∈ R^{d_model × d_ff}, b_1 ∈ R^{d_ff} : 1 層目の線形変換
    W_2 ∈ R^{d_ff × d_model}, b_2 ∈ R^{d_model} : 2 層目の線形変換
"""

from __future__ import annotations

from torch import Tensor, nn


class FeedForwardNetwork(nn.Module):
    """位置ごとに独立に適用される 2 層の順伝播ネットワーク。

    FFN(x) = W_2 * activation(W_1 x + b_1) + b_2

    系列の各位置(トークン)に対して同じ重み ``W_1, W_2`` を独立に適用する
    (位置間の情報混合は Multi-Head Attention 側が担い、このモジュールは
    各位置内の非線形変換のみを行う)。

    Args:
        d_model: 入出力の次元。
        d_ff: 中間層の次元。原論文(Vaswani et al., 2017)では d_ff = 4 * d_model。
        activation: 中間層の活性化関数。"relu"(原論文の設定)または "gelu"。
        dropout: 中間層の出力に適用する dropout 率。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation: nn.Module = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"未対応の activation: {activation!r}('relu' または 'gelu')")

    def forward(self, x: Tensor) -> Tensor:
        """順伝播。

        Args:
            x: 形状 ``(..., d_model)`` の入力。

        Returns:
            x と同じ形状 ``(..., d_model)`` の出力。
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
