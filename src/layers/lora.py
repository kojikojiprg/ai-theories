"""LoRA(Low-Rank Adaptation、低ランク適応)のスクラッチ実装。

Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 に基づく。
凍結した事前学習済みの重み W_0 に、低ランク行列の積による更新量を加える。

    h = W_0 x + (alpha / r) B A x

初期化は公式実装(``microsoft/LoRA`` の ``loralib``、``loralib/layers.py`` の
``Linear.reset_parameters``)に従う: A は ``nn.Linear`` の既定の初期化と同じ
``kaiming_uniform_(a=sqrt(5))``(一様分布 U(-1/sqrt(d_in), 1/sqrt(d_in)))、B は 0。
原論文本文(第 4.1 節)は「A をランダムなガウス分布で初期化する」と書いており、
分布の形が公式実装と異なる。いずれの場合も B = 0 のため、学習開始時の出力は
ベースモデルと一致する。

スケーリングは原論文の alpha / r のみを実装する(Kalajdzievski, 2023 の rsLoRA が
提案する alpha / sqrt(r) は実装しない)。

記号 / Notation:
    d_in  : ベース層の入力次元
    d_out : ベース層の出力次元
    r     : rank(低ランク行列の内側の次元、r << min(d_in, d_out))
    alpha : スケーリング定数(更新量に alpha / r を乗じる)
    A     : 形状 (r, d_in) の学習可能な行列(``lora_a``)
    B     : 形状 (d_out, r) の学習可能な行列(``lora_b``)

使い方 / Usage(012 で導入、016(SFT)などで再利用する)::

    from src.layers.lora import apply_lora

    torch.manual_seed(seed)  # A の初期化の乱数を固定する
    replaced = apply_lora(model, target_module_names=("w_q", "w_v"), rank=8, alpha=8.0)
    # model の LoRA 以外の全パラメータは requires_grad=False になる
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=...)
    ...  # 学習
    for name in replaced:
        model.get_submodule(name).merge()  # 推論用に W = W_0 + (alpha / r) B A へ畳み込む
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def compute_lora_parameter_count(d_in: int, d_out: int, rank: int) -> int:
    """1 行列あたりの LoRA の学習可能パラメータ数 r (d_in + d_out) を返す(閉形式)。"""
    return rank * (d_in + d_out)


class LoRALinear(nn.Module):
    """ベース層を合成(composition)で包み、低ランクの更新量を加える層。

    ベース層は ``nn.Linear`` に限定しない。``forward(x)`` を持ち、入出力次元が
    分かる任意のモジュール(013 で量子化したベース層など)を受け付ける。入出力次元は
    ``in_features``・``out_features`` 引数で与えるか、ベース層の同名の属性から取得する。

    ベース層のパラメータはすべて ``requires_grad=False`` にする(凍結)。

    Args:
        base_layer: 凍結するベース層(``forward(x)`` を持つモジュール)。
        rank: rank r(1 以上)。
        alpha: スケーリング定数 alpha。更新量に alpha / r を乗じる。
        in_features: 入力次元 d_in。``None`` の場合 ``base_layer.in_features`` を使う。
        out_features: 出力次元 d_out。``None`` の場合 ``base_layer.out_features`` を使う。
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank は 1 以上である必要がある: {rank}")
        self.base_layer = base_layer
        self.in_features = in_features if in_features is not None else base_layer.in_features
        self.out_features = out_features if out_features is not None else base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        reference = next(self.base_layer.parameters(), None)
        factory = {}
        if reference is not None and reference.is_floating_point():
            factory = {"device": reference.device, "dtype": reference.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, **factory))
        self.lora_b = nn.Parameter(torch.empty(self.out_features, rank, **factory))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """A を ``kaiming_uniform_(a=sqrt(5))``、B を 0 で初期化する(``loralib`` と同一)。"""
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def delta_weight(self) -> Tensor:
        """更新量 (alpha / r) B A(形状 ``(d_out, d_in)``)を返す。"""
        return self.scaling * (self.lora_b @ self.lora_a)

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base_layer(x)
        if self.merged:
            return base_output
        # B A x を (x A^T) B^T の順に計算し、d_out x d_in の行列を作らない。
        lora_output = (x @ self.lora_a.transpose(0, 1)) @ self.lora_b.transpose(0, 1)
        return base_output + self.scaling * lora_output

    def _require_weight(self) -> Tensor:
        weight = getattr(self.base_layer, "weight", None)
        if not isinstance(weight, Tensor):
            raise TypeError(
                "マージにはベース層が .weight(Tensor)を持つ必要がある: "
                f"{type(self.base_layer).__name__}"
            )
        return weight

    @torch.no_grad()
    def merge(self) -> None:
        """W = W_0 + (alpha / r) B A をベース層の重みに書き込む(推論時の追加遅延をなくす)。"""
        weight = self._require_weight()
        if self.merged:
            return
        weight.add_(self.delta_weight().to(weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """``merge()`` を取り消し、ベース層の重みを W_0 に戻す(浮動小数点の丸め誤差の範囲で)。"""
        weight = self._require_weight()
        if not self.merged:
            return
        weight.sub_(self.delta_weight().to(weight.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}"
        )


def apply_lora(
    model: nn.Module,
    target_module_names: tuple[str, ...] | list[str],
    rank: int,
    alpha: float,
) -> list[str]:
    """``model`` 内の対象モジュールを ``LoRALinear`` で包んで置換し、LoRA 以外の全パラメータを
    凍結する。

    対象は、完全修飾名の末尾の要素(属性名)が ``target_module_names`` のいずれかに
    一致するモジュールである(例: ``("w_q", "w_v")`` は全層の ``self_attn.w_q``・
    ``self_attn.w_v`` に一致する)。A の初期化は PyTorch の大域的な乱数を消費するため、
    再現性が必要な場合は呼び出し前に ``torch.manual_seed`` を呼ぶこと。

    Args:
        model: 置換対象のモデル(その場で書き換える)。
        target_module_names: 置換するモジュールの属性名。
        rank: rank r。
        alpha: スケーリング定数 alpha。

    Returns:
        置換したモジュールの完全修飾名のリスト(``model.get_submodule`` で取得できる)。

    Raises:
        ValueError: 対象のモジュールが 1 つも見つからない場合。
    """
    targets = set(target_module_names)
    matches = [
        name
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets and not isinstance(module, LoRALinear)
    ]
    if not matches:
        raise ValueError(f"対象のモジュールが見つからない: {sorted(targets)}")

    for p in model.parameters():
        p.requires_grad_(False)

    for name in matches:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank=rank, alpha=alpha))
    return matches
