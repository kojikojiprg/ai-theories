"""全パラメータ更新の学習ループで共通に使う、1 ステップの部品(019)。

画像分類(019)・CLIP(020)・テキスト埋め込み(024)・並べ替え(025・026)の学習ループは、
次の部品を同じ手順で使う。各ループがモジュール内に持っていた同一の処理を、本モジュールに統合した。

1. 行列形の重みにのみ重み減衰を掛けるための、パラメータの分割と 2 つの ``AdamW`` の構築
   (``build_weight_decay_split_optimizers()``)。
2. 損失スケーリングの unscale と、勾配の L2 ノルムの計算(``unscale_gradients_and_compute_norm()``)。
3. gradient clipping(``clip_gradients_to_threshold()``。ノルムが閾値を超えたら
   ``threshold / norm`` 倍する。007 と同じ定義)。
4. 更新(非有限値の勾配のステップは飛ばす)と、損失スケールの記録・更新
   (``step_optimizers_and_update_loss_scale()``)。
5. 学習後の、gradient clipping が発動したステップの判定(``flag_clipped_steps()``)。

各関数は、統合の前のループの中の記述と同じ演算を同じ順序で行う。

検証の範囲: CUDA・FP16 の autocast(自動混合精度)・動的損失スケーリングの経路は、CPU の学習ループでは
通らない。この経路のうち ``step_optimizers_and_update_loss_scale()`` は、統合の前のコードを写した
参照実装との比較で確かめた(``scripts/test_src_refactor_equivalence.py``。CPU で、損失スケーリングの
あり・なしと、非有限の勾配のあり・なしを比べた)。実際の旧コードとの GPU での比較はしていない。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler


def split_weight_decay_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """重み減衰を掛けるパラメータと掛けないパラメータに分ける。

    行列形(2 階以上)の重み(線形層・畳み込み層の重み)に重み減衰を掛け、バイアス・正規化層の
    パラメータ(1 階)と、モデルが ``no_weight_decay_parameter_names()`` で指定する行列形の
    パラメータ(ViT の [CLS] トークン・学習可能な位置埋め込み)には掛けない。

    導入: 019。

    Args:
        model: 分割するモデル。``requires_grad`` が偽のパラメータは除く。

    Returns:
        (重み減衰を掛けるパラメータ, 掛けないパラメータ)。
    """
    excluded = set()
    if hasattr(model, "no_weight_decay_parameter_names"):
        excluded = set(model.no_weight_decay_parameter_names())
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and name not in excluded:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return decay, no_decay


def build_weight_decay_split_optimizers(
    model: nn.Module, peak_learning_rate: float, weight_decay: float, foreach: bool = True
) -> tuple[list[AdamW], list[nn.Parameter]]:
    """重み減衰を掛ける ``AdamW`` と掛けない ``AdamW`` を作る(026 までのループに共通)。

    導入: 019。

    Args:
        model: 学習するモデル(``split_weight_decay_parameters()`` で分割する)。
        peak_learning_rate: 両方の ``AdamW`` の初期の学習率。
        weight_decay: 行列形の重みにのみ掛ける重み減衰。
        foreach: ``AdamW`` の更新を ``torch._foreach_*`` でまとめて行うか(既定は真。019 の画像分類の
            ループだけが偽で呼ぶ)。

    Returns:
        ``([重み減衰ありの AdamW, 重み減衰なしの AdamW], 全パラメータ(重み減衰ありを先頭に連結))``。
    """
    decay, no_decay = split_weight_decay_parameters(model)
    optimizers = [
        AdamW(decay, lr=peak_learning_rate, weight_decay=weight_decay, foreach=foreach),
        AdamW(no_decay, lr=peak_learning_rate, weight_decay=0.0, foreach=foreach),
    ]
    return optimizers, decay + no_decay


def unscale_gradients_and_compute_norm(
    parameters: list[nn.Parameter], scale: float | None
) -> Tensor:
    """勾配をその場でスケール値で除し、全パラメータの勾配を連結した L2 ノルムを返す。

    ``scale`` が ``None`` のときは除算をしない。返り値は GPU 上のスカラー(CPU に読み出さない)。

    導入: 019。

    Args:
        parameters: 勾配を持つパラメータ(勾配が ``None`` のものは除く)。
        scale: 損失スケール値(``None`` なら損失スケーリングを使わない)。

    Returns:
        形状 ``()`` の L2 ノルム。
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if scale is not None:
        torch._foreach_mul_(grads, 1.0 / scale)
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))


def clip_gradients_to_threshold(
    parameters: list[nn.Parameter], gradient_norm: Tensor, threshold: Tensor
) -> None:
    """ノルムが閾値を超えたら、勾配を ``threshold / norm`` 倍する(007 と同じ定義)。

    CPU に読み出さずに行う(``torch.where`` で係数を選ぶ)。

    導入: 019。

    Args:
        parameters: 勾配を持つパラメータ。
        gradient_norm: クリッピング前の全パラメータの勾配の L2 ノルム(形状 ``()``)。
        threshold: 閾値(``gradient_norm`` と同じデバイスの形状 ``()`` のテンソル)。
    """
    clip_coefficient = torch.where(
        gradient_norm > threshold, threshold / gradient_norm, torch.ones_like(gradient_norm)
    )
    torch._foreach_mul_([p.grad for p in parameters if p.grad is not None], clip_coefficient)


def step_optimizers_and_update_loss_scale(
    optimizers: Sequence[AdamW],
    gradient_norm: Tensor,
    loss_scaler: DynamicLossScaler | None,
    history: dict,
) -> bool:
    """勾配が有限なら全 ``optimizers`` で更新し、損失スケールを記録して更新する。

    損失スケーリングを使うとき(``loss_scaler`` が ``None`` でないとき)だけ、勾配のノルムが有限かを
    CPU に読み出して確かめる(ここで 1 回だけ GPU と同期する)。非有限なら更新を飛ばす。

    導入: 019。

    Args:
        optimizers: 更新する ``AdamW`` のリスト。
        gradient_norm: クリッピングの前のノルム(有限かどうかの判定に使う)。
        loss_scaler: 動的損失スケーラー(``None`` なら使わない)。
        history: ``"loss_scale"`` のリストを持つ学習の記録。このステップの損失スケール(使わない
            ときは 1.0)を追記する。

    Returns:
        更新を飛ばしたか(``found_inf``)。
    """
    found_inf = False
    if loss_scaler is not None:
        found_inf = not bool(torch.isfinite(gradient_norm))
    if not found_inf:
        for optimizer in optimizers:
            optimizer.step()
    if loss_scaler is not None:
        history["loss_scale"].append(loss_scaler.scale)
        loss_scaler.update(found_inf)
    else:
        history["loss_scale"].append(1.0)
    return found_inf


def flag_clipped_steps(gradient_norms: Sequence[float], threshold: float) -> list[bool]:
    """ステップごとに、gradient clipping が発動したか(ノルムが閾値を超えたか)を返す。

    導入: 019。

    Args:
        gradient_norms: ステップごとのクリッピング前のノルム。
        threshold: gradient clipping の閾値。

    Returns:
        ステップごとの bool のリスト(``gradient_norms`` と同じ長さ)。
    """
    return [g > threshold for g in gradient_norms]
