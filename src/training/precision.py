"""損失スケーリング(loss scaling)と FP32 マスター重み(master weights)の
スクラッチ実装(011)。

Micikevicius et al., "Mixed Precision Training", ICLR 2018 の手法をスクラッチで
実装する。演算ごとの精度割り当て(行列積は低精度、softmax・正規化層・損失の
reduction は FP32 で行う)自体は ``torch.autocast`` に委ねる(理論としては
011 ノートブックの理論セクションで説明し、ここでは実装しない)。

記号 / Notation:
    S (scale) : 損失に乗じるスケール値(loss scaling factor)
    g         : パラメータの勾配(``parameter.grad``)
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class StaticLossScaler:
    """静的損失スケーリング(static loss scaling)のスクラッチ実装。

    損失に固定のスケール値 :math:`S` を乗じて逆伝播することで、連鎖律により
    全パラメータの勾配も :math:`S` 倍になる(011 理論セクション 3 節の導出)。
    ``unscale_gradients`` で :math:`S` を除して元のスケールに戻す。

    Args:
        scale: 損失に乗じるスケール値 :math:`S`(2 のべき乗を推奨、011 5 節)。
    """

    def __init__(self, scale: float) -> None:
        if scale <= 0.0:
            raise ValueError(f"scale は正の値である必要がある: {scale}")
        self.scale = float(scale)

    def scale_loss(self, loss: Tensor) -> Tensor:
        """損失に現在のスケール値を乗じる。"""
        return loss * self.scale

    def unscale_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> bool:
        """全パラメータの勾配をその場でスケール値で除し、非有限値の有無を検出する。

        Args:
            parameters: 逆伝播済みのパラメータ(``.grad`` を持つもの)。

        Returns:
            unscale 後の勾配のいずれかに非有限値(NaN または Inf)が含まれていたか。
        """
        found_inf = False
        inv_scale = 1.0 / self.scale
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.detach().mul_(inv_scale)
            if not torch.isfinite(p.grad).all():
                found_inf = True
        return found_inf

    def update(self, found_inf: bool) -> None:
        """静的スケーリングはスケール値を更新しない(インターフェースの整合のためのみ存在)。"""


class DynamicLossScaler:
    """動的損失スケーリング(dynamic loss scaling)のスクラッチ実装。

    非有限値を検出したステップではスケール値を ``backoff_factor`` 倍に縮小し、
    ``growth_interval`` ステップ連続で非有限値が出なければ ``growth_factor`` 倍に
    拡大する(011 理論セクション 3 節)。``torch.cuda.amp.GradScaler`` と同じ設計
    (backoff は即時、growth は連続成功ステップ数のカウンタで管理する)だが、
    optimizer ごとの状態管理などは持たない最小限の実装である。

    Args:
        init_scale: 初期スケール値。
        growth_factor: growth 発生時にスケール値へ乗じる係数(1 より大きい)。
        backoff_factor: backoff 発生時にスケール値へ乗じる係数(0 より大きく 1 未満)。
        growth_interval: この回数だけ連続して非有限値が出なければ growth する。
    """

    def __init__(
        self,
        init_scale: float,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 100,
    ) -> None:
        if init_scale <= 0.0:
            raise ValueError(f"init_scale は正の値である必要がある: {init_scale}")
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._consecutive_finite_steps = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        """損失に現在のスケール値を乗じる。"""
        return loss * self.scale

    def unscale_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> bool:
        """全パラメータの勾配をその場でスケール値で除し、非有限値の有無を検出する。

        Args:
            parameters: 逆伝播済みのパラメータ(``.grad`` を持つもの)。

        Returns:
            unscale 後の勾配のいずれかに非有限値(NaN または Inf)が含まれていたか。
        """
        found_inf = False
        inv_scale = 1.0 / self.scale
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.detach().mul_(inv_scale)
            if not torch.isfinite(p.grad).all():
                found_inf = True
        return found_inf

    def update(self, found_inf: bool) -> None:
        """スケール値を backoff または growth する。

        非有限値を検出したステップでは即座に ``backoff_factor`` 倍し、連続成功
        カウンタを 0 にリセットする。検出しなかった場合はカウンタを 1 増やし、
        ``growth_interval`` に達したら ``growth_factor`` 倍してカウンタをリセットする。
        """
        if found_inf:
            self.scale *= self.backoff_factor
            self._consecutive_finite_steps = 0
        else:
            self._consecutive_finite_steps += 1
            if self._consecutive_finite_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._consecutive_finite_steps = 0


class MasterWeightOptimizer:
    """FP16 のモデル重みと FP32 の正本(master weights)を同期する optimizer のラッパー。

    ``model`` のパラメータ(FP16 で保持する想定)とは別に、同じ形状の FP32 の複製
    (master weights)を保持し、勾配の蓄積・optimizer の更新は master weights 側で
    行う。更新後、master weights を ``model`` のパラメータの dtype へキャストして
    コピーする。これにより、更新量が小さく FP16 の加算では丸め落ちてしまう場合でも
    (011 理論セクション 4 節)、正本である FP32 側には更新が正しく反映される。

    Args:
        model: FP16 で保持するパラメータを持つモデル。
        optimizer_factory: master weights のリストを受け取り optimizer を返す callable
            (例: ``functools.partial(torch.optim.SGD, lr=0.1)``)。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_factory,
    ) -> None:
        self.model = model
        self.master_params: list[Tensor] = [
            p.detach().clone().float().requires_grad_(True) for p in model.parameters()
        ]
        self.optimizer = optimizer_factory(self.master_params)

    def zero_grad(self) -> None:
        """master weights の勾配をリセットする。"""
        self.optimizer.zero_grad()

    def sync_gradients_to_master(self) -> None:
        """``model`` のパラメータ(FP16)の勾配を FP32 にキャストして master weights へコピーする。"""
        for master_p, model_p in zip(self.master_params, self.model.parameters(), strict=True):
            if model_p.grad is None:
                master_p.grad = None
            else:
                grad_fp32 = model_p.grad.detach().float()
                if master_p.grad is None:
                    master_p.grad = grad_fp32.clone()
                else:
                    master_p.grad.copy_(grad_fp32)

    @torch.no_grad()
    def step(self) -> None:
        """master weights(FP32)を 1 ステップ更新し、``model`` のパラメータへ反映する。

        更新は FP32 の master weights 上で行われるため、FP16 では丸め落ちる小さな
        更新量も正しく蓄積される。更新後、master weights を ``model`` のパラメータの
        dtype(FP16 を想定)へキャストしてコピーする(このキャストで初めて FP16 の
        丸めが 1 回だけ発生する。累積は常に FP32 側で行われる)。
        """
        self.optimizer.step()
        for master_p, model_p in zip(self.master_params, self.model.parameters(), strict=True):
            model_p.copy_(master_p.to(model_p.dtype))
