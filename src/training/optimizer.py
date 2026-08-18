"""AdamW(Decoupled Weight Decay Regularization)のスクラッチ実装。

Loshchilov, Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 に基づく。
比較対象として、重み減衰(weight decay)を勾配に混ぜる方式(L2 正則化、Adam の
「素朴な」重み減衰の実装、Kingma & Ba, "Adam: A Method for Stochastic Optimization",
ICLR 2015 の枠組みでよく使われる方式)の ``AdamWithL2Regularization`` も実装する。

両クラスは同一のインターフェース(``step()``・``zero_grad()``)を持ち、
``src/training/trainer.py`` の ``train_language_model()`` から交換可能に使える
(``torch.optim.Optimizer`` のサブクラスにはしないが、``step()``・``zero_grad()``
という最小限の呼び出し規約は合わせている)。

**AdamW と Adam(L2 正則化)の違い**: どちらも 1 次・2 次モーメント推定
(m: 勾配の指数移動平均、v: 勾配の二乗の指数移動平均)を使う点は同じだが、
重み減衰をどこに適用するかが異なる。

- ``AdamWithL2Regularization``: 重み減衰 λθ を **勾配に加算してから** モーメント推定に
  入れる(``g <- g + λθ``)。この結果、重み減衰の実効的な強度が 2 次モーメント
  推定 v によるスケーリング(``1/sqrt(v)``)の影響を受けてしまう。勾配の二乗が
  大きいパラメータ群ほど、同じ名目上の λ でも実際に適用される減衰は弱くなる。
- ``AdamW``: 重み減衰をモーメント推定を **介さずに** パラメータへ直接適用する
  (``θ <- θ - η λ θ`` をパラメータ更新式に別項として加える)。これにより、
  重み減衰の実効強度がパラメータ群間の勾配スケールから独立する
  (``compute_effective_decay_divergence``、``src/utils/statistics.py`` で直接検証する)。

記号 / Notation:
    θ (theta)     : パラメータ
    g             : 勾配(``theta.grad``)
    η (eta)       : 学習率(learning rate)
    β1, β2 (beta) : モーメント推定の指数移動平均の減衰率
    m             : 1 次モーメント推定(勾配の指数移動平均)
    v             : 2 次モーメント推定(勾配の二乗の指数移動平均)
    ε (eps)       : 数値安定化のための微小定数
    λ (lambda)    : 重み減衰係数(weight decay)
    t             : ステップ番号(1-indexed、バイアス補正に使う)
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class AdamW:
    """AdamW(decoupled weight decay)のスクラッチ実装。

    パラメータ更新式(ステップ ``t``、1-indexed):

    .. math::

        m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t &= m_t / (1 - \\beta_1^t) \\\\
        \\hat{v}_t &= v_t / (1 - \\beta_2^t) \\\\
        u_t &= \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon) \\\\
        \\theta_t &= \\theta_{t-1} - \\eta (u_t + \\lambda \\theta_{t-1})

    重み減衰の項 :math:`\\eta \\lambda \\theta_{t-1}` は :math:`\\hat{m}_t`・:math:`\\hat{v}_t`
    を一切経由しない(``torch.optim.AdamW`` と同一の定式化)。

    Args:
        params: 最適化対象のパラメータ(``nn.Parameter`` のイテラブル)。
        lr: 学習率 η。
        betas: モーメント推定の減衰率 (β1, β2)。
        eps: 数値安定化のための微小定数 ε。
        weight_decay: 重み減衰係数 λ。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params: list[torch.nn.Parameter] = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self._m: dict[int, Tensor] = {}
        self._v: dict[int, Tensor] = {}

    def zero_grad(self) -> None:
        """全パラメータの勾配を ``None`` にリセットする。"""
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        """全パラメータを 1 ステップ更新する。勾配が ``None`` のパラメータは無視する。"""
        self.step_count += 1
        t = self.step_count
        bias_correction1 = 1.0 - self.beta1**t
        bias_correction2 = 1.0 - self.beta2**t

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            key = id(p)
            if key not in self._m:
                self._m[key] = torch.zeros_like(p)
                self._v[key] = torch.zeros_like(p)
            m, v = self._m[key], self._v[key]

            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # 重み減衰は θ_{t-1} に対して乗法的に適用する(θ <- θ * (1 - ηλ))。
            # p.add_(u) の後に別項として p.add_(p, alpha=-lr*wd) を呼ぶと、
            # 2 回目の呼び出しがすでに更新済みの p を参照してしまい、θ_{t-1} では
            # なく θ_t に対して減衰をかけることになる(θ を 2 回連続でその場更新
            # (in-place)するために生じる誤り)。乗法的な適用はこの誤りを避けつつ、
            # ``torch.optim.AdamW`` の実装と数値的に一致する(小規模テストで確認済み)。
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-self.lr)

    def set_learning_rate(self, lr: float) -> None:
        """学習率を外部から更新する(warmup + cosine スケジュールとの併用のため)。"""
        self.lr = lr


class AdamWithL2Regularization:
    """L2 正則化を勾配に混ぜる方式の Adam(比較対象、AdamW ではない)のスクラッチ実装。

    パラメータ更新式(ステップ ``t``、1-indexed。重み減衰 λ をまず勾配に加算する点が
    ``AdamW`` との唯一の違い):

    .. math::

        \\tilde{g}_t &= g_t + \\lambda \\theta_{t-1} \\\\
        m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) \\tilde{g}_t \\\\
        v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) \\tilde{g}_t^2 \\\\
        \\hat{m}_t &= m_t / (1 - \\beta_1^t) \\\\
        \\hat{v}_t &= v_t / (1 - \\beta_2^t) \\\\
        \\theta_t &= \\theta_{t-1} - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}

    重み減衰の効果が :math:`\\hat{v}_t`(勾配のスケールに依存する 2 次モーメント推定)
    を経由するため、パラメータ群ごとに勾配のスケールが異なると実効的な減衰強度が
    乖離する(``AdamW`` との違いの理論的根拠、``compute_effective_decay_divergence``
    で直接測定する)。

    Args:
        params: 最適化対象のパラメータ(``nn.Parameter`` のイテラブル)。
        lr: 学習率 η。
        betas: モーメント推定の減衰率 (β1, β2)。
        eps: 数値安定化のための微小定数 ε。
        weight_decay: 重み減衰係数 λ(勾配に加算する L2 正則化の強度)。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params: list[torch.nn.Parameter] = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self._m: dict[int, Tensor] = {}
        self._v: dict[int, Tensor] = {}

    def zero_grad(self) -> None:
        """全パラメータの勾配を ``None`` にリセットする。"""
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        """全パラメータを 1 ステップ更新する。勾配が ``None`` のパラメータは無視する。"""
        self.step_count += 1
        t = self.step_count
        bias_correction1 = 1.0 - self.beta1**t
        bias_correction2 = 1.0 - self.beta2**t

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p
            key = id(p)
            if key not in self._m:
                self._m[key] = torch.zeros_like(p)
                self._v[key] = torch.zeros_like(p)
            m, v = self._m[key], self._v[key]

            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            p.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-self.lr)

    def set_learning_rate(self, lr: float) -> None:
        """学習率を外部から更新する(warmup + cosine スケジュールとの併用のため)。"""
        self.lr = lr
