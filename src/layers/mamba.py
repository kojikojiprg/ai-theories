"""Mamba の層(選択的な状態空間モデル、Selective State Space Model)のスクラッチ実装(023)。

Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
arXiv:2312.00752, 2023 の定式化(3 節・Algorithm 1・Algorithm 2)に従う。公式の CUDA カーネルを含む
パッケージは使わない。走査(scan)は **逐次ループ** であり、ハードウェアを意識した
並列走査(parallel scan)は実装しない。したがって実測の絶対速度は公式実装を代表しない。

**連続時間の状態空間モデル**(入力 ``x(t)`` から出力 ``y(t)`` への写像、状態 ``h(t)``):

.. math::

    h'(t) = A h(t) + B x(t), \\qquad y(t) = C h(t)

**ゼロ次ホールド(zero-order hold)による離散化**(入力を区間 ``[0, Δ]`` で定数とみなす):

.. math::

    \\bar{A} = \\exp(\\Delta A), \\qquad
    \\bar{B} = (\\Delta A)^{-1} (\\exp(\\Delta A) - I) \\cdot \\Delta B

``A`` が対角のとき、要素ごとに ``Ā = exp(ΔA)``、``B̄ = (exp(ΔA) - 1) / A * B`` になる
(``Δ`` は約分される)。公式実装の簡略形 ``B̄ = ΔB`` は、``B̄`` の ``Δ`` についての 1 次の近似
(``(exp(u) - 1) / u = 1 + u / 2 + ...`` の 1 項目)で、``Δ → 0`` で一致する。簡略形でも
``Ā = exp(ΔA)`` はそのまま使う。

**離散化した再帰**: ``h_t = Ā_t h_{t-1} + B̄_t x_t``、``y_t = C_t h_t``。

**選択性(selectivity)**: 選択的な構成では ``Δ``・``B``・``C`` を入力 ``x_t`` の関数にする
(``Δ_t = softplus(Linear(x_t) + bias)``、``B_t = Linear(x_t)``、``C_t = Linear(x_t)``)。
非選択の構成では ``Δ``・``B``・``C`` は入力によらない学習可能なパラメータで、系は時不変
(linear time-invariant)になる。それ以外の構成要素(射影・畳み込み・ゲート・``A``・``D``)は
2 つの構成で同一である。

記号 / Notation(原論文の表記に従う。ただし原論文はバッチサイズと行列 ``B`` の両方に ``B`` を
使うので、本モジュールのコードではバッチサイズを ``batch`` と書く):
    D (channels) : チャネル数(ブロック内の拡大後のチャネル数 ``d_inner = E * d_model``)
    L (length)   : 系列長
    N (state)    : 状態の次元(``state_dim``)
    E (expand)   : 拡大率(``d_inner = E * d_model``。022 のエキスパート数 ``E`` とは別の量)
    Δ (delta)    : 離散化の刻み幅
    R            : ``Δ`` の低ランク射影の次元(``delta_rank``、原論文の ``delta_rank``)
    K            : 因果的な畳み込みのカーネル幅(``conv_kernel``)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

DISCRETIZATION_METHODS = ("zero_order_hold", "euler")


def _at_least_float32(tensor: Tensor) -> Tensor:
    """FP16・BF16 は FP32 に上げ、FP32・FP64 はそのままにする(FP64 のテストを FP32
    に落とさないため)。"""
    return tensor.to(torch.promote_types(tensor.dtype, torch.float32))


class MambaState(NamedTuple):
    """推論時に受け渡す状態(系列の長さによらない定数サイズ)。

    Attributes:
        conv_buffer: 因果的な畳み込みの入力の直近 ``K - 1`` 個。形状 ``(batch, d_inner, K - 1)``。
        recurrent_state: 状態空間モデルの状態 ``h``。形状 ``(batch, d_inner, N)``。
    """

    conv_buffer: Tensor
    recurrent_state: Tensor


def discretize_state_space(
    delta: Tensor, a: Tensor, b: Tensor, method: str = "zero_order_hold"
) -> tuple[Tensor, Tensor]:
    """対角の ``A`` の連続時間の系を離散化する(``Ā`` と ``B̄`` を全時刻ぶん一括で返す)。

    Args:
        delta: 刻み幅 ``Δ``。形状 ``(batch, length, channels)``(正の値)。
        a: 対角の ``A``(チャネルごとに ``N`` 個の対角成分)。形状 ``(channels, state)``(0 でない値。
            Mamba では負の値)。
        b: 入力行列 ``B``。形状 ``(batch, length, channels, state)`` にブロードキャストできること
            (選択的な構成は ``(batch, length, 1, state)``、非選択の構成は
            ``(1, 1, channels, state)``)。
        method: ``"zero_order_hold"``(ゼロ次ホールド、``B̄ = (exp(ΔA) - 1) / A * B``)または
            ``"euler"``(簡略形、``B̄ = ΔB``)。どちらも ``Ā = exp(ΔA)``。

    Returns:
        ``(a_bar, b_bar)``。どちらも形状 ``(batch, length, channels, state)``。
        入力の型のまま計算する。
    """
    if method not in DISCRETIZATION_METHODS:
        raise ValueError(
            f"method は {DISCRETIZATION_METHODS} のいずれかである必要がある: {method!r}"
        )
    delta_a = delta.unsqueeze(-1) * a  # (batch, length, channels, state)
    a_bar = torch.exp(delta_a)
    # ゼロ次ホールド: (ΔA)^{-1} (exp(ΔA) - I) ΔB = A^{-1} (exp(ΔA) - I) B
    # (対角の A では要素ごとの式になり、Δ は約分される)。簡略形: B̄ = ΔB
    zero_order_hold_b = torch.expm1(delta_a) / a * b
    b_bar = zero_order_hold_b if method == "zero_order_hold" else delta.unsqueeze(-1) * b
    return a_bar, b_bar


def _scan_forward(
    a_bar: Tensor, bx: Tensor, c: Tensor, h0: Tensor, states: Tensor | None
) -> tuple[Tensor, Tensor]:
    """逐次ループの本体。``states`` が与えられたら、全時刻の状態 ``h_t`` をそこへ書き込む。"""
    length = a_bar.size(1)
    a_bar_steps = a_bar.unbind(dim=1)  # 時刻ごとの view
    bx_steps = bx.unbind(dim=1)
    c_index = [t if c.size(1) > 1 else 0 for t in range(length)]
    h = h0
    outputs = []
    for t in range(length):
        h = torch.addcmul(bx_steps[t], a_bar_steps[t], h)  # h_t = Ā_t h_{t-1} + B̄_t x_t
        if states is not None:
            states[:, t] = h
        outputs.append((h * c[:, c_index[t]]).sum(dim=-1))  # y_t = C_t h_t
    return torch.stack(outputs, dim=1), h


class _SelectiveScanFunction(torch.autograd.Function):
    """走査 ``h_t = Ā_t h_{t-1} + B̄_t x_t``、``y_t = C_t h_t`` の順伝播と、手で書いた逆伝播。

    ``Ā_t`` と ``B̄_t x_t`` を入力として受け取り(離散化は呼び出し側の通常の自動微分に任せる)、
    ループだけをこの関数で包む。時刻ごとの ``select`` を自動微分に任せると、逆伝播のたびに
    大きなテンソルを作り直すので、時間を逆順に回す式を直接書く:

    .. math::

        g_{h_t} = g_{y_t} C_t + \\bar{A}_{t+1} g_{h_{t+1}}, \\quad
        g_{\\bar{B}x_t} = g_{h_t}, \\quad g_{\\bar{A}_t} = g_{h_t} h_{t-1}, \\quad
        g_{C_t} = g_{y_t} h_t

    (``g_*`` は損失の ``*`` についての勾配。``g_{C_t}`` は ``C`` のブロードキャストされた
    軸について合計する。)二階微分には対応しない。
    """

    @staticmethod
    def forward(ctx, a_bar: Tensor, bx: Tensor, c: Tensor, h0: Tensor):
        states = torch.empty_like(bx)
        y, final_h = _scan_forward(a_bar, bx, c, h0, states)
        ctx.save_for_backward(a_bar, c, states, h0)
        return y, final_h.clone()

    @staticmethod
    def backward(ctx, grad_y: Tensor, grad_final_h: Tensor | None):
        a_bar, c, states, h0 = ctx.saved_tensors
        length = a_bar.size(1)
        grad_a_bar = torch.empty_like(a_bar)
        grad_bx = torch.empty_like(a_bar)
        grad_c = torch.zeros_like(c)
        g_h = torch.zeros_like(h0) if grad_final_h is None else grad_final_h.clone()
        for t in reversed(range(length)):
            g_y = grad_y[:, t].unsqueeze(-1)  # (batch, channels, 1)
            c_t = c[:, t if c.size(1) > 1 else 0]
            g_h = g_h + g_y * c_t  # この時刻の出力と、次の時刻から伝わる勾配の和
            grad_bx[:, t] = g_h
            h_prev = states[:, t - 1] if t > 0 else h0
            grad_a_bar[:, t] = g_h * h_prev
            g_c = g_y * states[:, t]
            if c.size(0) == 1:
                g_c = g_c.sum(dim=0, keepdim=True)
            if c.size(2) == 1:
                g_c = g_c.sum(dim=1, keepdim=True)
            if c.size(1) > 1:
                grad_c[:, t] = g_c
            else:
                grad_c[:, 0] += g_c
            g_h = g_h * a_bar[:, t]  # h_{t-1} についての勾配に変換する
        return grad_a_bar, grad_bx, grad_c, g_h


def selective_scan(
    x: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    skip_coefficient: Tensor | None = None,
    discretization: str = "zero_order_hold",
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """逐次ループによる走査 ``h_t = Ā_t h_{t-1} + B̄_t x_t``、``y_t = C_t h_t``。

    ``Ā_t`` と ``B̄_t x_t`` は全時刻ぶんを一括で計算し、Python のループの中は状態の更新と出力だけを
    行う。**計算は ``torch.autocast`` の中でも FP32 以上で行う**(入力が FP16・BF16・FP32 なら FP32、
    FP64 なら FP64)。勾配が不要なとき(推論)は、状態を保存しない素のループで計算する。

    Args:
        x: 入力。形状 ``(batch, length, channels)``。
        delta: 刻み幅 ``Δ``。形状 ``(batch, length, channels)``。
        a: 対角の ``A``。形状 ``(channels, state)``。
        b: 入力行列 ``B``。``(batch, length, channels, state)`` にブロードキャストできること。
        c: 出力行列 ``C``。同上。
        skip_coefficient: スキップ接続の係数 ``D``(形状 ``(channels,)``)。指定すると、
            状態空間モデルの出力に ``D * x`` を足す。
        discretization: ``"zero_order_hold"`` または ``"euler"``(``discretize_state_space``
            を参照)。
        initial_state: 初期状態 ``h_0``。形状 ``(batch, channels, state)``。``None`` なら 0。

    Returns:
        ``(y, final_state)``。``y`` の形状は ``(batch, length, channels)``、``final_state`` の形状は
        ``(batch, channels, state)``。
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        dtype = torch.promote_types(x.dtype, torch.float32)
        x, delta, a, b, c = (t.to(dtype) for t in (x, delta, a, b, c))
        batch, length, channels = x.shape
        a_bar, b_bar = discretize_state_space(delta, a, b, discretization)
        bx = b_bar * x.unsqueeze(-1)
        h0 = (
            torch.zeros(batch, channels, a.size(-1), dtype=dtype, device=x.device)
            if initial_state is None
            else initial_state.to(dtype)
        )
        needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (a_bar, bx, c, h0))
        if needs_grad:
            y, h = _SelectiveScanFunction.apply(a_bar, bx, c, h0)
        else:
            y, h = _scan_forward(a_bar, bx, c, h0, None)
        if skip_coefficient is not None:
            y = y + skip_coefficient.to(dtype) * x
    return y, h


def selective_scan_step(
    x_t: Tensor,
    delta_t: Tensor,
    a: Tensor,
    b_t: Tensor,
    c_t: Tensor,
    state: Tensor,
    skip_coefficient: Tensor | None = None,
    discretization: str = "zero_order_hold",
) -> tuple[Tensor, Tensor]:
    """1 ステップぶんの更新(推論用)。``selective_scan`` のループ 1 回と同じ計算をする。

    Args:
        x_t: 形状 ``(batch, channels)``。
        delta_t: 形状 ``(batch, channels)``。
        a: 形状 ``(channels, state)``。
        b_t: 形状 ``(batch, channels, state)`` にブロードキャストできること(``batch`` と
            ``channels`` は 1 でもよい)。
        c_t: 同上。
        state: 直前の状態 ``h_{t-1}``。形状 ``(batch, channels, state)``。
        skip_coefficient: スキップ接続の係数 ``D``。
        discretization: ``"zero_order_hold"`` または ``"euler"``。

    Returns:
        ``(y_t, h_t)``。
    """
    with torch.autocast(device_type=x_t.device.type, enabled=False):
        dtype = torch.promote_types(x_t.dtype, torch.float32)
        x_t, delta_t, a, b_t, c_t, state = (t.to(dtype) for t in (x_t, delta_t, a, b_t, c_t, state))
        a_bar, b_bar = discretize_state_space(
            delta_t.unsqueeze(1), a, b_t.unsqueeze(1), discretization
        )
        h = torch.addcmul(b_bar[:, 0] * x_t.unsqueeze(-1), a_bar[:, 0], state)
        y = (h * c_t).sum(dim=-1)
        if skip_coefficient is not None:
            y = y + skip_coefficient.to(dtype) * x_t
    return y, h


class MambaBlock(nn.Module):
    """Mamba ブロック(原論文 Figure 3)。

    入力 ``(batch, length, d_model)`` を線形射影で ``2 * d_inner`` に拡大して 2 つの枝に分け、一方を
    因果的な depthwise の 1 次元畳み込み → SiLU → 状態空間モデル(走査)に通し、他方を SiLU でゲートに
    して要素ごとに掛け、出力の線形射影で ``d_model`` に戻す。

    Args:
        d_model: モデルの隠れ次元。
        state_dim: 状態の次元 ``N``(原論文の既定値 16)。
        expand: 拡大率 ``E``(``d_inner = E * d_model``、原論文の既定値 2)。
        conv_kernel: 因果的な畳み込みのカーネル幅 ``K``(原論文の既定値 4)。
        delta_rank: ``Δ`` の低ランク射影の次元 ``R``。``None`` なら
            ``ceil(d_model / 16)``(原論文の既定)。
        selective: ``True`` なら ``Δ``・``B``・``C`` を入力の関数にする(選択的な構成)。
            ``False`` なら入力によらない学習可能なパラメータにする(非選択の構成。``Δ`` は
            チャネルごと、``B``・``C`` は ``(d_inner, N)``。原論文 Algorithm 1)。それ以外の
            構成要素は同一。
        discretization: ``"zero_order_hold"``(ゼロ次ホールド、主構成)または ``"euler"``(簡略形
            ``B̄ = ΔB``)。
        delta_min, delta_max, delta_init_floor: ``Δ`` の初期値の範囲。``Δ`` の初期値は
            ``[delta_min, delta_max]`` の対数一様分布から引き(``delta_init_floor`` 未満は
            切り上げ)、``softplus`` の逆関数で ``delta_projection`` のバイアス(非選択の構成では
            ``delta_bias``)に書き込む。
        use_checkpoint: ``True`` なら、学習時(勾配が有効なとき)にブロック全体を再計算する
            (activation checkpointing)。メモリ量を減らす代わりに計算量が増える。値は変わらない。
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        delta_rank: int | None = None,
        selective: bool = True,
        discretization: str = "zero_order_hold",
        delta_min: float = 1e-3,
        delta_max: float = 1e-1,
        delta_init_floor: float = 1e-4,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if discretization not in DISCRETIZATION_METHODS:
            raise ValueError(
                f"discretization は {DISCRETIZATION_METHODS} のいずれか: {discretization!r}"
            )
        self.d_model = d_model
        self.state_dim = state_dim
        self.expand = expand
        self.d_inner = expand * d_model
        self.conv_kernel = conv_kernel
        self.delta_rank = math.ceil(d_model / 16) if delta_rank is None else delta_rank
        self.selective = selective
        self.discretization = discretization
        self.use_checkpoint = use_checkpoint
        self.track_delta = False
        self.last_delta_mean: Tensor | None = (
            None  # 形状 (batch, length): チャネルについて平均した Δ
        )

        self.input_projection = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=conv_kernel, groups=self.d_inner, padding=0
        )
        self.output_projection = nn.Linear(self.d_inner, d_model, bias=False)
        # A = -exp(A_log)。S4D-real の初期化(状態 n の対角成分が -(n + 1))
        a_init = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.skip_coefficient = nn.Parameter(torch.ones(self.d_inner))

        if selective:
            self.selection_projection = nn.Linear(
                self.d_inner, self.delta_rank + 2 * state_dim, bias=False
            )
            self.delta_projection = nn.Linear(self.delta_rank, self.d_inner, bias=True)
            delta_std = self.delta_rank**-0.5
            nn.init.uniform_(self.delta_projection.weight, -delta_std, delta_std)
            delta_bias = self.delta_projection.bias
        else:
            self.delta_bias = nn.Parameter(torch.empty(self.d_inner))
            # S4D の初期化: B = 1、C は標準正規分布
            self.input_matrix = nn.Parameter(torch.ones(self.d_inner, state_dim))
            self.output_matrix = nn.Parameter(torch.randn(self.d_inner, state_dim))
            delta_bias = self.delta_bias
        with torch.no_grad():
            initial_delta = torch.exp(
                torch.rand(self.d_inner) * (math.log(delta_max) - math.log(delta_min))
                + math.log(delta_min)
            ).clamp(min=delta_init_floor)
            delta_bias.copy_(
                initial_delta + torch.log(-torch.expm1(-initial_delta))
            )  # softplus の逆関数

    @property
    def a_matrix(self) -> Tensor:
        """対角の ``A``(形状 ``(d_inner, N)``、負の値)。"""
        return -torch.exp(_at_least_float32(self.A_log))

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> MambaState:
        """ゼロの初期状態を返す。"""
        return MambaState(
            torch.zeros(batch_size, self.d_inner, self.conv_kernel - 1, device=device, dtype=dtype),
            torch.zeros(
                batch_size, self.d_inner, self.state_dim, device=device, dtype=torch.float32
            ),
        )

    def causal_convolution(self, conv_input: Tensor) -> Tensor:
        """因果的な depthwise の 1 次元畳み込み(``nn.Conv1d`` と同じ重みを、カーネル幅 ``K`` 個の
        シフトした積の和として明示的に計算する。バックエンドによらず同じ演算になる)。

        Args:
            conv_input: 先頭に過去の ``K - 1`` 個(またはゼロ)を足した入力。形状
                ``(batch, d_inner, K - 1 + length)``。

        Returns:
            形状 ``(batch, d_inner, length)``。出力 ``t`` は入力 ``t .. t + K - 1`` の重み付き和
            (重み ``w[:, 0, j]`` が入力 ``t + j`` に掛かる。``nn.Conv1d`` の相互相関と同じ)。
        """
        length = conv_input.size(-1) - (self.conv_kernel - 1)
        weight = self.conv1d.weight.squeeze(1)  # (d_inner, K)
        out = self.conv1d.bias.unsqueeze(-1) + conv_input[..., :length] * weight[:, 0:1]
        for j in range(1, self.conv_kernel):
            out = out + conv_input[..., j : j + length] * weight[:, j : j + 1]
        return out

    def selection_parameters(self, x_conv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """畳み込み後の入力から ``Δ``・``B``・``C`` を作る。

        Returns:
            ``(delta, b, c)``。``delta`` は形状 ``(batch, length, d_inner)``、``b``・``c`` は
            ``(batch, length, 1, N)``(選択的な構成)または ``(1, 1, d_inner, N)``(非選択の構成)。
        """
        batch, length, _ = x_conv.shape
        if self.selective:
            projected = self.selection_projection(x_conv)
            delta_raw, b, c = projected.split(
                [self.delta_rank, self.state_dim, self.state_dim], dim=-1
            )
            delta = functional.softplus(_at_least_float32(self.delta_projection(delta_raw)))
            return delta, b.unsqueeze(2), c.unsqueeze(2)
        delta = functional.softplus(_at_least_float32(self.delta_bias)).expand(batch, length, -1)
        return (
            delta,
            self.input_matrix.view(1, 1, *self.input_matrix.shape),
            self.output_matrix.view(1, 1, *self.output_matrix.shape),
        )

    def _forward_impl(
        self, x: Tensor, initial_state: MambaState | None
    ) -> tuple[Tensor, MambaState]:
        batch, length, _ = x.shape
        x_branch, z = self.input_projection(x).chunk(2, dim=-1)
        x_t = x_branch.transpose(1, 2)  # (batch, d_inner, length)
        if initial_state is None:
            conv_prefix = torch.zeros(
                batch, self.d_inner, self.conv_kernel - 1, device=x.device, dtype=x_t.dtype
            )
            recurrent_state = None
        else:
            conv_prefix, recurrent_state = initial_state
            conv_prefix = conv_prefix.to(x_t.dtype)
        conv_input = torch.cat([conv_prefix, x_t], dim=-1)  # 先頭に K - 1 個(過去 or 0)を足す
        x_conv = functional.silu(self.causal_convolution(conv_input).transpose(1, 2))
        delta, b, c = self.selection_parameters(x_conv)
        if self.track_delta and self.selective:
            self.last_delta_mean = delta.detach().mean(dim=-1)
        y, final_h = selective_scan(
            x_conv,
            delta,
            self.a_matrix,
            b,
            c,
            self.skip_coefficient,
            self.discretization,
            recurrent_state,
        )
        out = self.output_projection(y * functional.silu(z))
        new_state = MambaState(
            conv_input[..., -(self.conv_kernel - 1) :].detach(), final_h.detach()
        )
        return out, new_state

    def forward(
        self, x: Tensor, initial_state: MambaState | None = None, return_state: bool = False
    ) -> Tensor | tuple[Tensor, MambaState]:
        """順伝播。

        Args:
            x: 形状 ``(batch, length, d_model)``。
            initial_state: 直前までの状態(``None`` なら 0)。系列の続きを処理するときに使う。
            return_state: ``True`` なら ``(出力, 最後の状態)`` を返す。

        Returns:
            形状 ``(batch, length, d_model)`` の出力(``return_state=True`` なら状態との組)。
        """
        if (
            self.use_checkpoint
            and self.training
            and torch.is_grad_enabled()
            and initial_state is None
        ):
            out, state = checkpoint(self._forward_impl, x, None, use_reentrant=False)
        else:
            out, state = self._forward_impl(x, initial_state)
        return (out, state) if return_state else out

    def step(self, x_t: Tensor, state: MambaState) -> tuple[Tensor, MambaState]:
        """1 トークンぶんの推論(畳み込みのバッファと ``h`` を状態として受け渡す)。

        1 トークンあたりの時間も状態のサイズも系列の長さによらず一定である。

        Args:
            x_t: 形状 ``(batch, d_model)``。
            state: 直前の状態。

        Returns:
            ``(出力 (batch, d_model), 新しい状態)``。
        """
        x_branch, z = self.input_projection(x_t).chunk(2, dim=-1)  # (batch, d_inner)
        window = torch.cat([state.conv_buffer.to(x_branch.dtype), x_branch.unsqueeze(-1)], dim=-1)
        conv_out = (window * self.conv1d.weight.squeeze(1)).sum(dim=-1) + self.conv1d.bias
        x_conv = functional.silu(conv_out)  # (batch, d_inner)
        delta, b, c = self.selection_parameters(x_conv.unsqueeze(1))
        y, h = selective_scan_step(
            x_conv,
            delta[:, 0],
            self.a_matrix,
            b[:, 0],
            c[:, 0],
            state.recurrent_state,
            self.skip_coefficient,
            self.discretization,
        )
        out = self.output_projection(y * functional.silu(z))
        return out, MambaState(window[..., 1:], h)
