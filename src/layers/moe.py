"""MoE(Mixture of Experts)層のスクラッチ実装(022)。

Transformer Block の Feed-Forward Network(順伝播ネットワーク)を、E 個のエキスパート
(expert)と、トークンごとにエキスパートを選ぶルーター(router)に置き換える。各トークンは
E 個のうち k 個のエキスパートだけを通るので、総パラメータ数を E 倍にしても、トークンあたりの
計算量はエキスパート k 個分(とルーターの分)で済む(条件付き計算、conditional computation)。

対応する論文:

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer", ICLR 2017(top-k のゲート)
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and
  Automatic Sharding", ICLR 2021(top-2、エキスパートの容量とトークンの破棄)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple
  and Efficient Sparsity", JMLR 2022(top-1、負荷分散損失、capacity factor、ルーターの
  FP32 計算、ルーターの初期化の縮小)
- Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", 2022
  (router z-loss)

**エキスパートの形**: 各エキスパートは ``SwiGLUFeedForwardNetwork``(バイアスなし)と同じ形・
同じ大きさで、E 個分の重みを先頭の次元に積んだ 3 つのテンソルとして持つ。初期化は
``nn.Linear`` の既定(``kaiming_uniform_(a=sqrt(5))``)を、エキスパートごとに W・V・W_2 の順で
行う。したがって E = 1 の MoE 層を同じ乱数の状態から構築すると、エキスパートの重みは
``SwiGLUFeedForwardNetwork`` の重みと bit 単位で一致する(ルーターはその後に初期化する)。

**固定形状の振り分け・集約**: トークンを形状 ``(E, C, d_model)``(C はエキスパート 1 個の
容量)のテンソルに振り分け、全エキスパートを一括の行列積(``torch.bmm``)で計算する。
エキスパートごとの Python のループは使わない(E が大きいと、演算の起動の固定費が E に
比例して増えるため)。容量を超えた割り当ては破棄し、その位置の出力は 0 とする(呼び出し側の
残差接続だけが残る)。空いた枠は 0 で埋める(バイアスがないので、0 の入力の出力は 0 になる)。

**破棄の優先順位**: 割り当ては「第 1 候補の全トークン、第 2 候補の全トークン、...」の順に
並べ、同じ候補の中ではトークンの平坦化した順(バッチの先頭の系列から、系列の中では先頭の
位置から)に枠を埋める。あるトークンが破棄されるかどうかは、同じ系列の中ではそれより前の
位置のトークンの割り当てにしか依存しない(因果的な言語モデルで、未来のトークンの情報が
同じ系列の予測に混入しない)。ただし、バッチ内の他の系列には依存する。

**ルーターは FP32 で計算する**: ``torch.autocast`` の中でも、ルーターのロジット・softmax・
補助損失は FP32 で計算する(Fedus et al. の selective precision)。エキスパートの行列積は
autocast の精度に従う。

**補助損失と診断量**: ``forward()`` のたびに、補助損失(係数を掛ける前の値)を
``auxiliary_losses``、直近の呼び出しの診断量を ``last_routing``、トークンごとの割り当て先を
``last_expert_index`` に置く。``track_statistics`` を ``True`` にすると、学習時・評価時の
呼び出しごとの個数を別々に積算する(``routing_statistics()`` で読み、
``reset_statistics()`` で消す)。

記号 / Notation(Fedus et al. の表記に従う):
    N        : エキスパート数(コードでは ``num_experts``。docstring では、トークン数と
               区別するため E とも書く)
    T        : 1 回の呼び出しのトークン数(バッチサイズ x 系列長)
    k        : 1 トークンが通るエキスパートの数(``top_k``)
    W_r      : ルーターの重み(形状 ``(E, d_model)``、バイアスなし)
    h(x)     : ルーターのロジット W_r x
    p_i(x)   : ルーターの確率 softmax(h(x))_i
    f_i      : エキスパート i に割り当てられたトークンの割合(破棄の前、割り当て k T 個のうち)
    P_i      : ルーターの確率 p_i(x) のトークン平均
    C        : エキスパート 1 個の容量 ceil(capacity_factor * k * T / E)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.layers.activation import swish

LOAD_BALANCING_LOSS_NAME = "load_balancing"
ROUTER_Z_LOSS_NAME = "router_z"


def compute_expert_capacity(
    num_tokens: int, num_experts: int, top_k: int, capacity_factor: float
) -> int:
    """エキスパート 1 個の容量 C = ceil(capacity_factor * k * T / E) を返す。

    ``capacity_factor = 1`` は、割り当てが完全に一様なときにちょうど全部が収まる大きさである
    (Fedus et al., 2.2 節: expert capacity = (tokens per batch / number of experts) x
    capacity factor)。1 個のエキスパートが受け取れる割り当ては高々 T 個(各トークンは同じ
    エキスパートを 2 回選ばない)なので、T を上限とする。
    """
    capacity = math.ceil(capacity_factor * top_k * num_tokens / num_experts)
    return max(1, min(capacity, num_tokens))


def compute_dispatch_slots(
    expert_index: Tensor, num_experts: int, capacity: int
) -> tuple[Tensor, Tensor]:
    """各割り当てを、形状 ``(E, C)`` の枠の通し番号に対応させる。

    割り当て a(エキスパート ``expert_index[a]``)は、同じエキスパートへの割り当てのうち
    自分より前にあるものの個数を r として、r < C なら枠 ``expert_index[a] * C + r`` に入る。
    r >= C の割り当ては破棄し、枠の番号を ``E * C``(どのエキスパートにも属さない捨て枠)に
    する。

    Args:
        expert_index: 形状 ``(A,)`` の LongTensor(割り当てごとのエキスパートの番号。
            並び順が破棄の優先順位になる)。
        num_experts: エキスパート数 E。
        capacity: エキスパート 1 個の容量 C。

    Returns:
        (slot, keep) のタプル。``slot`` は形状 ``(A,)`` の LongTensor(枠の通し番号)、
        ``keep`` は形状 ``(A,)`` の bool Tensor(破棄されなかった割り当てが True)。
    """
    one_hot = torch.nn.functional.one_hot(expert_index, num_experts)  # (A, E)
    # 同じエキスパートへの割り当ての中での、0 始まりの順位
    rank = (one_hot.cumsum(dim=0) - 1).gather(1, expert_index[:, None]).squeeze(1)
    keep = rank < capacity
    slot = torch.where(
        keep, expert_index * capacity + rank, torch.full_like(rank, num_experts * capacity)
    )
    return slot, keep


class MixtureOfExpertsFeedForward(nn.Module):
    """top-k ルーティングの MoE 層(エキスパートは SwiGLU の順伝播ネットワーク)。

    y = sum_{i in TopK(p(x))} g_i(x) E_i(x),   E_i(x) = (Swish(x W_i) ⊙ x V_i) W_{2,i}

    ゲート g_i(x) は、既定ではルーターの確率 p_i(x) そのもの(Fedus et al. の式 2)。
    ``renormalize_gates=True`` では、選ばれた k 個の確率の和で割る(Shazeer et al.・
    Lepikhin et al. の top-k のゲート)。k = 1 で和で割るとゲートが恒等的に 1 になり、
    ルーターに勾配が流れなくなるので、その組み合わせは受け付けない。

    Args:
        d_model: 入出力の次元。
        d_ff: 各エキスパートの中間次元(``SwiGLUFeedForwardNetwork`` の ``d_ff`` と同じ意味)。
        num_experts: エキスパート数 E。
        top_k: 1 トークンが通るエキスパートの数 k。
        train_capacity_factor: 学習時(``self.training`` が True)の capacity factor。
        eval_capacity_factor: 評価時の capacity factor。
        beta: Swish の形状パラメータ。
        renormalize_gates: 選ばれた k 個の確率の和でゲートを割るか。
        router_init_scale: ルーターの重みの初期化の尺度 s。標準偏差 sqrt(s / d_model) の
            切断正規分布(2 標準偏差で切断)で初期化する。既定値 0.1 は Fedus et al. の
            「初期化の尺度を 1.0 から 0.1 に縮める」に従う。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 1,
        train_capacity_factor: float = 1.25,
        eval_capacity_factor: float = 2.0,
        beta: float = 1.0,
        renormalize_gates: bool = False,
        router_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k は 1 以上 num_experts 以下である必要がある: {top_k}")
        if renormalize_gates and top_k == 1:
            raise ValueError(
                "top_k = 1 でゲートを正規化すると常に 1 になり、ルーターに勾配が流れない"
            )
        if min(train_capacity_factor, eval_capacity_factor) <= 0.0:
            raise ValueError("capacity factor は正の値である必要がある")
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.train_capacity_factor = train_capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.beta = beta
        self.renormalize_gates = renormalize_gates

        # エキスパートの重み(nn.Linear と同じ (出力, 入力) の並びを、エキスパートの次元に積む)
        self.w = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.v = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        with torch.no_grad():
            for expert in range(num_experts):
                for weight in (self.w, self.v, self.w2):
                    nn.init.kaiming_uniform_(weight[expert], a=math.sqrt(5))
        # ルーター(バイアスなし)。エキスパートの後に初期化する
        self.router_weight = nn.Parameter(torch.empty(num_experts, d_model))
        std = math.sqrt(router_init_scale / d_model)
        nn.init.trunc_normal_(self.router_weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)

        self.auxiliary_losses: dict[str, Tensor] = {}
        self.last_routing: dict[str, Tensor] = {}
        self.last_expert_index: Tensor | None = None
        self.track_statistics = False
        self._statistics: dict[str, dict[str, Tensor]] = {}

    def capacity_factor(self) -> float:
        """現在のモード(学習 / 評価)の capacity factor。"""
        return self.train_capacity_factor if self.training else self.eval_capacity_factor

    def compute_router_logits(self, flat_x: Tensor) -> Tensor:
        """ルーターのロジット h(x) = W_r x を FP32 で計算する(autocast の中でも FP32)。"""
        with torch.autocast(device_type=flat_x.device.type, enabled=False):
            return torch.nn.functional.linear(flat_x.float(), self.router_weight.float())

    def forward(self, x: Tensor) -> Tensor:
        """順伝播。

        Args:
            x: 形状 ``(..., d_model)`` の入力。先頭の次元をすべて平坦化した T 個のトークンを
                1 つのバッチとして振り分ける。

        Returns:
            x と同じ形状の出力。破棄されたトークンの位置は 0。
        """
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        num_tokens = flat_x.size(0)
        num_experts, top_k = self.num_experts, self.top_k

        with torch.autocast(device_type=flat_x.device.type, enabled=False):
            logits = self.compute_router_logits(flat_x)  # (T, E)
            probabilities = torch.softmax(logits, dim=-1)
            top_probabilities, top_index = probabilities.topk(top_k, dim=-1)  # (T, k)
            gates = top_probabilities
            if self.renormalize_gates:
                gates = gates / gates.sum(dim=-1, keepdim=True)

        # 割り当てを「第 1 候補の全トークン、第 2 候補の全トークン、...」の順に並べる
        expert_index = top_index.t().reshape(-1)  # (k T,)
        token_index = torch.arange(num_tokens, device=flat_x.device).repeat(top_k)
        capacity = compute_expert_capacity(num_tokens, num_experts, top_k, self.capacity_factor())
        slot, keep = compute_dispatch_slots(expert_index, num_experts, capacity)

        # 振り分け: 枠ごとに、そこに入るトークンの番号を引く(空いた枠は 0 のトークン T を指す)
        num_slots = num_experts * capacity
        slot_token = torch.full((num_slots + 1,), num_tokens, device=flat_x.device)
        slot_token[slot] = token_index  # 破棄された割り当ては捨て枠 num_slots に書く
        padded_x = torch.cat([flat_x, flat_x.new_zeros(1, flat_x.size(1))], dim=0)
        expert_input = padded_x.index_select(0, slot_token[:num_slots]).view(
            num_experts, capacity, -1
        )

        # 全エキスパートを一括の行列積で計算する
        gate_hidden = swish(torch.bmm(expert_input, self.w.transpose(1, 2)), self.beta)
        hidden = gate_hidden * torch.bmm(expert_input, self.v.transpose(1, 2))
        expert_output = torch.bmm(hidden, self.w2.transpose(1, 2))  # (E, C, d_model)

        # 集約: 割り当てごとに枠の出力を引き(破棄された割り当ては 0 の行)、ゲートを掛けて足す
        padded_output = torch.cat(
            [
                expert_output.reshape(num_slots, -1),
                expert_output.new_zeros(1, expert_output.size(-1)),
            ],
            dim=0,
        )
        assignment_output = padded_output.index_select(0, slot) * gates.t().reshape(-1, 1)
        output = assignment_output.view(top_k, num_tokens, -1).sum(dim=0)

        self._record(logits, probabilities, top_index, keep)
        return output.view(shape)

    def _record(
        self, logits: Tensor, probabilities: Tensor, top_index: Tensor, keep: Tensor
    ) -> None:
        """補助損失(係数を掛ける前)と診断量を記録する。すべて FP32 の量から計算する。"""
        num_tokens = probabilities.size(0)
        num_assignments = top_index.numel()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            counts = torch.bincount(top_index.reshape(-1), minlength=self.num_experts)
            assignment_fraction = counts.to(probabilities.dtype) / num_assignments  # f_i
            mean_probability = probabilities.mean(dim=0)  # P_i
            # 負荷分散損失 N sum_i f_i P_i(Fedus et al. の式 4 の係数 alpha を除いたもの)。
            # f_i は argmax(微分できない)から作るので定数として扱い、勾配は P_i を通じて流れる
            load_balancing = self.num_experts * (assignment_fraction.detach() * mean_probability)
            # router z-loss (1 / T) sum_x (log sum_i exp h_i(x))^2(Zoph et al. の式 5)
            router_z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        self.auxiliary_losses = {
            LOAD_BALANCING_LOSS_NAME: load_balancing.sum(),
            ROUTER_Z_LOSS_NAME: router_z,
        }
        with torch.no_grad():
            max_probability = probabilities.max(dim=-1).values
            dropped = (~keep).sum()
            self.last_routing = {
                "assignment_fraction": assignment_fraction.detach(),
                "mean_router_probability": mean_probability.detach(),
                "dropped_fraction": dropped / num_assignments,
                "mean_max_probability": max_probability.mean(),
                "router_logit_norm": logits.norm(dim=-1).mean(),
            }
            self.last_expert_index = top_index.detach()
            # 個数は整数、確率の和は FP32 のまま積算する(MPS は FP64 を扱えないため。呼び出しは
            # 高々数千回で、FP32 の和の相対誤差は診断量の印字の桁より十分に小さい)
            if self.track_statistics:
                mode = "train" if self.training else "eval"
                update = {
                    "assignment_counts": counts,
                    "probability_sums": probabilities.sum(dim=0),
                    "dropped": dropped,
                    "assignments": torch.tensor(num_assignments, device=logits.device),
                    "tokens": torch.tensor(num_tokens, device=logits.device),
                    "max_probability_sum": max_probability.sum(),
                    "logit_norm_sum": logits.norm(dim=-1).sum(),
                }
                total = self._statistics.setdefault(mode, {})
                for name, value in update.items():
                    total[name] = total[name] + value if name in total else value.clone()

    def reset_statistics(self, mode: str | None = None) -> None:
        """積算した個数を消す(``mode`` が ``None`` なら学習時・評価時の両方)。"""
        if mode is None:
            self._statistics = {}
        else:
            self._statistics.pop(mode, None)

    def routing_statistics(self, mode: str) -> dict[str, object] | None:
        """``mode``(``"train"`` / ``"eval"``)の呼び出しで積算した診断量を返す。

        Returns:
            積算がなければ ``None``。あれば次のキーを持つ辞書:
            ``assignment_fraction``(f_i、長さ E のリスト)、``mean_router_probability``
            (P_i、長さ E のリスト)、``dropped_fraction``(破棄された割り当ての割合)、
            ``mean_max_probability``(トークンごとのルーターの確率の最大値の平均)、
            ``mean_router_logit_norm``(ルーターのロジットの L2 ノルムの平均)、``tokens``。
        """
        total = self._statistics.get(mode)
        if total is None:
            return None
        assignments = float(total["assignments"])
        tokens = float(total["tokens"])
        return {
            "assignment_fraction": (
                total["assignment_counts"].cpu().double() / assignments
            ).tolist(),
            "mean_router_probability": (total["probability_sums"].cpu().double() / tokens).tolist(),
            "dropped_fraction": float(total["dropped"]) / assignments,
            "mean_max_probability": float(total["max_probability_sum"]) / tokens,
            "mean_router_logit_norm": float(total["logit_norm_sum"]) / tokens,
            "tokens": int(tokens),
        }


def find_moe_layers(model: nn.Module) -> list[MixtureOfExpertsFeedForward]:
    """``model`` に含まれる MoE 層を、モジュールの登録順(浅い層から)に返す。"""
    return [m for m in model.modules() if isinstance(m, MixtureOfExpertsFeedForward)]


def set_statistics_tracking(model: nn.Module, enabled: bool) -> None:
    """``model`` の全 MoE 層の個数の積算を有効 / 無効にし、積算済みの個数を消す。"""
    for layer in find_moe_layers(model):
        layer.track_statistics = enabled
        layer.reset_statistics()
