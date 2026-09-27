"""PPO(Proximal Policy Optimization)による言語モデルの強化学習の部品(017)。

RLHF(Reinforcement Learning from Human Feedback)の第 3 段階を、Ziegler et al.(arXiv 2019)・
Stiennon et al.(NeurIPS 2020)・Ouyang et al.(NeurIPS 2022)の形で構成する部品をスクラッチ実装する。
生成(ロールアウト)と学習ループの呼び出しは 017 のノートブックが行う。

**エピソードと報酬**: プロンプト x を状態の起点とし、応答 y = (y_1, ..., y_T) の各トークンを
行動とする。トークン t の報酬は、KL ダイバージェンス(Kullback-Leibler divergence)のペナルティと、
最後のトークンでのみ与える報酬モデルのスコアの和である。

.. math::

    r_t = -\\beta \\left( \\log \\pi_{\\mathrm{old}}(y_t \\mid x, y_{<t})
          - \\log \\pi_{\\mathrm{ref}}(y_t \\mid x, y_{<t}) \\right)
          + [t = T] \\, r_\\phi(x, y)

``compute_kl_penalized_rewards()``。系列全体で和をとると、第 1 項は
``-beta log(pi_old(y | x) / pi_ref(y | x))`` になり、その期待値は ``-beta KL(pi_old || pi_ref)``
である。

**GAE**(Generalized Advantage Estimation、Schulman et al., ICLR 2016)``compute_gae()``:

.. math::

    \\delta_t = r_t + \\gamma V_{t+1} - V_t, \\qquad
    \\hat{A}_t = \\sum_{l=0}^{T-t} (\\gamma \\lambda)^l \\delta_{t+l}, \\qquad
    \\hat{R}_t = \\hat{A}_t + V_t

(``V_{T+1} = 0``。応答の終わりでエピソードが終わる。)

**クリップ付き目的関数**(Schulman et al., arXiv 2017)``ppo_clipped_policy_loss()``:

.. math::

    L^{\\mathrm{CLIP}}(\\theta) = \\mathbb{E}_t \\left[ \\min\\left( \\rho_t \\hat{A}_t,\\
        \\mathrm{clip}(\\rho_t, 1 - \\epsilon, 1 + \\epsilon) \\hat{A}_t \\right) \\right], \\qquad
    \\rho_t = \\frac{\\pi_\\theta(y_t \\mid \\cdot)}{\\pi_{\\mathrm{old}}(y_t \\mid \\cdot)}

損失としてはその符号を反転したものを最小化する。

**価値ヘッド** ``PolicyWithValueHead``: 方策(SFT + LoRA の言語モデル)の本体を共有し、最終正規化層の
後の隠れ状態から線形の層で状態価値 V を出す。状態 ``(x, y_{<t})`` の価値は、トークン y_t を予測する
位置(y_t の 1 つ前の位置)の隠れ状態から計算する。本体を共有するので、価値関数の損失の勾配も
方策の学習可能なパラメータ(LoRA)に流れる(方策と価値関数を別のネットワークにする実装もある。
共有は計算量を抑えるための簡略化である)。

**応答に揃えた配列**: 本モジュールの関数は、トークンごとの量を形状 ``(B, R)`` の「応答に揃えた」
配列で扱う(R はバッチ内の最大の応答長)。行 b の列 t は、応答の t 番目のトークン(系列の絶対位置
``P_b + t``、P_b は行 b のプロンプトの長さ)に対応する。``response_mask`` が False の列は
パディングであり、どの損失にも寄与しない。

記号 / Notation:
    B        : バッチ内のエピソード(系列)の数
    R        : バッチ内の最大の応答長
    beta     : KL 係数(ラベル生成の尺度 kappa とは別の量)
    gamma    : 割引率
    lambda   : GAE のパラメータ
    epsilon  : クリップの幅
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.models.gpt import GPTLanguageModel
from src.models.reward_model import compute_final_hidden_states
from src.training.trainer import OptimizerLike


def collate_rollout_sequences(
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """プロンプト + 応答を右パディングしてバッチにする。

    Returns:
        ``(token_ids, prompt_lengths, response_mask)``。形状は ``(B, S_max)``・``(B,)``・
        ``(B, R)``(``response_mask[b, t]`` は行 b の応答が t 番目のトークンを持つか)。
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts と responses の長さが一致しない")
    if any(len(r) == 0 for r in responses):
        raise ValueError("空の応答がある")
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses, strict=True)]
    max_response = max(len(r) for r in responses)
    token_ids = torch.full((len(prompts), max(lengths)), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((len(prompts), max_response), dtype=torch.bool)
    for row, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        sequence = list(prompt) + list(response)
        token_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        response_mask[row, : len(response)] = True
    prompt_lengths = torch.tensor([len(p) for p in prompts], dtype=torch.long)
    return token_ids, prompt_lengths, response_mask


def _response_positions(prompt_lengths: Tensor, response_mask: Tensor) -> Tensor:
    """応答の t 番目のトークンを予測する位置 ``P_b - 1 + t``(``(B, R)``)。パディングの列は 0。"""
    offsets = torch.arange(response_mask.size(1), device=prompt_lengths.device)
    positions = prompt_lengths[:, None] - 1 + offsets[None, :]
    return torch.where(response_mask, positions, torch.zeros_like(positions))


def _position_selector(positions: Tensor, sequence_length: int, dtype: torch.dtype) -> Tensor:
    """位置 ``positions``(``(B, R)``)を取り出す one-hot の行列 ``(B, R, S)``。

    ``selector @ x``(x は ``(B, S, ...)``)で各行の指定位置の値を取り出す。``gather`` や高度な
    インデックスの逆伝播(scatter・index_put の累積)は MPS に決定的な実装がないため、値が同じで
    逆伝播が行列積になる one-hot の積を使う。
    """
    columns = torch.arange(sequence_length, device=positions.device)
    return (positions[:, :, None] == columns[None, None, :]).to(dtype)


def gather_response_log_probs(
    logits: Tensor, token_ids: Tensor, prompt_lengths: Tensor, response_mask: Tensor
) -> Tensor:
    """応答のトークンごとの対数確率 ``log pi(y_t | x, y_{<t})``(``(B, R)``、パディングは 0)。

    Args:
        logits: 形状 ``(B, S, V)``(``token_ids`` を順伝播した出力)。
        token_ids: 形状 ``(B, S)``。
        prompt_lengths: 形状 ``(B,)``。
        response_mask: 形状 ``(B, R)``。
    """
    positions = _response_positions(prompt_lengths, response_mask)  # (B, R)
    selector = _position_selector(positions, logits.size(1), logits.dtype)
    selected = torch.bmm(selector, logits).float()  # (B, R, V)
    targets = token_ids.gather(1, positions + 1)  # 整数の添字(勾配を持たない)
    token_log_probs = -functional.cross_entropy(
        selected.reshape(-1, selected.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    return token_log_probs * response_mask


def gather_response_values(values: Tensor, prompt_lengths: Tensor, response_mask: Tensor) -> Tensor:
    """位置ごとの価値 ``(B, S)`` から、応答の t 番目のトークンの直前の状態の価値 ``(B, R)``。"""
    positions = _response_positions(prompt_lengths, response_mask)
    selector = _position_selector(positions, values.size(1), values.dtype)
    return torch.bmm(selector, values[:, :, None]).squeeze(-1) * response_mask


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """``mask`` が True の要素だけの平均(バッチ全体のトークンで平均する)。"""
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum()


def compute_kl_penalized_rewards(
    log_probs: Tensor,
    reference_log_probs: Tensor,
    scores: Tensor,
    response_mask: Tensor,
    beta: float,
) -> Tensor:
    """トークンごとの報酬 ``r_t = -beta (log pi_old - log pi_ref) + [t = T] r_phi``(``(B, R)``)。

    Args:
        log_probs: ロールアウト時の方策の対数確率 ``log pi_old``(``(B, R)``)。
        reference_log_probs: 参照方策の対数確率 ``log pi_ref``(``(B, R)``)。
        scores: 系列ごとの報酬モデルのスコア ``r_phi(x, y)``(``(B,)``)。
        response_mask: ``(B, R)``。各行は先頭から連続して True である前提。
        beta: KL 係数。
    """
    mask = response_mask.to(log_probs.dtype)
    rewards = -beta * (log_probs.detach() - reference_log_probs.detach()) * mask
    last = response_mask.sum(dim=1) - 1
    rows = torch.arange(rewards.size(0), device=rewards.device)
    rewards[rows, last] += scores.to(rewards.dtype)
    return rewards


def compute_gae(
    rewards: Tensor, values: Tensor, response_mask: Tensor, gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    """GAE によるアドバンテージ ``A_t`` と、価値関数の目標 ``R_t = A_t + V_t`` を返す。

    行ごとに、応答の最後のトークン(``V_{T+1} = 0``)から先頭へ向かって

    ``delta_t = r_t + gamma V_{t+1} - V_t``、``A_t = delta_t + gamma lambda A_{t+1}``

    を計算する。パディングの列の値は 0 とし、計算にも寄与しない(各行は先頭から連続して True で
    ある前提なので、パディングの列の次の値 ``V_{t+1}``・``A_{t+1}`` を 0 として扱えば正しい)。

    Returns:
        ``(advantages, returns)``。いずれも ``(B, R)``(勾配を持たない)。
    """
    mask = response_mask.to(values.dtype)
    rewards = rewards.detach() * mask
    values = values.detach() * mask
    advantages = torch.zeros_like(values)
    next_value = torch.zeros_like(values[:, 0])
    next_advantage = torch.zeros_like(values[:, 0])
    for t in range(values.size(1) - 1, -1, -1):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantage = (delta + gamma * lam * next_advantage) * mask[:, t]
        advantages[:, t] = advantage
        next_value = values[:, t] * mask[:, t]
        next_advantage = advantage
    return advantages, (advantages + values) * mask


def whiten_advantages(advantages: Tensor, response_mask: Tensor, eps: float = 1e-8) -> Tensor:
    """アドバンテージを、マスク内のトークン全体で平均 0・標準偏差 1 に標準化する。"""
    mean = masked_mean(advantages, response_mask)
    variance = masked_mean((advantages - mean) ** 2, response_mask)
    return (advantages - mean) / torch.sqrt(variance + eps) * response_mask


def ppo_clipped_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor]:
    """クリップ付き目的関数の符号を反転した損失と、クリップが効いたトークンの割合を返す。

    ``loss = -mean_t min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t)``(マスク内のトークンで平均)。
    「クリップが効いた」は、min が 2 つ目の項(クリップした項)をとり、かつ 2 つの項が異なる
    (rho_t がクリップの範囲の外にある)トークンとする。
    """
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    objective = torch.minimum(unclipped, clipped)
    clip_active = (clipped < unclipped) & response_mask
    clip_fraction = clip_active.float().sum() / response_mask.float().sum()
    return -masked_mean(objective, response_mask), clip_fraction.detach()


def value_function_loss(values: Tensor, returns: Tensor, response_mask: Tensor) -> Tensor:
    """価値関数の損失 ``0.5 mean_t (V_t - R_t)^2``(マスク内のトークンで平均)。"""
    return 0.5 * masked_mean((values - returns.detach()) ** 2, response_mask)


class PolicyWithValueHead(nn.Module):
    """方策(言語モデル)と、その本体を共有する価値ヘッド。

    Args:
        policy: 方策の ``GPTLanguageModel``(017 では SFT のマージ済みの重みに LoRA を掛けたもの)。
            生成(ロールアウト)には ``policy`` をそのまま使う。
        value_head_init_std: 価値ヘッドの重みの初期化の標準偏差(バイアスは 0)。
    """

    def __init__(self, policy: GPTLanguageModel, value_head_init_std: float = 0.0) -> None:
        super().__init__()
        self.policy = policy
        d_model = policy.token_embedding.embedding_dim
        self.value_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=value_head_init_std)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """``(logits (B, S, V), values (B, S))``。values[b, j] は位置 j までを見た状態の価値。"""
        hidden = compute_final_hidden_states(self.policy, token_ids)
        return self.policy.lm_head(hidden), self.value_head(hidden).squeeze(-1)


def ppo_update(
    agent: PolicyWithValueHead,
    rollout: dict[str, Tensor],
    optimizer: OptimizerLike,
    num_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """1 回のロールアウトのバッチで、方策と価値関数を ``num_epochs`` エポック更新する。

    Args:
        agent: 方策と価値ヘッド。
        rollout: ``token_ids``・``prompt_lengths``・``response_mask``
            (``collate_rollout_sequences()``)、
            ``old_log_probs``・``advantages``・``returns``(いずれも ``(B, R)``、同じデバイス上)。
        optimizer: 学習可能なパラメータ(LoRA と価値ヘッド)の optimizer。
        num_epochs: 同じロールアウトを使い回すエポック数。
        minibatch_size: 1 回の更新の系列数(B を割り切る)。
        clip_epsilon: クリップの幅。
        value_coefficient: 価値関数の損失の係数 c_V(全体の損失 = 方策の損失 + c_V x 価値の損失)。
        rng: ミニバッチの並べ替えに使う乱数生成器。

    Returns:
        全更新の平均: ``policy_loss``・``value_loss``・``clip_fraction``・``approx_kl``
        (旧方策と更新中の方策の対数確率の差の平均。``mean(log pi_old - log pi_theta)``)。
    """
    batch = rollout["token_ids"].size(0)
    if batch % minibatch_size != 0:
        raise ValueError(f"minibatch_size({minibatch_size})が B({batch})を割り切らない")
    agent.train()
    records: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "clip_fraction": [],
        "approx_kl": [],
    }
    for _ in range(num_epochs):
        order = rng.permutation(batch)
        for start in range(0, batch, minibatch_size):
            index = torch.as_tensor(
                order[start : start + minibatch_size], device=rollout["token_ids"].device
            )
            token_ids = rollout["token_ids"][index]
            prompt_lengths = rollout["prompt_lengths"][index]
            mask = rollout["response_mask"][index]
            # ミニバッチ内の最大の長さに切り詰める(右パディングの余りは計算しない)
            length = int((prompt_lengths + mask.sum(dim=1)).max())
            width = int(mask.sum(dim=1).max())
            token_ids, mask = token_ids[:, :length], mask[:, :width]
            logits, values = agent(token_ids)
            log_probs = gather_response_log_probs(logits, token_ids, prompt_lengths, mask)
            token_values = gather_response_values(values, prompt_lengths, mask)
            old_log_probs = rollout["old_log_probs"][index][:, :width]
            policy_loss, clip_fraction = ppo_clipped_policy_loss(
                log_probs,
                old_log_probs,
                rollout["advantages"][index][:, :width],
                mask,
                clip_epsilon,
            )
            value_loss = value_function_loss(
                token_values, rollout["returns"][index][:, :width], mask
            )
            loss = policy_loss + value_coefficient * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            records["policy_loss"].append(float(policy_loss.detach()))
            records["value_loss"].append(float(value_loss.detach()))
            records["clip_fraction"].append(float(clip_fraction))
            records["approx_kl"].append(
                float(masked_mean(old_log_probs - log_probs.detach(), mask))
            )
    return {key: float(np.mean(values)) for key, values in records.items()}
