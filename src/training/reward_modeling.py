"""報酬モデル(Reward Model)の Bradley-Terry 損失と学習ループ(017)。

選好の組 ``(x, y_w, y_l)``(x はプロンプト、y_w は選ばれた応答、y_l は選ばれなかった応答)に
対して、Bradley-Terry モデル ``P(y_w > y_l) = sigma(r_phi(x, y_w) - r_phi(x, y_l))`` の負の
対数尤度を最小化する(Christiano et al., NeurIPS 2017、Stiennon et al., NeurIPS 2020、
Ouyang et al., NeurIPS 2022)。

.. math::

    \\mathcal{L}(\\phi) = -\\frac{1}{B} \\sum_{i=1}^{B}
        \\log \\sigma\\left(r_\\phi(x_i, y_{w,i}) - r_\\phi(x_i, y_{l,i})\\right)

``-log sigma(z) = log(1 + e^{-z})`` を ``torch.logaddexp(0, -z)`` で計算する(z が大きな負の値でも
``exp`` があふれない)。損失は報酬の差 z にしか依存しないので、報酬に定数を足しても損失は
変わらない(報酬は定数の差を除いてしか識別できない)。

記号 / Notation:
    B     : 1 ステップの組の数
    phi   : 報酬モデルのパラメータ
    sigma : シグモイド関数
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from src.data.preference import collate_scored_sequences
from src.models.reward_model import RewardModel
from src.training.trainer import OptimizerLike, _set_optimizer_learning_rate


def bradley_terry_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """Bradley-Terry モデルの負の対数尤度の平均 ``mean(-log sigma(r_w - r_l))``。"""
    margin = chosen_rewards - rejected_rewards
    return torch.logaddexp(torch.zeros_like(margin), -margin).mean()


def train_reward_model(
    model: RewardModel,
    prompts: Sequence[Sequence[int]],
    chosen: Sequence[Sequence[int]],
    rejected: Sequence[Sequence[int]],
    batches: Sequence[Sequence[int]],
    optimizer: OptimizerLike,
    device: torch.device | str,
    learning_rate_schedule: Callable[[int], float] | None = None,
) -> dict[str, list]:
    """報酬モデルの学習ループ(fp32、gradient clipping なし)。

    ステップ ``s``(1-indexed)では、``batches[s - 1]`` の組について、選ばれた応答・選ばれなかった
    応答の系列(プロンプト + 応答)を 1 つのバッチにまとめて順伝播し(``2 B`` 系列)、
    ``bradley_terry_loss()`` で 1 回更新する。

    Args:
        model: 報酬モデル(学習するパラメータは ``requires_grad`` で指定する)。
        prompts: 組ごとのプロンプトのトークン列。
        chosen: 組ごとの選ばれた応答のトークン列(プロンプトを含まない)。
        rejected: 組ごとの選ばれなかった応答のトークン列。
        batches: ステップごとの組の添字(``make_epoch_batches()`` の出力)。
        optimizer: ``step()``・``zero_grad()`` を持つ optimizer。
        device: 学習に使うデバイス。
        learning_rate_schedule: ステップ番号(1-indexed)から学習率を返す callable。

    Returns:
        ステップごとの記録: ``step``、``loss``、``accuracy``(バッチ内で r_w > r_l だった組の割合。
        ラベルに対する一致率)、``mean_margin``(r_w - r_l の平均)、``learning_rate``。
    """
    if not len(prompts) == len(chosen) == len(rejected):
        raise ValueError("prompts・chosen・rejected の長さが一致しない")
    model = model.to(device)
    history: dict[str, list] = {
        "step": [],
        "loss": [],
        "accuracy": [],
        "mean_margin": [],
        "learning_rate": [],
    }
    for step, indices in enumerate(batches, start=1):
        model.train()
        current_learning_rate = None
        if learning_rate_schedule is not None:
            current_learning_rate = learning_rate_schedule(step)
            _set_optimizer_learning_rate(optimizer, current_learning_rate)
        sequences = [list(prompts[i]) + list(chosen[i]) for i in indices]
        sequences += [list(prompts[i]) + list(rejected[i]) for i in indices]
        token_ids, last_indices = collate_scored_sequences(sequences)
        rewards = model(token_ids.to(device), last_indices.to(device))
        chosen_rewards, rejected_rewards = rewards[: len(indices)], rewards[len(indices) :]
        loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        margin = (chosen_rewards - rejected_rewards).detach()
        history["step"].append(step)
        history["loss"].append(float(loss.detach()))
        history["accuracy"].append(float((margin > 0).float().mean()))
        history["mean_margin"].append(float(margin.mean()))
        history["learning_rate"].append(current_learning_rate)
    return history
