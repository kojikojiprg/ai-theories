"""直接選好最適化(Direct Preference Optimization)の損失と学習ループ(018)。

017 の RLHF(Reinforcement Learning from Human Feedback)は、選好データから報酬モデルを学習し、
その報酬を KL 正則化つきで強化学習により最大化した。DPO(Direct Preference Optimization、
Rafailov et al., NeurIPS 2023)は、KL 正則化つきの目的関数の最適解
``pi*(y | x) = pi_ref(y | x) exp(r(x, y) / beta) / Z(x)`` を報酬について解き、Bradley-Terry
モデルに代入して、報酬モデルを経由せずに方策を選好データから直接学習する。

**対数比のマージン**(Azar et al., AISTATS 2024 の記法)``compute_log_ratio_margin()``:

.. math::

    h_\\theta(x, y_w, y_l)
        = \\log \\frac{\\pi_\\theta(y_w \\mid x)}{\\pi_{\\mathrm{ref}}(y_w \\mid x)}
        - \\log \\frac{\\pi_\\theta(y_l \\mid x)}{\\pi_{\\mathrm{ref}}(y_l \\mid x)}

**DPO 損失** ``dpo_loss()``: ``-log sigma(beta h)``。``-log sigma(z) = log(1 + e^{-z})`` を
``torch.logaddexp(0, -z)`` で計算する(017 の ``bradley_terry_loss()`` と同じ扱い)。

**IPO 損失** ``ipo_loss()``(Identity Preference Optimization、Azar et al.):
``(h - 1 / (2 tau))^2``。マージンを有限の目標値 ``1 / (2 tau)`` へ回帰させる。

**系列の対数確率**: ``log pi(y | x) = sum_t log pi(y_t | x, y_{<t})`` は、応答部分のトークンの
条件付き対数確率の和であり、プロンプト(指示部分)のトークンは和に含めない(016 の損失マスクと同じ
区別)。バッチ化は 017 の ``collate_rollout_sequences()``(``src/training/ppo.py``)をそのまま使う。
トークンごとの対数確率は、017 の ``gather_response_log_probs()`` と同じ位置・同じ one-hot の積で
取り出すが、語彙への射影を応答の位置に限って行う(``sequence_log_prob_sums()`` の docstring。
既存の関数は変更しない)。

**CHES スコア**(centered hidden embedding similarity、Razin et al., ICLR 2025)
``compute_ches_statistics()``: 応答の t 番目のトークンを予測する位置の最終正規化層の後の隠れ状態
(語彙への射影の直前の隠れ状態)を ``h_t`` とし、応答全体の和を ``S(y) = sum_t h_t`` とすると、

.. math::

    \\mathrm{CHES}_x(y_w, y_l) = \\langle S(y_w), S(y_l) \\rangle - \\lVert S(y_w) \\rVert^2

長さで正規化した版は ``<S(y_w), S(y_l)> / (|y_w| |y_l|) - ||S(y_w)||^2 / |y_w|^2`` である。

記号 / Notation:
    x      : プロンプト(指示部分)
    y_w    : 選好された応答、y_l: 選好されなかった応答
    beta   : DPO の KL 正則化の係数
    tau    : IPO の KL 正則化の係数(DPO の beta に対応する)
    h      : 対数比のマージン
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.models.reward_model import compute_final_hidden_states
from src.training.ppo import collate_rollout_sequences
from src.training.trainer import OptimizerLike, _set_optimizer_learning_rate

LOSS_TYPES = ("dpo", "ipo")


def sequence_log_prob_sums(
    model: nn.Module, token_ids: Tensor, prompt_lengths: Tensor, response_mask: Tensor
) -> Tensor:
    """バッチの各系列の応答部分の対数確率の和 ``log pi(y | x)``(``(B,)``、勾配を保つ)。

    ``collate_rollout_sequences()`` の出力をそのまま受け取る。``model`` は ``GPTLanguageModel``
    (LoRA を掛けたものを含む)である。

    **計算の順序**: 017 の ``gather_response_log_probs()`` は全位置の logits ``(B, S, V)`` を
    作ってから応答の位置を取り出す。本関数は、最終正規化層の後の隠れ状態 ``(B, S, d_model)`` から
    応答のトークンを予測する位置 ``P_b - 1 + t`` を先に取り出し(017 と同じ one-hot の積。MPS でも
    逆伝播が決定的)、その位置だけを語彙へ射影する。``lm_head`` は位置ごとの線形写像なので値は同じで
    (行列積の丸めの範囲で。018 のノートブックで ``gather_response_log_probs()`` の経路と照合する)、
    語彙 V の大きさの中間テンソルが ``S / R`` 分の 1 になる。
    """
    hidden = compute_final_hidden_states(model, token_ids)  # (B, S, d)
    offsets = torch.arange(response_mask.size(1), device=token_ids.device)
    positions = prompt_lengths[:, None] - 1 + offsets[None, :]
    positions = torch.where(response_mask, positions, torch.zeros_like(positions))  # (B, R)
    columns = torch.arange(token_ids.size(1), device=token_ids.device)
    selector = (positions[:, :, None] == columns[None, None, :]).to(hidden.dtype)  # (B, R, S)
    logits = model.lm_head(torch.bmm(selector, hidden)).float()  # (B, R, V)
    targets = token_ids.gather(1, positions + 1)  # 整数の添字(勾配を持たない)
    token_log_probs = -functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    return (token_log_probs * response_mask).sum(dim=1)


@torch.no_grad()
def compute_response_log_prob_sums(
    model: nn.Module,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    device: torch.device | str,
    batch_size: int = 256,
) -> np.ndarray:
    """系列ごとの応答部分の対数確率の和 ``log pi(y | x)`` を float64 の配列で返す(勾配なし)。

    系列を長さの順に並べてからバッチにし(パディングを減らすため)、元の順に戻して返す。
    バッチの組み方は入力だけで決まるので、同じ入力なら同じバッチで計算される(決定的)。
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts と responses の長さが一致しない")
    was_training = model.training
    model.eval()
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses, strict=True)]
    order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
    values = np.empty(len(prompts), dtype=np.float64)
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            token_ids, prompt_lengths, mask = (
                t.to(device)
                for t in collate_rollout_sequences(
                    [prompts[i] for i in chunk], [responses[i] for i in chunk]
                )
            )
            sums = sequence_log_prob_sums(model, token_ids, prompt_lengths, mask)
            values[chunk] = sums.cpu().double().numpy()
    finally:
        model.train(was_training)
    return values


def precompute_reference_log_probs(
    reference: nn.Module,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    device: torch.device | str,
    batch_size: int = 256,
    cache: dict | None = None,
    cache_key: object = None,
) -> tuple[np.ndarray, bool]:
    """参照方策の対数確率の和 ``log pi_ref(y | x)`` を事前計算する(条件・シードをまたいで再利用)。

    参照方策は学習中に変わらないので、学習データの応答の ``log pi_ref`` は 1 度だけ計算すればよい。
    ``cache``(呼び出し側が持つ辞書)と ``cache_key``(参照方策と入力を識別する hashable な値)を
    渡すと、同じ鍵の 2 回目以降は計算せずに保存した値の複製を返す。鍵の作り方(参照方策の
    重みのハッシュと入力のハッシュを含めること)は呼び出し側の責任である。

    Returns:
        ``(values, hit)``。``values`` は float64 の配列、``hit`` はキャッシュから返したか。
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key].copy(), True
    values = compute_response_log_prob_sums(reference, prompts, responses, device, batch_size)
    if cache is not None and cache_key is not None:
        cache[cache_key] = values.copy()
    return values, False


def compute_log_ratio_margin(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
) -> Tensor:
    """対数比のマージン ``h = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))``。

    形状はいずれも ``(B,)``。
    """
    return (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)


def dpo_loss(margin: Tensor, beta: float) -> Tensor:
    """DPO 損失の平均 ``mean(-log sigma(beta h))``(``logaddexp(0, -beta h)`` で計算する)。"""
    z = beta * margin
    return torch.logaddexp(torch.zeros_like(z), -z).mean()


def ipo_loss(margin: Tensor, tau: float) -> Tensor:
    """IPO 損失の平均 ``mean((h - 1 / (2 tau))^2)``。"""
    return ((margin - 1.0 / (2.0 * tau)) ** 2).mean()


def preference_loss(margin: Tensor, loss_type: str, beta: float) -> Tensor:
    """``loss_type``(``"dpo"`` または ``"ipo"``)の損失。IPO では ``beta`` を tau として使う。"""
    if loss_type == "dpo":
        return dpo_loss(margin, beta)
    if loss_type == "ipo":
        return ipo_loss(margin, beta)
    raise ValueError(f"loss_type は {LOSS_TYPES} のいずれか: {loss_type!r}")


def train_direct_preference(
    model: nn.Module,
    prompts: Sequence[Sequence[int]],
    chosen: Sequence[Sequence[int]],
    rejected: Sequence[Sequence[int]],
    reference_chosen: Sequence[float] | np.ndarray,
    reference_rejected: Sequence[float] | np.ndarray,
    batches: Sequence[Sequence[int]],
    optimizer: OptimizerLike,
    loss_type: str,
    beta: float,
    device: torch.device | str,
    learning_rate_schedule: Callable[[int], float] | None = None,
    evaluation_steps: Sequence[int] = (),
    evaluate: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """DPO / IPO の学習ループ(fp32、gradient clipping なし)。

    ステップ ``s``(1-indexed)では、``batches[s - 1]`` の事例について、選好された応答・選好されな
    かった応答の系列を 1 つのバッチ(``2 B`` 系列)にまとめて方策で順伝播し、応答部分の対数確率の和
    から対数比のマージン ``h`` を求め、``loss_type`` の損失で 1 回更新する。参照方策の対数確率は
    事前計算した値(``reference_chosen``・``reference_rejected``、事例ごと)を使う。

    Args:
        model: 方策(学習するパラメータは ``requires_grad`` で指定する。018 では LoRA のみ)。
        prompts: 事例ごとのプロンプトのトークン列。
        chosen: 事例ごとの選好された応答のトークン列(プロンプトを含まない)。
        rejected: 事例ごとの選好されなかった応答のトークン列。
        reference_chosen: 事例ごとの ``log pi_ref(y_w | x)``。
        reference_rejected: 事例ごとの ``log pi_ref(y_l | x)``。
        batches: ステップごとの事例の添字(``make_epoch_batches()`` の出力)。
        optimizer: ``step()``・``zero_grad()`` を持つ optimizer。
        loss_type: ``"dpo"`` または ``"ipo"``。
        beta: DPO の beta(IPO では tau)。
        device: 学習に使うデバイス。
        learning_rate_schedule: ステップ番号(1-indexed)から学習率を返す callable。
        evaluation_steps: ``evaluate(model)`` を呼ぶステップ(そのステップの更新の後)。
            0 を含めると学習前にも呼ぶ。
        evaluate: 評価関数(学習用の組全体の対数比など)。戻り値を ``history["evaluations"][step]``
            に記録する。呼び出しの前後で ``model`` の train / eval の状態を保つ。

    Returns:
        ステップごとの記録(各リストの長さは ``len(batches)``): ``step``・``loss``・
        ``mean_margin``(バッチの h の平均)・``accuracy``(h > 0 の事例の割合)・
        ``chosen_log_ratio``(``log pi(y_w) - log pi_ref(y_w)`` の平均)・``rejected_log_ratio``・
        ``learning_rate``。加えて ``evaluations``(``{step: evaluate の戻り値}``)。
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type は {LOSS_TYPES} のいずれか: {loss_type!r}")
    count = len(prompts)
    if not (
        count == len(chosen) == len(rejected) == len(reference_chosen) == len(reference_rejected)
    ):
        raise ValueError("事例ごとの入力の長さが一致しない")
    if any(s < 0 or s > len(batches) for s in evaluation_steps):
        raise ValueError(f"evaluation_steps は 0〜{len(batches)} の範囲: {evaluation_steps}")
    if evaluation_steps and evaluate is None:
        raise ValueError("evaluation_steps を指定するときは evaluate が必要")
    reference_chosen_t = torch.as_tensor(np.asarray(reference_chosen), dtype=torch.float32)
    reference_rejected_t = torch.as_tensor(np.asarray(reference_rejected), dtype=torch.float32)
    model = model.to(device)
    history: dict = {
        key: []
        for key in (
            "step",
            "loss",
            "mean_margin",
            "accuracy",
            "chosen_log_ratio",
            "rejected_log_ratio",
            "learning_rate",
        )
    }
    history["evaluations"] = {}

    def run_evaluation(step: int) -> None:
        was_training = model.training
        history["evaluations"][step] = evaluate(model)
        model.train(was_training)

    if 0 in evaluation_steps:
        run_evaluation(0)
    for step, indices in enumerate(batches, start=1):
        model.train()
        current_learning_rate = None
        if learning_rate_schedule is not None:
            current_learning_rate = learning_rate_schedule(step)
            _set_optimizer_learning_rate(optimizer, current_learning_rate)
        size = len(indices)
        token_ids, prompt_lengths, mask = (
            t.to(device)
            for t in collate_rollout_sequences(
                [prompts[i] for i in indices] * 2,
                [chosen[i] for i in indices] + [rejected[i] for i in indices],
            )
        )
        sums = sequence_log_prob_sums(model, token_ids, prompt_lengths, mask)
        index = torch.as_tensor(list(indices), dtype=torch.long)
        reference_w = reference_chosen_t[index].to(device)
        reference_l = reference_rejected_t[index].to(device)
        margin = compute_log_ratio_margin(sums[:size], sums[size:], reference_w, reference_l)
        loss = preference_loss(margin, loss_type, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        margin = margin.detach()
        history["step"].append(step)
        history["loss"].append(float(loss.detach()))
        history["mean_margin"].append(float(margin.mean()))
        history["accuracy"].append(float((margin > 0).float().mean()))
        history["chosen_log_ratio"].append(float((sums[:size].detach() - reference_w).mean()))
        history["rejected_log_ratio"].append(float((sums[size:].detach() - reference_l).mean()))
        history["learning_rate"].append(current_learning_rate)
        if step in evaluation_steps:
            run_evaluation(step)
    return history


@torch.no_grad()
def compute_ches_statistics(
    model: nn.Module,
    prompts: Sequence[Sequence[int]],
    first: Sequence[Sequence[int]],
    second: Sequence[Sequence[int]],
    device: torch.device | str,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """組ごとの CHES スコアを計算するための量(Razin et al. の Definition 2・3)。

    応答 y の t 番目のトークン(t = 1, ..., |y|)を予測する位置 ``P - 1 + t``(P はプロンプトの長さ)
    の、最終正規化層の後の隠れ状態の和 ``S(y)`` を求め、組ごとに ``<S(y_1), S(y_2)>``・
    ``||S(y_1)||^2``・``||S(y_2)||^2``・応答の長さを返す。CHES は非対称で、y_1 を選好された応答と
    すると ``inner - norm_first``、y_2 を選好された応答とすると ``inner - norm_second`` である
    (``ches_from_statistics()``)。

    Returns:
        ``inner``・``norm_first``・``norm_second``・``length_first``・``length_second``
        (いずれも長さ N の float64 の配列)。
    """
    if not len(prompts) == len(first) == len(second):
        raise ValueError("prompts・first・second の長さが一致しない")
    was_training = model.training
    model.eval()
    sums: list[np.ndarray] = []
    try:
        for responses in (first, second):
            total = np.empty((len(prompts), model.token_embedding.embedding_dim))
            for start in range(0, len(prompts), batch_size):
                chunk = range(start, min(start + batch_size, len(prompts)))
                token_ids, prompt_lengths, mask = (
                    t.to(device)
                    for t in collate_rollout_sequences(
                        [prompts[i] for i in chunk], [responses[i] for i in chunk]
                    )
                )
                hidden = compute_final_hidden_states(model, token_ids)  # (b, S, d)
                offsets = torch.arange(mask.size(1), device=mask.device)
                positions = prompt_lengths[:, None] - 1 + offsets[None, :]  # (b, R)
                columns = torch.arange(hidden.size(1), device=mask.device)
                selector = (
                    (positions[:, :, None] == columns[None, None, :]) & mask[:, :, None]
                ).to(hidden.dtype)  # (b, R, S)
                summed = torch.bmm(selector.sum(dim=1, keepdim=True), hidden)[:, 0]  # (b, d)
                total[start : start + len(chunk)] = summed.cpu().double().numpy()
            sums.append(total)
    finally:
        model.train(was_training)
    return {
        "inner": np.einsum("nd,nd->n", sums[0], sums[1]),
        "norm_first": np.einsum("nd,nd->n", sums[0], sums[0]),
        "norm_second": np.einsum("nd,nd->n", sums[1], sums[1]),
        "length_first": np.array([len(r) for r in first], dtype=np.float64),
        "length_second": np.array([len(r) for r in second], dtype=np.float64),
    }


def ches_from_statistics(
    statistics: dict[str, np.ndarray], first_chosen: np.ndarray, normalized: bool = False
) -> np.ndarray:
    """``compute_ches_statistics()`` の出力から、向き(``first_chosen``)を指定した CHES を返す。

    ``first_chosen[i]`` が True なら y_1 を選好された応答とする。``normalized=True`` なら
    長さで正規化した版(Razin et al. の Definition 3)。
    """
    first_chosen = np.asarray(first_chosen, dtype=bool)
    inner = statistics["inner"]
    norm_w = np.where(first_chosen, statistics["norm_first"], statistics["norm_second"])
    if not normalized:
        return inner - norm_w
    length_w = np.where(first_chosen, statistics["length_first"], statistics["length_second"])
    length_l = np.where(first_chosen, statistics["length_second"], statistics["length_first"])
    return inner / (length_w * length_l) - norm_w / length_w**2
