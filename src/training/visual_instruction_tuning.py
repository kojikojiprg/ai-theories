"""LLaVA 型のモデルの 2 段階学習(特徴の整列 + 視覚指示チューニング)の学習ループと評価(021)。

第 1 段階(特徴の整列)も第 2 段階(視覚指示チューニング)も、損失は応答部分のトークンだけに掛ける
(016 の損失マスクと同じ。LLaVA の原論文の式 3 も、答えのトークンだけの自己回帰の対数尤度を
最大化する)。
2 つの段階の違いは、学習するパラメータ(第 1 段階は projection 層のみ、第 2 段階は projection 層と
言語モデルへの LoRA)とデータ(キャプション / 質問と答え)だけであり、学習ループは共通にする。

**損失の正規化の単位**: バッチ内の応答部分のトークン(終端記号を含む)の集合を R、トークン t の負の
対数尤度を ell_t とすると、損失は (1 / |R|) sum_{t in R} ell_t である(016 の損失マスクありの損失
``compute_instruction_tuning_loss(..., include_prompt_loss=False)`` と同じ値。021 で FP64 で
確かめる)。

画像 encoder は凍結しているので、パッチ特徴は画像ごとに一度だけ計算したもの(``features``、
形状 ``(画像の数, N, d_v)``)を添字で引いて使う。

記号 / Notation:
    B : バッチサイズ
    S : 系列長(全事例で共通の固定長)
    K : バッチ内の応答部分のトークン数 |R|
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from src.data.instruction import END_MARKER
from src.data.visual_instruction import EncodedVisualBatchData
from src.models.llava import LlavaStyleModel, greedy_generate_with_visual_tokens
from src.training.trainer import OptimizerLike, _set_optimizer_learning_rate


def response_loss(
    model: LlavaStyleModel, token_ids: Tensor, patch_features: Tensor, response_target_mask: Tensor
) -> dict[str, Tensor]:
    """応答部分のトークンの負の対数尤度の平均(逆伝播に使う損失)と、その合計・トークン数。"""
    logits, targets = model.response_logits(token_ids, patch_features, response_target_mask)
    # 半精度は FP32 に上げ、FP32・FP64 はそのまま計算する(FP64 の参照実装との比較のため)
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    total = functional.cross_entropy(logits, targets, reduction="sum")
    count = targets.numel()
    return {"loss": total / count, "sum": total.detach(), "count": count}


def train_visual_instruction(
    model: LlavaStyleModel,
    data: EncodedVisualBatchData,
    features: Tensor,
    batches: np.ndarray,
    optimizer: OptimizerLike,
    trainable_parameters: Sequence[torch.nn.Parameter],
    learning_rate_schedule: Callable[[int], float],
    gradient_clip_threshold: float | None,
    evaluation_steps: Sequence[int] = (),
    evaluation_fn: Callable[[LlavaStyleModel], dict] | None = None,
) -> dict[str, list]:
    """学習ループ(FP32)。ステップ ``s``(1-indexed)では ``batches[s - 1]`` の事例で 1 回更新する。

    gradient clipping は ``trainable_parameters`` のグローバルノルムで行い、適用前のノルムを常に
    記録する(016 の ``train_instruction_tuning()`` と同じ規則)。``evaluation_steps`` に含まれる
    ステップの更新の後に ``evaluation_fn(model)`` を呼び、その結果を記録する(途中の評価、診断量)。

    Args:
        model: ``LlavaStyleModel``。学習しないパラメータは ``requires_grad=False`` にしておく。
        data: ``encode_visual_examples()`` の出力(``features`` と同じデバイス上)。
        features: 形状 ``(画像の数, N, d_v)`` のパッチ特徴(凍結した画像 encoder の出力)。
        batches: 形状 ``(T, B)`` の事例の添字(``make_step_batches()`` の出力)。
        optimizer: ``step()``・``zero_grad()`` を持つ optimizer(``trainable_parameters`` で構築)。
        trainable_parameters: 学習するパラメータ。
        learning_rate_schedule: ステップ番号(1-indexed)から学習率を返す callable。
        gradient_clip_threshold: gradient clipping の閾値(``None`` なら無効)。
        evaluation_steps: 途中の評価を行うステップ。
        evaluation_fn: 途中の評価の関数。

    Returns:
        ステップごとの ``loss``・``gradient_norm``(クリッピングの前)・``learning_rate``・
        ``response_tokens``、途中の評価 ``evaluations``(ステップ -> 結果)。
    """
    history: dict[str, list] = {
        "loss": [],
        "gradient_norm": [],
        "learning_rate": [],
        "response_tokens": [],
    }
    evaluations = {}
    evaluation_steps = set(evaluation_steps)
    device = features.device
    batches_device = torch.as_tensor(batches, device=device)
    losses, norms = [], []
    for step in range(1, len(batches) + 1):
        model.train()
        learning_rate = learning_rate_schedule(step)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        index = batches_device[step - 1]
        token_ids = data.token_ids[index]
        parts = response_loss(
            model, token_ids, features[data.image_indices[index]], data.response_target_mask[index]
        )
        optimizer.zero_grad()
        parts["loss"].backward()
        gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad)
                    for p in trainable_parameters
                    if p.grad is not None
                ]
            )
        )
        if gradient_clip_threshold is not None:
            # 同期を避けるため、閾値を超えたかをホストに読み出さずに係数 min(1, c / ||g||) を掛ける
            scale = torch.clamp(gradient_clip_threshold / (gradient_norm + 1e-12), max=1.0)
            for p in trainable_parameters:
                if p.grad is not None:
                    p.grad.mul_(scale)
        optimizer.step()
        losses.append(parts["loss"].detach())
        norms.append(gradient_norm.detach())
        history["learning_rate"].append(learning_rate)
        history["response_tokens"].append(int(parts["count"]))
        if step in evaluation_steps and evaluation_fn is not None:
            evaluations[step] = evaluation_fn(model)
    history["loss"] = torch.stack(losses).cpu().tolist()
    history["gradient_norm"] = torch.stack(norms).cpu().tolist()
    history["evaluations"] = evaluations
    return history


@torch.no_grad()
def evaluate_response_negative_log_likelihood(
    model: LlavaStyleModel, data: EncodedVisualBatchData, features: Tensor, batch_size: int = 256
) -> dict[str, np.ndarray]:
    """教師強制で、事例ごとの応答部分の負の対数尤度の合計(nats、float64)とトークン数を返す。"""
    model.eval()
    sums, counts = [], []
    for start in range(0, len(data), batch_size):
        sl = slice(start, start + batch_size)
        token_ids = data.token_ids[sl]
        mask = data.response_target_mask[sl]
        hidden = model.hidden_states(token_ids, features[data.image_indices[sl]])[:, :-1]
        logits = model.language_model.lm_head(hidden)
        token_losses = functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            token_ids[:, 1:].reshape(-1),
            reduction="none",
        ).view(mask.shape)
        sums.append((token_losses * mask).sum(dim=1).cpu().double().numpy())
        counts.append(mask.sum(dim=1).cpu().numpy().astype(np.int64))
    return {"sum": np.concatenate(sums), "count": np.concatenate(counts)}


@torch.no_grad()
def generate_answers(
    model: LlavaStyleModel,
    data: EncodedVisualBatchData,
    features: Tensor,
    decode: Callable[[list[int]], str],
    max_new_tokens: int,
    image_indices: Tensor | None = None,
    batch_size: int = 256,
) -> list[str]:
    """各事例の指示部分から貪欲法で生成し、生成した応答部分の文字列を返す。

    ``image_indices`` を与えると、各事例の画像の代わりにその添字の画像の特徴を使う(画像を別の画像に
    差し替えた評価、視覚への依存の前提条件)。
    """
    lengths = data.prompt_lengths.tolist()
    rows = data.token_ids.cpu().tolist()
    prompts = [row[:length] for row, length in zip(rows, lengths, strict=True)]
    index = data.image_indices if image_indices is None else image_indices
    generated = greedy_generate_with_visual_tokens(
        model, prompts, features[index], decode, END_MARKER, max_new_tokens, batch_size
    )
    return [decode(tokens) for tokens in generated]
