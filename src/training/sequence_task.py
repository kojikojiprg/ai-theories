"""合成の系列課題(selective copying・induction heads)の学習ループと評価関数(023)。

``src/training/trainer.py`` の ``train_language_model()`` はコーパスからランダムな区間を
切り出す学習に結びついているので、ここでは **ステップごとに新しい事例を生成する** 学習ループを
別に用意する(同じ事例を繰り返さないので、学習データへの当てはまりと汎化の差の切り分けは
要らない)。損失は、目標が ``IGNORE_INDEX`` でない位置の交差エントロピーの平均である。
optimizer は 007 の ``AdamW``、学習率は warmup + cosine、gradient clipping は
``train_language_model()`` と同じ規則(全パラメータの勾配を連結した L2 ノルムが閾値を超えたら、
閾値 / ノルム を掛ける)である。ステップごとに CPU と同期しないように、損失・勾配ノルム・
発動の有無は GPU 上の値のまま積んで、最後にまとめて読み出す。

評価関数はすべて FP32、``model.eval()`` で行う。生成した答えそのもの(``answers`` /
``predictions``)を返すので、混同行列などの診断量を作れる。
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.data.synthetic_sequence import IGNORE_INDEX, SelectiveCopyingTask, make_rng
from src.training.optimizer import AdamW
from src.training.schedule import compute_warmup_cosine_learning_rate


def train_sequence_task(
    model: nn.Module,
    make_batch: Callable[[np.random.Generator], tuple[Tensor, Tensor]],
    num_steps: int,
    learning_rate: float,
    device: torch.device | str,
    seed: int,
    warmup_ratio: float = 0.1,
    min_learning_rate_ratio: float = 0.01,
    weight_decay: float = 0.0,
    gradient_clip_threshold: float | None = 1.0,
    eval_steps: Sequence[int] = (),
    evaluation_fn: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """ステップごとに新しい事例を生成して学習する。

    Args:
        model: 学習対象(``forward(token_ids) -> logits`` を持つ)。
        make_batch: 乱数の生成器を受け取り ``(inputs, targets)`` を返す関数。``inputs`` は形状
            ``(batch, length)`` の LongTensor、``targets`` は同じ形状で、損失を掛ける位置に答えの
            トークンの番号、それ以外に ``IGNORE_INDEX`` を置く。
        num_steps: 学習ステップ数。
        learning_rate: 学習率の最大値(warmup の後の値)。
        device: 学習に使うデバイス。
        seed: 学習データの乱数シード。ステップ ``s`` の事例は ``make_rng(seed, s)`` から生成する。
        warmup_ratio: 線形 warmup のステップ数の割合。
        min_learning_rate_ratio: cosine decay の下限の、最大値に対する比。
        weight_decay: AdamW の重み減衰。
        gradient_clip_threshold: gradient clipping の閾値(``None`` なら掛けない)。
        eval_steps: 学習の途中で ``evaluation_fn`` を呼ぶステップ(1 始まり)。
        evaluation_fn: モデルを受け取り、評価の結果の辞書を返す関数。

    Returns:
        ``train_loss``(ステップごとの損失、float64 の配列)、``gradient_norm``、
        ``clip_triggered``(bool の配列)、``eval_step``、``eval_results``(``evaluation_fn`` の
        戻り値のリスト)、``seconds`` を持つ辞書。
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    warmup_steps = max(1, round(warmup_ratio * num_steps))
    schedule = functools.partial(
        compute_warmup_cosine_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
        peak_learning_rate=learning_rate,
        min_learning_rate=learning_rate * min_learning_rate_ratio,
    )
    eval_step_set = set(eval_steps)
    losses: list[Tensor] = []
    gradient_norms: list[Tensor] = []
    eval_step_list: list[int] = []
    eval_results: list[dict] = []
    start = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        optimizer.set_learning_rate(schedule(step))
        inputs, targets = make_batch(make_rng(seed, step))
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        if gradient_clip_threshold is not None:
            scale = torch.clamp(gradient_clip_threshold / norm, max=1.0)
            torch._foreach_mul_(grads, scale)
        optimizer.step()
        losses.append(loss.detach())
        gradient_norms.append(norm.detach())
        if evaluation_fn is not None and step in eval_step_set:
            eval_step_list.append(step)
            eval_results.append(evaluation_fn(model))
    train_loss = torch.stack(losses).cpu().numpy().astype(np.float64)
    gradient_norm = torch.stack(gradient_norms).cpu().numpy().astype(np.float64)
    threshold = np.inf if gradient_clip_threshold is None else gradient_clip_threshold
    return {
        "train_loss": train_loss,
        "gradient_norm": gradient_norm,
        "clip_triggered": gradient_norm > threshold,
        "eval_step": eval_step_list,
        "eval_results": eval_results,
        "seconds": time.time() - start,
    }


@torch.no_grad()
def teacher_forced_evaluation(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device | str,
    batch_size: int = 256,
) -> dict:
    """正解の履歴を入力に与えたときの、目標が ``IGNORE_INDEX`` でない位置の評価(FP32)。

    Returns:
        ``nll``(負の対数尤度の平均、nats)と ``accuracy``(``argmax`` が目標に一致した位置の割合)。
    """
    was_training = model.training
    model.eval()
    total, correct, count = 0.0, 0, 0
    for start in range(0, inputs.size(0), batch_size):
        x = inputs[start : start + batch_size].to(device)
        y = targets[start : start + batch_size].to(device)
        logits = model(x).float()
        valid = y != IGNORE_INDEX
        total += float(
            functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        )
        correct += int(((logits.argmax(dim=-1) == y) & valid).sum())
        count += int(valid.sum())
    model.train(was_training)
    return {"nll": total / count, "accuracy": correct / count}


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    prefix: Tensor,
    num_new_tokens: int,
    device: torch.device | str,
    batch_size: int = 256,
) -> Tensor:
    """各ステップで文脈全体から次のトークンを貪欲に選んで生成する。形状 ``(N, num_new_tokens)``。"""
    was_training = model.training
    model.eval()
    chunks = []
    for start in range(0, prefix.size(0), batch_size):
        tokens = prefix[start : start + batch_size].to(device)
        for _ in range(num_new_tokens):
            next_token = model(tokens)[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        chunks.append(tokens[:, -num_new_tokens:].cpu())
    model.train(was_training)
    return torch.cat(chunks, dim=0)


def evaluate_selective_copying(
    model: nn.Module,
    tokens: Tensor,
    task: SelectiveCopyingTask,
    device: torch.device | str,
    batch_size: int = 256,
) -> dict:
    """Selective copying の評価。区切りまでを与え、答えを貪欲に生成して正解と比べる。

    Returns:
        ``answers``(生成した答え、形状 ``(N, num_data)``)、``correct``(同じ形の bool)、
        ``token_accuracy``(トークン単位の正解率)、``exact_match``(系列全体が一致した割合)、
        ``position_accuracy``(答えの位置ごとの正解率、形状 ``(num_data,)``)。
    """
    prefix = tokens[:, : task.input_length + 1]
    target = tokens[:, task.input_length + 1 :]
    answers = greedy_generate(model, prefix, task.num_data, device, batch_size)
    correct = (answers == target).numpy()
    return {
        "answers": answers.numpy(),
        "correct": correct,
        "token_accuracy": float(correct.mean()),
        "exact_match": float(correct.all(axis=1).mean()),
        "position_accuracy": correct.mean(axis=0),
    }


@torch.no_grad()
def evaluate_induction_heads(
    model: nn.Module,
    tokens: Tensor,
    answers: Tensor,
    device: torch.device | str,
    batch_size: int = 64,
) -> dict:
    """Induction heads の評価。末尾の位置で貪欲に選んだトークンを正解と比べる。

    Returns:
        ``predictions``(形状 ``(N,)``)、``correct``(形状 ``(N,)`` の bool)、``accuracy``。
    """
    was_training = model.training
    model.eval()
    predictions = []
    for start in range(0, tokens.size(0), batch_size):
        x = tokens[start : start + batch_size].to(device)
        predictions.append(model(x)[:, -1, :].argmax(dim=-1).cpu())
    model.train(was_training)
    predictions = torch.cat(predictions)
    correct = (predictions == answers).numpy()
    return {
        "predictions": predictions.numpy(),
        "correct": correct,
        "accuracy": float(correct.mean()),
    }
