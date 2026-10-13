"""可変長の cross-encoder の学習(026)。

025 の ``train_reranker()`` (``src/training/reranking.py``)と同じ損失・同じ学習の部品で、query
と passage が
**学習の前に作った可変長のトークン列の表**(右側をパディングして長さの配列を添えたもの)になる。

**損失**: 1 つの query に、正例 ``p^+`` 1 個と負例 ``p^-_1, ..., p^-_n`` ``n`` 個の候補を対応させ、
``n + 1`` 個の候補の上の softmax 交差エントロピーを最小にする。in-batch negatives は使わない。

**学習の組**(``sample_cloze_reranking_schedule()``): 026 の cross-encoder は、passage 単位の関連性
(正例は query の文を抜いた元の passage の 1 つだけ)で学習する。**学習の入力の分布を、推論時の候補に
揃える**ため、正例が第 1 段の上位 ``K`` 件に入る query だけを使い、負例も同じ上位 ``K`` 件から(
正例を
除いて)選ぶ。

記号 / Notation:
    B : 1 ステップの query の数、n : 負例の数、T : 学習ステップ数、K : 上位の件数
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from src.training.classification import (
    split_weight_decay_parameters,
    unscale_gradients_and_compute_norm,
)
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.reranking import softmax_cross_entropy_over_candidates
from src.training.schedule import compute_warmup_cosine_learning_rate


@dataclass
class ClozeRerankingSchedule:
    """学習の組(学習の前に全ステップぶんを作る)。

    Attributes:
        query_ids: 形状 ``(T, B)``。query の表(索引の passage と同じ番号)の添字。
        candidate_ids: 形状 ``(T, B, 1 + n)``。passage の表の添字(0 番目が正例)。
        num_usable: 学習に使える query の数(正例が上位 ``K`` 件に入る query)。
    """

    query_ids: np.ndarray
    candidate_ids: np.ndarray
    num_usable: int

    def digest(self) -> str:
        h = hashlib.sha256()
        for array in (self.query_ids, self.candidate_ids):
            h.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
        return h.hexdigest()


def usable_cloze_queries(mined_candidates: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """正例が上位の候補に入っている query の真偽 ``(len(positives),)``。"""
    return (mined_candidates == positives[:, None]).any(axis=1)


def sample_cloze_reranking_schedule(
    train_queries: np.ndarray,
    mined_candidates: np.ndarray,
    num_steps: int,
    batch_queries: int,
    num_negatives: int,
    seed: int,
) -> ClozeRerankingSchedule:
    """学習全体の組を、学習の前に作る(決定的)。

    Args:
        train_queries: 学習用の query の添字(昇順。正例は同じ添字の passage)。
        mined_candidates: 形状 ``(len(train_queries), K)``。各 query の第 1 段の上位 ``K`` 件(
        passage の添字)。
        num_steps: ``T``。
        batch_queries: ``B``。
        num_negatives: ``n``(``K - 1`` 以下)。
        seed: 乱数シード。
    """
    positives = np.asarray(train_queries)
    usable = np.nonzero(usable_cloze_queries(mined_candidates, positives))[0]
    if len(usable) < batch_queries:
        raise ValueError(f"学習に使える query が少なすぎる: {len(usable)} < B = {batch_queries}")
    if mined_candidates.shape[1] - 1 < num_negatives:
        raise ValueError("上位の件数が負例の数 + 1 より少ない")
    rng = np.random.default_rng([seed, 0])
    query_out = np.empty((num_steps, batch_queries), dtype=np.int64)
    candidate_out = np.empty((num_steps, batch_queries, 1 + num_negatives), dtype=np.int64)
    for step in range(num_steps):
        positions = usable[rng.choice(len(usable), size=batch_queries, replace=False)]
        for b, position in enumerate(positions.tolist()):
            positive = int(positives[position])
            others = mined_candidates[position][mined_candidates[position] != positive]
            chosen = others[rng.choice(len(others), size=num_negatives, replace=False)]
            query_out[step, b] = positive
            candidate_out[step, b, 0] = positive
            candidate_out[step, b, 1:] = chosen
    return ClozeRerankingSchedule(query_out, candidate_out, len(usable))


class VariableTables:
    """query と passage のトークンの表(デバイスに置く)と、長さの配列(CPU・デバイス)。"""

    def __init__(
        self,
        query_tokens: np.ndarray,
        query_lengths: np.ndarray,
        passage_tokens: np.ndarray,
        passage_lengths: np.ndarray,
        device: torch.device,
    ) -> None:
        self.query_tokens = torch.as_tensor(query_tokens, device=device)
        self.passage_tokens = torch.as_tensor(passage_tokens, device=device)
        self.query_lengths_cpu = np.asarray(query_lengths, dtype=np.int64)
        self.passage_lengths_cpu = np.asarray(passage_lengths, dtype=np.int64)
        self.query_lengths = torch.as_tensor(self.query_lengths_cpu, device=device)
        self.passage_lengths = torch.as_tensor(self.passage_lengths_cpu, device=device)
        self.device = device

    def step_inputs(self, query_ids: np.ndarray, candidate_ids: np.ndarray):
        """1 ステップぶんの ``(query 反復 (B(1+n), Lq'), passage (B(1+n), Lp'), 各長さ)``
        とスコアの形 ``(B, 1+n)``。"""
        batch, width = candidate_ids.shape
        ql = self.query_lengths_cpu[query_ids]
        pl = self.passage_lengths_cpu[candidate_ids]
        q = self.query_tokens[torch.as_tensor(query_ids, device=self.device)][:, : int(ql.max())]
        p = self.passage_tokens[torch.as_tensor(candidate_ids.reshape(-1), device=self.device)][
            :, : int(pl.max())
        ]
        q_rep = q[:, None, :].expand(batch, width, q.size(1)).reshape(batch * width, -1)
        ql_rep = (
            self.query_lengths[torch.as_tensor(query_ids, device=self.device)][:, None]
            .expand(batch, width)
            .reshape(-1)
        )
        pl_flat = self.passage_lengths[
            torch.as_tensor(candidate_ids.reshape(-1), device=self.device)
        ]
        return q_rep, ql_rep, p, pl_flat, (batch, width)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def candidate_scores(
    model: nn.Module,
    tables: VariableTables,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    use_fp16_autocast: bool,
) -> Tensor:
    q, ql, p, pl, shape = tables.step_inputs(query_ids, candidate_ids)
    with torch.autocast(
        device_type=tables.device.type, dtype=torch.float16, enabled=use_fp16_autocast
    ):
        scores = model(q, p, ql, pl)
    return scores.float().reshape(shape)


def train_reranker_variable(
    model: nn.Module,
    tables: VariableTables,
    schedule: ClozeRerankingSchedule,
    peak_learning_rate: float,
    warmup_steps: int,
    min_learning_rate: float,
    weight_decay: float,
    gradient_clip_threshold: float,
    use_fp16_autocast: bool = False,
    init_loss_scale: float = 2.0**16,
    loss_scale_growth_interval: int = 2000,
    evaluation_steps: tuple[int, ...] = (),
    evaluation_fn: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """可変長の cross-encoder を、``schedule`` の全ステップぶん学習する(全パラメータを更新する)。

    返り値は ``train_reranker()`` と同じ形式(
    ``"loss"``・``"gradient_norm"``・``"gradient_clip_triggered"``・
    ``"learning_rate"``・``"loss_scale"``・``"step_skipped"``・``"evaluations"``・``"data_stream_hash"``・
    ``"train_seconds"``)。
    """
    device = tables.device
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")
    num_steps = schedule.query_ids.shape[0]
    decay, no_decay = split_weight_decay_parameters(model)
    optimizers = [
        AdamW(decay, lr=peak_learning_rate, weight_decay=weight_decay, foreach=True),
        AdamW(no_decay, lr=peak_learning_rate, weight_decay=0.0, foreach=True),
    ]
    parameters = decay + no_decay
    loss_scaler = (
        DynamicLossScaler(init_loss_scale, growth_interval=loss_scale_growth_interval)
        if use_fp16_autocast
        else None
    )
    threshold = torch.tensor(gradient_clip_threshold, device=device)
    losses = torch.empty(num_steps, device=device)
    gradient_norms = torch.empty(num_steps, device=device)
    history: dict = {"learning_rate": [], "loss_scale": [], "step_skipped": [], "evaluations": {}}
    evaluation_steps = tuple(sorted(set(evaluation_steps)))
    train_seconds = 0.0
    start_time = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        learning_rate = compute_warmup_cosine_learning_rate(
            step, warmup_steps, num_steps, peak_learning_rate, min_learning_rate
        )
        for optimizer in optimizers:
            optimizer.set_learning_rate(learning_rate)
            optimizer.zero_grad()
        scores = candidate_scores(
            model,
            tables,
            schedule.query_ids[step - 1],
            schedule.candidate_ids[step - 1],
            use_fp16_autocast,
        )
        loss = softmax_cross_entropy_over_candidates(scores)
        (loss_scaler.scale_loss(loss) if loss_scaler is not None else loss).backward()
        gradient_norm = unscale_gradients_and_compute_norm(
            parameters, loss_scaler.scale if loss_scaler is not None else None
        )
        clip_coefficient = torch.where(
            gradient_norm > threshold, threshold / gradient_norm, torch.ones_like(gradient_norm)
        )
        torch._foreach_mul_([p.grad for p in parameters if p.grad is not None], clip_coefficient)
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
        with torch.no_grad():
            losses[step - 1] = loss.detach()
            gradient_norms[step - 1] = gradient_norm.detach()
        history["learning_rate"].append(learning_rate)
        history["step_skipped"].append(found_inf)
        if evaluation_fn is not None and step in evaluation_steps:
            _synchronize(device)
            train_seconds += time.time() - start_time
            history["evaluations"][step] = evaluation_fn(model)
            start_time = time.time()
    _synchronize(device)
    train_seconds += time.time() - start_time
    history["loss"] = losses.cpu().tolist()
    history["gradient_norm"] = gradient_norms.cpu().tolist()
    history["gradient_clip_triggered"] = [
        g > gradient_clip_threshold for g in history["gradient_norm"]
    ]
    history["data_stream_hash"] = schedule.digest()
    history["train_seconds"] = train_seconds
    return history


@torch.no_grad()
def compute_reranking_losses_variable(
    model: nn.Module,
    tables: VariableTables,
    schedule: ClozeRerankingSchedule,
    steps,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """``schedule`` の指定したステップの組での ``model`` の損失(勾配なし。1 ステップごと)。"""
    was_training = model.training
    model.eval()
    losses = []
    for step in steps:
        scores = candidate_scores(
            model, tables, schedule.query_ids[step], schedule.candidate_ids[step], use_fp16_autocast
        )
        losses.append(softmax_cross_entropy_over_candidates(scores))
    model.train(was_training)
    return torch.stack(losses).cpu().numpy().astype(np.float64)
