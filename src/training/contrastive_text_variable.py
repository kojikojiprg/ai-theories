"""可変長の組での、テキスト埋め込みの対照学習の学習ループ(026)。

024 の ``train_text_embedding()`` (``src/training/contrastive_text.py``)は、
学習用の記事の連結から固定
長の区間を切り出す。026 の Inverse Cloze Task は、抜いた文(query)と文を除いた文脈(passage)が
**学習の前に作った可変長のトークン列**(右側をパディングして長さの配列を添えたもの)になる。
このモジュールの ``train_text_embedding_from_batches()`` は、学習の部品(007 の ``AdamW``、warmup +
cosine の学習率、gradient clipping、011 の動的損失スケーリングと FP16 の autocast(CUDA のみ)、
Matryoshka Representation Learning の損失)を 024 と **同じ順序・同じ演算** で使う。
**すべての行が満長のとき**(024 の記事単位の定義の組)は、``model.pooled()`` と同じ経路になり、024 の
``train_text_embedding()`` と同じ損失の系列になる(026 のノートブックの単体テストで確かめる)。

``compute_pair_losses_from_batches()`` は、更新を行わずに同じ組・同じ損失を測る(学習の成立の前提条件
の分母。学習率 0 の学習の訓練損失と一致する)。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.models.text_embedding import TextEmbeddingModel, truncate_and_normalize
from src.models.variable_length import pooled_with_lengths
from src.training.classification import (
    split_weight_decay_parameters,
    unscale_gradients_and_compute_norm,
)
from src.training.contrastive_text import info_nce_loss, matryoshka_representation_loss
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def batches_digest(*arrays: np.ndarray) -> str:
    """組の系列(トークンと長さ)の SHA-256。同じシードの学習で組が揃うことの確認に使う。"""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


class VariableBatches:
    """学習の組(``(T, N, L)`` のトークンと ``(T, N)`` の長さ)をデバイスに置き、
    ステップごとに切り出す。"""

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
        self.num_steps, self.batch_size = self.query_lengths_cpu.shape
        self.query_width = self.query_tokens.size(2)
        self.passage_width = self.passage_tokens.size(2)
        self._query_max = self.query_lengths_cpu.max(axis=1)
        self._passage_max = self.passage_lengths_cpu.max(axis=1)
        self._query_full = (self.query_lengths_cpu == self.query_width).all(axis=1)
        self._passage_full = (self.passage_lengths_cpu == self.passage_width).all(axis=1)
        self.digest = batches_digest(query_tokens, query_lengths, passage_tokens, passage_lengths)

    def step(self, index: int) -> tuple[Tensor, Tensor, bool, Tensor, Tensor, bool]:
        """``(query, query の長さ, query が満長か, passage, passage の長さ, passage が満長か)``。"""
        qw, pw = int(self._query_max[index]), int(self._passage_max[index])
        return (
            self.query_tokens[index, :, :qw],
            self.query_lengths[index],
            bool(self._query_full[index]),
            self.passage_tokens[index, :, :pw],
            self.passage_lengths[index],
            bool(self._passage_full[index]),
        )


def _embed_pair(
    model: TextEmbeddingModel, batches: VariableBatches, index: int, use_fp16_autocast: bool
):
    device = batches.query_tokens.device
    q, ql, q_full, p, pl, p_full = batches.step(index)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast):
        pooled_queries = pooled_with_lengths(model, q, ql, full_length=q_full)
        pooled_passages = pooled_with_lengths(model, p, pl, full_length=p_full)
    return pooled_queries.float(), pooled_passages.float()


def _loss(
    pooled_queries: Tensor,
    pooled_passages: Tensor,
    temperature: float,
    dimensions: Sequence[int] | None,
):
    if dimensions:
        return matryoshka_representation_loss(
            pooled_queries, pooled_passages, dimensions, temperature
        )
    return info_nce_loss(
        truncate_and_normalize(pooled_queries), truncate_and_normalize(pooled_passages), temperature
    ), None


def train_text_embedding_from_batches(
    model: TextEmbeddingModel,
    batches: VariableBatches,
    peak_learning_rate: float,
    warmup_steps: int,
    min_learning_rate: float,
    weight_decay: float,
    gradient_clip_threshold: float,
    temperature: float,
    matryoshka_dimensions: Sequence[int] | None = None,
    use_fp16_autocast: bool = False,
    init_loss_scale: float = 2.0**16,
    loss_scale_growth_interval: int = 2000,
    evaluation_steps: tuple[int, ...] = (),
    evaluation_fn: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """``batches`` の全ステップ分、対照学習の損失で学習する(``train_text_embedding()`` と同じ手順)。

    返り値は ``train_text_embedding()`` と同じ形式の辞書(``"loss"``・``"gradient_norm"``・
    ``"gradient_clip_triggered"``・``"learning_rate"``・``"loss_scale"``・``"step_skipped"``・
    ``"in_batch_margin"``・``"dimension_losses"``(Matryoshka のとき)・``"evaluations"``・
    ``"data_stream_hash"``・``"train_seconds"``)。
    """
    device = batches.query_tokens.device
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")
    num_steps, batch_size = batches.num_steps, batches.batch_size
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
    margins = torch.empty(num_steps, device=device)
    dimension_losses = (
        torch.empty(num_steps, len(matryoshka_dimensions), device=device)
        if matryoshka_dimensions
        else None
    )
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
        for optimizer in optimizers:
            optimizer.zero_grad()
        pooled_queries, pooled_passages = _embed_pair(model, batches, step - 1, use_fp16_autocast)
        loss, per_dimension = _loss(
            pooled_queries, pooled_passages, temperature, matryoshka_dimensions
        )
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
            full_queries = truncate_and_normalize(pooled_queries.detach())
            full_passages = truncate_and_normalize(pooled_passages.detach())
            similarity = full_queries @ full_passages.t()
            positive = similarity.diagonal()
            hardest = similarity.masked_fill(
                torch.eye(batch_size, dtype=torch.bool, device=device), float("-inf")
            )
            margins[step - 1] = (positive - hardest.max(dim=1).values).mean()
            if dimension_losses is not None:
                dimension_losses[step - 1] = per_dimension
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
    history["in_batch_margin"] = margins.cpu().tolist()
    if dimension_losses is not None:
        history["dimension_losses"] = dimension_losses.cpu().numpy()
    history["data_stream_hash"] = batches.digest
    history["train_seconds"] = train_seconds
    return history


def compute_pair_losses_from_batches(
    model: TextEmbeddingModel,
    batches: VariableBatches,
    steps: Sequence[int],
    temperature: float,
    matryoshka_dimensions: Sequence[int] | None = None,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """``batches`` の指定したステップの組での ``model`` の損失(勾配なし。1 ステップごと)。

    ``train_text_embedding_from_batches()`` と同じ組・同じ損失で、更新を行わずに測る。
    呼び出しの前後で
    モデルのモードを変えない。
    """
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for step in steps:
            pooled_queries, pooled_passages = _embed_pair(model, batches, step, use_fp16_autocast)
            loss, _ = _loss(pooled_queries, pooled_passages, temperature, matryoshka_dimensions)
            losses.append(loss)
    model.train(was_training)
    return torch.stack(losses).cpu().numpy().astype(np.float64)
