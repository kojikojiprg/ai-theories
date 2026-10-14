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
from torch.nn import functional

from src.models.text_embedding import TextEmbeddingModel, truncate_and_normalize
from src.models.variable_length import pooled_with_lengths
from src.training.contrastive_text import info_nce_loss, matryoshka_representation_loss
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate
from src.training.update_step import (
    build_weight_decay_split_optimizers,
    clip_gradients_to_threshold,
    flag_clipped_steps,
    step_optimizers_and_update_loss_scale,
    unscale_gradients_and_compute_norm,
)
from src.utils.device import synchronize_device


def masked_info_nce_loss(
    query_embeddings: Tensor, passage_embeddings: Tensor, temperature: float, ignore: Tensor
) -> Tensor:
    """in-batch negatives の InfoNCE(query -> passage の方向)で、``ignore`` が真の組を負例から除く。

    ``ignore[i, j]`` が真なら、query ``i`` の候補から passage ``j`` を除く(対角は偽にする。
    正例は残す)。
    記事単位の定義では、同じ記事の他の組の passage は関連があるので、負例から除く。

    導入: 026。

    Args:
        query_embeddings: 形状 ``(N, d)`` の L2 正規化済みの query の埋め込み。
        passage_embeddings: 形状 ``(N, d)`` の L2 正規化済みの passage の埋め込み。
        temperature: 温度 ``tau``。
        ignore: 形状 ``(N, N)`` の bool。真の組を負例から除く(対角は偽)。

    Returns:
        損失(形状 ``()``)。
    """
    logits = query_embeddings @ passage_embeddings.t() / temperature
    logits = logits.masked_fill(ignore, float("-inf"))
    targets = torch.arange(logits.size(0), device=logits.device)
    return functional.cross_entropy(logits, targets)


def masked_matryoshka_representation_loss(
    pooled_queries: Tensor,
    pooled_passages: Tensor,
    dimensions: Sequence[int],
    temperature: float,
    ignore: Tensor,
) -> tuple[Tensor, Tensor]:
    """``matryoshka_representation_loss()`` の各次元の項を ``masked_info_nce_loss()``
    にしたもの。

    導入: 026。

    Args:
        pooled_queries: 形状 ``(N, d)`` の正規化前の query のベクトル。
        pooled_passages: 形状 ``(N, d)`` の正規化前の passage のベクトル。
        dimensions: 損失を取る次元の集合(先頭から切り詰める次元)。
        temperature: 温度 ``tau``。
        ignore: 形状 ``(N, N)`` の bool。真の組を負例から除く。

    Returns:
        ``(各次元の損失の等重みの平均, 次元ごとの損失 (len(dimensions),))``。
    """
    weights = [1.0 / len(dimensions)] * len(dimensions)
    per_dimension = torch.stack(
        [
            masked_info_nce_loss(
                truncate_and_normalize(pooled_queries, m),
                truncate_and_normalize(pooled_passages, m),
                temperature,
                ignore,
            )
            for m in dimensions
        ]
    )
    weight_tensor = torch.tensor(weights, device=per_dimension.device, dtype=per_dimension.dtype)
    return (weight_tensor * per_dimension).sum(), per_dimension.detach()


def batches_digest(*arrays: np.ndarray) -> str:
    """組の系列(トークンと長さ)の SHA-256。同じシードの学習で組が揃うことの確認に使う。

    導入: 026。

    Args:
        *arrays: ハッシュする配列(int64 に変換して連結する)。

    Returns:
        SHA-256 の 16 進文字列。
    """
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


class VariableBatches:
    """学習の組(``(T, N, L)`` のトークンと ``(T, N)`` の長さ)をデバイスに置き、
    ステップごとに切り出す。

    導入: 026。

    Args:
        query_tokens: 形状 ``(T, N, L_q)``。query のトークン(右側がパディング)。
        query_lengths: 形状 ``(T, N)``。query の有効な長さ。
        passage_tokens: 形状 ``(T, N, L_p)``。passage のトークン。
        passage_lengths: 形状 ``(T, N)``。passage の有効な長さ。
        device: 置くデバイス。
        article_ids: 形状 ``(T, N)`` の記事の番号(``mask_same_article`` のときのみ使う)。
        mask_same_article: 真なら、同じ記事の他の組の passage を負例から除く。
    """

    def __init__(
        self,
        query_tokens: np.ndarray,
        query_lengths: np.ndarray,
        passage_tokens: np.ndarray,
        passage_lengths: np.ndarray,
        device: torch.device,
        article_ids: np.ndarray | None = None,
        mask_same_article: bool = False,
    ) -> None:
        if mask_same_article and article_ids is None:
            raise ValueError("mask_same_article には article_ids が要る")
        self.mask_same_article = mask_same_article
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
        self.ignore_masks: Tensor | None = None
        extra = [np.zeros(1)]
        if mask_same_article:
            ids = torch.as_tensor(np.asarray(article_ids), device=device)
            same = ids[:, :, None] == ids[:, None, :]
            eye = torch.eye(self.batch_size, dtype=torch.bool, device=device)[None]
            self.ignore_masks = same & ~eye
            extra = [np.asarray(article_ids), np.ones(1)]
        self.digest = batches_digest(
            query_tokens, query_lengths, passage_tokens, passage_lengths, *extra
        )

    def ignore(self, index: int) -> Tensor | None:
        """ステップ ``index`` で、負例から除く組の真偽 ``(N, N)``(除かないなら ``None``)。"""
        return None if self.ignore_masks is None else self.ignore_masks[index]

    def negatives_per_row(self) -> np.ndarray:
        """ステップごとの、1 行あたりの負例の数の平均(同じ記事の組を除いた後)。"""
        if self.ignore_masks is None:
            return np.full(self.num_steps, self.batch_size - 1, dtype=np.float64)
        return (
            (self.batch_size - 1 - self.ignore_masks.sum(dim=2).float().mean(dim=1)).cpu().numpy()
        )

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
    ignore: Tensor | None = None,
):
    if ignore is not None:
        if dimensions:
            return masked_matryoshka_representation_loss(
                pooled_queries, pooled_passages, dimensions, temperature, ignore
            )
        loss = masked_info_nce_loss(
            truncate_and_normalize(pooled_queries),
            truncate_and_normalize(pooled_passages),
            temperature,
            ignore,
        )
        return loss, None
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

    導入: 026。

    Args:
        model: 学習する ``TextEmbeddingModel``(``batches`` と同じデバイス)。
        batches: 全ステップの組(``VariableBatches``)。
        peak_learning_rate: warmup の後の学習率の最大値。
        warmup_steps: 線形 warmup のステップ数。
        min_learning_rate: cosine decay の下限。
        weight_decay: 重み減衰(行列形の重みのみ)。
        gradient_clip_threshold: gradient clipping の閾値。
        temperature: InfoNCE の温度 ``tau``。
        matryoshka_dimensions: 指定すると、その次元の集合による Matryoshka Representation Learning
            の損失で学習する(``None`` なら全次元の InfoNCE)。
        use_fp16_autocast: 真なら FP16 の autocast と動的損失スケーリングを使う(CUDA のみ)。
        init_loss_scale: 動的損失スケーリングの初期スケール値。
        loss_scale_growth_interval: スケール値を増やすまでの連続成功ステップ数。
        evaluation_steps: 学習の途中で ``evaluation_fn`` を呼ぶステップ(更新の後)。
        evaluation_fn: モデルを受け取り、評価結果の辞書を返す関数。

    Returns:
        学習の記録の辞書(キーは上の説明のとおり)。
    """
    device = batches.query_tokens.device
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")
    num_steps, batch_size = batches.num_steps, batches.batch_size
    optimizers, parameters = build_weight_decay_split_optimizers(
        model, peak_learning_rate, weight_decay
    )
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
        ignore = batches.ignore(step - 1)
        loss, per_dimension = _loss(
            pooled_queries, pooled_passages, temperature, matryoshka_dimensions, ignore
        )
        (loss_scaler.scale_loss(loss) if loss_scaler is not None else loss).backward()
        gradient_norm = unscale_gradients_and_compute_norm(
            parameters, loss_scaler.scale if loss_scaler is not None else None
        )
        clip_gradients_to_threshold(parameters, gradient_norm, threshold)
        found_inf = step_optimizers_and_update_loss_scale(
            optimizers, gradient_norm, loss_scaler, history
        )
        with torch.no_grad():
            losses[step - 1] = loss.detach()
            gradient_norms[step - 1] = gradient_norm.detach()
            full_queries = truncate_and_normalize(pooled_queries.detach())
            full_passages = truncate_and_normalize(pooled_passages.detach())
            similarity = full_queries @ full_passages.t()
            positive = similarity.diagonal()
            excluded = torch.eye(batch_size, dtype=torch.bool, device=device)
            if ignore is not None:
                excluded = excluded | ignore
            hardest = similarity.masked_fill(excluded, float("-inf"))
            margins[step - 1] = (positive - hardest.max(dim=1).values).mean()
            if dimension_losses is not None:
                dimension_losses[step - 1] = per_dimension
        history["learning_rate"].append(learning_rate)
        history["step_skipped"].append(found_inf)
        if evaluation_fn is not None and step in evaluation_steps:
            synchronize_device(device)
            train_seconds += time.time() - start_time
            history["evaluations"][step] = evaluation_fn(model)
            start_time = time.time()
    synchronize_device(device)
    train_seconds += time.time() - start_time
    history["loss"] = losses.cpu().tolist()
    history["gradient_norm"] = gradient_norms.cpu().tolist()
    history["gradient_clip_triggered"] = flag_clipped_steps(
        history["gradient_norm"], gradient_clip_threshold
    )
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

    導入: 026。

    Args:
        model: 評価する ``TextEmbeddingModel``。
        batches: 組(``VariableBatches``)。
        steps: 損失を測るステップの番号(0 始まり)。
        temperature: InfoNCE の温度 ``tau``。
        matryoshka_dimensions: Matryoshka Representation Learning の次元の集合(``None`` なら全次元)
            。
        use_fp16_autocast: 真なら FP16 の autocast で計算する(CUDA のみ)。

    Returns:
        ステップごとの損失(形状 ``(len(steps),)`` の配列)。
    """
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for step in steps:
            pooled_queries, pooled_passages = _embed_pair(model, batches, step, use_fp16_autocast)
            loss, _ = _loss(
                pooled_queries,
                pooled_passages,
                temperature,
                matryoshka_dimensions,
                batches.ignore(step),
            )
            losses.append(loss)
    model.train(was_training)
    return torch.stack(losses).cpu().numpy().astype(np.float64)
