"""テキスト埋め込みの対照学習の損失と学習ループ(024)。

**損失**(記号は 024 のノートブックの 3 節に合わせる。``u_i``・``v_j`` は L2 正規化した query・
passage の埋め込み、``s_ij = u_i^T v_j`` はコサイン類似度、``N`` はバッチサイズ、``tau`` は温度):

- ``info_nce_loss()``: in-batch negatives の InfoNCE。i 番目の query の正例は i 番目の passage
  で、同じバッチの他の ``N - 1`` 個の passage を負例にする。
  ``L = -(1/N) sum_i log( exp(s_ii / tau) / sum_j exp(s_ij / tau) )``(query -> passage の方向)。
  ``symmetric=True`` なら passage -> query の方向も足して 2 で割る(020 の
  ``softmax_contrastive_loss()`` で ``logit_scale = log(1 / tau)`` としたものと一致する)。

- ``matryoshka_representation_loss()``: Matryoshka Representation Learning(Kusupati et al.,
  NeurIPS 2022)の損失。入れ子の次元の集合 ``M`` の各 ``m`` について、**プーリングした正規化前の
  ベクトルの先頭 ``m`` 次元を L2 正規化し直した** 埋め込みの InfoNCE を計算し、重み付きで足す。
  ``L = sum_{m in M} c_m L_m``。原論文は ``c_m = 1`` だが、本実装の既定は ``c_m = 1 / |M|``(平均)
  にする。全体が ``|M|`` 倍になるだけで最適解は変わらず、AdamW の更新は損失の尺度に依らない。違
  いが出るのは gradient clipping の閾値との関係だけで、平均にすると通常の InfoNCE と勾配のノルム
  の尺度が揃う。

**学習ループ**(``train_text_embedding()``): 019・020 と同じ部品を使う。007 の ``AdamW``
(``foreach=True``)、warmup + cosine の学習率、gradient clipping(クリッピング前の全パラメータの勾
配の L2 ノルムが閾値を超えたら縮める)、011 の ``DynamicLossScaler`` と FP16 の autocast(CUDA の
ときのみ)。encoder の順伝播のみを autocast の中で行い、埋め込みの正規化・類似度・損失は FP32 で
計算する。

**データ**: 学習全体の組(記事・query の開始位置・passage の開始位置、``sample_pair_schedule()``)
を学習の前に作ってデバイスに置き、各ステップでは切り出すだけにする。同じシードの学習どうしは、条
件によらず同じ組を同じ順序で見る。

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
from src.training.classification import (
    split_weight_decay_parameters,
    unscale_gradients_and_compute_norm,
)
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate


def info_nce_loss(
    query_embeddings: Tensor,
    passage_embeddings: Tensor,
    temperature: float,
    symmetric: bool = False,
) -> Tensor:
    """in-batch negatives の InfoNCE 損失(query -> passage の方向。``symmetric`` なら両方向の平均)。

    埋め込みは L2 正規化済みで、i 番目の query と i 番目の passage が正例の組。
    """
    logits = query_embeddings @ passage_embeddings.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = functional.cross_entropy(logits, targets)
    if symmetric:
        loss = 0.5 * (loss + functional.cross_entropy(logits.t(), targets))
    return loss


def matryoshka_representation_loss(
    pooled_queries: Tensor,
    pooled_passages: Tensor,
    dimensions: Sequence[int],
    temperature: float,
    weights: Sequence[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Matryoshka Representation Learning の損失 ``sum_m c_m L_m``。

    Args:
        pooled_queries: 形状 ``(N, d)``。プーリングした **正規化前の** query のベクトル。
        pooled_passages: 形状 ``(N, d)``。同、passage。
        dimensions: 入れ子の次元の集合 ``M``(各 ``m`` は ``d`` 以下)。
        temperature: 温度 ``tau``。
        weights: 次元ごとの重み ``c_m``。``None`` なら ``1 / |M|``。

    Returns:
        ``(全体の損失, 次元ごとの損失 L_m の形状 (|M|,) のテンソル)``。
    """
    if weights is None:
        weights = [1.0 / len(dimensions)] * len(dimensions)
    per_dimension = torch.stack(
        [
            info_nce_loss(
                truncate_and_normalize(pooled_queries, m),
                truncate_and_normalize(pooled_passages, m),
                temperature,
            )
            for m in dimensions
        ]
    )
    weight_tensor = torch.tensor(
        list(weights), device=per_dimension.device, dtype=per_dimension.dtype
    )
    return (weight_tensor * per_dimension).sum(), per_dimension.detach()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_text_embedding(
    model: TextEmbeddingModel,
    token_stream: Tensor,
    article_starts: np.ndarray,
    schedule: dict[str, np.ndarray],
    query_length: int,
    passage_length: int,
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
    """テキスト埋め込みモデルを、``schedule`` の全ステップ分、対照学習の損失で学習する。

    Args:
        model: ``TextEmbeddingModel``(``token_stream`` と同じデバイスに置いておく)。
        token_stream: 学習用の記事を連結したトークン ID(形状 ``(total,)``、int64、``model`` と
            同じデバイス)。
        article_starts: 学習用の各記事の ``token_stream`` での先頭の位置
            (``concatenate_articles()`` の返り値)。
        schedule: ``sample_pair_schedule()`` の返り値(``"article"``・``"query_start"``・
            ``"passage_start"``、形状 ``(T, N)``)。``T`` が学習ステップ数、``N`` がバッチサイズ。
        query_length: ``L_q``。
        passage_length: ``L_p``。
        peak_learning_rate: warmup の後の学習率の最大値。
        warmup_steps: 線形 warmup のステップ数。
        min_learning_rate: cosine decay の下限。
        weight_decay: 重み減衰(行列形の重みのみ)。
        gradient_clip_threshold: gradient clipping の閾値。
        temperature: InfoNCE の温度 ``tau``(固定)。
        matryoshka_dimensions: 指定すると、その次元の集合による Matryoshka Representation Learning
            の損失で学習する。``None`` なら通常の InfoNCE(全次元)で学習する。
        use_fp16_autocast: True なら encoder の順伝播を FP16 の autocast で行い、動的損失
            スケーリングを使う(CUDA のみ)。
        init_loss_scale: 動的損失スケーリングの初期スケール値。
        loss_scale_growth_interval: スケール値を growth するまでの連続成功ステップ数。
        evaluation_steps: 学習の途中で ``evaluation_fn`` を呼ぶステップ(そのステップの更新の後)。
        evaluation_fn: モデルを受け取り、評価結果の辞書を返す関数。

    Returns:
        学習の記録の辞書。``"loss"``・``"gradient_norm"``・``"gradient_clip_triggered"``・
        ``"learning_rate"``・``"loss_scale"``・``"step_skipped"``・``"in_batch_margin"`` は
        長さ ``T`` のリスト。``"dimension_losses"`` は Matryoshka Representation Learning のとき
        のみで、形状 ``(T, len(matryoshka_dimensions))``。``"evaluations"`` は ステップ ->
        ``evaluation_fn`` の返り値。``"data_stream_hash"`` は全ステップの組の位置の SHA-256、
        ``"train_seconds"`` は評価を除く学習ループの実時間。
    """
    device = token_stream.device
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")
    num_steps, batch_size = schedule["article"].shape
    starts = torch.as_tensor(
        np.asarray(article_starts)[schedule["article"]], device=device
    )  # (T, N): 各組の記事の先頭の位置
    query_begin = starts + torch.as_tensor(schedule["query_start"], device=device)
    passage_begin = starts + torch.as_tensor(schedule["passage_start"], device=device)
    query_offsets = torch.arange(query_length, device=device)
    passage_offsets = torch.arange(passage_length, device=device)

    digest = hashlib.sha256()
    for key in ("article", "query_start", "passage_start"):
        digest.update(np.ascontiguousarray(schedule[key], dtype=np.int64).tobytes())

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
    history: dict = {
        "learning_rate": [],
        "loss_scale": [],
        "step_skipped": [],
        "evaluations": {},
    }
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

        queries = token_stream[query_begin[step - 1][:, None] + query_offsets[None, :]]
        passages = token_stream[passage_begin[step - 1][:, None] + passage_offsets[None, :]]
        for optimizer in optimizers:
            optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
        ):
            pooled_queries = model.pooled(queries)
            pooled_passages = model.pooled(passages)
        pooled_queries, pooled_passages = pooled_queries.float(), pooled_passages.float()
        if matryoshka_dimensions:
            loss, per_dimension = matryoshka_representation_loss(
                pooled_queries, pooled_passages, matryoshka_dimensions, temperature
            )
        else:
            query_embeddings = truncate_and_normalize(pooled_queries)
            passage_embeddings = truncate_and_normalize(pooled_passages)
            loss = info_nce_loss(query_embeddings, passage_embeddings, temperature)
        (loss_scaler.scale_loss(loss) if loss_scaler is not None else loss).backward()

        gradient_norm = unscale_gradients_and_compute_norm(
            parameters, loss_scaler.scale if loss_scaler is not None else None
        )
        # 007 と同じ定義: ノルムが閾値を超えたら threshold / norm 倍する(CPU に読み出さずに行う)
        clip_coefficient = torch.where(
            gradient_norm > threshold, threshold / gradient_norm, torch.ones_like(gradient_norm)
        )
        torch._foreach_mul_([p.grad for p in parameters if p.grad is not None], clip_coefficient)

        found_inf = False
        if loss_scaler is not None:  # FP16 のときのみ、ここで 1 回だけ CPU と同期する
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
    history["data_stream_hash"] = digest.hexdigest()
    history["train_seconds"] = train_seconds
    return history
