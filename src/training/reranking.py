"""並べ替えモデルの学習(025)。

cross-encoder(``src/models/cross_encoder.py``)と dual encoder(024 の ``TextEmbeddingModel``)を、
**同じ組を同じ順序で見て、同じ損失の形で** 学習する。

**損失**: 1 つの query に、正例 ``p^+`` 1 個と負例 ``p^-_1, ..., p^-_n`` ``n``個の候補を対応させ、
``n + 1`` 個の候補の上の softmax 交差エントロピーを最小にする。

.. math::

    \\mathcal{L} = -\\frac{1}{B} \\sum_{b=1}^{B} \\log \\frac{\\exp(s_{b,0})}{\\sum_{j=0}^{n}
    \\exp(s_{b,j})}

``s_{b,j}`` は query ``b`` の候補 ``j`` のスコアで、``j = 0`` が正例。

- cross-encoder: ``s = g(q, p)``(スコアは線形のヘッドの出力)。
- dual encoder: ``s = f(q)^T f(p) / \\tau``(``f`` は L2 正規化した埋め込み、温度 ``\\tau`` は 024
と同じ 0.05 で固定)。

どちらも **in-batch negatives は使わない**(負例は query ごとの ``n`` 個だけ)。

**学習ループ**: 024 の ``train_text_embedding()`` と同じ部品(007 の ``AdamW``・warmup + cosine
の学習率・gradient clipping、011 の動的損失スケーリングと FP16 の autocast(CUDA のみ)、
019 の重み減衰の分割)を使う。本体の順伝播のみを autocast の中で行い、
スコアの計算(ヘッドの出力・内積・温度での割り算)と損失は FP32 で行う。

**同じ組での損失の測定**(``compute_reranking_losses()``): 学習と同じ組・同じ損失で、
更新を行わずに損失だけを測る(P1 の分母。学習率 0 の学習の訓練損失と一致する)。

記号 / Notation:
    B : 1 ステップの query の数、n : 負例の数、T : 学習ステップ数、L_q・L_p : query・passage の長さ
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from src.data.reranking import RerankingSchedule
from src.models.cross_encoder import CrossEncoder
from src.models.text_embedding import TextEmbeddingModel, truncate_and_normalize
from src.training.classification import (
    split_weight_decay_parameters,
    unscale_gradients_and_compute_norm,
)
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate


def softmax_cross_entropy_over_candidates(scores: Tensor) -> Tensor:
    """形状 ``(B, 1 + n)`` のスコア(0 列目が正例)の softmax 交差エントロピー(query の平均)。"""
    targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return functional.cross_entropy(scores, targets)


def compute_candidate_scores(
    model: nn.Module,
    kind: str,
    queries: Tensor,
    candidates: Tensor,
    temperature: float,
    use_fp16_autocast: bool,
) -> Tensor:
    """1 ステップぶんの候補のスコア ``(B, 1 + n)`` を FP32 で返す(温度は dual encoder のみ)。

    Args:
        model: ``CrossEncoder`` または ``TextEmbeddingModel``。
        kind: ``"cross_encoder"`` または ``"dual_encoder"``。
        queries: 形状 ``(B, L_q)``。
        candidates: 形状 ``(B, 1 + n, L_p)``(0 番目が正例)。
        temperature: dual encoder の温度。
        use_fp16_autocast: 真なら本体を FP16 の autocast で計算する(CUDA のみ)。
    """
    batch, width, passage_length = candidates.shape
    device_type = queries.device.type
    if kind == "cross_encoder":
        repeated = (
            queries[:, None, :].expand(batch, width, queries.size(1)).reshape(batch * width, -1)
        )
        flat = candidates.reshape(batch * width, passage_length)
        with torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_fp16_autocast
        ):
            scores = model(repeated, flat)
        return scores.float().reshape(batch, width)
    if kind == "dual_encoder":
        with torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_fp16_autocast
        ):
            pooled_queries = model.pooled(queries)
            pooled_passages = model.pooled(candidates.reshape(batch * width, passage_length))
        query_embeddings = truncate_and_normalize(pooled_queries.float())
        passage_embeddings = truncate_and_normalize(pooled_passages.float()).reshape(
            batch, width, -1
        )
        return torch.einsum("bd,bnd->bn", query_embeddings, passage_embeddings) / temperature
    raise ValueError(f"kind は 'cross_encoder' か 'dual_encoder': {kind!r}")


def schedule_tensors(
    schedule: RerankingSchedule, negative_kind: str, device: torch.device
) -> tuple[Tensor, Tensor]:
    """学習の組を ``(query の添字 (T, B)、候補の添字 (T, B, 1 + n))`` のテンソルにする(候補の
    0 番目が正例)。"""
    negatives = schedule.negatives(negative_kind)
    candidates = np.concatenate([schedule.positive[:, :, None], negatives], axis=2)
    return (
        torch.as_tensor(schedule.query, device=device),
        torch.as_tensor(candidates, device=device),
    )


def train_reranker(
    model: nn.Module,
    kind: str,
    query_tokens: Tensor,
    passage_tokens: Tensor,
    schedule: RerankingSchedule,
    negative_kind: str,
    peak_learning_rate: float,
    warmup_steps: int,
    min_learning_rate: float,
    weight_decay: float,
    gradient_clip_threshold: float,
    temperature: float,
    use_fp16_autocast: bool = False,
    init_loss_scale: float = 2.0**16,
    loss_scale_growth_interval: int = 2000,
    evaluation_steps: tuple[int, ...] = (),
    evaluation_fn: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """並べ替えモデルを、``schedule`` の全ステップぶん学習する(全パラメータを更新する)。

    Args:
        model: ``CrossEncoder`` または ``TextEmbeddingModel``(``query_tokens`` と同じデバイス)。
        kind: ``"cross_encoder"`` または ``"dual_encoder"``。
        query_tokens: 形状 ``(Q, L_q)``。コーパスの query のトークン(``schedule.query``
        の添字で引く)。
        passage_tokens: 形状 ``(P, L_p)``。コーパスの passage のトークン。
        schedule: ``sample_reranking_schedule()`` の返り値。
        negative_kind: ``"hard"`` または ``"random"``。
        peak_learning_rate, warmup_steps, min_learning_rate, weight_decay, gradient_clip_threshold:
            024 と同じ(warmup + cosine の学習率、行列形の重みにのみ重み減衰)。
        temperature: dual encoder の温度(cross-encoder では使わない)。
        use_fp16_autocast: 真なら本体を FP16 の autocast と動的損失スケーリングで学習する(CUDA
        のみ)。
        init_loss_scale, loss_scale_growth_interval: 動的損失スケーリングの設定。
        evaluation_steps: 学習の途中で ``evaluation_fn`` を呼ぶステップ(そのステップの更新の後)。
        evaluation_fn: モデルを受け取り、評価結果の辞書を返す関数。

    Returns:
        学習の記録の辞書。``"loss"``・``"gradient_norm"``・``"gradient_clip_triggered"``・``"learning_rate"``・
        ``"loss_scale"``・``"step_skipped"`` は長さ ``T`` のリスト、``"evaluations"`` は ステップ
        -> ``evaluation_fn`` の
        返り値、``"data_stream_hash"`` は全ステップの組の SHA-256、``"train_seconds"``
        は評価を除く学習ループの実時間。
    """
    device = query_tokens.device
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")
    query_ids, candidate_ids = schedule_tensors(schedule, negative_kind, device)
    num_steps = query_ids.size(0)
    digest = hashlib.sha256()
    for array in (schedule.query, schedule.positive, schedule.negatives(negative_kind)):
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())

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
        queries = query_tokens[query_ids[step - 1]]
        candidates = passage_tokens[candidate_ids[step - 1]]
        scores = compute_candidate_scores(
            model, kind, queries, candidates, temperature, use_fp16_autocast
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
    history["data_stream_hash"] = digest.hexdigest()
    history["train_seconds"] = train_seconds
    return history


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def compute_reranking_losses(
    model: nn.Module,
    kind: str,
    query_tokens: Tensor,
    passage_tokens: Tensor,
    schedule: RerankingSchedule,
    negative_kind: str,
    steps: range | list[int],
    temperature: float,
    use_fp16_autocast: bool = False,
) -> np.ndarray:
    """``schedule`` の指定したステップの組での ``model`` の損失(勾配なし。1 ステップごと)。

    ``train_reranker()`` と同じ組・同じ損失で、更新を行わずに損失だけを測る。
    学習の最後の区間の訓練損失を、**同じ組での学習前のモデルの損失** と比べるために使う。
    呼び出しの前後でモデルのモードを変えない。
    """
    device = query_tokens.device
    query_ids, candidate_ids = schedule_tensors(schedule, negative_kind, device)
    was_training = model.training
    model.eval()
    losses = []
    for step in steps:
        queries = query_tokens[query_ids[step]]
        candidates = passage_tokens[candidate_ids[step]]
        scores = compute_candidate_scores(
            model, kind, queries, candidates, temperature, use_fp16_autocast
        )
        losses.append(softmax_cross_entropy_over_candidates(scores))
    model.train(was_training)
    return torch.stack(losses).cpu().numpy().astype(np.float64)


def build_reranker(
    kind: str, backbone, separator_id: int, terminal_id: int, head_init_std: float = 0.02
) -> nn.Module:
    """``kind`` に応じて、008 の本体を包む並べ替えモデルを作る。

    ``"cross_encoder"`` は ``CrossEncoder``(終端の位置の出力へのスカラーのヘッド)、
    ``"dual_encoder"`` は 024 の C5 と同じ構成の ``TextEmbeddingModel``(平均プール・因果マスク)。
    """
    if kind == "cross_encoder":
        return CrossEncoder(backbone, separator_id, terminal_id, head_init_std)
    if kind == "dual_encoder":
        return TextEmbeddingModel(backbone, pooling="mean", attention="causal")
    raise ValueError(f"kind は 'cross_encoder' か 'dual_encoder': {kind!r}")
