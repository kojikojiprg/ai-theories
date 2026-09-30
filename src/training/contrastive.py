"""画像とテキストの対照学習(Contrastive Learning)の損失・学習ループ・評価(020)。

**損失**(記号は 020 のノートブックの 3 節に合わせる。u_i・v_j は L2 正規化した画像・テキストの
埋め込み、s_ij = u_i^T v_j はコサイン類似度、N はバッチサイズ):

- ``softmax_contrastive_loss()``: CLIP の対称な InfoNCE 損失(Radford et al., ICML 2021)。
  logits = exp(logit_scale) * s とし、画像 -> テキスト(行方向)とテキスト -> 画像(列方向)の
  交差エントロピーの平均をとる。
- ``sigmoid_contrastive_loss()``: SigLIP の sigmoid 損失(Zhai et al., ICCV 2023 の Algorithm 1)。
  logits = exp(t') * s + b、ラベル z_ij = 1(i = j)/ -1(i != j)として
  -(1/N) Σ_i Σ_j log σ(z_ij * logits_ij)。
- ``negclip_loss()``: NegCLIP(Yuksekgonul et al., ICLR 2023)。画像 -> テキスト方向の分母に、
  各正例の困難な負例(hard negative)のキャプションを加えた対称な InfoNCE 損失。困難な負例の
  画像は加えない。

**学習ループ**(``train_contrastive_model()``): 019 の ``train_image_classifier()``
(``src/training/classification.py``)と同じ部品を使う。007 の ``AdamW``(``foreach=True``)、
warmup + cosine の学習率、gradient clipping(クリッピング前の全パラメータの勾配の L2 ノルムが
閾値を超えたら縮める)、011 の ``DynamicLossScaler`` と FP16 の autocast(CUDA のときのみ)、
unscale と非有限値の検出を 1 回の同期で行う ``unscale_gradients_and_compute_norm()``。
encoder の順伝播のみを autocast の中で行い、埋め込みの正規化・類似度・損失は FP32 で計算する。

**データの順序**: バッチの添字は ``DistinctCaptionBatchSampler``(``src/data/synthetic_scenes.py``)
で、``data_seed`` で初期化した CPU の生成器から引く。損失の種類に依存しないので、同じ
``data_seed``・同じバッチサイズの学習どうしは、損失によらず同じ順序で同じシーンを見る
(020 の実験 A・C の対応のある差の前提)。

**評価**(``evaluate_retrieval()``・``evaluate_two_alternative()``): 常に FP32 で行う。
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from src.data.synthetic_scenes import DistinctCaptionBatchSampler, normalize_scene_images
from src.training.classification import (
    split_weight_decay_parameters,
    unscale_gradients_and_compute_norm,
)
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate

LOSS_TYPES = ("softmax", "sigmoid", "negclip")


def softmax_contrastive_loss(
    image_embeddings: Tensor, text_embeddings: Tensor, logit_scale: Tensor
) -> Tensor:
    """対称な InfoNCE 損失(CLIP)。埋め込みは L2 正規化済みで、i 番目の画像とテキストが正例の組。"""
    logits = logit_scale.exp() * image_embeddings @ text_embeddings.t()
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        functional.cross_entropy(logits, targets) + functional.cross_entropy(logits.t(), targets)
    )


def sigmoid_contrastive_loss(
    image_embeddings: Tensor, text_embeddings: Tensor, logit_scale: Tensor, logit_bias: Tensor
) -> Tensor:
    """sigmoid 損失(SigLIP)。全 N^2 組をそれぞれ独立な二値分類として扱い、和を N で割る。"""
    logits = logit_scale.exp() * image_embeddings @ text_embeddings.t() + logit_bias
    labels = 2.0 * torch.eye(logits.size(0), device=logits.device) - 1.0
    return -functional.logsigmoid(labels * logits).sum() / logits.size(0)


def negclip_loss(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    hard_negative_embeddings: Tensor,
    logit_scale: Tensor,
    positive_caption_ids: Tensor,
    hard_negative_caption_ids: Tensor,
) -> Tensor:
    """NegCLIP の損失。画像 -> テキスト方向の分母に困難な負例のキャプションを加える。

    困難な負例の列のうち、その行の画像の正例と同じキャプションになったもの(ある正例の順序の入れ替えが、
    バッチ内の別の正例のキャプションに一致する場合など)は、偽の負例になるので分母から除く(-inf
    にする)。

    Args:
        image_embeddings: 形状 ``(N, d_e)``。
        text_embeddings: 形状 ``(N, d_e)``(正例のキャプション)。
        hard_negative_embeddings: 形状 ``(M, d_e)``(バッチの全画像に共通の困難な負例)。
        logit_scale: log(1/τ)。
        positive_caption_ids: 形状 ``(N,)``。バッチ内で互いに異なる。
        hard_negative_caption_ids: 形状 ``(M,)``。
    """
    n = image_embeddings.size(0)
    scale = logit_scale.exp()
    all_texts = torch.cat([text_embeddings, hard_negative_embeddings], dim=0)
    logits_image_to_text = scale * image_embeddings @ all_texts.t()  # (N, N + M)
    column_ids = torch.cat([positive_caption_ids, hard_negative_caption_ids])
    false_negative = column_ids[None, :] == positive_caption_ids[:, None]
    false_negative[:, :n] &= ~torch.eye(n, dtype=torch.bool, device=false_negative.device)
    logits_image_to_text = logits_image_to_text.masked_fill(false_negative, float("-inf"))
    logits_text_to_image = scale * text_embeddings @ image_embeddings.t()
    targets = torch.arange(n, device=image_embeddings.device)
    return 0.5 * (
        functional.cross_entropy(logits_image_to_text, targets)
        + functional.cross_entropy(logits_text_to_image, targets)
    )


def in_batch_margin(image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
    """バッチ内の画像 -> テキストのマージン(正例の類似度 - 最も類似度の高い負例の類似度)の平均。"""
    similarity = image_embeddings @ text_embeddings.t()
    n = similarity.size(0)
    positive = similarity.diagonal()
    hardest = similarity.masked_fill(
        torch.eye(n, dtype=torch.bool, device=similarity.device), float("-inf")
    )
    return (positive - hardest.max(dim=1).values).mean()


def train_contrastive_model(
    model: nn.Module,
    images: Tensor,
    caption_token_ids: Tensor,
    scene_caption_ids: np.ndarray,
    copies: int,
    loss_type: str,
    num_steps: int,
    batch_size: int,
    peak_learning_rate: float,
    warmup_steps: int,
    min_learning_rate: float,
    weight_decay: float,
    gradient_clip_threshold: float,
    data_seed: int,
    hard_negative_ids: dict[str, np.ndarray] | None = None,
    hard_negative_allowed: np.ndarray | None = None,
    logit_scale_max: float | None = math.log(100.0),
    use_fp16_autocast: bool = False,
    init_loss_scale: float = 2.0**16,
    loss_scale_growth_interval: int = 2000,
    evaluation_steps: tuple[int, ...] = (),
    evaluation_fn: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """二重 encoder を ``num_steps`` ステップ、対照学習の損失で学習する。

    Args:
        model: ``CLIPDualEncoder``(``images`` と同じデバイスに置いておく)。
        images: 形状 ``(num_scenes, 3, 32, 32)`` の uint8 の学習用の画像(GPU 上にあってよい)。
        caption_token_ids: 形状 ``(num_captions, S)`` の全キャプションの符号化済みの系列
            (``images`` と同じデバイス)。
        scene_caption_ids: 各シーンのキャプションの番号(``repeat_caption_ids()`` の順)。
        copies: キャプションあたりのシーンの数。
        loss_type: ``"softmax"`` / ``"sigmoid"`` / ``"negclip"``。
        num_steps: 学習ステップ数。
        batch_size: バッチサイズ N。
        peak_learning_rate: warmup の後の学習率の最大値。
        warmup_steps: 線形 warmup のステップ数。
        min_learning_rate: cosine decay の下限。
        weight_decay: 重み減衰(行列形の重みのみ)。
        gradient_clip_threshold: gradient clipping の閾値。
        data_seed: バッチの添字の CPU の乱数生成器のシード。
        hard_negative_ids: ``"negclip"`` のとき必須。規則名 -> 各キャプションの困難な負例の
            キャプションの番号。
        hard_negative_allowed: 形状 ``(num_captions,)`` の bool。困難な負例として使ってよい
            キャプション(020 では、除外した (色, 形) の組を含まないキャプション)。False の
            キャプションは、テキスト encoder に入れず、分母にも加えない(添字の選択は CPU で行うので
            GPU との同期は起きない)。``None`` ならすべて使う。
        logit_scale_max: 各ステップの更新の後に ``logit_scale`` をこの値以下に切り詰める
            (CLIP の実装。``None`` なら切り詰めない)。
        use_fp16_autocast: True なら encoder の順伝播を FP16 の autocast で行い、
            動的損失スケーリングを使う(CUDA のみ)。
        init_loss_scale: 動的損失スケーリングの初期スケール値。
        loss_scale_growth_interval: スケール値を growth するまでの連続成功ステップ数。
        evaluation_steps: 学習の途中で ``evaluation_fn`` を呼ぶステップ(そのステップの更新の後)。
        evaluation_fn: モデルを受け取り、評価結果の辞書を返す関数。

    Returns:
        学習の記録の辞書: ``"loss"``・``"gradient_norm"``・``"learning_rate"``・``"logit_scale"``・
        ``"logit_bias"``・``"in_batch_margin"``・``"loss_scale"``・``"step_skipped"``
        (長さ ``num_steps`` のリスト)、``"evaluations"``、``"examples_seen"``、
        ``"data_stream_hash"``(全ステップのバッチの添字と学習後の生成器の状態の SHA-256)、
        ``"distinct_captions_in_every_batch"``、``"train_seconds"``、
        ``"text_caption_ids"``(学習中にテキスト encoder に入れた全系列のキャプションの番号の集合、
        昇順の numpy 配列)、``"hard_negatives_used"``・``"hard_negatives_generated"``
        (規則名 -> 分母に加えた数・規則が作った数。NegCLIP のときのみ)。
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type は {LOSS_TYPES} のいずれか: {loss_type!r}")
    if loss_type == "negclip" and hard_negative_ids is None:
        raise ValueError("negclip には hard_negative_ids が必要")
    device = images.device
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")
    generator = torch.Generator()
    generator.manual_seed(data_seed)
    sampler = DistinctCaptionBatchSampler(scene_caption_ids, copies, batch_size, generator)
    scene_caption_ids = np.asarray(scene_caption_ids)
    num_captions = caption_token_ids.size(0)
    if hard_negative_allowed is None:
        hard_negative_allowed = np.ones(num_captions, dtype=bool)
    text_caption_seen = np.zeros(num_captions, dtype=bool)  # テキスト encoder に入れたキャプション
    hard_negatives_used = {name: 0 for name in (hard_negative_ids or {})}
    hard_negatives_generated = {name: 0 for name in (hard_negative_ids or {})}

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

    def record() -> Tensor:
        return torch.empty(num_steps, device=device)

    losses, gradient_norms, margins, scales, biases = (
        record(),
        record(),
        record(),
        record(),
        record(),
    )
    history: dict = {"learning_rate": [], "loss_scale": [], "step_skipped": [], "evaluations": {}}
    digest = hashlib.sha256()
    evaluation_steps = tuple(sorted(set(evaluation_steps)))
    threshold = torch.tensor(gradient_clip_threshold, device=device)
    distinct = True

    train_seconds = 0.0
    start = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        learning_rate = compute_warmup_cosine_learning_rate(
            step, warmup_steps, num_steps, peak_learning_rate, min_learning_rate
        )
        for optimizer in optimizers:
            optimizer.set_learning_rate(learning_rate)

        positions = sampler.next_batch()
        digest.update(positions.numpy().tobytes())
        batch_scenes = positions.to(device, non_blocking=True)
        captions_cpu = scene_caption_ids[positions.numpy()]
        text_ids_cpu = captions_cpu
        if loss_type == "negclip":
            # 困難な負例の添字は CPU で作り、使ってはいけないキャプションをここで除く
            parts = []
            for name, rule in hard_negative_ids.items():
                candidates = rule[captions_cpu]
                keep = candidates[hard_negative_allowed[candidates]]
                hard_negatives_generated[name] += len(candidates)
                hard_negatives_used[name] += len(keep)
                parts.append(keep)
            hard_ids_cpu = np.concatenate(parts)
            text_ids_cpu = np.concatenate([captions_cpu, hard_ids_cpu])
            hard_ids = torch.as_tensor(hard_ids_cpu, device=device)
        text_caption_seen[text_ids_cpu] = True
        batch_captions = torch.as_tensor(captions_cpu, device=device)
        inputs = normalize_scene_images(images[batch_scenes])
        tokens = caption_token_ids[torch.as_tensor(text_ids_cpu, device=device)]

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
        ):
            image_features = model.vision.forward_features(inputs) @ model.image_projection
            text_features = model.text(tokens) @ model.text_projection
        image_embeddings = functional.normalize(image_features.float(), dim=-1)
        text_embeddings = functional.normalize(text_features.float(), dim=-1)
        positive_text = text_embeddings[:batch_size]
        if loss_type == "softmax":
            loss = softmax_contrastive_loss(image_embeddings, positive_text, model.logit_scale)
        elif loss_type == "sigmoid":
            loss = sigmoid_contrastive_loss(
                image_embeddings, positive_text, model.logit_scale, model.logit_bias
            )
        else:
            loss = negclip_loss(
                image_embeddings,
                positive_text,
                text_embeddings[batch_size:],
                model.logit_scale,
                batch_captions,
                hard_ids,
            )

        for optimizer in optimizers:
            optimizer.zero_grad()
        (loss_scaler.scale_loss(loss) if loss_scaler is not None else loss).backward()

        gradient_norm = unscale_gradients_and_compute_norm(
            parameters, loss_scaler.scale if loss_scaler is not None else None
        )
        clip_coefficient = torch.where(
            gradient_norm > threshold, threshold / gradient_norm, torch.ones_like(gradient_norm)
        )
        torch._foreach_mul_([p.grad for p in parameters if p.grad is not None], clip_coefficient)

        found_inf = False
        if loss_scaler is not None:  # ここで 1 回だけ CPU と同期する
            found_inf = not bool(torch.isfinite(gradient_norm))
        if not found_inf:
            for optimizer in optimizers:
                optimizer.step()
            if logit_scale_max is not None:
                with torch.no_grad():
                    model.logit_scale.clamp_(max=logit_scale_max)
        if loss_scaler is not None:
            history["loss_scale"].append(loss_scaler.scale)
            loss_scaler.update(found_inf)
        else:
            history["loss_scale"].append(1.0)

        with torch.no_grad():
            losses[step - 1] = loss.detach()
            gradient_norms[step - 1] = gradient_norm.detach()
            margins[step - 1] = in_batch_margin(image_embeddings, positive_text)
            scales[step - 1] = model.logit_scale.detach()
            biases[step - 1] = model.logit_bias.detach()
        history["learning_rate"].append(learning_rate)
        history["step_skipped"].append(found_inf)
        # バッチ内のキャプションが互いに異なることを、全ステップで CPU の添字から確かめる
        # (GPU との同期は起きない)
        distinct &= int(torch.unique(positions // copies).numel()) == batch_size

        if evaluation_fn is not None and step in evaluation_steps:
            _synchronize(device)
            train_seconds += time.time() - start
            history["evaluations"][step] = evaluation_fn(model)
            start = time.time()
    _synchronize(device)
    train_seconds += time.time() - start

    digest.update(generator.get_state().numpy().tobytes())
    history["loss"] = losses.cpu().tolist()
    history["gradient_norm"] = gradient_norms.cpu().tolist()
    history["in_batch_margin"] = margins.cpu().tolist()
    history["logit_scale"] = scales.cpu().tolist()
    history["logit_bias"] = biases.cpu().tolist()
    history["examples_seen"] = sampler.examples_drawn
    history["data_stream_hash"] = digest.hexdigest()
    history["distinct_captions_in_every_batch"] = distinct
    history["train_seconds"] = train_seconds
    history["text_caption_ids"] = np.flatnonzero(text_caption_seen)
    if loss_type == "negclip":
        history["hard_negatives_used"] = hard_negatives_used
        history["hard_negatives_generated"] = hard_negatives_generated
    return history


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def encode_images(model: nn.Module, images_uint8: Tensor, batch_size: int = 1024) -> Tensor:
    """L2 正規化した画像の埋め込み(FP32、評価モード)。"""
    was_training = model.training
    model.eval()
    parts = [
        model.encode_image(normalize_scene_images(images_uint8[s : s + batch_size]))
        for s in range(0, images_uint8.size(0), batch_size)
    ]
    model.train(was_training)
    return torch.cat(parts)


@torch.no_grad()
def encode_texts(model: nn.Module, token_ids: Tensor, batch_size: int = 4096) -> Tensor:
    """L2 正規化したテキストの埋め込み(FP32、評価モード)。"""
    was_training = model.training
    model.eval()
    parts = [
        model.encode_text(token_ids[s : s + batch_size])
        for s in range(0, token_ids.size(0), batch_size)
    ]
    model.train(was_training)
    return torch.cat(parts)


@torch.no_grad()
def evaluate_retrieval(
    image_embeddings: Tensor, candidate_text_embeddings: Tensor, true_candidate_index: Tensor
) -> dict:
    """画像 -> テキストの検索(zero-shot 分類と同じ形)の top-1 の正解率とマージン。

    各画像について、候補のキャプション全体とのコサイン類似度を計算し、最大の候補が正解なら正答とする。
    マージンは「正解の類似度 - 正解以外の候補の類似度の最大値」で、正のとき正答になる。

    Args:
        image_embeddings: 形状 ``(num_images, d_e)``。
        candidate_text_embeddings: 形状 ``(num_candidates, d_e)``。
        true_candidate_index: 形状 ``(num_images,)``。各画像の正解の候補の添字。

    Returns:
        ``{"accuracy", "num_images", "num_candidates", "mean_margin", "correct"}``(``correct`` は
        numpy の bool 配列)。
    """
    similarity = image_embeddings @ candidate_text_embeddings.t()
    rows = torch.arange(similarity.size(0), device=similarity.device)
    positive = similarity[rows, true_candidate_index]
    others = similarity.clone()
    others[rows, true_candidate_index] = float("-inf")
    margin = positive - others.max(dim=1).values
    correct = (margin > 0).cpu().numpy()
    return {
        "accuracy": float(correct.mean()),
        "num_images": int(similarity.size(0)),
        "num_candidates": int(similarity.size(1)),
        "mean_margin": float(margin.mean()),
        "correct": correct,
    }


@torch.no_grad()
def evaluate_two_alternative(
    image_embeddings: Tensor,
    all_text_embeddings: Tensor,
    positive_ids: Tensor,
    negative_ids: Tensor,
) -> np.ndarray:
    """2 択の正誤: 画像ごとに、正例のキャプションの類似度が負例のキャプションより大きければ正答。

    Args:
        image_embeddings: 形状 ``(num_images, d_e)``。
        all_text_embeddings: 形状 ``(num_captions, d_e)``(全キャプション)。
        positive_ids: 形状 ``(num_images,)``。正例のキャプションの番号。
        negative_ids: 形状 ``(num_images,)``。負例のキャプションの番号(正例とは異なる)。

    Returns:
        numpy の bool 配列(形状 ``(num_images,)``)。
    """
    positive = (image_embeddings * all_text_embeddings[positive_ids]).sum(dim=-1)
    negative = (image_embeddings * all_text_embeddings[negative_ids]).sum(dim=-1)
    return (positive > negative).cpu().numpy()


def modality_gap(image_embeddings: Tensor, text_embeddings: Tensor) -> float:
    """画像の埋め込みの重心とテキストの埋め込みの重心のユークリッド距離(Liang et al., NeurIPS 2022)
    。"""
    return float((image_embeddings.mean(dim=0) - text_embeddings.mean(dim=0)).norm())
