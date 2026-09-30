"""画像分類の学習ループと評価(019)。

007 の ``AdamW``(``src/training/optimizer.py``)と ``compute_warmup_cosine_learning_rate()``
(``src/training/schedule.py``)、gradient clipping(クリッピング前の全パラメータの勾配の L2 ノルムが
閾値を超えたら、閾値に一致するように縮める。007 と同じ定義)を使う。混合精度は 011 の
``DynamicLossScaler``(``src/training/precision.py``)と ``torch.autocast`` の FP16 で行い、
CUDA のときのみ有効にする(呼び出し側が ``use_fp16_autocast`` で指定する)。

**乱数の分離**: データの順序(``EpochShuffledBatchSampler``)とデータ拡張(``random_crop_and_flip``)の
乱数は、``data_seed`` で初期化した CPU の ``torch.Generator`` 1 つから引く。モデルの初期化の乱数
(呼び出し側がモデルを構築する前に ``torch.manual_seed`` で決める)とは別の生成器なので、
初期化の乱数の消費量がモデルの構造によって違っても、同じ ``data_seed`` の学習どうしでは
データの順序とデータ拡張が一致する(019 の実験 A の対応付きの差の前提)。

**損失スケーリングの unscale**: ``DynamicLossScaler.unscale_gradients()`` は、
パラメータごとに非有限値の有無を CPU に読み出す(パラメータの数だけ GPU との同期が起きる)。
本モジュールでは、同じスケール値での除算を ``torch._foreach_mul_`` でまとめて行い、
非有限値の検出を、勾配のノルム(クリッピングにも使う)が有限かどうかの 1 回の読み出しに
置き換える。勾配のいずれかの要素が NaN・Inf ならノルムも非有限になる。有限の勾配のノルムが
FP32 で溢れる場合も非有限と判定する(更新を飛ばす安全側)。スケール値の更新(backoff・growth)は
``DynamicLossScaler.update()`` をそのまま使う。unscale の結果が ``unscale_gradients()`` と
bit 単位で一致することは 019 のノートブックで確かめる。

**評価**: 評価は常に FP32 で行い(``torch.autocast`` を使わない)、学習の最終ステップの重みで
必ず評価できるように、学習関数は最終ステップの後の重みを持つモデルをそのまま残す。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from src.data.image import EpochShuffledBatchSampler, random_crop_and_flip, to_normalized_float
from src.training.optimizer import AdamW
from src.training.precision import DynamicLossScaler
from src.training.schedule import compute_warmup_cosine_learning_rate


def split_weight_decay_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """重み減衰を掛けるパラメータと掛けないパラメータに分ける。

    行列形(2 階以上)の重み(線形層・畳み込み層の重み)に重み減衰を掛け、バイアス・正規化層の
    パラメータ(1 階)と、モデルが ``no_weight_decay_parameter_names()`` で指定する行列形の
    パラメータ(ViT の [CLS] トークン・学習可能な位置埋め込み)には掛けない。

    Returns:
        (重み減衰を掛けるパラメータ, 掛けないパラメータ)。
    """
    excluded = set()
    if hasattr(model, "no_weight_decay_parameter_names"):
        excluded = set(model.no_weight_decay_parameter_names())
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and name not in excluded:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return decay, no_decay


def unscale_gradients_and_compute_norm(
    parameters: list[nn.Parameter], scale: float | None
) -> Tensor:
    """勾配をその場でスケール値で除し、全パラメータの勾配を連結した L2 ノルムを返す。

    ``scale`` が ``None`` のときは除算をしない。返り値は GPU 上のスカラー(CPU に読み出さない)。
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if scale is not None:
        torch._foreach_mul_(grads, 1.0 / scale)
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))


def train_image_classifier(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    train_indices: Tensor,
    num_steps: int,
    batch_size: int,
    peak_learning_rate: float,
    warmup_steps: int,
    min_learning_rate: float,
    weight_decay: float,
    gradient_clip_threshold: float,
    data_seed: int,
    crop_padding: int = 4,
    use_fp16_autocast: bool = False,
    init_loss_scale: float = 2.0**16,
    loss_scale_growth_interval: int = 2000,
    channels_last: bool = False,
    evaluation_steps: tuple[int, ...] = (),
    evaluation_fn: Callable[[nn.Module], dict] | None = None,
) -> dict:
    """画像分類器を ``num_steps`` ステップ学習する(交差エントロピー損失)。

    Args:
        model: 学習するモデル(``images`` と同じデバイスに置いておく)。
        images: 形状 ``(num, C, H, W)`` の uint8 の画像(データ全体。GPU 上にあってよい)。
        labels: 形状 ``(num,)`` の int64 のラベル(``images`` と同じデバイス)。
        train_indices: 学習に使う事例の ``images`` での添字(``images`` と同じデバイス、int64)。
        num_steps: 学習ステップ数 T。
        batch_size: バッチサイズ。
        peak_learning_rate: warmup の後の学習率の最大値。
        warmup_steps: 線形 warmup のステップ数。
        min_learning_rate: cosine decay の下限。
        weight_decay: 重み減衰(行列形の重みのみ、``split_weight_decay_parameters()``)。
        gradient_clip_threshold: gradient clipping の閾値(クリッピング前の勾配のノルムに対して)。
        data_seed: データの順序とデータ拡張の CPU の乱数生成器のシード。
        crop_padding: random crop のパディングの画素数。
        use_fp16_autocast: True なら ``torch.autocast(dtype=float16)`` と動的損失スケーリングを使う
            (CUDA のときのみ True にする)。
        init_loss_scale: 動的損失スケーリングの初期スケール値。
        loss_scale_growth_interval: スケール値を growth するまでの連続成功ステップ数。
        channels_last: True なら入力を channels_last のメモリ配置にする(畳み込みの高速化)。
        evaluation_steps: 学習の途中で ``evaluation_fn`` を呼ぶステップ(そのステップの更新の後)。
        evaluation_fn: モデルを受け取り、評価結果の辞書を返す関数。

    Returns:
        学習の記録の辞書:

        - ``"loss"``・``"gradient_norm"``(クリッピング前)・``"gradient_clip_triggered"``・
          ``"learning_rate"``・``"loss_scale"``・``"step_skipped"``: 長さ ``num_steps`` のリスト。
        - ``"evaluations"``: ステップ -> ``evaluation_fn`` の返り値。
        - ``"epochs_started"``: 始めたエポックの数。
        - ``"data_stream_hash"``: 全ステップのバッチの添字と、学習後のデータの乱数生成器の状態の
          SHA-256(同じ ``data_seed``・同じ事例数の学習どうしで一致する)。
        - ``"train_seconds"``: 学習ループの実時間(評価を除く)。
    """
    device = images.device
    generator = torch.Generator()
    generator.manual_seed(data_seed)
    sampler = EpochShuffledBatchSampler(int(train_indices.numel()), batch_size, generator)

    decay, no_decay = split_weight_decay_parameters(model)
    optimizers = [
        AdamW(decay, lr=peak_learning_rate, weight_decay=weight_decay),
        AdamW(no_decay, lr=peak_learning_rate, weight_decay=0.0),
    ]
    parameters = decay + no_decay
    loss_scaler = (
        DynamicLossScaler(init_loss_scale, growth_interval=loss_scale_growth_interval)
        if use_fp16_autocast
        else None
    )
    if use_fp16_autocast and device.type != "cuda":
        raise ValueError("use_fp16_autocast は CUDA のときのみ有効にする")

    losses = torch.empty(num_steps, device=device)
    gradient_norms = torch.empty(num_steps, device=device)
    history: dict = {
        "learning_rate": [],
        "loss_scale": [],
        "step_skipped": [],
        "evaluations": {},
    }
    digest = hashlib.sha256()
    evaluation_steps = tuple(sorted(set(evaluation_steps)))
    threshold = torch.tensor(gradient_clip_threshold, device=device)

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
        batch_indices = train_indices[positions.to(device, non_blocking=True)]
        inputs = random_crop_and_flip(images[batch_indices], crop_padding, generator)
        targets = labels[batch_indices]
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast
        ):
            logits = model(inputs)
        loss = functional.cross_entropy(logits.float(), targets)

        for optimizer in optimizers:
            optimizer.zero_grad()
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
        if loss_scaler is not None:  # ここで 1 回だけ CPU と同期する
            found_inf = not bool(torch.isfinite(gradient_norm))
        if not found_inf:
            for optimizer in optimizers:
                optimizer.step()
        if loss_scaler is not None:
            history["loss_scale"].append(loss_scaler.scale)
            loss_scaler.update(found_inf)
        else:
            history["loss_scale"].append(1.0)

        losses[step - 1] = loss.detach()
        gradient_norms[step - 1] = gradient_norm.detach()
        history["learning_rate"].append(learning_rate)
        history["step_skipped"].append(found_inf)

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
    history["gradient_clip_triggered"] = [
        g > gradient_clip_threshold for g in history["gradient_norm"]
    ]
    history["epochs_started"] = sampler.epochs_started
    history["data_stream_hash"] = digest.hexdigest()
    history["train_seconds"] = train_seconds
    return history


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate_image_classifier(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    batch_size: int = 500,
    image_transform: Callable[[Tensor], Tensor] | None = None,
    channels_last: bool = False,
) -> dict:
    """top-1 正解率と交差エントロピーを FP32 で評価する(データ拡張なし)。

    Args:
        model: 評価するモデル。
        images: 形状 ``(num, C, H, W)`` の uint8 の画像(モデルと同じデバイス)。
        labels: 形状 ``(num,)`` の int64 のラベル。
        batch_size: 評価のバッチサイズ(結果には影響しない)。
        image_transform: 標準化した画像に掛ける変換(パッチの並べ替えなど、診断量に使う)。
        channels_last: True なら入力を channels_last のメモリ配置にする。

    Returns:
        ``{"accuracy", "cross_entropy", "num_examples", "correct"}``。``correct`` は事例ごとの
        正誤(numpy の bool 配列)。
    """
    was_training = model.training
    model.eval()
    correct_parts, loss_sum = [], 0.0
    for start in range(0, images.size(0), batch_size):
        inputs = to_normalized_float(images[start : start + batch_size])
        if image_transform is not None:
            inputs = image_transform(inputs)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        targets = labels[start : start + batch_size]
        logits = model(inputs).float()
        loss_sum += float(functional.cross_entropy(logits, targets, reduction="sum"))
        correct_parts.append((logits.argmax(dim=-1) == targets).cpu())
    model.train(was_training)
    correct = torch.cat(correct_parts).numpy().astype(bool)
    return {
        "accuracy": float(correct.mean()),
        "cross_entropy": loss_sum / len(correct),
        "num_examples": int(len(correct)),
        "correct": correct,
    }


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """正解率のロジット log(p / (1 - p))(天井効果の診断に使う)。"""
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)
