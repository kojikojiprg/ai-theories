"""損失マスク(Loss Masking)に対応した SFT(Supervised Fine-Tuning)の学習関数(016)。

``src/data/instruction.py`` で符号化・バッチ化した指示データで、自己回帰言語モデルを
微調整する。``train_language_model()``(``src/training/trainer.py``)は連続したトークン列
からランダムな区間を切り出して全トークンに損失をかける事前学習用のループであり、事例ごとの
境界も損失マスクも扱わない。そのため既存の関数は変更せず、本モジュールを新規に置く。

**損失の正規化の単位**: バッチ内の応答部分のトークン(終端記号を含む)の集合を R、指示部分の
予測対象のトークンの集合を P、トークン t の負の対数尤度(negative log-likelihood)を
ell_t とする。

.. math::

    \\mathcal{L}_{\\mathrm{mask}} = \\frac{1}{|R|} \\sum_{t \\in R} \\ell_t, \\qquad
    \\mathcal{L}_{\\mathrm{unmask}} = \\frac{1}{|R|}
        \\left( \\sum_{t \\in R} \\ell_t + \\sum_{t \\in P} \\ell_t \\right)

どちらも応答部分のトークン数 |R| で割る。したがって応答部分の各トークンの勾配への寄与は
両者で同一であり、違いは指示部分の損失を足すかどうかだけになる(全トークン数 |R| + |P| で
割る一般的な実装では、指示が長いほど応答部分の勾配が |R| / (|R| + |P|) 倍に薄まる)。

記号 / Notation:
    B : バッチサイズ
    S : バッチ内の最大系列長
    V : 語彙サイズ
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.data.instruction import EncodedInstructionExample, collate_instruction_batch
from src.training.trainer import OptimizerLike, _set_optimizer_learning_rate


def compute_token_negative_log_likelihoods(logits: Tensor, token_ids: Tensor) -> Tensor:
    """位置 ``j`` の logits で位置 ``j + 1`` のトークンを予測したときの負の対数尤度(nats)。

    Args:
        logits: 形状 ``(B, S, V)``。
        token_ids: 形状 ``(B, S)``。

    Returns:
        形状 ``(B, S - 1)``。
    """
    predicted = logits[:, :-1, :]
    targets = token_ids[:, 1:]
    return functional.cross_entropy(
        predicted.reshape(-1, predicted.size(-1)).float(), targets.reshape(-1), reduction="none"
    ).view(targets.shape)


def compute_instruction_tuning_loss(
    logits: Tensor,
    token_ids: Tensor,
    prompt_target_mask: Tensor,
    response_target_mask: Tensor,
    include_prompt_loss: bool,
) -> dict[str, Tensor]:
    """損失マスクあり・なしの SFT の損失を、応答部分のトークン数 |R| で正規化して返す。

    Args:
        logits: 形状 ``(B, S, V)``。
        token_ids: 形状 ``(B, S)``(右パディング済み)。
        prompt_target_mask: 形状 ``(B, S - 1)`` の bool。指示部分の予測対象 P。
        response_target_mask: 形状 ``(B, S - 1)`` の bool。応答部分の予測対象 R。
        include_prompt_loss: True なら指示部分の損失も足す(損失マスクなし)。

    Returns:
        ``loss``(逆伝播に使うスカラー)、``response_sum``・``prompt_sum``(負の対数尤度の
        合計、nats、勾配を持たない)、``response_count``・``prompt_count``(トークン数)。
    """
    token_losses = compute_token_negative_log_likelihoods(logits, token_ids)
    response_mask = response_target_mask.to(token_losses.dtype)
    prompt_mask = prompt_target_mask.to(token_losses.dtype)
    response_sum = (token_losses * response_mask).sum()
    prompt_sum = (token_losses * prompt_mask).sum()
    response_count = response_mask.sum()
    numerator = response_sum + prompt_sum if include_prompt_loss else response_sum
    return {
        "loss": numerator / response_count,
        "response_sum": response_sum.detach(),
        "prompt_sum": prompt_sum.detach(),
        "response_count": response_count.detach(),
        "prompt_count": prompt_mask.sum().detach(),
    }


def make_epoch_batches(
    num_examples: int, num_steps: int, batch_size: int, seed: int
) -> list[list[int]]:
    """事例の添字のミニバッチを ``num_steps`` 個作る。

    エポックごとに事例の順序を並べ替え(乱数生成器は ``seed`` で初期化)、その並びを
    エポックをまたいで連結して先頭から ``batch_size`` 個ずつ切り出す。``num_examples`` が
    ``batch_size`` の倍数なら、ミニバッチはエポックの境界をまたがず、各エポックで全事例を
    ちょうど 1 回ずつ使う。

    Returns:
        長さ ``num_steps`` のリスト。各要素は長さ ``batch_size`` の添字のリスト。
    """
    rng = np.random.default_rng(seed)
    order: list[int] = []
    needed = num_steps * batch_size
    while len(order) < needed:
        order.extend(int(i) for i in rng.permutation(num_examples))
    return [order[i * batch_size : (i + 1) * batch_size] for i in range(num_steps)]


def train_instruction_tuning(
    model: nn.Module,
    examples: Sequence[EncodedInstructionExample],
    batches: Sequence[Sequence[int]],
    optimizer: OptimizerLike,
    include_prompt_loss: bool,
    device: torch.device | str,
    learning_rate_schedule: Callable[[int], float] | None = None,
    gradient_clip_threshold: float | None = None,
    pad_token_id: int = 0,
) -> dict[str, list]:
    """SFT の学習ループ(fp32)。

    ステップ ``s``(1-indexed)では ``batches[s - 1]`` の事例を右パディングしてバッチにし、
    ``compute_instruction_tuning_loss()`` の損失で 1 回更新する。gradient clipping・学習率
    スケジュールの扱いは ``train_language_model()`` と同じ(クリッピングはグローバルノルムで、
    適用前の勾配ノルムを常に記録する)。勾配ノルムは ``requires_grad`` のパラメータのみから
    計算する。

    Args:
        model: 学習対象のモデル(``forward(token_ids) -> logits``)。
        examples: 符号化した事例。
        batches: ステップごとの事例の添字(``make_epoch_batches()`` の出力)。
        optimizer: ``step()``・``zero_grad()`` を持つ optimizer(学習可能なパラメータのみで構築)。
        include_prompt_loss: True なら損失マスクなし(指示部分の損失も足す)。
        device: 学習に使うデバイス。
        learning_rate_schedule: ステップ番号(1-indexed)から学習率を返す callable。
        gradient_clip_threshold: gradient clipping の閾値(``None`` なら無効)。
        pad_token_id: パディングに使うトークン ID(損失に寄与しない)。

    Returns:
        ステップごとの記録: ``step``、``loss``(逆伝播に使った損失)、``response_loss``・
        ``prompt_loss``(トークンあたりの負の対数尤度、nats)、``response_tokens``・
        ``prompt_tokens``、``gradient_norm``(クリッピング適用前)、``gradient_clip_triggered``、
        ``learning_rate``。
    """
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    history: dict[str, list] = {
        "step": [],
        "loss": [],
        "response_loss": [],
        "prompt_loss": [],
        "response_tokens": [],
        "prompt_tokens": [],
        "gradient_norm": [],
        "gradient_clip_triggered": [],
        "learning_rate": [],
    }
    for step, indices in enumerate(batches, start=1):
        model.train()
        current_learning_rate = None
        if learning_rate_schedule is not None:
            current_learning_rate = learning_rate_schedule(step)
            _set_optimizer_learning_rate(optimizer, current_learning_rate)

        token_ids, prompt_mask, response_mask = collate_instruction_batch(
            [examples[i] for i in indices], pad_token_id
        )
        token_ids = token_ids.to(device)
        logits = model(token_ids)
        parts = compute_instruction_tuning_loss(
            logits, token_ids, prompt_mask.to(device), response_mask.to(device), include_prompt_loss
        )
        optimizer.zero_grad()
        parts["loss"].backward()

        gradient_norm = float(
            sum(p.grad.detach().pow(2).sum() for p in trainable if p.grad is not None) ** 0.5
        )
        clip_triggered = (
            gradient_clip_threshold is not None and gradient_norm > gradient_clip_threshold
        )
        if clip_triggered:
            scale = gradient_clip_threshold / gradient_norm
            for p in trainable:
                if p.grad is not None:
                    p.grad.detach().mul_(scale)
        optimizer.step()

        response_count = float(parts["response_count"])
        prompt_count = float(parts["prompt_count"])
        history["step"].append(step)
        history["loss"].append(float(parts["loss"].detach()))
        history["response_loss"].append(float(parts["response_sum"]) / response_count)
        history["prompt_loss"].append(float(parts["prompt_sum"]) / max(prompt_count, 1.0))
        history["response_tokens"].append(int(response_count))
        history["prompt_tokens"].append(int(prompt_count))
        history["gradient_norm"].append(gradient_norm)
        history["gradient_clip_triggered"].append(bool(clip_triggered))
        history["learning_rate"].append(current_learning_rate)
    return history


@torch.no_grad()
def evaluate_instruction_negative_log_likelihood(
    model: nn.Module,
    examples: Sequence[EncodedInstructionExample],
    device: torch.device | str,
    batch_size: int = 64,
    pad_token_id: int = 0,
) -> dict[str, np.ndarray]:
    """教師強制(teacher forcing)で、事例ごとの負の対数尤度の合計とトークン数を返す。

    事例の順に ``batch_size`` 個ずつ右パディングしてバッチにする(パディングは損失に寄与せず、
    因果マスクにより実トークンの logits も変えない)。

    Returns:
        ``response_sum``・``prompt_sum``(事例ごとの負の対数尤度の合計、nats、float64)、
        ``response_count``・``prompt_count``(事例ごとのトークン数、int64)。いずれも長さは
        ``len(examples)``。
    """
    model.eval()
    response_sum, prompt_sum, response_count, prompt_count = [], [], [], []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        token_ids, prompt_mask, response_mask = collate_instruction_batch(chunk, pad_token_id)
        token_ids = token_ids.to(device)
        token_losses = compute_token_negative_log_likelihoods(model(token_ids), token_ids)
        token_losses = (
            token_losses.cpu().double()
        )  # MPS は float64 を扱えないので CPU に移してから変換する
        response_sum.append((token_losses * response_mask).sum(dim=1).numpy())
        prompt_sum.append((token_losses * prompt_mask).sum(dim=1).numpy())
        response_count.append(response_mask.sum(dim=1).numpy().astype(np.int64))
        prompt_count.append(prompt_mask.sum(dim=1).numpy().astype(np.int64))
    return {
        "response_sum": np.concatenate(response_sum),
        "prompt_sum": np.concatenate(prompt_sum),
        "response_count": np.concatenate(response_count),
        "prompt_count": np.concatenate(prompt_count),
    }
