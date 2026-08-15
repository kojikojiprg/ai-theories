"""言語モデルの事前学習ループ・評価関数のスクラッチ実装。

**006 の時点では意図的に素朴な設定に留める。** 007(学習の安定化)で拡張する前提のため、
以下は実装しない: AdamW(Loshchilov & Hutter, ICLR 2019)、学習率スケジュール、
gradient clipping、mixed precision。Optimizer は Adam(Kingma & Ba, ICLR 2015)を
固定学習率で使い、演算は fp32(単精度浮動小数点)のみで行う。

勾配ノルムは実験 H(007 で扱う gradient clipping の要否の検証)で使うため、
clipping を行わずに測定のみ記録する。

記号 / Notation:
    B : 訓練バッチサイズ
    S : 系列長(sequence length)
    V : 語彙サイズ
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.data.text import get_random_batch
from src.utils.statistics import compute_bits_per_byte


def evaluate_bits_per_byte(
    model: nn.Module,
    evaluation_windows: Tensor,
    evaluation_mask: Tensor,
    total_bytes: int,
    device: torch.device | str,
    batch_size: int = 16,
) -> float:
    """検証テキスト全体を非重複窓で逐次評価し、bits-per-byte を計算する。

    ``evaluation_windows``・``evaluation_mask``(``make_evaluation_windows`` の出力、
    非重複の固定長窓とパディング位置を示すマスク)を先頭から順に固定サイズの
    ミニバッチに区切って処理する(**ランダムバッチによる評価はしない**、004 の原則)。
    各窓 ``[w_0, ..., w_{S-1}]`` について、位置 ``0 .. S-2`` の logits で位置
    ``1 .. S-1`` のトークンを予測する(先頭トークン ``w_0`` は窓内に左文脈を持たない
    ため予測対象に含めない)。パディング位置(``evaluation_mask`` が False)は損失の
    総和から除外する。

    ``total_bytes`` は、トークナイザ条件に関係なく **検証テキスト全体の UTF-8
    バイト数**(符号化前のテキストから直接計算した値)を渡す想定である(006、
    3.5 節。全トークナイザ条件で同一の定数になることが望ましい特性であり、
    ``make_evaluation_windows`` はこの値を計算しない)。

    Args:
        model: 評価対象の言語モデル(``forward(token_ids) -> logits`` を持つ)。
        evaluation_windows: 形状 ``(num_windows, S)`` の LongTensor。
        evaluation_mask: 形状 ``(num_windows, S)`` の bool Tensor(True が実トークン、
            False がパディング、``make_evaluation_windows`` の出力)。
        total_bytes: 検証テキスト全体の UTF-8 バイト数。
        device: 評価に使うデバイス。
        batch_size: 1 回の forward で処理する窓の数(メモリ制約に応じて調整する。
            窓の処理順序・集計結果には影響しない)。

    Returns:
        bits-per-byte(値が小さいほど圧縮効率が高い = モデルの予測性能が高い)。
    """
    model.eval()
    total_negative_log_likelihood_nats = 0.0
    with torch.no_grad():
        for start in range(0, evaluation_windows.size(0), batch_size):
            batch = evaluation_windows[start : start + batch_size].to(device)
            batch_mask = evaluation_mask[start : start + batch_size, 1:].to(device)  # (b, S-1)
            logits = model(batch)  # (b, S, V)
            predicted_logits = logits[:, :-1, :]
            targets = batch[:, 1:]
            per_token_nll = functional.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(batch_mask.shape)
            total_negative_log_likelihood_nats += (per_token_nll * batch_mask).sum().item()
    return compute_bits_per_byte(total_negative_log_likelihood_nats, total_bytes)


def train_language_model(
    model: nn.Module,
    train_token_ids: Tensor,
    evaluation_windows: Tensor,
    evaluation_mask: Tensor,
    total_eval_bytes: int,
    num_steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    eval_interval: int,
    device: torch.device | str,
    seed: int,
) -> dict[str, list[float]]:
    """Adam・固定学習率・fp32 の素朴な学習ループ。

    訓練データは ``get_random_batch``(``src/data/text.py``)でランダムな連続区間を
    切り出してミニバッチを作る(訓練はランダムサンプリングでよい。評価との違いは
    ``evaluate_bits_per_byte`` の docstring を参照)。

    Args:
        model: 学習対象の言語モデル(``forward(token_ids) -> logits`` を持つ)。
        train_token_ids: 訓練データの 1 次元 LongTensor(``encode_corpus`` の出力)。
        evaluation_windows: 検証用の非重複窓(``make_evaluation_windows`` の出力)。
        evaluation_mask: ``evaluation_windows`` に対応するパディングマスク
            (``make_evaluation_windows`` の出力、``evaluate_bits_per_byte`` にそのまま渡す)。
        total_eval_bytes: 検証テキスト全体の UTF-8 バイト数
            (``evaluate_bits_per_byte`` にそのまま渡す)。
        num_steps: 学習ステップ数。
        batch_size: 訓練バッチサイズ B。
        sequence_length: 訓練系列長 S。
        learning_rate: Adam の学習率(固定、スケジュールなし)。
        eval_interval: このステップ数ごとに検証 bits-per-byte を測定する。
        device: 学習に使うデバイス。
        seed: 乱数シード。関数の先頭で明示的に ``torch.manual_seed`` を呼び、
            バッチサンプリング用の ``torch.Generator`` にも同じ値を使う。

    Returns:
        以下のキーを持つ履歴の辞書:

        - ``"step"``: 学習ステップ番号のリスト(1-indexed)。
        - ``"train_loss"``: ステップごとの訓練損失(cross entropy、nats、バッチ平均)。
        - ``"gradient_norm"``: ステップごとの勾配ノルム(全パラメータの勾配を
          連結した L2 ノルム。clipping は行わず測定のみ、実験 H で使用)。
        - ``"eval_step"``: 検証を行ったステップ番号のリスト。
        - ``"eval_bits_per_byte"``: ``eval_step`` に対応する検証 bits-per-byte。
    """
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {
        "step": [],
        "train_loss": [],
        "gradient_norm": [],
        "eval_step": [],
        "eval_bits_per_byte": [],
    }

    for step in range(1, num_steps + 1):
        model.train()
        inputs, targets = get_random_batch(train_token_ids, batch_size, sequence_length, generator)
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        # clipping はせず、全パラメータの勾配を連結した L2 ノルムを測定のみ行う(実験 H)。
        gradient_norm_sq = sum(
            p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None
        )
        gradient_norm = float(gradient_norm_sq**0.5)

        optimizer.step()

        history["step"].append(step)
        history["train_loss"].append(loss.item())
        history["gradient_norm"].append(gradient_norm)

        if step % eval_interval == 0:
            bits_per_byte = evaluate_bits_per_byte(
                model, evaluation_windows, evaluation_mask, total_eval_bytes, device
            )
            history["eval_step"].append(step)
            history["eval_bits_per_byte"].append(bits_per_byte)

    return history
