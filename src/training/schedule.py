"""warmup + cosine 学習率スケジュールのスクラッチ実装。

大規模言語モデルの学習で広く使われる学習率スケジュール(例: Brown et al.,
"Language Models are Few-Shot Learners", NeurIPS 2020 の Appendix D)。
学習の序盤に学習率を線形に立ち上げる warmup と、その後コサイン曲線に沿って
減衰させる cosine decay を組み合わせる。

記号 / Notation:
    step               : 現在のステップ番号(1-indexed。``train_language_model`` の
                         ステップ番号と揃える)
    warmup_steps       : 線形 warmup に使うステップ数
    total_steps        : 学習全体のステップ数(cosine decay は
                         ``total_steps`` で最小学習率に達する)
    peak_learning_rate : warmup 終了時点で到達する学習率の最大値
    min_learning_rate  : cosine decay の下限として指定する学習率の最小値
"""

from __future__ import annotations

import math


def compute_warmup_cosine_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_learning_rate: float,
    min_learning_rate: float,
) -> float:
    """warmup + cosine スケジュールに従って、指定ステップの学習率を計算する。

    ``0 < step <= warmup_steps`` の区間では、学習率を線形に ``0`` から
    ``peak_learning_rate`` まで立ち上げる(warmup)。

    .. math::

        \\eta(s) = \\text{peak\\_lr} \\times s / \\text{warmup\\_steps}

    (以下、簡潔さのため ``step`` を :math:`s`、``peak_learning_rate`` を
    :math:`\\text{peak\\_lr}`、``min_learning_rate`` を :math:`\\text{min\\_lr}`
    と略記する。)

    ``warmup_steps < step`` の区間では、``min_learning_rate`` を下限としたコサイン
    曲線に沿って減衰させる(cosine decay)。``progress`` は warmup 終了後の
    経過割合(``total_steps`` で 1.0 に達する。それ以降は ``1.0`` に固定する)。

    .. math::

        p &= \\min(1, (s - w) / (T - w)) \\\\
        r &= \\text{peak\\_lr} - \\text{min\\_lr} \\\\
        \\eta(s) &= \\text{min\\_lr} + \\tfrac{1}{2} r (1 + \\cos(\\pi p))

    (:math:`w` = ``warmup_steps`` 、 :math:`T` = ``total_steps``)

    ``warmup_steps = 0`` の場合、warmup 区間は存在せず、``step = 1`` から直接
    cosine decay に入る(``progress`` の分母の ``total_steps - warmup_steps`` が
    ``total_steps`` に一致する)。

    Args:
        step: 現在のステップ番号(1-indexed)。
        warmup_steps: 線形 warmup に使うステップ数(0 以上)。
        total_steps: 学習全体のステップ数(``warmup_steps`` より大きい必要がある)。
        peak_learning_rate: warmup 終了時点で到達する学習率の最大値。
        min_learning_rate: cosine decay の下限として指定する学習率の最小値。

    Returns:
        指定ステップでの学習率。
    """
    if total_steps <= warmup_steps:
        raise ValueError(
            f"total_steps ({total_steps}) は warmup_steps ({warmup_steps}) より大きい必要がある"
        )

    if warmup_steps > 0 and step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_learning_rate + 0.5 * (peak_learning_rate - min_learning_rate) * (
        1.0 + math.cos(math.pi * progress)
    )
