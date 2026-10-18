"""時計と、1 反復の時間の費用モデル。

推論サーバの実行は 1 つのデバイスの上で順に進むので、要求の到着から終了までの時間は、
「実際に計算に使った時間の積算」と「何も実行する要求がなく、次の到着まで待った時間」の和で決まる。
本モジュールの時計は後者を実際には待たずに時刻を進める(仮想の時計)。計算の時間は、

- ``WallClock``: デバイスの同期を取ってから実時間を読む(実行)。
- ``ModelClock``: 1 回の forward の時間の費用モデル(``IterationCostModel``)で積算する(見積もり)。
  1 反復は 1 回の forward なので、反復の時間は費用モデルの 1 本の式で決まる。

のどちらかで測る。

導入: 027。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from src.utils.device import synchronize_device


class Clock:
    """時計の共通の約束。"""

    def start(self) -> None:
        raise NotImplementedError

    def now(self) -> float:
        """現在の時刻(秒。実行の開始が 0)。"""
        raise NotImplementedError

    def advance_to(self, time_seconds: float) -> None:
        """何も実行しない間の待ちを、実際には待たずに ``time_seconds`` まで進める。"""
        raise NotImplementedError

    def tick(self, rows: int, query_tokens: int) -> None:
        """1 回の forward が終わったことを知らせる(費用モデルの時計だけが時刻を進める)。

        Args:
            rows: forward の行数(prefill の行と decode の行の合計)。
            query_tokens: forward が処理した問い合わせのトークン数(行数 × 最長の問い合わせ)。
        """


class WallClock(Clock):
    """実時間の時計。時刻を読む前にデバイスの同期を取る(非同期の処理の完了を待つ)。"""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._origin = 0.0
        self._idle = 0.0

    def start(self) -> None:
        synchronize_device(self._device)
        self._origin = time.perf_counter()
        self._idle = 0.0

    def now(self) -> float:
        synchronize_device(self._device)
        return time.perf_counter() - self._origin + self._idle

    def advance_to(self, time_seconds: float) -> None:
        current = self.now()
        if time_seconds > current:
            self._idle += time_seconds - current


@dataclass(frozen=True)
class IterationCostModel:
    """1 回の forward の時間 [秒] を、1 本の一次式で表す費用モデル。

    ``時間 = fixed + per_row * 行数 + per_extra_token * (問い合わせのトークン数 - 行数)``。

    行数は prefill の行と decode の行の合計で、問い合わせのトークン数は **行数 × 最長の問い合わせの
    長さ**(短い行を埋めた分を含む。実行器が処理するトークン数そのもの)である。decode の行だけの
    forward では、問い合わせのトークン数が行数に等しいので、第 3 項は 0 になる。prefill の行が
    混ざると、その長さまで全行が埋められるので、第 3 項は ``行数 * (最長の長さ - 1)`` になる。
    小型モデルでは、固定の項(Python の処理とカーネルの起動)が支配的になりうる。

    Attributes:
        fixed: 固定の項(行数・トークン数によらない)。
        per_row: 1 行あたりの項。
        per_extra_token: 1 行に最初の 1 トークンより多い分として加わる、1 トークンあたりの項。
    """

    fixed: float
    per_row: float
    per_extra_token: float

    def forward_seconds(self, rows: int, query_tokens: int) -> float:
        return self.fixed + self.per_row * rows + self.per_extra_token * (query_tokens - rows)

    def decode_seconds(self, rows: int) -> float:
        """decode の行だけの forward(問い合わせのトークン数 = 行数)の時間。"""
        return self.forward_seconds(rows, rows)


def _fit_non_negative(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """係数が負にならない最小二乗解(固定の項は必ず含め、他の項の部分集合を総当たりする)。"""
    best, best_error = None, float("inf")
    for use_row in (True, False):
        for use_extra in (True, False):
            columns = [0] + ([1] if use_row else []) + ([2] if use_extra else [])
            coefficients, *_ = np.linalg.lstsq(design[:, columns], y, rcond=None)
            if np.any(coefficients < 0):
                continue
            error = float(((design[:, columns] @ coefficients - y) ** 2).sum())
            if error < best_error:
                full = np.zeros(3)
                full[columns] = coefficients
                best, best_error = full, error
    if best is None:  # すべての部分集合で負になる(固定の項だけでも)ことは通常ない。平均で代用する
        best = np.array([float(y.mean()), 0.0, 0.0])
    return best


def relative_errors(
    model: IterationCostModel, samples: Sequence[tuple[int, int, float]]
) -> list[float]:
    """測った ``(行数, 問い合わせのトークン数, 秒)`` に対する、費用モデルの予測の相対誤差
    ``(予測 - 実測) / 実測``。"""
    return [(model.forward_seconds(r, q) - t) / t for r, q, t in samples]


def fit_cost_model(
    samples: Sequence[tuple[int, int, float]],
) -> tuple[IterationCostModel, dict[str, float]]:
    """測った ``(行数, 問い合わせのトークン数, 秒)`` から、費用モデルを最小二乗法で作る。

    係数は負にならないようにする(時間が行数やトークン数とともに減る予測を避けるため)。

    Returns:
        ``(費用モデル, 診断量)``。診断量は、決定係数 ``r2``(1 に近いほど一次式が合う)、
        固定の項が行数 1 の decode の時間に占める割合 ``fixed_share``(1 に近いほど固定費が支配的)、
        予測の相対誤差の絶対値の最大 ``max_relative_error``。
    """
    rows = np.array([s[0] for s in samples], dtype=np.float64)
    tokens = np.array([s[1] for s in samples], dtype=np.float64)
    y = np.array([s[2] for s in samples], dtype=np.float64)
    design = np.stack([np.ones_like(rows), rows, tokens - rows], axis=1)
    fixed, per_row, per_extra = (float(c) for c in _fit_non_negative(design, y))
    model = IterationCostModel(fixed, per_row, per_extra)
    prediction = design @ np.array([fixed, per_row, per_extra])
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - prediction) ** 2).sum()) / total if total > 0 else 1.0
    diagnostics = {
        "r2": r2,
        "fixed_share": fixed / max(model.decode_seconds(1), 1e-12),
        "max_relative_error": float(max(abs(e) for e in relative_errors(model, samples))),
    }
    return model, diagnostics


class ModelClock(Clock):
    """費用モデルで時刻を積算する時計(見積もり用。実行しない)。"""

    def __init__(self, cost_model: IterationCostModel) -> None:
        self._cost = cost_model
        self._time = 0.0

    def start(self) -> None:
        self._time = 0.0

    def now(self) -> float:
        return self._time

    def advance_to(self, time_seconds: float) -> None:
        self._time = max(self._time, time_seconds)

    def tick(self, rows: int, query_tokens: int) -> None:
        self._time += self._cost.forward_seconds(rows, query_tokens)
