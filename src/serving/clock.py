"""時計と、1 反復の時間の費用モデル。

推論サーバの実行は 1 つのデバイスの上で順に進むので、要求の到着から終了までの時間は、
「実際に計算に使った時間の積算」と「何も実行する要求がなく、次の到着まで待った時間」の和で決まる。
本モジュールの時計は後者を実際には待たずに時刻を進める(仮想の時計)。計算の時間は、

- ``WallClock``: デバイスの同期を取ってから実時間を読む(実行)。
- ``ModelClock``: 1 反復の時間の費用モデル(``IterationCostModel``)で積算する(見積もり)。

のどちらかで測る。

導入: 027。
"""

from __future__ import annotations

import time
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

    def tick(self, kind: str, rows: int, query_tokens: int) -> None:
        """1 回の forward が終わったことを知らせる(費用モデルの時計だけが時刻を進める)。"""


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
    """1 反復の時間 [秒] の一次式。

    ``decode = decode_fixed + decode_per_row * 行数``、
    ``prefill = prefill_fixed + prefill_per_token * 問い合わせのトークン数(行数 × 最長の長さ)``。
    小型モデルでは、固定の項(Python の処理とカーネルの起動)が支配的になりうる。

    Attributes:
        decode_fixed, decode_per_row: decode の固定の項と、1 行あたりの項。
        prefill_fixed, prefill_per_token: prefill の固定の項と、1 トークンあたりの項。
    """

    decode_fixed: float
    decode_per_row: float
    prefill_fixed: float
    prefill_per_token: float

    def decode_seconds(self, rows: int) -> float:
        return self.decode_fixed + self.decode_per_row * rows

    def prefill_seconds(self, query_tokens: int) -> float:
        return self.prefill_fixed + self.prefill_per_token * query_tokens


def fit_cost_model(
    decode_samples: list[tuple[int, float]], prefill_samples: list[tuple[int, float]]
) -> tuple[IterationCostModel, dict[str, float]]:
    """測った ``(行数, 秒)`` と ``(トークン数, 秒)`` から、費用モデルを最小二乗法で作る。

    Returns:
        ``(費用モデル, 診断量)``。診断量は、決定係数 ``decode_r2``・``prefill_r2``(1 に近いほど
        一次式が合う)と、固定の項が行数 1 のときの時間に占める割合 ``decode_fixed_share``
        (1 に近いほど固定費が支配的)。
    """

    def fit(samples):
        x = np.array([s[0] for s in samples], dtype=np.float64)
        y = np.array([s[1] for s in samples], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        prediction = slope * x + intercept
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - prediction) ** 2).sum()) / total if total > 0 else 1.0
        return float(intercept), float(slope), r2

    decode_fixed, decode_per_row, decode_r2 = fit(decode_samples)
    prefill_fixed, prefill_per_token, prefill_r2 = fit(prefill_samples)
    model = IterationCostModel(decode_fixed, decode_per_row, prefill_fixed, prefill_per_token)
    diagnostics = {
        "decode_r2": decode_r2,
        "prefill_r2": prefill_r2,
        "decode_fixed_share": decode_fixed / max(model.decode_seconds(1), 1e-12),
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

    def tick(self, kind: str, rows: int, query_tokens: int) -> None:
        if kind == "prefill":
            self._time += self._cost.prefill_seconds(query_tokens)
        elif kind == "decode":
            self._time += self._cost.decode_seconds(rows)
        else:
            raise ValueError(f"未知の種類: {kind}")
