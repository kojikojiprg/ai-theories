"""計測。

計時は、デバイスの同期(``synchronize_device``)を取ってから始め、終わるときにも同期を取る。
途中で呼び出し側へ制御を渡している間(画面を更新している間)は計時を止めるので、画面の更新や、
ZeroGPU でのプロセス間の受け渡しの時間は含まれない。
"""

from __future__ import annotations

import time

import torch

from src.utils.device import synchronize_device


class ComputeClock:
    """計算に使った時間だけを積算する。デバイスの同期を取ってから計る。"""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._total = 0.0
        self._mark = 0.0

    def start(self) -> None:
        synchronize_device(self._device)
        self._mark = time.perf_counter()

    def stop(self) -> float:
        """計時を止め、ここまでの積算(秒)を返す。"""
        synchronize_device(self._device)
        self._total += time.perf_counter() - self._mark
        return self._total
