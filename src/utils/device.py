"""デバイスの同期(synchronization)。

学習ループの実時間を測るとき、評価や計測の直前に、デバイス上の非同期の処理の完了を待つために使う。
``src.utils`` のパッケージの初期化は torch を読み込まないが、本モジュールは torch を読み込むので、
``src.utils`` からは再エクスポートしない。
"""

from __future__ import annotations

import torch


def synchronize_device(device: torch.device) -> None:
    """デバイス上の非同期の処理がすべて終わるまで待つ(CPU では何もしない)。

    学習ループが ``time.time()`` で学習の実時間を測るとき、評価の関数を呼ぶ前と学習の終了時に
    呼ぶ(CUDA・MPS の処理は非同期で、同期しないと実時間が短く測れる)。019 以降の学習ループで
    共通に使う(各ループがモジュール内に持っていた同一の関数を統合した)。

    導入: 019。

    Args:
        device: 同期するデバイス。``cuda`` と ``mps`` のときだけ同期する。
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
