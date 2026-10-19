"""バッチのスケジューラ(どの要求をいつ実行に入れるか)。

推論サーバは、1 回の forward(1 反復)で複数の要求を同時に進める。どの要求を同じバッチに入れ、
いつ入れ替えるかの方針が、スループットと待ち時間を決める。

- **static batching**(``StaticBatchScheduler``): 要求の組を固定し、バッチの中で最も出力長が
  長い要求が終わるまで組を入れ替えない。短い要求が先に終わっても、その行は(捨てる出力のために)計算され続け、
  次のバッチを始められない。1 バッチの反復の数は ``max(O)``。
- **continuous batching**(``ContinuousBatchScheduler``、Orca の iteration-level scheduling、
  Yu et al., OSDI 2022): 反復ごとにスケジューリングする。終わった要求をすぐに抜き、待っている要求を
  その反復で入れる。入れた要求は、その反復で prefill(書き出しの全体を 1 回で処理して最初のトークンを
  生成)し、次の反復から decode(1 反復 1 トークン)に加わる。

どちらも先着順(待ち行列の先頭から)で、先頭の要求を入れられないときは、それより後ろの要求を
追い越して入れることはしない。

導入: 027。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from src.serving.kv_store import KeyValueStore
from src.serving.requests import RequestState


class Scheduler(ABC):
    """スケジューラの基底クラス。

    Attributes:
        max_batch_size: 同時に実行する要求の数の上限 B。
        retains_finished_rows: True のとき、終わった要求の行を、バッチ全体が終わるまで計算に残す。
    """

    retains_finished_rows: bool = False

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError(f"最大バッチサイズは 1 以上である必要がある: {max_batch_size}")
        self.max_batch_size = max_batch_size

    @abstractmethod
    def select(
        self, waiting: deque[RequestState], running: list[RequestState], store: KeyValueStore
    ) -> list[tuple[RequestState, int]]:
        """この反復で実行に入れる要求を選び、ストアに場所を割り当てる。

        Args:
            waiting: 到着済みで待っている要求(先頭が最も古い)。選んだ要求は取り除く。
            running: 既に実行中の行(static batching では、終わった行も含む)。
            store: KV キャッシュのストア。

        Returns:
            ``(要求, 既に計算済みで省けるトークンの数)`` のリスト(入れた順)。
        """

    def _fill(
        self, waiting: deque[RequestState], num_running: int, store: KeyValueStore
    ) -> list[tuple[RequestState, int]]:
        admitted: list[tuple[RequestState, int]] = []
        while waiting and num_running + len(admitted) < self.max_batch_size:
            state = waiting[0]
            if not store.can_admit(state, num_running + len(admitted)):
                break
            waiting.popleft()
            admitted.append((state, store.admit(state)))
        return admitted


class StaticBatchScheduler(Scheduler):
    """static batching。実行中のバッチが空になるまで、新しい要求を入れない。"""

    name = "static"
    retains_finished_rows = True

    def select(self, waiting, running, store):
        if running:
            return []
        return self._fill(waiting, 0, store)


class ContinuousBatchScheduler(Scheduler):
    """continuous batching。反復ごとに、空いた分だけ待っている要求を入れる。"""

    name = "continuous"

    def select(self, waiting, running, store):
        return self._fill(waiting, len(running), store)
