"""推論サーバの実行ループ(反復ごとのスケジューリング)。

1 回の反復(iteration)は次の順に進む。

1. 到着した要求を待ち行列に入れる。実行する要求も待っている要求もなければ、
   時計を次の到着まで進める。
2. スケジューラが、待っている要求から実行に入れるものを選び、ストアに場所を割り当てる。
3. **prefill**: 入れた要求の文脈(書き出し。退避から戻した要求は、それまでの生成も含む)を 1 回の
   forward で処理し、各要求の最初の(退避から戻した要求では次の)トークンを生成する。接頭辞が共有
   できるときは、計算済みの部分を省く。
4. **decode**: この反復の前から実行中だった要求について、直前に生成したトークンを入力に
   1 トークン進める。
   次のトークンの置き場所が足りなければ、最後に実行に入れた要求を **退避** する
   (Key / Value を捨てて待ち行列の
   先頭に戻す。再開のときは prefill し直す = 再計算)。
5. 終わった要求の場所を返す。static batching では、バッチ全体が終わるまで、終わった行も計算に残す。

1 つの要求は 1 反復に高々 1 トークンを生成する。したがって、反復あたりの「実行中の要求数」は、
その反復で生成したトークンの数に等しい(終わった行を残して計算する static batching では、
計算した行の数がそれより多い)。

導入: 027。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.serving.clock import Clock
from src.serving.kv_store import KeyValueStore, Row
from src.serving.requests import Request, RequestState, RequestStatus
from src.serving.scheduler import Scheduler


@dataclass(frozen=True)
class RequestRecord:
    """1 件の要求の結果(時刻は実行の開始を 0 とする秒)。"""

    request_id: int
    prompt_length: int
    output_length: int
    arrival_time: float
    admitted_time: float
    first_token_time: float
    finish_time: float
    num_preemptions: int

    @property
    def queue_wait(self) -> float:
        """到着から最初に実行に入るまでの待ち時間。"""
        return self.admitted_time - self.arrival_time

    @property
    def time_to_first_token(self) -> float:
        """到着から最初のトークンを生成するまでの時間(Time To First Token)。"""
        return self.first_token_time - self.arrival_time

    @property
    def time_per_output_token(self) -> float | None:
        """最初のトークンの後の、1 トークンあたりの時間(出力トークンあたりの時間)。

        出力長 1 では定義しない(None)。
        """
        if self.output_length < 2:
            return None
        return (self.finish_time - self.first_token_time) / (self.output_length - 1)


@dataclass(frozen=True)
class IterationRecord:
    """1 反復の記録。

    Attributes:
        start_time, end_time: 反復の開始と終了の時刻。
        prefill_rows, prefill_query_tokens: prefill の行数と、問い合わせのトークン数
            (行数 × 最長の長さ)。
        decode_rows: decode で計算した行数(static batching では終わった行を含む)。
        active_requests: この反復でトークンを生成した要求の数(実行中の要求数)。
        waiting: 反復の終わりに待っている要求の数。
        tokens_used, tokens_allocated, units_in_use: 反復の終わり(要求の場所を返す前)の
            メモリの使用状況。
        preemptions: ここまでの退避の累積回数。
    """

    start_time: float
    end_time: float
    prefill_rows: int
    prefill_query_tokens: int
    decode_rows: int
    active_requests: int
    waiting: int
    tokens_used: int
    tokens_allocated: int
    units_in_use: int
    preemptions: int


@dataclass
class RunResult:
    """1 回の実行の結果。

    Attributes:
        requests: 要求ごとの記録(``request_id`` の昇順)。
        iterations: 反復ごとの記録。
        generated: 要求ごとに生成したトークン。
        logits: ``record_logits`` が真のとき、要求ごと・生成したトークンごとの logits。
    """

    requests: list[RequestRecord]
    iterations: list[IterationRecord]
    generated: dict[int, list[int]]
    logits: dict[int, list[np.ndarray]] = field(default_factory=dict)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_length for r in self.requests)

    @property
    def makespan(self) -> float:
        """最初の到着から最後の完了までの時間。"""
        return max(r.finish_time for r in self.requests) - min(
            r.arrival_time for r in self.requests
        )

    @property
    def num_preemptions(self) -> int:
        return self.iterations[-1].preemptions if self.iterations else 0


class ServingEngine:
    """実行ループ。

    Args:
        scheduler: バッチのスケジューラ。
        store: KV キャッシュのストア。
        runner: モデルの実行器(``ModelRunner`` または ``DryRunner``)。
        clock: 時計(``WallClock`` または ``ModelClock``)。
        record_logits: True のとき、生成したトークンごとの logits を記録する(正しさの確認用)。
    """

    def __init__(
        self,
        scheduler: Scheduler,
        store: KeyValueStore,
        runner,
        clock: Clock,
        record_logits: bool = False,
    ) -> None:
        self.scheduler = scheduler
        self.store = store
        self.runner = runner
        self.clock = clock
        self.record_logits = record_logits

    def _emit(self, state: RequestState, token: int, logits, time_seconds: float) -> int:
        state.generated.append(int(token))
        if self.record_logits:
            state.logits.append(np.array(logits))
        if state.first_token_time is None:
            state.first_token_time = time_seconds
        if state.finished:
            state.finish_time = time_seconds
            state.status = RequestStatus.FINISHED
        return 1

    def _preempt(self, victim: RequestState, running: list[RequestState], waiting: deque) -> None:
        self.store.release(victim)
        running.remove(victim)
        victim.status = RequestStatus.WAITING
        victim.num_cached_tokens = 0
        victim.num_preemptions += 1
        waiting.appendleft(victim)

    def run(self, requests: Sequence[Request]) -> RunResult:
        """``requests`` をすべて処理し終えるまで実行する。"""
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("request_id が重複している")
        states = [RequestState(r) for r in requests]
        pending = deque(
            sorted(states, key=lambda s: (s.request.arrival_time, s.request.request_id))
        )
        waiting: deque[RequestState] = deque()
        running: list[RequestState] = []
        retains = self.scheduler.retains_finished_rows
        iterations: list[IterationRecord] = []
        admit_counter = 0
        preemptions = 0
        self.clock.start()

        while pending or waiting or running:
            now = self.clock.now()
            while pending and pending[0].request.arrival_time <= now:
                waiting.append(pending.popleft())
            if not waiting and not running:
                self.clock.advance_to(pending[0].request.arrival_time)
                continue

            start = now
            before = list(running)
            produced = 0
            preempted_now = False

            # --- prefill ---
            admitted = self.scheduler.select(waiting, running, self.store)
            prefill_rows = []
            contexts = []
            for state, cached in admitted:
                context = state.context_token_ids()
                contexts.append(context)
                prefill_rows.append(Row(state, context[cached:], cached))
            prefill_query_tokens = 0
            if prefill_rows:
                tokens, logits = self.runner.step(prefill_rows, self.store, self.record_logits)
                longest = max(len(row.query_token_ids) for row in prefill_rows)
                prefill_query_tokens = len(prefill_rows) * longest
                self.clock.tick("prefill", len(prefill_rows), prefill_query_tokens)
                time_prefill = self.clock.now()
                for i, (state, _) in enumerate(admitted):
                    state.num_cached_tokens = len(contexts[i])
                    self.store.after_prefill(state)
                    state.status = RequestStatus.RUNNING
                    state.admit_order = admit_counter
                    admit_counter += 1
                    if state.admitted_time is None:
                        state.admitted_time = start
                    running.append(state)
                    produced += self._emit(
                        state, tokens[i], logits[i] if logits is not None else None, time_prefill
                    )

            # --- decode ---
            ready: list[RequestState] = []
            for state in before:
                if state.status == RequestStatus.WAITING:
                    continue  # この反復の前半で退避された
                while not self.store.grow(state):
                    candidates = [s for s in running if not s.finished]
                    if retains or not candidates:
                        raise RuntimeError("メモリが足りず、退避できる要求もない(予算が小さすぎる)")
                    victim = max(candidates, key=lambda s: s.admit_order)
                    self._preempt(victim, running, waiting)
                    preemptions += 1
                    preempted_now = True
                    if victim in ready:
                        ready.remove(victim)
                    if victim is state:
                        break
                if state.status != RequestStatus.WAITING:
                    ready.append(state)
            if ready:
                rows = [Row(s, [s.last_token()], s.num_cached_tokens) for s in ready]
                tokens, logits = self.runner.step(rows, self.store, self.record_logits)
                self.clock.tick("decode", len(rows), len(rows))
                time_decode = self.clock.now()
                for i, state in enumerate(ready):
                    state.num_cached_tokens += 1
                    if not state.finished:
                        produced += self._emit(
                            state, tokens[i], logits[i] if logits is not None else None, time_decode
                        )

            if not prefill_rows and not ready and not preempted_now:
                raise RuntimeError(
                    "進行できない(待っている要求を実行に入れられず、実行中の要求もない。予算が小さすぎる)"
                )

            usage = self.store.usage()
            # --- 終わった要求の場所を返す ---
            if retains:
                if running and all(s.finished for s in running):
                    for state in running:
                        self.store.release(state)
                    running.clear()
            else:
                for state in [s for s in running if s.finished]:
                    self.store.release(state)
                    running.remove(state)

            iterations.append(
                IterationRecord(
                    start_time=start,
                    end_time=self.clock.now(),
                    prefill_rows=len(prefill_rows),
                    prefill_query_tokens=prefill_query_tokens,
                    decode_rows=len(ready),
                    active_requests=produced,
                    waiting=len(waiting),
                    tokens_used=usage.tokens_used,
                    tokens_allocated=usage.tokens_allocated,
                    units_in_use=usage.units_in_use,
                    preemptions=preemptions,
                )
            )

        records = [
            RequestRecord(
                request_id=s.request.request_id,
                prompt_length=s.request.prompt_length,
                output_length=s.request.output_length,
                arrival_time=s.request.arrival_time,
                admitted_time=s.admitted_time,
                first_token_time=s.first_token_time,
                finish_time=s.finish_time,
                num_preemptions=s.num_preemptions,
            )
            for s in sorted(states, key=lambda s: s.request.request_id)
        ]
        result = RunResult(
            requests=records,
            iterations=iterations,
            generated={s.request.request_id: list(s.generated) for s in states},
        )
        if self.record_logits:
            result.logits = {s.request.request_id: list(s.logits) for s in states}
        return result
