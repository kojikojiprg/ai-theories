"""常駐のエンジン: 反復(iteration)を 1 つずつ進める continuous batching / static batching。

``src/serving/engine.py`` の ``ServingEngine.run()`` は、到着時刻つきの要求の一覧を受け取って
最後まで走らせる形で、実行中に要求を受け付けられず、トークンを 1 つずつ外へ渡せず、行ごとの
デコードの設定を持てない。常駐させるには、この 3 つが要る。**``src/`` は変えず**、
``src/serving/`` の部品(``Scheduler``・``KeyValueStore``・``ServingModel``・``RequestState``・
``Row``)を組み合わせて、1 反復ずつ進める ``ResidentEngine`` をアプリ側に置く。

1 反復の手順は ``ServingEngine.run()`` と同じ(``src/serving/engine.py`` の docstring を参照)で、
**1 回の forward** で、実行中の要求の decode の行と、新しく入れた要求の prefill の行を進める。
違いは次の点だけである。

- 要求は ``submit()`` で実行中にも届く(別のスレッドから呼べる)。反復の頭で待ち行列に移す。
- 生成したトークンは、1 つずつ ``RequestSink`` に渡す(HTTP の層が、逐次表示の応答に使う)。
- 行ごとに ``DecodingSettings`` を持つ(貪欲法・temperature・top-k・top-p)。貪欲法の行は、
  ``ModelRunner`` と同じ ``logits.argmax`` で選ぶ。
- 要求の取り消しと時間の上限を、反復の頭で処理する。ブロックを返し、待ち行列から取り除く。
- メモリが足りず進行できない要求は、エンジンを止めずに、その要求だけを誤りで終える。
- 時計は実時間(``time.perf_counter``)で、到着の間の待ちは本当に待つ(``ServingEngine`` の仮想の時計
  ではない)。

``ServingEngine.run()`` と同じ負荷(全要求が先に届いている場合)を流すと、生成するトークンも反復ごとの
記録(行数・退避の回数・メモリの使用量)も一致する。この一致は
``tests/unit/test_serving_engine.py`` が確かめる(``src/`` の側が変わったときに、ここが黙って
食い違うのを防ぐ)。

スレッドの約束: ``step()`` と、デバイスに触れるすべての処理は、エンジンを動かす 1 本のスレッド
(``lab/serving/runtime.py`` の ``DeviceWorker``)だけが呼ぶ。``submit()``・``cancel()``・
``stats()``・``begin_drain()`` は、どのスレッドからでも呼べる(ロックで守る)。

導入: 段階 2-1(システムの仕組み。使った理論は 027・010・008)。
"""

from __future__ import annotations

import contextlib
import itertools
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from lab.features.generation.service import (
    METHOD_BEAM,
    METHOD_GREEDY,
    DecodingSettings,
    select_next_token,
)
from src.serving import (
    IterationRecord,
    KeyValueStore,
    Request,
    RequestState,
    RequestStatus,
    ReservedKeyValueStore,
    Row,
    Scheduler,
    ServingModel,
)

FINISH_LENGTH = "length"
FINISH_CANCELLED = "cancelled"
FINISH_TIMEOUT = "timeout"
FINISH_SHUTDOWN = "shutdown"
FINISH_ERROR = "error"

RECENT_WINDOW_SECONDS = 10.0
"""``recent_tokens_per_second`` を数える直近の時間。"""
RETRY_WINDOW_SECONDS = 30.0
"""再試行の目安(``Retry-After``)を見積もるときに、完了した要求を数える直近の時間。"""
RETRY_AFTER_LIMITS = (1, 60)
"""再試行の目安の下限と上限 [秒]。"""


# ---------------------------------------------------------------------------
# 受け付けの誤り(HTTP の層が、状態コードに対応させる)
# ---------------------------------------------------------------------------
class SubmitError(Exception):
    """要求を受け付けられなかった。``code`` は API の誤りの種類。"""

    code = "rejected"


class QueueFull(SubmitError):
    """待ち行列が上限に達している(HTTP 429)。``retry_after_seconds`` は再試行の目安 [秒]。"""

    code = "queue_full"

    def __init__(self, message: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class EngineClosed(SubmitError):
    """停止中、または故障していて受け付けない(HTTP 503)。"""

    code = "unavailable"


class RequestTooLarge(SubmitError):
    """この KV キャッシュの予算では、単独でも実行できない長さ(HTTP 422)。"""

    code = "request_too_large"


# ---------------------------------------------------------------------------
# 要求の結果と、トークンの受け取り口
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RequestOutcome:
    """1 件の要求の終わり方と時間(秒。到着 = ``submit()`` の呼び出し)。

    Attributes:
        finish_reason: ``length``(上限のトークン数まで生成した)・``cancelled``・``timeout``・
            ``shutdown``・``error``。
        message: ``length`` 以外のときの説明(日本語)。
        queue_wait_seconds: 到着から最初に実行に入るまで(実行に入らなかったときは None)。
        time_to_first_token_seconds: 到着から最初のトークンまで(生成しなかったときは None)。
        total_seconds: 到着から終わりまで。
        time_per_output_token_seconds: 最初のトークンの後の 1 トークンあたり
            (出力が 2 トークン未満のときは None)。
    """

    request_id: int
    finish_reason: str
    message: str | None
    prompt_tokens: int
    completion_tokens: int
    queue_wait_seconds: float | None
    time_to_first_token_seconds: float | None
    total_seconds: float
    time_per_output_token_seconds: float | None
    num_preemptions: int


class RequestSink(Protocol):
    """生成されたトークンと終わりの受け取り口。エンジンのスレッドから呼ばれる(短く返すこと)。"""

    def on_token(self, token_id: int, index: int) -> None: ...

    def on_finish(self, outcome: RequestOutcome) -> None: ...


@dataclass
class _Served:
    """1 件の要求の、エンジンの内側の記録。"""

    state: RequestState
    settings: DecodingSettings
    sink: RequestSink
    submitted_at: float
    deadline: float
    generator: torch.Generator | None = None
    started: bool = False
    admitted_at: float | None = None
    first_token_at: float | None = None
    closed: bool = False
    logits: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class _Snapshot:
    """反復の終わりに、エンジンのスレッドが作る状態の写し(他のスレッドは、これだけを読む)。"""

    waiting: int = 0
    running: int = 0
    tokens_used: int = 0
    tokens_allocated: int = 0
    units_in_use: int = 0


def _index_of(items: Iterable[RequestState], state: RequestState) -> int | None:
    for i, item in enumerate(items):
        if item is state:
            return i
    return None


def per_request_capacity(store: KeyValueStore) -> int:
    """1 件の要求が単独で使える KV キャッシュのトークン枠の数。"""
    if isinstance(store, ReservedKeyValueStore):
        return store.slot_length
    return store.capacity_tokens


class ResidentEngine:
    """反復を 1 つずつ進める常駐のエンジン(モジュールの docstring を参照)。

    Args:
        name: 識別名(モデルの名前など。統計に出す)。
        serving_model: 行ごとに位置の違うバッチを実行する実行器。
        scheduler: バッチのスケジューラ(continuous または static)。
        store: KV キャッシュのストア。
        max_queue_length: まだ一度も実行に入っていない要求の数の上限(超えると ``QueueFull``)。
        request_timeout_seconds: 到着から数えた 1 件の要求の時間の上限。
        record_logits: True のとき、生成したトークンごとの logits を記録する(正しさの確認用)。
        record_iterations: 残す反復の記録の数(0 で記録しない。試験用)。
        clock: 時計(秒)。試験で差し替える。
        on_iteration: 反復の頭(エンジンのスレッド)で呼ぶ関数。試験で反復を遅くするために使う。
    """

    def __init__(
        self,
        name: str,
        serving_model: ServingModel,
        scheduler: Scheduler,
        store: KeyValueStore,
        *,
        max_queue_length: int = 64,
        request_timeout_seconds: float = 60.0,
        record_logits: bool = False,
        record_iterations: int = 0,
        clock: Callable[[], float] = time.perf_counter,
        on_iteration: Callable[[], None] | None = None,
    ) -> None:
        if max_queue_length < 1 or request_timeout_seconds <= 0:
            raise ValueError("待ち行列の上限は 1 以上、時間の上限は正である必要がある")
        self.name = name
        self.serving_model = serving_model
        self.scheduler = scheduler
        self.store = store
        self.device = serving_model.device
        self.max_queue_length = max_queue_length
        self.request_timeout_seconds = request_timeout_seconds
        self.record_logits = record_logits
        self.max_context_length = min(
            serving_model.max_sequence_length, per_request_capacity(store)
        )
        self._clock = clock
        self._on_iteration = on_iteration
        self._origin = clock()
        self._retains = scheduler.retains_finished_rows

        # 再入できるロック: シグナルの処理が ``submit()`` の途中に割り込んでも固まらない
        self._lock = threading.RLock()
        self._intake: deque[_Served] = deque()
        self._cancels: dict[int, str] = {}
        self._draining = False
        self._failure: str | None = None
        self._wake: Callable[[], None] | None = None
        self._ids = itertools.count()
        self._unstarted = 0

        # 以下はエンジンのスレッドだけが触る
        self._waiting: deque[RequestState] = deque()
        self._running: list[RequestState] = []
        self._by_id: dict[int, _Served] = {}
        self._dead: set[int] = set()
        self._admit_counter = 0
        self.records: deque[IterationRecord] = deque(maxlen=record_iterations or 0)
        self._record_iterations = record_iterations > 0

        # 統計(ロックの下で更新・読み取り)
        self._counters = dict.fromkeys(
            ("accepted", "rejected", "completed", "cancelled", "timed_out", "failed", "shutdown"), 0
        )
        self._iterations = 0
        self._preemptions = 0
        self._output_tokens = 0
        self._busy_seconds = 0.0
        self._recent: deque[tuple[float, int]] = deque()
        self._completions: deque[float] = deque()
        self._snapshot = _Snapshot()
        # ``record_logits`` が真のとき、終わった要求ごとの logits(生成したトークンごと。試験用)
        self.recorded_logits: dict[int, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # どのスレッドからでも呼べる
    # ------------------------------------------------------------------
    def set_wake(self, wake: Callable[[], None] | None) -> None:
        """仕事が増えたときに呼ぶ関数(エンジンを動かすスレッドを起こす)。"""
        self._wake = wake

    def _notify(self) -> None:
        if self._wake is not None:
            self._wake()

    def submit(
        self, prompt_token_ids: Sequence[int], settings: DecodingSettings, sink: RequestSink
    ) -> int:
        """要求を待ち行列に入れ、要求の識別子を返す。

        Raises:
            ValueError: 設定と入力の組み合わせが生成できないもの(``DecodingSettings.validate``)。
            RequestTooLarge: この KV キャッシュの予算では、単独でも実行できない長さ。
            QueueFull: 待ち行列が上限に達している。
            EngineClosed: 停止中、または故障している。
        """
        if settings.method == METHOD_BEAM:
            raise ValueError("ビームサーチは常駐サーバでは使えません。")
        prompt = tuple(int(t) for t in prompt_token_ids)
        settings.validate(len(prompt), self.serving_model.max_sequence_length)
        if len(prompt) + settings.max_new_tokens > self.max_context_length:
            raise RequestTooLarge(
                f"書き出し {len(prompt)} + 生成 {settings.max_new_tokens} トークンは、"
                f"この KV キャッシュの予算({self.max_context_length} トークン)に載りません。"
            )
        now = self._clock()
        with self._lock:
            if self._failure is not None:
                raise EngineClosed(f"エンジンが故障しています: {self._failure}")
            if self._draining:
                raise EngineClosed("サーバは停止中で、新しい要求を受け付けません。")
            if self._unstarted >= self.max_queue_length:
                self._counters["rejected"] += 1
                raise QueueFull(
                    f"待ち行列が上限({self.max_queue_length} 件)に達しています。",
                    self._retry_after_locked(now),
                )
            request_id = next(self._ids)
            request = Request(
                request_id, prompt, settings.max_new_tokens, arrival_time=now - self._origin
            )
            served = _Served(
                RequestState(request),
                settings,
                sink,
                submitted_at=now,
                deadline=now + self.request_timeout_seconds,
            )
            self._intake.append(served)
            self._unstarted += 1
            self._counters["accepted"] += 1
        self._notify()
        return request_id

    def cancel(self, request_id: int, reason: str = FINISH_CANCELLED) -> None:
        """要求を取り消す(反復の頭で処理する。終わった要求や未知の識別子は何もしない)。"""
        with self._lock:
            self._cancels.setdefault(request_id, reason)
        self._notify()

    def begin_drain(self) -> None:
        """新しい要求を受け付けなくし、まだ一度も実行に入っていない要求を取り消す(実行中は終わらせる)。"""
        with self._lock:
            self._draining = True
        self._notify()

    def mark_failed(self, message: str) -> None:
        """故障として記録し、新しい要求を受け付けなくする。"""
        with self._lock:
            self._failure = message

    @property
    def failure(self) -> str | None:
        return self._failure

    def has_work(self) -> bool:
        """届いた要求、待っている要求、実行中の要求のどれかがあるか(エンジンのスレッドから呼ぶ)。"""
        with self._lock:
            pending = bool(self._intake or self._cancels)
        return pending or bool(self._waiting or self._running)

    def stats(self) -> dict:
        """状態の写し(待っている・実行中の要求数、KV キャッシュの使用量、これまでの数と速さ)。"""
        now = self._clock()
        with self._lock:
            counters = dict(self._counters)
            intake = len(self._intake)
            snapshot = self._snapshot
            iterations, preemptions = self._iterations, self._preemptions
            tokens, busy = self._output_tokens, self._busy_seconds
            recent = sum(n for t, n in self._recent if now - t <= RECENT_WINDOW_SECONDS)
            unstarted = self._unstarted
            draining, failure = self._draining, self._failure
        unit = "slot" if isinstance(self.store, ReservedKeyValueStore) else "block"
        return {
            "name": self.name,
            "scheduler": self.scheduler.name,
            "kv_manager": "reserved" if unit == "slot" else "paged",
            "max_batch_size": self.scheduler.max_batch_size,
            "state": "failed" if failure else ("stopping" if draining else "ready"),
            "waiting": snapshot.waiting + intake,
            "not_started": unstarted,
            "running": snapshot.running,
            "max_queue_length": self.max_queue_length,
            "kv_cache": {
                "unit": unit,
                "units_in_use": snapshot.units_in_use,
                "tokens_used": snapshot.tokens_used,
                "tokens_allocated": snapshot.tokens_allocated,
                "capacity_tokens": self.store.capacity_tokens,
            },
            "requests": counters,
            "iterations": iterations,
            "preemptions": preemptions,
            "output_tokens": tokens,
            "busy_seconds": busy,
            "tokens_per_second": tokens / busy if busy > 0 else None,
            "recent_tokens_per_second": recent / RECENT_WINDOW_SECONDS,
        }

    def _retry_after_locked(self, now: float) -> int:
        """再試行の目安 [秒]: 直近に完了した要求の数から、1 件分の空きができる時間を見積もる。"""
        low, high = RETRY_AFTER_LIMITS
        done = [t for t in self._completions if now - t <= RETRY_WINDOW_SECONDS]
        if not done:
            return low
        window = min(RETRY_WINDOW_SECONDS, max(now - self._origin, 1e-6))
        rate = len(done) / window
        return int(min(high, max(low, math.ceil(1.0 / rate))))

    # ------------------------------------------------------------------
    # エンジンのスレッドだけが呼ぶ
    # ------------------------------------------------------------------
    def step(self) -> bool:
        """届いた要求と取り消しを処理し、仕事があれば 1 反復進める。仕事があれば True。"""
        self._drain(self._clock())
        if not self._waiting and not self._running:
            self._publish()
            return False
        self._iterate()
        return True

    def abort_all(self, reason: str, message: str) -> None:
        """すべての要求を終え、ブロックを返す(停止の猶予が過ぎたとき、故障したとき)。"""
        with self._lock:
            arrivals = list(self._intake)
            self._intake.clear()
        for served in arrivals:
            self._by_id[served.state.request.request_id] = served
            self._waiting.append(served.state)
        for served in list(self._by_id.values()):
            self._terminate(served, reason, message)
        for state in list(self._running):
            self.store.release(state)
        self._running.clear()
        self._dead.clear()
        self._publish()

    def _done(self, state: RequestState) -> bool:
        return state.finished or state.request.request_id in self._dead

    def _drain(self, now: float) -> None:
        with self._lock:
            arrivals = list(self._intake)
            self._intake.clear()
            cancels = dict(self._cancels)
            self._cancels.clear()
            draining = self._draining
        for served in arrivals:
            self._by_id[served.state.request.request_id] = served
            self._waiting.append(served.state)
        for request_id, reason in cancels.items():
            served = self._by_id.get(request_id)
            if served is not None:
                self._terminate(served, reason, _MESSAGES.get(reason, "取り消されました。"))
        for served in list(self._by_id.values()):
            if now >= served.deadline:
                self._terminate(served, FINISH_TIMEOUT, _MESSAGES[FINISH_TIMEOUT])
            elif draining and not served.started:
                self._terminate(served, FINISH_SHUTDOWN, _MESSAGES[FINISH_SHUTDOWN])
        self._release_finished_batch()

    def _release_finished_batch(self) -> None:
        """static batching: バッチの全行が終わったら(取り消された行を含めて)ブロックを返す。"""
        if self._retains and self._running and all(self._done(s) for s in self._running):
            for state in self._running:
                self.store.release(state)
            self._running.clear()
            self._dead.clear()

    def _terminate(self, served: _Served, reason: str, message: str | None) -> None:
        """要求を途中で終える。待ち行列から外すか、ブロックを返す。"""
        state = served.state
        waiting_at = _index_of(self._waiting, state)
        if waiting_at is not None:
            del self._waiting[waiting_at]
        elif _index_of(self._running, state) is not None:
            if self._retains:
                self._dead.add(state.request.request_id)  # バッチが終わるまで、死んだ行として残す
            else:
                self.store.release(state)
                self._running.remove(state)
        self._close(served, reason, message)

    def _close(self, served: _Served, reason: str, message: str | None = None) -> None:
        """結果を通知する(1 回だけ)。"""
        if served.closed:
            return
        served.closed = True
        state = served.state
        request_id = state.request.request_id
        now = self._clock()
        self._by_id.pop(request_id, None)
        if self.record_logits:
            self.recorded_logits[request_id] = served.logits
        with self._lock:
            if not served.started:
                self._unstarted -= 1
            key = {
                FINISH_LENGTH: "completed",
                FINISH_CANCELLED: "cancelled",
                FINISH_TIMEOUT: "timed_out",
                FINISH_SHUTDOWN: "shutdown",
                FINISH_ERROR: "failed",
            }[reason]
            self._counters[key] += 1
            if reason == FINISH_LENGTH:
                self._completions.append(now)
                while self._completions and now - self._completions[0] > RETRY_WINDOW_SECONDS:
                    self._completions.popleft()
        count = len(state.generated)
        first = served.first_token_at
        per_token = None
        if first is not None and count >= 2:
            per_token = (now - first) / (count - 1)
        outcome = RequestOutcome(
            request_id=request_id,
            finish_reason=reason,
            message=message,
            prompt_tokens=state.request.prompt_length,
            completion_tokens=count,
            queue_wait_seconds=None
            if served.admitted_at is None
            else served.admitted_at - served.submitted_at,
            time_to_first_token_seconds=None if first is None else first - served.submitted_at,
            total_seconds=now - served.submitted_at,
            time_per_output_token_seconds=per_token,
            num_preemptions=state.num_preemptions,
        )
        with contextlib.suppress(Exception):  # 受け取り側の誤りで、エンジンを止めない
            served.sink.on_finish(outcome)

    def _preempt(self, victim: RequestState) -> None:
        """Key / Value を捨てて待ち行列の先頭に戻す(再開のときは文脈全体を prefill し直す)。"""
        self.store.release(victim)
        self._running.remove(victim)
        victim.status = RequestStatus.WAITING
        victim.num_cached_tokens = 0
        victim.num_preemptions += 1
        self._waiting.appendleft(victim)

    def _fail(self, state: RequestState, message: str) -> None:
        served = self._by_id.get(state.request.request_id)
        if served is not None:
            self._terminate(served, FINISH_ERROR, message)

    def _iterate(self) -> None:
        if self._on_iteration is not None:
            self._on_iteration()
        store, scheduler = self.store, self.scheduler
        running, waiting = self._running, self._waiting
        start = self._clock()
        limit = self.serving_model.max_sequence_length
        before = list(running)
        preempted_now = False

        # --- 1) decode の行の置き場所を用意する(実行中の要求が優先。古い要求から順に) ---
        ready: list[RequestState] = []
        for state in before:
            if state.status == RequestStatus.WAITING:
                continue  # この反復で、先に処理した要求のために退避された
            if self._done(state) and state.num_cached_tokens >= limit:
                continue  # 終わった行の位置がモデルの範囲を出た(static: 計算を続けない)
            abandoned = False
            while not store.grow(state):
                candidates = [s for s in running if not self._done(s)]
                if self._retains or not candidates:
                    self._fail(state, "KV キャッシュの予算が足りず、この要求を進められません。")
                    abandoned = True
                    break
                victim = max(candidates, key=lambda s: s.admit_order)
                self._preempt(victim)
                self._preemptions += 1
                preempted_now = True
                if (at := _index_of(ready, victim)) is not None:
                    del ready[at]
                if victim is state:
                    break
            if not abandoned and state.status != RequestStatus.WAITING:
                ready.append(state)

        # --- 2) 新しい要求を入れる(残りの場所から。退避が起きた反復では入れない) ---
        admitted = [] if preempted_now else scheduler.select(waiting, running, store)

        # --- 3) 1 回の forward(decode の行 + prefill の行) ---
        rows = [Row(s, [s.last_token()], s.num_cached_tokens) for s in ready]
        contexts = []
        for state, cached in admitted:
            context = state.context_token_ids()
            contexts.append(context)
            rows.append(Row(state, context[cached:], cached))
        if not rows:
            if not preempted_now and waiting:
                head = waiting[0]  # 退避もなく、待っている要求を入れられない: 先頭を誤りで終える
                self._fail(head, "KV キャッシュの予算が足りず、この要求を実行に入れられません。")
            self._release_finished_batch()
            self._publish()
            return

        now = self._clock()
        for state, _ in admitted:
            served = self._by_id[state.request.request_id]
            if not served.started:
                served.started = True
                served.admitted_at = now
                with self._lock:
                    self._unstarted -= 1
        forward_start = self._clock()
        layout = store.build_layout(rows)
        logits = self.serving_model.forward(layout, store)

        num_decode = len(ready)
        live: list[tuple[int, _Served]] = []
        for i, state in enumerate(ready):
            state.num_cached_tokens += 1
            if not self._done(state):
                live.append((i, self._by_id[state.request.request_id]))
        resumed_rows = 0
        for k, (state, _) in enumerate(admitted):
            resumed_rows += bool(state.generated)
            state.num_cached_tokens = len(contexts[k])
            store.after_prefill(state)
            state.status = RequestStatus.RUNNING
            state.admit_order = self._admit_counter
            self._admit_counter += 1
            running.append(state)
            live.append((num_decode + k, self._by_id[state.request.request_id]))

        tokens = self._choose_tokens(logits, live)  # デバイスの完了を待つ
        forward_end = self._clock()
        logits_array = logits.cpu().numpy() if self.record_logits else None
        for (i, served), token in zip(live, tokens, strict=True):
            self._emit(
                served, token, forward_end, None if logits_array is None else logits_array[i]
            )

        usage = store.usage()
        # --- 終わった要求の場所を返す ---
        if self._retains:
            self._release_finished_batch()
        else:
            for state in [s for s in running if s.finished]:
                store.release(state)
                running.remove(state)

        end = self._clock()
        self._publish(usage)
        with self._lock:
            self._iterations += 1
            self._busy_seconds += end - start
            self._output_tokens += len(live)
            self._recent.append((end, len(live)))
            while self._recent and end - self._recent[0][0] > RECENT_WINDOW_SECONDS:
                self._recent.popleft()
        if self._record_iterations:
            self.records.append(
                IterationRecord(
                    start_time=start - self._origin,
                    end_time=end - self._origin,
                    prefill_rows=len(admitted),
                    decode_rows=num_decode,
                    resumed_rows=resumed_rows,
                    prefill_tokens=sum(len(row.query_token_ids) for row in rows[num_decode:]),
                    query_tokens=len(rows) * max(len(row.query_token_ids) for row in rows),
                    active_requests=len(live),
                    waiting=len(waiting),
                    tokens_used=usage.tokens_used,
                    tokens_allocated=usage.tokens_allocated,
                    units_in_use=usage.units_in_use,
                    preemptions=self._preemptions,
                    forward_start_time=forward_start - self._origin,
                    forward_end_time=forward_end - self._origin,
                    layout_seconds=0.0,
                    idle_seconds=0.0,
                )
            )

    def _choose_tokens(self, logits: Tensor, live: list[tuple[int, _Served]]) -> list[int]:
        """生成する行の次のトークンを選ぶ。貪欲法の行は 1 回の argmax、サンプリングの行は行ごと。

        デバイスからホストへの転送は 1 回にまとめる。
        """
        if not live:
            return []
        greedy = [pos for pos, (_, s) in enumerate(live) if s.settings.method == METHOD_GREEDY]
        sampled = [pos for pos, (_, s) in enumerate(live) if s.settings.method != METHOD_GREEDY]
        parts: list[Tensor] = []
        if greedy:
            index = torch.tensor([live[pos][0] for pos in greedy], device=logits.device)
            parts.append(logits.index_select(0, index).argmax(dim=-1))
        for pos in sampled:
            i, served = live[pos]
            if served.generator is None:
                served.generator = torch.Generator(device=logits.device)
                served.generator.manual_seed(served.settings.seed)
            parts.append(
                select_next_token(logits[i : i + 1], served.settings, served.generator).reshape(1)
            )
        flat = torch.cat(parts).tolist()
        tokens = [0] * len(live)
        for pos, token in zip(greedy + sampled, flat, strict=True):
            tokens[pos] = int(token)
        return tokens

    def _emit(self, served: _Served, token: int, now: float, logits_row: np.ndarray | None) -> None:
        state = served.state
        state.generated.append(int(token))
        if logits_row is not None:
            served.logits.append(np.array(logits_row))
        if served.first_token_at is None:
            served.first_token_at = now
        with contextlib.suppress(Exception):  # 受け取り側の誤りで、エンジンを止めない
            served.sink.on_token(int(token), len(state.generated) - 1)
        if state.finished:
            state.status = RequestStatus.FINISHED
            self._close(served, FINISH_LENGTH)

    def _publish(self, usage=None) -> None:
        usage = usage or self.store.usage()
        snapshot = _Snapshot(
            waiting=len(self._waiting),
            running=sum(1 for s in self._running if not self._done(s)),
            tokens_used=usage.tokens_used,
            tokens_allocated=usage.tokens_allocated,
            units_in_use=usage.units_in_use,
        )
        with self._lock:
            self._snapshot = snapshot


_MESSAGES = {
    FINISH_CANCELLED: "取り消されました。",
    FINISH_TIMEOUT: "要求の時間の上限を超えたため、打ち切りました。",
    FINISH_SHUTDOWN: "サーバが停止するため、要求を取り消しました。",
}


def paged_store_blocks(max_batch_size: int, block_size: int, max_context: int) -> int:
    """最大バッチサイズの要求がそれぞれ最大の長さまで伸びても、退避が起きない数のブロック。"""
    return max_batch_size * -(-max_context // block_size)


__all__ = [
    "FINISH_CANCELLED",
    "FINISH_ERROR",
    "FINISH_LENGTH",
    "FINISH_SHUTDOWN",
    "FINISH_TIMEOUT",
    "EngineClosed",
    "QueueFull",
    "RequestOutcome",
    "RequestSink",
    "RequestTooLarge",
    "ResidentEngine",
    "SubmitError",
    "paged_store_blocks",
    "per_request_capacity",
]
