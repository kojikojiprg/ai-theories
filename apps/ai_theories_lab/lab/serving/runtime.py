"""エンジンを動かす 1 本のスレッドと、モデルの読み込み・起動・停止。

``DeviceWorker`` に相当する ``ServerRuntime`` は、**1 本のスレッド** (エンジンのスレッド)で、
デバイス(CUDA → MPS → CPU)とモデルを専有する。このスレッドが、(1) モデルとトークナイザの読み込み、
(2) モデルごとのエンジンの構築(KV キャッシュの確保)、(3) 反復のループ、のすべてを行う。HTTP の処理
(別のスレッド)は、要求をエンジンの待ち行列に入れ、生成されたトークンを受け取るだけで、
torch を呼ばない。

停止は 2 段階で、``begin_shutdown()`` が新しい要求の受け付けをやめ、まだ一度も実行に入っていない
要求を取り消す。実行中の要求は、猶予(``shutdown_grace_seconds``)の間は生成を続けて終わらせ、
猶予が過ぎても終わらない要求は取り消す。いずれの場合も、要求を取り消すときはブロックを返す。

導入: 段階 2-1(システムの仕組み。使った理論は 027・010)。
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from lab.core.artifacts import LoadedModel, Resources
from lab.serving.engine import (
    FINISH_ERROR,
    FINISH_SHUTDOWN,
    ResidentEngine,
    paged_store_blocks,
)
from src.serving import (
    ContinuousBatchScheduler,
    KeyValueStore,
    PagedKeyValueStore,
    ReservedKeyValueStore,
    Scheduler,
    ServingModel,
    StaticBatchScheduler,
)

SCHEDULERS = ("continuous", "static")
KV_MANAGERS = ("paged", "reserved")
SERVED_MODELS = ("main", "gqa")
"""API で選べるモデル(Hub のブランチ)。``sft-synthetic`` は指示への応答専用なので含めない。"""

STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_STOPPING = "stopping"
STATE_FAILED = "failed"

IDLE_WAIT_SECONDS = 0.05
"""仕事がないときに、スレッドが次の通知を待つ最長の時間 [秒]。"""


@dataclass(frozen=True)
class EngineConfig:
    """モデルごとのエンジンの設定(``serve.py`` の引数)。

    Attributes:
        scheduler: ``continuous``(continuous batching)または ``static``(static batching)。
        max_batch_size: 同時に実行する要求の数の上限。既定の 16 は 027 の 7.10 節による。
        kv_manager: ``paged``(PagedAttention のブロック管理)または ``reserved``(最大長の分を確保)。
        block_size: ブロックの大きさ(トークン)。``paged`` のとき。
        num_blocks: 物理ブロックの数。None のとき、最大バッチサイズの要求がそれぞれ最大の長さまで
            伸びても退避が起きない数にする。小さくすると、足りなくなったとき要求を退避する。
        prefix_caching: 接頭辞の共有。``paged`` のとき。
        max_queue_length: まだ一度も実行に入っていない要求の数の上限(超えると HTTP 429)。
        request_timeout_seconds: 1 件の要求の時間の上限(到着から数える。超えると取り消す)。
    """

    scheduler: str = "continuous"
    max_batch_size: int = 16
    kv_manager: str = "paged"
    block_size: int = 16
    num_blocks: int | None = None
    prefix_caching: bool = True
    max_queue_length: int = 64
    request_timeout_seconds: float = 60.0

    def validate(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler は {SCHEDULERS} のどれか: {self.scheduler!r}")
        if self.kv_manager not in KV_MANAGERS:
            raise ValueError(f"kv_manager は {KV_MANAGERS} のどれか: {self.kv_manager!r}")
        if self.max_batch_size < 1 or self.block_size < 1 or self.max_queue_length < 1:
            raise ValueError(
                "最大バッチサイズ・ブロックの大きさ・待ち行列の上限は 1 以上にしてください。"
            )
        if self.num_blocks is not None and self.num_blocks < 1:
            raise ValueError("ブロックの数は 1 以上にしてください。")
        if self.request_timeout_seconds <= 0:
            raise ValueError("要求の時間の上限は正にしてください。")


def build_scheduler(config: EngineConfig) -> Scheduler:
    cls = ContinuousBatchScheduler if config.scheduler == "continuous" else StaticBatchScheduler
    return cls(config.max_batch_size)


def build_store(serving_model: ServingModel, config: EngineConfig) -> KeyValueStore:
    common = {
        "num_layers": serving_model.num_layers,
        "num_key_value_heads": serving_model.num_key_value_heads,
        "head_dim": serving_model.head_dim,
        "device": serving_model.device,
        "dtype": serving_model.dtype,
    }
    if config.kv_manager == "reserved":
        return ReservedKeyValueStore(
            num_slots=config.max_batch_size, slot_length=serving_model.max_sequence_length, **common
        )
    num_blocks = config.num_blocks or paged_store_blocks(
        config.max_batch_size, config.block_size, serving_model.max_sequence_length
    )
    return PagedKeyValueStore(
        num_blocks=num_blocks,
        block_size=config.block_size,
        enable_prefix_caching=config.prefix_caching,
        **common,
    )


def build_engine(
    name: str, loaded: LoadedModel, config: EngineConfig, **engine_options
) -> ResidentEngine:
    """読み込み済みのモデルから、常駐のエンジンを作る(エンジンのスレッドで呼ぶ)。"""
    config.validate()
    serving_model = ServingModel(loaded.model)
    return ResidentEngine(
        name,
        serving_model,
        build_scheduler(config),
        build_store(serving_model, config),
        max_queue_length=config.max_queue_length,
        request_timeout_seconds=config.request_timeout_seconds,
        **engine_options,
    )


class ServerRuntime:
    """常駐サーバの実体(エンジンのスレッドと、その状態)。

    Args:
        loader: トークナイザとモデルを読み込む関数(エンジンのスレッドで呼ぶ)。
        config: モデルごとのエンジンの設定。
        shutdown_grace_seconds: 停止のとき、実行中の要求を終わらせるために待つ最長の時間 [秒]。
        model_keys: 提供するモデル。
        engine_options: ``ResidentEngine`` に渡す追加の引数(試験用)。
    """

    def __init__(
        self,
        loader: Callable[[], Resources],
        config: EngineConfig | None = None,
        *,
        shutdown_grace_seconds: float = 10.0,
        model_keys: tuple[str, ...] = SERVED_MODELS,
        engine_options: dict | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.validate()
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self.model_keys = model_keys
        self._loader = loader
        self._engine_options = engine_options or {}
        self._started_at = time.monotonic()
        self._state = STATE_LOADING
        self._error: str | None = None
        self._engines: dict[str, ResidentEngine] = {}
        self._tokenizer = None
        self._device_name = ""
        self._max_sequence_length = 0
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._signal = False
        self._stopping = False
        self._abort_at: float | None = None
        self._ready = threading.Event()

    # ------------------------------------------------------------------
    # どのスレッドからでも読める
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    def engine(self, key: str) -> ResidentEngine:
        return self._engines[key]

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """読み込みが終わる(または失敗する)まで待つ。準備ができていれば True。"""
        self._ready.wait(timeout)
        return self._state == STATE_READY

    def health(self) -> dict:
        return {
            "status": self._state,
            "error": self._error,
            "device": self._device_name or None,
            "models": list(self._engines) if self._engines else list(self.model_keys),
            "scheduler": self.config.scheduler,
            "kv_manager": self.config.kv_manager,
            "max_batch_size": self.config.max_batch_size,
            "uptime_seconds": time.monotonic() - self._started_at,
        }

    def stats(self) -> dict:
        return {
            "status": self._state,
            "uptime_seconds": time.monotonic() - self._started_at,
            "models": {key: engine.stats() for key, engine in self._engines.items()},
        }

    # ------------------------------------------------------------------
    # 起動と停止
    # ------------------------------------------------------------------
    def start(self) -> None:
        """エンジンのスレッドを起こす(読み込みはそのスレッドで始まる)。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._main, name="engine", daemon=True)
        self._thread.start()

    def begin_shutdown(self) -> None:
        """新しい要求の受け付けをやめ、未開始の要求を取り消す。実行中の要求は猶予まで続ける。"""
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            self._abort_at = time.monotonic() + self.shutdown_grace_seconds
            self._state = STATE_STOPPING if self._state == STATE_READY else self._state
        for engine in list(self._engines.values()):
            engine.begin_drain()
        self._notify()

    def shutdown(self) -> None:
        """停止して、エンジンのスレッドが終わるのを待つ。"""
        self.begin_shutdown()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.shutdown_grace_seconds + 10.0)

    def _notify(self) -> None:
        with self._condition:
            self._signal = True
            self._condition.notify()

    # ------------------------------------------------------------------
    # エンジンのスレッド
    # ------------------------------------------------------------------
    def _main(self) -> None:
        try:
            resources = self._loader()
            engines = {
                key: build_engine(key, resources.models[key], self.config, **self._engine_options)
                for key in self.model_keys
            }
            for engine in engines.values():
                engine.set_wake(self._notify)
            self._tokenizer = resources.tokenizer
            self._device_name = resources.device.type
            self._max_sequence_length = resources.models[self.model_keys[0]].max_sequence_length
            self._engines = engines
            with self._condition:
                if self._state == STATE_LOADING:
                    self._state = STATE_READY
                stopping = self._stopping
            if stopping:  # 読み込みの間に停止が要求された
                for engine in engines.values():
                    engine.begin_drain()
        except BaseException as error:  # 読み込みに失敗しても、サーバは健康確認に応答する
            self._error = f"{type(error).__name__}: {error}"
            traceback.print_exc()
            with self._condition:
                self._state = STATE_FAILED
            self._ready.set()
            return
        self._ready.set()
        self._loop()

    def _loop(self) -> None:
        engines = list(self._engines.values())
        aborted = False
        while True:
            progressed = False
            for engine in engines:
                try:
                    progressed |= engine.step()
                except Exception as error:  # 1 つのエンジンの故障で、他のエンジンを止めない
                    traceback.print_exc()
                    message = f"{type(error).__name__}: {error}"
                    engine.mark_failed(message)
                    try:
                        engine.abort_all(FINISH_ERROR, f"エンジンが故障しました: {message}")
                    except Exception:
                        traceback.print_exc()
            if self._stopping:
                if (
                    not aborted
                    and self._abort_at is not None
                    and time.monotonic() >= self._abort_at
                ):
                    aborted = True
                    for engine in engines:
                        engine.abort_all(
                            FINISH_SHUTDOWN, "停止の猶予が過ぎたため、取り消しました。"
                        )
                if not any(engine.has_work() for engine in engines):
                    return
            if not progressed:
                with self._condition:
                    if not self._signal:
                        self._condition.wait(IDLE_WAIT_SECONDS)
                    self._signal = False
