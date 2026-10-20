"""常駐サーバの試験用に、本物の uvicorn を別のスレッドで起動する道具(本物の HTTP の接続を使う)。"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
import uvicorn

from lab.core.artifacts import Resources
from lab.serving.api import create_app
from lab.serving.runtime import EngineConfig, ServerRuntime
from tests.helpers import free_port


@dataclass
class RunningServer:
    url: str
    runtime: ServerRuntime
    server: uvicorn.Server

    def client(self, **options) -> httpx.Client:
        return httpx.Client(base_url=self.url, **options)

    def stats(self, model: str = "main") -> dict:
        with self.client(timeout=10) as client:
            return client.get("/v1/stats").json()["models"][model]

    def wait_for(self, predicate, timeout: float = 15.0, model: str = "main") -> dict:
        """統計が条件を満たすまで待つ(満たさなければ失敗)。条件を満たした統計を返す。"""
        deadline = time.monotonic() + timeout
        stats = self.stats(model)
        while not predicate(stats):
            assert time.monotonic() < deadline, f"統計が条件を満たさない: {stats}"
            time.sleep(0.02)
            stats = self.stats(model)
        return stats


@contextmanager
def running_server(
    resources: Resources | None,
    config: EngineConfig | None = None,
    *,
    loader=None,
    wait_ready: bool = True,
    **runtime_options,
) -> Iterator[RunningServer]:
    """``resources``(または ``loader``)で ``ServerRuntime`` を作り、uvicorn で起動する。"""
    runtime = ServerRuntime(
        loader or (lambda: resources), config or EngineConfig(), **runtime_options
    )
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(runtime), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn が起動しない"
        time.sleep(0.01)
    if wait_ready:
        assert runtime.wait_until_ready(30), "モデルの読み込みが終わらない"
    try:
        yield RunningServer(f"http://127.0.0.1:{port}", runtime, server)
    finally:
        runtime.begin_shutdown()
        server.should_exit = True
        thread.join(30)
        assert not thread.is_alive(), "uvicorn が止まらない"
