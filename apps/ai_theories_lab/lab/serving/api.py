"""HTTP の API(FastAPI)。

端点(エンドポイント):

- ``POST /v1/generate``: 英語の書き出しの続きを生成する。``stream: true`` のときは、
  Server-Sent Events(``text/event-stream``)でトークンを 1 つずつ返す。
- ``GET /v1/health``: 起動と読み込みの状態(``loading`` / ``ready`` / ``stopping`` / ``failed``)。
- ``GET /v1/stats``: モデルごとの、待っている要求数・実行中の要求数・KV キャッシュの使用量・
  これまでのスループット。

HTTP の処理(別のスレッドではなく、イベントループ)は、要求を検査してエンジンの待ち行列に入れ、
エンジンのスレッドが生成したトークンを受け取って返すだけで、torch を呼ばない。トークンは
``asyncio.Queue`` にスレッドをまたいで渡す(``call_soon_threadsafe``)。キューに上限はなく、
遅いクライアントがエンジンを止めることはない(エンジンは、トークンを渡すだけで返る)。

クライアントが切断したら、要求を取り消して、ブロックを返す。逐次表示(Server-Sent Events)でも、
逐次でない応答でも同じ(0.25 秒ごとに切断を確かめる)。

誤りの応答は ``{"error": {"code": ..., "message": ...}}`` の形で、状態コードは次のとおり。

- 422: 要求の内容が誤っている(空の書き出し・長すぎる入力・不正な設定)。
- 429: 待ち行列が上限に達している。``Retry-After`` ヘッダと ``retry_after_seconds`` が再試行の目安。
- 503: 読み込み中・停止中・故障している。
- 504: 要求の時間の上限を超えた(逐次でない応答のとき。逐次表示では ``error`` イベント)。

導入: 段階 2-1(システムの仕組み。使った理論は 027・010・008)。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from lab.serving.contract import (
    DEFAULT_MAX_TOKENS,
    MAX_PROMPT_CHARACTERS,
    IncrementalDecoder,
    InvalidRequest,
    prepare_request,
)
from lab.serving.engine import (
    FINISH_CANCELLED,
    FINISH_ERROR,
    FINISH_LENGTH,
    FINISH_SHUTDOWN,
    FINISH_TIMEOUT,
    EngineClosed,
    QueueFull,
    RequestOutcome,
    RequestTooLarge,
    ResidentEngine,
)
from lab.serving.runtime import STATE_READY, ServerRuntime

DISCONNECT_POLL_SECONDS = 0.25
"""クライアントの切断を確かめる間隔 [秒]。"""
MAX_SEED = 2**32 - 1

FINISH_STATUS = {
    FINISH_TIMEOUT: (504, "timeout"),
    FINISH_SHUTDOWN: (503, "shutdown"),
    FINISH_ERROR: (500, "engine_error"),
    FINISH_CANCELLED: (499, "cancelled"),
}
"""``length`` 以外の終わり方を、逐次でない応答の状態コードと誤りの種類に対応させる。"""


class GenerateBody(BaseModel):
    """``POST /v1/generate`` の本文。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1, max_length=MAX_PROMPT_CHARACTERS, description="英語の書き出し"
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS,
        ge=1,
        le=256,
        description="生成するトークン数(終端記号がないので常にこの数)",
    )
    model: Literal["main", "gqa"] = Field("main", description="Hub のブランチ")
    method: Literal["greedy", "temperature", "top_k", "top_p"] = Field(
        "greedy", description="デコードの方法(008)"
    )
    temperature: float = Field(1.0, gt=0.0, le=10.0, description="temperature・top-k・top-p のとき")
    top_k: int = Field(40, ge=1, le=100_000, description="top-k のとき")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="top-p のとき")
    seed: int | None = Field(
        None, ge=0, le=MAX_SEED, description="乱数のシード。省略するとサーバが決め、応答に含める"
    )
    stream: bool = Field(False, description="true なら Server-Sent Events でトークンを逐次返す")


class ApiError(Exception):
    """誤りの応答(状態コード・種類・メッセージ)。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = dict(headers or {})
        self.extra = dict(extra or {})

    def response(self) -> JSONResponse:
        body = {"error": {"code": self.code, "message": self.message, **self.extra}}
        return JSONResponse(body, status_code=self.status_code, headers=self.headers)


class AsyncSink:
    """エンジンのスレッドから、イベントループの ``asyncio.Queue`` へトークンを渡す。"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def on_token(self, token_id: int, index: int) -> None:
        self._put(("token", (token_id, index)))

    def on_finish(self, outcome: RequestOutcome) -> None:
        self._put(("finish", outcome))

    def _put(self, item: tuple[str, Any]) -> None:
        with contextlib.suppress(RuntimeError):  # イベントループが閉じた(停止の途中)
            self._loop.call_soon_threadsafe(self.queue.put_nowait, item)


def server_sent_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def timings_of(outcome: RequestOutcome) -> dict:
    return {
        "queue_wait_seconds": outcome.queue_wait_seconds,
        "time_to_first_token_seconds": outcome.time_to_first_token_seconds,
        "time_per_output_token_seconds": outcome.time_per_output_token_seconds,
        "total_seconds": outcome.total_seconds,
    }


def usage_of(outcome: RequestOutcome) -> dict:
    return {
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
    }


def error_for(outcome: RequestOutcome) -> ApiError:
    status, code = FINISH_STATUS.get(outcome.finish_reason, (500, "engine_error"))
    return ApiError(status, code, outcome.message or "生成を完了できませんでした。")


def create_app(runtime: ServerRuntime) -> FastAPI:
    """``runtime`` を使う FastAPI のアプリを作る(起動でスレッドを起こし、終了で止める)。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.shutdown)

    app = FastAPI(
        title="AI Theories Lab 常駐サーバ",
        description=(
            "小型 GPT を常駐のエンジン(continuous batching・PagedAttention。027)で動かす "
            "HTTP の API。"
            "手元専用で、Hugging Face Spaces では動かない。"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        return error.response()

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"] if part != "body")
        message = f"要求の形式が正しくありません({where}): {first['msg']}"
        return ApiError(422, "invalid_request", message).response()

    def ensure_ready() -> None:
        if runtime.state != STATE_READY:
            message = {
                "loading": "モデルを読み込み中です。",
                "stopping": "サーバは停止中で、新しい要求を受け付けません。",
                "failed": f"起動に失敗しました: {runtime.error}",
            }.get(runtime.state, "サーバの準備ができていません。")
            raise ApiError(503, runtime.state, message)

    @app.get("/v1/health")
    async def health() -> JSONResponse:
        body = runtime.health()
        return JSONResponse(body, status_code=200 if runtime.state == STATE_READY else 503)

    @app.get("/v1/stats")
    async def stats() -> dict:
        return runtime.stats()

    @app.post("/v1/generate")
    async def generate(body: GenerateBody, request: Request) -> Response:
        ensure_ready()
        engine = runtime.engine(body.model)
        tokenizer = runtime.tokenizer
        try:
            prepared = await asyncio.to_thread(
                prepare_request,
                tokenizer.encode,
                runtime.max_sequence_length,
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                method=body.method,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                seed=body.seed,
            )
        except InvalidRequest as error:
            raise ApiError(422, "invalid_request", str(error)) from error

        sink = AsyncSink(asyncio.get_running_loop())
        try:
            request_id = engine.submit(prepared.prompt_token_ids, prepared.settings, sink)
        except QueueFull as error:
            raise ApiError(
                429,
                error.code,
                str(error),
                headers={"Retry-After": str(error.retry_after_seconds)},
                extra={"retry_after_seconds": error.retry_after_seconds},
            ) from error
        except EngineClosed as error:
            raise ApiError(503, error.code, str(error)) from error
        except RequestTooLarge as error:
            raise ApiError(422, error.code, str(error)) from error
        except ValueError as error:
            raise ApiError(422, "invalid_request", str(error)) from error

        meta = {
            "id": f"{body.model}-{request_id}",
            "model": body.model,
            "seed": prepared.settings.seed,
        }
        if body.stream:
            return StreamingResponse(
                stream_events(request, engine, request_id, sink, tokenizer.decode, meta),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await collect(request, engine, request_id, sink, tokenizer.decode, meta)

    return app


class DisconnectWatch:
    """クライアントの切断を、一定の間隔で確かめる(トークンが届き続けていても確かめる)。"""

    def __init__(self, request: Request) -> None:
        self._request = request
        self._checked = time.monotonic()

    async def lost(self) -> bool:
        now = time.monotonic()
        if now - self._checked < DISCONNECT_POLL_SECONDS:
            return False
        self._checked = now
        return await self._request.is_disconnected()


async def next_item(watch: DisconnectWatch, sink: AsyncSink) -> tuple[str, Any] | None:
    """次のトークンか終わりを待つ。クライアントが切断したら None。"""
    while True:
        if await watch.lost():
            return None
        try:
            return await asyncio.wait_for(sink.queue.get(), DISCONNECT_POLL_SECONDS)
        except TimeoutError:
            continue


async def collect(request, engine: ResidentEngine, request_id, sink, decode, meta) -> Response:
    """逐次でない応答: 終わるまで受け取って、1 つの JSON で返す。"""
    token_ids: list[int] = []
    finished = False
    watch = DisconnectWatch(request)
    try:
        while True:
            item = await next_item(watch, sink)
            if item is None:  # 切断された。応答は誰も読まないので、取り消して終える
                return Response(status_code=499)
            kind, value = item
            if kind == "token":
                token_ids.append(value[0])
                continue
            finished = True
            outcome: RequestOutcome = value
            if outcome.finish_reason != FINISH_LENGTH:
                raise error_for(outcome)
            return JSONResponse(
                {
                    **meta,
                    "text": decode(token_ids),
                    "token_ids": token_ids,
                    "finish_reason": outcome.finish_reason,
                    "usage": usage_of(outcome),
                    "timings": timings_of(outcome),
                    "preemptions": outcome.num_preemptions,
                }
            )
    finally:
        if not finished:
            engine.cancel(request_id)


async def stream_events(
    request, engine: ResidentEngine, request_id, sink, decode, meta
) -> AsyncIterator[bytes]:
    """逐次表示(Server-Sent Events): ``token`` を 1 つずつ、最後に ``done`` か ``error``。"""
    decoder = IncrementalDecoder(decode)
    finished = False
    watch = DisconnectWatch(request)
    try:
        while True:
            item = await next_item(watch, sink)
            if item is None:  # 切断された
                return
            kind, value = item
            if kind == "token":
                token_id, index = value
                text = decoder.push(token_id)
                yield server_sent_event(
                    "token", {"index": index, "token_id": token_id, "text": text}
                )
                continue
            finished = True
            outcome: RequestOutcome = value
            if outcome.finish_reason == FINISH_LENGTH:
                yield server_sent_event(
                    "done",
                    {
                        **meta,
                        "finish_reason": outcome.finish_reason,
                        "tail": decoder.finish(),
                        "usage": usage_of(outcome),
                        "timings": timings_of(outcome),
                        "preemptions": outcome.num_preemptions,
                    },
                )
            else:
                error = error_for(outcome)
                yield server_sent_event(
                    "error",
                    {
                        **meta,
                        "code": error.code,
                        "message": error.message,
                        "usage": usage_of(outcome),
                    },
                )
            return
    finally:
        if not finished:
            engine.cancel(request_id)
