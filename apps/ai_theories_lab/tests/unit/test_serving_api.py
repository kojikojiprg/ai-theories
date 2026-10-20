"""常駐サーバの HTTP の API の試験(本物の uvicorn を別のスレッドで起動し、HTTP の接続で呼ぶ)。

確かめること:

- 並行性: 同時に送った要求が、それぞれ自分の出力を受け取る(取り違えがない)。逐次表示で受け取った
  トークンを連結したものが、逐次でない応答と一致する。
- 運用の振る舞い: 待ち行列の上限(HTTP 429 と ``Retry-After``)・長すぎる入力の誤り・時間の上限・
  クライアントの切断による取り消し(ブロックがすべて戻る)・停止のときの扱い・読み込み中と故障の応答。

重みが乱数の代替アーティファクトで走るので、生成の中身に意味はない。生成したトークンは、1 要求ずつの
生成(``reference_generate``)と比べる。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from lab.core.artifacts import Resources
from lab.serving.runtime import EngineConfig
from tests.serving_helpers import assert_tokens_match_reference
from tests.serving_server import RunningServer, running_server

PROMPTS = [
    "The history of the city began when the river town grew into a market and a port.",
    "Albert Einstein was born in a small town, and the film was released in the spring.",
    "The unbelievable results were unexpectedly interesting for the tokenization of words.",
    "A short sentence.",
    "日本語の書き出しも、バイトに分けて扱える。",
    "The film was released in",
]


def generate(server: RunningServer, prompt: str, max_tokens: int = 8, **options) -> httpx.Response:
    with server.client(timeout=60) as client:
        return client.post(
            "/v1/generate", json={"prompt": prompt, "max_tokens": max_tokens, **options}
        )


def read_stream(
    server: RunningServer, prompt: str, max_tokens: int = 8, **options
) -> list[tuple[str, dict]]:
    """逐次表示の応答を、最後まで ``(イベント名, データ)`` の一覧にする。"""
    events: list[tuple[str, dict]] = []
    body = {"prompt": prompt, "max_tokens": max_tokens, "stream": True, **options}
    with server.client(timeout=60) as client, client.stream("POST", "/v1/generate", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events.extend(iter_events(r.iter_lines()))
    return events


def iter_events(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:
    name = None
    for line in lines:
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            yield name, json.loads(line.removeprefix("data: "))


@pytest.fixture(scope="module")
def server(resources: Resources) -> Iterator[RunningServer]:
    config = EngineConfig(max_batch_size=4, block_size=4)
    with running_server(resources, config) as running:
        yield running


# ---------------------------------------------------------------------------
# 基本の応答
# ---------------------------------------------------------------------------
def test_health_reports_the_state_and_the_device(server: RunningServer) -> None:
    with server.client(timeout=10) as client:
        response = client.get("/v1/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["models"] == ["main", "gqa"]
    assert body["scheduler"] == "continuous" and body["max_batch_size"] == 4
    assert body["device"] in {"cuda", "mps", "cpu"}


def test_a_non_streaming_response_has_the_text_the_usage_and_the_timings(
    server: RunningServer,
) -> None:
    response = generate(server, PROMPTS[0], 10, seed=3)
    body = response.json()
    assert response.status_code == 200
    assert body["finish_reason"] == "length"
    assert len(body["token_ids"]) == body["usage"]["completion_tokens"] == 10
    assert body["usage"]["prompt_tokens"] > 0 and body["seed"] == 3
    assert body["id"].startswith("main-")
    timings = body["timings"]
    assert 0 <= timings["queue_wait_seconds"] <= timings["time_to_first_token_seconds"]
    assert timings["time_to_first_token_seconds"] <= timings["total_seconds"]
    assert timings["time_per_output_token_seconds"] > 0


def test_the_server_picks_a_seed_when_none_is_given_and_reports_it(server: RunningServer) -> None:
    first = generate(server, PROMPTS[3], 6, method="top_p").json()
    again = generate(server, PROMPTS[3], 6, method="top_p", seed=first["seed"]).json()
    assert again["token_ids"] == first["token_ids"], "応答のシードで、同じ生成を再現できる"


@pytest.mark.parametrize("model", ["main", "gqa"])
def test_greedy_generation_matches_the_reference_for_both_models(
    resources: Resources, server: RunningServer, model: str
) -> None:
    loaded = resources.models[model]
    for prompt in PROMPTS[:3]:
        body = generate(server, prompt, 12, model=model).json()
        assert_tokens_match_reference(loaded, resources.tokenizer.encode(prompt), body["token_ids"])


# ---------------------------------------------------------------------------
# 並行性
# ---------------------------------------------------------------------------
def test_concurrent_requests_each_receive_their_own_output(
    resources: Resources, server: RunningServer
) -> None:
    """12 件を同時に送る(最大バッチサイズ 4 なので、実行中と待ちが混ざる)。書き出しも長さも違う。"""
    jobs = [(PROMPTS[i % len(PROMPTS)], 4 + (i * 3) % 11) for i in range(12)]
    barrier = threading.Barrier(len(jobs))

    def call(job: tuple[str, int]) -> dict:
        barrier.wait()
        response = generate(server, job[0], job[1])
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(call, jobs))
    loaded = resources.models["main"]
    for (prompt, max_tokens), body in zip(jobs, results, strict=True):
        assert len(body["token_ids"]) == max_tokens, "別の要求の出力を受け取った(長さが違う)"
        assert_tokens_match_reference(loaded, resources.tokenizer.encode(prompt), body["token_ids"])
    assert len({body["id"] for body in results}) == len(jobs)


def test_the_streamed_tokens_joined_equal_the_non_streaming_result(server: RunningServer) -> None:
    for prompt in (
        PROMPTS[0],
        PROMPTS[4],
    ):  # 英語と、マルチバイト文字(トークンが文字の途中で切れうる)
        plain = generate(server, prompt, 30).json()
        events = read_stream(server, prompt, 30)
        names = [name for name, _ in events]
        assert names == ["token"] * 30 + ["done"]
        tokens = [data for name, data in events if name == "token"]
        done = events[-1][1]
        assert [t["index"] for t in tokens] == list(range(30))
        assert [t["token_id"] for t in tokens] == plain["token_ids"]
        assert "".join(t["text"] for t in tokens) + done["tail"] == plain["text"]
        assert done["finish_reason"] == "length" and done["usage"]["completion_tokens"] == 30
        assert done["timings"]["time_to_first_token_seconds"] > 0


def test_concurrent_streams_do_not_mix_their_tokens(
    resources: Resources, server: RunningServer
) -> None:
    jobs = [(PROMPTS[i], 6 + 2 * i) for i in range(6)]
    barrier = threading.Barrier(len(jobs))

    def call(job: tuple[str, int]) -> list[tuple[str, dict]]:
        barrier.wait()
        return read_stream(server, job[0], job[1])

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(call, jobs))
    loaded = resources.models["main"]
    for (prompt, max_tokens), events in zip(jobs, results, strict=True):
        ids = [data["token_id"] for name, data in events if name == "token"]
        assert len(ids) == max_tokens and events[-1][0] == "done"
        assert_tokens_match_reference(loaded, resources.tokenizer.encode(prompt), ids)


def test_a_static_batching_server_also_gives_each_request_its_own_output(
    resources: Resources,
) -> None:
    config = EngineConfig(scheduler="static", max_batch_size=3, block_size=4)
    with running_server(resources, config) as static_server:
        jobs = [(PROMPTS[i], 3 + 4 * i) for i in range(5)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(
                pool.map(lambda job: generate(static_server, job[0], job[1]).json(), jobs)
            )
        for (prompt, max_tokens), body in zip(jobs, results, strict=True):
            assert len(body["token_ids"]) == max_tokens
            assert_tokens_match_reference(
                resources.models["main"], resources.tokenizer.encode(prompt), body["token_ids"]
            )
        stats = static_server.stats()
        assert stats["scheduler"] == "static" and stats["running"] == 0
        assert stats["kv_cache"]["units_in_use"] == 0


# ---------------------------------------------------------------------------
# 誤った要求
# ---------------------------------------------------------------------------
INVALID_BODIES = [
    pytest.param({"prompt": ""}, "invalid_request", id="empty-prompt"),
    pytest.param({"prompt": "   "}, "invalid_request", id="blank-prompt"),
    pytest.param({"prompt": "x" * 9000}, "invalid_request", id="too-many-characters"),
    pytest.param({"prompt": "word " * 400, "max_tokens": 8}, "invalid_request", id="long-prompt"),
    pytest.param(
        {"prompt": "The film", "max_tokens": 256}, "invalid_request", id="prompt-plus-output"
    ),
    pytest.param({"prompt": "The film", "max_tokens": 0}, "invalid_request", id="zero-tokens"),
    pytest.param({"prompt": "The film", "model": "nope"}, "invalid_request", id="unknown-model"),
    pytest.param({"prompt": "The film", "method": "beam"}, "invalid_request", id="beam-search"),
    pytest.param({"prompt": "The film", "top_p": 0}, "invalid_request", id="zero-p"),
    pytest.param({"prompt": "The film", "unknown_field": 1}, "invalid_request", id="extra-field"),
    pytest.param({"max_tokens": 4}, "invalid_request", id="missing-prompt"),
]


@pytest.mark.parametrize(("body", "code"), INVALID_BODIES)
def test_invalid_requests_are_refused_with_422_and_never_reach_the_engine(
    server: RunningServer, body: dict, code: str
) -> None:
    before = server.stats()["requests"]["accepted"]
    with server.client(timeout=10) as client:
        response = client.post("/v1/generate", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == code and error["message"]
    assert server.stats()["requests"]["accepted"] == before


def test_the_too_long_input_message_names_the_limit(
    resources: Resources, server: RunningServer
) -> None:
    response = generate(server, "word " * 400, 8)
    assert str(resources.models["main"].max_sequence_length) in response.json()["error"]["message"]


def test_a_non_json_body_is_a_422_too(server: RunningServer) -> None:
    with server.client(timeout=10) as client:
        response = client.post("/v1/generate", content=b"not json")
    assert response.status_code == 422 and "error" in response.json()


# ---------------------------------------------------------------------------
# 運用の振る舞い: 待ち行列の上限、時間の上限、切断
# ---------------------------------------------------------------------------
LONG = 200  # 試験の間、実行中のままにしておく長い生成
SLOW = {"on_iteration": lambda: time.sleep(0.01)}
"""1 反復を 10 ミリ秒以上にする(速い環境でも、生成が試験の途中で終わってしまわないように)。"""


def test_a_full_queue_returns_429_with_a_retry_hint_and_recovers(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=1, block_size=4, max_queue_length=1)
    with running_server(resources, config, engine_options=SLOW) as small:
        running = small.client(timeout=60)
        stream = running.stream(
            "POST", "/v1/generate", json={"prompt": PROMPTS[0], "max_tokens": LONG, "stream": True}
        )
        response = stream.__enter__()
        try:
            events = iter_events(response.iter_lines())  # 参照を残す(捨てると接続が閉じる)
            next(events)  # 最初のトークンが届いた = 実行中
            queued: list[httpx.Response] = []
            thread = threading.Thread(
                target=lambda: queued.append(generate(small, PROMPTS[1], 3)), daemon=True
            )
            thread.start()
            small.wait_for(lambda s: s["not_started"] == 1)  # 2 件目が待ち行列に入った

            refused = generate(small, PROMPTS[2], 3)
            assert refused.status_code == 429
            error = refused.json()["error"]
            assert error["code"] == "queue_full" and "上限" in error["message"]
            assert int(refused.headers["Retry-After"]) == error["retry_after_seconds"] >= 1
            assert small.stats()["requests"]["rejected"] == 1
        finally:
            stream.__exit__(None, None, None)  # 切断 = 実行中の要求を取り消す
            running.close()
        thread.join(30)  # 待っていた要求は、実行に入って終わる
        assert queued[0].status_code == 200
        small.wait_for(lambda s: s["running"] == 0 and s["waiting"] == 0)
        assert generate(small, PROMPTS[2], 3).status_code == 200, "空いたら受け付ける"


def test_disconnecting_a_stream_cancels_the_request_and_returns_its_blocks(
    resources: Resources,
) -> None:
    config = EngineConfig(max_batch_size=2, block_size=4)
    with running_server(resources, config, engine_options=SLOW) as local:
        with (
            local.client(timeout=60) as client,
            client.stream(
                "POST",
                "/v1/generate",
                json={"prompt": PROMPTS[0], "max_tokens": LONG, "stream": True},
            ) as response,
        ):
            lines = response.iter_lines()
            events = iter_events(lines)
            next(events)
            next(events)
            local.wait_for(lambda s: s["running"] == 1 and s["kv_cache"]["units_in_use"] > 0)
        stats = local.wait_for(lambda s: s["running"] == 0)
        assert stats["requests"]["cancelled"] == 1
        assert stats["kv_cache"]["units_in_use"] == 0, "切断した要求のブロックが戻っていない"
        assert stats["output_tokens"] < LONG, "切断の後も生成を続けた"
        engine = local.runtime.engine("main")
        engine.store.check_consistency()
        assert engine.store.num_free_blocks == engine.store.num_blocks


def test_disconnecting_a_non_streaming_request_cancels_it(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=2, block_size=4)
    with running_server(resources, config, engine_options=SLOW) as local:
        with pytest.raises(httpx.ReadTimeout):
            generate_with_timeout(local, PROMPTS[0], LONG, timeout=0.5)
        stats = local.wait_for(lambda s: s["running"] == 0 and s["requests"]["cancelled"] == 1)
        assert stats["kv_cache"]["units_in_use"] == 0


def generate_with_timeout(server: RunningServer, prompt: str, max_tokens: int, timeout: float):
    with server.client(timeout=timeout) as client:
        return client.post("/v1/generate", json={"prompt": prompt, "max_tokens": max_tokens})


def test_disconnecting_a_request_that_is_still_queued_cancels_it(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=1, block_size=4, max_queue_length=4)
    with running_server(resources, config, engine_options=SLOW) as local:
        holder = local.client(timeout=60)
        stream = holder.stream(
            "POST", "/v1/generate", json={"prompt": PROMPTS[0], "max_tokens": LONG, "stream": True}
        )
        response = stream.__enter__()
        try:
            events = iter_events(response.iter_lines())  # 参照を残す(捨てると接続が閉じる)
            next(events)
            with pytest.raises(
                httpx.ReadTimeout
            ):  # 実行中の要求が塞いでいるので、待つだけで返らない
                generate_with_timeout(local, PROMPTS[1], 3, timeout=0.5)
            stats = local.wait_for(lambda s: s["requests"]["cancelled"] == 1)
            assert stats["not_started"] == 0 and stats["running"] == 1
            assert stats["output_tokens"] > 0
        finally:
            stream.__exit__(None, None, None)
            holder.close()
        local.wait_for(lambda s: s["running"] == 0 and s["requests"]["cancelled"] == 2)
        assert local.stats()["kv_cache"]["units_in_use"] == 0


def test_a_request_over_the_time_limit_gets_504_or_an_error_event(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=2, block_size=4, request_timeout_seconds=0.05)
    with running_server(resources, config, engine_options=SLOW) as local:
        response = generate(local, PROMPTS[0], LONG)
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"
        events = read_stream(local, PROMPTS[0], LONG)
        assert events[-1][0] == "error" and events[-1][1]["code"] == "timeout"
        stats = local.wait_for(lambda s: s["running"] == 0)
        assert stats["requests"]["timed_out"] == 2
        assert stats["kv_cache"]["units_in_use"] == 0


# ---------------------------------------------------------------------------
# 起動と停止
# ---------------------------------------------------------------------------
def test_health_and_generate_answer_503_while_loading_then_serve(resources: Resources) -> None:
    release = threading.Event()

    def slow_loader() -> Resources:
        release.wait(30)
        return resources

    with running_server(None, loader=slow_loader, wait_ready=False) as loading:
        with loading.client(timeout=10) as client:
            health = client.get("/v1/health")
            refused = client.post("/v1/generate", json={"prompt": "The film", "max_tokens": 2})
        assert health.status_code == 503 and health.json()["status"] == "loading"
        assert refused.status_code == 503 and refused.json()["error"]["code"] == "loading"
        release.set()
        assert loading.runtime.wait_until_ready(30)
        assert generate(loading, "The film", 2).status_code == 200


def test_a_failed_load_is_reported_by_health_and_the_server_keeps_answering() -> None:
    def broken_loader() -> Resources:
        raise RuntimeError("読み込みに失敗した")

    with running_server(None, loader=broken_loader, wait_ready=False) as broken:
        broken.runtime.wait_until_ready(30)
        with broken.client(timeout=10) as client:
            health = client.get("/v1/health")
            refused = client.post("/v1/generate", json={"prompt": "The film", "max_tokens": 2})
        assert health.status_code == 503 and health.json()["status"] == "failed"
        assert "読み込みに失敗した" in health.json()["error"]
        assert refused.status_code == 503 and refused.json()["error"]["code"] == "failed"


def test_shutdown_finishes_running_requests_and_cancels_the_queued_ones(
    resources: Resources,
) -> None:
    config = EngineConfig(max_batch_size=1, block_size=4, max_queue_length=4)
    with running_server(resources, config, shutdown_grace_seconds=30, engine_options=SLOW) as local:
        results: dict[str, list[tuple[str, dict]]] = {}

        def stream(name: str, prompt: str, tokens: int) -> None:
            results[name] = read_stream(local, prompt, tokens)

        first = threading.Thread(target=stream, args=("running", PROMPTS[0], 40), daemon=True)
        first.start()
        local.wait_for(lambda s: s["running"] == 1)
        second = threading.Thread(target=stream, args=("queued", PROMPTS[1], 5), daemon=True)
        second.start()
        local.wait_for(lambda s: s["not_started"] == 1)

        local.runtime.begin_shutdown()
        refused = generate(local, PROMPTS[2], 3)
        assert refused.status_code == 503  # 停止中は新しい要求を受け付けない
        second.join(30)
        first.join(60)
        assert results["queued"][-1][0] == "error"
        assert results["queued"][-1][1]["code"] == "shutdown"
        assert results["running"][-1][0] == "done", "実行中の要求は、猶予の間に終わらせる"
        assert len([e for e in results["running"] if e[0] == "token"]) == 40
        assert local.runtime.engine("main").store.usage().units_in_use == 0


def test_shutdown_cancels_what_is_still_running_after_the_grace_period(
    resources: Resources,
) -> None:
    config = EngineConfig(max_batch_size=2, block_size=4)
    with running_server(
        resources, config, shutdown_grace_seconds=0.3, engine_options=SLOW
    ) as local:
        results: list[list[tuple[str, dict]]] = []
        thread = threading.Thread(
            target=lambda: results.append(read_stream(local, PROMPTS[0], LONG)), daemon=True
        )
        thread.start()
        local.wait_for(lambda s: s["running"] == 1)
        local.runtime.begin_shutdown()
        thread.join(30)
        assert results[0][-1][0] == "error" and results[0][-1][1]["code"] == "shutdown"
        deadline = time.monotonic() + 10
        while local.runtime.engine("main").has_work():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert local.runtime.engine("main").store.usage().units_in_use == 0


def test_stats_report_the_queue_the_cache_and_the_counters(server: RunningServer) -> None:
    before = server.stats()
    generate(server, PROMPTS[0], 7)
    after = server.wait_for(lambda s: s["running"] == 0)
    assert after["output_tokens"] - before["output_tokens"] == 7
    assert after["requests"]["completed"] - before["requests"]["completed"] == 1
    assert after["iterations"] > before["iterations"]
    assert after["kv_cache"]["capacity_tokens"] > 0 and after["kv_cache"]["unit"] == "block"
    assert after["tokens_per_second"] > 0 and after["max_queue_length"] == 64
    assert {"waiting", "running", "not_started", "preemptions", "busy_seconds"} <= set(after)
    with server.client(timeout=10) as client:
        body = client.get("/v1/stats").json()
    assert set(body["models"]) == {"main", "gqa"} and body["status"] == "ready"


def test_the_api_documentation_is_served(server: RunningServer) -> None:
    with server.client(timeout=10) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
    assert {"/v1/generate", "/v1/health", "/v1/stats"} <= set(schema["paths"])


def test_an_engine_failure_ends_its_requests_and_leaves_the_other_model_running(
    resources: Resources,
) -> None:
    """1 つのモデルのエンジンの反復で予期しない例外が起きたら、そのモデルの要求は
    ``engine_error`` で終わり、以後そのモデルへの要求は 503。もう 1 つのモデルは動き続ける。
    """
    armed = threading.Event()

    def break_once() -> None:
        if armed.is_set():
            armed.clear()
            raise RuntimeError("試験のための故障")

    config = EngineConfig(max_batch_size=2, block_size=4)
    with running_server(resources, config, engine_options={"on_iteration": break_once}) as local:
        assert generate(local, PROMPTS[0], 3).status_code == 200
        armed.set()
        failed = generate(local, PROMPTS[0], 30)
        assert failed.status_code == 500
        assert failed.json()["error"]["code"] == "engine_error"
        refused = generate(local, PROMPTS[0], 3)
        assert refused.status_code == 503 and refused.json()["error"]["code"] == "unavailable"
        stats = local.stats()
        assert stats["state"] == "failed" and stats["requests"]["failed"] == 1
        assert stats["kv_cache"]["units_in_use"] == 0
        other = generate(local, PROMPTS[0], 3, model="gqa")
        assert other.status_code == 200, "もう 1 つのモデルのエンジンは動き続ける"
