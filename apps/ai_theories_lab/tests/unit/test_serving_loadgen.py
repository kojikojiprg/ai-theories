"""負荷を掛ける道具(``lab/serving/loadgen.py``)の試験: 本物のサーバに本物の HTTP の同時要求を送る。

小さな負荷だけを流す(スループットの数値は、重みが乱数の代替アーティファクトでの値で、意味はない)。
確かめるのは、同じ条件なら同じ負荷になること、集計が要求ごとの結果と合うこと、
開いた負荷・閉じた負荷・逐次でない応答・待ち行列の上限での拒否が動くこと。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lab.core.artifacts import Resources
from lab.serving.loadgen import LoadSpec, main, make_plan, run_load, summarize
from lab.serving.runtime import EngineConfig
from tests.serving_server import RunningServer, running_server

SMALL = {"num_requests": 10, "output_mean": 6.0, "output_cv": 0.5, "output_max": 16}


@pytest.fixture(scope="module")
def server(resources: Resources) -> RunningServer:
    with running_server(resources, EngineConfig(max_batch_size=4, block_size=4)) as running:
        yield running


def test_the_same_conditions_give_the_same_load_and_a_different_seed_does_not() -> None:
    spec = LoadSpec(concurrency=4, rate=5.0, **SMALL)
    again = make_plan(spec)
    assert make_plan(spec) == again
    assert make_plan(LoadSpec(concurrency=4, rate=5.0, seed=1, **SMALL)) != again
    assert [p.send_at for p in again] == sorted(p.send_at for p in again)
    assert all(1 <= p.max_tokens <= SMALL["output_max"] for p in again)
    closed = make_plan(LoadSpec(concurrency=4, **SMALL))
    assert {p.send_at for p in closed} == {0.0}
    assert [p.prompt for p in closed] == [p.prompt for p in again]  # 内容は到着の仕方によらない


def test_a_spec_without_a_load_model_is_refused() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        make_plan(LoadSpec(**SMALL))
    with pytest.raises(ValueError, match="到着率"):
        make_plan(LoadSpec(rate=-1.0, **SMALL))


@pytest.mark.parametrize("stream", [True, False])
def test_a_closed_load_completes_every_request_and_the_summary_matches(
    server: RunningServer, stream: bool
) -> None:
    spec = LoadSpec(url=server.url, concurrency=4, stream=stream, **SMALL)
    results, elapsed = asyncio.run(run_load(spec))
    plan = make_plan(spec)
    assert [r.index for r in results] == list(range(spec.num_requests))
    assert all(r.ok for r in results)
    assert [r.completion_tokens for r in results] == [p.max_tokens for p in plan]
    summary = summarize(results, elapsed)
    assert summary["succeeded"] == summary["requests"] == spec.num_requests
    assert summary["output_tokens"] == sum(p.max_tokens for p in plan)
    assert summary["throughput_tokens_per_second"] > 0
    assert 0 < summary["time_to_first_token"]["mean"] <= summary["latency"]["mean"]
    assert summary["time_to_first_token"]["p50"] <= summary["time_to_first_token"]["p95"]
    assert summary["queue_wait"]["mean"] is not None
    assert not summary["errors"]


def test_an_open_load_sends_each_request_at_its_arrival_time(server: RunningServer) -> None:
    spec = LoadSpec(url=server.url, rate=40.0, **SMALL)
    results, _ = asyncio.run(run_load(spec))
    plan = make_plan(spec)
    assert all(r.ok for r in results)
    for result, planned in zip(results, plan, strict=True):
        assert result.sent_at >= planned.send_at - 0.01, "到着時刻より早く送った"


def test_requests_refused_by_a_full_queue_are_counted(resources: Resources) -> None:
    config = EngineConfig(max_batch_size=1, block_size=4, max_queue_length=2)
    with running_server(
        resources, config, engine_options={"on_iteration": lambda: __import__("time").sleep(0.01)}
    ) as small:
        spec = LoadSpec(url=small.url, rate=500.0, **SMALL)  # ほぼ同時に 10 件が届く
        results, elapsed = asyncio.run(run_load(spec))
        summary = summarize(results, elapsed)
        assert summary["rejected_429"] > 0
        assert summary["succeeded"] + summary["failed"] == spec.num_requests
        assert summary["errors"].get("429:queue_full") == summary["rejected_429"]
        assert small.stats()["requests"]["rejected"] == summary["rejected_429"]
        assert all(r.retry_after >= 1 for r in results if r.status == 429)


def test_the_command_line_prints_a_report_and_writes_json(
    server: RunningServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.json"
    code = main(
        [
            "--url", server.url, "--requests", "6", "--concurrency", "3", "--output-mean", "5",
            "--output-max", "12", "--json", str(output),
        ]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0
    for expected in (
        "continuous batching",
        "スループット",
        "最初のトークンまでの時間",
        "サーバの統計",
    ):
        assert expected in text
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["succeeded"] == 6 and saved["server"]["status"] == "ready"


def test_the_command_line_stops_when_the_server_is_not_ready(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--url", "http://127.0.0.1:9", "--requests", "1"]) == 2
    assert "準備ができていません" in capsys.readouterr().out
