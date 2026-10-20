"""API に同時に要求を送り、スループットと待ち時間を集計する道具(負荷を掛けるクライアント)。

``uv run python -m lab.serving.loadgen`` で動く。サーバ(``serve.py``)を別に起動しておき、本物の
HTTP の同時要求を送って測る。static batching と continuous batching を比べるには、サーバを
``--scheduler static`` と ``--scheduler continuous`` で順に起動し、**同じ引数(同じシード)** で
この道具を走らせる(要求の内容・出力長・到着時刻は、シードから決まる)。

- ``--concurrency C``: 閉じた負荷。C 個のクライアントが、前の要求が終わると次の要求を送る。
- ``--rate R``: 開いた負荷。到着率 R(要求 / 秒)の Poisson 過程で要求を送る(終わりを待たない)。
  ``--concurrency`` を同時に指定すると、同時に抱える要求の数の上限になる。
- 出力長は、027 と同じく平均と変動係数を指定した分布(``src/serving/workload.py``)から決める。

測る量(逐次表示で受け取る場合):

- スループット = 生成したトークンの総数 / 最初の送信から最後の完了までの時間(クライアントの実時間)。
- 最初のトークンまでの時間(Time To First Token)= 送信から最初の ``token`` イベントまで。
- 出力トークンあたりの時間(Time Per Output Token)= 最初のトークンの後の 1 トークンあたりの時間。
- 待ち行列の待ち時間 = サーバが測った、到着から最初に実行に入るまで(``done`` イベントの値)。

導入: 段階 2-1(システムの仕組み。使った理論は 027 の指標の定義)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import httpx
import numpy as np

from lab.core.passages import PASSAGE_TEXT
from src.serving import fit_output_length_distribution, poisson_arrival_times, sample_output_lengths

PASSAGE = PASSAGE_TEXT.split()
"""書き出しを切り出す英文(語に分けておく)。"""
PROMPT_WORDS = (8, 24)
"""書き出しの語数の範囲。"""
MAX_OUTPUT = 160
"""出力長の上限の既定(書き出しが最大の長さのときも、最大の文脈長 256 に収まる)。"""


@dataclass(frozen=True)
class LoadSpec:
    """負荷の条件。同じ条件(シード込み)なら、要求の内容・出力長・到着時刻が同じになる。"""

    url: str = "http://127.0.0.1:8000"
    num_requests: int = 64
    concurrency: int | None = None
    rate: float | None = None
    output_mean: float = 32.0
    output_cv: float = 0.5
    output_max: int = MAX_OUTPUT
    model: str = "main"
    method: str = "greedy"
    stream: bool = True
    seed: int = 0
    timeout_seconds: float = 120.0

    def validate(self) -> None:
        if self.num_requests < 1:
            raise ValueError("要求の数は 1 以上にしてください。")
        if self.rate is not None and self.rate <= 0:
            raise ValueError("到着率は正にしてください。")
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError("同時に送る要求の数は 1 以上にしてください。")
        if self.rate is None and self.concurrency is None:
            raise ValueError("--concurrency か --rate のどちらかを指定してください。")


@dataclass(frozen=True)
class PlannedRequest:
    index: int
    prompt: str
    max_tokens: int
    send_at: float  # 開始からの秒(開いた負荷のとき。閉じた負荷では 0)


def make_plan(spec: LoadSpec) -> list[PlannedRequest]:
    """要求の内容・出力長・到着時刻を、シードから決定的に作る。"""
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    distribution = fit_output_length_distribution(
        spec.output_mean, spec.output_cv, 1, spec.output_max
    )
    outputs = sample_output_lengths(distribution, spec.num_requests, rng)
    arrivals = (
        np.zeros(spec.num_requests)
        if spec.rate is None
        else poisson_arrival_times(
            spec.rate, spec.num_requests, np.random.default_rng([spec.seed, 7])
        )
    )
    plan = []
    for index in range(spec.num_requests):
        count = int(rng.integers(PROMPT_WORDS[0], PROMPT_WORDS[1] + 1))
        start = int(rng.integers(0, len(PASSAGE) - count))
        plan.append(
            PlannedRequest(
                index=index,
                prompt=" ".join(PASSAGE[start : start + count]),
                max_tokens=int(outputs[index]),
                send_at=float(arrivals[index]),
            )
        )
    return plan


@dataclass
class RequestResult:
    """1 件の要求の結果(時間は秒。クライアントの実時間)。"""

    index: int
    status: int | None = None
    error_code: str | None = None
    finish_reason: str | None = None
    completion_tokens: int = 0
    sent_at: float = 0.0
    finished_at: float = 0.0
    time_to_first_token: float | None = None
    time_per_output_token: float | None = None
    server_queue_wait: float | None = None
    server_time_to_first_token: float | None = None
    retry_after: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.finish_reason == "length"

    @property
    def latency(self) -> float:
        return self.finished_at - self.sent_at


async def send_request(
    client: httpx.AsyncClient, spec: LoadSpec, planned: PlannedRequest, origin: float
) -> RequestResult:
    result = RequestResult(planned.index, sent_at=time.perf_counter() - origin)
    body = {
        "prompt": planned.prompt,
        "max_tokens": planned.max_tokens,
        "model": spec.model,
        "method": spec.method,
        "seed": spec.seed * 100_003 + planned.index,
        "stream": spec.stream,
    }
    token_times: list[float] = []
    try:
        async with client.stream("POST", "/v1/generate", json=body) as response:
            result.status = response.status_code
            if response.status_code != 200:
                payload = json.loads((await response.aread()) or b"{}")
                error = payload.get("error", {})
                result.error_code = error.get("code")
                result.retry_after = error.get("retry_after_seconds")
            elif spec.stream:
                name = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data = json.loads(line.removeprefix("data: "))
                        now = time.perf_counter() - origin
                        if name == "token":
                            token_times.append(now)
                        elif name == "done":
                            result.finish_reason = data["finish_reason"]
                            result.completion_tokens = data["usage"]["completion_tokens"]
                            apply_server_timings(result, data["timings"])
                        elif name == "error":
                            result.status = None
                            result.error_code = data["code"]
                            result.completion_tokens = data["usage"]["completion_tokens"]
            else:
                data = json.loads(await response.aread())
                result.finish_reason = data["finish_reason"]
                result.completion_tokens = data["usage"]["completion_tokens"]
                apply_server_timings(result, data["timings"])
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        result.status = None
        result.error_code = type(error).__name__
    result.finished_at = time.perf_counter() - origin
    if token_times:
        result.time_to_first_token = token_times[0] - result.sent_at
        if len(token_times) >= 2:
            result.time_per_output_token = (token_times[-1] - token_times[0]) / (
                len(token_times) - 1
            )
    elif result.server_time_to_first_token is not None:
        result.time_to_first_token = result.server_time_to_first_token
    return result


def apply_server_timings(result: RequestResult, timings: dict) -> None:
    result.server_queue_wait = timings["queue_wait_seconds"]
    result.server_time_to_first_token = timings["time_to_first_token_seconds"]
    if not result.time_per_output_token:
        result.time_per_output_token = timings["time_per_output_token_seconds"]


async def run_load(spec: LoadSpec) -> tuple[list[RequestResult], float]:
    """負荷を掛け、要求ごとの結果と、全体の時間(最初の送信から最後の完了まで)を返す。"""
    plan = make_plan(spec)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(spec.timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(base_url=spec.url, timeout=timeout, limits=limits) as client:
        origin = time.perf_counter()
        results: list[RequestResult] = []
        if spec.rate is None:  # 閉じた負荷: C 個のクライアントが順に取る
            queue = iter(plan)

            async def worker() -> None:
                for planned in queue:
                    results.append(await send_request(client, spec, planned, origin))

            await asyncio.gather(*(worker() for _ in range(spec.concurrency or 1)))
        else:  # 開いた負荷: 到着時刻に送る
            gate = asyncio.Semaphore(spec.concurrency) if spec.concurrency else None

            async def arrive(planned: PlannedRequest) -> None:
                await asyncio.sleep(max(0.0, planned.send_at - (time.perf_counter() - origin)))
                if gate is None:
                    results.append(await send_request(client, spec, planned, origin))
                else:
                    async with gate:
                        results.append(await send_request(client, spec, planned, origin))

            await asyncio.gather(*(arrive(p) for p in plan))
        elapsed = time.perf_counter() - origin
    return sorted(results, key=lambda r: r.index), elapsed


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def summarize(results: Sequence[RequestResult], elapsed: float) -> dict:
    """結果を集計する。時間は秒。成功した要求だけで待ち時間を数える。"""
    ok = [r for r in results if r.ok]
    first = [r.time_to_first_token for r in ok if r.time_to_first_token is not None]
    per_token = [r.time_per_output_token for r in ok if r.time_per_output_token]
    wait = [r.server_queue_wait for r in ok if r.server_queue_wait is not None]
    latency = [r.latency for r in ok]
    tokens = sum(r.completion_tokens for r in results)
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            key = f"{r.status or 'error'}:{r.error_code}"
            errors[key] = errors.get(key, 0) + 1
    span = max((r.finished_at for r in results), default=0.0) - min(
        (r.sent_at for r in results), default=0.0
    )
    return {
        "requests": len(results),
        "succeeded": len(ok),
        "rejected_429": sum(1 for r in results if r.status == 429),
        "failed": len(results) - len(ok),
        "errors": errors,
        "elapsed_seconds": elapsed,
        "output_tokens": tokens,
        "throughput_tokens_per_second": tokens / span if span > 0 else None,
        "requests_per_second": len(ok) / span if span > 0 else None,
        "time_to_first_token": {
            "mean": float(np.mean(first)) if first else None,
            "p50": percentile(first, 50),
            "p95": percentile(first, 95),
            "p99": percentile(first, 99),
        },
        "time_per_output_token": {
            "mean": float(np.mean(per_token)) if per_token else None,
            "p50": percentile(per_token, 50),
            "p95": percentile(per_token, 95),
        },
        "queue_wait": {
            "mean": float(np.mean(wait)) if wait else None,
            "p95": percentile(wait, 95),
        },
        "latency": {
            "mean": float(np.mean(latency)) if latency else None,
            "p50": percentile(latency, 50),
            "p95": percentile(latency, 95),
        },
    }


def milliseconds(value: float | None) -> str:
    return "-" if value is None else f"{value * 1000:,.1f} ms"


def format_report(spec: LoadSpec, summary: dict, server: dict | None) -> str:
    load = (
        f"開いた負荷(到着率 {spec.rate:g} 要求/秒の Poisson 過程"
        + (f"、同時 {spec.concurrency} 件まで)" if spec.concurrency else ")")
        if spec.rate is not None
        else f"閉じた負荷(同時 {spec.concurrency} 件)"
    )
    lines = []
    if server is not None:
        lines.append(
            f"サーバ: {server['scheduler']} batching(最大バッチサイズ {server['max_batch_size']})、"
            f"KV キャッシュ {server['kv_manager']}、デバイス {server['device']}"
        )
    lines += [
        f"負荷: 要求 {spec.num_requests} 件、{load}、出力長 平均 {spec.output_mean:g}・"
        f"変動係数 {spec.output_cv:g}・上限 {spec.output_max}、モデル {spec.model}、"
        f"{spec.method}、{'逐次表示' if spec.stream else '逐次でない応答'}、シード {spec.seed}",
        f"成功 {summary['succeeded']} / {summary['requests']} 件"
        + (
            f"(待ち行列の上限で拒否 {summary['rejected_429']} 件)"
            if summary["rejected_429"]
            else ""
        )
        + (f"、失敗の内訳 {summary['errors']}" if summary["errors"] else ""),
    ]
    rate = summary["throughput_tokens_per_second"]
    lines.append(
        f"スループット: {rate:,.1f} トークン/秒({summary['output_tokens']} トークンを "
        f"{summary['elapsed_seconds']:.2f} 秒で)"
        if rate
        else "スループット: -"
    )
    first, per_token = summary["time_to_first_token"], summary["time_per_output_token"]
    wait, latency = summary["queue_wait"], summary["latency"]
    lines += [
        f"最初のトークンまでの時間: 平均 {milliseconds(first['mean'])}、"
        f"50% {milliseconds(first['p50'])}、95% {milliseconds(first['p95'])}、"
        f"99% {milliseconds(first['p99'])}",
        f"出力トークンあたりの時間: 平均 {milliseconds(per_token['mean'])}、"
        f"50% {milliseconds(per_token['p50'])}、95% {milliseconds(per_token['p95'])}",
        f"待ち行列の待ち時間(サーバの測定): 平均 {milliseconds(wait['mean'])}、"
        f"95% {milliseconds(wait['p95'])}",
        f"要求の全体の時間: 平均 {milliseconds(latency['mean'])}、"
        f"50% {milliseconds(latency['p50'])}、95% {milliseconds(latency['p95'])}",
    ]
    return "\n".join(lines)


def fetch(url: str, path: str) -> dict | None:
    try:
        return httpx.get(url + path, timeout=10.0).json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = LoadSpec()
    parser = argparse.ArgumentParser(
        description="API に同時に要求を送り、スループットと待ち時間を集計する"
    )
    parser.add_argument("--url", default=defaults.url, help="サーバの URL")
    parser.add_argument("--requests", type=int, default=defaults.num_requests)
    parser.add_argument(
        "--concurrency", type=int, default=None, help="同時に送る要求の数(閉じた負荷)"
    )
    parser.add_argument("--rate", type=float, default=None, help="到着率 [要求/秒](開いた負荷)")
    parser.add_argument("--output-mean", type=float, default=defaults.output_mean)
    parser.add_argument("--output-cv", type=float, default=defaults.output_cv, help="変動係数")
    parser.add_argument("--output-max", type=int, default=defaults.output_max)
    parser.add_argument("--model", choices=("main", "gqa"), default=defaults.model)
    parser.add_argument(
        "--method", choices=("greedy", "temperature", "top_k", "top_p"), default=defaults.method
    )
    parser.add_argument("--no-stream", action="store_true", help="逐次でない応答で受け取る")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--timeout", type=float, default=defaults.timeout_seconds)
    parser.add_argument("--json", default=None, help="集計をこのファイルに JSON で書く")
    args = parser.parse_args(argv)
    if args.rate is None and args.concurrency is None:
        args.concurrency = 16
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = LoadSpec(
        url=args.url.rstrip("/"),
        num_requests=args.requests,
        concurrency=args.concurrency,
        rate=args.rate,
        output_mean=args.output_mean,
        output_cv=args.output_cv,
        output_max=args.output_max,
        model=args.model,
        method=args.method,
        stream=not args.no_stream,
        seed=args.seed,
        timeout_seconds=args.timeout,
    )
    health = fetch(spec.url, "/v1/health")
    if health is None or health.get("status") != "ready":
        print(f"サーバ {spec.url} の準備ができていません: {health}")
        return 2
    before = fetch(spec.url, "/v1/stats")
    results, elapsed = asyncio.run(run_load(spec))
    after = fetch(spec.url, "/v1/stats")
    summary = summarize(results, elapsed)
    print(format_report(spec, summary, health))
    if before and after:
        b, a = before["models"][spec.model], after["models"][spec.model]
        iterations = a["iterations"] - b["iterations"]
        busy = a["busy_seconds"] - b["busy_seconds"]
        per_iteration = f"{busy / iterations * 1000:.2f} ms" if iterations else "-"
        print(
            f"サーバの統計(この負荷の前後の差): 反復 {iterations} 回、"
            f"1 反復の時間 平均 {per_iteration}、"
            f"退避 {a['preemptions'] - b['preemptions']} 回、"
            f"生成したトークン {a['output_tokens'] - b['output_tokens']}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(
                {"spec": asdict(spec), "server": health, "summary": summary},
                file,
                ensure_ascii=False,
                indent=2,
            )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
