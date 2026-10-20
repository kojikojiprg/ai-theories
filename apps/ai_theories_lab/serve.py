"""常駐サーバ(HTTP の API)の入口。**手元専用** で、Hugging Face Spaces では動かない。

画面(``app.py``)とは別のプロセスで動く。Spaces は ``app.py`` だけを起動するので、公開しても
このサーバは動かない。常駐のエンジンが複数の要求を同時に抱える continuous batching は、要求ごとに
別プロセスで GPU を割り当てる ZeroGPU とは合わないためである。

起動: このディレクトリで ``uv run python serve.py``(既定は ``http://127.0.0.1:8000``)。
static batching で起動して、同じ負荷を掛けて比べるには ``--scheduler static``。
負荷を掛ける道具は ``uv run python -m lab.serving.loadgen``。

停止(Ctrl-C): 新しい要求の受け付けをやめ、まだ実行に入っていない要求を取り消す。実行中の要求は
``--shutdown-grace`` 秒まで生成を続けて終わらせ、それでも終わらない要求は取り消す(ブロックを返す)。
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence

import uvicorn

from lab.core.artifacts import load_resources
from lab.serving.api import create_app
from lab.serving.runtime import KV_MANAGERS, SCHEDULERS, EngineConfig, ServerRuntime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = EngineConfig()
    parser = argparse.ArgumentParser(description="AI Theories Lab の常駐サーバ(手元専用)")
    parser.add_argument("--host", default="127.0.0.1", help="待ち受けるアドレス(既定は手元だけ)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULERS,
        default=defaults.scheduler,
        help="continuous batching か static batching",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=defaults.max_batch_size,
        help="同時に実行する要求の数の上限(既定の 16 は 027 の 7.10 節による)",
    )
    parser.add_argument(
        "--kv-manager",
        choices=KV_MANAGERS,
        default=defaults.kv_manager,
        help="paged(PagedAttention)か reserved(最大長の分を確保)",
    )
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument(
        "--kv-blocks",
        type=int,
        default=None,
        help="物理ブロックの数(省略すると退避が起きない数。小さくすると退避が起きる)",
    )
    parser.add_argument(
        "--no-prefix-caching", action="store_true", help="接頭辞の共有を使わない(paged のとき)"
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=defaults.max_queue_length,
        help="まだ実行に入っていない要求の数の上限(超えると 429)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=defaults.request_timeout_seconds,
        help="1 件の要求の時間の上限 [秒](到着から数える。超えると取り消す)",
    )
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=10.0,
        help="停止のとき、実行中の要求を終わらせるために待つ最長の時間 [秒]",
    )
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def config_from(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        scheduler=args.scheduler,
        max_batch_size=args.max_batch_size,
        kv_manager=args.kv_manager,
        block_size=args.block_size,
        num_blocks=args.kv_blocks,
        prefix_caching=not args.no_prefix_caching,
        max_queue_length=args.max_queue,
        request_timeout_seconds=args.request_timeout,
    )


class DrainingServer(uvicorn.Server):
    """最初の停止の合図で、エンジンの受け付けをやめ、未開始の要求を取り消す。

    uvicorn は、接続中の要求が終わるまで待ってから、アプリの終了処理を呼ぶ。その待ちの間に
    実行中の要求を終わらせられるよう、合図の時点でエンジンの停止の手順(猶予つき)を始める。
    """

    def __init__(self, config: uvicorn.Config, runtime: ServerRuntime) -> None:
        super().__init__(config)
        self._runtime = runtime

    def handle_exit(self, sig, frame) -> None:
        if not self.should_exit:
            # シグナルの処理の中ではロックを取らず、別のスレッドで始める
            threading.Thread(target=self._runtime.begin_shutdown, daemon=True).start()
        super().handle_exit(sig, frame)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from(args)
    runtime = ServerRuntime(load_resources, config, shutdown_grace_seconds=args.shutdown_grace)
    app = create_app(runtime)
    print(
        f"常駐サーバ: {config.scheduler} batching(最大バッチサイズ {config.max_batch_size})、"
        f"KV キャッシュ {config.kv_manager}、待ち行列の上限 {config.max_queue_length}、"
        f"要求の時間の上限 {config.request_timeout_seconds:g} 秒"
    )
    server = DrainingServer(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            timeout_graceful_shutdown=int(args.shutdown_grace) + 5,
        ),
        runtime,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
