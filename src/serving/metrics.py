"""実行結果(``RunResult``)から指標を計算する関数。

推論サーバの指標は、スループット(単位時間あたりに生成するトークンの数)と、要求ごとの待ち時間の
2 つの系統に分かれる。

- スループット = 生成したトークンの総数 / 最初の到着から最後の完了までの時間。
- 最初のトークンまでの時間(Time To First Token)= 到着から最初のトークンの生成まで。待ち行列の待ちと
  prefill の時間を含む。
- 出力トークンあたりの時間(Time Per Output Token)= 最初のトークンの後の、1 トークンあたりの時間。
  decode の 1 反復の時間と、同じ反復に prefill が割り込む分で決まる。
- 待ち行列の待ち時間 = 到着から最初に実行に入るまで。

導入: 027。
"""

from __future__ import annotations

import numpy as np

from src.serving.engine import RunResult


def throughput(result: RunResult) -> float:
    """生成したトークンの総数を、最初の到着から最後の完了までの時間で割った値 [トークン / 秒]。"""
    return result.total_output_tokens / result.makespan


def mean_active_requests(result: RunResult) -> float:
    """反復あたりの実行中の要求数(その反復でトークンを生成した要求の数)の平均。"""
    return float(np.mean([it.active_requests for it in result.iterations]))


def mean_computed_rows(result: RunResult) -> float:
    """反復あたりの計算した行数の平均(prefill と decode の行の合計)。

    終わった行を残して計算する方式では、その分を含む。
    """
    return float(np.mean([it.prefill_rows + it.decode_rows for it in result.iterations]))


def wasted_row_fraction(result: RunResult) -> float:
    """計算した行のうち、出力に使われなかった行の割合。

    static batching で、終わった行を計算し続けた分にあたる。
    """
    computed = sum(it.prefill_rows + it.decode_rows for it in result.iterations)
    return 1.0 - sum(it.active_requests for it in result.iterations) / computed


def latency_statistics(result: RunResult) -> dict[str, float]:
    """待ち時間の指標(秒)。

    Returns:
        ``queue_wait_mean``、``time_to_first_token_mean``・``_p50``・``_p95``、
        ``time_per_output_token_mean``(出力長 1 の要求は除く)。
    """
    wait = np.array([r.queue_wait for r in result.requests])
    first = np.array([r.time_to_first_token for r in result.requests])
    per_token = np.array(
        [r.time_per_output_token for r in result.requests if r.time_per_output_token is not None]
    )
    return {
        "queue_wait_mean": float(wait.mean()),
        "time_to_first_token_mean": float(first.mean()),
        "time_to_first_token_p50": float(np.percentile(first, 50)),
        "time_to_first_token_p95": float(np.percentile(first, 95)),
        "time_per_output_token_mean": float(per_token.mean()) if per_token.size else float("nan"),
    }


def memory_statistics(result: RunResult) -> dict[str, float]:
    """メモリの使用状況の指標(反復の平均)。

    Returns:
        ``tokens_used_mean``(実際に使っている枠)、``tokens_allocated_mean``(割り当てた枠)、
        ``utilization``(使っている枠の総和 / 割り当てた枠の総和)、
        ``wasted_tokens_mean``(割り当てたが使っていない枠の平均 = 内部の断片化)。
    """
    used = np.array([it.tokens_used for it in result.iterations], dtype=np.float64)
    allocated = np.array([it.tokens_allocated for it in result.iterations], dtype=np.float64)
    return {
        "tokens_used_mean": float(used.mean()),
        "tokens_allocated_mean": float(allocated.mean()),
        "utilization": float(used.sum() / allocated.sum()),
        "wasted_tokens_mean": float((allocated - used).mean()),
    }
