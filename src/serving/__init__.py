"""推論サーバ(continuous batching と PagedAttention)の実装(027)。

- ``requests``: 要求と要求ごとの状態。
- ``workload``: 合成の負荷(出力長の分布・到着過程・書き出し)。
- ``kv_store``: KV キャッシュの管理(最大長の分を前もって確保する方式と、
  PagedAttention のブロック管理)。
- ``runner``: 長さの違う要求をまとめて 1 回の forward で進める実行器。
- ``scheduler``: static batching と continuous batching。
- ``clock``: 時計と、1 反復の時間の費用モデル。
- ``engine``: 実行ループ(反復ごとのスケジューリング、退避)。
- ``metrics``: スループット・待ち時間・メモリの使用状況。
- ``reference``: 1 要求ずつの生成(正しさの基準)と数値の許容誤差。
"""

from src.serving.clock import IterationCostModel, ModelClock, WallClock, fit_cost_model
from src.serving.engine import IterationRecord, RequestRecord, RunResult, ServingEngine
from src.serving.kv_store import (
    BatchLayout,
    KeyValueStore,
    PagedKeyValueStore,
    ReservedKeyValueStore,
    Row,
    StoreUsage,
)
from src.serving.metrics import (
    latency_statistics,
    mean_active_requests,
    mean_computed_rows,
    memory_statistics,
    throughput,
    wasted_row_fraction,
)
from src.serving.reference import logits_tolerance, reference_generate
from src.serving.requests import Request, RequestState, RequestStatus
from src.serving.runner import DryRunner, ModelRunner, ServingModel
from src.serving.scheduler import ContinuousBatchScheduler, Scheduler, StaticBatchScheduler
from src.serving.workload import (
    OutputLengthDistribution,
    build_requests,
    expected_maximum,
    fit_output_length_distribution,
    maximum_coefficient_of_variation,
    moment_sampling_distribution,
    poisson_arrival_times,
    predicted_static_to_continuous_ratio,
    sample_output_lengths,
    sample_prompt_lengths,
)

__all__ = [
    "BatchLayout",
    "ContinuousBatchScheduler",
    "DryRunner",
    "IterationCostModel",
    "IterationRecord",
    "KeyValueStore",
    "ModelClock",
    "ModelRunner",
    "OutputLengthDistribution",
    "PagedKeyValueStore",
    "Request",
    "RequestRecord",
    "RequestState",
    "RequestStatus",
    "ReservedKeyValueStore",
    "Row",
    "RunResult",
    "Scheduler",
    "ServingEngine",
    "ServingModel",
    "StaticBatchScheduler",
    "StoreUsage",
    "WallClock",
    "build_requests",
    "expected_maximum",
    "fit_cost_model",
    "fit_output_length_distribution",
    "latency_statistics",
    "logits_tolerance",
    "maximum_coefficient_of_variation",
    "mean_active_requests",
    "mean_computed_rows",
    "memory_statistics",
    "moment_sampling_distribution",
    "poisson_arrival_times",
    "predicted_static_to_continuous_ratio",
    "reference_generate",
    "sample_output_lengths",
    "sample_prompt_lengths",
    "throughput",
    "wasted_row_fraction",
]
