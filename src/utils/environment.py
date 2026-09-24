"""実行環境の記録(014)。

ノートブックのセットアップセルで呼び、実行環境(GPU 名、torch・CUDA・cuDNN のバージョン、
リポジトリのコミットなど)をセル出力に残すための関数を提供する。結果・考察の本文で実行環境を
記述するときに、実行者の記録ではなくこのセル出力を出典にできるようにする。

各項目の値は次のいずれかである。

- 取得できた値(文字列・数値・bool)
- ``None``: その環境では該当しない項目(CPU での GPU 名、CUDA 非対応ビルドでの CUDA のバージョンなど)
- ``"取得できない (<例外の型名>)"``: 取得を試みて失敗した項目。例外は送出しない

個人を特定しうる情報(認証情報・環境変数の値・ユーザー名・ホスト名・ホームディレクトリのパス)は
取得しない。取得に失敗したときも、例外のメッセージ(パスを含みうる)は記録せず型名のみを残す。

使い方 / Usage::

    from src.utils.environment import print_execution_environment

    execution_environment = print_execution_environment()
"""

from __future__ import annotations

import datetime
import platform
import subprocess
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

# リポジトリルート(このファイルは src/utils/ 配下にある)。git の呼び出しはここをカレントに行う
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GIT_TIMEOUT_SECONDS = 10.0
_UNAVAILABLE_PREFIX = "取得できない"


def _try(getter: Callable[[], Any]) -> Any:
    """``getter()`` を呼び、失敗したときは例外を送出せず「取得できない」と記録する。"""
    try:
        return getter()
    except Exception as error:  # noqa: BLE001 — 環境情報の取得でセットアップセルを止めない
        return f"{_UNAVAILABLE_PREFIX} ({type(error).__name__})"


def _run_git(*arguments: str) -> str:
    """リポジトリルートをカレントディレクトリとして git を実行し、標準出力を返す。"""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=True,
    )
    return completed.stdout


def _select_device_type() -> str:
    """ノートブックと同じ優先順位(CUDA → MPS → CPU)でデバイスの種類を決める。"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_execution_environment(device: Any = None) -> dict[str, Any]:
    """実行環境を辞書として返す。どの項目も、取得に失敗しても例外を送出しない。

    Args:
        device: ノートブックで使用するデバイス(``torch.device`` または文字列)。
            ``None`` の場合は CUDA → MPS → CPU の優先順位で利用可能なものを選ぶ。

    Returns:
        次のキーを持つ辞書。

        - ``python_version``・``platform``: Python のバージョン、OS
        - ``torch_version``・``torch_cuda_version``・``cudnn_version``: torch のバージョン、
          torch がビルドされた CUDA のバージョン、cuDNN のバージョン
        - ``device_type``: 使用するデバイスの種類(``"cuda"``・``"mps"``・``"cpu"``)
        - ``gpu_name``・``gpu_compute_capability``・``gpu_total_memory_gib``: CUDA の場合のみ。
          それ以外は ``None``
        - ``git_commit``・``git_has_uncommitted_changes``: リポジトリの HEAD のコミットのハッシュ、
          未コミットの変更(未追跡のファイルを含む)があるか
        - ``executed_at_utc``: 実行日時(UTC、ISO 8601)
    """
    environment: dict[str, Any] = {}
    environment["python_version"] = _try(platform.python_version)
    # platform.platform() は OS・カーネル・アーキテクチャのみを含み、ホスト名は含まない
    environment["platform"] = _try(platform.platform)

    def torch_version() -> str:
        import torch

        return torch.__version__

    def torch_cuda_version() -> str | None:
        import torch

        return torch.version.cuda

    def cudnn_version() -> int | None:
        import torch

        return torch.backends.cudnn.version()

    environment["torch_version"] = _try(torch_version)
    environment["torch_cuda_version"] = _try(torch_cuda_version)
    environment["cudnn_version"] = _try(cudnn_version)

    def device_type() -> str:
        if device is None:
            return _select_device_type()
        return getattr(device, "type", None) or str(device).split(":")[0]

    environment["device_type"] = _try(device_type)

    environment["gpu_name"] = None
    environment["gpu_compute_capability"] = None
    environment["gpu_total_memory_gib"] = None
    if environment["device_type"] == "cuda":

        def cuda_device_index() -> int:
            import torch

            if device is not None and torch.device(device).index is not None:
                return torch.device(device).index
            return torch.cuda.current_device()

        def gpu_name() -> str:
            import torch

            return torch.cuda.get_device_name(cuda_device_index())

        def gpu_compute_capability() -> str:
            import torch

            major, minor = torch.cuda.get_device_capability(cuda_device_index())
            return f"{major}.{minor}"

        def gpu_total_memory_gib() -> float:
            import torch

            total_bytes = torch.cuda.get_device_properties(cuda_device_index()).total_memory
            return round(total_bytes / 2**30, 2)

        environment["gpu_name"] = _try(gpu_name)
        environment["gpu_compute_capability"] = _try(gpu_compute_capability)
        environment["gpu_total_memory_gib"] = _try(gpu_total_memory_gib)

    environment["git_commit"] = _try(lambda: _run_git("rev-parse", "HEAD").strip())
    environment["git_has_uncommitted_changes"] = _try(
        lambda: _run_git("status", "--porcelain").strip() != ""
    )
    environment["executed_at_utc"] = _try(
        lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    )
    return environment


_LABELS = {
    "python_version": "Python",
    "platform": "OS / platform",
    "torch_version": "torch",
    "torch_cuda_version": "torch のビルド時の CUDA",
    "cudnn_version": "cuDNN",
    "device_type": "デバイス / device",
    "gpu_name": "GPU 名 / GPU name",
    "gpu_compute_capability": "compute capability",
    "gpu_total_memory_gib": "GPU の総メモリ (GiB)",
    "git_commit": "コミット / git commit",
    "git_has_uncommitted_changes": "未コミットの変更 / uncommitted changes",
    "executed_at_utc": "実行日時 (UTC)",
}


def _display_width(text: str) -> int:
    """全角文字を幅 2 として数えた表示幅(桁揃え用)。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text)


def _format_value(value: Any) -> str:
    if value is None:
        return "該当なし"
    if isinstance(value, bool):
        return "あり" if value else "なし"
    return str(value)


def print_execution_environment(device: Any = None) -> dict[str, Any]:
    """実行環境を 1 項目 1 行で印字し、``describe_execution_environment()`` と同じ辞書を返す。

    Args:
        device: ``describe_execution_environment()`` と同じ。
    """
    environment = describe_execution_environment(device)
    width = max(_display_width(label) for label in _LABELS.values())
    print("実行環境 / Execution environment")
    for key, value in environment.items():
        label = _LABELS.get(key, key)
        padding = " " * (width - _display_width(label))
        print(f"  {label}{padding} : {_format_value(value)}")
    return environment
