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

**読み込み済みのパッケージの版の食い違いの検査**(``check_preloaded_package_versions()``。
017 の後に追加): Google Colab のカーネルは起動時に numpy などを読み込む。セットアップセルの
``uv pip install`` がディスク上のパッケージを別の版に入れ替えると、メモリ上の古い版と、後から
遅延読み込みされる新しい版のサブモジュールが混在する。その結果、例外になる(017 の Colab での
実行で、``numpy.testing`` が古い版の C 拡張と食い違った)か、黙って誤った動作をしうる。この関数は
``uv pip install`` の直後・``import torch`` の前に呼び、食い違いを検出したら停止する。そのため、
この関数とこのモジュールの import は **標準ライブラリのみ** を使う(torch は
``describe_execution_environment()`` などの関数の中で読み込む。``src.utils`` の初期化も
numpy・torch を読み込まない)。
"""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Mapping
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


# requirements.txt に載っていて読み込み済みであれば、必ず検査する配布物(PEP 503 で正規化した名前)
REQUIRED_VERSION_CHECK_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "torch", "scipy", "pandas")
DEFAULT_REQUIREMENTS_PATH = _REPOSITORY_ROOT / "requirements.txt"

# 版を比べない配布物(PEP 503 で正規化した名前)。モジュールの __version__ の表記が配布物の版の
# 表記と、normalize_version_for_comparison() の正規化(PEP 440 の 0 詰めなど)では吸収できない形で
# 異なるため、入れ替わりがなくても食い違いと判定されてしまう配布物のための一覧である。現時点では
# 該当なし(ロックファイルから入れたローカルの環境で requirements.txt の全配布物を import して
# 確認した)。項目を加えるときは、理由を 1 行のコメントで書く。numpy・torch・scipy・pandas は
# ここに入れない(REQUIRED_VERSION_CHECK_DISTRIBUTIONS との重複は
# check_preloaded_package_versions() が拒否する)。
VERSION_CHECK_EXCLUDED_DISTRIBUTIONS: frozenset[str] = frozenset()

_RESTART_INSTRUCTIONS = (
    "Google Colab のメニューの「ランタイム」→「セッションを再起動」を選び、"
    "再起動の後に「すべてのセルを実行」する"
    "(2 回目の git clone が出す「already exists」のメッセージは無視してよい)。"
)
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def normalize_distribution_name(name: str) -> str:
    """PEP 503 の正規化: 小文字にし、``-``・``_``・``.`` の連続を 1 つの ``-`` にする。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirement_names(path: str | Path) -> set[str]:
    """``requirements.txt`` に記載された配布物の名前(PEP 503 で正規化)の集合を返す。

    コメント(``#`` 以降)・空行・オプションの行(``-`` で始まる行)を除き、環境マーカー(``;`` 以降)・
    extras(``[...]``)・版の指定を取り除いた先頭の名前を取る。``uv export`` の出力は推移的な依存も
    含み、同じ配布物が環境マーカーの違いで複数行に現れることがある(名前の集合なので 1 つに
    まとまる)。
    """
    names: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT_NAME.match(line)
        if match:
            names.add(normalize_distribution_name(match.group(0)))
    return names


def normalize_version_for_comparison(version: str) -> str:
    """版を比べるための正規化(標準ライブラリのみ)。

    PEP 440 では ``2026.07.22`` と ``2026.7.22`` は同じ版を表すが、文字列としては異なる
    (``certifi`` の ``__version__`` は 0 詰めで書かれ、配布物の版は正規化されている)。
    ``packaging`` は検査の対象でありうるので使わず、次の正規化だけを行う。

    - 前後の空白と、先頭の ``v``(大文字小文字を問わない)を除く。
    - ローカル版(``+`` 以降、例: ``+cu126``)は正規化せず、そのまま比べる。
    - ``+`` より前の部分を小文字にし、``.`` で区切った各要素のうち数字のみから成るものは先頭の 0 を
      取り除く(``"07"`` は ``"7"``、``"0"`` は ``"0"`` のまま)。数字以外を含む要素(``0rc1``・
      ``post0`` など)は変えない。
    """
    version = version.strip()
    if version[:1] in ("v", "V"):
        version = version[1:]
    public, plus, local = version.partition("+")
    parts = [str(int(part)) if part.isdigit() else part for part in public.lower().split(".")]
    return ".".join(parts) + plus + local


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_preloaded_package_versions(
    modules: Mapping[str, Any] | None = None,
    requirements_path: str | Path | None = DEFAULT_REQUIREMENTS_PATH,
    module_distributions: Mapping[str, list[str]] | None = None,
    installed_version: Callable[[str], str | None] | None = None,
    required: tuple[str, ...] = REQUIRED_VERSION_CHECK_DISTRIBUTIONS,
    on_mismatch: str = "raise",
) -> list[str]:
    """``requirements.txt`` の配布物のうち読み込み済みのものの、版の食い違いがないことを確かめる。

    **対象の決め方**: ``requirements.txt`` に記載された配布物(名前は PEP 503 で正規化して比べる)の
    うち、その配布物が提供するトップレベルのモジュールが ``modules``(既定は ``sys.modules``)に
    読み込み済みのものを対象とする。モジュール名から配布物の名前への対応は
    ``importlib.metadata.packages_distributions()`` で引く(``PIL`` → ``Pillow`` のように異なる場合が
    ある)。複数の配布物が提供するモジュール(名前空間パッケージなど)は、版を 1 つに決められないので
    使わない。版が入れ替わりうるのは ``uv pip install -r requirements.txt`` がインストールした配布物
    だけなので、それ以外(Colab が起動時に読み込む無関係なパッケージ)は ``__version__`` の表記の
    違いによる誤検出の原因にしかならず、検査しない。

    **比べる版**: 読み込み済みのモジュールの ``__version__`` と、配布物のインストール済みの版
    (``importlib.metadata``)。``normalize_version_for_comparison()`` で正規化した文字列が一致すれば
    同じ版とみなす(先頭の ``v``・0 詰めの違いを吸収し、ローカル版はそのまま比べる)。
    ``__version__`` を持たないモジュールと、``VERSION_CHECK_EXCLUDED_DISTRIBUTIONS`` の配布物
    (正規化で吸収できない表記の違いで誤検出するもの。現時点では該当なし)は比べない。ただし
    ``required`` の配布物(numpy・torch・scipy・pandas)は、``requirements.txt`` に載っていて
    読み込み済みなら必ず検査し、どちらかの版を取得できない場合も食い違いとして扱う。

    ``requirements_path`` が読めない(または ``None``)の場合は、例外で止めずに ``required`` の
    配布物のみを検査し、その旨を印字する。

    Args:
        modules: 読み込み済みのモジュールの辞書(既定は ``sys.modules``。確認用に差し替えられる)。
        requirements_path: ``requirements.txt`` のパス(既定はリポジトリルートのもの)。
        module_distributions: トップレベルのモジュール名から配布物の名前の一覧への対応(既定は
            ``packages_distributions()``)。
        installed_version: 配布物の名前からインストール済みの版を返す関数(既定は
            ``importlib.metadata.version``。取得できなければ None)。
        required: ``requirements.txt`` に載っていて読み込み済みなら必ず検査する配布物。
        on_mismatch: 食い違いがあったときの動作。``"raise"``(既定)は ``RuntimeError`` を送出する。
            ``"restart"`` は食い違いと「再接続後、もう一度『すべてのセルを実行』すること」を印字し、
            標準出力を flush してから ``os.kill(os.getpid(), 9)`` でカーネル(このプロセス)を終了する
            (Google Colab ではカーネルが自動で起動し直し、古い版がメモリから消える。017 の本番で、
            手動の「セッションを再起動」では安定して解消できなかったため)。既定を ``"raise"`` に
            するのは、ローカルで意図せずプロセスを終了させないためである。

    Returns:
        検査した配布物の ``"名前==版"`` の一覧(食い違いがない場合)。1 行で印字もする。

    Raises:
        RuntimeError: 1 つでも食い違いがある場合(``on_mismatch="raise"``)。食い違った配布物と
            両方の版、対処を示す。
        ValueError: ``on_mismatch`` が ``"raise"``・``"restart"`` のいずれでもない場合。
    """
    if on_mismatch not in ("raise", "restart"):
        raise ValueError(f"on_mismatch は 'raise' または 'restart': {on_mismatch!r}")
    modules = sys.modules if modules is None else modules
    if module_distributions is None:
        module_distributions = importlib.metadata.packages_distributions()
    installed_version = _installed_version if installed_version is None else installed_version
    required_names = {normalize_distribution_name(n) for n in required}
    overlap = required_names & VERSION_CHECK_EXCLUDED_DISTRIBUTIONS
    if overlap:
        raise ValueError(
            f"必ず検査する配布物が、比べない配布物の一覧に入っている: {sorted(overlap)}"
        )
    try:
        if requirements_path is None:
            raise OSError("requirements_path が指定されていない")
        targets = read_requirement_names(requirements_path)
        source = "requirements.txt"
    except OSError as error:
        targets = set(required_names)
        source = f"{', '.join(sorted(required_names))} のみ"
        print(
            f"注意: requirements.txt を読めない({type(error).__name__})ため、"
            f"{', '.join(sorted(required_names))} のみを検査する。"
        )

    # 配布物(正規化した名前)-> その配布物だけが提供する、読み込み済みのトップレベルのモジュール
    loaded: dict[str, list[str]] = {}
    for module_name in sorted(list(modules)):
        if "." in module_name or module_name.startswith("_"):
            continue
        distributions = {
            normalize_distribution_name(d)
            for d in module_distributions.get(module_name, [module_name])
        }
        if len(distributions) != 1:
            continue
        (distribution,) = distributions
        if distribution in targets:
            loaded.setdefault(distribution, []).append(module_name)
    excluded = sorted(set(loaded) & VERSION_CHECK_EXCLUDED_DISTRIBUTIONS)
    for distribution in excluded:
        del loaded[distribution]

    checked: list[str] = []
    without_version: list[str] = []
    mismatches: list[str] = []
    for distribution in sorted(loaded):
        installed = installed_version(distribution)
        versions = {
            name: getattr(modules[name], "__version__", None) for name in loaded[distribution]
        }
        versions = {name: v for name, v in versions.items() if isinstance(v, str)}
        problems = []
        if not versions:
            without_version.append(distribution)
            if distribution in required_names:
                problems.append(
                    f"{distribution}: 読み込み済みの版を取得できない(__version__ がない)"
                )
        elif installed is None:
            problems.append(f"{distribution}: インストール済みの版を取得できない")
        else:
            problems += [
                f"{distribution}: 読み込み済み {version}(モジュール {name})/ "
                f"インストール済み {installed}"
                for name, version in versions.items()
                if normalize_version_for_comparison(version)
                != normalize_version_for_comparison(installed)
            ]
        mismatches += problems
        if versions and not problems:
            checked.append(f"{distribution}=={installed}")
    if mismatches:
        summary = (
            "読み込み済みのパッケージの版が、インストール済みの版と食い違っている"
            "(カーネルの起動時に読み込まれた版が、"
            "セットアップセルのインストールで入れ替わった):\n  " + "\n  ".join(mismatches)
        )
        if on_mismatch == "restart":
            print(summary)
            print("カーネルを終了する。再接続後、もう一度「すべてのセルを実行」すること。")
            sys.stdout.flush()
            os.kill(os.getpid(), 9)
            # os.kill が戻った場合(確認のために差し替えたときなど)は、下の例外で止める
        raise RuntimeError(summary + "\n対処: " + _RESTART_INSTRUCTIONS)
    listed = ", ".join(checked) if checked else "なし"
    not_loaded = len(targets) - len(loaded) - len(excluded)
    print(
        f"読み込み済みのパッケージの版の検査({source}): 食い違いなし。"
        f"検査した配布物 {len(checked)} 個({listed})、"
        f"未読み込みのため検査しなかった配布物 {not_loaded} 個、"
        f"__version__ がないため比べなかった配布物 {len(without_version)} 個"
        f"({', '.join(without_version) or 'なし'})、"
        f"表記の違いで誤検出するため比べなかった配布物 {len(excluded)} 個"
        f"({', '.join(excluded) or 'なし'})"
    )
    return checked
