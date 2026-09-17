"""010(KV キャッシュと推論の計算量)の生データを Hugging Face Hub へ昇格する
(Promoting Raw Data)。

``theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb`` が
``.rawdata/010_rawdata.json`` に書き出した、実験 A・B・C・D の判定の一次情報
(条件 x 水準 x 反復の生の実行時間、条件 x シードの学習曲線・bits-per-byte)を、
Hugging Face Hub の Dataset リポジトリ ``kojikojiprg/ai-theories-rawdata-010`` へ
昇格する、リポジトリ運用スクリプトである。

CLAUDE.md「生データの永続化」節の方針に従い、アップロード前にローカルファイルの
SHA-256 を再計算し、ノートブックのセル出力に印字された値(``--expected-sha256``で
指定する)と一致することを確認する。一致しない場合はアップロードを行わずに停止する
(ノートブックの生成物と昇格対象が食い違ったまま Hub に上げてしまう事故を防ぐため)。

``--filename``で Hub 上のファイル名を指定できる(既定は``010_rawdata.json``)。
前提条件不成立などの理由で本番実行をやり直した場合、旧実行の生データを別名
(例: ``010_rawdata_run1.json``)で昇格すれば、判定に使う最新の生データを上書きせずに
両方を残せる。別名で昇格したファイルをデータセットカードに「判定には使わない」旨とともに
記載したい場合は、本ファイル冒頭の``FILE_NOTES``にエントリを追加すること
(``promote_canonical_corpora.py``の``CORPUS_SPECS``と同じ、静的レジストリによる管理。
アップロードのたびに README.md を丸ごと再構築するため、レジストリに追加しておかないと
過去の注記が新しい README.md で失われる)。

実行例(リポジトリルートから):

    uv run python scripts/promote_rawdata_010.py --expected-sha256 <ノートブックの出力の値>
    uv run python scripts/promote_rawdata_010.py --expected-sha256 <値> --upload
    uv run python scripts/promote_rawdata_010.py --expected-sha256 <値> \
        --filename 010_rawdata_run1.json --upload

既定ではアップロードを行わない(dry-run)。実際に Hugging Face Hub へアップロードする
には``--upload``を明示的に指定し、リポジトリルートの``.env``に``HF_TOKEN``を設定して
おくこと(``.env.example``を参照)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TARGET_REPO_ID = "kojikojiprg/ai-theories-rawdata-010"
DEFAULT_RAWDATA_PATH = _REPO_ROOT / ".rawdata" / "010_rawdata.json"
DEFAULT_FILENAME = "010_rawdata.json"

# Hub 上のファイル名 -> 判定に使わない理由の注記。判定に使う通常の生データ
# (DEFAULT_FILENAME)は含めない。前提不成立などで本番実行をやり直した場合の
# 旧実行データをここに追加すると、データセットカードに「判定には使わない」旨が
# 反映される(スクリプトの docstring を参照)。
FILE_NOTES: dict[str, str] = {
    "010_rawdata_run1.json": (
        "前提条件 P0(ウォームアップ後の反復間の変動係数が 0.1 以下)が実験 A・B・C の"
        "いずれでも不成立だった 1 回目の本番実行(Google Colab T4 GPU)の生データ。"
        "**判定には使用しない。** 前提不成立と判断した根拠・修正内容は、ノートブック "
        "7.1 節の「旧実行の記録」を参照。"
    ),
}


def _load_dotenv(path: Path = _REPO_ROOT / ".env") -> None:
    """``.env``ファイルから環境変数を読み込む(依存を増やさないための自前実装)。

    既に環境変数として設定されている値は上書きしない。
    """
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _require_hf_token() -> str:
    """``HF_TOKEN``を環境変数(``.env``を含む)から取得する。未設定なら分かりやすく停止する。"""
    import os

    _load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN が設定されていない。リポジトリルートの .env に "
            "HF_TOKEN=<トークン> を設定すること(雛形は .env.example を参照)。"
        )
    return token


def compute_sha256(path: Path) -> str:
    """ファイルの SHA-256 を計算する(大きなファイルでもメモリに全展開しないよう
    チャンク単位で読む)。
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_rawdata(rawdata_path: Path, expected_sha256: str) -> dict:
    """生データファイルを読み込み、SHA-256 が期待値と一致することを確認する。

    一致しない場合はアップロードに進ませないため、``RuntimeError``を送出する
    (ノートブックのセル出力に印字された値と昇格対象のファイルが食い違ったまま
    Hub に上げてしまう事故を防ぐ、CLAUDE.md「生データの永続化」節)。

    Returns:
        読み込んだ生データ(``json.load``の結果)。
    """
    if not rawdata_path.exists():
        raise FileNotFoundError(
            f"生データファイルが見つからない: {rawdata_path}\n"
            "ノートブックの生データ書き出しセル(および Google Colab からの"
            "ダウンロードセル)を実行してから再度呼び出すこと。"
        )

    actual_sha256 = compute_sha256(rawdata_path)
    byte_count = rawdata_path.stat().st_size
    print(f"ファイル: {rawdata_path}")
    print(f"バイト数: {byte_count:,}")
    print(f"SHA-256(実測): {actual_sha256}")
    print(f"SHA-256(期待値): {expected_sha256}")

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "生データファイルの SHA-256 が期待値と一致しない。"
            "ノートブックのセル出力に印字された値と、アップロード対象のファイルが"
            "食い違っている可能性があるため、アップロードは行わない。"
        )
    print("[OK] SHA-256 がノートブックのセル出力の値と一致した")

    data = json.loads(rawdata_path.read_text(encoding="utf-8"))
    return data


def summarize_rawdata(data: dict) -> None:
    """生データの内容を要約して印字する(実験ごとの件数)。"""
    for key, value in data.items():
        if isinstance(value, list):
            print(f"  - {key}: {len(value)} 件")
        elif isinstance(value, dict):
            print(f"  - {key}: {len(value)} キー")
        else:
            print(f"  - {key}: {value!r}")


def build_dataset_card(filename: str, rawdata_sha256: str, byte_count: int) -> str:
    """Dataset リポジトリ用のデータセットカード(日本語メイン・英語併記)を作る。

    ``FILE_NOTES``に登録された全ファイル(このファイル自身がまだ未登録の場合は
    それも含める)を列挙することで、今回のアップロードで README.md を丸ごと
    上書きしても、過去に別名で昇格したファイル(前提不成立の旧実行データなど)の
    注記が失われないようにする。
    """
    known_files = dict(FILE_NOTES)
    known_files.setdefault(filename, None)

    files_section_lines = []
    for name in sorted(known_files, key=lambda n: (n != DEFAULT_FILENAME, n)):
        note = known_files[name]
        if note is None:
            files_section_lines.append(f"- `{name}`: 判定に使用する生データ。")
        else:
            files_section_lines.append(f"- `{name}`: {note}")
    files_section = "\n".join(files_section_lines)

    this_file_note = known_files[filename]
    verification_note = (
        f"`{filename}` は判定に使用しない(上記参照)。"
        if this_file_note is not None
        else f"`{filename}` はノートブックのセル出力に印字された値と一致することを"
        "確認済み(``promote_rawdata_010.py``が昇格前に再検証している)。"
    )

    return f"""---
language: en
license: cc-by-nc-4.0
tags:
- ai-theories
- raw-data
---

# ai-theories 010 生データ(KV キャッシュと推論の計算量)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。

[010. KV キャッシュと推論の計算量](https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb)
の実験 A・B・C・D における、事前宣言した判定基準の対比量を再計算できる最小の量
(判定の一次情報)を格納する。

研究・教育目的のデータであり、品質保証は行っていない。

## 内容

- 実験 A・B・C: 条件 x 水準 x 反復の生の実行時間。
- 実験 D: 条件(平均プール初期化 / ランダム初期化)x シードの学習曲線・
  bits-per-byte(初期化直後・追加学習後)。

## ファイル一覧

{files_section}

## 検証情報(今回アップロードした`{filename}`について)

- バイト数: {byte_count:,}
- SHA-256: `{rawdata_sha256}`

{verification_note}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rawdata-path",
        type=Path,
        default=DEFAULT_RAWDATA_PATH,
        help=f"生データファイルのパス(既定: {DEFAULT_RAWDATA_PATH})。",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        required=True,
        help="ノートブックのセル出力に印字された SHA-256(この値と一致しない場合は"
        "アップロードしない)。",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_FILENAME,
        help=f"Hub 上でのファイル名(既定: {DEFAULT_FILENAME})。前提条件不成立などで"
        "本番実行をやり直した場合、旧実行の生データを別名(例: 010_rawdata_run1.json)"
        "で昇格すると、判定に使う最新の生データを上書きせずに両方を残せる。",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Hugging Face Hub へ実際にアップロードする(既定は dry-run で行わない)。"
        "HF_TOKEN を .env に設定しておくこと。",
    )
    args = parser.parse_args()

    data = verify_rawdata(args.rawdata_path, args.expected_sha256)
    print("生データの内訳:")
    summarize_rawdata(data)

    if args.filename != DEFAULT_FILENAME and args.filename not in FILE_NOTES:
        print(
            f"[警告] --filename={args.filename!r} は FILE_NOTES に未登録。データセット"
            "カードに「判定には使わない」旨を残したい場合は、本スクリプト冒頭の "
            "FILE_NOTES にエントリを追加してから実行すること。"
        )

    byte_count = args.rawdata_path.stat().st_size
    card_text = build_dataset_card(args.filename, args.expected_sha256, byte_count)

    if not args.upload:
        print("アップロードをスキップした(--upload が指定されていないため、既定は dry-run)。")
        return

    token = _require_hf_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"アップロード中: {TARGET_REPO_ID} ({args.filename})")
    api.create_repo(repo_id=TARGET_REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(args.rawdata_path),
        path_in_repo=args.filename,
        repo_id=TARGET_REPO_ID,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=TARGET_REPO_ID,
        repo_type="dataset",
    )
    print(f"アップロード完了: https://huggingface.co/datasets/{TARGET_REPO_ID}")


if __name__ == "__main__":
    main()
