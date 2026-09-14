"""010(KV キャッシュと推論の計算量)の GQA uptraining 済みチェックポイントを
Hugging Face Hub へ昇格する(Promoting the GQA Uptrained Checkpoint)。

実験 D(平均プール初期化 + 追加学習、シード 0)の代表チェックポイントを、
``.rawdata/010_gqa_uptrained.pt`` から既存のモデルリポジトリ
``kojikojiprg/ai-theories-small-gpt-en`` の ``gqa`` ブランチへアップロードする。

**新規リポジトリは作らない。** CLAUDE.md「共有アーティファクトの管理方針」の
命名規則(設定値の違いはリポジトリ名ではなくブランチで管理する)に従い、Grouped-Query
Attention への変換という設定の違いはブランチ(``gqa``)で表現する。``main`` ブランチ
(008 がアップロードした通常の多頭注意機構チェックポイント)には一切変更を加えない。

トークナイザ・コーパスは同梱しない(CLAUDE.md「同梱の禁止」節)。モデルカードに、
必要なトークナイザのリポジトリ(``kojikojiprg/ai-theories-tokenizer-en``)を明記する。

実行例(リポジトリルートから):

    uv run python scripts/promote_gqa_uptrained_model.py --expected-sha256 <値>
    uv run python scripts/promote_gqa_uptrained_model.py --expected-sha256 <値> --upload

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

TARGET_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
TARGET_BRANCH = "gqa"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / ".rawdata" / "010_gqa_uptrained.pt"


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


def verify_checkpoint(checkpoint_path: Path, expected_sha256: str) -> None:
    """チェックポイントファイルの SHA-256 が期待値と一致することを確認する。

    一致しない場合はアップロードに進ませないため、``RuntimeError``を送出する
    (CLAUDE.md「生データの永続化」節の受け渡し検証と同じ考え方)。
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"チェックポイントファイルが見つからない: {checkpoint_path}\n"
            "ノートブックの生データ書き出しセル(および Google Colab からの"
            "ダウンロードセル)を実行してから再度呼び出すこと。"
        )

    actual_sha256 = compute_sha256(checkpoint_path)
    byte_count = checkpoint_path.stat().st_size
    print(f"ファイル: {checkpoint_path}")
    print(f"バイト数: {byte_count:,}")
    print(f"SHA-256(実測): {actual_sha256}")
    print(f"SHA-256(期待値): {expected_sha256}")

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "チェックポイントファイルの SHA-256 が期待値と一致しない。"
            "ノートブックのセル出力に印字された値と、アップロード対象のファイルが"
            "食い違っている可能性があるため、アップロードは行わない。"
        )
    print("[OK] SHA-256 がノートブックのセル出力の値と一致した")


def build_config(
    num_key_value_heads: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    sequence_length: int,
    vocabulary_size: int,
) -> dict:
    """``config.json``の内容を作る(008 がアップロードした ``main`` ブランチの
    ``config.json``に``num_key_value_heads``を加えた形式。008 の他フィールドは
    変換の前後で変わらないため引き継ぐ)。
    """
    return {
        "vocabulary_size": vocabulary_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "d_ff": d_ff,
        "sequence_length": sequence_length,
        "positional_encoding": "rope",
        "normalization": "rmsnorm",
        "feed_forward": "swiglu",
        "norm_first": True,
    }


def build_model_card(num_key_value_heads: int, num_heads: int, checkpoint_sha256: str) -> str:
    """モデルカード(日本語メイン・英語併記)を作る。"""
    return f"""---
language: en
license: mit
tags:
- ai-theories
- gpt
- gqa
- scratch-implementation
---

# ai-theories 標準小型 GPT モデル(英語・Grouped-Query Attention、ブランチ: `{TARGET_BRANCH}`)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
[kojikojiprg/ai-theories-small-gpt-en](https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en)
の `main` ブランチ(通常の多頭注意機構、ヘッド数 {num_heads})を起点に、
[010. KV キャッシュと推論の計算量](https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb)
の実験 D で Key / Value ヘッド数を {num_key_value_heads} に削減し(Grouped-Query
Attention)、平均プール初期化(Ainslie et al., "GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints", EMNLP 2023 の uptraining)の後に
追加学習したチェックポイント(シード 0 の代表例)。

研究・教育目的のモデルであり、品質保証は行っていない。商用・実運用での利用は想定しない。

## 構成

`config.json` を参照。`num_key_value_heads` が `num_heads` より小さい点のみ、
`main` ブランチと異なる。

**トークナイザはこのリポジトリ・ブランチには同梱していない。**
[kojikojiprg/ai-theories-tokenizer-en](https://huggingface.co/{TOKENIZER_REPO_ID})
を使用すること(`main` ブランチと共通)。

## 検証情報

- SHA-256(`model_state.pt`): `{checkpoint_sha256}`

## 関連ノートブック

- [010. KV キャッシュと推論の計算量](https://github.com/kojikojiprg/ai-theories/blob/main/theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"チェックポイントファイルのパス(既定: {DEFAULT_CHECKPOINT_PATH})。",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        required=True,
        help="ノートブックのセル出力に印字された SHA-256(この値と一致しない場合は"
        "アップロードしない)。",
    )
    parser.add_argument("--num-key-value-heads", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--vocabulary-size", type=int, required=True)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Hugging Face Hub へ実際にアップロードする(既定は dry-run で行わない)。"
        "HF_TOKEN を .env に設定しておくこと。",
    )
    args = parser.parse_args()

    verify_checkpoint(args.checkpoint_path, args.expected_sha256)

    config = build_config(
        num_key_value_heads=args.num_key_value_heads,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        sequence_length=args.sequence_length,
        vocabulary_size=args.vocabulary_size,
    )
    card_text = build_model_card(args.num_key_value_heads, args.num_heads, args.expected_sha256)
    print("config.json:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    if not args.upload:
        print("アップロードをスキップした(--upload が指定されていないため、既定は dry-run)。")
        return

    token = _require_hf_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"ブランチを確認・作成中: {TARGET_REPO_ID}@{TARGET_BRANCH}")
    api.create_branch(repo_id=TARGET_REPO_ID, branch=TARGET_BRANCH, exist_ok=True)

    print(f"アップロード中: {TARGET_REPO_ID}@{TARGET_BRANCH}")
    api.upload_file(
        path_or_fileobj=str(args.checkpoint_path),
        path_in_repo="model_state.pt",
        repo_id=TARGET_REPO_ID,
        revision=TARGET_BRANCH,
    )
    api.upload_file(
        path_or_fileobj=json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo="config.json",
        repo_id=TARGET_REPO_ID,
        revision=TARGET_BRANCH,
    )
    api.upload_file(
        path_or_fileobj=card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=TARGET_REPO_ID,
        revision=TARGET_BRANCH,
    )
    print(f"アップロード完了: https://huggingface.co/{TARGET_REPO_ID}/tree/{TARGET_BRANCH}")


if __name__ == "__main__":
    main()
