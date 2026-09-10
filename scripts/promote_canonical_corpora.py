"""共有コーパスアーティファクトの Hub への切り出し(Promoting Canonical Corpus Artifacts)。

複数のトピックで再利用しうるコーパスアーティファクトを、それぞれ独立した Hugging Face
Hub の Dataset リポジトリへ切り出すための、リポジトリ運用スクリプトである(`theories/`の
特定のトピックにも`apps/`の特定のアプリにも属さない)。

現時点で扱うコーパスは以下の 2 つだが、``CORPUS_SPECS``にスペックを追加するだけで
別の言語・別のマニフェストのコーパスを追加できる構造にしている
(``promote_canonical_tokenizers.py``の``ARTIFACTS``辞書と同じ考え方)。

| キー | 言語 | マニフェスト | 記事数 | 用途 |
|---|---|---|---|---|
| en | 英語(en) | ``en_009_scaling.json`` | 9826 | 009(スケーリング則)の学習グリッド用 |
| ja | 日本語(ja) | ``ja_006_pretraining.json`` | 80 | 006(小型 GPT の事前学習)の日本語条件用 |

BPE の学習・言語モデルの学習を伴わない、純粋な取得・分割・アップロードの処理であり
GPU を要しないため、本番スケールのままローカルで構築・検証する。記事の取得は Wikipedia
API を叩いて待つだけの処理であり、``load_wikipedia_corpus()``(``src/data/text.py``)が
記事単位でローカルにキャッシュするため、中断しても再実行時に取得済みの記事は
再取得されない。

実行例(リポジトリルートから):

    uv run python scripts/promote_canonical_corpora.py
    uv run python scripts/promote_canonical_corpora.py --only en
    uv run python scripts/promote_canonical_corpora.py --upload

既定ではアップロードを行わない(dry-run)。実際に Hugging Face Hub へアップロードする
には``--upload``を明示的に指定し、リポジトリルートの``.env``に``HF_TOKEN``を設定して
おくこと(``.env.example``を参照)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.text import (  # noqa: E402
    load_english_wikipedia_corpus,
    load_japanese_wikipedia_corpus,
    split_train_val_text,
    upload_corpus_artifact_to_hub,
)


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


OUTPUT_DIR = _REPO_ROOT / ".cache" / "promote_canonical_corpora"

# 将来、別の言語・別のマニフェストのコーパスを追加する場合は、このリストにスペックを
# 追加するだけでよい。リポジトリ名にはマニフェストの版数・記事数を含めない
# (命名規則。異なるマニフェストが必要になった場合はブランチで管理する、各アーティファクトの
# データセットカードに明記する)。
#
# "skip": True のスペックは取得・アップロードの両方をスキップする(009 第 3 ラウンド
# 修正: 英語コーパスのみ拡張し、日本語コーパスは変更しないため、取得に時間がかかる
# 日本語コーパスの再取得・再アップロードを避ける)。
CORPUS_SPECS = [
    {
        "key": "en",
        "language": "en",
        "repo_id": "kojikojiprg/ai-theories-corpus-en",
        "manifest": "en_009_scaling.json",
        "manifest_article_count": 9826,
        "validation_ratio": 0.05,
        # キャッシュディレクトリは言語とデータ源(英語版 Wikipedia)で命名し、トピック番号を
        # 含めない。009・006・008 など英語版 Wikipedia を使う全トピックがこのディレクトリの
        # 記事単位キャッシュ(wikipedia_en_articles/)を共有する。
        "cache_dir": _REPO_ROOT / ".cache" / "wikipedia_en",
        "loader": load_english_wikipedia_corpus,
        "usage_note": "009(スケーリング則)の学習グリッド用",
        "skip": False,
    },
    {
        "key": "ja",
        "language": "ja",
        "repo_id": "kojikojiprg/ai-theories-corpus-ja",
        "manifest": "ja_006_pretraining.json",
        "manifest_article_count": 80,
        "validation_ratio": 0.05,
        # キャッシュディレクトリは言語とデータ源(日本語版 Wikipedia)で命名し、トピック番号を
        # 含めない。006 本体(theories/02_pretraining/006_pretraining_small_gpt.ipynb)と
        # このディレクトリの記事単位キャッシュ(wikipedia_ja_articles/)を共有する。
        "cache_dir": _REPO_ROOT / ".cache" / "wikipedia_ja",
        "loader": load_japanese_wikipedia_corpus,
        "usage_note": "006(小型 GPT の事前学習)の日本語条件用",
        # corpus.json -> corpus.txt+metadata.json への形式移行(Colab の RAM 制約対応)に
        # 伴い、日本語コーパスの内容自体は変更しないが形式を英語コーパスと揃えるため、
        # 今回に限り一時的に skip=False とする(記事は既にキャッシュ済みのため再取得は
        # 発生せず、数分で完了する見込み)。この移行の再アップロードが完了したら、
        # 日本語コーパスは今後変更の予定がないため skip=True に戻してよい。
        "skip": False,
    },
]


def process_corpus(spec: dict) -> dict:
    """1 つのコーパススペックについて、取得・分割・``corpus.txt``+``metadata.json``の
    組み立てを行う。

    ``loader``(記事数を縮小せず本番のマニフェスト全体で呼び出す)でコーパスを取得し、
    ``validation_ratio``で訓練・検証分割の決定性を確認したうえで、アップロード用の
    ``corpus.txt``(コーパス全文のプレーンテキスト)と``metadata.json``
    (由来・バイト数などのメタデータ)を書き出す。

    ``corpus.txt``の書き出しは、``raw_text``をチャンクに区切って逐次``write()``する
    (英語コーパスは 545 MB あり、``json.dumps``でエスケープ済み文字列を新たに
    構築する旧実装(コーパス全文の二重持ちが発生していた)を避けるため)。
    """
    key = spec["key"]
    t0 = time.time()
    raw_text, fetch_metadata = spec["loader"](spec["cache_dir"], return_metadata=True)
    elapsed = time.time() - t0
    print(f"取得時間: {elapsed:.1f} s ({elapsed / 60:.1f} 分)")

    raw_bytes = len(raw_text.encode("utf-8"))
    print(f"raw_text: {len(raw_text):,} 文字 / {raw_bytes:,} バイト")
    print(
        f"取得できた記事数: {fetch_metadata['fetched_article_count']} / "
        f"{fetch_metadata['manifest_article_count']}"
    )
    if fetch_metadata["skipped_articles"]:
        print(f"[警告] {len(fetch_metadata['skipped_articles'])} 記事の取得に失敗した:")
        for a in fetch_metadata["skipped_articles"]:
            print(f"  - {a['title']}: {a['reason']}")
    else:
        print("すべての記事を取得できた(スキップなし)")

    assert fetch_metadata["manifest_article_count"] == spec["manifest_article_count"], (
        f"{key}: マニフェスト記事数が期待({spec['manifest_article_count']})と一致しない: "
        f"{fetch_metadata['manifest_article_count']}"
    )

    # 009 第 3 ラウンド修正: 取得失敗を警告のみで済ませず、その場で停止してアップロードに
    # 進ませない。取得失敗を放置すると、設計より小さいコーパスのまま学習グリッド(GPU 時間を
    # 要する)まで進んでしまい、前提条件 P3(データ再利用の上限)違反が学習後にしか判明しない
    # という無駄が生じるため。
    assert not fetch_metadata["skipped_articles"], (
        f"{key}: {len(fetch_metadata['skipped_articles'])} 件の記事取得に失敗した。"
        "スキップされた記事は上の警告を参照。既知の取得不能(nosuchrevid 等)であれば"
        "マニフェストから当該エントリを削除し、必要なら代替記事で補充してから再実行する"
        "(手順は src/data/wikipedia_manifests/README.md を参照)。アップロードは行わない。"
    )

    train_text, val_text = split_train_val_text(raw_text, spec["validation_ratio"])
    train_text_2, val_text_2 = split_train_val_text(raw_text, spec["validation_ratio"])
    assert train_text == train_text_2 and val_text == val_text_2, f"{key}: 分割が決定的でない"
    print(f"[OK] {key}: split_train_val_text の決定性を確認した")
    print(f"train_text: {len(train_text):,} 文字, val_text: {len(val_text):,} 文字")

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=_REPO_ROOT
    ).stdout.strip()

    metadata_payload = {
        "language": spec["language"],
        "manifest": spec["manifest"],
        "manifest_article_count": fetch_metadata["manifest_article_count"],
        "fetched_article_count": fetch_metadata["fetched_article_count"],
        "skipped_articles": fetch_metadata["skipped_articles"],
        "validation_ratio": spec["validation_ratio"],
        "raw_bytes": raw_bytes,
        "source_commit": source_commit,
    }

    artifact_dir = OUTPUT_DIR / key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    corpus_txt_path = artifact_dir / "corpus.txt"
    _CHUNK_CHARS = 10_000_000  # raw_text をこの文字数ごとに区切って逐次書き込む
    with open(corpus_txt_path, "w", encoding="utf-8") as f:
        for i in range(0, len(raw_text), _CHUNK_CHARS):
            f.write(raw_text[i : i + _CHUNK_CHARS])
    print(f"corpus.txt を書き出した: {corpus_txt_path} ({corpus_txt_path.stat().st_size:,} バイト)")
    assert corpus_txt_path.stat().st_size == raw_bytes, (
        f"{key}: corpus.txt のバイト数が raw_bytes と一致しない"
    )

    metadata_json_path = artifact_dir / "metadata.json"
    metadata_json_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"metadata.json を書き出した: {metadata_json_path}")

    return {
        "spec": spec,
        "payload": metadata_payload,
        "artifact_dir": artifact_dir,
        "corpus_txt_path": corpus_txt_path,
        "metadata_json_path": metadata_json_path,
    }


def build_dataset_card(payload: dict, spec: dict) -> str:
    """由来(マニフェスト名・記事数)・取得できなかった記事・ライセンス・バージョン管理の
    方針を含むデータセットカードを作成する。
    """
    skipped = payload["skipped_articles"]
    skipped_section = (
        "\n".join(f"- {a['title']}: {a['reason']}" for a in skipped)
        if skipped
        else "なし(すべての記事を取得できた)"
    )
    return f"""---
language: {payload["language"]}
license: cc-by-sa-4.0
tags:
- ai-theories
- corpus
- wikipedia
---

# ai-theories {payload["language"]} コーパス

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
{spec["usage_note"]}のコーパス。

## ライセンスについての注記

このデータセット自体のライセンスは **クリエイティブ・コモンズ 表示-継承 4.0 国際
(CC BY-SA 4.0)** である。`ai-theories`の他の成果物(トークナイザなど)は
`cc-by-nc-4.0`を採用しているが、本データセットの内容はフリー百科事典
『ウィキペディア(Wikipedia)』の本文そのもの(wikitext を平文に変換したのみで、
内容は改変していない)であり、Wikipedia 本文自体のライセンス(CC BY-SA 4.0、
表示・継承の条件)を継承する必要があるため、別のライセンスとしている。

## 由来

`src/data/wikipedia_manifests/{payload["manifest"]}`
(記事タイトル・リビジョン ID を固定したマニフェスト、
{payload["manifest_article_count"]} 記事)から、Wikimedia API
(`action=parse`、`oldid`でリビジョンを指定)で取得した。タイトルとリビジョン ID を
両方固定しているため、取得時点によらず同一の入力が得られる。

- マニフェスト記事数: {payload["manifest_article_count"]}
- 取得できた記事数: {payload["fetched_article_count"]}

**取得できなかった記事**:

{skipped_section}

## バージョン管理についての注記

このリポジトリ名(`{spec["repo_id"]}`)には、マニフェストの版数や記事数を含めない。
**将来、同じ言語で異なるマニフェスト(記事数や選定基準が異なるコーパス)が必要に
なった場合は、新しいリポジトリを作らずブランチで管理する。** 現在このリポジトリが
対象としているマニフェストは`{payload["manifest"]}`({payload["manifest_article_count"]}
記事)であり、これは上記「由来」節と本カードの記載からのみ判別できる(リポジトリ名
からは判別できない)。

研究・教育目的で構築したものであり、品質保証は行っていない。商用・実運用での利用は
想定しない。

## 構成

このリポジトリは`corpus.txt`と`metadata.json`の 2 ファイルからなる(以前は単一の
`corpus.json`だったが、コーパス全文を JSON 文字列として保持すると二重にメモリを
消費するため、Google Colab の RAM 制約(12 GB)に対応してプレーンテキストと
メタデータに分離した)。**コーパス全文が必要な場合は`corpus.txt`を、由来や
バイト数などのメタデータのみが必要な場合は`metadata.json`を読めばよい。**

`corpus.txt`は、取得できた記事本文を連結した全文をそのまま UTF-8 のプレーンテキスト
として格納したファイルである。

`metadata.json`は以下のフィールドを含む。

- `language`: 言語コード(`{payload["language"]}`)
- `manifest`: 記事タイトル・リビジョン ID の対応を記したマニフェストファイル名
  (`{payload["manifest"]}`)
- `manifest_article_count`: マニフェストに列挙された記事数
- `fetched_article_count`: 実際に取得できた記事数
- `skipped_articles`: 取得に失敗した記事(タイトル・理由)のリスト
- `validation_ratio`: {payload["validation_ratio"]}(`src/data/text.py`の
  `split_train_val_text()`で使う訓練・検証分割の比率)
- `raw_bytes`: `corpus.txt`の UTF-8 バイト数({payload["raw_bytes"]:,})
- `source_commit`: 取得時点の`ai-theories`リポジトリのコミットハッシュ
  (`{payload["source_commit"]}`)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Hugging Face Hub へ実際にアップロードする(既定は dry-run で行わない)。"
        "HF_TOKEN を .env に設定しておくこと。",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="処理対象のキーをカンマ区切りで絞る(例: --only en)。CORPUS_SPECS の "
        "skip フラグと併用でき、--only で選んだキーでも skip=True なら取得・"
        "アップロードは行わない。",
    )
    args = parser.parse_args()

    only_keys = set(args.only.split(",")) if args.only else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_artifacts: dict[str, dict] = {}
    for spec in CORPUS_SPECS:
        key = spec["key"]
        if only_keys is not None and key not in only_keys:
            continue

        print(f"=== {key} ===")
        if spec.get("skip"):
            print(f"[スキップ] {key}: skip=True のため取得・アップロードともに行わない。")
            print()
            continue

        artifact = process_corpus(spec)
        corpus_artifacts[key] = artifact
        print()

    for key, artifact in corpus_artifacts.items():
        card = build_dataset_card(artifact["payload"], artifact["spec"])
        card_path = artifact["artifact_dir"] / "README.md"
        card_path.write_text(card, encoding="utf-8")
        artifact["dataset_card_path"] = card_path
        print(f"{key}: データセットカードを書き出した: {card_path}")

    if args.upload:
        token = _require_hf_token()
        for key, artifact in corpus_artifacts.items():
            repo_id = artifact["spec"]["repo_id"]
            print(f"アップロード中: {key} -> {repo_id}")
            upload_corpus_artifact_to_hub(
                repo_id=repo_id,
                corpus_txt_path=artifact["corpus_txt_path"],
                metadata_json_path=artifact["metadata_json_path"],
                dataset_card_text=artifact["dataset_card_path"].read_text(encoding="utf-8"),
                token=token,
            )
            print(f"アップロード完了: https://huggingface.co/datasets/{repo_id}")
    else:
        print(
            "アップロードをスキップした(--upload が指定されていないため、既定は dry-run)。"
            "実際にアップロードするには --upload を指定し、.env に HF_TOKEN を設定すること。"
        )


if __name__ == "__main__":
    main()
