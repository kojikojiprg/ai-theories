"""共有トークナイザアーティファクトの Hub への切り出し(Promoting Canonical Tokenizer
Artifacts)。

複数のトピック(005・006・008・009 以降)で再利用するトークナイザを、それぞれ独立した
Hugging Face Hub リポジトリへ切り出す、リポジトリ運用スクリプトである。

対象とする 3 つのアーティファクトの由来・位置づけは以下の通り。

| 言語 / ドメイン | 種類 | 設定 | 由来 | 位置づけ |
|---|---|---|---|---|
| 英語(en) | バイトレベル BPE | 語彙サイズ 8192 | 008 のモデルリポジトリに同梱済みのものを
  そのまま再利用(再学習しない) | 006 の本番実験で選定された条件 |
| 日本語(ja) | 文字レベル(Character-level) | ― | 006 の本番実験で日本語の勝者となった
  条件。006 が実際に使ったコーパスから、006 と完全に同一の手順で再構築する | 006 の
  本番実験で選定された条件 |
| コード(code) | バイトレベル BPE | 語彙サイズ 8192、``max_chunk_bytes=64``(英語と同一
  設定) | ``load_code_corpus()``(リポジトリ自身の`src/`を連結するコーパス) | 暫定
  (provisional)。006 はコードドメインで言語モデルの学習を行っておらず「正解」が
  存在しないため、英語と同じ設定を仮採用する |

BPE の学習・文字レベル語彙の構築はいずれも GPU を必要としない処理であるため、本番
スケールのまま(記事数・コーパス量を縮小せずに)ローカルで構築・検証する。

**このスクリプトは既に Google Colab で実行済みであり、3 つのリポジトリは Hugging Face
Hub 上に存在する。** ``.ipynb``から``.py``への移行は形式の変更のみであり、再アップロード
は不要である。

実行例(リポジトリルートから):

    uv run python scripts/promote_canonical_tokenizers.py
    uv run python scripts/promote_canonical_tokenizers.py --only en,ja
    uv run python scripts/promote_canonical_tokenizers.py --upload

既定ではアップロードを行わない(dry-run)。実際に Hugging Face Hub へアップロードする
には``--upload``を明示的に指定し、リポジトリルートの``.env``に``HF_TOKEN``を設定して
おくこと(``.env.example``を参照)。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.text import (  # noqa: E402
    CharacterLevelTokenizer,
    load_code_corpus,
    load_wikipedia_corpus,
    save_character_level_tokenizer_json,
)
from src.data.tokenizer import (  # noqa: E402
    BPEIDTokenizer,
    learn_bpe,
    load_bpe_id_tokenizer_json,
    save_bpe_id_tokenizer_json,
    upload_tokenizer_artifact_to_hub,
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


CACHE_DIR = _REPO_ROOT / ".cache" / "promote_canonical_tokenizers"

EN_SOURCE_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"  # 008 のモデルリポジトリ
EN_TARGET_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
JA_TARGET_REPO_ID = "kojikojiprg/ai-theories-tokenizer-ja"
CODE_TARGET_REPO_ID = "kojikojiprg/ai-theories-tokenizer-code"

# 006 と同じキャッシュディレクトリ(.cache/006_corpus)を指定し、既に取得済みの記事
# キャッシュを再利用する(記事タイトル・リビジョン ID を固定しているため、再取得しても
# 内容は変わらない)。
JA_CACHE_DIR = _REPO_ROOT / ".cache" / "006_corpus"

_EXPECTED_JA_VOCAB_SIZE = 4654  # 006 の本番実行結果(セル出力: `ja/character: vocab_size=4654`)


def build_english_artifact() -> Path:
    """英語(en)アーティファクトを再パッケージする。

    **再学習は行わない。** 008 のモデルリポジトリ(``EN_SOURCE_REPO_ID``)から
    ``tokenizer.json``を取得し、その内容をそのまま(バイト単位で)新しいリポジトリ用に
    使う。
    """
    from huggingface_hub import hf_hub_download

    en_tokenizer_json_path = Path(
        hf_hub_download(repo_id=EN_SOURCE_REPO_ID, filename="tokenizer.json")
    )

    # パース可能であること・語彙サイズが期待通りであることを検証する(内容そのものの
    # シリアライズ形式は変えないよう、アップロード用ファイルはこのオブジェクトから
    # 再生成せず、ダウンロードしたファイルをそのままコピーする、下記参照)。
    en_tokenizer = load_bpe_id_tokenizer_json(en_tokenizer_json_path)
    print(f"英語トークナイザ取得元: {EN_SOURCE_REPO_ID}")
    print(f"vocab_size = {en_tokenizer.vocab_size}")
    assert en_tokenizer.vocab_size == 8192, "英語トークナイザの語彙サイズが 8192 と一致しない"

    # ラウンドトリップ検証(短いサンプルテキストで確認する。取得したトークナイザを
    # 再学習するわけではないため、008 の学習コーパス全体を再取得する必要はない)。
    roundtrip_sample = "The quick brown fox jumps over the lazy dog. Hello, world! 12345."
    assert en_tokenizer.decode(en_tokenizer.encode(roundtrip_sample)) == roundtrip_sample
    print("[OK] 英語トークナイザのラウンドトリップ検証")

    en_artifact_dir = CACHE_DIR / "en"
    en_artifact_dir.mkdir(parents=True, exist_ok=True)
    en_artifact_path = en_artifact_dir / "tokenizer.json"
    shutil.copy(en_tokenizer_json_path, en_artifact_path)
    print(f"アップロード用ファイルを書き出した: {en_artifact_path}")
    return en_artifact_dir


def build_japanese_artifact() -> tuple[Path, int]:
    """日本語(ja)アーティファクトを構築する。

    006 の 5.2 節・5.4 節(``build_tokenizers()``)は、``CharacterLevelTokenizer``を
    ``full_corpus[lang]``(訓練・検証に分割する **前** のコーパス全体)から構築している。
    ``VALIDATION_RATIO``による訓練・検証分割は言語モデルの学習・評価にのみ使われ、
    文字レベル語彙の構築には使われていない(検証テキストのみに出現する文字で符号化が
    失敗しないよう、コーパス全体を使う設計)。

    また 006 が実際に呼んでいるのは``load_wikipedia_corpus("ja", CACHE_DIR)``
    (既定マニフェスト``ja_006_pretraining.json``)であり、``load_japanese_corpus()``
    (005 時点の別マニフェスト``ja_005_tokenizer_legacy.json``を使う薄いラッパー)ではない。
    以下はこの実際の手順を再現する。
    """
    corpus_ja = load_wikipedia_corpus("ja", JA_CACHE_DIR)
    print(f"corpus_ja: {len(corpus_ja):,} 文字 / {len(corpus_ja.encode('utf-8')):,} バイト")

    # 006 の本番実行(LM_CORPUS_BYTES = 10**12、実質無制限)では、load_corpus_prefix() による
    # バイト数での切り詰めは no-op である(実際のコーパスサイズがこれを大きく下回るため)。
    # したがって corpus_ja は 006 の full_corpus["ja"] と完全に一致する。
    ja_tokenizer = CharacterLevelTokenizer(corpus_ja)
    print(f"vocab_size = {ja_tokenizer.vocab_size}")

    assert ja_tokenizer.vocab_size == _EXPECTED_JA_VOCAB_SIZE, (
        f"006 の本番実行結果({_EXPECTED_JA_VOCAB_SIZE})と一致しない: {ja_tokenizer.vocab_size}"
    )
    print(f"[OK] 006 の本番実行結果(vocab_size={_EXPECTED_JA_VOCAB_SIZE})と一致した")

    assert ja_tokenizer.decode(ja_tokenizer.encode(corpus_ja)) == corpus_ja
    print("[OK] 日本語トークナイザのラウンドトリップ検証(コーパス全体)")

    ja_artifact_dir = CACHE_DIR / "ja"
    ja_artifact_dir.mkdir(parents=True, exist_ok=True)
    save_character_level_tokenizer_json(ja_tokenizer, ja_artifact_dir / "tokenizer.json")
    print(f"アップロード用ファイルを書き出した: {ja_artifact_dir / 'tokenizer.json'}")
    return ja_artifact_dir, ja_tokenizer.vocab_size


def build_code_artifact() -> tuple[Path, int, int, str, bool]:
    """コード(code)アーティファクトを構築する(暫定扱い)。

    006 はコードドメインで言語モデルの学習・トークナイザ条件の比較を行っておらず、
    コードドメインにおける「正解」となる設定はまだ存在しない。ここでは英語と同じ設定
    (バイトレベル BPE、語彙サイズ 8192、``max_chunk_bytes=64``)を仮採用する。

    ``load_code_corpus()``はリポジトリ自身の``src/``を実行時点の内容で連結するため、
    取得結果は取得時点のリポジトリ内容に依存する。取得に使ったコミットハッシュを記録する。
    """
    code_corpus = load_code_corpus(_REPO_ROOT)
    code_corpus_bytes = len(code_corpus.encode("utf-8"))
    print(f"code_corpus: {len(code_corpus):,} 文字 / {code_corpus_bytes:,} バイト")

    code_commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=_REPO_ROOT
    ).stdout.strip()
    # load_code_corpus() が実際に読むのは src/ 配下のみなので、コミットハッシュとの
    # 対応が崩れているかどうかは src/ の変更有無だけで判定すれば十分(リポジトリ全体の
    # 変更有無で判定すると、src/ に無関係な変更で誤って「dirty」と報告してしまう)。
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    ).stdout
    code_is_dirty = git_status.strip() != ""
    print(f"commit hash: {code_commit_hash}")
    print(f"src/ に未コミットの変更があるか: {code_is_dirty}")

    code_bpe = learn_bpe(code_corpus, vocab_size=8192, byte_level=True, max_chunk_bytes=64)
    code_symbols = sorted(code_bpe.vocab)
    code_symbol_to_id = {s: i for i, s in enumerate(code_symbols)}
    code_tokenizer = BPEIDTokenizer(code_bpe, code_symbol_to_id)
    print(f"vocab_size = {code_tokenizer.vocab_size}")

    assert code_tokenizer.decode(code_tokenizer.encode(code_corpus)) == code_corpus
    print("[OK] コードトークナイザのラウンドトリップ検証(コーパス全体)")

    code_artifact_dir = CACHE_DIR / "code"
    code_artifact_dir.mkdir(parents=True, exist_ok=True)
    save_bpe_id_tokenizer_json(code_tokenizer, code_artifact_dir / "tokenizer.json")
    print(f"アップロード用ファイルを書き出した: {code_artifact_dir / 'tokenizer.json'}")
    return (
        code_artifact_dir,
        code_tokenizer.vocab_size,
        code_corpus_bytes,
        code_commit_hash,
        code_is_dirty,
    )


def build_model_cards(
    ja_vocab_size: int,
    code_vocab_size: int,
    code_corpus_bytes: int,
    code_commit_hash: str,
    code_is_dirty: bool,
) -> dict[str, str]:
    """3 つのアーティファクトそれぞれについて、日本語メイン・英語併記のモデルカードを
    作成する。由来トピックへのリンク、スクラッチ実装であること・研究教育目的であり
    品質保証がないこと、コードアーティファクトについては暫定扱いである旨と理由・
    コーパスのコミットハッシュを含める。ライセンスは 3 つとも``cc-by-nc-4.0``を採用する。
    """
    dirty_note = (
        "(取得時点で作業ツリーに未コミットの変更があったため、この commit hash が示す内容と\n"
        "厳密には一致しない可能性がある)"
        if code_is_dirty
        else ""
    )

    en_model_card = """---
language: en
license: cc-by-nc-4.0
tags:
- ai-theories
- tokenizer
- byte-level-bpe
---

# ai-theories 標準トークナイザ(英語・バイトレベル BPE)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
バイトレベル Byte Pair Encoding(BPE、語彙サイズ 8192)のスクラッチ実装によるトークナイザ。

研究・教育目的で構築したものであり、品質保証は行っていない。商用・実運用での利用は
想定しない。

## 由来

[006. 小型 GPT の事前学習](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/006_pretraining_small_gpt.ipynb)
の本番実験で、英語について 5 つのトークナイザ条件(文字レベル・バイトレベル BPE ×
4 語彙サイズ・Unigram 言語モデル)の中から選定された条件(バイトレベル BPE、語彙サイズ
8192)である。学習済みの語彙・マージ規則自体は
[008. デコーディング戦略](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb)
の学習済みモデルに同梱されていたものをそのまま引き継いでおり、再学習は行っていない。

## 構成

- 方式: バイトレベル BPE(byte-level Byte Pair Encoding)
- 語彙サイズ: 8192
- 事前分割チャンクの最大バイト数(`max_chunk_bytes`): 64

`tokenizer.json` は `merges`・`vocab`・`byte_level`・`chunk_split_mode`・`max_chunk_bytes`・
`symbol_to_id` を含む(`src/data/tokenizer.py` の `load_bpe_id_tokenizer_json()` で
読み込める)。
"""

    ja_model_card = f"""---
language: ja
license: cc-by-nc-4.0
tags:
- ai-theories
- tokenizer
- character-level
---

# ai-theories 標準トークナイザ(日本語・文字レベル)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
文字単位(character-level)のトークナイザのスクラッチ実装。

研究・教育目的で構築したものであり、品質保証は行っていない。商用・実運用での利用は
想定しない。

## 由来

[006. 小型 GPT の事前学習](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/006_pretraining_small_gpt.ipynb)
の本番実験で、日本語について 5 つのトークナイザ条件の中から選定された条件(文字レベル)
である。006 が実際に使った日本語版 Wikipedia コーパス(記事タイトル・リビジョン ID を
`src/data/wikipedia_manifests/ja_006_pretraining.json` に固定、約 24.58 MB)から、
006 と完全に同一の手順(訓練・検証に分割する前のコーパス全体から出現文字を語彙化する)
で再構築した。

## 構成

- 方式: 文字レベル(character-level)
- 語彙サイズ: {ja_vocab_size}(コーパスに出現したユニーク文字数、006 の本番
  実行結果と一致することを確認済み)

`tokenizer.json` は文字 → ID の対応表(`char_to_id`)のみを含む
(`src/data/text.py` の `load_character_level_tokenizer_json()` で読み込める)。
"""

    code_model_card = f"""---
language: en
license: cc-by-nc-4.0
tags:
- ai-theories
- tokenizer
- byte-level-bpe
- provisional
---

# ai-theories 標準トークナイザ(コード・バイトレベル BPE、暫定)

`ai-theories`(https://github.com/kojikojiprg/ai-theories)プロジェクトの成果物。
バイトレベル Byte Pair Encoding(BPE、語彙サイズ {code_vocab_size})の
スクラッチ実装によるトークナイザ。

研究・教育目的で構築したものであり、品質保証は行っていない。商用・実運用での利用は
想定しない。

## 暫定(provisional)扱いについて

**このアーティファクトは暫定扱いである。**
[006. 小型 GPT の事前学習](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/006_pretraining_small_gpt.ipynb)
はコードドメインで言語モデルの学習・トークナイザ条件の比較を行っておらず、コードドメイン
における「正解」となる設定(方式・語彙サイズ)はまだ確立していない。将来、コードドメインで
のトークナイザ比較実験が行われ、より適した設定が判明した場合はこのアーティファクトを
差し替える可能性がある。

現時点では、006 の本番実験で英語について選定された設定(バイトレベル BPE、語彙サイズ
8192、`max_chunk_bytes=64`)を仮採用している。

## 由来

[005. トークナイザ](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/005_tokenizer.ipynb)
で導入された `load_code_corpus()`(`src/data/text.py`)により取得したコーパス
(本リポジトリ自身の `src/` 配下の Python ソースコードを連結したもの)から学習した。

**このコーパスはリポジトリ自身の内容に依存する。** `load_code_corpus()` はリポジトリの
`src/` を実行時点の内容で連結するため、取得結果は取得時点のリポジトリ内容に依存し、
将来のコミットでは変化しうる。取得に使ったコミットハッシュ: `{code_commit_hash}`
{dirty_note}

## 構成

- 方式: バイトレベル BPE(byte-level Byte Pair Encoding)
- 語彙サイズ: {code_vocab_size}
- 事前分割チャンクの最大バイト数(`max_chunk_bytes`): 64
- 学習コーパスサイズ: {code_corpus_bytes:,} バイト
"""

    return {"en": en_model_card, "ja": ja_model_card, "code": code_model_card}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Hugging Face Hub へ実際にアップロードする(既定は dry-run で行わない)。"
        "HF_TOKEN を .env に設定しておくこと。3 リポジトリは Colab で実行済みのため、"
        "通常は再アップロード不要。",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="処理対象のキー(en, ja, code)をカンマ区切りで絞る(例: --only en,ja)。",
    )
    args = parser.parse_args()

    only_keys = set(args.only.split(",")) if args.only else {"en", "ja", "code"}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    artifact_dirs: dict[str, Path] = {}
    ja_vocab_size = _EXPECTED_JA_VOCAB_SIZE
    code_vocab_size = 8192
    code_corpus_bytes = 0
    code_commit_hash = ""
    code_is_dirty = False

    if "en" in only_keys:
        print("=== 英語(en) ===")
        artifact_dirs["en"] = build_english_artifact()
        print()

    if "ja" in only_keys:
        print("=== 日本語(ja) ===")
        artifact_dirs["ja"], ja_vocab_size = build_japanese_artifact()
        print()

    if "code" in only_keys:
        print("=== コード(code) ===")
        (
            artifact_dirs["code"],
            code_vocab_size,
            code_corpus_bytes,
            code_commit_hash,
            code_is_dirty,
        ) = build_code_artifact()
        print()

    cards = build_model_cards(
        ja_vocab_size=ja_vocab_size,
        code_vocab_size=code_vocab_size,
        code_corpus_bytes=code_corpus_bytes,
        code_commit_hash=code_commit_hash,
        code_is_dirty=code_is_dirty,
    )
    for key, artifact_dir in artifact_dirs.items():
        (artifact_dir / "README.md").write_text(cards[key], encoding="utf-8")
    print(
        f"{len(artifact_dirs)} 件のモデルカードを .cache に書き出した"
        "(アップロードはまだ行っていない)"
    )

    repo_ids = {"en": EN_TARGET_REPO_ID, "ja": JA_TARGET_REPO_ID, "code": CODE_TARGET_REPO_ID}

    if args.upload:
        token = _require_hf_token()
        for key, artifact_dir in artifact_dirs.items():
            repo_id = repo_ids[key]
            print(f"アップロード中: {repo_id}")
            upload_tokenizer_artifact_to_hub(
                repo_id=repo_id,
                tokenizer_json_path=artifact_dir / "tokenizer.json",
                model_card_text=(artifact_dir / "README.md").read_text(encoding="utf-8"),
                token=token,
            )
            print(f"アップロード完了: https://huggingface.co/{repo_id}")
    else:
        print(
            "アップロードをスキップした(--upload が指定されていないため、既定は dry-run)。"
            "3 リポジトリは既に Colab で実行済みのため、通常は再アップロード不要。"
        )


if __name__ == "__main__":
    main()
