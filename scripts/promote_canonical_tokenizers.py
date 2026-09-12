"""共有トークナイザアーティファクトの Hub への切り出し(Promoting Canonical Tokenizer
Artifacts)。

複数のトピック(005・006・008・009 以降)で再利用するトークナイザを、それぞれ独立した
Hugging Face Hub リポジトリへ切り出す、リポジトリ運用スクリプトである。

対象とする 3 つのアーティファクトの由来・位置づけは以下の通り。

| 言語 / ドメイン | 種類 | 設定 | 由来 | 位置づけ |
|---|---|---|---|---|
| 英語(en) | バイトレベル BPE | 語彙サイズ 8192 | 008 が実際に使っている学習レシピ
  (英語版 Wikipedia コーパス・訓練分割の先頭 8,000,000 文字・``max_chunk_bytes=64``)から
  ローカルで再学習する | 006 の本番実験で選定された条件 |
| 日本語(ja) | 文字レベル(Character-level) | ― | 006 の本番実験で日本語の勝者となった
  条件。006 が実際に使ったコーパスから、006 と完全に同一の手順で再構築する | 006 の
  本番実験で選定された条件 |
| コード(code) | バイトレベル BPE | 語彙サイズ 8192、``max_chunk_bytes=64``(英語と同一
  設定) | ``load_code_corpus()``(リポジトリ自身の`src/`を連結するコーパス) | 暫定
  (provisional)。006 はコードドメインで言語モデルの学習を行っておらず「正解」が
  存在しないため、英語と同じ設定を仮採用する |

3 つのアーティファクトはいずれも、他のトピックのモデルリポジトリ(``kojikojiprg/
ai-theories-small-gpt-en``など)からアーティファクトをダウンロードする形では構築しない。
学習レシピ自体から独立に再構築する(BPE の学習・文字レベル語彙の構築はいずれも GPU を
必要としない処理であるため、本番スケールのまま、記事数・コーパス量を縮小せずにローカルで
構築・検証できる)。

**このスクリプトは既に Google Colab で実行済みであり、3 つのリポジトリは Hugging Face
Hub 上に存在する。** ``.ipynb``から``.py``への移行は形式の変更のみであり、再アップロード
は不要である。

**英語(en)アーティファクトの構築方法についての注記**: 当初は 008 のモデルリポジトリに
同梱されていた``tokenizer.json``をそのまま複製していたが、これはモデルリポジトリへの
循環依存になる(共有トークナイザリポジトリを作り直す手段が、そのリポジトリの昇格元である
モデルリポジトリに依存してしまい、モデルリポジトリを削除・再作成すると参照できなくなる)。
008 の学習レシピが再現可能であることを検証済みのため、学習レシピからの再構築に変更した
(``build_english_artifact()``参照)。構築結果は Hub 上の既存アーティファクトと
完全一致することを検証する。

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
    split_train_val_text,
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

EN_TARGET_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
JA_TARGET_REPO_ID = "kojikojiprg/ai-theories-tokenizer-ja"
CODE_TARGET_REPO_ID = "kojikojiprg/ai-theories-tokenizer-code"

# 言語とデータ源(Wikipedia 英語版)で決まるキャッシュディレクトリを指定し、008 が
# 取得済みのコーパスキャッシュ(.cache/wikipedia_en)を再利用する(記事タイトル・
# リビジョン ID を固定しているため内容は変わらない)。
EN_CACHE_DIR = _REPO_ROOT / ".cache" / "wikipedia_en"

# 008 の実際の設定値(008 5.2 節・5.6 節、ノートブックを読んで確認済み。推測ではない)。
# 英語アーティファクトはこの設定から再学習して構築するため、008 側でこれらの値が
# 変わった場合はこの定数も追随させる必要がある。
EN_VALIDATION_RATIO = 0.05  # 008 の VALIDATION_RATIO(訓練・検証分割の比率)
EN_TOKENIZER_TRAIN_BYTES = 8_000_000  # 008 の TOKENIZER_TRAIN_BYTES(本番値)
EN_VOCAB_SIZE = 8192  # 008 の VOCAB_SIZE
EN_MAX_CHUNK_BYTES = 64  # 008 の MAX_CHUNK_BYTES

# 言語とデータ源(Wikipedia 日本語版)で決まるキャッシュディレクトリを指定し、既に
# 取得済みの記事キャッシュ(.cache/wikipedia_ja、記事タイトル・リビジョン ID を固定
# しているため内容は 006 取得時と変わらない)を再利用する。
JA_CACHE_DIR = _REPO_ROOT / ".cache" / "wikipedia_ja"

_EXPECTED_JA_VOCAB_SIZE = 4654  # 006 の本番実行結果(セル出力: `ja/character: vocab_size=4654`)


def _describe_mismatch(name: str, built_value: object, hub_value: object) -> str:
    """`built_value`と`hub_value`が異なる場合に、差分の内容を人間が読める形で返す。"""
    if isinstance(built_value, list) and isinstance(hub_value, list):
        if len(built_value) != len(hub_value):
            return f"{name}: 要素数が異なる(構築={len(built_value)}, Hub={len(hub_value)})"
        for i, (b, h) in enumerate(zip(built_value, hub_value, strict=True)):
            if b != h:
                return f"{name}: {i} 番目の要素から異なる(構築={b!r}, Hub={h!r})"
        return f"{name}: 不一致箇所を特定できなかった"
    if isinstance(built_value, set) and isinstance(hub_value, set):
        only_built = sorted(built_value - hub_value)[:5]
        only_hub = sorted(hub_value - built_value)[:5]
        return f"{name}: 構築側のみに含まれる例={only_built}, Hub側のみに含まれる例={only_hub}"
    if isinstance(built_value, dict) and isinstance(hub_value, dict):
        diff_keys = [k for k in built_value if built_value.get(k) != hub_value.get(k)]
        return f"{name}: 値が異なるキー数={len(diff_keys)}, 例={diff_keys[:5]}"
    return f"{name}: 構築={built_value!r}, Hub={hub_value!r}"


def _verify_english_artifact_matches_hub(built_tokenizer: BPEIDTokenizer) -> None:
    """構築したトークナイザが Hub 上の既存の``EN_TARGET_REPO_ID``と完全一致することを
    検証する。一致しない場合は差分を報告したうえで例外を送出し、アップロードに
    進ませない(呼び出し元の``build_english_artifact()``がこの関数より先にアップロード
    用ファイルを書き出すことはない)。
    """
    from huggingface_hub import hf_hub_download

    hub_path = Path(hf_hub_download(repo_id=EN_TARGET_REPO_ID, filename="tokenizer.json"))
    hub_tokenizer = load_bpe_id_tokenizer_json(hub_path)

    built_bpe, hub_bpe = built_tokenizer.bpe_tokenizer, hub_tokenizer.bpe_tokenizer
    checks = [
        ("merges", built_bpe.merges, hub_bpe.merges),
        ("vocab", built_bpe.vocab, hub_bpe.vocab),
        ("symbol_to_id", built_tokenizer.symbol_to_id, hub_tokenizer.symbol_to_id),
        ("byte_level", built_bpe.byte_level, hub_bpe.byte_level),
        ("max_chunk_bytes", built_bpe.max_chunk_bytes, hub_bpe.max_chunk_bytes),
        ("chunk_split_mode", built_bpe.chunk_split_mode, hub_bpe.chunk_split_mode),
    ]
    mismatches = [
        _describe_mismatch(name, built_value, hub_value)
        for name, built_value, hub_value in checks
        if built_value != hub_value
    ]
    if mismatches:
        detail = "\n".join(f"  - {m}" for m in mismatches)
        raise RuntimeError(
            f"構築したトークナイザが Hub 上の既存アーティファクト({EN_TARGET_REPO_ID})と"
            f"一致しない。アップロードは行わない。差分:\n{detail}"
        )
    print(
        f"[OK] 構築したトークナイザは Hub 上の既存アーティファクト"
        f"({EN_TARGET_REPO_ID})と完全一致した"
    )


def build_english_artifact() -> Path:
    """英語(en)アーティファクトを構築する。

    008 が実際に使っている学習レシピ(英語版 Wikipedia コーパス`en_006_pretraining.json`、
    訓練・検証分割``EN_VALIDATION_RATIO``、訓練分割の先頭``EN_TOKENIZER_TRAIN_BYTES``文字、
    ``EN_VOCAB_SIZE``、``byte_level=True``、``EN_MAX_CHUNK_BYTES``)から BPE を学習する。

    以前は 008 のモデルリポジトリから``tokenizer.json``をそのまま複製していたが、これは
    モデルリポジトリへの循環依存になるため(モジュールの docstring 参照)、学習レシピ
    からの再構築に変更した。構築結果は Hub 上の既存アーティファクトと完全一致するかを
    検証し、一致しない場合はアップロードせずに停止する(``_verify_english_artifact_matches_hub``)。
    """
    corpus_en = load_wikipedia_corpus("en", EN_CACHE_DIR)
    print(f"corpus_en: {len(corpus_en):,} 文字 / {len(corpus_en.encode('utf-8')):,} バイト")

    train_text_en, _val_text_en = split_train_val_text(corpus_en, EN_VALIDATION_RATIO)
    en_bpe = learn_bpe(
        train_text_en[:EN_TOKENIZER_TRAIN_BYTES],
        EN_VOCAB_SIZE,
        byte_level=True,
        max_chunk_bytes=EN_MAX_CHUNK_BYTES,
    )
    en_symbols = sorted(en_bpe.vocab)
    en_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}
    en_tokenizer = BPEIDTokenizer(en_bpe, en_symbol_to_id)
    print(f"vocab_size = {en_tokenizer.vocab_size}")
    assert en_tokenizer.vocab_size == 8192, "英語トークナイザの語彙サイズが 8192 と一致しない"

    roundtrip_sample = train_text_en[: min(len(train_text_en), 50_000)]
    assert en_tokenizer.decode(en_tokenizer.encode(roundtrip_sample)) == roundtrip_sample
    print(f"[OK] 英語トークナイザのラウンドトリップ検証(先頭 {len(roundtrip_sample):,} 文字)")

    _verify_english_artifact_matches_hub(en_tokenizer)

    en_artifact_dir = CACHE_DIR / "en"
    en_artifact_dir.mkdir(parents=True, exist_ok=True)
    save_bpe_id_tokenizer_json(en_tokenizer, en_artifact_dir / "tokenizer.json")
    print(f"アップロード用ファイルを書き出した: {en_artifact_dir / 'tokenizer.json'}")
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
8192)である。
[008. デコーディング戦略](https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/008_decoding_strategies.ipynb)
が実際に使っている学習レシピ(英語版 Wikipedia コーパス、訓練分割の先頭 800 万文字)から
再学習して構築しており、既存アーティファクトの語彙・マージ規則と完全一致することを
検証済みである。

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
