"""文字レベル(character-level)テキスト処理のスクラッチ実装。

サブワード分割(Byte Pair Encoding など)は 005(トークナイザ)で扱う。本モジュールは
文字単位の最小限の処理であり、004 の言語モデリング実験(条件比較が目的で、
トークナイザそのものの性能は比較対象ではない)に必要な入力パイプラインのみを提供する。

記号 / Notation:
    V : 語彙サイズ(vocabulary size、出現するユニーク文字数)
    B : バッチサイズ(batch size)
    S : 系列長(sequence length)
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Wikipedia コーパス(load_wikipedia_corpus)の既定の記事タイトル・リビジョン ID の
# 対応(JSON、``src/data/wikipedia_manifests/`` に言語・用途ごとにコミット済み)を
# 探す場所。
#
# 出典 / Source: フリー百科事典『ウィキペディア(Wikipedia)』
#     (https://<language>.wikipedia.org/、language は "ja"・"en" など)
# ライセンス / License: クリエイティブ・コモンズ 表示-継承 4.0 国際
#     (CC BY-SA 4.0、https://creativecommons.org/licenses/by-sa/4.0/deed.ja)
_WIKIPEDIA_MANIFEST_DIR = Path(__file__).parent / "wikipedia_manifests"

# Wikitext(MediaWiki のマークアップ)を平文に変換する際に、名前空間へのリンク
# ([[File:...]] 等)として扱い、まるごと削除する接頭辞。
_WIKI_NAMESPACE_LINK_RE = re.compile(
    r"^(File|Image|ファイル|画像|Category|カテゴリ):", re.IGNORECASE
)


def _remove_wikitext_templates(text: str) -> str:
    """``{{...}}`` 形式のテンプレート呼び出しを除去する(入れ子を考慮)。"""
    while True:
        new_text, n = re.subn(r"\{\{[^{}]*\}\}", "", text)
        if n == 0:
            return new_text
        text = new_text


def _strip_wikitext_links(text: str) -> str:
    """``[[...]]`` 形式のリンクを、入れ子(画像キャプション内のリンクなど)を
    考慮して展開・除去する。

    File/Image/Category 名前空間へのリンクは画像・カテゴリのメタデータなので
    まるごと削除し、通常のリンクは表示テキスト(最後の ``|`` 以降。無ければ
    リンク先そのもの)だけを残す。
    """
    result: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i : i + 2] == "[[":
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j : j + 2] == "[[":
                    depth += 1
                    j += 2
                elif text[j : j + 2] == "]]":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            inner_clean = _strip_wikitext_links(text[i + 2 : j - 2])
            if not _WIKI_NAMESPACE_LINK_RE.match(inner_clean):
                result.append(inner_clean.split("|")[-1])
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def _wikitext_to_plaintext(wikitext: str) -> str:
    """Wikitext(MediaWiki のマークアップ)を平文(プレーンテキスト)に変換する。

    ``action=parse&oldid=<リビジョンID>`` で取得した特定リビジョンの wikitext を
    対象に、コメント・``<ref>`` 参照・テンプレート・表・リンクなど代表的なマークアップ
    を正規表現ベースで除去する簡易的な変換であり、MediaWiki パーサの完全な再実装では
    ない。テンプレートは展開せずまるごと除去するため、数値などをテンプレート経由で
    挿入している箇所は本文から欠落することがある(005 の fertility 集計は文章全体の
    圧縮効率を見るものであり、この程度の局所的な欠落が結果を大きく左右しないと
    判断し許容する)。
    """
    text = wikitext
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = _remove_wikitext_templates(text)
    text = re.sub(r"(?s)\{\|.*?\|\}", "", text)
    text = _strip_wikitext_links(text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"(?m)^=+\s*(.*?)\s*=+$", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^[*#:;]+\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class CharacterLevelTokenizer:
    """文字単位のトークナイザ。

    与えられたテキストに出現する文字の集合から語彙を構築し、文字 <-> 整数 ID の
    相互変換のみを行う。サブワード分割(Byte Pair Encoding など、005 で扱う)は
    一切行わない、文字単位の最小限の処理である。

    Args:
        text: 語彙構築に使うテキスト。出現した文字だけが語彙に含まれる
            (未知文字への対応は持たない)。
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char: dict[int, str] = dict(enumerate(chars))

    def encode(self, text: str) -> list[int]:
        """文字列を整数 ID のリストに変換する。"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int] | Tensor) -> str:
        """整数 ID の列を文字列に変換する。"""
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return "".join(self.id_to_char[i] for i in ids)

    @classmethod
    def from_char_to_id(cls, char_to_id: dict[str, int]) -> CharacterLevelTokenizer:
        """文字 → ID の対応表から直接構築する(``load_character_level_tokenizer_json()``用)。

        通常のコンストラクタ(``__init__``)は語彙構築用のテキストを受け取るが、
        ここでは保存済みの対応表をそのまま復元する。
        """
        tokenizer = cls.__new__(cls)
        tokenizer.char_to_id = dict(char_to_id)
        tokenizer.id_to_char = {i: ch for ch, i in tokenizer.char_to_id.items()}
        tokenizer.vocab_size = len(tokenizer.char_to_id)
        return tokenizer


def save_character_level_tokenizer_json(
    tokenizer: CharacterLevelTokenizer, path: str | Path
) -> None:
    """``CharacterLevelTokenizer``を tokenizer.json 相当の形式でシリアライズして保存する
    (``scripts/promote_canonical_tokenizers.py``の依頼を参照)。

    文字 → ID の対応表(``char_to_id``)のみを保存する(``id_to_char``は``char_to_id``
    から一意に復元できるため冗長)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"char_to_id": tokenizer.char_to_id}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_character_level_tokenizer_json(path: str | Path) -> CharacterLevelTokenizer:
    """``save_character_level_tokenizer_json()``が書き出した JSON を読み込む。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CharacterLevelTokenizer.from_char_to_id(data["char_to_id"])


def load_tiny_shakespeare(cache_dir: str | Path) -> str:
    """Tiny Shakespeare データセットをダウンロードしてキャッシュする。

    Andrej Karpathy の char-rnn リポジトリで配布されているテキストファイルを取得する
    (https://github.com/karpathy/char-rnn)。2 回目以降の呼び出しはキャッシュから読む。

    Args:
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。

    Returns:
        テキスト全体を 1 つの文字列として返す。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "tiny_shakespeare.txt"

    if not cache_path.exists():
        urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, cache_path)  # noqa: S310

    return cache_path.read_text(encoding="utf-8")


def _fetch_wikipedia_revision_plaintext(
    language: str,
    title: str,
    revid: int,
    max_retries: int = 5,
    retry_wait_seconds: float = 5.0,
    request_timeout: float = 30.0,
) -> str:
    """指定言語版 Wikipedia の指定リビジョンの本文を取得し、平文に変換する。

    ``action=query&prop=extracts`` は常に最新版の本文を返し、``revids`` パラメータを
    指定しても無視される(実際にリクエストして確認済み: 数年前のリビジョン ID を
    指定しても最新版と同一の本文が返る)。特定のリビジョンを取得するには
    ``action=parse&oldid=<リビジョンID>`` を使う必要がある。この API は wikitext
    (MediaWiki のマークアップ)しか返さないため、``_wikitext_to_plaintext`` で
    平文に変換する(この変換ロジック自体は言語に依存しない一般的な MediaWiki
    マークアップの除去である)。

    Wikimedia API は (1) デフォルトの User-Agent(``python-urllib/...``)からの
    リクエストを 403 で拒否し、(2) 短時間に連続してリクエストすると 429
    (Too Many Requests)を返すことがある。そのため識別可能な User-Agent を
    付与し、各リクエストの間隔を空け、429 発生時は指数的に待機して再試行する。
    また ``urlopen`` にタイムアウトを指定しない場合、接続が確立した後にサーバー側で
    応答が止まる(スタール)と無期限にハングしうる(数百記事を連続取得する
    ``load_wikipedia_corpus`` では、この無期限ハングが 1 回でも起きると呼び出し全体が
    止まってしまうため、明示的なタイムアウトと再試行が必須)。

    HTTP ステータスが 200 でも、応答 JSON に ``"parse"`` キーがないことがある
    (記事が見つからない、リダイレクト、API 側のエラーオブジェクトを返す場合など。
    009 の本番実行で ``KeyError: 'parse'`` により全体が停止した不具合の修正)。
    この場合も 429・タイムアウトと同様に再試行対象として扱う。

    Args:
        language: Wikipedia の言語コード(``"ja"``・``"en"`` など)。
        title: 記事タイトル(取得自体には使わない。呼び出し側がキャッシュファイル名の
            対応付けに使うために受け取るだけの引数)。
        revid: 取得するリビジョン ID。
        max_retries: 429・タイムアウト・``"parse"``キー欠落発生時の最大再試行回数。
        retry_wait_seconds: 429・タイムアウト・``"parse"``キー欠落発生時の基本待機秒数
            (試行回数に比例して延びる)。
        request_timeout: 1 リクエストあたりの ``urlopen`` タイムアウト秒数。

    Raises:
        RuntimeError: ``max_retries``回再試行してもなお応答 JSON に``"parse"``キーが
            ない場合。呼び出し元(``load_wikipedia_corpus``)はこれを捕捉して当該記事を
            スキップする。
    """
    del title  # 取得は revid のみで行う(引数の説明は docstring を参照)
    params = urllib.parse.urlencode(
        {
            "action": "parse",
            "oldid": revid,
            "prop": "wikitext",
            "format": "json",
            "formatversion": 2,
        }
    )
    url = f"https://{language}.wikipedia.org/w/api.php?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-theories-wikipedia-corpus/1.0"}
    )

    for attempt in range(max_retries):
        is_last_attempt = attempt == max_retries - 1
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
            time.sleep(1.0)  # 連続リクエストによるレート制限(429)を避ける
            if "parse" in data:
                return _wikitext_to_plaintext(data["parse"]["wikitext"])
            if not is_last_attempt:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise RuntimeError(
                f"{language} revid={revid}: 応答 JSON に 'parse' キーがない"
                f"(error={data.get('error')!r})"
            )
        except urllib.error.HTTPError as e:
            if e.code == 429 and not is_last_attempt:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise
        except OSError:
            # タイムアウト(TimeoutError)、接続確立時の失敗(URLError、いずれも
            # OSError のサブクラス)、接続確立後にサーバーが応答を止めて切断する
            # RemoteDisconnected(http.client、これも OSError のサブクラス)を
            # まとめて一時的なネットワークエラーとして再試行する。
            if not is_last_attempt:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"{language} revid={revid}: 取得に失敗した(到達しないはずの分岐)")


def load_wikipedia_corpus(
    language: str,
    cache_dir: str | Path,
    manifest_path: str | Path | None = None,
    return_metadata: bool = False,
) -> str | tuple[str, dict]:
    """指定言語版 Wikipedia の固定記事集合を取得してキャッシュする(006、言語モデル
    事前学習用のコーパス取得)。

    出典: フリー百科事典『ウィキペディア(Wikipedia)』(https://<language>.wikipedia.org/、
    language は ``"ja"``・``"en"`` など)。ライセンス: クリエイティブ・コモンズ
    表示-継承 4.0 国際(CC BY-SA 4.0)。

    ``manifest_path`` の JSON(``{"記事タイトル": リビジョン ID, ...}``)に列挙された
    記事を、Wikimedia API(``action=parse``、``oldid`` でリビジョンを指定)を用いて
    取得し、wikitext を平文に変換したうえで連結する。タイトルとリビジョン ID を
    両方固定しているため、実行時点によらず同一の入力が得られる
    (``action=query&prop=extracts`` はリビジョンを固定できず最新版を返してしまうため
    使わない。``_fetch_wikipedia_revision_plaintext`` の docstring を参照)。

    記事ごとに個別のファイルへキャッシュする(``cache_dir/wikipedia_<language>_articles/``、
    ファイル名にリビジョン ID を含めるため、manifest を更新した場合は自動的に
    再取得される)。Wikimedia API のレート制限(429)により取得が一部の記事で
    失敗しても、呼び出しを再試行すれば取得済みの記事はキャッシュから読み、
    未取得の記事のみ再取得する。取得の進捗は 100 記事ごとに標準出力へ記録する
    (経過時間・推定残り時間を含む)。

    連結済みコーパス全体のキャッシュファイル名には、言語に加えて **``manifest_path``の
    ファイル名(拡張子除く)と記事数の両方を含める**(``cache_dir/wikipedia_<language>_
    <manifest のファイル名>_<記事数>.txt``)。以前は言語のみをキーにしていたため、
    同じ``cache_dir``に対して記事数を縮小したマニフェスト(スモークテスト用など)を
    指定しても、既に存在する全量取得のキャッシュファイルがそのまま読まれ、意図した
    縮小コーパスではなく全量コーパスが黙って返る不具合があった(この修正により解消)。
    記事単位のキャッシュ(``articles_dir``)はリビジョン ID ベースでマニフェストを
    またいで再利用できるため、このキーには含めない(異なるマニフェストで同じ記事の
    再取得は発生しない)。

    **記事単位の取得失敗は全体を止めない**(009 の本番実行が`KeyError: 'parse'`で
    全体停止した不具合の修正)。``_fetch_wikipedia_revision_plaintext``が
    ``max_retries``回再試行してもなお失敗した記事はスキップし、取得できた記事数・
    スキップした記事とその理由を``cache_dir/wikipedia_<language>_<manifest のファイル名>_
    <記事数>_fetch_metadata.json``に記録した上で、取得できた記事だけでコーパスを
    組み立てる(目標記事数を下回った場合はその旨を標準出力に警告として出す)。

    Args:
        language: Wikipedia の言語コード(``"ja"``・``"en"`` など)。
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。
        manifest_path: 記事タイトル・リビジョン ID の対応を記した JSON ファイルへの
            パス。``None``(既定値)の場合、
            ``src/data/wikipedia_manifests/<language>_006_pretraining.json``
            (UTF-8 で 20 MB 以上を目標に Wikipedia の長大記事(Longpages)から
            選定した記事集合、本リポジトリにコミット済み)を使う。
        return_metadata: True の場合、``(text, metadata)``のタプルを返す。
            ``metadata``は``{"manifest_article_count": int, "fetched_article_count":
            int, "skipped_articles": [{"title": str, "reason": str}, ...]}``を含む
            (``scripts/promote_canonical_corpora.py``で使用)。既定値``False``
            の場合は従来通り``text``のみを返す(005・006・008・009 の既存呼び出しの
            返り値は不変)。

    Returns:
        ``return_metadata=False``(既定)の場合、取得できた記事本文を連結した 1 つの
        文字列。``return_metadata=True``の場合は``(text, metadata)``のタプル。
    """
    if manifest_path is None:
        manifest_path = _WIKIPEDIA_MANIFEST_DIR / f"{language}_006_pretraining.json"
    manifest_path = Path(manifest_path)
    manifest: dict[str, int] = json.loads(manifest_path.read_text(encoding="utf-8"))

    cache_dir = Path(cache_dir)
    articles_dir = cache_dir / f"wikipedia_{language}_articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    # キャッシュファイル名にマニフェスト名・記事数を含める(マニフェストごとに衝突しない
    # ようにするため。以前はキャッシュファイル名が言語のみに依存しており、例えば記事数を
    # 縮小したマニフェスト(スモークテスト用)を指定しても、同じ cache_dir 内に既に存在する
    # 全量取得のキャッシュファイルがそのまま読まれ、意図した縮小コーパスではなく全量
    # コーパスが黙って返る不具合があった)。記事単位のキャッシュ(articles_dir)は
    # revid ベースでマニフェストをまたいで再利用できるため、このキーには含めない。
    cache_key = f"wikipedia_{language}_{manifest_path.stem}_{len(manifest)}"
    cache_path = cache_dir / f"{cache_key}.txt"
    metadata_path = cache_dir / f"{cache_key}_fetch_metadata.json"

    if not cache_path.exists():
        texts = []
        skipped_articles: list[dict[str, str]] = []
        total = len(manifest)
        start_time = time.time()
        for i, (title, revid) in enumerate(manifest.items()):
            article_path = articles_dir / f"{i:03d}_{revid}.txt"
            if not article_path.exists():
                try:
                    plaintext = _fetch_wikipedia_revision_plaintext(language, title, revid)
                except Exception as e:  # noqa: BLE001  # 記事単位の失敗理由を問わずスキップする
                    skipped_articles.append({"title": title, "reason": repr(e)})
                    continue
                article_path.write_text(plaintext, encoding="utf-8")
            texts.append(article_path.read_text(encoding="utf-8"))

            done = i + 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start_time
                remaining = elapsed / done * (total - done)
                print(
                    f"[{language}] {done}/{total} 記事処理済み"
                    f"(取得成功 {len(texts)} 件、スキップ {len(skipped_articles)} 件)、"
                    f"経過時間 {elapsed / 60:.1f} 分、推定残り時間 {remaining / 60:.1f} 分"
                )
        cache_path.write_text("\n".join(t for t in texts if t), encoding="utf-8")

        metadata = {
            "manifest_article_count": len(manifest),
            "fetched_article_count": len(texts),
            "skipped_articles": skipped_articles,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if skipped_articles:
            print(
                f"[警告] {language}: 目標 {len(manifest)} 記事中 {len(texts)} 記事のみ"
                f"取得できた(スキップ {len(skipped_articles)} 件)。"
                f"詳細は {metadata_path} を参照。"
            )
        else:
            print(f"{language}: {len(texts)}/{len(manifest)} 記事を取得した")

    text = cache_path.read_text(encoding="utf-8")
    if not return_metadata:
        return text

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        # metadata の記録を追加する前にキャッシュ済みだった corpus(全記事取得済みの
        # 前提で扱う。当時は全記事が揃わない限りキャッシュファイルを作らなかったため)。
        metadata = {
            "manifest_article_count": len(manifest),
            "fetched_article_count": len(manifest),
            "skipped_articles": [],
        }
    return text, metadata


def load_japanese_corpus(cache_dir: str | Path) -> str:
    """日本語コーパスを取得してキャッシュする(005 トークナイザの日本語ドメイン用)。

    ``load_wikipedia_corpus("ja", cache_dir, manifest_path=...)`` の薄いラッパー。
    005 で使った固定の 44 記事(``src/data/wikipedia_manifests/
    ja_005_tokenizer_legacy.json``、005 時点の記事タイトル・リビジョン ID の対応と
    同一)を対象とし、既定の引数値・返り値は 005 時点から変えていない(006 で
    ``load_wikipedia_corpus`` に一般化する際に、005 の呼び出し結果が変わらないことを
    検証済み)。

    Args:
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。

    Returns:
        取得した記事本文を連結した 1 つの文字列(005 時点と同一)。
    """
    manifest_path = _WIKIPEDIA_MANIFEST_DIR / "ja_005_tokenizer_legacy.json"
    return load_wikipedia_corpus("ja", cache_dir, manifest_path=manifest_path)


def load_japanese_wikipedia_corpus(
    cache_dir: str | Path, return_metadata: bool = False
) -> str | tuple[str, dict]:
    """日本語版 Wikipedia の記事集合からコーパスを取得してキャッシュする(006 の
    小型 GPT 事前学習用に選定したマニフェストを使う)。

    ``load_wikipedia_corpus("ja", cache_dir, manifest_path=...)`` の薄いラッパー。
    006 で使った 80 記事(``ja_006_pretraining.json``、UTF-8 で 20 MB 以上を目標に
    選定)を対象とする。``load_japanese_corpus()``(005 のトークナイザ用、
    ``ja_005_tokenizer_legacy.json``、44 記事)とは異なるマニフェストである点に注意。

    006 自体は``manifest_path``省略時の既定値(``load_wikipedia_corpus``の docstring
    参照)により本関数と同じマニフェストを直接呼び出しているため、本関数は既存の
    呼び出し結果と同一のテキストを返す。現時点で本関数を直接呼び出す既存ノートブックは
    ない(``scripts/promote_canonical_corpora.py``が、英語の
    ``load_english_wikipedia_corpus()``と対称にするために``loader``として使う)。

    Args:
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。
        return_metadata: ``load_wikipedia_corpus``にそのまま渡す(``scripts/
            promote_canonical_corpora.py``で使用)。

    Returns:
        ``load_wikipedia_corpus``と同じ(``return_metadata``の値に応じて文字列
        またはタプル)。
    """
    manifest_path = _WIKIPEDIA_MANIFEST_DIR / "ja_006_pretraining.json"
    return load_wikipedia_corpus(
        "ja", cache_dir, manifest_path=manifest_path, return_metadata=return_metadata
    )


def load_english_wikipedia_corpus(
    cache_dir: str | Path, return_metadata: bool = False
) -> str | tuple[str, dict]:
    """英語版 Wikipedia の記事集合からコーパスを取得してキャッシュする(009 スケーリング則
    の学習グリッド用に選定したマニフェストを使う)。

    ``load_wikipedia_corpus("en", cache_dir, manifest_path=...)`` の薄いラッパー。
    006 で使った 356 記事(``en_006_pretraining.json``)に、Wikipedia の
    Special:LongPages(長大記事一覧)から追加で選定した記事を加えた計 9826 記事
    (``src/data/wikipedia_manifests/en_009_scaling.json``、内訳は同ディレクトリの
    README を参照)を対象とする。006 のマニフェストを部分集合として含む形で
    拡張しているため、006・008 で既に取得済みの記事のキャッシュ
    (``wikipedia_en_articles/``)をそのまま再利用でき、追加で取得が必要なのは
    新規記事分のみである。

    Args:
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。
        return_metadata: ``load_wikipedia_corpus``にそのまま渡す(``scripts/
            promote_canonical_corpora.py``で使用)。既定値``False``の場合、
            009 の既存呼び出しの返り値は不変。

    Returns:
        ``load_wikipedia_corpus``と同じ(``return_metadata``の値に応じて文字列
        またはタプル)。
    """
    manifest_path = _WIKIPEDIA_MANIFEST_DIR / "en_009_scaling.json"
    return load_wikipedia_corpus(
        "en", cache_dir, manifest_path=manifest_path, return_metadata=return_metadata
    )


_EN_WIKIPEDIA_CORPUS_REPO_ID = "kojikojiprg/ai-theories-corpus-en"


def load_english_wikipedia_corpus_with_fallback(
    cache_dir: str | Path, return_metadata: bool = False
) -> str | tuple[str, dict]:
    """英語版 Wikipedia のコーパスを取得する。

    まず``kojikojiprg/ai-theories-corpus-en``(Hugging Face Hub の Dataset
    リポジトリ、``scripts/promote_canonical_corpora.py``でアップロードしたもの)
    から``corpus.txt``(コーパス全文のプレーンテキスト)と``metadata.json``
    (由来・バイト数などのメタデータ)を取得する。取得に失敗した場合
    (リポジトリが未作成・ネットワーク障害など)のみ、``load_english_wikipedia_corpus()``
    による Wikipedia API からの直接取得にフォールバックする。9826 記事の直接取得は
    Colab で数時間規模の時間を要するため(009、5.4 節)、Hub のデータセットが
    存在すればそれを優先する。

    **``corpus.json``(単一 JSON、``raw_text``フィールドにコーパス全文を格納)から
    ``corpus.txt``+``metadata.json``の 2 ファイル構成に変更した(009、Colab の RAM
    制約対応)。** JSON パースによる二重持ち(コーパス文字列とパース結果、約 1.1 GB)
    を避けるため、コーパス全文はプレーンテキストとして別ファイルに分離している。

    Args:
        cache_dir: 直接取得にフォールバックした場合のキャッシュ先ディレクトリ
            (``load_english_wikipedia_corpus()``にそのまま渡す)。
        return_metadata: True の場合、``(text, metadata)``のタプルを返す。
            ``metadata``は``{"source": "hub" または "direct", "raw_bytes": int,
            "manifest_article_count": int, "fetched_article_count": int,
            "skipped_articles": [...]}``を含む。取得元が Hub の場合、
            ``raw_bytes``・記事数はアップロード時に``metadata.json``へ記録された
            値(009 側で独立に再取得できない)であり、``len(text.encode("utf-8"))``
            と比較することで Hub からの取得が破損していないかを検証できる
            (Hub 経由の取得は決定的であるため、一致しなければ取得の破損を意味する)。
            既定値``False``の場合は従来通り``text``のみを返す(既存呼び出しの
            返り値は不変)。

    Returns:
        ``return_metadata=False``(既定)の場合、取得したコーパス全文。
        ``return_metadata=True``の場合は``(text, metadata)``のタプル。
        取得元(Hub / 直接取得のどちらだったか)は標準出力にも明記する。
    """
    try:
        from huggingface_hub import hf_hub_download

        corpus_path = hf_hub_download(
            repo_id=_EN_WIKIPEDIA_CORPUS_REPO_ID, filename="corpus.txt", repo_type="dataset"
        )
        metadata_path = hf_hub_download(
            repo_id=_EN_WIKIPEDIA_CORPUS_REPO_ID, filename="metadata.json", repo_type="dataset"
        )
        data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        print(f"コーパス取得元: {_EN_WIKIPEDIA_CORPUS_REPO_ID}(Hugging Face Hub)")
        text = Path(corpus_path).read_text(encoding="utf-8")
        if not return_metadata:
            return text
        metadata = {
            "source": "hub",
            "raw_bytes": data["raw_bytes"],
            "manifest_article_count": data["manifest_article_count"],
            "fetched_article_count": data["fetched_article_count"],
            "skipped_articles": data["skipped_articles"],
        }
        return text, metadata
    except Exception as e:  # noqa: BLE001  # Hub 取得の失敗理由を問わずフォールバックする
        print(
            f"Hub からの取得に失敗した({e!r})。Wikipedia API からの直接取得にフォールバックする。"
        )
        if not return_metadata:
            text = load_english_wikipedia_corpus(cache_dir)
            print("コーパス取得元: Wikipedia API(直接取得)")
            return text
        text, fetch_metadata = load_english_wikipedia_corpus(cache_dir, return_metadata=True)
        print("コーパス取得元: Wikipedia API(直接取得)")
        metadata = {
            "source": "direct",
            "raw_bytes": len(text.encode("utf-8")),
            "manifest_article_count": fetch_metadata["manifest_article_count"],
            "fetched_article_count": fetch_metadata["fetched_article_count"],
            "skipped_articles": fetch_metadata["skipped_articles"],
        }
        return text, metadata


def upload_corpus_artifact_to_hub(
    repo_id: str,
    corpus_txt_path: str | Path,
    metadata_json_path: str | Path,
    dataset_card_text: str,
    token: str,
) -> None:
    """指定した Dataset リポジトリを作成し(存在しなければ)、corpus.txt・metadata.json・
    データセットカードをアップロードする(``scripts/promote_canonical_corpora.py``で使用)。

    ``src/data/tokenizer.py``の``upload_tokenizer_artifact_to_hub()``と対になる関数
    (``repo_type="dataset"``である点、アップロードするファイル(``corpus.txt``+
    ``metadata.json``)が異なる)。呼び出し側が``DRY_RUN``・``IN_COLAB``のガードを
    行った上でのみ呼び出すこと。トークンの取得・環境変数への設定はここでは行わない
    (呼び出し側が用意して渡す)。

    以前の構成(単一の``corpus.json``)からの移行に伴い、アップロード後に古い
    ``corpus.json``がリポジトリに残っていれば削除する(``upload_file``はファイル単位の
    追加・上書きであり、構成を変えても明示的に削除しない限り古いファイルが残り続ける
    ため。CLAUDE.md「共有アーティファクトの管理方針」)。リポジトリが元々
    ``corpus.json``を含まない(初回アップロードなど)場合、削除は失敗するがこれは
    正常な状態のため無視する。
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(corpus_txt_path),
        path_in_repo="corpus.txt",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(metadata_json_path),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=dataset_card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    try:
        api.delete_file(path_in_repo="corpus.json", repo_id=repo_id, repo_type="dataset")
        print(f"{repo_id}: 旧形式の corpus.json を削除した")
    except Exception as e:  # noqa: BLE001  # 元々 corpus.json が無い場合を含め、削除失敗は無視する
        print(f"{repo_id}: corpus.json の削除をスキップした({e!r}、元々存在しない場合を含む)")


def load_code_corpus(repo_root: str | Path = ".") -> str:
    """コードコーパスを取得する(005 トークナイザのコードドメイン用)。

    外部データセットに依存せず、本リポジトリ自身の ``src/`` 配下の Python
    ソースコード(001〜004 でスクラッチ実装した層・学習ループなど)を 1 つの
    コーパスとして連結する。実在するコードであることが保証され、追加のダウンロードも
    不要である。

    Args:
        repo_root: リポジトリルートのパス。ノートブックの Colab セットアップセルで
            ``%cd ai-theories`` 済みであれば既定値(カレントディレクトリ)のままでよい。

    Returns:
        ``src/`` 配下の全 ``.py`` ファイルをパス順に連結した 1 つの文字列。
    """
    src_dir = Path(repo_root) / "src"
    paths = sorted(p for p in src_dir.rglob("*.py") if "__pycache__" not in p.parts)
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def split_train_val(ids: list[int], val_ratio: float = 0.1) -> tuple[Tensor, Tensor]:
    """文字 ID 列を学習用・検証用に分割する。

    時系列順を保ったまま末尾側を検証用に割り当てる(シャッフルしない)。

    Args:
        ids: ``CharacterLevelTokenizer.encode`` で得た整数 ID のリスト。
        val_ratio: 検証用に割り当てる割合。

    Returns:
        (train_ids, val_ids) のタプル。いずれも 1 次元の LongTensor。
    """
    data = torch.tensor(ids, dtype=torch.long)
    n_val = int(len(data) * val_ratio)
    return data[:-n_val], data[-n_val:]


def get_random_batch(
    data: Tensor | np.ndarray,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """連続する区間をランダムに切り出して、次トークン予測用のバッチを作る。

    Args:
        data: 1 次元の LongTensor(``split_train_val`` の出力など)、または
            ``numpy.ndarray``・``numpy.memmap``(``encode_text_to_memmap`` の出力)。
            後者の場合、切り出した区間だけをディスクから読み込む(``memmap`` の
            場合、コーパス全体を RAM に展開しない、009)。``Tensor`` を渡した場合の
            挙動は変更していない(後方互換性)。
        batch_size: バッチサイズ B。
        seq_len: 系列長 S。
        generator: 乱数生成に使う ``torch.Generator``(省略可、再現性のため)。

    Returns:
        (inputs, targets) のタプル。いずれも形状 ``(B, S)`` の LongTensor。
        ``targets`` は ``inputs`` を 1 つ右にずらしたもの(次トークン予測の教師信号)。
    """
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    if isinstance(data, Tensor):
        inputs = torch.stack([data[s : s + seq_len] for s in starts])
        targets = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    else:
        starts_list = starts.tolist()
        inputs = torch.from_numpy(
            np.stack([np.asarray(data[s : s + seq_len], dtype=np.int64) for s in starts_list])
        )
        targets = torch.from_numpy(
            np.stack(
                [np.asarray(data[s + 1 : s + seq_len + 1], dtype=np.int64) for s in starts_list]
            )
        )
    return inputs, targets


def split_train_val_text(text: str, validation_ratio: float) -> tuple[str, str]:
    """テキストを文字列の段階で学習用・検証用に分割する(006)。

    時系列順を保ったまま末尾側を検証用に割り当てる(シャッフルしない、
    ``split_train_val`` と同じ方針)。**トークン化の前に文字列のまま分割する** 点が
    ``split_train_val`` との違いであり、これが必須である理由は 2 つある。

    1. トークナイザ条件(文字レベル・BPE・語彙サイズの違いなど)をまたいで比較する
       とき、条件ごとにトークン化してから分割すると、同じトークン数の割合で切っても
       対応する文字範囲(検証テキストの中身そのもの)が条件ごとに変わってしまう。
       文字列の段階で分割しておけば、全条件が完全に同一の検証テキストに対する
       bits-per-byte を測ることになり、条件間で公平に比較できる。
    2. サブワード分割は文脈(前後の文字)によって同じ文字列でも異なる区切り方に
       なりうるため、先にトークン化してから ID 列を分割すると、学習用・検証用の
       境界をまたぐ位置でトークンが本来と異なる形に分割される可能性がある。

    Args:
        text: 分割対象のテキスト全体。
        validation_ratio: 検証用に割り当てる割合(0 以上 1 未満)。

    Returns:
        (train_text, val_text) のタプル。
    """
    n_val = int(len(text) * validation_ratio)
    return text[:-n_val], text[-n_val:]


def encode_corpus(tokenizer, text: str) -> Tensor:
    """トークナイザでコーパス全体を符号化して ID の配列を返す(006)。

    ``tokenizer`` は ``encode(text) -> Sequence[int]``(整数 ID の列を直接返す)を
    持つ想定(``CharacterLevelTokenizer`` と同じインターフェース)。005 の
    ``BPETokenizer``・``UnigramTokenizer`` は ``encode()`` が部分語シンボルの文字列
    (``list[str]``)を返し、シンボル <-> 整数 ID の対応を持たないため、そのままでは
    ここに渡せない(整数 ID を割り当てる語彙ラッパーの追加は 006 のノートブック
    段階で扱う)。

    Args:
        tokenizer: ``encode(text) -> Sequence[int]`` を持つトークナイザ。
        text: 符号化対象のテキスト全体。

    Returns:
        1 次元の LongTensor(トークン ID 列)。
    """
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


# 単語境界(非空白文字の直後に空白文字またはテキスト末尾が続く位置)にのみマッチする。
# ``encode_text_to_memmap`` がテキストをチャンクに分割する際、この境界でのみ分割する
# ことで、チャンクごとに符号化した結果を連結したものが、テキスト全体を一括で符号化した
# 結果と一致することを保証する(``src/data/tokenizer.py`` の ``pretokenize``、
# ``chunk_split_mode="whitespace"`` は、空白を単語の先頭に付与したうえで単語単位で
# 独立に符号化し、単語をまたぐマージを行わないため)。文字単位のトークナイザ
# (``CharacterLevelTokenizer``)では文字間の相互作用がそもそも無いため、この境界は
# より保守的な制約になるだけで結果は変わらない。
_WORD_BOUNDARY_RE = re.compile(r"\S\s")


def encode_text_to_memmap(
    tokenizer,
    text: str,
    memmap_path: str | Path,
    chunk_chars: int = 2_000_000,
    force: bool = False,
) -> np.memmap:
    """テキストを単語境界で分割しながら逐次符号化し、``uint16`` の``numpy.memmap``として
    ディスクに書き出す(009、Google Colab の RAM 制約(12 GB)への対応)。

    ``encode_corpus`` はトークン ID を Python の ``list[int]`` として保持してから
    ``torch.tensor`` に変換するため、大規模コーパス(1 億トークン超)では整数
    オブジェクトのオーバーヘッド(CPython では 1 個あたり約 28 バイト、256 を超える
    整数は使い回されない)により数 GB の RAM を消費する。本関数はテキストを
    ``chunk_chars`` 文字程度のチャンクに区切って順次 ``tokenizer.encode()`` を呼び出し、
    結果を直接``uint16``(2 バイト)の生バイナリとしてディスクへ逐次書き込むことで、
    ID 列全体を Python の ``list[int]`` として RAM に保持することを避ける。書き出した
    ファイルは``uint16``の生バイナリそのものであり、そのまま``numpy.memmap``として
    開ける。

    **チャンク分割は単語境界(``_WORD_BOUNDARY_RE``)でのみ行う。** これにより、
    チャンクごとに符号化した結果を連結したものは、テキスト全体を一括で
    ``tokenizer.encode()`` した結果(``encode_corpus`` の出力)と完全に一致する
    (根拠は``_WORD_BOUNDARY_RE``のコメントを参照)。

    ``memmap_path``と、その隣の``<memmap_path>.meta.json``(トークン数と、
    再利用可否を判定するためのフィンガープリント(テキスト長・先頭/末尾の一部・
    語彙サイズ)を記録)が既に存在し、フィンガープリントが一致する場合は再符号化を
    スキップする(セッション内でのキャッシュ再利用。フィンガープリントはテキスト
    全体のハッシュではなく先頭/末尾の一部に基づく簡易な同一性チェックであり、
    ``.cache/``が Colab のセッションをまたいで永続しないことを踏まえ、
    セッション内での再利用ができれば十分という方針(CLAUDE.md)に基づく)。

    Args:
        tokenizer: ``encode(text) -> Sequence[int]``と``vocab_size: int``を持つ
            トークナイザ(``encode_corpus``と同じ想定、``CharacterLevelTokenizer``・
            ``BPEIDTokenizer``など)。
        text: 符号化対象のテキスト全体。
        memmap_path: 書き出し先のパス。
        chunk_chars: 1 チャンクあたりの目安文字数。実際の分割位置は直近の単語境界に
            丸められるため、正確にこの文字数にはならない。
        force: True の場合、既存のキャッシュを無視して再符号化する。

    Returns:
        書き出した``uint16``配列を読み取り専用で開いた``numpy.memmap``
        (形状``(トークン数,)``)。

    Raises:
        AssertionError: ``tokenizer.vocab_size``が``uint16``の表現範囲
            (0〜65535)を超える場合、または符号化されたトークン ID がその範囲を
            超える場合。
    """
    vocab_size = tokenizer.vocab_size
    assert vocab_size <= 65536, (
        f"vocab_size={vocab_size} は uint16(0〜65535)の範囲を超えている。"
        "encode_text_to_memmap は uint16 での保持を前提としており、"
        "この語彙サイズでは使えない。"
    )

    memmap_path = Path(memmap_path)
    meta_path = memmap_path.with_name(memmap_path.name + ".meta.json")
    fingerprint = f"{len(text)}:{text[:1000]}:{text[-1000:]}"

    if not force and memmap_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint and meta.get("vocab_size") == vocab_size:
            token_count = meta["token_count"]
            print(
                f"[キャッシュ] {memmap_path} を再利用する({token_count:,} トークン、再符号化しない)"
            )
            return np.memmap(memmap_path, dtype=np.uint16, mode="r", shape=(token_count,))

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(text)
    total_tokens = 0
    with open(memmap_path, "wb") as f:
        pos = 0
        while pos < n:
            end = min(pos + chunk_chars, n)
            if end < n:
                m = _WORD_BOUNDARY_RE.search(text, end - 1)
                end = m.start() + 1 if m else n
            chunk_ids = tokenizer.encode(text[pos:end])
            if chunk_ids:
                arr = np.asarray(chunk_ids, dtype=np.int64)
                assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size, (
                    "符号化されたトークン ID が語彙サイズの範囲外(uint16 化の前提が崩れている)"
                )
                f.write(arr.astype(np.uint16).tobytes())
                total_tokens += arr.size
            pos = end

    meta_path.write_text(
        json.dumps(
            {"token_count": total_tokens, "fingerprint": fingerprint, "vocab_size": vocab_size}
        ),
        encoding="utf-8",
    )
    print(f"{memmap_path}: {total_tokens:,} トークンを uint16 memmap として書き出した")
    return np.memmap(memmap_path, dtype=np.uint16, mode="r", shape=(total_tokens,))


def make_evaluation_windows(
    token_ids: Tensor | np.ndarray,
    sequence_length: int,
) -> tuple[Tensor, Tensor]:
    """検証用の非重複窓(non-overlapping window)を作る(006)。

    ``token_ids`` を長さ ``sequence_length`` の窓に先頭から順に区切る(非重複)。
    末尾の端数トークンは **切り捨てず、パディングして最終窓を埋める**(006、当初は
    切り捨てていたが、トークナイザ条件によって fertility が異なり、切り捨てられる
    原文バイト数が条件ごとに変わってしまう問題があったため変更した)。パディング位置を
    示す bool マスクを併せて返す(True が実トークン、False がパディング)。

    パディング値には ``0`` を使う(語彙に含まれる有効な ID である必要はあるが、
    マスクにより損失計算から除外されるため、どの値でもよい)。

    bits-per-byte の分母(評価対象の UTF-8 バイト数)は、この関数では計算しない。
    トークナイザ条件によらず **検証テキスト全体の UTF-8 バイト数が全条件で同一の
    定数になる** ため(006、3.5 節)、呼び出し側が符号化前のテキストから直接
    ``len(text.encode("utf-8"))`` として求める(``evaluate_bits_per_byte`` に渡す)。

    Args:
        token_ids: 1 次元の LongTensor(``encode_corpus`` の出力)、または
            ``numpy.ndarray``・``numpy.memmap``(``encode_text_to_memmap`` の出力)。
            ``Tensor`` 以外が渡された場合、内部で ``Tensor`` に変換する。評価窓は
            検証データ全体を必要とするためそもそも全体を RAM に展開する前提であり、
            ``memmap`` を渡しても RAM 使用量の削減にはならない
            (検証データが巨大な場合は呼び出し側で分割を検討すること)。
        sequence_length: 窓の長さ S。

    Returns:
        (windows, mask) のタプル。いずれも形状 ``(num_windows, sequence_length)``。
        ``windows`` は LongTensor(末尾窓のパディング位置は ``0``)、``mask`` は
        bool の Tensor(True が実トークン、False がパディング)。
    """
    if not isinstance(token_ids, Tensor):
        token_ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64))
    n = len(token_ids)
    num_windows = -(-n // sequence_length)  # 切り上げ除算
    padded_length = num_windows * sequence_length
    pad_amount = padded_length - n

    if pad_amount > 0:
        padding = torch.zeros(pad_amount, dtype=token_ids.dtype)
        padded_ids = torch.cat([token_ids, padding])
        flat_mask = torch.cat(
            [torch.ones(n, dtype=torch.bool), torch.zeros(pad_amount, dtype=torch.bool)]
        )
    else:
        padded_ids = token_ids
        flat_mask = torch.ones(n, dtype=torch.bool)

    windows = padded_ids.view(num_windows, sequence_length)
    mask = flat_mask.view(num_windows, sequence_length)
    return windows, mask
