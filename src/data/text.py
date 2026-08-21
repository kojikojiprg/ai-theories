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

    Args:
        language: Wikipedia の言語コード(``"ja"``・``"en"`` など)。
        title: 記事タイトル(取得自体には使わない。呼び出し側がキャッシュファイル名の
            対応付けに使うために受け取るだけの引数)。
        revid: 取得するリビジョン ID。
        max_retries: 429・タイムアウト発生時の最大再試行回数。
        retry_wait_seconds: 429・タイムアウト発生時の基本待機秒数(試行回数に比例して延びる)。
        request_timeout: 1 リクエストあたりの ``urlopen`` タイムアウト秒数。
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
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
            time.sleep(1.0)  # 連続リクエストによるレート制限(429)を避ける
            wikitext = data["parse"]["wikitext"]
            return _wikitext_to_plaintext(wikitext)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise
        except OSError:
            # タイムアウト(TimeoutError)、接続確立時の失敗(URLError、いずれも
            # OSError のサブクラス)、接続確立後にサーバーが応答を止めて切断する
            # RemoteDisconnected(http.client、これも OSError のサブクラス)を
            # まとめて一時的なネットワークエラーとして再試行する。
            if attempt < max_retries - 1:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise
    return ""


def load_wikipedia_corpus(
    language: str,
    cache_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> str:
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
    未取得の記事のみ再取得する(1 記事も欠けずに揃うまで、コーパス全体のキャッシュ
    ファイルは作らない)。

    Args:
        language: Wikipedia の言語コード(``"ja"``・``"en"`` など)。
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。
        manifest_path: 記事タイトル・リビジョン ID の対応を記した JSON ファイルへの
            パス。``None``(既定値)の場合、
            ``src/data/wikipedia_manifests/<language>_006_pretraining.json``
            (UTF-8 で 20 MB 以上を目標に Wikipedia の長大記事(Longpages)から
            選定した記事集合、本リポジトリにコミット済み)を使う。

    Returns:
        取得した記事本文を連結した 1 つの文字列。
    """
    if manifest_path is None:
        manifest_path = _WIKIPEDIA_MANIFEST_DIR / f"{language}_006_pretraining.json"
    manifest: dict[str, int] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    cache_dir = Path(cache_dir)
    articles_dir = cache_dir / f"wikipedia_{language}_articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"wikipedia_{language}.txt"

    if not cache_path.exists():
        texts = []
        for i, (title, revid) in enumerate(manifest.items()):
            article_path = articles_dir / f"{i:03d}_{revid}.txt"
            if not article_path.exists():
                article_path.write_text(
                    _fetch_wikipedia_revision_plaintext(language, title, revid),
                    encoding="utf-8",
                )
            texts.append(article_path.read_text(encoding="utf-8"))
        cache_path.write_text("\n".join(t for t in texts if t), encoding="utf-8")

    return cache_path.read_text(encoding="utf-8")


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


def load_english_scaling_corpus(cache_dir: str | Path) -> str:
    """英語コーパスを取得してキャッシュする(009 スケーリング則の学習グリッド用)。

    ``load_wikipedia_corpus("en", cache_dir, manifest_path=...)`` の薄いラッパー。
    006 で使った 356 記事(``en_006_pretraining.json``)に、Wikipedia の
    Special:LongPages(長大記事一覧)から追加で選定した 644 記事を加えた計 1000 記事
    (``src/data/wikipedia_manifests/en_009_scaling.json``)を対象とする。006 の
    マニフェストを部分集合として含む形で拡張しているため、006・008 で既に取得済みの
    記事のキャッシュ(``wikipedia_en_articles/``)をそのまま再利用でき、追加で
    取得が必要なのは新規記事分のみである。

    Args:
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。

    Returns:
        取得した記事本文を連結した 1 つの文字列。
    """
    manifest_path = _WIKIPEDIA_MANIFEST_DIR / "en_009_scaling.json"
    return load_wikipedia_corpus("en", cache_dir, manifest_path=manifest_path)


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
    data: Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """連続する区間をランダムに切り出して、次トークン予測用のバッチを作る。

    Args:
        data: 1 次元の LongTensor(``split_train_val`` の出力など)。
        batch_size: バッチサイズ B。
        seq_len: 系列長 S。
        generator: 乱数生成に使う ``torch.Generator``(省略可、再現性のため)。

    Returns:
        (inputs, targets) のタプル。いずれも形状 ``(B, S)`` の LongTensor。
        ``targets`` は ``inputs`` を 1 つ右にずらしたもの(次トークン予測の教師信号)。
    """
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    inputs = torch.stack([data[s : s + seq_len] for s in starts])
    targets = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
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


def make_evaluation_windows(
    token_ids: Tensor,
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
        token_ids: 1 次元の LongTensor(``encode_corpus`` の出力)。
        sequence_length: 窓の長さ S。

    Returns:
        (windows, mask) のタプル。いずれも形状 ``(num_windows, sequence_length)``。
        ``windows`` は LongTensor(末尾窓のパディング位置は ``0``)、``mask`` は
        bool の Tensor(True が実トークン、False がパディング)。
    """
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
