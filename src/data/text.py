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

# 日本語コーパス(load_japanese_corpus)の取得元として使う日本語版 Wikipedia の記事。
# 分野を偏らせないよう、自然科学・人文・社会・技術・スポーツなど幅広いカテゴリから
# 選んでいる(005 トークナイザで使用)。
#
# 出典 / Source: フリー百科事典『ウィキペディア(Wikipedia)』日本語版
#     (https://ja.wikipedia.org/)
# ライセンス / License: クリエイティブ・コモンズ 表示-継承 4.0 国際
#     (CC BY-SA 4.0、https://creativecommons.org/licenses/by-sa/4.0/deed.ja)
#
# 記事本文は編集され続けるため、記事タイトルだけでは実行のたびに異なる文字列に
# なりうる(load_tiny_shakespeare や load_code_corpus とは違い、コーパスが厳密に
# 固定されない)。再現性を保つため、記事タイトルと合わせてリビジョン ID(特定時点の
# 版を指す ID)を固定する。各記事の当該リビジョンは
# ``https://ja.wikipedia.org/w/index.php?title=<記事名>&oldid=<リビジョンID>``
# で参照できる(以下のリビジョン ID はいずれも Wikimedia API に実際にアクセスして
# 取得した実在の値、取得日 2026-08-13)。
_JAPANESE_WIKIPEDIA_REVISIONS: dict[str, int] = {
    "日本": 110578005,
    "東京都": 110597729,
    "大阪府": 110565604,
    "京都府": 109765776,
    "日本語": 109750467,
    "英語": 110199747,
    "数学": 109208023,
    "物理学": 109540178,
    "化学": 108462219,
    "生物学": 110028225,
    "地学": 75704347,
    "天文学": 109540232,
    "宇宙": 110379900,
    "地球": 110170728,
    "気象学": 108617045,
    "医学": 108812371,
    "心理学": 110472899,
    "経済学": 110060128,
    "政治学": 110403427,
    "法学": 110414694,
    "歴史学": 110621288,
    "哲学": 110318691,
    "宗教": 109958573,
    "言語学": 109216659,
    "文学": 109467146,
    "音楽": 110093563,
    "映画": 109612030,
    "美術": 110191146,
    "建築": 110524075,
    "料理": 110081576,
    "農業": 110615080,
    "漁業": 109958685,
    "鉄道": 110480788,
    "自動車": 110315235,
    "航空機": 110600467,
    "コンピュータ": 110402036,
    "インターネット": 110324659,
    "人工知能": 110610279,
    "機械学習": 110120637,
    "深層学習": 55590908,
    "野球": 110528766,
    "サッカー": 110613233,
    "将棋": 109676436,
    "囲碁": 110232196,
    "オリンピック": 106676037,
}

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
    title: str, revid: int, max_retries: int = 5, retry_wait_seconds: float = 5.0
) -> str:
    """日本語版 Wikipedia の指定リビジョンの本文を取得し、平文に変換する。

    ``action=query&prop=extracts`` は常に最新版の本文を返し、``revids`` パラメータを
    指定しても無視される(実際にリクエストして確認済み: 数年前のリビジョン ID を
    指定しても最新版と同一の本文が返る)。特定のリビジョンを取得するには
    ``action=parse&oldid=<リビジョンID>`` を使う必要がある。この API は wikitext
    (MediaWiki のマークアップ)しか返さないため、``_wikitext_to_plaintext`` で
    平文に変換する。

    Wikimedia API は (1) デフォルトの User-Agent(``python-urllib/...``)からの
    リクエストを 403 で拒否し、(2) 短時間に連続してリクエストすると 429
    (Too Many Requests)を返すことがある。そのため識別可能な User-Agent を
    付与し、各リクエストの間隔を空け、429 発生時は指数的に待機して再試行する。
    """
    params = urllib.parse.urlencode(
        {
            "action": "parse",
            "oldid": revid,
            "prop": "wikitext",
            "format": "json",
            "formatversion": 2,
        }
    )
    url = f"https://ja.wikipedia.org/w/api.php?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-theories-tokenizer-notebook/1.0"}
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
            time.sleep(1.0)  # 連続リクエストによるレート制限(429)を避ける
            wikitext = data["parse"]["wikitext"]
            return _wikitext_to_plaintext(wikitext)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise
    return ""


def load_japanese_corpus(cache_dir: str | Path) -> str:
    """日本語コーパスを取得してキャッシュする(005 トークナイザの日本語ドメイン用)。

    出典: フリー百科事典『ウィキペディア(Wikipedia)』日本語版
    (https://ja.wikipedia.org/)。ライセンス: クリエイティブ・コモンズ 表示-継承 4.0
    国際(CC BY-SA 4.0)。固定の記事タイトル・リビジョン ID の対応
    (``_JAPANESE_WIKIPEDIA_REVISIONS``)から、日本語版 Wikipedia の MediaWiki API
    (``action=parse``、``oldid`` でリビジョンを指定)を用いて各記事の指定リビジョンの
    本文を取得し、wikitext を平文に変換したうえで連結する。

    タイトルとリビジョン ID を両方固定しているため、実行時点によらず同一の入力が
    得られる(``action=query&prop=extracts`` はリビジョンを固定できず最新版を返して
    しまうため使わない。関数の docstring を参照)。

    記事ごとに個別のファイルへキャッシュする(``cache_dir/japanese_wikipedia_articles/``、
    ファイル名にリビジョン ID を含めるため、``_JAPANESE_WIKIPEDIA_REVISIONS`` を
    更新した場合は自動的に再取得される)。Wikimedia API のレート制限(429)により
    取得が一部の記事で失敗しても、セルを再実行すれば取得済みの記事はキャッシュから
    読み、未取得の記事のみ再取得する(1 記事も欠けずに揃うまで、コーパス全体の
    キャッシュファイルは作らない)。

    Args:
        cache_dir: キャッシュ先ディレクトリ。存在しない場合は作成する。

    Returns:
        取得した記事本文を連結した 1 つの文字列。
    """
    cache_dir = Path(cache_dir)
    articles_dir = cache_dir / "japanese_wikipedia_articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "japanese_wikipedia.txt"

    if not cache_path.exists():
        texts = []
        for i, (title, revid) in enumerate(_JAPANESE_WIKIPEDIA_REVISIONS.items()):
            article_path = articles_dir / f"{i:03d}_{revid}.txt"
            if not article_path.exists():
                article_path.write_text(
                    _fetch_wikipedia_revision_plaintext(title, revid), encoding="utf-8"
                )
            texts.append(article_path.read_text(encoding="utf-8"))
        cache_path.write_text("\n".join(t for t in texts if t), encoding="utf-8")

    return cache_path.read_text(encoding="utf-8")


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
