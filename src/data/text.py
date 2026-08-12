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

# 日本語コーパス(load_japanese_corpus)の取得元として使う日本語版 Wikipedia の記事
# タイトル一覧(005 トークナイザで使用)。分野を偏らせないよう、自然科学・人文・
# 社会・技術・スポーツなど幅広いカテゴリから選んでいる。
_JAPANESE_WIKIPEDIA_TITLES = [
    "日本",
    "東京都",
    "大阪府",
    "京都府",
    "日本語",
    "英語",
    "数学",
    "物理学",
    "化学",
    "生物学",
    "地学",
    "天文学",
    "宇宙",
    "地球",
    "気象学",
    "医学",
    "心理学",
    "経済学",
    "政治学",
    "法学",
    "歴史学",
    "哲学",
    "宗教",
    "言語学",
    "文学",
    "音楽",
    "映画",
    "美術",
    "建築",
    "料理",
    "農業",
    "漁業",
    "鉄道",
    "自動車",
    "航空機",
    "コンピュータ",
    "インターネット",
    "人工知能",
    "機械学習",
    "深層学習",
    "野球",
    "サッカー",
    "将棋",
    "囲碁",
    "オリンピック",
]


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


def _fetch_wikipedia_extract(
    title: str, max_retries: int = 5, retry_wait_seconds: float = 5.0
) -> str:
    """日本語版 Wikipedia の 1 記事の本文(プレーンテキスト)を取得する。

    Wikimedia API は (1) デフォルトの User-Agent(``python-urllib/...``)からの
    リクエストを 403 で拒否し、(2) 短時間に連続してリクエストすると 429
    (Too Many Requests)を返すことがある。そのため識別可能な User-Agent を
    付与し、各リクエストの間隔を空け、429 発生時は指数的に待機して再試行する。
    """
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
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
            page = data["query"]["pages"][0]
            extract = page.get("extract", "")
            # explaintext=1 でも "== 節見出し ==" 形式のセクション見出しは平文のまま
            # 残るため、除去する(本文の自然な日本語文のみを残す)。
            return re.sub(r"(?m)^=+\s*.*?\s*=+$\n?", "", extract)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(retry_wait_seconds * (attempt + 1))
                continue
            raise
    return ""


def load_japanese_corpus(cache_dir: str | Path) -> str:
    """日本語コーパスを取得してキャッシュする(005 トークナイザの日本語ドメイン用)。

    日本語版 Wikipedia の MediaWiki API(``action=query``, ``prop=extracts``)を用いて、
    固定の記事タイトル一覧(``_JAPANESE_WIKIPEDIA_TITLES``)から本文(プレーンテキスト、
    Wikipedia のマークアップは除去済み)を取得して連結する。

    記事ごとに個別のファイルへキャッシュする(``cache_dir/japanese_wikipedia_articles/``)。
    Wikimedia API のレート制限(429)により取得が一部の記事で失敗しても、
    セルを再実行すれば取得済みの記事はキャッシュから読み、未取得の記事のみ再取得する
    (1 記事も欠けずに揃うまで、コーパス全体のキャッシュファイルは作らない)。

    Note:
        記事本文は将来的に編集され得るため、``load_tiny_shakespeare`` とは異なり
        厳密な意味での再現性(バイト単位での一致)は保証しない。005 の実験は
        語彙統計(fertility など)の相対比較が目的であり、コーパスが実行のたびに
        完全に同一である必要はない。

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
        for i, title in enumerate(_JAPANESE_WIKIPEDIA_TITLES):
            article_path = articles_dir / f"{i:03d}.txt"
            if not article_path.exists():
                article_path.write_text(_fetch_wikipedia_extract(title), encoding="utf-8")
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
