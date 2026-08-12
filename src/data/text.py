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

import urllib.request
from pathlib import Path

import torch
from torch import Tensor

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


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
