"""合成の系列課題 selective copying と induction heads の事例の生成(023)。

どちらも乱数の生成器(``numpy.random.Generator``)を引数に取り、シードから決定的に事例を作る。
学習ではステップごとに新しい事例を生成し(同じ事例を繰り返さない)、評価集合は学習とは別の
シードで固定して生成する。

**Selective copying**: Jing et al., "Gated Orthogonal Recurrent Units: On Learning to Forget",
Neural Computation 2019 の課題を、Gu & Dao, "Mamba", arXiv:2312.00752 の Section 4.1.1 が
選択性の検証に用いた形にしたもの。長さ ``input_length`` の入力の中のランダムな位置に
``num_data`` 個のデータのトークンが置かれ、残りはノイズのトークンである。区切りのトークンの
後に、データのトークンを **順序を保って** 出力する。系列は
``[入力 (input_length 個), 区切り, 答え (num_data 個)]`` で、損失と評価は答えのトークンの
予測にだけ掛ける(自己回帰。評価では答えを貪欲に 1 トークンずつ生成する)。

**Induction heads**: Olsson et al., "In-context Learning and Induction Heads",
Transformer Circuits Thread 2022 の課題を、Gu & Dao, "Mamba" の Section 4.1.2 の形にした
もの。ランダムなトークンの列の中に特別なトークンが 1 回現れ、その直後のトークンが答えである。
系列の末尾は再び特別なトークンで、その次に(最初に特別なトークンの直後にあったトークンを)
出力する。損失と評価は、末尾の位置の 1 回の予測にだけ掛ける。

語彙の番号(selective copying): ``0 .. num_symbols - 1`` がデータ、``num_symbols`` がノイズ、
``num_symbols + 1`` が区切り。語彙の番号(induction heads): ``0 .. num_symbols - 1`` が
ランダムなトークン、``num_symbols`` が特別なトークン。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

IGNORE_INDEX = -100
"""損失を掛けない位置の目標の値(``torch.nn.functional.cross_entropy`` の ``ignore_index``
の既定値)。"""


def make_rng(seed: int, *stream: int) -> np.random.Generator:
    """シードと添字の列から、
    決定的な乱数の生成器を作る(学習のステップごと・評価集合ごとに別の流れ)。"""
    return np.random.default_rng([seed, *stream])


@dataclass(frozen=True)
class SelectiveCopyingTask:
    """Selective copying の課題の定義。

    Attributes:
        num_symbols: データのトークンの種類の数。
        num_data: 1 事例に置くデータのトークンの数(答えの長さ)。
        input_length: 入力部分(データとノイズ)の長さ。
    """

    num_symbols: int
    num_data: int
    input_length: int

    def __post_init__(self) -> None:
        if not 1 <= self.num_data <= self.input_length:
            raise ValueError("num_data は 1 以上 input_length 以下である必要がある")

    @property
    def noise_token(self) -> int:
        return self.num_symbols

    @property
    def separator_token(self) -> int:
        return self.num_symbols + 1

    @property
    def vocabulary_size(self) -> int:
        return self.num_symbols + 2

    @property
    def total_length(self) -> int:
        return self.input_length + 1 + self.num_data

    @property
    def chance_accuracy(self) -> float:
        """答えのトークン 1 個を当てずっぽうで当てる確率(データのトークンの種類の数の逆数)。"""
        return 1.0 / self.num_symbols


def generate_selective_copying(
    task: SelectiveCopyingTask, batch_size: int, rng: np.random.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    """Selective copying の事例を ``batch_size`` 個生成する。

    Returns:
        ``(tokens, targets, data_positions)``。``tokens`` は形状 ``(batch_size, total_length)`` の
        LongTensor(入力・区切り・答え)、``targets`` は同じ形状で、位置 ``t`` に
        ``tokens[:, t + 1]`` を置く(答えのトークンの予測の位置のみ。それ以外は
        ``IGNORE_INDEX``)。``data_positions`` は形状 ``(batch_size, num_data)`` の LongTensor
        (入力部分のデータのトークンの位置、昇順)。
    """
    n, k, length = batch_size, task.num_data, task.input_length
    positions = np.sort(np.argsort(rng.random((n, length)), axis=1)[:, :k], axis=1)
    symbols = rng.integers(0, task.num_symbols, size=(n, k))
    inputs = np.full((n, length), task.noise_token, dtype=np.int64)
    np.put_along_axis(inputs, positions, symbols, axis=1)
    separator = np.full((n, 1), task.separator_token, dtype=np.int64)
    tokens = np.concatenate([inputs, separator, symbols.astype(np.int64)], axis=1)
    targets = np.full_like(tokens, IGNORE_INDEX)
    targets[:, length : length + k] = symbols  # 位置 length(区切り)が答えの 1 番目を予測する
    return (
        torch.from_numpy(tokens),
        torch.from_numpy(targets),
        torch.from_numpy(positions.astype(np.int64)),
    )


@dataclass(frozen=True)
class InductionHeadsTask:
    """Induction heads の課題の定義。

    Attributes:
        num_symbols: ランダムなトークンの種類の数。特別なトークンの番号は ``num_symbols``。
    """

    num_symbols: int

    @property
    def special_token(self) -> int:
        return self.num_symbols

    @property
    def vocabulary_size(self) -> int:
        return self.num_symbols + 1

    @property
    def chance_accuracy(self) -> float:
        """答えを当てずっぽうで当てる確率(ランダムなトークンの種類の数の逆数)。"""
        return 1.0 / self.num_symbols


def generate_induction_heads(
    task: InductionHeadsTask, length: int, batch_size: int, rng: np.random.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    """Induction heads の事例を ``batch_size`` 個、系列長 ``length`` で生成する。

    最初の特別なトークンの位置は ``[0, length - 3]`` の一様分布から引く(その直後のトークンは位置
    ``length - 2`` 以前のランダムなトークンで、末尾の特別なトークンではない)。

    Returns:
        ``(tokens, answers, first_positions)``。``tokens`` は形状 ``(batch_size, length)`` の
        LongTensor、``answers`` は形状 ``(batch_size,)`` の
        LongTensor(末尾の位置で予測すべきトークン)、
        ``first_positions`` は最初の特別なトークンの位置(形状 ``(batch_size,)``)。
    """
    if length < 4:
        raise ValueError(f"length は 4 以上である必要がある: {length}")
    n = batch_size
    tokens = rng.integers(0, task.num_symbols, size=(n, length)).astype(np.int64)
    first = rng.integers(0, length - 2, size=n)  # [0, length - 3]
    rows = np.arange(n)
    tokens[rows, first] = task.special_token
    answers = tokens[rows, first + 1].copy()
    tokens[:, length - 1] = task.special_token
    return (
        torch.from_numpy(tokens),
        torch.from_numpy(answers),
        torch.from_numpy(first.astype(np.int64)),
    )


def induction_heads_targets(tokens: Tensor, answers: Tensor) -> Tensor:
    """Induction heads の学習用の目標。末尾の位置にだけ答えを置き、他は ``IGNORE_INDEX`` にする。"""
    targets = torch.full_like(tokens, IGNORE_INDEX)
    targets[:, -1] = answers
    return targets
