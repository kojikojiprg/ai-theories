"""合成の選好データ(Preference Data)と真の報酬(017)。

016 の合成の指示データ(``src/data/instruction.py``)の上で、報酬モデル(Reward Model)と
RLHF(Reinforcement Learning from Human Feedback)の実験に使う選好データを **規則から** 作る。
人手の選好ラベルの代わりに、応答の良さを測る既知の関数(真の報酬 r*)を定め、その差から
Bradley-Terry モデルでラベルを確率的に抽選する。真の報酬が既知なので、学習した報酬モデルが
真の報酬をどこまで回復したか、最適化の圧力をかけたときに真の報酬がどう動くかを直接測れる。

**真の報酬** ``compute_true_reward()``: 生成した応答部分の文字列と正解の単語の列から、

.. math::

    r^*(y) = \\frac{1}{2} \\left( s(y) + f(y) \\right), \\qquad
    s(y) = 1 - \\frac{\\mathrm{lev}(u(y), a)}{\\max(|u(y)|, |a|, 1)}

とする。``u(y)`` は応答の本文(最初の ``###`` より前、なければ全体)を空白で区切った単語を
空白 1 つで連結した文字列、``a`` は正解の単語を空白 1 つで連結した文字列、``lev`` は文字単位の
編集距離(Levenshtein 距離)、``|.|`` は文字数、``f(y)`` は形式の遵守(016 の
``score_generated_response()`` と同じ判定)で 1 または 0 である。r* は [0, 1] の値をとり、
**r* = 1 となるのは 016 の意味で完全一致するときに限る**(s = 1 は単語の列が正解と一致すること、
f = 1 は形式の遵守と同値であるため)。正解 / 不正解の 2 値ではなく、正解に近い誤答ほど高くなる
段階的な値である。

**ラベルの抽選** ``sample_preference_labels()``: 尺度 kappa の Bradley-Terry モデル

.. math::

    P(y_1 \\succ y_2) = \\sigma\\left(\\kappa \\left(r^*(y_1) - r^*(y_2)\\right)\\right)

から、組ごとに独立にラベルを抽選する(sigma はシグモイド関数)。kappa は
``calibrate_label_scale()`` で、応答の組の真の報酬の差の分布だけから決める。

記号 / Notation:
    x      : プロンプト(指示部分)
    y      : 応答
    r*     : 真の報酬(``compute_true_reward()``)
    kappa  : ラベル生成の尺度の定数(KL 係数 beta とは別の量)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from src.data.instruction import DELIMITER_PREFIX, score_generated_response


def levenshtein_distance(a: str, b: str) -> int:
    """文字単位の編集距離(挿入・削除・置換をそれぞれコスト 1 とする Levenshtein 距離)。

    動的計画法の表を 1 行ずつ numpy で更新する。行 ``i`` の値 ``D[i, j]`` は

    .. math::

        D[i, j] = \\min(D[i-1, j] + 1,\\ D[i-1, j-1] + [a_i \\ne b_j],\\ D[i, j-1] + 1)

    で、第 3 項(同じ行の左隣への依存)は、先に第 1・2 項の最小値 ``t_j`` を求めてから
    ``D[i, j] = min_{k <= j} (t_k + (j - k))`` と書ける。これは ``t_k - k`` の累積最小値に
    ``j`` を足したものなので、``np.minimum.accumulate`` で 1 行をまとめて計算できる。
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    b_codes = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    columns = np.arange(len(b) + 1, dtype=np.int64)
    previous = columns.copy()
    for i, ch in enumerate(a, start=1):
        cost = (b_codes != ord(ch)).astype(np.int64)
        candidate = np.empty_like(previous)
        candidate[0] = i
        candidate[1:] = np.minimum(previous[1:] + 1, previous[:-1] + cost)
        previous = np.minimum.accumulate(candidate - columns) + columns
    return int(previous[-1])


def extract_response_words(generated_text: str) -> list[str]:
    """応答の本文(最初の ``###`` より前、なければ全体)を空白で区切った単語の列を返す。"""
    first = generated_text.find(DELIMITER_PREFIX)
    body = generated_text if first == -1 else generated_text[:first]
    return body.split()


def compute_true_reward(generated_text: str, answer: Sequence[str]) -> float:
    """真の報酬 r*(y) = (s(y) + f(y)) / 2 を返す(モジュールの docstring を参照)。

    Args:
        generated_text: 生成した応答部分の文字列(016 の採点と同じく、指示部分を含まない)。
        answer: 正解の単語の列。

    Returns:
        [0, 1] の値。016 の意味で完全一致するときに限り 1。
    """
    predicted = " ".join(extract_response_words(generated_text))
    target = " ".join(answer)
    similarity = 1.0 - levenshtein_distance(predicted, target) / max(len(predicted), len(target), 1)
    format_ok, _ = score_generated_response(generated_text, answer)
    return 0.5 * (similarity + float(format_ok))


def compute_bradley_terry_probability(reward_difference: np.ndarray, kappa: float) -> np.ndarray:
    """尺度 kappa の Bradley-Terry モデルの P(y_1 > y_2) = sigma(kappa (r*(y_1) - r*(y_2)))。"""
    z = kappa * np.asarray(reward_difference, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_label_scale(reward_differences: Sequence[float], target: float = 0.75) -> float:
    """ラベルの尺度 kappa を、真の報酬の差の分布だけから決める。

    差が 0 でない組について ``sigma(kappa |Delta r*|)`` の中央値が ``target`` になるように
    kappa を選ぶ。sigma は単調増加なので、中央値は ``|Delta r*|`` の中央値 ``m`` で決まり、

    .. math::

        \\kappa = \\frac{\\mathrm{logit}(\\mathrm{target})}{m}
                = \\frac{\\log(\\mathrm{target} / (1 - \\mathrm{target}))}{m}

    となる(target = 0.75 なら logit は log 3)。差が 0 の組(同点)は kappa によらず確率 0.5
    なので、中央値の計算から除く。学習した報酬モデルの結果は一切使わない。
    """
    if not 0.5 < target < 1.0:
        raise ValueError(f"target は (0.5, 1) の値である必要がある: {target}")
    magnitudes = np.abs(np.asarray(reward_differences, dtype=np.float64))
    magnitudes = magnitudes[magnitudes > 0]
    if magnitudes.size == 0:
        raise ValueError("差が 0 でない組がない(すべて同点)")
    return math.log(target / (1.0 - target)) / float(np.median(magnitudes))


def sample_preference_labels(
    first_rewards: Sequence[float],
    second_rewards: Sequence[float],
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """組ごとに Bradley-Terry モデルからラベルを抽選する。

    Returns:
        bool の配列。True なら 1 つ目の応答が選ばれた(y_w = y_1, y_l = y_2)。同点の組は確率
        0.5 で抽選される。
    """
    probabilities = compute_bradley_terry_probability(
        np.asarray(first_rewards, dtype=np.float64) - np.asarray(second_rewards, dtype=np.float64),
        kappa,
    )
    return rng.random(probabilities.shape[0]) < probabilities


def collate_scored_sequences(
    sequences: Sequence[Sequence[int]], pad_token_id: int = 0
) -> tuple[Tensor, Tensor]:
    """系列(プロンプト + 応答のトークン列)を右パディングしてバッチにし、最終トークンの位置を返す。

    右パディングなので、因果マスクにより実トークンの位置の隠れ状態はパディングの影響を受けない。
    報酬モデル(``src/models/reward_model.py``)は、各系列の最終トークンの位置の隠れ状態から
    スカラーを出す。

    Returns:
        ``(token_ids, last_indices)``。形状は ``(B, S_max)`` と ``(B,)``。
    """
    if any(len(s) == 0 for s in sequences):
        raise ValueError("空の系列がある")
    max_length = max(len(s) for s in sequences)
    token_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        token_ids[row, : len(sequence)] = torch.tensor(list(sequence), dtype=torch.long)
    last_indices = torch.tensor([len(s) - 1 for s in sequences], dtype=torch.long)
    return token_ids, last_indices
