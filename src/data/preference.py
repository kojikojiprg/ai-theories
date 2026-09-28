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


# --- 018(DPO)で追加: 重複させたラベル・決定的なラベル・組の類似度 ---


def sample_preference_label_matrix(
    first_rewards: Sequence[float],
    second_rewards: Sequence[float],
    kappa: float,
    num_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """組ごとに ``num_draws`` 回、独立に Bradley-Terry モデルからラベルを抽選する(018)。

    同じ組 ``(x, y_1, y_2)`` を K 回観測したことに相当する。経験的な選好確率
    ``p_hat = (y_1 が選ばれた回数) / K`` は、K 回の抽選が一致しないときに限り (0, 1) の値をとる。

    Returns:
        形状 ``(N, K)`` の bool の配列。``[i, k]`` が True なら、組 i の k 回目の観測で
        y_1 が選ばれた。
    """
    if num_draws < 1:
        raise ValueError(f"num_draws は 1 以上である必要がある: {num_draws}")
    probabilities = compute_bradley_terry_probability(
        np.asarray(first_rewards, dtype=np.float64) - np.asarray(second_rewards, dtype=np.float64),
        kappa,
    )
    return rng.random((probabilities.shape[0], num_draws)) < probabilities[:, None]


def deterministic_preference_label_matrix(
    first_rewards: Sequence[float], second_rewards: Sequence[float], num_draws: int
) -> np.ndarray:
    """真の報酬の高い方を選好とする決定的なラベルを ``num_draws`` 回重複させる(018)。

    ``sample_preference_label_matrix()`` で kappa -> 無限大とした極限に相当する。事例数を確率的な
    ラベルの条件と揃えるため、同じラベルを K 回重複させる。

    Raises:
        ValueError: 真の報酬が等しい組(Delta r* = 0)がある場合(決定的なラベルが定まらない)。
    """
    if num_draws < 1:
        raise ValueError(f"num_draws は 1 以上である必要がある: {num_draws}")
    difference = np.asarray(first_rewards, dtype=np.float64) - np.asarray(
        second_rewards, dtype=np.float64
    )
    if np.any(difference == 0):
        raise ValueError("真の報酬が等しい組がある(決定的なラベルが定まらない)")
    return np.repeat((difference > 0)[:, None], num_draws, axis=1)


def flatten_preference_labels(label_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(N, K)`` のラベルを、組の順・同じ組の中では観測の順の事例の列に展開する(018)。

    Returns:
        ``(pair_indices, first_chosen)``。長さ ``N K`` の配列で、事例 j は組 ``pair_indices[j]``、
        ``first_chosen[j]`` が True なら y_w = y_1, y_l = y_2。
    """
    label_matrix = np.asarray(label_matrix, dtype=bool)
    num_pairs, num_draws = label_matrix.shape
    return np.repeat(np.arange(num_pairs), num_draws), label_matrix.reshape(-1).copy()


def normalized_edit_distance(a: str, b: str) -> float:
    """文字単位の編集距離を長い方の文字数で割った値(018)。両方が空なら 0。値は [0, 1]。"""
    longest = max(len(a), len(b))
    return 0.0 if longest == 0 else levenshtein_distance(a, b) / longest


def select_similarity_stratified_pairs(
    strata_labels: Sequence[object],
    distances: Sequence[float],
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """層ごとに、類似度の近い組・遠い組を同数ずつ選ぶ(018)。

    各層(``strata_labels`` が等しい組の集合、018 では課題 x ``|Delta r*|`` の区間)の中で、組を
    ``distances``(正規化編集距離)の昇順に並べ(同じ値は元の添字の順)、先頭の ``q`` 個を近い組、
    末尾の ``q`` 個を遠い組とする。``q = floor(n_layer x fraction)``(``n_layer`` は層の大きさ)。
    ``fraction <= 0.5`` なので近い組と遠い組は重ならない。2 つの集合で層ごとの件数が一致するので、
    層を決める量(課題と真の報酬の差)の分布が揃う。

    Returns:
        ``(near_indices, far_indices, counts)``。添字は昇順。``counts`` は層ごとの
        ``{"near": q, "far": q, "size": n_layer}``。
    """
    if not 0.0 < fraction <= 0.5:
        raise ValueError(f"fraction は (0, 0.5] の値である必要がある: {fraction}")
    distances = np.asarray(distances, dtype=np.float64)
    if len(strata_labels) != distances.shape[0]:
        raise ValueError("strata_labels と distances の長さが一致しない")
    members: dict[object, list[int]] = {}
    for index, label in enumerate(strata_labels):
        members.setdefault(label, []).append(index)
    near: list[int] = []
    far: list[int] = []
    counts: dict = {}
    for label, indices in members.items():
        ordered = sorted(indices, key=lambda i: (distances[i], i))
        q = int(len(ordered) * fraction)
        near.extend(ordered[:q])
        far.extend(ordered[len(ordered) - q :] if q > 0 else [])
        counts[label] = {"near": q, "far": q, "size": len(ordered)}
    return np.array(sorted(near), dtype=np.int64), np.array(sorted(far), dtype=np.int64), counts
