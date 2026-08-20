"""デコーディング戦略(Decoding Strategies)のスクラッチ実装。

``GPTLanguageModel.generate()``(``src/models/gpt.py``)は貪欲法(greedy decoding)と
temperature サンプリングのみに対応する。本モジュールは、008 で新規に扱う
top-k サンプリング(Fan et al., "Hierarchical Neural Story Generation", ACL 2018)・
top-p サンプリング(Holtzman et al., "The Curious Case of Neural Text Degeneration",
ICLR 2020)・ビームサーチ(Wu et al., "Google's Neural Machine Translation System",
2016 のスタイルの長さペナルティ付きビームサーチ)をスクラッチ実装する。

``torch.topk``・``torch.sort`` などのテンソル演算プリミティブは使うが、
フィルタリング・探索のロジック自体は既存ライブラリの高水準 API に委譲しない。

記号 / Notation:
    V : 語彙サイズ(vocabulary_size)
    B : ビーム数(beam_size)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.models.gpt import GPTLanguageModel


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """上位 k 個以外の logits を ``-inf`` にする(top-k フィルタリング)。

    Fan et al., "Hierarchical Neural Story Generation", ACL 2018 の定義に基づく。
    形状 ``(..., V)`` の ``logits`` の最終次元(語彙次元)について、値の大きい順に
    上位 ``k`` 個を残し、残りを ``-inf`` に置き換える。``-inf`` にされた要素は、
    後段の ``softmax`` で確率 0 になる。

    Args:
        logits: 形状 ``(..., V)`` の logits。
        k: 残す候補の数(1 以上)。``V`` を超える場合は ``V`` に丸める
            (このとき出力は元の ``logits`` と一致する、フィルタなしと等価)。

    Returns:
        ``logits`` と同じ形状。上位 k 個以外が ``-inf`` に置き換えられたテンソル。

    Raises:
        ValueError: ``k`` が 1 未満の場合。
    """
    if k < 1:
        raise ValueError(f"k は 1 以上である必要がある: {k}")
    vocab_size = logits.size(-1)
    k = min(k, vocab_size)

    # 上位 k 番目(境界)の値を各分布ごとに求め、それ未満の要素を -inf にする。
    # k == vocab_size のとき threshold は最小値に一致し、logits < threshold を
    # 満たす要素は存在しないため、出力は元の logits と完全に一致する。
    kth_values = torch.topk(logits, k, dim=-1).values[..., -1:]
    return torch.where(logits < kth_values, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """累積確率が p 以上になる最小の集合(nucleus)以外の logits を ``-inf`` にする
    (top-p サンプリング / nucleus sampling)。

    Holtzman et al., "The Curious Case of Neural Text Degeneration", ICLR 2020 の
    定義に厳密に従う。確率の降順に並べた語彙 $x_{(1)}, x_{(2)}, \\dots$ について、
    nucleus 集合 $V^{(p)}$ を

    .. math::

        V^{(p)} = \\min \\left\\{ V' \\subseteq V \\;\\middle|\\;
            \\sum_{x \\in V'} P(x \\mid x_{1:i-1}) \\ge p \\right\\}

    を満たす最小の集合(確率降順に上位から累積和が p に達するまでの集合)として定義し、
    $V^{(p)}$ に含まれないトークンの logits を ``-inf`` にする。

    実装上は、確率降順に並べたときの「自分より上位のトークンまでの累積確率」
    (自分を含まない)が既に ``p`` を超えているトークンを nucleus の外側と判定する
    (この時点で自分を含めなくても累積確率が ``p`` に達しているため、自分を含めた
    最小集合には不要)。

    Args:
        logits: 形状 ``(..., V)`` の logits。
        p: 累積確率の下限(0 より大きく 1 以下)。

    Returns:
        ``logits`` と同じ形状。nucleus に含まれない要素が ``-inf`` に置き換えられたテンソル。
        ``p=1.0`` の場合、全確率の総和は 1.0 であり、どのトークンについても
        「自分より上位の累積確率」が 1.0 を超えることはないため、フィルタなしと等価になる。

    Raises:
        ValueError: ``p`` が (0, 1] の範囲外の場合。
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p は 0 より大きく 1 以下である必要がある: {p}")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs_exclusive = torch.cumsum(sorted_probs, dim=-1) - sorted_probs

    # 自分より上位の累積確率(自分を含まない)が既に p を超えているトークンを除外する。
    sorted_indices_to_remove = cumulative_probs_exclusive > p
    sorted_logits_filtered = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits_filtered)
    return filtered


def _gnmt_length_penalty(length: int, alpha: float) -> float:
    """Wu et al., 2016(GNMT)スタイルの長さペナルティ ``lp(Y)`` を計算する。

    .. math::

        lp(Y) = \\frac{(5 + |Y|)^{\\alpha}}{(5 + 1)^{\\alpha}}

    ``alpha=0`` のとき ``lp(Y) = 1``(正規化なし、系列長にかかわらず一定)になる。
    ``alpha`` が大きいほど長い系列のスコアを相対的に押し上げる(長さによる
    不利(対数尤度の和は系列が長いほど小さくなりやすい)を補正する強さが増す)。

    Args:
        length: 系列長 $|Y|$。
        alpha: 長さペナルティの強さ($\\alpha$、``beam_search`` の ``length_penalty``)。

    Returns:
        長さペナルティ ``lp(Y)``。
    """
    return ((5 + length) ** alpha) / (6.0**alpha)


@torch.no_grad()
def beam_search(
    model: GPTLanguageModel,
    prompt_ids: Tensor,
    beam_size: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
) -> list[tuple[Tensor, float]]:
    """ビームサーチ(beam search)による生成。

    各ステップで、現在の ``beam_size`` 個の候補系列それぞれについて次トークンの
    対数確率上位 ``beam_size`` 個を展開し(最大 ``beam_size^2`` 個の候補)、
    Wu et al., 2016(GNMT)スタイルの長さ正規化スコア

    .. math::

        \\mathrm{score}(Y) = \\frac{\\log P(Y)}{lp(Y)}

    (``length_penalty`` の docstring 参照、$\\log P(Y)$ は生成したトークンの
    対数確率の総和)が高い順に上位 ``beam_size`` 個を残す。EOS トークンによる早期終了は
    行わず、常に ``max_new_tokens`` ステップ生成する(``GPTLanguageModel`` は
    EOS トークンの概念を持たない)。

    ``beam_size=1`` の場合、各ステップで残る候補は常に 1 個(その時点での最尤トークンを
    追加した系列)になるため、``GPTLanguageModel.generate(temperature=0.0)``(貪欲法)と
    完全に同一の系列を生成する(不変条件アサーション、008 4 節)。

    Args:
        model: 生成に使う言語モデル(``forward(token_ids) -> logits`` を持つ)。
        prompt_ids: 形状 ``(1, S_0)`` の LongTensor(生成の起点となる文脈)。
            バッチサイズ 1 のみに対応する。
        beam_size: ビーム数 B。
        max_new_tokens: 生成するトークン数。
        length_penalty: 長さペナルティの強さ $\\alpha$(``_gnmt_length_penalty`` 参照)。
            ``0.0`` で正規化なし(対数尤度の総和をそのままスコアに使う)。

    Returns:
        ``(sequence, score)`` のタプルのリスト(長さ ``beam_size``、``score`` の
        降順にソート済み)。``sequence`` は形状 ``(1, S_0 + max_new_tokens)`` の
        LongTensor、``score`` は長さ正規化済みの対数尤度。

    Raises:
        ValueError: ``prompt_ids`` のバッチサイズが 1 でない場合。
    """
    if prompt_ids.size(0) != 1:
        raise ValueError(
            f"beam_search はバッチサイズ 1 のプロンプトのみに対応する: {prompt_ids.shape}"
        )

    was_training = model.training
    model.eval()
    try:
        beam_sequences: list[Tensor] = [prompt_ids]
        beam_log_probs: list[float] = [0.0]  # 正規化前の対数尤度の総和

        for _ in range(max_new_tokens):
            candidates: list[tuple[float, Tensor]] = []  # (正規化前 log prob, sequence)
            for seq, cum_log_prob in zip(beam_sequences, beam_log_probs, strict=True):
                context = seq[:, -model.max_sequence_length :]
                logits = model(context)
                next_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # (V,)
                num_candidates = min(beam_size, next_log_probs.size(-1))
                top_log_probs, top_ids = torch.topk(next_log_probs, num_candidates)
                for lp, token_id in zip(top_log_probs.tolist(), top_ids.tolist(), strict=True):
                    new_seq = torch.cat(
                        [seq, torch.tensor([[token_id]], device=seq.device, dtype=seq.dtype)],
                        dim=1,
                    )
                    candidates.append((cum_log_prob + lp, new_seq))

            new_length = candidates[0][1].size(1)
            lp_denom = _gnmt_length_penalty(new_length, length_penalty)
            candidates.sort(key=lambda item: item[0] / lp_denom, reverse=True)
            top_candidates = candidates[:beam_size]
            beam_log_probs = [c[0] for c in top_candidates]
            beam_sequences = [c[1] for c in top_candidates]
    finally:
        model.train(was_training)

    final_length = beam_sequences[0].size(1)
    lp_denom = _gnmt_length_penalty(final_length, length_penalty)
    results = [
        (seq, cum_log_prob / lp_denom)
        for seq, cum_log_prob in zip(beam_sequences, beam_log_probs, strict=True)
    ]
    results.sort(key=lambda item: item[1], reverse=True)
    return results
