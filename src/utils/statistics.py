"""隠れ状態・活性化・勾配・トークナイザの統計量を測定するユーティリティ。

``theories/01_foundations/004_normalization_and_activation.ipynb`` の実験 D・F、
``theories/02_pretraining/005_tokenizer.ipynb`` の各実験で使う。

記号 / Notation:
    d_model : 隠れ状態の特徴次元
    d_ff    : Feed-Forward Network の中間層次元
    N       : 集計対象のトークン数(バッチ × 系列長など)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn


def compute_mean_to_rms_ratio(hidden_states: Sequence[Tensor]) -> list[float]:
    """隠れ状態の平均の絶対値と二乗平均平方根(RMS)の比を層ごとに集計する。

    |mean(a)| / RMS(a) を各層の隠れ状態について特徴次元(最終次元 d_model 方向)で
    計算し、残りの次元(バッチ・系列長など)で平均する。この比が小さいほど、
    平均減算(re-centering)を省略したときに失われる情報が小さいことを意味する
    (RMSNorm が平行移動不変性(shift invariance)を失うことが実害になるかどうかの
    検証、理論セクションおよび実験 B・D を参照)。

    Args:
        hidden_states: 層ごとの隠れ状態のリスト。各要素は形状 ``(..., d_model)``。

    Returns:
        層ごとの比のリスト(``hidden_states`` と同じ長さ)。
    """
    ratios = []
    for h in hidden_states:
        mean = h.mean(dim=-1)
        rms = torch.sqrt(h.pow(2).mean(dim=-1))
        ratio = (mean.abs() / rms.clamp_min(1e-12)).mean().item()
        ratios.append(ratio)
    return ratios


def compute_always_negative_unit_ratio(
    preactivations: Sequence[Tensor],
    threshold: float = 0.0,
) -> list[float]:
    """Feed-Forward Network 中間層の
    「常に負のユニット(always-negative unit)」の割合を層ごとに集計する。

    検証データの全トークンにわたって活性化前(pre-activation)の値が一度も
    ``threshold`` を超えなかったユニットの割合を返す。

    **注意: この指標が「死んだユニット(dead unit)」、すなわち勾配が恒久的に
    伝わらなくなる状態を意味するのは、活性化関数が ReLU の場合に限る。** ReLU は
    負領域で導関数が恒等的に 0 になるため、常に負のユニットは実際に勾配を受け取れない。
    一方、GELU など負領域でも導関数が非零な活性化関数では、同じ「常に負」という条件を
    満たしていても勾配は伝わり続けるため、「死」を意味しない。異なる活性化関数間で
    この比率の値を単純に比較する際は、比率そのものではなく、実際に勾配が伝わっているか
    どうか(``compute_gradient_norm_by_unit_group`` 参照)まで確認すること(実験 F)。

    Args:
        preactivations: 層ごとの活性化前の値のリスト。各要素は形状 ``(N, d_ff)``
            (N は検証データの全トークン数、d_ff は中間層の次元)。
        threshold: 「正になった」とみなす閾値。

    Returns:
        層ごとの常に負のユニットの割合のリスト(0 以上 1 以下、``preactivations`` と同じ長さ)。
    """
    ratios = []
    for pre in preactivations:
        never_exceeded = pre.max(dim=0).values <= threshold
        ratios.append(never_exceeded.float().mean().item())
    return ratios


def compute_gradient_norm_by_unit_group(
    weight_grad: Tensor,
    always_negative_mask: Tensor,
) -> tuple[float, float]:
    """線形層の重み勾配を、常に負のユニット(always-negative unit)に対応する行と
    それ以外の行に分けて、それぞれのノルムを計算する。

    ``compute_always_negative_unit_ratio`` が測る比率だけでは、そのユニットが
    実際に勾配を受け取れているかどうかは分からない(同じ比率でも活性化関数によって
    意味が異なるため)。本関数は、常に負のユニットに対応する重み行の勾配ノルムを
    直接測ることで、これを代理指標の比較ではなく直接の検証にする(実験 F)。
    ReLU では常に負のユニットの勾配ノルムがほぼ厳密に 0 になることが期待される。

    Args:
        weight_grad: 線形層(順伝播ネットワークの 1 層目)の重み勾配、
            形状 ``(d_ff, d_model)``(``linear1.weight.grad`` を想定)。
        always_negative_mask: 形状 ``(d_ff,)`` の bool テンソル。True が
            「検証データ全体で活性化前の値が一度も正にならなかったユニット」を表す
            (``compute_always_negative_unit_ratio`` と同じ判定に基づく)。

    Returns:
        (always_negative_group_norm, other_group_norm) のタプル。
        該当ユニットが 1 つもない場合はそのグループのノルムを ``0.0`` とする。
    """
    always_negative_rows = weight_grad[always_negative_mask]
    other_rows = weight_grad[~always_negative_mask]
    always_negative_norm = (
        always_negative_rows.norm().item() if always_negative_rows.numel() > 0 else 0.0
    )
    other_norm = other_rows.norm().item() if other_rows.numel() > 0 else 0.0
    return always_negative_norm, other_norm


def compute_gradient_norm_per_layer(model: nn.Module) -> tuple[list[float], list[float]]:
    """層ごとの勾配ノルムを集計する。

    002(``theories/01_foundations/002_transformer_block.ipynb`` 実験 2)で導入した、
    「最終層を基準にした相対勾配」の手法を再利用する。生の勾配ノルムは損失の定義上
    各条件の出力スケールに交絡されるため、条件をまたいだ比較には使えない。
    条件間で比較したいときは、相対勾配(層 l の勾配ノルム ÷ 最終層の勾配ノルム)を使う。

    呼び出し前に、対象モデルの forward 内で各層の出力に ``retain_grad()`` を呼び、
    ``model.layer_outputs``(``list[Tensor]``、層順)へ格納しておく必要がある。
    その状態で ``loss.backward()`` を呼んだ後にこの関数を呼ぶ。

    Args:
        model: ``layer_outputs: list[Tensor]`` 属性を持つモデル
            (各要素は ``retain_grad()`` 済みで、逆伝播後に ``.grad`` を持つ)。

    Returns:
        (raw_norms, relative_norms) のタプル。いずれも層順のリストで、
        relative_norms は最終層の生の勾配ノルムで正規化した値。
    """
    layer_outputs: list[Tensor] = model.layer_outputs
    raw_norms = [out.grad.norm().item() for out in layer_outputs]
    final_norm = raw_norms[-1]
    relative_norms = [v / final_norm for v in raw_norms]
    return raw_norms, relative_norms


def compute_fertility(token_count: int, char_count: int) -> float:
    """fertility(1 文字あたりのトークン数)を計算する。

    Rust et al. (2021) の定義に基づく圧縮効率の指標。値が小さいほど、より少ない
    トークンで同じ文字数のテキストを表現できている(語彙サイズに対する圧縮効果が
    高い)ことを意味する。

    Note:
        原論文の fertility は主に単語単位(1 単語あたりのトークン数)で定義される
        ことが多いが、本リポジトリは日本語(単語境界が表層に現れない言語)を扱うため、
        分母を単語数ではなく文字数に統一する(005 の理論セクションを参照)。

    Args:
        token_count: トークナイザによる符号化後のトークン数。
        char_count: 符号化対象テキストの文字数。

    Returns:
        fertility(token_count / char_count)。
    """
    return token_count / char_count


def compute_unknown_rate(tokens: Sequence[str], unk_token: str = "<unk>") -> float:
    """トークン列に占める未知語(Out-of-Vocabulary)トークンの割合を計算する。

    Args:
        tokens: 符号化後のトークン列。
        unk_token: 未知語を表す特殊トークン。

    Returns:
        未知語トークンの割合(0 以上 1 以下)。``tokens`` が空の場合は 0.0。
    """
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t == unk_token) / len(tokens)


def compute_chunk_length_statistics(chunks: Sequence[str]) -> dict[str, float]:
    """チャンク(事前分割で得た単位)の長さ(文字数)の分布統計を計算する。

    空白による事前分割(pre-tokenization)が言語によって機能するかどうかを、
    チャンク長の分布として定量的に比較するために使う(005 の実験4)。

    Args:
        chunks: チャンク(部分文字列)のリスト。

    Returns:
        ``{"median": ..., "mean": ..., "p90": ..., "max": ...}`` の辞書
        (いずれも文字数単位)。
    """
    lengths = np.array([len(c) for c in chunks], dtype=float)
    return {
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": float(np.max(lengths)),
    }


def compute_exact_match_rate(
    sequences_a: Sequence[Sequence[str]],
    sequences_b: Sequence[Sequence[str]],
) -> tuple[float, list[int]]:
    """2 組の分割結果を項目ごとに比較し、完全一致率と不一致のインデックス一覧を返す。

    自作の Viterbi 分割と sentencepiece 自身の分割結果の比較、および BPE と
    Unigram 言語モデルの分割結果の比較に使う(005 の実験5)。

    Args:
        sequences_a: 分割結果 A(文の数だけ、部分語シンボル列を並べたもの)。
        sequences_b: 分割結果 B。``sequences_a`` と同じ長さである必要がある。

    Returns:
        (完全一致率, 不一致の項目インデックスのリスト) のタプル。
    """
    if len(sequences_a) != len(sequences_b):
        raise ValueError("sequences_a と sequences_b の要素数が一致しません")
    mismatches = [
        i
        for i, (a, b) in enumerate(zip(sequences_a, sequences_b, strict=True))
        if list(a) != list(b)
    ]
    match_rate = 1.0 - len(mismatches) / len(sequences_a)
    return match_rate, mismatches
