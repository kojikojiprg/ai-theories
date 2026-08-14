"""隠れ状態・活性化・勾配・トークナイザの統計量を測定するユーティリティ。

``theories/01_foundations/004_normalization_and_activation.ipynb`` の実験 D・F、
``theories/02_pretraining/005_tokenizer.ipynb`` の各実験で使う。

記号 / Notation:
    d_model : 隠れ状態の特徴次元
    d_ff    : Feed-Forward Network の中間層次元
    N       : 集計対象のトークン数(バッチ × 系列長など)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.data.tokenizer import try_decode_byte_level_symbol


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
    チャンク長の分布として定量的に比較するために使う(005 の実験5)。

    ``chunk_split_mode="whitespace"`` の下では、チャンクは空白の直前で区切られ、
    区切られた空白がチャンクの先頭に付く(``pretokenize`` を参照)。このため、
    インデントを持つコードのようにチャンク先頭に長い連続空白が付きうるドメインでは、
    チャンク長が「単語の長さ」ではなく「先頭の連続空白の長さ(コードであれば
    インデントの深さ)」を主に反映することがある。この 2 つの要因を切り分けられるよう、
    先頭の連続空白(``str.lstrip()`` が除去する、半角空白・タブ・改行を含む
    Unicode の空白文字)を除いた長さについても同じ統計を返す
    (キーに ``_without_leading_whitespace`` を付す)。

    Args:
        chunks: チャンク(部分文字列)のリスト。

    Returns:
        ``{"median": ..., "mean": ..., "p90": ..., "max": ...,
        "median_without_leading_whitespace": ..., "mean_without_leading_whitespace": ...,
        "p90_without_leading_whitespace": ..., "max_without_leading_whitespace": ...}``
        の辞書(いずれも文字数単位)。
    """
    lengths = np.array([len(c) for c in chunks], dtype=float)
    lengths_without_leading_whitespace = np.array([len(c.lstrip()) for c in chunks], dtype=float)
    return {
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": float(np.max(lengths)),
        "median_without_leading_whitespace": float(np.median(lengths_without_leading_whitespace)),
        "mean_without_leading_whitespace": float(np.mean(lengths_without_leading_whitespace)),
        "p90_without_leading_whitespace": float(
            np.percentile(lengths_without_leading_whitespace, 90)
        ),
        "max_without_leading_whitespace": float(np.max(lengths_without_leading_whitespace)),
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


def compute_segmentation_agreement_rate(
    segmentation_a: Sequence[str],
    segmentation_b: Sequence[str],
) -> float:
    """2 つの分割の一致度を、トークン境界の集合の Jaccard 係数として計算する。

    なぜ完全一致ではなく境界の Jaccard 係数か: BPE と Unigram 言語モデルのように
    符号化の慣習そのものが異なる(例: SentencePiece は語頭の空白を ``▁`` として
    トークンに含めるが、本リポジトリの自作 BPE はそうしない)2 つの分割方式を比較する
    場合、``compute_exact_match_rate`` のような文単位の完全一致(0 か 1 か)は、
    表層形の些細な流儀の違いだけで文全体が「不一致」になり、ほぼ常に一致率 0 に
    張り付いてしまう。これでは、2 つの分割が「どの程度似ているか」という度合いを
    測れない。トークン境界の位置集合どうしの Jaccard 係数を使うと、些細な流儀の
    違いがあっても分割の大部分が同じ位置で切れていれば高い値になり、部分的な
    一致の度合いを連続値として評価できる(005 の実験5-2)。

    具体的には、各分割について部分語の長さを先頭から累積し、トークンの境界となる
    文字位置の集合を作る(文字列先頭の位置 0 と末尾の位置は、どの分割でも自明に
    共通するため集合から除く)。2 つの境界位置集合 $A$, $B$ に対して

    $$
    J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}
    $$

    を返す。$A = B = \\emptyset$(両方とも 1 トークンのみで境界が 1 つも無い)場合は
    $J = 1.0$ とする(空集合どうしは定義上一致とみなす)。

    Args:
        segmentation_a: 分割結果 A(部分語シンボルのリスト)。
        segmentation_b: 分割結果 B。連結した文字列が ``segmentation_a`` の
            連結結果と一致する必要がある(異なる文字列を分割した結果を比較するのは
            不正な使い方であるため)。

    Returns:
        トークン境界集合の Jaccard 係数(0 以上 1 以下)。

    Raises:
        ValueError: 2 つの分割を連結した文字列が一致しない場合。
    """
    joined_a = "".join(segmentation_a)
    joined_b = "".join(segmentation_b)
    if joined_a != joined_b:
        raise ValueError(
            "segmentation_a と segmentation_b を連結した文字列が一致しません"
            f"(比較対象として不正): {joined_a!r} != {joined_b!r}"
        )

    def _boundaries(segmentation: Sequence[str]) -> set[int]:
        positions = set()
        cursor = 0
        for piece in segmentation:
            cursor += len(piece)
            positions.add(cursor)
        positions.discard(len(joined_a))  # 末尾は自明な境界なので除く
        return positions

    boundaries_a = _boundaries(segmentation_a)
    boundaries_b = _boundaries(segmentation_b)
    if not boundaries_a and not boundaries_b:
        return 1.0
    return len(boundaries_a & boundaries_b) / len(boundaries_a | boundaries_b)


def compute_character_coverage(
    vocabulary: Iterable[str],
    corpus: str,
) -> tuple[int, int, float]:
    """コーパスに出現するユニーク文字のうち、単一トークンとして語彙に含まれる
    ものの数を数える。

    語彙の各要素(シンボル)が「1 文字を表す単一トークン」かどうかは、次の手順で
    判定する。

    1. ``try_decode_byte_level_symbol``(``src/data/tokenizer.py``)で、そのシンボルを
       バイトレベル BPE の表現とみなして UTF-8 文字列へのデコードを試みる。
       成功し、かつ結果がちょうど 1 文字であれば、その文字を被覆しているとみなす。
       これは、複数バイトにまたがる文字(日本語など)を表すには、対応するバイト列の
       マージが完了している必要があることに対応する。
    2. 1 が失敗する場合(シンボルの文字がバイトレベル表現の定義域に無い場合。
       文字レベル語彙のシンボルは通常ここに該当する)、シンボルそのものが 1 文字の
       文字列であれば、その文字を被覆しているとみなす。

    この手順により、``byte_level`` の真偽を明示的に受け取らなくても、バイトレベル・
    文字レベルいずれの語彙に対しても正しく動作する。文字レベル初期語彙は学習コーパスの
    全ユニーク文字を初期シンボルとして含む(005 の 3.3 節)ため、被覆率は常に 1.0 になる
    (この関数の対照としての振る舞い)。

    Args:
        vocabulary: 語彙(``BPETokenizer.vocab`` などのシンボル文字列の集合)。
        corpus: 被覆率を測る対象のコーパス。

    Returns:
        (被覆した文字種数, コーパスのユニーク文字種数, 被覆率) のタプル。
        コーパスが空文字列の場合、被覆率は 1.0 とする。
    """
    corpus_chars = set(corpus)
    covered: set[str] = set()
    for symbol in vocabulary:
        decoded = try_decode_byte_level_symbol(symbol)
        if decoded is not None:
            if len(decoded) == 1:
                covered.add(decoded)
        elif len(symbol) == 1:
            covered.add(symbol)

    covered_in_corpus = covered & corpus_chars
    num_total = len(corpus_chars)
    num_covered = len(covered_in_corpus)
    coverage = num_covered / num_total if num_total else 1.0
    return num_covered, num_total, coverage
