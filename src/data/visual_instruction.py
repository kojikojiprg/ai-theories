"""合成のシーンに対する視覚指示データ(Visual Instruction Data)の生成・符号化・採点(021)。

020 の合成のシーン(``src/data/synthetic_scenes.py``。2 つの図形を左右または上下に並べた 32 x 32 の
画像と、その意味 ``SceneMeaning``)から、LLaVA(Liu et al., NeurIPS 2023)の 2 段階学習に使う
2 種類のデータを規則で作る。原論文は指示データを GPT-4 で生成したが、ここでは質問と正解を
シーンの意味から決定的に作る(016 の合成の指示データと同じ考え方)。

- **第 1 段階(特徴の整列)のデータ**: 画像 + 「画像を短く説明せよ」という指示 + 020 の正準形の
  キャプション(例: ``a red circle left of a blue square``)を応答とする事例。原論文の第 1 段階が
  画像とキャプションの組を単一の指示の形に直して使うのと同じ形である。
- **第 2 段階(視覚指示チューニング)のデータ**: 画像 + 質問 + 閉じた語彙の答え。質問の種類は
  ``QUESTION_TYPES``:

  - ``color``(属性): 形で指定した図形の色を答える(答えは 8 色のいずれか)。2 つの図形は色も形も
    異なるので、形で図形が一意に決まる。
  - ``spatial``(位置関係): 形で指定した図形 A が、もう一方の図形 B に対してどこにあるかを答える
    (答えは ``left``・``right``・``above``・``below`` のいずれか)。

  1 枚の画像から、色の質問を 2 個(各図形)、位置関係の質問を 2 個(各図形を A にしたもの)作る。

**テンプレート**(016 と同じ区切り記号を使い、トークナイザは変更しない)::

    ### Instruction:
    <image>
    {質問}
    ### Response: {答え}
    ### End

``<image>`` の位置に視覚トークン(projection 層の出力)を埋め込みとして差し込む。符号化では、
``### Instruction:\\n``(全事例で共通の接頭部分)・視覚トークンの枠・``\\n{質問}\\n### Response:``
(接尾部分)・`` {答え}\\n### End``(応答部分)を **別々に符号化して連結する**。視覚トークンの枠は
すべての事例で同じ位置(接頭部分の直後)から始まるので、バッチの中で埋め込みを差し込む位置が揃う。
枠の位置のトークン ID は埋め込みに置き換えられる仮の値(``placeholder_id``)であり、モデルの出力に
影響しない(``src/models/llava.py`` で確かめる)。

**テンプレートの既知性**: 各質問の種類のテンプレート(``COLOR_TEMPLATES``・``SPATIAL_TEMPLATES``)は、
画像の添字と図形の添字から決定的に割り当てる(``(image_index + k) mod 2``)。学習用・評価用のどちらの
画像の集合でも、すべてのテンプレートが同じ割合で現れ、答えの語彙もすべて学習用に現れる。評価用の
事例は「学習で見たテンプレートと答えの語彙で、学習で見ていない画像(別の乱数シードで描いたもの)に
答える」ものになり、テンプレートや答えが学習で既知かどうかが条件間・質問の種類間で揃う。

記号 / Notation:
    N_v : 視覚トークンの数(パッチトークン全体なら N = HW / P^2、平均プールなら 1)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.data.instruction import END_MARKER, INSTRUCTION_HEADER, RESPONSE_HEADER
from src.data.synthetic_scenes import COLOR_NAMES, RELATIONS, SHAPES, SceneMeaning, caption_text

QUESTION_TYPES: tuple[str, ...] = ("color", "spatial")

COLOR_TEMPLATES: tuple[str, ...] = (
    "What color is the {target}?",
    "Which color does the {target} have?",
)
SPATIAL_TEMPLATES: tuple[str, ...] = (
    "Where is the {target} relative to the {reference}?",
    "Is the {target} left of, right of, above or below the {reference}?",
)
CAPTION_INSTRUCTION = "Describe the image briefly."
"""第 1 段階(特徴の整列)の指示文(全事例で共通)。"""

SPATIAL_ANSWERS: tuple[str, ...] = ("left", "right", "above", "below")
ANSWER_VOCABULARY: dict[str, tuple[str, ...]] = {"color": COLOR_NAMES, "spatial": SPATIAL_ANSWERS}

PROMPT_PREFIX = INSTRUCTION_HEADER + "\n"
"""視覚トークンの枠の前に置く接頭部分(全事例で共通)。"""


@dataclass(frozen=True)
class VisualQuestion:
    """画像 1 枚に対する質問 1 個(またはキャプションの事例 1 個)。

    Attributes:
        image_index: 画像の添字(画像の集合の中での位置)。
        question_type: ``"color"`` / ``"spatial"`` / ``"caption"``(第 1 段階)。
        question: 質問の文字列(キャプションの事例では ``CAPTION_INSTRUCTION``)。
        answer: 正解の文字列(空白区切りの単語の列)。
        template_index: テンプレートの番号(キャプションの事例では 0)。
    """

    image_index: int
    question_type: str
    question: str
    answer: str
    template_index: int


def spatial_answer(meaning: SceneMeaning, target_is_first: bool) -> str:
    """図形 A(``target_is_first`` なら 1 つ目の図形)が図形 B に対してどこにあるか。

    1 つ目の図形は左(左右の配置)または上(上下の配置)にある(正準形、020)。
    """
    if RELATIONS[meaning.relation] == "left of":
        return "left" if target_is_first else "right"
    return "above" if target_is_first else "below"


def build_questions(meanings: Sequence[SceneMeaning]) -> list[VisualQuestion]:
    """各画像(意味)から、色の質問 2 個・位置関係の質問 2 個を作る(画像の順、各画像の中は固定の順)。

    テンプレートは ``(image_index + k) mod 2``(``k`` は図形の添字 0・1)で割り当てる。各画像で
    両方のテンプレートが 1 回ずつ使われるので、どの画像の部分集合でもテンプレートの割合は等しい。
    """
    questions = []
    for i, m in enumerate(meanings):
        objects = (m.first, m.second)
        for k in range(2):
            target = SHAPES[objects[k][1]]
            t = (i + k) % len(COLOR_TEMPLATES)
            questions.append(
                VisualQuestion(
                    i,
                    "color",
                    COLOR_TEMPLATES[t].format(target=target),
                    COLOR_NAMES[objects[k][0]],
                    t,
                )
            )
        for k in range(2):
            target, reference = SHAPES[objects[k][1]], SHAPES[objects[1 - k][1]]
            t = (i + k) % len(SPATIAL_TEMPLATES)
            questions.append(
                VisualQuestion(
                    i,
                    "spatial",
                    SPATIAL_TEMPLATES[t].format(target=target, reference=reference),
                    spatial_answer(m, target_is_first=(k == 0)),
                    t,
                )
            )
    return questions


def build_caption_examples(meanings: Sequence[SceneMeaning]) -> list[VisualQuestion]:
    """第 1 段階のデータ: 各画像に、指示 ``CAPTION_INSTRUCTION`` と正準形のキャプションの応答を
    付ける。"""
    return [
        VisualQuestion(i, "caption", CAPTION_INSTRUCTION, caption_text(m), 0)
        for i, m in enumerate(meanings)
    ]


def format_suffix(question: str) -> str:
    """視覚トークンの枠の後ろの部分(質問と ``### Response:``)。"""
    return f"\n{question}\n{RESPONSE_HEADER}"


def format_visual_response(answer: str) -> str:
    """応答部分(先頭の空白から終端記号まで)。016 の ``format_response()`` と同じ形。"""
    return f" {answer}\n{END_MARKER}"


def language_prior_upper_bound(questions: Sequence[VisualQuestion]) -> float:
    """画像を見ずに質問の文字列だけから答えを決める戦略が、この事例の集合で達成できる正解率の上界。

    質問の文字列ごとに、最も多い答えを返す戦略(集合の中で最適)の正解率である。
    """
    by_text: dict[str, Counter] = {}
    for q in questions:
        by_text.setdefault(q.question, Counter())[q.answer] += 1
    return sum(max(c.values()) for c in by_text.values()) / len(questions)


@dataclass(frozen=True)
class EncodedVisualBatchData:
    """符号化して固定長にそろえた事例の集合(全事例を 1 つのテンソルに持つ)。

    Attributes:
        token_ids: 形状 ``(n, S)`` の int64。視覚トークンの枠は ``placeholder_id``、右側は
            パディング。
        response_target_mask: 形状 ``(n, S - 1)`` の bool。位置 ``j`` の出力が応答部分のトークン
            ``token_ids[:, j + 1]`` を予測するとき True(016 の集合 R、終端記号を含む)。
        prompt_lengths: 形状 ``(n,)``。指示部分(接頭部分 + 枠 + 接尾部分)のトークン数。
        lengths: 形状 ``(n,)``。応答部分を含む系列長。
        image_indices: 形状 ``(n,)``。各事例の画像の添字。
        visual_start: 視覚トークンの枠の開始位置(全事例で共通)。
        num_visual_tokens: 枠の長さ N_v。
    """

    token_ids: Tensor
    response_target_mask: Tensor
    prompt_lengths: Tensor
    lengths: Tensor
    image_indices: Tensor
    visual_start: int
    num_visual_tokens: int

    def to(self, device: torch.device | str) -> EncodedVisualBatchData:
        return EncodedVisualBatchData(
            self.token_ids.to(device),
            self.response_target_mask.to(device),
            self.prompt_lengths.to(device),
            self.lengths.to(device),
            self.image_indices.to(device),
            self.visual_start,
            self.num_visual_tokens,
        )

    def __len__(self) -> int:
        return self.token_ids.size(0)


def encode_visual_examples(
    questions: Sequence[VisualQuestion],
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    num_visual_tokens: int,
    sequence_length: int,
    placeholder_id: int,
    pad_id: int,
) -> EncodedVisualBatchData:
    """事例を符号化して、長さ ``sequence_length`` にそろえた 1 つのテンソルにする。

    接頭部分・接尾部分・応答部分を別々に符号化して連結する。それぞれの復号が元の文字列と一致する
    ことを確かめる(可逆性)。系列長が ``sequence_length`` を超える事例があれば例外にする。
    """
    prefix_ids = encode(PROMPT_PREFIX)
    if decode(prefix_ids) != PROMPT_PREFIX:
        raise ValueError("接頭部分の復号が元の文字列と一致しない")
    visual_start = len(prefix_ids)
    rows, prompt_lengths, lengths = [], [], []
    suffix_cache: dict[str, list[int]] = {}
    response_cache: dict[str, list[int]] = {}
    for q in questions:
        if q.question not in suffix_cache:
            suffix_ids = encode(format_suffix(q.question))
            if decode(suffix_ids) != format_suffix(q.question):
                raise ValueError(f"接尾部分の復号が元の文字列と一致しない: {q.question!r}")
            suffix_cache[q.question] = suffix_ids
        if q.answer not in response_cache:
            response_ids = encode(format_visual_response(q.answer))
            if decode(response_ids) != format_visual_response(q.answer):
                raise ValueError(f"応答部分の復号が元の文字列と一致しない: {q.answer!r}")
            response_cache[q.answer] = response_ids
        prompt = prefix_ids + [placeholder_id] * num_visual_tokens + suffix_cache[q.question]
        row = prompt + response_cache[q.answer]
        if len(row) > sequence_length:
            raise ValueError(f"系列長 {len(row)} が sequence_length {sequence_length} を超える")
        rows.append(row)
        prompt_lengths.append(len(prompt))
        lengths.append(len(row))
    n = len(rows)
    token_ids = torch.full((n, sequence_length), pad_id, dtype=torch.int64)
    response_mask = torch.zeros((n, sequence_length - 1), dtype=torch.bool)
    for r, row in enumerate(rows):
        token_ids[r, : len(row)] = torch.tensor(row, dtype=torch.int64)
        # 位置 j の出力が位置 j + 1 の応答のトークンを予測する(prompt_length <= j + 1 < length)
        response_mask[r, prompt_lengths[r] - 1 : lengths[r] - 1] = True
    return EncodedVisualBatchData(
        token_ids=token_ids,
        response_target_mask=response_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int64),
        image_indices=torch.tensor([q.image_index for q in questions], dtype=torch.int64),
        visual_start=visual_start,
        num_visual_tokens=num_visual_tokens,
    )


def make_step_batches(num_examples: int, num_steps: int, batch_size: int, seed: int) -> np.ndarray:
    """形状 ``(num_steps, batch_size)`` の事例の添字(エポックごとに並べ替えて連結し、先頭から
    切り出す)。

    016 の ``make_epoch_batches()`` と同じ規則(``num_examples`` が ``batch_size`` の倍数なら、
    ミニバッチはエポックの境界をまたがない)を、numpy の配列で返す版。
    """
    rng = np.random.default_rng(seed)
    needed = num_steps * batch_size
    order = np.concatenate(
        [rng.permutation(num_examples) for _ in range(-(-needed // num_examples))]
    )
    return order[:needed].reshape(num_steps, batch_size).astype(np.int64)
