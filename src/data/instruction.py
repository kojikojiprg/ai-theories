"""合成の指示データ(Instruction Data)の生成・テンプレートの適用・符号化・採点(016)。

016(SFT: Supervised Fine-Tuning、指示チューニング)で使う、入力と課題から応答が決定的に
決まる合成の指示データを扱う。入力は英単語の列(既定で 3〜8 語)であり、課題は単語の
並べ替え・複製・選択の 4 種類(``TASK_NAMES``)である。文字単位の操作は BPE(Byte Pair
Encoding)の部分語分割と相性が悪い(1 文字が 1 トークンにならない)ため避け、単語を
単位とする操作だけを使う。単語は、先頭に空白を付けた形(``" " + word``)で 008 の
トークナイザの **単一トークン** になるものから選ぶ(``WORD_VOCABULARY``。単一トークンで
あることの確認は、トークナイザを受け取る ``check_single_token_words()`` で行う)。

テンプレート(チャットテンプレート)は既存の語彙で表せる文字列だけを使い、トークナイザは
変更しない::

    ### Instruction:
    {前置き(長い水準のみ)} {指示文}
    Words: w1 w2 w3
    ### Response: v1 v2 v3
    ### End

``### Response:`` までを指示部分(プロンプト)、その直後の空白から終端記号 ``### End``
までを応答部分とする。区切り記号(``### Instruction:``・``### Response:``・``### End``)は
いずれも ``###`` で始まる。

指示部分と応答部分は **別々に符号化して連結する**(``encode_instruction_example()``)。
損失マスクの境界がトークンの境界と一致し、応答部分のトークン列が指示部分(短い水準・長い
水準)によらず同一になる。

記号 / Notation:
    K   : 課題の数(``len(TASK_NAMES)``)
    x   : 入力(単語の列)
    y   : 応答(単語の列)
    X   : 評価用の入力の集合
    c*  : 指示を無視して入力だけから出力を決める決定的な戦略が、評価集合で達成できる
          完全一致率の上界(``compute_instruction_agnostic_upper_bound()``)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

INSTRUCTION_HEADER = "### Instruction:"
RESPONSE_HEADER = "### Response:"
END_MARKER = "### End"
DELIMITER_PREFIX = "###"
"""区切り記号はすべてこの文字列で始まる(形式の遵守の判定に使う、``score_generated_response()``)。"""

WORDS_LABEL = "Words:"

TASK_NAMES: tuple[str, ...] = ("reverse", "repeat_twice", "odd_positions", "rotate_left")
"""課題の名前。``apply_task()`` がそれぞれの応答を決める。

- ``reverse``: 単語の順序を逆にする。
- ``repeat_twice``: 各単語を 2 回ずつ続けて書く(順序は保つ)。
- ``odd_positions``: 奇数番目(1 番目・3 番目・...)の単語だけを残す。
- ``rotate_left``: 先頭の単語を末尾に移す(左に 1 つ回転する)。

「先頭と末尾の単語を入れ替える」課題は、3 語の入力で ``reverse`` と応答が一致するため
採用しない(016 の 5 節)。上の 4 課題は、互いに異なる 3 語以上の入力に対して応答がすべて
異なる(``reverse`` と ``rotate_left`` は長さが同じだが、先頭の単語が w_n と w_2 で異なる)。
"""

# 先頭に空白を付けた形で 008 のトークナイザ(kojikojiprg/ai-theories-tokenizer-en)の単一トークンに
# なる英単語(名詞を中心に選んだ)。語彙から暴力・犯罪などを連想させる語を除いた。
WORD_VOCABULARY: tuple[str, ...] = tuple(
    """
    ability account action actor address age agency agent agreement alliance amount analysis analyst
    anatomy angel animal arm army art article assembly assistant attorney author award ball ballot
    bank bar base baseball basis bill billion birth black block blue board body book box brain
    branch brother budget building bus business cabinet camp campaign candidate capital car carbon
    card career case cat cell center century chair chairman champion chance character chief child
    children citizen city class climate club coach coalition coin college colour column command
    comment committee community company condition congress contact contest contract control cost
    council country county course court crowd custom data date daughter day deal debate debut
    decision defense degree delegate deputy design dialect director discovery district division
    doctor document draft dress drive ear economy editor education effect effort election embassy
    emergency energy engine engineer episode equipment error escape event evidence evolution
    exchange expert face fact faculty family farm fashion father feature field film fire flag flight
    food foot football force fossil founder friend fuel function fund future gain game gas gender
    genus ghost girl goal gold governor graduate green ground group growth guard hand head health
    hearing history home honor hospital host hour house housing human hunter husband idea identity
    impact incident industry inflation influence inning interest interview issue job journal judge
    key labor ladder land law leader league leg level life light limit link location lock loss
    majority manager market marriage match material mayor meaning medal media medicine meeting
    member method million minister ministry mission model money month moon mother movement music
    name nation network news night nominee novel object offer office officer official oil opinion
    option origin owner park partner party patient pattern peace period person petition phase
    physician pitch pitcher plan plant plate platform player playoff policy poll port post power
    practice presence president press pressure price problem process producer product professor
    program progress project property race radio rain rally range rank rate rating reason record red
    reform region relief report request research resident result review risk ritual road rock rocket
    role round rugby rule runner safety school science score season seat secretary section sector
    security senator sense series service share ship show sign silence silver site situation
    software soul source space speaker species speech sport staff stage standard star state
    statement station status story street student study style subject suburb success summit supply
    support surgeon survey swing symbol system table target tariff tartan tax teacher team
    television tent territory test texture thread tool tooth town track trade tradition train
    training transfer travel treatment trial trip trust umpire uniform union unit vaccine vehicle
    version vessel veteran victory video view village visit vote walk warning water weather week
    white wife wind winner witness woman world writer yard year yellow zone
    """.split()  # noqa: SIM905  # 単語の一覧は空白区切りのほうが読みやすい
)

# 学習に使う指示文(課題ごとに 4 個)。短い水準のテンプレートはこれをそのまま使う。
SEEN_INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "reverse": (
        "Reverse the order of the words.",
        "Write the words in reverse order.",
        "List the words from the last one to the first one.",
        "Output the words backwards.",
    ),
    "repeat_twice": (
        "Repeat each word twice.",
        "Write every word two times in a row.",
        "Say each word two times, keeping the original order.",
        "Double each word.",
    ),
    "odd_positions": (
        "Keep only the words in odd positions.",
        "Write the first, third, fifth and seventh words.",
        "Output every other word, beginning with the first one.",
        "Remove the words in even positions.",
    ),
    "rotate_left": (
        "Move the first word to the end.",
        "Put the first word after the last word.",
        "Rotate the words to the left by one position.",
        "Shift every word one place to the left and wrap the first word around to the end.",
    ),
}

# 学習に使わない指示文(課題ごとに 2 個、実験 A の診断量「未見テンプレートでの完全一致率」用)。
UNSEEN_INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "reverse": (
        "Put the words in the opposite order.",
        "Read the words from the end to the beginning and write them down.",
    ),
    "repeat_twice": (
        "Write each word, and then write it again.",
        "Copy every word two times without changing the order.",
    ),
    "odd_positions": (
        "Skip every second word.",
        "Write only the 1st, 3rd, 5th and 7th words.",
    ),
    "rotate_left": (
        "Take the first word and place it after all the others.",
        "Begin with the second word and finish with the first word.",
    ),
}

# 長い水準の前置き(課題によらない冗長な文章)。課題の意味を変えず、課題を特定する情報も
# 含まない。長い水準の指示文は「前置き + 空白 + 短い水準の指示文」とする。
PREAMBLES: tuple[str, ...] = (
    "You are a careful helper who works with short lists of English words. "
    "In this exercise you will read a short description of an operation and a list of words. "
    "Please read the description slowly and make sure that you understand it before you begin "
    "to write. Your answer should contain only words from the list, written in lowercase letters "
    "and separated by single spaces, without extra comments or punctuation.",
    "The following exercise is about rearranging and copying words, and it does not require any "
    "outside knowledge. A small list of common English words is given below, together with one "
    "simple operation that should be applied to the list. Apply the operation exactly as it is "
    "described, without adding or removing anything else, and write the new list on a single line "
    "with single spaces between the words.",
    "Welcome to a short exercise. There is one list of ordinary English words and one operation "
    "that tells you how to transform the list into a new list of words. The operation is always "
    "simple, but it has to be followed precisely, so please think about it before you answer. "
    "Write only the transformed list, keep every word in lowercase, and do not include any other "
    "text in your answer.",
    "Please help with the small exercise described below. It uses a list of English words that "
    "were chosen at random, so the words are not related to each other and they do not form a "
    "sentence. Follow the operation stated after this paragraph and produce the new list of words "
    "that it describes. Write the words in lowercase with single spaces between them, and avoid "
    "any comments before or after the list.",
)


def apply_task(task: str, words: Sequence[str]) -> tuple[str, ...]:
    """課題 ``task`` を入力 ``words`` に適用した応答(正解の単語の列)を返す。"""
    words = tuple(words)
    if task == "reverse":
        return words[::-1]
    if task == "repeat_twice":
        return tuple(w for word in words for w in (word, word))
    if task == "odd_positions":
        return words[::2]  # 1-indexed で奇数番目 = 0-indexed で偶数の添字
    if task == "rotate_left":
        return words[1:] + words[:1]
    raise ValueError(f"未知の課題: {task!r}")


def format_prompt(instruction: str, words: Sequence[str], preamble: str | None = None) -> str:
    """指示部分(プロンプト)の文字列を返す。``preamble`` を与えると長い水準になる。"""
    body = instruction if preamble is None else f"{preamble} {instruction}"
    listed = "".join(" " + w for w in words)
    return f"{INSTRUCTION_HEADER}\n{body}\n{WORDS_LABEL}{listed}\n{RESPONSE_HEADER}"


def format_response(answer: Sequence[str]) -> str:
    """応答部分の文字列(先頭の空白から終端記号まで)を返す。"""
    return "".join(" " + w for w in answer) + "\n" + END_MARKER


def template_words() -> set[str]:
    """テンプレート(区切り記号・指示文・前置き)に現れる英単語の集合(小文字)を返す。

    入力の単語がテンプレートの単語と重ならないことを確かめるのに使う(入力の単語が指示文の
    単語と一致すると、どこまでが指示でどこからが入力かが曖昧になるため)。
    """
    texts = [INSTRUCTION_HEADER, RESPONSE_HEADER, END_MARKER, WORDS_LABEL, *PREAMBLES]
    texts += [
        t for group in (SEEN_INSTRUCTIONS, UNSEEN_INSTRUCTIONS) for ts in group.values() for t in ts
    ]
    found: set[str] = set()
    for text in texts:
        token = ""
        for ch in text.lower() + " ":
            if "a" <= ch <= "z":
                token += ch
            elif token:
                found.add(token)
                token = ""
    return found


def check_single_token_words(encode: Callable[[str], list[int]], words: Sequence[str]) -> list[str]:
    """``" " + word`` が単一トークンにならない単語のリストを返す(空なら全単語が単一トークン)。"""
    return [w for w in words if len(encode(" " + w)) != 1]


@dataclass(frozen=True)
class InstructionExample:
    """指示データの 1 事例(入力・課題・テンプレートの選択)。

    Attributes:
        words: 入力 x(単語の列)。
        task: 課題の名前(``TASK_NAMES`` のいずれか)。
        template_index: 指示文の番号(``unseen`` に応じて ``SEEN_INSTRUCTIONS`` または
            ``UNSEEN_INSTRUCTIONS`` の添字)。
        preamble_index: 長い水準で使う前置きの番号(``PREAMBLES`` の添字)。
        unseen: True なら学習に使わない指示文(``UNSEEN_INSTRUCTIONS``)を使う。
    """

    words: tuple[str, ...]
    task: str
    template_index: int
    preamble_index: int
    unseen: bool = False

    @property
    def instruction(self) -> str:
        table = UNSEEN_INSTRUCTIONS if self.unseen else SEEN_INSTRUCTIONS
        return table[self.task][self.template_index]

    @property
    def answer(self) -> tuple[str, ...]:
        return apply_task(self.task, self.words)

    def prompt(self, long: bool) -> str:
        preamble = PREAMBLES[self.preamble_index] if long else None
        return format_prompt(self.instruction, self.words, preamble)

    def response(self) -> str:
        return format_response(self.answer)


def sample_word_sequences(
    rng: np.random.Generator,
    count: int,
    vocabulary: Sequence[str],
    min_length: int,
    max_length: int,
    exclude: set[tuple[str, ...]] | None = None,
) -> list[tuple[str, ...]]:
    """互いに異なる単語の列を ``count`` 個生成する。

    列の長さは ``[min_length, max_length]`` から一様に選び、単語は ``vocabulary`` から
    非復元抽出する(1 つの列の中に同じ単語は現れない)。生成する列どうし、および
    ``exclude`` に含まれる列とは重複しない(重複した場合は引き直す)。
    """
    exclude = set() if exclude is None else set(exclude)
    sequences: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    while len(sequences) < count:
        length = int(rng.integers(min_length, max_length + 1))
        indices = rng.choice(len(vocabulary), size=length, replace=False)
        sequence = tuple(vocabulary[int(i)] for i in indices)
        if sequence in seen or sequence in exclude:
            continue
        seen.add(sequence)
        sequences.append(sequence)
    return sequences


def build_training_examples(
    inputs: Sequence[tuple[str, ...]], rng: np.random.Generator
) -> list[InstructionExample]:
    """学習用の事例を作る(入力 1 つにつき事例 1 つ)。

    課題は、先頭から K 個ずつの区切りごとに ``TASK_NAMES`` の並べ替えを割り当てる。
    したがって、K の倍数の長さの任意の先頭部分で課題の数が均等になる(016 の実験 C で、
    入れ子にした先頭 N 個の部分集合を使うため)。指示文・前置きの番号は一様に選ぶ。
    """
    k = len(TASK_NAMES)
    examples = []
    for start in range(0, len(inputs), k):
        order = rng.permutation(k)
        for offset, words in enumerate(inputs[start : start + k]):
            task = TASK_NAMES[int(order[offset])]
            examples.append(
                InstructionExample(
                    words=tuple(words),
                    task=task,
                    template_index=int(rng.integers(len(SEEN_INSTRUCTIONS[task]))),
                    preamble_index=int(rng.integers(len(PREAMBLES))),
                )
            )
    return examples


def build_evaluation_examples(
    inputs: Sequence[tuple[str, ...]], rng: np.random.Generator, unseen: bool = False
) -> list[InstructionExample]:
    """評価用の事例を作る。各入力を K 個の課題すべてと組にする(直積)。

    事例は入力の順、同じ入力の中では ``TASK_NAMES`` の順に並ぶ(事例 ``i`` の入力は
    ``inputs[i // K]``、課題は ``TASK_NAMES[i % K]``)。指示文・前置きの番号は一様に選ぶ。
    """
    table = UNSEEN_INSTRUCTIONS if unseen else SEEN_INSTRUCTIONS
    examples = []
    for words in inputs:
        for task in TASK_NAMES:
            examples.append(
                InstructionExample(
                    words=tuple(words),
                    task=task,
                    template_index=int(rng.integers(len(table[task]))),
                    preamble_index=int(rng.integers(len(PREAMBLES))),
                    unseen=unseen,
                )
            )
    return examples


def compute_instruction_agnostic_upper_bound(
    answers_by_input: Sequence[Sequence[Sequence[str]]],
) -> float:
    """指示を無視して入力だけから出力を決める決定的な戦略の、完全一致率の上界 c* を返す。

    評価集合が「入力 x の集合 X」と「K 個の課題」の直積であるとき、入力 x に対して
    指示を見ない戦略が出せる応答は 1 つだけなので、x の K 事例のうち正解にできるのは
    ``max_y n_x(y)`` 個が最大である(``n_x(y)`` は x に対して応答 y を正解とする課題の数)。

    .. math::

        c^* = \\frac{1}{|X|} \\sum_{x \\in X} \\max_y \\frac{n_x(y)}{K}

    Args:
        answers_by_input: 入力ごとの、K 個の課題の正解(単語の列)のリスト。

    Returns:
        c*。課題間で応答がすべて異なれば 1/K。
    """
    total = 0.0
    for answers in answers_by_input:
        counts = Counter(tuple(a) for a in answers)
        total += max(counts.values()) / len(answers)
    return total / len(answers_by_input)


@dataclass(frozen=True)
class EncodedInstructionExample:
    """符号化した事例。``token_ids = 指示部分のトークン列 + 応答部分のトークン列``。

    Attributes:
        token_ids: 連結したトークン列。
        prompt_length: 指示部分のトークン数(``token_ids[:prompt_length]`` が指示部分)。
    """

    token_ids: tuple[int, ...]
    prompt_length: int

    @property
    def response_ids(self) -> tuple[int, ...]:
        return self.token_ids[self.prompt_length :]


def encode_instruction_example(
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    prompt: str,
    response: str,
) -> EncodedInstructionExample:
    """指示部分と応答部分を **別々に符号化して連結する**。

    別々に符号化するので、損失マスクの境界(``prompt_length``)は必ずトークンの境界と一致する。
    連結したトークン列の復号が全文(``prompt + response``)と一致することを確かめる。
    """
    prompt_ids = encode(prompt)
    response_ids = encode(response)
    token_ids = tuple(prompt_ids + response_ids)
    if decode(list(token_ids)) != prompt + response:
        raise ValueError("連結したトークン列の復号が全文と一致しない")
    return EncodedInstructionExample(token_ids=token_ids, prompt_length=len(prompt_ids))


def collate_instruction_batch(
    examples: Sequence[EncodedInstructionExample], pad_token_id: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """事例のリストを右パディングしてバッチにし、指示部分・応答部分の予測対象のマスクを返す。

    位置 ``j`` の logits は位置 ``j + 1`` のトークンを予測する(予測対象は ``token_ids[:, 1:]``)。
    予測対象のうち、

    - 指示部分(集合 P): ``1 <= j + 1 < prompt_length``。系列の先頭のトークンは左の文脈を
      持たないので予測対象に含めない。
    - 応答部分(集合 R): ``prompt_length <= j + 1 < len(token_ids)``。終端記号を含む。

    パディングは系列の右側にだけ置く。因果マスクにより実トークンの位置はそれより右の
    パディングを参照しないので、パディングは実トークンの logits を変えない。パディングの位置は
    どちらのマスクにも含まれず、損失に寄与しない。

    Returns:
        ``(token_ids, prompt_target_mask, response_target_mask)``。形状はそれぞれ
        ``(B, S_max)``、``(B, S_max - 1)``、``(B, S_max - 1)``(マスクは bool)。
    """
    max_length = max(len(e.token_ids) for e in examples)
    token_ids = torch.full((len(examples), max_length), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((len(examples), max_length - 1), dtype=torch.bool)
    response_mask = torch.zeros((len(examples), max_length - 1), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.token_ids)
        token_ids[row, :length] = torch.tensor(example.token_ids, dtype=torch.long)
        prompt_mask[row, : example.prompt_length - 1] = True
        response_mask[row, example.prompt_length - 1 : length - 1] = True
    return token_ids, prompt_mask, response_mask


def score_generated_response(generated_text: str, answer: Sequence[str]) -> tuple[bool, bool]:
    """生成した応答部分の文字列を採点し、``(形式の遵守, 完全一致)`` を返す。

    - **形式の遵守**: 終端記号 ``### End`` が現れ、それより前に他の区切り記号が現れないこと。
      区切り記号はすべて ``###`` で始まるので、「最初に現れる ``###`` が終端記号の先頭である
      こと」と同値である。上限トークン数以内に現れたかどうかは、生成を上限トークン数で打ち切った
      文字列を渡すことで判定される。
    - **完全一致**: 形式を遵守し、かつ終端記号より前の文字列を空白で区切った単語の列が正解と
      一致すること(空白の正規化: 先頭・末尾の空白を除き、連続する空白・改行を 1 つの区切りと
      みなす。``text.split()`` による)。
    """
    first = generated_text.find(DELIMITER_PREFIX)
    format_ok = first != -1 and generated_text.startswith(END_MARKER, first)
    if not format_ok:
        return False, False
    return True, generated_text[:first].split() == list(answer)
