"""指示への応答の画面(016 SFT・017 報酬モデルと RLHF)。

処理は ``service.py`` にあり、ここは画面の組み立てと、処理の結果の表示用の整形だけを行う。
理論の数値(016 の評価)は、ノートブックのセル出力から転記したもので、
このアプリのモデルをここで測り直したものではない。
"""

from __future__ import annotations

import time

import gradio as gr

from lab.core.artifacts import Resources
from lab.core.device import gpu
from lab.features.instruction.service import (
    TASK_NAMES,
    TEMPLATE_UNSEEN,
    InstructionOutcome,
    instruction_choices,
    parse_words,
    random_words,
    respond,
    template_kind_of,
)
from lab.ui.layout import Page
from lab.ui.links import notebook_link
from lab.ui.markdown import paragraphs

PAGE_TITLE = "指示への応答 / Instruction Following"

# @spaces.GPU の duration(秒)。最大 32 トークンの貪欲法なので短い。
INSTRUCTION_GPU_SECONDS = 15

TASK_CHOICES = [
    ("reverse — 単語の順序を逆にする", "reverse"),
    ("repeat_twice — 各単語を 2 回ずつ続けて書く", "repeat_twice"),
    ("odd_positions — 奇数番目の単語だけを残す", "odd_positions"),
    ("rotate_left — 先頭の単語を末尾に移す", "rotate_left"),
]
TEMPLATE_KIND_CHOICES = [
    ("見たことのあるテンプレート / Seen (学習に使った指示文)", "seen"),
    ("未見のテンプレート / Unseen (学習に使っていない指示文)", "unseen"),
]

INSTRUCTION_MARKDOWN = paragraphs(
    "017 が Hugging Face Hub の`sft-synthetic`ブランチに上げたモデルで、"
    f"{notebook_link('016')} の合成課題に答えます。"
    "課題は、英単語の列に対する 4 種類の操作で、正解は規則で決まります。"
    f"このモデルは {notebook_link('017')} が 016 の条件 1 のレシピ"
    "(損失マスクあり・LoRA・2,048 ステップ)で学習し直し、"
    "LoRA(Low-Rank Adaptation)を重みにマージしたものです。",
    "**自由な会話ができるモデルではありません。** "
    "合成課題だけで SFT したため、一般の文章の言語モデルとしての性能は SFT の前より"
    "大きく下がりました(017 で測った英語 Wikipedia の bits-per-byte は 1.668 から 3.152 へ悪化)。",
    "**見たことのあるテンプレートと未見のテンプレート**(016 の実験 A、5 シード)",
    "| 指示文 | 完全一致率 |\n"
    "|---|---|\n"
    "| 学習に使った言い回し(見たことのあるテンプレート) | 平均 0.804(条件 1) |\n"
    "| 学習に使っていない言い回し(未見のテンプレート) | 0.022〜0.078 |",
    "基準値は 0.25 です。これは、指示を無視して入力の単語だけから出力を決める戦略が達成できる"
    "完全一致率の上界で、未見のテンプレートの成績はそれをすべて下回りました。"
    "つまりこのモデルが「指示を読んで出力を切り替える」のは、学習で見た言い回しの範囲に限られ、"
    "未見の言い回しでは課題を識別できていません。下で 2 つを切り替えて試してください"
    "(数値は 016 の評価集合での値で、ここで入力する個々の例の成績ではありません)。",
)


def format_instruction_outcome(task: str, instruction: str, outcome: InstructionOutcome) -> str:
    """結果(判定・プロンプト・応答・正解)を Markdown にする。"""
    kind = template_kind_of(task, instruction)
    kind_label = "未見のテンプレート" if kind == TEMPLATE_UNSEEN else "見たことのあるテンプレート"
    if outcome.exact_match:
        verdict = "✅ 完全一致(形式を守り、単語の列が正解と一致)"
    elif outcome.format_ok:
        verdict = "⚠️ 形式は守ったが、単語の列が正解と違う"
    else:
        verdict = (
            "❌ 形式を守れなかった"
            "(終端記号`### End`で終わっていない、または途中に別の区切り記号が出た)"
        )
    answer = " ".join(outcome.answer)
    response = outcome.response_text.strip() or "(空)"
    return (
        f"### {verdict}\n\n"
        f"- 課題: `{task}` / 指示文: {kind_label}\n"
        f"- 正解(規則で決まる): `{answer}`\n"
        f"- モデルの応答(貪欲法、{outcome.num_new_tokens} トークン):\n\n"
        f"```text\n{response}\n```\n"
    )


def update_instruction_dropdown(task: str, kind: str) -> tuple[gr.Dropdown, str, str]:
    """課題・指示文の種類を変えたとき、指示文の選択肢を入れ替え、前の応答の表示を消す。"""
    choices = instruction_choices(task, kind)
    return gr.Dropdown(choices=choices, value=choices[0]), "", ""


def shuffle_words() -> str:
    """画面の「入れ替える」ボタン用。シードは呼び出しごとに変える。"""
    return random_words(time.time_ns() % (2**31))


def build_page(resources: Resources) -> None:
    """指示への応答のページを、``gr.Tab`` の中に組み立てる。"""

    @gpu(duration=INSTRUCTION_GPU_SECONDS)
    def answer_instruction(task: str, instruction: str, words_text: str) -> tuple[str, str]:
        """合成課題の指示に、``sft-synthetic`` のモデルが貪欲法で答える。"""
        try:
            parsed = parse_words(words_text)
            outcome = respond(
                resources.models["sft-synthetic"],
                resources.tokenizer,
                resources.device,
                task,
                instruction,
                parsed.words,
            )
        except ValueError as error:
            raise gr.Error(str(error)) from error
        report = format_instruction_outcome(task, instruction, outcome)
        if parsed.warnings:
            report += "\n" + "\n".join(f"> 注意: {warning}" for warning in parsed.warnings) + "\n"
        return outcome.prompt, report

    gr.Markdown(INSTRUCTION_MARKDOWN)
    with gr.Row():
        with gr.Column():
            task = gr.Dropdown(choices=TASK_CHOICES, value=TASK_NAMES[0], label="課題 / Task")
            template_kind = gr.Radio(
                choices=TEMPLATE_KIND_CHOICES,
                value="seen",
                label="指示文の種類 / Template kind",
            )
            first_choices = instruction_choices(TASK_NAMES[0], "seen")
            instruction = gr.Dropdown(
                choices=first_choices, value=first_choices[0], label="指示文 / Instruction"
            )
            words_text = gr.Textbox(
                label="入力の単語(英語、3〜8 個、空白区切り) / Words",
                value=random_words(0),
                lines=2,
            )
            shuffle_button = gr.Button("単語を入れ替える / Shuffle words")
            answer_button = gr.Button("応答を生成する / Answer", variant="primary")
        with gr.Column():
            prompt_preview = gr.Textbox(
                label="モデルに入力した指示部分 / Prompt", lines=6, interactive=False
            )
            answer_report = gr.Markdown()

    for selector in (task, template_kind):
        selector.change(
            update_instruction_dropdown,
            [task, template_kind],
            [instruction, prompt_preview, answer_report],
            api_visibility="private",
        )
    shuffle_button.click(shuffle_words, outputs=words_text, api_visibility="private")
    answer_button.click(
        answer_instruction,
        inputs=[task, instruction, words_text],
        outputs=[prompt_preview, answer_report],
        api_name="answer_instruction",
    )


PAGE = Page(PAGE_TITLE, build_page)
