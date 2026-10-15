"""Text Generation Playground(Hugging Face Spaces の ZeroGPU で動く Gradio アプリ)。

ai-theories でスクラッチ実装した小型 GPT で、次の 2 つを試す。

- タブ 1 文章の生成: 008 のデコーディング戦略と、010 の KV キャッシュ・Grouped-Query Attention。
- タブ 2 指示への応答: 016 の合成の指示データと、017 が上げた SFT(Supervised Fine-Tuning)のモデル。

**ZeroGPU の作法**

- ``spaces`` は、CUDA を使うパッケージ(torch など)より前に import する(``spaces`` が要求する)。
- モデルはモジュールの読み込み時に 1 回だけ読み込み、デバイスへ載せる(ZeroGPU の推奨)。
- GPU を使う処理は ``@spaces.GPU`` を付けた関数の中で行う。ZeroGPU の外(手元の MPS や CPU)では、
  この装飾子は何もせず、同じファイルがそのまま動く。

手元での起動: このディレクトリで ``uv run python app.py``。
"""

import spaces  # ZeroGPU: CUDA を使うパッケージ(torch など)より前に import する

# isort: split

from collections.abc import Iterator

import gradio as gr

from playground import text
from playground.artifacts import load_resources
from playground.generation import (
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TOP_P,
    DecodingSettings,
    run_beam_search,
    stream_generation,
)
from playground.instruction import (
    TASK_NAMES,
    instruction_choices,
    parse_words,
    random_words,
    respond,
)

# --- 起動時に 1 回だけ読み込む(トークナイザと 3 ブランチのモデルを、同時にメモリへ載せる) ---
RESOURCES = load_resources()

# @spaces.GPU の duration(秒)。ZeroGPU は、閲覧者の 1 日の利用枠の残りが
# これ以上あるかを確かめて割り当てる。
# 小さなモデルなので、最大のトークン数・ビーム数でも数秒で終わる。短いほど待ち行列で優先される。
GENERATION_GPU_SECONDS = 30
COMPARISON_GPU_SECONDS = 60
INSTRUCTION_GPU_SECONDS = 15

SPACE_TITLE = "Text Generation Playground"


def _decode(token_ids) -> str:
    return RESOURCES.tokenizer.decode(list(token_ids))


def _build_settings(
    method: str,
    use_cache: bool,
    temperature: float,
    top_k: float,
    top_p: float,
    beam_size: float,
    length_penalty: float,
    max_new_tokens: float,
    seed: float | None,
) -> DecodingSettings:
    # Gradio のスライダー・数値欄は float で届くことがあるので、整数の設定は int にそろえる。
    return DecodingSettings(
        method=method,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        beam_size=int(beam_size),
        length_penalty=float(length_penalty),
        seed=0 if seed is None else int(seed),
        use_cache=bool(use_cache),
    )


def _generate_events(
    model_key: str, prompt: str, settings: DecodingSettings, label: str
) -> Iterator[tuple[str, str, str]]:
    """1 つのデコードの方法で生成し、``(本文, 速さの表示, 補足)`` を途中経過ごとに返す。

    ``@spaces.GPU`` を付けた関数の中から呼ぶ(GPU の割り当てが済んだ後の計算だけを測る)。
    """
    loaded = RESOURCES.models[model_key]
    device = RESOURCES.device
    prompt_ids = RESOURCES.tokenizer.encode(prompt)
    try:
        settings.validate(len(prompt_ids), loaded.max_sequence_length)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    if settings.method == METHOD_BEAM:
        yield prompt, f"**{label}**: ビームサーチを計算中…(最後にまとめて表示します)", ""
        outcome = run_beam_search(loaded.model, prompt_ids, settings, device)
        body, details = text.format_beam_outcome(outcome, _decode, prompt)
        stats = text.format_generation_stats(
            label, settings.max_new_tokens, outcome.compute_seconds, device.type
        )
        yield body, stats, details
        return

    for progress in stream_generation(loaded.model, prompt_ids, settings, device):
        body = prompt + _decode(progress.new_token_ids)
        if progress.finished:
            stats = text.format_generation_stats(
                label, len(progress.new_token_ids), progress.compute_seconds, device.type
            )
        else:
            stats = f"**{label}**: 生成中… {len(progress.new_token_ids)} トークン"
        yield body, stats, ""


@spaces.GPU(duration=GENERATION_GPU_SECONDS)
def generate_text(
    prompt: str,
    model_key: str,
    use_cache: bool,
    method: str,
    temperature: float,
    top_k: float,
    top_p: float,
    beam_size: float,
    length_penalty: float,
    max_new_tokens: float,
    seed: float | None,
) -> Iterator[tuple[str, str, str]]:
    """タブ 1: 書き出しの続きを生成する(逐次表示)。"""
    settings = _build_settings(
        method,
        use_cache,
        temperature,
        top_k,
        top_p,
        beam_size,
        length_penalty,
        max_new_tokens,
        seed,
    )
    yield from _generate_events(model_key, prompt, settings, "生成結果")


@spaces.GPU(duration=COMPARISON_GPU_SECONDS)
def compare_methods(
    prompt: str,
    model_key: str,
    use_cache: bool,
    method_a: str,
    method_b: str,
    temperature: float,
    top_k: float,
    top_p: float,
    beam_size: float,
    length_penalty: float,
    max_new_tokens: float,
    seed: float | None,
) -> Iterator[tuple[str, str, str, str, str, str]]:
    """タブ 1: 同じ入力・同じシード・同じ設定で、デコードの方法 2 つを順に生成して並べる。"""
    if method_a == method_b:
        raise gr.Error("比べる 2 つのデコードの方法を、別のものにしてください。")
    state = {"a": ("", "", ""), "b": ("", "", "")}
    for side, method in (("a", method_a), ("b", method_b)):
        settings = _build_settings(
            method,
            use_cache,
            temperature,
            top_k,
            top_p,
            beam_size,
            length_penalty,
            max_new_tokens,
            seed,
        )
        for event in _generate_events(model_key, prompt, settings, f"方法 {side.upper()}"):
            state[side] = event
            yield (*state["a"], *state["b"])


@spaces.GPU(duration=INSTRUCTION_GPU_SECONDS)
def answer_instruction(task: str, instruction: str, words_text: str) -> tuple[str, str]:
    """タブ 2: 合成課題の指示に、``sft-synthetic`` のモデルが貪欲法で答える。"""
    try:
        parsed = parse_words(words_text)
        outcome = respond(
            RESOURCES.models["sft-synthetic"],
            RESOURCES.tokenizer,
            RESOURCES.device,
            task,
            instruction,
            parsed.words,
        )
    except ValueError as error:
        raise gr.Error(str(error)) from error
    report = text.format_instruction_outcome(task, instruction, outcome)
    if parsed.warnings:
        report += "\n" + "\n".join(f"> 注意: {warning}" for warning in parsed.warnings) + "\n"
    return outcome.prompt, report


def update_instruction_dropdown(task: str, kind: str) -> tuple[gr.Dropdown, str, str]:
    """課題・指示文の種類を変えたとき、指示文の選択肢を入れ替え、前の応答の表示を消す。"""
    choices = instruction_choices(task, kind)
    return gr.Dropdown(choices=choices, value=choices[0]), "", ""


def shuffle_words() -> str:
    # 画面の「入れ替える」ボタン用。シードは呼び出しごとに変える。
    import time

    return random_words(time.time_ns() % (2**31))


def build_demo() -> gr.Blocks:
    main = RESOURCES.models["main"]
    with gr.Blocks(title=SPACE_TITLE) as demo:
        gr.Markdown(text.intro_markdown(main))
        with gr.Accordion("読み込んだモデル / Loaded Models", open=False):
            gr.Markdown(text.describe_models(RESOURCES.models, RESOURCES.device))

        with gr.Tabs():
            # ---------------- タブ 1: 文章の生成 ----------------
            with gr.Tab("文章の生成 / Text Generation"):
                gr.Markdown(text.generation_tab_markdown())
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="書き出し(英語) / Prompt",
                            value=text.EXAMPLE_PROMPTS[0],
                            lines=3,
                            max_length=2000,
                        )
                        gr.Examples(
                            examples=[[p] for p in text.EXAMPLE_PROMPTS],
                            inputs=[prompt],
                            label="入力の例 / Examples",
                        )
                        model_key = gr.Radio(
                            choices=text.MODEL_CHOICES,
                            value="main",
                            label="モデル / Model (Hub のブランチ)",
                        )
                        use_cache = gr.Checkbox(
                            value=True,
                            label="KV キャッシュを使う / Use KV cache",
                            info="ビームサーチでは使われません。",
                        )
                        method = gr.Dropdown(
                            choices=text.METHOD_CHOICES,
                            value=METHOD_TOP_P,
                            label="デコードの方法 / Decoding method",
                        )
                        temperature = gr.Slider(
                            0.1,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="temperature(temperature・top-k・top-p のとき)",
                        )
                        top_k = gr.Slider(1, 200, value=40, step=1, label="k(top-k のとき)")
                        top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="p(top-p のとき)")
                        beam_size = gr.Slider(
                            1, 8, value=4, step=1, label="ビーム数(ビームサーチのとき)"
                        )
                        length_penalty = gr.Slider(
                            0.0,
                            2.0,
                            value=1.0,
                            step=0.1,
                            label="長さペナルティ(ビームサーチのとき)",
                        )
                        max_new_tokens = gr.Slider(
                            1,
                            main.max_sequence_length,
                            value=64,
                            step=1,
                            label="最大トークン数 / Max new tokens",
                            info=(
                                "KV キャッシュを使うときは、書き出しとの和が "
                                f"{main.max_sequence_length} 以下"
                            ),
                        )
                        seed = gr.Number(
                            value=0,
                            precision=0,
                            minimum=0,
                            label="乱数のシード / Seed(サンプリングのとき)",
                        )
                    with gr.Column(scale=3):
                        generate_button = gr.Button("生成する / Generate", variant="primary")
                        output_text = gr.Textbox(
                            label="生成結果 / Output(書き出しに続けて表示)", lines=12
                        )
                        output_stats = gr.Markdown()
                        output_details = gr.Markdown()
                        gr.Markdown(text.TIMING_NOTE)

                shared_inputs = [
                    prompt,
                    model_key,
                    use_cache,
                    method,
                    temperature,
                    top_k,
                    top_p,
                    beam_size,
                    length_penalty,
                    max_new_tokens,
                    seed,
                ]
                generate_button.click(
                    generate_text,
                    inputs=shared_inputs,
                    outputs=[output_text, output_stats, output_details],
                    api_name="generate_text",
                )

                with gr.Accordion(
                    "2 つのデコードの方法を並べて比べる / Compare Two Methods", open=False
                ):
                    gr.Markdown(text.COMPARE_MARKDOWN)
                    with gr.Row():
                        method_a = gr.Dropdown(
                            choices=text.METHOD_CHOICES, value=METHOD_GREEDY, label="方法 A"
                        )
                        method_b = gr.Dropdown(
                            choices=text.METHOD_CHOICES, value=METHOD_TOP_P, label="方法 B"
                        )
                    gr.Examples(
                        examples=[[text.DEGENERATION_EXAMPLE_PROMPT, METHOD_GREEDY, METHOD_TOP_P]],
                        inputs=[prompt, method_a, method_b],
                        label="退化(繰り返し)を見る例: 貪欲法と top-p",
                    )
                    compare_button = gr.Button("並べて生成する / Compare", variant="primary")
                    with gr.Row():
                        with gr.Column():
                            text_a = gr.Textbox(label="方法 A の生成結果", lines=10)
                            stats_a = gr.Markdown()
                            details_a = gr.Markdown()
                        with gr.Column():
                            text_b = gr.Textbox(label="方法 B の生成結果", lines=10)
                            stats_b = gr.Markdown()
                            details_b = gr.Markdown()
                    compare_button.click(
                        compare_methods,
                        inputs=[
                            prompt,
                            model_key,
                            use_cache,
                            method_a,
                            method_b,
                            temperature,
                            top_k,
                            top_p,
                            beam_size,
                            length_penalty,
                            max_new_tokens,
                            seed,
                        ],
                        outputs=[text_a, stats_a, details_a, text_b, stats_b, details_b],
                        api_name="compare_methods",
                    )

            # ---------------- タブ 2: 指示への応答 ----------------
            with gr.Tab("指示への応答 / Instruction Following"):
                gr.Markdown(text.instruction_tab_markdown())
                with gr.Row():
                    with gr.Column():
                        task = gr.Dropdown(
                            choices=text.TASK_CHOICES, value=TASK_NAMES[0], label="課題 / Task"
                        )
                        template_kind = gr.Radio(
                            choices=text.TEMPLATE_KIND_CHOICES,
                            value="seen",
                            label="指示文の種類 / Template kind",
                        )
                        first_choices = instruction_choices(TASK_NAMES[0], "seen")
                        instruction = gr.Dropdown(
                            choices=first_choices,
                            value=first_choices[0],
                            label="指示文 / Instruction",
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

        gr.Markdown(text.about_markdown())
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
