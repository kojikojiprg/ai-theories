"""文章の生成の画面(008 デコーディング戦略・010 KV キャッシュ)。

処理は ``service.py`` にあり、ここは画面の組み立てと、処理の結果の表示用の整形だけを行う。
GPU を使う処理は ``@gpu`` を付けた関数の中で行う(ZeroGPU の外では何もしない)。
"""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from lab.core.artifacts import Resources
from lab.core.device import gpu
from lab.features.generation.service import (
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TEMPERATURE,
    METHOD_TOP_K,
    METHOD_TOP_P,
    DecodingSettings,
    GenerationEvent,
    check_generation_request,
    generate_events,
)
from lab.ui.layout import Page
from lab.ui.links import notebook_link
from lab.ui.markdown import items, paragraphs

PAGE_TITLE = "文章の生成 / Text Generation"

# @spaces.GPU の duration(秒)。ZeroGPU は、閲覧者の 1 日の利用枠の残りが
# これ以上あるかを確かめて割り当てる。小さなモデルなので、最大のトークン数・ビーム数でも
# 数秒で終わる。短いほど待ち行列で優先される。
GENERATION_GPU_SECONDS = 30
COMPARISON_GPU_SECONDS = 60

# 画面の選択肢(表示名, 内部の値)
METHOD_CHOICES = [
    ("貪欲法 / Greedy decoding", METHOD_GREEDY),
    ("temperature サンプリング / Temperature sampling", METHOD_TEMPERATURE),
    ("top-k サンプリング / Top-k sampling", METHOD_TOP_K),
    ("top-p サンプリング / Top-p (nucleus) sampling", METHOD_TOP_P),
    ("ビームサーチ / Beam search", METHOD_BEAM),
]
MODEL_CHOICES = [
    ("main — 標準の多頭注意機構 / Multi-Head Attention (008)", "main"),
    ("gqa — Grouped-Query Attention に変換して追加学習 (010)", "gqa"),
]

# 入力の例(モデルが英語で学習しているため英語)
EXAMPLE_PROMPTS = [
    "The history of the city began when",
    "Albert Einstein was born in",
    "The film was released in",
]
# 008 実験 A の退化(繰り返し)を見るための組: 同じ入力・同じシードで貪欲法と top-p を並べる。
DEGENERATION_EXAMPLE_PROMPT = "The history of the town of"

GENERATION_MARKDOWN = paragraphs(
    "英語の書き出しを入力すると、小型 GPT が続きを 1 トークンずつ生成します。理論は "
    f"{notebook_link('008')} と {notebook_link('010')} です。",
    "**デコードの方法**(008)",
    items(
        "**貪欲法**(Greedy decoding): 常に最も確率の高いトークンを選びます。"
        "同じ入力なら毎回同じ結果で、同じ句の繰り返し(退化、Degeneration)が起きやすくなります。",
        "**temperature サンプリング**: `softmax(logits / T)`から抽選します。"
        "T が大きいほど多様に、小さいほど貪欲法に近くなります。",
        "**top-k サンプリング**: 確率の上位 k 個に絞ってから抽選します(Fan et al., 2018)。",
        "**top-p サンプリング**(nucleus sampling): "
        "累積確率が p に達する最小の集合に絞ってから抽選します(Holtzman et al., 2020)。",
        "**ビームサーチ**(Beam search): 長さ正規化した対数尤度の高い候補を、"
        "ビーム数だけ残しながら探索します。`src`の実装は KV キャッシュと逐次表示に"
        "対応していないため、KV キャッシュの設定は使われず、結果は最後にまとめて表示します。",
    ),
    "008 の実験 A(24 トークンの書き出し 8 個から 64 トークンを生成)では、"
    "4-gram 重複率の平均が貪欲法・ビームサーチで 0.5205、temperature・top-p で 0.0002 でした"
    "(このアプリのモデルを測り直した値ではありません)。",
    "**KV キャッシュ**(010): 過去のトークンの Key / Value を保存して、"
    "新しいトークン 1 個ぶんだけを計算します。"
    "貪欲法では、使っても使わなくても生成結果は完全に一致します。"
    "`gqa`は Key / Value のヘッド数を減らした版で、KV キャッシュのメモリが小さくなります"
    "(上の「読み込んだモデル」の表)。",
)

COMPARE_MARKDOWN = paragraphs(
    "同じ書き出し・同じ乱数のシード・同じ設定で、デコードの方法を 2 つ並べて生成します"
    "(temperature・k・p・ビーム数などは上の設定を共通で使います)。"
    "貪欲法と top-p を並べると、繰り返しの違いが見えやすくなります。",
)

TIMING_NOTE = (
    "速さは、条件ごとに 1 回だけ測った参考値です(ウォームアップなし・繰り返しなし)。"
    "デバイスの同期を取ったうえで、GPU の割り当てにかかる時間と画面の更新の時間を除いて"
    "測っています。同じシードでもデバイスが違うと、サンプリングの結果は変わりえます。"
)


def format_generation_stats(label: str, num_tokens: int, seconds: float, device: str) -> str:
    """生成したトークン数・時間・毎秒のトークン数(1 回の計測の参考値)。"""
    rate = num_tokens / seconds if seconds > 0 else float("inf")
    return (
        f"**{label}**: 生成したトークン数 {num_tokens} / 時間 {seconds:.3f} 秒 / "
        f"毎秒 {rate:.1f} トークン({device}・1 回の計測)"
    )


def format_beam_candidates(candidates: tuple[tuple[float, str], ...]) -> str:
    """ビームサーチの候補の一覧(スコアの降順)。"""
    lines = [f"ビームサーチの候補(長さ正規化した対数尤度の降順、{len(candidates)} 件):"]
    for rank, (score, continuation) in enumerate(candidates, start=1):
        tail = continuation.replace("\n", " ")
        lines.append(f"{rank}. スコア {score:.3f}: …{tail[-60:]}")
    return "\n".join(lines)


def describe_event(event: GenerationEvent, label: str, device: str) -> tuple[str, str, str]:
    """生成の途中経過を、``(本文, 速さの表示, 補足)`` にする。"""
    if event.waiting_for_beam:
        return event.text, f"**{label}**: ビームサーチを計算中…(最後にまとめて表示します)", ""
    if not event.finished:
        return event.text, f"**{label}**: 生成中… {event.num_new_tokens} トークン", ""
    assert event.compute_seconds is not None
    stats = format_generation_stats(label, event.num_new_tokens, event.compute_seconds, device)
    details = format_beam_candidates(event.beam_candidates) if event.beam_candidates else ""
    return event.text, stats, details


def build_settings(
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
    """画面の値から設定を作る。Gradio のスライダー・数値欄は float で届くことがあるので、
    整数の設定は int にそろえる。"""
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


def build_page(resources: Resources) -> None:
    """文章の生成のページを、``gr.Tab`` の中に組み立てる。

    画面のイベントの関数(``generate_text``・``compare_methods``)は、GPU を使わずに判定できる
    入力の検査を先に行い、誤りなら ``gr.Error`` にする。検査を通った入力だけを、``@gpu`` を付けた
    関数(``*_on_gpu``。計算だけを行う)に渡す。誤った入力で ZeroGPU の利用枠を使わないため。
    """
    main = resources.models["main"]
    device = resources.device.type

    def check_inputs(model_key: str, prompt: str, *settings_list: DecodingSettings) -> None:
        """入力の検査(GPU を使わない)。誤りは ``gr.Error``。"""
        try:
            for settings in settings_list:
                check_generation_request(
                    resources.models[model_key], resources.tokenizer, prompt, settings
                )
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def run(
        model_key: str, prompt: str, settings: DecodingSettings, label: str
    ) -> Iterator[tuple[str, str, str]]:
        for event in generate_events(
            resources.models[model_key], resources.tokenizer, resources.device, prompt, settings
        ):
            yield describe_event(event, label, device)

    @gpu(duration=GENERATION_GPU_SECONDS)
    def generate_text_on_gpu(
        prompt: str, model_key: str, settings: DecodingSettings
    ) -> Iterator[tuple[str, str, str]]:
        """検査済みの入力で、書き出しの続きを生成する(逐次表示)。"""
        yield from run(model_key, prompt, settings, "生成結果")

    @gpu(duration=COMPARISON_GPU_SECONDS)
    def compare_methods_on_gpu(
        prompt: str, model_key: str, settings_a: DecodingSettings, settings_b: DecodingSettings
    ) -> Iterator[tuple[str, str, str, str, str, str]]:
        """検査済みの入力で、デコードの方法 2 つを順に(同じ入力・同じシードで)生成して並べる。"""
        state = {"a": ("", "", ""), "b": ("", "", "")}
        for side, settings in (("a", settings_a), ("b", settings_b)):
            for described in run(model_key, prompt, settings, f"方法 {side.upper()}"):
                state[side] = described
                yield (*state["a"], *state["b"])

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
        """書き出しの続きを生成する(逐次表示)。"""
        settings = build_settings(
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
        check_inputs(model_key, prompt, settings)
        yield from generate_text_on_gpu(prompt, model_key, settings)

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
        """同じ入力・同じシード・同じ設定で、デコードの方法 2 つを順に生成して並べる。"""
        if method_a == method_b:
            raise gr.Error("比べる 2 つのデコードの方法を、別のものにしてください。")
        settings_a, settings_b = (
            build_settings(
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
            for method in (method_a, method_b)
        )
        check_inputs(model_key, prompt, settings_a, settings_b)  # 2 つとも、生成の前に検査する
        yield from compare_methods_on_gpu(prompt, model_key, settings_a, settings_b)

    gr.Markdown(GENERATION_MARKDOWN)
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="書き出し(英語) / Prompt",
                value=EXAMPLE_PROMPTS[0],
                lines=3,
                max_length=2000,
            )
            gr.Examples(
                examples=[[p] for p in EXAMPLE_PROMPTS],
                inputs=[prompt],
                label="入力の例 / Examples",
            )
            model_key = gr.Radio(
                choices=MODEL_CHOICES, value="main", label="モデル / Model (Hub のブランチ)"
            )
            use_cache = gr.Checkbox(
                value=True,
                label="KV キャッシュを使う / Use KV cache",
                info="ビームサーチでは使われません。",
            )
            method = gr.Dropdown(
                choices=METHOD_CHOICES, value=METHOD_TOP_P, label="デコードの方法 / Decoding method"
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
            beam_size = gr.Slider(1, 8, value=4, step=1, label="ビーム数(ビームサーチのとき)")
            length_penalty = gr.Slider(
                0.0, 2.0, value=1.0, step=0.1, label="長さペナルティ(ビームサーチのとき)"
            )
            max_new_tokens = gr.Slider(
                1,
                main.max_sequence_length,
                value=64,
                step=1,
                label="最大トークン数 / Max new tokens",
                info=(
                    f"KV キャッシュを使うときは、書き出しとの和が {main.max_sequence_length} 以下"
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
            output_text = gr.Textbox(label="生成結果 / Output(書き出しに続けて表示)", lines=12)
            output_stats = gr.Markdown()
            output_details = gr.Markdown()
            gr.Markdown(TIMING_NOTE)

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

    with gr.Accordion("2 つのデコードの方法を並べて比べる / Compare Two Methods", open=False):
        gr.Markdown(COMPARE_MARKDOWN)
        with gr.Row():
            method_a = gr.Dropdown(choices=METHOD_CHOICES, value=METHOD_GREEDY, label="方法 A")
            method_b = gr.Dropdown(choices=METHOD_CHOICES, value=METHOD_TOP_P, label="方法 B")
        gr.Examples(
            examples=[[DEGENERATION_EXAMPLE_PROMPT, METHOD_GREEDY, METHOD_TOP_P]],
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


PAGE = Page(PAGE_TITLE, build_page)
