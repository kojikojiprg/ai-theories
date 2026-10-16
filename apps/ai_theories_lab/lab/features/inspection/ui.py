"""モデルの中を見る画面(001 注意機構・005 トークナイザ・008 デコーディング戦略)。

処理は ``service.py`` にあり、ここは画面の組み立てと、結果の表示用の整形
(色分けの HTML・表・図)だけを行う。モデルを動かす処理は 1 回(「解析する」)だけで、
結果は ``gr.State`` に持つ。層・ヘッド・位置を選び直す
ときは、持っている結果から表示を作り直すだけで、モデルは動かさない(ZeroGPU の利用枠を使わない)。
"""

from __future__ import annotations

import html
import math

import gradio as gr
from matplotlib.figure import Figure

from lab.core.artifacts import Resources
from lab.core.device import gpu
from lab.features.inspection.figures import attention_figure
from lab.features.inspection.service import (
    MAX_INSPECTION_TOKENS,
    ChunkTrace,
    ModelAnalysis,
    PositionView,
    Tokenization,
    analyze,
    check_analysis_request,
    describe_position,
    readable_symbol,
    select_attention,
    tokenize,
)
from lab.ui.layout import Page
from lab.ui.links import notebook_link
from lab.ui.markdown import items, paragraphs

PAGE_TITLE = "モデルの中を見る / Inside the Model"

# @spaces.GPU の duration(秒)。最大 128 トークンの順伝播 1 回なので短い。
INSPECTION_GPU_SECONDS = 10

EXAMPLE_TEXTS = [
    "The unbelievable tokenization of uncommon words",
    "The history of the city began when the river town grew into a market and a port.",
    "Albert Einstein was born in Ulm, in the Kingdom of Württemberg.",
]

INSPECTION_MARKDOWN = paragraphs(
    "英語の文を入力すると、小型 GPT(`main`)の中で何が起きているかを、3 つの見方で表示します。"
    f"理論は {notebook_link('001')}・{notebook_link('005')}・{notebook_link('008')} です。",
    items(
        "**トークン分割**(005): 文がどのトークンに分かれるか。語が、学習した順位の高い結合の"
        "ルール(マージ規則)で結合されていく過程も、段階ごとに見られます。",
        "**注意の重み**(001): 層とヘッドを選んで、どのトークンがどのトークンに"
        "注意を向けたかを見ます。",
        "**トークンごとの確率**(008): 各位置で、モデルが次のトークンに付けた確率と、"
        "次のトークンの分布のエントロピー(予測の迷いの大きさ)を見ます。",
    ),
    "このモデルは、英語の Wikipedia だけで学習した小さなモデル(約 520 万パラメータ)です。"
    "一般的な知識や、英語以外の言語、記号の多い文は、ほとんど学習していません。"
    f"入力は {MAX_INSPECTION_TOKENS} トークンまでです。",
)

TOKENIZATION_MARKDOWN = paragraphs(
    "### 1. トークン分割 / Tokenization(005)",
    "文は、バイトレベルの BPE(Byte Pair Encoding)でトークンに分けられます。"
    "空白は直後の語の先頭に付き(`␣`で表示)、改行は`⏎`で表します。"
    "色は隣り合うトークンを区別するためだけのもので、小さな数字がトークン ID です。",
)
MERGE_MARKDOWN = (
    "語(事前分割で得た 1 単位)を選ぶと、1 バイトずつに分けた状態から、結合のルールが学習した順位の"
    "高いものから順に適用されて、トークンになるまでの過程が見られます。"
)
ATTENTION_MARKDOWN = paragraphs(
    "### 2. 注意の重み / Attention Weights(001)",
    "行(Query)のトークンが、列(Key)のどのトークンに、どれだけ注意を向けたか。"
    "右上(未来のトークン)は、因果マスクにより見られないので白です。各行の和は 1 です。"
    "ここで表示する値は、モデルの中で実際に Value との積に使われた注意の重みです"
    "(各層の注意機構が返す値を、順伝播の途中で読み取っています)。"
    f"位置エンコーディングの RoPE(Rotary Position Embedding、{notebook_link('003')})を"
    "適用した Query・Key から計算された値です。",
)
PROBABILITY_MARKDOWN = paragraphs(
    "### 3. トークンごとの確率 / Next-Token Probability(008)",
    "各位置は、「そのトークンまでを見た時点」を表します。その時点でモデルが、**次の実際のトークン**に"
    "付けた確率(`p`)を、橙の濃さで示します(確率が低く、予測しにくかったほど濃い)。"
    "下の青い棒は、その時点の次のトークンの分布のエントロピー(`H`、nats)で、"
    "棒が長いほど分布が広く、迷っていることを表します(最大は語彙の一様分布の ln V)。"
    "位置を選ぶと、その時点の上位の候補を表で見られます。",
)

_TOKEN_HUES = (210, 30, 130, 280, 55, 340, 170, 90)


# ---------------------------------------------------------------------------
# 表示用の整形(Gradio に依存しない。試験から直接呼べる)
# ---------------------------------------------------------------------------
def tokens_html(tokenization: Tokenization) -> str:
    """トークンを色分けして並べた HTML(小さな数字がトークン ID)。

    入力の文字列は、すべてエスケープする。
    """
    chips = []
    for piece in tokenization.pieces:
        hue = _TOKEN_HUES[piece.index % len(_TOKEN_HUES)]
        chips.append(
            f'<span title="位置 {piece.index} / ID {piece.token_id}" '
            f'style="display:inline-flex;flex-direction:column;align-items:center;margin:2px;'
            f"padding:3px 6px;border-radius:6px;background:hsla({hue},70%,55%,0.30);"
            f'font-family:ui-monospace,monospace;line-height:1.25;">'
            f'<span style="white-space:pre;font-size:15px;">{html.escape(piece.text)}</span>'
            f'<span style="font-size:10px;opacity:0.7;">{piece.token_id}</span></span>'
        )
    return f'<div style="line-height:1.1;">{"".join(chips)}</div>'


def tokenization_summary(tokenization: Tokenization) -> str:
    """``N 文字 → M トークン`` などの 1 行の要約。"""
    num_chars = len(tokenization.text)
    num_bytes = len(tokenization.text.encode("utf-8"))
    num_tokens = len(tokenization.pieces)
    return (
        f"**{num_chars} 文字({num_bytes} バイト)→ {num_tokens} トークン**"
        f"({len(tokenization.chunks)} 個の語に事前分割)"
    )


def chunk_choices(tokenization: Tokenization) -> list[tuple[str, int]]:
    """語(チャンク)の選択肢 ``(表示, 番号)``。結合が起きる語に、結合の段数を添える。"""
    choices = []
    for index, trace in enumerate(tokenization.chunks):
        label = "".join(readable_symbol(s) for s in trace.initial_symbols)
        choices.append((f"{index}: {label}({len(trace.steps)} 段階)", index))
    return choices


def default_chunk(tokenization: Tokenization) -> int:
    """最初に表示する語: 結合の段数が最も多い(過程が見どころの)語。"""
    return max(range(len(tokenization.chunks)), key=lambda i: len(tokenization.chunks[i].steps))


MERGE_HEADERS = ["段階", "結合したペア", "規則の順位", "結合後のシンボル列"]


def merge_rows(trace: ChunkTrace) -> list[list[str]]:
    """1 つの語の結合の過程を、表の行にする。"""

    def joined(symbols) -> str:
        return " | ".join(readable_symbol(s) for s in symbols)

    rows = [["0(開始)", "—", "—", joined(trace.initial_symbols)]]
    for number, step in enumerate(trace.steps, start=1):
        merged = readable_symbol(step.pair[0] + step.pair[1])
        pair = f"{readable_symbol(step.pair[0])} + {readable_symbol(step.pair[1])} → {merged}"
        rows.append([str(number), pair, str(step.rank), joined(step.symbols_after)])
    if not trace.steps:
        rows.append(["—", "(結合できるペアがない)", "—", joined(trace.final_symbols)])
    return rows


def probability_html(
    analysis: ModelAnalysis, tokenization: Tokenization, selected: int | None = None
) -> str:
    """位置ごとの確率(橙の濃さ)とエントロピー(青い棒)を並べた HTML。選んだ位置は枠で囲む。"""
    chips = []
    for index, piece in enumerate(tokenization.pieces):
        probability = float(analysis.next_probability[index])
        entropy = float(analysis.entropy[index])
        if math.isnan(probability):
            alpha, p_text = 0.05, "p=—"
            title = f"位置 {index}: 実際の次のトークンはない(続きの予測のみ)"
        else:
            surprisal_bits = -math.log2(max(probability, 1e-12))
            alpha = 0.08 + 0.57 * min(1.0, surprisal_bits / 10.0)
            p_text = f"p={probability:.3f}"
            rank = int(analysis.next_rank[index])
            title = f"位置 {index}: 次のトークンの確率 {probability:.4f}、順位 {rank}"
        bar = max(2.0, 100.0 * entropy / analysis.max_entropy)
        border = "hsl(260,80%,55%)" if index == selected else "transparent"
        chips.append(
            f'<span title="{html.escape(title)}、エントロピー {entropy:.2f} nats" '
            f'style="display:inline-flex;flex-direction:column;margin:2px;padding:3px 6px;'
            f"min-width:48px;border-radius:6px;border:2px solid {border};"
            f"background:hsla(25,90%,50%,{alpha:.3f});font-family:ui-monospace,monospace;"
            f'line-height:1.25;">'
            f'<span style="white-space:pre;font-size:15px;">{html.escape(piece.text)}</span>'
            f'<span style="font-size:10px;opacity:0.8;">{p_text}</span>'
            f'<span style="font-size:10px;opacity:0.8;">H={entropy:.2f}</span>'
            f'<span style="display:block;height:5px;border-radius:3px;margin-top:2px;'
            f'background:hsla(210,20%,50%,0.25);">'
            f'<span style="display:block;height:5px;border-radius:3px;width:{bar:.1f}%;'
            f'background:hsla(210,80%,50%,0.9);"></span></span></span>'
        )
    return f'<div style="line-height:1.1;">{"".join(chips)}</div>'


def position_choices(tokenization: Tokenization) -> list[tuple[str, int]]:
    """位置の選択肢 ``(表示, 位置)``。"""
    return [(f"{piece.index}: {piece.text}", piece.index) for piece in tokenization.pieces]


CANDIDATE_HEADERS = ["順位", "トークン", "ID", "確率"]


def candidate_rows(view: PositionView) -> list[list[str]]:
    """位置の上位の候補を、表の行にする。"""
    return [[str(c.rank), c.text, str(c.token_id), f"{c.probability:.4f}"] for c in view.candidates]


def position_summary(view: PositionView, max_entropy: float) -> str:
    """選んだ位置の説明(Markdown)。"""
    lines = [
        f"位置 {view.position}(`{view.context_text}`)までを見た時点の、次のトークンの予測です。"
        f"エントロピーは {view.entropy:.2f} nats(最大 {max_entropy:.2f})です。"
    ]
    if view.actual_text is None:
        lines.append("入力の最後なので、実際の次のトークンはありません。下の表は続きの候補です。")
    else:
        lines.append(
            f"実際の次のトークンは`{view.actual_text}`で、確率は {view.actual_probability:.4f}、"
            f"順位は {view.actual_rank} 位でした。"
        )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
def layer_choices(num_layers: int) -> list[tuple[str, int]]:
    """層の選択肢 ``(表示, 番号)``。"""
    return [(f"層 {i} / Layer {i}", i) for i in range(num_layers)]


def head_choices(num_heads: int) -> list[tuple[str, str]]:
    """ヘッドの選択肢 ``(表示, 値)``。

    値は、全ヘッドの平均が ``"mean"``、それ以外はヘッド番号の文字列。
    """
    return [("全ヘッドの平均 / Mean of all heads", "mean")] + [
        (f"ヘッド {k} / Head {k}", str(k)) for k in range(num_heads)
    ]


def build_page(resources: Resources) -> None:
    """モデルの中を見るページを、``gr.Tab`` の中に組み立てる。"""
    tokenizer = resources.tokenizer
    main = resources.models["main"]

    def run_tokenization(text: str):
        """文をトークンに分ける(モデルは動かさない)。"""
        try:
            tokenization = tokenize(tokenizer, text)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        default = default_chunk(tokenization)
        return (
            tokenization,
            None,  # 前の解析の結果は捨てる
            tokens_html(tokenization),
            tokenization_summary(tokenization),
            gr.Dropdown(choices=chunk_choices(tokenization), value=default),
            merge_rows(tokenization.chunks[default]),
        )

    @gpu(duration=INSPECTION_GPU_SECONDS)
    def run_model_on_gpu(tokenization: Tokenization) -> ModelAnalysis:
        """検査済みのトークン列をモデルに 1 回通す(GPU を使う処理は、ここだけ)。"""
        return analyze(main, resources.device, tokenization.token_ids)

    def run_model(tokenization: Tokenization | None) -> ModelAnalysis:
        """「解析する」の本体。入力の検査(GPU を使わない)を先に行い、誤りは ``gr.Error``。"""
        try:
            checked = check_analysis_request(tokenization)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return run_model_on_gpu(checked)

    def render_attention(
        analysis: ModelAnalysis | None, tokenization: Tokenization | None, layer: int, head: str
    ) -> Figure | None:
        if analysis is None or tokenization is None:
            return None
        matrix = select_attention(analysis, int(layer), None if head == "mean" else int(head))
        which = "mean of all heads" if head == "mean" else f"head {head}"
        return attention_figure(matrix, tokenization.token_texts, f"Layer {layer}, {which}")

    def render_position(
        analysis: ModelAnalysis | None, tokenization: Tokenization | None, position: int | None
    ):
        if analysis is None or tokenization is None or position is None:
            return "", [], ""
        view = describe_position(analysis, tokenizer, int(position))
        return (
            probability_html(analysis, tokenization, int(position)),
            candidate_rows(view),
            position_summary(view, analysis.max_entropy),
        )

    def render_after_analysis(
        analysis: ModelAnalysis | None,
        tokenization: Tokenization | None,
        layer: int,
        head: str,
    ):
        """解析の直後に、位置の選択肢を作り直し、選んでいる層・ヘッドと、最後の位置で表示する。"""
        if analysis is None or tokenization is None:
            return None, gr.Dropdown(), "", [], ""
        last = analysis.num_tokens - 1
        figure = render_attention(analysis, tokenization, layer, head)
        probability, rows, summary = render_position(analysis, tokenization, last)
        return (
            figure,
            gr.Dropdown(choices=position_choices(tokenization), value=last),
            probability,
            rows,
            summary,
        )

    def render_merge_steps(tokenization: Tokenization | None, chunk_index: int | None):
        if tokenization is None or chunk_index is None:
            return []
        return merge_rows(tokenization.chunks[int(chunk_index)])

    # ---- 部品の配置 ----
    gr.Markdown(INSPECTION_MARKDOWN)
    text_box = gr.Textbox(
        label="英語の文 / Sentence",
        value=EXAMPLE_TEXTS[0],
        lines=3,
        max_length=2000,
    )
    gr.Examples(
        examples=[[t] for t in EXAMPLE_TEXTS], inputs=[text_box], label="入力の例 / Examples"
    )
    analyze_button = gr.Button("解析する / Analyze", variant="primary")
    tokenization_state = gr.State(None)
    analysis_state = gr.State(None)

    gr.Markdown(TOKENIZATION_MARKDOWN)
    summary_md = gr.Markdown()
    tokens_view = gr.HTML()
    gr.Markdown(MERGE_MARKDOWN)
    chunk_dropdown = gr.Dropdown(choices=[], label="語 / Word")
    merge_table = gr.Dataframe(
        headers=MERGE_HEADERS, value=[], interactive=False, wrap=True, label="結合の過程 / Merges"
    )

    gr.Markdown(ATTENTION_MARKDOWN)
    with gr.Row():
        layer_dropdown = gr.Dropdown(
            choices=layer_choices(main.model.num_layers), value=0, label="層 / Layer"
        )
        head_dropdown = gr.Dropdown(
            choices=head_choices(main.model.num_heads), value="mean", label="ヘッド / Head"
        )
    attention_plot = gr.Plot(label="注意の重み / Attention")

    gr.Markdown(PROBABILITY_MARKDOWN)
    probability_view = gr.HTML()
    position_dropdown = gr.Dropdown(choices=[], label="位置 / Position")
    position_md = gr.Markdown()
    candidates_table = gr.Dataframe(
        headers=CANDIDATE_HEADERS,
        value=[],
        interactive=False,
        label="次のトークンの上位の候補 / Top Candidates",
    )

    # ---- イベント ----
    (
        analyze_button.click(
            run_tokenization,
            inputs=[text_box],
            outputs=[
                tokenization_state,
                analysis_state,
                tokens_view,
                summary_md,
                chunk_dropdown,
                merge_table,
            ],
            api_name="inspect_tokenize",
        )
        .success(
            run_model,
            inputs=[tokenization_state],
            outputs=[analysis_state],
            api_name="inspect_model",
        )
        .success(
            render_after_analysis,
            inputs=[analysis_state, tokenization_state, layer_dropdown, head_dropdown],
            outputs=[
                attention_plot,
                position_dropdown,
                probability_view,
                candidates_table,
                position_md,
            ],
            api_name="inspect_render",
        )
    )
    for selector in (layer_dropdown, head_dropdown):
        selector.change(
            render_attention,
            inputs=[analysis_state, tokenization_state, layer_dropdown, head_dropdown],
            outputs=[attention_plot],
            api_visibility="private",
        )
    position_dropdown.change(
        render_position,
        inputs=[analysis_state, tokenization_state, position_dropdown],
        outputs=[probability_view, candidates_table, position_md],
        api_visibility="private",
    )
    chunk_dropdown.change(
        render_merge_steps,
        inputs=[tokenization_state, chunk_dropdown],
        outputs=[merge_table],
        api_visibility="private",
    )


PAGE = Page(PAGE_TITLE, build_page)
