"""画面に表示する文言と、表示用の整形。

日本語を主とし、専門用語は英語を併記する。理論の数値(008 の実験 A、016 の評価)は、
各ノートブックのセル出力から転記したもので、このアプリのモデルをここで測り直したものではない。
"""

from __future__ import annotations

import torch

from playground.artifacts import LoadedModel
from playground.generation import (
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TEMPERATURE,
    METHOD_TOP_K,
    METHOD_TOP_P,
    BeamOutcome,
)
from playground.instruction import TEMPLATE_UNSEEN, InstructionOutcome, template_kind_of

REPOSITORY_URL = "https://github.com/kojikojiprg/ai-theories"
_NOTEBOOKS = f"{REPOSITORY_URL}/blob/main/theories"
NOTEBOOK_URLS = {
    "008": f"{_NOTEBOOKS}/02_pretraining/008_decoding_strategies.ipynb",
    "010": f"{_NOTEBOOKS}/03_efficient_training/010_kv_cache_and_inference_compute.ipynb",
    "016": f"{_NOTEBOOKS}/04_alignment/016_supervised_fine_tuning.ipynb",
    "017": f"{_NOTEBOOKS}/04_alignment/017_reward_model_and_rlhf.ipynb",
}

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

# 入力の例(モデルが英語で学習しているため英語)
EXAMPLE_PROMPTS = [
    "The history of the city began when",
    "Albert Einstein was born in",
    "The film was released in",
]
# 008 実験 A の退化(繰り返し)を見るための組: 同じ入力・同じシードで貪欲法と top-p を並べる。
DEGENERATION_EXAMPLE_PROMPT = "The history of the town of"


def _paragraphs(*blocks: str) -> str:
    """段落(空行で区切る)を連結する。段落の中は改行せず、隣接する文字列リテラルで書く。

    Markdown は段落の中の改行を空白として描画する。日本語の文の途中に空白が入らないように、
    ソースの上での折り返しは文字列リテラルの連結で行い、文字列としては改行しない。
    """
    return "\n\n".join(blocks)


def _items(*items: str) -> str:
    """箇条書き(各項目は改行しない)。"""
    return "\n".join(f"- {item}" for item in items)


INTRO_MARKDOWN = _paragraphs(
    "# Text Generation Playground",
    "[ai-theories](%(repo)s) でスクラッチ実装した小型 GPT"
    "(Generative Pre-trained Transformer、英語、約 %(params)sパラメータ)を動かして、"
    "**デコーディング戦略**(Decoding Strategy)・**KV キャッシュ**(Key-Value Cache)・"
    "**指示チューニング**(Instruction Tuning)の違いを手元で確かめるアプリです。",
    _items(
        "研究・教育目的のモデルで、品質の保証はありません。事実の正しさも保証しません。",
        "英語で学習したモデルです。入力は英語にしてください。",
        "ChatGPT のように会話できるモデルではありません。"
        "タブ 1 は書き出しの続きを書くだけで、タブ 2 は決まった形式の指示にだけ答えます。",
    ),
)

TAB_GENERATION_MARKDOWN = _paragraphs(
    "英語の書き出しを入力すると、小型 GPT が続きを 1 トークンずつ生成します。理論は "
    "[008 デコーディング戦略](%(n008)s) と "
    "[010 KV キャッシュと推論の計算量](%(n010)s) です。",
    "**デコードの方法**(008)",
    _items(
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

COMPARE_MARKDOWN = _paragraphs(
    "同じ書き出し・同じ乱数のシード・同じ設定で、デコードの方法を 2 つ並べて生成します"
    "(temperature・k・p・ビーム数などは上の設定を共通で使います)。"
    "貪欲法と top-p を並べると、繰り返しの違いが見えやすくなります。",
)

TIMING_NOTE = (
    "速さは、条件ごとに 1 回だけ測った参考値です(ウォームアップなし・繰り返しなし)。"
    "デバイスの同期を取ったうえで、GPU の割り当てにかかる時間と画面の更新の時間を除いて"
    "測っています。同じシードでもデバイスが違うと、サンプリングの結果は変わりえます。"
)

TAB_INSTRUCTION_MARKDOWN = _paragraphs(
    "017 が Hugging Face Hub の`sft-synthetic`ブランチに上げたモデルで、"
    "[016 SFT(指示チューニング)](%(n016)s) の合成課題に答えます。"
    "課題は、英単語の列に対する 4 種類の操作で、正解は規則で決まります。"
    "このモデルは [017 報酬モデルと RLHF](%(n017)s) が 016 の条件 1 のレシピ"
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

ABOUT_MARKDOWN = _paragraphs(
    "**ソースコードと理論**: [kojikojiprg/ai-theories](%(repo)s)。"
    "モデルとトークナイザは Hugging Face Hub の "
    "[ai-theories-small-gpt-en](https://huggingface.co/kojikojiprg/ai-theories-small-gpt-en)"
    "(ブランチ`main`・`gqa`・`sft-synthetic`)と "
    "[ai-theories-tokenizer-en](https://huggingface.co/kojikojiprg/ai-theories-tokenizer-en)"
    " です。",
)


def format_parameter_count(num_parameters: int) -> str:
    """パラメータ数を ``520 万`` のような日本語の単位で表す。"""
    return f"{num_parameters / 10_000:,.0f} 万"


def intro_markdown(main: LoadedModel) -> str:
    return INTRO_MARKDOWN % {
        "repo": REPOSITORY_URL,
        "params": format_parameter_count(main.num_parameters),
    }


def generation_tab_markdown() -> str:
    return TAB_GENERATION_MARKDOWN % {"n008": NOTEBOOK_URLS["008"], "n010": NOTEBOOK_URLS["010"]}


def instruction_tab_markdown() -> str:
    return TAB_INSTRUCTION_MARKDOWN % {"n016": NOTEBOOK_URLS["016"], "n017": NOTEBOOK_URLS["017"]}


def about_markdown() -> str:
    return ABOUT_MARKDOWN % {"repo": REPOSITORY_URL}


def describe_models(models: dict[str, LoadedModel], device: torch.device) -> str:
    """読み込んだモデルの一覧(Markdown の表)。3 ブランチを同時に載せたときの合計も示す。"""
    rows = [
        "| ブランチ | Query ヘッド数 | Key / Value ヘッド数 | パラメータ数 | 重みのメモリ |"
        " KV キャッシュ(1 トークンあたり) |",
        "|---|---|---|---|---|---|",
    ]
    total_bytes = 0
    for key, loaded in models.items():
        config = loaded.config
        key_value_heads = config.get("num_key_value_heads") or config["num_heads"]
        total_bytes += loaded.parameter_bytes
        rows.append(
            f"| `{key}` | {config['num_heads']} | {key_value_heads} |"
            f" {loaded.num_parameters:,} | {loaded.parameter_bytes / 2**20:.1f} MiB |"
            f" {loaded.key_value_cache_bytes_per_token:,} バイト |"
        )
    rows.append("")
    rows.append(
        f"3 ブランチを同時に`{device.type}`へ載せています"
        f"(重みの合計 {total_bytes / 2**20:.1f} MiB)。"
    )
    return "\n".join(rows)


def format_generation_stats(label: str, num_tokens: int, seconds: float, device: str) -> str:
    """生成したトークン数・時間・毎秒のトークン数(1 回の計測の参考値)。"""
    rate = num_tokens / seconds if seconds > 0 else float("inf")
    return (
        f"**{label}**: 生成したトークン数 {num_tokens} / 時間 {seconds:.3f} 秒 / "
        f"毎秒 {rate:.1f} トークン({device}・1 回の計測)"
    )


def format_beam_outcome(outcome: BeamOutcome, decode, prompt: str) -> tuple[str, str]:
    """ビームサーチの結果を ``(最良の候補の本文, 他の候補を含む補足)`` にする。"""
    best = outcome.candidates[0]
    text = prompt + decode(list(best.new_token_ids))
    lines = [f"ビームサーチの候補(長さ正規化した対数尤度の降順、{len(outcome.candidates)} 件):"]
    for rank, candidate in enumerate(outcome.candidates, start=1):
        tail = decode(list(candidate.new_token_ids)).replace("\n", " ")
        lines.append(f"{rank}. スコア {candidate.score:.3f}: …{tail[-60:]}")
    return text, "\n".join(lines)


def format_instruction_outcome(task: str, instruction: str, outcome: InstructionOutcome) -> str:
    """タブ 2 の結果(判定・プロンプト・応答・正解)を Markdown にする。"""
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
