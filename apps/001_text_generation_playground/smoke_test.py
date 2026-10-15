"""スモークテスト(動作確認)。``uv run python smoke_test.py [--hub]``

学習や重い計算は行わない。確かめるのは、コードの経路と、理論が要求する不変条件である。

**既定(代替のアーティファクト)**: Hugging Face Hub を使わない。構造だけが本物と同じで重みが乱数の
モデル(``main``・``gqa``・``sft-synthetic``)と、小さな語彙のトークナイザを、一時ディレクトリに
Hugging Face のキャッシュと同じ配置で作り、オフラインモード(``HF_HUB_OFFLINE=1``)で
アプリを読み込む。
アプリのコードは、Hub から取得する経路(``hf_hub_download``)をそのまま通る。重みが乱数なので、
**生成される文章や応答の中身には意味がなく、モデルの性能は何も確かめられない。**

**``--hub``**: Hub の本物のモデル・トークナイザを使う(ネットワークが要る)。既定の確認に加えて、
本物の重みでしか確かめられない項目(チェックポイントの SHA-256、合成課題の正解率、退化の例が
実際に退化するか)を確かめる。

どちらのモードでも、``app.py`` を ``gradio_client`` から呼ぶ確認を含む。
``--serve`` を付けると、テストはせずにアプリだけを起動する(画面の確認用)。
"""

import argparse
import hashlib
import os
import resource
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--hub", action="store_true", help="Hub の本物のアーティファクトを使う")
parser.add_argument(
    "--serve",
    action="store_true",
    help="テストはせず、アプリだけを起動する(画面の確認用。既定では代替のアーティファクトを使う)",
)
ARGS = parser.parse_args()

# Hugging Face の環境変数は、huggingface_hub を import する前に決める必要がある。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
STAND_IN_DIRECTORY: Path | None = None
if not ARGS.hub:
    STAND_IN_DIRECTORY = Path(tempfile.mkdtemp(prefix="playground_stand_in_"))
    os.environ["HF_HUB_CACHE"] = str(STAND_IN_DIRECTORY / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from playground import text  # noqa: E402
from playground.artifacts import (  # noqa: E402
    MODEL_REPO_ID,
    TOKENIZER_REPO_ID,
    build_model,
)
from playground.generation import (  # noqa: E402
    METHOD_BEAM,
    METHOD_GREEDY,
    METHOD_TEMPERATURE,
    METHOD_TOP_K,
    METHOD_TOP_P,
    DecodingSettings,
    run_beam_search,
    select_next_token,
    stream_generation,
)
from playground.instruction import (  # noqa: E402
    MAX_NEW_TOKENS,
    MAX_WORDS,
    MIN_WORDS,
    TEMPLATE_KINDS,
    InstructionOutcome,
    instruction_choices,
    parse_words,
    random_words,
    respond,
    template_kind_of,
)
from src.data.instruction import (  # noqa: E402
    SEEN_INSTRUCTIONS,
    TASK_NAMES,
    UNSEEN_INSTRUCTIONS,
    WORD_VOCABULARY,
    apply_task,
    format_prompt,
    sample_word_sequences,
)
from src.data.tokenizer import (  # noqa: E402
    BPEIDTokenizer,
    learn_bpe,
    save_bpe_id_tokenizer_json,
)
from src.generation import beam_search, top_k_filter, top_p_filter  # noqa: E402
from src.models.gpt import GPTLanguageModel  # noqa: E402

# 010 の PROD_CONFIG(本物の main の config.json の内容。010 のセル出力)
PRODUCTION_CONFIG = {
    "vocabulary_size": 8192,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 1024,
    "swiglu_d_ff": 683,
    "sequence_length": 256,
    "dropout": 0.0,
    "tie_embeddings": True,
    "positional_encoding": "rope",
    "normalization": "rmsnorm",
    "feed_forward": "swiglu",
    "norm_first": True,
}
GQA_KEY_VALUE_HEADS = 4  # 010 の実験 D の gqa_mean_pool
# アップロードしたチェックポイントの SHA-256(017 の 6.14 節、010 の実験 D のセル出力)
EXPECTED_SHA256 = {
    "sft-synthetic": "5814fbc024a27adc1624ee9a718fbbd5a7e28f5d20a0df40c8d4633126a91ab5",
    "gqa": "f23e68cd4a2865f01612f20af10482fe7146deb84d16df18a78489e43037b9ca",
}

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    """条件を確かめ、結果を 1 行で印字する。失敗しても続けて、最後にまとめて報告する。"""
    print(("OK:   " if condition else "FAIL: ") + message)
    if not condition:
        FAILURES.append(message)


# ---------------------------------------------------------------------------
# 代替のアーティファクト(既定のモードのみ)
# ---------------------------------------------------------------------------
def _write_repository(
    cache_dir: Path, repo_id: str, revisions: dict[str, dict[str, bytes]]
) -> None:
    """Hugging Face のキャッシュと同じ配置(refs/・snapshots/)でリポジトリを書く。"""
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    for revision, files in revisions.items():
        commit = hashlib.sha1(f"{repo_id}@{revision}".encode()).hexdigest()
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / revision).write_text(commit, encoding="utf-8")
        for name, content in files.items():
            (snapshot / name).write_bytes(content)


def build_stand_in_artifacts(directory: Path) -> None:
    """構造だけが本物と同じ、乱数の重みのモデル 3 つと、小さな語彙のトークナイザを作る。"""
    import json

    corpus = " ".join(
        [
            "The history of the city began when the river town grew into a market and a port.",
            "Albert Einstein was born in a small town, and the film was released in the spring.",
            "### Instruction: ### Response: ### End Words:",
            *(
                t
                for table in (SEEN_INSTRUCTIONS, UNSEEN_INSTRUCTIONS)
                for ts in table.values()
                for t in ts
            ),
            *WORD_VOCABULARY,
            *WORD_VOCABULARY,
        ]
    )
    bpe = learn_bpe(corpus, 700, byte_level=True, max_chunk_bytes=64)
    symbol_to_id = {symbol: i for i, symbol in enumerate(sorted(bpe.vocab))}
    tokenizer = BPEIDTokenizer(bpe, symbol_to_id)
    tokenizer_path = directory / "tokenizer.json"
    save_bpe_id_tokenizer_json(tokenizer, tokenizer_path)

    config = {**PRODUCTION_CONFIG, "vocabulary_size": tokenizer.vocab_size}
    revisions: dict[str, dict[str, bytes]] = {}
    for seed, (revision, key_value_heads) in enumerate(
        (("main", None), ("gqa", GQA_KEY_VALUE_HEADS), ("sft-synthetic", None))
    ):
        revision_config = dict(config)
        if key_value_heads is not None:
            revision_config["num_key_value_heads"] = key_value_heads
        torch.manual_seed(seed)
        model = build_model(revision_config)
        checkpoint_path = directory / f"{revision}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        revisions[revision] = {
            "config.json": json.dumps(revision_config).encode("utf-8"),
            "model_state.pt": checkpoint_path.read_bytes(),
        }
    cache_dir = directory / "hub"
    _write_repository(cache_dir, MODEL_REPO_ID, revisions)
    _write_repository(
        cache_dir, TOKENIZER_REPO_ID, {"main": {"tokenizer.json": tokenizer_path.read_bytes()}}
    )


if STAND_IN_DIRECTORY is not None:
    build_stand_in_artifacts(STAND_IN_DIRECTORY)
    print(f"代替のアーティファクトを作った(Hub は使わない): {STAND_IN_DIRECTORY}")
else:
    print("Hub の本物のアーティファクトを使う")

import app  # noqa: E402  アプリの読み込み(モデルとトークナイザの読み込みを含む)

RESOURCES = app.RESOURCES
TOKENIZER = RESOURCES.tokenizer
DEVICE = RESOURCES.device
MAIN = RESOURCES.models["main"]


# ---------------------------------------------------------------------------
# 参照実装(008 のノートブックの generate_with_filter() と同じ手順)
# ---------------------------------------------------------------------------
@torch.no_grad()
def reference_generate_with_filter(
    model: GPTLanguageModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=prompt_ids.device)
        generator.manual_seed(seed)
    token_ids = prompt_ids
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.max_sequence_length :]
        next_logits = model(context)[:, -1, :]
        if top_k is not None:
            next_logits = top_k_filter(next_logits, top_k)
        if top_p is not None:
            next_logits = top_p_filter(next_logits, top_p)
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids


def generate_ids(model_key: str, prompt: str, settings: DecodingSettings) -> list[int]:
    """アプリと同じ関数で生成し、生成したトークン ID(書き出しを除く)を返す。"""
    loaded = RESOURCES.models[model_key]
    prompt_ids = TOKENIZER.encode(prompt)
    if settings.method == METHOD_BEAM:
        return list(
            run_beam_search(loaded.model, prompt_ids, settings, DEVICE).candidates[0].new_token_ids
        )
    progress = list(stream_generation(loaded.model, prompt_ids, settings, DEVICE))
    return list(progress[-1].new_token_ids)


PROMPTS = [*text.EXAMPLE_PROMPTS, text.DEGENERATION_EXAMPLE_PROMPT]


# ---------------------------------------------------------------------------
# 1. 読み込み: 3 ブランチを同時にメモリへ載せる
# ---------------------------------------------------------------------------
def test_loading() -> None:
    print("\n== 1. 読み込み ==")
    check(set(RESOURCES.models) == {"main", "gqa", "sft-synthetic"}, "3 ブランチを読み込んだ")
    total_bytes = sum(m.parameter_bytes for m in RESOURCES.models.values())
    print(text.describe_models(RESOURCES.models, DEVICE))
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 if sys.platform != "darwin" else 1024 * 1024
    )
    print(
        f"重みの合計 {total_bytes / 2**20:.1f} MiB / "
        f"このプロセスのメモリの最大値 {peak_rss_mib:.0f} MiB"
    )
    if DEVICE.type == "cuda":
        print(f"CUDA に載っている量: {torch.cuda.memory_allocated() / 2**20:.1f} MiB")
    check(
        all(
            next(m.model.parameters()).device.type == DEVICE.type for m in RESOURCES.models.values()
        ),
        f"全モデルが {DEVICE.type} 上にある",
    )
    check(all(not m.model.training for m in RESOURCES.models.values()), "全モデルが評価モード")
    gqa, main = RESOURCES.models["gqa"], RESOURCES.models["main"]
    check(
        gqa.model.num_key_value_heads < main.model.num_key_value_heads
        and gqa.key_value_cache_bytes_per_token < main.key_value_cache_bytes_per_token,
        "gqa は Key / Value ヘッド数と KV キャッシュのバイト数が main より小さい",
    )
    check(
        main.model.token_embedding.weight is main.model.lm_head.weight,
        "重み共有(トークン埋め込みと出力層)が保たれている",
    )


# ---------------------------------------------------------------------------
# 2. 生成: KV キャッシュの有無で貪欲法の結果が一致する(010 の不変条件)ほか
# ---------------------------------------------------------------------------
def test_generation() -> None:
    print("\n== 2. 生成 ==")
    for model_key in ("main", "gqa"):
        loaded = RESOURCES.models[model_key]
        for prompt in PROMPTS:
            prompt_ids = TOKENIZER.encode(prompt)
            tensor = torch.tensor([prompt_ids], device=DEVICE)
            with_cache = generate_ids(
                model_key, prompt, DecodingSettings(METHOD_GREEDY, 48, use_cache=True)
            )
            without_cache = generate_ids(
                model_key, prompt, DecodingSettings(METHOD_GREEDY, 48, use_cache=False)
            )
            reference_cache = loaded.model.generate(tensor, 48, temperature=0.0, use_cache=True)
            reference_plain = loaded.model.generate(tensor, 48, temperature=0.0, use_cache=False)
            beam_one = generate_ids(
                model_key,
                prompt,
                DecodingSettings(METHOD_BEAM, 48, beam_size=1, length_penalty=1.0),
            )
            check(
                with_cache == without_cache,
                f"[{model_key}] 貪欲法: KV キャッシュの有無で一致({prompt!r})",
            )
            check(
                with_cache == reference_cache[0, len(prompt_ids) :].tolist()
                and without_cache == reference_plain[0, len(prompt_ids) :].tolist(),
                f"[{model_key}] 貪欲法: GPTLanguageModel.generate() と一致({prompt!r})",
            )
            check(
                with_cache == beam_one,
                f"[{model_key}] 貪欲法: ビーム数 1 のビームサーチと一致({prompt!r})",
            )

    # サンプリング: 008 の generate_with_filter() と、同じシードで同じトークンになる
    # (KV キャッシュなし)
    prompt = text.EXAMPLE_PROMPTS[0]
    prompt_ids = TOKENIZER.encode(prompt)
    tensor = torch.tensor([prompt_ids], device=DEVICE)
    cases = (
        (METHOD_TEMPERATURE, {"temperature": 0.8}, {"temperature": 0.8}),
        (METHOD_TOP_K, {"temperature": 1.0, "top_k": 40}, {"temperature": 1.0, "top_k": 40}),
        (METHOD_TOP_P, {"temperature": 1.0, "top_p": 0.9}, {"temperature": 1.0, "top_p": 0.9}),
    )
    for method, settings_kwargs, reference_kwargs in cases:
        settings = DecodingSettings(method, 40, seed=7, use_cache=False, **settings_kwargs)
        mine = generate_ids("main", prompt, settings)
        reference = reference_generate_with_filter(
            MAIN.model, tensor, 40, seed=7, **reference_kwargs
        )[0, len(prompt_ids) :].tolist()
        check(mine == reference, f"{method}: 008 の generate_with_filter() と同じシードで一致")
        cached = generate_ids(
            "main", prompt, DecodingSettings(method, 40, seed=7, use_cache=True, **settings_kwargs)
        )
        agreement = float(np.mean([a == b for a, b in zip(mine, cached, strict=True)]))
        print(f"      (参考) {method}: KV キャッシュの有無でトークンが一致した割合 {agreement:.3f}")

    # シードの再現性
    settings = DecodingSettings(METHOD_TOP_P, 32, temperature=1.5, top_p=0.95, seed=3)
    first = generate_ids("main", prompt, settings)
    second = generate_ids("main", prompt, settings)
    other = generate_ids(
        "main", prompt, DecodingSettings(METHOD_TOP_P, 32, temperature=1.5, top_p=0.95, seed=4)
    )
    check(first == second, "同じシードで同じ生成結果")
    check(first != other, "別のシードで別の生成結果")

    # フィルタの確認: top-k=1 は貪欲法と同じ、top-p の極小は貪欲法と同じ
    greedy = generate_ids("main", prompt, DecodingSettings(METHOD_GREEDY, 32))
    check(
        generate_ids("main", prompt, DecodingSettings(METHOD_TOP_K, 32, top_k=1, seed=9)) == greedy,
        "top-k の k=1 は貪欲法と同じ",
    )
    check(
        generate_ids("main", prompt, DecodingSettings(METHOD_TOP_P, 32, top_p=1e-6, seed=9))
        == greedy,
        "top-p の p が極小なら貪欲法と同じ",
    )

    # logits から次のトークンを選ぶ関数の単体確認(貪欲法はシードによらない)
    logits = torch.tensor([[0.1, 3.0, 0.2, 2.9]], device=DEVICE)
    chosen = {
        int(select_next_token(logits, DecodingSettings(METHOD_GREEDY, 1), None).item())
        for _ in range(5)
    }
    check(chosen == {1}, "select_next_token: 貪欲法は argmax")

    # ビームサーチ: 候補はスコアの降順で、ビーム数だけ返る
    outcome = run_beam_search(
        MAIN.model, TOKENIZER.encode(prompt), DecodingSettings(METHOD_BEAM, 24, beam_size=3), DEVICE
    )
    scores = [c.score for c in outcome.candidates]
    check(
        len(scores) == 3 and scores == sorted(scores, reverse=True),
        "ビームサーチ: 候補が 3 件でスコアの降順",
    )
    reference_beam = beam_search(
        MAIN.model, torch.tensor([TOKENIZER.encode(prompt)], device=DEVICE), 3, 24, 1.0
    )
    check(
        list(outcome.candidates[0].new_token_ids)
        == reference_beam[0][0][0, len(TOKENIZER.encode(prompt)) :].tolist(),
        "ビームサーチ: src の beam_search() と同じ最良の候補",
    )


# ---------------------------------------------------------------------------
# 3. 速さの表示
# ---------------------------------------------------------------------------
def test_timing() -> None:
    print("\n== 3. 速さの表示 ==")
    prompt_ids = TOKENIZER.encode(text.EXAMPLE_PROMPTS[0])
    settings = DecodingSettings(METHOD_GREEDY, 24)
    sleep_per_step = 0.02
    started = time.perf_counter()
    last = None
    count = 0
    for progress in stream_generation(MAIN.model, prompt_ids, settings, DEVICE):
        time.sleep(sleep_per_step)  # 画面の更新に時間がかかる場合を模す
        last = progress
        count += 1
    wall = time.perf_counter() - started
    check(
        count == 24 and last.finished and len(last.new_token_ids) == 24,
        "途中経過が 1 トークンずつ 24 回返る",
    )
    check(
        0 < last.compute_seconds <= wall - 0.9 * sleep_per_step * count,
        f"計算時間({last.compute_seconds:.3f} 秒)に、途中経過を返している間の待ち"
        f"({sleep_per_step * count:.2f} 秒)が含まれない",
    )
    line = text.format_generation_stats("生成結果", 24, last.compute_seconds, DEVICE.type)
    print("      " + line)
    check(
        "生成したトークン数 24" in line and "時間" in line and "毎秒" in line,
        "生成したトークン数・時間・毎秒のトークン数を表示する",
    )


# ---------------------------------------------------------------------------
# 4. 入力の検証
# ---------------------------------------------------------------------------
def test_validation() -> None:
    print("\n== 4. 入力の検証 ==")
    limit = MAIN.max_sequence_length

    def raises(settings: DecodingSettings, prompt_length: int) -> bool:
        try:
            settings.validate(prompt_length, limit)
        except ValueError:
            return True
        return False

    check(raises(DecodingSettings(METHOD_GREEDY, 8), 0), "空の書き出しは拒否する")
    check(raises(DecodingSettings(METHOD_GREEDY, 8), limit + 1), "長すぎる書き出しは拒否する")
    check(
        raises(DecodingSettings(METHOD_GREEDY, 200, use_cache=True), 100),
        "KV キャッシュあり: 書き出し + 最大トークン数が長すぎれば拒否する",
    )
    check(
        not raises(DecodingSettings(METHOD_GREEDY, 200, use_cache=False), 100),
        "KV キャッシュなし: 直近の文脈で先へ進めるので許す",
    )
    check(
        not raises(DecodingSettings(METHOD_BEAM, 200, use_cache=True), 100),
        "ビームサーチは KV キャッシュの長さの制約を受けない",
    )
    check(raises(DecodingSettings(METHOD_TOP_P, 8, top_p=0.0), 4), "top-p の p=0 は拒否する")
    check(raises(DecodingSettings(METHOD_TOP_K, 8, top_k=0), 4), "top-k の k=0 は拒否する")
    check(
        raises(DecodingSettings(METHOD_TEMPERATURE, 8, temperature=0.0), 4),
        "temperature=0 は拒否する",
    )
    check(raises(DecodingSettings(METHOD_TOP_P, 8, seed=-1), 4), "負のシードは拒否する")
    check(raises(DecodingSettings("unknown", 8), 4), "未知のデコードの方法は拒否する")


# ---------------------------------------------------------------------------
# 5. タブ 2(指示への応答)
# ---------------------------------------------------------------------------
def test_instruction() -> None:
    print("\n== 5. 指示への応答 ==")
    for task in TASK_NAMES:
        check(
            len(instruction_choices(task, "seen")) == 4,
            f"{task}: 見たことのあるテンプレートは 4 個",
        )
        check(len(instruction_choices(task, "unseen")) == 2, f"{task}: 未見のテンプレートは 2 個")
        check(
            all(template_kind_of(task, t) == "seen" for t in instruction_choices(task, "seen"))
            and all(
                template_kind_of(task, t) == "unseen" for t in instruction_choices(task, "unseen")
            ),
            f"{task}: 指示文から種類が逆引きできる",
        )
    check(set(TEMPLATE_KINDS) == {"seen", "unseen"}, "指示文の種類は seen / unseen")

    def raises_value_error(function, *args) -> bool:
        try:
            function(*args)
        except ValueError:
            return True
        return False

    check(
        raises_value_error(template_kind_of, "reverse", "Do something else."),
        "一覧にない指示文は拒否する",
    )
    check(raises_value_error(parse_words, "one two"), "単語が少なすぎれば拒否する")
    check(
        raises_value_error(parse_words, " ".join(["cat"] * (MAX_WORDS + 1))),
        "単語が多すぎれば拒否する",
    )
    check(raises_value_error(parse_words, "cat dog 3d"), "英字以外を含めば拒否する")
    parsed = parse_words("Cat, ghost  house")
    check(
        parsed.words == ("cat", "ghost", "house") and not parsed.warnings,
        "大文字・カンマ・連続する空白を正規化する",
    )
    check(bool(parse_words("cat dog zzzz").warnings), "学習の語彙の外の単語には注意を出す")
    check(bool(parse_words("cat cat dog").warnings), "重複した単語には注意を出す")

    words = random_words(0).split()
    check(
        random_words(0) == random_words(0)
        and MIN_WORDS <= len(words) <= MAX_WORDS
        and set(words) <= set(WORD_VOCABULARY)
        and len(set(words)) == len(words),
        "ランダムな単語はシードで決まり、語彙の中の重複しない 3〜8 語",
    )

    sft = RESOURCES.models["sft-synthetic"]
    for task in TASK_NAMES:
        instruction = instruction_choices(task, "seen")[0]
        outcome = respond(sft, TOKENIZER, DEVICE, task, instruction, tuple(words))
        check(
            outcome.prompt == format_prompt(instruction, words)
            and outcome.answer == apply_task(task, words)
            and outcome.num_new_tokens <= MAX_NEW_TOKENS,
            f"{task}: プロンプトは短い水準の形式で、正解は規則どおり、"
            f"生成は {MAX_NEW_TOKENS} トークン以内",
        )

    # 判定の表示(モデルの出力に依らず、3 通りの判定が表示できること)
    base = {"prompt": "p", "answer": ("a", "b"), "num_new_tokens": 4}
    for format_ok, exact, marker in (
        (True, True, "✅"),
        (True, False, "⚠️"),
        (False, False, "❌"),
    ):
        outcome = InstructionOutcome(
            response_text=" a b\n### End", format_ok=format_ok, exact_match=exact, **base
        )
        report = text.format_instruction_outcome(
            "reverse", instruction_choices("reverse", "unseen")[0], outcome
        )
        check(
            marker in report and "未見のテンプレート" in report and "`a b`" in report,
            f"判定の表示 {marker}",
        )


# ---------------------------------------------------------------------------
# 6. app.py を gradio_client から呼ぶ
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_app_via_client() -> None:
    print("\n== 6. gradio_client から app.py を呼ぶ ==")
    from gradio_client import Client

    _, url, _ = app.demo.launch(
        prevent_thread_lock=True, server_name="127.0.0.1", server_port=_free_port(), quiet=True
    )
    try:
        client = Client(url, verbose=False)

        def call_generate(prompt: str, model_key: str, use_cache: bool, method: str, **overrides):
            arguments = {
                "temperature": 1.0,
                "top_k": 40,
                "top_p": 0.9,
                "beam_size": 4,
                "length_penalty": 1.0,
                "max_new_tokens": 32,
                "seed": 0,
            } | overrides
            job = client.submit(
                prompt,
                model_key,
                use_cache,
                method,
                arguments["temperature"],
                arguments["top_k"],
                arguments["top_p"],
                arguments["beam_size"],
                arguments["length_penalty"],
                arguments["max_new_tokens"],
                arguments["seed"],
                api_name="/generate_text",
            )
            return job.result(), len(job.outputs())

        prompt = text.EXAMPLE_PROMPTS[0]
        for model_key in ("main", "gqa"):
            (cached, stats, _), updates = call_generate(prompt, model_key, True, METHOD_GREEDY)
            (plain, _, _), _ = call_generate(prompt, model_key, False, METHOD_GREEDY)
            check(
                cached.startswith(prompt) and len(cached) > len(prompt),
                f"[{model_key}] 生成した続きが書き出しの後に表示される",
            )
            check(
                cached == plain,
                f"[{model_key}] 貪欲法: KV キャッシュの有無で、画面に返る生成結果が一致",
            )
            check("毎秒" in stats and "トークン" in stats, f"[{model_key}] 速さの表示が返る")
            check(updates > 1, f"[{model_key}] 逐次表示: 途中経過が複数回返る({updates} 回)")
        for method in (METHOD_TEMPERATURE, METHOD_TOP_K, METHOD_TOP_P):
            (body, stats, _), _ = call_generate(prompt, "main", True, method, seed=5)
            check(body.startswith(prompt) and "毎秒" in stats, f"{method}: 生成できる")
        (body, stats, details), _ = call_generate(prompt, "main", True, METHOD_BEAM, beam_size=3)
        check(
            "ビームサーチの候補" in details and "毎秒" in stats,
            "ビームサーチ: 候補の一覧と速さが返る",
        )

        compared = client.predict(
            prompt,
            "main",
            True,
            METHOD_GREEDY,
            METHOD_TOP_P,
            1.0,
            40,
            0.9,
            4,
            1.0,
            32,
            0,
            api_name="/compare_methods",
        )
        text_a, stats_a, _, text_b, stats_b, _ = compared
        check(
            text_a.startswith(prompt) and text_b.startswith(prompt) and text_a != text_b,
            "比較: 2 つの方法の結果が並ぶ",
        )
        check("毎秒" in stats_a and "毎秒" in stats_b, "比較: それぞれの速さが表示される")
        greedy_alone, _ = call_generate(prompt, "main", True, METHOD_GREEDY)
        check(greedy_alone[0] == text_a, "比較の方法 A(貪欲法)は、単独で生成した結果と一致")

        instruction = instruction_choices("reverse", "seen")[0]
        shown_prompt, report = client.predict(
            "reverse", instruction, "cat dog house tree", api_name="/answer_instruction"
        )
        check(
            shown_prompt == format_prompt(instruction, ("cat", "dog", "house", "tree"))
            and "正解" in report
            and "tree house dog cat" in report,
            "指示への応答: プロンプトと正解が表示される",
        )
    finally:
        app.demo.close()


# ---------------------------------------------------------------------------
# 7. 本物のアーティファクトでのみ確かめる項目(--hub)
# ---------------------------------------------------------------------------
def repeated_ngram_rate(token_ids: list[int], n: int = 4) -> float:
    """n-gram のうち、同じ n-gram が既に出ているものの割合(008 の重複率と同じ考え方)。"""
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def test_hub_only() -> None:
    from huggingface_hub import hf_hub_download

    print("\n== 7. 本物のアーティファクトでの確認(--hub) ==")
    for key, loaded in RESOURCES.models.items():
        expected = dict(PRODUCTION_CONFIG)
        if key == "gqa":
            expected["num_key_value_heads"] = GQA_KEY_VALUE_HEADS
        check(
            all(loaded.config.get(k) == v for k, v in expected.items()),
            f"[{key}] config.json が 010 の PROD_CONFIG(と gqa の Key / Value ヘッド数)と一致",
        )
    for key, expected_digest in EXPECTED_SHA256.items():
        path = hf_hub_download(MODEL_REPO_ID, "model_state.pt", revision=key)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        check(
            digest == expected_digest,
            f"[{key}] model_state.pt の SHA-256 がノートブックの記録と一致",
        )
    check(
        TOKENIZER.vocab_size == PRODUCTION_CONFIG["vocabulary_size"],
        "トークナイザの語彙サイズが 8192",
    )

    # 合成課題: 見たことのあるテンプレートで正解率が高く、未見のテンプレートでは低い(016 の傾向)
    rng = np.random.default_rng(2026)
    inputs = sample_word_sequences(rng, 50, WORD_VOCABULARY, MIN_WORDS, MAX_WORDS)
    sft = RESOURCES.models["sft-synthetic"]
    rates = {}
    for kind in ("seen", "unseen"):
        exact = 0
        total = 0
        for index, words in enumerate(inputs):
            for task in TASK_NAMES:
                choices = instruction_choices(task, kind)
                instruction = choices[index % len(choices)]
                exact += respond(sft, TOKENIZER, DEVICE, task, instruction, words).exact_match
                total += 1
        rates[kind] = exact / total
        print(f"      {kind}: 完全一致率 {rates[kind]:.3f}({total} 事例)")
    check(
        rates["seen"] >= 0.5,
        "見たことのあるテンプレートの完全一致率が 0.5 以上(016 の平均は 0.804)",
    )
    check(
        rates["unseen"] < 0.25,
        "未見のテンプレートの完全一致率が基準値 0.25 を下回る(016 は 0.022〜0.078)",
    )

    # 退化の例: 貪欲法の繰り返しが top-p より多い
    prompt = text.DEGENERATION_EXAMPLE_PROMPT
    greedy = generate_ids("main", prompt, DecodingSettings(METHOD_GREEDY, 64))
    top_p_rates = [
        repeated_ngram_rate(
            generate_ids("main", prompt, DecodingSettings(METHOD_TOP_P, 64, top_p=0.9, seed=seed))
        )
        for seed in range(5)
    ]
    greedy_rate = repeated_ngram_rate(greedy)
    print(
        f"      {prompt!r}: 4-gram 重複率 貪欲法 {greedy_rate:.3f} / "
        f"top-p(0.9, 5 シード)平均 {float(np.mean(top_p_rates)):.3f}"
    )
    check(greedy_rate > float(np.mean(top_p_rates)), "退化の例: 貪欲法の重複率が top-p より高い")


def main() -> int:
    if ARGS.serve:
        app.demo.launch()
        return 0
    test_loading()
    test_generation()
    test_timing()
    test_validation()
    test_instruction()
    test_app_via_client()
    if ARGS.hub:
        test_hub_only()
    else:
        print(
            "\n注意: 代替のアーティファクト(乱数の重み)での確認なので、モデルの性能に関わる項目"
            "(SHA-256、合成課題の正解率、退化の例)は確かめていない。--hub で確かめられる。"
        )
    if STAND_IN_DIRECTORY is not None:
        shutil.rmtree(STAND_IN_DIRECTORY, ignore_errors=True)
    print()
    if FAILURES:
        print(f"失敗 {len(FAILURES)} 件:")
        for message in FAILURES:
            print(f"  - {message}")
        return 1
    print("すべての確認が通った")
    return 0


if __name__ == "__main__":
    sys.exit(main())
