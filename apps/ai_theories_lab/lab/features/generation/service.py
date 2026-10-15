"""文章の生成の処理(Gradio に依存しない)。

008 のノートブックにある ``generate_with_filter()`` と同じ手順(top-k・top-p のフィルタを
温度で割る前の logits にかけ、そのあと softmax からサンプリングする)を、次の 2 点を足して
実装する。

- **KV キャッシュ(010)の有無を選べる。** 使う場合の手順は
  ``GPTLanguageModel.generate(use_cache=True)`` と同じ(prefill の後、新しいトークンを
  1 個ずつ順伝播する)。使わない場合は ``use_cache=False`` と同じく、毎ステップ直近
  ``max_sequence_length`` トークンの文脈全体を再計算する。
- **トークンを 1 個生成するたびに途中経過を返す**(画面の逐次表示のため)。

top-k・top-p のフィルタ(``top_k_filter``・``top_p_filter``)とビームサーチ
(``beam_search``)は ``src/generation/decoding.py``(008)のものをそのまま使う。
ビームサーチは ``src`` の実装が KV キャッシュと逐次の返却に対応していないため、
結果を最後にまとめて返す。

計算時間の測り方は ``lab.core.timing`` を参照。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from lab.core.artifacts import LoadedModel
from lab.core.timing import ComputeClock
from src.data.tokenizer import BPEIDTokenizer
from src.generation import beam_search, top_k_filter, top_p_filter
from src.generation.cache import KeyValueCache
from src.models.gpt import GPTLanguageModel

METHOD_GREEDY = "greedy"
METHOD_TEMPERATURE = "temperature"
METHOD_TOP_K = "top_k"
METHOD_TOP_P = "top_p"
METHOD_BEAM = "beam"
DECODING_METHODS = (METHOD_GREEDY, METHOD_TEMPERATURE, METHOD_TOP_K, METHOD_TOP_P, METHOD_BEAM)
SAMPLING_METHODS = (METHOD_TEMPERATURE, METHOD_TOP_K, METHOD_TOP_P)

MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class DecodingSettings:
    """デコードの方法とその設定。方法に関係しない設定は使われない。

    Attributes:
        method: ``DECODING_METHODS`` のいずれか。
        max_new_tokens: 生成するトークン数(終端記号を持たないモデルなので、常にこの数だけ生成する)。
        temperature: 温度(temperature・top-k・top-p のサンプリングで使う)。
        top_k: top-k の k。
        top_p: top-p の p。
        beam_size: ビーム数(ビームサーチのみ)。
        length_penalty: 長さペナルティの強さ(ビームサーチのみ)。
        seed: サンプリングの乱数シード(サンプリングのみ)。
        use_cache: KV キャッシュを使うか(ビームサーチでは使われない)。
    """

    method: str
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.9
    beam_size: int = 4
    length_penalty: float = 1.0
    seed: int = 0
    use_cache: bool = True

    def validate(self, prompt_length: int, max_sequence_length: int) -> None:
        """設定と入力の長さの組み合わせが生成できるものかを確かめる。

        Raises:
            ValueError: 生成できない場合。メッセージは画面にそのまま表示できる日本語。
        """
        if self.method not in DECODING_METHODS:
            raise ValueError(f"未知のデコードの方法: {self.method!r}")
        if prompt_length < 1:
            raise ValueError("書き出しが空です。英語の文を入力してください。")
        if prompt_length > max_sequence_length:
            raise ValueError(
                f"書き出しが長すぎます({prompt_length} トークン)。"
                f"{max_sequence_length} トークン以下にしてください。"
            )
        if self.max_new_tokens < 1:
            raise ValueError("最大トークン数は 1 以上にしてください。")
        if self.method in SAMPLING_METHODS:
            if not self.temperature > 0.0:
                raise ValueError("温度は 0 より大きくしてください。")
            if not 0 <= self.seed <= MAX_SEED:
                raise ValueError(f"乱数のシードは 0 以上 {MAX_SEED} 以下にしてください。")
        if self.method == METHOD_TOP_K and self.top_k < 1:
            raise ValueError("k は 1 以上にしてください。")
        if self.method == METHOD_TOP_P and not 0.0 < self.top_p <= 1.0:
            raise ValueError("p は 0 より大きく 1 以下にしてください。")
        if self.method == METHOD_BEAM:
            if self.beam_size < 1:
                raise ValueError("ビーム数は 1 以上にしてください。")
            if self.length_penalty < 0.0:
                raise ValueError("長さペナルティは 0 以上にしてください。")
        elif self.use_cache and prompt_length + self.max_new_tokens > max_sequence_length:
            raise ValueError(
                "KV キャッシュを使う場合、書き出しのトークン数と最大トークン数の和が "
                f"{max_sequence_length} 以下である必要があります"
                f"(今は {prompt_length} + {self.max_new_tokens})。"
                "最大トークン数を減らすか、KV キャッシュを使わない設定にしてください"
                "(使わない場合は、直近の文脈だけを使って先へ進みます)。"
            )


@dataclass(frozen=True)
class GenerationProgress:
    """生成の途中経過。

    Attributes:
        new_token_ids: ここまでに生成したトークン(書き出しは含まない)。
        compute_seconds: ここまでの計算時間(途中経過を返している間は含まない)。
        finished: 最後のトークンまで生成したか。
    """

    new_token_ids: tuple[int, ...]
    compute_seconds: float
    finished: bool


@dataclass(frozen=True)
class BeamCandidate:
    """ビームサーチの候補 1 つ。``score`` は長さ正規化済みの対数尤度。"""

    new_token_ids: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class BeamOutcome:
    """ビームサーチの結果(``candidates`` はスコアの降順)と、計算時間。"""

    candidates: tuple[BeamCandidate, ...]
    compute_seconds: float


def select_next_token(
    logits: Tensor, settings: DecodingSettings, generator: torch.Generator | None
) -> Tensor:
    """形状 ``(1, V)`` の logits から次のトークン(形状 ``(1, 1)``)を選ぶ。

    008 の ``generate_with_filter()`` と同じく、top-k・top-p のフィルタは温度で割る前の
    logits にかける。
    貪欲法は argmax(温度・シードは使わない)。
    """
    if settings.method == METHOD_GREEDY:
        return logits.argmax(dim=-1, keepdim=True)
    if settings.method == METHOD_TOP_K:
        logits = top_k_filter(logits, settings.top_k)
    elif settings.method == METHOD_TOP_P:
        logits = top_p_filter(logits, settings.top_p)
    probs = torch.softmax(logits / settings.temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def stream_generation(
    model: GPTLanguageModel,
    prompt_ids: list[int],
    settings: DecodingSettings,
    device: torch.device,
) -> Iterator[GenerationProgress]:
    """貪欲法・サンプリングで 1 トークンずつ生成し、そのたびに途中経過を返す。

    ビームサーチは対象外(``run_beam_search`` を使う)。最後の途中経過は ``finished=True`` で、
    ``compute_seconds`` はその生成全体の計算時間になる。
    """
    if settings.method == METHOD_BEAM:
        raise ValueError("ビームサーチは run_beam_search() を使う")
    settings.validate(len(prompt_ids), model.max_sequence_length)

    generator = None
    if settings.method in SAMPLING_METHODS:
        generator = torch.Generator(device=device)
        generator.manual_seed(settings.seed)

    was_training = model.training
    model.eval()
    try:
        clock = ComputeClock(device)
        clock.start()
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        cache = KeyValueCache(model.num_layers) if settings.use_cache else None
        if cache is not None:
            # prefill: 書き出し全体を 1 回の順伝播でキャッシュに書き込む。
            positions = torch.arange(tokens.size(1), device=device)
            logits = model(tokens, positions=positions, kv_cache=cache)
        else:
            logits = model(tokens[:, -model.max_sequence_length :])

        new_ids: list[int] = []
        for step in range(settings.max_new_tokens):
            next_token = select_next_token(logits[:, -1, :], settings, generator)
            new_ids.append(int(next_token.item()))
            finished = step == settings.max_new_tokens - 1
            elapsed = clock.stop()
            yield GenerationProgress(tuple(new_ids), elapsed, finished)
            if finished:
                return
            clock.start()
            if cache is not None:
                # decode: 新しいトークン 1 個だけを順伝播する。
                position = torch.tensor([cache.length], device=device)
                logits = model(next_token, positions=position, kv_cache=cache)
            else:
                tokens = torch.cat([tokens, next_token], dim=1)
                logits = model(tokens[:, -model.max_sequence_length :])
    finally:
        model.train(was_training)


def run_beam_search(
    model: GPTLanguageModel,
    prompt_ids: list[int],
    settings: DecodingSettings,
    device: torch.device,
) -> BeamOutcome:
    """ビームサーチ(``src/generation/decoding.py``)で生成する。KV キャッシュは使われない。"""
    if settings.method != METHOD_BEAM:
        raise ValueError("run_beam_search() はビームサーチ用")
    settings.validate(len(prompt_ids), model.max_sequence_length)

    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    clock = ComputeClock(device)
    clock.start()
    results = beam_search(
        model,
        prompt,
        beam_size=settings.beam_size,
        max_new_tokens=settings.max_new_tokens,
        length_penalty=settings.length_penalty,
    )
    elapsed = clock.stop()
    candidates = tuple(
        BeamCandidate(tuple(sequence[0, len(prompt_ids) :].tolist()), float(score))
        for sequence, score in results
    )
    return BeamOutcome(candidates, elapsed)


@dataclass(frozen=True)
class GenerationEvent:
    """画面に渡す生成の途中経過。トークンの ID は文字列に復号してある。

    Attributes:
        text: 書き出しに続けて表示する本文。
        num_new_tokens: ここまでに生成したトークン数。
        compute_seconds: 計算時間(``finished`` のときだけ値を持つ)。
        finished: 生成が終わったか。
        waiting_for_beam: ビームサーチを計算中で、結果がまだない(最初の 1 回だけ真)。
        beam_candidates: ビームサーチの候補(スコアの降順)の ``(スコア, 続きの文字列)``。
            最後だけ値を持つ。
    """

    text: str
    num_new_tokens: int
    compute_seconds: float | None
    finished: bool
    waiting_for_beam: bool = False
    beam_candidates: tuple[tuple[float, str], ...] = ()


def generate_events(
    loaded: LoadedModel,
    tokenizer: BPEIDTokenizer,
    device: torch.device,
    prompt: str,
    settings: DecodingSettings,
    decode: Callable[[list[int]], str] | None = None,
) -> Iterator[GenerationEvent]:
    """1 つのデコードの方法で生成し、途中経過(``GenerationEvent``)を返す。

    ``@spaces.GPU`` を付けた関数の中から呼ぶ(GPU の割り当てが済んだ後の計算だけを測る)。

    Raises:
        ValueError: 設定・入力が生成できないもの(メッセージは画面に表示できる日本語)。
    """
    decode = decode or tokenizer.decode
    prompt_ids = tokenizer.encode(prompt)
    settings.validate(len(prompt_ids), loaded.max_sequence_length)

    if settings.method == METHOD_BEAM:
        yield GenerationEvent(prompt, 0, None, False, waiting_for_beam=True)
        outcome = run_beam_search(loaded.model, prompt_ids, settings, device)
        candidates = tuple((c.score, decode(list(c.new_token_ids))) for c in outcome.candidates)
        best = outcome.candidates[0]
        yield GenerationEvent(
            prompt + decode(list(best.new_token_ids)),
            len(best.new_token_ids),
            outcome.compute_seconds,
            True,
            beam_candidates=candidates,
        )
        return

    for progress in stream_generation(loaded.model, prompt_ids, settings, device):
        yield GenerationEvent(
            prompt + decode(list(progress.new_token_ids)),
            len(progress.new_token_ids),
            progress.compute_seconds if progress.finished else None,
            progress.finished,
        )
