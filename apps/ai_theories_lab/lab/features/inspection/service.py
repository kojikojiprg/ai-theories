"""モデルの中を見る機能の処理(Gradio に依存しない)。

対象は 008 の小型 GPT(``main``)で、英語の文を入れると、次の 3 つを調べる。

1. **トークン分割(005)**: 共有トークナイザ(バイトレベル BPE)が文をどのトークンに分けるか。
   1 つの語(事前分割で得た 1 単位)が、学習順のマージ規則で結合されていく過程を、段階ごとに記録する。
2. **注意の重み(001)**: 各層・各ヘッドの注意の重み ``(層, ヘッド, Query の位置, Key の位置)``。
3. **トークンごとの確率(008)**: 入力の各位置で、モデルが次のトークンに付けた確率・
   分布のエントロピー・上位の候補。

**注意の重みの取り出し方。** ``src/`` は変更しない。``MultiHeadAttention.forward`` は
``(出力, 注意の重み)`` を返し、``DecoderBlock`` が受け取る(``GPTLanguageModel.forward`` は
捨てる)ので、各層の ``self_attn`` に **forward hook** を付けて、返り値の注意の重みを
横取りする。これはモデルの中で実際に Value との積に使われた値そのもの
(``scaled_dot_product_attention`` が返す ``attn_weights``)で、再計算した値ではない。
hook は値を読むだけで、順伝播の結果は変わらない(試験で bit 単位の一致を確かめる)。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from lab.core.artifacts import LoadedModel
from src.data.tokenizer import (
    BPEIDTokenizer,
    BPETokenizer,
    pretokenize,
    try_decode_byte_level_symbol,
)
from src.models.gpt import GPTLanguageModel

MAX_INSPECTION_TOKENS = 128
"""調べられる最大のトークン数(注意の重みの図が読めるように、モデルの最大長 256 より小さくする)。"""

TOP_CANDIDATES = 10
"""位置ごとに記録する、次のトークンの上位の候補の数。"""


# ---------------------------------------------------------------------------
# 1. トークン分割(005)
# ---------------------------------------------------------------------------
def readable_symbol(symbol: str) -> str:
    """バイトレベル BPE のシンボルを、人が読める文字列にする。

    空白は ``␣``、改行は ``⏎``、タブは ``⇥`` で表す。マルチバイト文字の一部のバイトだけが結合された
    シンボル(UTF-8 として復号できないもの)は、そのままの文字を ``‹ ›`` で囲む。
    """
    decoded = try_decode_byte_level_symbol(symbol)
    if decoded is None:
        return f"‹{symbol}›"
    return decoded.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥")


def merge_symbol_pair(
    symbols: Sequence[str], pair: tuple[str, str], merged: str
) -> tuple[str, ...]:
    """シンボル列の中の ``pair`` に一致する隣接ペアを、左から順に(重複せず)すべて
    ``merged`` にする。

    ``BPETokenizer.encode_chunk`` が使う結合(``src/data/tokenizer.py``)と同じ規則。
    """
    result: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


@dataclass(frozen=True)
class MergeStep:
    """BPE の符号化の 1 段階。

    Attributes:
        rank: 結合したペアのマージ規則の順位(学習した順、0 が最優先)。
        pair: 結合した 2 つのシンボル。
        symbols_after: 結合した後のシンボル列。
    """

    rank: int
    pair: tuple[str, str]
    symbols_after: tuple[str, ...]


@dataclass(frozen=True)
class ChunkTrace:
    """事前分割で得た 1 単位(チャンク。英語では、直前の空白を含む 1 語)の、結合の過程。

    Attributes:
        chunk: チャンクの文字列。
        initial_symbols: 結合の前の、1 バイト 1 シンボルの列。
        steps: 結合の段階(最優先のマージ規則から順に)。
        final_symbols: 結合が終わった後のシンボル列(= このチャンクのトークン)。
        token_ids: ``final_symbols`` のトークン ID。
    """

    chunk: str
    initial_symbols: tuple[str, ...]
    steps: tuple[MergeStep, ...]
    final_symbols: tuple[str, ...]
    token_ids: tuple[int, ...]


def trace_chunk(tokenizer: BPEIDTokenizer, chunk: str) -> ChunkTrace:
    """1 チャンクの BPE の符号化を、結合の段階ごとに記録する。

    ``BPETokenizer.encode_chunk`` と同じ手順(隣り合うペアのうち、最も早く学習されたマージ規則の
    ペアを、すべての出現で結合する。結合できるペアがなくなるまで繰り返す)を、途中経過を残しながら
    行う。結果は ``encode_chunk`` の結果と一致する(一致しなければ ``RuntimeError``)。
    """
    bpe: BPETokenizer = tokenizer.bpe_tokenizer
    # バイト列を 1 バイト 1 シンボルへ戻す手順は、src の非公開のメソッドを使う(再実装しない。
    # src は pyproject.toml の rev で固定しているので、この呼び出しが黙って変わることはない)。
    initial = tuple(bpe._initial_symbols(chunk))
    symbols = initial
    steps: list[MergeStep] = []
    ranks = bpe.merge_ranks
    while len(symbols) > 1:
        ranked = [
            (ranks[pair], pair) for pair in zip(symbols, symbols[1:], strict=False) if pair in ranks
        ]
        if not ranked:
            break
        rank, best_pair = min(ranked)
        symbols = merge_symbol_pair(symbols, best_pair, best_pair[0] + best_pair[1])
        steps.append(MergeStep(rank, best_pair, symbols))

    if list(symbols) != bpe.encode_chunk(chunk):
        raise RuntimeError(f"結合の過程の記録が encode_chunk() の結果と一致しない: {chunk!r}")
    return ChunkTrace(
        chunk=chunk,
        initial_symbols=initial,
        steps=tuple(steps),
        final_symbols=symbols,
        token_ids=tuple(tokenizer.symbol_to_id[s] for s in symbols),
    )


@dataclass(frozen=True)
class TokenPiece:
    """入力の 1 トークン。

    Attributes:
        index: 入力の中での位置(0 始まり)。
        token_id: トークン ID。
        symbol: バイトレベルのシンボル。
        text: 人が読める表示(``readable_symbol``)。
        chunk_index: このトークンが属するチャンクの番号。
    """

    index: int
    token_id: int
    symbol: str
    text: str
    chunk_index: int


@dataclass(frozen=True)
class Tokenization:
    """入力の文のトークン分割。"""

    text: str
    chunks: tuple[ChunkTrace, ...]
    pieces: tuple[TokenPiece, ...]

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(piece.token_id for piece in self.pieces)

    @property
    def token_texts(self) -> tuple[str, ...]:
        return tuple(piece.text for piece in self.pieces)


def tokenize(tokenizer: BPEIDTokenizer, text: str) -> Tokenization:
    """文をトークンに分け、チャンクごとの結合の過程を記録する。

    Raises:
        ValueError: 入力が空、またはトークン数が ``MAX_INSPECTION_TOKENS`` を超える場合
            (メッセージは画面に表示できる日本語)。
    """
    if not text.strip():
        raise ValueError("文が空です。英語の文を入力してください。")
    bpe = tokenizer.bpe_tokenizer
    chunks = tuple(
        trace_chunk(tokenizer, chunk)
        for chunk in pretokenize(text, bpe.chunk_split_mode, bpe.max_chunk_bytes)
    )
    pieces: list[TokenPiece] = []
    for chunk_index, trace in enumerate(chunks):
        for symbol, token_id in zip(trace.final_symbols, trace.token_ids, strict=True):
            pieces.append(
                TokenPiece(len(pieces), token_id, symbol, readable_symbol(symbol), chunk_index)
            )
    if len(pieces) > MAX_INSPECTION_TOKENS:
        raise ValueError(
            f"文が長すぎます({len(pieces)} トークン)。"
            f"{MAX_INSPECTION_TOKENS} トークン以下にしてください。"
        )
    return Tokenization(text, chunks, tuple(pieces))


# ---------------------------------------------------------------------------
# 2・3. 注意の重みとトークンごとの確率(001・008)
# ---------------------------------------------------------------------------
@contextmanager
def capture_attention(model: GPTLanguageModel) -> Iterator[dict[int, Tensor]]:
    """各層の ``self_attn`` が返す注意の重みを、forward hook で集める(読むだけで、値は変えない)。

    ``MultiHeadAttention.forward`` は ``(出力, 注意の重み)`` を返す。注意の重みの形状は
    ``(B, h, S_q, S_k)``。``with`` の中で ``model(...)`` を実行すると、層の番号から注意の重みへの
    辞書が埋まる(テンソルは計算グラフから切り離してある)。
    """
    captured: dict[int, Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(module, args, output) -> None:
            captured[layer_index] = output[1].detach()

        return hook

    for layer_index, block in enumerate(model.blocks):
        handles.append(block.self_attn.register_forward_hook(make_hook(layer_index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@dataclass(frozen=True)
class ModelAnalysis:
    """モデルに文を通した結果(すべて CPU 上の numpy 配列)。

    記号: ``S`` はトークン数、``L`` は層数、``h`` は Query のヘッド数、``K`` は上位の候補の数。
    位置 ``i`` は「入力の ``i`` 番目のトークンまでを見た時点」で、そこでの出力は、
    ``i + 1`` 番目のトークン(次のトークン)の予測である。

    Attributes:
        token_ids: 入力のトークン ID(長さ ``S``)。
        attention: 注意の重み ``(L, h, S, S)``。``attention[l, k, i, j]`` は、層 ``l``・
            ヘッド ``k`` で、位置 ``i``(Query)が位置 ``j``(Key)に向けた重み。
            因果マスクにより ``j > i`` は 0。
        entropy: 位置ごとの、次のトークンの分布のエントロピー(nats、長さ ``S``)。
        next_probability: 位置ごとの、実際の次のトークンに付けた確率(長さ ``S``)。
            最後の位置には実際の次のトークンがないので NaN。
        next_rank: 実際の次のトークンの順位(1 が最上位、長さ ``S``)。最後の位置は 0。
        top_ids: 位置ごとの上位の候補のトークン ID ``(S, K)``(確率の降順)。
        top_probabilities: ``top_ids`` の確率 ``(S, K)``。
        vocabulary_size: 語彙サイズ(エントロピーの上限 ``ln V`` の計算に使う)。
    """

    token_ids: tuple[int, ...]
    attention: np.ndarray
    entropy: np.ndarray
    next_probability: np.ndarray
    next_rank: np.ndarray
    top_ids: np.ndarray
    top_probabilities: np.ndarray
    vocabulary_size: int

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_layers(self) -> int:
        return int(self.attention.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.attention.shape[1])

    @property
    def max_entropy(self) -> float:
        """一様な分布のエントロピー ``ln V``(nats)。"""
        return math.log(self.vocabulary_size)


@torch.no_grad()
def analyze(
    loaded: LoadedModel,
    device: torch.device,
    token_ids: Sequence[int],
    top_candidates: int = TOP_CANDIDATES,
) -> ModelAnalysis:
    """トークン列をモデルに 1 回通し、注意の重みと、位置ごとの次のトークンの分布を調べる。

    ``@gpu`` を付けた関数の中から呼ぶ。結果は CPU 上の numpy 配列で返す(ZeroGPU のプロセス間の
    受け渡しに、GPU 上のテンソルを残さないため)。

    Raises:
        ValueError: トークン列が空、または ``MAX_INSPECTION_TOKENS`` を超える場合。
    """
    num_tokens = len(token_ids)
    if not 1 <= num_tokens <= MAX_INSPECTION_TOKENS:
        raise ValueError(
            f"トークン数は 1 以上 {MAX_INSPECTION_TOKENS} 以下にしてください(今は {num_tokens})。"
        )
    model = loaded.model
    tokens = torch.tensor([list(token_ids)], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    try:
        with capture_attention(model) as captured:
            logits = model(tokens)[0]  # (S, V)
    finally:
        model.train(was_training)

    attention = torch.stack([captured[i][0] for i in range(model.num_layers)])  # (L, h, S, S)
    probabilities = torch.softmax(logits.float(), dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, min(top_candidates, logits.size(-1)))

    next_probability = torch.full((num_tokens,), float("nan"), device=device)
    next_rank = torch.zeros(num_tokens, dtype=torch.long, device=device)
    if num_tokens > 1:
        actual_next = tokens[0, 1:]  # 位置 i の次のトークン(i = 0 .. S - 2)
        positions = torch.arange(num_tokens - 1, device=device)
        actual_probability = probabilities[positions, actual_next]
        next_probability[:-1] = actual_probability
        # 順位 = 実際のトークンより確率の高いトークンの数 + 1
        next_rank[:-1] = (probabilities[:-1] > actual_probability[:, None]).sum(dim=-1) + 1

    return ModelAnalysis(
        token_ids=tuple(int(t) for t in token_ids),
        attention=attention.float().cpu().numpy(),
        entropy=entropy.cpu().numpy(),
        next_probability=next_probability.cpu().numpy(),
        next_rank=next_rank.cpu().numpy().astype(np.int32),
        top_ids=top_ids.cpu().numpy().astype(np.int32),
        top_probabilities=top_probabilities.cpu().numpy(),
        vocabulary_size=int(logits.size(-1)),
    )


def select_attention(analysis: ModelAnalysis, layer: int, head: int | None) -> np.ndarray:
    """層 ``layer`` のヘッド ``head`` の注意の重み ``(S, S)`` を返す。

    ``head`` が ``None`` なら、全ヘッドの平均。

    Raises:
        ValueError: 層・ヘッドの番号が範囲外の場合。
    """
    if not 0 <= layer < analysis.num_layers:
        raise ValueError(f"層は 0 以上 {analysis.num_layers - 1} 以下です: {layer}")
    if head is None:
        return analysis.attention[layer].mean(axis=0)
    if not 0 <= head < analysis.num_heads:
        raise ValueError(f"ヘッドは 0 以上 {analysis.num_heads - 1} 以下です: {head}")
    return analysis.attention[layer, head]


@dataclass(frozen=True)
class Candidate:
    """次のトークンの候補 1 つ。"""

    rank: int
    token_id: int
    text: str
    probability: float


@dataclass(frozen=True)
class PositionView:
    """1 つの位置(入力の ``position`` 番目のトークンまでを見た時点)の、次のトークンの予測。

    Attributes:
        position: 位置(0 始まり)。
        context_text: この位置のトークンの表示。
        entropy: 分布のエントロピー(nats)。
        candidates: 上位の候補(確率の降順)。
        actual_text: 実際の次のトークンの表示(最後の位置は ``None``)。
        actual_probability: 実際の次のトークンに付けた確率(最後の位置は ``None``)。
        actual_rank: 実際の次のトークンの順位(最後の位置は ``None``)。
    """

    position: int
    context_text: str
    entropy: float
    candidates: tuple[Candidate, ...]
    actual_text: str | None
    actual_probability: float | None
    actual_rank: int | None


def describe_position(
    analysis: ModelAnalysis, tokenizer: BPEIDTokenizer, position: int
) -> PositionView:
    """位置 ``position`` の、次のトークンの上位の候補と、実際の次のトークンの確率・順位をまとめる。

    Raises:
        ValueError: 位置が範囲外の場合。
    """
    if not 0 <= position < analysis.num_tokens:
        raise ValueError(f"位置は 0 以上 {analysis.num_tokens - 1} 以下です: {position}")
    candidates = tuple(
        Candidate(
            rank + 1,
            int(token_id),
            readable_symbol(tokenizer.id_to_symbol[int(token_id)]),
            float(probability),
        )
        for rank, (token_id, probability) in enumerate(
            zip(analysis.top_ids[position], analysis.top_probabilities[position], strict=True)
        )
    )
    is_last = position == analysis.num_tokens - 1
    actual_id = None if is_last else analysis.token_ids[position + 1]
    return PositionView(
        position=position,
        context_text=readable_symbol(tokenizer.id_to_symbol[analysis.token_ids[position]]),
        entropy=float(analysis.entropy[position]),
        candidates=candidates,
        actual_text=None
        if actual_id is None
        else readable_symbol(tokenizer.id_to_symbol[actual_id]),
        actual_probability=None if is_last else float(analysis.next_probability[position]),
        actual_rank=None if is_last else int(analysis.next_rank[position]),
    )
