"""LLaVA 型の Vision-Language 連結(021)。

Liu et al., "Visual Instruction Tuning", NeurIPS 2023(LLaVA)の構成に従い、凍結した画像 encoder の
パッチ特徴を projection 層で言語モデルの埋め込み空間に写像し、テキストのトークン埋め込みの列に
差し込んで、言語モデルに入力する。

.. math::

    Z_v = g(X_v), \\qquad H_v = W \\cdot Z_v

- ``g``: 画像 encoder。020 の CLIP の画像 encoder(019 の ``VisionTransformer``)の、**最終層の
  一つ前の層** の出力(最終正規化の前の残差の流れ)のうち、[CLS] トークンを除いたパッチトークンの
  格子全体を使う(原論文と同じ選び方。``vision_patch_features()``)。
- ``W``: projection 層(``VisualProjection``)。原論文の LLaVA は線形写像、LLaVA-1.5(Liu et al.,
  CVPR 2024)は GELU をはさむ 2 層の多層パーセプトロン(Multi-Layer Perceptron)を使う。021 の実験では
  線形写像のみを使う(2 層の版は理論の対応のために実装だけを置く)。
- 言語モデル: 008 の ``GPTLanguageModel``。トークン ID の代わりに埋め込みの列から順伝播する
  ``language_model_hidden_states()`` を置き、既存の ``forward()`` は変更しない(埋め込みに
  ``token_embedding(token_ids)`` を渡すと ``forward()`` と bit 単位で一致することを 021 で
  確かめる)。

視覚トークンは、テキストの系列の中の固定の位置(``visual_start`` から長さ N_v の枠)に差し込む
(``src/data/visual_instruction.py`` の符号化で、枠の位置は全事例で共通)。視覚トークンの取り方は
``pooling`` で切り替える。

- ``"patch"``: パッチトークンの格子全体(N_v = N = HW / P^2)。
- ``"mean"``: 同じ層のパッチトークンの平均プール(N_v = 1)。

記号 / Notation:
    N   : パッチの数(HW / P^2)
    d_v : 画像 encoder の隠れ次元(019 の D)
    d   : 言語モデルの隠れ次元(d_model)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from src.generation.cache import KeyValueCache
from src.layers.activation import gelu_exact
from src.layers.attention import create_causal_mask
from src.models.gpt import GPTLanguageModel
from src.models.vit import VisionTransformer

POOLING_TYPES = ("patch", "mean")
PROJECTION_TYPES = ("linear", "mlp")


@torch.no_grad()
def vision_patch_features(vision: VisionTransformer, images: Tensor, num_blocks: int) -> Tensor:
    """先頭から ``num_blocks`` 個のブロックを通した後のパッチトークン(形状 ``(B, N, d_v)``)。

    ``num_blocks = L - 1`` が最終層の一つ前の層(原論文の選び方)。最終正規化は掛けない
    (Hugging Face の ``hidden_states[-2]`` と同じく、残差の流れそのもの)。[CLS] トークン(位置 0)は
    除く。
    画像 encoder は凍結して使うので、勾配を計算しない。

    Args:
        vision: 019 の ``VisionTransformer``。
        images: 形状 ``(B, C, H, W)`` の float(020 と同じ正規化済みのもの)。
        num_blocks: 通すブロックの数(1 以上 L 以下)。
    """
    if not 1 <= num_blocks <= len(vision.blocks):
        raise ValueError(f"num_blocks は 1 以上 {len(vision.blocks)} 以下: {num_blocks}")
    x = vision.embed(images)
    for block in vision.blocks[:num_blocks]:
        x, _ = block(x)
    return x[:, 1:]


class VisualProjection(nn.Module):
    """視覚特徴を言語モデルの埋め込み空間に写す projection 層。

    Args:
        in_features: 画像 encoder の隠れ次元 d_v。
        out_features: 言語モデルの隠れ次元 d。
        kind: ``"linear"``(LLaVA: H_v = W Z_v + b)または ``"mlp"``(LLaVA-1.5: 線形 -> GELU ->
            線形)。
    """

    def __init__(self, in_features: int, out_features: int, kind: str = "linear") -> None:
        super().__init__()
        if kind not in PROJECTION_TYPES:
            raise ValueError(f"kind は {PROJECTION_TYPES} のいずれか: {kind!r}")
        self.kind = kind
        if kind == "linear":
            self.layers = nn.ModuleList([nn.Linear(in_features, out_features)])
        else:
            self.layers = nn.ModuleList(
                [nn.Linear(in_features, out_features), nn.Linear(out_features, out_features)]
            )

    def forward(self, features: Tensor) -> Tensor:
        x = self.layers[0](features)
        if self.kind == "mlp":
            x = self.layers[1](gelu_exact(x))
        return x


def language_model_hidden_states(
    model: GPTLanguageModel,
    embeddings: Tensor,
    positions: Tensor | None = None,
    kv_cache: KeyValueCache | None = None,
) -> Tensor:
    """埋め込みの列から、最終正規化層の後の隠れ状態 ``(B, S, d)`` を返す(``lm_head`` の直前)。

    ``GPTLanguageModel.forward()`` の計算のうち、トークン埋め込みの後から最終正規化層までを
    同じ手順で行う(因果マスクは ``kv_cache`` が空のときだけ作る。``forward()`` と同じ規則)。
    """
    _, seq_len, _ = embeddings.shape
    cache_is_empty = kv_cache is None or kv_cache.length == 0
    if cache_is_empty and seq_len > model.max_sequence_length:
        raise ValueError(
            f"系列長 ({seq_len}) が max_sequence_length ({model.max_sequence_length}) を超えている"
        )
    causal_mask = create_causal_mask(seq_len, device=embeddings.device) if cache_is_empty else None
    hidden = embeddings
    for i, block in enumerate(model.blocks):
        hidden, _, _ = block(
            hidden, None, tgt_mask=causal_mask, positions=positions, kv_cache=kv_cache, layer_idx=i
        )
    return model.final_norm(hidden)


class LlavaStyleModel(nn.Module):
    """projection 層 + 言語モデル(画像 encoder は持たず、事前に計算したパッチ特徴を受け取る)。

    画像 encoder は凍結するので、パッチ特徴(``vision_patch_features()`` の出力)は画像ごとに
    一度だけ計算して再利用できる。本クラスは、その特徴から視覚トークンを作り、テキストの
    トークン埋め込みの枠に差し込んで言語モデルに通す。

    Args:
        language_model: 008 の ``GPTLanguageModel``(LoRA を掛けたものでもよい)。
        projection: ``VisualProjection``。
        visual_start: 視覚トークンの枠の開始位置(全事例で共通)。
        pooling: ``"patch"`` または ``"mean"``。
    """

    def __init__(
        self,
        language_model: GPTLanguageModel,
        projection: VisualProjection,
        visual_start: int,
        pooling: str = "patch",
    ) -> None:
        super().__init__()
        if pooling not in POOLING_TYPES:
            raise ValueError(f"pooling は {POOLING_TYPES} のいずれか: {pooling!r}")
        self.language_model = language_model
        self.projection = projection
        self.visual_start = visual_start
        self.pooling = pooling

    def visual_tokens(self, patch_features: Tensor) -> Tensor:
        """パッチ特徴 ``(B, N, d_v)`` から視覚トークン ``(B, N_v, d)`` を作る。"""
        if self.pooling == "mean":
            patch_features = patch_features.mean(dim=1, keepdim=True)
        return self.projection(patch_features)

    def input_embeddings(self, token_ids: Tensor, patch_features: Tensor) -> Tensor:
        """トークン埋め込みの列の枠 ``[visual_start, visual_start + N_v)`` を視覚トークンに
        置き換える。

        枠の位置のトークン ID は使われない(置き換えられる)。
        """
        visual = self.visual_tokens(patch_features)
        end = self.visual_start + visual.size(1)
        text = self.language_model.token_embedding(token_ids)
        return torch.cat([text[:, : self.visual_start], visual, text[:, end:]], dim=1)

    def hidden_states(self, token_ids: Tensor, patch_features: Tensor) -> Tensor:
        """最終正規化層の後の隠れ状態 ``(B, S, d)``。"""
        return language_model_hidden_states(
            self.language_model, self.input_embeddings(token_ids, patch_features)
        )

    def forward(self, token_ids: Tensor, patch_features: Tensor) -> Tensor:
        """全位置の logits ``(B, S, V)``(検証用。学習は応答の位置だけを語彙に射影する)。"""
        return self.language_model.lm_head(self.hidden_states(token_ids, patch_features))

    def response_logits(
        self, token_ids: Tensor, patch_features: Tensor, response_target_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """応答部分のトークンを予測する位置だけの logits ``(K, V)`` と、その正解 ``(K,)``。

        ``response_target_mask``(形状 ``(B, S - 1)``)が True の位置 ``j`` の隠れ状態だけを語彙に
        射影する(018 と同じく、語彙への射影の計算量を応答のトークン数 K に比例させる)。
        """
        hidden = self.hidden_states(token_ids, patch_features)[:, :-1]
        selected = hidden[response_target_mask]
        targets = token_ids[:, 1:][response_target_mask]
        return self.language_model.lm_head(selected), targets


@torch.no_grad()
def greedy_generate_with_visual_tokens(
    model: LlavaStyleModel,
    prompts: Sequence[Sequence[int]],
    patch_features: Tensor,
    decode: Callable[[list[int]], str],
    stop_text: str,
    max_new_tokens: int,
    batch_size: int = 256,
) -> list[list[int]]:
    """視覚トークンを差し込んだ指示部分から、終端記号の文字列が現れるまで貪欲法で生成する。

    016 の ``greedy_generate_until_stop()`` と同じ規則(指示部分の長さごとにバッチにしてパディングを
    使わない、KV キャッシュで新しいトークン 1 個ずつ順伝播する、終端記号の文字列が現れた事例から
    止める)。
    prefill だけを埋め込みの列から行い(視覚トークンを含む)、decode の各ステップは生成したトークンの
    ID から ``GPTLanguageModel.forward()`` で行う。

    Args:
        model: ``LlavaStyleModel``。
        prompts: 事例ごとの指示部分のトークン ID の列(視覚トークンの枠を含む)。
        patch_features: 形状 ``(len(prompts), N, d_v)``。事例ごとのパッチ特徴。
        decode: トークン ID の列を文字列に復号する関数。
        stop_text: 停止の条件とする文字列。
        max_new_tokens: 1 事例あたりの生成トークン数の上限。
        batch_size: 同じ長さの指示部分をまとめる 1 回の順伝播の事例数の上限。

    Returns:
        事例ごとの生成トークン列(指示部分を含まない)。
    """
    if len(prompts) != patch_features.size(0):
        raise ValueError("prompts と patch_features の事例数が一致しない")
    lm = model.language_model
    longest = max(len(p) for p in prompts)
    if longest + max_new_tokens > lm.max_sequence_length:
        raise ValueError(
            f"指示部分の長さ({longest})+ 上限トークン数({max_new_tokens})が "
            f"max_sequence_length({lm.max_sequence_length})を超える"
        )
    device = patch_features.device
    was_training = model.training
    model.eval()
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_length[len(prompt)].append(index)
    outputs: list[list[int] | None] = [None] * len(prompts)
    try:
        for length in sorted(by_length):
            members = by_length[length]
            for start in range(0, len(members), batch_size):
                chunk = members[start : start + batch_size]
                context = torch.tensor(
                    [list(prompts[i]) for i in chunk], dtype=torch.long, device=device
                )
                index = torch.tensor(chunk, device=device)
                kv_cache = KeyValueCache(lm.num_layers)
                embeddings = model.input_embeddings(context, patch_features[index])
                positions = torch.arange(length, device=device)
                hidden = language_model_hidden_states(lm, embeddings, positions, kv_cache)
                next_tokens = lm.lm_head(hidden[:, -1]).argmax(dim=-1)
                generated: list[list[int]] = [[] for _ in chunk]
                finished = [False] * len(chunk)
                for step in range(max_new_tokens):
                    values = next_tokens.tolist()
                    for row in range(len(chunk)):
                        if finished[row]:
                            continue
                        generated[row].append(values[row])
                        if stop_text in decode(generated[row]):
                            finished[row] = True
                    if all(finished) or step == max_new_tokens - 1:
                        break
                    step_position = torch.tensor([kv_cache.length], device=device)
                    logits = lm(next_tokens[:, None], positions=step_position, kv_cache=kv_cache)
                    next_tokens = logits[:, -1].argmax(dim=-1)
                for i, tokens in zip(chunk, generated, strict=True):
                    outputs[i] = tokens
    finally:
        model.train(was_training)
    return [list(o) for o in outputs]
