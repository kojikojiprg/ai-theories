"""小型 GPT を包むテキスト埋め込みモデル(024)。

008 の小型 GPT(``GPTLanguageModel``)の **最終正規化層の後の隠れ状態** ``H``(形状 ``(B, S, d)``)
から、プーリング(pooling)で 1 本のベクトルを作り、L2 正規化したものを文の埋め込み(sentence
embedding)とする。射影層は置かない(埋め込みの次元は ``d_model`` のまま)。query と passage は **
同じ重み** の encoder で符号化する(dual encoder、重みの共有)。

**プーリング**(``pooling``):

- ``"mean"``: 全位置の平均 ``mean_t H_t``。

- ``"last"``: 終端位置の出力 ``H_{S-1}``。008 のトークナイザには終端の特殊トークンがないので、入
  力の **最後の位置**(固定長でパディングを使わないので、最後のトークンの位置)の出力を使う。因果
  マスクのもとでは、位置 ``t`` の出力は位置 ``t`` までのトークンしか見ていない。入力の全体を見て
  いるのは最後の位置だけである。

**注意マスク**(``attention``):

- ``"causal"``: 008 の事前学習と同じ因果マスク。

- ``"bidirectional"``: マスクを外し、全位置が全位置を参照する。008 の事前学習では見ていない入力
  になる(言語モデルとして学習された重みを、双方向の注意で使う)。LLM2Vec(BehnamGhader et al.,
  COLM 2024)は、この切り替えと追加の学習で decoder 型の言語モデルを埋め込みモデルに転用した。

**Matryoshka Representation Learning**(Kusupati et al., NeurIPS 2022)のための次元の切り詰め: プ
ーリングで得た **正規化前の** ベクトル ``z`` の先頭 ``m`` 次元を取り出して L2 正規化し直す
(``truncate_and_normalize()``)。``m = d`` のときは通常の埋め込みと一致する。

既存の ``GPTLanguageModel`` は変更しない。隠れ状態の計算(``compute_hidden_states()``)は
``GPTLanguageModel.forward()`` の語彙への射影の直前までと同じ手順で、``attention="causal"`` のと
き ``src/models/reward_model.py`` の ``compute_final_hidden_states()`` と bit 単位で一致する(024
で確かめる)。

記号 / Notation:
    B : バッチサイズ
    S : 系列長
    d : 埋め込みの次元(``d_model``)
    m : 切り詰めた後の次元

"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from src.layers.attention import create_causal_mask
from src.models.gpt import GPTLanguageModel

POOLING_MODES = ("mean", "last")
ATTENTION_MODES = ("causal", "bidirectional")


def compute_hidden_states(
    model: GPTLanguageModel, token_ids: Tensor, attention: str = "causal"
) -> Tensor:
    """最終正規化層の後の隠れ状態 ``(B, S, d)`` を返す(語彙への射影の直前)。

    トークン埋め込み -> ``DecoderBlock`` を ``num_layers`` 段 -> 最終正規化層。
    ``attention="causal"`` は ``GPTLanguageModel.forward()`` と同じ因果マスク、
    ``"bidirectional"`` はマスクなし。
    位置エンコーディング(RoPE)は、どちらも位置 0 から ``S - 1`` までの連番で与える。
    """
    if attention not in ATTENTION_MODES:
        raise ValueError(f"attention は {ATTENTION_MODES} のいずれか: {attention!r}")
    _, seq_len = token_ids.shape
    if seq_len > model.max_sequence_length:
        raise ValueError(
            f"系列長 ({seq_len}) が max_sequence_length ({model.max_sequence_length}) を超えている"
        )
    hidden = model.token_embedding(token_ids)
    mask = create_causal_mask(seq_len, device=token_ids.device) if attention == "causal" else None
    for i, block in enumerate(model.blocks):
        hidden, _, _ = block(
            hidden, None, tgt_mask=mask, positions=None, kv_cache=None, layer_idx=i
        )
    return model.final_norm(hidden)


def truncate_and_normalize(pooled: Tensor, dimension: int | None = None) -> Tensor:
    """正規化前のベクトルの先頭 ``dimension`` 次元を取り出して、L2 正規化する。

    ``dimension=None`` なら全次元。切り詰めた後に **再正規化する** (Kusupati et al.)。
    """
    if dimension is not None:
        pooled = pooled[..., :dimension]
    return functional.normalize(pooled, dim=-1)


class TextEmbeddingModel(nn.Module):
    """小型 GPT による dual encoder の encoder(query と passage で重みを共有する)。

    Args:
        backbone: 本体の ``GPTLanguageModel``(008 の重みを読み込んだもの、またはランダム初期化)。
            語彙への射影 ``lm_head`` は埋め込みの計算に使わない(重み共有によりトークン埋め込みと
            同じテンソルなので、トークン埋め込みとしては学習される)。
        pooling: ``"mean"`` または ``"last"``。
        attention: ``"causal"`` または ``"bidirectional"``。
    """

    def __init__(
        self, backbone: GPTLanguageModel, pooling: str = "mean", attention: str = "causal"
    ) -> None:
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling は {POOLING_MODES} のいずれか: {pooling!r}")
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention は {ATTENTION_MODES} のいずれか: {attention!r}")
        self.backbone = backbone
        self.pooling = pooling
        self.attention = attention

    @property
    def embedding_dimension(self) -> int:
        return int(self.backbone.token_embedding.embedding_dim)

    def pooled(self, token_ids: Tensor) -> Tensor:
        """プーリングした **正規化前の** ベクトル ``(B, d)``。"""
        hidden = compute_hidden_states(self.backbone, token_ids, self.attention)
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, -1, :]

    def pooled_at_positions(self, token_ids: Tensor, positions: Sequence[int]) -> Tensor:
        """指定した位置の出力 ``(B, len(positions), d)``。

        プーリング方式によらない。診断量に使う。
        """
        hidden = compute_hidden_states(self.backbone, token_ids, self.attention)
        return hidden[:, list(positions), :]

    def forward(self, token_ids: Tensor, dimension: int | None = None) -> Tensor:
        """L2 正規化した埋め込み ``(B, dimension または d)``。"""
        return truncate_and_normalize(self.pooled(token_ids), dimension)


@torch.no_grad()
def encode_tokens(
    model: TextEmbeddingModel,
    token_ids: Tensor,
    batch_size: int,
    dimensions: Sequence[int | None] = (None,),
    positions: Sequence[int] | None = None,
) -> dict:
    """固定長のトークン列の集合を FP32 で符号化する(評価用、勾配なし)。

    Args:
        model: 符号化するモデル(呼び出しの間 ``eval()`` にし、終わったら元の状態に戻す)。
        token_ids: 形状 ``(n, S)``(``model`` と同じデバイス、または CPU。バッチごとにデバイスへ
            移す)。
        batch_size: 1 回の順伝播の系列の数(結果には影響しない)。
        dimensions: 切り詰める次元の一覧(``None`` は全次元)。
        positions: 指定すると、プーリング方式の代わりに、その位置の出力それぞれを正規化した
            埋め込みも返す。

    Returns:
        ``{"by_dimension": {次元: 形状 (n, 次元) の埋め込み}, "by_position": {位置: 形状 (n, d)
        の埋め込み}}``。``by_dimension`` の次元は ``None`` を ``d`` に置き換えた整数をキーとする。
        ``by_position`` は ``positions`` を指定したときのみ。埋め込みはモデルと同じデバイスに置く。
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    full = model.embedding_dimension
    keys = [full if dimension is None else int(dimension) for dimension in dimensions]
    by_dimension: dict[int, list[Tensor]] = {k: [] for k in keys}
    by_position: dict[int, list[Tensor]] = {int(p): [] for p in positions or ()}
    try:
        for start in range(0, token_ids.size(0), batch_size):
            batch = token_ids[start : start + batch_size].to(device)
            pooled = model.pooled(batch).float()
            for key in keys:
                by_dimension[key].append(truncate_and_normalize(pooled, key))
            if positions is not None:
                outputs = model.pooled_at_positions(batch, positions).float()
                for index, position in enumerate(positions):
                    by_position[int(position)].append(truncate_and_normalize(outputs[:, index]))
    finally:
        model.train(was_training)
    result = {"by_dimension": {k: torch.cat(v) for k, v in by_dimension.items()}}
    if positions is not None:
        result["by_position"] = {k: torch.cat(v) for k, v in by_position.items()}
    return result
