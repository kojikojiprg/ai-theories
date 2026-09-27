"""報酬モデル(Reward Model)と、言語モデルの最終正規化後の隠れ状態の取り出し(017)。

報酬モデルは、SFT 済みの言語モデルの本体(トークン埋め込み・``DecoderBlock`` の積層・最終正規化層)
に、スカラーを出す線形のヘッドを付けたものである(Ouyang et al., NeurIPS 2022 の InstructGPT は、
SFT モデルの語彙への射影を取り除いてスカラーを出す層に置き換えた)。

.. math::

    r_\\phi(x, y) = w^\\top h_{\\mathrm{last}}(x, y) + b

``h_last`` は、プロンプト x と応答 y を連結した系列の **最終トークンの位置** の、最終正規化層の
後の隠れ状態(次元 d_model)である。因果マスクにより、最終トークンの位置は系列全体を参照できる。
phi は本体とヘッドのすべてのパラメータ(017 では全パラメータを学習する)。

``compute_final_hidden_states()`` は ``GPTLanguageModel.forward()`` の計算のうち、語彙への射影
(``lm_head``)の直前までを同じ手順で行う。既存の ``forward()`` は変更しない(017 で、
``model.lm_head(compute_final_hidden_states(model, x))`` が ``model(x)`` と bit 単位で一致する
ことを確かめる)。PPO(``src/training/ppo.py``)の価値ヘッドも、同じ関数で方策の本体の隠れ状態を
取り出す。

記号 / Notation:
    B       : バッチサイズ
    S       : 系列長
    d_model : 隠れ状態の次元
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.data.preference import collate_scored_sequences
from src.layers.attention import create_causal_mask
from src.models.gpt import GPTLanguageModel


def compute_final_hidden_states(model: GPTLanguageModel, token_ids: Tensor) -> Tensor:
    """最終正規化層の後の隠れ状態 ``(B, S, d_model)`` を返す(``lm_head`` の直前)。

    ``GPTLanguageModel.forward(token_ids)``(``positions=None``・``kv_cache=None``)と同じ手順:
    トークン埋め込み -> 因果マスク付きの ``DecoderBlock`` を ``num_layers`` 段 -> 最終正規化層。
    """
    _, seq_len = token_ids.shape
    if seq_len > model.max_sequence_length:
        raise ValueError(
            f"系列長 ({seq_len}) が max_sequence_length ({model.max_sequence_length}) を超えている"
        )
    hidden = model.token_embedding(token_ids)
    causal_mask = create_causal_mask(seq_len, device=token_ids.device)
    for i, block in enumerate(model.blocks):
        hidden, _, _ = block(
            hidden, None, tgt_mask=causal_mask, positions=None, kv_cache=None, layer_idx=i
        )
    return model.final_norm(hidden)


class RewardModel(nn.Module):
    """言語モデルの本体 + スカラーのヘッドによる報酬モデル。

    Args:
        backbone: 本体にする ``GPTLanguageModel``(SFT 済みの重みを複製して渡す)。語彙への射影
            ``lm_head`` は報酬の計算に使わない(重み共有によりトークン埋め込みと同じテンソルなので、
            埋め込みとしては学習される)。
        head_init_std: ヘッドの重み w の初期化の標準偏差(バイアス b は 0 で初期化)。乱数は
            呼び出し前の ``torch.manual_seed`` で固定する。
    """

    def __init__(self, backbone: GPTLanguageModel, head_init_std: float) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = backbone.token_embedding.embedding_dim
        self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, token_ids: Tensor, last_indices: Tensor) -> Tensor:
        """各系列の報酬 ``(B,)`` を返す。``last_indices[b]`` は系列 b の最終トークンの位置。"""
        hidden = compute_final_hidden_states(self.backbone, token_ids)
        # 最終トークンの位置の隠れ状態を one-hot の重みの和で取り出す。高度なインデックス
        # (hidden[rows, last_indices])の逆伝播は index_put(accumulate=True)になり、MPS では
        # 決定的な実装がない(torch.use_deterministic_algorithms(True) で例外になる)。one-hot の
        # 和は値が同じで、逆伝播も要素ごとの積なので、どのデバイスでも決定的である。
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        selector = (positions[None, :] == last_indices[:, None]).to(hidden.dtype)  # (B, S)
        last_hidden = (selector[:, :, None] * hidden).sum(dim=1)
        return self.head(last_hidden).squeeze(-1)


@torch.no_grad()
def score_sequences(
    reward_model: RewardModel,
    sequences: Sequence[Sequence[int]],
    device: torch.device | str,
    batch_size: int = 256,
) -> np.ndarray:
    """系列(プロンプト + 応答のトークン列)ごとの報酬を float64 の配列で返す。

    系列を長さの順に並べてからバッチにし(パディングを減らすため)、元の順に戻して返す。
    右パディングは因果マスクにより最終トークンの位置の隠れ状態を変えない。
    """
    was_training = reward_model.training
    reward_model.eval()
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    scores = np.empty(len(sequences), dtype=np.float64)
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            token_ids, last_indices = collate_scored_sequences([sequences[i] for i in chunk])
            values = reward_model(token_ids.to(device), last_indices.to(device))
            scores[chunk] = values.float().cpu().double().numpy()
    finally:
        reward_model.train(was_training)
    return scores
