"""cross-encoder による並べ替えのモデル(025)。

**dual encoder との違い**: dual encoder(024)は query と passage を別々に符号化し、
スコアを内積``f(q)^T g(p)`` で求める。スコアが ``q`` の関数と ``p`` の関数の積に分解できるので、
passage の埋め込みを事前に計算して索引にできるが、passage の情報を
1 本のベクトルに詰める制約がある。cross-encoder は query と passage を **連結して 1 本の入力**
にし、すべてのトークンの間の注意(attention)で相互作用を計算してスコア ``g(q, p)`` を出す。query
ごとに、すべての候補との組を符号化する必要があるので、全件には使えず、第
1 段が絞り込んだ候補の並べ替え(reranking)に使う(Nogueira & Cho, "Passage Re-ranking with BERT",
2019)。

**入力の列**: ``[query のトークン] [区切り] [passage のトークン] [終端]``。
008 のトークナイザの語彙は 256 個のバイトと 7,936 個のマージで、特殊トークンがない。
**語彙を増やさず**、コーパスに一度も現れない単一バイトのトークン(制御文字)を区切りと終端に使う(区
切り: 0x1F、終端: 0x1E。025 のノートブックで、コーパスでの出現回数が 0 であることを確かめる)。
すべての列が同じ長さ(``L_q + 1 + L_p + 1``)なのでパディングは使わない。

**終端の位置の出力を読む理由**: 008 の小型 GPT は因果マスクを使うので、位置 ``t``の隠れ状態は位置
``t`` までのトークンにだけ依存する。**query 全体と passage 全体の両方を見ている位置は、
列の最後の位置(終端のトークン)だけ** である(query のトークンの位置は passage を見ておらず、passage
の途中の位置は query 全体と passage の前半だけを見ている)。そこで、
最終正規化層の後の終端の位置の隠れ状態 ``h_end`` に線形の層を付けてスカラーのスコアにする。

.. math::

    g(q, p) = w^\\top h_{\\mathrm{end}}(q, p) + b

``w`` は ``d_model`` 次元の重み、``b`` はバイアスである(017 の報酬モデルと同じ形)。
語彙への射影``lm_head`` は使わない(重み共有によりトークン埋め込みと同じテンソルなので、
埋め込みとしては学習される)。

**更新するパラメータ**: 本体のすべてのパラメータとヘッド(025 の実験 D の dual encoder の条件も、
本体のすべてのパラメータを更新する)。

記号 / Notation:
    L_q : query の長さ、L_p : passage の長さ、d_model : 隠れ状態の次元、B : バッチサイズ
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.gpt import GPTLanguageModel
from src.models.text_embedding import compute_hidden_states


def build_cross_encoder_inputs(
    query_tokens: Tensor, passage_tokens: Tensor, separator_id: int, terminal_id: int
) -> Tensor:
    """query と passage を ``[query] [区切り] [passage] [終端]`` の列に連結する。

    Args:
        query_tokens: 形状 ``(B, L_q)``。
        passage_tokens: 形状 ``(B, L_p)``(``query_tokens`` と同じ ``B``。組ごとに 1 対 1 に対応)。
        separator_id: 区切りのトークンの ID。
        terminal_id: 終端のトークンの ID。

    Returns:
        形状 ``(B, L_q + 1 + L_p + 1)`` の整数のテンソル。
    """
    batch = query_tokens.size(0)
    separator = torch.full(
        (batch, 1), separator_id, dtype=query_tokens.dtype, device=query_tokens.device
    )
    terminal = torch.full(
        (batch, 1), terminal_id, dtype=query_tokens.dtype, device=query_tokens.device
    )
    return torch.cat(
        [query_tokens, separator, passage_tokens.to(query_tokens.dtype), terminal], dim=1
    )


class CrossEncoder(nn.Module):
    """小型 GPT の本体 + 終端の位置の出力へのスカラーのヘッドによる cross-encoder。

    Args:
        backbone: 本体にする ``GPTLanguageModel``(008 の重みを読み込んだもの)。
        separator_id: 区切りのトークンの ID。
        terminal_id: 終端のトークンの ID。
        head_init_std: ヘッドの重み ``w`` の初期化の標準偏差(バイアス ``b`` は 0)。
        乱数は呼び出し前の
            ``torch.manual_seed`` で固定する。
    """

    def __init__(
        self,
        backbone: GPTLanguageModel,
        separator_id: int,
        terminal_id: int,
        head_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.separator_id = separator_id
        self.terminal_id = terminal_id
        d_model = backbone.token_embedding.embedding_dim
        self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, query_tokens: Tensor, passage_tokens: Tensor) -> Tensor:
        """組ごとのスコア ``(B,)``。ヘッドは FP32 で計算する。"""
        inputs = build_cross_encoder_inputs(
            query_tokens, passage_tokens, self.separator_id, self.terminal_id
        )
        hidden = compute_hidden_states(self.backbone, inputs, "causal")
        with torch.autocast(device_type=inputs.device.type, enabled=False):  # ヘッドは常に FP32
            return self.head(hidden[:, -1, :].float()).squeeze(-1)
