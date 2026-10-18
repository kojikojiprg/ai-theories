"""長さの違う要求をまとめて 1 回の forward で進める処理(モデルの実行)。

``GPTLanguageModel.forward`` と ``KeyValueCache`` は、バッチの全行が同じ位置・同じ長さであることを
前提にする(``positions`` は全行共通の ``(S,)``、キャッシュは全行が同じ長さ)。推論サーバでは、
バッチの中の要求が別々の位置にいて(RoPE の位置が行ごとに違う)、KV キャッシュの長さも行ごとに違う。
そこで本モジュールは、**学習済みの重みと、既存の層(正規化・順伝播ネットワーク・射影)をそのまま使い**、
注意機構の部分だけを、行ごとの位置と KV キャッシュの置き場所(``src/serving/kv_store.py``)を扱う
形で書き直す。既存の ``src/`` のコードは変更しない。

1 回の forward は、次の手順を層ごとに行う。

1. 入力 ``x`` を正規化(``norm1``)し、Query・Key・Value に射影する(``w_q``・``w_k``・``w_v``)。
2. 行ごとの絶対位置で RoPE を適用する(``cos_cached[位置]`` を行ごとに引く)。
3. 新しい Key / Value を、KV キャッシュの置き場所(スロットまたはブロック表が指す場所)に書き込む。
4. 各行の過去の Key / Value を添字で集め(足りない部分は埋める)、位置に基づくマスク
   (``集めた位置 j <= Query の位置``)をかけて注意機構を計算する。
5. 出力射影(``w_o``)と順伝播ネットワーク(``norm3`` → ``feed_forward``)を残差接続で足す。

最後の層の後は、各行の最後の有効な位置だけに最終正規化と語彙への射影をかける(次のトークンを選ぶのに
必要なのはそこだけ)。

**prefill の行と decode の行が同じ forward に混ざる場合**(Orca の selective batching に倣う):
行ごとに問い合わせの長さが違う(prefill の行は未計算の書き出し、decode の行は 1 トークン)ので、
全行を最長の ``Q`` まで埋める。埋め方と、埋めた部分の除き方は次のとおりである。

- 入力: 短い行の末尾は、トークン 0・位置 0 で埋める(埋めた位置は有限の値で計算されるだけ)。
- 書き込み: 埋めた位置の Key / Value は書き込まない(有効なトークンの添字だけを ``write_*`` に
  入れる)。
- マスク: 集めた位置 ``j`` が Query の絶対位置以下のときだけ参加させる。prefill の行の Query の
  位置は 1 つずつ増えるので、この規則がそのまま因果マスクになる(接頭辞の共有で先頭が計算済みの
  行も、Query の位置が絶対位置なので同じ規則で正しい)。埋めた Query の位置は 0 なので先頭の Key
  だけを見る(結果は使わない)。集めた Key / Value の、行の長さより後ろの埋めた部分は、位置が
  Query の位置より後ろなので除かれる。
- 出力: 各行の最後の有効な位置(``last_index``)の隠れ状態だけを取り出す。埋めた位置の出力は
  使わない。
- 計算量: decode の行も最長の ``Q`` まで埋めるので、処理するトークン数は 行数 × ``Q`` になる
  (費用モデルは、これを ``clock.py`` の式で見積もる)。

導入: 027。

記号 / Notation:
    R   : 行数(バッチの中の要求の数)
    Q   : 今回の forward の最長の問い合わせの長さ(prefill の行は未計算の書き出しの長さ、
          decode の行は 1)
    Lkv : 集めた Key / Value の長さ
    h   : Query のヘッド数、g : Key / Value のヘッド数、d_k : ヘッドあたりの次元
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel
from src.serving.kv_store import BatchLayout, KeyValueStore, Row


class ServingModel:
    """``GPTLanguageModel`` を、行ごとに位置の違うバッチで実行するための実行器。

    Args:
        model: 実行するモデル。注意機構は RoPE(周波数の変換なし)、正規化前置、交差注意なしであること
            (008 の小型 GPT がこの構成)。評価モードにして使う。

    Raises:
        ValueError: 想定と違う構成のモデルのとき(黙って別の計算をしないため)。
    """

    def __init__(self, model: GPTLanguageModel) -> None:
        model.eval()
        self.model = model
        first = model.blocks[0].self_attn
        rope = first.positional_transform
        if not isinstance(rope, RotaryPositionEmbedding) or rope.frequency_scaling is not None:
            raise ValueError("RoPE(周波数の変換なし)を注入したモデルだけに対応する")
        for block in model.blocks:
            if not block.norm_first or block.use_cross_attention:
                raise ValueError("正規化前置・交差注意なしの DecoderBlock だけに対応する")
            attn = block.self_attn
            if attn.positional_transform is not rope or attn.attention_score_bias is not None:
                raise ValueError(
                    "全層で同じ RoPE を共有し、注意のスコアへのバイアスがないモデルだけに対応する"
                )
        if rope.cos_cached.size(0) < model.max_sequence_length:
            raise ValueError("RoPE の cos / sin が最大の文脈長まで事前計算されていない")
        self.rope = rope
        self.num_layers = model.num_layers
        self.num_heads = first.num_heads
        self.num_key_value_heads = first.num_key_value_heads
        self.head_dim = first.d_k
        self.d_model = first.d_model
        self.max_sequence_length = model.max_sequence_length
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype

    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        rows, length, _ = x.shape
        return x.view(rows, length, num_heads, self.head_dim).transpose(1, 2)

    @torch.no_grad()
    def forward(self, layout: BatchLayout, store: KeyValueStore) -> Tensor:
        """配置 ``layout`` の 1 回分を実行し、各行の最後の有効な位置の logits を返す。

        Returns:
            形状 ``(R, V)`` の logits。
        """
        model = self.model
        x = model.token_embedding(layout.token_ids)  # (R, Q, d_model)
        rows, query_length, _ = x.shape
        cos = self.rope.cos_cached[layout.positions][:, None].to(dtype=x.dtype)  # (R, 1, Q, d_k)
        sin = self.rope.sin_cached[layout.positions][:, None].to(dtype=x.dtype)
        key_length = layout.gather_first.size(1)
        key_index = torch.arange(key_length, device=x.device)
        # 集めた位置 j が Query の絶対位置以下のときだけ参加させる(埋めた部分は位置より後ろ)
        allowed = (key_index[None, None, :] <= layout.positions[:, :, None])[
            :, None
        ]  # (R, 1, Q, Lkv)
        group = self.num_heads // self.num_key_value_heads
        scale = math.sqrt(self.head_dim)

        for layer, block in enumerate(model.blocks):
            attn = block.self_attn
            normed = block.norm1(x)
            q = self._split_heads(attn.w_q(normed), self.num_heads)  # (R, h, Q, d_k)
            k = self._split_heads(attn.w_k(normed), self.num_key_value_heads)  # (R, g, Q, d_k)
            v = self._split_heads(attn.w_v(normed), self.num_key_value_heads)
            rotate = RotaryPositionEmbedding._rotate_half
            q = q * cos + rotate(q) * sin
            k = k * cos + rotate(k) * sin

            # 新しい Key / Value を、置き場所(スロットまたはブロック表が指す場所)に書き込む
            store.keys[layer][layout.write_first, layout.write_second] = k.transpose(1, 2)[
                layout.write_row, layout.write_column
            ]
            store.values[layer][layout.write_first, layout.write_second] = v.transpose(1, 2)[
                layout.write_row, layout.write_column
            ]
            # 各行の過去の Key / Value を集める
            gathered_k = store.keys[layer][
                layout.gather_first, layout.gather_second
            ]  # (R, Lkv, g, d_k)
            gathered_v = store.values[layer][layout.gather_first, layout.gather_second]
            keys = gathered_k.transpose(1, 2)  # (R, g, Lkv, d_k)
            values = gathered_v.transpose(1, 2)
            if group > 1:
                keys = keys.repeat_interleave(group, dim=1)
                values = values.repeat_interleave(group, dim=1)

            scores = torch.matmul(q, keys.transpose(-2, -1)) / scale  # (R, h, Q, Lkv)
            scores = scores.masked_fill(~allowed, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, values)  # (R, h, Q, d_k)
            merged = context.transpose(1, 2).reshape(rows, query_length, self.d_model)
            x = x + attn.w_o(merged)
            x = x + block.feed_forward(block.norm3(x))

        last = x[torch.arange(rows, device=x.device), layout.last_index]  # (R, d_model)
        return model.lm_head(model.final_norm(last))


class ModelRunner:
    """行の集まりを 1 回の forward で進め、次のトークン(貪欲法)を返す。

    Args:
        serving_model: 実行器。
    """

    def __init__(self, serving_model: ServingModel) -> None:
        self.serving_model = serving_model
        self.device = serving_model.device

    def step(
        self, rows: Sequence[Row], store: KeyValueStore, return_logits: bool = False
    ) -> tuple[list[int], np.ndarray | None]:
        """``rows`` を 1 回の forward で進める(Key / Value は ``store`` に書き込まれる)。

        Returns:
            ``(次のトークンのリスト, logits)``。logits は ``return_logits`` が真のときだけ
            ``(R, V)`` の numpy 配列(計算した型のまま)。
        """
        layout = store.build_layout(rows)
        logits = self.serving_model.forward(layout, store)
        tokens = logits.argmax(dim=-1).tolist()  # デバイスの完了を待つ
        return tokens, (logits.cpu().numpy() if return_logits else None)


class DryRunner:
    """モデルを実行せずに、スケジューラとメモリ管理の動作だけを進める実行器(見積もり用)。

    出力長は要求に割り当てた長さで決まるので、トークンの値は割り当て・退避・終了の判断に影響しない。
    そのため、同じ負荷・同じスケジューラ・同じストアで、実行と同じ反復の列が得られる(オフラインの場合)。
    """

    device = torch.device("cpu")

    def step(
        self, rows: Sequence[Row], store: KeyValueStore, return_logits: bool = False
    ) -> tuple[list[int], np.ndarray | None]:
        return [0] * len(rows), None
