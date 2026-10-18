"""要求(request)と、要求ごとの状態の管理。

推論サーバが扱う 1 件の要求は、書き出し(プロンプト)のトークン列、生成するトークンの数
(出力長)、到着時刻の 3 つで決まる。出力長は、生成の途中の終端トークンでは止めず、要求に
割り当てた長さで止める(推論サーバの評価で一般的な扱い。出力長の分布を設計どおりに制御するため)。

``RequestState`` は、要求が待ち行列に入ってから終わるまでの状態(生成済みのトークン、KV キャッシュに
書き込み済みのトークン数、各時刻)を持つ。KV キャッシュの置き場所そのもの(スロットやブロック表)は
``src/serving/kv_store.py`` の側が持ち、ここには含めない。

導入: 027。

記号 / Notation:
    P : 書き出しの長さ(トークン数)
    O : 出力長(生成するトークン数)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RequestStatus(Enum):
    """要求の状態。"""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class Request:
    """1 件の要求。

    Args:
        request_id: 要求の識別子(1 回の実行の中で一意)。
        prompt_token_ids: 書き出しのトークン列(長さ P、1 以上)。
        output_length: 生成するトークンの数 O(1 以上)。
        arrival_time: 到着時刻(秒)。一括処理(offline)では全要求が 0。
    """

    request_id: int
    prompt_token_ids: tuple[int, ...]
    output_length: int
    arrival_time: float = 0.0

    def __post_init__(self) -> None:
        if len(self.prompt_token_ids) < 1:
            raise ValueError("書き出しは 1 トークン以上である必要がある")
        if self.output_length < 1:
            raise ValueError(f"出力長は 1 以上である必要がある: {self.output_length}")
        if self.arrival_time < 0:
            raise ValueError(f"到着時刻は 0 以上である必要がある: {self.arrival_time}")

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_length(self) -> int:
        """書き出しと出力を合わせた長さ(P + O)。"""
        return self.prompt_length + self.output_length


@dataclass
class RequestState:
    """実行中の要求の状態。

    Attributes:
        request: 元の要求。
        generated: これまでに生成したトークン(最大 O 個)。
        num_cached_tokens: KV キャッシュに Key / Value を書き込み済みのトークンの数。
            この値が、次に書き込むトークンの絶対位置になる。
        status: 待っている / 実行中 / 終了。
        admit_order: 最後に実行に入れた順番(新しいほど大きい。退避の対象を選ぶのに使う)。
        admitted_time: 最初に実行に入れた時刻。
        first_token_time: 最初のトークンを生成した時刻。
        finish_time: 最後のトークンを生成した時刻。
        num_preemptions: 退避(メモリ不足による一時的な取り下げ)された回数。
        logits: ``record_logits`` が真のとき、生成したトークンごとの logits(語彙の次元)。
    """

    request: Request
    generated: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    admit_order: int = -1
    admitted_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    num_preemptions: int = 0
    logits: list[np.ndarray] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return len(self.generated) >= self.request.output_length

    def context_token_ids(self) -> list[int]:
        """KV キャッシュに載せる対象の列(書き出しと、これまでに生成したトークン)。

        退避された要求を再び実行に入れるときは、この全体を prefill し直す(再計算)。
        """
        return list(self.request.prompt_token_ids) + self.generated

    def last_token(self) -> int:
        """次の decode の入力になるトークン(直前に生成したトークン)。"""
        return self.generated[-1]
