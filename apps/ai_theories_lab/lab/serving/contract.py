"""API の要求の検査と、生成したトークンの逐次復号(torch に依存しない)。

HTTP の層(``api.py``)が使う純粋な Python の関数をまとめる。要求の検査は、トークナイザと
モデルの最大の文脈長だけで判定でき、エンジンの待ち行列に入れる前に行う。

導入: 段階 2-1(システムの仕組み。使った理論は 008 のデコードの方法)。
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass

from lab.features.generation.service import (
    MAX_SEED,
    METHOD_GREEDY,
    METHOD_TEMPERATURE,
    METHOD_TOP_K,
    METHOD_TOP_P,
    DecodingSettings,
)

API_METHODS = (METHOD_GREEDY, METHOD_TEMPERATURE, METHOD_TOP_K, METHOD_TOP_P)
"""API で選べるデコードの方法(ビームサーチは、行をまとめて進める常駐のエンジンでは使えない)。"""
MAX_PROMPT_CHARACTERS = 8000
"""書き出しの文字数の上限(トークン分割の時間が、極端に長くならないようにする)。"""
DEFAULT_MAX_TOKENS = 64


class InvalidRequest(ValueError):
    """要求の内容が誤っている(HTTP 422)。メッセージは日本語で、そのまま利用者に返す。"""


@dataclass(frozen=True)
class PreparedRequest:
    """検査を通った要求。

    Attributes:
        prompt_token_ids: 書き出しのトークン列。
        settings: 行ごとのデコードの設定(``max_new_tokens`` を含む)。
    """

    prompt_token_ids: list[int]
    settings: DecodingSettings


def draw_seed() -> int:
    """シードの指定がないときに、サーバが決める乱数のシード(応答に含めるので、再現できる)。"""
    return secrets.randbelow(MAX_SEED + 1)


def prepare_request(
    encode: Callable[[str], list[int]],
    max_sequence_length: int,
    *,
    prompt: str,
    max_tokens: int,
    method: str,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None,
) -> PreparedRequest:
    """書き出しをトークンに分け、デコードの設定と合わせて検査する。

    Raises:
        InvalidRequest: 生成できない要求(書き出しと生成の合計が最大の文脈長を超える場合も含む)。
    """
    if method not in API_METHODS:
        raise InvalidRequest(f"デコードの方法は {', '.join(API_METHODS)} のどれかにしてください。")
    if not prompt.strip():
        raise InvalidRequest("書き出しが空です。英語の文を入力してください。")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise InvalidRequest(f"書き出しは {MAX_PROMPT_CHARACTERS} 文字以下にしてください。")
    prompt_ids = encode(prompt)
    settings = DecodingSettings(
        method=method,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=draw_seed() if seed is None else seed,
    )
    try:
        settings.validate(len(prompt_ids), max_sequence_length)
    except ValueError as error:
        raise InvalidRequest(str(error)) from error
    return PreparedRequest(prompt_ids, settings)


class IncrementalDecoder:
    """トークンを 1 つずつ受け取り、増えた文字列を返す。

    バイトレベルの BPE は、1 つのトークンがマルチバイト文字の途中で切れることがある。その末尾の
    不完全な部分(復号すると置換文字 U+FFFD になる)は、次のトークンが続きを持ってくるまで保留する。
    保留した分は ``finish()`` で返す。``push`` の戻り値と ``finish()`` の戻り値を連結すると、
    全トークンを一度に復号した文字列と一致する。
    """

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self._decode = decode
        self._ids: list[int] = []
        self._sent = 0

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._decode(self._ids)
        end = len(text)
        while end > self._sent and text[end - 1] == "�":
            end -= 1
        delta = text[self._sent : end]
        self._sent = end
        return delta

    def finish(self) -> str:
        text = self._decode(self._ids)
        delta = text[self._sent :]
        self._sent = len(text)
        return delta
