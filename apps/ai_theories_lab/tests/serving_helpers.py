"""常駐サーバの試験で共有する関数(エンジンを 1 反復ずつ同期的に進める道具と、基準との比較)。"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from lab.core.artifacts import LoadedModel, Resources
from lab.features.generation.service import DecodingSettings
from lab.serving.engine import RequestOutcome, ResidentEngine
from lab.serving.runtime import EngineConfig, build_engine
from src.serving import logits_tolerance, reference_generate


@dataclass
class Collector:
    """``RequestSink``: 受け取ったトークンと終わりを貯める。"""

    tokens: list[int] = field(default_factory=list)
    outcome: RequestOutcome | None = None
    finished: threading.Event = field(default_factory=threading.Event)

    def on_token(self, token_id: int, index: int) -> None:
        assert index == len(self.tokens), "トークンの番号が連続していない"
        self.tokens.append(token_id)

    def on_finish(self, outcome: RequestOutcome) -> None:
        assert self.outcome is None, "終わりが 2 回通知された"
        self.outcome = outcome
        self.finished.set()


def greedy(max_new_tokens: int) -> DecodingSettings:
    return DecodingSettings(method="greedy", max_new_tokens=max_new_tokens)


def make_engine(
    resources: Resources,
    config: EngineConfig | None = None,
    model_key: str = "main",
    **options,
) -> ResidentEngine:
    """アプリと同じ経路(``build_engine``)で、試験用のエンジンを作る。"""
    return build_engine(model_key, resources.models[model_key], config or EngineConfig(), **options)


def drive(engine: ResidentEngine, max_steps: int = 100_000) -> int:
    """仕事がなくなるまで、エンジンを 1 反復ずつ進める。進めた回数を返す。"""
    steps = 0
    while engine.step():
        steps += 1
        assert steps < max_steps, "エンジンが終わらない"
    return steps


def random_prompt(rng: np.random.Generator, length: int, vocabulary_size: int) -> list[int]:
    return [int(t) for t in rng.integers(0, vocabulary_size, size=length)]


def check_against_reference(
    loaded: LoadedModel,
    prompt: Sequence[int],
    tokens: Sequence[int],
    logits: Sequence[np.ndarray],
) -> None:
    """エンジンを通した貪欲法の生成が、1 要求ずつの生成(010 の KV キャッシュ)と一致すること。

    logits の差は、計算する型の丸めの単位と値の大きさから導いた許容誤差で比べる
    (``logits_tolerance``)。
    上位 2 つの logits の差が許容誤差の 2 倍以下の手(僅差の手)でトークンが分かれた場合は、
    その手までの logits を比べて、それ以降(文脈が違ってくる)は比べない。
    """
    expected, reference = reference_generate(loaded.model, prompt, len(tokens))
    assert len(logits) == len(tokens)
    tolerance = logits_tolerance(loaded.model, reference)
    for step, (token, expected_token) in enumerate(zip(tokens, expected, strict=True)):
        difference = float(np.abs(np.asarray(logits[step]) - reference[step]).max())
        assert difference <= tolerance, (
            f"{step} 番目の手の logits の差 {difference:.3e} が許容誤差 {tolerance:.3e} を超えた"
        )
        if token != expected_token:
            top2 = np.sort(reference[step])[-2:]
            assert top2[1] - top2[0] <= 2 * tolerance, (
                f"{step} 番目の手でトークンが違う({token} と {expected_token})が、僅差の手ではない"
            )
            return


def assert_tokens_match_reference(
    loaded: LoadedModel, prompt: Sequence[int], tokens: Sequence[int]
) -> None:
    """生成したトークン列が、1 要求ずつの貪欲法の生成と一致すること(logits が手元にないとき)。

    トークンが分かれた手が、上位 2 つの logits の差が許容誤差の 2 倍以下の僅差の手であれば許す
    (それ以降は文脈が違ってくるので比べない)。
    """
    expected, reference = reference_generate(loaded.model, prompt, len(tokens))
    tolerance = logits_tolerance(loaded.model, reference)
    for step, (token, expected_token) in enumerate(zip(tokens, expected, strict=True)):
        if token != expected_token:
            top2 = np.sort(reference[step])[-2:]
            assert top2[1] - top2[0] <= 2 * tolerance, (
                f"{step} 番目の手でトークンが違う({token} と {expected_token})が、僅差の手ではない"
            )
            return
