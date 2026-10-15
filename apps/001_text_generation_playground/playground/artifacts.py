"""Hugging Face Hub のモデル・トークナイザの取得と、モデルの構築。

取得するもの(いずれも共有アーティファクトで、``kojikojiprg`` の公開リポジトリ):

- モデル ``kojikojiprg/ai-theories-small-gpt-en``(ブランチごとに ``config.json`` と
  ``model_state.pt``)
    - ``main``: 008 のデコーディング戦略の実験が使う、事前学習済みの小型 GPT(多頭注意機構)。
    - ``gqa``: 010 の実験 D で Key / Value ヘッド数を減らして追加学習した
      Grouped-Query Attention 版。
    - ``sft-synthetic``: 017 が上げた、016 の合成の指示データで SFT(Supervised Fine-Tuning)した版。
- トークナイザ ``kojikojiprg/ai-theories-tokenizer-en``(``tokenizer.json``。
  モデルには同梱されない)。

モデルの構築は、010・016・017 のノートブックが ``config.json`` から組み立てている手順と同じ
(RoPE・RMSNorm・SwiGLU の小型 GPT)。構造が想定と違う ``config.json`` は、黙って別の構造のモデルを
作らないように例外にする。
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from src.data.tokenizer import BPEIDTokenizer, load_bpe_id_tokenizer_from_hub
from src.layers.feedforward import SwiGLUFeedForwardNetwork
from src.layers.normalization import RMSNorm
from src.layers.positional_encoding import RotaryPositionEmbedding
from src.models.gpt import GPTLanguageModel

MODEL_REPO_ID = "kojikojiprg/ai-theories-small-gpt-en"
TOKENIZER_REPO_ID = "kojikojiprg/ai-theories-tokenizer-en"
CHECKPOINT_FILENAME = "model_state.pt"
CONFIG_FILENAME = "config.json"

# 構築できる構造(これ以外の config.json は例外にする)
EXPECTED_ARCHITECTURE = {
    "positional_encoding": "rope",
    "normalization": "rmsnorm",
    "feed_forward": "swiglu",
}


@dataclass(frozen=True)
class ModelSpec:
    """Hub 上のモデルの 1 つのブランチ。

    Attributes:
        key: コードと画面の内部で使う識別子(ブランチ名と同じ)。
        revision: Hub のブランチ名。
    """

    key: str
    revision: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("main", "main"),
    ModelSpec("gqa", "gqa"),
    ModelSpec("sft-synthetic", "sft-synthetic"),
)


@dataclass(frozen=True)
class LoadedModel:
    """読み込み済みのモデル。

    Attributes:
        spec: 読み込んだブランチ。
        config: ``config.json`` の内容。
        model: 評価モード・勾配なしの ``GPTLanguageModel``(``device`` 上)。
        num_parameters: 重み共有を 1 回と数えたパラメータ数。
        parameter_bytes: 重み共有を 1 回と数えたパラメータのバイト数。
        key_value_cache_bytes_per_token: 1 トークンあたりの KV キャッシュのバイト数
            (``2 * L * g * d_k * 要素のバイト数``、010 の閉形式)。
    """

    spec: ModelSpec
    config: dict
    model: GPTLanguageModel
    num_parameters: int
    parameter_bytes: int
    key_value_cache_bytes_per_token: int

    @property
    def max_sequence_length(self) -> int:
        return self.model.max_sequence_length


@dataclass(frozen=True)
class Resources:
    """起動時に 1 回だけ読み込むもの。

    Attributes:
        device: 生成に使うデバイス。
        tokenizer: 英語のバイトレベル BPE トークナイザ。
        models: ``ModelSpec.key`` から読み込み済みのモデルへの対応。
    """

    device: torch.device
    tokenizer: BPEIDTokenizer
    models: dict[str, LoadedModel]


def select_device() -> torch.device:
    """CUDA → MPS(Metal Performance Shaders)→ CPU の順に、使えるデバイスを選ぶ。

    ZeroGPU では、``@spaces.GPU`` の関数の外(モジュールの読み込み時)でも
    ``torch.cuda.is_available()`` が真になる(CUDA のエミュレーション)。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict) -> GPTLanguageModel:
    """``config.json`` から小型 GPT を構築する(重みは未読み込み)。

    010 の ``build_production_model()`` と同じ手順。``num_key_value_heads`` があれば
    Grouped-Query Attention になる(``gqa`` ブランチ)。
    """
    for name, expected in EXPECTED_ARCHITECTURE.items():
        if config.get(name) != expected:
            raise ValueError(
                f"config.json の {name} が想定({expected!r})と違う: {config.get(name)!r}。"
                "このアプリが構築できる構造ではない。"
            )
    d_model = config["d_model"]
    d_k = d_model // config["num_heads"]
    return GPTLanguageModel(
        vocabulary_size=config["vocabulary_size"],
        d_model=d_model,
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_sequence_length=config["sequence_length"],
        positional_transform=RotaryPositionEmbedding(d_k, max_position=config["sequence_length"]),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(
            SwiGLUFeedForwardNetwork, d_model, config["swiglu_d_ff"]
        ),
        tie_embeddings=config["tie_embeddings"],
        dropout=config["dropout"],
        norm_first=config.get("norm_first", True),
        num_key_value_heads=config.get("num_key_value_heads"),
    )


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """重み共有(トークン埋め込みと出力層)を 1 回と数えた ``(パラメータ数, バイト数)``。

    ``Module.parameters()`` は同じ ``Parameter`` を 1 回しか返さないので、そのまま足せばよい。
    """
    parameters = list(model.parameters())
    total = sum(p.numel() for p in parameters)
    total_bytes = sum(p.numel() * p.element_size() for p in parameters)
    return total, total_bytes


def load_model(spec: ModelSpec, device: torch.device) -> LoadedModel:
    """Hub のブランチからモデルを読み込み、評価モードにして ``device`` へ載せる。"""
    config_path = hf_hub_download(MODEL_REPO_ID, CONFIG_FILENAME, revision=spec.revision)
    checkpoint_path = hf_hub_download(MODEL_REPO_ID, CHECKPOINT_FILENAME, revision=spec.revision)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = build_model(config)
    # state_dict(テンソルの辞書)だけを読む。任意のコードを実行しうる pickle の読み込みは許さない。
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)

    num_parameters, parameter_bytes = count_parameters(model)
    d_k = config["d_model"] // config["num_heads"]
    key_value_heads = config.get("num_key_value_heads") or config["num_heads"]
    element_bytes = next(model.parameters()).element_size()
    cache_bytes_per_token = 2 * config["num_layers"] * key_value_heads * d_k * element_bytes
    return LoadedModel(
        spec=spec,
        config=config,
        model=model,
        num_parameters=num_parameters,
        parameter_bytes=parameter_bytes,
        key_value_cache_bytes_per_token=cache_bytes_per_token,
    )


def load_resources() -> Resources:
    """トークナイザと 3 ブランチのモデルを、同時にメモリへ載せる(起動時に 1 回だけ呼ぶ)。"""
    device = select_device()
    tokenizer, _ = load_bpe_id_tokenizer_from_hub(TOKENIZER_REPO_ID)
    models = {spec.key: load_model(spec, device) for spec in MODEL_SPECS}
    for loaded in models.values():
        if loaded.config["vocabulary_size"] != tokenizer.vocab_size:
            raise ValueError(
                f"{loaded.spec.key}: 語彙サイズ({loaded.config['vocabulary_size']})が"
                f"トークナイザ({tokenizer.vocab_size})と一致しない"
            )
    return Resources(device=device, tokenizer=tokenizer, models=models)
