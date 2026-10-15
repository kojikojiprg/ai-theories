"""代替のアーティファクト(乱数の重み)。

構造だけが本物と同じで重みが乱数のモデル(``main``・``gqa``・``sft-synthetic``)と、小さな語彙の
トークナイザを、Hugging Face のキャッシュと同じ配置(``refs/``・``snapshots/``)で作る。
オフラインで読み込めば、アプリのコードは Hub から取得する経路(``hf_hub_download``)をそのまま通る。
重みが乱数なので、**生成される文章や応答の中身には意味がなく、モデルの性能は何も確かめられない。**
確かめるのは、コードの経路と、理論が要求する不変条件である。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from lab.core.artifacts import MODEL_REPO_ID, TOKENIZER_REPO_ID, build_model
from src.data.instruction import SEEN_INSTRUCTIONS, UNSEEN_INSTRUCTIONS, WORD_VOCABULARY
from src.data.tokenizer import BPEIDTokenizer, learn_bpe, save_bpe_id_tokenizer_json
from tests.constants import GQA_KEY_VALUE_HEADS, PRODUCTION_CONFIG

STAND_IN_VOCABULARY_SIZE = 700

# 注意の重みやトークン分割の試験が使う、いくつかの英文(語彙の学習にも使う)
TRAINING_SENTENCES = [
    "The history of the city began when the river town grew into a market and a port.",
    "Albert Einstein was born in a small town, and the film was released in the spring.",
    "The unbelievable results were unexpectedly interesting for the tokenization of words.",
]


def _write_repository(
    cache_dir: Path, repo_id: str, revisions: dict[str, dict[str, bytes]]
) -> None:
    """Hugging Face のキャッシュと同じ配置(refs/・snapshots/)でリポジトリを書く。"""
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    for revision, files in revisions.items():
        commit = hashlib.sha1(f"{repo_id}@{revision}".encode()).hexdigest()
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / revision).write_text(commit, encoding="utf-8")
        for name, content in files.items():
            (snapshot / name).write_bytes(content)


def build_stand_in_artifacts(directory: Path) -> Path:
    """代替のアーティファクトを ``directory`` に作り、Hugging Face のキャッシュの
    ディレクトリを返す。
    """
    corpus = " ".join(
        [
            *TRAINING_SENTENCES,
            "### Instruction: ### Response: ### End Words:",
            *(
                t
                for table in (SEEN_INSTRUCTIONS, UNSEEN_INSTRUCTIONS)
                for ts in table.values()
                for t in ts
            ),
            *WORD_VOCABULARY,
            *WORD_VOCABULARY,
        ]
    )
    bpe = learn_bpe(corpus, STAND_IN_VOCABULARY_SIZE, byte_level=True, max_chunk_bytes=64)
    symbol_to_id = {symbol: i for i, symbol in enumerate(sorted(bpe.vocab))}
    tokenizer = BPEIDTokenizer(bpe, symbol_to_id)
    tokenizer_path = directory / "tokenizer.json"
    save_bpe_id_tokenizer_json(tokenizer, tokenizer_path)

    config = {**PRODUCTION_CONFIG, "vocabulary_size": tokenizer.vocab_size}
    revisions: dict[str, dict[str, bytes]] = {}
    for seed, (revision, key_value_heads) in enumerate(
        (("main", None), ("gqa", GQA_KEY_VALUE_HEADS), ("sft-synthetic", None))
    ):
        revision_config = dict(config)
        if key_value_heads is not None:
            revision_config["num_key_value_heads"] = key_value_heads
        torch.manual_seed(seed)
        model = build_model(revision_config)
        checkpoint_path = directory / f"{revision}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        revisions[revision] = {
            "config.json": json.dumps(revision_config).encode("utf-8"),
            "model_state.pt": checkpoint_path.read_bytes(),
        }
    cache_dir = directory / "hub"
    _write_repository(cache_dir, MODEL_REPO_ID, revisions)
    _write_repository(
        cache_dir, TOKENIZER_REPO_ID, {"main": {"tokenizer.json": tokenizer_path.read_bytes()}}
    )
    return cache_dir
