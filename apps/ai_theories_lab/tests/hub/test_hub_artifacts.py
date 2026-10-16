"""本物のアーティファクトが、ノートブックの記録と一致すること。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from lab.core.artifacts import MODEL_REPO_ID, Resources
from tests.constants import EXPECTED_SHA256, GQA_KEY_VALUE_HEADS, PRODUCTION_CONFIG


@pytest.mark.parametrize("key", ["main", "gqa", "sft-synthetic"])
def test_config_matches_the_production_config_of_010(hub_resources: Resources, key: str) -> None:
    expected = dict(PRODUCTION_CONFIG)
    if key == "gqa":
        expected["num_key_value_heads"] = GQA_KEY_VALUE_HEADS
    config = hub_resources.models[key].config
    assert all(config.get(name) == value for name, value in expected.items())


@pytest.mark.parametrize("key", sorted(EXPECTED_SHA256))
def test_checkpoint_sha256_matches_the_notebook_record(key: str) -> None:
    path = hf_hub_download(MODEL_REPO_ID, "model_state.pt", revision=key)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == EXPECTED_SHA256[key]


def test_tokenizer_vocabulary_size_is_8192(hub_resources: Resources) -> None:
    assert hub_resources.tokenizer.vocab_size == PRODUCTION_CONFIG["vocabulary_size"]
