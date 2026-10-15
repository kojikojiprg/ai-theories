"""Hub からの読み込み(3 ブランチを同時にメモリへ載せる)。"""

from __future__ import annotations

from lab.core.artifacts import Resources


def test_three_branches_are_loaded(resources: Resources) -> None:
    assert set(resources.models) == {"main", "gqa", "sft-synthetic"}


def test_all_models_are_on_the_selected_device(resources: Resources) -> None:
    for loaded in resources.models.values():
        assert next(loaded.model.parameters()).device.type == resources.device.type


def test_all_models_are_in_eval_mode(resources: Resources) -> None:
    assert all(not loaded.model.training for loaded in resources.models.values())


def test_gqa_has_fewer_key_value_heads_and_a_smaller_cache(resources: Resources) -> None:
    main, gqa = resources.models["main"], resources.models["gqa"]
    assert gqa.model.num_key_value_heads < main.model.num_key_value_heads
    assert gqa.key_value_cache_bytes_per_token < main.key_value_cache_bytes_per_token


def test_tied_embeddings_are_preserved(resources: Resources) -> None:
    model = resources.models["main"].model
    assert model.token_embedding.weight is model.lm_head.weight
