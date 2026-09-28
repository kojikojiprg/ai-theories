"""汎用ユーティリティ(utilities)。

``from src.utils import compute_fertility`` のように、``statistics``・``visualization`` の関数を
パッケージから直接 import できる(001〜005 のノートブックが使う形)。ただし、これらの
サブモジュールは **名前が参照されたときに初めて読み込む**(PEP 562 のモジュールの
``__getattr__``)。パッケージの初期化で numpy・torch・matplotlib を読み込まないためである。
``src.utils.environment`` の ``check_preloaded_package_versions()`` は、セットアップセルで torch
などを読み込む前に呼ぶので、``src.utils`` の初期化がそれらを読み込んではならない。
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRIBUTES: dict[str, str] = {
    "compute_always_negative_unit_ratio": "src.utils.statistics",
    "compute_bits_per_byte": "src.utils.statistics",
    "compute_character_coverage": "src.utils.statistics",
    "compute_chunk_length_statistics": "src.utils.statistics",
    "compute_exact_match_rate": "src.utils.statistics",
    "compute_fertility": "src.utils.statistics",
    "compute_gradient_norm_by_unit_group": "src.utils.statistics",
    "compute_gradient_norm_per_layer": "src.utils.statistics",
    "compute_mean_to_rms_ratio": "src.utils.statistics",
    "compute_perplexity": "src.utils.statistics",
    "compute_segmentation_agreement_rate": "src.utils.statistics",
    "compute_unknown_rate": "src.utils.statistics",
    "count_non_embedding_parameters": "src.utils.statistics",
    "plot_attention_heatmap": "src.utils.visualization",
    "plot_bar_by_layer": "src.utils.visualization",
    "plot_dual_axis_curves": "src.utils.visualization",
    "plot_function_curves": "src.utils.visualization",
    "plot_grouped_bar": "src.utils.visualization",
    "plot_learning_curves": "src.utils.visualization",
    "plot_learning_curves_multi_seed": "src.utils.visualization",
    "plot_multi_head_attention": "src.utils.visualization",
    "plot_seed_scatter": "src.utils.visualization",
}


def __getattr__(name: str) -> Any:
    """``_LAZY_ATTRIBUTES`` の名前が参照されたときに、対応するサブモジュールを読み込んで返す。"""
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # 2 回目以降は通常の属性として見つかる
    return value


__all__ = [
    "compute_always_negative_unit_ratio",
    "compute_bits_per_byte",
    "compute_character_coverage",
    "compute_chunk_length_statistics",
    "compute_exact_match_rate",
    "compute_fertility",
    "compute_gradient_norm_by_unit_group",
    "compute_gradient_norm_per_layer",
    "compute_mean_to_rms_ratio",
    "compute_perplexity",
    "compute_segmentation_agreement_rate",
    "compute_unknown_rate",
    "count_non_embedding_parameters",
    "plot_attention_heatmap",
    "plot_bar_by_layer",
    "plot_dual_axis_curves",
    "plot_function_curves",
    "plot_grouped_bar",
    "plot_learning_curves",
    "plot_learning_curves_multi_seed",
    "plot_multi_head_attention",
    "plot_seed_scatter",
]
