"""theories/・apps/ から再利用する学習ループ。"""

from src.training.trainer import evaluate_bits_per_byte, train_language_model

__all__ = [
    "evaluate_bits_per_byte",
    "train_language_model",
]
