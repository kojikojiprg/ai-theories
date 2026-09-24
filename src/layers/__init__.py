"""ニューラルネットワークの層(layers)。"""

from src.layers.activation import gelu_exact, gelu_tanh_approximation, swish
from src.layers.attention import (
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask,
    scaled_dot_product_attention,
)
from src.layers.feedforward import FeedForwardNetwork, SwiGLUFeedForwardNetwork
from src.layers.flash_attention import (
    FlashAttentionFunction,
    count_block_pairs,
    flash_attention_backward,
    flash_attention_forward,
)
from src.layers.lora import LoRALinear, apply_lora, compute_lora_parameter_count
from src.layers.normalization import LayerNormalization, RMSNorm
from src.layers.positional_encoding import (
    ALiBiPositionBias,
    AttentionScoreBias,
    LearnedAbsolutePositionalEmbedding,
    QueryKeyPositionalTransform,
    RotaryPositionEmbedding,
    ShawRelativePositionBias,
    SinusoidalPositionalEncoding,
    T5RelativePositionBias,
)
from src.layers.transformer_block import DecoderBlock, EncoderBlock

__all__ = [
    "ALiBiPositionBias",
    "AttentionScoreBias",
    "DecoderBlock",
    "EncoderBlock",
    "FeedForwardNetwork",
    "FlashAttentionFunction",
    "LayerNormalization",
    "LearnedAbsolutePositionalEmbedding",
    "LoRALinear",
    "MultiHeadAttention",
    "QueryKeyPositionalTransform",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "ShawRelativePositionBias",
    "SinusoidalPositionalEncoding",
    "SwiGLUFeedForwardNetwork",
    "T5RelativePositionBias",
    "apply_lora",
    "compute_lora_parameter_count",
    "count_block_pairs",
    "create_causal_mask",
    "create_padding_mask",
    "flash_attention_backward",
    "flash_attention_forward",
    "gelu_exact",
    "gelu_tanh_approximation",
    "scaled_dot_product_attention",
    "swish",
]
