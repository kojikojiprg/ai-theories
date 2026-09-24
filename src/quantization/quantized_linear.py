"""量子化した重みを保持する線形層(``QuantizedLinear``)。

013(量子化の基礎)で導入する。パック済みの符号とスケールを buffer として保持し
(学習可能なパラメータとしては持たない。重みは凍結される)、順伝播のたびに FP32 へ
逆量子化してから行列積を行う。

    y = x dequant(W_q)^T + bias

**融合カーネル(逆量子化と行列積を 1 つのカーネルで行うもの)は使わない。** 順伝播ごとに
逆量子化した FP32 の重み全体をメモリ上に作るため、この実装は元の ``nn.Linear`` より
速くならない(重みのみ量子化が decode を速くする理由は 013 の理論セクションで扱う)。
目的は、量子化による精度への影響と、格納するバイト数を測ることである。

QLoRA(Dettmers et al., NeurIPS 2023)の計算データ型は BF16 だが、本実装は FP32 で
逆量子化・行列積を行う(Google Colab の T4 は BF16 の行列積をハードウェアで持たないため)。

012 の ``LoRALinear`` のベース層として使える(``in_features``・``out_features`` 属性を持ち、
パラメータを持たない場合は FP32 の buffer から LoRA の行列の device・dtype が決まる)。
勾配は逆量子化後の重みを経由して入力側(と LoRA の行列)へ流れ、符号・スケール自体には
流れない。

使い方 / Usage::

    from src.quantization.quantized_linear import quantize_linear_layers

    names = quantize_linear_layers(
        model.blocks,
        method="nf4",
        bits=4,
        granularity="block",
        block_size=64,
        double_quantization=True,
    )
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.quantization.quantize import (
    DoubleQuantizedScale,
    QuantizedWeight,
    compute_nf4_codebook,
    quantize_weight,
)

_STORAGE_PREFIX = "quantized_"


class QuantizedLinear(nn.Module):
    """量子化した重みで順伝播する線形層(重みは凍結、buffer として保持する)。

    Args:
        quantized_weight: 量子化した重み(形状 ``(out_features, in_features)``)。
        bias: バイアス(``None`` 可)。持つ場合は ``requires_grad=False`` の FP32 パラメータ
            として保持する。
    """

    def __init__(self, quantized_weight: QuantizedWeight, bias: Tensor | None = None) -> None:
        super().__init__()
        self.out_features, self.in_features = quantized_weight.shape
        self._metadata = {
            "method": quantized_weight.method,
            "bits": quantized_weight.bits,
            "granularity": quantized_weight.granularity,
            "block_size": quantized_weight.block_size,
            "shape": quantized_weight.shape,
            "double_quantization_block_size": (
                quantized_weight.double_quantized_scale.block_size
                if quantized_weight.double_quantized_scale is not None
                else None
            ),
        }
        self._storage_names = []
        for name, tensor in quantized_weight.storage_tensors().items():
            self.register_buffer(_STORAGE_PREFIX + name, tensor.clone())
            self._storage_names.append(name)
        # NF4 の符号語は全行列で共有する定数であり、格納バイト数に含めない(state_dict にも
        # 保存しない)。device の移動に追従させるため non-persistent な buffer にする。
        if quantized_weight.method == "nf4":
            self.register_buffer("codebook", compute_nf4_codebook(), persistent=False)
        else:
            self.codebook = None
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, **quantize_kwargs) -> QuantizedLinear:
        """``nn.Linear`` の重みを ``quantize_weight(**quantize_kwargs)`` で量子化して作る。"""
        quantized = quantize_weight(linear.weight.detach().cpu(), **quantize_kwargs)
        layer = cls(quantized, linear.bias)
        return layer.to(linear.weight.device)

    def _storage(self, name: str) -> Tensor | None:
        return getattr(self, _STORAGE_PREFIX + name, None) if name in self._storage_names else None

    def quantized_weight(self) -> QuantizedWeight:
        """現在の buffer(device 上)から ``QuantizedWeight`` を組み立てる。"""
        double_quantized = None
        if self._metadata["double_quantization_block_size"] is not None:
            double_quantized = DoubleQuantizedScale(
                codes=self._storage("double_quantized_scale_codes"),
                scale=self._storage("double_quantized_scale_scale"),
                offset=self._storage("double_quantized_scale_offset"),
                block_size=self._metadata["double_quantization_block_size"],
            )
        return QuantizedWeight(
            method=self._metadata["method"],
            bits=self._metadata["bits"],
            granularity=self._metadata["granularity"],
            block_size=self._metadata["block_size"],
            shape=self._metadata["shape"],
            codes=self._storage("codes"),
            scale=self._storage("scale"),
            zero_point=self._storage("zero_point"),
            double_quantized_scale=double_quantized,
        )

    def dequantized_weight(self) -> Tensor:
        """逆量子化した FP32 の重み(形状 ``(out_features, in_features)``)。"""
        return self.quantized_weight().dequantize(codebook=self.codebook)

    def forward(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.dequantized_weight().to(x.dtype), self.bias)

    def storage_bytes(self) -> int:
        """重みの格納バイト数の実測値(符号・スケール。符号語・バイアスは含めない)。"""
        return sum(
            t.numel() * t.element_size() for t in (self._storage(n) for n in self._storage_names)
        )

    def extra_repr(self) -> str:
        meta = self._metadata
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"method={meta['method']}, bits={meta['bits']}, granularity={meta['granularity']}, "
            f"block_size={meta['block_size']}, "
            f"double_quantization_block_size={meta['double_quantization_block_size']}"
        )


def quantize_linear_layers(
    module: nn.Module,
    precomputed: Mapping[str, QuantizedWeight] | None = None,
    **quantize_kwargs,
) -> list[str]:
    """``module`` 内のすべての ``nn.Linear`` を ``QuantizedLinear`` に置き換える(その場で)。

    Args:
        module: 置換対象(例: ``GPTLanguageModel.blocks``。埋め込みと重みを共有する出力層を
            量子化しないよう、呼び出し側が範囲を渡す)。
        precomputed: 完全修飾名(``module`` からの相対名)から量子化済みの重みへの対応。
            指定した場合は量子化をやり直さずに使う(条件・シードをまたぐキャッシュ)。
        **quantize_kwargs: ``quantize_weight`` に渡す引数(``precomputed`` がない層に使う)。

    Returns:
        置換した層の完全修飾名(``module`` からの相対名)のリスト。
    """
    names = [name for name, child in module.named_modules() if isinstance(child, nn.Linear)]
    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        linear = getattr(parent, attr)
        if precomputed is not None and name in precomputed:
            layer = QuantizedLinear(precomputed[name], linear.bias).to(linear.weight.device)
        else:
            layer = QuantizedLinear.from_linear(linear, **quantize_kwargs)
        setattr(parent, attr, layer)
    return names


@torch.no_grad()
def quantize_linear_weights(module: nn.Module, **quantize_kwargs) -> dict[str, QuantizedWeight]:
    """``module`` 内のすべての ``nn.Linear`` の重みを量子化した結果を CPU 上で返す
    (``quantize_linear_layers(precomputed=...)`` に渡すキャッシュを作る)。"""
    return {
        name: quantize_weight(child.weight.detach().cpu(), **quantize_kwargs)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    }
