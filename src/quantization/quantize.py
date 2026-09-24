"""重みの量子化(Quantization)・逆量子化(Dequantization)のスクラッチ実装。

013(量子化の基礎)で導入する。次の 3 方式を、テンソル単位・チャネル単位・ブロック単位の
粒度で扱う。

- absmax 方式(対称、Symmetric): 刻み幅 Delta = max|w| / (2^(b-1) - 1)、
  符号 q = round(w / Delta) in [-(2^(b-1) - 1), 2^(b-1) - 1]、逆量子化 w_hat = q Delta。
  Dettmers et al., "QLoRA", NeurIPS 2023 の式 (1)(INT8 では [-127, 127])と同じ。
- ゼロ点(Zero-Point)付きの非対称方式(Asymmetric): Jacob et al., "Quantization and
  Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018
  の r = S (q - Z)。値域 [min(w_min, 0), max(w_max, 0)] を 2^b - 1 等分し
  (0 を厳密に表現できるよう値域に 0 を含める)、Z = round(-w_min / S) とする。
- NF4(4-bit NormalFloat): QLoRA の分位点量子化。ブロックごとに max|w| で [-1, 1] に
  正規化し、16 個の符号語のうち最も近いものに丸める。符号語は ``compute_nf4_codebook()``
  で標準正規分布の分位点から導出する。

二重量子化(Double Quantization、QLoRA 第 3 節)は、NF4 のブロックごとのスケール
(max|w|、FP32)をさらに量子化する。原論文は第 2 段に 8 ビット浮動小数点(FP8)を使うが、
本実装は第 2 段を **8 ビットの absmax 方式(INT8)** で量子化する(平均を引いてから
ブロック単位で量子化する手順と、ビット数・ブロックサイズは原論文と同じ)。格納バイト数は
第 2 段の形式によらず同じである。

4 ビットの符号は 2 個を 1 つの ``uint8`` に詰めて格納する(``pack_4bit``)。8 ビットの
符号は 1 個を 1 つの ``uint8`` に格納する。それ以外のビット幅(2, 3, 5, 6, 7)は擬似量子化
(量子化 -> 逆量子化による誤差の測定)専用であり、1 符号 1 バイトの非パック格納とする
(格納バイト数の閉形式は主張しない)。

ブロック単位の粒度では、テンソルを行優先(row-major)で 1 次元に平坦化し、先頭から
``block_size`` 要素ずつ区切る(QLoRA 第 2 節と同じ)。要素数が ``block_size`` で割り切れない
場合、最後のブロックは短い(統計量の計算では、最後のブロックの末尾要素を複製して詰めるため、
max・min は変わらない)。

記号 / Notation:
    b      : ビット幅
    N      : テンソルの要素数
    B      : ブロックサイズ(第 1 段)
    B_2    : 二重量子化の第 2 段のブロックサイズ
    G      : スケールのグループ数(テンソル単位 1、チャネル単位 d_out、ブロック単位 ceil(N / B))
    Delta  : 量子化の刻み幅(``scale``)

使い方 / Usage::

    from src.quantization.quantize import quantize_weight

    q = quantize_weight(
        w, method="nf4", bits=4, granularity="block", block_size=64, double_quantization=True
    )
    w_hat = q.dequantize()  # 形状は w と同じ、FP32
    q.storage_bytes()  # 実際に格納しているテンソルのバイト数
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Literal

import torch
from torch import Tensor

Method = Literal["absmax", "zero_point", "nf4"]
Granularity = Literal["tensor", "channel", "block"]

# NF4 の 16 個の符号語の参照値(一次情報からの転記。記憶からではない)。
# 出典 1: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023,
#         arXiv:2305.14314v1, Appendix E "NormalFloat 4-bit data type"。
# 出典 2: bitsandbytes, bitsandbytes/functional.py の get_4bit_type("nf4")
#         (https://github.com/bitsandbytes-foundation/bitsandbytes、
#         commit 833649043474794b8fe7a4136e0c40faf077b2e0 で確認)。
# 2 つの出典の値は完全に一致する。
NF4_CODEBOOK_REFERENCE: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

# NF4 の最近傍の符号語を決める 15 個の閾値の参照値(一次情報からの転記)。
# 出典: bitsandbytes, csrc/kernels.cu の dQuantizeNF4
# (commit 833649043474794b8fe7a4136e0c40faf077b2e0)。
# 隣り合う符号語(FP32)の FP64 での中点であり、x > 閾値 なら上側の符号語に割り当てる
# (中点ちょうどの値は下側の符号語になる)。昇順に並べ替えて転記した。
NF4_THRESHOLDS_REFERENCE: tuple[float, ...] = (
    -0.8480964004993439,
    -0.6106329262256622,
    -0.4599952697753906,
    -0.33967943489551544,
    -0.23460740596055984,
    -0.13791173323988914,
    -0.045525018125772476,
    0.03979014977812767,
    0.1202552504837513,
    0.2035212516784668,
    0.2920137718319893,
    0.3893125355243683,
    0.5016634166240692,
    0.6427869200706482,
    0.8614784181118011,
)

# NF4 の最も外側の分位点の確率。bitsandbytes の create_normal_map の既定値
# (offset=0.9677083)をそのまま使う。閉形式の値 compute_nf4_offset() = 0.96770833... を
# 小数第 7 位で丸めたものであり、参照値 NF4_CODEBOOK_REFERENCE はこの丸めた値から
# 生成されている(丸める前の値を使うと、FP32 で最大 3e-7 ずれる)。
NF4_OFFSET = 0.9677083

# 二重量子化の既定値(QLoRA 第 3 節: 第 2 段は 8 ビット、ブロックサイズ 256)
DOUBLE_QUANTIZATION_BITS = 8
DOUBLE_QUANTIZATION_BLOCK_SIZE = 256


def compute_nf4_offset(num_bits: int = 4) -> float:
    """NF4 の最も外側の分位点の確率 delta を返す。

    QLoRA 式 (4) は、2^k + 1 個の等間隔な確率点 i / (2^k + 1) の分位点の中点を符号語とする。
    負側は 2^(k-1) 個・正側は 2^(k-1) + 1 個の符号語を別々に作るため、最も外側の確率点は
    負側で 1 - 1 / (2 (2^k - 1))、正側で 1 - 1 / (2 * 2^k) となる。bitsandbytes の
    ``create_normal_map`` の既定値 ``offset=0.9677083`` はこの 2 つの平均
    (k = 4: (1 - 1/30 + 1 - 1/32) / 2 = 0.96770833...)を小数第 7 位で丸めたものである。
    本関数は丸める前の閉形式の値を返す(符号語の導出には丸めた ``NF4_OFFSET`` を使う)。
    """
    levels = 2**num_bits
    return 0.5 * ((1.0 - 1.0 / (2 * (levels - 1))) + (1.0 - 1.0 / (2 * levels)))


def compute_nf4_codebook(offset: float | None = None) -> Tensor:
    """NF4 の 16 個の符号語を、標準正規分布の分位点から導出する(QLoRA 式 (4)・第 3 節)。

    手順(bitsandbytes の ``create_normal_map(offset, use_extra_value=True)`` と同じ):

    1. 正側: 確率 [offset, 0.5] を 9 点に等分し、端点 0.5 を除く 8 点の分位点
       Q(p)(Q は標準正規分布の分位点関数)を取る(8 個の正の値)。
    2. 負側: 確率 [offset, 0.5] を 8 点に等分し、端点 0.5 を除く 7 点について -Q(p) を取る
       (7 個の負の値)。
    3. 0 を 1 個加え(正負の両方に現れる 0 のうち 1 つを除いた結果、0 を厳密に表現できる)、
       昇順に並べ、最大値(= 正側の最大値)で割って [-1, 1] に正規化する。

    分位点関数は ``torch.special.ndtri``(FP64)で計算する。確率点の等分は bitsandbytes と
    同じく ``torch.linspace`` の FP32 で行い、最後に FP32 へ丸めてから正規化する
    (参照値 ``NF4_CODEBOOK_REFERENCE`` と bit 単位で比較するため、丸めの順序を揃える)。

    Args:
        offset: 最も外側の分位点の確率。``None`` の場合は ``NF4_OFFSET``。

    Returns:
        形状 ``(16,)`` の FP32 テンソル(昇順)。
    """
    offset = NF4_OFFSET if offset is None else offset
    positive_probabilities = torch.linspace(offset, 0.5, 9)[:-1].to(torch.float64)
    negative_probabilities = torch.linspace(offset, 0.5, 8)[:-1].to(torch.float64)
    positive = torch.special.ndtri(positive_probabilities)
    negative = -torch.special.ndtri(negative_probabilities)
    values = torch.cat([positive, torch.zeros(1, dtype=torch.float64), negative]).to(torch.float32)
    values = values.sort().values
    return values / values.max()


def pack_4bit(codes: Tensor) -> Tensor:
    """0〜15 の符号(``uint8``、1 次元)を 2 個ずつ 1 バイトに詰める。

    偶数番目の要素を下位 4 ビット、奇数番目の要素を上位 4 ビットに置く。要素数が奇数の場合は
    末尾に 0 を 1 個補う(``unpack_4bit`` に元の要素数を渡して取り除く)。
    """
    if codes.dtype != torch.uint8 or codes.dim() != 1:
        raise ValueError("codes は 1 次元の uint8 テンソルである必要がある")
    if codes.numel() and int(codes.max()) > 15:
        raise ValueError("4 ビットの符号は 0〜15 である必要がある")
    if codes.numel() % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    return codes[0::2] | (codes[1::2] << 4)


def unpack_4bit(packed: Tensor, num_elements: int) -> Tensor:
    """``pack_4bit`` の逆変換。先頭 ``num_elements`` 個の符号(``uint8``)を返す。"""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack([low, high], dim=1).reshape(-1)[:num_elements]


def _num_groups(shape: tuple[int, ...], granularity: Granularity, block_size: int | None) -> int:
    num_elements = math.prod(shape)
    if granularity == "tensor":
        return 1
    if granularity == "channel":
        return shape[0]
    if granularity == "block":
        if block_size is None or block_size < 1:
            raise ValueError("ブロック単位の粒度には 1 以上の block_size が必要である")
        return -(-num_elements // block_size)
    raise ValueError(f"未知の粒度: {granularity!r}")


def _group_length(shape: tuple[int, ...], granularity: Granularity, block_size: int | None) -> int:
    num_elements = math.prod(shape)
    if granularity == "tensor":
        return num_elements
    if granularity == "channel":
        return num_elements // shape[0]
    return block_size


def _grouped(flat: Tensor, group_length: int) -> Tensor:
    """平坦化したテンソルを先頭から ``group_length`` 要素ずつ区切り、(G, group_length) に並べる。

    要素数が ``group_length`` で割り切れない場合は、末尾要素を複製して最後のグループを詰める
    (同じグループに既にある値の複製なので、そのグループの max・min は変わらない)。
    """
    num_groups = -(-flat.numel() // group_length)
    pad = group_length * num_groups - flat.numel()
    if pad:
        flat = torch.cat([flat, flat[-1:].expand(pad)])
    return flat.view(num_groups, group_length)


@dataclass
class DoubleQuantizedScale:
    """二重量子化したスケール(QLoRA 第 3 節)。

    第 1 段のスケール c(正の FP32、G 個)から平均 mu を引き、``block_size`` 個ずつの
    ブロックで 8 ビットの absmax 方式により量子化する。逆量子化は
    c_hat = q_2 Delta_2 + mu(Delta_2 は第 2 段のブロックごとの刻み幅)。

    Attributes:
        codes: 形状 ``(G,)`` の ``uint8``(符号 q_2 + 127)。
        scale: 形状 ``(ceil(G / block_size),)`` の FP32(第 2 段の刻み幅 Delta_2)。
        offset: 形状 ``()`` の FP32(平均 mu)。
        block_size: 第 2 段のブロックサイズ B_2。
    """

    codes: Tensor
    scale: Tensor
    offset: Tensor
    block_size: int

    def dequantize(self) -> Tensor:
        num_groups = self.codes.numel()
        levels = 2 ** (DOUBLE_QUANTIZATION_BITS - 1) - 1
        grouped = _grouped(self.codes.to(torch.float32) - levels, self.block_size)
        values = (grouped * self.scale[:, None]).reshape(-1)[:num_groups]
        return values + self.offset

    def storage_tensors(self) -> dict[str, Tensor]:
        return {"codes": self.codes, "scale": self.scale, "offset": self.offset}


def double_quantize_scale(
    scale: Tensor, block_size: int = DOUBLE_QUANTIZATION_BLOCK_SIZE
) -> DoubleQuantizedScale:
    """第 1 段のスケールを二重量子化する(平均を引いてから 8 ビット absmax 方式)。"""
    scale = scale.to(torch.float32)
    offset = scale.mean()
    centered = scale - offset
    grouped = _grouped(centered, block_size)
    levels = 2 ** (DOUBLE_QUANTIZATION_BITS - 1) - 1
    step = grouped.abs().amax(dim=1) / levels
    safe_step = torch.where(step > 0, step, torch.ones_like(step))
    codes = torch.round(grouped / safe_step[:, None]).clamp(-levels, levels)
    codes = (codes + levels).to(torch.uint8).reshape(-1)[: centered.numel()]
    return DoubleQuantizedScale(codes=codes, scale=step, offset=offset, block_size=block_size)


@dataclass
class QuantizedWeight:
    """量子化した重み(符号・スケールなど、格納するテンソル一式とメタデータ)。

    Attributes:
        method: ``"absmax"``・``"zero_point"``・``"nf4"``。
        bits: ビット幅 b(NF4 は 4)。
        granularity: ``"tensor"``・``"channel"``・``"block"``。
        block_size: ブロック単位のときのブロックサイズ B(それ以外は ``None``)。
        shape: 元のテンソルの形状。
        codes: 符号(``uint8``)。b = 4 のときパック済み(N / 2 バイト)、それ以外は N バイト。
        scale: FP32 のスケール(形状 ``(G,)``)。absmax・ゼロ点方式では刻み幅 Delta、NF4 では
            ブロックの max|w|。二重量子化した場合は ``None``。
        zero_point: ゼロ点方式のゼロ点 Z(``uint8``、形状 ``(G,)``)。それ以外は ``None``。
        double_quantized_scale: 二重量子化したスケール(``scale`` の代わりに格納する)。
    """

    method: Method
    bits: int
    granularity: Granularity
    block_size: int | None
    shape: tuple[int, ...]
    codes: Tensor
    scale: Tensor | None
    zero_point: Tensor | None = None
    double_quantized_scale: DoubleQuantizedScale | None = None

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def packed(self) -> bool:
        return self.bits == 4

    @property
    def num_groups(self) -> int:
        return _num_groups(self.shape, self.granularity, self.block_size)

    def unpacked_codes(self) -> Tensor:
        if self.packed:
            return unpack_4bit(self.codes, self.num_elements)
        return self.codes

    def group_scale(self) -> Tensor:
        """各グループのスケール(二重量子化した場合は逆量子化した値)を FP32 で返す。"""
        if self.double_quantized_scale is not None:
            return self.double_quantized_scale.dequantize()
        return self.scale

    def dequantize(self, codebook: Tensor | None = None) -> Tensor:
        """逆量子化した FP32 テンソル(形状 ``shape``)を返す。

        Args:
            codebook: NF4 の符号語(``None`` の場合は ``compute_nf4_codebook()``)。
                ``QuantizedLinear`` はデバイス上に置いた符号語を渡して再計算を避ける。
        """
        codes = self.unpacked_codes()
        scale = self.group_scale()
        if self.method == "nf4":
            if codebook is None:
                codebook = compute_nf4_codebook().to(codes.device)
            normalized = codebook[codes.long()]
        elif self.method == "absmax":
            normalized = codes.to(torch.float32) - (2 ** (self.bits - 1))
        elif self.method == "zero_point":
            normalized = codes.to(torch.float32)
        else:
            raise ValueError(f"未知の方式: {self.method!r}")
        grouped = _grouped(normalized, _group_length(self.shape, self.granularity, self.block_size))
        if self.method == "zero_point":
            grouped = grouped - self.zero_point.to(torch.float32)[:, None]
        values = (grouped * scale[:, None]).reshape(-1)[: self.num_elements]
        return values.view(self.shape)

    def storage_tensors(self) -> dict[str, Tensor]:
        """格納するテンソルの一覧(NF4 の符号語は全行列で共有する定数のため含めない)。"""
        tensors = {"codes": self.codes}
        if self.scale is not None:
            tensors["scale"] = self.scale
        if self.zero_point is not None:
            tensors["zero_point"] = self.zero_point
        if self.double_quantized_scale is not None:
            for key, value in self.double_quantized_scale.storage_tensors().items():
                tensors[f"double_quantized_scale_{key}"] = value
        return tensors

    def storage_bytes(self) -> int:
        """実際に格納しているテンソルのバイト数(実測値、``numel * element_size`` の総和)。"""
        return sum(t.numel() * t.element_size() for t in self.storage_tensors().values())

    def to(self, device: torch.device | str) -> QuantizedWeight:
        moved = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                moved[field.name] = value.to(device)
            elif isinstance(value, DoubleQuantizedScale):
                moved[field.name] = replace(
                    value,
                    codes=value.codes.to(device),
                    scale=value.scale.to(device),
                    offset=value.offset.to(device),
                )
        return replace(self, **moved)


def quantize_weight(
    weight: Tensor,
    method: Method = "absmax",
    bits: int = 8,
    granularity: Granularity = "tensor",
    block_size: int | None = None,
    double_quantization: bool = False,
    double_quantization_block_size: int = DOUBLE_QUANTIZATION_BLOCK_SIZE,
) -> QuantizedWeight:
    """重みを量子化する。

    Args:
        weight: 量子化するテンソル(チャネル単位の粒度では第 0 次元を出力チャネルとみなす。
            ``nn.Linear.weight`` の形状 ``(d_out, d_in)`` では各行が 1 チャネル)。
        method: ``"absmax"``(対称)・``"zero_point"``(非対称)・``"nf4"``。
        bits: ビット幅 b(2〜8)。``method="nf4"`` では 4 のみ。
        granularity: ``"tensor"``・``"channel"``・``"block"``。
        block_size: ブロック単位のときのブロックサイズ B。
        double_quantization: True の場合、スケールを二重量子化する(``method="nf4"`` のみ)。
        double_quantization_block_size: 二重量子化の第 2 段のブロックサイズ B_2。

    Returns:
        ``QuantizedWeight``。符号化・統計量の計算は FP32 で行う。
    """
    if not 2 <= bits <= 8:
        raise ValueError(f"bits は 2〜8 である必要がある: {bits}")
    if method == "nf4" and bits != 4:
        raise ValueError("NF4 は 4 ビットのみ")
    if double_quantization and method != "nf4":
        raise ValueError("二重量子化は NF4 のスケール(max|w|)にのみ対応する")
    if granularity != "block":
        block_size = None

    shape = tuple(weight.shape)
    flat = weight.detach().to(torch.float32).reshape(-1)
    grouped = _grouped(flat, _group_length(shape, granularity, block_size))
    zero_point = None

    if method == "absmax":
        levels = 2 ** (bits - 1) - 1
        scale = grouped.abs().amax(dim=1) / levels
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(grouped / safe_scale[:, None]).clamp(-levels, levels)
        # 符号 q in [-levels, levels] を q + 2^(b-1) in [1, 2^b - 1] として符号なしで格納する
        codes = codes + 2 ** (bits - 1)
    elif method == "zero_point":
        levels = 2**bits - 1
        low = grouped.amin(dim=1).clamp(max=0.0)
        high = grouped.amax(dim=1).clamp(min=0.0)
        scale = (high - low) / levels
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.round(-low / safe_scale).clamp(0, levels)
        codes = (torch.round(grouped / safe_scale[:, None]) + zero[:, None]).clamp(0, levels)
        zero_point = zero.to(torch.uint8)
    elif method == "nf4":
        scale = grouped.abs().amax(dim=1)
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalized = grouped / safe_scale[:, None]
        codes = nearest_codebook_index(normalized, compute_nf4_codebook())
    else:
        raise ValueError(f"未知の方式: {method!r}")

    codes = codes.to(torch.uint8).reshape(-1)[: flat.numel()]
    if bits == 4:
        codes = pack_4bit(codes)
    quantized = QuantizedWeight(
        method=method,
        bits=bits,
        granularity=granularity,
        block_size=block_size,
        shape=shape,
        codes=codes,
        scale=scale,
        zero_point=zero_point,
    )
    if double_quantization:
        quantized.double_quantized_scale = double_quantize_scale(
            scale, block_size=double_quantization_block_size
        )
        quantized.scale = None
    return quantized


def compute_codebook_thresholds(codebook: Tensor) -> Tensor:
    """昇順の符号語から、隣り合う符号語の中点(最近傍の境界)を返す。

    中点は FP64 で計算してから符号語の dtype に丸める(bitsandbytes の ``dQuantizeNF4`` の
    閾値と同じ値になる。FP32 のまま (a + b) / 2 を計算すると、丸めにより 1 ulp ずれることがある)。
    """
    wide = codebook.to(torch.float64)
    return ((wide[1:] + wide[:-1]) / 2).to(codebook.dtype)


def nearest_codebook_index(values: Tensor, codebook: Tensor) -> Tensor:
    """各要素に最も近い符号語の番号を返す(隣り合う符号語の中点で区切る二分探索)。

    境界の規則は bitsandbytes の ``dQuantizeNF4`` と同じで、値が閾値より大きければ上側の
    符号語に割り当てる(閾値ちょうどの値は下側の符号語になる)。要素数 N に対して
    O(N log K)(K は符号語の数)。
    """
    thresholds = compute_codebook_thresholds(codebook).to(values.device)
    return torch.bucketize(values, thresholds, right=False)


def compute_quantized_storage_bytes(
    num_elements: int,
    bits: int,
    num_groups: int,
    scale_bytes: int = 4,
    zero_point: bool = False,
    double_quantization_block_size: int | None = None,
) -> int:
    """パック格納のバイト数の閉形式(4・8 ビットのみ)。

    - 符号: b N / 8(N は偶数を想定)
    - スケール: 二重量子化なしなら scale_bytes G。二重量子化ありなら 1 G(8 ビットの符号)
      + 4 ceil(G / B_2)(第 2 段の FP32 スケール)+ 4(FP32 の平均 mu)
    - ゼロ点: 1 G(``uint8``)
    """
    if bits not in (4, 8):
        raise ValueError("パック格納の閉形式は 4・8 ビットのみ")
    total = bits * num_elements // 8
    if double_quantization_block_size is None:
        total += scale_bytes * num_groups
    else:
        total += num_groups + 4 * (-(-num_groups // double_quantization_block_size)) + 4
    if zero_point:
        total += num_groups
    return total


def compute_effective_bits_per_parameter(
    bits: int,
    block_size: int,
    scale_bits: int = 32,
    double_quantization_block_size: int | None = None,
    double_quantization_bits: int = DOUBLE_QUANTIZATION_BITS,
) -> float:
    """スケールの格納オーバーヘッドを含めた 1 パラメータあたりの実効ビット数(閉形式)。

    二重量子化なし: b + s / B。二重量子化あり: b + b_2 / B + s / (B B_2)
    (s はスケールのビット数、b_2 は第 2 段のビット数。N が B B_2 で割り切れ、テンソルごとの
    平均 mu の 32 ビットを無視できる場合の値。QLoRA 第 3 節)。
    """
    if double_quantization_block_size is None:
        return bits + scale_bits / block_size
    return (
        bits
        + double_quantization_bits / block_size
        + scale_bits / (block_size * double_quantization_block_size)
    )
