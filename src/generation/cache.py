"""KV キャッシュ(KV Cache)のスクラッチ実装。

自己回帰生成(autoregressive generation)では、ステップ t で追加される Key / Value は
それ以前のステップの計算結果と完全に同じ値になる(RoPE のような位置依存の変換も、
各トークンの絶対位置だけで決まるため、後続ステップで系列が伸びても値は変わらない)。
``KeyValueCache`` は、層ごとにこの Key / Value を保持し、新しいステップの Key / Value を
追記しながら「これまでの全ステップ分」を返す。

理論的な導出・メモリ量の閉形式は
``theories/03_efficient_training/010_kv_cache_and_inference_compute.ipynb`` の
理論セクションを参照。

記号 / Notation:
    B : バッチサイズ
    L : 層数(num_layers)
    g : Key / Value ヘッド数(num_key_value_heads。多頭注意機構では h と同じ)
    T : 現在キャッシュされている系列長
    d_k : ヘッドあたりの次元
"""

from __future__ import annotations

import torch
from torch import Tensor


class KeyValueCache:
    """層ごとの Key / Value を保持し、追記(append)と取得を行うキャッシュ。

    状態(過去の Key / Value)をレイヤー(``MultiHeadAttention``)の外に出す設計にしている
    理由は 2 つある。

    1. 条件・シードをまたいで同じレイヤーインスタンスを使い回す実験(実験 A〜D)では、
       レイヤー自身が状態を持つと、ある条件の計測が次の条件に漏れる(前の生成の
       Key / Value が残ったまま次の計測を始めてしまう)おそれがある。キャッシュを
       呼び出し側が明示的に生成・破棄することで、この漏れを構造的に防ぐ。
    2. メモリ量(``memory_bytes``)を集計するには、全層の Key / Value を一箇所で
       把握できる必要がある。レイヤーごとに状態を分散させると、集計のために
       モデル全体を走査しなければならない。

    Args:
        num_layers: 層数 L。
    """

    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers は 1 以上である必要がある: {num_layers}")
        self.num_layers = num_layers
        self._keys: list[Tensor | None] = [None] * num_layers
        self._values: list[Tensor | None] = [None] * num_layers

    @property
    def length(self) -> int:
        """現在キャッシュされている系列長 T。まだ何も追記していない場合は 0。"""
        first_key = self._keys[0]
        return 0 if first_key is None else first_key.size(-2)

    def update(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """層 ``layer_idx`` に新しいステップの Key / Value を追記し、これまでの
        全ステップ分(過去 + 今回)を返す。

        Args:
            layer_idx: 層番号(0-indexed)。
            key: 形状 ``(B, g, S_new, d_k)`` の新しい Key。
            value: 形状 ``(B, g, S_new, d_k)`` の新しい Value(``key`` と同じ形状)。

        Returns:
            (full_key, full_value) のタプル。いずれも形状 ``(B, g, T, d_k)``
            (``T`` はこの呼び出し後の ``length``)。
        """
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx が範囲外: {layer_idx} (num_layers={self.num_layers})")

        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = key
            self._values[layer_idx] = value
        else:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key], dim=-2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value], dim=-2)
        return self._keys[layer_idx], self._values[layer_idx]

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """層 ``layer_idx`` の現在の Key / Value を返す(まだ空なら ``None``)。"""
        key = self._keys[layer_idx]
        if key is None:
            return None
        return key, self._values[layer_idx]

    def memory_bytes(self) -> int:
        """現在キャッシュしている全層の Key / Value の合計バイト数を実測する。

        閉形式 ``2 * B * L * T * (g * d_k) * bytes_per_element``
        (``compute_key_value_cache_memory_bytes``、``src/utils/statistics.py``)との
        一致を検証する対照として使う。
        """
        total = 0
        for key, value in zip(self._keys, self._values, strict=True):
            if key is None:
                continue
            total += key.numel() * key.element_size()
            total += value.numel() * value.element_size()
        return total
