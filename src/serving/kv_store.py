"""KV キャッシュの管理(メモリの割り当て)。

推論サーバは、同時に処理する要求ごとに、過去の Key / Value を保持する場所を割り当てる。
割り当て方で、
同じメモリの量で同時に処理できる要求の数が変わる。本モジュールは 2 つの方式を実装する。

1. **最大長の分を前もって確保する方式**(``ReservedKeyValueStore``): 要求を受け入れるときに、モデルの
   最大の文脈長の分(``slot_length`` トークン)を 1 つのスロットとして確保する。
   出力長が事前に分からない
   ので、最大まで伸びる場合に備える。実際に使う長さ ``P + O - 1`` が最大より短いと、確保した枠の
   残りは使われない(**内部の断片化**)。スロットの数が上限になる。
2. **PagedAttention の方式**(``PagedKeyValueStore``、Kwon et al., SOSP 2023):
   KV キャッシュを固定長の
   **ブロック**(``block_size`` トークン)に分け、要求ごとの **ブロック表**(論理ブロック番号 → 物理
   ブロック番号)で参照する。要求が伸びるたびに必要なブロックを 1 つずつ割り当てるので、無駄は要求
   あたり最後のブロックの空きだけ(最大で ``block_size - 1`` トークン、つまりブロック 1 つ分未満)。
   物理ブロックには **参照数** を持たせ、複数の要求がブロックを共有できる(接頭辞の共有 = prefix
   caching)。共有しているブロックに書き込むときは、先に複製して自分専用にする
   (**書き込み時の複製**、copy-on-write)。メモリが足りなくなったときは、呼び出し側(``engine.py``)が
   要求を退避して(再計算)、ブロックを解放する。

どちらの方式も、物理的な置き場所は ``(first, second)`` の 2 つの添字で指す 4 階のテンソル
(``keys[layer]`` の形状 ``(A, B, g, d_k)``)にそろえてある。スロット方式では ``(スロット, 位置)``、
ブロック方式では ``(物理ブロック, ブロック内の位置)``。注意機構の側は、方式の違いを知らずに
同じ手順(添字で書き込む・添字で集める)で読み書きできる。

導入: 027。

記号 / Notation:
    L   : 層数
    g   : Key / Value ヘッド数
    d_k : ヘッドあたりの次元
    b   : ブロックの大きさ(``block_size``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.serving.requests import RequestState


@dataclass(frozen=True)
class Row:
    """1 回の forward に入れる 1 行(1 つの要求の、今回処理する部分)。

    Attributes:
        state: 要求の状態。
        query_token_ids: 今回処理するトークン(prefill では書き出しの未計算の部分、
            decode では 1 トークン)。
        start_position: ``query_token_ids[0]`` の絶対位置(RoPE の位置)。
    """

    state: RequestState
    query_token_ids: list[int]
    start_position: int


@dataclass(frozen=True)
class BatchLayout:
    """1 回の forward の入力の配置(全層で共有する)。

    形状の記号は R = 行数、Q = 今回の最長の問い合わせ、Lkv = 集めた Key / Value の長さ。

    Attributes:
        token_ids: ``(R, Q)`` の入力トークン(短い行の末尾は 0 で埋める)。
        positions: ``(R, Q)`` の絶対位置(埋めた部分は 0)。
        write_first, write_second: 新しい Key / Value の書き込み先(``keys[layer][first, second]``)。
            埋めた部分を除く、有効なトークンの数 N の 1 次元。
        write_row, write_column: 書き込む値を ``(R, Q, ...)`` から取り出す添字(長さ N)。
        gather_first, gather_second: 注意機構が集める Key / Value の添字。形状 ``(R, Lkv)``。
        last_index: 各行の、最後の有効な問い合わせの位置 ``(R,)``。
    """

    token_ids: Tensor
    positions: Tensor
    write_first: Tensor
    write_second: Tensor
    write_row: Tensor
    write_column: Tensor
    gather_first: Tensor
    gather_second: Tensor
    last_index: Tensor


@dataclass(frozen=True)
class StoreUsage:
    """ある時点のメモリの使用状況。

    Attributes:
        tokens_used: Key / Value を実際に保持しているトークン枠の数(共有ブロックは 1 回と数える)。
        tokens_allocated: 要求に割り当てているトークン枠の数(使っていない枠を含む)。
        units_in_use: 使用中のスロット数、または使用中の物理ブロック数。
    """

    tokens_used: int
    tokens_allocated: int
    units_in_use: int


class KeyValueStore(ABC):
    """KV キャッシュの置き場所と割り当ての共通の基底クラス。

    Args:
        num_layers: 層数 L。
        num_key_value_heads: Key / Value ヘッド数 g。
        head_dim: ヘッドあたりの次元 d_k。
        device: テンソルを置くデバイス。
        dtype: 保持する型。
        materialize: False のとき、テンソルを確保しない(割り当ての動作だけを見る見積もり用。
            ``build_layout`` は呼べない)。
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device | str | None,
        dtype: torch.dtype,
        materialize: bool = True,
    ) -> None:
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.materialize = materialize
        self.keys: Tensor | None = None
        self.values: Tensor | None = None
        self._active: dict[int, RequestState] = {}

    # ----- 割り当て -----

    @property
    @abstractmethod
    def capacity_tokens(self) -> int:
        """このストアが持つトークン枠の総数(予算)。"""

    @abstractmethod
    def can_admit(self, state: RequestState, num_running: int) -> bool:
        """要求 ``state`` の文脈を載せる場所を、今割り当てられるか。

        Args:
            state: 実行に入れたい要求。
            num_running: 既に実行中の要求の数(成長の余白の見積もりに使う)。
        """

    @abstractmethod
    def admit(self, state: RequestState) -> int:
        """要求の文脈を載せる場所を割り当てる(``can_admit`` が真のときだけ呼ぶ)。

        Returns:
            既に Key / Value が載っていて、計算を省けるトークンの数(先頭から数える)。
        """

    @abstractmethod
    def grow(self, state: RequestState) -> bool:
        """次の 1 トークン(位置 ``state.num_cached_tokens``)の書き込み先を用意する。

        Returns:
            用意できたら True。メモリが足りなければ False(呼び出し側が退避する)。
        """

    @abstractmethod
    def release(self, state: RequestState) -> None:
        """要求に割り当てた場所をすべて返す。"""

    def after_prefill(self, state: RequestState) -> None:  # noqa: B027
        """prefill で Key / Value を書き込んだ直後の処理(接頭辞の登録など)。既定は何もしない。"""

    @abstractmethod
    def usage(self) -> StoreUsage:
        """現在のメモリの使用状況。"""

    # ----- 添字の組み立て -----

    @abstractmethod
    def _write_indices(
        self, state: RequestState, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """位置 ``positions`` の書き込み先 ``(first, second)``。"""

    @abstractmethod
    def _gather_indices(
        self, states: Sequence[RequestState], max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """各要求の Key / Value を集める添字 ``(first, second)``。

        形状は ``(R, Lkv)``(``Lkv >= max_length``)。
        """

    def build_layout(self, rows: Sequence[Row]) -> BatchLayout:
        """行の集まりから、forward の入力の配置を作る(添字は 1 回の転送でデバイスに送る)。"""
        if not self.materialize:
            raise RuntimeError("materialize=False のストアでは配置を作れない")
        num_rows = len(rows)
        query_max = max(len(row.query_token_ids) for row in rows)
        token_ids = np.zeros((num_rows, query_max), dtype=np.int64)
        positions = np.zeros((num_rows, query_max), dtype=np.int64)
        last_index = np.empty(num_rows, dtype=np.int64)
        write_first, write_second, write_row, write_column = [], [], [], []
        max_length = 0
        for i, row in enumerate(rows):
            count = len(row.query_token_ids)
            token_ids[i, :count] = row.query_token_ids
            row_positions = row.start_position + np.arange(count, dtype=np.int64)
            positions[i, :count] = row_positions
            first, second = self._write_indices(row.state, row_positions)
            write_first.append(first)
            write_second.append(second)
            write_row.append(np.full(count, i, dtype=np.int64))
            write_column.append(np.arange(count, dtype=np.int64))
            last_index[i] = count - 1
            max_length = max(max_length, row.start_position + count)
        gather_first, gather_second = self._gather_indices([row.state for row in rows], max_length)
        arrays = [
            token_ids,
            positions,
            last_index,
            np.concatenate(write_first),
            np.concatenate(write_second),
            np.concatenate(write_row),
            np.concatenate(write_column),
            gather_first,
            gather_second,
        ]
        packed = torch.from_numpy(np.concatenate([a.ravel() for a in arrays])).to(self.device)
        views, offset = [], 0
        for array in arrays:
            views.append(packed[offset : offset + array.size].view(array.shape))
            offset += array.size
        return BatchLayout(
            token_ids=views[0],
            positions=views[1],
            last_index=views[2],
            write_first=views[3],
            write_second=views[4],
            write_row=views[5],
            write_column=views[6],
            gather_first=views[7],
            gather_second=views[8],
        )

    def _allocate_pools(self, first_dim: int, second_dim: int) -> None:
        if self.materialize:
            shape = (
                self.num_layers,
                first_dim,
                second_dim,
                self.num_key_value_heads,
                self.head_dim,
            )
            self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)


class ReservedKeyValueStore(KeyValueStore):
    """最大長の分を前もって確保する方式。

    Args:
        num_slots: スロットの数(同時に処理できる要求の数の上限)。
        slot_length: 1 スロットのトークン枠の数(モデルの最大の文脈長)。
    """

    name = "reserved"

    def __init__(
        self,
        *,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        num_slots: int,
        slot_length: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        materialize: bool = True,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            materialize=materialize,
        )
        if num_slots < 1 or slot_length < 1:
            raise ValueError("num_slots と slot_length は 1 以上である必要がある")
        self.num_slots = num_slots
        self.slot_length = slot_length
        self._free_slots = list(range(num_slots - 1, -1, -1))
        self._slot_of: dict[int, int] = {}
        self._allocate_pools(num_slots, slot_length)

    @property
    def capacity_tokens(self) -> int:
        return self.num_slots * self.slot_length

    def can_admit(self, state: RequestState, num_running: int) -> bool:
        return bool(self._free_slots)

    def admit(self, state: RequestState) -> int:
        total = state.request.total_length
        if total - 1 > self.slot_length:
            raise ValueError(f"要求の長さ {total} がスロットの長さ {self.slot_length} を超える")
        slot = self._free_slots.pop()
        self._slot_of[state.request.request_id] = slot
        self._active[state.request.request_id] = state
        return 0

    def grow(self, state: RequestState) -> bool:
        if state.num_cached_tokens >= self.slot_length:
            raise ValueError("スロットの長さを超える位置には書き込めない")
        return True

    def release(self, state: RequestState) -> None:
        slot = self._slot_of.pop(state.request.request_id)
        del self._active[state.request.request_id]
        self._free_slots.append(slot)

    def usage(self) -> StoreUsage:
        used = sum(s.num_cached_tokens for s in self._active.values())
        return StoreUsage(used, len(self._active) * self.slot_length, len(self._active))

    def _write_indices(self, state, positions):
        slot = self._slot_of[state.request.request_id]
        return np.full(positions.shape, slot, dtype=np.int64), positions

    def _gather_indices(self, states, max_length):
        slots = np.array([self._slot_of[s.request.request_id] for s in states], dtype=np.int64)
        first = np.broadcast_to(slots[:, None], (len(states), max_length))
        second = np.broadcast_to(np.arange(max_length, dtype=np.int64)[None, :], first.shape)
        return np.ascontiguousarray(first), np.ascontiguousarray(second)


class PagedKeyValueStore(KeyValueStore):
    """PagedAttention の方式(ブロック表・参照数・書き込み時の複製・接頭辞の共有)。

    Args:
        num_blocks: 物理ブロックの数(予算 = ``num_blocks * block_size`` トークン)。
        block_size: ブロックの大きさ b(トークン)。
        enable_prefix_caching: True のとき、書き出しの先頭から同じ内容が続く完全なブロックを、
            既に計算済みの物理ブロックとして共有する。共有できるのは完全に埋まったブロックだけ
            (内容が確定していて書き込まれない)。要求が終わって参照数が 0 になっても、登録済みの
            ブロックはすぐには捨てず、メモリが足りなくなったときに古いものから捨てる。
        headroom_blocks_per_running: 要求を受け入れた後に、実行中の要求 1 つあたり何ブロックの空きを
            残すか(各要求が次にブロックを必要とするときのために。退避の頻度を抑える)。実行中の
            要求がないときは適用しない。
    """

    name = "paged"

    def __init__(
        self,
        *,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        materialize: bool = True,
        enable_prefix_caching: bool = False,
        headroom_blocks_per_running: int = 0,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            materialize=materialize,
        )
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks と block_size は 1 以上である必要がある")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.headroom_blocks_per_running = headroom_blocks_per_running
        self._reference_count = [0] * num_blocks
        self._free_blocks = list(range(num_blocks - 1, -1, -1))
        self._tables: dict[int, list[int]] = {}
        # 接頭辞の共有: 先頭からそのブロックの終わりまでのトークン列 → 物理ブロック
        self._prefix_index: dict[tuple[int, ...], int] = {}
        self._registered_key: dict[int, tuple[int, ...]] = {}
        # 参照数が 0 で、登録済みのため残しているブロック(古い順)
        self._evictable: OrderedDict[int, None] = OrderedDict()
        self.num_copy_on_write = 0
        self.num_prefix_hit_blocks = 0
        self._allocate_pools(num_blocks, block_size)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    # ----- 物理ブロックの割り当て -----

    def _available_blocks(self) -> int:
        return len(self._free_blocks) + len(self._evictable)

    def _allocate_block(self) -> int | None:
        if self._free_blocks:
            block = self._free_blocks.pop()
        elif self._evictable:
            block, _ = self._evictable.popitem(last=False)
            del self._prefix_index[self._registered_key.pop(block)]
        else:
            return None
        assert self._reference_count[block] == 0
        self._reference_count[block] = 1
        return block

    def _drop_reference(self, block: int) -> None:
        if self._reference_count[block] < 1:
            raise RuntimeError(f"参照数が 0 のブロック {block} を解放しようとした")
        self._reference_count[block] -= 1
        if self._reference_count[block] == 0:
            if block in self._registered_key:
                self._evictable[block] = None
            else:
                self._free_blocks.append(block)

    def reference_count(self, block: int) -> int:
        return self._reference_count[block]

    def block_table(self, state: RequestState) -> list[int]:
        return list(self._tables[state.request.request_id])

    @property
    def num_free_blocks(self) -> int:
        """今すぐ割り当てられるブロック数(空きと、捨ててよい登録済みのブロック)。"""
        return self._available_blocks()

    # ----- 接頭辞の共有 -----

    def _shared_blocks(self, context: Sequence[int]) -> list[int]:
        """文脈の先頭から、既に計算済みの物理ブロックで置き換えられる部分。

        少なくとも 1 トークンは計算し直す(最後のトークンの logits が要るため)ので、共有するのは
        ``(長さ - 1) // b`` ブロックまで。
        """
        if not self.enable_prefix_caching:
            return []
        shared = []
        for i in range((len(context) - 1) // self.block_size):
            key = tuple(context[: (i + 1) * self.block_size])
            block = self._prefix_index.get(key)
            if block is None:
                break
            shared.append(block)
        return shared

    def can_admit(self, state: RequestState, num_running: int) -> bool:
        context = state.context_token_ids()
        shared = self._shared_blocks(context)
        needed = -(-len(context) // self.block_size) - len(shared)
        # 共有するブロックのうち、参照数が 0 で捨てられる側にあるものは、共有で使用中になる
        revived = sum(1 for b in shared if self._reference_count[b] == 0)
        available = self._available_blocks() - revived
        headroom = self.headroom_blocks_per_running * (num_running + 1) if num_running else 0
        return available - needed >= headroom

    def admit(self, state: RequestState) -> int:
        context = state.context_token_ids()
        shared = self._shared_blocks(context)
        table = []
        for block in shared:
            if self._reference_count[block] == 0:
                del self._evictable[block]
            self._reference_count[block] += 1
            table.append(block)
        self.num_prefix_hit_blocks += len(shared)
        for _ in range(-(-len(context) // self.block_size) - len(shared)):
            block = self._allocate_block()
            if block is None:
                for taken in table:
                    self._drop_reference(taken)
                raise RuntimeError("can_admit を確かめずに admit を呼んだ(ブロックが足りない)")
            table.append(block)
        self._tables[state.request.request_id] = table
        self._active[state.request.request_id] = state
        return len(shared) * self.block_size

    def after_prefill(self, state: RequestState) -> None:
        if not self.enable_prefix_caching:
            return
        context = state.context_token_ids()
        table = self._tables[state.request.request_id]
        for i in range(len(context) // self.block_size):
            block = table[i]
            if block in self._registered_key:
                continue
            key = tuple(context[: (i + 1) * self.block_size])
            if key not in self._prefix_index:
                self._prefix_index[key] = block
                self._registered_key[block] = key

    def grow(self, state: RequestState) -> bool:
        table = self._tables[state.request.request_id]
        index = state.num_cached_tokens // self.block_size
        if index == len(table):
            block = self._allocate_block()
            if block is None:
                return False
            table.append(block)
            return True
        block = table[index]
        if self._reference_count[block] > 1:
            # 書き込み時の複製: 共有しているブロックに書く前に、自分専用の複製を作る
            fresh = self._allocate_block()
            if fresh is None:
                return False
            if self.materialize:
                self.keys[:, fresh] = self.keys[:, block]
                self.values[:, fresh] = self.values[:, block]
            self._reference_count[block] -= 1
            table[index] = fresh
            self.num_copy_on_write += 1
        return True

    def fork(self, parent: RequestState, child: RequestState) -> None:
        """``parent`` のブロック表を ``child`` と共有する(参照数を増やすだけで、内容は複製しない)。

        並列サンプリングなど、同じ文脈から複数の続きを生成する場合の入り口。どちらかが共有している
        ブロックに書き込むとき(``grow``)に、書き込み時の複製が起きる。
        """
        table = list(self._tables[parent.request.request_id])
        for block in table:
            self._reference_count[block] += 1
        self._tables[child.request.request_id] = table
        self._active[child.request.request_id] = child
        child.num_cached_tokens = parent.num_cached_tokens

    def release(self, state: RequestState) -> None:
        table = self._tables.pop(state.request.request_id)
        del self._active[state.request.request_id]
        for block in table:
            self._drop_reference(block)

    def usage(self) -> StoreUsage:
        # 共有しているブロックは 1 回と数える(ブロックごとの埋まっている数は、参照する要求の最大)
        filled: dict[int, int] = {}
        size = self.block_size
        for request_id, state in self._active.items():
            for i, block in enumerate(self._tables[request_id]):
                count = min(size, max(state.num_cached_tokens - i * size, 0))
                filled[block] = max(filled.get(block, 0), count)
        return StoreUsage(sum(filled.values()), len(filled) * size, len(filled))

    def _write_indices(self, state, positions):
        table = np.asarray(self._tables[state.request.request_id], dtype=np.int64)
        return table[positions // self.block_size], positions % self.block_size

    def _gather_indices(self, states, max_length):
        size = self.block_size
        num_blocks = -(-max_length // size)
        padded = np.zeros(
            (len(states), num_blocks), dtype=np.int64
        )  # 足りない分は物理ブロック 0 を指す(マスクされる)
        for i, state in enumerate(states):
            table = self._tables[state.request.request_id][:num_blocks]
            padded[i, : len(table)] = table
        positions = np.arange(num_blocks * size, dtype=np.int64)
        first = padded[:, positions // size]
        second = np.ascontiguousarray(np.broadcast_to(positions % size, first.shape))
        return first, second

    def check_consistency(self) -> None:
        """参照数・空きの一覧・ブロック表の整合を確かめる(試験用。壊れていれば AssertionError)。"""
        counts = [0] * self.num_blocks
        for table in self._tables.values():
            for block in table:
                counts[block] += 1
        assert counts == self._reference_count, "参照数がブロック表の数と一致しない"
        free = set(self._free_blocks)
        evictable = set(self._evictable)
        assert len(free) == len(self._free_blocks), "空きの一覧に重複がある"
        assert not (free & evictable), "空きと捨ててよいブロックが重なる"
        in_use = {b for b, c in enumerate(counts) if c > 0}
        assert not (in_use & (free | evictable)), "使用中のブロックが空きの一覧にある"
        assert in_use | free | evictable == set(range(self.num_blocks)), (
            "どの一覧にもないブロックがある"
        )
        for block in evictable:
            assert block in self._registered_key
        for key, block in self._prefix_index.items():
            assert self._registered_key[block] == key
