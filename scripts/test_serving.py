"""``src/serving/`` の単体テスト。

小さな乱数の重みのモデルを CPU で動かし、次を確かめる(Hugging Face Hub は使わない)。

- 負荷の生成: 出力長の分布の平均・変動係数、最大値の期待値、Poisson 過程。
- KV キャッシュの管理: ブロックの確保と解放の整合、参照数、書き込み時の複製(copy-on-write)、
  接頭辞の共有と捨て方、最大長の分を前もって確保する方式のスロット。
- スケジューラ: static batching と continuous batching の反復の数・入れ替え・先着順。
- 正しさ: エンジンを通した生成が、010 の KV キャッシュによる 1 要求ずつの生成と、
  トークン列で一致し、
  logits の差が計算する型から導いた許容誤差以内であること(長さの違う要求の混在、途中での入れ替え、
  終わった行を残す方式、接頭辞の共有、退避と再計算、ブロックの大きさ、Grouped-Query Attention、
  到着の間の待ちを含む)。

使い方 / Usage::

    uv run python scripts/test_serving.py
"""

from __future__ import annotations

import functools
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layers.feedforward import SwiGLUFeedForwardNetwork  # noqa: E402
from src.layers.normalization import LayerNormalization, RMSNorm  # noqa: E402
from src.layers.positional_encoding import RotaryPositionEmbedding  # noqa: E402
from src.models.gpt import GPTLanguageModel  # noqa: E402
from src.serving import (  # noqa: E402
    ContinuousBatchScheduler,
    DryRunner,
    IterationCostModel,
    ModelClock,
    ModelRunner,
    PagedKeyValueStore,
    Request,
    RequestState,
    ReservedKeyValueStore,
    ServingEngine,
    ServingModel,
    StaticBatchScheduler,
    WallClock,
    build_requests,
    expected_maximum,
    fit_cost_model,
    fit_output_length_distribution,
    latency_statistics,
    logits_tolerance,
    maximum_coefficient_of_variation,
    mean_active_requests,
    memory_statistics,
    moment_sampling_distribution,
    poisson_arrival_times,
    predicted_static_to_continuous_ratio,
    reference_generate,
    sample_output_lengths,
    sample_prompt_lengths,
    throughput,
    wasted_row_fraction,
)

CPU = torch.device("cpu")
VOCABULARY = 64
D_MODEL = 32
LAYERS = 2
HEADS = 4
MAX_LENGTH = 64


def tiny_model(seed: int = 0, num_key_value_heads: int | None = None) -> GPTLanguageModel:
    """乱数の重みの小さな GPT(RoPE・RMSNorm・SwiGLU。008 の小型 GPT と同じ構成の縮小版)。

    重みを大きめに初期化して、logits の上位 2 つの差が丸め誤差より十分大きくなるようにする。
    """
    torch.manual_seed(seed)
    model = GPTLanguageModel(
        vocabulary_size=VOCABULARY,
        d_model=D_MODEL,
        num_layers=LAYERS,
        num_heads=HEADS,
        d_ff=64,
        max_sequence_length=MAX_LENGTH,
        positional_transform=RotaryPositionEmbedding(D_MODEL // HEADS, max_position=MAX_LENGTH),
        normalization_factory=RMSNorm,
        feed_forward_factory=functools.partial(SwiGLUFeedForwardNetwork, D_MODEL, 43),
        num_key_value_heads=num_key_value_heads,
    )
    for parameter in model.parameters():
        if parameter.dim() >= 2:
            torch.nn.init.normal_(parameter, 0.0, 0.3)
    return model.eval()


def random_requests(
    count: int, seed: int = 0, prompt_range=(3, 10), output_range=(1, 12), arrival=None
) -> list[Request]:
    rng = np.random.default_rng(seed)
    requests = []
    for i in range(count):
        prompt = tuple(int(t) for t in rng.integers(0, VOCABULARY, rng.integers(*prompt_range)))
        output = int(rng.integers(output_range[0], output_range[1] + 1))
        requests.append(Request(i, prompt, output, 0.0 if arrival is None else arrival[i]))
    return requests


def paged_store(model, num_blocks=64, block_size=4, prefix=False, headroom=0, materialize=True):
    attn = model.blocks[0].self_attn
    return PagedKeyValueStore(
        num_layers=model.num_layers,
        num_key_value_heads=attn.num_key_value_heads,
        head_dim=attn.d_k,
        num_blocks=num_blocks,
        block_size=block_size,
        device=CPU,
        enable_prefix_caching=prefix,
        headroom_blocks_per_running=headroom,
        materialize=materialize,
    )


def reserved_store(model, num_slots=4, materialize=True):
    attn = model.blocks[0].self_attn
    return ReservedKeyValueStore(
        num_layers=model.num_layers,
        num_key_value_heads=attn.num_key_value_heads,
        head_dim=attn.d_k,
        num_slots=num_slots,
        slot_length=model.max_sequence_length,
        device=CPU,
        materialize=materialize,
    )


def dry_run(scheduler, store, requests, cost=None):
    cost = cost or IterationCostModel(1e-3, 1e-4, 2e-3, 1e-5)
    return ServingEngine(scheduler, store, DryRunner(), ModelClock(cost)).run(requests)


class EquivalenceMixin:
    """エンジンを通した生成が 1 要求ずつの生成と一致することを確かめる。"""

    def assert_matches_reference(self, model, requests, result, check_logits=True):
        serving_model_dtype = next(model.parameters()).dtype
        self.assertEqual(serving_model_dtype, torch.float32)
        for request in requests:
            tokens, logits = reference_generate(
                model, request.prompt_token_ids, request.output_length
            )
            self.assertEqual(
                result.generated[request.request_id], tokens, f"要求 {request.request_id}"
            )
            if check_logits:
                got = np.stack(result.logits[request.request_id])
                self.assertEqual(got.dtype, logits.dtype)
                tolerance = logits_tolerance(model, logits)
                self.assertLessEqual(float(np.abs(got - logits).max()), tolerance)
                top2 = np.sort(logits, axis=-1)[:, -2:]
                # トークンの一致が logits の近さから保証される
                # (上位 2 つの差が許容誤差の 2 倍より大きい)
                self.assertTrue(np.all(top2[:, 1] - top2[:, 0] > 2 * tolerance))

    def run_engine(self, model, scheduler, store, requests, clock=None):
        runner = ModelRunner(ServingModel(model))
        engine = ServingEngine(
            scheduler, store, runner, clock or WallClock(CPU), record_logits=True
        )
        return engine.run(requests)


class WorkloadTest(unittest.TestCase):
    def test_fitted_distribution_matches_mean_and_variation(self):
        for variation in (0.25, 0.5, 0.75, 1.0):
            d = fit_output_length_distribution(48, variation, 1, 192)
            self.assertAlmostEqual(d.mean, 48, places=6)
            self.assertAlmostEqual(d.coefficient_of_variation, variation, places=6)
            self.assertAlmostEqual(float(d.probabilities.sum()), 1.0, places=12)
            self.assertEqual((d.minimum, d.maximum), (1, 192))

    def test_small_scale_distribution_keeps_structure(self):
        d = fit_output_length_distribution(6, 1.0, 1, 24)
        self.assertAlmostEqual(d.mean, 6, places=6)
        self.assertAlmostEqual(d.coefficient_of_variation, 1.0, places=6)

    def test_infeasible_variation_is_rejected(self):
        limit = maximum_coefficient_of_variation(48, 1, 192)
        self.assertAlmostEqual(limit, math.sqrt(47 * 144) / 48)
        with self.assertRaises(ValueError):
            fit_output_length_distribution(48, limit + 0.01, 1, 192)
        with self.assertRaises(ValueError):
            fit_output_length_distribution(300, 0.5, 1, 192)

    def test_expected_maximum_matches_monte_carlo(self):
        d = fit_output_length_distribution(48, 0.75, 1, 192)
        rng = np.random.default_rng(0)
        samples = rng.choice(d.support, size=(200_000, 16), p=d.probabilities).max(axis=1)
        standard_error = samples.std() / math.sqrt(len(samples))
        self.assertLess(abs(samples.mean() - expected_maximum(d, 16)), 4 * standard_error)

    def test_expected_maximum_of_one_is_mean(self):
        d = fit_output_length_distribution(48, 0.5, 1, 192)
        self.assertAlmostEqual(expected_maximum(d, 1), d.mean, places=9)

    def test_predicted_ratio_increases_with_variation_and_batch_size(self):
        ratios = [
            predicted_static_to_continuous_ratio(
                fit_output_length_distribution(48, variation, 1, 192), 16
            )
            for variation in (0.25, 0.5, 0.75, 1.0)
        ]
        self.assertEqual(ratios, sorted(ratios))
        self.assertGreater(ratios[0], 1.0)
        d = fit_output_length_distribution(48, 0.5, 1, 192)
        self.assertLess(
            predicted_static_to_continuous_ratio(d, 4), predicted_static_to_continuous_ratio(d, 16)
        )

    def test_sampling_is_reproducible_and_in_range(self):
        d = fit_output_length_distribution(48, 1.0, 1, 192)
        a = sample_output_lengths(d, 500, np.random.default_rng(3))
        b = sample_output_lengths(d, 500, np.random.default_rng(3))
        c = sample_output_lengths(d, 500, np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertTrue(a.min() >= 1 and a.max() <= 192)

    def test_prompt_lengths_are_uniform_integers_in_range(self):
        lengths = sample_prompt_lengths(16, 48, 5000, np.random.default_rng(0))
        self.assertTrue(lengths.min() == 16 and lengths.max() == 48)
        self.assertLess(abs(lengths.mean() - 32), 0.5)

    def test_poisson_arrivals(self):
        times = poisson_arrival_times(4.0, 20_000, np.random.default_rng(0))
        self.assertTrue(np.all(np.diff(times) > 0))
        gaps = np.diff(np.concatenate([[0.0], times]))
        standard_error = gaps.std() / math.sqrt(len(gaps))
        self.assertLess(abs(gaps.mean() - 0.25), 4 * standard_error)
        with self.assertRaises(ValueError):
            poisson_arrival_times(0.0, 3, np.random.default_rng(0))

    def test_moment_sampling_distribution_scale(self):
        d = fit_output_length_distribution(48, 0.5, 1, 192)
        means, variations = moment_sampling_distribution(d, 400, 4000, np.random.default_rng(0))
        expected_std = d.standard_deviation / math.sqrt(400)
        self.assertLess(abs(means.std() / expected_std - 1.0), 0.1)
        self.assertLess(abs(variations.mean() - 0.5), 0.02)

    def test_build_requests_orders_by_arrival_and_shares_prefix(self):
        pool = [list(range(i, i + 20)) for i in range(5)]
        rng = np.random.default_rng(0)
        requests = build_requests(
            pool, [4, 6, 5], [3, 2, 7], [0.9, 0.1, 0.5], rng, shared_prefix=[100, 101]
        )
        self.assertEqual([r.request_id for r in requests], [0, 1, 2])
        self.assertEqual([r.arrival_time for r in requests], [0.1, 0.5, 0.9])
        self.assertEqual([r.output_length for r in requests], [2, 7, 3])
        self.assertTrue(all(r.prompt_token_ids[:2] == (100, 101) for r in requests))
        self.assertEqual([r.prompt_length for r in requests], [2 + 6, 2 + 5, 2 + 4])
        with self.assertRaises(ValueError):
            build_requests(pool, [30], [1], [0.0], rng)

    def test_request_validation(self):
        with self.assertRaises(ValueError):
            Request(0, (), 3)
        with self.assertRaises(ValueError):
            Request(0, (1,), 0)


class PagedStoreTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()

    def state(self, request_id, prompt_length, output_length=4, token=1, tokens=None):
        prompt = tuple(tokens) if tokens is not None else (token,) * prompt_length
        return RequestState(Request(request_id, prompt, output_length))

    def test_admit_allocates_ceil_blocks_and_release_restores_everything(self):
        store = paged_store(self.model, num_blocks=16, block_size=4, materialize=False)
        a, b = self.state(0, 9), self.state(1, 4)
        self.assertTrue(store.can_admit(a, 0))
        self.assertEqual(store.admit(a), 0)
        store.admit(b)
        self.assertEqual(len(store.block_table(a)), 3)  # ceil(9 / 4)
        self.assertEqual(len(store.block_table(b)), 1)
        self.assertEqual(store.num_free_blocks, 16 - 4)
        store.check_consistency()
        store.release(a)
        store.release(b)
        self.assertEqual(store.num_free_blocks, 16)
        store.check_consistency()

    def test_waste_is_less_than_one_block_per_request(self):
        store = paged_store(self.model, num_blocks=64, block_size=8, materialize=False)
        states = [self.state(i, 5 + 3 * i) for i in range(6)]
        for s in states:
            store.admit(s)
            s.num_cached_tokens = s.request.prompt_length
        usage = store.usage()
        self.assertEqual(usage.tokens_used, sum(s.request.prompt_length for s in states))
        self.assertLess(usage.tokens_allocated - usage.tokens_used, len(states) * 8)
        for s in states:
            waste = len(store.block_table(s)) * 8 - s.num_cached_tokens
            self.assertTrue(0 <= waste < 8)

    def test_grow_allocates_only_at_block_boundaries(self):
        store = paged_store(self.model, num_blocks=8, block_size=4, materialize=False)
        s = self.state(0, 3)
        store.admit(s)
        s.num_cached_tokens = 3
        free_before = store.num_free_blocks
        self.assertTrue(store.grow(s))  # 位置 3 は最初のブロックの最後
        self.assertEqual(store.num_free_blocks, free_before)
        s.num_cached_tokens = 4
        self.assertTrue(store.grow(s))  # 位置 4 は 2 つ目のブロックの先頭
        self.assertEqual(store.num_free_blocks, free_before - 1)
        store.check_consistency()

    def test_can_admit_false_when_full_and_grow_fails(self):
        store = paged_store(self.model, num_blocks=2, block_size=4, materialize=False)
        a = self.state(0, 8)
        self.assertTrue(store.can_admit(a, 0))
        store.admit(a)
        a.num_cached_tokens = 8
        self.assertFalse(store.can_admit(self.state(1, 1), 1))
        self.assertFalse(store.grow(a))
        with self.assertRaises(RuntimeError):
            store.admit(self.state(1, 1))
        store.check_consistency()

    def test_headroom_blocks_admission_only_when_something_is_running(self):
        store = paged_store(self.model, num_blocks=6, block_size=4, headroom=2, materialize=False)
        s = self.state(0, 4)
        self.assertTrue(store.can_admit(s, 0))  # 何も実行していなければ余白を求めない
        store.admit(s)
        self.assertTrue(store.can_admit(self.state(1, 4), 1))  # 空き 5 - 必要 1 = 4 >= 2 * 2
        for i in range(2, 4):
            store.admit(self.state(i, 4))
        self.assertFalse(store.can_admit(self.state(9, 4), 3))  # 空き 3 - 必要 1 = 2 < 2 * 4

    def test_double_release_is_detected(self):
        store = paged_store(self.model, materialize=False)
        s = self.state(0, 4)
        store.admit(s)
        store.release(s)
        with self.assertRaises(KeyError):
            store.release(s)

    def test_copy_on_write_copies_contents_and_isolates_writes(self):
        store = paged_store(self.model, num_blocks=8, block_size=4)
        parent = self.state(0, 6)
        store.admit(parent)
        parent.num_cached_tokens = 6
        marker = torch.arange(float(store.keys[0, 0].numel())).view_as(store.keys[0, 0])
        shared_block = store.block_table(parent)[1]  # 2 つ目のブロックは 2 トークンだけ埋まる
        store.keys[:, shared_block] = marker
        child = self.state(1, 6)
        store.fork(parent, child)
        self.assertEqual(store.reference_count(shared_block), 2)
        self.assertEqual(store.block_table(child), store.block_table(parent))
        self.assertEqual(child.num_cached_tokens, 6)

        self.assertTrue(store.grow(child))  # 共有している部分的なブロックへの書き込み = 複製
        self.assertEqual(store.num_copy_on_write, 1)
        new_block = store.block_table(child)[1]
        self.assertNotEqual(new_block, shared_block)
        self.assertEqual(store.block_table(child)[0], store.block_table(parent)[0])
        self.assertEqual(store.reference_count(shared_block), 1)
        self.assertEqual(store.reference_count(new_block), 1)
        self.assertEqual(store.reference_count(store.block_table(parent)[0]), 2)
        torch.testing.assert_close(store.keys[:, new_block], store.keys[:, shared_block])
        store.keys[:, new_block, 2] = -1.0  # 子が自分の複製に書いても親は変わらない
        torch.testing.assert_close(
            store.keys[:, shared_block], marker.expand_as(store.keys[:, shared_block])
        )
        self.assertTrue(store.grow(parent))  # 親は既に自分専用になったので、複製は起きない
        self.assertEqual(store.num_copy_on_write, 1)
        store.check_consistency()
        store.release(parent)
        store.release(child)
        self.assertEqual(store.num_free_blocks, 8)
        store.check_consistency()

    def test_fork_then_writing_at_boundary_needs_no_copy(self):
        store = paged_store(self.model, num_blocks=8, block_size=4, materialize=False)
        parent = self.state(0, 8)
        store.admit(parent)
        parent.num_cached_tokens = 8
        child = self.state(1, 8)
        store.fork(parent, child)
        self.assertTrue(store.grow(child))
        self.assertTrue(store.grow(parent))
        self.assertEqual(store.num_copy_on_write, 0)
        full = store.block_table(parent)[:2]
        self.assertEqual(full, store.block_table(child)[:2])
        self.assertTrue(all(store.reference_count(b) == 2 for b in full))
        self.assertNotEqual(store.block_table(parent)[2], store.block_table(child)[2])
        store.check_consistency()

    def test_prefix_sharing_reuses_full_blocks_only(self):
        store = paged_store(self.model, num_blocks=16, block_size=4, prefix=True, materialize=False)
        common = list(range(1, 12))  # 11 トークン = 完全なブロック 2 つ + 3 トークン
        first = self.state(0, 0, tokens=common + [50, 51])
        self.assertEqual(store.admit(first), 0)
        first.num_cached_tokens = len(common) + 2
        store.after_prefill(first)
        second = self.state(1, 0, tokens=common + [60])
        self.assertEqual(store.admit(second), 8)  # 共有できるのは完全な 2 ブロック
        shared = store.block_table(second)[:2]
        self.assertEqual(shared, store.block_table(first)[:2])
        self.assertTrue(all(store.reference_count(b) == 2 for b in shared))
        self.assertNotEqual(store.block_table(second)[2], store.block_table(first)[2])
        self.assertEqual(store.num_prefix_hit_blocks, 2)
        store.check_consistency()

    def test_prefix_sharing_recomputes_at_least_one_token(self):
        store = paged_store(self.model, num_blocks=16, block_size=4, prefix=True, materialize=False)
        prompt = list(range(1, 9))  # ちょうど 2 ブロック
        a = self.state(0, 0, tokens=prompt)
        store.admit(a)
        a.num_cached_tokens = 8
        store.after_prefill(a)
        b = self.state(1, 0, tokens=prompt)
        self.assertEqual(store.admit(b), 4)  # 同じ内容でも最後のブロックは計算し直す
        c = self.state(2, 0, tokens=prompt[:5])
        self.assertEqual(store.admit(c), 4)

    def test_released_prefix_blocks_stay_cached_and_are_evicted_oldest_first(self):
        store = paged_store(self.model, num_blocks=6, block_size=4, prefix=True, materialize=False)

        def serve(request_id, base):
            state = self.state(request_id, 0, tokens=list(range(base, base + 9)))
            cached = store.admit(state)
            state.num_cached_tokens = 9
            store.after_prefill(state)
            store.release(state)
            return cached

        self.assertEqual(serve(0, 1), 0)
        self.assertEqual(serve(1, 1), 8)  # 解放後も残っていて再利用できる
        self.assertEqual(
            store.num_free_blocks, 6
        )  # 参照数 0 の登録済みブロックは割り当て可能に数える
        store.check_consistency()
        self.assertEqual(serve(2, 100), 0)  # 別の内容は新しいブロックを使う
        self.assertEqual(serve(3, 200), 0)  # 6 ブロックでは足りず、古い 1 の接頭辞を捨てる
        self.assertEqual(serve(4, 1), 0)
        store.check_consistency()

    def test_usage_counts_shared_blocks_once(self):
        store = paged_store(self.model, num_blocks=16, block_size=4, prefix=True, materialize=False)
        common = list(range(1, 9))
        a = self.state(0, 0, tokens=common + [20])
        store.admit(a)
        a.num_cached_tokens = 9
        store.after_prefill(a)
        b = self.state(1, 0, tokens=common + [30])
        store.admit(b)
        b.num_cached_tokens = 9
        usage = store.usage()
        self.assertEqual(usage.units_in_use, 2 + 1 + 1)  # 共有 2 ブロック + 各自の最後のブロック
        self.assertEqual(usage.tokens_used, 8 + 1 + 1)

    def test_consistency_after_a_full_dry_run_and_no_leak(self):
        rng = np.random.default_rng(0)
        requests = [
            Request(
                i,
                tuple(int(t) for t in rng.integers(0, 50, rng.integers(2, 20))),
                int(rng.integers(1, 15)),
            )
            for i in range(30)
        ]
        for prefix in (False, True):
            store = paged_store(
                self.model, num_blocks=20, block_size=4, prefix=prefix, materialize=False
            )
            dry_run(ContinuousBatchScheduler(6), store, requests)
            store.check_consistency()
            self.assertEqual(store.num_free_blocks, 20)


class ReservedStoreTest(unittest.TestCase):
    def test_slots_are_exclusive_and_returned(self):
        model = tiny_model()
        store = reserved_store(model, num_slots=2, materialize=False)
        states = [RequestState(Request(i, (1, 2, 3), 4)) for i in range(3)]
        self.assertTrue(store.can_admit(states[0], 0))
        store.admit(states[0])
        store.admit(states[1])
        self.assertFalse(store.can_admit(states[2], 2))
        self.assertEqual(store.capacity_tokens, 2 * MAX_LENGTH)
        usage = store.usage()
        self.assertEqual(usage.tokens_allocated, 2 * MAX_LENGTH)
        states[0].num_cached_tokens = 5
        self.assertEqual(store.usage().tokens_used, 5)
        store.release(states[0])
        self.assertTrue(store.can_admit(states[2], 1))
        self.assertEqual(store.usage().units_in_use, 1)

    def test_request_longer_than_slot_is_rejected(self):
        model = tiny_model()
        store = reserved_store(model, num_slots=1, materialize=False)
        with self.assertRaises(ValueError):
            store.admit(RequestState(Request(0, tuple(range(40)), 40)))


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()

    def requests_with_outputs(self, outputs, prompt=4, arrival=None):
        return [
            Request(
                i,
                tuple((i + j) % VOCABULARY for j in range(prompt)),
                o,
                0.0 if arrival is None else arrival[i],
            )
            for i, o in enumerate(outputs)
        ]

    def test_static_iterations_equal_sum_of_batch_maxima(self):
        outputs = [3, 9, 2, 5, 7, 1, 4, 4, 8, 2]
        for batch in (2, 3, 4):
            store = paged_store(self.model, num_blocks=128, materialize=False)
            result = dry_run(
                StaticBatchScheduler(batch), store, self.requests_with_outputs(outputs)
            )
            expected = sum(max(outputs[i : i + batch]) for i in range(0, len(outputs), batch))
            self.assertEqual(len(result.iterations), expected)

    def test_static_keeps_finished_rows_and_wastes_computation(self):
        outputs = [2, 8, 2, 8]
        store = paged_store(self.model, num_blocks=128, materialize=False)
        result = dry_run(StaticBatchScheduler(4), store, self.requests_with_outputs(outputs))
        self.assertEqual(len(result.iterations), 8)
        rows = [it.prefill_rows + it.decode_rows for it in result.iterations]
        self.assertEqual(rows, [4] * 8)  # 終わった行も計算し続ける
        self.assertEqual([it.active_requests for it in result.iterations], [4, 4, 2, 2, 2, 2, 2, 2])
        self.assertAlmostEqual(wasted_row_fraction(result), 1 - 20 / 32)

    def test_static_does_not_admit_until_batch_is_empty(self):
        store = paged_store(self.model, num_blocks=128, materialize=False)
        result = dry_run(StaticBatchScheduler(2), store, self.requests_with_outputs([6, 1, 1, 1]))
        first_batch_end = max(r.finish_time for r in result.requests[:2])
        self.assertGreaterEqual(
            min(r.admitted_time for r in result.requests[2:]), first_batch_end - 1e-9
        )

    def test_continuous_equals_static_when_all_lengths_equal(self):
        outputs = [5] * 8
        counts = []
        for scheduler in (StaticBatchScheduler(4), ContinuousBatchScheduler(4)):
            store = paged_store(self.model, num_blocks=128, materialize=False)
            counts.append(
                len(dry_run(scheduler, store, self.requests_with_outputs(outputs)).iterations)
            )
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], 10)

    def test_continuous_refills_freed_slots_and_never_exceeds_the_limit(self):
        outputs = [2, 9, 2, 9, 2, 9, 2, 9, 2, 9, 2, 9]
        store = paged_store(self.model, num_blocks=256, materialize=False)
        result = dry_run(ContinuousBatchScheduler(4), store, self.requests_with_outputs(outputs))
        self.assertTrue(all(it.active_requests <= 4 for it in result.iterations))
        self.assertLess(len(result.iterations), 9 + 9 + 9)  # static なら 3 バッチで 27 反復
        # 反復の終わりに待っている要求がある間は、実行中の要求数が上限に保たれる
        busy = [it for it in result.iterations if it.waiting > 0]
        self.assertTrue(busy)
        self.assertTrue(all(it.active_requests == 4 for it in busy))

    def test_continuous_iterations_are_between_the_ideal_and_static(self):
        rng = np.random.default_rng(0)
        outputs = [int(x) for x in rng.integers(1, 20, 24)]
        batch = 4
        store = paged_store(self.model, num_blocks=512, materialize=False)
        result = dry_run(
            ContinuousBatchScheduler(batch), store, self.requests_with_outputs(outputs)
        )
        static_iterations = sum(max(outputs[i : i + batch]) for i in range(0, len(outputs), batch))
        self.assertGreaterEqual(len(result.iterations), math.ceil(sum(outputs) / batch))
        self.assertLess(len(result.iterations), static_iterations)

    def test_first_come_first_served(self):
        store = paged_store(self.model, num_blocks=256, materialize=False)
        result = dry_run(ContinuousBatchScheduler(2), store, self.requests_with_outputs([4] * 6))
        admitted = [r.admitted_time for r in result.requests]
        self.assertEqual(admitted, sorted(admitted))

    def test_timestamps_are_ordered_and_every_request_gets_its_length(self):
        outputs = [1, 4, 2, 7, 3]
        store = paged_store(self.model, num_blocks=256, materialize=False)
        result = dry_run(ContinuousBatchScheduler(2), store, self.requests_with_outputs(outputs))
        for record, output in zip(result.requests, outputs, strict=True):
            self.assertLessEqual(record.arrival_time, record.admitted_time)
            self.assertLessEqual(record.admitted_time, record.first_token_time)
            self.assertLessEqual(record.first_token_time, record.finish_time)
            self.assertEqual(len(result.generated[record.request_id]), output)
        self.assertEqual(result.requests[0].first_token_time, result.requests[0].finish_time)
        self.assertIsNone(result.requests[0].time_per_output_token)

    def test_idle_time_is_skipped_until_the_next_arrival(self):
        cost = IterationCostModel(0.01, 0.0, 0.01, 0.0)
        store = paged_store(self.model, num_blocks=64, materialize=False)
        requests = self.requests_with_outputs([3, 3], arrival=[0.0, 5.0])
        result = dry_run(ContinuousBatchScheduler(2), store, requests, cost)
        self.assertAlmostEqual(result.requests[1].admitted_time, 5.0)
        self.assertAlmostEqual(result.requests[1].queue_wait, 0.0)
        self.assertAlmostEqual(result.requests[1].time_to_first_token, 0.01)
        self.assertAlmostEqual(result.makespan, 5.0 + 0.03)

    def test_dry_run_reproduces_the_real_schedule(self):
        requests = random_requests(10, seed=2)
        model = self.model
        real = ServingEngine(
            ContinuousBatchScheduler(3),
            paged_store(model),
            ModelRunner(ServingModel(model)),
            WallClock(CPU),
        ).run(requests)
        dry = ServingEngine(
            ContinuousBatchScheduler(3),
            paged_store(model, materialize=False),
            DryRunner(),
            ModelClock(IterationCostModel(1e-3, 0, 1e-3, 0)),
        ).run(requests)

        def schedule(result):
            return [
                (i.prefill_rows, i.prefill_query_tokens, i.decode_rows, i.active_requests)
                for i in result.iterations
            ]

        self.assertEqual(schedule(real), schedule(dry))

    def test_budget_smaller_than_one_request_stops_instead_of_looping(self):
        store = paged_store(self.model, num_blocks=1, block_size=4, materialize=False)
        with self.assertRaises(RuntimeError):
            dry_run(ContinuousBatchScheduler(2), store, self.requests_with_outputs([3], prompt=9))

    def test_preemption_happens_when_blocks_run_out_and_all_requests_finish(self):
        store = paged_store(self.model, num_blocks=10, block_size=2, materialize=False)
        result = dry_run(
            ContinuousBatchScheduler(6), store, self.requests_with_outputs([12] * 6, prompt=4)
        )
        self.assertGreater(result.num_preemptions, 0)
        self.assertTrue(all(len(result.generated[i]) == 12 for i in range(6)))
        self.assertEqual(sum(r.num_preemptions for r in result.requests), result.num_preemptions)
        # 退避されるのは最後に実行に入れた要求で、最初に入れた要求は退避されない
        self.assertEqual(result.requests[0].num_preemptions, 0)
        self.assertGreater(result.requests[-1].num_preemptions, 0)
        store.check_consistency()
        self.assertEqual(store.num_free_blocks, 10)


class MetricsTest(unittest.TestCase):
    def test_throughput_latency_and_memory(self):
        model = tiny_model()
        cost = IterationCostModel(0.01, 0.0, 0.01, 0.0)
        requests = [Request(i, (1, 2, 3, 4), 5) for i in range(4)]
        reserved = dry_run(
            ContinuousBatchScheduler(4), reserved_store(model, 4, False), requests, cost
        )
        self.assertAlmostEqual(throughput(reserved), 20 / reserved.makespan)
        self.assertAlmostEqual(mean_active_requests(reserved), 4.0)
        stats = latency_statistics(reserved)
        self.assertAlmostEqual(stats["queue_wait_mean"], 0.0)
        self.assertGreater(stats["time_per_output_token_mean"], 0.0)
        memory = memory_statistics(reserved)
        self.assertAlmostEqual(memory["tokens_allocated_mean"], 4 * MAX_LENGTH)
        self.assertLess(memory["utilization"], 0.2)  # 長さ 8 前後の要求に 64 トークンを確保している
        paged = dry_run(
            ContinuousBatchScheduler(4),
            paged_store(model, 64, 4, materialize=False),
            requests,
            cost,
        )
        self.assertGreater(memory_statistics(paged)["utilization"], 0.8)
        self.assertLess(memory_statistics(paged)["wasted_tokens_mean"], 4 * 4)

    def test_cost_model_fit_recovers_a_line(self):
        decode = [(n, 0.004 + 0.0002 * n) for n in (1, 2, 4, 8, 16)]
        prefill = [(n, 0.006 + 0.00003 * n) for n in (10, 50, 100, 400)]
        model, diagnostics = fit_cost_model(decode, prefill)
        self.assertAlmostEqual(model.decode_fixed, 0.004, places=9)
        self.assertAlmostEqual(model.decode_per_row, 0.0002, places=9)
        self.assertAlmostEqual(model.prefill_per_token, 0.00003, places=9)
        self.assertAlmostEqual(diagnostics["decode_r2"], 1.0)
        self.assertAlmostEqual(diagnostics["decode_fixed_share"], 0.004 / 0.0042)

    def test_model_clock_accumulates_and_never_goes_back(self):
        clock = ModelClock(IterationCostModel(0.01, 0.001, 0.02, 0.0001))
        clock.start()
        clock.tick("decode", 4, 4)
        clock.tick("prefill", 2, 30)
        self.assertAlmostEqual(clock.now(), 0.014 + 0.023)
        clock.advance_to(0.001)
        self.assertAlmostEqual(clock.now(), 0.037)
        clock.advance_to(1.0)
        self.assertAlmostEqual(clock.now(), 1.0)
        with self.assertRaises(ValueError):
            clock.tick("other", 1, 1)


class EquivalenceTest(EquivalenceMixin, unittest.TestCase):
    def test_continuous_with_paged_store_and_swapping(self):
        model = tiny_model()
        requests = random_requests(9, seed=1)
        result = self.run_engine(model, ContinuousBatchScheduler(3), paged_store(model), requests)
        self.assert_matches_reference(model, requests, result)
        self.assertGreater(len(result.iterations), 0)

    def test_continuous_with_reserved_store(self):
        model = tiny_model()
        requests = random_requests(7, seed=3)
        result = self.run_engine(
            model, ContinuousBatchScheduler(3), reserved_store(model, 3), requests
        )
        self.assert_matches_reference(model, requests, result)

    def test_static_with_finished_rows_kept(self):
        model = tiny_model()
        requests = random_requests(7, seed=4, output_range=(1, 14))
        for store in (paged_store(model, num_blocks=128), reserved_store(model, 3)):
            result = self.run_engine(model, StaticBatchScheduler(3), store, requests)
            self.assert_matches_reference(model, requests, result)

    def test_paged_and_reserved_generate_identical_tokens(self):
        model = tiny_model(seed=1)
        requests = random_requests(8, seed=5)
        paged = self.run_engine(model, ContinuousBatchScheduler(4), paged_store(model), requests)
        reserved = self.run_engine(
            model, ContinuousBatchScheduler(4), reserved_store(model, 4), requests
        )
        self.assertEqual(paged.generated, reserved.generated)

    def test_block_sizes_do_not_change_the_result(self):
        model = tiny_model(seed=2)
        requests = random_requests(6, seed=6)
        baseline = None
        for block_size in (1, 2, 3, 4, 7, 16, 64):
            store = paged_store(model, num_blocks=-(-512 // block_size), block_size=block_size)
            result = self.run_engine(model, ContinuousBatchScheduler(3), store, requests)
            self.assert_matches_reference(model, requests, result)
            baseline = baseline or result.generated
            self.assertEqual(result.generated, baseline)

    def test_prefix_sharing_matches_reference_and_reuses_blocks(self):
        model = tiny_model(seed=3)
        rng = np.random.default_rng(7)
        system = tuple(int(t) for t in rng.integers(0, VOCABULARY, 17))  # 4 ブロック + 1 トークン
        requests = [
            Request(
                i,
                system + tuple(int(t) for t in rng.integers(0, VOCABULARY, rng.integers(1, 6))),
                int(rng.integers(1, 9)),
            )
            for i in range(8)
        ]
        shared_store = paged_store(model, num_blocks=64, block_size=4, prefix=True)
        result = self.run_engine(model, ContinuousBatchScheduler(2), shared_store, requests)
        self.assert_matches_reference(model, requests, result)
        self.assertGreaterEqual(
            shared_store.num_prefix_hit_blocks, 4 * 6
        )  # 最初の 2 件の後は 4 ブロックずつ再利用
        plain = self.run_engine(
            model,
            ContinuousBatchScheduler(2),
            paged_store(model, num_blocks=64, block_size=4),
            requests,
        )
        self.assertEqual(result.generated, plain.generated)
        shared_store.check_consistency()

    def test_prefix_sharing_in_one_prefill_batch_and_with_identical_prompts(self):
        model = tiny_model(seed=4)
        prompt = tuple(range(1, 13))
        requests = [Request(i, prompt, 5) for i in range(4)]
        store = paged_store(model, num_blocks=64, block_size=4, prefix=True)
        result = self.run_engine(model, ContinuousBatchScheduler(4), store, requests)
        self.assert_matches_reference(model, requests, result)
        late = [Request(i, prompt, 5, arrival_time=100.0) for i in range(4, 6)]
        store = paged_store(model, num_blocks=64, block_size=4, prefix=True)
        result = self.run_engine(model, ContinuousBatchScheduler(4), store, requests[:2] + late)
        self.assert_matches_reference(model, requests[:2] + late, result)
        self.assertGreater(store.num_prefix_hit_blocks, 0)

    def test_preemption_with_recompute_matches_reference(self):
        model = tiny_model(seed=5)
        requests = random_requests(8, seed=8, prompt_range=(4, 8), output_range=(6, 14))
        store = paged_store(model, num_blocks=16, block_size=2)
        result = self.run_engine(model, ContinuousBatchScheduler(8), store, requests)
        self.assertGreater(result.num_preemptions, 0)
        self.assert_matches_reference(model, requests, result)
        store.check_consistency()
        self.assertEqual(store.num_free_blocks, 16)

    def test_preemption_keeps_first_token_time_of_the_first_generation(self):
        model = tiny_model(seed=5)
        requests = random_requests(8, seed=8, prompt_range=(4, 8), output_range=(6, 14))
        store = paged_store(model, num_blocks=16, block_size=2)
        result = self.run_engine(model, ContinuousBatchScheduler(8), store, requests)
        for record in result.requests:
            self.assertLessEqual(record.first_token_time, record.finish_time)
            if record.num_preemptions:
                self.assertLess(record.first_token_time, record.finish_time)

    def test_grouped_query_attention_model(self):
        model = tiny_model(seed=6, num_key_value_heads=2)
        requests = random_requests(6, seed=9)
        for store in (paged_store(model), reserved_store(model, 3)):
            result = self.run_engine(model, ContinuousBatchScheduler(3), store, requests)
            self.assert_matches_reference(model, requests, result)

    def test_arrivals_with_idle_gaps(self):
        model = tiny_model(seed=7)
        arrival = [0.0, 0.0, 0.01, 50.0, 50.0, 120.0]
        requests = random_requests(6, seed=10, arrival=arrival)
        result = self.run_engine(model, ContinuousBatchScheduler(2), paged_store(model), requests)
        self.assert_matches_reference(model, requests, result)
        self.assertGreaterEqual(result.requests[5].first_token_time, 120.0)
        self.assertLess(result.requests[5].queue_wait, 5.0)

    def test_output_length_one_and_long_requests(self):
        model = tiny_model(seed=8)
        requests = [
            Request(0, (1, 2, 3), 1),
            Request(1, (4,), 40),
            Request(2, tuple(range(20)), 1),
            Request(3, tuple(range(5, 25)), 30),
        ]
        for scheduler in (ContinuousBatchScheduler(3), StaticBatchScheduler(3)):
            result = self.run_engine(model, scheduler, paged_store(model, num_blocks=128), requests)
            self.assert_matches_reference(model, requests, result)

    def test_reference_matches_generate_with_cache(self):
        model = tiny_model(seed=9)
        prompt = (3, 1, 4, 1, 5, 9, 2)
        tokens, _ = reference_generate(model, prompt, 10)
        generated = model.generate(torch.tensor([prompt]), 10, temperature=0.0, use_cache=True)
        self.assertEqual(tokens, generated[0, len(prompt) :].tolist())

    def test_tolerance_is_derived_from_dtype_and_magnitude(self):
        model = tiny_model()
        logits = np.full((3, VOCABULARY), 2.0, dtype=np.float32)
        small = logits_tolerance(model, logits)
        larger = logits_tolerance(model, logits * 100)
        self.assertAlmostEqual(
            larger / small, 100.0, places=6
        )  # 大きさの尺度(最大|logit|)に比例する
        eps32 = float(torch.finfo(torch.float32).eps)
        self.assertGreater(small, 10 * eps32)
        self.assertLess(small, 1e-2)
        with self.assertRaises(TypeError):
            logits_tolerance(model, logits.astype(np.float64))

    def test_unsupported_models_are_rejected(self):
        no_rope = GPTLanguageModel(
            VOCABULARY, D_MODEL, 1, HEADS, 64, MAX_LENGTH, normalization_factory=RMSNorm
        )
        with self.assertRaises(ValueError):
            ServingModel(no_rope)
        post_norm = GPTLanguageModel(
            VOCABULARY,
            D_MODEL,
            1,
            HEADS,
            64,
            MAX_LENGTH,
            positional_transform=RotaryPositionEmbedding(D_MODEL // HEADS, max_position=MAX_LENGTH),
            normalization_factory=LayerNormalization,
            norm_first=False,
        )
        with self.assertRaises(ValueError):
            ServingModel(post_norm)


if __name__ == "__main__":
    unittest.main(verbosity=1)
