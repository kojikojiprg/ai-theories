"""``src/`` の整理(関数の移動・統合)が、挙動を変えていないことの試験。

参照のコミット(整理の前のコミット)を ``git worktree`` で取り出し、同じ入力で計算した結果を、
参照のコミットと現在の作業ツリーで **それぞれ別のサブプロセスで** 計算して、bit 単位で比べる。
入力はすべて CPU で軽く終わる小さなもの(学習ループは数ステップ)である。

比べるもの:

- ブートストラップの重み・比の分布(``paired_bootstrap_ratio_of_sums``・
  ``paired_cluster_bootstrap_ratio_of_sums``・``cluster_bootstrap_weights``)。
- 検索の順位と nDCG@10(``rank_candidates``・``rank_reranked_candidates``)。
- 6 つの学習ループ(``train_image_classifier``・``train_contrastive_model``・
  ``train_text_embedding``・``train_text_embedding_from_batches``・``train_reranker``・
  ``train_reranker_variable``)の学習の記録と学習後の重み。
- ``split_weight_decay_parameters``・``unscale_gradients_and_compute_norm``(移動した関数)。

参照の実行を 2 回繰り返して差がないこと(参照の実行が決定的であること)も確かめる。

使い方(リポジトリのルートで)::

    uv run python scripts/test_src_refactor_equivalence.py --base e8706ec

損失スケーリング(CUDA のみ)の経路は CPU の学習ループでは通らないので、
``step_optimizers_and_update_loss_scale`` は、統合の前のループに書かれていた記述を
そのまま写した参照の実装と、直接比べる(同じ試験の中で行う)。
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
IGNORED_KEYS = {"train_seconds"}  # 実時間は再現しない


# ---------------------------------------------------------------------------
# 結果の正規化(bit 単位の比較のため、バイト列にする)
# ---------------------------------------------------------------------------
def normalize(value):
    """値を、bit 単位で比較できる入れ子のタプル・辞書にする。"""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return ("tensor", str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", str(array.dtype), array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return normalize(np.asarray(value))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    if isinstance(value, int | str | bytes | type(None)):
        return (type(value).__name__, value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            {f.name: normalize(getattr(value, f.name)) for f in dataclasses.fields(value)},
        )
    if isinstance(value, dict):
        return (
            "dict",
            {str(k): normalize(v) for k, v in value.items() if k not in IGNORED_KEYS},
        )
    if isinstance(value, list | tuple):
        return (type(value).__name__, [normalize(v) for v in value])
    raise TypeError(f"正規化できない型: {type(value)}")


# ---------------------------------------------------------------------------
# 試験の入力と実行(参照のコミットでも作業ツリーでも同じソースで実行する)
# ---------------------------------------------------------------------------
def tiny_gpt(vocabulary_size: int = 32):
    import torch  # noqa: F401

    from src.models.gpt import GPTLanguageModel

    return GPTLanguageModel(
        vocabulary_size=vocabulary_size,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_sequence_length=24,
    )


def evaluation_probe(model):
    """評価の関数の代わり。評価の時点の重みの要約を返す。"""
    import torch

    return {
        "weight_sum": float(sum(p.detach().double().sum() for p in model.parameters())),
        "training": bool(torch.is_grad_enabled()),
    }


def common_hyperparameters():
    return {
        "peak_learning_rate": 1e-2,
        "warmup_steps": 2,
        "min_learning_rate": 1e-3,
        "weight_decay": 0.1,
        "gradient_clip_threshold": 0.05,  # 小さい値にして、clipping が発動する経路も通す
    }


def case_bootstrap():
    import numpy as np

    from src.retrieval.cost_curve import cluster_bootstrap_weights
    from src.utils.statistics import (
        paired_bootstrap_ratio_of_sums,
        paired_cluster_bootstrap_ratio_of_sums,
    )

    rng = np.random.default_rng(0)
    numerators = rng.uniform(0.5, 2.0, size=(3, 37))
    denominators = rng.uniform(1.0, 3.0, size=37)
    clusters = rng.integers(0, 9, size=37).tolist()
    out = {}
    for resamples, chunk in ((1200, 500), (7, 500), (1200, 64)):
        out[f"unit_{resamples}_{chunk}"] = paired_bootstrap_ratio_of_sums(
            numerators, denominators, resamples, seed=3, chunk_size=chunk
        )
        out[f"unit1d_{resamples}_{chunk}"] = paired_bootstrap_ratio_of_sums(
            numerators[0], denominators, resamples, seed=3, chunk_size=chunk
        )
        out[f"cluster_{resamples}_{chunk}"] = paired_cluster_bootstrap_ratio_of_sums(
            numerators, denominators, clusters, resamples, seed=3, chunk_size=chunk
        )
        out[f"weights_{resamples}_{chunk}"] = cluster_bootstrap_weights(
            clusters, resamples, seed=5, chunk_size=chunk
        )
    out["weights_strings"] = cluster_bootstrap_weights([f"a{c}" for c in clusters], 600, seed=1)
    return out


def case_ranking():
    import numpy as np
    import torch

    from src.retrieval.evaluation import rank_candidates
    from src.retrieval.reranking import rank_reranked_candidates

    rng = np.random.default_rng(1)
    num_queries, num_passages = 40, 60
    passage_articles = np.sort(rng.integers(0, 12, size=num_passages))
    query_sources = rng.integers(0, num_passages, size=num_queries)
    query_articles = passage_articles[query_sources]
    # 同じ記事の他の passage が 1 つもない query を避ける
    keep = np.array([(passage_articles == a).sum() >= 2 for a in query_articles], dtype=bool)
    query_sources, query_articles = query_sources[keep], query_articles[keep]
    scores = torch.as_tensor(rng.normal(size=(len(query_sources), num_passages)).round(1))
    out = {
        "ranking": rank_candidates(
            scores, query_articles, passage_articles, query_sources, chunk_size=7
        )
    }
    positives = np.array([(passage_articles == a).sum() - 1 for a in query_articles])
    for k in (5, 15):  # 候補が 10 件に満たない場合と、満たす場合
        candidate_ids = np.stack(
            [rng.choice(num_passages, size=k, replace=False) for _ in query_sources]
        )
        candidate_scores = rng.normal(size=(len(query_sources), k)).round(1)
        out[f"reranked_{k}"] = rank_reranked_candidates(
            candidate_scores, candidate_ids, query_articles, passage_articles, positives
        )
    out["constant"] = rank_reranked_candidates(
        np.zeros((len(query_sources), 15)),
        candidate_ids,
        query_articles,
        passage_articles,
        positives,
    )
    return out


def case_moved_functions():
    import torch

    from src.models.vit import VisionTransformer
    from src.training.classification import (
        split_weight_decay_parameters,
        unscale_gradients_and_compute_norm,
    )

    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=8, patch_size=4, num_classes=3, d_model=16, num_layers=1, num_heads=2, d_ff=32
    )
    decay, no_decay = split_weight_decay_parameters(model)
    names = {id(p): n for n, p in model.named_parameters()}
    out = {
        "decay": [names[id(p)] for p in decay],
        "no_decay": [names[id(p)] for p in no_decay],
    }
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100.0
    out["norm_scaled"] = unscale_gradients_and_compute_norm(list(model.parameters()), 64.0)
    out["grads_scaled"] = [p.grad.clone() for p in model.parameters()]
    out["norm_plain"] = unscale_gradients_and_compute_norm(list(model.parameters()), None)
    return out


def case_classification():
    import torch

    from src.models.vit import VisionTransformer
    from src.training.classification import train_image_classifier

    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=8, patch_size=4, num_classes=3, d_model=16, num_layers=1, num_heads=2, d_ff=32
    )
    images = torch.randint(0, 256, (24, 3, 8, 8), dtype=torch.uint8)
    labels = torch.randint(0, 3, (24,))
    history = train_image_classifier(
        model,
        images,
        labels,
        torch.arange(24),
        num_steps=6,
        batch_size=4,
        data_seed=7,
        crop_padding=2,
        evaluation_steps=(3,),
        evaluation_fn=evaluation_probe,
        **common_hyperparameters(),
    )
    return {"history": history, "state": model.state_dict()}


def tiny_clip():
    from src.models.clip import CausalTextTransformer, CLIPDualEncoder
    from src.models.vit import VisionTransformer

    vision = VisionTransformer(
        image_size=32, patch_size=16, num_classes=3, d_model=16, num_layers=1, num_heads=2, d_ff=32
    )
    text = CausalTextTransformer(
        vocabulary_size=20,
        context_length=6,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        end_token_id=1,
    )
    return CLIPDualEncoder(vision, text, embedding_dim=8)


def case_contrastive():
    import numpy as np
    import torch

    from src.training.contrastive import train_contrastive_model

    out = {}
    num_captions, copies = 8, 2
    scene_caption_ids = np.repeat(np.arange(num_captions), copies)
    generator = torch.Generator().manual_seed(11)
    images = torch.randint(
        0, 256, (num_captions * copies, 3, 32, 32), dtype=torch.uint8, generator=generator
    )
    captions = torch.randint(2, 20, (num_captions, 6), generator=generator)
    captions[:, -1] = 1  # 終端
    hard = {
        "shift": (np.arange(num_captions) + 1) % num_captions,
        "reverse": num_captions - 1 - np.arange(num_captions),
    }
    allowed = np.array([True, True, False, True, True, True, False, True])
    for loss_type in ("softmax", "sigmoid", "negclip"):
        torch.manual_seed(0)
        model = tiny_clip()
        history = train_contrastive_model(
            model,
            images,
            captions,
            scene_caption_ids,
            copies,
            loss_type,
            num_steps=5,
            batch_size=4,
            data_seed=3,
            hard_negative_ids=hard if loss_type == "negclip" else None,
            hard_negative_allowed=allowed if loss_type == "negclip" else None,
            evaluation_steps=(2,),
            evaluation_fn=evaluation_probe,
            **common_hyperparameters(),
        )
        out[loss_type] = {"history": history, "state": model.state_dict()}
    return out


def tiny_text_embedding():
    from src.models.text_embedding import TextEmbeddingModel

    return TextEmbeddingModel(tiny_gpt(), pooling="mean", attention="causal")


def case_contrastive_text():
    import numpy as np
    import torch

    from src.training.contrastive_text import train_text_embedding

    generator = torch.Generator().manual_seed(2)
    num_articles, article_length = 6, 12
    token_stream = torch.randint(0, 32, (num_articles * article_length,), generator=generator)
    article_starts = np.arange(num_articles) * article_length
    rng = np.random.default_rng(0)
    steps, batch = 5, 4
    schedule = {
        "article": np.stack(
            [rng.choice(num_articles, size=batch, replace=False) for _ in range(steps)]
        ),
        "query_start": rng.integers(0, 7, size=(steps, batch)),
        "passage_start": rng.integers(0, 5, size=(steps, batch)),
    }
    out = {}
    for name, dimensions in (("plain", None), ("matryoshka", (8, 16))):
        torch.manual_seed(0)
        model = tiny_text_embedding()
        history = train_text_embedding(
            model,
            token_stream,
            article_starts,
            schedule,
            query_length=4,
            passage_length=6,
            temperature=0.05,
            matryoshka_dimensions=dimensions,
            evaluation_steps=(2,),
            evaluation_fn=evaluation_probe,
            **common_hyperparameters(),
        )
        out[name] = {"history": history, "state": model.state_dict()}
    return out


def case_contrastive_text_variable():
    import numpy as np
    import torch

    from src.training.contrastive_text_variable import (
        VariableBatches,
        train_text_embedding_from_batches,
    )

    rng = np.random.default_rng(4)
    steps, batch, query_width, passage_width = 5, 4, 5, 7
    query_tokens = rng.integers(0, 32, size=(steps, batch, query_width))
    passage_tokens = rng.integers(0, 32, size=(steps, batch, passage_width))
    query_lengths = rng.integers(2, query_width + 1, size=(steps, batch))
    passage_lengths = rng.integers(2, passage_width + 1, size=(steps, batch))
    article_ids = rng.integers(0, 3, size=(steps, batch))
    device = torch.device("cpu")
    out = {}
    for name, masked, dimensions in (("plain", False, None), ("masked", True, (8, 16))):
        batches = VariableBatches(
            query_tokens,
            query_lengths,
            passage_tokens,
            passage_lengths,
            device,
            article_ids=article_ids if masked else None,
            mask_same_article=masked,
        )
        torch.manual_seed(0)
        model = tiny_text_embedding()
        history = train_text_embedding_from_batches(
            model,
            batches,
            temperature=0.05,
            matryoshka_dimensions=dimensions,
            evaluation_steps=(2,),
            evaluation_fn=evaluation_probe,
            **common_hyperparameters(),
        )
        out[name] = {"history": history, "state": model.state_dict(), "digest": batches.digest}
    return out


def case_reranking():
    import numpy as np
    import torch

    from src.data.reranking import RerankingSchedule
    from src.models.cross_encoder import CrossEncoder
    from src.training.reranking import train_reranker

    generator = torch.Generator().manual_seed(5)
    query_tokens = torch.randint(3, 32, (10, 4), generator=generator)
    passage_tokens = torch.randint(3, 32, (30, 6), generator=generator)
    rng = np.random.default_rng(6)
    steps, batch, negatives = 5, 3, 3
    schedule = RerankingSchedule(
        query=rng.integers(0, 10, size=(steps, batch)),
        positive=rng.integers(0, 30, size=(steps, batch)),
        hard_negatives=rng.integers(0, 30, size=(steps, batch, negatives)),
        random_negatives=rng.integers(0, 30, size=(steps, batch, negatives)),
        num_same_article_removed=0,
        num_mined_considered=0,
        num_fallback_queries=0,
    )
    out = {}
    for kind in ("cross_encoder", "dual_encoder"):
        for negative_kind in ("hard", "random"):
            torch.manual_seed(0)
            if kind == "cross_encoder":
                model = CrossEncoder(tiny_gpt(), separator_id=1, terminal_id=2)
            else:
                model = tiny_text_embedding()
            history = train_reranker(
                model,
                kind,
                query_tokens,
                passage_tokens,
                schedule,
                negative_kind,
                temperature=0.05,
                evaluation_steps=(2,),
                evaluation_fn=evaluation_probe,
                **common_hyperparameters(),
            )
            out[f"{kind}_{negative_kind}"] = {"history": history, "state": model.state_dict()}
    return out


def case_reranking_variable():
    import numpy as np
    import torch

    from src.models.variable_length import VariableLengthCrossEncoder
    from src.training.reranking_variable import (
        ClozeRerankingSchedule,
        VariableTables,
        train_reranker_variable,
    )

    rng = np.random.default_rng(8)
    query_tokens = rng.integers(3, 32, size=(10, 5))
    passage_tokens = rng.integers(3, 32, size=(30, 7))
    tables = VariableTables(
        query_tokens,
        rng.integers(2, 6, size=10),
        passage_tokens,
        rng.integers(2, 8, size=30),
        torch.device("cpu"),
    )
    steps, batch, width = 5, 3, 4
    schedule = ClozeRerankingSchedule(
        query_ids=rng.integers(0, 10, size=(steps, batch)),
        candidate_ids=rng.integers(0, 30, size=(steps, batch, width)),
        num_usable=10,
    )
    torch.manual_seed(0)
    model = VariableLengthCrossEncoder(tiny_gpt(), separator_id=1, terminal_id=2)
    history = train_reranker_variable(
        model,
        tables,
        schedule,
        evaluation_steps=(2,),
        evaluation_fn=evaluation_probe,
        **common_hyperparameters(),
    )
    return {"history": history, "state": model.state_dict()}


CASES = {
    "bootstrap": case_bootstrap,
    "ranking": case_ranking,
    "moved_functions": case_moved_functions,
    "classification": case_classification,
    "contrastive": case_contrastive,
    "contrastive_text": case_contrastive_text,
    "contrastive_text_variable": case_contrastive_text_variable,
    "reranking": case_reranking,
    "reranking_variable": case_reranking_variable,
}


def run_case_in_subprocess(root: Path, name: str) -> object:
    """``root`` を ``src`` の取得元にして、サブプロセスで ``name`` の試験を計算する。"""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.pkl"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-case",
            name,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{root} の {name} が失敗した:\n{completed.stderr[-3000:]}")
        return pickle.loads(output.read_bytes())


def first_difference(a, b, path: str = "") -> str | None:
    """2 つの正規化済みの値の最初の差の位置(同じなら ``None``)。"""
    if type(a) is not type(b):
        return f"{path}: 型が違う"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: キーが違う {sorted(set(a) ^ set(b))}"
        for key in a:
            difference = first_difference(a[key], b[key], f"{path}/{key}")
            if difference:
                return difference
        return None
    if isinstance(a, list | tuple):
        if len(a) != len(b):
            return f"{path}: 長さが違う {len(a)} と {len(b)}"
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            difference = first_difference(x, y, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if a == b else f"{path}: 値が違う"


def reference_loss_scale_block(optimizers, gradient_norm, loss_scaler, history):
    """統合の前のループに書かれていた、更新と損失スケールの記述(そのまま写したもの)。"""
    import torch

    found_inf = False
    if loss_scaler is not None:
        found_inf = not bool(torch.isfinite(gradient_norm))
    if not found_inf:
        for optimizer in optimizers:
            optimizer.step()
    if loss_scaler is not None:
        history["loss_scale"].append(loss_scaler.scale)
        loss_scaler.update(found_inf)
    else:
        history["loss_scale"].append(1.0)
    return found_inf


def test_loss_scale_block() -> None:
    """``step_optimizers_and_update_loss_scale`` が、写した参照の記述と一致することを確かめる。

    損失スケーリングのあり・なしと、有限・非有限の勾配の組み合わせで比べる。
    """
    import torch

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from src.training.precision import DynamicLossScaler
    from src.training.update_step import (
        build_weight_decay_split_optimizers,
        clip_gradients_to_threshold,
        flag_clipped_steps,
        step_optimizers_and_update_loss_scale,
    )

    def run(use_helper: bool, use_scaler: bool, make_nonfinite_at: int):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        optimizers, parameters = build_weight_decay_split_optimizers(model, 1e-2, 0.1)
        scaler = DynamicLossScaler(2.0**4, growth_interval=2) if use_scaler else None
        history = {"loss_scale": []}
        skipped, norms = [], []
        for step in range(7):
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss = model(torch.randn(3, 4)).pow(2).mean()
            (scaler.scale_loss(loss) if scaler is not None else loss).backward()
            if step == make_nonfinite_at:
                parameters[0].grad.fill_(float("inf"))
            from src.training.update_step import unscale_gradients_and_compute_norm

            norm = unscale_gradients_and_compute_norm(
                parameters, scaler.scale if scaler is not None else None
            )
            threshold = torch.tensor(0.05)
            norms.append(float(norm))
            clip_gradients_to_threshold(parameters, norm, threshold)
            if use_helper:
                found = step_optimizers_and_update_loss_scale(optimizers, norm, scaler, history)
            else:
                found = reference_loss_scale_block(optimizers, norm, scaler, history)
            skipped.append(found)
        clipped = flag_clipped_steps(norms, 0.05)
        return normalize((skipped, history, clipped, model.state_dict(), norms))

    for use_scaler in (False, True):
        for nonfinite in (-1, 3):
            difference = first_difference(
                run(False, use_scaler, nonfinite), run(True, use_scaler, nonfinite)
            )
            assert difference is None, f"損失スケールの記述が一致しない: {difference}"
    print("一致 loss_scale_block(損失スケーリングあり・なし × 非有限の勾配あり・なし)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="e8706ec", help="参照のコミット")
    parser.add_argument("--run-case", help="(内部用)サブプロセスで 1 つの試験を計算する")
    parser.add_argument("--output", help="(内部用)結果の書き出し先")
    arguments = parser.parse_args()

    if arguments.run_case:
        sys.path.insert(0, str(Path.cwd()))
        import src

        assert Path(src.__file__).resolve().is_relative_to(Path.cwd().resolve()), (
            "src の取得元が違う"
        )
        result = normalize(CASES[arguments.run_case]())
        Path(arguments.output).write_bytes(pickle.dumps(result))
        return 0

    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "reference"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(reference), arguments.base],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        failures = 0
        try:
            for name in CASES:
                first = run_case_in_subprocess(reference, name)
                second = run_case_in_subprocess(reference, name)
                deterministic = first_difference(first, second)
                current = run_case_in_subprocess(REPOSITORY_ROOT, name)
                difference = first_difference(first, current)
                status = "一致" if difference is None else f"不一致 {difference}"
                note = (
                    "" if deterministic is None else f"(参照の 2 回の実行が不一致: {deterministic})"
                )
                print(f"{status:<6} {name} {note}")
                failures += difference is not None or deterministic is not None
            test_loss_scale_block()
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(reference)],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
            )
    print("すべて一致" if failures == 0 else f"不一致 {failures} 件")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
