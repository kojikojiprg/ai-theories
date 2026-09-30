"""画像分類のデータ(CIFAR-10)の取得・分割・部分集合と、GPU 上のデータ拡張。

CIFAR-10(Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", Technical Report,
2009)は、32 × 32 画素のカラー画像 60,000 枚(訓練 50,000 枚・テスト 10,000 枚、10 クラス、
各クラス同数)からなる。取得は torchvision に委ね、キャッシュは ``.cache/cifar10/`` に置く
(入力から決定的に再生成でき、失われても再取得するだけで済むため)。

Google Colab の無料枠は CPU の数が少なく、``DataLoader`` のワーカーでの前処理が律速になりやすい。
そこで、データ全体を uint8 のまま GPU のテンソルに載せ、ミニバッチの取り出し・データ拡張
(パディング 4 の random crop と左右反転)・正規化をすべて GPU 上のテンソル演算で行う。
データの順序とデータ拡張の乱数は CPU の ``torch.Generator`` から引く(初期化の乱数とは別の
生成器にし、デバイスによらず同じ順序・同じ拡張になるようにする)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
# 訓練集合 50,000 枚のチャネルごとの平均・標準偏差(広く使われている値)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_CIFAR10_CACHE_DIR = Path(".cache") / "cifar10"


@dataclass
class ImageClassificationData:
    """画像分類のデータ一式(画像は uint8、形状 ``(num, C, H, W)``)。"""

    train_images: Tensor
    train_labels: Tensor
    test_images: Tensor
    test_labels: Tensor


def load_cifar10(cache_dir: str | Path = DEFAULT_CIFAR10_CACHE_DIR) -> ImageClassificationData:
    """CIFAR-10 を取得して uint8 のテンソルで返す(初回のみダウンロードする)。

    torchvision の ``CIFAR10(download=True)`` は、アーカイブの MD5 を照合してから展開する。
    キャッシュが既にあれば、照合のうえで再ダウンロードしない。

    Args:
        cache_dir: キャッシュのディレクトリ(既定値 ``.cache/cifar10``)。

    Returns:
        訓練 50,000 枚・テスト 10,000 枚の画像(``(num, 3, 32, 32)``、uint8)とラベル(int64)。
    """
    from torchvision.datasets import CIFAR10  # torchvision はこの関数でのみ使う

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for split, train in (("train", True), ("test", False)):
        dataset = CIFAR10(root=str(cache_dir), train=train, download=True)
        images = torch.from_numpy(np.ascontiguousarray(dataset.data)).permute(0, 3, 1, 2)
        tensors[f"{split}_images"] = images.contiguous()
        tensors[f"{split}_labels"] = torch.tensor(dataset.targets, dtype=torch.int64)
    return ImageClassificationData(**tensors)


def split_class_balanced_validation(
    labels: np.ndarray, per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """訓練集合から、各クラス ``per_class`` 枚の検証集合を切り出す。

    Args:
        labels: 訓練集合のラベル(形状 ``(num,)``)。
        per_class: 検証集合に入れる各クラスの枚数。
        seed: 切り出しの乱数シード(学習のシードとは独立に固定する)。

    Returns:
        (学習に使う添字, 検証集合の添字)。どちらも昇順に並べた int64 の配列。
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    validation = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if len(members) < per_class:
            raise ValueError(
                f"クラス {c} の枚数 {len(members)} が per_class={per_class} より少ない"
            )
        validation.append(rng.permutation(members)[:per_class])
    validation_indices = np.sort(np.concatenate(validation)).astype(np.int64)
    train_indices = np.setdiff1d(np.arange(len(labels)), validation_indices).astype(np.int64)
    return train_indices, validation_indices


def make_nested_class_balanced_subsets(
    labels: np.ndarray, pool_indices: np.ndarray, fractions: tuple[float, ...], seed: int
) -> dict[float, np.ndarray]:
    """クラス均衡で入れ子の部分集合を作る(小さい部分集合は大きい部分集合に含まれる)。

    各クラスの添字を 1 回だけ乱数で並べ替え、割合 f の部分集合にはその先頭
    ``floor(f * n_c)`` 枚を入れる(n_c はクラス c の枚数)。並べ替えを全割合で共有するので、
    部分集合は割合の順に入れ子になる。

    Args:
        labels: 全体のラベル(``pool_indices`` の添字で引く)。
        pool_indices: 部分集合を選ぶ元の添字(検証集合を除いた訓練集合など)。
        fractions: 割合の組(各値は (0, 1])。
        seed: 部分集合を選ぶ乱数シード(学習のシードとは独立に固定する)。

    Returns:
        割合 -> 部分集合の添字(昇順、int64)の辞書。
    """
    labels = np.asarray(labels)
    pool_indices = np.asarray(pool_indices)
    rng = np.random.default_rng(seed)
    pool_labels = labels[pool_indices]
    orders = {c: rng.permutation(pool_indices[pool_labels == c]) for c in np.unique(pool_labels)}
    subsets = {}
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"割合は (0, 1] の範囲: {fraction}")
        chosen = [order[: int(np.floor(fraction * len(order)))] for order in orders.values()]
        subsets[fraction] = np.sort(np.concatenate(chosen)).astype(np.int64)
    return subsets


class EpochShuffledBatchSampler:
    """エポックごとに並べ替えた添字を、エポックの境界をまたいで一定の大きさのバッチに切り出す。

    データの枚数がバッチサイズで割り切れない場合も、すべてのバッチが同じ大きさになる
    (エポックの末尾の端数は次のエポックの先頭とつながる)。1 エポックの中では各事例を
    ちょうど 1 回ずつ使う。

    Args:
        num_examples: データの枚数。
        batch_size: バッチサイズ。
        generator: 並べ替えに使う CPU の ``torch.Generator``。
    """

    def __init__(self, num_examples: int, batch_size: int, generator: torch.Generator) -> None:
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.generator = generator
        self._buffer = torch.empty(0, dtype=torch.int64)
        self.epochs_started = 0

    def next_batch(self) -> Tensor:
        """次のバッチの添字(0 以上 ``num_examples`` 未満、CPU の int64)を返す。"""
        while self._buffer.numel() < self.batch_size:
            permutation = torch.randperm(self.num_examples, generator=self.generator)
            self._buffer = torch.cat([self._buffer, permutation])
            self.epochs_started += 1
        batch, self._buffer = self._buffer[: self.batch_size], self._buffer[self.batch_size :]
        return batch


def normalize_images(images: Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD) -> Tensor:
    """[0, 1] の画素値(float)をチャネルごとに標準化する。"""
    mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device)[None, :, None, None]
    std_t = torch.tensor(std, dtype=images.dtype, device=images.device)[None, :, None, None]
    return (images - mean_t) / std_t


def to_normalized_float(images_uint8: Tensor) -> Tensor:
    """uint8 の画像を [0, 1] の float32 にして標準化する(データ拡張なし、評価用)。"""
    return normalize_images(images_uint8.float() / 255.0)


def random_crop_and_flip(images_uint8: Tensor, padding: int, generator: torch.Generator) -> Tensor:
    """パディング ``padding`` の random crop と左右反転を GPU 上で行い、標準化して返す。

    画像の周囲を 0(黒)で ``padding`` 画素ずつ埋め、元の大きさの窓を一様に選んで切り出す
    (torchvision の ``RandomCrop(size, padding)`` と同じ分布)。その後、確率 1/2 で左右を
    反転する。窓の位置と反転の有無は CPU の ``generator`` から引き、画像のあるデバイスへ送る。

    Args:
        images_uint8: 形状 ``(B, C, H, W)`` の uint8 の画像(GPU 上にあってよい)。
        padding: 各辺に足す画素数。
        generator: CPU の ``torch.Generator``。

    Returns:
        形状 ``(B, C, H, W)`` の標準化した float32 の画像。
    """
    batch_size, channels, height, width = images_uint8.shape
    offsets = torch.randint(0, 2 * padding + 1, (batch_size, 2), generator=generator)
    flips = torch.rand(batch_size, generator=generator) < 0.5
    device = images_uint8.device
    offsets = offsets.to(device, non_blocking=True)
    flips = flips.to(device, non_blocking=True)

    images = images_uint8.float() / 255.0
    padded = functional.pad(images, (padding, padding, padding, padding))  # 0 で埋める
    rows = offsets[:, 0, None] + torch.arange(height, device=device)  # (B, H)
    cols = offsets[:, 1, None] + torch.arange(width, device=device)  # (B, W)
    batch_index = torch.arange(batch_size, device=device)[:, None, None, None]
    channel_index = torch.arange(channels, device=device)[None, :, None, None]
    cropped = padded[batch_index, channel_index, rows[:, None, :, None], cols[:, None, None, :]]
    flipped = torch.where(flips[:, None, None, None], cropped.flip(-1), cropped)
    return normalize_images(flipped)
