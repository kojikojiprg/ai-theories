"""2 つの図形からなる合成のシーンと、規則で生成するキャプション(020)。

32 x 32 の RGB 画像に、色と形を持つ図形を 2 つ、左右または上下に並べて描き、その内容を
``a red circle left of a blue square`` のような規則のキャプションで表す。CLIP
(Radford et al., ICML 2021)の対照学習と、bag-of-words 化(Yuksekgonul et al., ICLR 2023)の
検証のためのデータである。

**シーンの意味**: シーンの意味は ``(relation, first, second)`` の 3 つ組で決まる。``relation`` は
``"left of"``(左右の配置)または ``"above"``(上下の配置)、``first``・``second`` は
``(color, shape)`` の組で、``first`` が左(上)、``second`` が右(下)の図形である。
キャプションは意味から一意に決まる **正準形** とし、``"right of"``・``"below"`` は使わない。
そのため、キャプションの文字列が異なることと意味が異なることは同値になり、「A left of B」と
「B right of A」のような同義の言い換えが負例に混ざらない(``caption_meaning()`` で確かめる)。

**2 つの図形は色も形も異なる**: 色が同じなら色の入れ替えが、形も色も同じなら順序の入れ替えが
元のキャプションと同じ文字列になり、困難な負例(hard negative)にならないためである。

**描画**: numpy でシーンのまとまりごとにベクトル化して描く。乱数はすべて
``np.random.default_rng(seed)`` から引くので、同じシードからは bit 単位で同じ画像が再生成される。

**未見の組み合わせ**: (色, 形) の組の一部 ``H`` を学習データのどちらの図形にも出さず、評価集合にのみ
含める(``holdout_pairs()``)。各色・各形は ``H`` の外の組で学習データに現れるので、未見なのは
「組み合わせ」だけである。

**困難な負例の生成規則**(``swap_attributes()``・``swap_order()``):

- 属性の入れ替え: 2 つの図形の色を交換する(形と位置関係はそのまま)。
- 順序の入れ替え: 関係語を保ったまま 2 つの図形を交換する。

どちらも元のキャプションと同じ単語の多重集合(bag of words)を持つので、語順と属性の結びつきを
区別しないモデルには元のキャプションと見分けられない。

記号 / Notation:
    K : 色の数、S : 形の数、R : 位置関係の数(2)
    H : 学習データから除外する (色, 形) の組の集合
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import Tensor

IMAGE_SIZE = 32

# 色(RGB の基準値)。描画では図形ごとに各チャネルへ一様な揺らぎを加える
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (50, 90, 230),
    "yellow": (230, 215, 40),
    "magenta": (210, 50, 210),
    "cyan": (40, 205, 215),
    "orange": (245, 140, 30),
    "white": (235, 235, 235),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "diamond", "cross")
RELATIONS: tuple[str, ...] = ("left of", "above")  # 0: 左右の配置、1: 上下の配置(正準形のみ)
COLOR_NAMES: tuple[str, ...] = tuple(COLORS)

PAD_TOKEN, START_TOKEN, END_TOKEN = "<pad>", "<sot>", "<eot>"
FUNCTION_WORDS: tuple[str, ...] = ("a", "left", "of", "above")


@dataclass(frozen=True)
class SceneMeaning:
    """シーンの意味。``first``・``second`` は (色の番号, 形の番号)。``first`` が左(上)の図形。"""

    relation: int
    first: tuple[int, int]
    second: tuple[int, int]


def caption_text(meaning: SceneMeaning) -> str:
    """意味から正準形のキャプションを作る(例: ``a red circle left of a blue square``)。"""
    (c1, s1), (c2, s2) = meaning.first, meaning.second
    return (
        f"a {COLOR_NAMES[c1]} {SHAPES[s1]} {RELATIONS[meaning.relation]} "
        f"a {COLOR_NAMES[c2]} {SHAPES[s2]}"
    )


def caption_meaning(caption: str) -> SceneMeaning:
    """キャプションを意味に戻す。

    ``"right of"``・``"below"`` も受け付け、左右・上下を入れ替えた正準形の意味に直す
    (「B right of A」は「A left of B」と同じ意味になる)。同義の言い換えを判定するために使う。
    """
    words = caption.split()
    if words[0] != "a" or "a" not in words[1:]:
        raise ValueError(f"キャプションの形式が違う: {caption!r}")
    second_a = words.index("a", 1)
    first_words, relation_words, second_words = words[1:3], words[3:second_a], words[second_a + 1 :]
    if len(first_words) != 2 or len(second_words) != 2:
        raise ValueError(f"キャプションの形式が違う: {caption!r}")
    first = (COLOR_NAMES.index(first_words[0]), SHAPES.index(first_words[1]))
    second = (COLOR_NAMES.index(second_words[0]), SHAPES.index(second_words[1]))
    relation_text = " ".join(relation_words)
    if relation_text in RELATIONS:
        return SceneMeaning(RELATIONS.index(relation_text), first, second)
    if relation_text == "right of":
        return SceneMeaning(RELATIONS.index("left of"), second, first)
    if relation_text == "below":
        return SceneMeaning(RELATIONS.index("above"), second, first)
    raise ValueError(f"未知の関係語: {relation_text!r}")


def is_valid_meaning(meaning: SceneMeaning) -> bool:
    """2 つの図形の色も形も異なること(困難な負例が元と異なる文字列になる条件)。"""
    return meaning.first[0] != meaning.second[0] and meaning.first[1] != meaning.second[1]


def enumerate_meanings() -> list[SceneMeaning]:
    """有効な意味をすべて列挙する(関係 -> 1 つ目の図形 -> 2 つ目の図形の辞書式順)。"""
    objects = list(product(range(len(COLOR_NAMES)), range(len(SHAPES))))
    meanings = [
        SceneMeaning(r, a, b) for r in range(len(RELATIONS)) for a in objects for b in objects
    ]
    return [m for m in meanings if is_valid_meaning(m)]


def swap_attributes(meaning: SceneMeaning) -> SceneMeaning:
    """属性の入れ替え: 2 つの図形の色を交換する(形・位置関係はそのまま)。"""
    (c1, s1), (c2, s2) = meaning.first, meaning.second
    return SceneMeaning(meaning.relation, (c2, s1), (c1, s2))


def swap_order(meaning: SceneMeaning) -> SceneMeaning:
    """順序の入れ替え: 関係語を保ったまま 2 つの図形を交換する。"""
    return SceneMeaning(meaning.relation, meaning.second, meaning.first)


HARD_NEGATIVE_RULES = {"attribute_swap": swap_attributes, "order_swap": swap_order}


def holdout_pairs() -> frozenset[tuple[int, int]]:
    """学習データから除外する (色, 形) の組 ``H``。

    色 ``i`` に形 ``i mod S`` を対応させる(``i = 0, ..., K - 1``)。K = 8・S = 5 では 40 組のうち
    8 組(20%)になり、各色がちょうど 1 回ずつ ``H`` に入る。各色は残りの S - 1 個の形と、各形は
    少なくとも K - 2 個の色と、学習データで組になる。
    """
    return frozenset((i, i % len(SHAPES)) for i in range(len(COLOR_NAMES)))


class CaptionVocabulary:
    """規則の集合から決まる単語単位のトークナイザ(Word-level Tokenizer)。

    語彙は特殊トークン(``<pad>``・``<sot>``・``<eot>``)、機能語、色、形の順に固定する。
    符号化の結果は ``[<sot>, 単語..., <eot>, <pad>...]`` で、長さ ``context_length`` にそろえる。
    """

    def __init__(self, context_length: int = 10) -> None:
        self.tokens: tuple[str, ...] = (
            PAD_TOKEN,
            START_TOKEN,
            END_TOKEN,
            *FUNCTION_WORDS,
            *COLOR_NAMES,
            *SHAPES,
        )
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        assert len(self.token_to_id) == len(self.tokens), "語彙に重複がある"
        self.context_length = context_length
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.start_id = self.token_to_id[START_TOKEN]
        self.end_id = self.token_to_id[END_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, caption: str) -> list[int]:
        ids = [self.start_id] + [self.token_to_id[w] for w in caption.split()] + [self.end_id]
        if len(ids) > self.context_length:
            raise ValueError(f"系列長 {len(ids)} が context_length {self.context_length} を超える")
        return ids + [self.pad_id] * (self.context_length - len(ids))

    def decode(self, ids: Sequence[int]) -> str:
        words = []
        for i in ids:
            token = self.tokens[int(i)]
            if token == START_TOKEN:
                continue
            if token == END_TOKEN:
                break
            words.append(token)
        return " ".join(words)

    def encode_batch(self, captions: Sequence[str]) -> Tensor:
        """キャプションの列を形状 ``(len(captions), context_length)`` の int64 テンソルにする。"""
        return torch.tensor([self.encode(c) for c in captions], dtype=torch.int64)


@dataclass
class CaptionUniverse:
    """全キャプションの一覧と、学習用・未見の組み合わせの評価用への分割。

    Attributes:
        meanings: 有効な意味の全列挙(添字がキャプションの番号)。
        captions: 各意味の正準形のキャプション。
        token_ids: 形状 ``(num_captions, context_length)`` の符号化済みの系列。
        holdout: 除外する (色, 形) の組 ``H``。
        train_caption_ids: どちらの図形も ``H`` に入らないキャプションの番号(昇順)。
        unseen_caption_ids: 少なくとも一方の図形が ``H`` に入るキャプションの番号(昇順)。
        hard_negative_ids: 規則名 -> 各キャプションの困難な負例のキャプションの番号
            (形状 ``(num_captions,)``)。
    """

    meanings: list[SceneMeaning]
    captions: list[str]
    token_ids: Tensor
    holdout: frozenset[tuple[int, int]]
    train_caption_ids: np.ndarray
    unseen_caption_ids: np.ndarray
    hard_negative_ids: dict[str, np.ndarray]

    @property
    def num_captions(self) -> int:
        return len(self.meanings)


def build_caption_universe(vocabulary: CaptionVocabulary) -> CaptionUniverse:
    """全キャプションを列挙し、未見の組み合わせで分割し、困難な負例の対応表を作る。"""
    meanings = enumerate_meanings()
    captions = [caption_text(m) for m in meanings]
    index = {m: i for i, m in enumerate(meanings)}
    holdout = holdout_pairs()
    uses_holdout = np.array([m.first in holdout or m.second in holdout for m in meanings])
    hard_negative_ids = {
        name: np.array([index[rule(m)] for m in meanings], dtype=np.int64)
        for name, rule in HARD_NEGATIVE_RULES.items()
    }
    return CaptionUniverse(
        meanings=meanings,
        captions=captions,
        token_ids=vocabulary.encode_batch(captions),
        holdout=holdout,
        train_caption_ids=np.flatnonzero(~uses_holdout).astype(np.int64),
        unseen_caption_ids=np.flatnonzero(uses_holdout).astype(np.int64),
        hard_negative_ids=hard_negative_ids,
    )


@dataclass
class SceneSet:
    """描画したシーンのまとまり。

    Attributes:
        images: 形状 ``(num, 3, 32, 32)`` の uint8 の画像。
        caption_ids: 各画像のキャプションの番号(``CaptionUniverse.meanings`` の添字)。
        visible_pixels: 形状 ``(num, 2)``。描画後に見えている各図形の画素数(図形の重なりの確認用)。
    """

    images: Tensor
    caption_ids: np.ndarray
    visible_pixels: np.ndarray


# 図形の大きさ・位置の揺らぎ(画素)
RADIUS_RANGE = (4.5, 6.0)  # 図形の「半径」(形ごとの係数を掛けて大きさを決める)
GAP_RANGE = (14.0, 17.0)  # 2 つの図形の中心の間隔(並べる方向)
SHIFT_RANGE = 1.5  # 並べる方向の、2 つの図形の組全体のずれ
CROSS_AXIS_CENTER_RANGE = (9.0, 23.0)  # 並べる方向と直交する方向の中心
CROSS_AXIS_JITTER = 1.5  # 直交する方向の、図形ごとのずれ(2 つの差は最大 3 画素)
COLOR_JITTER = 18  # 各チャネルの一様な揺らぎの幅(±)
BACKGROUND_LEVEL_RANGE = (30, 90)  # 背景の灰色の水準
BACKGROUND_NOISE_STD = 10.0  # 背景と図形に加える画素ごとのガウス雑音の標準偏差


def _shape_masks(
    dx: np.ndarray, dy: np.ndarray, radius: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """塗る画素を決める。

    Args:
        dx, dy: 図形の中心からの変位(形状 ``(M, H, W)``)。
        radius, shape: 図形の「半径」と形の番号(形状 ``(M, 1, 1)``)。
    """
    ax, ay = np.abs(dx), np.abs(dy)
    circle = dx**2 + dy**2 <= radius**2
    square = np.maximum(ax, ay) <= 0.85 * radius
    # 上向きの三角形: 頂点が dy = -r、底辺が dy = 0.8 r、底辺の半幅 0.95 r
    triangle = (dy >= -radius) & (dy <= 0.8 * radius) & (ax <= (dy + radius) * (0.95 / 1.8))
    diamond = ax + ay <= 1.1 * radius
    arm = 0.38 * radius
    cross = ((ax <= arm) & (ay <= radius)) | ((ay <= arm) & (ax <= radius))
    stacked = np.stack([circle, square, triangle, diamond, cross])  # (S, M, H, W)
    assert stacked.shape[0] == len(SHAPES)
    return np.take_along_axis(stacked, shape[None, :, :, :].astype(np.int64), axis=0)[0]


def render_scenes(
    universe: CaptionUniverse, caption_ids: np.ndarray, seed: int, chunk_size: int = 2048
) -> SceneSet:
    """キャプションの番号の列から、それぞれの意味を満たすシーンを描く(numpy でベクトル化)。

    左右の配置では 1 つ目の図形の中心が 2 つ目より 14〜17 画素左にあり、上下の差は 3 画素以内。
    上下の配置はこれを転置した配置である。そのため、左右の配置のシーンに「above」は成り立たず、
    キャプションの関係語は配置から一意に決まる。

    Args:
        universe: ``build_caption_universe()`` の結果。
        caption_ids: 描くシーンのキャプションの番号(形状 ``(num,)``)。
        seed: 乱数シード(同じシードと同じ ``caption_ids`` からは同じ画像が再生成される)。
        chunk_size: 一度に描くシーンの数(メモリ量のみに影響し、結果には影響しない)。

    Returns:
        ``SceneSet``。
    """
    caption_ids = np.asarray(caption_ids, dtype=np.int64)
    num = len(caption_ids)
    rng = np.random.default_rng(seed)
    # 乱数は全シーン分をまとめて先に引く(chunk_size によらず同じ画像になる)
    relation = np.array([universe.meanings[i].relation for i in caption_ids])
    colors = np.array(
        [[universe.meanings[i].first[0], universe.meanings[i].second[0]] for i in caption_ids]
    )
    shapes = np.array(
        [[universe.meanings[i].first[1], universe.meanings[i].second[1]] for i in caption_ids]
    )
    radius = rng.uniform(*RADIUS_RANGE, size=(num, 2))
    gap = rng.uniform(*GAP_RANGE, size=num)
    shift = rng.uniform(-SHIFT_RANGE, SHIFT_RANGE, size=num)
    cross_center = rng.uniform(*CROSS_AXIS_CENTER_RANGE, size=num)
    cross_jitter = rng.uniform(-CROSS_AXIS_JITTER, CROSS_AXIS_JITTER, size=(num, 2))
    color_jitter = rng.integers(-COLOR_JITTER, COLOR_JITTER + 1, size=(num, 2, 3))
    background_level = rng.uniform(*BACKGROUND_LEVEL_RANGE, size=num)
    noise = rng.normal(0.0, BACKGROUND_NOISE_STD, size=(num, 3, IMAGE_SIZE, IMAGE_SIZE))

    # 並べる方向(左右なら x、上下なら y)の座標と、直交する方向の座標
    along = np.stack([16.0 - gap / 2 + shift, 16.0 + gap / 2 + shift], axis=1)  # (num, 2)
    across = cross_center[:, None] + cross_jitter  # (num, 2)
    horizontal = relation == RELATIONS.index("left of")
    center_x = np.where(horizontal[:, None], along, across)
    center_y = np.where(horizontal[:, None], across, along)
    base_colors = np.array(list(COLORS.values()), dtype=np.int64)
    object_rgb = np.clip(base_colors[colors] + color_jitter, 0, 255)  # (num, 2, 3)

    coordinate = np.arange(IMAGE_SIZE) + 0.5  # 画素の中心
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    images = np.empty((num, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    visible = np.empty((num, 2), dtype=np.int64)
    for start in range(0, num, chunk_size):
        sl = slice(start, min(start + chunk_size, num))
        m = sl.stop - sl.start
        canvas = np.broadcast_to(
            background_level[sl, None, None, None], (m, 3, IMAGE_SIZE, IMAGE_SIZE)
        ).copy()
        owner = np.full((m, IMAGE_SIZE, IMAGE_SIZE), -1, dtype=np.int64)
        for k in range(2):  # 2 つ目の図形を後に描く(重なった画素は 2 つ目が上になる)
            dx = xx[None] - center_x[sl, k, None, None]
            dy = yy[None] - center_y[sl, k, None, None]
            mask = _shape_masks(dx, dy, radius[sl, k, None, None], shapes[sl, k, None, None])
            canvas = np.where(
                mask[:, None], object_rgb[sl, k, :, None, None].astype(np.float64), canvas
            )
            owner = np.where(mask, k, owner)
        canvas = canvas + noise[sl]
        images[sl] = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
        visible[sl] = np.stack(
            [(owner == 0).sum(axis=(1, 2)), (owner == 1).sum(axis=(1, 2))], axis=1
        )
    return SceneSet(
        images=torch.from_numpy(images), caption_ids=caption_ids, visible_pixels=visible
    )


def repeat_caption_ids(caption_ids: np.ndarray, copies: int) -> np.ndarray:
    """各キャプションを ``copies`` 回ずつ並べた番号の列(同じ番号が連続する)。"""
    return np.repeat(np.asarray(caption_ids, dtype=np.int64), copies)


class DistinctCaptionBatchSampler:
    """各バッチのキャプションがすべて異なるように、シーンの添字のバッチを切り出す。

    キャプションの集合が小さい合成データでは、無作為に切り出したバッチに同じキャプションのシーンが
    複数入りやすい。対照学習ではそれらが互いの負例になり、正しいキャプションを負例として押し下げる
    (偽の負例、false negative)。バッチが大きいほど起きやすいので、バッチサイズの効果と交絡する。
    そこで、各ステップで学習用のキャプションから ``batch_size`` 個を重複なく一様に選び、選んだ
    キャプションごとに、そのキャプションのシーンを巡回的に 1 つずつ取り出す(各キャプションのシーンの
    順序は、使い切るたびに並べ替え直す)。

    シーンは ``scene_caption_ids`` に同じ数 ``copies`` ずつ、キャプションの番号順に並んでいることを
    前提とする(``repeat_caption_ids()`` の出力の順)。乱数はすべて CPU の ``generator`` から引き、
    モデルには依存しない。

    Args:
        scene_caption_ids: 各シーンのキャプションの番号。
        copies: キャプションあたりのシーンの数。
        batch_size: バッチサイズ N(学習用のキャプションの数以下)。
        generator: CPU の ``torch.Generator``。
    """

    def __init__(
        self,
        scene_caption_ids: np.ndarray,
        copies: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> None:
        scene_caption_ids = np.asarray(scene_caption_ids)
        num_captions = len(scene_caption_ids) // copies
        if num_captions * copies != len(scene_caption_ids):
            raise ValueError("シーンの数がキャプションあたりの数で割り切れない")
        grouped = scene_caption_ids.reshape(num_captions, copies)
        if not (grouped == grouped[:, :1]).all() or len(np.unique(grouped[:, 0])) != num_captions:
            raise ValueError("シーンがキャプションごとに連続して並んでいない")
        if batch_size > num_captions:
            raise ValueError(f"batch_size {batch_size} がキャプションの数 {num_captions} を超える")
        self.num_captions = num_captions
        self.copies = copies
        self.batch_size = batch_size
        self.generator = generator
        self._order = torch.stack(
            [torch.randperm(copies, generator=generator) for _ in range(num_captions)]
        )
        self._pointer = torch.zeros(num_captions, dtype=torch.int64)
        self.examples_drawn = 0

    def next_batch(self) -> Tensor:
        """次のバッチのシーンの添字(CPU の int64、形状 ``(batch_size,)``)を返す。"""
        chosen = torch.randperm(self.num_captions, generator=self.generator)[: self.batch_size]
        within = self._order[chosen, self._pointer[chosen]]
        self._pointer[chosen] += 1
        for c in chosen[
            self._pointer[chosen] == self.copies
        ].tolist():  # 使い切ったキャプションを並べ替え直す
            self._order[c] = torch.randperm(self.copies, generator=self.generator)
            self._pointer[c] = 0
        self.examples_drawn += self.batch_size
        return chosen * self.copies + within


def normalize_scene_images(images_uint8: Tensor) -> Tensor:
    """uint8 の画像を [-1, 1] の float32 にする(平均 0.5・標準偏差 0.5 の標準化)。"""
    return images_uint8.float() / 127.5 - 1.0
