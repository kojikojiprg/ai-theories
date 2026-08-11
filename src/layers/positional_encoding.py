"""位置エンコーディング(Positional Encoding)のスクラッチ実装。

Transformer Block(Multi-Head Attention + Feed-Forward Network)は入力の並び替えに
対して置換同変(permutation equivariant)であり、単体では系列の順序を区別できない。
位置エンコーディングは、この順序情報をモデルに注入するための仕組みである。

本モジュールには、注入点の異なる複数方式を統一的なインターフェースの下に実装する
(詳細は`theories/01_foundations/003_positional_encoding_rope.ipynb`の理論・実装方針を参照)。

- 入力埋め込みへの加算方式: ``SinusoidalPositionalEncoding``(002 で導入)、
  ``LearnedAbsolutePositionalEmbedding``(003 で追加)
- Query・Key を変換する方式(``QueryKeyPositionalTransform`` のサブクラス):
  ``RotaryPositionEmbedding``(003 で追加)
- Attention スコアにバイアスを加える方式(``AttentionScoreBias`` のサブクラス):
  ``ShawRelativePositionBias``、``T5RelativePositionBias``、``ALiBiPositionBias``
  (いずれも 003 で追加)

記号 / Notation:
    pos, m, n : 系列中の位置(0-indexed)。m は Query 側、n は Key 側を表す。
    i         : 次元インデックス
    d_model   : モデルの隠れ次元
    d_k       : 1 ヘッドあたりの Query / Key の次元
    h         : ヘッド数
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """正弦波(sinusoidal)位置エンコーディング。

    Vaswani et al., "Attention Is All You Need", NeurIPS 2017, 3.5 節の定義:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    各位置 pos に対して長さ d_model のベクトル PE(pos, :) を 1 つ定め、
    トークン埋め込みに加算して使う。位置ごとに異なる周波数の sin / cos を
    偶数・奇数次元に交互に割り当てることで、各位置が一意なパターンを持つ。

    Args:
        d_model: モデルの隠れ次元。
        max_len: 事前に計算しておく位置の最大数。
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        # 10000^(2i/d_model) を exp/log で計算(オーバーフロー回避のため対数空間で計算)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model / 2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 学習パラメータではないが device / dtype をモデルと合わせたいので buffer として登録する。
        # (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """入力の埋め込みに位置エンコーディングを加算する。

        Args:
            x: 形状 ``(B, S, d_model)`` のトークン埋め込み。``S <= max_len`` が必要。

        Returns:
            x と同じ形状の、位置エンコーディングを加算したテンソル。
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class LearnedAbsolutePositionalEmbedding(nn.Module):
    """学習可能な絶対位置埋め込み(Learned Absolute Positional Embedding)。

    位置 m ごとに独立な埋め込みベクトル(行列 ``P`` の m 行目)を学習し、
    入力埋め込みに加算する。BERT(Devlin et al., NAACL 2019)などが採用する方式。

    正弦波方式と異なりパラメータ数は d_model * max_len に比例し、max_len を
    超える位置に対応する行が存在しないため、原理的に長さの外挿(length extrapolation)
    ができない(詳細はノートブック 003 の理論セクション参照)。

    Args:
        d_model: モデルの隠れ次元。
        max_len: 学習時に想定する最大系列長 L_max。
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embeddings = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """入力埋め込みに位置 m の埋め込み P_m を加算する。

        Args:
            x: 形状 ``(B, S, d_model)`` のトークン埋め込み。
            positions: 形状 ``(S,)`` の絶対位置インデックス。``None`` のときは
                0 から S - 1 までの連番として扱う。

        Returns:
            x と同じ形状のテンソル。
        """
        seq_len = x.size(1)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        if int(positions.max().item()) >= self.max_len:
            raise ValueError(
                f"position が max_len ({self.max_len}) を超えている: "
                f"max(positions) = {int(positions.max().item())}"
            )
        return x + self.position_embeddings[positions]


class QueryKeyPositionalTransform(nn.Module, ABC):
    """Query・Key を、内積を取る前に位置に応じて変換するインターフェース。

    Attention スコア e_mn = q_m · k_n / sqrt(d_k) を計算する前に、Query・Key
    そのものへ位置情報を作用させる方式(RoPE 等)の共通インターフェース。
    """

    @abstractmethod
    def apply(
        self, query: Tensor, key: Tensor, positions: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Query・Key に位置変換を適用する。

        Args:
            query: 形状 ``(..., S_q, d_k)`` の Query。
            key: 形状 ``(..., S_k, d_k)`` の Key。
            positions: Query 側の絶対位置インデックス(形状 ``(S_q,)``)。``None``
                のときは 0 から S_q - 1 までの連番として扱う。KV キャッシュを
                用いた逐次推論(トピック 010)では、生成の各ステップで Query の
                絶対位置がキャッシュ長だけずれるため、これを外部から指定できる
                ようにしている。

        Returns:
            変換後の (query, key) のタプル。形状は入力と同じ。
        """
        raise NotImplementedError


class AttentionScoreBias(nn.Module, ABC):
    """Attention スコアに位置バイアスを加えるインターフェース。

    Attention スコア e_mn = q_m · k_n / sqrt(d_k) の計算後、softmax を取る前に
    加算する、位置のみに依存するバイアス項(T5・ALiBi 等)の共通インターフェース。
    """

    @abstractmethod
    def bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Attention スコアに加えるバイアス行列を返す。

        Args:
            query_length: Query 側の系列長 S_q。
            key_length: Key 側の系列長 S_k。
            device: 生成先デバイス。
            dtype: 生成するテンソルの dtype。

        Returns:
            形状 ``(h, S_q, S_k)`` または ``(1, S_q, S_k)``(ブロードキャスト可能)
            のバイアス行列。
        """
        raise NotImplementedError


class RotaryPositionEmbedding(QueryKeyPositionalTransform):
    """RoPE(Rotary Position Embedding、Su et al., Neurocomputing 2024)。

    Query・Key を d_k / 2 個の 2 次元部分空間に分割し、部分空間 i を角周波数
    theta_i = base^(-2i / d_k) で位置 m だけ回転させる。内積 (R_m q)^T (R_n k) は
    相対位置 n - m のみに依存する(導出はノートブック 003 の理論セクション参照)。

    効率的な実装として、ブロック対角の回転行列との明示的な行列積は行わず、
    次元を前半・後半に分割して入れ替える ``rotate_half`` と cos / sin の
    要素ごとの積で等価な計算を行う(GPT-NeoX / LLaMA 系の実装で広く使われる
    変形で、部分空間の次元ペアの取り方(i 番目と i + d_k/2 番目を組にする)が
    原論文の隣接ペア(2i 番目と 2i+1 番目)と異なるだけで、Q・K に同一の置換を
    適用するため内積の理論的性質(相対位置のみへの依存)は保たれる)。

    記号 / Notation:
        d_k  : ヘッドあたりの次元(偶数である必要がある)
        base : 角周波数の基数(既定値 10000)
        m, n : 位置インデックス(m: Query 側、n: Key 側)
        theta_i (i = 0, ..., d_k/2 - 1) : 部分空間 i の角周波数

    Args:
        d_k: ヘッドあたりの次元。偶数である必要がある。
        base: 角周波数の基数。原論文の既定値である 10000 を用いる。base を
            変更すると部分空間ごとの回転速度の分布が変わり、これは長文脈拡張
            (トピック 014)のスケーリング手法群の出発点になる。
        max_position: cos / sin を事前計算しておく最大位置。これを超える位置が
            要求された場合は自動的に再計算して拡張する。
    """

    def __init__(self, d_k: int, base: float = 10000.0, max_position: int = 2048) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k は偶数である必要がある: {d_k}")
        self.d_k = d_k
        self.base = base

        inv_freq = base ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)  # (d_k/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = 0
        self._build_cache(max_position)

    def _set_buffer(self, name: str, tensor: Tensor) -> None:
        if name in self._buffers:
            del self._buffers[name]
        self.register_buffer(name, tensor, persistent=False)

    def _build_cache(self, max_position: int) -> None:
        positions = torch.arange(max_position, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)  # (max_position, d_k/2)
        # rotate_half 実装との対応のため、前半・後半に同じ周波数を複製する。
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_position, d_k)
        self._set_buffer("cos_cached", emb.cos())
        self._set_buffer("sin_cached", emb.sin())
        self.max_position = max_position

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """(..., d_k) の後半を反転して前半と入れ替える: [x1, x2] -> [-x2, x1]。"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply(
        self, query: Tensor, key: Tensor, positions: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        seq_len = query.size(-2)
        if positions is None:
            positions = torch.arange(seq_len, device=query.device)

        max_pos = int(positions.max().item()) + 1
        if max_pos > self.max_position:
            self._build_cache(max_pos)

        cos = self.cos_cached[positions].to(dtype=query.dtype, device=query.device)
        sin = self.sin_cached[positions].to(dtype=query.dtype, device=query.device)

        q = query * cos + self._rotate_half(query) * sin
        k = key * cos + self._rotate_half(key) * sin
        return q, k


class ShawRelativePositionBias(AttentionScoreBias):
    """Shaw et al. (NAACL 2018) 方式の相対位置エンコーディング(Key 側の項のみ)。

    原論文の定義は e_mn = (x_m W^Q)(x_n W^K + a^K_mn)^T / sqrt(d_k) であり、
    a^K_mn = w^K_{clip(n-m, k_clip)}(clip は相対距離を [-k_clip, k_clip] に
    切り詰める関数)。展開すると e_mn = q_m・k_n/sqrt(d_k) + q_m・a^K_mn/sqrt(d_k)
    となり、第 2 項は Query の内容 q_m に依存する。この内容依存性のため、
    位置のみに依存する``AttentionScoreBias.bias()``(T5・ALiBi と共通のインター
    フェース)では計算できず、``relative_vectors()`` で相対位置ベクトル a^K_mn の
    みを返し、``MultiHeadAttention`` 側で Query との内積を直接計算する(詳細は
    ノートブック 003 の実装方針セクション参照)。

    原論文は Value 側にも a^V_mn を加算する定式化(第 2 項は Attention 重みで
    Value を集約する式に直接介入するため、スコアへの加算という枠組みに収まらない)
    だが、後続研究では寄与が小さいとして省略されることが多く、本モジュールでは
    実装しない(ノートブック 003 内で検証コードとして直接実装する)。

    記号 / Notation:
        d_k          : ヘッドあたりの次元
        k_clip       : 考慮する最大相対距離(max_relative_position)
        m, n         : 位置インデックス(m: Query 側、n: Key 側)
        w^K          : 学習可能パラメータ、形状 (2 * k_clip + 1, d_k)

    Args:
        d_k: ヘッドあたりの次元。
        max_relative_position: 最大相対距離 k_clip。これを超える相対距離は
            符号を保ったまま k_clip に切り詰められる。
    """

    def __init__(self, d_k: int, max_relative_position: int = 16) -> None:
        super().__init__()
        self.d_k = d_k
        self.max_relative_position = max_relative_position
        num_positions = 2 * max_relative_position + 1
        self.relative_key_embeddings = nn.Parameter(torch.zeros(num_positions, d_k))
        nn.init.xavier_uniform_(self.relative_key_embeddings)

    def relative_vectors(
        self,
        query_length: int,
        key_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """相対位置ベクトル a^K_mn を形状 ``(S_q, S_k, d_k)`` で返す。"""
        m = torch.arange(query_length, device=device)[:, None]
        n = torch.arange(key_length, device=device)[None, :]
        relative_position = torch.clamp(
            n - m, -self.max_relative_position, self.max_relative_position
        )
        indices = relative_position + self.max_relative_position
        return self.relative_key_embeddings[indices].to(dtype=dtype)

    def bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        raise NotImplementedError(
            "ShawRelativePositionBias の a^K_mn は Query の内容 q_m に依存するため、"
            "位置のみに依存する bias() では計算できない。MultiHeadAttention は "
            "isinstance 判定により relative_vectors() を用いて Query との内積を "
            "直接計算する。"
        )


class T5RelativePositionBias(AttentionScoreBias):
    """T5(Raffel et al., JMLR 2020)の相対位置バイアス。

    Attention スコアに、ヘッドごとに学習可能なスカラー b_{h, bucket(n-m)} を
    加算する。相対距離 n - m を対数スケールでバケット化することで、近距離は
    細かく、遠距離は粗く扱い、少ないパラメータで長距離の相対位置も表現できる。
    Shaw et al. 方式との違いは、(1) 加算する項がベクトルではなくスカラーである
    こと、(2) 相対距離をバケット化すること、(3) 層間でパラメータを共有すること
    が多いこと(このクラス自体は 1 層分のバイアスのみを持つ)。

    記号 / Notation:
        h             : ヘッド数(num_heads)
        num_buckets   : バケット数
        max_distance  : バケット化の対数スケールが有効な最大距離
        bidirectional : True なら双方向(Encoder の自己注意)、False なら
                        因果的(Decoder の自己注意)なバケット化を行う

    Args:
        num_heads: ヘッド数 h。
        num_buckets: バケット数。
        max_distance: これを超える相対距離は最後のバケットに丸められる。
        bidirectional: 双方向注意かどうか。
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor,
        bidirectional: bool,
        num_buckets: int,
        max_distance: int,
    ) -> Tensor:
        """相対距離 n - m を対数スケールのバケット番号に変換する(T5 原実装のアルゴリズム)。"""
        relative_buckets = torch.zeros_like(relative_position)
        if bidirectional:
            num_buckets //= 2
            relative_buckets = relative_buckets + (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # 半分のバケットは距離をそのまま使う(近距離、線形スケール)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 残り半分は対数スケールで max_distance まで広げる(遠距離)
        relative_position_if_large = (
            max_exact
            + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets = relative_buckets + torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def relative_position_bucket(self, relative_position: Tensor) -> Tensor:
        """相対距離 n - m を、このインスタンスの設定(num_buckets・max_distance・
        bidirectional)でバケット番号に変換する(`_relative_position_bucket` の
        public ラッパー。ノートブックなど外部からバケット番号を確認したい場合に使う)。
        """
        return self._relative_position_bucket(
            relative_position, self.bidirectional, self.num_buckets, self.max_distance
        )

    def bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        context_position = torch.arange(query_length, device=device)[:, None]  # m
        memory_position = torch.arange(key_length, device=device)[None, :]  # n
        relative_position = memory_position - context_position  # n - m, (S_q, S_k)
        relative_position_bucket = self.relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)  # (S_q, S_k, h)
        values = values.permute(2, 0, 1)  # (h, S_q, S_k)
        return values.to(dtype=dtype)


class ALiBiPositionBias(AttentionScoreBias):
    """ALiBi(Attention with Linear Biases、Press et al., ICLR 2022)。

    Attention スコアに -m_h * (m - n) を加算する。m_h はヘッド h ごとの固定の
    傾き(学習可能パラメータを持たない)で、幾何数列
    m_h = 2^(-8h/H) (h = 1, ..., H、H = num_heads) として定める。ヘッドごとに
    異なる傾きを持たせることで、ヘッドごとに異なる距離スケールを担当させる。

    学習可能パラメータを持たないため、任意の系列長にそのまま適用でき、外挿
    (length extrapolation)を明示的な設計目標とした手法である(原論文タイトル
    "Train Short, Test Long" の通り)。

    記号 / Notation:
        H    : ヘッド数(num_heads)
        m_h  : ヘッド h の傾き
        m, n : 位置インデックス(m: Query 側、n: Key 側)

    Args:
        num_heads: ヘッド数 H。
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(self._compute_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _compute_slopes(num_heads: int) -> list[float]:
        """m_h = 2^(-8h/H) の幾何数列を計算する。

        H が 2 のべき乗でない場合は、原論文の公式実装に従い、直近の 2 のべき乗
        で計算した数列を補間して残りのヘッド分を埋める。
        """

        def _power_of_2_slopes(n: int) -> list[float]:
            start = 2.0 ** (-8.0 / n)
            return [start ** (h + 1) for h in range(n)]

        if math.log2(num_heads).is_integer():
            return _power_of_2_slopes(num_heads)

        closest_pow2 = 2 ** math.floor(math.log2(num_heads))
        base_slopes = _power_of_2_slopes(closest_pow2)
        extra_slopes = _power_of_2_slopes(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
        return base_slopes + extra_slopes

    def bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        m = torch.arange(query_length, device=device, dtype=torch.float32)[:, None]
        n = torch.arange(key_length, device=device, dtype=torch.float32)[None, :]
        relative_position = m - n  # (S_q, S_k)
        slopes = self.slopes.to(device=device)
        bias = -slopes[:, None, None] * relative_position[None, :, :]  # (h, S_q, S_k)
        return bias.to(dtype=dtype)
