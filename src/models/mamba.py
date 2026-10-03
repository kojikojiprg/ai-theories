"""Mamba による自己回帰言語モデルのスクラッチ実装(023)。

トークン埋め込み → [RMSNorm → ``MambaBlock`` → 残差接続] の積層 → 最終 RMSNorm → 出力層
(語彙への射影)。正規化前置(Pre-Layer Normalization)、埋め込みと出力層の重み共有、
埋め込みの初期化(標準偏差 0.02)は、008 の ``GPTLanguageModel`` に揃えた。

**``GPTLanguageModel`` と揃えなかった点**:

- 位置エンコーディングを持たない(再帰の構造が順序の情報を担う)。
- 順伝播ネットワークを持たない(``MambaBlock`` が注意機構と順伝播ネットワークの両方の役割を
  担う。原論文は同じパラメータ数で Transformer の約 2 倍の層数を積む)。
- 層の初期化は各層の既定のまま(公式実装が行う、出力射影の重みの 1 / sqrt(層数) への縮小は
  行わない)。
- 推論では、系列の長さに比例して増える KV キャッシュの代わりに、層ごとの定数サイズの状態
  (畳み込みのバッファと ``h``)を持ち回る(``step()``・``generate()``)。

Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
arXiv:2312.00752, 2023。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from src.layers.mamba import MambaBlock, MambaState
from src.layers.normalization import RMSNorm


class MambaLanguageModel(nn.Module):
    """Mamba の自己回帰言語モデル。

    Args:
        vocabulary_size: 語彙サイズ。
        d_model: モデルの隠れ次元。
        num_layers: ``MambaBlock`` の層数。
        state_dim, expand, conv_kernel, delta_rank, selective, discretization, use_checkpoint:
            ``MambaBlock`` にそのまま渡す。
        normalization_factory: 正規化層の factory(既定は ``RMSNorm``)。
        tie_embeddings: ``True`` なら埋め込み行列と出力層の重みを共有する。
    """

    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        delta_rank: int | None = None,
        selective: bool = True,
        discretization: str = "zero_order_hold",
        use_checkpoint: bool = False,
        normalization_factory: Callable[[int], nn.Module] | None = None,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.num_layers = num_layers
        norm_fn = normalization_factory or RMSNorm
        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)  # 008 と同じ
        self.norms = nn.ModuleList([norm_fn(d_model) for _ in range(num_layers)])
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    state_dim=state_dim,
                    expand=expand,
                    conv_kernel=conv_kernel,
                    delta_rank=delta_rank,
                    selective=selective,
                    discretization=discretization,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = norm_fn(d_model)
        self.lm_head = nn.Linear(d_model, vocabulary_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        token_ids: Tensor,
        states: list[MambaState] | None = None,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[MambaState]]:
        """順伝播。

        Args:
            token_ids: 形状 ``(batch, length)`` の LongTensor。
            states: 層ごとの直前の状態(``None`` なら 0)。続きを処理するときに使う。
            return_states: ``True`` なら ``(logits, 層ごとの最後の状態)`` を返す。

        Returns:
            形状 ``(batch, length, vocabulary_size)`` の logits(``return_states=True``
            なら状態との組)。
        """
        hidden = self.token_embedding(token_ids)
        new_states: list[MambaState] = []
        for i, (norm, block) in enumerate(zip(self.norms, self.blocks, strict=True)):
            initial = None if states is None else states[i]
            out, state = block(norm(hidden), initial_state=initial, return_state=True)
            hidden = hidden + out
            new_states.append(state)
        logits = self.lm_head(self.final_norm(hidden))
        return (logits, new_states) if return_states else logits

    def initial_states(self, batch_size: int, device: torch.device) -> list[MambaState]:
        """層ごとのゼロの初期状態を返す。"""
        dtype = self.token_embedding.weight.dtype
        return [block.initial_state(batch_size, device, dtype) for block in self.blocks]

    def step(self, token_ids: Tensor, states: list[MambaState]) -> tuple[Tensor, list[MambaState]]:
        """1 トークンぶんの推論。

        Args:
            token_ids: 形状 ``(batch,)`` の LongTensor。
            states: 層ごとの直前の状態。

        Returns:
            ``(logits (batch, vocabulary_size), 新しい状態)``。
        """
        hidden = self.token_embedding(token_ids)
        new_states: list[MambaState] = []
        for norm, block, state in zip(self.norms, self.blocks, states, strict=True):
            out, new_state = block.step(norm(hidden), state)
            hidden = hidden + out
            new_states.append(new_state)
        return self.lm_head(self.final_norm(hidden)), new_states

    def auxiliary_losses(self) -> dict[str, Tensor]:
        """補助損失を持たない(``train_language_model()`` との互換のため)。"""
        return {}

    def set_delta_tracking(self, enabled: bool) -> None:
        """各層の直近の順伝播の ``Δ``(チャネルについての平均)を、``last_delta_mean`` に記録するか。

        Args:
            enabled: ``True`` なら記録する(選択的な構成のブロックのみ)。
        """
        for block in self.blocks:
            block.track_delta = enabled
            if not enabled:
                block.last_delta_mean = None

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
        use_state: bool = True,
    ) -> Tensor:
        """自己回帰的にトークンを生成する。

        ``use_state=True``(既定値)では、起点の文脈を 1 回の順伝播で処理して層ごとの状態を作り、
        以降は 1 トークンずつ ``step()`` で状態を更新する(1 トークンあたりの計算量が文脈の
        長さによらず一定)。``use_state=False`` では、生成ステップごとに文脈全体を再計算する。
        ``temperature=0.0`` の貪欲法の生成結果は、どちらでも一致する(単体テストで確かめる)。

        Args:
            token_ids: 形状 ``(batch, length_0)`` の LongTensor(生成の起点となる文脈)。
            max_new_tokens: 生成するトークン数。
            temperature: サンプリング温度。``0.0`` で貪欲法。
            seed: サンプリングの乱数シード。
            use_state: 状態を持ち回って生成するか。

        Returns:
            形状 ``(batch, length_0 + max_new_tokens)`` の LongTensor。
        """
        if temperature < 0.0:
            raise ValueError(f"temperature は 0 以上である必要がある: {temperature}")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=token_ids.device)
            generator.manual_seed(seed)

        def _sample(next_token_logits: Tensor) -> Tensor:
            if temperature == 0.0:
                return next_token_logits.argmax(dim=-1, keepdim=True)
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1, generator=generator)

        was_training = self.training
        self.eval()
        try:
            if not use_state:
                for _ in range(max_new_tokens):
                    next_token = _sample(self(token_ids)[:, -1, :])
                    token_ids = torch.cat([token_ids, next_token], dim=1)
                return token_ids
            if max_new_tokens < 1:
                return token_ids
            logits, states = self(token_ids, return_states=True)
            next_token = _sample(logits[:, -1, :])
            token_ids = torch.cat([token_ids, next_token], dim=1)
            for _ in range(max_new_tokens - 1):
                logits, states = self.step(token_ids[:, -1], states)
                next_token = _sample(logits)
                token_ids = torch.cat([token_ids, next_token], dim=1)
        finally:
            self.train(was_training)
        return token_ids
