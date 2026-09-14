"""GPT(Generative Pre-trained Transformer)スタイルの decoder-only 自己回帰
言語モデル(Autoregressive Language Model)のスクラッチ実装。

Radford et al., "Improving Language Understanding by Generative Pre-Training", 2018
(GPT)、および Radford et al., "Language Models are Unsupervised Multitask Learners",
2019(GPT-2)の decoder-only アーキテクチャに基づく。トークン埋め込み → 因果マスク
(causal mask)付き Transformer Block(``DecoderBlock``、``use_cross_attention=False``)
の積層 → 最終正規化層 → 出力層(語彙への射影)という構成を、001〜004 でスクラッチ
実装した部品(``MultiHeadAttention``・``DecoderBlock``・``LayerNormalization``)を
再利用して組み立てる。

**正規化前置(Pre-Layer Normalization)を採用する**(``DecoderBlock`` の
``norm_first=True``、002 の知見に従う。深い層でも学習が安定するため)。

**位置エンコーディングは注入方式のみを使う**(``positional_transform``、例えば
003 の ``RotaryPositionEmbedding``)。002 のような正弦波位置エンコーディングや
学習可能な絶対位置埋め込みをモジュール内部に持たない。``positional_transform``
が ``None``(既定値)の場合、モデルは明示的な位置情報を一切持たない(因果マスクに
よる非対称性のみが位置の手がかりになる)。呼び出し側が明示的に位置エンコーディング
方式を選ぶ設計であり、003 の「``self_attn`` 属性を ``positional_transform`` を
指定した ``MultiHeadAttention`` に差し替える」手法をレイヤーごとに適用することで
実現する。

**重み共有(tie_embeddings)**: トークン埋め込み行列と出力層(``lm_head``)の重みを
共有できる(Press & Wolf, "Using the Output Embedding to Improve Language Models",
EACL 2017、Inan et al., "Tying Word Vectors and Word Classifiers", ICLR 2017)。
入力埋め込みと出力射影がいずれも「トークン ID <-> d_model 次元ベクトル」の対応を
学習する点で役割が近く、重みを共有することでパラメータ数を削減しつつ性能を
落とさないことが示されている。

記号 / Notation:
    V         : 語彙サイズ(vocabulary_size)
    d_model   : モデルの隠れ次元
    d_ff      : Feed-Forward Network の中間層次元
    h         : Multi-Head Attention のヘッド数(num_heads)
    L         : Transformer Block の層数(num_layers)
    B         : バッチサイズ
    S         : 系列長(sequence length)
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn

from src.generation.cache import KeyValueCache
from src.layers.attention import MultiHeadAttention, create_causal_mask
from src.layers.normalization import LayerNormalization
from src.layers.positional_encoding import QueryKeyPositionalTransform
from src.layers.transformer_block import DecoderBlock


class GPTLanguageModel(nn.Module):
    """decoder-only Transformer による自己回帰言語モデル(GPT スタイル)。

    構成: トークン埋め込み -> 因果マスク付き ``DecoderBlock``
    (``use_cross_attention=False``)を ``num_layers`` 段 -> 最終正規化層 ->
    出力層(語彙への射影)。

    Args:
        vocabulary_size: 語彙サイズ V。
        d_model: モデルの隠れ次元。
        num_layers: ``DecoderBlock`` の層数 L。
        num_heads: Multi-Head Attention のヘッド数 h。
        d_ff: Feed-Forward Network の中間層次元。
        max_sequence_length: モデルが扱う最大系列長。``forward`` に渡す系列長
            (``token_ids`` の第 2 次元)がこれを超える場合はエラーにする。
        positional_transform: 各層の ``self_attn`` に注入する位置変換
            (003 の ``QueryKeyPositionalTransform`` のサブクラス、例:
            ``RotaryPositionEmbedding``)。全層で同一インスタンスを共有する
            (RoPE は学習パラメータを持たないため共有して問題ない)。``None``
            (既定値)の場合、位置エンコーディングを一切注入しない。
        normalization_factory: ``DecoderBlock``・最終正規化層にそのまま渡す
            正規化層の factory(004 参照)。``None``(既定値)の場合は
            ``LayerNormalization`` を使う。
        feed_forward_factory: ``DecoderBlock`` にそのまま渡す順伝播ネットワークの
            factory(004 参照)。``None``(既定値)の場合は ReLU を使う
            標準の Feed-Forward Network になる。
        tie_embeddings: True(既定値)の場合、トークン埋め込み行列と出力層
            (``lm_head``)の重みを共有する(Press & Wolf 2017、Inan et al. 2017)。
        dropout: Attention 重み・各サブレイヤー出力に適用する dropout 率。
        norm_first: True(既定値)で正規化前置(Pre-Layer Normalization)、False で
            正規化後置(Post-Layer Normalization、007 で追加)。006 までは
            ``DecoderBlock`` に常に ``norm_first=True`` を渡していた(002 の知見に
            従い、深い層でも学習が安定するため)。007 では、002 で観測した
            正規化後置の勾配の不均衡・004 で観測した正規化欠如条件でのシード間
            ばらつきの増大を踏まえ、意図的に不安定性を誘発する条件として
            ``norm_first=False`` を使う。``DecoderBlock`` 自体は 004 の時点で
            ``norm_first`` 引数を既に持っており(``src/layers/transformer_block.py``)、
            本引数はそれを ``GPTLanguageModel`` の呼び出し側に透過するのみである。
        num_key_value_heads: 各層の ``MultiHeadAttention`` に渡す Key / Value
            ヘッド数(010 で追加、``num_heads`` を割り切れる必要がある)。``None``
            (既定値)の場合 ``num_key_value_heads = num_heads``(通常の多頭注意機構、
            001〜009 と完全に同一の挙動)。``num_key_value_heads < num_heads`` の
            とき Grouped-Query Attention / Multi-Query Attention になる
            (``convert_attention_to_grouped_query`` で既存モデルから変換する
            場合はこの引数を直接使わず、変換後のモデルの各層が既にこのヘッド数で
            構築されている)。
    """

    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_sequence_length: int,
        positional_transform: QueryKeyPositionalTransform | None = None,
        normalization_factory: Callable[[int], nn.Module] | None = None,
        feed_forward_factory: Callable[[], nn.Module] | None = None,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
        norm_first: bool = True,
        num_key_value_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_heads
        )

        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        # nn.Embedding の既定初期化(標準正規分布、標準偏差 1)は次元でスケールされて
        # おらず、そのままでは埋め込みベクトルのノルムが大きくなりすぎる
        # (norm ~ sqrt(d_model))。tie_embeddings=True の場合、この行列がそのまま
        # 出力層 lm_head の重みにもなるため、スケールの悪い初期化が「ランダム初期化
        # 直後の損失が ln(V) に一致するはず」という前提(実験 A)を崩す
        # (実測: 標準偏差 1 のままだと V=2045 で loss ≈ 54 対 ln(V) ≈ 7.6 という
        # 大きな乖離が生じた)。GPT-2(Radford et al., 2019)にならい標準偏差 0.02 の
        # 正規分布で初期化する。
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                self._build_block(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    normalization_factory,
                    feed_forward_factory,
                    positional_transform,
                    norm_first,
                    num_key_value_heads,
                )
                for _ in range(num_layers)
            ]
        )
        norm_fn = normalization_factory or LayerNormalization
        self.final_norm = norm_fn(d_model)
        self.lm_head = nn.Linear(d_model, vocabulary_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _build_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        normalization_factory: Callable[[int], nn.Module] | None,
        feed_forward_factory: Callable[[], nn.Module] | None,
        positional_transform: QueryKeyPositionalTransform | None,
        norm_first: bool = True,
        num_key_value_heads: int | None = None,
    ) -> DecoderBlock:
        """交差注意を持たない ``DecoderBlock`` を 1 層分構築する。

        ``positional_transform``・``num_key_value_heads``(010 で追加)の少なくとも
        一方が指定された場合、003 と同じ手法(``self_attn`` 属性をこれらを注入した
        ``MultiHeadAttention`` に差し替える)で組み込む。両方とも ``None`` の場合、
        ``DecoderBlock`` が既定で構築する ``self_attn``(通常の多頭注意機構)を
        そのまま使う(001〜009 と完全に同一の乱数消費・挙動になる)。``DecoderBlock``
        自体は変更しない。
        """
        block = DecoderBlock(
            d_model,
            num_heads,
            d_ff,
            dropout=dropout,
            norm_first=norm_first,
            normalization_factory=normalization_factory,
            feed_forward_factory=feed_forward_factory,
            use_cross_attention=False,
        )
        if positional_transform is not None or num_key_value_heads is not None:
            block.self_attn = MultiHeadAttention(
                d_model,
                num_heads,
                dropout=dropout,
                positional_transform=positional_transform,
                num_key_value_heads=num_key_value_heads,
            )
        return block

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor | None = None,
        kv_cache: KeyValueCache | None = None,
    ) -> Tensor:
        """順伝播。

        Args:
            token_ids: 形状 ``(B, S)`` の LongTensor(トークン ID 列)。``kv_cache``
                が ``None`` または空の場合、``S`` は ``max_sequence_length`` 以下
                である必要がある(``kv_cache`` が空でない場合の制約は
                ``kv_cache`` の説明を参照)。
            positions: 絶対位置インデックス(形状 ``(S,)``)。``positional_transform``
                に透過する(``None`` のときは 0 から S - 1 までの連番)。
            kv_cache: KV キャッシュ(``KeyValueCache``、010 で追加)。``None``
                (既定値)の場合、001〜009 と完全に同一の計算になる。指定する場合、
                **``kv_cache`` が空のとき(最初の呼び出し、prefill)は ``token_ids``
                に複数トークンを渡せる**(通常の因果マスクを構築する)。**``kv_cache``
                が空でないとき(2 回目以降の呼び出し、decode)は ``token_ids`` は
                1 トークン(``S = 1``)のみを渡す想定**(``GPTLanguageModel.generate``
                の decode ループを参照)。この場合、渡された 1 トークンはキャッシュ
                済みの全トークンより後ろの位置にあるため、マスクなし(全キャッシュに
                参加)で正しい計算になる。空でない ``kv_cache`` に複数トークンを渡すと、
                本実装は因果マスクを構築しないため不正な(未来を参照する)計算になる。

        Returns:
            形状 ``(B, S, V)`` の logits。
        """
        _, seq_len = token_ids.shape
        cache_is_empty = kv_cache is None or kv_cache.length == 0
        if cache_is_empty and seq_len > self.max_sequence_length:
            raise ValueError(
                f"系列長 ({seq_len}) が max_sequence_length "
                f"({self.max_sequence_length}) を超えている"
            )

        hidden = self.token_embedding(token_ids)
        # kv_cache が空でない(decode ステップ)場合、新規トークンは既存キャッシュより
        # 常に後ろの位置にあるため、マスクなし(全キャッシュに参加)で正しい計算になる。
        # マスクが必要なのは、複数トークンをまとめて渡す prefill(kv_cache が空、
        # または kv_cache=None の通常計算)のときのみ。
        causal_mask = (
            None if not cache_is_empty else create_causal_mask(seq_len, device=token_ids.device)
        )
        for i, block in enumerate(self.blocks):
            hidden, _, _ = block(
                hidden,
                None,
                tgt_mask=causal_mask,
                positions=positions,
                kv_cache=kv_cache,
                layer_idx=i,
            )
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        """自己回帰的にトークンを生成する。

        ``temperature=0.0`` の場合は貪欲法(greedy decoding、常に最尤トークンを選ぶ)、
        それ以外は temperature sampling(``softmax(logits / temperature)`` から
        多項分布サンプリング)を行う。top-k・top-p(nucleus sampling)・beam search は
        実装しない(008 で扱う)。

        ``use_cache=False``(既定値)の場合、生成ステップごとに直近
        ``max_sequence_length`` トークンの文脈全体を再計算する(006〜008 と完全に
        同一の挙動)。``use_cache=True``(010 で追加)の場合、``KeyValueCache``
        (``src/generation/cache.py``)を使い、初回のみ起点の文脈全体を
        prefill(まとめて 1 回の順伝播でキャッシュを構築)し、以降は新規トークン
        1 個のみを順伝播する(``GPTLanguageModel.forward`` の ``kv_cache`` 引数の
        decode ステップの制約を参照)。同じ ``temperature``・``seed`` の下で、
        ``use_cache`` の有無によらず貪欲法(``temperature=0.0``)の生成結果は
        完全に一致する(010 実験 A の前提となる不変条件)。

        Args:
            token_ids: 形状 ``(B, S_0)`` の LongTensor(生成の起点となる文脈)。
            max_new_tokens: 生成するトークン数。
            temperature: サンプリング温度。``0.0`` で貪欲法になる。
            seed: サンプリングに使う乱数シード(省略可、再現性のため)。
                ``temperature=0.0`` のときは使われない。
            use_cache: True の場合 KV キャッシュを使って生成する(010 で追加)。
                ``use_cache=True`` の場合、``S_0 + max_new_tokens`` が
                ``max_sequence_length`` を超えるとエラーにする(``use_cache=False``
                のような、直近 ``max_sequence_length`` トークンへの自動的な
                トランケーションは行わない。KV キャッシュはキャッシュした全トークンを
                保持し続ける設計のため、スライディングウィンドウでの切り詰めは
                本トピックの対象外である)。

        Returns:
            形状 ``(B, S_0 + max_new_tokens)`` の LongTensor
            (``token_ids`` に生成トークンを連結したもの)。
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
            if not use_cache:
                for _ in range(max_new_tokens):
                    context = token_ids[:, -self.max_sequence_length :]
                    logits = self(context)
                    next_token = _sample(logits[:, -1, :])
                    token_ids = torch.cat([token_ids, next_token], dim=1)
                return token_ids

            initial_length = token_ids.size(1)
            if initial_length + max_new_tokens > self.max_sequence_length:
                raise ValueError(
                    "use_cache=True では、生成後の系列長"
                    f"({initial_length + max_new_tokens})が max_sequence_length"
                    f"({self.max_sequence_length})を超える生成には対応しない"
                    "(スライディングウィンドウでの切り詰めは対象外)"
                )
            if max_new_tokens < 1:
                return token_ids

            kv_cache = KeyValueCache(self.num_layers)

            # prefill: 起点の文脈全体を 1 回の順伝播でキャッシュに書き込む。
            prefill_positions = torch.arange(initial_length, device=token_ids.device)
            logits = self(token_ids, positions=prefill_positions, kv_cache=kv_cache)
            next_token = _sample(logits[:, -1, :])
            token_ids = torch.cat([token_ids, next_token], dim=1)

            # decode: 以降は新規トークン 1 個のみを順伝播する。
            for _ in range(max_new_tokens - 1):
                step_position = torch.tensor([kv_cache.length], device=token_ids.device)
                logits = self(token_ids[:, -1:], positions=step_position, kv_cache=kv_cache)
                next_token = _sample(logits[:, -1, :])
                token_ids = torch.cat([token_ids, next_token], dim=1)
        finally:
            self.train(was_training)
        return token_ids


def _mean_pool_key_value_weight(
    weight: Tensor, num_heads: int, num_key_value_heads: int, d_k: int
) -> Tensor:
    """多頭注意機構の ``w_k``・``w_v`` 重み(形状 ``(num_heads * d_k, d_model)``)を、
    ``num_heads / num_key_value_heads`` ヘッドずつのグループで平均プールし、形状
    ``(num_key_value_heads * d_k, d_model)`` に縮小する(Ainslie et al., EMNLP 2023
    の uptraining、理論セクション参照)。グループ ``i`` は連続するヘッド
    ``[i * group_size, (i+1) * group_size)`` から作る(``MultiHeadAttention.forward``
    の Grouped-Query Attention 展開(``repeat_interleave``)と対応するグループ分けを
    揃える必要がある)。
    """
    d_model = weight.size(-1)
    group_size = num_heads // num_key_value_heads
    reshaped = weight.view(num_heads, d_k, d_model)
    grouped = reshaped.view(num_key_value_heads, group_size, d_k, d_model).mean(dim=1)
    return grouped.reshape(num_key_value_heads * d_k, d_model).contiguous()


def convert_attention_to_grouped_query(
    model: GPTLanguageModel,
    num_key_value_heads: int,
    init: str,
    seed: int | None = None,
) -> GPTLanguageModel:
    """既存の多頭注意機構チェックポイントから、Key / Value ヘッド数を削減した
    Grouped-Query Attention(``1 < num_key_value_heads < num_heads``)または
    Multi-Query Attention(``num_key_value_heads = 1``)モデルへ変換する(010、
    Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from
    Multi-Head Checkpoints", EMNLP 2023)。

    ``Query`` 射影(``w_q``)・出力射影(``w_o``)、および Attention 以外の全ての
    重み(埋め込み・Feed-Forward Network・正規化層)は変更せずそのままコピーする。
    変更するのは各層の ``w_k``・``w_v``(形状が ``num_heads * d_k`` から
    ``num_key_value_heads * d_k`` へ縮小する)のみである。

    Args:
        model: 変換元の学習済み ``GPTLanguageModel``(``num_key_value_heads`` が
            ``num_heads`` と等しい、通常の多頭注意機構であることを想定)。
        num_key_value_heads: 変換後の Key / Value ヘッド数。``model`` の
            ``num_heads`` を割り切れる必要がある。
        init: ``"mean_pool"`` の場合、各グループ内の元のヘッドの重みを平均プールして
            初期化する(uptraining の初期化、追加学習により全ヘッドの平均から
            グループごとに特殊化していくことを期待する手法)。``"random"`` の場合、
            Xavier 一様分布によるランダム初期化にする(平均プール初期化の効果を
            測る対照条件、010 実験 D)。
        seed: ``init="random"`` のときの乱数シード(``init="mean_pool"`` のときは
            使われない、決定的な処理のため)。

    Returns:
        変換後の新しい ``GPTLanguageModel`` インスタンス(``model`` 自体は変更しない)。

    Raises:
        ValueError: ``init`` が ``"mean_pool"``・``"random"`` のいずれでもない場合、
            または ``num_heads`` が ``num_key_value_heads`` で割り切れない場合。
    """
    if init not in ("mean_pool", "random"):
        raise ValueError(f'init は "mean_pool" または "random" である必要がある: {init!r}')
    if model.num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_heads ({model.num_heads}) は num_key_value_heads "
            f"({num_key_value_heads}) で割り切れる必要がある"
        )

    new_model = copy.deepcopy(model)
    new_model.num_key_value_heads = num_key_value_heads
    generator = None
    if init == "random" and seed is not None:
        generator = torch.Generator().manual_seed(seed)

    for block in new_model.blocks:
        old_attn: MultiHeadAttention = block.self_attn
        num_heads = old_attn.num_heads
        d_k = old_attn.d_k
        new_attn = MultiHeadAttention(
            old_attn.d_model,
            num_heads,
            dropout=old_attn.dropout.p,
            bias=old_attn.w_q.bias is not None,
            positional_transform=old_attn.positional_transform,
            attention_score_bias=old_attn.attention_score_bias,
            num_key_value_heads=num_key_value_heads,
        )
        new_attn.w_q.load_state_dict(old_attn.w_q.state_dict())
        new_attn.w_o.load_state_dict(old_attn.w_o.state_dict())

        if init == "mean_pool":
            with torch.no_grad():
                new_attn.w_k.weight.copy_(
                    _mean_pool_key_value_weight(
                        old_attn.w_k.weight.data, num_heads, num_key_value_heads, d_k
                    )
                )
                new_attn.w_v.weight.copy_(
                    _mean_pool_key_value_weight(
                        old_attn.w_v.weight.data, num_heads, num_key_value_heads, d_k
                    )
                )
                if old_attn.w_k.bias is not None:
                    new_attn.w_k.bias.copy_(
                        _mean_pool_key_value_weight(
                            old_attn.w_k.bias.data[:, None], num_heads, num_key_value_heads, 1
                        ).squeeze(-1)
                    )
                    new_attn.w_v.bias.copy_(
                        _mean_pool_key_value_weight(
                            old_attn.w_v.bias.data[:, None], num_heads, num_key_value_heads, 1
                        ).squeeze(-1)
                    )
        else:  # init == "random"
            with torch.no_grad():
                nn.init.xavier_uniform_(new_attn.w_k.weight, generator=generator)
                nn.init.xavier_uniform_(new_attn.w_v.weight, generator=generator)
                if old_attn.w_k.bias is not None:
                    nn.init.zeros_(new_attn.w_k.bias)
                    nn.init.zeros_(new_attn.w_v.bias)

        block.self_attn = new_attn

    return new_model
