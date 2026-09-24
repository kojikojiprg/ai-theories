"""言語モデルの事前学習ループ・評価関数のスクラッチ実装。

**006 の時点では意図的に素朴な設定に留めていた**(AdamW(Loshchilov & Hutter,
ICLR 2019)・学習率スケジュール・gradient clipping・mixed precision のいずれも
実装しない、Optimizer は Adam(Kingma & Ba, ICLR 2015)を固定学習率で使う)。
007(学習の安定化)で、AdamW・warmup + cosine スケジュール・gradient clipping を
``optimizer``・``learning_rate_schedule``・``gradient_clip_threshold`` 引数として
追加した(mixed precision は 03_efficient_training の別トピックで扱うため対象外。
演算は引き続き fp32(単精度浮動小数点)のみで行う)。**これら 3 引数を渡さない場合、
006 と完全に同一の挙動になる**(後方互換性を検証済み、007 5 節)。

011(混合精度学習)で、``autocast_dtype``(``torch.autocast`` による演算ごとの
精度割り当て)・``loss_scaler``(``src.training.precision`` の
``StaticLossScaler`` / ``DynamicLossScaler``)引数を追加した。**これらに加えて
007 までの引数のいずれも渡さない場合、006・007 と完全に同一の挙動になる**
(後方互換性、011 で検証)。1 ステップの処理順序は「スケールした損失で逆伝播 ->
unscale -> 非有限値の検出 -> unscale 後の勾配に gradient clipping -> 更新または
スキップ -> スケール値の更新」であり、``loss_scaler`` が ``None`` の場合は
非有限値が検出されることが無いため(``found_inf`` が常に ``False``)、007 までの
処理順序に一致する。

勾配ノルムは実験 H(006)・007 の主張 1〜4 で使うため、``gradient_clip_threshold``
の指定の有無によらず、**クリッピング適用前(ただし loss scaling の unscale 後)の
値を常に記録する**(007 2-3 節、011 でも同じ方針を踏襲)。

``evaluate_at_final_step`` 引数は、008 の検証 bits-per-byte の記録で判明した問題への
対処として追加した。``eval_interval`` が ``num_steps`` を割り切らない場合、途中の評価の
最後は最終ステップと一致しない(008 では 2181 ステップの学習で評価の最後が 2000 ステップ
だった)。``True`` を渡すと最終ステップでも評価し、``eval_bits_per_byte`` の末尾が必ず
学習を終えた重みの値になる。既定値 ``False`` では 006〜013 と完全に同一の挙動になる。

記号 / Notation:
    B : 訓練バッチサイズ
    S : 系列長(sequence length)
    V : 語彙サイズ
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.data.text import get_random_batch
from src.utils.statistics import compute_bits_per_byte

OptimizerLike = object
"""``step()``・``zero_grad()`` を持つ optimizer の型注釈用エイリアス
(``torch.optim.Optimizer`` および ``src.training.optimizer`` のスクラッチ実装の
いずれも受け付けることを示す、構造的な型制約は課さない)。"""


def _set_optimizer_learning_rate(optimizer: OptimizerLike, lr: float) -> None:
    """optimizer の種類によらず学習率を更新する。

    ``src.training.optimizer`` のスクラッチ実装(``set_learning_rate`` メソッドを
    持つ)と ``torch.optim.Optimizer``(``param_groups`` を持つ)の両方に対応する。
    """
    if hasattr(optimizer, "set_learning_rate"):
        optimizer.set_learning_rate(lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = lr


def evaluate_bits_per_byte(
    model: nn.Module,
    evaluation_windows: Tensor,
    evaluation_mask: Tensor,
    total_bytes: int,
    device: torch.device | str,
    batch_size: int = 16,
) -> float:
    """検証テキスト全体を非重複窓で逐次評価し、bits-per-byte を計算する。

    ``evaluation_windows``・``evaluation_mask``(``make_evaluation_windows`` の出力、
    非重複の固定長窓とパディング位置を示すマスク)を先頭から順に固定サイズの
    ミニバッチに区切って処理する(**ランダムバッチによる評価はしない**、004 の原則)。
    各窓 ``[w_0, ..., w_{S-1}]`` について、位置 ``0 .. S-2`` の logits で位置
    ``1 .. S-1`` のトークンを予測する(先頭トークン ``w_0`` は窓内に左文脈を持たない
    ため予測対象に含めない)。パディング位置(``evaluation_mask`` が False)は損失の
    総和から除外する。

    ``total_bytes`` は、トークナイザ条件に関係なく **検証テキスト全体の UTF-8
    バイト数**(符号化前のテキストから直接計算した値)を渡す想定である(006、
    3.5 節。全トークナイザ条件で同一の定数になることが望ましい特性であり、
    ``make_evaluation_windows`` はこの値を計算しない)。

    Args:
        model: 評価対象の言語モデル(``forward(token_ids) -> logits`` を持つ)。
        evaluation_windows: 形状 ``(num_windows, S)`` の LongTensor。
        evaluation_mask: 形状 ``(num_windows, S)`` の bool Tensor(True が実トークン、
            False がパディング、``make_evaluation_windows`` の出力)。
        total_bytes: 検証テキスト全体の UTF-8 バイト数。
        device: 評価に使うデバイス。
        batch_size: 1 回の forward で処理する窓の数(メモリ制約に応じて調整する。
            窓の処理順序・集計結果には影響しない)。

    Returns:
        bits-per-byte(値が小さいほど圧縮効率が高い = モデルの予測性能が高い)。
    """
    model.eval()
    total_negative_log_likelihood_nats = 0.0
    with torch.no_grad():
        for start in range(0, evaluation_windows.size(0), batch_size):
            batch = evaluation_windows[start : start + batch_size].to(device)
            batch_mask = evaluation_mask[start : start + batch_size, 1:].to(device)  # (b, S-1)
            logits = model(batch)  # (b, S, V)
            predicted_logits = logits[:, :-1, :]
            targets = batch[:, 1:]
            per_token_nll = functional.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(batch_mask.shape)
            total_negative_log_likelihood_nats += (per_token_nll * batch_mask).sum().item()
    return compute_bits_per_byte(total_negative_log_likelihood_nats, total_bytes)


def train_language_model(
    model: nn.Module,
    train_token_ids: Tensor,
    evaluation_windows: Tensor,
    evaluation_mask: Tensor,
    total_eval_bytes: int,
    num_steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    eval_interval: int,
    device: torch.device | str,
    seed: int,
    optimizer: OptimizerLike | None = None,
    learning_rate_schedule: Callable[[int], float] | None = None,
    gradient_clip_threshold: float | None = None,
    autocast_dtype: torch.dtype | None = None,
    loss_scaler: object | None = None,
    evaluate_at_final_step: bool = False,
) -> dict[str, list[float]]:
    """Adam・固定学習率・fp32 の学習ループ(007 で AdamW・学習率スケジュール・
    gradient clipping に対応、後方互換性あり)。

    訓練データは ``get_random_batch``(``src/data/text.py``)でランダムな連続区間を
    切り出してミニバッチを作る(訓練はランダムサンプリングでよい。評価との違いは
    ``evaluate_bits_per_byte`` の docstring を参照)。

    Args:
        model: 学習対象の言語モデル(``forward(token_ids) -> logits`` を持つ)。
        train_token_ids: 訓練データの 1 次元 LongTensor(``encode_corpus`` の出力)、
            または``numpy.memmap``(``encode_text_to_memmap`` の出力、009)。
            ``get_random_batch``(``src/data/text.py``)がいずれの型にも対応する。
        evaluation_windows: 検証用の非重複窓(``make_evaluation_windows`` の出力)。
        evaluation_mask: ``evaluation_windows`` に対応するパディングマスク
            (``make_evaluation_windows`` の出力、``evaluate_bits_per_byte`` にそのまま渡す)。
        total_eval_bytes: 検証テキスト全体の UTF-8 バイト数
            (``evaluate_bits_per_byte`` にそのまま渡す)。
        num_steps: 学習ステップ数。
        batch_size: 訓練バッチサイズ B。
        sequence_length: 訓練系列長 S。
        learning_rate: 固定学習率。``optimizer`` が ``None`` の場合、この値で
            ``torch.optim.Adam`` を構築する(006 と同一の挙動)。``optimizer`` が
            指定された場合、この引数は無視される(optimizer 自身が保持する学習率、
            または ``learning_rate_schedule`` が使われる)。
        eval_interval: このステップ数ごとに検証 bits-per-byte を測定する。
        device: 学習に使うデバイス。
        seed: 乱数シード。関数の先頭で明示的に ``torch.manual_seed`` を呼び、
            バッチサンプリング用の ``torch.Generator`` にも同じ値を使う。
        optimizer: 学習に使う optimizer(``step()``・``zero_grad()`` を持つ、
            ``src.training.optimizer`` のスクラッチ実装または
            ``torch.optim.Optimizer`` のインスタンス)。``None``(既定値)の場合、
            ``torch.optim.Adam(model.parameters(), lr=learning_rate)`` を使う
            (006 と同一の挙動、後方互換性)。呼び出し側が ``model.parameters()``
            から構築済みのインスタンスを渡す。
        learning_rate_schedule: ステップ番号(1-indexed)から学習率を返す callable
            (``compute_warmup_cosine_learning_rate`` を ``functools.partial`` で
            束縛したものを想定、``src/training/schedule.py``)。``None``(既定値)の
            場合は固定学習率のまま(006 と同一の挙動)。指定された場合、毎ステップ
            ``optimizer`` の学習率をこの関数の戻り値で上書きする。
        gradient_clip_threshold: グローバルノルムでの gradient clipping の閾値。
            ``None``(既定値)の場合は無効(006 と同一の挙動、勾配ノルムの測定のみ
            行う)。指定された場合、``optimizer.step()`` の前に全パラメータの勾配を
            ``min(1, gradient_clip_threshold / gradient_norm)`` でスケーリングする
            (``loss_scaler`` を併用する場合は unscale 後の勾配に適用する、011 6 節)。
        autocast_dtype: ``torch.autocast`` に渡す演算ごとの精度(``torch.float16``
            など)。``None``(既定値)の場合は ``torch.autocast`` を使わず、順伝播・
            損失計算を fp32 のまま行う(006・007 と完全に同一の挙動、011 で追加)。
        loss_scaler: 損失スケーリング(``src.training.precision`` の
            ``StaticLossScaler`` / ``DynamicLossScaler``、``scale_loss()``・
            ``unscale_gradients()``・``update()`` を持つ)。``None``(既定値)の
            場合は損失スケーリングを行わない(011 で追加)。
        evaluate_at_final_step: ``True`` の場合、``step % eval_interval == 0`` の
            ステップに加えて、最終ステップ(``step == num_steps``)でも検証する
            (同じステップで 2 回評価することはない)。``False``(既定値)の場合は
            ``step % eval_interval == 0`` のステップでのみ検証する(006〜013 と完全に
            同一の挙動)。``eval_interval`` が ``num_steps`` を割り切らないとき、
            ``False`` では ``eval_bits_per_byte`` の末尾が学習途中の値になる点に注意する。

    Returns:
        以下のキーを持つ履歴の辞書:

        - ``"step"``: 学習ステップ番号のリスト(1-indexed)。
        - ``"train_loss"``: ステップごとの訓練損失(cross entropy、nats、バッチ平均、
          loss scaling 適用前の値)。
        - ``"gradient_norm"``: ステップごとの勾配ノルム(全パラメータの勾配を
          連結した L2 ノルム、**gradient clipping 適用前(loss scaling の unscale
          後)の値**。clipping の有無に関わらず常に記録する、007 2-3 節)。
        - ``"gradient_clip_triggered"``: ステップごとに gradient clipping が実際に
          発動したか(``gradient_norm > gradient_clip_threshold``)を示す bool の
          リスト。``gradient_clip_threshold`` が ``None`` の場合は常に ``False``。
          gradient clipping を有効にした条件で実際に発動しているかどうかの前提条件
          (precondition)の検証に使う(007 前提条件 P2)。
        - ``"loss_step_delta"``: 直前ステップとの訓練損失の差(``train_loss[i] -
          train_loss[i-1]``)。最初のステップは比較対象が無いため ``0.0``。
          最大単一ステップ損失上昇幅(007 主張 3・4)の算出に使う。
        - ``"learning_rate"``: ステップごとに実際に使われた学習率
          (``learning_rate_schedule`` 指定時はその出力、それ以外は固定値)。
          更新をスキップしたステップ(``step_skipped`` が ``True``)でも、学習率
          スケジュールはステップ数どおりに進む(スキップは optimizer の更新の
          みを止め、スケジュールの進行とは独立である、011 6 節)。
        - ``"loss_scale"``: ステップごとに損失に乗じたスケール値。``loss_scaler``
          が ``None`` の場合は常に ``1.0``。
        - ``"step_skipped"``: ステップごとに、unscale 後の勾配に非有限値
          (NaN または Inf)が検出され optimizer の更新をスキップしたかを示す bool。
          ``loss_scaler`` が ``None`` の場合は常に ``False``。
        - ``"eval_step"``: 検証を行ったステップ番号のリスト(``evaluate_at_final_step``
          が ``False`` の場合、末尾が ``num_steps`` とは限らない)。
        - ``"eval_bits_per_byte"``: ``eval_step`` に対応する検証 bits-per-byte
          (評価は常に fp32 で行う、``autocast_dtype`` の指定によらない)。
    """
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device_type = torch.device(device).type

    history: dict[str, list[float]] = {
        "step": [],
        "train_loss": [],
        "gradient_norm": [],
        "gradient_clip_triggered": [],
        "loss_step_delta": [],
        "learning_rate": [],
        "loss_scale": [],
        "step_skipped": [],
        "eval_step": [],
        "eval_bits_per_byte": [],
    }

    previous_loss: float | None = None
    for step in range(1, num_steps + 1):
        model.train()

        if learning_rate_schedule is not None:
            current_lr = learning_rate_schedule(step)
            _set_optimizer_learning_rate(optimizer, current_lr)
        else:
            current_lr = learning_rate

        inputs, targets = get_random_batch(train_token_ids, batch_size, sequence_length, generator)
        inputs, targets = inputs.to(device), targets.to(device)

        if autocast_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits = model(inputs)
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
        else:
            logits = model(inputs)
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        # loss scaling: 連鎖律により、損失を S 倍すると全パラメータの勾配も S 倍になる
        # (011 理論セクション 3 節)。loss_scaler が None の場合はスケーリングなし。
        scaled_loss = loss_scaler.scale_loss(loss) if loss_scaler is not None else loss

        optimizer.zero_grad()
        scaled_loss.backward()

        # unscale: loss_scaler が None の場合、found_inf は常に False になり、
        # 以降の処理は 007 までと完全に同一の挙動になる。
        found_inf = (
            loss_scaler.unscale_gradients(model.parameters()) if loss_scaler is not None else False
        )

        # クリッピング適用前の、全パラメータの勾配(unscale 後)を連結した L2 ノルムを
        # 常に測定する(007 2-3 節、クリッピングの効果を測定するため適用前の値を残す)。
        gradient_norm_sq = sum(
            p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None
        )
        gradient_norm = float(gradient_norm_sq**0.5)

        # 発動の判定基準は「クリッピング適用前の勾配ノルムが閾値を超えたか」であり、
        # gradient_clip_threshold が None(クリッピング無効)の条件では常に False とする
        # (007 前提条件 P2: gradient clipping を有効にした条件で実際に発動しているかの検証に使う)。
        clip_triggered = (
            gradient_clip_threshold is not None and gradient_norm > gradient_clip_threshold
        )
        if clip_triggered:
            clip_scale = gradient_clip_threshold / gradient_norm
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(clip_scale)

        # 非有限値を検出したステップは optimizer の更新をスキップする(011 6 節)。
        # loss_scaler が None の場合、found_inf は常に False のため常に更新する
        # (007 までと同一の挙動)。
        if not found_inf:
            optimizer.step()

        if loss_scaler is not None:
            loss_scaler.update(found_inf)

        loss_value = loss.item()
        loss_step_delta = 0.0 if previous_loss is None else loss_value - previous_loss
        previous_loss = loss_value

        history["step"].append(step)
        history["train_loss"].append(loss_value)
        history["gradient_norm"].append(gradient_norm)
        history["gradient_clip_triggered"].append(clip_triggered)
        history["loss_step_delta"].append(loss_step_delta)
        history["learning_rate"].append(current_lr)
        history["loss_scale"].append(loss_scaler.scale if loss_scaler is not None else 1.0)
        history["step_skipped"].append(found_inf)

        if step % eval_interval == 0 or (evaluate_at_final_step and step == num_steps):
            bits_per_byte = evaluate_bits_per_byte(
                model, evaluation_windows, evaluation_mask, total_eval_bytes, device
            )
            history["eval_step"].append(step)
            history["eval_bits_per_byte"].append(bits_per_byte)

    return history
