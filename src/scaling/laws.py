"""スケーリング則(Scaling Laws)の解析パイプラインのスクラッチ実装。

Kaplan et al., "Scaling Laws for Neural Language Models" (2020) の Approach 2
(IsoFLOP プロファイル)・Hoffmann et al., "Training Compute-Optimal Large Language
Models" (Chinchilla, NeurIPS 2022) の Approach 3(パラメトリックあてはめ)に基づく
解析手順をスクラッチ実装する。線形の最小二乗解(``numpy.linalg.lstsq``)・連立一次
方程式の求解(``numpy.linalg.solve``)は numpy の数値線形代数ルーチンを利用するが、
非線形あてはめ(飽和べき乗則・Chinchilla のパラメトリックあてはめ)は
Levenberg-Marquardt 法(ヤコビアンは中心差分による数値微分)をスクラッチ実装し、
外部の最適化ライブラリ(``scipy.optimize`` 等)には委譲しない。

記号 / Notation:
    N           : 非埋め込みパラメータ数(non-embedding parameter count)
    D           : 訓練トークン数
    C           : 計算量予算(浮動小数点演算回数)
    L           : 検証損失(009 では bits-per-byte)
    V           : 語彙サイズ、d_model: モデルの隠れ次元、n_layer: 層数、n_ctx: 系列長
    E, A, B     : Chinchilla パラメトリックあてはめ L(N,D) = E + A/N^alpha + B/D^beta の係数
    alpha, beta : 同あてはめの指数パラメータ
    a, b        : 計算量最適配分の指数(N_opt(C) ∝ C^a、D_opt(C) ∝ C^b)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# トークンあたりの計算量
# ---------------------------------------------------------------------------


def estimate_flops_per_token(
    non_embedding_params: float,
    d_model: int,
    n_layer: int,
    n_ctx: int,
    vocabulary_size: int,
    include_output_layer: bool = True,
    include_attention_seq_term: bool = False,
) -> float:
    """1 トークンあたりの計算量(浮動小数点演算回数)を見積もる。

    順伝播 1 パラメータあたり 2 回の演算、逆伝播を含めて 3 倍という標準的な近似
    (Kaplan et al., 2020)から、本体(embedding 層と出力層を除く Transformer 本体)の
    計算量を ``6N`` とする。出力層(重み共有により埋め込み行列と同一の行列)を含める
    場合は ``6 V d_model`` を加算し、Attention 機構の系列長依存項を含める場合は
    ``12 n_layer n_ctx d_model`` を加算する(009 5.2 節の導出)。

    Args:
        non_embedding_params: 非埋め込みパラメータ数 N(``count_non_embedding_parameters``
            の実測値を渡す想定)。
        d_model: モデルの隠れ次元。
        n_layer: Transformer Block の層数。
        n_ctx: 系列長(注意機構の系列長依存項の計算にのみ使う)。
        vocabulary_size: 語彙サイズ V(出力層の計算量にのみ使う)。
        include_output_layer: True の場合、出力層の計算量 ``6 V d_model`` を加える
            (Porian et al., 2024 の要因 1 に対応する切り替え)。
        include_attention_seq_term: True の場合、注意機構の系列長依存項
            ``12 n_layer n_ctx d_model`` を加える。

    Returns:
        1 トークンあたりの推定計算量(FLOPs)。
    """
    flops = 6.0 * non_embedding_params
    if include_output_layer:
        flops += 6.0 * vocabulary_size * d_model
    if include_attention_seq_term:
        flops += 12.0 * n_layer * n_ctx * d_model
    return flops


# ---------------------------------------------------------------------------
# べき乗則あてはめ(対数空間の線形最小二乗)
# ---------------------------------------------------------------------------


@dataclass
class PowerLawFit:
    """``fit_power_law`` の結果。y = coefficient * x^exponent。"""

    exponent: float
    coefficient: float
    r_squared: float
    residuals: np.ndarray


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> PowerLawFit:
    """対数空間の最小二乗によるべき乗則あてはめ y = a x^b。

    ``log y = log a + b log x`` を線形最小二乗(``numpy.linalg.lstsq``)であてはめる。

    Args:
        x: 正の実数値の系列。
        y: 正の実数値の系列(``x`` と同じ長さ)。

    Returns:
        PowerLawFit(exponent=b、coefficient=a、r_squared=決定係数(対数空間)、
        residuals=対数空間の残差 ``log y - (log a + b log x)``)。
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        raise ValueError("fit_power_law は正の値のみを扱える")

    log_x = np.log(x_arr)
    log_y = np.log(y_arr)
    design = np.stack([log_x, np.ones_like(log_x)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    exponent, log_coefficient = coeffs

    predicted = design @ coeffs
    residuals = log_y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return PowerLawFit(
        exponent=float(exponent),
        coefficient=float(np.exp(log_coefficient)),
        r_squared=r_squared,
        residuals=residuals,
    )


# ---------------------------------------------------------------------------
# Levenberg-Marquardt 法(非線形最小二乗のスクラッチ実装)
# ---------------------------------------------------------------------------


def _numerical_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray], params: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """中心差分による残差関数のヤコビアン(数値微分)。"""
    n_params = len(params)
    base = residual_fn(params)
    jacobian = np.zeros((len(base), n_params))
    for j in range(n_params):
        step = eps * max(abs(params[j]), 1.0)
        params_plus = params.copy()
        params_plus[j] += step
        params_minus = params.copy()
        params_minus[j] -= step
        jacobian[:, j] = (residual_fn(params_plus) - residual_fn(params_minus)) / (2 * step)
    return jacobian


def _levenberg_marquardt(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    params0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Levenberg-Marquardt 法による(重み付き)非線形最小二乗のスクラッチ実装。

    ヤコビアンは中心差分による数値微分(``_numerical_jacobian``)で計算する。
    damping パラメータ(``lambda_``)を、コストが改善すれば縮小、改善しなければ
    拡大しながら反復する標準的な LM の手続き。``weights`` を指定すると重み付き
    最小二乗になる(IRLS による Huber 損失の近似あてはめに使う、
    ``_fit_nonlinear_huber`` 参照)。

    Args:
        residual_fn: パラメータベクトルを受け取り残差ベクトルを返す関数。
        params0: 初期パラメータ。
        max_iter: 最大反復回数。
        tol: パラメータ更新量(相対ノルム)がこの値を下回ったら収束とみなす。
        weights: 残差ごとの重み(``None`` は全て 1、通常の最小二乗)。

    Returns:
        (収束後のパラメータ, 収束したかどうか) のタプル。
    """
    params = np.array(params0, dtype=float)
    residuals = residual_fn(params)
    if weights is None:
        weights = np.ones_like(residuals)
    cost = float(np.sum(weights * residuals**2))

    lambda_ = 1e-3
    converged = False

    for _ in range(max_iter):
        jacobian = _numerical_jacobian(residual_fn, params)
        w_sqrt = np.sqrt(weights)
        jw = jacobian * w_sqrt[:, None]
        rw = residuals * w_sqrt

        jtj = jw.T @ jw
        neg_jtr = -(jw.T @ rw)  # Gauss-Newton 方向: delta = -(J^T J)^-1 J^T r

        step_accepted = False
        for _ in range(30):
            damped = jtj + lambda_ * np.diag(np.diag(jtj) + 1e-12)
            try:
                delta = np.linalg.solve(damped, neg_jtr)
            except np.linalg.LinAlgError:
                lambda_ *= 10
                continue

            new_params = params + delta
            new_residuals = residual_fn(new_params)
            with np.errstate(over="ignore", invalid="ignore"):
                new_cost = float(np.sum(weights * new_residuals**2))
            if np.isnan(new_cost):
                new_cost = np.inf

            if new_cost < cost:
                if np.linalg.norm(delta) < tol * (np.linalg.norm(params) + tol):
                    converged = True
                params, residuals, cost = new_params, new_residuals, new_cost
                lambda_ = max(lambda_ / 10, 1e-12)
                step_accepted = True
                break
            lambda_ *= 10

        if not step_accepted:
            # damping を増やしても改善しない -> 局所的な最小点に達したとみなす
            converged = True
            break
        if converged:
            break

    return params, converged


# ---------------------------------------------------------------------------
# 飽和べき乗則あてはめ
# ---------------------------------------------------------------------------


@dataclass
class SaturatingPowerLawFit:
    """``fit_saturating_power_law`` の結果。L(x) = l_inf + (x_c / x)^alpha。"""

    l_inf: float
    x_c: float
    alpha: float
    converged: bool
    residuals: np.ndarray


def fit_saturating_power_law(x: Sequence[float], y: Sequence[float]) -> SaturatingPowerLawFit:
    """飽和べき乗則 L(x) = l_inf + (x_c / x)^alpha をあてはめる。

    Levenberg-Marquardt 法(``_levenberg_marquardt``)による非線形最小二乗。
    正値制約を持つ ``x_c``・``alpha`` は対数空間でパラメータ化して最適化することで
    自動的に満たす。初期値を複数(l_inf の初期割合 x alpha の初期値)組み合わせて試す
    multi-start とし、最良の残差平方和を与える解を採用する(単一の初期値では局所解に
    陥りやすいため)。

    Args:
        x: 正の実数値の系列(D、訓練トークン数を想定)。
        y: 対応する損失の系列(L)。``x`` と同じ長さ。

    Returns:
        SaturatingPowerLawFit。
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        l_inf, log_x_c, log_alpha = params
        x_c = np.exp(log_x_c)
        alpha = np.exp(log_alpha)
        with np.errstate(over="ignore"):
            # multi-start の探索中に極端なパラメータを試すことがあり、その際の
            # オーバーフロー(inf)は次善のコスト値として自然に棄却されるため無害。
            pred = l_inf + (x_c / x_arr) ** alpha
        return pred - y_arr

    y_min = float(y_arr.min())
    x_median = float(np.median(x_arr))

    best_params, best_cost, best_converged = None, np.inf, False
    for l_inf_init_frac in (0.5, 0.8, 0.95):
        for alpha_init in (0.1, 0.3, 0.5, 0.8):
            params0 = np.array(
                [y_min * l_inf_init_frac, np.log(x_median), np.log(alpha_init)], dtype=float
            )
            params, converged = _levenberg_marquardt(residual_fn, params0)
            cost = float(np.sum(residual_fn(params) ** 2))
            if cost < best_cost:
                best_params, best_cost, best_converged = params, cost, converged

    l_inf, log_x_c, log_alpha = best_params
    return SaturatingPowerLawFit(
        l_inf=float(l_inf),
        x_c=float(np.exp(log_x_c)),
        alpha=float(np.exp(log_alpha)),
        converged=best_converged,
        residuals=residual_fn(best_params),
    )


# ---------------------------------------------------------------------------
# IsoFLOP プロファイルの放物線あてはめ
# ---------------------------------------------------------------------------


@dataclass
class IsoFLOPParabolaFit:
    """``fit_isoflop_parabola`` の結果。L = a (log N)^2 + b (log N) + c。"""

    a: float
    b: float
    c: float
    vertex_log_n: float
    vertex_loss: float
    is_interior: bool


def fit_isoflop_parabola(log_n: Sequence[float], loss: Sequence[float]) -> IsoFLOPParabolaFit:
    """(log N, L) に 2 次多項式をあてはめ、頂点を閉形式で求める。

    ``L = a (log N)^2 + b (log N) + c`` を線形最小二乗(``numpy.linalg.lstsq``)で
    あてはめ、頂点 ``log N* = -b / (2a)`` を解析解で求める(Hoffmann et al., 2022,
    Approach 2)。``a <= 0``(下に凸でない、極小点を持たない)場合は頂点を求められない
    ため ``is_interior=False`` とする。

    Args:
        log_n: log N の値の系列(IsoFLOP プロファイル、同一計算量予算での複数
            model サイズに対応)。
        loss: 対応する損失の系列。``log_n`` と同じ長さ。

    Returns:
        IsoFLOPParabolaFit(a, b, c: 係数、vertex_log_n, vertex_loss: 頂点の座標、
        is_interior: 頂点が ``log_n`` の範囲の内点かどうか)。
    """
    log_n_arr = np.asarray(log_n, dtype=float)
    loss_arr = np.asarray(loss, dtype=float)
    design = np.stack([log_n_arr**2, log_n_arr, np.ones_like(log_n_arr)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, loss_arr, rcond=None)
    a, b, c = (float(v) for v in coeffs)

    if a <= 0:
        return IsoFLOPParabolaFit(
            a=a, b=b, c=c, vertex_log_n=float("nan"), vertex_loss=float("nan"), is_interior=False
        )

    vertex_log_n = -b / (2 * a)
    vertex_loss = a * vertex_log_n**2 + b * vertex_log_n + c
    is_interior = bool(log_n_arr.min() < vertex_log_n < log_n_arr.max())

    return IsoFLOPParabolaFit(
        a=a,
        b=b,
        c=c,
        vertex_log_n=vertex_log_n,
        vertex_loss=vertex_loss,
        is_interior=is_interior,
    )


# ---------------------------------------------------------------------------
# Chinchilla パラメトリックあてはめ(対数空間の Huber 損失、IRLS)
# ---------------------------------------------------------------------------


def _huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_r = np.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quadratic, linear)


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Huber 損失を IRLS(Iteratively Reweighted Least Squares)で近似するための重み。

    ``|r| <= delta`` では通常の二乗損失(重み 1)、``|r| > delta`` では重み
    ``delta / |r|`` とすることで、重み付き二乗和 ``sum(w_i r_i^2)`` が Huber 損失の
    1 回の反復における局所近似になる(標準的な robust regression の手法)。
    """
    abs_r = np.abs(residuals)
    weights = np.ones_like(residuals)
    mask = abs_r > delta
    weights[mask] = delta / abs_r[mask]
    return weights


def _fit_nonlinear_huber(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    params0: np.ndarray,
    huber_delta: float,
    n_irls_iter: int = 15,
    lm_iter_per_step: int = 8,
) -> tuple[np.ndarray, bool]:
    """IRLS による Huber 損失の非線形あてはめ。

    各反復で現在の残差から Huber 重み(``_huber_weights``)を計算し、その重みを
    固定した重み付き最小二乗を Levenberg-Marquardt 法(``_levenberg_marquardt``)で
    解く、という 2 段階の反復を繰り返す。
    """
    params = np.array(params0, dtype=float)
    converged = False
    for _ in range(n_irls_iter):
        residuals = residual_fn(params)
        weights = _huber_weights(residuals, huber_delta)
        new_params, _ = _levenberg_marquardt(
            residual_fn, params, max_iter=lm_iter_per_step, weights=weights
        )
        param_change = np.linalg.norm(new_params - params) / (np.linalg.norm(params) + 1e-12)
        params = new_params
        if param_change < 1e-8:
            converged = True
            break
    return params, converged


@dataclass
class ChinchillaParametricFit:
    """``fit_chinchilla_parametric`` の結果。L(N,D) = e + a_coef/N^alpha + b_coef/D^beta。"""

    e: float
    a_coef: float
    b_coef: float
    alpha: float
    beta: float
    converged: bool
    residuals: np.ndarray


def fit_chinchilla_parametric(
    n: Sequence[float],
    d: Sequence[float],
    loss: Sequence[float],
    huber_delta: float = 0.05,
    init_alpha_grid: Sequence[float] = (0.2, 0.35, 0.5),
    init_beta_grid: Sequence[float] = (0.2, 0.35, 0.5),
) -> ChinchillaParametricFit:
    """Chinchilla のパラメトリックあてはめ L(N,D) = E + A/N^alpha + B/D^beta を
    対数空間の Huber 損失であてはめる(Hoffmann et al., 2022, Approach 3)。

    ``E, A, B, alpha, beta`` が正であるという制約を、対数空間でパラメータ化する
    ことで自動的に満たす(``params = [log E, log A, log B, log alpha, log beta]``
    を実際の最適化変数とし、非制約の Levenberg-Marquardt を IRLS 経由で適用する、
    ``_fit_nonlinear_huber`` 参照)。

    Chinchilla のパラメトリックあてはめは初期値に敏感であることが知られている
    (Besiroglu et al., "Chinchilla Scaling: A replication attempt", 2024)ため、
    ``init_alpha_grid`` x ``init_beta_grid`` の全組み合わせを初期値として試す
    multi-start を行い、最良の Huber 損失を与える解を採用する。

    Args:
        n: 非埋め込みパラメータ数 N の測定値の系列。
        d: 訓練トークン数 D の測定値の系列(``n`` と同じ長さ)。
        loss: 対応する損失 L の測定値の系列。
        huber_delta: Huber 損失の閾値(対数空間の残差に対して適用)。
        init_alpha_grid: multi-start に使う alpha の初期値候補。
        init_beta_grid: multi-start に使う beta の初期値候補。

    Returns:
        ChinchillaParametricFit。
    """
    n_arr = np.asarray(n, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    log_loss = np.log(np.asarray(loss, dtype=float))

    def residual_fn(params: np.ndarray) -> np.ndarray:
        log_e, log_a, log_b, log_alpha, log_beta = params
        e, a_coef, b_coef = np.exp(log_e), np.exp(log_a), np.exp(log_b)
        alpha, beta = np.exp(log_alpha), np.exp(log_beta)
        pred = e + a_coef / n_arr**alpha + b_coef / d_arr**beta
        return np.log(pred) - log_loss

    loss_min = float(np.exp(log_loss).min())
    best_params, best_cost, best_converged = None, np.inf, False
    for alpha_init in init_alpha_grid:
        for beta_init in init_beta_grid:
            params0 = np.array(
                [
                    np.log(max(loss_min * 0.5, 1e-6)),
                    np.log(max(loss_min * float(n_arr.mean()) ** alpha_init * 0.5, 1e-6)),
                    np.log(max(loss_min * float(d_arr.mean()) ** beta_init * 0.5, 1e-6)),
                    np.log(alpha_init),
                    np.log(beta_init),
                ],
                dtype=float,
            )
            params, converged = _fit_nonlinear_huber(residual_fn, params0, huber_delta)
            cost = float(np.sum(_huber_loss(residual_fn(params), huber_delta)))
            if cost < best_cost:
                best_params, best_cost, best_converged = params, cost, converged

    log_e, log_a, log_b, log_alpha, log_beta = best_params
    return ChinchillaParametricFit(
        e=float(np.exp(log_e)),
        a_coef=float(np.exp(log_a)),
        b_coef=float(np.exp(log_b)),
        alpha=float(np.exp(log_alpha)),
        beta=float(np.exp(log_beta)),
        converged=best_converged,
        residuals=residual_fn(best_params),
    )


def compute_optimal_allocation_exponents(alpha: float, beta: float) -> tuple[float, float]:
    """計算量最適配分の指数を求める(Hoffmann et al., 2022 Approach 3、009 5.3 節の導出)。

    制約 ``C = 6ND`` の下で ``L(N,D) = E + A/N^alpha + B/D^beta`` を最小化すると、
    ``N_opt(C) ∝ C^a``・``D_opt(C) ∝ C^b``(``a = beta/(alpha+beta)``、
    ``b = alpha/(alpha+beta)``)が導かれる。この導出は ``C`` が ``ND`` に比例する
    ことを前提とする(出力層を含む数え方では成り立たない、009 5.3 節)。

    Args:
        alpha, beta: ``fit_chinchilla_parametric`` で得た指数。

    Returns:
        (a, b) のタプル。
    """
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    return a, b


# ---------------------------------------------------------------------------
# IsoFLOP プロファイルからの計算量最適フロンティアの再構成
# ---------------------------------------------------------------------------


@dataclass
class GridPoint:
    """学習グリッドの 1 セル(1 つの d_model x 1 つの計算量予算)の実測値。"""

    d_model: int
    non_embedding_params: float
    tokens_trained: float
    loss: float


@dataclass
class FrontierResult:
    """``reconstruct_optimal_frontier`` の結果。"""

    power_law_fit: PowerLawFit | None
    frontier_log_c: list[float]
    frontier_log_n_opt: list[float]
    saturating_fits: dict[int, SaturatingPowerLawFit]
    num_interior_budgets: int
    profiles: dict[float, dict[str, object]] = field(default_factory=dict)


def reconstruct_optimal_frontier(
    grid: Sequence[GridPoint],
    target_compute_budgets: Sequence[float],
    flops_per_token_by_d_model: dict[int, float],
) -> FrontierResult:
    """学習グリッドから計算量最適フロンティア N_opt(C) を再構成する。

    009 6.4 節の解析パイプラインのうち、以下を行う。

    1. ``d_model`` ごとに、実測の ``(D, L)`` の点へ飽和べき乗則(``fit_saturating_power_law``)
       をあてはめ、``L(D)`` の連続曲線を得る。
    2. 目標計算量予算 ``C`` ごとに、各 ``d_model`` で ``D = C / flops_per_token`` を求め、
       その ``D`` が実測 ``D`` の範囲の **内側にある** ``d_model`` のみを使って
       ``(log N, L)`` に放物線(``fit_isoflop_parabola``)をあてはめ、頂点から
       ``N_opt(C)`` を得る(外挿はしない。範囲外の ``d_model`` はその ``C`` の
       プロファイルから除外する)。
    3. ``log N_opt`` 対 ``log C`` のべき乗則あてはめ(``fit_power_law``)から
       指数を推定する。

    Args:
        grid: 学習グリッドの全セル(``GridPoint`` のリスト)。
        target_compute_budgets: 目標計算量予算 C の系列。
        flops_per_token_by_d_model: ``{d_model: 1 トークンあたりの計算量}``
            (``estimate_flops_per_token`` の結果、数え方を固定した上で ``d_model``
            ごとに事前計算しておく)。

    Returns:
        FrontierResult。``power_law_fit`` はフロンティア推定に使えた計算量予算が
        2 点未満の場合 ``None``。
    """
    by_d_model: dict[int, list[GridPoint]] = {}
    for point in grid:
        by_d_model.setdefault(point.d_model, []).append(point)

    saturating_fits: dict[int, SaturatingPowerLawFit] = {}
    d_ranges: dict[int, tuple[float, float]] = {}
    n_by_d_model: dict[int, float] = {}
    for d_model, points in by_d_model.items():
        d_values = [p.tokens_trained for p in points]
        l_values = [p.loss for p in points]
        saturating_fits[d_model] = fit_saturating_power_law(d_values, l_values)
        d_ranges[d_model] = (min(d_values), max(d_values))
        n_by_d_model[d_model] = float(np.mean([p.non_embedding_params for p in points]))

    frontier_log_c: list[float] = []
    frontier_log_n_opt: list[float] = []
    profiles: dict[float, dict[str, object]] = {}

    for c in target_compute_budgets:
        log_n_candidates: list[float] = []
        loss_candidates: list[float] = []
        for d_model in by_d_model:
            f = flops_per_token_by_d_model[d_model]
            d_required = c / f
            d_min, d_max = d_ranges[d_model]
            if not (d_min <= d_required <= d_max):
                continue
            fit = saturating_fits[d_model]
            predicted_loss = fit.l_inf + (fit.x_c / d_required) ** fit.alpha
            log_n_candidates.append(float(np.log(n_by_d_model[d_model])))
            loss_candidates.append(float(predicted_loss))

        if len(log_n_candidates) < 3:
            continue

        parabola = fit_isoflop_parabola(log_n_candidates, loss_candidates)
        profiles[c] = {
            "log_n": log_n_candidates,
            "loss": loss_candidates,
            "parabola_coeffs": (parabola.a, parabola.b, parabola.c),
            "vertex": (
                (parabola.vertex_log_n, parabola.vertex_loss) if parabola.is_interior else None
            ),
        }
        if not parabola.is_interior:
            continue

        frontier_log_c.append(float(np.log(c)))
        frontier_log_n_opt.append(parabola.vertex_log_n)

    power_law_fit = None
    if len(frontier_log_c) >= 2:
        power_law_fit = fit_power_law(
            list(np.exp(frontier_log_c)), list(np.exp(frontier_log_n_opt))
        )

    return FrontierResult(
        power_law_fit=power_law_fit,
        frontier_log_c=frontier_log_c,
        frontier_log_n_opt=frontier_log_n_opt,
        saturating_fits=saturating_fits,
        num_interior_budgets=len(frontier_log_c),
        profiles=profiles,
    )


# ---------------------------------------------------------------------------
# パラメトリックブートストラップ
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """``bootstrap_scaling_analysis`` の結果(各指数のブートストラップ標本)。"""

    a_body: np.ndarray
    a_body_output: np.ndarray
    a_approach3: np.ndarray
    delta_b: np.ndarray
    delta_c: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    num_successful: int
    num_attempted: int


def bootstrap_scaling_analysis(
    grid: Sequence[GridPoint],
    target_compute_budgets: Sequence[float],
    flops_per_token_body: dict[int, float],
    flops_per_token_body_output: dict[int, float],
    noise_std: float,
    n_bootstrap: int,
    huber_delta: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """パラメトリックブートストラップによる指数推定の不確かさの評価(009 6.4 節)。

    測定したシード間標準偏差 ``noise_std`` の正規ノイズを各ランの ``L`` に加え、
    IsoFLOP プロファイルによる指数推定(数え方「本体のみ」「本体+出力層」の 2 通り、
    ``reconstruct_optimal_frontier``)と Chinchilla パラメトリックあてはめによる
    指数推定(``fit_chinchilla_parametric`` に相当する処理)を **同一のノイズ付与
    された標本** に対して行う。これにより実験 B・C の対比量(差分)を同一反復内の
    値として計算でき、独立に求めた 2 つの標準偏差を合成する場合に生じる分散の
    過大評価(相関の無視)を避けられる。

    Chinchilla のパラメトリックあてはめは計算コストが高いため、ノイズなしの点推定で
    収束した解を全反復の初期値(warm start)として使い、multi-start は行わない
    (ノイズの大きさが小さければ最適解の近傍に留まるとみなせる)。

    Args:
        grid: 学習グリッドの全セル(実測値)。
        target_compute_budgets: フロンティア推定に使う目標計算量予算の系列。
        flops_per_token_body: 数え方「本体のみ」の ``{d_model: トークンあたり計算量}``。
        flops_per_token_body_output: 数え方「本体+出力層」の同様の辞書。
        noise_std: ``L`` に加えるノイズの標準偏差(ノイズ床の実測値)。
        n_bootstrap: ブートストラップの反復回数。
        huber_delta: Chinchilla あてはめの Huber 損失の閾値。
        seed: 乱数シード。

    Returns:
        BootstrapResult。IsoFLOP フロンティアのあてはめが失敗した反復(内点予算が
        2 未満で ``power_law_fit`` が得られない)はスキップし、``num_successful``に
        実際に成功した反復数、``num_attempted``に試行した反復数を記録する。
    """
    rng = np.random.default_rng(seed)

    n_values = np.array([p.non_embedding_params for p in grid], dtype=float)
    d_values = np.array([p.tokens_trained for p in grid], dtype=float)
    l_values = np.array([p.loss for p in grid], dtype=float)
    d_models = [p.d_model for p in grid]

    warm_fit = fit_chinchilla_parametric(n_values, d_values, l_values, huber_delta=huber_delta)
    warm_params = np.array(
        [
            np.log(warm_fit.e),
            np.log(warm_fit.a_coef),
            np.log(warm_fit.b_coef),
            np.log(warm_fit.alpha),
            np.log(warm_fit.beta),
        ]
    )

    a_body_list, a_bo_list, a3_list, alpha_list, beta_list = [], [], [], [], []

    for _ in range(n_bootstrap):
        noisy_l = l_values + rng.normal(0.0, noise_std, size=l_values.shape)
        noisy_l = np.clip(noisy_l, 1e-6, None)  # bits-per-byte は正である必要がある
        noisy_grid = [
            GridPoint(d_model=dm, non_embedding_params=n, tokens_trained=d, loss=nl)
            for dm, n, d, nl in zip(d_models, n_values, d_values, noisy_l, strict=True)
        ]

        result_body = reconstruct_optimal_frontier(
            noisy_grid, target_compute_budgets, flops_per_token_body
        )
        result_bo = reconstruct_optimal_frontier(
            noisy_grid, target_compute_budgets, flops_per_token_body_output
        )
        if result_body.power_law_fit is None or result_bo.power_law_fit is None:
            continue

        def residual_fn_local(
            params: np.ndarray, n_values=n_values, d_values=d_values, noisy_l=noisy_l
        ) -> np.ndarray:
            log_e, log_a, log_b, log_alpha, log_beta = params
            e, a_coef, b_coef = np.exp(log_e), np.exp(log_a), np.exp(log_b)
            alpha, beta = np.exp(log_alpha), np.exp(log_beta)
            pred = e + a_coef / n_values**alpha + b_coef / d_values**beta
            return np.log(pred) - np.log(noisy_l)

        chinchilla_params, _ = _fit_nonlinear_huber(residual_fn_local, warm_params, huber_delta)
        _, _, _, log_alpha3, log_beta3 = chinchilla_params
        alpha_i, beta_i = float(np.exp(log_alpha3)), float(np.exp(log_beta3))
        a3, _ = compute_optimal_allocation_exponents(alpha_i, beta_i)

        a_body_list.append(result_body.power_law_fit.exponent)
        a_bo_list.append(result_bo.power_law_fit.exponent)
        a3_list.append(a3)
        alpha_list.append(alpha_i)
        beta_list.append(beta_i)

    a_body_arr = np.array(a_body_list)
    a_bo_arr = np.array(a_bo_list)
    a3_arr = np.array(a3_list)
    return BootstrapResult(
        a_body=a_body_arr,
        a_body_output=a_bo_arr,
        a_approach3=a3_arr,
        delta_b=a_bo_arr - a_body_arr,
        delta_c=a3_arr - a_body_arr,
        alpha=np.array(alpha_list),
        beta=np.array(beta_list),
        num_successful=len(a_body_list),
        num_attempted=n_bootstrap,
    )
