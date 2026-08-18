"""可視化(visualization)ユーティリティ。

ノートブック間で再利用する描画ヘルパをまとめる。

Note:
    Colab の matplotlib には日本語フォントが入っていないため、
    図中のラベル・タイトルは英語で記述する(説明は Markdown セル側で日本語にする)。
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def _to_numpy(weights: torch.Tensor | np.ndarray) -> np.ndarray:
    """Tensor / ndarray を計算グラフから切り離した ndarray に変換する。"""
    if isinstance(weights, torch.Tensor):
        return weights.detach().cpu().numpy()
    return np.asarray(weights)


def plot_attention_heatmap(
    weights: torch.Tensor | np.ndarray,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    annotate: bool = False,
    colorbar: bool = True,
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> Axes:
    """Attention 重み行列を 1 枚のヒートマップとして描画する。

    Args:
        weights: 形状 ``(S_q, S_k)`` の Attention 重み。行方向(Query)の和が 1。
        x_labels: 横軸(Key 側)のラベル。
        y_labels: 縦軸(Query 側)のラベル。
        title: 図のタイトル。
        ax: 描画先の Axes。None なら新規作成する。
        cmap: カラーマップ名。
        annotate: 各セルに数値を書き込むか(小さい行列向け)。
        colorbar: カラーバーを表示するか。
        vmin, vmax: カラースケールの下限・上限。

    Returns:
        描画に使った Axes。
    """
    matrix = _to_numpy(weights)
    if matrix.ndim != 2:
        raise ValueError(f"weights は 2 次元 (S_q, S_k) である必要がある: shape={matrix.shape}")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    n_query, n_key = matrix.shape
    ax.set_xticks(np.arange(n_key))
    ax.set_yticks(np.arange(n_query))
    ax.set_xticklabels(x_labels if x_labels is not None else np.arange(n_key), rotation=90)
    ax.set_yticklabels(y_labels if y_labels is not None else np.arange(n_query))
    ax.set_xlabel("Key position (attended to)")
    ax.set_ylabel("Query position (attending)")
    if title is not None:
        ax.set_title(title, fontsize=11)

    if annotate:
        threshold = matrix.max() / 2.0
        for i in range(n_query):
            for j in range(n_key):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] < threshold else "black",
                )

    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_multi_head_attention(
    weights: torch.Tensor | np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    n_cols: int = 4,
    cmap: str = "viridis",
    annotate: bool = False,
    figsize_per_head: tuple[float, float] = (3.4, 3.0),
    vmax: float | None = None,
) -> Figure:
    """ヘッドごとの Attention 重みを並べて描画する。

    Args:
        weights: 形状 ``(h, S_q, S_k)`` の Attention 重み
            (``(B, h, S_q, S_k)`` の場合は先頭バッチを自動で取り出す)。
        labels: Query / Key 共通のトークンラベル(自己注意を想定)。
        title: 図全体のタイトル。
        n_cols: 1 行あたりのヘッド数。
        cmap: カラーマップ名。
        annotate: 各セルに数値を書き込むか。
        figsize_per_head: ヘッド 1 つあたりの図サイズ。
        vmax: カラースケールの上限。None の場合は全ヘッドの最大値を使う。
            図どうしを比較したいときは ``1.0`` などに固定する。

    Returns:
        生成した Figure。
    """
    matrix = _to_numpy(weights)
    if matrix.ndim == 4:
        matrix = matrix[0]
    if matrix.ndim != 3:
        raise ValueError(f"weights は (h, S_q, S_k) である必要がある: shape={matrix.shape}")

    n_heads = matrix.shape[0]
    n_cols = min(n_cols, n_heads)
    n_rows = int(np.ceil(n_heads / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_head[0] * n_cols, figsize_per_head[1] * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    if vmax is None:
        vmax = float(matrix.max())
    for head in range(n_heads):
        plot_attention_heatmap(
            matrix[head],
            x_labels=labels,
            y_labels=labels,
            title=f"Head {head + 1}",
            ax=axes[head],
            cmap=cmap,
            annotate=annotate,
            colorbar=False,
            vmax=vmax,
        )
    for unused in range(n_heads, len(axes)):
        axes[unused].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_seed_scatter(
    values_by_condition: dict[str, list[float]],
    title: str = "",
    ylabel: str = "Value",
    ax: Axes | None = None,
    log_scale: bool = False,
) -> Axes:
    """条件ごとに、乱数シード(seed)ごとの値を平均・範囲とともに散布図で描画する。

    複数シードで実行した実験(004 の実験 C・E・G など)で、平均値だけでなく
    個々のシードの値をすべて重ねて表示するために使う。条件間で最小値〜最大値の
    区間が重なるかどうかを目視で判定できるようにすることが目的。

    Args:
        values_by_condition: ``{条件名: シードごとの値のリスト}`` の辞書。
        title: 図のタイトル。
        ylabel: 縦軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。
        log_scale: 縦軸を対数スケールにするか。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(values_by_condition) + 2.0), 4.2))

    rng = np.random.default_rng(0)
    for x, (_name, values) in enumerate(values_by_condition.items()):
        values_arr = np.asarray(values, dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values_arr))
        ax.scatter(
            np.full(len(values_arr), x) + jitter,
            values_arr,
            color="tab:gray",
            alpha=0.7,
            s=22,
            zorder=2,
            label="individual seeds" if x == 0 else None,
        )
        mean = values_arr.mean()
        low, high = values_arr.min(), values_arr.max()
        ax.errorbar(
            [x],
            [mean],
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color="tab:red",
            capsize=4,
            markersize=7,
            zorder=3,
            label="mean (min-max range)" if x == 0 else None,
        )

    ax.set_xticks(range(len(values_by_condition)))
    ax.set_xticklabels(list(values_by_condition.keys()), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return ax


def plot_learning_curves_multi_seed(
    histories_by_condition: dict[str, list[dict]],
    step_key: str = "step",
    value_key: str = "evaluation_loss",
    title: str = "Learning curve",
    xlabel: str = "Step",
    ylabel: str = "Loss",
    ax: Axes | None = None,
    log_scale: bool = False,
) -> Axes:
    """条件ごとに、複数シードの学習曲線をすべて重ねて描画する。

    条件ごとに色を揃え、シードごとの線は細く半透明にすることで、シード間のばらつきを
    見せつつ条件間の違いも読み取れるようにする。凡例には条件名のみを表示し、
    シード番号は出さない(`ax.plot([], [], ...)`でシードごとの線とは別に凡例専用の
    ダミー線を 1 本だけ登録する)。

    Args:
        histories_by_condition: ``{条件名: シードごとの history 辞書のリスト}``。
            各 history は`train_character_level_language_model()`の戻り値のように
            ``step_key``・``value_key``をキーに持つ辞書を想定する。
        step_key: history から横軸の値を取り出すキー。
        value_key: history から縦軸の値を取り出すキー。
        title: 図のタイトル。
        xlabel: 横軸ラベル。
        ylabel: 縦軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。
        log_scale: 縦軸を対数スケールにするか。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, histories) in enumerate(histories_by_condition.items()):
        color = color_cycle[i % len(color_cycle)]
        for history in histories:
            ax.plot(
                history[step_key],
                history[value_key],
                color=color,
                alpha=0.35,
                linewidth=1.0,
            )
        ax.plot([], [], color=color, linewidth=2.0, label=name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_bar_by_layer(
    values_by_condition: dict[str, list[float]],
    title: str = "",
    ylabel: str = "Value",
    xlabel: str = "Layer index (1 = closest to input)",
    ax: Axes | None = None,
) -> Axes:
    """層ごとの統計量を、条件ごとに並べた棒グラフで描画する。

    隠れ状態の平均/RMS 比(実験 D)や死んだユニット(dead unit)の割合(実験 F)など、
    層番号を横軸にした条件間比較に使う。

    Args:
        values_by_condition: ``{条件名: 層ごとの値のリスト}`` の辞書。
            すべての条件で層数(リストの長さ)が揃っている必要がある。
        title: 図のタイトル。
        ylabel: 縦軸ラベル。
        xlabel: 横軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))

    n_conditions = len(values_by_condition)
    n_layers = len(next(iter(values_by_condition.values())))
    layer_positions = np.arange(1, n_layers + 1)
    bar_width = 0.8 / n_conditions

    for i, (name, values) in enumerate(values_by_condition.items()):
        offset = (i - (n_conditions - 1) / 2) * bar_width
        ax.bar(layer_positions + offset, values, width=bar_width, label=name)

    ax.set_xticks(layer_positions)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return ax


def plot_function_curves(
    x: torch.Tensor | np.ndarray,
    curves: dict[str, torch.Tensor | np.ndarray],
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "f(x)",
    ax: Axes | None = None,
) -> Axes:
    """複数の関数の値(または導関数)を同一の x 軸上に重ねて描画する。

    活性化関数の形状比較(004 の実験 A、ReLU / GELU / Swish とその導関数)に使う。

    Args:
        x: 形状 ``(N,)`` の共通の横軸の値。
        curves: ``{系列名: f(x) の値}`` の辞書。各値は x と同じ形状 ``(N,)``。
        title: 図のタイトル。
        xlabel: 横軸ラベル。
        ylabel: 縦軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.2))

    x_np = _to_numpy(x)
    for name, values in curves.items():
        ax.plot(x_np, _to_numpy(values), label=name, linewidth=1.8)

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_learning_curves(
    histories: dict[str, list[float]],
    title: str = "Learning curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    ax: Axes | None = None,
    log_scale: bool = False,
) -> Axes:
    """学習曲線(複数系列)を折れ線で描画する。

    Args:
        histories: ``{系列名: 値のリスト}`` の辞書。
        title: 図のタイトル。
        xlabel: 横軸ラベル。
        ylabel: 縦軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。
        log_scale: 縦軸を対数スケールにするか。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))

    for name, values in histories.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_grouped_bar(
    values_by_series: dict[str, dict[str, float]],
    title: str = "",
    ylabel: str = "Value",
    xlabel: str = "",
    ax: Axes | None = None,
    log_scale: bool = False,
    noise_band: float | None = None,
) -> Axes:
    """カテゴリ(x 軸)ごとに複数系列の値を並べた棒グラフを描画する。

    ドメイン(英語・日本語・コードなど)× 条件(chunk_split_mode など)のように、
    2 つの離散軸を持つ値を比較する際に使う(005 の実験4など)。

    Args:
        values_by_series: ``{系列名: {カテゴリ名: 値}}`` の辞書。全系列でカテゴリの
            集合が揃っている必要がある(最初の系列のキー順を x 軸の順序として使う)。
        title: 図のタイトル。
        ylabel: 縦軸ラベル。
        xlabel: 横軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。
        log_scale: 縦軸を対数スケールにするか。
        noise_band: 測定ノイズ床(noise floor)の半値幅(006 で追加)。``None``
            (既定値、従来の呼び出しはこの値のまま影響を受けない)の場合は描画しない。
            指定した場合、``values_by_series`` の **最初の系列を基準条件(baseline)**
            とみなし(x 軸の順序を決めるのと同じ系列)、カテゴリごとにその値
            ``± noise_band`` の範囲を棒の背後に灰色の帯として描画する。他の系列の
            値がこの帯からはみ出しているかどうかで、条件間の差が測定ノイズ床を
            超えているかを目視で判定できるようにする(複数シードでの繰り返し測定の
            ばらつきなどを noise_band として渡す想定、``plot_seed_scatter`` の
            min-max range と同種の意図)。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))

    series_names = list(values_by_series.keys())
    categories = list(next(iter(values_by_series.values())).keys())
    n_series = len(series_names)
    x = np.arange(len(categories))
    bar_width = 0.8 / n_series

    if noise_band is not None:
        baseline_series = values_by_series[series_names[0]]
        for i, category in enumerate(categories):
            baseline_value = baseline_series[category]
            ax.add_patch(
                Rectangle(
                    (x[i] - 0.4, baseline_value - noise_band),
                    0.8,
                    2 * noise_band,
                    facecolor="gray",
                    alpha=0.25,
                    edgecolor="none",
                    zorder=0,
                    label="noise floor" if i == 0 else None,
                )
            )

    for i, name in enumerate(series_names):
        offset = (i - (n_series - 1) / 2) * bar_width
        values = [values_by_series[name][c] for c in categories]
        ax.bar(x + offset, values, width=bar_width, label=name, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return ax


def plot_gradient_norm_trace(
    gradient_norms: Sequence[float],
    clip_threshold: float | None = None,
    title: str = "",
    xlabel: str = "Step",
    ylabel: str = "Gradient norm",
    ax: Axes | None = None,
    log_scale: bool = False,
) -> Axes:
    """勾配ノルムの時系列を折れ線で描画し、gradient clipping 閾値を水平線で重畳表示する(007)。

    ``gradient_norms`` は ``train_language_model``(``src/training/trainer.py``)の
    history の ``"gradient_norm"``(クリッピング適用前の値)を想定する。学習中の
    勾配ノルムのスパイクと、それが ``clip_threshold`` に対してどの程度の大きさかを
    目視で確認する(007 主張 3、gradient clipping の効果の検証)。

    Args:
        gradient_norms: ステップごとの勾配ノルム(クリッピング適用前)。
        clip_threshold: gradient clipping の閾値。``None``(既定値)の場合、
            水平線は描画しない(クリッピングを使わない条件を想定)。
        title: 図のタイトル。
        xlabel: 横軸ラベル。
        ylabel: 縦軸ラベル。
        ax: 描画先の Axes。None なら新規作成する。
        log_scale: 縦軸を対数スケールにするか。

    Returns:
        描画に使った Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))

    steps = np.arange(1, len(gradient_norms) + 1)
    ax.plot(steps, gradient_norms, color="tab:blue", linewidth=1.2, label="gradient norm")
    if clip_threshold is not None:
        ax.axhline(
            clip_threshold,
            color="tab:red",
            linewidth=1.4,
            linestyle="--",
            label=f"clip threshold ({clip_threshold:.3g})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_dual_axis_curves(
    x: Sequence[float],
    left_curves: dict[str, Sequence[float]],
    right_curves: dict[str, Sequence[float]],
    xlabel: str = "",
    left_ylabel: str = "",
    right_ylabel: str = "",
    title: str = "",
    ax: Axes | None = None,
    log_x: bool = False,
) -> Axes:
    """左右で異なる尺度を持つ 2 種類の系列を、共通の x 軸上に重ねて描画する。

    埋め込み行列のパラメータ数(語彙サイズに比例して増加)と、系列長に依存する
    演算量(語彙サイズが増えるほど fertility が下がり減少する)のように、
    スケールの異なる量を同一の x 軸(語彙サイズなど)上で比較する際に使う
    (005 の実験6、語彙サイズと計算量のトレードオフ)。

    Args:
        x: 共通の横軸の値。
        left_curves: 左側 y 軸に描く ``{系列名: 値}`` の辞書。
        right_curves: 右側 y 軸に描く ``{系列名: 値}`` の辞書。
        xlabel: 横軸ラベル。
        left_ylabel: 左側縦軸ラベル。
        right_ylabel: 右側縦軸ラベル。
        title: 図のタイトル。
        ax: 描画先の Axes(左側)。None なら新規作成する。
        log_x: 横軸を対数スケールにするか。

    Returns:
        左側の Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    right_ax = ax.twinx()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lines = []
    for i, (name, values) in enumerate(left_curves.items()):
        (line,) = ax.plot(
            x,
            values,
            color=color_cycle[i % len(color_cycle)],
            linestyle="-",
            marker="o",
            label=f"{name} (left)",
        )
        lines.append(line)
    n_left = len(left_curves)
    for i, (name, values) in enumerate(right_curves.items()):
        (line,) = right_ax.plot(
            x,
            values,
            color=color_cycle[(n_left + i) % len(color_cycle)],
            linestyle="--",
            marker="s",
            label=f"{name} (right)",
        )
        lines.append(line)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(left_ylabel)
    right_ax.set_ylabel(right_ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    return ax
