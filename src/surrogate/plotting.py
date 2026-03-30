"""Plotting utilities for surrogate model diagnostics.

Requires the optional ``plot`` dependency group::

    uv sync --group plot
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .model import SurrogateModel


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with:  uv sync --group plot"
        ) from exc
    return plt


def _import_seaborn():
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "seaborn is required for this plot. Install it with:  uv sync --group plot"
        ) from exc
    return sns


_STYLE_APPLIED = False

# Colorblind-safe palette (Okabe-Ito, widely recommended for accessibility).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def _apply_style() -> None:
    """Apply a clean, accessible plot style (called once on first plot)."""
    global _STYLE_APPLIED  # noqa: PLW0603
    if _STYLE_APPLIED:
        return

    sns = _import_seaborn()
    sns.set_theme(
        style="ticks",
        context="notebook",
        palette=PALETTE,
        font_scale=1.05,
        rc={
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        },
    )
    _STYLE_APPLIED = True


def parity_plot(
    model: SurrogateModel,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Predicted-vs-actual parity plot using LOO cross-validation.

    Points on the diagonal indicate perfect predictions. Systematic
    deviations reveal regions where the surrogate is biased. This is
    the standard first check after fitting  -- use it to decide whether
    the model is accurate enough or needs more training data.

    Args:
        model: A fitted ``SurrogateModel``.
        ax: Matplotlib axes to draw on. Created if ``None``.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()
    _apply_style()
    loo = model.loo_predict()
    pred = loo["mean"]
    actual = loo["actual"]
    cols = pred.columns

    n_out = len(cols)
    if ax is not None:
        fig = ax.get_figure()
        axes = [ax] if n_out == 1 else [ax]
    else:
        fig, axes = plt.subplots(
            1,
            n_out,
            figsize=(5 * n_out, 4.5),
            squeeze=False,
            layout="constrained",
        )
        axes = axes.ravel()

    for i, col in enumerate(cols):
        a = axes[i] if i < len(axes) else axes[0]
        colour = PALETTE[i % len(PALETTE)]
        a.scatter(
            actual[col],
            pred[col],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            s=40,
            color=colour,
            zorder=3,
        )
        lo = min(actual[col].min(), pred[col].min())
        hi = max(actual[col].max(), pred[col].max())
        margin = (hi - lo) * 0.05
        a.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            color="#888888",
            ls="--",
            lw=1,
            zorder=2,
        )
        a.set_xlabel("Actual")
        a.set_ylabel("Predicted (LOO)")
        a.set_title(col)
        a.set_aspect("equal", adjustable="datalim")

    return fig


def calibration_plot(
    model: SurrogateModel,
    *,
    n_levels: int = 20,
    ax: Axes | None = None,
) -> Figure:
    """Uncertainty calibration curve from LOO predictions.

    For each nominal confidence level, computes the fraction of LOO
    observations that fall within the corresponding credible interval.
    A well-calibrated model follows the diagonal. If the curve falls
    below the diagonal the model is overconfident (intervals too narrow);
    above means it is underconfident. Use this to assess whether the
    reported uncertainty can be trusted for decision-making.

    Args:
        model: A fitted ``SurrogateModel``.
        n_levels: Number of confidence levels to evaluate.
        ax: Matplotlib axes to draw on. Created if ``None``.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()
    _apply_style()
    loo = model.loo_predict()
    pred = loo["mean"].values
    std = loo["std"].values
    actual = loo["actual"].values

    levels = np.linspace(0.05, 0.99, n_levels)
    observed_coverage = np.empty(len(levels))

    for i, level in enumerate(levels):
        z = _norm_ppf((1 + level) / 2)
        lower = pred - z * std
        upper = pred + z * std
        inside = (actual >= lower) & (actual <= upper)
        observed_coverage[i] = inside.mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    else:
        fig = ax.get_figure()

    ax.fill_between(
        [0, 1],
        [0, 1],
        alpha=0.08,
        color="#888888",
        label="_nolegend_",
    )
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1, label="Ideal")
    ax.plot(
        levels,
        observed_coverage,
        "o-",
        markersize=5,
        color=PALETTE[0],
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="Model",
    )
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Uncertainty calibration")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig


def slice_plot(
    model: SurrogateModel,
    X_centre: pd.DataFrame,
    column: str,
    *,
    n_points: int = 100,
    lower: float | None = None,
    upper: float | None = None,
    output_columns: list[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """1D slice plot  -- sweep one input, fix others at centre values.

    Useful for understanding the effect of a single input on each output
    while holding everything else constant. The 95% credible interval
    ribbon shows where the surrogate is uncertain  -- wide ribbons indicate
    sparse training data in that region.

    Args:
        model: A fitted ``SurrogateModel``.
        X_centre: Single-row DataFrame with the central input values.
        column: Name of the input column to sweep.
        n_points: Number of points along the sweep.
        lower: Lower bound for the sweep (defaults to training min).
        upper: Upper bound for the sweep (defaults to training max).
        output_columns: Which output columns to plot (all if ``None``).
        ax: Matplotlib axes to draw on. Created if ``None``.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()
    _apply_style()

    centre = X_centre.iloc[[0]]
    X_sweep = pd.concat([centre] * n_points, ignore_index=True)

    if lower is None or upper is None:
        train_col = model._encoder.inverse_transform(model._train_X)[column]
        if lower is None:
            lower = float(train_col.min())
        if upper is None:
            upper = float(train_col.max())

    X_sweep[column] = np.linspace(lower, upper, n_points)
    result = model.predict(X_sweep)

    cols = output_columns or list(result["mean"].columns)
    n_out = len(cols)

    if ax is not None:
        fig = ax.get_figure()
        axes = [ax]
    else:
        fig, axes = plt.subplots(
            1,
            n_out,
            figsize=(5 * n_out, 4.5),
            squeeze=False,
            layout="constrained",
        )
        axes = axes.ravel()

    x_vals = X_sweep[column].values
    for i, col in enumerate(cols):
        a = axes[i] if i < len(axes) else axes[0]
        colour = PALETTE[i % len(PALETTE)]
        mean = result["mean"][col].values
        lo = result["lower_95"][col].values
        hi = result["upper_95"][col].values

        a.plot(x_vals, mean, color=colour, lw=1.8, label="Mean")
        a.fill_between(x_vals, lo, hi, alpha=0.2, color=colour, label="95% CI")
        a.set_xlabel(column)
        a.set_ylabel(col)
        a.set_title(f"{col} vs {column}")
        a.legend()

    return fig


def correlation_heatmap(
    model: SurrogateModel,
    X: pd.DataFrame,
    *,
    point_index: int = 0,
    n_samples: int = 200,
    ax: Axes | None = None,
) -> Figure:
    """Heatmap of output correlations at a given input point.

    Shows how outputs co-vary according to the surrogate's posterior.
    Strong correlations suggest shared underlying drivers. Most useful
    with three or more outputs, or when deciding whether outputs can
    be modelled independently.

    Args:
        model: A fitted ``SurrogateModel``.
        X: Input DataFrame (correlations estimated at ``point_index``).
        point_index: Which row of ``X`` to show correlations for.
        n_samples: Number of posterior draws for correlation estimation.
        ax: Matplotlib axes to draw on. Created if ``None``.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()
    sns = _import_seaborn()
    _apply_style()

    corr = model.predict_correlation(X, n_samples=n_samples)
    matrix = corr[point_index]
    cols = model._scaler._columns
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    else:
        fig = ax.get_figure()

    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        xticklabels=cols,
        yticklabels=cols,
        vmin=-1,
        vmax=1,
        cmap="PRGn",
        ax=ax,
    )
    ax.set_title(f"Output correlation (point {point_index})")

    return fig


def convergence_plot(
    metrics: list[dict],
    *,
    metric: str = "r2",
    ax: Axes | None = None,
) -> Figure:
    """Plot LOO metric vs training set size for active learning convergence.

    Shows how model accuracy improves as new points are added via
    ``suggest_next``. Use this to decide when to stop collecting data
     -- a plateau indicates diminishing returns from additional experiments.

    Args:
        metrics: List of dicts as returned by ``model.score()``, one per
            active learning iteration. Each must contain the given *metric*
            key.
        metric: Which metric to plot (``"r2"`` or ``"rmse"``).
        ax: Matplotlib axes to draw on. Created if ``None``.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()
    _apply_style()

    if not metrics:
        raise ValueError("metrics list is empty")

    cols = list(metrics[0][metric].keys())
    n_iters = range(1, len(metrics) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    else:
        fig = ax.get_figure()

    for col in cols:
        values = [m[metric][col] for m in metrics]
        ax.plot(n_iters, values, "o-", markersize=4, label=col)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Active learning convergence ({metric.upper()})")
    ax.legend()

    return fig


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (avoids scipy dependency)."""
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
