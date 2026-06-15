"""Reusable plotting helpers shared across phases."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.regimes.rule_based import REGIME_ORDER, regime_episodes
from raa.utils.viz import REGIME_COLORS


def shade_regimes(ax: plt.Axes, regime: pd.Series, alpha: float = 0.18) -> None:
    """Shade the background of a time-series axis by regime episode."""
    eps = regime_episodes(regime)
    for _, row in eps.iterrows():
        color = REGIME_COLORS.get(row["regime"], "#cccccc")
        # extend the band to the end of the episode's final month
        end = row["end"] + pd.offsets.MonthEnd(1)
        ax.axvspan(row["start"], end, color=color, alpha=alpha, lw=0)


def regime_legend_handles():
    """Patch handles for a regime colour legend."""
    from matplotlib.patches import Patch

    return [Patch(facecolor=REGIME_COLORS[r], alpha=0.5, label=r) for r in REGIME_ORDER]


def corr_heatmap(ax: plt.Axes, corr: pd.DataFrame, title: str, annot: bool = True) -> None:
    """Render a correlation matrix as a blue-white-red heatmap on ``ax``."""
    data = corr.to_numpy()
    im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(title, fontsize=10)
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black" if abs(v) < 0.6 else "white",
                    )
    return im


def grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str = "",
    ylabel: str = "",
    regime_colors: bool = True,
) -> None:
    """Grouped bar chart: index = categories on x, columns = series (regimes)."""
    cats = list(df.index)
    series = list(df.columns)
    n = len(series)
    x = np.arange(len(cats))
    width = 0.8 / max(1, n)
    for k, s in enumerate(series):
        color = REGIME_COLORS.get(s) if regime_colors else None
        ax.bar(x + (k - (n - 1) / 2) * width, df[s].to_numpy(), width, label=s, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=min(n, 4))
