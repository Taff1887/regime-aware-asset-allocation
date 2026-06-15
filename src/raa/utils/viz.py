"""Plotting style and helpers for publication-quality figures.

A single, consistent house style is applied so every figure across the report
shares fonts, grid, palette and DPI. Colours are colour-blind friendly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from raa.utils.config import settings

# Colour-blind-friendly qualitative palette (Okabe-Ito inspired).
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#999999",  # grey
]

# Fixed colours for the four macro regimes, used consistently everywhere.
REGIME_COLORS: dict[str, str] = {
    "Goldilocks": "#2E8B57",   # sea green  (low inflation, high growth)
    "Reflation": "#E69F00",    # amber      (alt label for overheating)
    "Overheating": "#E69F00",  # amber      (high inflation, high growth)
    "Stagflation": "#C0392B",  # red        (high inflation, low growth)
    "Recession": "#5D6D7E",    # slate grey (low inflation, low growth)
}


def set_style() -> None:
    """Apply the project-wide matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "figure.figsize": (10, 6),
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def save_fig(fig: plt.Figure, name: str, subdir: str | None = None) -> Path:
    """Save ``fig`` as a PNG under the figures directory and return the path."""
    out_dir = settings.figures_dir if subdir is None else settings.figures_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# Apply the style as soon as the module is imported.
set_style()
