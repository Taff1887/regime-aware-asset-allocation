"""Fixed-income deep dive over the long sample (~1980-2026, real data).

Answers the questions an insurer's (fixed-income-heavy) committee cares about:
- how does each bond type behave in each macro regime?
- when does credit beat government? when do global bonds beat US?
- when does long duration beat short? when do TIPS beat nominal?
- does the bond-equity hedge fail in inflation regimes over the *long* history
  (including the 1980s), not just in 2022?

Run with::

    uv run python -m raa.analysis.fixed_income
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.analysis import by_regime
from raa.regimes.rule_based import REGIME_ORDER, classify_regimes
from raa.reporting.charts import regime_legend_handles, shade_regimes
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import REGIME_COLORS, save_fig

ORDER = ["VFISX", "VFITX", "VUSTX", "VBMFX", "VFICX", "VWESX", "VWEHX", "RPIBX", "VIPSX", "VFINX"]
LABEL = {
    "VFISX": "Short Treasury", "VFITX": "Int Treasury", "VUSTX": "Long Treasury",
    "VBMFX": "US Aggregate", "VFICX": "Int IG credit", "VWESX": "Long IG credit",
    "VWEHX": "High Yield", "RPIBX": "Global ex-US", "VIPSX": "TIPS", "VFINX": "US Equity",
}
# (A, B, label) -> annualised mean of (A - B) by regime; positive => A beats B
SPREADS = [
    ("VWESX", "VUSTX", "Long credit − Long Treasury"),
    ("VFICX", "VFITX", "Int credit − Int Treasury"),
    ("VWEHX", "VWESX", "High yield − IG credit"),
    ("RPIBX", "VBMFX", "Global ex-US − US Agg"),
    ("VUSTX", "VFISX", "Long − Short Treasury (duration)"),
    ("VIPSX", "VFITX", "TIPS − nominal Treasury"),
]
BOND_EQUITY = [("VBMFX", "US Aggregate"), ("VUSTX", "Long Treasury"), ("VWESX", "Long IG credit")]


def _outdir():
    d = settings.reports_dir / "fixed_income"
    d.mkdir(parents=True, exist_ok=True)
    return d


def spread_by_regime(rets: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    reg = regime.dropna().astype(str)
    idx = rets.index.intersection(reg.index)
    rets, reg = rets.loc[idx], reg.loc[idx]
    rows = {}
    for a, b, label in SPREADS:
        if a not in rets or b not in rets:
            continue
        sp = (rets[a] - rets[b]).dropna()
        row = {}
        for r in REGIME_ORDER:
            s = sp.loc[reg.reindex(sp.index) == r]
            row[r] = float(s.mean() * 12) if len(s) >= 6 else np.nan
        row["All"] = float(sp.mean() * 12)
        rows[label] = row
    return pd.DataFrame(rows).T


def bond_equity_corr_by_regime(rets: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    reg = regime.dropna().astype(str)
    idx = rets.index.intersection(reg.index)
    rets, reg = rets.loc[idx], reg.loc[idx]
    rows = {}
    for tic, label in BOND_EQUITY:
        if tic not in rets or "VFINX" not in rets:
            continue
        row = {}
        for r in [*REGIME_ORDER, "All"]:
            sub = rets if r == "All" else rets.loc[reg == r]
            s = sub[[tic, "VFINX"]].dropna()
            row[r] = float(s[tic].corr(s["VFINX"])) if len(s) >= 12 else np.nan
        rows[label] = row
    return pd.DataFrame(rows).T


def run() -> dict:
    out = _outdir()
    rets = read_parquet(settings.processed_dir / "returns_monthly_fi.parquet")
    macro = read_parquet(settings.processed_dir / "macro_monthly_long.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly_long.parquet")["rf"]
    panel = classify_regimes(macro)
    regime = panel["regime"]

    # per-(fund, regime) metrics
    tables = by_regime.build_metric_tables(rets, regime, rf=rf)
    for name, tbl in tables.items():
        write_csv(tbl.reindex(ORDER).round(4), out / f"fi_{name}_by_regime.csv")

    spreads = spread_by_regime(rets, regime)
    be_corr = bond_equity_corr_by_regime(rets, regime)
    counts = by_regime.regime_counts(regime.loc[regime.index.intersection(rets.index)])
    write_csv(spreads.round(4), out / "fi_spreads_by_regime.csv")
    write_csv(be_corr.round(3), out / "fi_bond_equity_corr.csv")
    write_csv(counts, out / "fi_regime_months.csv")

    _fig_heatmap(tables["sharpe"], "Sharpe ratio", "01_fi_sharpe_by_regime", -1.5, 1.5)
    _fig_heatmap(tables["ann_return"] * 100, "Annualised return %", "02_fi_return_by_regime", -10, 15)
    _fig_spreads(spreads)
    _fig_timeline(panel, rets.index.min())
    _fig_bond_equity(rets, regime, be_corr)

    summary = {
        "sample": f"{rets.index.min().date()}..{rets.index.max().date()}",
        "regime_months_in_fi_sample": counts["months"].to_dict(),
        "sharpe_by_regime": tables["sharpe"].reindex(ORDER)[REGIME_ORDER].round(2).to_dict(),
        "spreads_by_regime": spreads[REGIME_ORDER].round(4).to_dict(),
        "bond_equity_corr": be_corr[REGIME_ORDER].round(3).to_dict(),
    }
    with open(out / "fixed_income_findings.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("FIXED-INCOME DEEP DIVE  (sample {})", summary["sample"])
    logger.info("Regime months in FI sample: {}", summary["regime_months_in_fi_sample"])
    logger.info("Spreads (ann.) by regime:\n{}", spreads[REGIME_ORDER].round(3).to_string())
    logger.info("Bond-equity correlation by regime:\n{}", be_corr[REGIME_ORDER].round(2).to_string())
    logger.info("=" * 70)
    return summary


# ------------------------------------------------------------------------- figs
def _fig_heatmap(tbl: pd.DataFrame, title: str, fname: str, vmin: float, vmax: float) -> None:
    sub = tbl.reindex(ORDER)[REGIME_ORDER].astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(sub.to_numpy(), cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(REGIME_ORDER)), REGIME_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(sub.index)), [LABEL.get(t, t) for t in sub.index])
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.to_numpy()[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Fixed income: {title} by regime\n(real data, ~1980-2026)")
    fig.colorbar(im, ax=ax, shrink=0.7, label=title)
    save_fig(fig, fname, subdir="fixed_income")


def _fig_spreads(spreads: pd.DataFrame) -> None:
    sub = (spreads[REGIME_ORDER] * 100).astype(float)
    cats = list(sub.index)
    x = np.arange(len(cats))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for k, r in enumerate(REGIME_ORDER):
        ax.bar(x + (k - 1.5) * w, sub[r].to_numpy(), w, label=r, color=REGIME_COLORS[r])
    ax.set_xticks(x, cats, rotation=20, ha="right", fontsize=9)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Annualised outperformance (%)")
    ax.set_title("When does each fixed-income trade win? (A − B, annualised, by regime)")
    ax.legend(fontsize=8, ncol=4)
    save_fig(fig, "03_fi_spreads_by_regime", subdir="fixed_income")


def _fig_timeline(panel: pd.DataFrame, start) -> None:
    p = panel.loc[panel.index >= pd.Timestamp("1965-01-31")]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(p.index, p["infl_signal"], color="#D55E00", lw=1.2, label="Inflation: CPI YoY (3m)")
    ax.plot(p.index, p["growth_signal"], color="#0072B2", lw=1.0, alpha=0.8, label="Growth: IndPro YoY (3m)")
    ax.axhline(0, color="#999", lw=0.7)
    shade_regimes(ax, p["regime"])
    ax.axvline(start, color="#222", ls=":", lw=1.2)
    ax.text(start, ax.get_ylim()[1] * 0.9, " bond-fund data begins", fontsize=8)
    ax.set_title("Long-sample macro regimes (1965-2026): the Great Inflation gives real inflation-regime history")
    ax.set_ylabel("Year-on-year %")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + regime_legend_handles(), labels + REGIME_ORDER, fontsize=7, ncol=3, loc="upper right")
    save_fig(fig, "04_fi_regime_timeline_long", subdir="fixed_income")


def _fig_bond_equity(rets: pd.DataFrame, regime: pd.Series, be_corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    # rolling corr
    if {"VBMFX", "VFINX"} <= set(rets):
        roll = rets["VBMFX"].rolling(36).corr(rets["VFINX"]).dropna()
        axes[0].plot(roll.index, roll.values, color="#222", lw=1.3)
        axes[0].axhline(0, color="#999", lw=0.8)
        shade_regimes(axes[0], regime.reindex(roll.index))
        axes[0].set_title("Rolling 36m bond-equity correlation (US Agg vs S&P 500)")
        axes[0].set_ylabel("Correlation")
        axes[0].legend(handles=regime_legend_handles(), fontsize=7, ncol=2, loc="upper left")
    # by regime bars
    sub = be_corr[REGIME_ORDER].astype(float)
    x = np.arange(len(sub.index))
    w = 0.2
    for k, r in enumerate(REGIME_ORDER):
        axes[1].bar(x + (k - 1.5) * w, sub[r].to_numpy(), w, label=r, color=REGIME_COLORS[r])
    axes[1].set_xticks(x, sub.index, rotation=15, ha="right", fontsize=8)
    axes[1].axhline(0, color="#444", lw=0.8)
    axes[1].set_title("Bond-equity correlation by regime")
    axes[1].legend(fontsize=7, ncol=2)
    save_fig(fig, "05_fi_bond_equity_corr", subdir="fixed_income")


if __name__ == "__main__":
    run()
