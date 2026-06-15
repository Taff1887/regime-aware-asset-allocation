"""Phase 1: the simple four-quadrant regime analysis (go/no-go checkpoint).

Establishes whether simple Growth x Inflation regimes carry information for asset
behaviour *before* any advanced modelling. Produces:

- per-(asset, regime) return / vol / Sharpe / drawdown / VaR / ES tables
- full correlation matrices within each regime
- key diversifier-pair correlations across regimes
- an average-pairwise-correlation diversification index by regime
- regime timeline, episodes, transition matrix
- publication-quality figures

Run with::

    uv run python -m raa.analysis.phase1
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from raa.analysis import by_regime
from raa.data.universe import label_map
from raa.regimes.rule_based import (
    REGIME_ORDER,
    classify_regimes,
    regime_episodes,
    transition_matrix,
)
from raa.reporting.charts import (
    corr_heatmap,
    grouped_bars,
    regime_legend_handles,
    shade_regimes,
)
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import save_fig

# Representative cross-asset set for readable charts/correlation matrices.
ANALYSIS_SET = ["SPY", "EEM", "IEF", "TLT", "LQD", "HYG", "DBC", "GLD", "VNQ"]
# Subset highlighted in the Sharpe-by-regime chart.
HIGHLIGHT = ["SPY", "IEF", "TLT", "LQD", "HYG", "DBC", "GLD"]


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    macro = read_parquet(settings.processed_dir / "macro_monthly.parquet")
    rets = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly.parquet")["rf"]
    return macro, rets, rf


def _outdir(sub: str):
    d = settings.reports_dir / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def run() -> dict:
    settings.ensure_dirs()
    macro, rets, rf = _load()
    reports = _outdir("phase1")

    # --- 1. Classify regimes (core, point-in-time) -----------------------------
    panel = classify_regimes(macro)
    regime = panel["regime"]
    # Persist a committed, FMP-free regime label file + the full panel.
    write_csv(panel[["growth_signal", "infl_signal", "regime"]], settings.processed_dir / "regime_labels.csv")
    panel.to_parquet(settings.processed_dir / "regime_panel.parquet")

    counts = by_regime.regime_counts(regime)
    episodes = regime_episodes(regime)
    trans = transition_matrix(regime)
    write_csv(counts, reports / "regime_counts.csv")
    write_csv(episodes, reports / "regime_episodes.csv", index=False)
    write_csv(trans, reports / "transition_matrix.csv")

    logger.info("Regime month counts:\n{}", counts.to_string())

    # --- 2. Per-(asset, regime) metric tables ----------------------------------
    tables = by_regime.build_metric_tables(rets, regime, rf=rf)
    for name, tbl in tables.items():
        write_csv(tbl.round(4), reports / f"asset_{name}_by_regime.csv")

    # --- 3. Correlation matrices + diversification -----------------------------
    corr = by_regime.regime_corr_matrices(rets, regime, assets=ANALYSIS_SET)
    for label, mat in corr.items():
        write_csv(mat.round(3), reports / f"corr_{label}.csv")
    pair_corr = by_regime.key_pair_correlations(rets, regime)
    write_csv(pair_corr.round(3), reports / "key_pair_correlations.csv")
    div_index = by_regime.diversification_index(rets, regime, assets=ANALYSIS_SET)
    write_csv(div_index.round(3).to_frame("avg_pairwise_corr"), reports / "diversification_index.csv")

    # --- 4. Figures ------------------------------------------------------------
    _fig_timeline(panel)
    _fig_regime_counts(counts)
    _fig_sharpe_by_regime(tables["sharpe"])
    _fig_annreturn_heatmap(tables["ann_return"])
    _fig_corr_small_multiples(corr)
    _fig_rolling_equity_bond(rets, regime)
    _fig_key_pairs(pair_corr)
    _fig_diversification(div_index)

    summary = _summarise(tables, pair_corr, div_index, counts)
    _write_findings(summary, reports)
    return summary


# --------------------------------------------------------------------------- figs
def _fig_timeline(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    ax = axes[0]
    ax.plot(panel.index, panel["growth_signal"], color="#0072B2", lw=1.3, label="Growth: IndPro YoY (3m)")
    ax.plot(panel.index, panel["infl_signal"], color="#D55E00", lw=1.3, label="Inflation: CPI YoY (3m)")
    ax.plot(panel.index, panel["growth_ref"], color="#0072B2", lw=0.8, ls="--", alpha=0.6)
    ax.plot(panel.index, panel["infl_ref"], color="#D55E00", lw=0.8, ls="--", alpha=0.6)
    shade_regimes(ax, panel["regime"])
    ax.set_title("Macro regime timeline: Growth x Inflation signals vs point-in-time trend")
    ax.set_ylabel("Year-on-year %")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + regime_legend_handles(), labels + REGIME_ORDER,
              loc="upper left", fontsize=7, ncol=3)

    ax2 = axes[1]
    if "fedfunds" in panel:
        ax2.plot(panel.index, panel["fedfunds"], color="#444", lw=1.0, label="Fed funds rate")
    if "slope_10y_3m" in panel:
        ax2.plot(panel.index, panel["slope_10y_3m"], color="#009E73", lw=1.0, label="10y-3m slope")
    shade_regimes(ax2, panel["regime"])
    ax2.axhline(0, color="#999", lw=0.7)
    ax2.set_ylabel("%")
    ax2.legend(loc="upper left", fontsize=8)
    save_fig(fig, "01_regime_timeline", subdir="phase1")


def _fig_regime_counts(counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    from raa.utils.viz import REGIME_COLORS

    ax.bar(counts.index, counts["months"], color=[REGIME_COLORS[r] for r in counts.index])
    for i, (_r, row) in enumerate(counts.iterrows()):
        ax.text(i, row["months"], f"{int(row['months'])}\n{row['share']:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Months per regime (full sample)")
    ax.set_ylabel("Months")
    save_fig(fig, "02_regime_counts", subdir="phase1")


def _fig_sharpe_by_regime(sharpe_tbl: pd.DataFrame) -> None:
    sub = sharpe_tbl.loc[[a for a in HIGHLIGHT if a in sharpe_tbl.index], REGIME_ORDER]
    sub.index = [label_map().get(a, a).split("(")[0].strip() for a in sub.index]
    fig, ax = plt.subplots(figsize=(11, 6))
    grouped_bars(ax, sub, title="Sharpe ratio by asset class and regime", ylabel="Annualised Sharpe")
    save_fig(fig, "03_sharpe_by_regime", subdir="phase1")


def _fig_annreturn_heatmap(ret_tbl: pd.DataFrame) -> None:
    sub = ret_tbl.loc[[a for a in ANALYSIS_SET if a in ret_tbl.index], REGIME_ORDER] * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-25, vmax=25)
    ax.set_xticks(range(len(REGIME_ORDER)), REGIME_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(sub.index)), sub.index)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.to_numpy()[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("Annualised return by asset and regime")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Ann. return %")
    save_fig(fig, "04_annreturn_heatmap", subdir="phase1")


def _fig_corr_small_multiples(corr: dict[str, pd.DataFrame]) -> None:
    labels = [r for r in REGIME_ORDER if r in corr]
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    im = None
    for ax, label in zip(axes, labels, strict=False):
        im = corr_heatmap(ax, corr[label], f"{label}")
    fig.suptitle("Cross-asset correlation matrix within each regime", y=1.02, fontsize=13, weight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="correlation")
    save_fig(fig, "05_corr_by_regime", subdir="phase1")


def _fig_rolling_equity_bond(rets: pd.DataFrame, regime: pd.Series) -> None:
    if "SPY" not in rets or "IEF" not in rets:
        return
    roll = rets["SPY"].rolling(24).corr(rets["IEF"]).dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll.index, roll.values, color="#222", lw=1.4)
    ax.axhline(0, color="#999", lw=0.8)
    shade_regimes(ax, regime.reindex(roll.index))
    ax.set_title("Rolling 24-month equity-bond correlation (SPY vs IEF) with regime shading")
    ax.set_ylabel("Correlation")
    ax.legend(handles=regime_legend_handles(), loc="upper left", fontsize=8, ncol=4)
    save_fig(fig, "06_rolling_equity_bond_corr", subdir="phase1")


def _fig_key_pairs(pair_corr: pd.DataFrame) -> None:
    sub = pair_corr[REGIME_ORDER]  # index = pair names, cols = regimes
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped_bars(ax, sub, title="Key diversifier-pair correlations across regimes", ylabel="Correlation")
    save_fig(fig, "07_key_pair_correlations", subdir="phase1")


def _fig_diversification(div_index: pd.Series) -> None:
    from raa.utils.viz import REGIME_COLORS

    d = div_index.reindex([r for r in REGIME_ORDER if r in div_index.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(d.index, d.values, color=[REGIME_COLORS[r] for r in d.index])
    for i, v in enumerate(d.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Diversification index: average pairwise correlation by regime\n(higher = less diversification available)")
    ax.set_ylabel("Avg pairwise correlation")
    save_fig(fig, "08_diversification_index", subdir="phase1")


# ----------------------------------------------------------------------- summary
def _summarise(tables, pair_corr, div_index, counts) -> dict:
    eq_bond = pair_corr.loc["Equity / Intermediate Treasury", REGIME_ORDER] if "Equity / Intermediate Treasury" in pair_corr.index else None
    sharpe = tables["sharpe"]
    return {
        "regime_counts": counts["months"].to_dict(),
        "equity_bond_corr_by_regime": eq_bond.round(3).to_dict() if eq_bond is not None else {},
        "diversification_index": div_index.round(3).to_dict(),
        "spy_sharpe": sharpe.loc["SPY", REGIME_ORDER].round(2).to_dict() if "SPY" in sharpe.index else {},
        "ief_sharpe": sharpe.loc["IEF", REGIME_ORDER].round(2).to_dict() if "IEF" in sharpe.index else {},
        "gld_sharpe": sharpe.loc["GLD", REGIME_ORDER].round(2).to_dict() if "GLD" in sharpe.index else {},
        "dbc_sharpe": sharpe.loc["DBC", REGIME_ORDER].round(2).to_dict() if "DBC" in sharpe.index else {},
    }


def _write_findings(summary: dict, reports) -> None:
    import json

    with open(reports / "phase1_findings.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("PHASE 1 FINDINGS (go/no-go)")
    logger.info("Regime months: {}", summary["regime_counts"])
    logger.info("Equity/Bond corr by regime (SPY/IEF): {}", summary["equity_bond_corr_by_regime"])
    logger.info("Diversification index by regime: {}", summary["diversification_index"])
    logger.info("SPY Sharpe by regime: {}", summary["spy_sharpe"])
    logger.info("IEF Sharpe by regime: {}", summary["ief_sharpe"])
    logger.info("GLD Sharpe by regime: {}", summary["gld_sharpe"])
    logger.info("DBC Sharpe by regime: {}", summary["dbc_sharpe"])
    logger.info("=" * 70)


if __name__ == "__main__":
    run()
