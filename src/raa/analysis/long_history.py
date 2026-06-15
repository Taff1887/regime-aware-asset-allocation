"""Long-history portfolio backtest (~1990-2026) on real mutual-fund proxies.

The headline ETF backtest (§12.1) is limited to 2005-2026 because the youngest
ETF sleeves only list from 2007-08. Here we rebuild the same strategies on
long-history, low-cost mutual-fund proxies (real total-return NAVs) so the charts
and tables span the maximum real period. Real-asset sleeves (EM, gold) enter
point-in-time via ETFs when their data begins. Some funds are actively managed,
so returns are net of fees and carry modest manager effects — directionally
representative of the asset class, documented as a caveat. No simulated data.

Run with::

    uv run python -m raa.analysis.long_history
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.analysis import by_regime, metrics
from raa.analysis.performance import evaluate_many
from raa.data.fixed_income import _fetch_returns_wide  # helper below
from raa.portfolio.backtest import (
    regime_overlay_strategy,
    regime_strategy,
    run_backtest,
    static_strategy,
)
from raa.regimes.rule_based import REGIME_ORDER, classify_regimes, confirm_regime
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import PALETTE, save_fig

# Optimiser universe (fund proxies + ETFs that join point-in-time).
LONG_BT_ASSETS = [
    "VFINX", "PRITX", "EEM",                       # equities (US, intl, EM)
    "VFISX", "VFITX", "VUSTX",                     # Treasuries (short/int/long)
    "VFICX", "VWESX", "VWEHX",                     # IG int, IG long, high yield
    "RPIBX", "FRESX", "VIPSX", "DBC", "GLD",       # global bonds, REITs, TIPS, commodities, gold
]
EXTRA_FUNDS = ["PRITX", "FRESX"]                   # not in the FI panel; fetch here
SIXTY40 = {"VFINX": 0.6, "VBMFX": 0.4}            # US equity / US aggregate bond
LONG_RISK = {"VFINX", "PRITX", "EEM", "VWEHX", "FRESX", "DBC"}
LONG_DEF = {"VFISX", "VFITX", "VUSTX", "GLD"}
START_LONG = "1990-01-31"
COST_BPS = 10.0
REBAL = 3
LABEL = {
    "VFINX": "US Equity", "PRITX": "Intl Equity", "EEM": "EM Equity",
    "VFISX": "Short Tsy", "VFITX": "Int Tsy", "VUSTX": "Long Tsy",
    "VFICX": "Int IG", "VWESX": "Long IG", "VWEHX": "High Yield",
    "RPIBX": "Global bonds", "FRESX": "REITs", "VIPSX": "TIPS",
    "DBC": "Commodities", "GLD": "Gold",
}
CRISES = {
    "1990 recession": ("1990-07-31", "1991-03-31"),
    "1994 bond crash": ("1994-02-28", "1994-12-31"),
    "Dot-com 2000-02": ("2000-09-30", "2002-09-30"),
    "GFC 2007-09": ("2007-11-30", "2009-02-28"),
    "Euro 2011": ("2011-05-31", "2012-06-30"),
    "COVID 2020": ("2020-02-29", "2020-03-31"),
    "Inflation 2022": ("2022-01-31", "2022-10-31"),
}


def _outdir():
    d = settings.reports_dir / "long_history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def assemble_long_returns() -> pd.DataFrame:
    """Combine FI-fund returns + extra funds + EEM/GLD ETFs into one panel."""
    fi = read_parquet(settings.processed_dir / "returns_monthly_fi.parquet")
    etf = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    extra = _fetch_returns_wide(EXTRA_FUNDS)
    panel = fi.join(extra, how="outer")
    for t in ["EEM", "GLD", "DBC"]:
        if t in etf:
            panel = panel.join(etf[[t]], how="outer")
    # Data-quality guard: a monthly return with |r|>100% is a vendor error for
    # any diversified fund/ETF here (e.g. a mis-adjusted NAV) — drop it. This is
    # cleaning of bad real data, not simulation.
    bad = panel.abs() > 1.0
    n_bad = int(bad.to_numpy().sum())
    if n_bad:
        logger.warning("Dropping {} impossible monthly returns (|r|>100%, vendor errors)", n_bad)
        panel = panel.mask(bad)
    out = settings.processed_dir / "returns_monthly_long.parquet"
    panel.sort_index().to_parquet(out)
    return panel.sort_index()


def build_strategies(returns: pd.DataFrame, regime_conf: pd.Series) -> dict:
    common = dict(cost_bps=COST_BPS, start=START_LONG, rebalance_every=REBAL)
    specs = {
        "60/40": static_strategy("fixed", ["VFINX", "VBMFX"], fixed=SIXTY40),
        "Equal-Weight": static_strategy("equal", LONG_BT_ASSETS),
        "Inverse-Vol": static_strategy("inverse_vol", LONG_BT_ASSETS),
        "ERC (Risk Parity)": static_strategy("erc", LONG_BT_ASSETS),
        "Min-Variance": static_strategy("min_var", LONG_BT_ASSETS),
        "Max-Diversification": static_strategy("max_div", LONG_BT_ASSETS),
        "Max-Sharpe (MVO)": static_strategy("max_sharpe", LONG_BT_ASSETS),
        "Regime ERC": regime_strategy("erc", LONG_BT_ASSETS, regime_conf),
        "Regime Max-Sharpe": regime_strategy("max_sharpe", LONG_BT_ASSETS, regime_conf),
        "Regime Risk Overlay": regime_overlay_strategy(
            LONG_BT_ASSETS, regime_conf, derisk=0.5, risk_sleeve=LONG_RISK, defensive_sleeve=LONG_DEF
        ),
    }
    return {name: run_backtest(returns, fn, name=name, **common) for name, fn in specs.items()}


def run() -> dict:
    out = _outdir()
    returns = assemble_long_returns()
    macro = read_parquet(settings.processed_dir / "macro_monthly_long.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly_long.parquet")["rf"]
    regime = classify_regimes(macro)["regime"]
    regime_conf = confirm_regime(regime, k=3)

    logger.info("Long-history backtest from {} over {} assets", START_LONG, len(LONG_BT_ASSETS))
    strategies = build_strategies(returns, regime_conf)
    perf = evaluate_many(strategies, rf=rf, benchmark_name="60/40")
    order = ["ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
             "var_5", "es_5", "hit_rate", "avg_turnover"]
    perf = perf[[c for c in order if c in perf.columns]]
    write_csv(perf.round(4), out / "long_strategy_performance.csv")

    nets = pd.DataFrame({k: v["net"] for k, v in strategies.items() if not v["net"].empty})
    crisis_dd = _crisis_drawdowns(nets)
    write_csv((crisis_dd * 100).round(1), out / "long_crisis_drawdowns.csv")

    # per-regime asset Sharpe over the long sample
    sharpe_tbl = by_regime.build_metric_tables(returns[LONG_BT_ASSETS], regime, rf=rf)["sharpe"]
    write_csv(sharpe_tbl.reindex(LONG_BT_ASSETS).round(3), out / "long_asset_sharpe_by_regime.csv")

    _fig_equity(strategies)
    _fig_perf_bar(perf)
    _fig_crisis(crisis_dd)
    _fig_asset_sharpe(sharpe_tbl)

    summary = {
        "sample": f"{nets.index.min().date()}..{nets.index.max().date()}",
        "sharpe": perf["sharpe"].round(3).to_dict(),
        "ann_return": perf["ann_return"].round(4).to_dict(),
        "max_drawdown": perf["max_drawdown"].round(3).to_dict(),
        "crisis_drawdowns": (crisis_dd * 100).round(1).to_dict(),
    }
    with open(out / "long_history_findings.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("LONG-HISTORY BACKTEST  ({})", summary["sample"])
    logger.info("Sharpe: {}", summary["sharpe"])
    logger.info("Max drawdown: {}", summary["max_drawdown"])
    logger.info("Crisis drawdowns (%):\n{}", (crisis_dd * 100).round(1).to_string())
    logger.info("=" * 70)
    return summary


def _crisis_drawdowns(nets: pd.DataFrame) -> pd.DataFrame:
    strats = ["60/40", "ERC (Risk Parity)", "Regime Risk Overlay", "Regime Max-Sharpe"]
    rows = {}
    for name, (s, e) in CRISES.items():
        w = nets.loc[pd.Timestamp(s) - pd.Timedelta(days=31): pd.Timestamp(e) + pd.Timedelta(days=31)]
        rows[name] = {st: metrics.max_drawdown(w[st]) for st in strats if st in w}
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- figs
def _cum(net):
    return (1 + net).cumprod()


def _fig_equity(strategies: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(strategies.items()):
        if res["net"].empty:
            continue
        ls = "--" if name.startswith("Regime") else "-"
        lw = 2.0 if name.startswith("Regime") else 1.3
        c = _cum(res["net"])
        ax.plot(c.index, c.values, label=name, lw=lw, ls=ls, color=PALETTE[i % len(PALETTE)])
    ax.set_yscale("log")
    ax.set_title("Long-history cumulative growth of $1 (net of costs, log scale, 1990-2026)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, "01_long_equity_curves", subdir="long_history")


def _fig_perf_bar(perf: pd.DataFrame) -> None:
    s = perf["sharpe"].sort_values()
    colors = ["#C0392B" if n.startswith("Regime") else "#0072B2" for n in s.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s.index, s.values, color=colors)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_title("Long-history Sharpe by strategy (1990-2026, red = regime-aware)")
    ax.set_xlabel("Annualised Sharpe")
    save_fig(fig, "02_long_sharpe_ranking", subdir="long_history")


def _fig_crisis(dd: pd.DataFrame) -> None:
    data = dd * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(data.columns))
    n = len(data.index)
    w = 0.8 / n
    for k, strat in enumerate(data.index):
        ax.bar(x + (k - (n - 1) / 2) * w, data.loc[strat].to_numpy(), w, label=strat)
    ax.set_xticks(x, data.columns, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Max drawdown %")
    ax.set_title("Drawdown through crises (1990-2026): static vs regime-aware")
    ax.legend(fontsize=8)
    save_fig(fig, "03_long_crisis_drawdowns", subdir="long_history")


def _fig_asset_sharpe(tbl: pd.DataFrame) -> None:
    sub = tbl.reindex(LONG_BT_ASSETS)[REGIME_ORDER].astype(float)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sub.to_numpy(), cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(REGIME_ORDER)), REGIME_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(sub.index)), [LABEL.get(t, t) for t in sub.index])
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.to_numpy()[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Asset-class Sharpe by regime (long history)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Sharpe")
    save_fig(fig, "04_long_asset_sharpe", subdir="long_history")


if __name__ == "__main__":
    run()
