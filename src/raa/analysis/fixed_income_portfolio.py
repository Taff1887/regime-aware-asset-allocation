"""Fixed-income-ONLY portfolio construction (the bond analogue of stock picking).

Treats fixed-income *sleeves* as the building blocks and asks how best to allocate
*within* a bond book — the question an insurer (≈90% fixed income) actually faces.
Sleeves map to the underlying risks:

- rates / duration : short / intermediate / long Treasuries (VFISX/VFITX/VUSTX)
- credit           : intermediate & long IG, high yield (VFICX/VWESX/VWEHX)
- inflation        : TIPS (VIPSX)
- currency / global: unhedged international bonds (RPIBX)

We backtest 1995-2026 (real fund data), net of costs, point-in-time, versus the
US Aggregate (VBMFX) benchmark, comparing naive (1/N), optimiser-based, heuristic
building-block and a regime "credit de-risk" overlay. Calmar (return per unit of
drawdown) is reported alongside Sharpe.

Run with::

    uv run python -m raa.analysis.fixed_income_portfolio
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.analysis import metrics
from raa.analysis.performance import evaluate_many
from raa.portfolio.backtest import (
    regime_overlay_strategy,
    regime_strategy,
    run_backtest,
    static_strategy,
)
from raa.regimes.rule_based import classify_regimes, confirm_regime
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import PALETTE, save_fig

FI_ASSETS = ["VFISX", "VFITX", "VUSTX", "VFICX", "VWESX", "VWEHX", "RPIBX", "VIPSX"]
DOMESTIC = ["VFISX", "VFITX", "VUSTX", "VFICX", "VWESX", "VWEHX", "VIPSX"]  # ex-global
FI_RISK = {"VFICX", "VWESX", "VWEHX", "RPIBX"}      # credit + currency (the "risk-on" sleeves)
FI_DEF = {"VFISX", "VFITX", "VUSTX", "VIPSX"}       # governments + inflation-linked
BENCH = "US Aggregate (VBMFX)"
START = "1995-01-31"
COST_BPS = 10.0
REBAL = 3
CRISES = {
    "Dot-com 2000-02": ("2000-09-30", "2002-09-30"),
    "GFC 2007-09": ("2007-11-30", "2009-02-28"),
    "Euro 2011": ("2011-05-31", "2012-06-30"),
    "COVID 2020": ("2020-02-29", "2020-03-31"),
    "Inflation 2022": ("2022-01-31", "2022-10-31"),
}


def _outdir():
    d = settings.reports_dir / "fi_portfolio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_strategies(returns: pd.DataFrame, regime_conf: pd.Series) -> dict:
    common = dict(cost_bps=COST_BPS, start=START, rebalance_every=REBAL)
    specs = {
        BENCH: static_strategy("fixed", ["VBMFX"], fixed={"VBMFX": 1.0}),
        "Equal-Weight (1/N)": static_strategy("equal", FI_ASSETS),
        "Treasury Ladder (govt only)": static_strategy(
            "fixed", ["VFISX", "VFITX", "VUSTX"], fixed={"VFISX": 1 / 3, "VFITX": 1 / 3, "VUSTX": 1 / 3}
        ),
        "Credit Tilt": static_strategy(
            "fixed", ["VFITX", "VUSTX", "VFICX", "VWESX", "VWEHX"],
            fixed={"VFITX": 0.15, "VUSTX": 0.10, "VFICX": 0.25, "VWESX": 0.25, "VWEHX": 0.25},
        ),
        "Domestic 1/N (no global)": static_strategy("equal", DOMESTIC),
        "Inverse-Vol": static_strategy("inverse_vol", FI_ASSETS),
        "ERC (Risk Parity)": static_strategy("erc", FI_ASSETS),
        "Min-Variance": static_strategy("min_var", FI_ASSETS),
        "Max-Sharpe (MVO)": static_strategy("max_sharpe", FI_ASSETS),
        "Regime ERC": regime_strategy("erc", FI_ASSETS, regime_conf),
        "Regime Credit De-risk": regime_overlay_strategy(
            FI_ASSETS, regime_conf, derisk=0.5, risk_sleeve=FI_RISK, defensive_sleeve=FI_DEF
        ),
    }
    return {name: run_backtest(returns, fn, name=name, **common) for name, fn in specs.items()}


def run() -> dict:
    out = _outdir()
    returns = read_parquet(settings.processed_dir / "returns_monthly_fi.parquet")
    macro = read_parquet(settings.processed_dir / "macro_monthly_long.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly_long.parquet")["rf"]
    regime_conf = confirm_regime(classify_regimes(macro)["regime"], k=3)

    logger.info("Fixed-income-only backtest from {} over {} sleeves", START, len(FI_ASSETS))
    strategies = build_strategies(returns, regime_conf)
    perf = evaluate_many(strategies, rf=rf, benchmark_name=BENCH)
    order = ["ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
             "var_5", "es_5", "avg_turnover", "info_ratio"]
    perf = perf[[c for c in order if c in perf.columns]]
    write_csv(perf.round(4), out / "fi_portfolio_performance.csv")

    nets = pd.DataFrame({k: v["net"] for k, v in strategies.items() if not v["net"].empty})
    crisis_dd = _crisis_drawdowns(nets)
    write_csv((crisis_dd * 100).round(1), out / "fi_portfolio_crisis_drawdowns.csv")

    _fig_equity(strategies)
    _fig_sharpe_calmar(perf)
    _fig_crisis(crisis_dd)

    summary = {
        "sample": f"{nets.index.min().date()}..{nets.index.max().date()}",
        "sharpe": perf["sharpe"].round(3).to_dict(),
        "calmar": perf["calmar"].round(3).to_dict(),
        "max_drawdown": perf["max_drawdown"].round(3).to_dict(),
        "crisis_drawdowns": (crisis_dd * 100).round(1).to_dict(),
    }
    with open(out / "fi_portfolio_findings.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("FIXED-INCOME-ONLY PORTFOLIOS ({})", summary["sample"])
    logger.info("Performance:\n{}", perf.round(3)[["ann_return", "ann_vol", "sharpe", "calmar", "max_drawdown", "avg_turnover"]].to_string())
    logger.info("Crisis drawdowns (%):\n{}", (crisis_dd * 100).round(1).to_string())
    logger.info("=" * 70)
    return summary


def _crisis_drawdowns(nets: pd.DataFrame) -> pd.DataFrame:
    strats = [BENCH, "Treasury Ladder (govt only)", "Credit Tilt", "Regime Credit De-risk"]
    rows = {}
    for name, (s, e) in CRISES.items():
        w = nets.loc[pd.Timestamp(s) - pd.Timedelta(days=31): pd.Timestamp(e) + pd.Timedelta(days=31)]
        rows[name] = {st: metrics.max_drawdown(w[st]) for st in strats if st in w}
    return pd.DataFrame(rows)


def _cum(net):
    return (1 + net).cumprod()


def _fig_equity(strategies: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(strategies.items()):
        if res["net"].empty:
            continue
        ls = "--" if name.startswith("Regime") else ("-." if name == BENCH else "-")
        lw = 2.2 if name.startswith("Regime") else (2.0 if name == BENCH else 1.2)
        c = _cum(res["net"])
        ax.plot(c.index, c.values, label=name, lw=lw, ls=ls, color=PALETTE[i % len(PALETTE)])
    ax.set_yscale("log")
    ax.set_title("Fixed-income-only portfolios: growth of $1 (net of costs, 1995-2026)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, "01_fi_portfolio_equity", subdir="fi_portfolio")


def _fig_sharpe_calmar(perf: pd.DataFrame) -> None:
    p = perf.sort_values("sharpe")
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y - 0.2, p["sharpe"], 0.4, color="#0072B2", label="Sharpe")
    ax.barh(y + 0.2, p["calmar"], 0.4, color="#D55E00", label="Calmar")
    ax.set_yticks(y, p.index, fontsize=9)
    for i, (s, c) in enumerate(zip(p["sharpe"], p["calmar"], strict=False)):
        ax.text(s, i - 0.2, f" {s:.2f}", va="center", fontsize=7)
        ax.text(c, i + 0.2, f" {c:.2f}", va="center", fontsize=7)
    ax.set_title("Fixed-income portfolios: Sharpe and Calmar")
    ax.legend()
    save_fig(fig, "02_fi_portfolio_sharpe_calmar", subdir="fi_portfolio")


def _fig_crisis(dd: pd.DataFrame) -> None:
    data = dd * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(data.columns))
    n = len(data.index)
    w = 0.8 / n
    for k, strat in enumerate(data.index):
        ax.bar(x + (k - (n - 1) / 2) * w, data.loc[strat].to_numpy(), w, label=strat)
    ax.set_xticks(x, data.columns, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Max drawdown %")
    ax.set_title("Fixed-income portfolios: drawdown through crises")
    ax.legend(fontsize=8)
    save_fig(fig, "03_fi_portfolio_crisis", subdir="fi_portfolio")


if __name__ == "__main__":
    run()
