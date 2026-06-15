"""Phase 3: portfolio construction + out-of-sample backtesting.

Compares static strategic allocations against regime-aware allocations on real
historical data, point-in-time and net of transaction costs. Crisis, currency
and statistical-validation analyses live in their own modules and are invoked by
:func:`run` once available.

Run with::

    uv run python -m raa.analysis.phase3
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from raa.analysis import crisis, currency, metrics, validation
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

# Multi-asset investable universe for the portfolio backtest (USD total return).
BT_ASSETS = [
    "SPY", "EWA", "EWU", "IEV", "EWJ", "EEM",   # equities
    "SHY", "IEF", "TLT",                          # treasuries
    "LQD", "HYG",                                 # credit
    "IGF", "RWO", "DBC", "GLD",                   # real assets
]
START = "2005-01-31"
COST_BPS = 10.0
REBAL = 3  # quarterly rebalancing


def _outdir():
    d = settings.reports_dir / "phase3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_strategies(returns: pd.DataFrame, regime_conf: pd.Series) -> dict:
    """Construct and backtest the full static + regime-aware strategy set."""
    common = dict(cost_bps=COST_BPS, start=START, rebalance_every=REBAL)
    specs = {
        # --- Static benchmarks ---
        "60/40": static_strategy("fixed", BT_ASSETS, fixed={"SPY": 0.6, "IEF": 0.4}),
        "Equal-Weight": static_strategy("equal", BT_ASSETS),
        "Inverse-Vol": static_strategy("inverse_vol", BT_ASSETS),
        "ERC (Risk Parity)": static_strategy("erc", BT_ASSETS),
        "Min-Variance": static_strategy("min_var", BT_ASSETS),
        "Max-Diversification": static_strategy("max_div", BT_ASSETS),
        "Max-Sharpe (MVO)": static_strategy("max_sharpe", BT_ASSETS),
        # --- Regime-aware ---
        "Regime ERC": regime_strategy("erc", BT_ASSETS, regime_conf),
        "Regime Min-Variance": regime_strategy("min_var", BT_ASSETS, regime_conf),
        "Regime Max-Sharpe": regime_strategy("max_sharpe", BT_ASSETS, regime_conf),
        "Regime Risk Overlay": regime_overlay_strategy(BT_ASSETS, regime_conf, derisk=0.5),
    }
    results = {}
    for name, fn in specs.items():
        results[name] = run_backtest(returns, fn, name=name, **common)
    return results


def run() -> dict:
    settings.ensure_dirs()
    out = _outdir()
    returns = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly.parquet")["rf"]
    macro = read_parquet(settings.processed_dir / "macro_monthly.parquet")

    regime = classify_regimes(macro)["regime"]
    regime_conf = confirm_regime(regime, k=3)

    logger.info("Running {} strategies over {} assets from {}", 9, len(BT_ASSETS), START)
    strategies = build_strategies(returns, regime_conf)

    perf = evaluate_many(strategies, rf=rf, benchmark_name="60/40")
    order = [
        "ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
        "var_5", "es_5", "hit_rate", "avg_turnover", "tracking_error", "info_ratio",
        "up_capture", "down_capture",
    ]
    perf = perf[[c for c in order if c in perf.columns]]
    write_csv(perf.round(4), out / "strategy_performance.csv")
    logger.info("Performance table:\n{}", perf[["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"]].round(3).to_string())

    # Persist net return series for downstream (validation, report).
    nets = pd.DataFrame({k: v["net"] for k, v in strategies.items() if not v["net"].empty})
    nets.to_parquet(settings.processed_dir / "strategy_returns.parquet")

    _fig_equity_curves(strategies)
    _fig_drawdowns(strategies)
    _fig_sharpe_bar(perf)
    _fig_weights(strategies.get("Regime Risk Overlay"))

    # --- Crisis, currency, statistical validation ---
    logger.info("--- Crisis stress tests ---")
    crisis_res = crisis.analyze(nets)
    logger.info("--- Currency hedging (USD investor) ---")
    currency_res = currency.analyze()
    logger.info("--- Statistical validation (block bootstrap) ---")
    valid_res = validation.validate(nets, rf)

    summary = {
        "assets": BT_ASSETS,
        "sample_start": str(nets.index.min().date()),
        "sample_end": str(nets.index.max().date()),
        "sharpe": perf["sharpe"].round(3).to_dict(),
        "max_drawdown": perf["max_drawdown"].round(3).to_dict(),
        "ann_return": perf["ann_return"].round(4).to_dict(),
        "avg_turnover": perf["avg_turnover"].round(3).to_dict() if "avg_turnover" in perf else {},
        "crisis_eqbond_during": crisis_res["corr"]["eqbond_during"].round(2).to_dict(),
        "currency_hedging": currency_res["comparison"].round(3).to_dict(),
        "validation_sharpe_diff": valid_res["sharpe_diff"].to_dict(orient="records"),
    }
    with open(out / "phase3_findings.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("PHASE 3 (core backtest) FINDINGS")
    logger.info("Sample: {} .. {}", summary["sample_start"], summary["sample_end"])
    logger.info("Sharpe: {}", summary["sharpe"])
    logger.info("Max drawdown: {}", summary["max_drawdown"])
    logger.info("=" * 70)
    return {"summary": summary, "strategies": strategies, "perf": perf}


# ------------------------------------------------------------------------- figs
def _cum(net: pd.Series) -> pd.Series:
    return (1 + net).cumprod()


def _fig_equity_curves(strategies: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(strategies.items()):
        if res["net"].empty:
            continue
        ls = "--" if name.startswith("Regime") else "-"
        lw = 2.0 if name.startswith("Regime") else 1.3
        ax.plot(_cum(res["net"]).index, _cum(res["net"]).values, label=name, lw=lw, ls=ls,
                color=PALETTE[i % len(PALETTE)])
    ax.set_yscale("log")
    ax.set_title("Cumulative growth of $1 (net of costs, log scale)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, "01_equity_curves", subdir="phase3")


def _fig_drawdowns(strategies: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in ["60/40", "ERC (Risk Parity)", "Regime ERC", "Regime Min-Variance"]:
        res = strategies.get(name)
        if not res or res["net"].empty:
            continue
        dd = metrics.drawdown_curve(res["net"]) * 100
        ax.plot(dd.index, dd.values, label=name, lw=1.4)
    ax.set_title("Drawdowns: static vs regime-aware")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=9)
    save_fig(fig, "02_drawdowns", subdir="phase3")


def _fig_sharpe_bar(perf: pd.DataFrame) -> None:
    s = perf["sharpe"].sort_values()
    colors = ["#C0392B" if n.startswith("Regime") else "#0072B2" for n in s.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s.index, s.values, color=colors)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_title("Sharpe ratio by strategy (red = regime-aware)")
    ax.set_xlabel("Annualised Sharpe")
    save_fig(fig, "03_sharpe_ranking", subdir="phase3")


def _fig_weights(res: dict | None) -> None:
    if not res or res.get("weights") is None or res["weights"].empty:
        return
    w = res["weights"].clip(lower=0)
    w = w.loc[:, (w.abs().sum() > 0.01)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(w.index, w.T.values, labels=w.columns, colors=[PALETTE[i % len(PALETTE)] for i in range(w.shape[1])])
    ax.set_title(f"Weight evolution: {res['name']}")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save_fig(fig, "04_regime_weights", subdir="phase3")


if __name__ == "__main__":
    run()
