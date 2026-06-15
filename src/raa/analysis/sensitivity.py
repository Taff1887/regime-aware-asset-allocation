"""Sensitivity / robustness analysis for the regime framework and overlay.

Varies the parameters a sceptical investment committee would probe:
- regime threshold method (expanding / rolling / full median);
- the overlay de-risk factor;
- transaction-cost assumption;
- the regime confirmation lag (whipsaw filter).

Everything uses real historical data and the same point-in-time backtest engine.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from raa.analysis import metrics
from raa.analysis.by_regime import diversification_index, key_pair_correlations
from raa.analysis.phase3 import BT_ASSETS, REBAL, START
from raa.portfolio.backtest import regime_overlay_strategy, run_backtest, static_strategy
from raa.regimes.rule_based import REGIME_ORDER, classify_regimes, confirm_regime
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import save_fig


def _sharpe_dd(net: pd.Series, rf: pd.Series, turn: pd.Series) -> dict:
    return {
        "sharpe": metrics.sharpe(net, rf),
        "ann_return": metrics.ann_return(net),
        "max_drawdown": metrics.max_drawdown(net),
        "avg_turnover": float(turn.mean()),
    }


def run() -> dict:
    out = settings.reports_dir / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    returns = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly.parquet")["rf"]
    macro = read_parquet(settings.processed_dir / "macro_monthly.parquet")
    regime = classify_regimes(macro)["regime"]
    regime_conf = confirm_regime(regime, k=3)

    # 1. Threshold method robustness (descriptive: diversification + eq/bond corr)
    thr_rows = {}
    for m in ["expanding_median", "rolling_median", "full_median"]:
        reg = classify_regimes(macro, threshold=m)["regime"]
        di = diversification_index(returns, reg, assets=["SPY", "EEM", "IEF", "TLT", "LQD", "HYG", "DBC", "GLD", "VNQ"])
        pc = key_pair_correlations(returns, reg).loc["Equity / Intermediate Treasury"]
        thr_rows[m] = {
            **{f"divcorr_{k}": v for k, v in di.round(3).items()},
            **{f"eqbond_{k}": pc[k] for k in REGIME_ORDER},
        }
    thr = pd.DataFrame(thr_rows).T
    write_csv(thr.round(3), out / "sensitivity_threshold.csv")

    # 2. De-risk factor sensitivity
    derisk_rows = {}
    for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
        fn = regime_overlay_strategy(BT_ASSETS, regime_conf, derisk=d)
        bt = run_backtest(returns, fn, cost_bps=10, start=START, rebalance_every=REBAL, name=f"derisk_{d}")
        derisk_rows[d] = _sharpe_dd(bt["net"], rf, bt["turnover"])
    derisk = pd.DataFrame(derisk_rows).T
    derisk.index.name = "derisk_factor"
    write_csv(derisk.round(4), out / "sensitivity_derisk.csv")

    # 3. Transaction-cost sensitivity (overlay vs static ERC)
    cost_rows = {}
    erc_fn = static_strategy("erc", BT_ASSETS)
    ov_fn = regime_overlay_strategy(BT_ASSETS, regime_conf, derisk=0.5)
    for c in [0, 5, 10, 25, 50]:
        erc = run_backtest(returns, erc_fn, cost_bps=c, start=START, rebalance_every=REBAL, name="erc")
        ov = run_backtest(returns, ov_fn, cost_bps=c, start=START, rebalance_every=REBAL, name="ov")
        cost_rows[c] = {"ERC_sharpe": metrics.sharpe(erc["net"], rf),
                        "Overlay_sharpe": metrics.sharpe(ov["net"], rf)}
    cost = pd.DataFrame(cost_rows).T
    cost.index.name = "cost_bps"
    write_csv(cost.round(4), out / "sensitivity_cost.csv")

    # 4. Confirmation-lag sensitivity
    conf_rows = {}
    for k in [1, 3, 6]:
        rc = confirm_regime(regime, k=k)
        fn = regime_overlay_strategy(BT_ASSETS, rc, derisk=0.5)
        bt = run_backtest(returns, fn, cost_bps=10, start=START, rebalance_every=REBAL, name=f"k{k}")
        conf_rows[k] = _sharpe_dd(bt["net"], rf, bt["turnover"])
    conf = pd.DataFrame(conf_rows).T
    conf.index.name = "confirm_k"
    write_csv(conf.round(4), out / "sensitivity_confirm.csv")

    _fig(derisk, cost)
    logger.info("Threshold robustness:\n{}", thr.round(2).to_string())
    logger.info("De-risk sensitivity:\n{}", derisk.round(3).to_string())
    logger.info("Cost sensitivity:\n{}", cost.round(3).to_string())
    logger.info("Confirm-lag sensitivity:\n{}", conf.round(3).to_string())
    return {"threshold": thr, "derisk": derisk, "cost": cost, "confirm": conf}


def _fig(derisk: pd.DataFrame, cost: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(derisk.index, derisk["sharpe"], "o-", color="#0072B2", label="Sharpe")
    ax2 = ax.twinx()
    ax2.plot(derisk.index, derisk["max_drawdown"] * 100, "s--", color="#C0392B", label="Max DD %")
    ax.set_xlabel("De-risk factor (1 = no de-risk)")
    ax.set_ylabel("Sharpe", color="#0072B2")
    ax2.set_ylabel("Max drawdown %", color="#C0392B")
    ax.set_title("Overlay sensitivity to de-risk strength")

    axc = axes[1]
    axc.plot(cost.index, cost["ERC_sharpe"], "o-", label="ERC (static)")
    axc.plot(cost.index, cost["Overlay_sharpe"], "s-", label="Regime Risk Overlay")
    axc.set_xlabel("Transaction cost (bps per unit turnover)")
    axc.set_ylabel("Sharpe")
    axc.set_title("Robustness to transaction costs")
    axc.legend(fontsize=9)
    save_fig(fig, "11_sensitivity", subdir="phase3")


if __name__ == "__main__":
    run()
