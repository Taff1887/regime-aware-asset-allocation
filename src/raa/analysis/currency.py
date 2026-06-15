"""Currency hedging analysis from a USD investor's perspective.

The USD-listed foreign-equity ETFs (EWA, EWU, IEV, EWJ) are *unhedged*: their USD
return embeds both the local-market move and the currency move. We reconstruct
hedged and partially-hedged returns by stripping the FX return:

    R_unhedged(USD) ~= R_local + R_fx
    R_hedged        =  R_unhedged - hedge_ratio * R_fx

This ignores hedging carry (forward points ~ rate differential); the assumption
is stated explicitly. We then compare fully hedged / unhedged / 50% hedged global
equity baskets on risk and return, quantify the FX volatility contribution, and
test whether hedging matters more in particular regimes.

All inputs are REAL historical data (FMP spot FX + ETF total returns).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.analysis import metrics
from raa.data.fmp_client import FMPClient
from raa.data.market import _paginate_light
from raa.regimes.rule_based import REGIME_ORDER, classify_regimes
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import REGIME_COLORS, save_fig

FOREIGN_EQ = {"EWA": "AUDUSD", "EWU": "GBPUSD", "IEV": "EURUSD", "EWJ": "JPYUSD"}
BASKET = ["SPY", "EWA", "EWU", "IEV", "EWJ"]  # equal-weight global equity


def fx_monthly_returns(client: FMPClient | None = None) -> pd.DataFrame:
    client = client or FMPClient()
    out = {}
    for pair in set(FOREIGN_EQ.values()):
        lvl = _paginate_light(client, pair, "1990-01-01", "2026-12-31")
        if not lvl.empty:
            out[pair] = lvl.resample("ME").last().pct_change()
    return pd.DataFrame(out)


def hedged_basket(returns: pd.DataFrame, fx: pd.DataFrame, hedge_ratio: float) -> pd.Series:
    """Equal-weight global equity basket at a given hedge ratio (SPY is USD)."""
    cols = [a for a in BASKET if a in returns]
    adj = returns[cols].copy()
    for asset, pair in FOREIGN_EQ.items():
        if asset in adj and pair in fx:
            adj[asset] = adj[asset] - hedge_ratio * fx[pair].reindex(adj.index)
    return adj.mean(axis=1)


def compare_hedging(returns: pd.DataFrame, fx: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rows = {}
    for label, hr in [("Unhedged", 0.0), ("50% Hedged", 0.5), ("Fully Hedged", 1.0)]:
        b = hedged_basket(returns, fx, hr).dropna()
        rows[label] = {
            "ann_return": metrics.ann_return(b),
            "ann_vol": metrics.ann_vol(b),
            "sharpe": metrics.sharpe(b, rf),
            "max_drawdown": metrics.max_drawdown(b),
        }
    return pd.DataFrame(rows).T


def fx_vol_contribution(returns: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """For each foreign sleeve: local vol, FX vol, unhedged vol and the
    correlation of local return with the currency."""
    rows = {}
    for asset, pair in FOREIGN_EQ.items():
        if asset not in returns or pair not in fx:
            continue
        unh = returns[asset]
        fxr = fx[pair].reindex(unh.index)
        loc = (unh - fxr).dropna()
        df = pd.concat([loc.rename("loc"), fxr.rename("fx")], axis=1).dropna()
        rows[asset] = {
            "local_vol": metrics.ann_vol(df["loc"]),
            "fx_vol": metrics.ann_vol(df["fx"]),
            "unhedged_vol": metrics.ann_vol(unh),
            "corr_local_fx": float(df["loc"].corr(df["fx"])),
        }
    return pd.DataFrame(rows).T


def hedging_by_regime(returns: pd.DataFrame, fx: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """Unhedged vs hedged basket annualised vol within each regime."""
    unh = hedged_basket(returns, fx, 0.0)
    hed = hedged_basket(returns, fx, 1.0)
    reg = regime.dropna().astype(str)
    idx = unh.index.intersection(reg.index)
    rows = {}
    for label in REGIME_ORDER:
        mask = reg.loc[idx] == label
        rows[label] = {
            "unhedged_vol": metrics.ann_vol(unh.loc[idx][mask]),
            "hedged_vol": metrics.ann_vol(hed.loc[idx][mask]),
        }
    return pd.DataFrame(rows).T


def analyze() -> dict:
    out = settings.reports_dir / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    returns = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly.parquet")["rf"]
    macro = read_parquet(settings.processed_dir / "macro_monthly.parquet")
    regime = classify_regimes(macro)["regime"]

    fx = fx_monthly_returns()
    comp = compare_hedging(returns, fx, rf)
    contrib = fx_vol_contribution(returns, fx)
    byreg = hedging_by_regime(returns, fx, regime)
    write_csv(comp.round(4), out / "currency_hedging_comparison.csv")
    write_csv(contrib.round(4), out / "currency_fx_vol_contribution.csv")
    write_csv(byreg.round(4), out / "currency_hedging_by_regime.csv")

    _fig_hedging(comp, byreg)
    logger.info("Currency hedging comparison:\n{}", comp.round(3).to_string())
    return {"comparison": comp, "contribution": contrib, "by_regime": byreg}


def _fig_hedging(comp: pd.DataFrame, byreg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(comp.index))
    ax.bar(x - 0.2, comp["ann_vol"] * 100, 0.4, label="Ann vol %", color="#0072B2")
    ax.bar(x + 0.2, comp["sharpe"], 0.4, label="Sharpe", color="#D55E00")
    ax.set_xticks(x, comp.index)
    ax.set_title("Global equity basket: hedged vs unhedged (USD investor)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    xr = np.arange(len(byreg.index))
    ax2.bar(xr - 0.2, byreg["unhedged_vol"] * 100, 0.4, label="Unhedged vol %", color="#999")
    ax2.bar(xr + 0.2, byreg["hedged_vol"] * 100, 0.4, label="Hedged vol %",
            color=[REGIME_COLORS[r] for r in byreg.index])
    ax2.set_xticks(xr, byreg.index, rotation=20, ha="right")
    ax2.set_title("FX impact on equity volatility by regime")
    ax2.legend(fontsize=9)
    save_fig(fig, "08_currency_hedging", subdir="phase3")
