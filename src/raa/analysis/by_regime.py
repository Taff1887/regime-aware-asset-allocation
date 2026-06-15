"""Per-regime asset-class statistics and correlation analysis.

Given a monthly return panel and a monthly regime label, compute:

- per-(asset, regime) annualised return / vol / Sharpe / drawdown / VaR / ES;
- full correlation matrices within each regime;
- key diversifier pairs (equity-bond, credit-equity, commodity-equity, gold-equity)
  across regimes;
- an average-pairwise-correlation "diversification index" by regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raa.analysis import metrics
from raa.regimes.rule_based import REGIME_ORDER

# Canonical diversifier pairs examined across regimes.
KEY_PAIRS: list[tuple[str, str, str]] = [
    ("SPY", "IEF", "Equity / Intermediate Treasury"),
    ("SPY", "TLT", "Equity / Long Treasury"),
    ("SPY", "LQD", "Equity / IG Credit"),
    ("SPY", "HYG", "Equity / High Yield"),
    ("SPY", "DBC", "Equity / Commodities"),
    ("SPY", "GLD", "Equity / Gold"),
    ("SPY", "VNQ", "Equity / REITs"),
    ("HYG", "IEF", "High Yield / Treasury"),
]


def align(returns: pd.DataFrame, regime: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align a return panel to a regime label series on the month-end index."""
    reg = regime.dropna()
    idx = returns.index.intersection(reg.index)
    return returns.loc[idx], reg.loc[idx].astype(str)


def regime_counts(regime: pd.Series) -> pd.DataFrame:
    """Months and time-share per regime."""
    r = regime.dropna().astype(str)
    counts = r.value_counts().reindex(REGIME_ORDER, fill_value=0)
    out = pd.DataFrame({"months": counts})
    out["share"] = out["months"] / out["months"].sum()
    return out


def build_metric_tables(
    returns: pd.DataFrame,
    regime: pd.Series,
    rf: pd.Series | None = None,
    metric_fns: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a dict ``metric -> DataFrame(index=asset, cols=regimes + 'All')``."""
    rets, reg = align(returns, regime)
    if metric_fns is None:
        metric_fns = {
            "ann_return": metrics.ann_return,
            "ann_vol": metrics.ann_vol,
            "sharpe": lambda s: metrics.sharpe(s, rf),
            "max_drawdown": metrics.max_drawdown,
            "var_5": metrics.hist_var,
            "es_5": metrics.expected_shortfall,
            "hit_rate": metrics.hit_rate,
        }

    cols = [*REGIME_ORDER, "All"]
    tables = {m: pd.DataFrame(index=rets.columns, columns=cols, dtype=float) for m in metric_fns}

    for label in cols:
        mask = slice(None) if label == "All" else (reg == label)
        sub = rets.loc[mask] if label != "All" else rets
        for asset in rets.columns:
            series = sub[asset].dropna()
            for m, fn in metric_fns.items():
                tables[m].loc[asset, label] = fn(series) if len(series) >= 3 else np.nan
    return tables


def regime_corr_matrices(
    returns: pd.DataFrame, regime: pd.Series, assets: list[str] | None = None, min_obs: int = 12
) -> dict[str, pd.DataFrame]:
    """Full correlation matrix within each regime (and overall)."""
    rets, reg = align(returns, regime)
    if assets:
        rets = rets[[a for a in assets if a in rets.columns]]
    out: dict[str, pd.DataFrame] = {}
    for label in [*REGIME_ORDER, "All"]:
        sub = rets if label == "All" else rets.loc[reg == label]
        sub = sub.dropna(axis=1, how="all")
        if len(sub) >= min_obs:
            out[label] = sub.corr()
    return out


def key_pair_correlations(
    returns: pd.DataFrame, regime: pd.Series, pairs: list[tuple[str, str, str]] | None = None
) -> pd.DataFrame:
    """Correlation of canonical diversifier pairs across regimes."""
    pairs = pairs or KEY_PAIRS
    rets, reg = align(returns, regime)
    rows = []
    for a, b, name in pairs:
        if a not in rets or b not in rets:
            continue
        row = {"pair": name, "a": a, "b": b}
        for label in [*REGIME_ORDER, "All"]:
            sub = rets if label == "All" else rets.loc[reg == label]
            s = sub[[a, b]].dropna()
            row[label] = float(s[a].corr(s[b])) if len(s) >= 12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def diversification_index(
    returns: pd.DataFrame, regime: pd.Series, assets: list[str] | None = None, min_obs: int = 12
) -> pd.Series:
    """Average off-diagonal correlation among ``assets`` within each regime.

    Higher => assets move together => *less* diversification available.
    """
    mats = regime_corr_matrices(returns, regime, assets=assets, min_obs=min_obs)
    out = {}
    for label, mat in mats.items():
        n = mat.shape[0]
        if n < 2:
            continue
        off = mat.where(~np.eye(n, dtype=bool))
        out[label] = float(np.nanmean(off.to_numpy()))
    return pd.Series(out).reindex([*REGIME_ORDER, "All"]).dropna()
