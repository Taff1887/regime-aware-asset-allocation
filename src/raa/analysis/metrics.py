"""Performance and risk metrics computed from periodic (monthly) returns.

All functions operate on simple-return series. ``periods_per_year`` defaults to
12 (monthly). Risk-adjusted ratios accept a risk-free series or scalar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PPY = 12  # monthly


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).dropna().astype(float)


def ann_return(returns: pd.Series, ppy: int = PPY) -> float:
    """Geometric annualised return."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return np.nan
    return growth ** (ppy / len(r)) - 1.0


def ann_vol(returns: pd.Series, ppy: int = PPY) -> float:
    """Annualised volatility (sample std, ddof=1)."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(ppy))


def _excess(returns: pd.Series, rf: pd.Series | float | None) -> pd.Series:
    r = _clean(returns)
    if rf is None:
        return r
    if np.isscalar(rf):
        return r - float(rf)
    rf_aligned = pd.Series(rf).reindex(r.index).fillna(0.0)
    return r - rf_aligned


def sharpe(returns: pd.Series, rf: pd.Series | float | None = None, ppy: int = PPY) -> float:
    """Annualised Sharpe ratio on excess returns."""
    ex = _excess(returns, rf).dropna()
    if len(ex) < 2 or ex.std(ddof=1) == 0:
        return np.nan
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(ppy))


def sortino(returns: pd.Series, rf: pd.Series | float | None = None, ppy: int = PPY) -> float:
    """Annualised Sortino ratio (downside deviation vs 0 excess return)."""
    ex = _excess(returns, rf).dropna()
    downside = ex[ex < 0]
    if downside.empty:
        return np.nan
    dd = np.sqrt((downside**2).mean())
    if dd == 0:
        return np.nan
    return float(ex.mean() / dd * np.sqrt(ppy))


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Drawdown path from a return series (0 = at high-water mark)."""
    r = _clean(returns)
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (most negative point of the drawdown curve)."""
    dd = drawdown_curve(returns)
    return float(dd.min()) if not dd.empty else np.nan


def calmar(returns: pd.Series, ppy: int = PPY) -> float:
    """Annualised return divided by the absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return ann_return(returns, ppy) / abs(mdd)


def hist_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Value-at-Risk at confidence ``1-alpha`` (reported as a loss, negative)."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    return float(np.quantile(r, alpha))


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall / CVaR: mean of returns at or below the VaR threshold."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    return float(tail.mean()) if not tail.empty else float(var)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return."""
    r = _clean(returns)
    return float((r > 0).mean()) if not r.empty else np.nan


def summary_stats(
    returns: pd.Series, rf: pd.Series | float | None = None, ppy: int = PPY
) -> dict[str, float]:
    """Full metric bundle for one return series."""
    r = _clean(returns)
    return {
        "n_obs": int(len(r)),
        "ann_return": ann_return(r, ppy),
        "ann_vol": ann_vol(r, ppy),
        "sharpe": sharpe(r, rf, ppy),
        "sortino": sortino(r, rf, ppy),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, ppy),
        "var_5": hist_var(r, 0.05),
        "es_5": expected_shortfall(r, 0.05),
        "hit_rate": hit_rate(r),
        "skew": float(r.skew()) if len(r) > 2 else np.nan,
        "worst_month": float(r.min()) if not r.empty else np.nan,
        "best_month": float(r.max()) if not r.empty else np.nan,
    }
