"""Risk-factor decomposition: asset betas, factor behaviour by regime, and
portfolio variance attribution to macro factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raa.analysis import metrics
from raa.regimes.rule_based import REGIME_ORDER


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS with intercept. Returns (coefs[intercept, *betas], r_squared)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, r2


def factor_betas(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: list[str] | None = None,
    assets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Multivariate regression of each asset on the factor set.

    Returns ``(betas[asset x factor], r2[asset], alpha_ann[asset])``.
    """
    factor_cols = factor_cols or [c for c in factors.columns if factors[c].notna().any()]
    assets = assets or list(returns.columns)
    F = factors[factor_cols].dropna()
    betas = pd.DataFrame(index=assets, columns=factor_cols, dtype=float)
    r2s, alphas = {}, {}
    for a in assets:
        df = pd.concat([returns[a], F], axis=1, join="inner").dropna()
        if len(df) < 24:
            continue
        y = df[a].to_numpy()
        X = df[factor_cols].to_numpy()
        coef, r2 = _ols(y, X)
        alphas[a] = coef[0] * 12.0  # annualised intercept
        betas.loc[a, factor_cols] = coef[1:]
        r2s[a] = r2
    return betas, pd.Series(r2s), pd.Series(alphas)


def factor_perf_by_regime(
    factors: pd.DataFrame, regime: pd.Series, factor_cols: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Annualised mean, vol and info-ratio of each factor within each regime."""
    factor_cols = factor_cols or list(factors.columns)
    reg = regime.dropna().astype(str)
    idx = factors.index.intersection(reg.index)
    F, reg = factors.loc[idx, factor_cols], reg.loc[idx]

    mean = pd.DataFrame(index=factor_cols, columns=[*REGIME_ORDER, "All"], dtype=float)
    vol = mean.copy()
    ir = mean.copy()
    for label in [*REGIME_ORDER, "All"]:
        sub = F if label == "All" else F.loc[reg == label]
        for fc in factor_cols:
            s = sub[fc].dropna()
            if len(s) < 6:
                continue
            mean.loc[fc, label] = metrics.ann_return(s)
            vol.loc[fc, label] = metrics.ann_vol(s)
            ir.loc[fc, label] = metrics.sharpe(s)
    return {"mean": mean, "vol": vol, "ir": ir}


def portfolio_factor_risk(
    weights: dict[str, float],
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: list[str] | None = None,
) -> pd.Series:
    """Attribute a portfolio's variance to macro factors.

    Builds the portfolio return, regresses on factors, then splits variance into
    each factor's contribution (b_i * (Sigma_F b)_i) plus idiosyncratic residual.
    Returns a Series of variance shares summing to ~1.
    """
    factor_cols = factor_cols or [c for c in factors.columns if factors[c].notna().any()]
    w = pd.Series(weights)
    cols = [c for c in w.index if c in returns.columns]
    w = w[cols] / w[cols].sum()
    port = (returns[cols] * w).sum(axis=1)

    df = pd.concat([port.rename("port"), factors[factor_cols]], axis=1, join="inner").dropna()
    if len(df) < 24:
        return pd.Series(dtype=float)
    y = df["port"].to_numpy()
    X = df[factor_cols].to_numpy()
    coef, _ = _ols(y, X)
    b = coef[1:]
    sigma_f = np.cov(X, rowvar=False)
    systematic = b @ sigma_f @ b
    total = float(np.var(y, ddof=1))
    contrib = b * (sigma_f @ b)  # per-factor contribution to systematic variance
    out = {fc: contrib[i] / total for i, fc in enumerate(factor_cols)}
    out["idiosyncratic"] = max(0.0, (total - systematic) / total)
    return pd.Series(out)
