"""Portfolio construction methods.

All optimisers are long-only, fully invested (weights sum to 1) and accept an
optional per-asset weight cap. They operate on a covariance matrix (and, for
mean-variance, an expected-return vector) estimated from *real historical*
returns. Covariance can be Ledoit-Wolf shrunk for stability.

Methods:
- ``equal_weight``          : 1/N
- ``fixed_weight``          : a supplied static mix (e.g. 60/40)
- ``inverse_vol``           : risk weighting by 1/sigma (naive risk parity)
- ``equal_risk_contribution`` : true ERC / risk parity
- ``min_variance``          : global minimum-variance
- ``max_diversification``   : maximise the diversification ratio
- ``max_sharpe``            : mean-variance tangency (needs expected returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def sample_cov(returns: pd.DataFrame, shrinkage: bool = True) -> pd.DataFrame:
    """Covariance of complete-case monthly returns (optionally LW-shrunk)."""
    X = returns.dropna()
    if X.shape[0] < 6 or X.shape[1] < 2:
        return X.cov()
    if shrinkage and X.shape[0] > X.shape[1] + 2:
        cov = LedoitWolf().fit(X.to_numpy()).covariance_
        return pd.DataFrame(cov, index=X.columns, columns=X.columns)
    return X.cov()


def _solve(obj, n: int, w_max: float = 1.0, w0: np.ndarray | None = None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, w_max)] * n
    if w0 is None:
        w0 = np.full(n, 1.0 / n)
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def equal_weight(assets: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(assets), index=assets)


def fixed_weight(weights: dict[str, float]) -> pd.Series:
    s = pd.Series(weights, dtype=float)
    return s / s.sum()


def inverse_vol(cov: pd.DataFrame) -> pd.Series:
    iv = 1.0 / np.sqrt(np.diag(cov))
    iv = iv / iv.sum()
    return pd.Series(iv, index=cov.index)


def min_variance(cov: pd.DataFrame, w_max: float = 1.0) -> pd.Series:
    S = cov.to_numpy()
    w = _solve(lambda w: w @ S @ w, len(cov), w_max)
    return pd.Series(w, index=cov.index)


def max_diversification(cov: pd.DataFrame, w_max: float = 1.0) -> pd.Series:
    S = cov.to_numpy()
    sig = np.sqrt(np.diag(S))

    def neg_dr(w):
        v = np.sqrt(w @ S @ w)
        return -(w @ sig) / v if v > 0 else 0.0

    w = _solve(neg_dr, len(cov), w_max)
    return pd.Series(w, index=cov.index)


def equal_risk_contribution(cov: pd.DataFrame, w_max: float = 1.0) -> pd.Series:
    """True ERC: each asset contributes equally to portfolio variance."""
    S = cov.to_numpy()

    def obj(w):
        rc = w * (S @ w)
        return float(np.sum((rc[:, None] - rc[None, :]) ** 2))

    w0 = inverse_vol(cov).to_numpy()
    w = _solve(obj, len(cov), w_max, w0=w0)
    return pd.Series(w, index=cov.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0, w_max: float = 1.0) -> pd.Series:
    """Mean-variance tangency portfolio (long-only)."""
    S = cov.to_numpy()
    m = mu.reindex(cov.index).to_numpy()

    def neg_sharpe(w):
        v = np.sqrt(w @ S @ w)
        return -((w @ m) - rf) / v if v > 0 else 0.0

    w = _solve(neg_sharpe, len(cov), w_max)
    return pd.Series(w, index=cov.index)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Percentage contribution of each asset to total portfolio variance."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    S = cov.to_numpy()
    pv = w @ S @ w
    if pv <= 0:
        return pd.Series(0.0, index=cov.index)
    rc = w * (S @ w) / pv
    return pd.Series(rc, index=cov.index)
