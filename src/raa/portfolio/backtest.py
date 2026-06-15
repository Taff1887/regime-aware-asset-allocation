"""Out-of-sample backtesting engine and strategy factories.

The engine is strictly point-in-time: at each rebalance date ``t`` weights are a
function of data available **up to and including ``t``** only, and are applied to
the *next* period's realised returns. Assets enter the investable set when they
have enough history (no look-ahead, no pre-inception data). Transaction costs are
charged on turnover; weights are held between rebalances.

Static strategies use unconditional (expanding/rolling) moments. Regime-aware
strategies use moments estimated from prior months in the *current* regime,
falling back to the full history when a regime has too few observations.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from raa.portfolio import construct
from raa.utils.logging import logger

WeightFn = Callable[[pd.DataFrame, pd.Timestamp], "pd.Series | None"]


def available_assets(hist: pd.DataFrame, assets: list[str], min_obs: int = 24) -> list[str]:
    """Assets with at least ``min_obs`` non-NaN observations so far."""
    return [a for a in assets if a in hist.columns and hist[a].notna().sum() >= min_obs]


def run_backtest(
    returns: pd.DataFrame,
    weight_fn: WeightFn,
    cost_bps: float = 10.0,
    min_obs: int = 36,
    rebalance_every: int = 1,
    start: str | None = None,
    name: str = "strategy",
) -> dict:
    """Run a monthly backtest. Returns net/gross return Series, weights, turnover."""
    R = returns.sort_index()
    dates = R.index
    i0 = max(min_obs, R.index.get_indexer([pd.Timestamp(start)])[0] if start else min_obs)
    w_prev: pd.Series | None = None
    w_cur: pd.Series | None = None
    rows, weights_hist, turn_hist = [], {}, {}

    for i in range(i0, len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        if (i - i0) % rebalance_every == 0 or w_cur is None:
            hist = R.loc[:t]
            w_new = weight_fn(hist, t)
            if w_new is not None and w_new.sum() > 0:
                w_cur = w_new.reindex(R.columns).fillna(0.0)
        if w_cur is None:
            continue
        r_next = R.loc[t1, w_cur.index].fillna(0.0)
        gross = float((w_cur * r_next).sum())
        turn = 0.0 if w_prev is None else float((w_cur - w_prev).abs().sum())
        cost = turn * cost_bps / 1e4
        rows.append((t1, gross - cost, gross))
        weights_hist[t1] = w_cur
        turn_hist[t1] = turn
        w_prev = w_cur

    if not rows:
        return {"name": name, "net": pd.Series(dtype=float)}
    idx = [r[0] for r in rows]
    net = pd.Series([r[1] for r in rows], index=idx, name=name)
    gross = pd.Series([r[2] for r in rows], index=idx, name=name)
    weights = pd.DataFrame(weights_hist).T
    turnover = pd.Series(turn_hist, name="turnover")
    logger.info("  backtest {:22s}: {} months {}..{} | avg turnover {:.1%}",
                name, len(net), idx[0].date(), idx[-1].date(), turnover.mean())
    return {"name": name, "net": net, "gross": gross, "weights": weights, "turnover": turnover}


# --------------------------------------------------------------------- factories
def static_strategy(
    method: str,
    assets: list[str],
    shrinkage: bool = True,
    window: int | None = None,
    w_max: float = 1.0,
    fixed: dict[str, float] | None = None,
    min_asset_obs: int = 24,
) -> WeightFn:
    """Weight function for an unconditional (non-regime) strategy."""

    def fn(hist: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
        cols = available_assets(hist, assets, min_asset_obs)
        if not cols:
            return None
        if method == "fixed":
            fcols = [a for a in cols if a in fixed]
            return construct.fixed_weight({a: fixed[a] for a in fcols}) if fcols else None
        if method == "equal":
            return construct.equal_weight(cols)
        sub = hist[cols]
        if window:
            sub = sub.iloc[-window:]
        cov = construct.sample_cov(sub, shrinkage)
        cols = list(cov.index)
        if method == "inverse_vol":
            return construct.inverse_vol(cov)
        if method == "erc":
            return construct.equal_risk_contribution(cov, w_max)
        if method == "min_var":
            return construct.min_variance(cov, w_max)
        if method == "max_div":
            return construct.max_diversification(cov, w_max)
        if method == "max_sharpe":
            mu = sub[cols].mean()
            return construct.max_sharpe(mu, cov, w_max=w_max)
        raise ValueError(f"unknown method {method!r}")

    return fn


def regime_strategy(
    method: str,
    assets: list[str],
    regimes: pd.Series,
    shrinkage: bool = True,
    min_regime_obs: int = 24,
    w_max: float = 1.0,
    min_asset_obs: int = 24,
) -> WeightFn:
    """Weight function using moments conditioned on the current regime."""
    reg = regimes.dropna().astype(str)

    def fn(hist: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
        cols = available_assets(hist, assets, min_asset_obs)
        if not cols:
            return None
        cur = reg.asof(t)
        sub = hist[cols]
        if pd.notna(cur):
            mask = reg.reindex(hist.index).astype(str) == str(cur)
            cond = hist.loc[mask.fillna(False), cols]
            if cond.dropna().shape[0] >= min_regime_obs:
                sub = cond
        cov = construct.sample_cov(sub, shrinkage)
        ccols = list(cov.index)
        if method == "min_var":
            return construct.min_variance(cov, w_max)
        if method == "erc":
            return construct.equal_risk_contribution(cov, w_max)
        if method == "max_sharpe":
            mu = sub[ccols].mean()
            return construct.max_sharpe(mu, cov, w_max=w_max)
        raise ValueError(f"unknown method {method!r}")

    return fn


# Risk-asset vs defensive sleeves for the transparent de-risking overlay.
RISK_SLEEVE = {"SPY", "EWA", "EWU", "IEV", "EWJ", "EEM", "HYG", "DBC", "IGF", "RWO"}
DEFENSIVE_SLEEVE = {"SHY", "IEF", "TLT", "GLD"}
ADVERSE_REGIMES = {"Stagflation", "Recession"}


def regime_overlay_strategy(
    assets: list[str],
    regimes: pd.Series,
    derisk: float = 0.5,
    shrinkage: bool = True,
    min_asset_obs: int = 24,
    risk_sleeve: set[str] | None = None,
    defensive_sleeve: set[str] | None = None,
) -> WeightFn:
    """Transparent, implementable overlay: hold an ERC base, but in adverse macro
    regimes (Stagflation/Recession) cut the risk-asset sleeve by ``1-derisk`` and
    move the freed weight into defensives (Treasuries + gold), proportionally.

    ``derisk`` is a fixed, non-optimised parameter (sensitivity tested separately).
    ``risk_sleeve`` / ``defensive_sleeve`` let other universes (e.g. mutual-fund
    proxies) define their own sleeves; they default to the ETF sets.
    """
    reg = regimes.dropna().astype(str)
    risk_sleeve = risk_sleeve or RISK_SLEEVE
    defensive_sleeve = defensive_sleeve or DEFENSIVE_SLEEVE

    def fn(hist: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
        cols = available_assets(hist, assets, min_asset_obs)
        if not cols:
            return None
        cov = construct.sample_cov(hist[cols], shrinkage)
        w = construct.equal_risk_contribution(cov)
        cur = reg.asof(t)
        if pd.notna(cur) and str(cur) in ADVERSE_REGIMES:
            risk_cols = [a for a in w.index if a in risk_sleeve]
            def_cols = [a for a in w.index if a in defensive_sleeve]
            freed = w[risk_cols].sum() * (1.0 - derisk)
            if risk_cols and def_cols and w[def_cols].sum() > 0:
                w[risk_cols] = w[risk_cols] * derisk
                w[def_cols] = w[def_cols] + freed * (w[def_cols] / w[def_cols].sum())
        return w / w.sum()

    return fn
