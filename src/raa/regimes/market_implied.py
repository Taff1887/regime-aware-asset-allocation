"""Market-implied regime detection: risk-on / neutral / risk-off.

Rather than macro fundamentals, this infers the prevailing environment from what
markets are pricing: equity implied vol (VIX), FX vol, and the funding spread
(TED). A standardised composite "stress index" is split into three states.

The descriptive classification below standardises over the full sample; the
backtest uses an expanding-window version to remain point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRESS_COLS = ["vix", "fx_vol", "ted_spread"]
MARKET_STATES = ["Risk-On", "Neutral", "Risk-Off"]


def stress_index(market: pd.DataFrame, cols: list[str] | None = None, expanding: bool = False) -> pd.Series:
    """Composite market-stress z-score (mean of standardised stress columns)."""
    cols = [c for c in (cols or STRESS_COLS) if c in market]
    X = market[cols]
    if expanding:
        mu = X.expanding(min_periods=24).mean()
        sd = X.expanding(min_periods=24).std(ddof=0)
        z = (X - mu) / sd
    else:
        z = (X - X.mean()) / X.std(ddof=0)
    return z.mean(axis=1, skipna=True).rename("stress")


def market_regime(
    market: pd.DataFrame, lower: float = -0.5, upper: float = 0.5, expanding: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Classify each month into Risk-On / Neutral / Risk-Off from the stress index."""
    stress = stress_index(market, expanding=expanding)
    reg = pd.Series(
        np.where(stress > upper, "Risk-Off", np.where(stress < lower, "Risk-On", "Neutral")),
        index=stress.index,
        name="market_regime",
    )
    reg[stress.isna()] = np.nan
    reg = pd.Categorical(reg, categories=MARKET_STATES)
    return pd.Series(reg, index=stress.index, name="market_regime"), stress
