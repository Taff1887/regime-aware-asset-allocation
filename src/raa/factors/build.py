"""Construct macro risk-factor return/indicator series.

We build transparent, mostly *tradable* factor proxies from the asset and market
panels. Each factor isolates one macro risk:

Tradable long/short or excess-return factors
--------------------------------------------
- ``growth``    : equity risk premium      = SPY - rf
- ``rates``     : duration / term premium  = TLT - rf  (long Treasuries)
- ``credit``    : credit / default premium = HYG - IEF (HY minus duration)
- ``inflation`` : inflation beta           = 0.5*(DBC + GLD) - IEF
- ``commodity`` : commodity premium        = DBC - rf
- ``currency``  : USD strength             = - mean(FX-vs-USD returns)

Observable risk indicators (non-tradable; used as risk drivers, not return premia)
----------------------------------------------------------------------------------
- ``volatility``: change in VIX            (risk-off when positive)
- ``liquidity`` : change in TED spread     (funding stress when positive)

Note the common sample is gated by the youngest ingredient (HYG/DBC ~2007-08).
"""

from __future__ import annotations

import pandas as pd

from raa.utils.config import settings
from raa.utils.io import read_parquet, write_parquet
from raa.utils.logging import logger

TRADABLE_FACTORS = ["growth", "rates", "credit", "inflation", "commodity", "currency"]
INDICATOR_FACTORS = ["volatility", "liquidity"]

FACTOR_LABELS = {
    "growth": "Growth / Equity (SPY-rf)",
    "rates": "Rates / Duration (TLT-rf)",
    "credit": "Credit (HYG-IEF)",
    "inflation": "Inflation (real assets - bonds)",
    "commodity": "Commodity (DBC-rf)",
    "currency": "Currency / USD strength",
    "volatility": "Volatility (ΔVIX)",
    "liquidity": "Liquidity (ΔTED)",
}


def build_factors(
    returns: pd.DataFrame | None = None,
    rf: pd.Series | None = None,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the monthly factor panel."""
    if returns is None:
        returns = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    if rf is None:
        rf = read_parquet(settings.processed_dir / "rf_monthly.parquet")["rf"]
    if market is None:
        market = read_parquet(settings.processed_dir / "market_monthly.parquet")

    rf = rf.reindex(returns.index).fillna(0.0)

    def ex(col: str) -> pd.Series:
        return returns[col] - rf if col in returns else pd.Series(index=returns.index, dtype=float)

    f = pd.DataFrame(index=returns.index)
    f["growth"] = ex("SPY")
    f["rates"] = ex("TLT")
    f["credit"] = (returns["HYG"] - returns["IEF"]) if {"HYG", "IEF"} <= set(returns) else pd.NA
    f["inflation"] = (
        0.5 * (returns["DBC"] + returns["GLD"]) - returns["IEF"]
        if {"DBC", "GLD", "IEF"} <= set(returns)
        else pd.NA
    )
    f["commodity"] = ex("DBC")
    f["currency"] = market["usd_ret"].reindex(returns.index) if "usd_ret" in market else pd.NA
    f["volatility"] = market["vix_chg"].reindex(returns.index) if "vix_chg" in market else pd.NA
    f["liquidity"] = market["ted_spread"].reindex(returns.index).diff() if "ted_spread" in market else pd.NA

    f.index.name = "date"
    return f


def collect_factors() -> pd.DataFrame:
    settings.ensure_dirs()
    f = build_factors()
    out = settings.processed_dir / "factors_monthly.parquet"
    write_parquet(f, out)
    logger.info("Saved factor panel: {} rows x {} cols -> {}", len(f), f.shape[1], out)
    return f


if __name__ == "__main__":
    collect_factors()
