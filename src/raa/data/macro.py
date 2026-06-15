"""Macro-economic series for regime detection and factor analysis.

All series are sourced from FMP and resampled to a common **month-end** index.
We deliberately keep the macro inputs simple and transparent:

- Inflation : CPI year-on-year (%)
- Growth    : Industrial-production year-on-year (%), plus real-GDP YoY and the
              12-month change in the unemployment rate as corroborating signals
- Policy    : Federal-funds rate
- Curve     : 3m / 2y / 10y / 30y Treasury yields and curve slopes

Publication lag is handled downstream in regime detection (signals are lagged so
no point-in-time look-ahead occurs).
"""

from __future__ import annotations

import pandas as pd

from raa.data.fmp_client import FMPClient
from raa.utils.config import settings
from raa.utils.io import write_parquet
from raa.utils.logging import logger

START = "1990-01-01"
END = "2026-12-31"


def _indicator_series(records: list[dict]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").to_numpy(),
        index=df["date"],
    ).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _to_month_end(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    return s.resample("ME").last()


def fetch_treasury_monthly(client: FMPClient, start: str = START, end: str = END) -> pd.DataFrame:
    """Month-end Treasury curve and slopes."""
    rows = client.treasury_rates(start, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = {"month3": "y3m", "year2": "y2", "year5": "y5", "year10": "y10", "year30": "y30"}
    df = df[list(keep)].rename(columns=keep).apply(pd.to_numeric, errors="coerce")
    m = df.resample("ME").last()
    m["slope_10y_3m"] = m["y10"] - m["y3m"]
    m["slope_10y_2y"] = m["y10"] - m["y2"]
    return m


def build_macro_monthly(
    client: FMPClient | None = None, start: str = START, end: str = END
) -> pd.DataFrame:
    """Assemble the full month-end macro panel."""
    client = client or FMPClient()
    logger.info("Fetching macro indicators from FMP ({}..{})", start, end)

    cpi = _to_month_end(_indicator_series(client.economic_indicator("CPI", start, end)))
    indpro = _to_month_end(
        _indicator_series(
            client.economic_indicator("industrialProductionTotalIndex", start, end)
        )
    )
    unemp = _to_month_end(
        _indicator_series(client.economic_indicator("unemploymentRate", start, end))
    )
    fedfunds = _to_month_end(
        _indicator_series(client.economic_indicator("federalFunds", start, end))
    )
    gdp = _indicator_series(client.economic_indicator("realGDP", start, end))

    macro = pd.DataFrame(
        {
            "cpi": cpi,
            "indpro": indpro,
            "unemployment": unemp,
            "fedfunds": fedfunds,
        }
    )

    # Year-on-year transforms (12 monthly periods).
    macro["cpi_yoy"] = macro["cpi"].pct_change(12) * 100.0
    macro["indpro_yoy"] = macro["indpro"].pct_change(12) * 100.0
    macro["unemployment_chg_12m"] = macro["unemployment"].diff(12)

    # Real-GDP YoY is quarterly; forward-fill to month-end (a step series).
    if not gdp.empty:
        gdp_yoy = (gdp.pct_change(4) * 100.0).rename("gdp_yoy")
        macro = macro.join(
            gdp_yoy.resample("ME").last().ffill(), how="left"
        )

    # Treasury curve.
    tsy = fetch_treasury_monthly(client, start, end)
    if not tsy.empty:
        macro = macro.join(tsy, how="left")

    macro = macro.sort_index()
    macro.index.name = "date"
    return macro


def collect_macro(client: FMPClient | None = None) -> pd.DataFrame:
    """Build and persist the macro panel to ``data/processed/macro_monthly.parquet``."""
    settings.ensure_dirs()
    macro = build_macro_monthly(client)
    out = settings.processed_dir / "macro_monthly.parquet"
    write_parquet(macro, out)
    logger.info("Saved macro panel: {} rows x {} cols -> {}", len(macro), macro.shape[1], out)
    return macro


if __name__ == "__main__":
    collect_macro()
