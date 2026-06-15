"""Long-history fixed-income data: bond mutual funds + extended macro.

Builds, from real FMP data:
- ``returns_monthly_fi``  : month-end total returns of the bond-fund universe
- ``macro_monthly_long``  : macro panel back to 1950 (for long-sample regimes)
- ``rf_monthly_long``     : risk-free proxy from the effective fed funds rate
"""

from __future__ import annotations

import pandas as pd

from raa.data.fmp_client import FMPClient
from raa.data.macro import build_macro_monthly
from raa.data.prices import _fetch_symbol_full
from raa.utils.config import settings
from raa.utils.io import read_yaml, write_parquet
from raa.utils.logging import logger

START = "1970-01-01"
END = "2026-12-31"


def load_fi_universe() -> dict:
    return read_yaml(settings.config_dir / "fixed_income.yaml")


def fund_tickers() -> list[str]:
    return [f["ticker"] for f in load_fi_universe()["funds"]]


def fund_meta() -> pd.DataFrame:
    return pd.DataFrame(load_fi_universe()["funds"]).set_index("ticker")


def fetch_fi_returns(client: FMPClient, syms: list[str]) -> pd.DataFrame:
    frames = []
    for sym in syms:
        rows = _fetch_symbol_full(client, sym, START, END)
        if not rows:
            logger.warning("No data for fund {}", sym)
            continue
        df = pd.DataFrame(rows)[["date", "adjClose"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = sym
        frames.append(df.rename(columns={"adjClose": "adj_close"}))
        logger.info("  {:6s}: {} bars ({} .. {})", sym, len(df), df["date"].min().date(), df["date"].max().date())
    long = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    wide = long.pivot(index="date", columns="ticker", values="adj_close").sort_index().reindex(columns=syms)
    monthly_px = wide.resample("ME").last()
    return monthly_px.pct_change()


def _fetch_returns_wide(syms: list[str]) -> pd.DataFrame:
    """Convenience: fetch a list of symbols and return wide monthly returns."""
    return fetch_fi_returns(FMPClient(), syms)


def build_long_rf(client: FMPClient) -> pd.Series:
    ff = client.economic_indicator("federalFunds", "1950-01-01", END)
    d = pd.DataFrame(ff)
    d["date"] = pd.to_datetime(d["date"])
    s = pd.Series(pd.to_numeric(d["value"], errors="coerce").to_numpy(), index=d["date"]).sort_index().resample("ME").last()
    rf = (1.0 + s / 100.0) ** (1.0 / 12.0) - 1.0
    rf.name = "rf"
    return rf


def collect_fixed_income(client: FMPClient | None = None) -> None:
    settings.ensure_dirs()
    client = client or FMPClient()
    logger.info("Fetching {} long-history bond funds", len(fund_tickers()))
    rets = fetch_fi_returns(client, fund_tickers())
    write_parquet(rets, settings.processed_dir / "returns_monthly_fi.parquet")
    logger.info("Saved returns_monthly_fi: {} rows ({}..{})", len(rets), rets.index.min().date(), rets.index.max().date())

    macro = build_macro_monthly(client, start="1950-01-01")
    write_parquet(macro, settings.processed_dir / "macro_monthly_long.parquet")
    logger.info("Saved macro_monthly_long: {} rows from {}", len(macro), macro.index.min().date())

    rf = build_long_rf(client)
    write_parquet(rf.to_frame("rf"), settings.processed_dir / "rf_monthly_long.parquet")
    logger.info("Saved rf_monthly_long: {} rows from {}", len(rf), rf.index.min().date())


if __name__ == "__main__":
    collect_fixed_income()
