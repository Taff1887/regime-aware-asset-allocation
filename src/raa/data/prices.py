"""Price and return panels for the asset universe.

Prices are dividend-adjusted (total-return) daily closes from FMP. We build:

- ``prices_wide``    : daily total-return price levels (index=date, cols=tickers)
- ``returns_daily``  : daily simple returns
- ``returns_monthly``: month-end simple returns (primary analysis frequency)
- ``rf_monthly``     : risk-free monthly return from the 3m Treasury yield
"""

from __future__ import annotations

import pandas as pd

from raa.data.fmp_client import FMPClient
from raa.data.universe import tickers
from raa.utils.config import settings
from raa.utils.io import write_parquet
from raa.utils.logging import logger

START = "1990-01-01"
END = "2026-12-31"


_MAX_BARS = 5000  # FMP EOD page size; older history requires walking the cursor back


def _fetch_symbol_full(
    client: FMPClient, sym: str, start: str, end: str, max_pages: int = 12
) -> list[dict]:
    """Fetch a symbol's full history, paginating the ``to`` cursor backwards.

    The FMP EOD endpoint returns at most ~5000 bars ending at ``to`` and ignores
    ``from`` beyond that window. We therefore page backwards: each request's
    earliest date becomes the next request's ``to`` until we reach ``start`` (or
    a short page signals inception).
    """
    collected: dict[str, dict] = {}
    cursor_to = end
    for _ in range(max_pages):
        rows = client.historical_prices(sym, start, cursor_to, series="dividend-adjusted")
        if not rows:
            break
        for r in rows:
            collected[r["date"]] = r
        earliest = min(r["date"] for r in rows)
        if len(rows) < _MAX_BARS or earliest <= start:
            break
        # Step the cursor one day before the earliest bar seen.
        cursor_to = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return list(collected.values())


def fetch_prices_long(
    client: FMPClient, syms: list[str], start: str = START, end: str = END
) -> pd.DataFrame:
    """Long DataFrame ``[ticker, date, adj_close]`` for the given symbols (full history)."""
    frames = []
    for sym in syms:
        rows = _fetch_symbol_full(client, sym, start, end)
        if not rows:
            logger.warning("No price data for {}", sym)
            continue
        df = pd.DataFrame(rows)[["date", "adjClose"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = sym
        df = df.rename(columns={"adjClose": "adj_close"})
        frames.append(df)
        logger.info("  {:5s}: {} bars ({} .. {})", sym, len(df), df["date"].min().date(),
                    df["date"].max().date())
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "adj_close"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])


def build_price_panels(client: FMPClient | None = None) -> dict[str, pd.DataFrame]:
    """Fetch the universe and return daily/monthly price + return panels."""
    client = client or FMPClient()
    syms = tickers(core_only=False)
    logger.info("Fetching {} asset price series from FMP", len(syms))

    long = fetch_prices_long(client, syms)
    wide = (
        long.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .reindex(columns=syms)
    )

    returns_daily = wide.pct_change()
    monthly_px = wide.resample("ME").last()
    returns_monthly = monthly_px.pct_change()

    rf_monthly = build_rf_monthly(client)

    return {
        "prices_wide": wide,
        "returns_daily": returns_daily,
        "returns_monthly": returns_monthly,
        "rf_monthly": rf_monthly.to_frame("rf"),
    }


def build_rf_monthly(client: FMPClient | None = None, start: str = START, end: str = END) -> pd.Series:
    """Monthly risk-free return from the 3-month constant-maturity Treasury yield."""
    client = client or FMPClient()
    rows = client.treasury_rates(start, end)
    if not rows:
        return pd.Series(dtype=float, name="rf")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    y3m = (
        pd.Series(pd.to_numeric(df["month3"], errors="coerce").to_numpy(), index=df["date"])
        .sort_index()
        .resample("ME")
        .last()
    )
    # Annual yield (%) -> monthly compounded return.
    rf = (1.0 + y3m / 100.0) ** (1.0 / 12.0) - 1.0
    rf.name = "rf"
    return rf


def collect_prices(client: FMPClient | None = None) -> dict[str, pd.DataFrame]:
    """Build and persist all price/return panels to ``data/processed``."""
    settings.ensure_dirs()
    panels = build_price_panels(client)
    for name, df in panels.items():
        out = settings.processed_dir / f"{name}.parquet"
        write_parquet(df, out)
        logger.info("Saved {}: {} rows x {} cols", name, len(df), df.shape[1])
    return panels


if __name__ == "__main__":
    collect_prices()
