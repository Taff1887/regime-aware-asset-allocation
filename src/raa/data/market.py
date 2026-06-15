"""Market-implied stress indicators: equity vol, FX vol, funding spread, USD.

These feed the market-implied regime model and several risk factors:

- ``vix``          : CBOE VIX level (equity implied volatility)
- ``vix_chg``      : month-on-month change in VIX
- ``fx_vol``       : average realised vol across major USD pairs (annualised)
- ``usd_ret``      : USD strength = minus the mean monthly return of FX-vs-USD
- ``ted_spread``   : 3m CD rate minus 3m T-bill (funding/liquidity stress proxy)
- ``recession_prob``: smoothed US recession probability (corroborating only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raa.data.fmp_client import FMPClient
from raa.data.universe import fx_pairs
from raa.utils.config import settings
from raa.utils.io import write_parquet
from raa.utils.logging import logger

START = "1990-01-01"
END = "2026-12-31"
_MAX_BARS = 5000


def _paginate_light(client: FMPClient, symbol: str, start: str, end: str, max_pages: int = 12) -> pd.Series:
    """Paginated daily 'light' (price) series -> Series indexed by date."""
    collected: dict[str, float] = {}
    cursor = end
    for _ in range(max_pages):
        rows = client.get("historical-price-eod/light", symbol=symbol, **{"from": start, "to": cursor})
        if not rows:
            break
        for r in rows:
            collected[r["date"]] = r.get("price")
        earliest = min(r["date"] for r in rows)
        if len(rows) < _MAX_BARS or earliest <= start:
            break
        cursor = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not collected:
        return pd.Series(dtype=float, name=symbol)
    s = pd.Series(collected, name=symbol)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def build_market_monthly(client: FMPClient | None = None, start: str = START, end: str = END) -> pd.DataFrame:
    client = client or FMPClient()
    logger.info("Fetching market-stress indicators (VIX, FX, CD rate)")

    # VIX (equity implied vol)
    vix_d = _paginate_light(client, "^VIX", start, end)
    vix_m = vix_d.resample("ME").last()

    # FX realised vol + USD strength
    pairs = [p["pair"] for p in fx_pairs()]
    fx_levels = {}
    for p in pairs:
        s = _paginate_light(client, p, start, end)
        if not s.empty:
            fx_levels[p] = s
    fx_df = pd.DataFrame(fx_levels)
    fx_ret_d = fx_df.pct_change()
    # annualised realised vol per month from daily returns, averaged across pairs
    fx_vol = (fx_ret_d.resample("ME").std() * np.sqrt(21 * 12)).mean(axis=1) if not fx_df.empty else pd.Series(dtype=float)
    fx_ret_m = fx_df.resample("ME").last().pct_change()
    usd_ret = -fx_ret_m.mean(axis=1) if not fx_df.empty else pd.Series(dtype=float)  # USD strengthens when pairs fall

    # 3m CD rate vs 3m T-bill -> TED-like funding spread
    cd = client.economic_indicator("3MonthOr90DayRatesAndYieldsCertificatesOfDeposit", start, end)
    cd_s = pd.Series(dtype=float)
    if cd:
        d = pd.DataFrame(cd)
        d["date"] = pd.to_datetime(d["date"])
        cd_s = pd.Series(pd.to_numeric(d["value"], errors="coerce").to_numpy(), index=d["date"]).sort_index().resample("ME").last()
    tsy = client.treasury_rates(start, end)
    tbill = pd.Series(dtype=float)
    if tsy:
        t = pd.DataFrame(tsy)
        t["date"] = pd.to_datetime(t["date"])
        tbill = pd.Series(pd.to_numeric(t["month3"], errors="coerce").to_numpy(), index=t["date"]).sort_index().resample("ME").last()
    ted = (cd_s - tbill).dropna() if not cd_s.empty and not tbill.empty else pd.Series(dtype=float)

    # Recession probability (corroborating)
    rp = client.economic_indicator("smoothedUSRecessionProbabilities", start, end)
    rp_s = pd.Series(dtype=float)
    if rp:
        d = pd.DataFrame(rp)
        d["date"] = pd.to_datetime(d["date"])
        rp_s = pd.Series(pd.to_numeric(d["value"], errors="coerce").to_numpy(), index=d["date"]).sort_index().resample("ME").last()

    market = pd.DataFrame(
        {
            "vix": vix_m,
            "vix_chg": vix_m.diff(),
            "fx_vol": fx_vol,
            "usd_ret": usd_ret,
            "ted_spread": ted,
            "recession_prob": rp_s,
        }
    ).sort_index()
    market.index.name = "date"
    return market


def collect_market(client: FMPClient | None = None) -> pd.DataFrame:
    settings.ensure_dirs()
    market = build_market_monthly(client)
    out = settings.processed_dir / "market_monthly.parquet"
    write_parquet(market, out)
    logger.info("Saved market panel: {} rows x {} cols -> {}", len(market), market.shape[1], out)
    return market


if __name__ == "__main__":
    collect_market()
