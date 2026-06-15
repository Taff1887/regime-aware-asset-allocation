"""A thin, robust client for the Financial Modeling Prep (FMP) ``stable`` API.

Three properties make this safe for large, reproducible pulls:

1. **On-disk caching** - every response is cached as JSON under ``data/cache``
   keyed by a hash of the endpoint and parameters (the API key is excluded from
   the key). Re-running the pipeline is therefore instant and offline-friendly.
2. **Retries with backoff** - transient errors (timeouts, 429, 5xx) are retried
   with exponential backoff via ``tenacity``.
3. **Client-side rate limiting** - calls are throttled to ``fmp_calls_per_minute``
   so we stay within plan limits even across many symbols.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from raa.utils.config import settings
from raa.utils.io import hash_key, read_json, write_json
from raa.utils.logging import logger


class FMPError(RuntimeError):
    """Raised when the FMP API returns an error payload or status."""


class FMPClient:
    """Caching, throttled, retrying wrapper around the FMP REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        calls_per_minute: int | None = None,
        use_cache: bool = True,
    ) -> None:
        self.api_key = api_key or settings.fmp_api_key
        self.base_url = (base_url or settings.fmp_base_url).rstrip("/")
        self.calls_per_minute = calls_per_minute or settings.fmp_calls_per_minute
        self.use_cache = use_cache
        self._min_interval = 60.0 / max(1, self.calls_per_minute)
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._session = requests.Session()

        if not self.api_key:
            logger.warning(
                "FMP_API_KEY is empty - live calls will fail. "
                "Copy .env.example to .env and add your key."
            )
        settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- caching
    def _cache_path(self, endpoint: str, params: dict[str, Any]):
        slug = endpoint.replace("/", "_")
        # Sort params (minus the key) so cache lookups are order-independent.
        key_parts = sorted(f"{k}={v}" for k, v in params.items() if k != "apikey")
        return settings.cache_dir / f"{slug}__{hash_key(slug, *key_parts)}.json"

    # ------------------------------------------------------------- throttling
    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    # ------------------------------------------------------------- networking
    @retry(
        retry=retry_if_exception_type((requests.RequestException, FMPError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(settings.fmp_max_retries),
        reraise=True,
    )
    def _request(self, endpoint: str, params: dict[str, Any]) -> Any:
        self._throttle()
        url = f"{self.base_url}/{endpoint}"
        resp = self._session.get(url, params={**params, "apikey": self.api_key}, timeout=40)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise FMPError(f"{endpoint}: HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and ("Error Message" in data or "error" in data):
            raise FMPError(f"{endpoint}: {data}")
        return data

    def get(self, endpoint: str, **params: Any) -> Any:
        """Return parsed JSON for ``endpoint``, using the disk cache when possible."""
        params = {k: v for k, v in params.items() if v is not None}
        cache_path = self._cache_path(endpoint, params)
        if self.use_cache and cache_path.exists():
            return read_json(cache_path)

        data = self._request(endpoint, params)
        if self.use_cache:
            write_json(data, cache_path)
        return data

    # ----------------------------------------------------- typed convenience API
    def economic_indicator(
        self, name: str, start: str = "1990-01-01", end: str = "2026-12-31"
    ) -> list[dict[str, Any]]:
        """Macro series (e.g. ``CPI``, ``realGDP``, ``federalFunds``)."""
        data = self.get("economic-indicators", name=name, **{"from": start, "to": end})
        return data if isinstance(data, list) else []

    def treasury_rates(
        self, start: str = "1990-01-01", end: str = "2026-12-31"
    ) -> list[dict[str, Any]]:
        """Daily US Treasury constant-maturity curve (1m .. 30y)."""
        data = self.get("treasury-rates", **{"from": start, "to": end})
        return data if isinstance(data, list) else []

    def historical_prices(
        self,
        symbol: str,
        start: str = "1990-01-01",
        end: str = "2026-12-31",
        series: str = "dividend-adjusted",
    ) -> list[dict[str, Any]]:
        """Daily EOD prices. ``series='dividend-adjusted'`` gives total-return prices."""
        endpoint = f"historical-price-eod/{series}"
        data = self.get(endpoint, symbol=symbol, **{"from": start, "to": end})
        return data if isinstance(data, list) else []
