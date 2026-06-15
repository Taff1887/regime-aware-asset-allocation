"""Orchestrate the full data pull: macro panel + price/return panels.

Run with::

    uv run python -m raa.data.collect

All responses are cached under ``data/cache`` so subsequent runs are instant and
work offline. Processed panels are written to ``data/processed``.
"""

from __future__ import annotations

from raa.data.fmp_client import FMPClient
from raa.data.macro import collect_macro
from raa.data.prices import collect_prices
from raa.utils.config import settings
from raa.utils.logging import logger


def main() -> None:
    settings.ensure_dirs()
    if not settings.fmp_api_key:
        logger.error("No FMP_API_KEY set. Copy .env.example to .env and add your key.")
        return

    client = FMPClient()
    logger.info("=== Collecting macro data ===")
    collect_macro(client)
    logger.info("=== Collecting price data ===")
    collect_prices(client)
    logger.info("Data collection complete. Processed panels in {}", settings.processed_dir)


if __name__ == "__main__":
    main()
