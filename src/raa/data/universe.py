"""Asset-universe definitions, loaded from ``config/universe.yaml``."""

from __future__ import annotations

from dataclasses import dataclass

from raa.utils.config import settings
from raa.utils.io import read_yaml


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str
    sub_class: str
    region: str
    currency: str
    inception: str
    core: bool


def _load_raw() -> dict:
    return read_yaml(settings.config_dir / "universe.yaml")


def load_universe(core_only: bool = False) -> list[Asset]:
    """Return the list of :class:`Asset` definitions.

    Parameters
    ----------
    core_only:
        If ``True``, return only the long-history ``core`` subset (data ~2004+).
    """
    raw = _load_raw()
    assets = [Asset(**a) for a in raw["assets"]]
    if core_only:
        assets = [a for a in assets if a.core]
    return assets


def tickers(core_only: bool = False) -> list[str]:
    return [a.ticker for a in load_universe(core_only=core_only)]


def fx_pairs() -> list[dict]:
    return _load_raw().get("fx", [])


def base_currency() -> str:
    return _load_raw().get("base_currency", "USD")


def asset_class_map(core_only: bool = False) -> dict[str, str]:
    """Map ticker -> asset_class."""
    return {a.ticker: a.asset_class for a in load_universe(core_only=core_only)}


def label_map(core_only: bool = False) -> dict[str, str]:
    """Map ticker -> human-readable name."""
    return {a.ticker: a.name for a in load_universe(core_only=core_only)}
