"""Central configuration.

All runtime settings (the FMP key, rate limits, derived directory paths) are
resolved here so the rest of the codebase never reads environment variables or
hard-codes a path. Settings are loaded once from ``.env`` via pydantic-settings
and exposed through the module-level singleton :data:`settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three parents up from src/raa/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Typed runtime configuration, populated from ``.env`` + environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Financial Modeling Prep ---
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    fmp_calls_per_minute: int = 300
    fmp_max_retries: int = 5

    # --- Logging ---
    log_level: str = "INFO"

    # ------------------------------------------------------------------ paths
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        """On-disk cache for raw FMP JSON responses."""
        return self.data_dir / "cache"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def config_dir(self) -> Path:
        return PROJECT_ROOT / "config"

    @property
    def figures_dir(self) -> Path:
        return PROJECT_ROOT / "figures"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    def ensure_dirs(self) -> None:
        """Create every output directory the pipeline writes to."""
        for p in (
            self.raw_dir,
            self.cache_dir,
            self.processed_dir,
            self.figures_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
