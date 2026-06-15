"""Small, dependency-light I/O helpers used across the pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def hash_key(*parts: Any) -> str:
    """Deterministic short hash of arbitrary parts, for cache filenames.

    Order-sensitive. ``parts`` are stringified then hashed; the FMP API key is
    deliberately never passed in so cache files are key-independent.
    """
    blob = "|".join(str(p) for p in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def read_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    return path
