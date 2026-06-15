"""Project-wide logger (loguru), configured once on import."""

from __future__ import annotations

import sys

from loguru import logger

from raa.utils.config import settings

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def configure_logging(level: str | None = None) -> None:
    """Reset sinks and attach a single coloured stderr sink."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=(level or settings.log_level).upper(),
        format=_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )


configure_logging()

__all__ = ["logger", "configure_logging"]
