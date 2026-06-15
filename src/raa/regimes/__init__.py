"""Regime detection.

The **core** framework is rule-based (Growth x Inflation quadrants) and lives in
:mod:`raa.regimes.rule_based`. Supplementary, data-driven methods (HMM,
clustering, market-implied) are added in Phase 2 as robustness checks.
"""

from raa.regimes.rule_based import (
    REGIME_ORDER,
    classify_regimes,
    regime_series,
)

__all__ = ["REGIME_ORDER", "classify_regimes", "regime_series"]
