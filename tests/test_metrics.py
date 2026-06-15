"""Unit tests for performance/risk metrics (deterministic toy inputs)."""

import numpy as np
import pandas as pd
import pytest

from raa.analysis import metrics


def test_ann_return_constant():
    r = pd.Series([0.01] * 12)
    assert metrics.ann_return(r) == pytest.approx(1.01**12 - 1)


def test_ann_vol_zero_for_constant():
    r = pd.Series([0.01] * 24)
    assert metrics.ann_vol(r) == 0.0


def test_sharpe_sign_and_rf():
    r = pd.Series([0.02, 0.01, 0.03, 0.0, 0.02] * 6)
    assert metrics.sharpe(r) > 0
    assert metrics.sharpe(r, rf=0.05) < metrics.sharpe(r, rf=0.0)


def test_max_drawdown_known_path():
    r = pd.Series([1.0, -0.5])  # +100% then -50%
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_var_es_ordering():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.05, 1000))
    var = metrics.hist_var(r, 0.05)
    es = metrics.expected_shortfall(r, 0.05)
    assert es <= var < 0


def test_summary_stats_keys():
    r = pd.Series([0.01, -0.02, 0.03, 0.0] * 6)
    s = metrics.summary_stats(r)
    for k in ["ann_return", "ann_vol", "sharpe", "max_drawdown", "var_5", "es_5", "hit_rate"]:
        assert k in s
