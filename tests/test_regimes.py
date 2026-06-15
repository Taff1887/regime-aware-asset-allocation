"""Unit tests for regime classification and helpers."""

import pandas as pd

from raa.regimes.rule_based import (
    REGIME_ORDER,
    classify_regimes,
    confirm_regime,
    transition_matrix,
)


def _macro():
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    # growth: first 6 low, last 6 high; inflation: alternating low/high
    indpro = [0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10]
    cpi = [0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10]
    return pd.DataFrame({"indpro_yoy": indpro, "cpi_yoy": cpi}, index=idx)


def test_quadrant_mapping_full_median():
    panel = classify_regimes(_macro(), smooth=1, threshold="full_median", min_periods=1, lag=0)
    reg = panel["regime"].astype(str)
    # row idx 6: growth high (10), inflation low (0) -> Goldilocks
    assert reg.iloc[6] == "Goldilocks"
    # row 7: growth high, inflation high -> Overheating
    assert reg.iloc[7] == "Overheating"
    # row 0: growth low, inflation low -> Recession
    assert reg.iloc[0] == "Recession"
    # row 1: growth low, inflation high -> Stagflation
    assert reg.iloc[1] == "Stagflation"


def test_confirm_regime_filters_whipsaw():
    seq = ["Goldilocks"] * 5 + ["Stagflation"] + ["Goldilocks"] * 5
    s = pd.Series(seq, index=pd.date_range("2000-01-31", periods=11, freq="ME"))
    conf = confirm_regime(s, k=3).astype(str)
    # the single-month Stagflation flip should not be confirmed
    assert (conf == "Stagflation").sum() == 0
    assert conf.iloc[-1] == "Goldilocks"


def test_transition_matrix_rows_sum_to_one():
    panel = classify_regimes(_macro(), smooth=1, threshold="full_median", min_periods=1, lag=0)
    tm = transition_matrix(panel["regime"])
    row_sums = tm.sum(axis=1)
    # rows with any transitions sum to 1
    for r in REGIME_ORDER:
        if tm.loc[r].sum() > 0:
            assert abs(row_sums[r] - 1.0) < 1e-9
