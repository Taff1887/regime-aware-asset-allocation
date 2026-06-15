"""Unit tests for portfolio construction and the backtest engine."""

import numpy as np
import pandas as pd

from raa.portfolio import construct
from raa.portfolio.backtest import available_assets, run_backtest, static_strategy


def _cov(vols, corr=0.0):
    vols = np.array(vols)
    n = len(vols)
    c = np.full((n, n), corr)
    np.fill_diagonal(c, 1.0)
    cov = c * np.outer(vols, vols)
    return pd.DataFrame(cov, index=[f"A{i}" for i in range(n)], columns=[f"A{i}" for i in range(n)])


def test_equal_weight_sums_to_one():
    w = construct.equal_weight(["A", "B", "C"])
    assert abs(w.sum() - 1.0) < 1e-12
    assert (w == 1 / 3).all()


def test_min_variance_favours_low_vol_asset():
    cov = _cov([0.05, 0.20])  # A0 much lower vol
    w = construct.min_variance(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()
    assert w["A0"] > w["A1"]


def test_erc_identity_is_equal_weight():
    cov = _cov([0.1, 0.1, 0.1], corr=0.0)  # identical, uncorrelated
    w = construct.equal_risk_contribution(cov)
    assert np.allclose(w.to_numpy(), 1 / 3, atol=1e-2)


def test_risk_contributions_sum_to_one():
    cov = _cov([0.1, 0.2, 0.15], corr=0.3)
    w = construct.inverse_vol(cov)
    rc = construct.risk_contributions(w, cov)
    assert abs(rc.sum() - 1.0) < 1e-9


def test_available_assets_respects_min_obs():
    idx = pd.date_range("2000-01-31", periods=30, freq="ME")
    df = pd.DataFrame({"A": range(30), "B": [np.nan] * 20 + list(range(10))}, index=idx)
    avail = available_assets(df, ["A", "B"], min_obs=24)
    assert avail == ["A"]


def test_backtest_fixed_weights_zero_turnover():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(
        {"SPY": rng.normal(0.005, 0.04, 60), "IEF": rng.normal(0.002, 0.015, 60)}, index=idx
    )
    fn = static_strategy("fixed", ["SPY", "IEF"], fixed={"SPY": 0.6, "IEF": 0.4})
    res = run_backtest(rets, fn, cost_bps=10, min_obs=12, rebalance_every=1, name="6040")
    assert not res["net"].empty
    # constant target weights => ~zero turnover after the first rebalance
    assert res["turnover"].iloc[1:].abs().mean() < 1e-6
