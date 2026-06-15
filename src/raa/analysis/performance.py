"""Strategy performance evaluation, including benchmark-relative metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raa.analysis import metrics


def capture(strategy: pd.Series, benchmark: pd.Series, up: bool = True) -> float:
    """Up/down capture: mean strategy return in benchmark up/down months,
    divided by the benchmark's mean in those months."""
    df = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    mask = df["b"] > 0 if up else df["b"] < 0
    if mask.sum() == 0 or df.loc[mask, "b"].mean() == 0:
        return np.nan
    return float(df.loc[mask, "s"].mean() / df.loc[mask, "b"].mean())


def active_stats(strategy: pd.Series, benchmark: pd.Series, ppy: int = 12) -> dict[str, float]:
    df = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 12:
        return {"tracking_error": np.nan, "info_ratio": np.nan, "beta": np.nan,
                "up_capture": np.nan, "down_capture": np.nan}
    active = df["s"] - df["b"]
    te = float(active.std(ddof=1) * np.sqrt(ppy))
    ir = float(active.mean() / active.std(ddof=1) * np.sqrt(ppy)) if active.std(ddof=1) > 0 else np.nan
    beta = float(np.cov(df["s"], df["b"])[0, 1] / np.var(df["b"], ddof=1))
    return {
        "tracking_error": te,
        "info_ratio": ir,
        "beta": beta,
        "up_capture": capture(df["s"], df["b"], up=True),
        "down_capture": capture(df["s"], df["b"], up=False),
    }


def evaluate(
    net: pd.Series,
    rf: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    """Full performance bundle for a strategy return series."""
    out = metrics.summary_stats(net, rf)
    if turnover is not None and len(turnover):
        out["avg_turnover"] = float(turnover.reindex(net.index).mean())
    if benchmark is not None:
        out.update(active_stats(net, benchmark))
    return out


def evaluate_many(
    strategies: dict[str, dict],
    rf: pd.Series | None = None,
    benchmark_name: str | None = None,
) -> pd.DataFrame:
    """Evaluate a dict of backtest results -> tidy metric table (strategies x metrics)."""
    bench = strategies[benchmark_name]["net"] if benchmark_name else None
    rows = {}
    for name, res in strategies.items():
        if "net" not in res or res["net"].empty:
            continue
        rows[name] = evaluate(res["net"], rf=rf, benchmark=bench, turnover=res.get("turnover"))
    return pd.DataFrame(rows).T
