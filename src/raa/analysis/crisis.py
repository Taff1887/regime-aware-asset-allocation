"""Historical crisis stress tests.

For each major crisis we measure, on REAL data:
- asset-class total return and max drawdown during the event;
- how correlations behaved pre / during / post (did diversifiers fail?);
- how static vs regime-aware portfolios drew down and recovered.

Correlations and drawdowns use daily returns (more observations, better for short
events like the COVID crash); performance uses compounded daily returns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.analysis import metrics
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import save_fig

# Event windows (during). Pre/post are the 12 months either side.
CRISES: dict[str, tuple[str, str]] = {
    "GFC 2007-09": ("2007-11-01", "2009-02-28"),
    "Euro Debt 2011": ("2011-05-01", "2012-06-30"),
    "China/Oil 2015-16": ("2015-06-01", "2016-02-29"),
    "COVID 2020": ("2020-02-19", "2020-03-31"),
    "Inflation Shock 2022": ("2022-01-01", "2022-10-31"),
}
DISPLAY_ASSETS = ["SPY", "EEM", "IEF", "TLT", "LQD", "HYG", "DBC", "GLD", "RWO"]
CRISIS_STRATS = ["60/40", "ERC (Risk Parity)", "Regime Risk Overlay", "Regime Max-Sharpe"]


def _win(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return daily.loc[start:end]


def _total_return(daily: pd.Series) -> float:
    d = daily.dropna()
    return float((1 + d).prod() - 1) if len(d) else np.nan


def asset_performance(daily: pd.DataFrame) -> pd.DataFrame:
    """Total return and max drawdown of each asset within each crisis window."""
    rows = {}
    for name, (s, e) in CRISES.items():
        w = _win(daily, s, e)
        rows[name] = {a: _total_return(w[a]) for a in DISPLAY_ASSETS if a in w}
    tr = pd.DataFrame(rows)
    return tr


def avg_pairwise_corr(daily: pd.DataFrame, assets: list[str]) -> float:
    """Mean off-diagonal correlation, using assets present in the window and
    pairwise-complete observations (so assets that post-date an early crisis
    simply drop out rather than voiding the whole matrix)."""
    cols = [a for a in assets if a in daily and daily[a].notna().sum() >= 10]
    if len(cols) < 2:
        return np.nan
    c = daily[cols].corr(min_periods=10).to_numpy()
    off = c[~np.eye(c.shape[0], dtype=bool)]
    return float(np.nanmean(off))


def correlation_regimes(daily: pd.DataFrame) -> pd.DataFrame:
    """Average pairwise correlation pre / during / post each crisis, plus the
    equity-bond (SPY/IEF) correlation."""
    rows = []
    for name, (s, e) in CRISES.items():
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        pre = daily.loc[s_ts - pd.Timedelta(days=365):s_ts]
        during = daily.loc[s_ts:e_ts]
        post = daily.loc[e_ts:e_ts + pd.Timedelta(days=365)]
        row = {"crisis": name}
        for label, w in [("pre", pre), ("during", during), ("post", post)]:
            row[f"avg_corr_{label}"] = avg_pairwise_corr(w, DISPLAY_ASSETS)
            sub = w[["SPY", "IEF"]].dropna()
            row[f"eqbond_{label}"] = float(sub["SPY"].corr(sub["IEF"])) if len(sub) >= 10 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("crisis")


def strategy_drawdowns(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """Max drawdown and total return of each strategy within each crisis window
    (monthly strategy returns)."""
    rows = {}
    for name, (s, e) in CRISES.items():
        # widen slightly to capture the monthly trough
        w = strategy_returns.loc[pd.Timestamp(s) - pd.Timedelta(days=31): pd.Timestamp(e) + pd.Timedelta(days=31)]
        rows[name] = {
            strat: metrics.max_drawdown(w[strat]) for strat in CRISIS_STRATS if strat in w
        }
    return pd.DataFrame(rows)


def analyze(strategy_returns: pd.DataFrame | None = None) -> dict:
    out = settings.reports_dir / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    daily = read_parquet(settings.processed_dir / "returns_daily.parquet")

    perf = asset_performance(daily)
    corr = correlation_regimes(daily)
    write_csv((perf * 100).round(1), out / "crisis_asset_returns.csv")
    write_csv(corr.round(3), out / "crisis_correlations.csv")

    dd = None
    if strategy_returns is not None and not strategy_returns.empty:
        dd = strategy_drawdowns(strategy_returns)
        write_csv((dd * 100).round(1), out / "crisis_strategy_drawdowns.csv")

    _fig_asset_returns(perf)
    _fig_correlations(corr)
    if dd is not None:
        _fig_strategy_dd(dd)

    logger.info("Crisis equity/bond correlation (during):\n{}", corr[["eqbond_pre", "eqbond_during", "eqbond_post"]].round(2).to_string())
    return {"asset_perf": perf, "corr": corr, "strategy_dd": dd}


def _fig_asset_returns(perf: pd.DataFrame) -> None:
    data = (perf * 100).reindex(DISPLAY_ASSETS)
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(data.to_numpy(), cmap="RdYlGn", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(data.index)), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.to_numpy()[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("Asset total return through each crisis (%)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="total return %")
    save_fig(fig, "05_crisis_asset_returns", subdir="phase3")


def _fig_correlations(corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(corr.index))
    w = 0.27
    for k, ph in enumerate(["pre", "during", "post"]):
        axes[0].bar(x + (k - 1) * w, corr[f"avg_corr_{ph}"].to_numpy(), w, label=ph)
        axes[1].bar(x + (k - 1) * w, corr[f"eqbond_{ph}"].to_numpy(), w, label=ph)
    for ax, title in zip(axes, ["Average pairwise correlation", "Equity-bond correlation (SPY/IEF)"]):
        ax.set_xticks(x, corr.index, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Did diversification hold through crises? (pre / during / post)", y=1.02, weight="bold")
    save_fig(fig, "06_crisis_correlations", subdir="phase3")


def _fig_strategy_dd(dd: pd.DataFrame) -> None:
    data = (dd * 100)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(data.columns))
    n = len(data.index)
    w = 0.8 / n
    for k, strat in enumerate(data.index):
        ax.bar(x + (k - (n - 1) / 2) * w, data.loc[strat].to_numpy(), w, label=strat)
    ax.set_xticks(x, data.columns, rotation=20, ha="right")
    ax.set_ylabel("Max drawdown %")
    ax.set_title("Portfolio max drawdown through each crisis: static vs regime-aware")
    ax.legend(fontsize=8)
    save_fig(fig, "07_crisis_strategy_drawdowns", subdir="phase3")
