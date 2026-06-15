"""Statistical validation via block bootstrap of REAL historical returns.

We do NOT simulate synthetic price paths. The circular block bootstrap resamples
contiguous blocks of the *actual observed* monthly returns (preserving short-run
autocorrelation), which yields confidence intervals for the Sharpe ratio and a
significance test for the Sharpe *difference* between strategies. Strategies are
resampled jointly (same block draws) so cross-correlation is preserved.

It also summarises regime persistence (how sticky regimes are), which underpins
whether regime signals are actionable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.regimes.rule_based import REGIME_ORDER, classify_regimes, regime_episodes, transition_matrix
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import save_fig

PPY = 12


def _boot_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    idx: list[int] = []
    while len(idx) < n:
        s = int(rng.integers(0, n))
        idx.extend(((s + np.arange(block)) % n).tolist())
    return np.array(idx[:n])


def _sharpe_cols(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    sd[sd == 0] = np.nan
    return mu / sd * np.sqrt(PPY)


def bootstrap_sharpe(
    excess: pd.DataFrame, n_boot: int = 5000, block: int = 6, seed: int = 42
) -> np.ndarray:
    """Return an (n_boot x n_strategy) array of bootstrap Sharpe ratios."""
    rng = np.random.default_rng(seed)
    arr = excess.to_numpy()
    n = len(arr)
    out = np.empty((n_boot, arr.shape[1]))
    for b in range(n_boot):
        out[b] = _sharpe_cols(arr[_boot_indices(n, block, rng)])
    return out


def validate(
    strategy_returns: pd.DataFrame,
    rf: pd.Series,
    pairs: list[tuple[str, str]] | None = None,
    n_boot: int = 5000,
    block: int = 6,
) -> dict:
    out = settings.reports_dir / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    excess = strategy_returns.sub(rf.reindex(strategy_returns.index), axis=0).dropna()
    cols = list(excess.columns)
    boots = bootstrap_sharpe(excess, n_boot=n_boot, block=block)

    point = _sharpe_cols(excess.to_numpy())
    ci = pd.DataFrame(
        {
            "sharpe": point,
            "ci_low": np.nanpercentile(boots, 2.5, axis=0),
            "ci_high": np.nanpercentile(boots, 97.5, axis=0),
        },
        index=cols,
    )
    write_csv(ci.round(3), out / "validation_sharpe_ci.csv")

    pairs = pairs or [
        ("Regime Risk Overlay", "ERC (Risk Parity)"),
        ("Regime Max-Sharpe", "Max-Sharpe (MVO)"),
        ("Regime Risk Overlay", "60/40"),
    ]
    rows = []
    for a, b in pairs:
        if a not in cols or b not in cols:
            continue
        ia, ib = cols.index(a), cols.index(b)
        diff = boots[:, ia] - boots[:, ib]
        rows.append(
            {
                "A": a, "B": b,
                "sharpe_diff": point[ia] - point[ib],
                "ci_low": float(np.nanpercentile(diff, 2.5)),
                "ci_high": float(np.nanpercentile(diff, 97.5)),
                "p_A_not_better": float(np.mean(diff <= 0)),
            }
        )
    diff_tbl = pd.DataFrame(rows)
    write_csv(diff_tbl.round(3), out / "validation_sharpe_diff.csv", index=False)

    persist = _regime_persistence()
    write_csv(persist.round(3), out / "validation_regime_persistence.csv")

    _fig_sharpe_ci(ci)
    if not diff_tbl.empty:
        _fig_diff_dist(boots, cols, pairs[0])

    logger.info("Sharpe 95% CIs:\n{}", ci.round(2).to_string())
    logger.info("Sharpe-difference tests:\n{}", diff_tbl.round(3).to_string())
    return {"sharpe_ci": ci, "sharpe_diff": diff_tbl, "persistence": persist}


def _regime_persistence() -> pd.DataFrame:
    macro = read_parquet(settings.processed_dir / "macro_monthly.parquet")
    regime = classify_regimes(macro)["regime"]
    tm = transition_matrix(regime, normalize=True)
    eps = regime_episodes(regime)
    rows = {}
    for r in REGIME_ORDER:
        stay = float(tm.loc[r, r]) if r in tm.index else np.nan
        durs = eps.loc[eps["regime"] == r, "months"]
        rows[r] = {
            "stay_prob": stay,
            "avg_duration_m": float(durs.mean()) if len(durs) else np.nan,
            "n_episodes": int(len(durs)),
        }
    return pd.DataFrame(rows).T


def _fig_sharpe_ci(ci: pd.DataFrame) -> None:
    c = ci.sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(c))
    ax.errorbar(c["sharpe"], y, xerr=[c["sharpe"] - c["ci_low"], c["ci_high"] - c["sharpe"]],
                fmt="o", color="#0072B2", ecolor="#888", capsize=3)
    ax.set_yticks(y, c.index)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_title("Bootstrap 95% confidence intervals for Sharpe ratio\n(block bootstrap of real monthly returns)")
    ax.set_xlabel("Sharpe ratio")
    save_fig(fig, "09_sharpe_confidence_intervals", subdir="phase3")


def _fig_diff_dist(boots: np.ndarray, cols: list[str], pair: tuple[str, str]) -> None:
    a, b = pair
    diff = boots[:, cols.index(a)] - boots[:, cols.index(b)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diff, bins=60, color="#009E73", alpha=0.8)
    ax.axvline(0, color="#C0392B", lw=1.2, label="zero (no difference)")
    ax.axvline(float(np.mean(diff)), color="#222", lw=1.2, ls="--", label="mean difference")
    ax.set_title(f"Bootstrap distribution of Sharpe difference:\n{a} minus {b}")
    ax.set_xlabel("Sharpe difference")
    ax.legend(fontsize=9)
    save_fig(fig, "10_sharpe_diff_distribution", subdir="phase3")
