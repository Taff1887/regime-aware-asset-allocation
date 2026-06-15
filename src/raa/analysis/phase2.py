"""Phase 2: supplementary regimes (HMM, clustering, market-implied) + risk-factor
decomposition.

Run with::

    uv run python -m raa.analysis.phase2
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raa.factors.build import FACTOR_LABELS, build_factors
from raa.factors.decomposition import (
    factor_betas,
    factor_perf_by_regime,
    portfolio_factor_risk,
)
from raa.regimes.market_implied import MARKET_STATES, market_regime
from raa.regimes.rule_based import REGIME_ORDER, classify_regimes, regime_episodes
from raa.regimes.statistical import (
    agreement,
    build_feature_matrix,
    fit_clustering,
    fit_hmm,
    optimal_assignment,
)
from raa.utils.config import settings
from raa.utils.io import read_parquet, write_csv
from raa.utils.logging import logger
from raa.utils.viz import REGIME_COLORS, save_fig

MARKET_COLORS = {"Risk-On": "#2E8B57", "Neutral": "#BBBBBB", "Risk-Off": "#C0392B"}
CORE_SET = ["SPY", "EEM", "IEF", "TLT", "LQD", "HYG", "DBC", "GLD", "VNQ"]


def _outdir():
    d = settings.reports_dir / "phase2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _regime_strip(ax, regime: pd.Series, colors: dict, title: str) -> None:
    r = regime.dropna()
    grp = (r.astype(str) != r.astype(str).shift()).cumsum()
    for _, idx in r.groupby(grp).groups.items():
        idx = pd.DatetimeIndex(idx)
        label = str(r.loc[idx[0]])
        end = idx[-1] + pd.offsets.MonthEnd(1)
        ax.axvspan(idx[0], end, color=colors.get(label, "#ccc"), alpha=0.8, lw=0)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)


def run() -> dict:
    settings.ensure_dirs()
    out = _outdir()
    macro = read_parquet(settings.processed_dir / "macro_monthly.parquet")
    market = read_parquet(settings.processed_dir / "market_monthly.parquet")
    returns = read_parquet(settings.processed_dir / "returns_monthly.parquet")
    rf = read_parquet(settings.processed_dir / "rf_monthly.parquet")["rf"]

    rule = classify_regimes(macro)["regime"]

    # --- Factors --------------------------------------------------------------
    factors = build_factors(returns, rf, market)
    factors.to_parquet(settings.processed_dir / "factors_monthly.parquet")
    fcols = [c for c in factors.columns if factors[c].notna().sum() > 24]

    betas, r2, alpha = factor_betas(returns, factors, factor_cols=fcols, assets=CORE_SET)
    write_csv(betas.round(3), out / "factor_betas.csv")
    write_csv(r2.round(3).to_frame("r2"), out / "factor_r2.csv")

    perf = factor_perf_by_regime(factors, rule, factor_cols=fcols)
    for k, tbl in perf.items():
        write_csv(tbl.round(4), out / f"factor_{k}_by_regime.csv")

    pf6040 = portfolio_factor_risk({"SPY": 0.6, "IEF": 0.4}, returns, factors, fcols)
    pfew = portfolio_factor_risk({a: 1.0 for a in CORE_SET}, returns, factors, fcols)
    pf_risk = pd.DataFrame({"60/40": pf6040, "Equal-Weight": pfew})
    write_csv(pf_risk.round(3), out / "portfolio_factor_risk.csv")

    # --- HMM + clustering -----------------------------------------------------
    raw, Xz = build_feature_matrix(macro, market)
    hmm = fit_hmm(raw, Xz, n_states=4)
    hmm_named, _ = optimal_assignment(hmm["states"], rule)
    write_csv(hmm["transmat"].round(3), out / "hmm_transition_matrix.csv")
    write_csv(
        pd.DataFrame({"persistence": hmm["persistence"], "exp_duration_m": hmm["expected_duration_months"]}).round(2),
        out / "hmm_persistence.csv",
    )
    write_csv(hmm["profiles"].round(2), out / "hmm_state_profiles.csv")
    km = fit_clustering(raw, Xz, "kmeans")
    gmm = fit_clustering(raw, Xz, "gmm")
    agg = fit_clustering(raw, Xz, "agglomerative")

    agree = pd.DataFrame(
        {
            "HMM": agreement(rule, hmm["states"]),
            "KMeans": agreement(rule, km),
            "GMM": agreement(rule, gmm),
            "Agglomerative": agreement(rule, agg),
        }
    ).T
    write_csv(agree.round(3), out / "regime_method_agreement.csv")

    # --- Market-implied -------------------------------------------------------
    mkt_reg, stress = market_regime(market)
    mret = _market_regime_returns(returns, mkt_reg)
    write_csv(mret.round(4), out / "market_regime_asset_returns.csv")

    # --- Figures --------------------------------------------------------------
    _fig_betas(betas)
    _fig_factor_ir(perf["ir"])
    _fig_pf_risk(pf_risk)
    _fig_method_timelines(rule, hmm_named, mkt_reg)
    _fig_agreement(agree)
    _fig_market_stress(market, stress, mkt_reg)
    _fig_market_regime_returns(mret)

    summary = {
        "factor_r2_mean": float(r2.mean()),
        "pf_factor_risk_6040": pf6040.round(3).to_dict(),
        "hmm_vs_rule": agreement(rule, hmm["states"]),
        "kmeans_vs_rule": agreement(rule, km),
        "gmm_vs_rule": agreement(rule, gmm),
        "hmm_persistence": hmm["persistence"].round(2).to_dict(),
        "factor_ir_by_regime": perf["ir"].round(2).to_dict(),
        "market_regime_spy": mret.loc["SPY"].round(4).to_dict() if "SPY" in mret.index else {},
    }
    with open(out / "phase2_findings.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("PHASE 2 FINDINGS")
    logger.info("Mean factor-model R^2 across core assets: {:.2f}", summary["factor_r2_mean"])
    logger.info("60/40 variance from factors: {}", summary["pf_factor_risk_6040"])
    logger.info("HMM vs rule-based agreement: {}", summary["hmm_vs_rule"])
    logger.info("KMeans vs rule-based: {}", summary["kmeans_vs_rule"])
    logger.info("HMM state persistence: {}", summary["hmm_persistence"])
    logger.info("Market-regime SPY mean monthly return: {}", summary["market_regime_spy"])
    logger.info("=" * 70)
    return summary


def _market_regime_returns(returns: pd.DataFrame, mkt_reg: pd.Series) -> pd.DataFrame:
    reg = mkt_reg.dropna().astype(str)
    idx = returns.index.intersection(reg.index)
    rets, reg = returns.loc[idx], reg.loc[idx]
    rows = {}
    for a in CORE_SET:
        if a not in rets:
            continue
        rows[a] = {st: rets.loc[reg == st, a].mean() for st in MARKET_STATES}
    return pd.DataFrame(rows).T.reindex(columns=MARKET_STATES)


# ------------------------------------------------------------------------- figs
def _fig_betas(betas: pd.DataFrame) -> None:
    b = betas.astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.nanmax(np.abs(b.to_numpy()))
    im = ax.imshow(b.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(b.columns)), [FACTOR_LABELS.get(c, c) for c in b.columns], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(b.index)), b.index)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            v = b.to_numpy()[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Asset exposures (betas) to macro risk factors")
    fig.colorbar(im, ax=ax, shrink=0.7, label="beta")
    save_fig(fig, "01_factor_betas", subdir="phase2")


def _fig_factor_ir(ir: pd.DataFrame) -> None:
    sub = ir[REGIME_ORDER].astype(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(sub.to_numpy(), cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(REGIME_ORDER)), REGIME_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(sub.index)), [FACTOR_LABELS.get(c, c) for c in sub.index], fontsize=8)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.to_numpy()[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Factor information ratio by regime\n(which macro factors pay in each regime)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="ann. info ratio")
    save_fig(fig, "02_factor_ir_by_regime", subdir="phase2")


def _fig_pf_risk(pf_risk: pd.DataFrame) -> None:
    df = pf_risk.fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(df.columns))
    x = range(len(df.columns))
    for fac in df.index:
        ax.bar(x, df.loc[fac].to_numpy() * 100, bottom=bottom, label=FACTOR_LABELS.get(fac, fac))
        bottom += df.loc[fac].to_numpy() * 100
    ax.set_xticks(list(x), df.columns)
    ax.set_ylabel("% of portfolio variance")
    ax.set_title("Portfolio variance attributed to macro factors")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save_fig(fig, "03_portfolio_factor_risk", subdir="phase2")


def _fig_method_timelines(rule, hmm_labels, mkt_reg) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.5), sharex=True)
    _regime_strip(axes[0], rule, REGIME_COLORS, "Rule-based")
    _regime_strip(axes[1], hmm_labels, REGIME_COLORS, "HMM (4-state)")
    _regime_strip(axes[2], mkt_reg, MARKET_COLORS, "Market-implied")
    axes[0].set_title("Regime classification across methods")
    from matplotlib.patches import Patch

    h1 = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    h2 = [Patch(facecolor=MARKET_COLORS[r], label=r) for r in MARKET_STATES]
    axes[2].legend(handles=h1 + h2, ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    save_fig(fig, "04_method_timelines", subdir="phase2")


def _fig_agreement(agree: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agree.index, agree["adjusted_rand"].to_numpy(), color="#0072B2", alpha=0.8, label="Adjusted Rand")
    ax.bar(agree.index, agree["match_rate"].to_numpy(), color="#D55E00", alpha=0.5, label="Match rate")
    ax.set_title("Agreement of data-driven regimes with the rule-based model")
    ax.set_ylabel("Score")
    ax.legend()
    for i, v in enumerate(agree["adjusted_rand"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    save_fig(fig, "05_method_agreement", subdir="phase2")


def _fig_market_stress(market, stress, mkt_reg) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(stress.index, stress.values, color="#222", lw=1.2, label="Stress index (z)")
    ax.axhline(0.5, color="#C0392B", ls="--", lw=0.8)
    ax.axhline(-0.5, color="#2E8B57", ls="--", lw=0.8)
    eps = regime_episodes(mkt_reg.rename("regime")) if mkt_reg.notna().any() else None
    if eps is not None:
        for _, row in eps.iterrows():
            end = row["end"] + pd.offsets.MonthEnd(1)
            ax.axvspan(row["start"], end, color=MARKET_COLORS.get(row["regime"], "#ccc"), alpha=0.12, lw=0)
    ax.set_title("Market-implied stress index (VIX + FX vol + funding spread)")
    ax.set_ylabel("Composite z-score")
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, "06_market_stress", subdir="phase2")


def _fig_market_regime_returns(mret: pd.DataFrame) -> None:
    df = (mret * 100).astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df.index))
    w = 0.25
    for k, st in enumerate(MARKET_STATES):
        ax.bar(x + (k - 1) * w, df[st].to_numpy(), w, label=st, color=MARKET_COLORS[st])
    ax.set_xticks(x, df.index, rotation=45, ha="right")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Mean monthly return %")
    ax.set_title("Mean asset return by market-implied regime")
    ax.legend()
    save_fig(fig, "07_market_regime_returns", subdir="phase2")


if __name__ == "__main__":
    run()
