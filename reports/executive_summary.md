# Regime-Aware Strategic Asset Allocation — Executive Summary

**For:** CIO · Investment Committee · Head of Multi-Asset · Insurance Portfolio Team
**Length:** ~3 pages · **Full report:** [`report.md`](report.md)

---

## Research objective

Determine whether conditioning strategic asset allocation on the **macroeconomic regime** — defined simply and transparently — produces materially better, *implementable* portfolio outcomes than a static policy allocation. The study deliberately prioritises **simple, explainable, robust** methods over model sophistication, and tests every conclusion out-of-sample, net of costs, on **real historical data only** (Financial Modeling Prep; no simulated data).

## Methodology in brief

- A **four-quadrant Growth × Inflation regime model** (Goldilocks, Overheating, Stagflation, Recession), classified point-in-time from CPI and industrial-production trends. Hidden Markov, clustering and market-stress models are run only as robustness checks.
- A globally diversified, USD-based asset universe of liquid ETF total-return proxies (equities, three Treasury durations, IG/HY credit, infrastructure, REITs, commodities, gold), 1993–2026 where available.
- Per-regime return/risk/correlation statistics; an eight-factor macro risk decomposition; and a costed, point-in-time backtest of seven static and four regime-aware strategies, plus crisis stress tests, currency-hedging analysis and bootstrap significance testing.

## Main findings

1. **Asset behaviour is strongly regime-dependent.** Equities: Sharpe 0.82 (Goldilocks), 1.03 (Recession/recovery), **−0.17 (Stagflation)**. Long Treasuries: 1.03 (Goldilocks), **−0.54 (Overheating)**. Gold: **1.12 (Stagflation)**. Commodities: **1.09 (Overheating)**. The ranking of asset classes reshuffles materially by regime.

2. **Diversification weakens exactly when inflation is the problem.** The equity–bond correlation is **−0.55 in Goldilocks** but turns **positive (+0.13 to +0.16) in inflationary regimes**. Average cross-asset correlation rises from **0.22 (Goldilocks) to 0.40 (Stagflation)**.

3. **The 2022 inflation shock is the proof.** The equity–bond hedge held through the GFC (−0.47), Euro crisis (−0.69), China 2015 (−0.45) and COVID (−0.54) — but **broke in 2022 (+0.13)**: long Treasuries fell −34%, gold −11%, equities −18%, and **only commodities rose (+21%)**.

4. **60/40 is not balanced by risk:** ~**95% of its variance comes from a single equity/growth factor**, versus a far more even spread for a diversified equal-weight portfolio.

5. **Regime awareness pays as risk management, not alpha.** Conditioning the covariance on the regime adds nothing; conditioning *return forecasts* hurts. But a transparent **de-risking overlay cut maximum drawdown ~24%** (−15.0% vs −19.6% base) and reduced the **GFC drawdown to −10.9%** (vs 60/40's −28.9%) and **COVID to −1.2%** — with low turnover and robustness to costs. The Sharpe gain, however, is **not statistically significant** on 21 years of data.

6. **Fixed-income deep dive (1980–2026, real long-history funds) — for a bond-heavy book.** Over 45 years including the Great Inflation, the **bond–equity correlation is highest in Stagflation** (US aggregate +0.41, long credit +0.43) — the hedge failure is a *regime feature, not a 2022 fluke*. The dominant fixed-income lever is **duration** (long beats short ~+8%/yr in Goldilocks, −4%/yr in Overheating), **credit and high yield lag in Stagflation** (not defensive), **unhedged global bonds lag US by ~9%/yr in Overheating** (strong dollar), and **TIPS hedge inflation surprises but not the 2022-style real-rate shock**. No nominal bond reliably hedges a stagflationary equity drawdown — the structural case for a real-asset sleeve alongside the bond book.

## Portfolio implications

- The diversifier that protects equities **depends on the regime**: bonds in disinflationary environments, **real assets (gold, commodities) in inflationary ones**. A static allocation reliant on bonds for ballast is implicitly betting on a disinflationary world.
- For a USD investor, **hedging foreign-currency exposure improved risk-adjusted returns** (Sharpe 0.40 → 0.47; volatility 15.3% → 12.7%), with the largest FX drag in Stagflation.

## Key recommendations

1. **Set SAA assumptions conditionally.** Stress-test the policy portfolio against the *Stagflation* correlation matrix; stop assuming the equity–bond hedge works in an inflation shock. *(Highest confidence.)*
2. **Hold a structural real-asset allocation** (gold/commodities) as cheap insurance against the one regime in which everything else correlates. *(High confidence.)*
3. **Run a modest, rules-based regime risk overlay** — de-risk ~50% of the equity sleeve into defensives in confirmed adverse regimes, rebalanced quarterly — for drawdown control, *not* for expected alpha. Ensure the defensive sleeve includes real assets so it also defends against inflation shocks. *(Moderate confidence.)*
4. **Hedge foreign-currency exposure** (fully or substantially) from a USD base.
5. **Avoid complexity that doesn't pay:** regime-conditional return forecasting and high-frequency tactical timing destroyed value after costs.

## Limitations

Liquid proxies span 1993–2026 (full universe from 2005), covering four crises but only ~1.5 inflation regimes, so inflation-regime conclusions and Sharpe-difference tests have limited statistical power. Regimes are US-anchored; ETF proxies differ from indices; per-country bond curves are incompletely represented (documented, not hidden); 60/40's strong showing is partly an artefact of US-equity outperformance in this era.

**Bottom line:** macro regimes should reshape how an institution sets and stress-tests its SAA assumptions and how it diversifies — and can support a simple, transparent risk overlay — but they are a risk-management and assumption tool, not an alpha engine.
