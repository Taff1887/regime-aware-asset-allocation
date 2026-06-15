"""Rule-based macro-regime classification: the transparent core of the project.

Each month is placed into one of four quadrants defined by the **level of growth**
and the **level of inflation** relative to their own point-in-time trend:

                       Inflation LOW        Inflation HIGH
    Growth HIGH        Goldilocks           Overheating
    Growth LOW         Recession            Stagflation

Design choices (all explicit, no look-ahead):

- **Growth signal**  : industrial-production YoY (%), 3-month smoothed.
- **Inflation signal**: CPI YoY (%), 3-month smoothed.
- **Threshold**      : the *expanding median* of each signal using only data up
  to month ``t`` (``min_periods`` guards the early sample). "High" = at/above the
  trend median. This adapts to the sample and needs no hand-set cutoff.
- **Publication lag**: signals are shifted forward ``lag`` months so the regime
  assigned to month ``t`` uses only macro data released by then (point-in-time).

Corroborating series (real-GDP YoY, the 12-month change in unemployment) are kept
on the output for charts and robustness, but the default classification uses the
two headline signals so it stays explainable to an investment committee.
"""

from __future__ import annotations

import pandas as pd

REGIME_ORDER: list[str] = ["Goldilocks", "Overheating", "Stagflation", "Recession"]


def _smooth(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean() if window and window > 1 else s


def _expanding_median(s: pd.Series, min_periods: int) -> pd.Series:
    return s.expanding(min_periods=min_periods).median()


def classify_regimes(
    macro: pd.DataFrame,
    growth_col: str = "indpro_yoy",
    infl_col: str = "cpi_yoy",
    smooth: int = 3,
    threshold: str = "expanding_median",
    ref_window: int = 120,
    min_periods: int = 36,
    lag: int = 1,
) -> pd.DataFrame:
    """Classify each month into a Growth x Inflation regime.

    Parameters
    ----------
    macro:
        Month-end macro panel (see :func:`raa.data.macro.build_macro_monthly`).
    growth_col, infl_col:
        Columns used as the growth and inflation signals.
    smooth:
        Moving-average window (months) applied to each signal before classifying.
    threshold:
        ``"expanding_median"`` (point-in-time, default), ``"rolling_median"``
        (uses ``ref_window``), or ``"full_median"`` (whole-sample, descriptive).
    ref_window:
        Window for the rolling-median threshold.
    min_periods:
        Minimum observations before the expanding threshold is defined.
    lag:
        Months to shift the (already point-in-time) signals to reflect macro
        publication delay. ``lag=1`` => regime at ``t`` uses data through ``t-1``.

    Returns
    -------
    DataFrame indexed by month-end with signal/threshold/flag columns and a
    categorical ``regime`` column ordered per :data:`REGIME_ORDER`.
    """
    if growth_col not in macro or infl_col not in macro:
        raise KeyError(f"macro panel missing {growth_col!r} or {infl_col!r}")

    g = _smooth(macro[growth_col], smooth)
    p = _smooth(macro[infl_col], smooth)

    if threshold == "expanding_median":
        g_ref = _expanding_median(g, min_periods)
        p_ref = _expanding_median(p, min_periods)
    elif threshold == "rolling_median":
        g_ref = g.rolling(ref_window, min_periods=min_periods).median()
        p_ref = p.rolling(ref_window, min_periods=min_periods).median()
    elif threshold == "full_median":
        g_ref = pd.Series(g.median(), index=g.index)
        p_ref = pd.Series(p.median(), index=p.index)
    else:
        raise ValueError(f"unknown threshold method: {threshold!r}")

    out = pd.DataFrame(
        {
            "growth_signal": g,
            "infl_signal": p,
            "growth_ref": g_ref,
            "infl_ref": p_ref,
        }
    )
    # Apply publication lag (point-in-time).
    if lag:
        out = out.shift(lag)

    out["growth_high"] = out["growth_signal"] >= out["growth_ref"]
    out["infl_high"] = out["infl_signal"] >= out["infl_ref"]

    def _label(row: pd.Series) -> str | float:
        if pd.isna(row["growth_ref"]) or pd.isna(row["infl_ref"]):
            return pd.NA
        gh, ih = bool(row["growth_high"]), bool(row["infl_high"])
        if gh and not ih:
            return "Goldilocks"
        if gh and ih:
            return "Overheating"
        if not gh and ih:
            return "Stagflation"
        return "Recession"

    out["regime"] = out.apply(_label, axis=1)
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=False)

    # Carry corroborating series through for charts (unshifted; descriptive only).
    for col in ("gdp_yoy", "unemployment_chg_12m", "slope_10y_3m", "fedfunds"):
        if col in macro:
            out[col] = macro[col]
    return out


def regime_series(macro: pd.DataFrame, **kwargs) -> pd.Series:
    """Convenience: just the categorical ``regime`` series."""
    return classify_regimes(macro, **kwargs)["regime"]


def regime_episodes(regime: pd.Series) -> pd.DataFrame:
    """Collapse a monthly regime series into contiguous episodes.

    Returns one row per episode: ``regime, start, end, months``.
    """
    r = regime.dropna()
    if r.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "months"])
    grp = (r.astype(str) != r.astype(str).shift()).cumsum()
    rows = []
    for _, idx in r.groupby(grp).groups.items():
        idx = pd.DatetimeIndex(idx)
        rows.append(
            {
                "regime": str(r.loc[idx[0]]),
                "start": idx[0],
                "end": idx[-1],
                "months": len(idx),
            }
        )
    return pd.DataFrame(rows)


def transition_matrix(regime: pd.Series, normalize: bool = True) -> pd.DataFrame:
    """Month-to-month regime transition counts (or probabilities)."""
    r = regime.dropna().astype(str)
    nxt = r.shift(-1)
    pairs = pd.DataFrame({"from": r, "to": nxt}).dropna()
    mat = pairs.groupby(["from", "to"]).size().unstack(fill_value=0)
    mat = mat.reindex(index=REGIME_ORDER, columns=REGIME_ORDER, fill_value=0)
    if normalize:
        row_sums = mat.sum(axis=1).replace(0, pd.NA)
        mat = mat.div(row_sums, axis=0).fillna(0.0)
    return mat
