"""Supplementary data-driven regime detection: HMM and clustering.

These are *robustness checks* on the rule-based core, not the headline model. We
ask: do regimes emerge naturally from the macro data, and do they line up with
the transparent Growth x Inflation quadrants?

Comparison method matters. Detected states are arbitrary integers, so:

- structural agreement with the rule-based model is measured with the
  permutation-invariant **Adjusted Rand Index** on the raw partitions;
- for an interpretable, non-degenerate naming we map states to the four quadrants
  with an **optimal (Hungarian) assignment** that maximises overlap with the
  rule-based labels.

Each state is also described by its mean macro features, which is the real
economic content independent of any naming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from raa.regimes.rule_based import REGIME_ORDER

FEATURES = ["cpi_yoy", "indpro_yoy", "unemployment_chg_12m", "slope_10y_3m", "vix"]


def build_feature_matrix(
    macro: pd.DataFrame, market: pd.DataFrame, features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw, standardised)`` feature frames with NaNs dropped."""
    features = features or FEATURES
    df = macro.join(market[["vix"]], how="left") if "vix" in market else macro.copy()
    X = df[features].dropna()
    Xz = (X - X.mean()) / X.std(ddof=0)
    return X, Xz


def fit_hmm(raw: pd.DataFrame, Xz: pd.DataFrame, n_states: int = 4, seed: int = 42) -> dict:
    """Fit a Gaussian HMM; return raw states, transition matrix, state profiles."""
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states, covariance_type="full", n_iter=400, random_state=seed
    )
    model.fit(Xz.to_numpy())
    states = model.predict(Xz.to_numpy())
    names = [f"S{i}" for i in range(n_states)]
    transmat = pd.DataFrame(model.transmat_, index=names, columns=names)
    persistence = pd.Series(np.diag(model.transmat_), index=names)
    expected_duration = pd.Series(1.0 / (1.0 - np.diag(model.transmat_)), index=names)
    states_s = pd.Series(states, index=Xz.index, name="hmm_state")
    profiles = raw.assign(_s=states).groupby("_s").mean()
    profiles.index = [f"S{i}" for i in profiles.index]
    return {
        "states": states_s,
        "transmat": transmat,
        "persistence": persistence,
        "expected_duration_months": expected_duration,
        "loglik": float(model.score(Xz.to_numpy())),
        "profiles": profiles,
    }


def fit_clustering(
    raw: pd.DataFrame, Xz: pd.DataFrame, method: str = "kmeans", k: int = 4, seed: int = 42
) -> pd.Series:
    """Cluster the standardised features; return raw integer cluster labels."""
    X = Xz.to_numpy()
    if method == "kmeans":
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    elif method == "gmm":
        labels = GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit_predict(X)
    elif method == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    else:
        raise ValueError(f"unknown clustering method {method!r}")
    return pd.Series(labels, index=Xz.index, name=method)


def optimal_assignment(states: pd.Series, reference: pd.Series) -> tuple[pd.Series, dict]:
    """Map raw states to quadrant names maximising overlap with ``reference``."""
    df = pd.concat([states.rename("s"), reference.astype(str).rename("r")], axis=1).dropna()
    cont = pd.crosstab(df["s"], df["r"]).reindex(columns=REGIME_ORDER, fill_value=0)
    row_ind, col_ind = linear_sum_assignment(-cont.to_numpy())
    mapping = {cont.index[r]: cont.columns[c] for r, c in zip(row_ind, col_ind)}
    for s in cont.index:  # states beyond the 4 quadrants -> their argmax overlap
        if s not in mapping:
            mapping[s] = cont.columns[int(cont.loc[s].to_numpy().argmax())]
    named = states.map(mapping).astype(pd.CategoricalDtype(categories=REGIME_ORDER))
    return named, mapping


def agreement(reference: pd.Series, states: pd.Series) -> dict[str, float]:
    """ARI (permutation-invariant) + best-assignment match rate vs ``reference``."""
    df = pd.concat([reference.astype(str).rename("r"), states.rename("s")], axis=1).dropna()
    if df.empty:
        return {"adjusted_rand": np.nan, "match_rate": np.nan, "n": 0}
    ari = float(adjusted_rand_score(df["r"].astype("category").cat.codes, df["s"]))
    _, mapping = optimal_assignment(df["s"], df["r"])
    matched = df["s"].map(mapping).astype(str)
    match = float((matched == df["r"]).mean())
    return {"adjusted_rand": ari, "match_rate": match, "n": int(len(df))}
