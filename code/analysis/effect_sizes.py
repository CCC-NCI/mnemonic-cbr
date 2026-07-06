"""Cohen's d for pairwise contrasts.

Spec §3.6: "Cohen's d for headline pairwise contrasts."

This module computes Cohen's d (pooled-SD version) for every pair of
levels of a factor, on a given outcome. The manuscript can then cite
the d for whichever pair becomes the headline finding.

Cohen's d (pooled) for two groups with means m_a, m_b and standard
deviations s_a, s_b and sizes n_a, n_b:

    s_pooled = sqrt( ((n_a-1) s_a^2 + (n_b-1) s_b^2) / (n_a + n_b - 2) )
    d        = (m_a - m_b) / s_pooled

Conventional interpretation: |d| ≈ 0.2 small, ≈ 0.5 medium, ≈ 0.8 large.
"""

from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Pooled-SD Cohen's d. Returns NaN if either group has < 2 obs."""
    a = group_a.dropna().to_numpy(dtype=float)
    b = group_b.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def pairwise_cohens_d(
    df: pd.DataFrame,
    factor: str,
    outcome: str,
) -> pd.DataFrame:
    """Compute Cohen's d for every pair of levels of `factor` on `outcome`.

    Returns rows sorted by |d| descending.
    """
    data = df[[factor, outcome]].dropna()
    levels = sorted(data[factor].unique())
    rows = []
    for a, b in combinations(levels, 2):
        d = cohens_d(
            data[data[factor] == a][outcome],
            data[data[factor] == b][outcome],
        )
        rows.append({
            "level_a":  a,
            "level_b":  b,
            "n_a":      int((data[factor] == a).sum()),
            "n_b":      int((data[factor] == b).sum()),
            "mean_a":   float(data[data[factor] == a][outcome].mean()),
            "mean_b":   float(data[data[factor] == b][outcome].mean()),
            "d":        d,
            "abs_d":    abs(d) if pd.notna(d) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_d", ascending=False).reset_index(drop=True)
    return out


def magnitude_label(d: float) -> str:
    """Conventional Cohen's d magnitude label."""
    if pd.isna(d):
        return "n/a"
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"
