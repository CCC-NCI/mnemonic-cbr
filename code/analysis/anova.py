"""Two-way Architecture × Persona ANOVA.

Spec §3.6: "Two-way ANOVA on R5 and on (R1+R2+R3)/3 separately."

Implementation uses scipy.stats f_oneway and a hand-rolled two-way
calculation rather than statsmodels — keeps dependencies minimal.

Returns a small dict per ANOVA with:
  - sum_sq_arch, sum_sq_persona, sum_sq_interaction, sum_sq_residual
  - df_arch, df_persona, df_interaction, df_residual
  - F_arch, F_persona, F_interaction
  - p_arch, p_persona, p_interaction
  - partial_eta_squared_arch / persona / interaction
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def two_way_anova(
    df: pd.DataFrame,
    outcome: str = "R5",
    factor_a: str = "architecture",
    factor_b: str = "persona",
) -> Dict[str, float]:
    """Balanced two-way ANOVA with interaction term.

    Drops rows where the outcome is NaN. Caller is responsible for
    ensuring the design is approximately balanced; for the spec's
    150 cases × 5 personas × 5 archs × 3 folds × 2 legs design, it is.
    """
    data = df[[factor_a, factor_b, outcome]].dropna()
    if data.empty:
        raise ValueError(f"No non-null observations of outcome {outcome!r}")

    grand_mean = data[outcome].mean()
    levels_a = data[factor_a].unique()
    levels_b = data[factor_b].unique()
    a_means = data.groupby(factor_a)[outcome].mean()
    b_means = data.groupby(factor_b)[outcome].mean()
    cell_means = data.groupby([factor_a, factor_b])[outcome].mean()

    n_total = len(data)
    n_a = len(levels_a)
    n_b = len(levels_b)

    # Sums of squares.
    ss_total = float(((data[outcome] - grand_mean) ** 2).sum())

    ss_a = 0.0
    for level in levels_a:
        n_level = (data[factor_a] == level).sum()
        ss_a += n_level * (a_means[level] - grand_mean) ** 2

    ss_b = 0.0
    for level in levels_b:
        n_level = (data[factor_b] == level).sum()
        ss_b += n_level * (b_means[level] - grand_mean) ** 2

    ss_cells = 0.0
    for (la, lb), m in cell_means.items():
        n_cell = ((data[factor_a] == la) & (data[factor_b] == lb)).sum()
        ss_cells += n_cell * (m - grand_mean) ** 2

    ss_interaction = ss_cells - ss_a - ss_b
    ss_residual = ss_total - ss_cells

    # Degrees of freedom.
    df_a = n_a - 1
    df_b = n_b - 1
    df_ab = df_a * df_b
    df_residual = n_total - n_a * n_b

    ms_a = ss_a / df_a if df_a > 0 else np.nan
    ms_b = ss_b / df_b if df_b > 0 else np.nan
    ms_ab = ss_interaction / df_ab if df_ab > 0 else np.nan
    ms_residual = ss_residual / df_residual if df_residual > 0 else np.nan

    f_a = ms_a / ms_residual if ms_residual and not np.isnan(ms_residual) else np.nan
    f_b = ms_b / ms_residual if ms_residual and not np.isnan(ms_residual) else np.nan
    f_ab = ms_ab / ms_residual if ms_residual and not np.isnan(ms_residual) else np.nan

    p_a = 1 - stats.f.cdf(f_a, df_a, df_residual) if not np.isnan(f_a) else np.nan
    p_b = 1 - stats.f.cdf(f_b, df_b, df_residual) if not np.isnan(f_b) else np.nan
    p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_residual) if not np.isnan(f_ab) else np.nan

    eta_a = ss_a / (ss_a + ss_residual) if ss_a + ss_residual > 0 else np.nan
    eta_b = ss_b / (ss_b + ss_residual) if ss_b + ss_residual > 0 else np.nan
    eta_ab = ss_interaction / (ss_interaction + ss_residual) if ss_interaction + ss_residual > 0 else np.nan

    return {
        "outcome": outcome,
        "n": n_total,
        "ss_a": ss_a, "ss_b": ss_b,
        "ss_interaction": ss_interaction, "ss_residual": ss_residual,
        "df_a": df_a, "df_b": df_b,
        "df_interaction": df_ab, "df_residual": df_residual,
        "ms_a": ms_a, "ms_b": ms_b, "ms_interaction": ms_ab, "ms_residual": ms_residual,
        "F_a": float(f_a), "F_b": float(f_b), "F_interaction": float(f_ab),
        "p_a": float(p_a), "p_b": float(p_b), "p_interaction": float(p_ab),
        "eta2p_a": float(eta_a),
        "eta2p_b": float(eta_b),
        "eta2p_interaction": float(eta_ab),
    }


def format_anova_table(anova: Dict[str, float],
                       factor_a_name: str = "Architecture",
                       factor_b_name: str = "Persona") -> pd.DataFrame:
    """Format an ANOVA result dict as a publication-style table."""
    rows = [
        {
            "Source":   factor_a_name,
            "SS":       anova["ss_a"],
            "df":       anova["df_a"],
            "MS":       anova["ms_a"],
            "F":        anova["F_a"],
            "p":        anova["p_a"],
            "η²p":      anova["eta2p_a"],
        },
        {
            "Source":   factor_b_name,
            "SS":       anova["ss_b"],
            "df":       anova["df_b"],
            "MS":       anova["ms_b"],
            "F":        anova["F_b"],
            "p":        anova["p_b"],
            "η²p":      anova["eta2p_b"],
        },
        {
            "Source":   f"{factor_a_name} × {factor_b_name}",
            "SS":       anova["ss_interaction"],
            "df":       anova["df_interaction"],
            "MS":       anova["ms_interaction"],
            "F":        anova["F_interaction"],
            "p":        anova["p_interaction"],
            "η²p":      anova["eta2p_interaction"],
        },
        {
            "Source":   "Residual",
            "SS":       anova["ss_residual"],
            "df":       anova["df_residual"],
            "MS":       anova["ms_residual"],
            "F":        np.nan,
            "p":        np.nan,
            "η²p":      np.nan,
        },
    ]
    return pd.DataFrame(rows)
