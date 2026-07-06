"""Aggregation from SQLite manifest → manuscript-ready dataframes.

Spec §3.4: report R5, (R1+R2+R3)/3, R4, R6 separately.
Spec §3.6: per-cell means, per-persona marginals, per-architecture
marginals. Each marginal averages across the orthogonal factors.

This module returns pandas DataFrames; export.py turns them into
CSV / Markdown / LaTeX.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Outcomes reported separately per spec §3.4.
OUTCOMES = ("R5", "quality_composite", "R4", "R6")

# Pretty labels for the four reporting buckets.
OUTCOME_LABELS = {
    "R5":                "R5 (student trajectory)",
    "quality_composite": "Quality (R1+R2+R3)/3",
    "R4":                "R4 (domain accuracy)",
    "R6":                "R6 (strategy fidelity)",
}

# Short labels for narrow-layout tables (fit on a page).
OUTCOME_SHORT_LABELS = {
    "R5":                "R5",
    "quality_composite": "Quality",
    "R4":                "R4",
    "R6":                "R6",
}


def load_manifest(db_path) -> pd.DataFrame:
    """Read all dialogues + scores from a SQLite manifest into a DataFrame."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Manifest not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT case_id, persona, architecture, student_leg, "
            "R1, R2, R3, R4, R5, R6, quality_composite, "
            "judge_provider, judge_model "
            "FROM dialogues",
            con,
        )
    finally:
        con.close()
    return df


def _mean_sd(series: pd.Series) -> pd.Series:
    """Return ('n', 'mean', 'sd') over a series, ignoring NaN."""
    s = series.dropna()
    return pd.Series(
        {"n": int(len(s)), "mean": float(s.mean()) if len(s) else np.nan,
         "sd":  float(s.std(ddof=1)) if len(s) > 1 else np.nan}
    )


def per_persona_table(df: pd.DataFrame,
                      outcomes: Iterable[str] = OUTCOMES) -> pd.DataFrame:
    """Per-persona marginals: mean across cases × architectures × legs."""
    rows = []
    for persona, sub in df.groupby("persona"):
        row = {"persona": persona}
        for outcome in outcomes:
            stats = _mean_sd(sub[outcome])
            row[f"{outcome}_n"]    = stats["n"]
            row[f"{outcome}_mean"] = stats["mean"]
            row[f"{outcome}_sd"]   = stats["sd"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("R5_mean", ascending=False).reset_index(drop=True)


def per_architecture_table(df: pd.DataFrame,
                           outcomes: Iterable[str] = OUTCOMES) -> pd.DataFrame:
    """Per-architecture marginals: mean across cases × personas × legs."""
    rows = []
    for arch, sub in df.groupby("architecture"):
        row = {"architecture": arch}
        for outcome in outcomes:
            stats = _mean_sd(sub[outcome])
            row[f"{outcome}_n"]    = stats["n"]
            row[f"{outcome}_mean"] = stats["mean"]
            row[f"{outcome}_sd"]   = stats["sd"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("R5_mean", ascending=False).reset_index(drop=True)


def narrow_marginals(
    wide_df: pd.DataFrame,
    index_col: str,
    outcomes: Iterable[str] = OUTCOMES,
    ndigits: int = 2,
) -> pd.DataFrame:
    """Compact 'mean (SD)' layout for the per-persona / per-architecture
    marginal tables.

    Reduces the 13-column wide form (index + n/mean/sd × 4 outcomes) to a
    5-column form: index + one cell per outcome formatted as 'mean (sd)'.
    The n is dropped from the table proper because it is constant across
    rows at this stage; report it once in the table caption.

    Use this for LaTeX export when the wide form would overflow the page.
    The CSV and Markdown exports can continue to use the wide form as the
    source of truth.
    """
    out = pd.DataFrame()
    out[index_col] = wide_df[index_col].astype(str)
    for outcome in outcomes:
        label = OUTCOME_SHORT_LABELS.get(outcome, outcome)
        mean_col = f"{outcome}_mean"
        sd_col = f"{outcome}_sd"

        def _fmt(row, m=mean_col, s=sd_col):
            mv = row[m]
            sv = row[s]
            if pd.isna(mv):
                return "—"
            if pd.isna(sv):
                return f"{mv:.{ndigits}f}"
            return f"{mv:.{ndigits}f} ({sv:.{ndigits}f})"

        out[label] = wide_df.apply(_fmt, axis=1)
    return out


def cell_table(df: pd.DataFrame,
               outcome: str = "R5") -> pd.DataFrame:
    """Per-cell (architecture × persona) means for one outcome.

    Returns a wide table with personas as columns. Useful for the
    Architecture × Persona interaction tables in the manuscript.
    """
    cell = df.groupby(["architecture", "persona"])[outcome].mean().reset_index()
    return cell.pivot(index="architecture", columns="persona", values=outcome)


def cross_student_variance(df: pd.DataFrame,
                           outcome: str = "R5") -> pd.DataFrame:
    """Cross-student variance for one outcome.

    For each (case, persona, architecture) cell, compute |R_a - R_b|
    where R_a is the leg_a score and R_b the leg_b score. Robustness
    check per spec §3.5.

    Returns one row per (persona, architecture) with mean abs diff
    and the within-cell correlation between leg_a and leg_b.
    """
    pivot = df.pivot_table(
        index=["case_id", "persona", "architecture"],
        columns="student_leg",
        values=outcome,
    )
    if "leg_a" not in pivot.columns or "leg_b" not in pivot.columns:
        return pd.DataFrame(columns=["persona", "architecture", "mean_abs_diff", "pearson_r", "n"])
    diff = (pivot["leg_a"] - pivot["leg_b"]).abs()
    diff = diff.dropna()
    out_rows = []
    grouped = diff.groupby(level=["persona", "architecture"])
    for (persona, arch), s in grouped:
        sub = pivot.loc[(slice(None), persona, arch), :].dropna()
        if len(sub) >= 2:
            r = sub["leg_a"].corr(sub["leg_b"])
        else:
            r = np.nan
        out_rows.append(
            {
                "persona": persona,
                "architecture": arch,
                "mean_abs_diff": float(s.mean()),
                "pearson_r": float(r) if pd.notna(r) else np.nan,
                "n": int(len(s)),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["persona", "architecture"]).reset_index(drop=True)
