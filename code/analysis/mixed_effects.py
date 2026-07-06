"""
mixed_effects.py

Two-way Architecture x Persona ANOVA refit as a mixed-effects model with
case_id and student_leg as crossed random effects.

Motivation
----------
The manuscript's original two-way ANOVAs treated 1,250 dialogues as
independent observations (df_residual = 1223).  Dialogues are nested in
25 cases and crossed with 2 student legs.  Case-level clustering will
deflate p-values; the reviewer of v5 named this as the single most
likely quantitative referee demand.

Model
-----
For each outcome y (R5, R1, R2, R3, R4, R6):

    y_ijkm = mu
             + alpha_i (architecture, fixed, 5 levels)
             + beta_j  (persona, fixed, 5 levels)
             + (alpha*beta)_ij (fixed interaction)
             + u_k (case, random, ~N(0, sigma^2_case))
             + v_m (student leg, random, ~N(0, sigma^2_leg))
             + epsilon_ijkm  (~N(0, sigma^2_resid))

Fit with `statsmodels.formula.api.mixedlm` using crossed variance
components on a dummy grouping variable, then run type-III Wald tests
on the fixed-effect coefficient blocks corresponding to architecture,
persona, and their interaction.

Because a Wald chi-square with k numerator df is a scaled F with
denominator df = infinity, we report both the chi-square/df figure and
the equivalent F approximation (F = chi2 / df, denom df = inf), along
with the variance components as an intraclass proxy.

Author: Dietmar Janetzko, NCI Cloud Competency Centre
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.formula.api as smf


OUTCOMES = ("R5", "R2", "R1", "R3", "R4", "R6")


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load the phase-B/D manifest into a data frame keyed on cell attributes."""
    with sqlite3.connect(str(manifest_path)) as conn:
        df = pd.read_sql_query(
            "SELECT case_id, persona, architecture, student_leg, "
            "       R1, R2, R3, R4, R5, R6 FROM dialogues",
            conn,
        )
    return df


# --------------------------------------------------------------------------- #
# Wald-test helper for a factor block
# --------------------------------------------------------------------------- #

def _factor_wald(res, prefix: str) -> tuple[float, float, int]:
    """
    Joint Wald test on all coefficients whose parameter names start with
    ``prefix`` AND do not contain ':' (i.e. are pure main-effect terms,
    not interaction terms).  Returns (chi2, p_value, df).
    """
    idx = [i for i, name in enumerate(res.params.index)
           if name.startswith(prefix) and ":" not in name]
    if not idx:
        return float("nan"), float("nan"), 0
    contrasts = np.zeros((len(idx), len(res.params)))
    for row, ii in enumerate(idx):
        contrasts[row, ii] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wt = res.wald_test(contrasts, use_f=False)
    chi2 = float(np.asarray(wt.statistic).squeeze())
    p    = float(np.asarray(wt.pvalue).squeeze())
    df   = int(len(idx))
    return chi2, p, df


def _variance_components(res) -> dict[str, float]:
    scale = float(res.scale)
    out = {"sigma2_resid": scale}
    try:
        for i, name in enumerate(res.model.exog_vc.names):
            out[f"sigma2_{name}"] = float(res.vcomp[i]) * scale
    except AttributeError:
        # older statsmodels API fallback
        for i, name in enumerate(("case", "leg")):
            out[f"sigma2_{name}"] = float(res.vcomp[i]) * scale
    return out


# --------------------------------------------------------------------------- #
# Model fit
# --------------------------------------------------------------------------- #

def fit_mixed_two_way(
    df: pd.DataFrame,
    outcome: str,
    case_col: str = "case_id",
    leg_col:  str = "student_leg",
    arch_col: str = "architecture",
    persona_col: str = "persona",
) -> dict[str, Any]:
    """
    Fit y ~ arch * persona + (1|case) + (1|leg) and run Wald tests on the
    fixed-effect blocks.
    """
    data = df[[case_col, leg_col, arch_col, persona_col, outcome]].dropna().copy()
    data[outcome] = data[outcome].astype(float)
    data["dummy"] = 1

    formula = f"{outcome} ~ C({arch_col}) * C({persona_col})"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula, data, groups=data["dummy"],
            vc_formula={
                "case": f"0 + C({case_col})",
                "leg":  f"0 + C({leg_col})",
            },
            re_formula="0",
        )
        res = model.fit(reml=True, method="lbfgs")

    # Wald tests on the three fixed-effect blocks.
    chi2_a, p_a, df_a = _factor_wald(res, f"C({arch_col})")
    chi2_b, p_b, df_b = _factor_wald(res, f"C({persona_col})")
    # Interaction terms are named C(arch)[T.x]:C(persona)[T.y]; they contain a colon.
    idx_int = [
        i for i, n in enumerate(res.params.index)
        if ":" in n and n.startswith(f"C({arch_col})")
    ]
    if idx_int:
        contrasts = np.zeros((len(idx_int), len(res.params)))
        for r, ii in enumerate(idx_int):
            contrasts[r, ii] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wt = res.wald_test(contrasts, use_f=False)
        chi2_ab = float(np.asarray(wt.statistic).squeeze())
        p_ab    = float(np.asarray(wt.pvalue).squeeze())
        df_ab   = int(len(idx_int))
    else:
        chi2_ab, p_ab, df_ab = float("nan"), float("nan"), 0

    vc = _variance_components(res)
    total_var = sum(vc.values())
    icc_case = vc.get("sigma2_case", 0.0) / total_var if total_var else float("nan")
    icc_leg  = vc.get("sigma2_leg",  0.0) / total_var if total_var else float("nan")

    # F approximation (Wald chi2 / df, denom df = infinity).
    def _as_F(chi2: float, df: int) -> float:
        if df <= 0 or np.isnan(chi2):
            return float("nan")
        return chi2 / df

    return {
        "outcome":     outcome,
        "n_dialogues": int(len(data)),
        "n_cases":     int(data[case_col].nunique()),
        "n_legs":      int(data[leg_col].nunique()),
        "arch": {
            "chi2": chi2_a, "df": df_a, "p": p_a, "F_approx": _as_F(chi2_a, df_a),
        },
        "persona": {
            "chi2": chi2_b, "df": df_b, "p": p_b, "F_approx": _as_F(chi2_b, df_b),
        },
        "interaction": {
            "chi2": chi2_ab, "df": df_ab, "p": p_ab, "F_approx": _as_F(chi2_ab, df_ab),
        },
        "variance_components": vc,
        "icc_case": float(icc_case),
        "icc_leg":  float(icc_leg),
        "converged": bool(getattr(res, "converged", True)),
        "log_likelihood": float(res.llf) if hasattr(res, "llf") else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Manuscript-ready table formatter
# --------------------------------------------------------------------------- #

def format_mixed_table(result: dict[str, Any]) -> pd.DataFrame:
    """One-row-per-effect summary for LaTeX/CSV export."""
    rows = []
    for label, key in (("Architecture", "arch"),
                        ("Persona",      "persona"),
                        ("Architecture x Persona", "interaction")):
        e = result[key]
        rows.append({
            "Source": label,
            "chi2":   e["chi2"],
            "df":     e["df"],
            "F_approx": e["F_approx"],
            "p":      e["p"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli():
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir",  type=Path, required=True)
    p.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
    return p.parse_args()


def main() -> None:
    args = _cli()
    df = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for outcome in args.outcomes:
        print(f"[mixed_effects] fitting {outcome} ...", flush=True)
        res = fit_mixed_two_way(df, outcome=outcome)
        summary[outcome] = res
        table = format_mixed_table(res)
        table.to_csv(args.out_dir / f"mixed_{outcome}.csv", index=False)
        print(table.to_string(index=False))
        print(f"  variance components: {res['variance_components']}")
        print(f"  ICC(case) = {res['icc_case']:.3f}   ICC(leg) = {res['icc_leg']:.3f}")
        print()
    import json
    with open(args.out_dir / "mixed_effects_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[mixed_effects] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
