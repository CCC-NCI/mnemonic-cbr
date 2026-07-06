"""
human_validation.py

End-to-end analysis pipeline for the Prolific human-rater validation study.

Reads:
  validation-study/data/Responses.csv                    (per-event rating log)
  validation-study/sampling_frame/sampling_frame_manifest.json
  validation-study/sentinels/sentinel_clear_*.json       (for expected bands)
  results/phaseB_smoke_5turn/manifest.sqlite             (LLM scores for the 1,250 dialogues)

Writes:
  results/human_validation/participants_summary.csv      (one row per rater)
  results/human_validation/dialogue_ratings.csv          (rater x dialogue long form)
  results/human_validation/dialogue_means.csv            (per-dialogue human mean vs LLM)
  results/human_validation/stage1_pool_reliability.json
  results/human_validation/stage2_llm_vs_human.json
  results/human_validation/stage3_rank_preservation.json
  results/human_validation/REPORT.md                     (manuscript-ready summary)

Stages:
  Stage 1  Pool reliability                              ICC(2,1) + ICC(2,k) on
                                                        the human rater x dialogue
                                                        long table via two-way
                                                        random-effects variance
                                                        components (statsmodels
                                                        mixedlm).  Sparse-safe.
  Stage 2  LLM-to-human agreement                        Per-dialogue human mean vs
                                                        LLM single rating, Pearson +
                                                        Spearman + ICC(2,1).  Reports
                                                        both primary outcome R5 and
                                                        secondary anchored R2.
  Stage 3  Architecture rank preservation                Aggregate mean R5 by
                                                        architecture from humans vs
                                                        LLM.  Report ranks and Spearman.

Exclusion rules (pre-registered, OSF §3.5):
  strict     'more than one sentinel item outside its pass band'
  sensitivity relax clear_flat R5 pass band from {1,2} to {1,2,3} to reflect
             empirically observed lay-rater reading of the sentinel

Both exclusion regimes are reported side-by-side.  The strict version is the
pre-registered analysis; the sensitivity version is a pre-declared robustness check.

Author: Dietmar Janetzko, NCI Cloud Competency Centre
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

REPO_ROOT   = Path(__file__).resolve().parents[3]  # up to IJAIED3/
VS_ROOT     = REPO_ROOT / "validation-study"
MB_ROOT     = REPO_ROOT / "mnemonic-cbr"
RESULTS_DIR = MB_ROOT / "results" / "human_validation"

DEFAULT_RESPONSES = VS_ROOT / "data" / "Responses.csv"
DEFAULT_FRAME     = VS_ROOT / "sampling_frame" / "sampling_frame_manifest.json"
DEFAULT_LLM_DB    = MB_ROOT / "results" / "phaseB_smoke_5turn" / "manifest.sqlite"

# Pre-Prolific pilot / test sessions the researcher confirmed to exclude.
# Everything up to and including 'Prolific-Test' (JDYSDN7X) is a test.
TEST_PARTICIPANTS: set[str] = {
    "5JQ98RLE",  # "Test two"
    "QWZT3VVE",  # "Test three"
    "UXVYZJNP",  # "FULL TEST 1"
    "MLHNDY69",  # "TEST THREE"
    "FNRMNFN6",  # "TEAST TEST"
    "JDYSDN7X",  # "Prolific-Test"
}

# Sentinel pass bands.  Strict is pre-registered (OSF §3.5); sensitivity relaxes
# clear_flat R5 to include 3, reflecting the surface-progress reading humans
# take of the S2 line "it's a triangular based pyramid".
SENTINEL_BANDS_STRICT: dict[str, dict[str, set[int]]] = {
    "sentinel_clear_improvement": {"r5": {4, 5}, "r2": {1, 2, 3}},
    "sentinel_clear_flat":        {"r5": {1, 2}, "r2": {2, 3, 4}},
}
SENTINEL_BANDS_SENS: dict[str, dict[str, set[int]]] = {
    "sentinel_clear_improvement": {"r5": {4, 5}, "r2": {1, 2, 3}},
    "sentinel_clear_flat":        {"r5": {1, 2, 3}, "r2": {2, 3, 4}},
}

# Practice trial expected scores (see instrument/index.html).
PRACTICE_EXPECTED: dict[str, dict[str, int]] = {
    "practice_mid":     {"r5": 3, "r2": 3},
    "practice_r2_rule": {"r5": 4, "r2": 2},
}
# Practice deviation exclusion: fail = |observed - expected| > 2 on either axis
# in BOTH practice trials.  Pre-registered in OSF §3.5.
PRACTICE_DEV_TOL = 2

# Bayesian decision bands for Stage 2 point estimates.
DECISION_BANDS: list[tuple[str, float, float]] = [
    ("strong",     0.75, 1.01),
    ("adequate",   0.60, 0.75),
    ("partial",    0.40, 0.60),
    ("divergence", -1.01, 0.40),
]

ARCHITECTURES: list[str] = [
    "baseline", "pure_ai", "pure_cbr_tpl", "pure_cbr_llm", "hybrid",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _label_for(x: float) -> str:
    for name, lo, hi in DECISION_BANDS:
        if lo <= x < hi:
            return name
    return "divergence"


def _to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _bootstrap_ci(
    values: np.ndarray,
    stat: callable,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI over a 1-D array of paired observations (rows)."""
    if len(values) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx_pool = np.arange(len(values))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        try:
            boot[b] = stat(values[idx])
        except Exception:
            boot[b] = np.nan
    boot = boot[~np.isnan(boot)]
    if boot.size < 100:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


# --------------------------------------------------------------------------- #
# Data ingestion
# --------------------------------------------------------------------------- #

@dataclass
class Loaded:
    events:          pd.DataFrame   # all rating events, test participants removed
    practice:        pd.DataFrame
    dialogue_ratings: pd.DataFrame  # frame + sentinel ratings, one row per event
    session_ends:    pd.DataFrame
    frame_items:     pd.DataFrame   # sampling frame with llm_R5, llm_R2, arch, ...
    llm_scores:      pd.DataFrame   # all 1,250 phase-B dialogues (arch aggregation)


def load_all(
    responses_path: Path = DEFAULT_RESPONSES,
    frame_path:     Path = DEFAULT_FRAME,
    llm_db_path:    Path = DEFAULT_LLM_DB,
) -> Loaded:
    """Load Prolific event log + sampling frame + LLM scores."""
    events = pd.read_csv(responses_path, dtype=str, keep_default_na=False)
    events = events[~events["participant_code"].isin(TEST_PARTICIPANTS)].copy()

    # Coerce numeric columns for downstream analysis.
    for col in ("r5", "r2", "expected_r5", "expected_r2",
                "sequence_position", "rating_time_seconds"):
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")

    practice = events[events["event_type"] == "practice_rating"].copy()
    dialogue = events[events["event_type"] == "dialogue_rating"].copy()
    sessions = events[events["event_type"] == "session_end"].copy()

    # Sampling frame → dataframe.
    with open(frame_path) as fh:
        manifest = json.load(fh)
    frame_items = pd.DataFrame(manifest["items"])
    # Convenience join key: blind_id used as dialogue_id in the survey.
    frame_items["dialogue_key"] = frame_items["blind_id"]

    # LLM scores for architecture-level aggregation (Stage 3).
    with sqlite3.connect(llm_db_path) as conn:
        llm = pd.read_sql_query(
            "SELECT case_id, persona, architecture, student_leg, "
            "       R2, R5 FROM dialogues",
            conn,
        )
    llm = llm.rename(columns={"R2": "llm_R2", "R5": "llm_R5"})

    return Loaded(
        events=events,
        practice=practice,
        dialogue_ratings=dialogue,
        session_ends=sessions,
        frame_items=frame_items,
        llm_scores=llm,
    )


# --------------------------------------------------------------------------- #
# Exclusion rules
# --------------------------------------------------------------------------- #

@dataclass
class ParticipantSummary:
    code:                    str
    n_practice:              int
    n_dialogue_frame:        int
    n_dialogue_sentinel:     int
    completed:               bool
    practice_fail:           bool
    strict_sentinel_fails:   int
    sens_sentinel_fails:     int
    excluded_strict:         bool
    excluded_sens:           bool
    sentinel_details:        dict[str, Any] = field(default_factory=dict)


def _sentinel_fails(
    dialogue_ratings: pd.DataFrame,
    pcode: str,
    bands: dict[str, dict[str, set[int]]],
) -> tuple[int, dict[str, Any]]:
    """Count sentinel-item fails for one participant under given bands."""
    n_fails = 0
    detail: dict[str, Any] = {}
    for sentinel_id, band in bands.items():
        row = dialogue_ratings[
            (dialogue_ratings["participant_code"] == pcode)
            & (dialogue_ratings["dialogue_id"] == sentinel_id)
        ]
        if row.empty:
            detail[sentinel_id] = {"seen": False}
            continue
        r5 = _to_int(row.iloc[0]["r5"])
        r2 = _to_int(row.iloc[0]["r2"])
        r5_fail = r5 is not None and r5 not in band["r5"]
        r2_fail = r2 is not None and r2 not in band["r2"]
        n_fails += int(r5_fail) + int(r2_fail)
        detail[sentinel_id] = {
            "seen": True, "r5": r5, "r2": r2,
            "r5_fail": r5_fail, "r2_fail": r2_fail,
        }
    return n_fails, detail


def _practice_fail(practice: pd.DataFrame, pcode: str) -> bool:
    """Pre-registered practice-deviation rule: |obs - exp| > tol on either axis
    in BOTH trials → fail."""
    rows = practice[practice["participant_code"] == pcode]
    if rows.empty:
        return False  # no practice data — treat as inclusion (they never got that far)
    fails_per_trial = 0
    for trial_id, exp in PRACTICE_EXPECTED.items():
        r = rows[rows["practice_id"] == trial_id]
        if r.empty:
            continue
        r5 = _to_int(r.iloc[0]["r5"])
        r2 = _to_int(r.iloc[0]["r2"])
        if r5 is None and r2 is None:
            continue
        dev_r5 = abs(r5 - exp["r5"]) if r5 is not None else 0
        dev_r2 = abs(r2 - exp["r2"]) if r2 is not None else 0
        if dev_r5 > PRACTICE_DEV_TOL or dev_r2 > PRACTICE_DEV_TOL:
            fails_per_trial += 1
    return fails_per_trial >= 2


def build_participant_summary(loaded: Loaded) -> pd.DataFrame:
    frame_ids = set(loaded.frame_items["blind_id"])
    sentinel_ids = set(SENTINEL_BANDS_STRICT.keys())
    completers = set(loaded.session_ends["participant_code"])

    # Union of every real (non-test) participant code that appears anywhere
    # in the event log, so drop-outs who cleared practice but never rated a
    # dialogue still get a row.
    all_codes = set(loaded.events["participant_code"])
    grouped = dict(list(loaded.dialogue_ratings.groupby("participant_code")))

    rows: list[dict[str, Any]] = []
    for pcode in sorted(all_codes):
        evts = grouped.get(pcode, loaded.dialogue_ratings.iloc[0:0])
        n_frame = int(evts["dialogue_id"].isin(frame_ids).sum()) if len(evts) else 0
        n_sent  = int(evts["dialogue_id"].isin(sentinel_ids).sum()) if len(evts) else 0
        practice_n = int((loaded.practice["participant_code"] == pcode).sum())
        strict_fails, strict_det = _sentinel_fails(
            loaded.dialogue_ratings, pcode, SENTINEL_BANDS_STRICT)
        sens_fails, sens_det = _sentinel_fails(
            loaded.dialogue_ratings, pcode, SENTINEL_BANDS_SENS)
        pfail = _practice_fail(loaded.practice, pcode)
        summary = ParticipantSummary(
            code=pcode,
            n_practice=practice_n,
            n_dialogue_frame=n_frame,
            n_dialogue_sentinel=n_sent,
            completed=pcode in completers,
            practice_fail=pfail,
            strict_sentinel_fails=strict_fails,
            sens_sentinel_fails=sens_fails,
            excluded_strict=(strict_fails > 1) or pfail,
            excluded_sens=(sens_fails > 1) or pfail,
            sentinel_details={"strict": strict_det, "sensitivity": sens_det},
        )
        rows.append(vars(summary))
    return pd.DataFrame(rows).sort_values("code").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 1: Pool reliability (ICC(2,1) and ICC(2,k))
# --------------------------------------------------------------------------- #

def _stage1_icc_variance(
    df_long: pd.DataFrame,
    outcome: str,
) -> dict[str, float]:
    """Two-way random-effects ICC on the sparse rater x dialogue long table.

    Fits    y = mu + rater + dialogue + eps
    with    rater ~ N(0, sigma^2_r), dialogue ~ N(0, sigma^2_t), eps ~ N(0, sigma^2_e)
    using   statsmodels MixedLM with crossed variance components.

    ICC(2,1) = sigma^2_t / (sigma^2_t + sigma^2_r + sigma^2_e)
    ICC(2,k) = k*sigma^2_t / (k*sigma^2_t + sigma^2_r + sigma^2_e)      (k = mean raters/target)
    """
    import statsmodels.formula.api as smf  # local import: heavy dep

    d = df_long[["participant_code", "dialogue_key", outcome]].dropna().copy()
    d[outcome] = d[outcome].astype(float)
    if len(d) < 20:
        return {"n": len(d), "icc_2_1": float("nan"), "icc_2_k": float("nan"),
                "sigma2_target": float("nan"), "sigma2_rater": float("nan"),
                "sigma2_error": float("nan"), "k_mean": float("nan")}
    d["dummy"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(
            f"{outcome} ~ 1", d, groups=d["dummy"],
            vc_formula={
                "rater":  "0 + C(participant_code)",
                "target": "0 + C(dialogue_key)",
            },
            re_formula="0",
        )
        res = md.fit(reml=True, method="lbfgs")

    scale = float(res.scale)
    # vcomp entries are variance components expressed on the same variance scale
    # as residual (statsmodels normalises them by scale, so multiply back).
    try:
        vc = {name: float(res.vcomp[i]) * scale
              for i, name in enumerate(res.model.exog_vc.names)}
    except AttributeError:
        vc = {name: float(res.vcomp[i]) * scale for i, name in enumerate(["rater", "target"])}

    s2_r = vc.get("rater", 0.0)
    s2_t = vc.get("target", 0.0)
    s2_e = scale
    denom = s2_t + s2_r + s2_e
    icc_2_1 = s2_t / denom if denom > 0 else float("nan")
    # k = mean # of raters per target actually observed
    k_mean = d.groupby("dialogue_key")["participant_code"].nunique().mean()
    denom_k = k_mean * s2_t + s2_r + s2_e
    icc_2_k = (k_mean * s2_t) / denom_k if denom_k > 0 else float("nan")

    return {
        "n_ratings":     int(len(d)),
        "n_raters":      int(d["participant_code"].nunique()),
        "n_targets":     int(d["dialogue_key"].nunique()),
        "k_mean":        float(k_mean),
        "sigma2_target": float(s2_t),
        "sigma2_rater":  float(s2_r),
        "sigma2_error":  float(s2_e),
        "icc_2_1":       float(icc_2_1),
        "icc_2_k":       float(icc_2_k),
    }


def stage1_pool_reliability(
    dialogue_ratings: pd.DataFrame,
    frame_items:      pd.DataFrame,
    included_codes:   set[str],
) -> dict[str, Any]:
    """Compute pool reliability on the frame ratings from included participants."""
    frame_ids = set(frame_items["blind_id"])
    df = dialogue_ratings[
        dialogue_ratings["participant_code"].isin(included_codes)
        & dialogue_ratings["dialogue_id"].isin(frame_ids)
    ].rename(columns={"dialogue_id": "dialogue_key"}).copy()
    out: dict[str, Any] = {}
    for outcome in ("r5", "r2"):
        out[outcome] = _stage1_icc_variance(df, outcome)
    out["_notes"] = (
        "Two-way random-effects ICC via statsmodels MixedLM variance components. "
        "Sparse rater×dialogue design; k_mean = observed raters per dialogue."
    )
    return out


# --------------------------------------------------------------------------- #
# Stage 2: LLM-to-human agreement
# --------------------------------------------------------------------------- #

def _icc_2_1_between(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1) between two paired series, two-way random single measure.

    Treat each row as a target; the two columns (human_mean, llm) as two 'raters'.
    """
    if len(x) < 3:
        return float("nan")
    df = pd.DataFrame({"target": np.arange(len(x)), "R1": x, "R2": y})
    long = df.melt(id_vars="target", var_name="rater", value_name="score")
    grand = long["score"].mean()
    n = len(x)
    k = 2
    MSR = (k * ((df[["R1", "R2"]].mean(axis=1) - grand) ** 2).sum()) / (n - 1)
    MSC = (n * ((df[["R1", "R2"]].mean(axis=0) - grand) ** 2).sum()) / (k - 1)
    # residual mean square
    rows_mean = df[["R1", "R2"]].mean(axis=1)
    cols_mean = df[["R1", "R2"]].mean(axis=0)
    ss_e = 0.0
    for i in range(n):
        for j, col in enumerate(("R1", "R2")):
            ss_e += (df[col].iloc[i] - rows_mean.iloc[i] - cols_mean.iloc[j] + grand) ** 2
    MSE = ss_e / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if denom <= 0:
        return float("nan")
    return float((MSR - MSE) / denom)


def stage2_llm_vs_human(
    dialogue_ratings: pd.DataFrame,
    frame_items:      pd.DataFrame,
    included_codes:   set[str],
    min_raters:       int = 2,
) -> dict[str, Any]:
    """Per-dialogue human mean vs LLM single rating on R5 (primary) and R2."""
    frame_ids = set(frame_items["blind_id"])
    df = dialogue_ratings[
        dialogue_ratings["participant_code"].isin(included_codes)
        & dialogue_ratings["dialogue_id"].isin(frame_ids)
    ].copy()

    # Aggregate per dialogue.
    agg = (
        df.groupby("dialogue_id")
          .agg(n_raters=("participant_code", "nunique"),
               human_r5=("r5", "mean"),
               human_r2=("r2", "mean"))
          .reset_index()
          .rename(columns={"dialogue_id": "blind_id"})
    )
    agg = agg[agg["n_raters"] >= min_raters]
    merged = agg.merge(
        frame_items[["blind_id", "llm_R5", "llm_R2", "architecture",
                     "case_id", "persona", "student_leg"]],
        on="blind_id", how="inner",
    )

    out: dict[str, Any] = {"n_dialogues_used": int(len(merged)),
                            "min_raters":      min_raters}
    for outcome in ("r5", "r2"):
        h = merged[f"human_{outcome}"].to_numpy(dtype=float)
        m = merged[f"llm_R{outcome[-1]}"].to_numpy(dtype=float)
        if len(h) < 3 or np.std(h) == 0 or np.std(m) == 0:
            out[outcome] = {"n": len(h), "insufficient_data": True}
            continue
        pearson_r, pearson_p = stats.pearsonr(h, m)
        spearman_r, spearman_p = stats.spearmanr(h, m)
        icc = _icc_2_1_between(h, m)
        # Bootstrap CIs on the three statistics.
        paired = np.column_stack([h, m])
        ci_r  = _bootstrap_ci(paired, lambda a: stats.pearsonr(a[:, 0], a[:, 1])[0])
        ci_rho = _bootstrap_ci(paired, lambda a: stats.spearmanr(a[:, 0], a[:, 1])[0])
        ci_icc = _bootstrap_ci(paired, lambda a: _icc_2_1_between(a[:, 0], a[:, 1]))
        out[outcome] = {
            "n":            int(len(h)),
            "pearson_r":    float(pearson_r),
            "pearson_p":    float(pearson_p),
            "pearson_ci":   [float(ci_r[0]),  float(ci_r[1])],
            "spearman_rho": float(spearman_r),
            "spearman_p":   float(spearman_p),
            "spearman_ci":  [float(ci_rho[0]), float(ci_rho[1])],
            "icc_2_1":      float(icc),
            "icc_2_1_ci":   [float(ci_icc[0]), float(ci_icc[1])],
            "human_mean":   float(np.mean(h)),
            "llm_mean":     float(np.mean(m)),
            "mean_abs_gap": float(np.mean(np.abs(h - m))),
            "band":         _label_for(pearson_r),
        }
    out["_dialogue_means"] = merged.to_dict(orient="records")
    return out


# --------------------------------------------------------------------------- #
# Stage 3: Architecture rank preservation
# --------------------------------------------------------------------------- #

def stage3_rank_preservation(
    dialogue_ratings: pd.DataFrame,
    frame_items:      pd.DataFrame,
    llm_scores:       pd.DataFrame,
    included_codes:   set[str],
) -> dict[str, Any]:
    """Compare architecture ordering under human vs LLM scoring."""
    frame_ids = set(frame_items["blind_id"])
    df = dialogue_ratings[
        dialogue_ratings["participant_code"].isin(included_codes)
        & dialogue_ratings["dialogue_id"].isin(frame_ids)
    ].copy()
    # Join to arch via frame manifest.
    df = df.merge(
        frame_items[["blind_id", "architecture"]],
        left_on="dialogue_id", right_on="blind_id", how="left",
    )
    human_arch = (
        df.groupby("architecture")["r5"]
          .agg(["mean", "count", "std"])
          .rename(columns={"mean": "human_r5_mean",
                           "count": "human_n",
                           "std":  "human_r5_sd"})
    )
    llm_arch = (
        llm_scores.groupby("architecture")["llm_R5"]
                  .agg(["mean", "count", "std"])
                  .rename(columns={"mean": "llm_r5_mean",
                                   "count": "llm_n",
                                   "std":  "llm_r5_sd"})
    )
    joined = human_arch.join(llm_arch, how="outer").reset_index()
    joined["human_rank"] = joined["human_r5_mean"].rank(ascending=False, method="min")
    joined["llm_rank"]   = joined["llm_r5_mean"].rank(ascending=False, method="min")
    # Sub-frame with both means for correlation.
    both = joined.dropna(subset=["human_r5_mean", "llm_r5_mean"])
    if len(both) >= 3:
        rho, p = stats.spearmanr(both["human_r5_mean"], both["llm_r5_mean"])
        tau, tau_p = stats.kendalltau(both["human_rank"], both["llm_rank"])
    else:
        rho, p, tau, tau_p = (float("nan"),) * 4
    return {
        "per_architecture": joined.round(3).to_dict(orient="records"),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "kendall_tau": float(tau), "kendall_p":  float(tau_p),
        "n_architectures": int(len(both)),
    }


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def _fmt_ci(low: float, high: float) -> str:
    if any(np.isnan([low, high])):
        return "n/a"
    return f"[{low:.2f}, {high:.2f}]"


def build_report(
    participants: pd.DataFrame,
    stage1_strict: dict[str, Any], stage2_strict: dict[str, Any], stage3_strict: dict[str, Any],
    stage1_sens:   dict[str, Any], stage2_sens:   dict[str, Any], stage3_sens:   dict[str, Any],
) -> str:
    n_total     = len(participants)
    n_complete  = int(participants["completed"].sum())
    n_incl_str  = int(((~participants["excluded_strict"]) & participants["completed"]).sum())
    n_incl_sens = int(((~participants["excluded_sens"])   & participants["completed"]).sum())

    lines: list[str] = []
    lines.append("# Human validation study — analysis report")
    lines.append("")
    lines.append(f"- Prolific participants delivered (post-test filter): **{n_total}**")
    lines.append(f"- Sessions reaching `session_end`: **{n_complete}**")
    lines.append(f"- Included after **strict** exclusion (pre-registered): **{n_incl_str}**")
    lines.append(f"- Included after **sensitivity** exclusion (relaxed clear_flat R5 band): **{n_incl_sens}**")
    lines.append("")
    lines.append("## Stage 1 — Pool reliability (ICC on human rater × dialogue)")
    lines.append("")
    for label, s1 in (("strict", stage1_strict), ("sensitivity", stage1_sens)):
        for outcome in ("r5", "r2"):
            m = s1[outcome]
            lines.append(
                f"- **{label}** / {outcome.upper()}: "
                f"ICC(2,1) = {m['icc_2_1']:.2f}, ICC(2,k̄={m['k_mean']:.1f}) = {m['icc_2_k']:.2f}   "
                f"(n_ratings={m['n_ratings']}, n_raters={m['n_raters']}, n_targets={m['n_targets']})"
            )
        lines.append("")
    lines.append("## Stage 2 — LLM-to-human agreement (per-dialogue)")
    lines.append("")
    for label, s2 in (("strict", stage2_strict), ("sensitivity", stage2_sens)):
        for outcome in ("r5", "r2"):
            m = s2.get(outcome, {})
            if m.get("insufficient_data"):
                lines.append(f"- **{label}** / {outcome.upper()}: insufficient data.")
                continue
            lines.append(
                f"- **{label}** / {outcome.upper()}: "
                f"Pearson r = {m['pearson_r']:.2f} {_fmt_ci(*m['pearson_ci'])}, "
                f"Spearman ρ = {m['spearman_rho']:.2f} {_fmt_ci(*m['spearman_ci'])}, "
                f"ICC(2,1) = {m['icc_2_1']:.2f} {_fmt_ci(*m['icc_2_1_ci'])}   "
                f"(n_dialogues={m['n']}, mean human = {m['human_mean']:.2f}, "
                f"mean LLM = {m['llm_mean']:.2f}, "
                f"mean |Δ| = {m['mean_abs_gap']:.2f}; band: **{m['band']}**)"
            )
        lines.append("")
    lines.append("## Stage 3 — Architecture rank preservation (R5)")
    lines.append("")
    for label, s3 in (("strict", stage3_strict), ("sensitivity", stage3_sens)):
        lines.append(f"### {label.capitalize()}")
        lines.append("")
        lines.append("| architecture | human mean R5 | human n | LLM mean R5 | LLM n | human rank | LLM rank |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in s3["per_architecture"]:
            arch = row["architecture"]
            hm = row.get("human_r5_mean"); hn = row.get("human_n")
            lm = row.get("llm_r5_mean");   ln = row.get("llm_n")
            hr = row.get("human_rank");    lr = row.get("llm_rank")
            def _f(x, fmt="{:.2f}"):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "—"
                return fmt.format(x) if isinstance(x, float) else str(x)
            lines.append(f"| {arch} | {_f(hm)} | {_f(hn, '{:.0f}')} | {_f(lm)} | "
                         f"{_f(ln, '{:.0f}')} | {_f(hr, '{:.0f}')} | {_f(lr, '{:.0f}')} |")
        lines.append("")
        lines.append(
            f"Spearman ρ (arch means) = {s3['spearman_rho']:.2f} (p = {s3['spearman_p']:.3f}); "
            f"Kendall τ (arch ranks) = {s3['kendall_tau']:.2f} (p = {s3['kendall_p']:.3f})."
        )
        lines.append("")
    lines.append("---")
    lines.append("Generated by `analysis/human_validation.py`.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(
    responses_path: Path = DEFAULT_RESPONSES,
    frame_path:     Path = DEFAULT_FRAME,
    llm_db_path:    Path = DEFAULT_LLM_DB,
    out_dir:        Path = RESULTS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[human_validation] loading {responses_path.name}", file=sys.stderr)
    loaded = load_all(responses_path, frame_path, llm_db_path)

    participants = build_participant_summary(loaded)
    participants.to_csv(out_dir / "participants_summary.csv", index=False)

    # Long-form rater × dialogue table for the record (helps reviewers).
    frame_ids = set(loaded.frame_items["blind_id"])
    long = loaded.dialogue_ratings[
        loaded.dialogue_ratings["dialogue_id"].isin(frame_ids | set(SENTINEL_BANDS_STRICT))
    ][["participant_code", "dialogue_id", "r5", "r2", "sequence_position",
       "rating_time_seconds"]].copy()
    long.to_csv(out_dir / "dialogue_ratings.csv", index=False)

    for regime, exclude_col in (("strict", "excluded_strict"),
                                 ("sensitivity", "excluded_sens")):
        incl = set(participants.loc[
            (~participants[exclude_col]) & participants["completed"], "code"])
        print(f"[human_validation] {regime}: included={len(incl)}", file=sys.stderr)
        s1 = stage1_pool_reliability(loaded.dialogue_ratings, loaded.frame_items, incl)
        s2 = stage2_llm_vs_human(loaded.dialogue_ratings, loaded.frame_items, incl)
        s3 = stage3_rank_preservation(loaded.dialogue_ratings, loaded.frame_items,
                                       loaded.llm_scores, incl)
        for stage, obj in (("stage1_pool_reliability", s1),
                           ("stage2_llm_vs_human",     s2),
                           ("stage3_rank_preservation", s3)):
            with open(out_dir / f"{stage}__{regime}.json", "w") as fh:
                json.dump(obj, fh, indent=2, default=str)
        # Persist the per-dialogue means as CSV for the strict regime only
        # (sensitivity table is a superset and doesn't need duplication).
        if regime == "strict":
            dm = pd.DataFrame(s2.get("_dialogue_means", []))
            if not dm.empty:
                dm.to_csv(out_dir / "dialogue_means.csv", index=False)
        if regime == "strict":
            s1_strict, s2_strict, s3_strict = s1, s2, s3
        else:
            s1_sens, s2_sens, s3_sens = s1, s2, s3

    report = build_report(participants,
                          s1_strict, s2_strict, s3_strict,
                          s1_sens,   s2_sens,   s3_sens)
    (out_dir / "REPORT.md").write_text(report)
    print(f"[human_validation] wrote outputs to {out_dir}", file=sys.stderr)


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    p.add_argument("--frame",     type=Path, default=DEFAULT_FRAME)
    p.add_argument("--llm-db",    type=Path, default=DEFAULT_LLM_DB)
    p.add_argument("--out-dir",   type=Path, default=RESULTS_DIR)
    return p


if __name__ == "__main__":
    args = _cli().parse_args()
    run(args.responses, args.frame, args.llm_db, args.out_dir)
