"""A/B compare two student-mode runs.

Reads two manifests (one per student_mode) and reports whether the
cbr_grounded student materially changes the rubric scores relative
to the pure_ai student.

Joins on (case_id, persona, architecture, student_leg). Cells present
in both manifests are paired. For each rubric item, reports:
  - mean of mode A
  - mean of mode B
  - mean delta (B − A)
  - paired Cohen's d
  - per-architecture and per-persona breakdowns

Writes a Markdown summary at:
  results/student_mode_comparison.md
plus a timestamped archive copy.

Usage:

    python code/experiments/compare_student_modes.py \\
        --pure-ai     results/phaseB_smoke/manifest.sqlite \\
        --cbr-grounded results/phaseB_smoke_cbr/manifest.sqlite

If invoked with no arguments, defaults to the conventional locations.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from experiments._run_logger import (                  # noqa: E402
    run_with_logging,
    set_log_path,
    utc_stamp,
)


ITEMS = ("R1", "R2", "R3", "R4", "R5", "R6", "quality_composite")
ITEM_LABELS = {
    "R1": "R1 misconception engagement",
    "R2": "R2 cognitive demand",
    "R3": "R3 scaffolding fit",
    "R4": "R4 domain accuracy",
    "R5": "R5 student trajectory",
    "R6": "R6 strategy fidelity",
    "quality_composite": "Quality composite (R1+R2+R3)/3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--pure-ai",
        default="results/phaseB_smoke/manifest.sqlite",
        help="Manifest for student_mode=pure_ai (default: "
        "results/phaseB_smoke/manifest.sqlite)",
    )
    p.add_argument(
        "--cbr-grounded",
        default="results/phaseB_smoke_cbr/manifest.sqlite",
        help="Manifest for student_mode=cbr_grounded (default: "
        "results/phaseB_smoke_cbr/manifest.sqlite)",
    )
    p.add_argument(
        "--out",
        default="results/student_mode_comparison.md",
        help="Output Markdown summary",
    )
    return p.parse_args()


def load_real_rows(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT case_id, persona, architecture, student_leg, "
        "R1, R2, R3, R4, R5, R6, quality_composite, judge_provider "
        "FROM dialogues "
        "WHERE LOWER(COALESCE(judge_provider, '')) != 'stub'",
        conn,
    )
    conn.close()
    return df


def paired_cohens_d(diffs: pd.Series) -> float:
    diffs = diffs.dropna()
    if len(diffs) < 2:
        return float("nan")
    sd = diffs.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diffs.mean() / sd)


def magnitude_label(d: float) -> str:
    if pd.isna(d):
        return "n/a"
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def main() -> int:
    args = parse_args()
    pure_path = Path(args.pure_ai)
    cbr_path = Path(args.cbr_grounded)

    out_dir = Path(args.out).parent
    set_log_path(out_dir / "logs" / f"compare_student_modes_{utc_stamp()}.txt")

    lines: List[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("# Student-mode A/B comparison")
    out("")
    out(f"- Generated:    {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    out(f"- pure_ai:      `{pure_path}`")
    out(f"- cbr_grounded: `{cbr_path}`")
    out("")

    df_pure = load_real_rows(pure_path)
    df_cbr = load_real_rows(cbr_path)
    out(f"## Row counts")
    out("")
    out(f"- pure_ai real rows:       {len(df_pure)}")
    out(f"- cbr_grounded real rows:  {len(df_cbr)}")
    out("")

    if df_pure.empty or df_cbr.empty:
        out("ERROR: one of the manifests has no real rows. "
            "Run both modes before invoking this script.")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 2

    # Join on the four cell-key columns.
    keys = ["case_id", "persona", "architecture", "student_leg"]
    merged = df_pure.merge(
        df_cbr, on=keys, suffixes=("_pure", "_cbr"), how="inner"
    )
    out(f"## Paired cells (present in BOTH manifests)")
    out("")
    out(f"- Paired cell count: {len(merged)}")
    if len(merged) == 0:
        out("\nERROR: no cells were paired. Run both modes on the same "
            "(case × persona × architecture × leg) matrix first.")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 2
    out("")

    # Headline table — overall mean and paired d per rubric item.
    rows = []
    for item in ITEMS:
        a = merged[f"{item}_pure"]
        b = merged[f"{item}_cbr"]
        diffs = (b - a).dropna()
        rows.append(
            {
                "item": ITEM_LABELS[item],
                "mean_pure": float(a.mean()) if a.notna().any() else float("nan"),
                "mean_cbr": float(b.mean()) if b.notna().any() else float("nan"),
                "mean_delta_cbr_minus_pure": float(diffs.mean()) if len(diffs) else float("nan"),
                "paired_d": paired_cohens_d(diffs),
                "n_paired": int(len(diffs)),
            }
        )
    headline = pd.DataFrame(rows)
    headline["magnitude"] = headline["paired_d"].apply(magnitude_label)

    out("## Headline: per-item paired comparison")
    out("")
    out("```")
    out(headline.to_string(index=False))
    out("```")
    out("")

    # Decision summary.
    r5 = headline[headline["item"] == ITEM_LABELS["R5"]].iloc[0]
    quality = headline[headline["item"] == ITEM_LABELS["quality_composite"]].iloc[0]
    out("## Reading the headline")
    out("")
    out(f"- R5 (student trajectory): cbr−pure delta = "
        f"{r5['mean_delta_cbr_minus_pure']:+.2f}, paired d = "
        f"{r5['paired_d']:+.2f} ({r5['magnitude']}).")
    out(f"- Quality composite:        cbr−pure delta = "
        f"{quality['mean_delta_cbr_minus_pure']:+.2f}, paired d = "
        f"{quality['paired_d']:+.2f} ({quality['magnitude']}).")
    out("")
    out("Decision rule:")
    out("- |paired d| on R5 ≥ 0.5  → cbr_grounded materially shifts the "
        "trajectory signal; adopt cbr_grounded as canonical for Phase D.")
    out("- |paired d| on R5 < 0.2  → no material shift; keep pure_ai as "
        "the spec design; document the test as a negative result.")
    out("- 0.2 ≤ |paired d| < 0.5 → ambiguous; run on a larger matrix or "
        "report both in the manuscript with a discussion paragraph.")
    out("")

    # Per-architecture breakdown on R5.
    out("## R5 by architecture (paired)")
    out("")
    per_arch = []
    for arch, sub in merged.groupby("architecture"):
        a = sub["R5_pure"]
        b = sub["R5_cbr"]
        diffs = (b - a).dropna()
        per_arch.append(
            {
                "architecture": arch,
                "n": int(len(diffs)),
                "mean_pure_R5": float(a.mean()) if a.notna().any() else float("nan"),
                "mean_cbr_R5": float(b.mean()) if b.notna().any() else float("nan"),
                "delta": float(diffs.mean()) if len(diffs) else float("nan"),
                "paired_d": paired_cohens_d(diffs),
            }
        )
    per_arch_df = pd.DataFrame(per_arch).sort_values("delta", ascending=False)
    out("```")
    out(per_arch_df.to_string(index=False))
    out("```")
    out("")

    # Per-persona breakdown on R5.
    out("## R5 by persona (paired)")
    out("")
    per_pers = []
    for persona, sub in merged.groupby("persona"):
        a = sub["R5_pure"]
        b = sub["R5_cbr"]
        diffs = (b - a).dropna()
        per_pers.append(
            {
                "persona": persona,
                "n": int(len(diffs)),
                "mean_pure_R5": float(a.mean()) if a.notna().any() else float("nan"),
                "mean_cbr_R5": float(b.mean()) if b.notna().any() else float("nan"),
                "delta": float(diffs.mean()) if len(diffs) else float("nan"),
                "paired_d": paired_cohens_d(diffs),
            }
        )
    per_pers_df = pd.DataFrame(per_pers).sort_values("delta", ascending=False)
    out("```")
    out(per_pers_df.to_string(index=False))
    out("```")
    out("")

    # Write.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Timestamped archive copy.
    stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_path = out_path.with_name(
        out_path.stem + f"_{stamp}" + out_path.suffix
    )
    archive_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"Archived as {archive_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="compare_student_modes"))
