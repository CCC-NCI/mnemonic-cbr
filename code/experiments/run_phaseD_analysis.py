"""Phase D analysis — manifest → manuscript tables.

Reads a SQLite manifest produced by run_phaseB_smoke.py or
run_phaseD_main.py and writes manuscript-ready CSV, Markdown, and
LaTeX outputs under {out_dir}/tables/.

Usage:

    python code/experiments/run_phaseD_analysis.py \
        --manifest results/phaseB_smoke/manifest.sqlite \
        --out      results/phaseB_smoke/analysis

The output directory will contain:

    analysis_summary.md                   Single human-readable summary
                                          of all printed output, with a
                                          UTC timestamp in the filename
                                          plus a copy at analysis_summary.md
    tables/
      per_persona_means.{csv,md,tex}
      per_architecture_means.{csv,md,tex}
      cell_R5.{csv,md,tex}                Architecture × Persona cell means for R5
      cell_quality.{csv,md,tex}           Architecture × Persona cell means for quality composite
      cross_student_variance.{csv,md,tex} Cross-leg robustness on R5
      anova_R5.{csv,md,tex}
      anova_quality.{csv,md,tex}
      cohens_d_architecture_R5.{csv,md,tex}
      cohens_d_persona_R5.{csv,md,tex}

Each .tex file is a complete `table` environment with caption and label;
copy into the manuscript with `\\input{path/to/file.tex}` or paste.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from analysis.aggregate import (                       # noqa: E402
    OUTCOMES,
    OUTCOME_LABELS,
    load_manifest,
    per_persona_table,
    per_architecture_table,
    cell_table,
    cross_student_variance,
    narrow_marginals,
)
from analysis.anova import two_way_anova, format_anova_table  # noqa: E402
from analysis.effect_sizes import pairwise_cohens_d, magnitude_label  # noqa: E402
from analysis.export import (                          # noqa: E402
    export_csv,
    export_markdown,
    export_latex,
    export_all,
)
from experiments._run_logger import (                  # noqa: E402
    run_with_logging,
    set_log_path,
    utc_stamp,
)


# ---------------------------------------------------------------------
# Tiny "tee" helper: write to stdout and to a buffer for the summary md
# ---------------------------------------------------------------------

class _Tee:
    """Captures lines for the summary file while also printing them."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, *args, code_block: bool = False) -> None:
        text = " ".join(str(a) for a in args)
        print(text)
        if code_block:
            # Wrap monospaced blocks in fenced code so the markdown
            # preserves alignment.
            self.lines.append("```")
            self.lines.append(text)
            self.lines.append("```")
        else:
            self.lines.append(text)

    def blank(self) -> None:
        print()
        self.lines.append("")

    def heading(self, level: int, text: str) -> None:
        print(text)
        self.lines.append(("#" * level) + " " + text)

    def write_summary(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--manifest",
        default="results/phaseB_smoke/manifest.sqlite",
        help="Path to SQLite manifest from Phase B/D run",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <manifest dir>/analysis)",
    )
    p.add_argument(
        "--filter",
        choices=("real", "stub", "all"),
        default="real",
        help="Which rows to include (default: real, judge_provider != 'stub')",
    )
    return p.parse_args()


def write_analysis_summary(
    manifest,
    out=None,
    filter_mode: str = "real",
) -> int:
    """Programmatic entry point — runs the analysis and writes outputs.

    Returns 0 on success, non-zero on error.
    """
    class _Args:
        pass

    args = _Args()
    args.manifest = manifest
    args.out = out
    args.filter = filter_mode
    return _run_analysis(args)


def main() -> int:
    args = parse_args()
    # When run as a script, tee the console output into a per-run log
    # file under <analysis_dir>/logs/. When called via
    # write_analysis_summary() from another script, the parent run's
    # log_path is already registered and set_log_path is a no-op.
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out) if args.out else manifest_path.parent / "analysis"
    set_log_path(out_dir / "logs" / f"run_phaseD_analysis_{utc_stamp()}.txt")
    return _run_analysis(args)


def _run_analysis(args) -> int:
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out) if args.out else manifest_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    now_iso = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    now_stamp = now_iso.replace(":", "").replace("-", "")

    out = _Tee()
    out.heading(1, "Phase D analysis summary")
    out.blank()
    out(f"- Generated: {now_iso}")
    out(f"- Manifest:  `{manifest_path}`")
    out(f"- Filter:    `{args.filter}`")
    out.blank()
    out("---")
    out.blank()

    out.heading(2, "Manifest summary")
    df_all = load_manifest(manifest_path)
    if args.filter == "real":
        df = df_all[df_all["judge_provider"].fillna("").str.lower() != "stub"].copy()
    elif args.filter == "stub":
        df = df_all[df_all["judge_provider"].fillna("").str.lower() == "stub"].copy()
    else:
        df = df_all.copy()
    out(f"- Total rows in manifest: {len(df_all)}")
    out(f"- Rows after filter:      {len(df)}")
    if len(df) == 0:
        out("")
        out(f"ERROR: no rows match filter={args.filter}. Either run more "
            f"dialogues (USE_REAL_LLMS=1 ...) or pass --filter all to "
            f"include stub rows.")
        out.write_summary(out_dir / "analysis_summary.md")
        return 2
    if len(df) < 25:
        out("")
        out(f"NOTE: only {len(df)} rows survived the filter — ANOVA and "
            f"Cohen's d are computable but statistical interpretation is "
            f"limited at this scale. Treat the output as plumbing "
            f"verification rather than findings.")
    out(f"- Personas:      {sorted(df['persona'].unique())}")
    out(f"- Architectures: {sorted(df['architecture'].unique())}")
    out(f"- Student legs:  {sorted(df['student_leg'].unique())}")
    out.blank()

    # ---- Marginals ----
    # The wide form (n + mean + sd per outcome) is the source of truth and
    # is exported as CSV and Markdown. The narrow form (single 'mean (SD)'
    # cell per outcome) is exported as LaTeX because the wide form has too
    # many columns to fit on a manuscript page.
    out.heading(2, "Per-persona marginals")
    pp = per_persona_table(df)
    pp_narrow = narrow_marginals(pp, index_col="persona")
    n_per_persona = int(pp["R5_n"].iloc[0]) if "R5_n" in pp.columns and len(pp) else 0
    pp_caption = (
        f"Per-persona means across architectures, cases, and student legs "
        f"(n = {n_per_persona} per persona). Cells are 'mean (SD)'. "
        f"R5 is the primary outcome (student trajectory); Quality is the "
        f"(R1+R2+R3)/3 composite; R4 is domain accuracy; R6 is strategy fidelity."
    )
    export_csv(pp, tables_dir / "per_persona_means.csv")
    export_markdown(pp, tables_dir / "per_persona_means.md")
    export_latex(
        pp_narrow, tables_dir / "per_persona_means.tex",
        caption=pp_caption, label="tab:per-persona-means",
    )
    out(pp.to_string(index=False), code_block=True)
    out.blank()

    out.heading(2, "Per-architecture marginals")
    pa = per_architecture_table(df)
    pa_narrow = narrow_marginals(pa, index_col="architecture")
    n_per_arch = int(pa["R5_n"].iloc[0]) if "R5_n" in pa.columns and len(pa) else 0
    pa_caption = (
        f"Per-architecture means across personas, cases, and student legs "
        f"(n = {n_per_arch} per architecture). Cells are 'mean (SD)'. "
        f"Same outcome conventions as the per-persona table."
    )
    export_csv(pa, tables_dir / "per_architecture_means.csv")
    export_markdown(pa, tables_dir / "per_architecture_means.md")
    export_latex(
        pa_narrow, tables_dir / "per_architecture_means.tex",
        caption=pa_caption, label="tab:per-arch-means",
    )
    out(pa.to_string(index=False), code_block=True)
    out.blank()

    # ---- Cell means ----
    out.heading(2, "Architecture × Persona cell means")
    for outcome in ("R5", "quality_composite"):
        cell = cell_table(df, outcome=outcome).reset_index()
        suffix = "R5" if outcome == "R5" else "quality"
        export_all(
            cell, tables_dir, f"cell_{suffix}",
            caption=f"Architecture × Persona cell means for {OUTCOME_LABELS[outcome]}. "
                    f"Each cell averages across cases and student legs.",
            label=f"tab:cell-{suffix}",
        )
    out("Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.")
    out.blank()

    # ---- ANOVA ----
    n_arch = df["architecture"].nunique()
    n_pers = df["persona"].nunique()
    out.heading(2, "Two-way ANOVA (Architecture × Persona)")
    if n_arch < 2 or n_pers < 2:
        out(f"Skipped — need ≥ 2 architectures and ≥ 2 personas; "
            f"got {n_arch} architecture(s), {n_pers} persona(s).")
        out.blank()
    else:
        for outcome in ("R5", "quality_composite"):
            result = two_way_anova(df, outcome=outcome,
                                   factor_a="architecture", factor_b="persona")
            table = format_anova_table(result, factor_a_name="Architecture",
                                       factor_b_name="Persona")
            suffix = "R5" if outcome == "R5" else "quality"
            export_all(
                table, tables_dir, f"anova_{suffix}",
                caption=f"Two-way ANOVA on {OUTCOME_LABELS[outcome]} with "
                        f"Architecture and Persona as factors.",
                label=f"tab:anova-{suffix}",
                ndigits=3,
            )
            out(f"- **{outcome}**: "
                f"F_arch={result['F_a']:.2f} (p={result['p_a']:.3g}, "
                f"η²p={result['eta2p_a']:.3f}); "
                f"F_persona={result['F_b']:.2f} (p={result['p_b']:.3g}, "
                f"η²p={result['eta2p_b']:.3f}); "
                f"F_interaction={result['F_interaction']:.2f} "
                f"(p={result['p_interaction']:.3g}).")
        out.blank()

    # ---- Cohen's d ----
    out.heading(2, "Pairwise Cohen's d on R5")
    for factor in ("architecture", "persona"):
        d_table = pairwise_cohens_d(df, factor=factor, outcome="R5")
        d_table["magnitude"] = d_table["d"].apply(magnitude_label)
        export_all(
            d_table, tables_dir, f"cohens_d_{factor}_R5",
            caption=f"Pairwise Cohen's d on R5 between levels of {factor}.",
            label=f"tab:cohensd-{factor}-R5",
            ndigits=3,
        )
        if not d_table.empty:
            top = d_table.iloc[0]
            out(f"- **{factor}**: largest |d|={top['abs_d']:.2f} "
                f"({top['level_a']} vs {top['level_b']}, {top['magnitude']})")
    out.blank()

    # ---- Cross-student robustness ----
    out.heading(2, "Cross-student variance on R5 (leg_a vs leg_b)")
    csv = cross_student_variance(df, outcome="R5")
    if not csv.empty:
        export_all(
            csv, tables_dir, "cross_student_variance",
            caption="Cross-student variance on R5: per-cell mean absolute "
                    "difference between leg_a (OpenAI student) and leg_b "
                    "(Anthropic student), and per-cell Pearson r.",
            label="tab:cross-student",
        )
        mean_diff = csv["mean_abs_diff"].mean()
        out(f"- Mean |Δ R5| across cells = {mean_diff:.2f}")
    else:
        out("- Skipped (no cells with both legs scored).")
    out.blank()

    out.heading(2, "Output locations")
    out(f"- Tables: `{tables_dir}/`")
    out(f"- Summary: `{out_dir / 'analysis_summary.md'}` "
        f"(also archived as `analysis_summary_{now_stamp}.md`)")

    out.write_summary(out_dir / "analysis_summary.md")
    # Also write a timestamped copy so successive analyses can be compared
    out.write_summary(out_dir / f"analysis_summary_{now_stamp}.md")

    return 0


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="run_phaseD_analysis"))
