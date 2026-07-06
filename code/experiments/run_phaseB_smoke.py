"""Phase B-proper smoke — 250 dialogues, full cell matrix.

Spec reference: REBUILD_SPECIFICATION_v3.md §5 Phase B (B.4 end-to-end
smoke test). IMPLEMENTATION_PLAN §5 Phase B, §6.1 (staging).

The full Phase B smoke covers every cell in the cross-product:

    5 cases × 5 personas × 5 architectures × 2 student legs = 250 dialogues

Each dialogue is run end-to-end: load case → run four-turn dialogue
→ two-pass rubric scoring → persist JSON + manifest row.

This script is one notch up from run_phaseB_plumbing.py — same stack,
larger matrix, real model assignments by default (USE_REAL_LLMS=1
swaps stubs for real providers).

Modes:

  • Default (no env flag):
        Stub mode, no API spend, 250 cells run in ~10 seconds.
        Useful for verifying the matrix coverage and the resume logic.

  • USE_REAL_LLMS=1 with OPENAI_API_KEY + ANTHROPIC_API_KEY:
        Spec model assignments. Teacher = GPT-3.5-turbo,
        student leg A = GPT-4o-mini, student leg B = Claude Haiku 3.5,
        judge primary = Claude Sonnet. Estimated cost ~$5–15.

  • USE_REAL_LLMS=1 with only ANTHROPIC_API_KEY:
        OpenAI-assigned roles fall back to Anthropic with a warning
        (model overlap acknowledged in IMPLEMENTATION_PLAN §6.1).
        Lower cost than full spec assignment but methodologically
        weaker; outputs should be flagged in the manuscript if used.

Resume: rerunning the same --out directory skips cells already present
in the manifest. So if a real-API run is interrupted at cell 137, the
next invocation picks up at cell 138.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from dialogue.llm_provider import for_role                       # noqa: E402
from dialogue.loop import run_dialogue                            # noqa: E402
from dialogue.personas import PERSONAS                            # noqa: E402
from dialogue.retrieval import clean_mnemonic_engine              # noqa: E402
from dialogue.student import StudentSimulator                     # noqa: E402
from dialogue.teacher import ARCHITECTURES, TeacherGenerator      # noqa: E402

from experiments.eedi_loader import load_n_usable_cases           # noqa: E402
from experiments.persist import DialogueStore                     # noqa: E402
from experiments._run_logger import (                              # noqa: E402
    run_with_logging,
    set_log_path,
    utc_stamp,
)

from scoring.rubric import RubricScorer                           # noqa: E402


# ---------------------------------------------------------------------
# Cell matrix
# ---------------------------------------------------------------------

def build_full_matrix(n_cases: int = 5) -> List[Tuple[int, str, str, str]]:
    """Return the full 250-cell matrix: n_cases × personas × architectures × legs."""
    cells: List[Tuple[int, str, str, str]] = []
    for case_idx in range(n_cases):
        for persona in PERSONAS:
            for architecture in ARCHITECTURES:
                for leg in ("leg_a", "leg_b"):
                    cells.append((case_idx, persona, architecture, leg))
    return cells


# ---------------------------------------------------------------------
# Cost reporting
# ---------------------------------------------------------------------

# Per-call cost estimates. These are conservative — actual cost depends
# on token counts which we don't measure precisely in this version of
# the runner. The Phase C token audit (see plan §5 Phase C and spec §11)
# replaces these estimates with measured numbers.
#
# Per LLM call assumption: ~600 input tokens + ~150 output tokens.
# Token prices (USD per 1M tokens) reflect spec-era Anthropic /
# OpenAI pricing; update via the env vars below if needed.

_INPUT_PRICE_PER_M = {
    "gpt-3.5-turbo": 0.50,
    "gpt-4o-mini":   0.15,
    "claude-haiku-4-5":  0.80,
    "claude-haiku-3-5":  0.80,
    "claude-sonnet-4-5": 3.00,
    "claude-sonnet-4-6": 3.00,
    "stub": 0.0,
}
_OUTPUT_PRICE_PER_M = {
    "gpt-3.5-turbo": 1.50,
    "gpt-4o-mini":   0.60,
    "claude-haiku-4-5":  4.00,
    "claude-haiku-3-5":  4.00,
    "claude-sonnet-4-5": 15.00,
    "claude-sonnet-4-6": 15.00,
    "stub": 0.0,
}

_AVG_INPUT_TOKENS_PER_CALL = 600
_AVG_OUTPUT_TOKENS_PER_CALL = 150


def estimate_call_cost(model: str) -> float:
    """Return USD estimate per single call to a model."""
    inp = _INPUT_PRICE_PER_M.get(model, 1.0)
    out = _OUTPUT_PRICE_PER_M.get(model, 5.0)
    cost = (
        (_AVG_INPUT_TOKENS_PER_CALL / 1_000_000) * inp
        + (_AVG_OUTPUT_TOKENS_PER_CALL / 1_000_000) * out
    )
    return cost


def estimate_run_cost(
    n_cells: int,
    teacher_model: str,
    student_a_model: str,
    student_b_model: str,
    judge_model: str,
) -> float:
    """Crude pre-run cost estimate.

    Per cell:
      - 1 teacher call at turn 1
      - 1 student call at turn 2
      - 1 teacher call at turn 3 (unless pure_cbr_tpl)
      - 2 judge calls (Pass 1 and Pass 2)

    We treat the matrix as if 4/5 of cells use the LLM teacher (4 of 5
    architectures) and 1/5 use the template branch. Student legs are
    split 50/50 between leg_a (OpenAI) and leg_b (Anthropic).
    """
    # 4 of 5 architectures use the LLM teacher.
    teacher_calls = n_cells * (4 / 5) * 2  # 2 teacher calls per LLM-using cell
    # Student call at turn 2 for every cell.
    student_a_calls = (n_cells / 2)  # half the cells use leg_a
    student_b_calls = (n_cells / 2)  # half the cells use leg_b
    # Judge: 2 passes per cell.
    judge_calls = n_cells * 2

    return (
        teacher_calls * estimate_call_cost(teacher_model)
        + student_a_calls * estimate_call_cost(student_a_model)
        + student_b_calls * estimate_call_cost(student_b_model)
        + judge_calls * estimate_call_cost(judge_model)
    )


# ---------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------

def already_done(store: DialogueStore, case_id: str, persona: str,
                 architecture: str, student_leg: str,
                 real_mode: bool = False) -> bool:
    """Has this cell already been run in a compatible mode?

    In stub mode, any existing row counts as done.
    In real mode, rows whose judge_provider is 'stub' are NOT counted —
    they were produced under stubs and must be re-run with real LLMs.
    This prevents the silent-skip-of-stub-data trap when switching from
    stub-mode verification to real-API execution against the same --out
    directory.
    """
    rows = store.query(
        "SELECT judge_provider FROM dialogues WHERE case_id=? AND persona=? "
        "AND architecture=? AND student_leg=? LIMIT 1",
        (case_id, persona, architecture, student_leg),
    )
    if not rows:
        return False
    if not real_mode:
        return True
    # Real mode: only count as done if the persisted row came from a
    # non-stub provider.
    judge_provider = (rows[0].get("judge_provider") or "").lower()
    return judge_provider not in {"", "stub"}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--data",
        default="data/all_train.csv",
        help="Path to EEDI CSV (default: data/all_train.csv)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output root for JSON + SQLite. "
        "Default: results/phaseB_smoke for student_mode=pure_ai; "
        "results/phaseB_smoke_cbr for student_mode=cbr_grounded "
        "(set to avoid clobbering the pure_ai manifest).",
    )
    p.add_argument(
        "--student-mode",
        choices=("pure_ai", "cbr_grounded"),
        default="pure_ai",
        help="Student simulator mode. pure_ai (default) is the spec "
        "design: student LLM receives only the misconception label and "
        "dialogue history. cbr_grounded retrieves K=2 same-misconception "
        "cases and injects their wrong-answer texts into the student "
        "prompt — see spec §3.5 A/B test.",
    )
    p.add_argument(
        "--n-cases",
        type=int,
        default=5,
        help="Number of EEDI cases × personas × architectures × legs "
        "(default 5 → 250 cells)",
    )
    p.add_argument(
        "--case-base-size",
        type=int,
        default=500,
        help="How many cases to load into the retrieval base "
        "(default 500; verified empirically as the smallest size at "
        "which embedding retrieval reliably finds topic-aligned cases "
        "on the BIDMAS query — 200 was not enough, see Stage-2 "
        "first-real-mode results)",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Truncate the cell matrix to the first N cells "
        "(default: full 250). Use small values for cheap real-mode "
        "test runs: --max-cells 1 ≈ $0.01, --max-cells 10 ≈ $0.10, "
        "--max-cells 50 ≈ $0.50.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Number of turns per dialogue (default 4). "
        "5 ends on the student (s-t-s-t-s); 6 ends on the teacher "
        "(s-t-s-t-s-t). Cost rises ~20%% per extra turn. The default "
        "output directory is suffixed with '_<N>turn' when N != 4 "
        "so the runs do not collide with the 4-turn manifest.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; re-run every cell even if present in manifest",
    )
    p.add_argument(
        "--skip-inspect",
        action="store_true",
        help="Skip the auto-run of inspect_dialogues.py after the smoke. "
        "By default, dialogues_inspection.md is written to {out}/ when "
        "the smoke finishes.",
    )
    p.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip the auto-run of run_phaseD_analysis.py after the smoke. "
        "By default, analysis_summary.md and the tables/ directory are "
        "written to {out}/analysis/ when the smoke finishes.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (INFO level)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Auto-derive --out based on student mode and turn budget if the
    # user didn't set one. Suffixes prevent runs with different
    # configurations from clobbering each other's manifests.
    if args.out is None:
        base = "results/phaseB_smoke"
        if args.student_mode != "pure_ai":
            base += "_cbr"
        if args.max_turns != 4:
            base += f"_{args.max_turns}turn"
        args.out = base

    # Tee stdout/stderr into a per-run log file so the full console
    # output is preserved alongside the dialogues and manifest.
    log_path = set_log_path(
        Path(args.out) / "logs" / f"run_phaseB_smoke_{utc_stamp()}.txt"
    )
    print(f"log file           = {log_path}")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    real_mode = os.getenv("USE_REAL_LLMS", "0").strip().lower() in {"1", "true", "yes", "on"}
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    print("=" * 72)
    print("Phase B-proper smoke — full 250-cell matrix")
    print("=" * 72)
    print(f"mode               = {'REAL' if real_mode else 'STUB (no API calls)'}")
    if real_mode:
        print(f"OPENAI_API_KEY     = {'set' if has_openai_key else 'NOT SET — will fall back to Anthropic'}")
        print(f"ANTHROPIC_API_KEY  = {'set' if has_anthropic_key else 'NOT SET'}")
    print(f"student mode       = {args.student_mode}")
    print(f"turns per dialogue = {args.max_turns}")
    print(f"output root        = {args.out}")
    print(f"data file          = {args.data}")
    full_matrix_size = args.n_cases * len(PERSONAS) * len(ARCHITECTURES) * 2
    effective_size = min(args.max_cells, full_matrix_size) if args.max_cells else full_matrix_size
    matrix_note = ""
    if args.max_cells and args.max_cells < full_matrix_size:
        matrix_note = f" (truncated to first {effective_size} via --max-cells)"
    print(f"cases × personas × archs × legs = "
          f"{args.n_cases} × {len(PERSONAS)} × {len(ARCHITECTURES)} × 2 = "
          f"{full_matrix_size} cells{matrix_note}")
    print(f"case base          = {args.case_base_size}")
    print(f"resume             = {'off' if args.no_resume else 'on'}")
    print()

    # 1. Load cases.
    print("[1/5] Loading EEDI cases...")
    cases_ext = load_n_usable_cases(n=args.case_base_size, filepath=args.data)
    if len(cases_ext) < args.n_cases:
        print(
            f"ERROR: requested {args.n_cases} cases but only "
            f"{len(cases_ext)} usable cases found.",
            file=sys.stderr,
        )
        return 2
    legacy_case_base = [ce.case for ce in cases_ext]
    target_cases = cases_ext[: args.n_cases]
    print(f"      loaded {len(cases_ext)} cases; using first {args.n_cases} as targets.")
    print()

    # 2. Build cleaned retrieval engine once.
    print("[2/5] Building cleaned mnemonic engine...")
    from dialogue.embeddings import is_available as _emb_available
    engine = clean_mnemonic_engine(legacy_case_base, n_chunks=8)
    retrieval_path = "embeddings (cosine)" if _emb_available() else "legacy 4-feature euclidean (fallback)"
    print(f"      Base similarity = {retrieval_path}.")
    print()

    # 3. Build providers + scorer.
    print("[3/5] Wiring providers and scorer...")
    teacher_provider = for_role("teacher")
    student_provider_a = for_role("student_leg_a")
    student_provider_b = for_role("student_leg_b")
    judge_provider = for_role("judge_primary")
    scorer = RubricScorer(primary_provider=judge_provider)
    print(f"      teacher       = {teacher_provider}")
    print(f"      student_leg_a = {student_provider_a}")
    print(f"      student_leg_b = {student_provider_b}")
    print(f"      judge_primary = {judge_provider}")
    print()

    # 4. Cost estimate (real-mode only) + stub-data warning.
    cells = build_full_matrix(args.n_cases)
    if args.max_cells is not None and args.max_cells < len(cells):
        cells = cells[: args.max_cells]

    # If real mode and the manifest already contains stub rows, warn.
    store_preview = DialogueStore(root=args.out)
    try:
        stub_rows = len(store_preview.query(
            "SELECT 1 FROM dialogues WHERE LOWER(COALESCE(judge_provider, ''))='stub'"
        ))
    finally:
        store_preview.close()

    if real_mode and stub_rows > 0:
        print(
            f"NOTE: {stub_rows} cells in {args.out}/manifest.sqlite were "
            f"produced in STUB mode. Real-mode resume logic will re-run "
            f"those cells with real LLMs (--no-resume would re-run "
            f"every cell)."
        )
        print()

    if real_mode:
        est_cost = estimate_run_cost(
            n_cells=len(cells),
            teacher_model=teacher_provider.model,
            student_a_model=student_provider_a.model,
            student_b_model=student_provider_b.model,
            judge_model=judge_provider.model,
        )
        print(f"[4/5] Cost estimate (REAL mode): ~${est_cost:,.2f} for {len(cells)} cells")
        print(f"       Press Ctrl-C in the next 5 seconds to abort.")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("Aborted by user.", file=sys.stderr)
            return 130
    else:
        print(f"[4/5] Stub mode — no API spend.")
    print()

    # 5. Run cells.
    print(f"[5/5] Running {len(cells)} cells...")
    store = DialogueStore(root=args.out)
    skipped = 0
    completed = 0
    degraded = 0
    started_at = time.time()

    try:
        for i, (case_idx, persona, arch, leg) in enumerate(cells, start=1):
            target_case = target_cases[case_idx]

            if not args.no_resume and already_done(
                store, target_case.id, persona, arch, leg,
                real_mode=real_mode,
            ):
                skipped += 1
                continue

            teacher = TeacherGenerator(
                architecture=arch,
                persona=persona,
                provider=(None if arch == "pure_cbr_tpl" else teacher_provider),
                mnemonic_engine=engine,
                case_base=legacy_case_base,
                k_retrieve=3,
            )
            student = StudentSimulator(
                provider=(student_provider_a if leg == "leg_a" else student_provider_b),
                leg_name=leg,
                student_mode=args.student_mode,
                mnemonic_engine=(engine if args.student_mode == "cbr_grounded" else None),
                case_base=(legacy_case_base if args.student_mode == "cbr_grounded" else None),
            )

            try:
                state = run_dialogue(
                    case_ext=target_case,
                    persona=persona,
                    architecture=arch,
                    student_leg=leg,
                    teacher=teacher,
                    student=student,
                    max_turns=args.max_turns,
                    seed=42 + i,
                )
                score = scorer.score(state)
            except KeyboardInterrupt:
                print("\nInterrupted; partial progress preserved in manifest.")
                raise
            except Exception as e:
                print(f"  [{i}/{len(cells)}] FAILED ({type(e).__name__}: {e})")
                degraded += 1
                continue

            store.write(state, score)
            completed += 1
            ok = (
                len(state.turn_history) == args.max_turns
                and score.R5 is not None
                and score.R6 is not None
            )
            if not ok:
                degraded += 1

            if i % 25 == 0 or i == len(cells):
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{i}/{len(cells)}] completed={completed} "
                    f"skipped={skipped} degraded={degraded} "
                    f"({rate:.1f} cells/sec)"
                )
        total = store.count()
    finally:
        store.close()

    elapsed = time.time() - started_at
    print()
    print("-" * 72)
    print("Phase B-proper run summary")
    print("-" * 72)
    print(f"cells in matrix    = {len(cells)}")
    print(f"completed this run = {completed}")
    print(f"skipped (resumed)  = {skipped}")
    print(f"degraded           = {degraded}")
    print(f"manifest rows      = {total}")
    print(f"elapsed            = {elapsed:.1f}s")
    print()

    # Structural checks — these are hard gates. If any of them fails,
    # the run did not produce usable data and we abort before the
    # post-actions. Degradation is reported separately as a quality
    # warning (see below) and only fails the run above a high
    # threshold, because 1-5% degradation is normal under real-API
    # conditions (transient 503s, malformed judge JSON, etc.).
    structural_checks = [
        ("manifest row count >= cells run",       total >= len(cells)),
        ("JSON dir exists",                       (Path(args.out) / "dialogues").is_dir()),
        ("SQLite manifest exists",                (Path(args.out) / "manifest.sqlite").is_file()),
    ]
    structural_ok = True
    for label, ok in structural_checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}")
        structural_ok = structural_ok and ok

    # Degradation as a soft / hard gate.
    degraded_rate = (degraded / len(cells)) if len(cells) else 0.0
    DEGRADED_HARD_FAIL_THRESHOLD = 0.25  # 25% of cells
    if degraded == 0:
        print("  [PASS] no degraded cells")
        degraded_ok = True
    elif degraded_rate < DEGRADED_HARD_FAIL_THRESHOLD:
        print(
            f"  [WARN] {degraded} cells degraded "
            f"({degraded_rate * 100:.1f}% of {len(cells)}) — "
            f"under the {DEGRADED_HARD_FAIL_THRESHOLD * 100:.0f}% "
            f"hard-fail threshold; continuing"
        )
        degraded_ok = True
    else:
        print(
            f"  [FAIL] {degraded} cells degraded "
            f"({degraded_rate * 100:.1f}% of {len(cells)}) — "
            f"above the {DEGRADED_HARD_FAIL_THRESHOLD * 100:.0f}% "
            f"hard-fail threshold"
        )
        degraded_ok = False

    print()
    if not structural_ok:
        print("Structural checks FAILED — skipping post-actions.", file=sys.stderr)
        return 1
    if degraded_ok:
        if degraded == 0:
            print("All Phase B-proper checks PASSED.")
        else:
            print(
                f"Phase B-proper finished with {degraded} degraded cell(s) "
                f"(within tolerance)."
            )
    print()

    # ---- Auto post-actions ----
    manifest_path = Path(args.out) / "manifest.sqlite"
    if not args.skip_inspect:
        try:
            from experiments.inspect_dialogues import write_inspection_md
            md_path = write_inspection_md(
                manifest=manifest_path,
                filter_mode="real",
            )
            if md_path:
                print(f"[post] dialogues inspection → {md_path}")
        except Exception as e:
            print(f"[post] inspect_dialogues failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

    if not args.skip_analyze:
        try:
            from experiments.run_phaseD_analysis import write_analysis_summary
            rc = write_analysis_summary(
                manifest=manifest_path,
                filter_mode="real",
            )
            if rc == 0:
                print(f"[post] analysis summary → "
                      f"{Path(args.out) / 'analysis' / 'analysis_summary.md'}")
            else:
                print(f"[post] analysis returned non-zero: {rc}", file=sys.stderr)
        except Exception as e:
            print(f"[post] run_phaseD_analysis failed: {type(e).__name__}: {e}",
                  file=sys.stderr)

    # Propagate degradation hard-fail as the final exit code, AFTER the
    # post-actions have run so the user always gets the inspection +
    # analysis tables even on a soft-failed run.
    return 0 if degraded_ok else 1


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="run_phaseB_smoke"))
