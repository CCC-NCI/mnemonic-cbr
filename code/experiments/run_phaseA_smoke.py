"""Phase A smoke test — single case, deterministic stubs, no LLM.

Spec exit condition (REBUILD_SPECIFICATION_v3.md §5 Phase A):
    "Wire up a single test case running end-to-end with placeholder
     LLM calls (returning fixed strings). Verify the dialogue loop
     works."

This script does exactly that for one (case, persona, architecture,
student_leg) cell. Run it from the mnemonic-cbr/ directory:

    cd mnemonic-cbr
    python code/experiments/run_phaseA_smoke.py

Or any architecture/persona:

    python code/experiments/run_phaseA_smoke.py --architecture pure_cbr_tpl
    python code/experiments/run_phaseA_smoke.py --persona socratic --architecture hybrid

All architectures work in Phase A (the LLM-using ones go through stubs).
pure_cbr_tpl is the most informative because it actually exercises
retrieval (no LLM stub to mask what's happening).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _setup_path():
    """Make 'cbr', 'dialogue', and 'experiments' importable when the
    script is run from mnemonic-cbr/ (the most natural cwd)."""
    here = Path(__file__).resolve().parent  # .../code/experiments
    code_dir = here.parent                   # .../code
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from experiments._run_logger import (                         # noqa: E402
    run_with_logging,
    set_log_path,
    utc_stamp,
)

from dialogue.llm_provider import StubProvider                # noqa: E402
from dialogue.loop import run_dialogue                        # noqa: E402
from dialogue.retrieval import clean_mnemonic_engine          # noqa: E402
from dialogue.student import StudentSimulator                 # noqa: E402
from dialogue.teacher import ARCHITECTURES, TeacherGenerator  # noqa: E402
from dialogue.personas import PERSONAS                        # noqa: E402

from experiments.eedi_loader import (                         # noqa: E402
    load_first_usable_case,
    load_n_usable_cases,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--data",
        default="data/all_train.csv",
        help="Path to EEDI CSV (default: data/all_train.csv)",
    )
    p.add_argument(
        "--persona",
        default="experiential",
        choices=list(PERSONAS),
        help="Teaching persona for the teacher",
    )
    p.add_argument(
        "--architecture",
        default="hybrid",
        choices=list(ARCHITECTURES),
        help="Teacher architecture branch",
    )
    p.add_argument(
        "--student-leg",
        default="leg_a",
        choices=["leg_a", "leg_b"],
        help="Which student leg label to record in the dialogue",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Spec fixes 4; --max-turns is provided for debugging only",
    )
    p.add_argument(
        "--case-base-size",
        type=int,
        default=200,
        help="How many cases to load into the retrieval base "
        "(default 200; small values like 20 produce ~random retrieval "
        "regardless of the embedding model because the base lacks "
        "topical matches; 200 has been verified to yield topic-aware "
        "retrieval on the BIDMAS query)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_log_path(
        Path("results/phaseA_smoke/logs")
        / f"run_phaseA_smoke_{utc_stamp()}.txt"
    )
    print("=" * 72)
    print("Phase A smoke test — dialogue layer, stubs only, no LLM")
    print("=" * 72)
    print(f"persona      = {args.persona}")
    print(f"architecture = {args.architecture}")
    print(f"student leg  = {args.student_leg}")
    print(f"max turns    = {args.max_turns}")
    print(f"data file    = {args.data}")
    print(f"case base    = {args.case_base_size} cases loaded for retrieval")
    print()

    # 1. Load enough cases for a non-trivial retrieval base.
    #    The dialogue runs on the first case; the rest provide the
    #    case base for retrieval-using architectures.
    print("[1/4] Loading EEDI cases...")
    cases_ext = load_n_usable_cases(n=args.case_base_size, filepath=args.data)
    if not cases_ext:
        print("ERROR: no usable cases in the EEDI file.", file=sys.stderr)
        sys.exit(1)
    target_case = cases_ext[0]
    legacy_case_base = [ce.case for ce in cases_ext]
    print(f"      loaded {len(cases_ext)} cases; target case id={target_case.id}")
    print(f"      target misconception: {target_case.misconception!r}")
    print(f"      target question (first 80 chars): "
          f"{' '.join(target_case.problem_text.split())[:80]!r}")
    print(f"      target student answer: {target_case.student_answer_text!r}")
    print()

    # 2. Build a cleaned mnemonic engine (only needed for retrieval-using
    #    architectures, but cheap and harmless otherwise).
    print("[2/4] Building cleaned mnemonic engine (outcome-leakage shim)...")
    needs_engine = args.architecture in ("pure_cbr_llm", "pure_cbr_tpl", "hybrid")
    if needs_engine:
        from dialogue.embeddings import is_available as _emb_available
        # n_chunks=5 because 20 cases is small; KMeans would warn otherwise.
        engine = clean_mnemonic_engine(legacy_case_base, n_chunks=5)
        retrieval_path = "embeddings (cosine)" if _emb_available() else "legacy 4-feature euclidean (fallback)"
        print(f"      engine built. Base similarity = {retrieval_path}.")
    else:
        engine = None
        print(f"      engine not needed for architecture={args.architecture}")
    print()

    # 3. Build teacher and student with stub providers.
    print("[3/4] Wiring stub-based teacher and student...")
    teacher_provider = (
        None if args.architecture == "pure_cbr_tpl"
        else StubProvider(role="teacher")
    )
    teacher = TeacherGenerator(
        architecture=args.architecture,
        persona=args.persona,
        provider=teacher_provider,
        mnemonic_engine=engine,
        case_base=legacy_case_base if needs_engine else [],
        k_retrieve=3,
    )
    student = StudentSimulator(
        provider=StubProvider(role=f"student_{args.student_leg}"),
        leg_name=args.student_leg,
    )
    print(f"      teacher = {teacher.architecture}/{teacher.persona} "
          f"(provider={'stub' if teacher_provider else 'none — pure_cbr_tpl'})")
    print(f"      student = stub provider, leg={student.leg_name}")
    print()

    # 4. Run the dialogue.
    print("[4/4] Running dialogue loop...")
    state = run_dialogue(
        case_ext=target_case,
        persona=args.persona,
        architecture=args.architecture,
        student_leg=args.student_leg,
        teacher=teacher,
        student=student,
        max_turns=args.max_turns,
        seed=42,
    )
    print()
    print("-" * 72)
    print("Dialogue transcript")
    print("-" * 72)
    print(f"case_id            = {state.case_id}")
    print(f"misconception      = {state.misconception_label!r}")
    print(f"persona            = {state.persona}")
    print(f"architecture       = {state.architecture}")
    print(f"student_leg        = {state.student_leg}")
    print(f"max_turns          = {state.max_turns}")
    print(f"actual turn count  = {len(state.turn_history)}")
    print()
    for turn in state.turn_history:
        print(f"  Turn {turn.turn_index} [{turn.speaker:>7s}]: {turn.text}")
    print()

    # Exit-condition verification.
    print("-" * 72)
    print("Phase A exit-condition checks")
    print("-" * 72)
    checks = [
        ("turn count == max_turns", len(state.turn_history) == state.max_turns),
        ("turn 0 is student",       state.turn_history[0].speaker == "student"),
        ("turn 1 is teacher",       state.turn_history[1].speaker == "teacher"),
        ("turn 2 is student",       state.turn_history[2].speaker == "student"),
        ("turn 3 is teacher",       state.turn_history[3].speaker == "teacher"),
        ("turn 0 is deterministic", not state.turn_history[0].text.startswith("[STUB:")),
        ("turn 1 came from a stub" if args.architecture != "pure_cbr_tpl"
         else "turn 1 is template-wrap (no stub)",
         (state.turn_history[1].text.startswith("[STUB:")
          if args.architecture != "pure_cbr_tpl"
          else not state.turn_history[1].text.startswith("[STUB:"))),
    ]
    all_pass = True
    for label, ok in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}")
        all_pass = all_pass and ok
    print()
    if all_pass:
        print("All Phase A exit-condition checks PASSED.")
        sys.exit(0)
    print("One or more checks FAILED.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="run_phaseA_smoke"))
