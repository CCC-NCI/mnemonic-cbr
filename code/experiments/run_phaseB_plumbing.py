"""Phase B plumbing smoke — full stack, end-to-end.

Spec reference: IMPLEMENTATION_PLAN §5 Phase B and §6.1 (Phase
B-plumbing).

Exercises:
    EEDI loader → dialogue loop → RubricScorer → DialogueStore

Default mode is stub: zero API calls, deterministic outputs, the goal
is to verify that the full pipeline executes end-to-end and produces
the expected artefacts (a JSON file per dialogue + a SQLite manifest
row per dialogue).

Real-API mode: set USE_REAL_LLMS=1 in the environment. Anthropic
serves every role in this mode (model-overlap acknowledged in
IMPLEMENTATION_PLAN §6.1; outputs are discarded). ANTHROPIC_API_KEY
must be set. Cost ~$0.05 for the default 5-dialogue run.

Usage:

    # Stub-mode smoke (default, no API calls)
    python code/experiments/run_phaseB_plumbing.py

    # One dialogue only (fastest)
    python code/experiments/run_phaseB_plumbing.py --n 1

    # Real Anthropic, 5 dialogues
    USE_REAL_LLMS=1 ANTHROPIC_API_KEY=sk-... python code/experiments/run_phaseB_plumbing.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--data",
        default="data/all_train.csv",
        help="Path to EEDI CSV (default: data/all_train.csv)",
    )
    p.add_argument(
        "--out",
        default="results/phaseB_plumbing",
        help="Output root for JSON + SQLite (default: results/phaseB_plumbing)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of (case × persona × architecture × student_leg) cells "
        "to run. Default 5 — minimal coverage of the routing matrix.",
    )
    p.add_argument(
        "--case-base-size",
        type=int,
        default=100,
        help="How many cases to load into the retrieval base "
        "(default 100; large enough to yield topical retrieval under "
        "the embedding similarity path)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (INFO level)",
    )
    return p.parse_args()


def design_cells(n: int) -> List[Tuple[int, str, str, str]]:
    """Pick n (case_index, persona, architecture, student_leg) cells.

    For n <= 5 we deliberately spread across architectures so that
    every branch of teacher.py gets exercised at least once. After
    that we cycle.
    """
    # Order so that with n=5 we hit each architecture exactly once.
    base = [
        (0, "experiential",  "hybrid",       "leg_a"),
        (0, "socratic",      "pure_cbr_tpl", "leg_b"),
        (0, "rule_based",    "baseline",     "leg_a"),
        (0, "constructive",  "pure_ai",      "leg_b"),
        (0, "traditional",   "pure_cbr_llm", "leg_a"),
    ]
    cells = []
    for i in range(n):
        case_idx, persona, arch, leg = base[i % len(base)]
        # If n > 5, vary the case index too so we don't run the same
        # case 5 times.
        case_idx = i // len(base)
        cells.append((case_idx, persona, arch, leg))
    return cells


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    log_path = set_log_path(
        Path(args.out) / "logs" / f"run_phaseB_plumbing_{utc_stamp()}.txt"
    )

    real_mode = os.getenv("USE_REAL_LLMS", "0").strip().lower() in {"1", "true", "yes", "on"}
    print("=" * 72)
    print("Phase B plumbing smoke")
    print("=" * 72)
    print(f"mode         = {'REAL (Anthropic)' if real_mode else 'STUB (no API calls)'}")
    print(f"output root  = {args.out}")
    print(f"log file     = {log_path}")
    print(f"data file    = {args.data}")
    print(f"n cells      = {args.n}")
    print(f"case base    = {args.case_base_size}")
    print()

    # 1. Load cases.
    print("[1/4] Loading EEDI cases...")
    cases_ext = load_n_usable_cases(n=args.case_base_size, filepath=args.data)
    if not cases_ext:
        print("ERROR: no usable cases.", file=sys.stderr)
        return 2
    legacy_case_base = [ce.case for ce in cases_ext]
    print(f"      loaded {len(cases_ext)} cases.")
    print()

    # 2. Build cleaned retrieval engine once (shared across cells).
    print("[2/4] Building cleaned mnemonic engine...")
    from dialogue.embeddings import is_available as _emb_available
    engine = clean_mnemonic_engine(legacy_case_base, n_chunks=5)
    retrieval_path = "embeddings (cosine)" if _emb_available() else "legacy 4-feature euclidean (fallback)"
    print(f"      Base similarity = {retrieval_path}.")
    print()

    # 3. Build providers + scorer (shared across cells where possible).
    print("[3/4] Wiring providers and scorer...")
    teacher_provider = for_role("teacher")
    student_provider_a = for_role("student_leg_a")
    student_provider_b = for_role("student_leg_b")
    judge_provider = for_role("judge_primary")
    scorer = RubricScorer(primary_provider=judge_provider)
    print(f"      teacher      = {teacher_provider}")
    print(f"      student_leg_a= {student_provider_a}")
    print(f"      student_leg_b= {student_provider_b}")
    print(f"      judge_primary= {judge_provider}")
    print()

    # 4. Run cells and persist.
    print(f"[4/4] Running {args.n} cells...")
    cells = design_cells(args.n)
    store = DialogueStore(root=args.out)
    failures = 0
    try:
        for i, (case_idx, persona, arch, leg) in enumerate(cells, start=1):
            target_case = cases_ext[case_idx % len(cases_ext)]
            print(
                f"  [{i}/{args.n}] case={target_case.id} "
                f"persona={persona} arch={arch} leg={leg}"
            )

            # Build a per-cell teacher (architecture differs per cell).
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
            )

            state = run_dialogue(
                case_ext=target_case,
                persona=persona,
                architecture=arch,
                student_leg=leg,
                teacher=teacher,
                student=student,
                max_turns=4,
                seed=42 + i,
            )
            score = scorer.score(state)
            path = store.write(state, score)

            ok = (
                len(state.turn_history) == 4
                and score.R5 is not None
                and score.R6 is not None
            )
            marker = "OK" if ok else "DEGRADED"
            print(
                f"        → {marker}: R5={score.R5} "
                f"quality={_fmt(score.quality_composite)} "
                f"R6={score.R6}  ({path.name})"
            )
            if not ok:
                failures += 1

        total = store.count()
    finally:
        store.close()

    print()
    print("-" * 72)
    print("Phase B plumbing checks")
    print("-" * 72)
    checks = [
        ("manifest row count == cells run", total == len(cells)),
        ("no degraded cells",               failures == 0),
        ("JSON dir exists",                 (Path(args.out) / "dialogues").is_dir()),
        ("SQLite manifest exists",          (Path(args.out) / "manifest.sqlite").is_file()),
    ]
    all_pass = True
    for label, ok in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}")
        all_pass = all_pass and ok
    print()
    print(f"Persisted {total} dialogue(s) under {args.out}/")
    if all_pass:
        print("All Phase B plumbing checks PASSED.")
        return 0
    print("One or more checks FAILED.", file=sys.stderr)
    return 1


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="run_phaseB_plumbing"))
