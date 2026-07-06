"""Phase C sanity-check battery.

Spec reference: REBUILD_SPECIFICATION_v3.md §4.

Runs checks 1, 3, 4, 5 against an existing manifest. Check 2 (inter-
judge ICC) is deferred until a Gemini key is available. Writes a
human-readable summary at `{manifest_dir}/sanity_summary.md` plus
the raw results as `{manifest_dir}/sanity_summary.json`.

Check 4 and Check 5 require Anthropic API calls (the primary judge
re-scores adversarial variants). USE_REAL_LLMS=1 is required for
those two; otherwise they fall back to the stub judge and the result
is reported with a clear caveat.

Usage:

    USE_REAL_LLMS=1 python code/experiments/run_sanity_battery.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import List


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--manifest",
        default="results/phaseB_smoke/manifest.sqlite",
        help="Path to SQLite manifest (default: "
        "results/phaseB_smoke/manifest.sqlite)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: same dir as the manifest.",
    )
    p.add_argument(
        "--n-adversarial",
        type=int,
        default=40,
        help="Number of dialogues to use for adversarial checks 4 and 5 "
        "(default 40, per spec).",
    )
    p.add_argument(
        "--check1-case-id",
        default="case_0",
        help="Case to use for persona discriminability check (default: case_0)",
    )
    p.add_argument(
        "--check1-architecture",
        default="hybrid",
        help="Architecture to use for Check 1 (default: hybrid)",
    )
    p.add_argument(
        "--skip-check4",
        action="store_true",
        help="Skip Check 4 (saves API cost; useful for replay).",
    )
    p.add_argument(
        "--skip-check5",
        action="store_true",
        help="Skip Check 5 (saves API cost; useful for replay).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    set_log_path(out_dir / "logs" / f"run_sanity_battery_{utc_stamp()}.txt")

    from sanity.checks import (
        check1_persona_discriminability,
        check3_r5_contamination,
        check4_shuffle_persona,
        check5_dialogue_order,
    )

    real_mode = os.getenv("USE_REAL_LLMS", "0").lower() in {"1", "true", "yes", "on"}

    lines: List[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("# Phase C sanity-check battery")
    out("")
    out(f"- Generated: {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    out(f"- Manifest:  `{manifest_path}`")
    out(f"- Real mode: {real_mode} (checks 4 and 5 need this)")
    out("")
    out("---")
    out("")

    results = []

    # Check 1
    out("## Check 1 — Persona discriminability")
    r1 = check1_persona_discriminability(
        manifest_path,
        target_case_id=args.check1_case_id,
        architecture=args.check1_architecture,
    )
    results.append(r1)
    out(f"- **Pass:** `{r1['pass']}`")
    out(f"- {r1['summary']}")
    out("")

    # Check 3
    out("## Check 3 — R5 contamination (length / sentiment)")
    r3 = check3_r5_contamination(manifest_path)
    results.append(r3)
    out(f"- **Pass:** `{r3['pass']}`")
    out(f"- {r3['summary']}")
    out("")

    # Check 4 (costs API). Skip cleanly under stub mode rather than
    # reporting a misleading FAIL: the stub judge returns near-identical
    # scores for the original and the persona-shuffled dialogues, so the
    # drop is meaningless. The gate decision treats a stub-skip as
    # "not computable" (pass=None), distinct from an actual FAIL.
    out("## Check 4 — Shuffle-persona (adversarial; needs real judge)")
    if args.skip_check4:
        out("- **Skipped** by --skip-check4 flag.")
        results.append({"name": "Check 4", "pass": None, "summary": "skipped by flag"})
    elif not real_mode:
        out("- **Skipped (stub mode)** — needs USE_REAL_LLMS=1 to call "
            "the real judge for adversarial re-scoring; under stubs the "
            "drop is uninformative. Re-run with `USE_REAL_LLMS=1` "
            "(~$0.40, ~1 min).")
        results.append({"name": "Check 4", "pass": None,
                        "summary": "skipped (stub mode; needs USE_REAL_LLMS=1)"})
    else:
        r4 = check4_shuffle_persona(manifest_path, n=args.n_adversarial)
        results.append(r4)
        out(f"- **Pass:** `{r4['pass']}`")
        out(f"- {r4['summary']}")
    out("")

    # Check 5 (costs API). Same stub-skip logic as Check 4.
    out("## Check 5 — Dialogue-order corruption (adversarial; needs real judge)")
    if args.skip_check5:
        out("- **Skipped** by --skip-check5 flag.")
        results.append({"name": "Check 5", "pass": None, "summary": "skipped by flag"})
    elif not real_mode:
        out("- **Skipped (stub mode)** — needs USE_REAL_LLMS=1 to call "
            "the real judge for adversarial re-scoring; under stubs the "
            "drop is uninformative. Re-run with `USE_REAL_LLMS=1` "
            "(~$0.40, ~1 min).")
        results.append({"name": "Check 5", "pass": None,
                        "summary": "skipped (stub mode; needs USE_REAL_LLMS=1)"})
    else:
        r5 = check5_dialogue_order(manifest_path, n=args.n_adversarial)
        results.append(r5)
        out(f"- **Pass:** `{r5['pass']}`")
        out(f"- {r5['summary']}")
    out("")

    out("---")
    out("")
    out("## Gate decision")
    out("")
    # A check is a HARD fail only if pass == False. pass == None means
    # the check was skipped (by flag or because stub mode prevented it
    # from running meaningfully); skips do not flip the exit code.
    hard_fails = [r["name"] for r in results if r.get("pass") is False]
    skips = [r["name"] for r in results if r.get("pass") is None]
    all_runnable_passed = not hard_fails

    if all_runnable_passed and not skips:
        out("**All implemented checks PASSED.** The methodology has cleared "
            "its pre-registered sanity gate. (Check 2 — inter-judge ICC — "
            "is deferred pending a Google/Gemini key.)")
    elif all_runnable_passed and skips:
        out(f"**All runnable checks PASSED.** Not yet run: {', '.join(skips)}. "
            f"To complete the gate, re-run with `USE_REAL_LLMS=1` so the "
            f"adversarial checks call the real judge.")
    else:
        out("**Gate not fully cleared.** "
            + (f"Failed: {', '.join(hard_fails)}. " if hard_fails else "")
            + (f"Not yet run: {', '.join(skips)}." if skips else ""))

    out("")
    out(f"_Output also written as JSON: `{out_dir / 'sanity_summary.json'}`._")
    out("")

    summary_md = out_dir / "sanity_summary.md"
    summary_json = out_dir / "sanity_summary.json"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nWrote {summary_md}")
    print(f"Wrote {summary_json}")
    # Exit non-zero ONLY on a hard fail. A skipped check is not a fail
    # (the user is informed they need to re-run with real LLMs).
    return 0 if all_runnable_passed else 1


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="run_sanity_battery"))
