"""Check 5 floor-compression probe.

Runs sanity Check 5 (dialogue-order corruption) on a subset of
dialogues whose original R5 is at or above a threshold (default 3),
where there is enough "trajectory headroom" for the 0.5-point drop
criterion to be detectable.

Reports as a robustness analysis *alongside* the strict full-sample
Check 5 in `sanity_summary.md` — does not overwrite that file.

Usage:

    USE_REAL_LLMS=1 TOKENIZERS_PARALLELISM=false \\
        python code/experiments/run_check5_probe.py

Optional flags:

    --manifest PATH    Manifest to read (default: results/phaseB_smoke/manifest.sqlite)
    --min-r5 INT       R5 floor for the filtered sample (default: 3)
    --n INT            Number of dialogues to sample (default: 40)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path


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
        help="Manifest to read from",
    )
    p.add_argument(
        "--min-r5",
        type=int,
        default=3,
        help="R5 floor for the filtered sample (default 3)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=40,
        help="Number of dialogues to sample (default 40)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output Markdown file (default: same dir as manifest, "
        "filename check5_probe_summary.md)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    out_path = (
        Path(args.out) if args.out
        else manifest_path.parent / "check5_probe_summary.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    set_log_path(
        out_path.parent / "logs" / f"run_check5_probe_{utc_stamp()}.txt"
    )

    from sanity.checks import check5_dialogue_order

    real_mode = os.getenv("USE_REAL_LLMS", "0").lower() in {"1", "true", "yes", "on"}

    lines = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("# Check 5 floor-compression probe")
    out("")
    out(f"- Generated: {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    out(f"- Manifest:  `{manifest_path}`")
    out(f"- Real mode: {real_mode}")
    out(f"- Filter:    R5 >= {args.min_r5}")
    out(f"- Target n:  {args.n}")
    out("")
    out("---")
    out("")
    out("## Result")
    out("")

    result = check5_dialogue_order(
        manifest_path,
        n=args.n,
        min_R5=args.min_r5,
    )

    out(f"- **Pass:** `{result['pass']}`")
    out(f"- {result['summary']}")
    out("")

    if result["pass"] is True:
        out("**Interpretation.** The pre-registered 0.5-point drop "
            "criterion is met on dialogues with sufficient R5 headroom. "
            "The framework's R5 is order-sensitive when the trajectory "
            "signal is large enough; the failure of the full-sample "
            "Check 5 is attributable to floor compression on a "
            "left-skewed R5 distribution rather than to insensitivity "
            "of R5 to dialogue order.")
    elif result["pass"] is False:
        out("**Interpretation.** Even on the high-R5 subset, the drop "
            "does not reach 0.5. R5's order-sensitivity is genuinely "
            "weaker than the spec's pre-registered threshold expected. "
            "Document in Limitations.")
    else:
        out("**Interpretation.** The probe could not run — likely too "
            "few dialogues with R5 >= the filter threshold. Lower "
            "the filter or expand the manifest.")
    out("")

    out("---")
    out("")
    out("_The strict full-sample Check 5 result is preserved in "
        f"`{manifest_path.parent / 'sanity_summary.md'}`._")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nWrote {out_path}")
    print(f"Wrote {json_path}")
    return 0 if result["pass"] is True else 1


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="run_check5_probe"))
