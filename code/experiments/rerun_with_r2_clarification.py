"""LLM judge rerun with R2 disambiguation appended.

OSF pre registration reference: §6.1, fourth methodological commitment.
Action plan reference: §1.1a (registered with the OSF deposit).

Why this rerun exists
---------------------

The human rating instrument adds one disambiguation paragraph to the R2
item, directing the human rater to source R2 from what the teacher
asks rather than from how the student responds. The LLM judge's
original Pass 1 prompt does not contain this paragraph. Stage 2 of the
validation study enters the LLM as one additional rater alongside the
human pool in an ICC. That comparison is only clean if both sides
answered the same item. With the disambiguation on the human side
only, a low Stage 2 ICC would conflate judge invalidity with wording
asymmetry, which is exactly the confound the verbatim equivalence
requirement existed to prevent.

The rerun therefore scores the 150 dialogue sampling frame subset
with the disambiguation appended to the LLM judge's Pass 1 prompt.
The Stage 2 ICC against the human pool is then computed against
these rerun scores.

Scope
-----

Only the 150 dialogues in the sampling frame are rescored, and only
on R5 and R2 (the two anchored items). R1, R3, R4 are not rescored
because they are not entered into Stage 2; the LLM only items
continue to be reported against the original Phase D scores.

Outputs are written under
`results/check2_icc_rerun_with_r2_clarification/`, one JSON per
dialogue plus a summary CSV, mirroring the layout of
`run_check2.py` so the analysis pipeline can read them with a
single path swap.

Usage
-----

    export ANTHROPIC_API_KEY="..."
    python code/experiments/rerun_with_r2_clarification.py \\
        --sampling-manifest ../validation-study/sampling_frame/sampling_frame_manifest.json \\
        --dialogues-dir results/phaseB_smoke_5turn/dialogues \\
        --out results/check2_icc_rerun_with_r2_clarification \\
        --judge-model claude-sonnet-4-5

Cost (June 2026 prices)
-----------------------

150 dialogues, two pass scoring per dialogue, Claude Sonnet at
$3/M input + $15/M output tokens, ~1500 input + ~150 output tokens
per call. Total approximately USD 1.50 wall clock about 10 minutes.

The rerun is intentionally cheap because it only runs once, before
the Prolific round opens. If a Stage 2 sensitivity check later
suggests the disambiguation shifts LLM scores by less than 0.3
points on R5 or R2 over a 50 dialogue audit subset, the original
Phase D scores can be reused for Stage 2 as well and the deposit
amended to record the equivalence finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from dialogue.personas import PERSONA_SHORT_DESCRIPTIONS            # noqa: E402
from scoring.prompts import (                                        # noqa: E402
    PASS1_PROMPT,
    PASS2_PROMPT,
    format_turns_block,
)
from scoring import parse as _parse                                  # noqa: E402


logger = logging.getLogger("rerun_with_r2_clarification")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)


# ---------------------------------------------------------------------
# The disambiguation paragraph
# ---------------------------------------------------------------------

R2_DISAMBIGUATION = (
    "For R2, judge from what the teacher asks, regardless of how the "
    "student responds. A teacher who demands reasoning but is met "
    "with a confused student still scores high on R2. A teacher who "
    "states the answer but happens to provoke a thoughtful student "
    "response still scores low on R2."
)


def make_pass1_prompt_with_disambiguation(
    misconception_label: str, turns_block: str,
) -> str:
    """Build the Pass 1 prompt with the R2 disambiguation inserted
    immediately below the R2 item definition.

    The original PASS1_PROMPT contains the line
        R2. Cognitive demand: Does the teacher require the student
        to reason, or just state the answer?
    and we insert the disambiguation as the next line. Both the
    human instrument and this rerun see the same disambiguation in
    the same location relative to the R2 item.
    """
    base = PASS1_PROMPT.format(
        misconception_label=misconception_label,
        turns_block=turns_block,
    )
    target = (
        "R2. Cognitive demand: Does the teacher require the student to reason,\n"
        "    or just state the answer?"
    )
    replacement = target + "\n    " + R2_DISAMBIGUATION
    if target not in base:
        # PASS1_PROMPT formatting changed; refuse to silently emit a
        # different prompt than registered.
        raise RuntimeError(
            "Could not locate the R2 item in the Pass 1 prompt template. "
            "The disambiguation insertion is registered against the v3 "
            "prompt layout. Update both this script and the OSF deposit."
        )
    return base.replace(target, replacement)


# ---------------------------------------------------------------------
# Anthropic Sonnet judge wrapper
# ---------------------------------------------------------------------

class SonnetJudge:
    def __init__(self, model: str):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic not installed. Run: pip install anthropic"
            ) from e
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY env var is not set")
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 600) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return str(response.content[0])


# ---------------------------------------------------------------------
# Sampling manifest loader
# ---------------------------------------------------------------------

def load_sampling_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sampling manifest not found at {path}")
    with path.open() as fh:
        manifest = json.load(fh)
    items = manifest.get("items", [])
    logger.info("Loaded sampling manifest n=%d from %s", len(items), path)
    return items


def load_dialogue_json(dialogues_dir: Path, file_path: str) -> Dict[str, Any]:
    rel = file_path
    if rel.startswith("dialogues/"):
        rel = rel[len("dialogues/"):]
    full = dialogues_dir / rel
    if not full.exists():
        raise FileNotFoundError(f"Dialogue not found at {full}")
    with full.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------
# Scoring one dialogue
# ---------------------------------------------------------------------

def score_one_dialogue(judge: SonnetJudge,
                       dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """Rescore one dialogue using Pass 1 with disambiguation and Pass 2
    unchanged. Returns a dict with R5 and R2 and the original R1, R3,
    R4, R6 reused from the dialogue's existing score so a downstream
    consumer can see all six items in one row.
    """
    state = dialogue["state"]
    misconception_label = state["misconception_label"]
    persona = state["persona"]
    turn_history = state["turn_history"]
    turns_block = format_turns_block(turn_history)
    persona_desc = PERSONA_SHORT_DESCRIPTIONS.get(persona, persona)

    pass1_prompt = make_pass1_prompt_with_disambiguation(
        misconception_label, turns_block,
    )
    pass2_prompt = PASS2_PROMPT.format(
        persona=persona,
        persona_short_description=persona_desc,
        turns_block=turns_block,
    )

    pass1_text = judge.generate(pass1_prompt, temperature=0.2, max_tokens=600)
    pass2_text = judge.generate(pass2_prompt, temperature=0.2, max_tokens=600)

    pass1 = _parse.parse_pass1(pass1_text)
    pass2 = _parse.parse_pass2(pass2_text)

    return {
        "rerun_R1": pass1["R1"],
        "rerun_R2": pass1["R2"],
        "rerun_R3": pass1["R3"],
        "rerun_R4": pass1["R4"],
        "rerun_R5": pass1["R5"],
        "rerun_R6": pass2["R6"],
        "judge_model": judge.model,
        "pass1_text": pass1_text,
        "pass2_text": pass2_text,
        "pass1_error": pass1["error"],
        "pass2_error": pass2["error"],
    }


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "blind_id", "case_id", "persona", "architecture", "student_leg",
    "original_R1", "original_R2", "original_R3", "original_R4", "original_R5", "original_R6",
    "rerun_R1", "rerun_R2", "rerun_R3", "rerun_R4", "rerun_R5", "rerun_R6",
    "judge_model", "pass1_error", "pass2_error", "file_path",
]


def load_existing(summary_csv: Path) -> set:
    if not summary_csv.exists():
        return set()
    out = set()
    with summary_csv.open() as fh:
        for row in csv.DictReader(fh):
            out.add(row["blind_id"])
    return out


def append_summary(summary_csv: Path, row: Dict[str, Any]) -> None:
    existed = summary_csv.exists()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        if not existed:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sampling-manifest",
        type=Path,
        required=True,
        help="Path to validation-study/sampling_frame/sampling_frame_manifest.json.",
    )
    p.add_argument(
        "--dialogues-dir",
        type=Path,
        default=Path("results/phaseB_smoke_5turn/dialogues"),
        help="Path to Phase D dialogue JSONs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-5",
        help="Sonnet model name (default %(default)s).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Sleep seconds between API calls.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N manifest items (smoke).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "rerun_scores.csv"
    per_dialogue_dir = out_dir / "rerun_scores"

    items = load_sampling_manifest(args.sampling_manifest)
    if args.limit:
        items = items[:args.limit]

    judge = SonnetJudge(model=args.judge_model)
    logger.info("Sonnet judge: %s", args.judge_model)

    done = load_existing(summary_csv)
    if done:
        logger.info("Resume: %d already rescored, will skip", len(done))

    n_scored = n_skipped = n_errors = 0
    t0 = time.time()
    for i, item in enumerate(items, start=1):
        blind_id = item["blind_id"]
        if blind_id in done:
            n_skipped += 1
            continue
        try:
            dialogue = load_dialogue_json(args.dialogues_dir, item["file_path"])
            orig = dialogue.get("score", {})
            rerun = score_one_dialogue(judge, dialogue)
            row = {
                "blind_id": blind_id,
                "case_id": item.get("case_id", ""),
                "persona": item.get("persona", ""),
                "architecture": item.get("architecture", ""),
                "student_leg": item.get("student_leg", ""),
                "original_R1": orig.get("R1"),
                "original_R2": orig.get("R2"),
                "original_R3": orig.get("R3"),
                "original_R4": orig.get("R4"),
                "original_R5": orig.get("R5"),
                "original_R6": orig.get("R6"),
                "rerun_R1": rerun["rerun_R1"],
                "rerun_R2": rerun["rerun_R2"],
                "rerun_R3": rerun["rerun_R3"],
                "rerun_R4": rerun["rerun_R4"],
                "rerun_R5": rerun["rerun_R5"],
                "rerun_R6": rerun["rerun_R6"],
                "judge_model": rerun["judge_model"],
                "pass1_error": rerun["pass1_error"],
                "pass2_error": rerun["pass2_error"],
                "file_path": item.get("file_path", ""),
            }
            append_summary(summary_csv, row)

            per_dialogue_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "blind_id": blind_id,
                "sampling_manifest_item": item,
                "original_score": orig,
                "rerun_score": {k: v for k, v in rerun.items()
                                if k != "judge_model"},
                "judge_model": rerun["judge_model"],
            }
            with (per_dialogue_dir / f"{blind_id}.json").open("w") as fh:
                json.dump(payload, fh, indent=2)
            n_scored += 1
            logger.info(
                "[%d/%d] %s rescored (R5 %s -> %s, R2 %s -> %s)",
                i, len(items), blind_id,
                orig.get("R5"), rerun["rerun_R5"],
                orig.get("R2"), rerun["rerun_R2"],
            )
        except Exception as e:
            n_errors += 1
            logger.exception("[%d/%d] %s failed: %s", i, len(items), blind_id, e)
        time.sleep(args.sleep)

    logger.info(
        "Done. scored=%d skipped=%d errors=%d elapsed=%.1fs",
        n_scored, n_skipped, n_errors, time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
