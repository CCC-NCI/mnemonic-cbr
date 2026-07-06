"""Check 2 inter judge reliability runner.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.4 (inter judge ICC on a
50 dialogue subset) and §4 (Sanity Check Battery, Check 2).

The check answers one question. The primary judge in this work is
Claude Sonnet. Does a judge from a different model family agree with
Claude Sonnet on the six rubric items R1 to R6? If two model families
land in the same place on the same dialogues then the rubric scores
are not an artefact of one provider's prompt tuning.

This script does not touch live participants and needs no ethics
approval. It only re scores dialogues that have already been written
to disk under results/phaseB_smoke_5turn/dialogues/, using a second
LLM judge from a different provider family (Google Gemini by default).

Workflow

  1. Load the 50 dialogue stratified subset that select_validation_subset.py
     already produced (results/phaseB_smoke_5turn/validation_subset/
     validation_subset.csv) and pick the first --n-dialogues rows.
     The subset is already stratified across architecture, persona,
     and student leg, so the first 50 rows give a balanced sample.
  2. For each selected dialogue, re run the two pass rubric prompt
     against the Gemini API. The Pass 1 prompt is persona blind and
     covers R1 to R5. The Pass 2 prompt is persona visible and covers
     R6 only. The prompts are imported verbatim from scoring/prompts.py
     so the second judge sees exactly what the primary judge saw.
  3. Persist the Gemini score for each dialogue to a per dialogue JSON
     under results/check2_icc/gemini_scores/, plus a summary CSV at
     results/check2_icc/gemini_scores.csv with one row per dialogue.
  4. Pair each Gemini score with the existing Claude Sonnet score
     read from the source dialogue JSON and compute ICC(2,1) per
     rubric item using pingouin. ICC(2,1) is the appropriate model
     for a two way random effects single rater absolute agreement
     case, which is what a comparison of two LLM judges on the same
     dialogues is.
  5. Write the ICC table to results/check2_icc/icc_table.csv and a
     human readable markdown version at results/check2_icc/icc_table.md
     that can be pasted into §sec:results-gates of the manuscript.

The script is resume safe. If gemini_scores.csv already exists then
dialogues that already have a Gemini score are skipped on the next
invocation.

Run modes

  • Default GOOGLE_API_KEY in the environment:
        python -m experiments.run_check2 \\
            --subset results/phaseB_smoke_5turn/validation_subset/validation_subset.csv \\
            --dialogues-dir results/phaseB_smoke_5turn/dialogues \\
            --out results/check2_icc \\
            --n-dialogues 50

  • Resume an interrupted run with the same command. Dialogues whose
    Gemini score is already in gemini_scores.csv are skipped.

  • Different secondary judge family. The script defaults to Gemini
    but accepts --secondary openai or --secondary anthropic for a
    sensitivity check. The third family run is the headline result.

Cost estimate

  Gemini 1.5 Flash, 50 dialogues, two passes each, ~800 input tokens
  plus ~150 output tokens per call: ≈ 100 API calls, well under one
  US dollar at June 2026 prices. Confirm in the Google Cloud console
  before running on a paid project.

Dependencies

  pip install google-generativeai pingouin pandas
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from dialogue.personas import PERSONA_SHORT_DESCRIPTIONS              # noqa: E402
from scoring.prompts import (                                          # noqa: E402
    PASS1_PROMPT,
    PASS2_PROMPT,
    format_turns_block,
)
from scoring import parse as _parse                                    # noqa: E402


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger("run_check2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)


# ---------------------------------------------------------------------
# Secondary judge providers
# ---------------------------------------------------------------------

# The default secondary family is Google Gemini. The two alternatives
# (openai, anthropic) exist for sensitivity checks if Gemini quota is
# exhausted or to confirm that a single non Anthropic provider behaves
# the way Gemini does. Whichever family is chosen is recorded in the
# output CSV so the manuscript can name the actual judge used.

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


class SecondaryJudge:
    """Thin wrapper that exposes generate(prompt) -> str for the
    selected secondary provider. The classes already in
    dialogue/llm_provider.py cover OpenAI and Anthropic; this file
    adds a Gemini wrapper because Gemini is not yet wired into the
    main provider dispatch.
    """

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "google-generativeai not installed. "
                    "Run: pip install google-generativeai"
                ) from e
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY (or GEMINI_API_KEY) env var is not set"
                )
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)
        if self.provider == "openai":
            try:
                import openai  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "openai not installed. Run: pip install openai"
                ) from e
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY env var is not set")
            import openai
            return openai.OpenAI(api_key=api_key)
        if self.provider == "anthropic":
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
            return anthropic.Anthropic(api_key=api_key)
        raise ValueError(f"Unknown secondary provider: {self.provider!r}")

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 400) -> str:
        """Generate with 429 aware retry.

        Free tier quotas (Gemini in particular) bounce when the per
        minute or per day request budget is exhausted. The error
        message includes a retry hint such as "retry in 42.84s". We
        parse it, sleep that long, and try again up to a small bound.
        """
        max_retries = 6
        for attempt in range(max_retries):
            try:
                return self._generate_once(prompt, temperature, max_tokens)
            except Exception as e:
                err_str = str(e)
                type_name = type(e).__name__
                is_rate_limit = (
                    "429" in err_str
                    or "ResourceExhausted" in type_name
                    or "RateLimitError" in type_name
                    or "rate limit" in err_str.lower()
                )
                if not is_rate_limit or attempt == max_retries - 1:
                    raise
                # Daily quotas can never be cleared by retrying inside
                # the same calendar day, so fail fast instead of burning
                # six minutes of retries.
                is_daily_quota = (
                    "PerDay" in err_str
                    or "RequestsPerDay" in err_str
                )
                if is_daily_quota:
                    raise RuntimeError(
                        "Daily quota exhausted. Free tier on this model "
                        "is capped per day. Enable billing on the Google "
                        "Cloud project, or switch to --secondary openai "
                        "or --secondary anthropic for Check 2."
                    ) from e
                # Try to parse "retry in 42.84s" from the error text.
                m = re.search(r"retry in ([\d.]+)\s*s", err_str)
                wait = float(m.group(1)) if m else 30.0
                wait = min(wait + 2.0, 90.0)
                logger.warning(
                    "429 rate limit hit, sleeping %.1fs and retrying "
                    "(attempt %d of %d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
        raise RuntimeError("max retries exceeded on rate limit")

    def _generate_once(self, prompt: str, temperature: float,
                       max_tokens: int) -> str:
        if self.provider == "gemini":
            import google.generativeai as genai  # type: ignore
            response = self._client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            # response.text is a "quick accessor" that raises when the
            # candidate has no Part, which happens for finish_reason 2
            # (MAX_TOKENS) on thinking models like gemini-2.5-pro.
            # Walk the candidate structure manually so the run does not
            # crash on a single empty response.
            try:
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    parts = getattr(content, "parts", None) or []
                    text_chunks = [getattr(p, "text", "") for p in parts]
                    joined = "".join(t for t in text_chunks if t)
                    if joined:
                        return joined
                    # No text but there is a finish_reason — surface it
                    # as a parsable error so the row records the failure
                    # rather than aborting the whole run.
                    fr = getattr(candidates[0], "finish_reason", None)
                    return (
                        '{"error": "no text parts returned, '
                        f'finish_reason={fr}"}}'
                    )
                return ""
            except Exception:
                return ""
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        if self.provider == "anthropic":
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
        raise ValueError(f"Unknown provider: {self.provider!r}")


# ---------------------------------------------------------------------
# Subset loading
# ---------------------------------------------------------------------

def load_subset(subset_csv: Path, n_dialogues: int) -> List[Dict[str, str]]:
    """Read validation_subset.csv and return the first n rows.

    The CSV is already stratified across architecture, persona, and
    student leg, so the first n rows preserve the stratification
    proportionally for any reasonable n in [10, 150].
    """
    if not subset_csv.exists():
        raise FileNotFoundError(
            f"Subset CSV not found at {subset_csv}. "
            f"Run select_validation_subset.py first."
        )
    rows: List[Dict[str, str]] = []
    with subset_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    if n_dialogues > len(rows):
        raise ValueError(
            f"--n-dialogues={n_dialogues} but subset CSV only has {len(rows)} rows"
        )
    return rows[:n_dialogues]


def load_dialogue_json(dialogues_dir: Path, file_path: str) -> Dict[str, Any]:
    """Load one dialogue JSON. The file_path column in the subset CSV
    is relative to the phaseB_smoke_5turn results directory, e.g.
    'dialogues/case_8__traditional__baseline__leg_a.json'.
    """
    # If file_path already starts with 'dialogues/' strip it and
    # resolve under dialogues_dir.
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

def score_one_dialogue(judge: SecondaryJudge,
                       dialogue: Dict[str, Any],
                       max_tokens: int = 400) -> Dict[str, Any]:
    """Run Pass 1 and Pass 2 prompts against the secondary judge and
    return a dict with R1..R6 plus error fields.
    """
    state = dialogue["state"]
    misconception_label = state["misconception_label"]
    persona = state["persona"]
    turn_history = state["turn_history"]

    persona_desc = PERSONA_SHORT_DESCRIPTIONS.get(persona, persona)
    turns_block = format_turns_block(turn_history)

    pass1_prompt = PASS1_PROMPT.format(
        misconception_label=misconception_label,
        turns_block=turns_block,
    )
    pass2_prompt = PASS2_PROMPT.format(
        persona=persona,
        persona_short_description=persona_desc,
        turns_block=turns_block,
    )

    pass1_text = judge.generate(pass1_prompt, temperature=0.2,
                                max_tokens=max_tokens)
    pass2_text = judge.generate(pass2_prompt, temperature=0.2,
                                max_tokens=max_tokens)

    pass1 = _parse.parse_pass1(pass1_text)
    pass2 = _parse.parse_pass2(pass2_text)

    return {
        "R1": pass1["R1"],
        "R2": pass1["R2"],
        "R3": pass1["R3"],
        "R4": pass1["R4"],
        "R5": pass1["R5"],
        "R6": pass2["R6"],
        "justification_pass1": pass1.get("justification", ""),
        "justification_pass2": pass2.get("justification", ""),
        "judge_provider": judge.provider,
        "judge_model": judge.model,
        "pass1_error": pass1.get("error", ""),
        "pass2_error": pass2.get("error", ""),
        "pass1_raw_text": pass1.get("raw_text", ""),
        "pass2_raw_text": pass2.get("raw_text", ""),
    }


# ---------------------------------------------------------------------
# Persistence (resume safe)
# ---------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "blind_id",
    "case_id",
    "persona",
    "architecture",
    "student_leg",
    "claude_R1",
    "claude_R2",
    "claude_R3",
    "claude_R4",
    "claude_R5",
    "claude_R6",
    "gemini_R1",
    "gemini_R2",
    "gemini_R3",
    "gemini_R4",
    "gemini_R5",
    "gemini_R6",
    "judge_provider",
    "judge_model",
    "pass1_error",
    "pass2_error",
    "file_path",
]


def load_existing_blind_ids(summary_csv: Path) -> set:
    if not summary_csv.exists():
        return set()
    ids = set()
    with summary_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ids.add(row["blind_id"])
    return ids


def append_summary_row(summary_csv: Path, row: Dict[str, Any]) -> None:
    file_existed = summary_csv.exists()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        if not file_existed:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})


def write_per_dialogue_json(per_dialogue_dir: Path,
                            blind_id: str,
                            payload: Dict[str, Any]) -> None:
    per_dialogue_dir.mkdir(parents=True, exist_ok=True)
    out = per_dialogue_dir / f"{blind_id}.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------
# ICC computation
# ---------------------------------------------------------------------

def compute_icc_table(summary_csv: Path) -> "pandas.DataFrame":
    """Return a long form DataFrame with one row per rubric item and
    ICC(2,1) plus its 95 percent CI.
    """
    try:
        import pandas as pd
        import pingouin as pg
    except ImportError as e:
        raise ImportError(
            "pingouin + pandas required for ICC. "
            "Run: pip install pingouin pandas"
        ) from e

    df = pd.read_csv(summary_csv)
    items = ["R1", "R2", "R3", "R4", "R5", "R6"]
    rows = []
    for item in items:
        claude_col = f"claude_{item}"
        gemini_col = f"gemini_{item}"
        # Drop rows where either rater is missing (parse failure).
        paired = df[[claude_col, gemini_col]].dropna()
        if len(paired) < 5:
            rows.append({
                "rubric_item": item,
                "n_paired": len(paired),
                "icc_2_1": None,
                "ci_low": None,
                "ci_high": None,
                "note": "n<5, not enough dialogues for a stable ICC",
            })
            continue
        # Build long format that pingouin expects.
        long_df = pd.DataFrame({
            "target": list(range(len(paired))) * 2,
            "rater":  ["claude"] * len(paired) + ["gemini"] * len(paired),
            "rating": list(paired[claude_col]) + list(paired[gemini_col]),
        })
        icc = pg.intraclass_corr(
            data=long_df, targets="target", raters="rater", ratings="rating"
        )
        # ICC2 row is the two way random absolute agreement single rater model.
        icc2_row = icc[icc["Type"] == "ICC2"].iloc[0]
        ci_low, ci_high = icc2_row["CI95%"]
        rows.append({
            "rubric_item": item,
            "n_paired": len(paired),
            "icc_2_1": float(icc2_row["ICC"]),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "note": "",
        })
    return pd.DataFrame(rows)


def write_icc_outputs(icc_df: "pandas.DataFrame", out_dir: Path) -> None:
    icc_csv = out_dir / "icc_table.csv"
    icc_md = out_dir / "icc_table.md"
    icc_df.to_csv(icc_csv, index=False)

    # Pretty markdown for paste into §sec:results-gates.
    lines = []
    lines.append("# Check 2 Inter Judge ICC")
    lines.append("")
    lines.append("Primary judge: Claude Sonnet. Secondary judge: as listed in the")
    lines.append("CSV. Model is ICC(2,1) two way random effects, absolute")
    lines.append("agreement, single rater. Decision bands per spec §3.4:")
    lines.append("strong ≥ 0.75, adequate 0.60 to 0.74, partial 0.40 to 0.59,")
    lines.append("divergence < 0.40.")
    lines.append("")
    lines.append("| Rubric item | n paired | ICC(2,1) | 95 percent CI |")
    lines.append("|-------------|---------:|---------:|:--------------|")
    for _, r in icc_df.iterrows():
        if r["icc_2_1"] is None or (isinstance(r["icc_2_1"], float) and r["icc_2_1"] != r["icc_2_1"]):
            lines.append(
                f"| {r['rubric_item']} | {int(r['n_paired'])} | -- | -- "
                f"({r['note']}) |"
            )
        else:
            lines.append(
                f"| {r['rubric_item']} | {int(r['n_paired'])} | "
                f"{r['icc_2_1']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |"
            )
    lines.append("")
    icc_md.write_text("\n".join(lines))
    logger.info("Wrote ICC outputs to %s and %s", icc_csv, icc_md)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subset",
        type=Path,
        default=Path("results/phaseB_smoke_5turn/validation_subset/validation_subset.csv"),
        help="Path to the stratified subset CSV (default: %(default)s).",
    )
    p.add_argument(
        "--dialogues-dir",
        type=Path,
        default=Path("results/phaseB_smoke_5turn/dialogues"),
        help="Directory containing the dialogue JSONs (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/check2_icc"),
        help="Output directory (default: %(default)s).",
    )
    p.add_argument(
        "--n-dialogues",
        type=int,
        default=50,
        help="Number of dialogues to score (default: %(default)s).",
    )
    p.add_argument(
        "--secondary",
        choices=("gemini", "openai", "anthropic"),
        default="gemini",
        help="Secondary judge family (default: %(default)s).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default model name for the chosen secondary.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between API calls (default: %(default)s).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help=(
            "Max output tokens per API call (default: %(default)s). "
            "Bump to 4096 or higher for thinking models such as "
            "gemini-2.5-pro that spend output tokens on internal reasoning."
        ),
    )
    p.add_argument(
        "--icc-only",
        action="store_true",
        help="Skip scoring; only recompute ICC from an existing summary CSV.",
    )
    p.add_argument(
        "--list-gemini-models",
        action="store_true",
        help="List Gemini models the current GOOGLE_API_KEY can call, then exit.",
    )
    return p.parse_args()


def list_gemini_models() -> int:
    """Print the Gemini models the current key can call.

    Useful when a model name 404s — model availability depends on the
    project, region, and tier of the key.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        print(f"google-generativeai not installed: {e}")
        return 2
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY (or GEMINI_API_KEY) env var is not set")
        return 2
    genai.configure(api_key=api_key)
    print(f"{'name':50s}  supports generateContent?")
    print("-" * 75)
    for m in genai.list_models():
        supports = "generateContent" in (m.supported_generation_methods or [])
        print(f"{m.name:50s}  {supports}")
    return 0


def main() -> int:
    args = parse_args()

    if args.list_gemini_models:
        return list_gemini_models()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "gemini_scores.csv"
    per_dialogue_dir = out_dir / "gemini_scores"

    if args.icc_only:
        if not summary_csv.exists():
            logger.error("--icc-only but %s not found", summary_csv)
            return 2
        icc_df = compute_icc_table(summary_csv)
        write_icc_outputs(icc_df, out_dir)
        return 0

    # Pick default model per provider if user did not override.
    if args.model:
        model = args.model
    elif args.secondary == "gemini":
        model = DEFAULT_GEMINI_MODEL
    elif args.secondary == "openai":
        model = DEFAULT_OPENAI_MODEL
    else:
        model = DEFAULT_ANTHROPIC_MODEL

    logger.info("Secondary judge: %s / %s", args.secondary, model)
    judge = SecondaryJudge(provider=args.secondary, model=model)

    rows = load_subset(args.subset, args.n_dialogues)
    logger.info("Loaded %d dialogues from %s", len(rows), args.subset)

    already_done = load_existing_blind_ids(summary_csv)
    if already_done:
        logger.info("Resume: %d dialogues already scored, will skip",
                    len(already_done))

    n_scored = 0
    n_skipped = 0
    n_errors = 0
    t0 = time.time()
    for i, subset_row in enumerate(rows, start=1):
        blind_id = subset_row["blind_id"]
        if blind_id in already_done:
            n_skipped += 1
            continue

        try:
            dialogue = load_dialogue_json(args.dialogues_dir,
                                          subset_row["file_path"])
            primary_score = dialogue.get("score", {})
            secondary = score_one_dialogue(judge, dialogue,
                                           max_tokens=args.max_tokens)

            summary_row = {
                "blind_id": blind_id,
                "case_id": subset_row.get("case_id", ""),
                "persona": subset_row.get("persona", ""),
                "architecture": subset_row.get("architecture", ""),
                "student_leg": subset_row.get("student_leg", ""),
                "claude_R1": primary_score.get("R1"),
                "claude_R2": primary_score.get("R2"),
                "claude_R3": primary_score.get("R3"),
                "claude_R4": primary_score.get("R4"),
                "claude_R5": primary_score.get("R5"),
                "claude_R6": primary_score.get("R6"),
                "gemini_R1": secondary.get("R1"),
                "gemini_R2": secondary.get("R2"),
                "gemini_R3": secondary.get("R3"),
                "gemini_R4": secondary.get("R4"),
                "gemini_R5": secondary.get("R5"),
                "gemini_R6": secondary.get("R6"),
                "judge_provider": secondary["judge_provider"],
                "judge_model": secondary["judge_model"],
                "pass1_error": secondary["pass1_error"],
                "pass2_error": secondary["pass2_error"],
                "file_path": subset_row.get("file_path", ""),
            }
            append_summary_row(summary_csv, summary_row)
            write_per_dialogue_json(per_dialogue_dir, blind_id, {
                "blind_id": blind_id,
                "subset_row": subset_row,
                "primary_score": primary_score,
                "secondary_score": secondary,
            })
            n_scored += 1
            logger.info("[%d/%d] %s scored (R1..R6 = %s)",
                        i, len(rows), blind_id,
                        [secondary[k] for k in ("R1","R2","R3","R4","R5","R6")])
        except Exception as e:
            n_errors += 1
            logger.exception("[%d/%d] %s failed: %s", i, len(rows), blind_id, e)
        time.sleep(args.sleep)

    elapsed = time.time() - t0
    logger.info(
        "Done. scored=%d skipped=%d errors=%d elapsed=%.1fs",
        n_scored, n_skipped, n_errors, elapsed,
    )

    if n_scored + len(already_done) >= 5:
        icc_df = compute_icc_table(summary_csv)
        write_icc_outputs(icc_df, out_dir)
    else:
        logger.warning("Too few scored dialogues for ICC; skipping ICC step")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
