"""Select a stratified validation subset for the human-rater study.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.4 (human-rater validation
of R5) and VALIDATION_STUDY_MATERIALS.md §10.

Draws ~100--200 dialogues from a manifest, stratified across persona
× architecture × case-difficulty band. Exports:

  - validation_subset.csv     One row per selected dialogue with
                              identifier + LLM-judge R5 (held back
                              from raters, retained for analysis).
  - validation_subset.md      Survey-ready rendering of each
                              selected dialogue, in the same format
                              the survey will present.
  - validation_subset.html    Same as the .md but in HTML for
                              pasting into Qualtrics / Google Forms.

Dialogues for raters are blinded to:
  - the persona claim (Pass 1 mirror)
  - the architecture
  - the LLM judge's R5 rating

Identifiers are persistent so ratings can be joined back to the
manifest later.

Usage:

    python code/experiments/select_validation_subset.py \\
        --manifest results/phaseB_smoke/manifest.sqlite \\
        --n 150

Optional flags:
    --attention-checks INT   Number of attention-check sentinels to
                             interleave (default 10, ~1 per 15
                             dialogues).
    --seed INT               Stratification seed (default 42).
    --out PATH               Output directory (default:
                             {manifest_dir}/validation_subset).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--manifest",
        default="results/phaseB_smoke_5turn/manifest.sqlite",
        help="Path to SQLite manifest (default: "
        "results/phaseB_smoke_5turn/manifest.sqlite). The default is the "
        "five-turn run reported as the primary result in the manuscript; "
        "pass --manifest results/phaseB_smoke/manifest.sqlite to draw "
        "from the four-turn ablation run instead.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=150,
        help="Target subset size before attention checks (default 150)",
    )
    p.add_argument(
        "--attention-checks-csv",
        default=None,
        help="Path to a researcher-curated CSV of attention-check "
        "dialogues. Columns: blind_id (or case_id+persona+architecture+"
        "student_leg), intended_rating (1-5), check_type (A or B), "
        "rationale. If omitted, NO attention checks are added and a "
        "warning is printed --- using auto-extracted (judge-derived) "
        "attention checks would circularly penalise raters who disagree "
        "with the LLM judge in either direction, which is the disagreement "
        "the validation study is designed to measure.",
    )
    p.add_argument(
        "--allow-auto-attention-checks",
        action="store_true",
        help="DEPRECATED. Falls back to the original auto-extraction "
        "(R5=1 -> type A, R5=5 -> type B) and emits a strong warning. "
        "Kept only for backwards-compatible reproduction of earlier runs.",
    )
    p.add_argument(
        "--auto-attention-checks-n",
        type=int,
        default=10,
        help="Only used with --allow-auto-attention-checks (default 10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Stratification seed (default 42)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: {manifest_dir}/validation_subset)",
    )
    return p.parse_args()


def _real_rows(manifest_path: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(manifest_path))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT * FROM dialogues "
        "WHERE LOWER(COALESCE(judge_provider,'')) != 'stub' "
        "  AND R5 IS NOT NULL"
    ).fetchall())
    conn.close()
    return [dict(r) for r in rows]


def _load_dialogue(manifest_dir: Path, row: Dict[str, Any]) -> Optional[dict]:
    p = manifest_dir / row.get("file_path", "")
    if not p.exists():
        from experiments.persist import _filename_for_keys
        p = manifest_dir / "dialogues" / _filename_for_keys(
            row["case_id"], row["persona"], row["architecture"], row["student_leg"]
        )
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _construct_word_count(state: dict) -> int:
    """Proxy for case difficulty: word count of the misconception label
    plus the question text. Used only for stratification banding."""
    text = state.get("misconception_label", "") or ""
    turns = state.get("turn_history", []) or []
    if turns:
        text += " " + (turns[0].get("text", "") if isinstance(turns[0], dict)
                       else turns[0].text)
    return len(text.split())


def _difficulty_band(word_count: int, thresholds=(20, 35)) -> str:
    if word_count < thresholds[0]:
        return "short"
    if word_count < thresholds[1]:
        return "medium"
    return "long"


def _r5_band(r5) -> str:
    """Coarse R5 banding for stratified sampling. Three bands so the
    rater sees the full rating range, not just the modal middle:
      - low:  R5 in {1, 2}
      - mid:  R5 == 3
      - high: R5 in {4, 5}
    """
    try:
        r = int(round(float(r5)))
    except (TypeError, ValueError):
        return "mid"
    if r <= 2:
        return "low"
    if r == 3:
        return "mid"
    return "high"


def _stratify_and_sample(
    rows: List[Dict[str, Any]],
    manifest_dir: Path,
    n_target: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Stratify rows by (persona, architecture, difficulty_band, R5_band)
    and draw a sample whose R5 distribution is balanced across low/mid/high.

    Balancing on R5_band matters for ICC: if the rater sees only mid-range
    dialogues (which dominate the manifest after the floor lifted to ~2.67),
    they have no full-scale anchor points and ICC is depressed. Balancing
    the sample ensures the rater encounters dialogues across the full
    rating range.
    """
    rng = random.Random(seed)
    # Build stratum -> rows. R5_band is included in the key so that the
    # sample isn't skewed toward the modal middle of the R5 distribution.
    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        payload = _load_dialogue(manifest_dir, row)
        if payload is None:
            continue
        wc = _construct_word_count(payload["state"])
        diff_band = _difficulty_band(wc)
        r5_band = _r5_band(row.get("R5"))
        key = (row["persona"], row["architecture"], diff_band, r5_band)
        row["_difficulty_band"] = diff_band
        row["_word_count"] = wc
        row["_r5_band"] = r5_band
        strata[key].append(row)

    # Allocate equally across the three R5 bands first (the variable we
    # most care about for ICC), then proportionally across the
    # persona/architecture/difficulty sub-strata within each R5 band.
    per_r5_band_target = max(1, n_target // 3)
    by_r5_band: Dict[str, Dict[tuple, List[Dict[str, Any]]]] = {
        "low": defaultdict(list),
        "mid": defaultdict(list),
        "high": defaultdict(list),
    }
    for key, group in strata.items():
        persona, arch, diff_band, r5_band = key
        by_r5_band[r5_band][(persona, arch, diff_band)] = group

    selected: List[Dict[str, Any]] = []
    for r5_band, sub_strata in by_r5_band.items():
        total_in_band = sum(len(v) for v in sub_strata.values())
        if total_in_band == 0:
            continue
        for sub_key, group in sub_strata.items():
            rng.shuffle(group)
            take = max(1, round(per_r5_band_target * len(group) / total_in_band))
            take = min(take, len(group))
            selected.extend(group[:take])

    # Trim to target with random pruning if over.
    if len(selected) > n_target:
        rng.shuffle(selected)
        selected = selected[:n_target]
    return selected


def _attention_check_rows_from_csv(
    rows: List[Dict[str, Any]],
    csv_path: Path,
) -> List[Dict[str, Any]]:
    """Load researcher-curated attention checks from a CSV file.

    Required CSV columns:
      blind_id          (preferred — joins to the existing manifest blind_id)
      OR (case_id, persona, architecture, student_leg)  — natural key
      intended_rating   1..5
      check_type        "A" (obvious-failure, expect 1) or "B" (obvious-resolution, expect 5)
      rationale         one-line justification (kept for audit, not shown to raters)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"attention-checks CSV not found: {csv_path}")
    # Build lookup by both blind_id and natural key for flexibility.
    by_blind: Dict[str, Dict[str, Any]] = {_blind_id(r): r for r in rows}
    by_natural: Dict[tuple, Dict[str, Any]] = {
        (r["case_id"], r["persona"], r["architecture"], r["student_leg"]): r
        for r in rows
    }
    out: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for spec in reader:
            row = None
            bid = (spec.get("blind_id") or "").strip()
            if bid and bid in by_blind:
                row = by_blind[bid]
            else:
                natural = (
                    (spec.get("case_id") or "").strip(),
                    (spec.get("persona") or "").strip(),
                    (spec.get("architecture") or "").strip(),
                    (spec.get("student_leg") or "").strip(),
                )
                if natural in by_natural:
                    row = by_natural[natural]
            if row is None:
                print(
                    f"WARN: attention-check spec did not match any manifest row: {spec}",
                    file=sys.stderr,
                )
                continue
            intended = (spec.get("intended_rating") or "").strip()
            ctype = (spec.get("check_type") or "").strip().upper()
            row["_attention_check"] = f"{ctype}_intended_{intended}"
            row["_attention_rationale"] = (spec.get("rationale") or "").strip()
            out.append(row)
    return out


def _attention_check_rows_auto(
    rows: List[Dict[str, Any]],
    n_each: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """DEPRECATED. Auto-extract attention-check sentinels from LLM-judge
    ratings (R5=1 -> type A, R5=5 -> type B). This is circular for a
    validation study --- raters who correctly disagree with the LLM
    judge would be penalised --- and is retained only behind the
    --allow-auto-attention-checks flag for backwards-compatible
    reproduction.
    """
    rng = random.Random(seed + 1)
    sorted_by_r5 = sorted(rows, key=lambda r: r["R5"])
    type_a_pool = [r for r in sorted_by_r5 if r["R5"] == 1]
    type_b_pool = [r for r in sorted_by_r5 if r["R5"] == 5]
    rng.shuffle(type_a_pool)
    rng.shuffle(type_b_pool)
    out: List[Dict[str, Any]] = []
    for r in type_a_pool[:n_each]:
        r["_attention_check"] = "A_intended_1_AUTO"
        out.append(r)
    for r in type_b_pool[:n_each]:
        r["_attention_check"] = "B_intended_5_AUTO"
        out.append(r)
    return out


def _format_dialogue_md(row: Dict[str, Any], payload: dict) -> str:
    state = payload["state"]
    misconception = state.get("misconception_label", "(unspecified)")
    turns = state.get("turn_history", []) or []
    lines = []
    lines.append(f"### Dialogue `{row['case_id']}` "
                 f"(rater-blind identifier: `{_blind_id(row)}`)")
    lines.append("")
    lines.append(f"**Misconception:** {misconception}")
    lines.append("")
    for t in turns:
        if isinstance(t, dict):
            speaker = t.get("speaker")
            text = t.get("text")
            idx = t.get("turn_index")
        else:
            speaker = t.speaker
            text = t.text
            idx = t.turn_index
        lines.append(f"**Turn {idx} ({speaker}):** {text}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**How much learning took place in this dialogue?**")
    lines.append("")
    lines.append("- (1) No learning evident")
    lines.append("- (2) Minimal learning")
    lines.append("- (3) Partial learning")
    lines.append("- (4) Substantial learning")
    lines.append("- (5) Complete learning")
    lines.append("")
    lines.append("Brief justification (optional, one sentence): _______")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _format_dialogue_html(row: Dict[str, Any], payload: dict) -> str:
    state = payload["state"]
    misconception = state.get("misconception_label", "(unspecified)")
    turns = state.get("turn_history", []) or []
    parts = []
    parts.append('<div class="dialogue" style="margin:1em 0; padding:1em; '
                 'border:1px solid #ccc; border-radius:6px;">')
    parts.append(f'<p><strong>Dialogue ID:</strong> '
                 f'<code>{_blind_id(row)}</code></p>')
    parts.append(f'<p><strong>Misconception:</strong> '
                 f'{_escape(misconception)}</p>')
    parts.append('<hr/>')
    for t in turns:
        if isinstance(t, dict):
            speaker = t.get("speaker")
            text = t.get("text")
            idx = t.get("turn_index")
        else:
            speaker = t.speaker
            text = t.text
            idx = t.turn_index
        parts.append(f'<p><strong>Turn {idx} ({speaker}):</strong> '
                     f'{_escape(text)}</p>')
    parts.append('<hr/>')
    parts.append('<p><strong>How much learning took place in this '
                 'dialogue?</strong></p>')
    parts.append('<ol>')
    for label in ("No learning evident", "Minimal learning",
                  "Partial learning", "Substantial learning",
                  "Complete learning"):
        parts.append(f'<li>{label}</li>')
    parts.append('</ol>')
    parts.append('<p>Brief justification (optional, one sentence): '
                 '________________________________________________</p>')
    parts.append('</div>')
    return "\n".join(parts)


def _blind_id(row: Dict[str, Any]) -> str:
    # Persistent rater-blind identifier: hash-shorten persona/arch/leg.
    import hashlib
    raw = f"{row['case_id']}|{row['persona']}|{row['architecture']}|{row['student_leg']}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"D{h}"


def _escape(s: Optional[str]) -> str:
    if s is None:
        return ""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br/>"))


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest_dir = manifest_path.parent
    out_dir = Path(args.out) if args.out else manifest_dir / "validation_subset"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _real_rows(manifest_path)
    print(f"Loaded {len(rows)} real-mode rows with R5 from manifest.")
    if not rows:
        print("ERROR: no rows to stratify.", file=sys.stderr)
        return 2

    # Decide attention-check source before stratifying the main sample,
    # because the attention-check pool is subtracted from the budget.
    if args.attention_checks_csv:
        attn = _attention_check_rows_from_csv(rows, Path(args.attention_checks_csv))
        print(f"Loaded {len(attn)} researcher-curated attention checks "
              f"from {args.attention_checks_csv}.")
    elif args.allow_auto_attention_checks:
        print("WARNING: --allow-auto-attention-checks is DEPRECATED. "
              "Auto-extracted attention checks (R5=1/R5=5) use the LLM "
              "judge's own ratings as ground truth, which penalises "
              "raters who legitimately disagree with the judge --- the "
              "disagreement signal this validation study is designed to "
              "measure. Use --attention-checks-csv with a hand-curated "
              "list instead. Proceeding under deprecation.",
              file=sys.stderr)
        attn = _attention_check_rows_auto(
            rows, args.auto_attention_checks_n // 2, args.seed,
        )
        print(f"Added {len(attn)} AUTO-extracted attention checks "
              f"(flagged with the AUTO suffix in the CSV).")
    else:
        attn = []
        print("NOTE: no attention checks added. Supply --attention-checks-csv "
              "with a researcher-curated list to enable rater-quality gating "
              "(strongly recommended). See VALIDATION_STUDY_MATERIALS.md §6.")

    # Stratified main sample (budget is what remains after attention checks).
    n_main = max(0, args.n - len(attn))
    selected = _stratify_and_sample(rows, manifest_dir, n_main, args.seed)
    print(f"Selected {len(selected)} main dialogues "
          f"(target {n_main}; stratified by persona × architecture × "
          f"difficulty_band × R5_band, with R5_band balanced across "
          f"low/mid/high to broaden the rater-visible rating range).")
    all_selected = selected + attn

    # Shuffle once for the master ordering (per-rater randomisation
    # happens in the survey tool).
    rng = random.Random(args.seed + 2)
    rng.shuffle(all_selected)

    # Export CSV (the analyst's record; held back from raters).
    csv_path = out_dir / "validation_subset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "blind_id", "case_id", "persona", "architecture", "student_leg",
            "llm_R5", "llm_quality_composite", "llm_R6",
            "difficulty_band", "r5_band", "word_count",
            "attention_check_type", "attention_check_rationale",
            "file_path",
        ])
        for row in all_selected:
            writer.writerow([
                _blind_id(row), row["case_id"], row["persona"],
                row["architecture"], row["student_leg"],
                row.get("R5"), row.get("quality_composite"), row.get("R6"),
                row.get("_difficulty_band"), row.get("_r5_band"),
                row.get("_word_count"),
                row.get("_attention_check") or "",
                row.get("_attention_rationale") or "",
                row.get("file_path", ""),
            ])

    # Export survey-ready Markdown.
    md_path = out_dir / "validation_subset.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Validation subset — dialogues for human rating\n\n")
        fh.write(f"_Generated by select_validation_subset.py. "
                 f"Total dialogues: {len(all_selected)}._\n\n")
        fh.write("Paste the contents below into your survey tool "
                 "(Qualtrics / Google Forms / LimeSurvey). Each "
                 "dialogue is presented with the same wording, the "
                 "5-point Likert scale, and a free-text "
                 "justification field. Identifiers are silent — keep "
                 "them in the survey's hidden metadata to match back "
                 "to the manifest.\n\n")
        fh.write("---\n\n")
        for row in all_selected:
            payload = _load_dialogue(manifest_dir, row)
            if payload is None:
                continue
            fh.write(_format_dialogue_md(row, payload))

    # Export survey-ready HTML.
    html_path = out_dir / "validation_subset.html"
    with html_path.open("w", encoding="utf-8") as fh:
        fh.write("<!doctype html><html><head><meta charset='utf-8'>"
                 "<title>Validation subset</title></head><body>\n")
        fh.write("<h1>Validation subset — dialogues for human rating</h1>\n")
        fh.write(f"<p><em>Total dialogues: {len(all_selected)}</em></p>\n")
        for row in all_selected:
            payload = _load_dialogue(manifest_dir, row)
            if payload is None:
                continue
            fh.write(_format_dialogue_html(row, payload))
        fh.write("</body></html>\n")

    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    print()
    print(f"Total dialogues in subset: {len(all_selected)} "
          f"({len(selected)} main + {len(attn)} attention checks).")
    print(f"Each dialogue should be rated by ≥3 raters; recommended "
          f"survey-tool design: present each rater with ~20 of these "
          f"dialogues drawn at random.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
