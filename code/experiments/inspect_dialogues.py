"""Inspect persisted dialogues — Markdown dump for easy reading.

Reads a Phase B/D SQLite manifest, joins to the persisted JSON dialogue
files, and writes a single Markdown report with each dialogue's turns,
rubric scores, and judge justifications.

Default output path: {manifest_dir}/dialogues_inspection.md

Default filter: real (rows where judge_provider != 'stub'). Most of the
time you want to see the dialogues you actually paid for; use
--filter all or --filter stub to include / restrict to stub rows.

Usage:

    # All real dialogues from a Phase B run
    python code/experiments/inspect_dialogues.py \\
        --manifest results/phaseB_smoke/manifest.sqlite

    # Only socratic + hybrid dialogues
    python code/experiments/inspect_dialogues.py \\
        --manifest results/phaseB_smoke/manifest.sqlite \\
        --persona socratic --architecture hybrid

    # One specific cell
    python code/experiments/inspect_dialogues.py \\
        --manifest results/phaseB_smoke/manifest.sqlite \\
        --case-id case_0 --persona socratic --architecture baseline --leg leg_a
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Optional


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
        help="Path to SQLite manifest "
        "(default: results/phaseB_smoke/manifest.sqlite, matching the "
        "default --out of run_phaseB_smoke.py)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output Markdown file. Default: {manifest_dir}/dialogues_inspection.md",
    )
    p.add_argument(
        "--filter",
        choices=("real", "stub", "all"),
        default="real",
        help="Which rows to include (default: real, i.e. judge_provider != 'stub')",
    )
    p.add_argument("--case-id", default=None, help="Filter by case_id")
    p.add_argument("--persona", default=None, help="Filter by persona")
    p.add_argument("--architecture", default=None, help="Filter by architecture")
    p.add_argument("--leg", default=None, help="Filter by student_leg")
    p.add_argument(
        "--max",
        type=int,
        default=None,
        help="Cap the number of dialogues rendered (after filtering)",
    )
    return p.parse_args()


def write_inspection_md(
    manifest,
    out=None,
    filter_mode: str = "real",
    case_id: Optional[str] = None,
    persona: Optional[str] = None,
    architecture: Optional[str] = None,
    leg: Optional[str] = None,
    max_count: Optional[int] = None,
) -> Optional[Path]:
    """Programmatic entry point — same effect as the CLI without argparse.

    Returns the written path, or None if the manifest is missing or empty.
    """
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        return None
    manifest_dir = manifest_path.parent
    out_path = Path(out) if out else manifest_dir / "dialogues_inspection.md"

    # Mirror the argparse Namespace shape that fetch_rows + load_dialogue_json expect.
    class _Args:
        pass

    args = _Args()
    args.filter = filter_mode
    args.case_id = case_id
    args.persona = persona
    args.architecture = architecture
    args.leg = leg
    args.max = max_count

    rows = fetch_rows(manifest_path, args)

    header_lines = [
        "# Dialogue inspection",
        "",
        f"- Generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Manifest: `{manifest_path}`",
        f"- Filter: `{filter_mode}`"
        + (f", case_id=`{case_id}`" if case_id else "")
        + (f", persona=`{persona}`" if persona else "")
        + (f", architecture=`{architecture}`" if architecture else "")
        + (f", leg=`{leg}`" if leg else ""),
        f"- Dialogues rendered: {len(rows)}",
        "",
        "---",
        "",
    ]

    if not rows:
        header_lines.append("_No dialogues matched the filter._")
        out_path.write_text("\n".join(header_lines), encoding="utf-8")
        return out_path

    body: list = []
    skipped = 0
    for row in rows:
        payload = load_dialogue_json(manifest_dir, row)
        if payload is None:
            skipped += 1
            continue
        body.append(render_dialogue_md(row, payload))

    out_path.write_text(
        "\n".join(header_lines) + "\n".join(body),
        encoding="utf-8",
    )
    return out_path


def fetch_rows(manifest_path: Path, args: argparse.Namespace) -> List[dict]:
    conn = sqlite3.connect(str(manifest_path))
    conn.row_factory = sqlite3.Row
    where = []
    params: List = []
    if args.filter == "real":
        where.append("LOWER(COALESCE(judge_provider, '')) != 'stub'")
    elif args.filter == "stub":
        where.append("LOWER(COALESCE(judge_provider, '')) = 'stub'")
    if args.case_id:
        where.append("case_id = ?")
        params.append(args.case_id)
    if args.persona:
        where.append("persona = ?")
        params.append(args.persona)
    if args.architecture:
        where.append("architecture = ?")
        params.append(args.architecture)
    if args.leg:
        where.append("student_leg = ?")
        params.append(args.leg)
    sql = "SELECT * FROM dialogues"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY case_id, persona, architecture, student_leg"
    if args.max:
        sql += f" LIMIT {int(args.max)}"
    cursor = conn.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def load_dialogue_json(manifest_dir: Path, row: dict) -> Optional[dict]:
    file_path = manifest_dir / row.get("file_path", "")
    if not file_path.exists():
        # Try the conventional naming as a fallback.
        from experiments.persist import _filename_for_keys

        file_path = manifest_dir / "dialogues" / _filename_for_keys(
            row["case_id"], row["persona"], row["architecture"], row["student_leg"]
        )
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def render_dialogue_md(row: dict, payload: dict) -> str:
    state = payload.get("state", {})
    score = payload.get("score") or {}
    turns = state.get("turn_history", [])

    lines: List[str] = []
    lines.append(
        f"## {row['case_id']} / {row['persona']} / {row['architecture']} / {row['student_leg']}"
    )
    lines.append("")
    misconception = state.get("misconception_label") or row.get("misconception", "")
    lines.append(f"**Misconception:** {misconception}")
    lines.append("")
    lines.append(
        f"**Judge:** {score.get('judge_provider', '?')} "
        f"(model: {score.get('judge_model', '?')})"
    )
    lines.append("")
    lines.append("### Turns")
    lines.append("")
    for t in turns:
        lines.append(f"**Turn {t['turn_index']} [{t['speaker']}]**")
        lines.append("")
        lines.append(_md_escape_block(t["text"]))
        lines.append("")
    lines.append("### Rubric")
    lines.append("")
    lines.append("| Item | Score |")
    lines.append("|---|---|")
    rubric_labels = [
        ("R1", "R1 misconception engagement"),
        ("R2", "R2 cognitive demand"),
        ("R3", "R3 scaffolding fit"),
        ("R4", "R4 domain accuracy"),
        ("R5", "R5 student trajectory"),
        ("R6", "R6 strategy fidelity"),
    ]
    for key, label in rubric_labels:
        v = score.get(key)
        lines.append(f"| {label} | {_fmt_int(v)} |")
    q = score.get("quality_composite")
    lines.append(f"| Quality composite (R1+R2+R3)/3 | {_fmt_float(q)} |")
    lines.append("")

    j1 = score.get("justification_pass1") or ""
    j2 = score.get("justification_pass2") or ""
    if j1 or j2:
        lines.append("### Judge justifications")
        lines.append("")
        if j1:
            lines.append(f"**Pass 1 (R1–R5, persona-blind):** {j1}")
            lines.append("")
        if j2:
            lines.append(f"**Pass 2 (R6, persona-visible):** {j2}")
            lines.append("")

    err1 = score.get("pass1_error") or ""
    err2 = score.get("pass2_error") or ""
    if err1 or err2:
        lines.append("### Parsing errors")
        lines.append("")
        if err1:
            lines.append(f"- Pass 1: {err1}")
        if err2:
            lines.append(f"- Pass 2: {err2}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _md_escape_block(text: str) -> str:
    """Render a turn's text as a Markdown quoted block, preserving LaTeX
    fragments without breaking the rendering."""
    if not text:
        return "> (empty)"
    # Quote each line. Use a Markdown blockquote (`> `) prefix.
    return "\n".join(f"> {line}" for line in text.splitlines())


def _fmt_int(v) -> str:
    if v is None:
        return "—"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return str(v)


def _fmt_float(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    # Log under <out_dir>/logs/; if --out was given as a file path, use
    # its parent, else default to the manifest's directory.
    log_dir = (
        Path(args.out).parent if args.out else manifest_path.parent
    )
    set_log_path(log_dir / "logs" / f"inspect_dialogues_{utc_stamp()}.txt")

    out_path = write_inspection_md(
        manifest=manifest_path,
        out=args.out,
        filter_mode=args.filter,
        case_id=args.case_id,
        persona=args.persona,
        architecture=args.architecture,
        leg=args.leg,
        max_count=args.max,
    )
    if out_path is None:
        print("ERROR: nothing written.", file=sys.stderr)
        return 2
    # Count rows for the friendly message.
    rows = fetch_rows(manifest_path, args)
    print(f"Wrote {out_path} ({len(rows)} dialogues)")
    return 0


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="inspect_dialogues"))
