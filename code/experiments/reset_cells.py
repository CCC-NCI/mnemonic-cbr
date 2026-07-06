"""Reset (delete) selected cells from the manifest + JSON files.

Lets you surgically clear a subset of dialogues so that the next
run_phaseB_smoke.py invocation regenerates them. Filters match
inspect_dialogues.py for consistency.

Dry-run by default — prints what would be deleted but doesn't touch
anything. Pass --apply to actually delete.

Usage:

    # See what would be deleted (no destruction)
    python code/experiments/reset_cells.py \\
        --persona socratic --architecture-prefix pure_cbr

    # Actually delete (real-mode socratic pure_cbr_* cells)
    python code/experiments/reset_cells.py \\
        --persona socratic --architecture-prefix pure_cbr --apply

    # Reset everything in the manifest (use with care!)
    python code/experiments/reset_cells.py --filter all --apply

    # Reset stub rows only (clean slate before a real-mode run)
    python code/experiments/reset_cells.py --filter stub --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--manifest",
        default="results/phaseB_smoke/manifest.sqlite",
        help="Path to SQLite manifest "
        "(default: results/phaseB_smoke/manifest.sqlite)",
    )
    p.add_argument(
        "--filter",
        choices=("real", "stub", "all"),
        default="real",
        help="Which rows are candidates for deletion (default: real)",
    )
    p.add_argument("--case-id", default=None, help="Filter by case_id")
    p.add_argument("--persona", default=None, help="Filter by persona")
    p.add_argument("--architecture", default=None, help="Filter by exact architecture")
    p.add_argument(
        "--architecture-prefix",
        default=None,
        help="Filter by architecture LIKE 'PREFIX%%' (e.g. pure_cbr matches "
        "pure_cbr_llm and pure_cbr_tpl)",
    )
    p.add_argument("--leg", default=None, help="Filter by student_leg")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the deletion (default is dry-run)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest_dir = manifest_path.parent

    where: List[str] = []
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
    if args.architecture_prefix:
        where.append("architecture LIKE ?")
        params.append(args.architecture_prefix + "%")
    if args.leg:
        where.append("student_leg = ?")
        params.append(args.leg)

    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    select_sql = (
        "SELECT case_id, persona, architecture, student_leg, file_path, "
        "judge_provider "
        "FROM dialogues" + where_clause +
        " ORDER BY case_id, persona, architecture, student_leg"
    )
    delete_sql = "DELETE FROM dialogues" + where_clause

    conn = sqlite3.connect(str(manifest_path))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(select_sql, params).fetchall())

    print(f"Manifest: {manifest_path}")
    print(f"Filter: {args.filter}"
          + (f", case_id={args.case_id}" if args.case_id else "")
          + (f", persona={args.persona}" if args.persona else "")
          + (f", architecture={args.architecture}" if args.architecture else "")
          + (f", architecture-prefix={args.architecture_prefix}" if args.architecture_prefix else "")
          + (f", leg={args.leg}" if args.leg else ""))
    print(f"Matched: {len(rows)} row(s)")
    print()

    if not rows:
        print("Nothing to do.")
        conn.close()
        return 0

    for r in rows:
        judge = r["judge_provider"] or ""
        print(f"  {r['case_id']:>10s} / {r['persona']:>13s} / "
              f"{r['architecture']:>14s} / {r['student_leg']:>5s} "
              f"(judge: {judge})")
    print()

    if not args.apply:
        print("Dry-run (no changes). Re-run with --apply to delete.")
        conn.close()
        return 0

    # Delete JSON files first (so a crash mid-deletion leaves a recoverable state).
    deleted_files = 0
    missing_files = 0
    for r in rows:
        if not r["file_path"]:
            continue
        p = manifest_dir / r["file_path"]
        if p.exists():
            p.unlink()
            deleted_files += 1
        else:
            missing_files += 1

    # Then delete manifest rows.
    cur = conn.execute(delete_sql, params)
    deleted_rows = cur.rowcount
    conn.commit()
    conn.close()

    print(f"Deleted {deleted_rows} manifest row(s); {deleted_files} JSON file(s)"
          + (f"; {missing_files} JSON file(s) already missing" if missing_files else "")
          + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
