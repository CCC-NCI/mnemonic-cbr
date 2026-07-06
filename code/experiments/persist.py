"""Dialogue + score persistence.

Spec reference: IMPLEMENTATION_PLAN §6.4 — JSON-per-dialogue plus a
SQLite manifest for query-ability.

Why both:
- JSON-per-dialogue: each dialogue is a self-contained file that a
  human can open and read. Easy hand-inspection in Phase B and C;
  easy to ship as the public artefact (~26,500 files) for Phase E.
- SQLite manifest: one row per dialogue with the key columns and the
  rubric scores. Enables fast filtering ("all hybrid + experiential
  dialogues with R5 >= 4") without scanning thousands of JSON files.

Idempotent: writing the same (case_id, persona, architecture,
student_leg) overwrites the JSON file and upserts the SQLite row.

Stdlib only: json + sqlite3.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dialogue.state import DialogueState
    from scoring.rubric import RubricScore


# ---------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------

class _Encoder(json.JSONEncoder):
    """Encode numpy arrays and DialogueState/Turn objects cleanly."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)


def _state_to_dict(state) -> dict:
    """Convert a DialogueState to a plain dict for JSON serialisation."""
    return {
        "case_id": state.case_id,
        "misconception_label": state.misconception_label,
        "misconception_features": (
            state.misconception_features.tolist()
            if hasattr(state.misconception_features, "tolist")
            else list(state.misconception_features)
        ),
        "persona": state.persona,
        "architecture": state.architecture,
        "student_leg": state.student_leg,
        "max_turns": state.max_turns,
        "metadata": state.metadata,
        "turn_history": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "turn_index": t.turn_index,
            }
            for t in state.turn_history
        ],
    }


# ---------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS dialogues (
    case_id          TEXT NOT NULL,
    persona          TEXT NOT NULL,
    architecture     TEXT NOT NULL,
    student_leg      TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    written_at_utc   TEXT NOT NULL,

    misconception    TEXT,
    max_turns        INTEGER,
    turn_count       INTEGER,

    R1               INTEGER,
    R2               INTEGER,
    R3               INTEGER,
    R4               INTEGER,
    R5               INTEGER,
    R6               INTEGER,
    quality_composite REAL,
    judge_provider   TEXT,
    judge_model      TEXT,
    pass1_error      TEXT,
    pass2_error      TEXT,

    PRIMARY KEY (case_id, persona, architecture, student_leg)
);
CREATE INDEX IF NOT EXISTS idx_persona      ON dialogues(persona);
CREATE INDEX IF NOT EXISTS idx_architecture ON dialogues(architecture);
CREATE INDEX IF NOT EXISTS idx_R5           ON dialogues(R5);
"""


_UPSERT_SQL = """\
INSERT INTO dialogues (
    case_id, persona, architecture, student_leg, file_path,
    written_at_utc, misconception, max_turns, turn_count,
    R1, R2, R3, R4, R5, R6, quality_composite,
    judge_provider, judge_model, pass1_error, pass2_error
) VALUES (
    :case_id, :persona, :architecture, :student_leg, :file_path,
    :written_at_utc, :misconception, :max_turns, :turn_count,
    :R1, :R2, :R3, :R4, :R5, :R6, :quality_composite,
    :judge_provider, :judge_model, :pass1_error, :pass2_error
)
ON CONFLICT (case_id, persona, architecture, student_leg) DO UPDATE SET
    file_path          = excluded.file_path,
    written_at_utc     = excluded.written_at_utc,
    misconception      = excluded.misconception,
    max_turns          = excluded.max_turns,
    turn_count         = excluded.turn_count,
    R1                 = excluded.R1,
    R2                 = excluded.R2,
    R3                 = excluded.R3,
    R4                 = excluded.R4,
    R5                 = excluded.R5,
    R6                 = excluded.R6,
    quality_composite  = excluded.quality_composite,
    judge_provider     = excluded.judge_provider,
    judge_model        = excluded.judge_model,
    pass1_error        = excluded.pass1_error,
    pass2_error        = excluded.pass2_error
;
"""


# ---------------------------------------------------------------------
# DialogueStore — primary API
# ---------------------------------------------------------------------

class DialogueStore:
    """Manages JSON files + the SQLite manifest under a root directory.

    Typical use:

        store = DialogueStore(root="results/phaseB_plumbing")
        for cell in cells:
            state = run_dialogue(...)
            score = scorer.score(state)
            store.write(state, score)
        store.close()

    Idempotent: re-running the same cell overwrites the JSON file and
    upserts the manifest row. Safe to point at an existing directory.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.dialogues_dir = self.root / "dialogues"
        self.dialogues_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "manifest.sqlite"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "DialogueStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- write ----------

    def write(self, state, score: Optional["RubricScore"] = None) -> Path:
        """Persist a dialogue (+ optional score) and update the manifest."""
        filename = self._filename_for(state)
        file_path = self.dialogues_dir / filename

        payload = {
            "state": _state_to_dict(state),
            "score": score.to_dict() if score is not None else None,
            "written_at_utc": _utc_now(),
        }
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, cls=_Encoder)

        self._upsert_manifest(state, score, file_path)
        return file_path

    # ---------- read ----------

    def read(
        self,
        case_id: str,
        persona: str,
        architecture: str,
        student_leg: str,
    ) -> Optional[dict]:
        filename = _filename_for_keys(case_id, persona, architecture, student_leg)
        path = self.dialogues_dir / filename
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM dialogues")
        return int(cur.fetchone()[0])

    def query(self, sql: str, params: tuple = ()) -> list:
        """Run a SELECT against the manifest. Read-only convenience."""
        cur = self._conn.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ---------- internals ----------

    def _filename_for(self, state) -> str:
        return _filename_for_keys(
            state.case_id, state.persona, state.architecture, state.student_leg
        )

    def _upsert_manifest(self, state, score, file_path: Path) -> None:
        row = {
            "case_id": state.case_id,
            "persona": state.persona,
            "architecture": state.architecture,
            "student_leg": state.student_leg,
            "file_path": str(file_path.relative_to(self.root)),
            "written_at_utc": _utc_now(),
            "misconception": state.misconception_label,
            "max_turns": state.max_turns,
            "turn_count": len(state.turn_history),
            "R1": None, "R2": None, "R3": None,
            "R4": None, "R5": None, "R6": None,
            "quality_composite": None,
            "judge_provider": None,
            "judge_model": None,
            "pass1_error": None,
            "pass2_error": None,
        }
        if score is not None:
            row.update(
                R1=score.R1, R2=score.R2, R3=score.R3,
                R4=score.R4, R5=score.R5, R6=score.R6,
                quality_composite=score.quality_composite,
                judge_provider=score.judge_provider,
                judge_model=score.judge_model,
                pass1_error=score.pass1_error or None,
                pass2_error=score.pass2_error or None,
            )
        self._conn.execute(_UPSERT_SQL, row)
        self._conn.commit()


# ---------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------

def _safe(component: str) -> str:
    """Lightly sanitise a filename component."""
    return component.replace("/", "_").replace(" ", "_")


def _filename_for_keys(
    case_id: str, persona: str, architecture: str, student_leg: str
) -> str:
    return (
        f"{_safe(case_id)}__{_safe(persona)}__{_safe(architecture)}__"
        f"{_safe(student_leg)}.json"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
