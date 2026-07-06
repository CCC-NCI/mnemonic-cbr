"""DialogueState, Turn, and CaseExt dataclasses.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.1.

DialogueState carries the full state of a single (case, persona,
architecture, student_leg) dialogue. Turn budget is fixed at 4 turns:
student(0) → teacher(1) → student(2) → teacher(3). No early termination.

CaseExt extends the legacy Case dataclass with two optional fields
needed by the spec's deterministic turn-1 template (§3.2):
problem_text and student_answer_text. The legacy Case is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Import the legacy Case as the underlying record. This is the central
# reuse from the v1 code per IMPLEMENTATION_PLAN §3.1.
try:
    from cbr.mnemonic_augmentation import Case as _LegacyCase
except ImportError:
    # Allow this module to be imported even when sys.path doesn't include
    # the legacy package — useful for unit tests of state alone.
    _LegacyCase = None  # type: ignore


@dataclass
class CaseExt:
    """Wraps a legacy Case with the two text fields the spec's turn-1
    template needs.

    Why a wrapper rather than editing the legacy Case: the legacy file
    stays untouched as reviewer reference. CaseExt is the unit that
    flows through the dialogue layer.
    """

    case: Any  # legacy Case object (typed Any to keep this importable standalone)
    problem_text: str = ""
    student_answer_text: str = ""

    # Convenience pass-throughs so callers can write case_ext.misconception
    # rather than case_ext.case.misconception.
    @property
    def id(self) -> str:
        return self.case.id

    @property
    def features(self) -> np.ndarray:
        return self.case.features

    @property
    def misconception(self) -> str:
        return self.case.misconception

    @property
    def intervention(self) -> Dict:
        return self.case.intervention


@dataclass
class Turn:
    """A single utterance in a dialogue.

    speaker:    "student" or "teacher"
    text:       the utterance
    turn_index: 0-based position; 0 is the deterministic student turn
    """

    speaker: str
    text: str
    turn_index: int


@dataclass
class DialogueState:
    """All state for a single dialogue.

    Fields mirror spec §3.1 exactly, with the addition of a metadata
    dict for provenance (timestamps, model versions, seeds — populated
    by the orchestrator, not the dialogue layer).
    """

    case_id: str
    misconception_label: str
    misconception_features: np.ndarray
    persona: str
    architecture: str
    student_leg: str  # "leg_a" or "leg_b"
    turn_history: List[Turn] = field(default_factory=list)
    max_turns: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Helpers used by the dialogue loop and prompt builders ---

    def history_as_text(self) -> str:
        """Render the turn history as a transcript for prompt insertion."""
        if not self.turn_history:
            return "(no turns yet)"
        lines = []
        for turn in self.turn_history:
            lines.append(f"  Turn {turn.turn_index} ({turn.speaker}): {turn.text}")
        return "\n".join(lines)

    def is_complete(self) -> bool:
        return len(self.turn_history) >= self.max_turns

    def whose_turn(self) -> str:
        """Whose turn is it to speak next? Strict alternation starting
        with the student at index 0.

        index 0 → student, index 1 → teacher, index 2 → student, ...
        """
        next_index = len(self.turn_history)
        return "student" if next_index % 2 == 0 else "teacher"


# Re-export for convenience
__all__ = ["CaseExt", "Turn", "DialogueState"]
