"""Four-turn dialogue orchestrator.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.1.

Fixed turn budget. Alternation pattern:
    Turn 0: student  (deterministic, no LLM)
    Turn 1: teacher
    Turn 2: student
    Turn 3: teacher

No early termination. Every (case, persona, architecture, student_leg)
cell produces a dialogue of identical length, removing length confounds.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Optional

from dialogue.state import DialogueState, Turn
from dialogue.student import initial_student_utterance

if TYPE_CHECKING:
    from dialogue.state import CaseExt
    from dialogue.student import StudentSimulator
    from dialogue.teacher import TeacherGenerator


def run_dialogue(
    case_ext: "CaseExt",
    persona: str,
    architecture: str,
    student_leg: str,
    teacher: "TeacherGenerator",
    student: "StudentSimulator",
    max_turns: int = 4,
    seed: Optional[int] = None,
) -> DialogueState:
    """Run a single dialogue end-to-end and return the final state.

    Spec §3.1 fixes max_turns=4. The argument is exposed for the
    documented contingency in §3.1 to extend to 6 turns if Phase C
    persona differentiation is weak; not for run-by-run variation.
    """
    state = DialogueState(
        case_id=case_ext.id,
        misconception_label=case_ext.misconception,
        misconception_features=case_ext.features,
        persona=persona,
        architecture=architecture,
        student_leg=student_leg,
        max_turns=max_turns,
        metadata={
            "started_at": _dt.datetime.utcnow().isoformat() + "Z",
            "seed": seed,
            "teacher_temperature": teacher.temperature,
            "student_temperature": student.temperature,
            "k_retrieve": teacher.k_retrieve,
            "student_mode": getattr(student, "student_mode", "pure_ai"),
        },
    )

    # Turn 0: deterministic student utterance, no LLM.
    state.turn_history.append(
        Turn(speaker="student", text=initial_student_utterance(case_ext), turn_index=0)
    )

    # Turns 1..max_turns-1: alternate teacher / student.
    while not state.is_complete():
        next_index = len(state.turn_history)
        whose = state.whose_turn()
        if whose == "teacher":
            text = teacher.next_utterance(state, case_ext)
            state.turn_history.append(
                Turn(speaker="teacher", text=text, turn_index=next_index)
            )
        else:
            text = student.next_utterance(state, case_ext)
            state.turn_history.append(
                Turn(speaker="student", text=text, turn_index=next_index)
            )

    state.metadata["ended_at"] = _dt.datetime.utcnow().isoformat() + "Z"
    return state
