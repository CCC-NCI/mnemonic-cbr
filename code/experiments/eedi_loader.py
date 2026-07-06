"""EEDI → CaseExt loader for the rebuild.

The data file in mnemonic-cbr/data/all_train.csv is question-level
metadata (one row per question, 10,842 rows). Columns observed:

  QuestionId, ConstructId, ConstructName, SubjectId, SubjectName,
  CorrectAnswer (one of 'A'-'D'), QuestionText,
  AnswerAText, AnswerBText, AnswerCText, AnswerDText,
  MisconceptionAId..D, MisconceptionAName..D,
  source, OriginalQuestionId.

Each row encodes the question plus, for each wrong answer letter, the
misconception that wrong answer demonstrates. So a single row gives
us up to three distinct "student picked answer X with misconception Y"
cases — one per wrong-answer letter that has a non-empty misconception.

For the rebuild, a CaseExt represents one such (question, wrong
answer, misconception) triple. The legacy create_case_from_eedi
collapses to "MisconceptionAName" specifically; here we expand the
choice to pick the first available wrong-answer-with-misconception.

This module exists in experiments/ rather than dialogue/ because it
bridges between the dataset format and the dialogue layer's data
model. Loaders are experiment-layer concerns.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from dialogue.state import CaseExt

# Lazy import of the legacy Case so this module is importable even
# when the cbr/ package isn't on the path.
def _legacy_case(**kwargs):
    from cbr.mnemonic_augmentation import Case
    return Case(**kwargs)


_ANSWER_LETTERS = ("A", "B", "C", "D")


def _safe_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _pick_wrong_answer(row: pd.Series) -> Optional[Tuple[str, str, str]]:
    """Return (letter, answer_text, misconception_name) for the first
    wrong answer in the row that has a non-empty misconception. None
    if the row has no usable wrong-answer-with-misconception.
    """
    correct = _safe_str(row.get("CorrectAnswer")).upper()
    for letter in _ANSWER_LETTERS:
        if letter == correct:
            continue
        misc_name = _safe_str(row.get(f"Misconception{letter}Name"))
        if not misc_name:
            continue
        answer_text = _safe_str(row.get(f"Answer{letter}Text"))
        if not answer_text:
            continue
        return letter, answer_text, misc_name
    return None


def _build_intervention_text(
    construct_name: str,
    correct_answer_letter: str,
    correct_answer_text: str,
    wrong_answer_text: str,
    misconception_name: str,
) -> str:
    """Synthesise a substantive intervention description from the
    EEDI row data.

    This is option (c) from IMPLEMENTATION_PLAN: deterministic synthesis
    from EEDI columns. No LLM, no hand-written exemplars. Honest about
    its source: every field is from the EEDI question metadata.

    The output is a short, content-rich string that gives the
    pure_cbr_llm and hybrid teacher architectures something real to
    render. It is intentionally a description of the case rather than
    pedagogical advice — the teacher LLM still does the pedagogical
    work; this is the substrate.
    """
    parts = []
    if construct_name:
        parts.append(f"Topic: {construct_name}.")
    if correct_answer_text:
        parts.append(
            f"Correct answer ({correct_answer_letter or '?'}): "
            f"{correct_answer_text}."
        )
    if wrong_answer_text:
        parts.append(
            f"Student selected: {wrong_answer_text}."
        )
    if misconception_name:
        parts.append(
            f"Underlying misconception: {misconception_name}."
        )
    parts.append(
        "Intervention focus: surface why the selected answer is wrong "
        "and walk through the reasoning that leads to the correct answer."
    )
    return " ".join(parts)


def _features_for(row: pd.Series, misconception_count: int) -> np.ndarray:
    """Same 4-feature layout as the legacy create_case_from_eedi (which
    is what MnemonicAugmentation expects)."""

    def safe_numeric(value, default=0):
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    return np.array(
        [
            safe_numeric(row.get("QuizId", 0)) % 1000,
            safe_numeric(row.get("QuestionId", 0)) % 1000,
            float(misconception_count),
            float(len(_safe_str(row.get("ConstructName")))),
        ],
        dtype=np.float64,
    )


def case_ext_from_row(row: pd.Series, case_id: str) -> Optional[CaseExt]:
    """Build a CaseExt from one EEDI row. Returns None if the row has
    no usable wrong-answer-with-misconception.
    """
    picked = _pick_wrong_answer(row)
    if picked is None:
        return None
    wrong_letter, wrong_answer_text, misconception_name = picked

    # Count populated misconceptions for the 4-feature vector.
    misconception_count = sum(
        1 for letter in _ANSWER_LETTERS
        if _safe_str(row.get(f"Misconception{letter}Name"))
    )

    features = _features_for(row, misconception_count)

    # Extract the correct-answer text for the intervention content.
    correct_letter = _safe_str(row.get("CorrectAnswer")).upper()
    correct_answer_text = ""
    if correct_letter in _ANSWER_LETTERS:
        correct_answer_text = _safe_str(row.get(f"Answer{correct_letter}Text"))
    construct_name = _safe_str(row.get("ConstructName"))

    intervention_text = _build_intervention_text(
        construct_name=construct_name,
        correct_answer_letter=correct_letter,
        correct_answer_text=correct_answer_text,
        wrong_answer_text=wrong_answer_text,
        misconception_name=misconception_name,
    )

    legacy = _legacy_case(
        id=case_id,
        features=features,
        misconception=misconception_name,
        intervention={
            "strategy": construct_name or "unknown",
            "complexity": len(construct_name),
            "construct_name": construct_name,
            "correct_answer_letter": correct_letter,
            "correct_answer_text": correct_answer_text,
            "wrong_answer_letter": wrong_letter,
            "wrong_answer_text": wrong_answer_text,
            "misconception_name": misconception_name,
            "intervention_text": intervention_text,
            "source": "deterministic_from_eedi_columns",
        },
        # outcome and utility_score are set to neutral defaults. The
        # cleaned mnemonic engine (dialogue.retrieval) no longer
        # consults these in retrieval shaping.
        outcome=0.5,
        utility_score=0.5,
    )

    return CaseExt(
        case=legacy,
        problem_text=_safe_str(row.get("QuestionText")),
        student_answer_text=wrong_answer_text,
    )


def load_first_usable_case(
    filepath: str = "data/all_train.csv",
) -> CaseExt:
    """Phase A helper: load the first row from the EEDI CSV that has a
    usable wrong-answer-with-misconception, and return it as a CaseExt.

    Used by run_phaseA_smoke.py to demonstrate the loop end-to-end on
    one real case.
    """
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        case_ext = case_ext_from_row(row, case_id=f"case_{idx}")
        if case_ext is not None:
            return case_ext
    raise RuntimeError(
        f"No usable wrong-answer-with-misconception found in {filepath}"
    )


def load_n_usable_cases(n: int, filepath: str = "data/all_train.csv") -> List[CaseExt]:
    """Phase B+ helper: return the first n usable CaseExt objects.

    Not used in Phase A; included here so Phase B doesn't have to add
    another loader module.
    """
    df = pd.read_csv(filepath)
    out: List[CaseExt] = []
    for idx, row in df.iterrows():
        case_ext = case_ext_from_row(row, case_id=f"case_{idx}")
        if case_ext is not None:
            out.append(case_ext)
            if len(out) >= n:
                break
    return out
