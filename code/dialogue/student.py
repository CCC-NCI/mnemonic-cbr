"""Student layer.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.2, §3.5.

Two public surfaces:

- initial_student_utterance(case_ext) -> str
    Deterministic turn-0 utterance. No LLM. Uses the spec template
    "I have {problem}. I got {answer} because {reasoning}." with
    reasoning drawn from reasoning_templates.template_for().

- class StudentSimulator
    Wraps an LLMProvider for turns 2, 4, 6, ... (everything after
    turn 0). Spec temperature 0.3.

    Two student modes are supported:
      * "pure_ai" (default): the student LLM receives only the
        misconception label and the dialogue history. This is the
        original spec design.
      * "cbr_grounded": before each student LLM call, K=2 cases with
        the same misconception family are retrieved from the case
        base and surfaced in the prompt as "examples of other students
        with this misconception". The student is instructed to keep
        the misconception structure of those real examples. This
        reduces LLM-prior dependence in the student turn.

    The two modes are compared as an A/B test (see spec §3.5 and
    manuscript §4.5). The default is pure_ai so that no existing
    runs are silently invalidated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from dialogue.reasoning_templates import template_for

if TYPE_CHECKING:  # avoid hard import for static analysis
    from dialogue.state import CaseExt, DialogueState
    from dialogue.llm_provider import LLMProvider
    from cbr.mnemonic_augmentation import Case, MnemonicAugmentation


STUDENT_MODES = ("pure_ai", "cbr_grounded")


_STUDENT_PROMPT_TEMPLATE = """\
You are simulating a UK secondary school student (Years 7-11, ages
11-16) who currently holds the following mathematical misconception:

  Misconception label: {misconception_label}
  Example error: {example_from_eedi}
{cbr_grounding_block}
You are talking with a teacher who is trying to help you understand
the topic better.

Conversation so far:
{turn_history_as_text}

Respond as the student would respond next. Your response should:
- be 1-3 sentences in the voice of an 11-16 year old
- reflect your current understanding, including any partial
  correction the teacher has prompted
- ask questions or express confusion where realistic
- NEVER suddenly claim full understanding unless the teacher has
  walked you through the correction explicitly

Your next utterance:"""


_CBR_GROUNDING_HEADER = """
Examples of other students with the same misconception
(drawn from the case base; their wrong answers below show how this
misconception typically manifests):
"""


_CBR_GROUNDING_FOOTER = """
Keep the misconception structure visible in your responses.
Do not suddenly claim full understanding unless the teacher has
walked you through the correction explicitly.
"""


def initial_student_utterance(case_ext) -> str:
    """Build the deterministic turn-0 student utterance.

    Spec §3.2 template:
        "I have {case.problem}. I got {case.student_answer} because
         {brief_reasoning_template}."

    If problem_text or student_answer_text are empty (e.g., when the
    CaseExt wasn't populated from EEDI properly), fall back to a
    misconception-label-only utterance so the smoke test still works.
    """
    reasoning = template_for(case_ext.misconception)
    if case_ext.problem_text and case_ext.student_answer_text:
        # Single-line the problem text to keep the turn-0 utterance
        # readable. The full QuestionText sometimes spans multiple
        # lines in EEDI; we collapse whitespace.
        problem = " ".join(case_ext.problem_text.split())
        # Add a period after the problem only if it doesn't already
        # end with terminal punctuation (EEDI questions often end in '?').
        if not problem.endswith((".", "?", "!")):
            problem = problem + "."
        answer = case_ext.student_answer_text.strip().rstrip(".")
        return f"I have: {problem} I got {answer} because {reasoning}"
    # Fallback when case is sparse — surfaces in unit tests with bare Cases.
    return f"I'm working on a problem about {case_ext.misconception}, and {reasoning}"


class StudentSimulator:
    """Generates student utterances for turns after turn 0.

    Construction params:
      provider:           LLMProvider for the student model.
      leg_name:           "leg_a" or "leg_b" (purely a label).
      temperature:        0.3 by spec.
      student_mode:       "pure_ai" (default) or "cbr_grounded".
      mnemonic_engine:    Required if student_mode == "cbr_grounded".
      case_base:          Required if student_mode == "cbr_grounded".
      k_grounding:        Number of same-misconception cases to surface
                          in the prompt under cbr_grounded mode.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        leg_name: str = "leg_a",
        temperature: float = 0.3,
        student_mode: str = "pure_ai",
        mnemonic_engine: Optional["MnemonicAugmentation"] = None,
        case_base: Optional[List["Case"]] = None,
        k_grounding: int = 2,
    ):
        if student_mode not in STUDENT_MODES:
            raise ValueError(
                f"Unknown student_mode {student_mode!r}; "
                f"expected one of {STUDENT_MODES}"
            )
        if student_mode == "cbr_grounded":
            if mnemonic_engine is None or not case_base:
                raise ValueError(
                    "cbr_grounded student_mode requires both mnemonic_engine "
                    "and case_base; pass them in or use student_mode='pure_ai'"
                )
        self.provider = provider
        self.leg_name = leg_name
        self.temperature = temperature
        self.student_mode = student_mode
        self.mnemonic_engine = mnemonic_engine
        self.case_base = case_base or []
        self.k_grounding = k_grounding

    def next_utterance(self, state: "DialogueState", case_ext) -> str:
        prompt = self._build_prompt(state, case_ext)
        return self.provider.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=200,
        )

    def _build_prompt(self, state: "DialogueState", case_ext) -> str:
        if case_ext.problem_text and case_ext.student_answer_text:
            example = (
                f"Problem: {' '.join(case_ext.problem_text.split())} | "
                f"Student's wrong answer: {case_ext.student_answer_text}"
            )
        else:
            example = "(no specific example provided)"

        grounding_block = ""
        if self.student_mode == "cbr_grounded":
            retrieved = self._retrieve_grounding(case_ext)
            grounding_block = self._format_grounding(retrieved)

        return _STUDENT_PROMPT_TEMPLATE.format(
            misconception_label=state.misconception_label,
            example_from_eedi=example,
            cbr_grounding_block=grounding_block,
            turn_history_as_text=state.history_as_text(),
        )

    # ---- CBR grounding (only used when student_mode == "cbr_grounded") ----

    def _retrieve_grounding(self, case_ext) -> List:
        """Return top-K cases for student grounding.

        Uses the same engine + similarity as the teacher, but selects
        cases that share the misconception family. Falls back to plain
        top-K if the misconception filter yields too few hits.
        """
        if not self.mnemonic_engine or not self.case_base:
            return []
        from cbr.mnemonic_augmentation import Case

        query = Case(
            id=f"student_query_{case_ext.id}",
            features=case_ext.features,
            misconception=case_ext.misconception,
            intervention={},
            outcome=0.5,
            utility_score=0.5,
        )
        # Score every non-self case.
        scored = []
        for c in self.case_base:
            if c.id == case_ext.id:
                continue
            sim = self.mnemonic_engine.enhanced_similarity(query, c)
            scored.append((sim, c))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Prefer cases whose misconception shares any word with the query
        # misconception. Embedding cosine already ranks them up, but the
        # filter sharpens "same family" vs "topically adjacent".
        query_words = _misconception_words(case_ext.misconception)
        family = []
        rest = []
        for sim, c in scored:
            if _misconception_words(c.misconception) & query_words:
                family.append((sim, c))
            else:
                rest.append((sim, c))
        ordered = family + rest
        return [c for _, c in ordered[: self.k_grounding]]

    @staticmethod
    def _format_grounding(retrieved) -> str:
        if not retrieved:
            return ""
        lines = [_CBR_GROUNDING_HEADER.strip()]
        for i, c in enumerate(retrieved, start=1):
            intv = getattr(c, "intervention", None) or {}
            wrong = intv.get("wrong_answer_text", "") if isinstance(intv, dict) else ""
            misc = c.misconception or "(unknown)"
            if not wrong:
                wrong = "(no recorded wrong answer)"
            lines.append(
                f"  {i}. A student with the misconception "
                f"\"{_short(misc, 90)}\" chose: \"{_short(wrong, 90)}\""
            )
        lines.append(_CBR_GROUNDING_FOOTER.strip())
        return "\n" + "\n".join(lines) + "\n"


def _misconception_words(label: str) -> set:
    """Lower-case content words from a misconception label. Used for
    'same-family' filtering. Stop words excluded so trivial overlaps
    (e.g. "the", "a") don't count as a match."""
    if not label:
        return set()
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "by",
        "with", "is", "are", "was", "were", "be", "as", "for", "that",
        "this", "these", "those", "it", "its", "but", "not", "no",
        "does", "do", "did", "has", "have", "had", "when", "than", "from",
    }
    return {
        w.lower().strip(",.;:!?\"'()[]")
        for w in label.split()
        if w.lower().strip(",.;:!?\"'()[]") and w.lower().strip(",.;:!?\"'()[]") not in stop
    }


def _short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"
