"""Two-pass rubric judge.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.4.

Pass 1: persona-blind, scores R1..R5 + brief justification.
Pass 2: persona-visible, scores R6 only + brief justification.

The two passes are independent LLM calls; the judge does not carry
context between them. Judge temperature is 0.2 (spec §11) to maximise
within-judge consistency.

The four reporting buckets (spec §3.4):
  - R5 alone              — primary outcome
  - (R1 + R2 + R3) / 3    — pedagogical quality composite
  - R4                    — domain accuracy sanity statistic
  - R6                    — strategy fidelity adherence statistic

RubricScore exposes these as convenience properties so callers can
aggregate without re-doing the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any, Dict, Optional

from dialogue.personas import PERSONA_SHORT_DESCRIPTIONS
from scoring.prompts import PASS1_PROMPT, PASS2_PROMPT, format_turns_block
from scoring import parse as _parse

if TYPE_CHECKING:
    from dialogue.state import DialogueState
    from dialogue.llm_provider import LLMProvider


MISSING_SCORE: Optional[int] = None


@dataclass
class RubricScore:
    """One scored dialogue. All R-values are int in [1, 5] or None if
    parsing failed.

    The four reporting buckets are exposed as properties.
    """

    R1: Optional[int] = MISSING_SCORE
    R2: Optional[int] = MISSING_SCORE
    R3: Optional[int] = MISSING_SCORE
    R4: Optional[int] = MISSING_SCORE
    R5: Optional[int] = MISSING_SCORE
    R6: Optional[int] = MISSING_SCORE

    justification_pass1: str = ""
    justification_pass2: str = ""

    judge_provider: str = ""
    judge_model: str = ""

    # Error fields propagated from parse.py. Empty string == no error.
    pass1_error: str = ""
    pass2_error: str = ""
    pass1_raw_text: str = ""
    pass2_raw_text: str = ""

    # --- Reporting-bucket properties (spec §3.4) ---

    @property
    def primary_outcome(self) -> Optional[int]:
        """R5 — primary outcome."""
        return self.R5

    @property
    def quality_composite(self) -> Optional[float]:
        """(R1 + R2 + R3) / 3 — pedagogical quality composite.

        Returns None if any of R1/R2/R3 is None; the aggregator decides
        whether to drop or impute.
        """
        if None in (self.R1, self.R2, self.R3):
            return None
        return (self.R1 + self.R2 + self.R3) / 3.0

    @property
    def domain_accuracy(self) -> Optional[int]:
        """R4 — sanity statistic."""
        return self.R4

    @property
    def strategy_fidelity(self) -> Optional[int]:
        """R6 — adherence statistic."""
        return self.R6

    @property
    def is_complete(self) -> bool:
        """True iff all six items were parsed successfully."""
        return None not in (
            self.R1, self.R2, self.R3, self.R4, self.R5, self.R6
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Derived fields for downstream queries / persistence.
        d["quality_composite"] = self.quality_composite
        d["is_complete"] = self.is_complete
        return d


class RubricScorer:
    """Two-pass rubric scorer.

    The primary provider is used for both Pass 1 and Pass 2 in a normal
    run. The secondary provider is used for the inter-judge ICC subset
    (50 dialogues, Phase C/D) via score_secondary().
    """

    def __init__(
        self,
        primary_provider: "LLMProvider",
        temperature: float = 0.2,
        secondary_provider: Optional["LLMProvider"] = None,
    ):
        self.primary = primary_provider
        self.temperature = temperature
        self.secondary = secondary_provider

    def score(self, state: "DialogueState") -> RubricScore:
        """Run the two-pass scoring with the primary judge."""
        return self._score_with(self.primary, state)

    def score_secondary(self, state: "DialogueState") -> RubricScore:
        """Run with the secondary judge — for inter-judge ICC.

        Used only on the 50-dialogue subset described in spec §3.4. If
        no secondary provider is configured, raises.
        """
        if self.secondary is None:
            raise RuntimeError(
                "No secondary provider configured for score_secondary()"
            )
        return self._score_with(self.secondary, state)

    # --- Internals ---

    def _score_with(self, provider: "LLMProvider", state: "DialogueState") -> RubricScore:
        pass1_text = self._call_pass1(provider, state)
        pass2_text = self._call_pass2(provider, state)
        pass1 = _parse.parse_pass1(pass1_text)
        pass2 = _parse.parse_pass2(pass2_text)
        return RubricScore(
            R1=pass1["R1"], R2=pass1["R2"], R3=pass1["R3"],
            R4=pass1["R4"], R5=pass1["R5"],
            R6=pass2["R6"],
            justification_pass1=pass1["justification"],
            justification_pass2=pass2["justification"],
            judge_provider=provider.provider,
            judge_model=provider.model,
            pass1_error=pass1["error"],
            pass2_error=pass2["error"],
            pass1_raw_text=pass1["raw_text"],
            pass2_raw_text=pass2["raw_text"],
        )

    def _call_pass1(self, provider: "LLMProvider", state: "DialogueState") -> str:
        prompt = PASS1_PROMPT.format(
            misconception_label=state.misconception_label,
            turns_block=format_turns_block(state.turn_history),
        )
        return provider.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=400,
        )

    def _call_pass2(self, provider: "LLMProvider", state: "DialogueState") -> str:
        persona_desc = PERSONA_SHORT_DESCRIPTIONS.get(
            state.persona, state.persona
        )
        prompt = PASS2_PROMPT.format(
            persona=state.persona,
            persona_short_description=persona_desc,
            turns_block=format_turns_block(state.turn_history),
        )
        return provider.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=400,
        )
