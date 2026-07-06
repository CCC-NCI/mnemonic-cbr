"""Teacher layer.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.3.

TeacherGenerator with five architecture branches:

  baseline       — canonical sentence per persona, shared LLM render
  pure_cbr_llm   — retrieved interventions, shared LLM render
                   ("no elaboration beyond retrieved content")
  pure_cbr_tpl   — retrieved interventions, template wrap, NO LLM
  pure_ai        — persona prompt only, shared LLM render
  hybrid         — persona prompt + retrieved interventions, shared
                   LLM render

The shared rendering prompt is identical across the four LLM-using
branches; what varies is (a) the "content source" inserted into the
prompt and (b) a one-line constraint on how the LLM should use it.
This is the §3.3 design — isolating content source from rendering
style.

For Phase A, the LLM call is a stub (deterministic string). For
Phase B-plumbing with Anthropic-only keys, the provider is a real
Anthropic Haiku client. For Phase B-proper, the provider becomes
OpenAI GPT-3.5-turbo per the spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from dialogue.personas import (
    CANONICAL_BASELINE_SENTENCES,
    PERSONA_PRINCIPLES,
    assert_known_persona,
)

if TYPE_CHECKING:
    from dialogue.state import CaseExt, DialogueState
    from dialogue.llm_provider import LLMProvider
    from cbr.mnemonic_augmentation import Case, MnemonicAugmentation


ARCHITECTURES = (
    "baseline",
    "pure_cbr_llm",
    "pure_cbr_tpl",
    "pure_ai",
    "hybrid",
)


# Shared rendering prompt template. The {content_block} and
# {content_source_instruction} slots are what differ by architecture.
_RENDERING_PROMPT = """\
You are rendering a teacher utterance using the {persona} teaching
strategy.

{persona_principles}

The student holds the following misconception:
  {misconception_label}

{content_block}

Conversation so far:
{turn_history_as_text}

Render the next teacher utterance. It must be 1-3 sentences, follow
the {persona} approach, and respond to the student's most recent
turn. {content_source_instruction}

Your next utterance:"""


# Per-architecture content-source constraint (the §3.3 table, last
# column). pure_cbr_tpl is not here because it doesn't use the LLM.
_CONTENT_SOURCE_INSTRUCTION = {
    "baseline": (
        "Render the canonical baseline sentence above with minimal "
        "rewording. Do not add new pedagogical content."
    ),
    "pure_cbr_llm": (
        "Use ONLY the retrieved interventions above as your content "
        "source. Do not add new pedagogical content beyond what is "
        "given in the retrieved cases."
    ),
    "pure_ai": (
        "Generate a persona-conditioned response from your own "
        "pedagogical knowledge. There are no retrieved cases."
    ),
    "hybrid": (
        "Integrate the retrieved interventions above with your own "
        "pedagogical knowledge to produce the response."
    ),
}


def assert_known_architecture(architecture: str) -> None:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. Expected one of: {ARCHITECTURES}"
        )


class TeacherGenerator:
    """Generates teacher utterances for one (persona, architecture) pair.

    Construction is per-cell, not per-dialogue: one instance is reused
    across all dialogues for the same (persona, architecture) so that
    the case_base / mnemonic_engine setup cost is amortised.
    """

    def __init__(
        self,
        architecture: str,
        persona: str,
        provider: Optional["LLMProvider"] = None,
        mnemonic_engine: Optional["MnemonicAugmentation"] = None,
        case_base: Optional[List["Case"]] = None,
        k_retrieve: int = 3,
        temperature: float = 0.7,
    ):
        assert_known_architecture(architecture)
        assert_known_persona(persona)
        self.architecture = architecture
        self.persona = persona
        self.provider = provider
        self.mnemonic_engine = mnemonic_engine
        self.case_base = case_base or []
        self.k_retrieve = k_retrieve
        self.temperature = temperature
        # Sanity: required dependencies per architecture.
        if architecture == "pure_cbr_tpl":
            if mnemonic_engine is None or not self.case_base:
                raise ValueError(
                    "pure_cbr_tpl requires both mnemonic_engine and case_base"
                )
            # provider may be None for pure_cbr_tpl; that's the whole point.
        elif architecture in ("pure_cbr_llm", "hybrid"):
            if mnemonic_engine is None or not self.case_base or provider is None:
                raise ValueError(
                    f"{architecture} requires mnemonic_engine, case_base, and provider"
                )
        elif architecture in ("baseline", "pure_ai"):
            if provider is None:
                raise ValueError(f"{architecture} requires a provider")

    def next_utterance(self, state: "DialogueState", case_ext) -> str:
        """Generate the teacher's next utterance."""
        if self.architecture == "pure_cbr_tpl":
            return self._template_wrap(state, case_ext)
        prompt = self._build_rendering_prompt(state, case_ext)
        return self.provider.generate(  # type: ignore[union-attr]
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=200,
        )

    # --- Content gathering per architecture ---

    def _build_rendering_prompt(self, state, case_ext) -> str:
        if self.architecture == "baseline":
            content_block = (
                "Canonical baseline sentence to render:\n"
                f"  {CANONICAL_BASELINE_SENTENCES[self.persona]}"
            )
        elif self.architecture == "pure_ai":
            content_block = (
                "No retrieved cases. Rely on the persona principles "
                "above and the dialogue history."
            )
        elif self.architecture in ("pure_cbr_llm", "hybrid"):
            retrieved = self._retrieve(case_ext)
            content_block = self._format_retrieved(retrieved)
        else:
            # Should be unreachable due to constructor validation.
            raise AssertionError(f"unexpected architecture {self.architecture!r}")

        return _RENDERING_PROMPT.format(
            persona=self.persona,
            persona_principles=PERSONA_PRINCIPLES[self.persona],
            misconception_label=state.misconception_label,
            content_block=content_block,
            turn_history_as_text=state.history_as_text(),
            content_source_instruction=_CONTENT_SOURCE_INSTRUCTION[self.architecture],
        )

    def _template_wrap(self, state, case_ext) -> str:
        """No-LLM branch for pure_cbr_tpl.

        Stitches retrieved interventions into a template string. No
        rendering, no rewording. The §3.3 table calls for exactly this.
        """
        retrieved = self._retrieve(case_ext)
        if not retrieved:
            return (
                "Let's think about this together. "
                "(No similar cases were retrieved for this misconception.)"
            )
        pieces = []
        for i, c in enumerate(retrieved, start=1):
            text = self._intervention_text(c)
            pieces.append(f"  {i}. {text}")
        body = "\n".join(pieces)
        return f"Let's think about this. From similar cases:\n{body}"

    # --- CBR retrieval ---

    def _retrieve(self, case_ext) -> List:
        """Return top-k cases from the case base, ranked by the
        cleaned mnemonic-enhanced similarity (see dialogue.retrieval).
        """
        if self.mnemonic_engine is None or not self.case_base:
            return []
        # Build a query Case from case_ext for similarity comparison.
        # The legacy Case is the unit the engine expects.
        from cbr.mnemonic_augmentation import Case

        query_case = Case(
            id=f"query_{case_ext.id}",
            features=case_ext.features,
            misconception=case_ext.misconception,
            intervention={},
            outcome=0.5,  # placeholder; the cleaned engine no longer
                         # consults this field for retrieval shaping.
            utility_score=0.5,
        )
        scored = []
        for c in self.case_base:
            # Skip self-retrieval if the query coincidentally matches a
            # case in the base (cheap defensive check).
            if c.id == case_ext.id:
                continue
            sim = self.mnemonic_engine.enhanced_similarity(query_case, c)
            scored.append((sim, c))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [c for _, c in scored[: self.k_retrieve]]

    @staticmethod
    def _intervention_text(case) -> str:
        """Extract intervention text for a retrieved case.

        Prefers the deterministically-synthesised intervention_text
        populated by experiments/eedi_loader.py::case_ext_from_row
        (option (c) in IMPLEMENTATION_PLAN). Falls back to a brief
        placeholder for cases that lack it (e.g., legacy test cases
        constructed without the loader).
        """
        intervention = getattr(case, "intervention", None) or {}
        text = ""
        if isinstance(intervention, dict):
            text = intervention.get("intervention_text", "") or ""
        if text:
            return text
        misconception = getattr(case, "misconception", None) or "unknown misconception"
        return (
            f"For students with '{misconception}', explain the underlying "
            f"rule and try a worked example."
        )

    @staticmethod
    def _format_retrieved(retrieved) -> str:
        if not retrieved:
            return "No similar cases were retrieved."
        lines = ["Retrieved similar cases (use these as your content source):"]
        for i, c in enumerate(retrieved, start=1):
            text = TeacherGenerator._intervention_text(c)
            lines.append(f"  {i}. Misconception: {c.misconception}")
            lines.append(f"     Intervention: {text}")
        return "\n".join(lines)
