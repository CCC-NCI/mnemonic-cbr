"""Provider-agnostic LLM wrapper.

Spec reference: REBUILD_SPECIFICATION_v3.md §3 (roles and providers),
§11 (cost scenarios).

Two concrete providers are exposed:

  - StubProvider — deterministic, role-aware, no API calls. Used in
    Phase A and in the full-stack stub smoke test before any spend.
    For judge roles, returns valid JSON so the parsing layer exercises.

  - AnthropicProvider — real Anthropic API client. Used for
    Phase B-plumbing (5 dialogues, ~$0.05) and for the
    Anthropic-assigned roles in Phase B-proper.

OpenAI and Google providers will be added when the corresponding
keys arrive. The for_role() dispatcher reads a single env flag
(USE_REAL_LLMS=1) to switch between stub mode and real mode.

Design goals:
- Single generate(prompt, ...) interface so callers don't know which
  provider is behind it.
- Role-to-provider mapping lives in one config (see for_role); to
  swap providers you change one place.
- Stubs include the role and a salient context snippet in the return
  string so a smoke-test reader can tell at a glance which call
  produced which utterance.
- Judge stubs detect Pass 1 vs Pass 2 by sniffing the prompt and emit
  valid JSON with deterministic-but-varied integer scores so the
  parse layer is fully exercised by stub runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base interface. Concrete subclasses override generate()."""

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider!r}, model={self.model!r})"


# Regex to detect Pass-1 vs Pass-2 judge prompts from their content.
# Cheap, robust to wording changes within the same template family.
_PASS1_MARKERS = re.compile(r"\bR5\.\s+Student trajectory", re.IGNORECASE)
_PASS2_MARKERS = re.compile(r"\bR6\.\s+Strategy fidelity", re.IGNORECASE)


def _stable_score(seed_text: str, salt: str) -> int:
    """Deterministic 1..5 from a text-and-salt hash. Used by the judge
    stub to produce non-constant but reproducible scores."""
    h = hashlib.sha256((salt + seed_text).encode("utf-8")).digest()
    return 1 + (h[0] % 5)


class StubProvider(LLMProvider):
    """Deterministic stub used in Phase A and in stub-mode smoke tests.

    For non-judge roles: returns a short text string with role + a
    prompt snippet.

    For judge roles: sniffs the prompt to detect Pass 1 vs Pass 2 and
    returns valid JSON with deterministic 1..5 integer scores.
    """

    def __init__(self, role: str):
        super().__init__(provider="stub", model="stub")
        self.role = role
        self._counter = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        self._counter += 1

        # Judge role: emit valid JSON so the parse layer is exercised.
        if self.role.startswith("judge_"):
            if _PASS1_MARKERS.search(prompt):
                return json.dumps(
                    {
                        "R1": _stable_score(prompt, f"R1-{self.role}"),
                        "R2": _stable_score(prompt, f"R2-{self.role}"),
                        "R3": _stable_score(prompt, f"R3-{self.role}"),
                        "R4": _stable_score(prompt, f"R4-{self.role}"),
                        "R5": _stable_score(prompt, f"R5-{self.role}"),
                        "brief_justification": (
                            f"[STUB:{self.role}] deterministic pass-1 score "
                            f"#{self._counter}"
                        ),
                    }
                )
            if _PASS2_MARKERS.search(prompt):
                return json.dumps(
                    {
                        "R6": _stable_score(prompt, f"R6-{self.role}"),
                        "brief_justification": (
                            f"[STUB:{self.role}] deterministic pass-2 score "
                            f"#{self._counter}"
                        ),
                    }
                )
            # Unrecognised judge prompt — fall through to text stub.

        # Non-judge: text stub with role + prompt snippet.
        first_line = next(
            (ln.strip() for ln in prompt.splitlines() if ln.strip()),
            "(empty prompt)",
        )
        snippet = first_line[:60] + ("…" if len(first_line) > 60 else "")
        return f"[STUB:{self.role}#{self._counter}] {snippet}"


# ---------------------------------------------------------------------
# Real Anthropic provider
# ---------------------------------------------------------------------

# Default Anthropic models per role. Phase B-proper will keep these
# for Anthropic-assigned roles (student_leg_b, judge_primary,
# judge_secondary if Anthropic stands in for Gemini); OpenAI roles
# get GPT models once the key arrives.
DEFAULT_ANTHROPIC_MODELS = {
    "teacher": "claude-haiku-4-5",
    "student_leg_a": "claude-haiku-4-5",
    "student_leg_b": "claude-haiku-4-5",
    "judge_primary": "claude-sonnet-4-5",
    "judge_secondary": "claude-sonnet-4-5",
}


class AnthropicProvider(LLMProvider):
    """Real Anthropic API client.

    The legacy `cbr/llm_client.py` already imports the `anthropic`
    package; we use the same approach here. API key from
    ANTHROPIC_API_KEY env var unless overridden.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            ) from e
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY env var is not set and no api_key argument given"
            )
        super().__init__(provider="anthropic", model=model, api_key=key)
        import anthropic  # local import after the check above
        self._client = anthropic.Anthropic(api_key=key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        response = self._client.messages.create(**kwargs)
        # Anthropic returns a list of content blocks; the text block
        # is typically the first one in our usage.
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        # Fallback if the type discriminator isn't where we expect it.
        return str(response.content[0])


# ---------------------------------------------------------------------
# Real OpenAI provider
# ---------------------------------------------------------------------

# Default OpenAI models per spec §3 / §11 (Scenario B):
#   teacher       → GPT-3.5-turbo
#   student_leg_a → GPT-4o-mini
DEFAULT_OPENAI_MODELS = {
    "teacher": "gpt-3.5-turbo",
    "student_leg_a": "gpt-4o-mini",
}


class OpenAIProvider(LLMProvider):
    """Real OpenAI API client.

    Uses the same `openai` python client the legacy `cbr/llm_client.py`
    imports. API key from OPENAI_API_KEY env var unless overridden.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY env var is not set and no api_key argument given"
            )
        super().__init__(provider="openai", model=model, api_key=key)
        import openai
        self._client = openai.OpenAI(api_key=key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------
# Role dispatch
# ---------------------------------------------------------------------

# Role names per spec §3.
ROLES = (
    "teacher",
    "student_leg_a",
    "student_leg_b",
    "judge_primary",
    "judge_secondary",
)

# Spec §3.5 / §11 role-to-provider assignment for Phase B-proper /
# Scenario B. teacher and student_leg_a go to OpenAI; student_leg_b
# and the judges go to Anthropic. judge_secondary (Gemini Pro) is
# planned for the 50-dialogue ICC subset and remains Anthropic-fallback
# until a Google key is wired in.
_SPEC_PROVIDER_FOR_ROLE = {
    "teacher": "openai",
    "student_leg_a": "openai",
    "student_leg_b": "anthropic",
    "judge_primary": "anthropic",
    "judge_secondary": "anthropic",  # Gemini stub for now
}


def _truthy(env_value: Optional[str]) -> bool:
    if env_value is None:
        return False
    return env_value.strip().lower() in {"1", "true", "yes", "on"}


def for_role(role: str) -> LLMProvider:
    """Return the configured provider for a role.

    Mode selection:
      - default: StubProvider (Phase A, deterministic, no API calls).
      - USE_REAL_LLMS=1: real provider.
          * Per-role assignment from spec (teacher/student_leg_a → OpenAI;
            student_leg_b/judge_* → Anthropic).
          * If OPENAI_API_KEY is missing, OpenAI-assigned roles fall back
            to Anthropic with a warning. Acknowledged in
            IMPLEMENTATION_PLAN §6.1 as a model-overlap caveat.
      - LLM_PROVIDER_OVERRIDE=anthropic: force Anthropic for every role
        (Phase B-plumbing mode). Same caveat applies.

    Per-role model override via env var, e.g.,
      OPENAI_MODEL_teacher=gpt-4o
      ANTHROPIC_MODEL_judge_primary=claude-opus-4-6
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}. Expected one of: {ROLES}")

    if not _truthy(os.getenv("USE_REAL_LLMS")):
        return StubProvider(role=role)

    override = (os.getenv("LLM_PROVIDER_OVERRIDE") or "").strip().lower()
    assigned = _SPEC_PROVIDER_FOR_ROLE.get(role, "anthropic")
    if override == "anthropic":
        assigned = "anthropic"
    elif override == "openai":
        assigned = "openai" if role in DEFAULT_OPENAI_MODELS else "anthropic"

    if assigned == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning(
                f"for_role({role!r}): OPENAI_API_KEY not set; "
                f"falling back to Anthropic. Model overlap is a known "
                f"limitation under this fallback."
            )
            assigned = "anthropic"

    if assigned == "openai":
        model = os.getenv(
            f"OPENAI_MODEL_{role}", DEFAULT_OPENAI_MODELS[role]
        )
        logger.info(f"for_role({role!r}) → OpenAIProvider(model={model!r})")
        return OpenAIProvider(model=model)

    # Anthropic (either by spec assignment, by override, or by fallback)
    model = os.getenv(
        f"ANTHROPIC_MODEL_{role}", DEFAULT_ANTHROPIC_MODELS[role]
    )
    logger.info(f"for_role({role!r}) → AnthropicProvider(model={model!r})")
    return AnthropicProvider(model=model)
