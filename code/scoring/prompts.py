"""Judge prompt templates.

Lifted verbatim from REBUILD_SPECIFICATION_v3.md §3.4 with placeholders
left as Python str.format() slots.

The judge does NOT carry context between passes. Pass 1 is
persona-blind (R1-R5). Pass 2 is persona-visible (R6 only).

Phase C may tighten the wording based on what failure modes show up
in the pilot inspection — see IMPLEMENTATION_PLAN §8 (deferred items).
The starting text is the spec text.
"""

from __future__ import annotations


PASS1_PROMPT = """\
You are an expert in mathematics education research. You will read a
short tutoring dialogue between a teacher and a UK secondary school
student, then score the dialogue on five rubric items.

You will NOT be told what teaching strategy the teacher claims to be
using. Score the dialogue on its substantive properties only.

The student's initial misconception:
  {misconception_label}

The dialogue:
{turns_block}

Score each item from 1 (poor) to 5 (excellent). Be strict; reserve 5
for genuinely strong instances.

R1. Misconception engagement: Does the teacher acknowledge and address
    the specific misconception (not just generic correction)?
R2. Cognitive demand: Does the teacher require the student to reason,
    or just state the answer?
R3. Scaffolding fit: Is the support level calibrated to what the
    student has demonstrated?
R4. Domain accuracy: Is the mathematical content correct?
R5. Student trajectory: Does the student's reasoning visibly improve
    across turns? Consider whether the dialogue plausibly corrects the
    misconception, judging only from what the student says.

Respond with a JSON object: {{"R1": <int>, "R2": <int>, "R3": <int>,
"R4": <int>, "R5": <int>, "brief_justification": "<one sentence>"}}."""


PASS2_PROMPT = """\
You are an expert in mathematics education research. You will read a
short tutoring dialogue and assess whether the teacher's behavior is
consistent with a stated teaching strategy.

The claimed teaching strategy: {persona}
Brief description of the {persona} strategy: {persona_short_description}

The dialogue:
{turns_block}

R6. Strategy fidelity: How faithfully does the teacher instantiate the
    {persona} strategy? Score 1 (not at all) to 5 (textbook example).

Respond with a JSON object: {{"R6": <int>, "brief_justification": "<one
sentence>"}}."""


def format_turns_block(turn_history) -> str:
    """Format a list of Turn objects (or dicts of the same shape) as the
    {turns_block} slot value. Handles both forms because dialogues come
    in as dataclasses from the dialogue loop and as dicts when reloaded
    from persisted JSON.
    """
    if not turn_history:
        return "  (no turns)"
    lines = []
    for turn in turn_history:
        if isinstance(turn, dict):
            idx = turn.get("turn_index")
            speaker = turn.get("speaker")
            text = turn.get("text")
        else:
            idx = turn.turn_index
            speaker = turn.speaker
            text = turn.text
        lines.append(f"  Turn {idx} ({speaker}): {text}")
    return "\n".join(lines)
