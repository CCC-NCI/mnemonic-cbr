"""Mechanical turn-1 reasoning template per misconception type.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.2.

Turn 1 of every dialogue is deterministic, not LLM-generated. The spec
template is:

    "I have {case.problem}. I got {case.student_answer} because
     {brief_reasoning_template}."

This module maps a misconception label (or substring of one) to a
short, mechanical "because ..." clause that reads like an 11-16 year
old explaining their wrong-answer reasoning.

The starter set below covers the most common misconception families.
Phase B will enumerate the actual misconception labels present in the
EEDI sample and extend this dictionary. Until then, _DEFAULT handles
unknowns.
"""

from __future__ import annotations

# Map: keyword (substring, lowercased) → reasoning clause.
# First match wins (insertion order matters).
_TEMPLATES = {
    # Fraction arithmetic
    "add numerator": "I added the numerators and added the denominators.",
    "fraction": "I treated the fractions like ordinary numbers.",
    # Decimal arithmetic
    "decimal": "I treated the decimal like a whole number and ignored the point.",
    "place value": "I lined up the numbers from the left instead of by place value.",
    # Algebra
    "distribut": "I only multiplied the first term inside the brackets.",
    "negative": "I forgot that two negatives multiply to a positive.",
    "exponent": "I added the exponents instead of multiplying.",
    "bidmas": "I worked left to right instead of following the order of operations.",
    "bodmas": "I worked left to right instead of following the order of operations.",
    "order of operations": "I worked left to right instead of following the order of operations.",
    # Geometry / measurement
    "perimeter": "I confused perimeter with area.",
    "area": "I added the side lengths instead of multiplying.",
    "angle": "I assumed all the angles in the figure are equal.",
    # Percentages / ratios
    "percent": "I treated the percent as if it were already a fraction over 1.",
    "ratio": "I added the parts of the ratio as if they were a total.",
    # Number sense
    "round": "I rounded too early in the calculation.",
    "estimate": "I rounded each number separately and added the rounded values.",
    # Equations / inequalities
    "inequalit": "I flipped the inequality the wrong way when multiplying by a negative.",
    "equation": "I moved the term across without changing its sign.",
}

_DEFAULT = "I'm not sure I worked it out the right way."


def template_for(misconception_label: str) -> str:
    """Return a short 'because ...' reasoning clause for a misconception.

    Matching is case-insensitive substring matching against the keys
    in _TEMPLATES. The first matching key wins, so order keys from more
    specific to more general.

    Falls back to _DEFAULT if nothing matches.
    """
    if not misconception_label:
        return _DEFAULT
    needle = misconception_label.lower()
    for key, template in _TEMPLATES.items():
        if key in needle:
            return template
    return _DEFAULT
