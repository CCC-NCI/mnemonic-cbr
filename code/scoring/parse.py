"""JSON parsing for judge responses with graceful fallback.

LLMs occasionally:
- wrap their JSON in Markdown fences (```json ... ```);
- prepend explanatory prose before the JSON;
- emit trailing commas or single quotes.

Strategy (cheap → expensive):
  1. Try strict json.loads on the raw text.
  2. Strip Markdown fences and retry.
  3. Find the first '{' and last '}' and try that substring.
  4. Give up; return ParseFailure with the raw text attached.

A single retry-on-different-strategy is enough for almost all real
LLM outputs. The rubric is small (5 or 6 integer fields plus a
short string), so heroics aren't justified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass
class ParseFailure:
    """Returned in place of a parsed dict when the response can't be parsed."""

    raw_text: str
    error: str


def _try_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return result if isinstance(result, dict) else None


def parse_judge_response(text: str) -> Dict[str, Any] | ParseFailure:
    """Best-effort parse of a judge JSON response.

    Returns a dict on success, or ParseFailure on failure.
    """
    if not text or not isinstance(text, str):
        return ParseFailure(raw_text=str(text), error="empty or non-string response")

    # Strategy 1: strict.
    direct = _try_load(text)
    if direct is not None:
        return direct

    # Strategy 2: strip Markdown fences.
    fence_match = _FENCE_RE.match(text.strip())
    if fence_match:
        fenced = _try_load(fence_match.group(1))
        if fenced is not None:
            return fenced

    # Strategy 3: extract between first '{' and last '}'.
    open_idx = text.find("{")
    close_idx = text.rfind("}")
    if open_idx != -1 and close_idx > open_idx:
        substring = text[open_idx : close_idx + 1]
        bracketed = _try_load(substring)
        if bracketed is not None:
            return bracketed

    return ParseFailure(
        raw_text=text,
        error="could not extract valid JSON object from response",
    )


def _coerce_int_1_5(value: Any) -> Optional[int]:
    """Coerce a value to an int in [1, 5], or None if not valid."""
    if value is None:
        return None
    try:
        i = int(value)
    except (TypeError, ValueError):
        try:
            i = int(round(float(value)))
        except (TypeError, ValueError):
            return None
    if 1 <= i <= 5:
        return i
    return None


def extract_rubric_items(
    parsed: Dict[str, Any], expected_keys: Iterable[str]
) -> Dict[str, Optional[int]]:
    """Pull out the expected rubric integers from a parsed dict.

    Missing keys map to None. Out-of-range values map to None. The
    caller decides what to do with Nones (skip aggregation, retry, or
    mark cell as missing).
    """
    return {key: _coerce_int_1_5(parsed.get(key)) for key in expected_keys}


def parse_pass1(text: str) -> Dict[str, Any]:
    """Parse a Pass-1 response. Returns a dict with keys R1..R5 (each
    int or None), justification (str or ''), and error (str or '').

    Never raises. Errors are encoded in the 'error' field.
    """
    parsed = parse_judge_response(text)
    if isinstance(parsed, ParseFailure):
        return {
            "R1": None,
            "R2": None,
            "R3": None,
            "R4": None,
            "R5": None,
            "justification": "",
            "error": parsed.error,
            "raw_text": parsed.raw_text,
        }
    scores = extract_rubric_items(parsed, ("R1", "R2", "R3", "R4", "R5"))
    return {
        **scores,
        "justification": str(parsed.get("brief_justification", "") or ""),
        "error": "",
        "raw_text": text,
    }


def parse_pass2(text: str) -> Dict[str, Any]:
    """Parse a Pass-2 response. Returns a dict with R6 (int or None),
    justification, error, raw_text. Never raises.
    """
    parsed = parse_judge_response(text)
    if isinstance(parsed, ParseFailure):
        return {
            "R6": None,
            "justification": "",
            "error": parsed.error,
            "raw_text": parsed.raw_text,
        }
    scores = extract_rubric_items(parsed, ("R6",))
    return {
        **scores,
        "justification": str(parsed.get("brief_justification", "") or ""),
        "error": "",
        "raw_text": text,
    }
