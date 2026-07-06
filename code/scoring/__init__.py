"""Mnemonic-CBR Rebuild — scoring package (v2).

Two-pass rubric judge per REBUILD_SPECIFICATION_v3.md §3.4.

Public surface:
  - RubricScore, MISSING_SCORE      (rubric.py)
  - RubricScorer                     (rubric.py)
  - parse_pass1, parse_pass2,
    ParseFailure                     (parse.py)
  - PASS1_PROMPT, PASS2_PROMPT       (prompts.py)
"""
