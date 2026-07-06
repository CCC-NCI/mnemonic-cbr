"""Sanity-check battery for the Phase C gate.

Spec reference: REBUILD_SPECIFICATION_v3.md §4.

Five checks, each with a pre-registered pass/fail criterion. This
package implements checks 1, 3, 4, and 5. Check 2 (inter-judge ICC)
is deferred until a Gemini key is available.

Public surface:
    check1_persona_discriminability(manifest_path, ...) -> dict
    check3_r5_contamination(manifest_path) -> dict
    check4_shuffle_persona(manifest_path, n=40, judge_provider=...) -> dict
    check5_dialogue_order(manifest_path, n=40, judge_provider=...) -> dict
"""
