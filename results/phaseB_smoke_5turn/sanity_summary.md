# Phase C sanity-check battery

- Generated: 2026-05-17T19:17:18Z
- Manifest:  `results/phaseB_smoke_5turn/manifest.sqlite`
- Real mode: True (checks 4 and 5 need this)

---

## Check 1 — Persona discriminability
- **Pass:** `True`
- Mean pairwise cosine = 0.638 (pass <= 0.85: True); max = 0.833 (pass <= 0.95: True). Observation (not gating): socratic-vs-rule_based = 0.488 (vs median 0.641: below).

## Check 3 — R5 contamination (length / sentiment)
- **Pass:** `True`
- r(R5, length) = -0.205 (pass |r|<=0.5: True); r(R5, sentiment) = 0.300 (pass: True).

## Check 4 — Shuffle-persona (adversarial; needs real judge)
- **Pass:** `True`
- n = 40; mean R6 original = 3.17; mean R6 shuffled = 1.85; drop = +1.32 (pass drop >= 1.0: True).

## Check 5 — Dialogue-order corruption (adversarial; needs real judge)
- **Pass:** `False`
- n = 40; mean R5 original = 2.58; mean R5 swapped = 2.45; drop = +0.12 (pass drop >= 0.5: False).

---

## Gate decision

**Gate not fully cleared.** Failed: Check 5: dialogue-order corruption. 

_Output also written as JSON: `results/phaseB_smoke_5turn/sanity_summary.json`._

