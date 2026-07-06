# Phase C sanity-check battery

- Generated: 2026-05-12T13:10:58Z
- Manifest:  `results/phaseB_smoke/manifest.sqlite`
- Real mode: True (checks 4 and 5 need this)

---

## Check 1 — Persona discriminability
- **Pass:** `True`
- Mean pairwise cosine = 0.580 (pass <= 0.85: True); max = 0.752 (pass <= 0.95: True). Observation (not gating): socratic-vs-rule_based = 0.752 (vs median 0.546: at or above).

## Check 3 — R5 contamination (length / sentiment)
- **Pass:** `True`
- r(R5, length) = -0.294 (pass |r|<=0.5: True); r(R5, sentiment) = 0.278 (pass: True).

## Check 4 — Shuffle-persona (adversarial; needs real judge)
- **Pass:** `True`
- n = 40; mean R6 original = 3.10; mean R6 shuffled = 1.73; drop = +1.38 (pass drop >= 1.0: True).

## Check 5 — Dialogue-order corruption (adversarial; needs real judge)
- **Pass:** `False`
- n = 40; mean R5 original = 1.85; mean R5 swapped = 1.50; drop = +0.35 (pass drop >= 0.5: False).

---

## Gate decision

**Gate not fully cleared.** Failed: Check 5: dialogue-order corruption. 

_Output also written as JSON: `results/phaseB_smoke/sanity_summary.json`._

