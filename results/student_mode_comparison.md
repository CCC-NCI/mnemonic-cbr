# Student-mode A/B comparison

- Generated:    2026-05-12T08:54:54Z
- pure_ai:      `results/phaseB_smoke/manifest.sqlite`
- cbr_grounded: `results/phaseB_smoke_cbr/manifest.sqlite`

## Row counts

- pure_ai real rows:       250
- cbr_grounded real rows:  250

## Paired cells (present in BOTH manifests)

- Paired cell count: 250

## Headline: per-item paired comparison

```
                          item  mean_pure  mean_cbr  mean_delta_cbr_minus_pure  paired_d  n_paired  magnitude
   R1 misconception engagement   2.120000  2.136000                      0.016  0.018417       250 negligible
           R2 cognitive demand   1.980000  1.972000                     -0.008 -0.012148       250 negligible
            R3 scaffolding fit   1.784000  1.848000                      0.064  0.080097       250 negligible
            R4 domain accuracy   3.020000  2.996000                     -0.024 -0.014336       250 negligible
         R5 student trajectory   1.804000  1.812000                      0.008  0.007147       250 negligible
          R6 strategy fidelity   2.664000  2.640000                     -0.024 -0.020483       250 negligible
Quality composite (R1+R2+R3)/3   1.961333  1.985333                      0.024  0.036809       250 negligible
```

## Reading the headline

- R5 (student trajectory): cbr−pure delta = +0.01, paired d = +0.01 (negligible).
- Quality composite:        cbr−pure delta = +0.02, paired d = +0.04 (negligible).

Decision rule:
- |paired d| on R5 ≥ 0.5  → cbr_grounded materially shifts the trajectory signal; adopt cbr_grounded as canonical for Phase D.
- |paired d| on R5 < 0.2  → no material shift; keep pure_ai as the spec design; document the test as a negative result.
- 0.2 ≤ |paired d| < 0.5 → ambiguous; run on a larger matrix or report both in the manuscript with a discussion paragraph.

## R5 by architecture (paired)

```
architecture  n  mean_pure_R5  mean_cbr_R5  delta  paired_d
      hybrid 50          2.00         2.14   0.14  0.110949
     pure_ai 50          2.10         2.18   0.08  0.069099
    baseline 50          1.74         1.74   0.00  0.000000
pure_cbr_llm 50          1.96         1.90  -0.06 -0.052167
pure_cbr_tpl 50          1.22         1.10  -0.12 -0.214891
```

## R5 by persona (paired)

```
     persona  n  mean_pure_R5  mean_cbr_R5  delta  paired_d
  rule_based 50          1.62         1.74   0.12  0.150192
constructive 50          1.92         2.00   0.08  0.070176
    socratic 50          1.88         1.88   0.00  0.000000
experiential 50          1.72         1.64  -0.08 -0.061963
 traditional 50          1.88         1.80  -0.08 -0.063538
```

