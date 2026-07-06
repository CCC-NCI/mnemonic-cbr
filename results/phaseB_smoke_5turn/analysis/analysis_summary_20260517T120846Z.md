# Phase D analysis summary

- Generated: 2026-05-17T12:08:46Z
- Manifest:  `results/phaseB_smoke_5turn/manifest.sqlite`
- Filter:    `real`

---

## Manifest summary
- Total rows in manifest: 50
- Rows after filter:      50
- Personas:      ['constructive', 'experiential', 'rule_based', 'socratic', 'traditional']
- Architectures: ['baseline', 'hybrid', 'pure_ai', 'pure_cbr_llm', 'pure_cbr_tpl']
- Student legs:  ['leg_a', 'leg_b']

## Per-persona marginals
```
     persona  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
experiential  10.0      2.5 0.849837                 10.0                1.633333              0.456773  10.0      3.0 1.414214  10.0      2.9 1.449138
    socratic  10.0      2.2 1.135292                 10.0                2.266667              0.978787  10.0      4.3 1.059350  10.0      3.7 1.494434
constructive  10.0      2.1 0.994429                 10.0                1.766667              0.567646  10.0      3.7 1.418136  10.0      2.3 1.251666
 traditional  10.0      2.1 1.286684                 10.0                1.600000              0.516398  10.0      3.6 1.349897  10.0      1.6 0.516398
  rule_based  10.0      2.0 1.054093                 10.0                1.500000              0.360041  10.0      3.2 1.549193  10.0      3.0 1.414214
```

## Per-architecture marginals
```
architecture  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
    baseline  10.0      2.5 1.269296                 10.0                1.800000              1.135292  10.0      3.5 1.509231  10.0      3.0 1.632993
      hybrid  10.0      2.5 0.971825                 10.0                2.033333              0.291865  10.0      3.6 1.577621  10.0      3.3 1.159502
pure_cbr_llm  10.0      2.3 1.159502                 10.0                1.933333              0.210819  10.0      4.5 0.971825  10.0      3.3 1.159502
     pure_ai  10.0      1.9 0.994429                 10.0                2.000000              0.222222  10.0      3.8 1.398412  10.0      2.9 1.197219
pure_cbr_tpl  10.0      1.7 0.674949                 10.0                1.000000              0.000000  10.0      2.4 0.516398  10.0      1.0 0.000000
```

## Architecture × Persona cell means
Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.

## Two-way ANOVA (Architecture × Persona)
- **R5**: F_arch=1.29 (p=0.299, η²p=0.172); F_persona=0.36 (p=0.833, η²p=0.055); F_interaction=1.29 (p=0.274).
- **quality_composite**: F_arch=28.76 (p=4.99e-09, η²p=0.821); F_persona=14.19 (p=3.58e-06, η²p=0.694); F_interaction=7.70 (p=4.17e-06).

## Pairwise Cohen's d on R5
- **architecture**: largest |d|=0.96 (hybrid vs pure_cbr_tpl, large)
- **persona**: largest |d|=0.52 (experiential vs rule_based, medium)

## Cross-student variance on R5 (leg_a vs leg_b)
- Mean |Δ R5| across cells = 1.16

## Output locations
- Tables: `results/phaseB_smoke_5turn/analysis/tables/`
- Summary: `results/phaseB_smoke_5turn/analysis/analysis_summary.md` (also archived as `analysis_summary_20260517T120846Z.md`)
