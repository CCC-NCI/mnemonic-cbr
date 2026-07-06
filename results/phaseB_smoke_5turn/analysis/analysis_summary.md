# Phase D analysis summary

- Generated: 2026-05-17T17:53:27Z
- Manifest:  `results/phaseB_smoke_5turn/manifest.sqlite`
- Filter:    `real`

---

## Manifest summary
- Total rows in manifest: 1250
- Rows after filter:      1250
- Personas:      ['constructive', 'experiential', 'rule_based', 'socratic', 'traditional']
- Architectures: ['baseline', 'hybrid', 'pure_ai', 'pure_cbr_llm', 'pure_cbr_tpl']
- Student legs:  ['leg_a', 'leg_b']

## Per-persona marginals
```
     persona  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
constructive 249.0 2.951807 1.381651                249.0                2.311914              0.929160 249.0 3.875502 1.466206 249.0 2.967871 1.461527
    socratic 250.0 2.892000 1.308121                250.0                2.474667              0.994195 250.0 4.068000 1.396847 250.0 3.860000 1.560172
 traditional 250.0 2.620000 1.312437                250.0                1.990667              0.837754 250.0 3.452000 1.595581 250.0 2.084000 1.008470
experiential 249.0 2.534137 1.263571                249.0                1.898260              0.716799 249.0 3.160643 1.520789 250.0 3.128000 1.613079
  rule_based 250.0 2.348000 1.312167                250.0                1.654667              0.618283 250.0 3.048000 1.661547 250.0 2.988000 1.561284
```

## Per-architecture marginals
```
architecture  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
      hybrid 250.0 2.920000 1.332932                250.0                2.434667              0.756025 250.0 3.872000 1.502213 250.0 3.400000 1.311059
pure_cbr_llm 249.0 2.803213 1.334042                249.0                2.319946              0.720165 249.0 3.783133 1.524384 250.0 3.280000 1.356821
     pure_ai 249.0 2.791165 1.336734                249.0                2.423025              0.773373 249.0 3.847390 1.560863 249.0 3.650602 1.299197
    baseline 250.0 2.708000 1.301349                250.0                2.154667              0.845540 250.0 3.956000 1.406405 250.0 3.700000 1.345079
pure_cbr_tpl 250.0 2.124000 1.217602                250.0                1.000000              0.000000 250.0 2.148000 1.063305 250.0 1.000000 0.000000
```

## Architecture × Persona cell means
Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.

## Two-way ANOVA (Architecture × Persona)
- **R5**: F_arch=14.97 (p=6e-12, η²p=0.047); F_persona=9.60 (p=1.21e-07, η²p=0.030); F_interaction=1.54 (p=0.0784).
- **quality_composite**: F_arch=256.42 (p=1.11e-16, η²p=0.456); F_persona=74.96 (p=1.11e-16, η²p=0.197); F_interaction=8.92 (p=1.11e-16).

## Pairwise Cohen's d on R5
- **architecture**: largest |d|=0.62 (hybrid vs pure_cbr_tpl, medium)
- **persona**: largest |d|=0.45 (constructive vs rule_based, small)

## Cross-student variance on R5 (leg_a vs leg_b)
- Mean |Δ R5| across cells = 1.31

## Output locations
- Tables: `results/phaseB_smoke_5turn/analysis/tables/`
- Summary: `results/phaseB_smoke_5turn/analysis/analysis_summary.md` (also archived as `analysis_summary_20260517T175327Z.md`)
