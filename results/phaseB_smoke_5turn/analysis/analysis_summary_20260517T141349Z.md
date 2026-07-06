# Phase D analysis summary

- Generated: 2026-05-17T14:13:49Z
- Manifest:  `results/phaseB_smoke_5turn/manifest.sqlite`
- Filter:    `real`

---

## Manifest summary
- Total rows in manifest: 250
- Rows after filter:      250
- Personas:      ['constructive', 'experiential', 'rule_based', 'socratic', 'traditional']
- Architectures: ['baseline', 'hybrid', 'pure_ai', 'pure_cbr_llm', 'pure_cbr_tpl']
- Student legs:  ['leg_a', 'leg_b']

## Per-persona marginals
```
     persona  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
    socratic  50.0 2.680000 1.376775                 50.0                2.520000              1.067262  50.0  4.10000 1.474269  50.0 3.700000 1.555110
constructive  49.0 2.469388 1.430356                 49.0                2.210884              0.929690  49.0  3.44898 1.646476  49.0 2.428571 1.241639
 traditional  50.0 2.440000 1.342553                 50.0                2.033333              0.952976  50.0  3.26000 1.700060  50.0 1.780000 0.615779
experiential  50.0 2.220000 1.093394                 50.0                1.806667              0.680436  50.0  2.70000 1.619398  50.0 2.660000 1.520070
  rule_based  50.0 2.120000 1.255843                 50.0                1.733333              0.728431  50.0  2.70000 1.798525  50.0 2.740000 1.425855
```

## Per-architecture marginals
```
architecture  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
      hybrid  50.0 2.660000 1.408574                 50.0                2.560000              0.919529  50.0 3.640000 1.723310  50.0 3.000000 1.261680
pure_cbr_llm  50.0 2.520000 1.403203                 50.0                2.360000              0.699336  50.0 3.640000 1.723310  50.0 2.940000 1.331104
    baseline  50.0 2.440000 1.264266                 50.0                2.026667              0.915574  50.0 3.640000 1.613211  50.0 3.260000 1.396935
     pure_ai  49.0 2.265306 1.319336                 49.0                2.360544              0.744849  49.0 3.326531 1.760537  49.0 3.122449 1.317078
pure_cbr_tpl  50.0 2.040000 1.087217                 50.0                1.000000              0.000000  50.0 1.960000 1.105829  50.0 1.000000 0.000000
```

## Architecture × Persona cell means
Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.

## Two-way ANOVA (Architecture × Persona)
- **R5**: F_arch=1.65 (p=0.162, η²p=0.029); F_persona=1.40 (p=0.236, η²p=0.024); F_interaction=0.50 (p=0.947).
- **quality_composite**: F_arch=44.72 (p=1.11e-16, η²p=0.444); F_persona=11.67 (p=1.25e-08, η²p=0.173); F_interaction=2.24 (p=0.00497).

## Pairwise Cohen's d on R5
- **architecture**: largest |d|=0.49 (hybrid vs pure_cbr_tpl, small)
- **persona**: largest |d|=0.42 (rule_based vs socratic, small)

## Cross-student variance on R5 (leg_a vs leg_b)
- Mean |Δ R5| across cells = 1.33

## Output locations
- Tables: `results/phaseB_smoke_5turn/analysis/tables/`
- Summary: `results/phaseB_smoke_5turn/analysis/analysis_summary.md` (also archived as `analysis_summary_20260517T141349Z.md`)
