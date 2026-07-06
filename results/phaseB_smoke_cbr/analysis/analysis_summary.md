# Phase D analysis summary

- Generated: 2026-05-12T08:52:42Z
- Manifest:  `results/phaseB_smoke_cbr/manifest.sqlite`
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
constructive  50.0     2.00 1.106567                 50.0                2.093333              0.849276  50.0     3.12 1.709995  50.0     2.46 1.281103
    socratic  50.0     1.88 1.081194                 50.0                2.366667              1.015191  50.0     3.64 1.625812  50.0     3.62 1.627443
 traditional  50.0     1.80 0.989743                 50.0                1.960000              0.912598  50.0     2.96 1.817826  50.0     1.74 0.632778
  rule_based  50.0     1.74 1.065412                 50.0                1.706667              0.612197  50.0     2.48 1.693204  50.0     2.76 1.478554
experiential  50.0     1.64 0.898070                 50.0                1.800000              0.690066  50.0     2.78 1.515767  50.0     2.62 1.537159
```

## Per-architecture marginals
```
architecture  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
     pure_ai  50.0     2.18 1.155113                 50.0                2.386667              0.825775  50.0     2.92 1.700420  50.0     3.00 1.428571
      hybrid  50.0     2.14 1.143036                 50.0                2.386667              0.681335  50.0     3.44 1.716189  50.0     3.10 1.328648
pure_cbr_llm  50.0     1.90 0.839096                 50.0                2.300000              0.762648  50.0     3.40 1.784285  50.0     2.98 1.406835
    baseline  50.0     1.74 1.084398                 50.0                1.853333              0.738203  50.0     3.60 1.551826  50.0     3.12 1.364865
pure_cbr_tpl  50.0     1.10 0.303046                 50.0                1.000000              0.000000  50.0     1.62 0.830294  50.0     1.00 0.000000
```

## Architecture × Persona cell means
Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.

## Two-way ANOVA (Architecture × Persona)
- **R5**: F_arch=10.82 (p=4.86e-08, η²p=0.161); F_persona=1.06 (p=0.376, η²p=0.019); F_interaction=1.70 (p=0.0478).
- **quality_composite**: F_arch=46.16 (p=1.11e-16, η²p=0.451); F_persona=8.83 (p=1.22e-06, η²p=0.136); F_interaction=1.96 (p=0.0164).

## Pairwise Cohen's d on R5
- **architecture**: largest |d|=1.28 (pure_ai vs pure_cbr_tpl, large)
- **persona**: largest |d|=0.36 (constructive vs experiential, small)

## Cross-student variance on R5 (leg_a vs leg_b)
- Mean |Δ R5| across cells = 0.79

## Output locations
- Tables: `results/phaseB_smoke_cbr/analysis/tables/`
- Summary: `results/phaseB_smoke_cbr/analysis/analysis_summary.md` (also archived as `analysis_summary_20260512T085242Z.md`)
