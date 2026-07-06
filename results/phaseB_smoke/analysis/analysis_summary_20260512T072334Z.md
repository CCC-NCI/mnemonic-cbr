# Phase D analysis summary

- Generated: 2026-05-12T07:23:34Z
- Manifest:  `results/phaseB_smoke/manifest.sqlite`
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
constructive  50.0     1.92 1.046666                 50.0                2.020000              0.800822  50.0     3.10 1.787142  50.0     2.50 1.343921
    socratic  50.0     1.88 0.961292                 50.0                2.300000              0.906640  50.0     3.84 1.516710  50.0     3.66 1.572856
 traditional  50.0     1.88 1.154229                 50.0                2.033333              0.857804  50.0     3.04 1.737462  50.0     1.96 0.698687
experiential  50.0     1.72 0.969746                 50.0                1.780000              0.561864  50.0     2.90 1.631951  50.0     2.68 1.621790
  rule_based  50.0     1.62 0.945235                 50.0                1.673333              0.622663  50.0     2.22 1.555635  50.0     2.52 1.343769
```

## Per-architecture marginals
```
architecture  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
     pure_ai  50.0     2.10 1.054630                 50.0                2.386667              0.694520  50.0     3.14 1.829576  50.0     3.38 1.353604
      hybrid  50.0     2.00 1.178030                 50.0                2.306667              0.645374  50.0     3.22 1.810288  50.0     2.90 1.164965
pure_cbr_llm  50.0     1.96 0.968061                 50.0                2.266667              0.631738  50.0     3.48 1.681108  50.0     2.88 1.334625
    baseline  50.0     1.74 1.065412                 50.0                1.846667              0.700534  50.0     3.62 1.510372  50.0     3.16 1.447870
pure_cbr_tpl  50.0     1.22 0.418452                 50.0                1.000000              0.000000  50.0     1.64 0.802038  50.0     1.00 0.000000
```

## Architecture × Persona cell means
Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.

## Two-way ANOVA (Architecture × Persona)
- **R5**: F_arch=6.61 (p=4.8e-05, η²p=0.105); F_persona=0.88 (p=0.477, η²p=0.015); F_interaction=1.23 (p=0.247).
- **quality_composite**: F_arch=56.53 (p=1.11e-16, η²p=0.501); F_persona=10.16 (p=1.41e-07, η²p=0.153); F_interaction=2.01 (p=0.0134).

## Pairwise Cohen's d on R5
- **architecture**: largest |d|=1.10 (pure_ai vs pure_cbr_tpl, large)
- **persona**: largest |d|=0.30 (constructive vs rule_based, small)

## Cross-student variance on R5 (leg_a vs leg_b)
- Mean |Δ R5| across cells = 0.82

## Output locations
- Tables: `results/phaseB_smoke/analysis/tables/`
- Summary: `results/phaseB_smoke/analysis/analysis_summary.md` (also archived as `analysis_summary_20260512T072334Z.md`)
