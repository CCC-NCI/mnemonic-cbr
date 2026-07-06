# Phase D analysis summary

- Generated: 2026-05-12T12:08:56Z
- Manifest:  `results/phaseB_smoke/manifest.sqlite`
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
constructive 250.0    2.316 1.276976                250.0                2.152000              0.841795 250.0    3.556 1.600346 250.0    2.888 1.437959
    socratic 250.0    2.008 1.130064                250.0                2.266667              0.878114 250.0    3.864 1.452565 250.0    3.784 1.575842
experiential 250.0    1.956 1.141612                250.0                1.854667              0.701715 250.0    3.052 1.531889 250.0    3.008 1.659806
 traditional 250.0    1.952 1.074390                250.0                1.960000              0.749952 250.0    3.336 1.597910 250.0    2.184 1.059544
  rule_based 250.0    1.908 1.069632                250.0                1.653333              0.565086 250.0    2.956 1.672499 250.0    3.028 1.573888
```

## Per-architecture marginals
```
architecture  R5_n  R5_mean    R5_sd  quality_composite_n  quality_composite_mean  quality_composite_sd  R4_n  R4_mean    R4_sd  R6_n  R6_mean    R6_sd
     pure_ai 250.0    2.332 1.125541                250.0                2.312000              0.655521 250.0    3.624 1.623937 250.0    3.664 1.301491
      hybrid 250.0    2.244 1.229026                250.0                2.298667              0.669104 250.0    3.536 1.615905 250.0    3.352 1.281666
pure_cbr_llm 250.0    2.168 1.135311                250.0                2.270667              0.626702 250.0    3.652 1.545461 250.0    3.264 1.330181
    baseline 250.0    2.096 1.181681                250.0                2.005333              0.746234 250.0    4.048 1.316293 250.0    3.612 1.438657
pure_cbr_tpl 250.0    1.300 0.678115                250.0                1.000000              0.000000 250.0    1.904 0.872829 250.0    1.000 0.000000
```

## Architecture × Persona cell means
Wrote `tables/cell_R5.{csv,md,tex}` and `tables/cell_quality.{csv,md,tex}`.

## Two-way ANOVA (Architecture × Persona)
- **R5**: F_arch=37.62 (p=1.11e-16, η²p=0.109); F_persona=5.90 (p=0.000106, η²p=0.019); F_interaction=1.99 (p=0.0115).
- **quality_composite**: F_arch=264.48 (p=1.11e-16, η²p=0.463); F_persona=49.34 (p=1.11e-16, η²p=0.139); F_interaction=6.80 (p=4.22e-15).

## Pairwise Cohen's d on R5
- **architecture**: largest |d|=1.11 (pure_ai vs pure_cbr_tpl, large)
- **persona**: largest |d|=0.35 (constructive vs rule_based, small)

## Cross-student variance on R5 (leg_a vs leg_b)
- Mean |Δ R5| across cells = 0.87

## Output locations
- Tables: `results/phaseB_smoke/analysis/tables/`
- Summary: `results/phaseB_smoke/analysis/analysis_summary.md` (also archived as `analysis_summary_20260512T120856Z.md`)
