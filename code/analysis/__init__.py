"""Analysis package — turns the Phase B/D manifest into manuscript tables.

Spec reference: REBUILD_SPECIFICATION_v3.md §3.6 (aggregation),
§3.4 (separate reporting: R5 alone, (R1+R2+R3)/3 composite, R4 and R6
as descriptives).

Public surface:
  - load_manifest(path)              (aggregate.py)
  - per_persona_table(df)            (aggregate.py)
  - per_architecture_table(df)       (aggregate.py)
  - cell_table(df)                   (aggregate.py)
  - two_way_anova(df, outcome)       (anova.py)
  - pairwise_cohens_d(df, factor, outcome)   (effect_sizes.py)
  - export_csv, export_markdown,
    export_latex                     (export.py)
"""
