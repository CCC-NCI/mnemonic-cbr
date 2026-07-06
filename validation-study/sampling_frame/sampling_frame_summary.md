# Sampling Frame Summary

- n: 150
- per stratum target: 10
- seed: 42
- band definition: {'low': [1, 2], 'mid': [3], 'high': [4, 5]}
- SHA 256: `cb679ed22e5f81784a79c049098947b93f17236b6373775dfe6d14b2a3996904`

## Per stratum counts

| Architecture | low | mid | high | total |
|--------------|----:|----:|-----:|------:|
| baseline | 10 | 10 | 10 | 30 |
| hybrid | 10 | 10 | 10 | 30 |
| pure_ai | 10 | 10 | 10 | 30 |
| pure_cbr_llm | 10 | 10 | 10 | 30 |
| pure_cbr_tpl | 10 | 10 | 10 | 30 |

## Persona balance

| Persona | n |
|---------|--:|
| traditional | 31 |
| socratic | 31 |
| constructive | 30 |
| experiential | 29 |
| rule_based | 29 |

## Student leg balance

| Leg | n |
|-----|--:|
| leg_a | 74 |
| leg_b | 76 |

## Tercile fallback substitutions

None. Every (architecture, R5 band) stratum had at least the per stratum target available under the primary band definition.
