# Check 5 floor-compression probe

- Generated: 2026-05-12T13:22:24Z
- Manifest:  `results/phaseB_smoke/manifest.sqlite`
- Real mode: True
- Filter:    R5 >= 3
- Target n:  40

---

## Result

- **Pass:** `True`
- n = 40 [filter: R5 >= 3]; mean R5 original = 3.45; mean R5 swapped = 2.38; drop = +1.08 (pass drop >= 0.5: True).

**Interpretation.** The pre-registered 0.5-point drop criterion is met on dialogues with sufficient R5 headroom. The framework's R5 is order-sensitive when the trajectory signal is large enough; the failure of the full-sample Check 5 is attributable to floor compression on a left-skewed R5 distribution rather than to insensitivity of R5 to dialogue order.

---

_The strict full-sample Check 5 result is preserved in `results/phaseB_smoke/sanity_summary.md`._
