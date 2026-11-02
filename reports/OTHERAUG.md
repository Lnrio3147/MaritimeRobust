# otheraug: pre-registered predictions O1-O3

Arm `otheraug_s17`: augmentation families disjoint from the fifteen test corruption families, same 200-epoch recipe, same 601-condition grid.

| arm | clean | singles | pairs |
|---|---:|---:|---:|
| clean | 95.47 | 74.77 | 57.29 |
| AugMix | 95.36 | 88.95 | 78.72 |
| lattice k=15 | 94.28 | 91.80 | 87.88 |
| otheraug_s17 | 95.86 | 85.57 | 69.85 |

| id | registered | measured | verdict |
|---|---|---|---|
| O1 | pairs <= 82.88 (k=15 minus 5 pp) | 69.85 (gap +18.04 pp) | holds |
| O2 | held-out excess at s3 > +1 pp | +5.78 pp | holds |
| O3 | clean >= 95.0 | 95.86 | holds |

Context: held-out excess at s3 is +7.86 pp for clean, -0.17 pp for k=15, +1.97 pp for AugMix.

**Verdict.** O1 and O2 both hold: compound accuracy tracks whether the evaluated families were trained on, not augmentation strength. Section 4.7 gains this arm as a control; the claim is unchanged.
