# Additivity-constrained training

delta = e(a+b) - |E_a ∪ E_b|; acc = accuracy under the named condition.
Means over the cells present for each arm (fold0/1/2 × seed17/29).

## Clean accuracy

| arm | clean acc | matched_haze_5 | matched_glare_5 |
|---|---:|---:|---:|
| baseline | 0.85 | 0.82 | 0.74 |
| a0 (control) | 0.85 | 0.83 | 0.75 |
| a1 (additivity) | 0.85 | 0.82 | 0.74 |

## Compound accuracy and excess

| arm | haze+glare_5 acc | delta | haze+cloud_5 acc | delta | haze+glare_4 acc | delta |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.59 | +9.9 | 0.76 | -0.6 | 0.76 | -0.4 |
| a0 (control) | 0.63 | +8.3 | 0.78 | -2.1 | 0.77 | +0.0 |
| a1 (additivity) | 0.61 | +8.7 | 0.75 | +0.6 | 0.76 | +0.9 |

## Paired a1 - a0 per cell

- **matched_haze+glare_5 acc**: mean -1.29 pp (-4.4 .. +2.4), 2/6 positive
- **matched_haze+cloud_5 acc**: mean -2.75 pp (-6.8 .. +2.4), 1/6 positive
- **clean acc**: mean +0.08 pp (-1.0 .. +2.4), 2/6 positive
- **haze+glare_5 delta**: a0 +8.3 -> a1 +8.7 (+0.4)
- **haze+cloud_5 delta**: a0 -2.1 -> a1 +0.6 (+2.7)
- **haze+glare_4 delta**: a0 +0.0 -> a1 +0.9 (+0.9)
