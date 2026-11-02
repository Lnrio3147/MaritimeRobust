# CIFAR-10-C: compound accuracy gain, with pair-level intervals

Gain over `clean_s17` on the 525 pair conditions, in percentage points.
Intervals are percentile bootstrap over the 105 corruption pairs (20000 resamples), which keeps the five severities of a pair together.

| arm | gain (pp) | 95% CI | worst pair | best pair |
|---|---:|---|---:|---:|
| lattice_k15_s17 | +30.60 | [+28.47, +32.69] | +4.4 | +58.3 |
| lattice_k40_s17 | +31.23 | [+29.14, +33.32] | +4.7 | +58.8 |
| singles_s17 | +29.48 | [+27.47, +31.50] | +4.8 | +57.9 |
| lattice_k15_s29 | +30.50 | [+28.44, +32.60] | +4.4 | +58.4 |
| singles_s29 | +29.29 | [+27.30, +31.33] | +4.8 | +57.8 |
| augmix_s17 | +21.43 | [+20.19, +22.65] | +4.0 | +36.4 |
| augmix_s29 | +21.44 | [+20.15, +22.73] | +3.9 | +38.5 |
| otheraug_s17 | +12.56 | [+11.44, +13.69] | +2.2 | +26.7 |
| ncf_c_a_s17 | +20.17 | [+17.98, +22.52] | +5.2 | +57.1 |
| ncf_s_a_s17 | +18.35 | [+16.00, +20.78] | +3.7 | +55.8 |
| ncf_c_a_r2_s17 | +20.33 | [+18.03, +22.74] | +5.2 | +57.2 |
| ncf_c_b_r2_s17 | +20.77 | [+18.58, +23.03] | +4.9 | +57.0 |

## lattice_k15_s17 minus augmix_s17

Mean +9.17 pp, 95% CI [+8.00, +10.37]; lattice_k15_s17 is higher on 104 of 105 pairs.

## Across seeds

- lattice_k15: s17: +30.60, s29: +30.50 pp; spread 0.10 pp
- singles: s17: +29.48, s29: +29.29 pp; spread 0.19 pp
- augmix: s17: +21.43, s29: +21.44 pp; spread 0.01 pp
