# Coverage study

Nested pair pools over the six calibrated families; three pairs held out of every pool. All arms apply a compound to half the batch and a clean view to the other half, so compound exposure is constant and only pair diversity varies.

## Held-out lattice pairs

| arm | held-out delta (pp) | clean acc | vs baseline (paired, pp) |
|---|---:|---:|---:|
| baseline (k=0) | +8.2 | 84.95 | — |
| c3 | -2.4 | 85.92 | -11.7 |

## Per-pair held-out delta

| pair | baseline (k=0) | c3 |
|---|---:|---:|
| defocus+rain | +16.3 | -1.3 |
| motion+impulse | +3.8 | -7.9 |
| noise+pixelate | +4.5 | +2.1 |

## Transfer: paper grid (never in any lattice)

| arm | haze+glare_5 acc | delta | haze+cloud_5 acc | delta | clean acc |
|---|---:|---:|---:|---:|---:|
| baseline (k=0) | 59.2 | +9.9 | 76.4 | -0.6 | 85.0 |
| c3 | 33.8 | +13.3 | 34.6 | +12.6 | 85.1 |
