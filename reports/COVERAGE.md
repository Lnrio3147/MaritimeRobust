# Coverage study

Nested pair pools over the six calibrated families; three pairs held out of every pool. All arms apply a compound to half the batch and a clean view to the other half, so compound exposure is constant and only pair diversity varies.

## Held-out lattice pairs

| arm | held-out delta (pp) | clean acc | vs baseline (paired, pp) |
|---|---:|---:|---:|
| baseline (k=0) | +8.2 | 84.95 | — |
| c3 | -1.8 | 86.25 | -10.0 |
| c6 | +2.0 | 86.08 | -6.1 |
| c12 | +1.4 | 86.41 | -6.8 |
| s6 (singles control) | +7.2 | 86.73 | -1.0 |

## Pairing versus mere exposure (c12 - s6 (singles control))

- held-out delta: **-5.77 pp** (-8.4 .. -2.3), 6/6 cells favour c12
- clean accuracy: -0.32 pp (c12 86.41 vs s6 (singles control) 86.73)
- held-out pairs delta difference: **-5.77 pp** over 6 cells
- pool pairs delta difference: **-5.12 pp** over 6 cells

## Per-pair held-out delta

| pair | baseline (k=0) | c3 | c6 | c12 | s6 (singles control) |
|---|---:|---:|---:|---:|---:|
| defocus+rain | +16.3 | +0.3 | +3.1 | -0.1 | +9.1 |
| motion+impulse | +3.8 | -7.0 | +1.1 | +1.2 | +5.7 |
| noise+pixelate | +4.5 | +1.5 | +1.9 | +3.2 | +6.7 |

## Transfer: paper grid (never in any lattice)

| arm | haze+glare_5 acc | delta | haze+cloud_5 acc | delta | clean acc |
|---|---:|---:|---:|---:|---:|
| baseline (k=0) | 59.2 | +9.9 | 76.4 | -0.6 | 85.0 |
| c3 | 38.5 | +12.4 | 38.8 | +12.1 | 86.2 |
| c6 | 37.6 | +12.2 | 41.8 | +5.9 | 86.1 |
| c12 | 36.4 | +10.7 | 40.7 | +9.2 | 86.4 |
| s6 (singles control) | 37.7 | +13.8 | 43.6 | +8.0 | 86.6 |
