# Coverage study

Nested pair pools over the six calibrated families; three pairs held out of every pool. All arms apply a compound to half the batch and a clean view to the other half, so compound exposure is constant and only pair diversity varies.

## Held-out lattice pairs

| arm | held-out delta (pp) | clean acc | vs baseline (paired, pp) |
|---|---:|---:|---:|
| baseline (k=0) | +8.2 | 84.95 | — |
| c3 | -1.8 | 86.25 | -10.0 |
| c6 | +2.0 | 86.08 | -6.1 |
| c12 | +1.4 | 86.41 | -6.8 |
| c12b (lattice+recipe) | +4.5 | 85.84 | -3.7 |
| s6 (singles control) | +7.2 | 86.73 | -1.0 |
| s6b (singles+recipe) | +9.7 | 86.33 | +1.5 |
| s6h (damage-matched singles) | +1.9 | 86.08 | -6.3 |

## Pairing versus mere exposure (c12 - s6 (singles control))

- held-out delta: **-5.77 pp** (-8.4 .. -2.3), 6/6 cells favour c12
- clean accuracy: -0.32 pp (c12 86.41 vs s6 (singles control) 86.73)
- held-out pairs delta difference: **-5.77 pp** over 6 cells
- pool pairs delta difference: **-5.12 pp** over 6 cells

## Pairing versus mere exposure (c12b (lattice+recipe) - s6b (singles+recipe))

- held-out delta: **-5.12 pp** (-9.4 .. +0.3), 5/6 cells favour c12b (lattice+recipe)
- clean accuracy: -0.49 pp (c12b (lattice+recipe) 85.84 vs s6b (singles+recipe) 86.33)
- held-out pairs delta difference: **-5.12 pp** over 6 cells
- pool pairs delta difference: **-5.99 pp** over 6 cells

## Pairing versus mere exposure (c3 - s6h (damage-matched singles))

- held-out delta: **-3.67 pp** (-7.1 .. -1.8), 6/6 cells favour c3
- clean accuracy: +0.16 pp (c3 86.25 vs s6h (damage-matched singles) 86.08)
- held-out pairs delta difference: **-3.67 pp** over 6 cells
- pool pairs delta difference: **-1.86 pp** over 6 cells

## Per-pair held-out delta

| pair | baseline (k=0) | c3 | c6 | c12 | c12b (lattice+recipe) | s6 (singles control) | s6b (singles+recipe) | s6h (damage-matched singles) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| defocus+rain | +16.3 | +0.3 | +3.1 | -0.1 | +5.8 | +9.1 | +12.5 | +0.5 |
| motion+impulse | +3.8 | -7.0 | +1.1 | +1.2 | +1.9 | +5.7 | +7.3 | +4.9 |
| noise+pixelate | +4.5 | +1.5 | +1.9 | +3.2 | +5.9 | +6.7 | +9.1 | +0.4 |

## Transfer: paper grid (never in any lattice)

| arm | haze+glare_5 acc | delta | haze+cloud_5 acc | delta | clean acc |
|---|---:|---:|---:|---:|---:|
| baseline (k=0) | 59.2 | +9.9 | 76.4 | -0.6 | 85.0 |
| c3 | 38.5 | +12.4 | 38.8 | +12.1 | 86.2 |
| c6 | 37.6 | +12.2 | 41.8 | +5.9 | 86.1 |
| c12 | 36.4 | +10.7 | 40.7 | +9.2 | 86.4 |
| c12b (lattice+recipe) | 56.1 | +11.0 | 72.0 | +2.0 | 85.8 |
| s6 (singles control) | 40.0 | +12.5 | 44.6 | +8.4 | 86.7 |
| s6b (singles+recipe) | 57.4 | +12.0 | 72.2 | +2.7 | 86.3 |
