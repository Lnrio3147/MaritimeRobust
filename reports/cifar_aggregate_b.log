# CIFAR-10-C: compound excess over the union baseline

`delta = e(a+b) - |E_a u E_b|`, 105 unordered pairs x 5 severities, 10000 test images per condition. `super` is the accuracy-space superadditivity `e(a+b) - e(a) - e(b)` that the compound-corruption literature reports; the two disagree, and the disagreement is the point.

| arm | cells | delta mean | median | p10 .. p90 | > 0 | > +2pp | super mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_s17 | 525 | +3.63 | +2.19 | -9.5 .. +18.8 | 62% | 50% | -7.76 |
| lattice_k15 | 525 | +0.83 | -0.46 | -1.2 .. +3.5 | 35% | 16% | -4.29 |
| lattice_k40 | 525 | +0.50 | -0.50 | -1.2 .. +3.0 | 34% | 14% | -4.62 |
| singles | 525 | +2.41 | +0.34 | -0.9 .. +6.5 | 56% | 31% | -2.17 |

## Damage law

If delta were a function of how much damage the compound carries, a damage-only predictor would be enough. Spearman correlations over all cells, with the composite error as the damage measure.

| arm | spearman(delta, compound error) | p | spearman(super, compound error) | p |
|---|---:|---:|---:|---:|
| clean_s17 | +0.26 | 1.0e-09 | -0.19 | 7.1e-06 |
| lattice_k15 | +0.68 | 1.1e-73 | +0.55 | 1.8e-43 |
| lattice_k40 | +0.73 | 7.6e-88 | +0.63 | 9.3e-59 |
| singles | +0.89 | 2.4e-185 | +0.85 | 1.7e-148 |

## By severity

| arm | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| clean_s17 | +0.66 | +2.93 | +4.90 | +6.58 | +3.06 |
| lattice_k15 | -0.75 | -0.69 | -0.19 | +0.77 | +5.01 |
| lattice_k40 | -0.84 | -0.68 | -0.42 | +0.65 | +3.80 |
| singles | -0.62 | -0.20 | +0.75 | +2.76 | +9.35 |

### By family — clean_s17

Mean delta over the ten compounds each family appears in, per severity.

| family | s1 | s2 | s3 | s4 | s5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| brightness | -0.8 | -0.7 | +0.0 | +0.9 | +3.8 | +0.6 |
| contrast | +1.1 | +9.3 | +18.2 | +23.2 | +2.7 | +10.9 |
| defocus_blur | -0.8 | +0.6 | +4.1 | +11.6 | +3.7 | +3.9 |
| elastic_transform | -0.3 | -0.2 | +2.0 | +5.3 | +3.2 | +2.0 |
| fog | +0.1 | +4.3 | +11.0 | +17.9 | +19.1 | +10.5 |
| frost | +1.1 | +4.3 | +8.4 | +9.6 | +9.7 | +6.6 |
| gaussian_noise | +0.2 | -2.6 | -7.0 | -7.2 | -9.5 | -5.2 |
| glass_blur | +3.2 | +5.4 | +5.9 | +3.2 | +2.7 | +4.1 |
| impulse_noise | -1.6 | -2.4 | -2.6 | -4.0 | -9.5 | -4.0 |
| jpeg_compression | +3.1 | +5.8 | +6.9 | +6.6 | +2.6 | +5.0 |
| motion_blur | +2.1 | +7.9 | +12.6 | +13.9 | +10.5 | +9.4 |
| pixelate | -0.2 | +1.8 | +2.4 | +3.1 | -1.2 | +1.2 |
| shot_noise | -0.1 | +0.4 | -2.9 | -3.1 | -7.2 | -2.6 |
| snow | +1.4 | +5.7 | +6.3 | +7.8 | +8.0 | +5.8 |
| zoom_blur | +1.5 | +4.3 | +8.3 | +10.2 | +7.4 | +6.3 |

### By family — lattice_k15

Mean delta over the ten compounds each family appears in, per severity.

| family | s1 | s2 | s3 | s4 | s5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| brightness | -0.8 | -1.1 | -1.0 | -1.0 | +0.6 | -0.6 |
| contrast | -0.8 | -0.7 | +0.0 | +2.2 | +19.4 | +4.0 |
| defocus_blur | -0.6 | -0.7 | -0.4 | +0.1 | +4.4 | +0.6 |
| elastic_transform | -1.1 | -0.7 | -0.5 | +0.4 | +3.3 | +0.3 |
| fog | -0.7 | -0.7 | -0.0 | +2.1 | +12.9 | +2.7 |
| frost | -0.8 | -0.8 | -0.2 | +0.1 | +3.3 | +0.3 |
| gaussian_noise | -0.8 | -0.9 | -0.3 | +0.5 | +2.0 | +0.1 |
| glass_blur | -0.8 | -0.8 | +0.1 | +1.7 | +4.4 | +0.9 |
| impulse_noise | -0.6 | -0.2 | -0.1 | +1.5 | +5.2 | +1.2 |
| jpeg_compression | -0.6 | +0.1 | +1.0 | +2.2 | +5.7 | +1.7 |
| motion_blur | -0.7 | -0.6 | +0.2 | +0.4 | +4.9 | +0.8 |
| pixelate | -0.7 | -0.8 | -0.7 | -0.5 | +1.3 | -0.3 |
| shot_noise | -0.6 | -1.0 | -0.6 | +0.2 | +1.8 | -0.1 |
| snow | -0.8 | -0.6 | +0.1 | +1.3 | +2.7 | +0.5 |
| zoom_blur | -0.8 | -0.8 | -0.4 | +0.2 | +3.4 | +0.3 |

### By family — lattice_k40

Mean delta over the ten compounds each family appears in, per severity.

| family | s1 | s2 | s3 | s4 | s5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| brightness | -0.9 | -1.1 | -1.2 | -1.0 | +0.5 | -0.7 |
| contrast | -0.9 | -0.7 | -0.6 | +0.7 | +11.7 | +2.1 |
| defocus_blur | -0.8 | -0.7 | -0.7 | +0.0 | +3.1 | +0.2 |
| elastic_transform | -1.2 | -0.9 | -0.6 | +0.4 | +2.5 | +0.0 |
| fog | -0.9 | -0.8 | -0.5 | +1.6 | +9.6 | +1.8 |
| frost | -0.9 | -0.6 | -0.4 | +0.2 | +2.2 | +0.1 |
| gaussian_noise | -0.8 | -0.7 | -0.5 | +0.7 | +2.2 | +0.2 |
| glass_blur | -1.0 | -0.8 | -0.2 | +1.6 | +3.2 | +0.5 |
| impulse_noise | -0.6 | -0.3 | -0.2 | +1.5 | +4.8 | +1.0 |
| jpeg_compression | -0.7 | +0.1 | +1.0 | +2.3 | +6.2 | +1.8 |
| motion_blur | -1.0 | -0.7 | -0.1 | +0.5 | +3.5 | +0.5 |
| pixelate | -0.8 | -0.8 | -0.8 | -0.4 | +1.0 | -0.4 |
| shot_noise | -0.7 | -0.8 | -0.5 | +0.4 | +1.9 | +0.1 |
| snow | -0.7 | -0.5 | -0.2 | +1.1 | +2.2 | +0.4 |
| zoom_blur | -0.8 | -0.8 | -0.7 | +0.1 | +2.4 | +0.0 |

### By family — singles

Mean delta over the ten compounds each family appears in, per severity.

| family | s1 | s2 | s3 | s4 | s5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| brightness | -0.7 | -0.9 | -0.7 | -0.3 | +3.0 | +0.1 |
| contrast | -0.8 | -0.2 | +1.1 | +4.9 | +28.9 | +6.8 |
| defocus_blur | -0.7 | -0.5 | +0.2 | +1.4 | +8.8 | +1.8 |
| elastic_transform | -0.7 | -0.3 | +0.3 | +2.3 | +6.6 | +1.7 |
| fog | -0.8 | -0.5 | +0.6 | +3.9 | +18.4 | +4.3 |
| frost | -0.6 | -0.1 | +1.0 | +1.7 | +6.6 | +1.7 |
| gaussian_noise | -0.7 | -0.1 | +1.2 | +2.8 | +6.0 | +1.8 |
| glass_blur | -0.5 | +0.3 | +1.3 | +4.7 | +8.8 | +2.9 |
| impulse_noise | -0.6 | +0.1 | +1.0 | +4.4 | +10.0 | +3.0 |
| jpeg_compression | -0.2 | +0.9 | +2.3 | +5.0 | +10.1 | +3.6 |
| motion_blur | -0.5 | +0.1 | +1.6 | +2.7 | +9.6 | +2.7 |
| pixelate | -0.7 | -0.5 | -0.3 | +0.7 | +4.6 | +0.7 |
| shot_noise | -0.7 | -0.7 | +0.6 | +1.9 | +5.2 | +1.3 |
| snow | -0.7 | -0.2 | +0.6 | +2.9 | +5.6 | +1.6 |
| zoom_blur | -0.5 | -0.3 | +0.7 | +2.4 | +8.1 | +2.1 |
