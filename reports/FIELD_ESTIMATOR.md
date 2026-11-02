# Field estimator: cross-content transfer (NCF phase 1)

Estimator trained 12000 steps on the continuous prior of `cifar_fields.sample_field` — physical parameters only, no family names. Training objective is equivariant (one field, two contents, estimates must agree; weight 1.0). Identifiability on clean input: mean K 0.976 (target 1), mean |N| 0.0210 (target 0), corr(N, x) +0.196 (target 0). Transfer measured on 64 held-out CIFAR-10 test images: estimate the field from image *i*'s degraded version, apply it to image *j*, compare with *j*'s true degradation.

| condition | PSNR identity | PSNR transfer | MSE removed |
|---|---:|---:|---:|
| gaussian_noise | 22.16 | 18.80 | -1.199 |
| shot_noise | 23.79 | 20.45 | -1.296 |
| impulse_noise | 20.25 | 17.40 | -0.968 |
| defocus_blur | 31.29 | 29.38 | -1.369 |
| glass_blur | 22.87 | 22.40 | -0.143 |
| motion_blur | 22.93 | 22.18 | -0.302 |
| zoom_blur | 23.14 | 22.51 | -0.257 |
| snow | 18.49 | 19.33 | +0.144 |
| frost | 13.63 | 17.28 | +0.521 |
| fog | 18.08 | 18.89 | +0.052 |
| brightness | 18.36 | 19.63 | +0.208 |
| contrast | 19.30 | 19.69 | -0.096 |
| elastic_transform | 19.67 | 19.59 | -0.027 |
| pixelate | 29.55 | 28.59 | -0.502 |
| jpeg_compression | 28.63 | 28.20 | -0.225 |
| frost+snow | 10.55 | 14.80 | +0.601 |
| fog+contrast | 15.80 | 17.13 | +0.221 |
| gaussian_noise+shot_noise | 19.93 | 16.71 | -1.130 |

Families with positive transfer: 4 of 15.
