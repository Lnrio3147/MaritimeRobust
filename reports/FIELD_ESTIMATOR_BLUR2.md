# Field estimator: cross-content transfer (NCF phase 1)

Estimator trained 12000 steps on the continuous prior of `cifar_fields.sample_field` — physical parameters only, no family names. Training objective is equivariant (one field, two contents, estimates must agree; weight 1.0). Identifiability on clean input: mean K 0.974 (target 1), mean |N| 0.0246 (target 0), corr(N, x) +0.254 (target 0). Transfer measured on 64 held-out CIFAR-10 test images: estimate the field from image *i*'s degraded version, apply it to image *j*, compare with *j*'s true degradation.

| condition | PSNR identity | PSNR transfer | MSE removed |
|---|---:|---:|---:|
| gaussian_noise | 22.19 | 18.73 | -1.326 |
| shot_noise | 23.62 | 20.29 | -1.409 |
| impulse_noise | 20.32 | 17.49 | -0.991 |
| defocus_blur | 30.91 | 32.25 | -0.809 |
| glass_blur | 22.16 | 21.46 | -0.270 |
| motion_blur | 22.14 | 23.07 | +0.083 |
| zoom_blur | 22.40 | 23.61 | +0.186 |
| snow | 18.67 | 19.71 | +0.180 |
| frost | 14.23 | 18.19 | +0.539 |
| fog | 17.68 | 18.63 | +0.118 |
| brightness | 18.34 | 19.67 | +0.189 |
| contrast | 18.38 | 19.16 | -0.084 |
| elastic_transform | 19.41 | 19.45 | -0.005 |
| pixelate | 28.99 | 27.45 | -0.972 |
| jpeg_compression | 28.57 | 27.57 | -0.570 |
| frost+snow | 10.05 | 14.83 | +0.640 |
| fog+contrast | 14.99 | 17.07 | +0.345 |
| gaussian_noise+shot_noise | 19.88 | 16.83 | -1.052 |

Families with positive transfer: 6 of 15.
