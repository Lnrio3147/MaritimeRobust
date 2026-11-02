# Field estimator: cross-content transfer (NCF phase 1)

Estimator trained 12000 steps on the continuous prior of `cifar_fields.sample_field` — physical parameters only, no family names. Training objective is equivariant (one field, two contents, estimates must agree; weight 1.0). Identifiability on clean input: mean K 0.978 (target 1), mean |N| 0.0216 (target 0), corr(N, x) +0.184 (target 0). Transfer measured on 64 held-out CIFAR-10 test images: estimate the field from image *i*'s degraded version, apply it to image *j*, compare with *j*'s true degradation.

| condition | PSNR identity | PSNR transfer | MSE removed |
|---|---:|---:|---:|
| gaussian_noise | 22.22 | 21.75 | -0.188 |
| shot_noise | 23.59 | 22.86 | -0.374 |
| impulse_noise | 20.39 | 19.84 | -0.213 |
| defocus_blur | 30.91 | 29.35 | -1.811 |
| glass_blur | 22.24 | 21.64 | -0.273 |
| motion_blur | 22.24 | 21.75 | -0.318 |
| zoom_blur | 22.40 | 21.94 | -0.258 |
| snow | 18.64 | 19.37 | +0.056 |
| frost | 13.64 | 17.48 | +0.530 |
| fog | 17.83 | 18.34 | +0.067 |
| brightness | 18.34 | 19.28 | +0.079 |
| contrast | 18.38 | 18.89 | -0.139 |
| elastic_transform | 19.50 | 19.21 | -0.123 |
| pixelate | 28.99 | 27.60 | -1.411 |
| jpeg_compression | 28.57 | 27.81 | -0.663 |
| frost+snow | 10.91 | 15.02 | +0.584 |
| fog+contrast | 15.11 | 16.38 | +0.223 |
| gaussian_noise+shot_noise | 19.88 | 19.50 | -0.122 |

Families with positive transfer: 4 of 15.
