# Field estimator: cross-content transfer (NCF phase 1)

Estimator trained 6000 steps on the continuous prior of `cifar_fields.sample_field` — physical parameters only, no family names. Transfer measured on 128 held-out CIFAR-10 test images: estimate the field from image *i*'s degraded version, apply it to image *j*, compare with *j*'s true degradation.

| condition | PSNR identity | PSNR transfer | MSE removed |
|---|---:|---:|---:|
| gaussian_noise | 22.19 | 9.56 | -19.318 |
| shot_noise | 23.93 | 9.65 | -30.856 |
| impulse_noise | 20.33 | 9.54 | -11.870 |
| defocus_blur | 31.11 | 10.24 | -158.643 |
| glass_blur | 22.70 | 10.07 | -21.944 |
| motion_blur | 22.58 | 10.44 | -21.277 |
| zoom_blur | 22.87 | 10.38 | -23.034 |
| snow | 18.77 | 9.51 | -8.745 |
| frost | 14.03 | 11.02 | -1.693 |
| fog | 18.00 | 13.57 | -2.454 |
| brightness | 18.10 | 9.70 | -7.001 |
| contrast | 18.83 | 14.37 | -3.333 |
| elastic_transform | 19.85 | 10.21 | -11.234 |
| pixelate | 29.32 | 9.98 | -120.870 |
| jpeg_compression | 28.83 | 9.91 | -96.842 |
| frost+snow | 10.03 | 11.54 | +0.142 |
| fog+contrast | 15.61 | 17.88 | +0.118 |
| gaussian_noise+shot_noise | 19.99 | 9.43 | -11.584 |

Families with positive transfer: 0 of 15.
