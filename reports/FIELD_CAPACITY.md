# Field capacity probe (NCF phase 0)

32 clean CIFAR-10 test images, severity 3. (K, N) fitted per image on image *i* (300 Adam steps, TV 0.01/0.01) and evaluated on a different image. `fit` is the in-sample reference and is expected to look perfect — with K = 1 the additive field can copy any target — so only the transfer column says anything.

| condition | PSNR identity | PSNR in-sample | PSNR transfer | MSE removed |
|---|---:|---:|---:|---:|
| gaussian_noise | 22.16 | 39.71 | 18.96 | -1.094 |
| shot_noise | 23.47 | 39.72 | 20.25 | -1.138 |
| impulse_noise | 20.41 | 40.08 | 17.96 | -0.795 |
| defocus_blur | 31.37 | 39.78 | 27.57 | -2.459 |
| glass_blur | 23.04 | 39.62 | 19.81 | -1.366 |
| motion_blur | 23.02 | 39.72 | 19.71 | -1.503 |
| zoom_blur | 23.15 | 39.68 | 19.92 | -1.414 |
| snow | 18.70 | 39.86 | 21.00 | +0.340 |
| frost | 14.06 | 39.14 | 19.84 | +0.580 |
| fog | 18.06 | 39.94 | 16.13 | -0.726 |
| brightness | 18.08 | 39.15 | 27.12 | +0.764 |
| contrast | 19.36 | 39.95 | 17.18 | -0.898 |
| elastic_transform | 20.25 | 39.54 | 17.21 | -1.284 |
| pixelate | 30.08 | 39.27 | 25.89 | -2.628 |
| jpeg_compression | 29.47 | 39.49 | 25.36 | -1.945 |
| frost+snow | 10.63 | 36.65 | 18.10 | +0.770 |
| gaussian_noise+shot_noise | 19.80 | 39.71 | 16.78 | -1.010 |
| jpeg_compression+pixelate | 27.13 | 39.41 | 23.52 | -1.569 |
