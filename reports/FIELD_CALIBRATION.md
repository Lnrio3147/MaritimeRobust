# Field-bank calibration

PSNR of the deployment pool's conditions against their clean originals (the strength the training augmentation should reach), and PSNR of images degraded by sampled bank fields under different application counts.

## Deployment pool (real CIFAR-10-C conditions)

| severity | mean PSNR over 15 families |
|---|---:|
| 1 | 27.27 |
| 3 | 21.96 |
| 5 | 19.37 |

## Sampled bank fields

| applications m | fields composed | mean PSNR | 10th pct |
|---|---:|---:|---:|
| 1 | 1 | 40.00 | 20.61 |
| 1 | 2 | 29.50 | 16.55 |
| 2 | 1 | 30.07 | 15.38 |
| 2 | 2 | 20.48 | 11.85 |
| 3 | 1 | 30.25 | 11.58 |
| 3 | 2 | 21.64 | 9.34 |

Closest to the single-family severity-3 pool (21.96 dB): m=3, composed=2.

