# Field-bank calibration

PSNR of the deployment pool's conditions against their clean originals (the strength the training augmentation should reach), and PSNR of images degraded by sampled bank fields under different application counts.

## Deployment pool (real CIFAR-10-C conditions)

| severity | mean PSNR over 15 families |
|---|---:|
| 1 | 27.21 |
| 3 | 21.93 |
| 5 | 19.41 |

## Sampled bank fields

| applications m | fields composed | mean PSNR | 10th pct |
|---|---:|---:|---:|
| 1 | 1 | 34.42 | 19.22 |
| 1 | 2 | 26.79 | 16.60 |
| 2 | 1 | 25.96 | 14.78 |
| 2 | 2 | 18.99 | 11.05 |
| 3 | 1 | 26.35 | 11.91 |
| 3 | 2 | 17.99 | 8.98 |

Closest to the single-family severity-3 pool (21.93 dB): m=2, composed=2.

