# Port fidelity against the published CIFAR-10-C archive

Model accuracy on published vs regenerated images, per severity (percentage points; regenerated minus published). The archive is not bit-reproducible upstream, so agreement in accuracy is the check that matters: the same corruption family at the same calibrated severity.

Worst absolute difference: **1.17 pp** (tolerance 1.0 pp) — REVIEW.

| corruption | s1 | s2 | s3 | s4 | s5 | mean | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| brightness | +0.00 | +0.01 | +0.00 | -0.01 | -0.01 | -0.00 | ok |
| contrast | +0.00 | -0.01 | +0.00 | +0.02 | +0.01 | +0.00 | ok |
| defocus_blur | +0.02 | +0.01 | +0.03 | +0.00 | +0.01 | +0.01 | ok |
| elastic_transform | -0.02 | +0.29 | +0.57 | +0.40 | -0.57 | +0.13 | ok |
| fog | -0.01 | +0.22 | +0.25 | +0.17 | -0.45 | +0.04 | ok |
| frost | -0.01 | +0.36 | +0.32 | +0.60 | -0.12 | +0.23 | ok |
| gaussian_noise | -0.36 | -0.04 | +0.22 | -0.10 | +0.61 | +0.07 | ok |
| glass_blur | -0.34 | -0.28 | +0.48 | +0.40 | +0.36 | +0.12 | ok |
| impulse_noise | +0.06 | -0.03 | -0.50 | +0.24 | +0.35 | +0.02 | ok |
| jpeg_compression | +0.15 | +0.03 | -0.36 | -0.29 | -0.68 | -0.23 | ok |
| motion_blur | +0.17 | +0.64 | -0.40 | -0.06 | -0.55 | -0.04 | ok |
| pixelate | -0.02 | +0.00 | +0.00 | +0.02 | +0.00 | +0.00 | ok |
| shot_noise | +0.46 | +0.55 | +0.12 | -0.14 | +0.37 | +0.27 | ok |
| snow | -0.31 | -0.04 | -0.15 | -1.17 | -0.40 | -0.41 | ok |
| zoom_blur | +0.01 | -0.01 | -0.01 | -0.02 | +0.00 | -0.01 | ok |
