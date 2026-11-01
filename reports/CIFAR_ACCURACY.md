# CIFAR-10-C: absolute accuracy per condition family

Percent correct, averaged over the conditions in each family. The excess measures elsewhere in this project are relative to the same model's own singles; this table is the absolute number.

| arm | clean | singles | pairs | held-out pairs s3 |
|---|---:|---:|---:|---:|
| clean_s17 | 95.47 | 74.77 | 57.29 | 61.14 |
| lattice_k15_s17 | 94.28 | 91.80 | 87.88 | 89.86 |
| lattice_k40_s17 | 94.16 | 91.95 | 88.52 | 90.16 |
| singles_s17 | 94.99 | 92.30 | 86.77 | 90.04 |
| lattice_k15_s29 | 94.31 | 91.73 | 87.78 | 90.08 |
| singles_s29 | 95.03 | 92.26 | 86.58 | 89.93 |
| augmix_s17 | 95.36 | 88.95 | 78.72 | 84.84 |

Relative to the clean reference model, on the same cells:

| arm | Δ pairs | Δ held-out pairs s3 | Δ clean |
|---|---:|---:|---:|
| lattice_k15_s17 | +30.60 | +28.72 | -1.19 |
| lattice_k40_s17 | +31.23 | +29.02 | -1.31 |
| singles_s17 | +29.48 | +28.90 | -0.48 |
| lattice_k15_s29 | +30.50 | +28.94 | -1.16 |
| singles_s29 | +29.29 | +28.79 | -0.44 |
| augmix_s17 | +21.43 | +23.70 | -0.11 |
