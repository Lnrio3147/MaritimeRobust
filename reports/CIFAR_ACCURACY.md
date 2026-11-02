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
| augmix_s29 | 95.85 | 89.39 | 78.73 | 84.96 |
| otheraug_s17 | 95.86 | 85.57 | 69.85 | 75.65 |
| ncf_c_a_s17 | 95.08 | 87.75 | 77.46 | 82.84 |
| ncf_s_a_s17 | 95.08 | 87.09 | 75.63 | 79.97 |
| ncf_c_a_r2_s17 | 95.21 | 87.98 | 77.61 | 83.06 |
| ncf_c_b_r2_s17 | 94.80 | 87.89 | 78.06 | 83.28 |

Relative to the clean reference model, on the same cells:

| arm | Δ pairs | Δ held-out pairs s3 | Δ clean |
|---|---:|---:|---:|
| lattice_k15_s17 | +30.60 | +28.72 | -1.19 |
| lattice_k40_s17 | +31.23 | +29.02 | -1.31 |
| singles_s17 | +29.48 | +28.90 | -0.48 |
| lattice_k15_s29 | +30.50 | +28.94 | -1.16 |
| singles_s29 | +29.29 | +28.79 | -0.44 |
| augmix_s17 | +21.43 | +23.70 | -0.11 |
| augmix_s29 | +21.44 | +23.82 | +0.38 |
| otheraug_s17 | +12.56 | +14.51 | +0.39 |
| ncf_c_a_s17 | +20.17 | +21.70 | -0.39 |
| ncf_s_a_s17 | +18.35 | +18.83 | -0.39 |
| ncf_c_a_r2_s17 | +20.33 | +21.91 | -0.26 |
| ncf_c_b_r2_s17 | +20.77 | +22.14 | -0.67 |
