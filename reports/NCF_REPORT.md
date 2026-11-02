# NCF arm adjudication

Accuracy (%) on the same 601-condition grid as Table 1. `paired-held` is the mean over pair conditions containing fog, frost or snow -- the families excluded from the tier-B field bank and never named to the estimator in any bank.

| arm | clean | singles | pairs | pairs w/ held-out families | pairs w/o |
|---|---:|---:|---:|---:|---:|
| clean_s17 | 95.47 | 74.77 | 57.29 | 58.99 | 56.28 |
| augmix_s17 | 95.36 | 88.95 | 78.72 | 78.46 | 78.87 |
| singles_s17 | 94.99 | 92.30 | 86.77 | 86.84 | 86.73 |
| lattice_k15_s17 | 94.28 | 91.80 | 87.88 | 87.64 | 88.03 |
| ncf_s_a_s17 | 95.08 | 87.09 | 75.63 | 77.23 | 74.69 |
| ncf_c_a_s17 | 95.08 | 87.75 | 77.46 | 78.48 | 76.86 |
| ncf_c_a_r2_s17 | 95.21 | 87.98 | 77.61 | 79.57 | 76.46 |
| ncf_c_b_r2_s17 | 94.80 | 87.89 | 78.06 | 79.01 | 77.49 |
| ncf_c_ab_r2_s17 | 95.05 | 90.76 | 84.66 | 84.19 | 84.94 |
| ncf_c_bb_r2_s17 | 94.55 | 90.88 | 85.43 | 84.96 | 85.71 |
