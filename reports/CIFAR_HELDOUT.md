# CIFAR-10-C coverage arms, held-out pairs

## clean_s17

- held-out pairs, all severities: delta +5.73 pp
- held-out pairs, s3: delta +7.86 pp (super +1.82)
- all pairs, s3: delta +4.90 pp
## lattice_k15_s17

- held-out pairs, all severities: delta +1.25 pp
- held-out pairs, s3: delta -0.17 pp (super -5.03)
- all pairs, s3: delta -0.19 pp
## lattice_k40_s17

- held-out pairs, all severities: delta +0.69 pp
- held-out pairs, s3: delta -0.42 pp (super -5.29)
- all pairs, s3: delta -0.42 pp
## singles_s17

- held-out pairs, all severities: delta +2.84 pp
- held-out pairs, s3: delta +0.33 pp (super -3.97)
- all pairs, s3: delta +0.75 pp
## lattice_k15_s29

- held-out pairs, all severities: delta +1.13 pp
- held-out pairs, s3: delta -0.48 pp (super -5.34)
- all pairs, s3: delta -0.29 pp
## singles_s29

- held-out pairs, all severities: delta +2.93 pp
- held-out pairs, s3: delta +0.41 pp (super -3.83)
- all pairs, s3: delta +0.88 pp
## augmix_s17

- held-out pairs, all severities: delta +4.47 pp
- held-out pairs, s3: delta +1.97 pp (super -2.24)
- all pairs, s3: delta +2.95 pp
## augmix_s29

- held-out pairs, all severities: delta +5.03 pp
- held-out pairs, s3: delta +2.44 pp (super -1.55)
- all pairs, s3: delta +3.41 pp
## otheraug_s17

- held-out pairs, all severities: delta +6.60 pp
- held-out pairs, s3: delta +5.78 pp (super +1.25)
- all pairs, s3: delta +7.57 pp
## ncf_c_a_s17

- held-out pairs, all severities: delta +3.81 pp
- held-out pairs, s3: delta +1.71 pp (super -3.03)
- all pairs, s3: delta +3.22 pp
## ncf_s_a_s17

- held-out pairs, all severities: delta +4.62 pp
- held-out pairs, s3: delta +3.64 pp (super -1.20)
- all pairs, s3: delta +4.67 pp
## ncf_c_a_r2_s17

- held-out pairs, all severities: delta +3.65 pp
- held-out pairs, s3: delta +1.55 pp (super -3.05)
- all pairs, s3: delta +3.40 pp
## ncf_c_b_r2_s17

- held-out pairs, all severities: delta +3.91 pp
- held-out pairs, s3: delta +1.48 pp (super -3.62)
- all pairs, s3: delta +2.86 pp

## Frozen contrasts

- lattice_k15_s17 minus clean_s17 (held-out, s3): -8.02 pp
- lattice_k40_s17 minus clean_s17 (held-out, s3): -8.27 pp
- B2 paired-minus-singles, seed 17 (held-out, s3): -0.49 pp (threshold: < -1 pp)
- B2 paired-minus-singles, seed 29 (held-out, s3): -0.89 pp (threshold: < -1 pp)
- B2 across 2 seeds: mean -0.69 pp, range -0.89 … -0.49 pp
