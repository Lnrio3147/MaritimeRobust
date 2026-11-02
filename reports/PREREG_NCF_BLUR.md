# Pre-registration: kernel (convolution) term in the field bank

Frozen before the blur-head estimator's bank is built and before any arm using it is
trained. Date: 2026-11-02. The estimator checkpoint and the bank do not exist yet at
write time.

## What is being tested

`docs/MODULE_NCF.md` §8.5 结论 5 locates the representation boundary at convolution:
photometric and noise families reach the supervised level from estimated fields alone
(shot 92.6 vs 92.3 supervised), while every convolution family sits below AugMix
(motion 83.2 vs 92.6, glass 75.7 vs 88.4), and the compound arm is therefore 77.46
against 87.88 for the k=15 lattice.

This arm adds one term to the field: a per-image anisotropic Gaussian kernel
`(sigma_x, sigma_y, theta)`, predicted by a head on the same trunk and applied before
the gain/additive field — the order the degradation generator uses. Nothing else
changes: same prior, same training recipe, same 601-condition grid, same bank builder.

## Predictions

**B2 (identifiability, checked first).** On 512 held-out synthetic draws from the
training prior, the predicted `sigma` correlates with the true `sigma` at
`r >= +0.50` per axis, and the MAE is at least 25% below the predict-the-mean
baseline. On real CIFAR-10-C at severity 3, mean predicted `sigma_x` is
`<= 0.40 px` for the eight non-convolution families (gaussian/shot/impulse noise,
fog, frost, snow, brightness, contrast, elastic) and `>= 0.80 px` for at least three
of {defocus_blur, motion_blur, zoom_blur}.

**B1 (representation).** With the blur term in the bank and application count
re-calibrated by PSNR (not by accuracy), the compound tier-A arm rises by
`>= +3.0 pp` over `ncf_c_a_r2` (77.61), i.e. to `>= 80.6`, and the five convolution
families rise by `>= +4.0 pp` each over their no-blur values in the per-family table.

**B3 (clean cost).** Clean accuracy of the blur arm stays `>= 94.5`.

## Falsification rules, written before the run

- **B2 fails** → the bank's kernel parameters are noise, any change in B1 is
  attributable to something else, and no B1 result may be reported with the
  representation interpretation. The arm is then reported as a failed extension.
- **B2 holds, B1 fails** → convolution is not the whole boundary; the residual gap
  is not a representation issue and the module's scope stays as measured in §8.5.
- **B1 holds but the gain is < 2 pp** → the representation fix is real but too small
  to change the module's standing against AugMix (78.72); report as a scope note,
  not a contribution.

No prediction above depends on accuracy for any threshold or hyperparameter.
