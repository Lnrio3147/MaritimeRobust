# Capacity axis: compound excess versus backbone size

delta = e(a+b) - |E_a ∪ E_b|, matched-marginal protocol, mean over cells (fold0/1/2 × seed17/29).

| pair | sev | convnext_tiny | convnext_small | convnext_base |
|---|---:|---:|---:|---:|
| haze+cloud | 2 | -2.2 (6) | -1.3 (6) | -1.5 (6) |
| haze+cloud | 3 | -2.3 (6) | -2.8 (6) | -0.3 (6) |
| haze+cloud | 4 | -1.4 (6) | -1.9 (6) | -1.0 (6) |
| haze+cloud | 5 | -0.6 (6) | +0.2 (6) | +0.2 (6) |
| haze+glare | 2 | -1.1 (6) | -1.3 (6) | -0.8 (6) |
| haze+glare | 3 | -0.6 (6) | -1.4 (6) | -0.3 (6) |
| haze+glare | 4 | -0.4 (6) | -0.4 (6) | -0.1 (6) |
| haze+glare | 5 | +9.9 (6) | +9.7 (6) | +11.2 (6) |

## Severity-5 excess only

- **haze+cloud**: tiny -0.6, small +0.2, base +0.2
- **haze+glare**: tiny +9.9, small +9.7, base +11.2

## Frozen-feature probe

A linear head cannot overfit 412 images the way a 28M-parameter fine-tune can, so a probe isolates whether the excess lives in the representation itself.

**delta is not comparable across models of different accuracy.** Its ceiling is the remaining headroom ``100 - err_union``, and a weaker model exhausts more of it on the single corruptions. The normalised column divides by that headroom; read the capacity claim off *that*, not off the raw delta.

| backbone | probe clean | fine-tuned clean | hg5 delta | hg5 headroom | hg5 normalised | hc5 delta | cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| convnext_tiny | 72.3 | 85.0 | +9.7 | 50.5 | 19.2% | +2.9 | 1 |
| convnext_small | 73.6 | 85.2 | +4.7 | 48.9 | 9.6% | -3.1 | 6 |
| convnext_base | 74.8 | 85.5 | +4.0 | 49.0 | 8.1% | -2.0 | 6 |

Fine-tuned tiny, same normalisation: 14.3%.
