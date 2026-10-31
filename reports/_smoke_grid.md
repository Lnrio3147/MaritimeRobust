# CIFAR-10-C: compound excess over the union baseline

`delta = e(a+b) - |E_a u E_b|`, 105 unordered pairs x 5 severities, 15000 test images per condition. `super` is the accuracy-space superadditivity `e(a+b) - e(a) - e(b)` that the compound-corruption literature reports; the two disagree, and the disagreement is the point.

| arm | cells | delta mean | median | p10 .. p90 | > 0 | > +2pp | super mean |
|---|---:|---:|---:|---:|---:|---:|---:|

## Damage law

If delta were a function of how much damage the compound carries, a damage-only predictor would be enough. Spearman correlations over all cells, with the composite error as the damage measure.

| arm | spearman(delta, compound error) | p | spearman(super, compound error) | p |
|---|---:|---:|---:|---:|

## By severity

| arm | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
