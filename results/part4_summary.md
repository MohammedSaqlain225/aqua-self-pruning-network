# Part 4 results (test accuracy in percent, mean +/- sd over 8 seeds)

Dense baseline (0% sparsity): test acc 98.16 +/- 0.49 (mean +/- sd, n=8)

| target sparsity | magnitude | taylor | fisher | magnitude+regrow | fisher+regrow |
|---|---|---|---|---|---|
| 0.50 (achieved 0.5000) | 97.99 +/- 0.69 | 97.95 +/- 0.57 | 97.95 +/- 0.42 | 97.78 +/- 0.54 | 98.26 +/- 0.25 |
| 0.75 (achieved 0.7500) | 98.19 +/- 0.66 | 97.99 +/- 0.36 | 97.95 +/- 0.44 | 98.23 +/- 0.47 | 98.26 +/- 0.32 |
| 0.90 (achieved 0.9000) | 98.12 +/- 0.46 | 98.06 +/- 0.49 | 98.33 +/- 0.30 | 98.09 +/- 0.46 | 97.99 +/- 0.64 |
| 0.95 (achieved 0.9500) | 98.02 +/- 0.60 | 97.64 +/- 0.45 | 98.06 +/- 0.56 | 97.88 +/- 0.59 | 98.09 +/- 0.85 |
| 0.98 (achieved 0.9800) | 97.57 +/- 0.53 | 97.33 +/- 0.33 | 97.33 +/- 1.00 | 97.57 +/- 0.59 | 97.50 +/- 0.39 |
| 0.99 (achieved 0.9900) | 96.74 +/- 0.88 | 96.35 +/- 0.62 | 96.56 +/- 0.73 | 96.77 +/- 0.59 | 97.26 +/- 0.94 |

## Paired comparison vs magnitude (same seeds => same init & batch order)

diff = acc(method) - acc(magnitude) in percentage points; 95% paired bootstrap CI; W/L/T = seeds where method is better/worse/tied

| sparsity | method | mean diff (pp) | 95% CI | W/L/T | CI excludes 0? |
|---|---|---|---|---|---|
| 0.50 | taylor | -0.03 | [-0.28, +0.21] | 4/4/0 | no |
| 0.50 | fisher | -0.03 | [-0.38, +0.31] | 4/3/1 | no |
| 0.50 | fisher+regrow | +0.28 | [-0.17, +0.66] | 5/2/1 | no |
| 0.50 | magnitude+regrow | -0.21 | [-0.42, +0.00] | 1/4/3 | no |
| 0.75 | taylor | -0.21 | [-0.80, +0.35] | 3/4/1 | no |
| 0.75 | fisher | -0.24 | [-0.66, +0.14] | 3/5/0 | no |
| 0.75 | fisher+regrow | +0.07 | [-0.38, +0.45] | 4/2/2 | no |
| 0.75 | magnitude+regrow | +0.03 | [-0.35, +0.45] | 3/4/1 | no |
| 0.90 | taylor | -0.07 | [-0.66, +0.45] | 4/3/1 | no |
| 0.90 | fisher | +0.21 | [-0.28, +0.69] | 5/3/0 | no |
| 0.90 | fisher+regrow | -0.14 | [-0.83, +0.52] | 4/3/1 | no |
| 0.90 | magnitude+regrow | -0.03 | [-0.28, +0.21] | 3/4/1 | no |
| 0.95 | taylor | -0.38 | [-0.94, +0.10] | 3/4/1 | no |
| 0.95 | fisher | +0.03 | [-0.49, +0.52] | 4/2/2 | no |
| 0.95 | fisher+regrow | +0.07 | [-0.38, +0.66] | 2/3/3 | no |
| 0.95 | magnitude+regrow | -0.14 | [-0.52, +0.21] | 2/4/2 | no |
| 0.98 | taylor | -0.24 | [-0.59, +0.14] | 3/5/0 | no |
| 0.98 | fisher | -0.24 | [-0.94, +0.42] | 3/4/1 | no |
| 0.98 | fisher+regrow | -0.07 | [-0.31, +0.21] | 3/4/1 | no |
| 0.98 | magnitude+regrow | +0.00 | [-0.56, +0.45] | 3/1/4 | no |
| 0.99 | taylor | -0.38 | [-1.15, +0.45] | 3/4/1 | no |
| 0.99 | fisher | -0.17 | [-0.69, +0.42] | 3/5/0 | no |
| 0.99 | fisher+regrow | +0.52 | [-0.14, +1.15] | 6/2/0 | no |
| 0.99 | magnitude+regrow | +0.03 | [-0.59, +0.63] | 4/3/1 | no |

## Real cost (seed 0, fisher+regrow models; MACs/sample; latency = best-of-5 mean, this machine, single thread)

| model | MACs dense-times-zero | MACs nnz | MACs compacted | FLOP red. (nnz) | alive params | compacted shapes | lat B=1 dense / csc / compact (us) | lat B=64 (us) |
|---|---|---|---|---|---|---|---|---|
| dense@0.0 | 50432 | 50432 | 50432 | 1.0x | 50826 | [[64, 256], [256, 128], [128, 10]] | 14 / 184 / 14 | 145 / 31655 / 167 |
| fisher+regrow@0.5 | 50432 | 25216 | 48570 | 2.0x | 25610 | [[64, 256], [256, 121], [121, 10]] | 14 / 107 / 13 | 164 / 13326 / 175 |
| fisher+regrow@0.75 | 50432 | 12608 | 47645 | 4.0x | 13002 | [[64, 251], [251, 121], [121, 10]] | 15 / 64 / 13 | 161 / 5604 / 163 |
| fisher+regrow@0.9 | 50432 | 5043 | 30854 | 10.0x | 5437 | [[64, 164], [164, 117], [117, 10]] | 14 / 40 / 12 | 165 / 1228 / 119 |
| fisher+regrow@0.95 | 50432 | 2522 | 18056 | 20.0x | 2916 | [[64, 113], [113, 88], [88, 10]] | 13 / 30 / 9 | 165 / 614 / 75 |
| fisher+regrow@0.98 | 50432 | 1009 | 8850 | 50.0x | 1403 | [[64, 63], [63, 66], [66, 10]] | 13 / 23 / 8 | 158 / 290 / 48 |
| fisher+regrow@0.99 | 50432 | 504 | 4946 | 100.1x | 898 | [[64, 39], [39, 50], [50, 10]] | 14 / 21 / 7 | 159 / 177 / 35 |

## Falsifiable claim

At 99% sparsity (achieved 0.9900), Fisher-saliency pruning (no regrowth) reaches 96.56% test accuracy vs 96.74% for magnitude pruning, mean over 8 paired seeds (diff -0.17 pp, 95% paired-bootstrap CI [-0.69, +0.42], fisher better/worse/tied on 3/5/0 seeds); dense reference 98.16%.
