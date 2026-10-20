# The Self-Pruning Network (AQUA challenge)

A reverse-mode autodiff engine, MLP library, Adam optimizer, and a gradual
self-pruning training loop - all in pure Python + NumPy. The network removes its
own least-important connections during training, and every efficiency claim is
backed by committed numbers.

**scikit-learn usage (stated explicitly, as required):** `sklearn.datasets.load_digits`
is used *only* to load the dataset (it ships inside the wheel; no network needed).
No sklearn model, optimizer, metric or gradient code is used. No PyTorch / TF / JAX /
Keras / HuggingFace. Autodiff, backward pass, pruning, optimizers and training loop are
my own code.

## Install

```bash
uv venv && uv pip install -r requirements.txt     # or: pip install -r requirements.txt
```
Python >= 3.10. Everything is seeded; numbers reproduce from a clean clone.

## Commands

| What | Command | Time |
|---|---|---|
| Gradient-check + masked-weight + pruner tests | `python -m pytest -q` *(or, with no pytest: `python -m tests.run_tests`)* | ~3 s |
| Part 2: dense training, learning curve, init ablation | `python -m train.run_part2` | ~10 s |
| Part 3: self-pruning run (90% target) | `python -m train.run_part3` | ~5 s |
| Part 4: Pareto sweep + baselines + cost + claim | `python -m train.run_part4 --fresh` | ~25 min (1 CPU) |
| Part 4 tables/plots only (from committed raw numbers) | `python -m train.run_part4 --analyze` | ~5 s |
| Validate the importance criterion vs true loss change | `python -m prune.validate_criterion` | ~10 s |

## Layout

```
engine/tensor.py        autodiff: Tensor, ops, backward(), masked(), softmax-CE
nn/layers.py            Linear (with mask), ReLU/Tanh/GELU, Sequential, make_mlp
nn/optim.py             Adam, SGD+momentum - mask-aware (per-slot state, reset on mask change)
prune/pruner.py         cubic schedule, magnitude/taylor/fisher criteria, regrowth
prune/sparse_infer.py   CSC sparse forward, dead-neuron compaction, MAC/latency accounting
prune/validate_criterion.py   score vs measured loss increase
train/                  data, loop, run_part{2,3,4}.py
tests/                  gradcheck suite, masked-weight tests, pruner + sparse tests
results/                committed plots + raw numbers (part4_raw.jsonl, part4_summary.md, ...)
DESIGN.md               derivations and design answers
```

## Results

See `results/part4_summary.md` (full tables), `results/part4_pareto.png`, and the
headline numbers in the **Results summary** section at the bottom of this file.

## Results summary (all numbers from committed files; digits dataset, 8 seeds)

Dense baseline: **98.16 +/- 0.49 %** test accuracy (360 test samples, so one sample = 0.28 pp).

| target sparsity | magnitude | fisher | fisher+regrow |
|---|---|---|---|
| 90% | 98.12 +/- 0.46 | 98.33 +/- 0.30 | 97.99 +/- 0.64 |
| 95% | 98.02 +/- 0.60 | 98.06 +/- 0.56 | 98.09 +/- 0.85 |
| 99% | 96.74 +/- 0.88 | 96.56 +/- 0.73 | 97.26 +/- 0.94 |

Full tables, all five methods and paired statistics: `results/part4_summary.md`. Plot: `results/part4_pareto.png`.

### Falsifiable claims (honest version)

1. **Supported:** "Removing 99% of weights (504 of 50,432 remain, verified as exact zeros) costs about 1.4-1.6 pp of test accuracy (96.6-97.3% vs 98.16% dense), mean over 8 seeds."
2. **Not supported:** that saliency pruning beats magnitude pruning *on this task*. Mean differences are within +/-0.5 pp and every paired 95% bootstrap CI includes 0 (e.g. 99% sparsity, fisher vs magnitude: -0.17 pp, CI [-0.69, +0.42], better on 3/8 seeds). The best-looking result, fisher+regrow at 99% (+0.52 pp, CI [-0.14, +1.15], 6/8 seeds), is suggestive, not significant. Digits is easy and the test set small; a harder dataset and more seeds would be needed to resolve a difference of this size.
3. **Supported (small sample):** the Fisher score tracks measured single-weight loss increase better than magnitude (Spearman 0.65 vs 0.51, `results/criterion_validation.json`).
4. **Cost:** MACs per sample fall 20x at 95% and 100x at 99%. Our unstructured NumPy CSC kernel is *slower* than dense BLAS up to ~98% sparsity; dense-times-zero is no speedup. Compacting dead neurons into smaller dense matrices gave real wall-clock gains (99%, batch 64: 35 us vs 159 us; batch 1: 7 vs 14 us; 90%: ~1.4x). Latencies are machine-dependent (see `results/part4_cost.json`).

### Notes on process
* Hyper-parameters were tuned on the validation split only; test accuracy is reported, not tuned on.
* A bug found during tuning: the per-layer minimum-connection floor exceeded the whole keep budget at 98% sparsity and caused training collapse. The floor is now capped at half of the budget (`prune/pruner.py`).
