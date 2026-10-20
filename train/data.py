"""Dataset: sklearn's bundled 8x8 digits (1797 samples, 64 features, 10 classes).

scikit-learn is used ONLY to load this dataset (`load_digits`). No sklearn
model, optimizer, metric, or gradient code is used anywhere.
The data ships inside the sklearn wheel, so no network access is required.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits


def load_data(split_seed: int = 0):
    """Fixed stratification-free split: 1257 train / 180 val / 360 test.

    The split depends only on `split_seed` (kept constant across experiments),
    NOT on the model seed, so seed-to-seed variance reflects init + batch order.
    Hyper-parameters were chosen on `val`; `test` is only reported.
    """
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0               # pixels are 0..16
    y = d.target.astype(np.int64)
    idx = np.random.default_rng(split_seed).permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr, n_va = 1257, 180
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    Xte, yte = X[n_tr + n_va:], y[n_tr + n_va:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8            # standardise with TRAIN stats only
    f = lambda a: (a - mu) / sd
    return (f(Xtr), ytr), (f(Xva), yva), (f(Xte), yte)
