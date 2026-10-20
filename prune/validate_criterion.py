"""Empirical check of the derivation: does each score predict the TRUE loss increase
from deleting one weight?

    python -m prune.validate_criterion

For a trained dense net we (a) collect gradient statistics, (b) delete each of
N random weights one at a time and measure the actual change in training loss,
(c) report Spearman rank correlation between each criterion and that true dL.
"""
import json
import os

import numpy as np

from engine import Tensor, softmax_cross_entropy
from prune.pruner import SelfPruner
from train.data import load_data
from train.loop import Config, train


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(n_probe=600, seed=0):
    data = load_data(); (Xtr, ytr) = data[0]
    net, _ = train(Config(seed=seed, epochs=30), data)
    ws = net.prunable_weights()
    rng = np.random.default_rng(seed)
    # gradient statistics at the trained point (minibatch grads, as used during pruning)
    res = {}
    trackers = {c: SelfPruner(ws, 0.9, c, ema=0.9) for c in ("magnitude", "taylor", "fisher")}
    for b in range(0, len(Xtr), 64):
        for w in ws: w.zero_grad()
        softmax_cross_entropy(net(Tensor(Xtr[b:b+64])), ytr[b:b+64]).backward()
        trackers["fisher"].observe()
    for c in ("magnitude", "taylor"):                       # share the same statistics
        trackers[c].g_bar, trackers[c].g2, trackers[c].n_obs = trackers["fisher"].g_bar, trackers["fisher"].g2, trackers["fisher"].n_obs
    L0 = float(softmax_cross_entropy(net(Tensor(Xtr)), ytr).data)
    sizes = [w.data.size for w in ws]
    probes = []
    while len(probes) < n_probe:
        li = rng.choice(len(ws), p=np.array(sizes) / sum(sizes))
        probes.append((li, int(rng.integers(sizes[li]))))
    true = []
    for li, k in probes:
        w = ws[li]; flat = w.data.reshape(-1); old = flat[k]
        flat[k] = 0.0
        true.append(float(softmax_cross_entropy(net(Tensor(Xtr)), ytr).data) - L0)
        flat[k] = old
    true = np.array(true)
    for c, tr in trackers.items():
        sc = tr.scores()
        s = np.array([sc[li].reshape(-1)[k] for li, k in probes])
        res[c] = {"spearman_vs_true_dL": spearman(s, true)}
    res["n_probe"], res["base_loss"] = n_probe, L0
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/criterion_validation.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
