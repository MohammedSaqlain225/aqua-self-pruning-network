"""Part 3: one self-pruning run (default 90% target, Fisher-saliency + regrowth).

    python -m train.run_part3 [--sparsity 0.9] [--criterion fisher] [--regrow 0.1]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.data import load_data
from train.loop import Config, train

OUT = "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", default="fisher", choices=["magnitude", "taylor", "fisher"])
    ap.add_argument("--regrow", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = load_data()
    cfg = Config(target_sparsity=a.sparsity, criterion=a.criterion, regrow_frac=a.regrow, seed=a.seed)
    net, res = train(cfg, data, verbose=True)

    # --- honesty checks on the final model -------------------------------------
    alive = total = 0
    for w in net.prunable_weights():
        assert np.all(w.data[w.mask == 0] == 0.0), "pruned raw weight is not an exact zero"
        alive += int(np.count_nonzero(w.data)); total += w.data.size
    print(f"achieved sparsity (exact zeros) = {1 - alive/total:.4f}  | alive={alive}/{total}")
    print(f"test_acc={res['test_acc']:.4f}  val_acc={res['val_acc']:.4f}  total mask churn={res['history']['churn'][-1]}")

    dense = json.load(open(f"{OUT}/part2_dense.json"))["dense"] if os.path.exists(f"{OUT}/part2_dense.json") else None
    if dense:
        print(f"dense reference: test_acc={dense['test_acc']:.4f}  -> accuracy cost = {100*(dense['test_acc']-res['test_acc']):+.2f} pp")
    json.dump(res, open(f"{OUT}/part3_selfprune.json", "w"), indent=1)

    h = res["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h["epoch"], h["sparsity"]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("sparsity")
    ax[0].set_title("Cubic sparsity ramp (measured from masks)")
    ax[1].plot(h["epoch"], h["val_acc"], label="val acc"); ax[1].plot(h["epoch"], h["train_acc"], label="train acc")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0.8, 1.01); ax[1].legend(); ax[1].set_title("Accuracy while pruning")
    plt.tight_layout(); plt.savefig(f"{OUT}/part3_selfprune_curve.png", dpi=130)


if __name__ == "__main__":
    main()
