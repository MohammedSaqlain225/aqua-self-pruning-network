"""Part 2: dense MLP with Adam. Learning curve + weight-init ablation + NaN guard.

    python -m train.run_part2
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import Tensor
from nn.layers import make_mlp
from train.data import load_data
from train.loop import Config, train

OUT = "results"


def activation_stats(init, data, seed=0):
    """Std of pre-activations at each layer on one batch (forward-only, untrained)."""
    X = data[0][0][:256]
    net = make_mlp([64, 256, 128, 10], np.random.default_rng(seed), "relu", init)
    x, stds = Tensor(X), []
    for l in net.layers:
        x = l(x)
        if hasattr(l, "weight"):
            stds.append(float(x.data.std()))
    return stds


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_data()
    cfg = Config(seed=0)
    print("== Dense baseline (Adam) ==")
    net, res = train(cfg, data, verbose=True)
    print(f"final: test_acc={res['test_acc']:.4f} val_acc={res['val_acc']:.4f}  (no NaNs: guarded by FloatingPointError)")

    # initialisation ablation ------------------------------------------------
    abl = {}
    for init in ["he", "naive"]:
        c = Config(seed=0, init=init, epochs=30)
        try:
            _, r = train(c, data)
            abl[init] = {"first_epoch_train_loss": r["history"]["train_loss"][0],
                         "val_acc": r["val_acc"], "layer_preact_std": activation_stats(init, data), "nan": False}
        except FloatingPointError:
            abl[init] = {"nan": True, "layer_preact_std": activation_stats(init, data)}
    print("== init ablation ==")
    for k, v in abl.items():
        print(k, v)

    json.dump({"dense": res, "init_ablation": abl}, open(f"{OUT}/part2_dense.json", "w"), indent=1)

    h = res["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h["epoch"], h["train_loss"], label="train"); ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(h["epoch"], h["train_acc"], label="train"); ax[1].plot(h["epoch"], h["val_acc"], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].set_title("Accuracy")
    plt.tight_layout(); plt.savefig(f"{OUT}/part2_learning_curve.png", dpi=130)
    print(f"wrote {OUT}/part2_dense.json and {OUT}/part2_learning_curve.png")


if __name__ == "__main__":
    main()
