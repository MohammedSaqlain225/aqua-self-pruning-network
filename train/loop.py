"""Mini-batched training loop (own implementation) with optional self-pruning."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn.layers import make_mlp
from nn.optim import Adam, SGD


@dataclass
class Config:
    sizes: tuple = (64, 256, 128, 10)
    activation: str = "relu"
    init: str | None = None
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-3
    optimizer: str = "adam"
    seed: int = 0
    # pruning ---------------------------------------------------------------
    target_sparsity: float = 0.0
    criterion: str = "fisher"          # magnitude | taylor | fisher
    prune_start_frac: float = 0.15     # fraction of total steps before pruning starts
    prune_end_frac: float = 0.70       # sparsity reaches target here; rest = fine-tune
    prune_every: int = 10              # steps between mask updates
    regrow_frac: float = 0.0           # fraction of kept connections swapped per update (decays to 0)
    min_layer_density: float = 0.03
    ema: float = 0.9


def accuracy(net, X, y):
    return float((net(Tensor(X)).data.argmax(1) == y).mean())


def evaluate(net, X, y):
    logits = net(Tensor(X))
    return float(softmax_cross_entropy(logits, y).data), float((logits.data.argmax(1) == y).mean())


def sparsity_of(net):
    ws = net.prunable_weights()
    alive = sum(float(w.mask.sum()) for w in ws)
    total = sum(w.mask.size for w in ws)
    return 1.0 - alive / total


def train(cfg: Config, data, verbose=False):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    rng = np.random.default_rng(cfg.seed)
    net = make_mlp(list(cfg.sizes), rng, cfg.activation, cfg.init)
    params = net.parameters()
    opt = Adam(params, lr=cfg.lr) if cfg.optimizer == "adam" else SGD(params, lr=cfg.lr)

    pruner = None
    steps_per_epoch = int(np.ceil(len(Xtr) / cfg.batch_size))
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.target_sparsity > 0:
        from prune.pruner import SelfPruner
        pruner = SelfPruner(net.prunable_weights(), cfg.target_sparsity, cfg.criterion,
                            t_start=int(cfg.prune_start_frac * total_steps),
                            t_end=int(cfg.prune_end_frac * total_steps),
                            every=cfg.prune_every, regrow_frac=cfg.regrow_frac,
                            min_layer_density=cfg.min_layer_density, ema=cfg.ema)

    hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "sparsity", "churn"]}
    step, t0 = 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        perm = rng.permutation(len(Xtr))
        losses = []
        for b in range(0, len(Xtr), cfg.batch_size):
            idx = perm[b:b + cfg.batch_size]
            opt.zero_grad()
            loss = softmax_cross_entropy(net(Tensor(Xtr[idx])), ytr[idx])
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            if pruner is not None:
                pruner.observe()                 # stats from this batch's gradients
            opt.step()
            if pruner is not None:
                pruner.maybe_update(step)        # masks change; optimizer resyncs on next step
            losses.append(float(loss.data)); step += 1
        trl, tra = evaluate(net, Xtr, ytr)
        val_l, val_a = evaluate(net, Xva, yva)
        for k, v in zip(hist, [epoch, trl, tra, val_l, val_a, sparsity_of(net),
                               pruner.churn if pruner else 0]):
            hist[k].append(v)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d}  loss {trl:.4f}  train_acc {tra:.4f}  val_acc {val_a:.4f}  sparsity {hist['sparsity'][-1]:.3f}")
    te_l, te_a = evaluate(net, Xte, yte)
    return net, {"history": hist, "test_loss": te_l, "test_acc": te_a, "val_acc": hist["val_acc"][-1],
                 "sparsity": sparsity_of(net), "seconds": time.time() - t0, "config": asdict(cfg)}
