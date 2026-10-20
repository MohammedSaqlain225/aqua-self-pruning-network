"""Composable layers built only on engine.Tensor."""
from __future__ import annotations

import numpy as np

from engine import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """y = x @ W_eff + b,  W_eff = W * mask  (W has shape (in, out)).

    init='he'     : N(0, 2/fan_in)   -- keeps Var(activation) ~ constant through ReLU
    init='xavier' : N(0, 2/(fan_in+fan_out)) -- for tanh/sigmoid/linear
    init='naive'  : N(0, 1)          -- deliberately bad, used in the init ablation
    """

    def __init__(self, n_in, n_out, rng: np.random.Generator, init="he", prunable=True):
        std = {"he": np.sqrt(2.0 / n_in),
               "xavier": np.sqrt(2.0 / (n_in + n_out)),
               "naive": 1.0}[init]
        self.weight = Tensor(rng.normal(0.0, std, size=(n_in, n_out)), requires_grad=True)
        self.bias = Tensor(np.zeros(n_out), requires_grad=True)
        self.prunable = prunable
        if prunable:
            self.weight.mask = np.ones((n_in, n_out))

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        w = self.weight.masked(self.weight.mask) if self.weight.mask is not None else self.weight
        return x @ w + self.bias            # (N,out) + (out,) -> broadcast


class ReLU(Module):
    def forward(self, x): return x.relu()


class Tanh(Module):
    def forward(self, x): return x.tanh()


class GELU(Module):
    def forward(self, x): return x.gelu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def linears(self):
        return [l for l in self.layers if isinstance(l, Linear)]

    def prunable_weights(self):
        return [l.weight for l in self.linears() if l.prunable]


def make_mlp(sizes, rng, activation="relu", init=None):
    act = {"relu": ReLU, "tanh": Tanh, "gelu": GELU}[activation]
    init = init or ("xavier" if activation == "tanh" else "he")
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        layers.append(Linear(a, b, rng, init=("xavier" if last else init)))
        if not last:
            layers.append(act())
    return Sequential(*layers)
