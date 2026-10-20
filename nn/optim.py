"""Optimizers that are correct in the presence of pruning masks.

The bug we are guarding against
-------------------------------
A naive optimizer treats a masked weight as "just a weight whose gradient
happens to be zero".  With momentum / Adam that is wrong in three ways:

 1. m (and v) are NOT zero for a weight pruned mid-training, so the update
    -lr * m_hat / (sqrt(v_hat)+eps) keeps pushing the *raw* weight for many
    steps after pruning (momentum "ghost updates").
 2. When the connection is later revived, the raw weight and its (m, v) are
    stale: the first step uses statistics from a different era of training.
 3. Adam's bias-correction uses a single global step t.  A freshly revived
    slot has no history, so (1 - beta^t) ~ 1 under-corrects and its first
    updates are far smaller than lr; correct behaviour needs a PER-SLOT t.

Our fix: whenever a slot's mask changes (pruned OR revived) we zero its
weight, m, v and its step counter t; every step we multiply the gradient and the
update by the mask. A revived slot therefore starts exactly like a brand-new
parameter initialised at 0.
"""
from __future__ import annotations

import numpy as np


class _MaskAware:
    def _sync_mask(self, p, st):
        """Detect mask changes since the last step and reset state for them."""
        cur = np.ones_like(p.data) if p.mask is None else p.mask
        if st["mask"] is None:
            st["mask"] = cur.copy()
            return
        changed = cur != st["mask"]
        if changed.any():
            for k in st["bufs"]:
                st[k][changed] = 0.0
            p.data[changed] = 0.0            # pruned -> exact zero; revived -> starts at 0
            st["mask"] = cur.copy()
        p.data *= cur                        # pruned slots are exactly 0 in raw storage too


class SGD(_MaskAware):
    def __init__(self, params, lr=0.05, momentum=0.9):
        self.params, self.lr, self.mu = list(params), lr, momentum
        self.state = {id(p): {"v": np.zeros_like(p.data), "mask": None, "bufs": ["v"]} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            st = self.state[id(p)]
            self._sync_mask(p, st)
            if p.grad is None:
                continue
            g = p.grad if p.mask is None else p.grad * p.mask
            st["v"] = self.mu * st["v"] + g
            upd = self.lr * st["v"]
            if p.mask is not None:
                st["v"] *= p.mask
                upd = upd * p.mask
            p.data -= upd


class Adam(_MaskAware):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, betas[0], betas[1], eps
        self.state = {id(p): {"m": np.zeros_like(p.data), "v": np.zeros_like(p.data),
                              "t": np.zeros_like(p.data), "mask": None,
                              "bufs": ["m", "v", "t"]} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            st = self.state[id(p)]
            self._sync_mask(p, st)
            if p.grad is None:
                continue
            alive = np.ones_like(p.data) if p.mask is None else p.mask
            g = p.grad * alive
            st["t"] += alive                                   # per-slot step counter
            st["m"] = self.b1 * st["m"] + (1 - self.b1) * g
            st["v"] = self.b2 * st["v"] + (1 - self.b2) * g * g
            t = np.maximum(st["t"], 1.0)
            m_hat = st["m"] / (1 - self.b1 ** t)
            v_hat = st["v"] / (1 - self.b2 ** t)
            upd = self.lr * m_hat / (np.sqrt(v_hat) + self.eps) * alive
            st["m"] *= alive
            st["v"] *= alive
            p.data -= upd
            p.data *= alive
