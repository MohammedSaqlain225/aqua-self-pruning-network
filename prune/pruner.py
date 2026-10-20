"""Gradual self-pruning with importance criteria and optional regrowth.

Importance criteria (all estimate  dL = L(w -> 0) - L(w)  for removing a slot):

  magnitude : |w|                      (baseline; ignores the loss entirely)
  taylor    : | w * gbar |             first-order term of the Taylor expansion
  fisher    : 0.5 * F * w^2            second-order term with diagonal empirical
                                       Fisher F = EMA(g^2) as the Hessian proxy

Derivation (also in DESIGN.md):
  L(w - w e_i) ~ L(w) - g_i w_i + 0.5 H_ii w_i^2
  => dL_i ~ -g_i w_i + 0.5 H_ii w_i^2
  Near a minimum E[g] ~ 0 so the first-order term vanishes (and its sign is
  noise), leaving 0.5 H_ii w_i^2 with H_ii ~ F_ii = E[g_i^2] (Gauss-Newton /
  Fisher for cross-entropy).  `taylor` keeps only the first term, `fisher` the second.
"""
from __future__ import annotations

import numpy as np


def cubic_sparsity(step, t0, t1, s_final, s_init=0.0):
    """Zhu & Gupta (2017):  s_t = s_f + (s_i - s_f) * (1 - (t-t0)/(t1-t0))^3."""
    if step < t0:
        return s_init
    if step >= t1:
        return s_final
    frac = (step - t0) / (t1 - t0)
    return s_final + (s_init - s_final) * (1.0 - frac) ** 3


class SelfPruner:
    def __init__(self, weights, target_sparsity, criterion="fisher", t_start=0, t_end=1000,
                 every=10, regrow_frac=0.0, min_layer_density=0.03, ema=0.9):
        assert criterion in ("magnitude", "taylor", "fisher")
        self.ws = list(weights)
        self.s_final, self.criterion = target_sparsity, criterion
        self.t0, self.t1, self.every = t_start, t_end, every
        self.regrow_frac, self.min_density, self.ema = regrow_frac, min_layer_density, ema
        self.N = sum(w.data.size for w in self.ws)
        self.g_bar = [np.zeros_like(w.data) for w in self.ws]    # EMA of dL/dW_eff (signed)
        self.g2 = [np.zeros_like(w.data) for w in self.ws]       # EMA of (dL/dW_eff)^2  (Fisher diag)
        self.n_obs = 0
        self.churn = 0                                           # cumulative # revived slots

    # ------------------------------------------------------------ statistics
    def observe(self):
        """Call after backward(): uses dense_grad so pruned slots are tracked too."""
        a = self.ema
        self.n_obs += 1
        for i, w in enumerate(self.ws):
            g = w.dense_grad if w.dense_grad is not None else np.zeros_like(w.data)
            self.g_bar[i] = a * self.g_bar[i] + (1 - a) * g
            self.g2[i] = a * self.g2[i] + (1 - a) * g * g

    def _debias(self, x):                                        # EMA zero-init bias correction
        return x / (1.0 - self.ema ** max(self.n_obs, 1))

    def scores(self):
        out = []
        for i, w in enumerate(self.ws):
            if self.criterion == "magnitude":
                s = np.abs(w.data)
            elif self.criterion == "taylor":
                s = np.abs(w.data * self._debias(self.g_bar[i]))
            else:
                s = 0.5 * self._debias(self.g2[i]) * w.data ** 2
            out.append(s)
        return out

    # --------------------------------------------------------------- updates
    def current_sparsity(self):
        return 1.0 - sum(w.mask.sum() for w in self.ws) / self.N

    def maybe_update(self, step):
        if step < self.t0 or step > self.t1 or (step - self.t0) % self.every != 0:
            return
        self.update(step)

    def update(self, step):
        s_t = cubic_sparsity(step, self.t0, self.t1, self.s_final)
        keep_total = int(round((1.0 - s_t) * self.N))
        # sparsity never decreases below what is already achieved
        keep_total = min(keep_total, int(sum(w.mask.sum() for w in self.ws)))
        # regrowth budget decays linearly to 0 at t1 so the mask settles before fine-tuning
        decay = max(0.0, 1.0 - (step - self.t0) / max(1, self.t1 - self.t0))
        n_regrow = int(round(self.regrow_frac * decay * keep_total))
        n_prune_keep = keep_total - n_regrow

        old = [w.mask.copy() for w in self.ws]
        sc = self.scores()
        flat = np.concatenate([np.where(old[i] > 0, sc[i], -np.inf).ravel() for i in range(len(self.ws))])
        # protect each layer's top `min_density` slots so no layer is wiped out
        # the floor may use at most half of the keep budget (otherwise it overrides the target)
        floors = [max(1, int(np.ceil(self.min_density * w.data.size))) for w in self.ws]
        scale = min(1.0, 0.5 * keep_total / max(1, sum(floors)))
        offset = 0
        for i, w in enumerate(self.ws):
            k = max(1, int(scale * floors[i]))
            seg = flat[offset:offset + w.data.size]
            k = min(k, int((seg > -np.inf).sum()))
            if k > 0:
                top = np.argpartition(-seg, k - 1)[:k]
                flat[offset + top] = np.inf
            offset += w.data.size
        keep_idx = np.argpartition(-flat, n_prune_keep - 1)[:n_prune_keep] if n_prune_keep > 0 else np.array([], int)
        new_flat = np.zeros(self.N)
        new_flat[keep_idx] = 1.0

        # regrow: among previously-pruned slots take largest |EMA gradient|
        # (first-order loss decrease available if the slot is revived from 0)
        if n_regrow > 0:
            gscore = np.concatenate([np.where(old[i] > 0, -np.inf, np.abs(self._debias(self.g_bar[i]))).ravel()
                                     for i in range(len(self.ws))])
            n_avail = int((gscore > -np.inf).sum())
            n_regrow = min(n_regrow, n_avail)
            if n_regrow > 0:
                ridx = np.argpartition(-gscore, n_regrow - 1)[:n_regrow]
                new_flat[ridx] = 1.0

        offset = 0
        for i, w in enumerate(self.ws):
            m = new_flat[offset:offset + w.data.size].reshape(w.data.shape)
            offset += w.data.size
            self.churn += int(((m > 0) & (old[i] == 0)).sum())
            w.mask = m                      # optimizer notices the change and resets state
            w.data *= m                     # pruned raw weights become an exact 0
