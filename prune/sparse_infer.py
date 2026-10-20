"""Sparse-aware inference paths (no dense-times-zero) and cost accounting.

Three forward paths over the same trained, pruned network:
  dense     : full matmul with the zeros still in the matrices  (what a naive
              "pruned" model does: same work as the unpruned model)
  csc       : our own CSC gather/segment-sum kernel; work ~ nnz
  compacted : dead neurons physically removed -> smaller *dense* matrices that
              run on BLAS.  This is the form that gives real wall-clock wins
              in NumPy without a sparse BLAS.
"""
from __future__ import annotations

import time

import numpy as np


def extract(net):
    """-> list of (W_eff, bias) with pruned slots as exact zeros. ReLU MLP only."""
    out = []
    for l in net.linears():
        W = l.weight.data * (l.weight.mask if l.weight.mask is not None else 1.0)
        out.append((W.copy(), l.bias.data.copy()))
    return out


def dense_forward(layers, X):
    for i, (W, b) in enumerate(layers):
        X = X @ W + b
        if i < len(layers) - 1:
            X = np.maximum(X, 0.0)
    return X


class CSCLayer:
    """Weights stored column-major (per output unit): rows, vals, column pointers."""
    def __init__(self, W, b):
        self.n_in, self.n_out = W.shape
        cols, rows = np.nonzero(W.T)                    # sorted by column
        self.rows, self.vals = rows, W[rows, cols]
        counts = np.bincount(cols, minlength=self.n_out)
        self.nonempty = counts > 0
        self.starts = (np.cumsum(counts) - counts)[self.nonempty]
        self.b = b
        self.nnz = len(self.vals)

    def __call__(self, X):
        out = np.zeros((X.shape[0], self.n_out))
        if self.nnz:
            contrib = X[:, self.rows] * self.vals       # (B, nnz): work proportional to nnz
            out[:, self.nonempty] = np.add.reduceat(contrib, self.starts, axis=1)
        return out + self.b


class SparseNet:
    def __init__(self, layers):
        self.layers = [CSCLayer(W, b) for W, b in layers]

    def __call__(self, X):
        for i, l in enumerate(self.layers):
            X = l(X)
            if i < len(self.layers) - 1:
                X = np.maximum(X, 0.0)
        return X


def compact(layers):
    """Remove neurons that cannot influence the output, *exactly* (ReLU nets).

    * hidden unit with no alive OUTGOING weight  -> its value is never used: drop.
    * hidden unit with no alive INCOMING weight  -> output is the constant
      relu(bias): fold  relu(b_j) * W_next[j, :]  into the next layer's bias, drop.
    Iterates to a fixed point because each removal can orphan neighbours.
    """
    L = [(W.copy(), b.copy()) for W, b in layers]
    changed = True
    while changed:
        changed = False
        for k in range(len(L) - 1):
            W, b = L[k]; Wn, bn = L[k + 1]
            no_out = ~np.any(Wn != 0, axis=1)
            no_in = ~np.any(W != 0, axis=0)
            drop = no_out | no_in
            if drop.any():
                const = no_in & ~no_out
                bn = bn + (np.maximum(b[const], 0.0)[:, None] * Wn[const]).sum(axis=0)
                keep = ~drop
                L[k] = (W[:, keep], b[keep]); L[k + 1] = (Wn[keep], bn)
                changed = True
    return L


def count_macs(layers, sparse=False):
    """Multiply-accumulates per sample (FLOPs = 2 x MACs)."""
    return int(sum((np.count_nonzero(W) if sparse else W.size) for W, _ in layers))


def count_params(layers, sparse=False):
    return int(sum((np.count_nonzero(W) if sparse else W.size) + b.size for W, b in layers))


def bench(fn, X, repeats=30, warmup=3):
    for _ in range(warmup):
        fn(X)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(X)
        best = min(best, (time.perf_counter() - t0) / repeats)
    return best


def cost_report(net, Xtest, batch_sizes=(1, 64, 512)):
    layers = extract(net)
    comp = compact(layers)
    sn = SparseNet(layers)
    ref = dense_forward(layers, Xtest)
    assert np.allclose(sn(Xtest), ref, atol=1e-9), "CSC forward disagrees with dense"
    assert np.allclose(dense_forward(comp, Xtest), ref, atol=1e-9), "compaction changed the function"
    rep = {
        "macs_dense_model": count_macs(layers),
        "macs_sparse_nnz": count_macs(layers, sparse=True),
        "macs_compacted_dense": count_macs(comp),
        "params_dense_model": count_params(layers),
        "params_alive": count_params(layers, sparse=True),
        "compacted_shapes": [W.shape for W, _ in comp],
        "latency_seconds": {},
    }
    rep["flop_reduction_nnz"] = rep["macs_dense_model"] / max(1, rep["macs_sparse_nnz"])
    for B in batch_sizes:
        X = np.tile(Xtest, (int(np.ceil(B / len(Xtest))), 1))[:B]
        rep["latency_seconds"][str(B)] = {
            "dense_times_zero": bench(lambda x: dense_forward(layers, x), X),
            "csc_sparse": bench(sn, X),
            "compacted_dense": bench(lambda x: dense_forward(comp, x), X),
        }
    return rep
