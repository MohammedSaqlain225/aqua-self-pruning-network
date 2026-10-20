"""Minimal reverse-mode autodiff engine on NumPy arrays.

Design notes
------------
* Every op creates a new Tensor that remembers its parents and a closure
  (`_backward`) that pushes `out.grad` into the parents' `.grad`.
* `backward()` does an iterative DFS topological sort (no recursion limit),
  then walks the nodes in reverse order.
* Gradients ACCUMULATE (`_acc`), which is required whenever a tensor is used
  more than once (e.g. `x * x`).
* Broadcasting: every binary op un-broadcasts the incoming gradient back to the
  parent's shape with `_unbroadcast` (sum over the broadcast axes).
* `masked(mask)` is the pruning primitive; see its docstring for the exact
  gradient convention.
"""
from __future__ import annotations

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` over the axes that NumPy broadcast so it has `shape`."""
    while grad.ndim > len(shape):          # leading axes that were prepended
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):    # axes that were stretched from 1
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _acc(t: "Tensor", g: np.ndarray) -> None:
    if not t.requires_grad:
        return
    t.grad = g if t.grad is None else t.grad + g


def _as_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    def __init__(self, data, requires_grad: bool = False, _parents=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        # Pruning support (only meaningful for parameters) ------------------
        self.mask: np.ndarray | None = None        # 1.0 = alive, 0.0 = pruned
        self.dense_grad: np.ndarray | None = None  # dL/dW_eff incl. pruned slots
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    # ------------------------------------------------------------------ misc
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = None
        self.dense_grad = None

    def _new(self, data, parents, op):
        rg = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=rg, _parents=parents if rg else (), _op=op)

    # ------------------------------------------------------------ elementwise
    def __add__(self, other):
        other = _as_tensor(other)
        out = self._new(self.data + other.data, (self, other), "add")

        def _backward():
            _acc(self, _unbroadcast(out.grad, self.shape))
            _acc(other, _unbroadcast(out.grad, other.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = self._new(self.data * other.data, (self, other), "mul")

        def _backward():
            _acc(self, _unbroadcast(out.grad * other.data, self.shape))
            _acc(other, _unbroadcast(out.grad * self.data, other.shape))
        out._backward = _backward
        return out

    def __neg__(self):
        out = self._new(-self.data, (self,), "neg")

        def _backward():
            _acc(self, -out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = self._new(self.data / other.data, (self, other), "div")

        def _backward():
            _acc(self, _unbroadcast(out.grad / other.data, self.shape))
            _acc(other, _unbroadcast(-out.grad * self.data / other.data ** 2, other.shape))
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    def __pow__(self, p: float):
        assert isinstance(p, (int, float)), "only constant exponents supported"
        out = self._new(self.data ** p, (self,), f"pow{p}")

        def _backward():
            _acc(self, out.grad * p * self.data ** (p - 1))
        out._backward = _backward
        return out

    def exp(self):
        out = self._new(np.exp(self.data), (self,), "exp")

        def _backward():
            _acc(self, out.grad * out.data)
        out._backward = _backward
        return out

    def log(self):
        out = self._new(np.log(self.data), (self,), "log")

        def _backward():
            _acc(self, out.grad / self.data)
        out._backward = _backward
        return out

    # ----------------------------------------------------------------- matmul
    def __matmul__(self, other):
        other = _as_tensor(other)
        assert self.data.ndim == 2 and other.data.ndim == 2, "matmul supports 2-D only"
        out = self._new(self.data @ other.data, (self, other), "matmul")

        def _backward():
            _acc(self, out.grad @ other.data.T)
            _acc(other, self.data.T @ out.grad)
        out._backward = _backward
        return out

    # ------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        out = self._new(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _acc(self, np.broadcast_to(g, self.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---------------------------------------------------------- non-linearity
    def relu(self):
        out = self._new(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            _acc(self, out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def tanh(self):
        out = self._new(np.tanh(self.data), (self,), "tanh")

        def _backward():
            _acc(self, out.grad * (1.0 - out.data ** 2))
        out._backward = _backward
        return out

    def sigmoid(self):
        # numerically stable: never exponentiate a large positive number
        e = np.exp(-np.abs(self.data))
        s = np.where(self.data >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        out = self._new(s, (self,), "sigmoid")

        def _backward():
            _acc(self, out.grad * out.data * (1.0 - out.data))
        out._backward = _backward
        return out

    def gelu(self):
        """tanh approximation of GELU."""
        x = self.data
        t = np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3))
        out = self._new(0.5 * x * (1.0 + t), (self,), "gelu")

        def _backward():
            du = _SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * x ** 2)
            d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du
            _acc(self, out.grad * d)
        out._backward = _backward
        return out

    # ---------------------------------------------------------------- masking
    def masked(self, mask: np.ndarray):
        """Return W_eff = W * mask (a *real* zero in the forward pass).

        Gradient convention (see DESIGN.md Q2):
          * `self.grad`       += g * mask  -> pruned slots get EXACTLY 0, so the
            optimizer receives no signal for them.
          * `self.dense_grad` += g         -> dL/dW_eff at every slot, including
            pruned ones: what a pruned weight *would* feel if revived at 0.
            This is the right signal for regrowth.
        Both are exact derivatives; they answer different questions.
        """
        mask = np.asarray(mask, dtype=np.float64)
        out = self._new(self.data * mask, (self,), "masked")

        def _backward():
            _acc(self, out.grad * mask)
            if self.requires_grad:
                self.dense_grad = out.grad if self.dense_grad is None else self.dense_grad + out.grad
        out._backward = _backward
        return out

    # --------------------------------------------------------------- backward
    def backward(self, grad: np.ndarray | None = None):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:                         # iterative post-order DFS
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        for node in reversed(topo):
            node._backward()


# ------------------------------------------------------------------- losses
def softmax(logits: Tensor) -> np.ndarray:
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Fused, numerically stable softmax + mean cross-entropy.

    loss = mean_i [ logsumexp(z_i) - z_i[y_i] ], max subtracted first.
    dL/dz = (softmax(z) - onehot(y)) / N
    """
    y = np.asarray(y, dtype=np.int64)
    n = logits.shape[0]
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1))[:, None]
    loss = -logp[np.arange(n), y].mean()
    out = logits._new(loss, (logits,), "softmax_ce")

    def _backward():
        p = np.exp(logp)
        p[np.arange(n), y] -= 1.0
        _acc(logits, out.grad * p / n)
    out._backward = _backward
    return out
