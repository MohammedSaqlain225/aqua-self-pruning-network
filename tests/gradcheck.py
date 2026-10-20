import numpy as np
from engine import Tensor


def numerical_grad(f, arrays, eps=1e-6):
    """Central finite differences of scalar f(*arrays) wrt every array."""
    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]
            a[i] = old + eps; fp = f(*arrays)
            a[i] = old - eps; fm = f(*arrays)
            a[i] = old
            g[i] = (fp - fm) / (2 * eps)
        grads.append(g)
    return grads


def check(fn, *arrays, tol=1e-6):
    """fn maps Tensors -> scalar Tensor. Compares autodiff vs finite differences."""
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    fn(*ts).backward()
    num = numerical_grad(lambda *xs: float(fn(*[Tensor(x) for x in xs]).data), [a.copy() for a in arrays])
    for t, n in zip(ts, num):
        assert t.grad is not None and t.grad.shape == t.shape, "missing/mis-shaped grad"
        err = np.max(np.abs(t.grad - n) / (1.0 + np.abs(n)))
        assert err < tol, f"gradcheck failed: rel err {err:.2e}"
