"""Part 1 correctness requirement: masked weights, their gradients, and optimizer state.

Each test states the invariant it protects.  `NaiveAdam` below is the
implementation most people write (mask only the gradient, global step t, no
state reset); the tests show it violates the invariants and ours does not.
"""
import numpy as np
from engine import Tensor, softmax_cross_entropy
from nn.layers import Linear, make_mlp
from nn.optim import Adam, SGD
from .gradcheck import numerical_grad

R = np.random.default_rng(0)


class NaiveAdam:
    """Mask only the gradient; keep moments; single global t. (The common bug.)"""
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad * (p.mask if p.mask is not None else 1.0)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _layer_and_data(n_in=6, n_out=4, n=8):
    lin = Linear(n_in, n_out, np.random.default_rng(3))
    x = Tensor(R.normal(size=(n, n_in)))
    y = R.integers(0, n_out, n)
    return lin, x, y


def _loss_grad(lin, x, y):
    lin.weight.zero_grad(); lin.bias.zero_grad()
    loss = softmax_cross_entropy(lin(x), y)
    loss.backward()
    return float(loss.data)


def _random_mask(shape, keep=0.5, seed=0):
    return (np.random.default_rng(seed).random(shape) < keep).astype(np.float64)


# --------------------------------------------------------------- forward / grads
def test_masked_weight_contributes_exactly_zero_to_forward():
    lin, x, y = _layer_and_data()
    lin.weight.mask = _random_mask(lin.weight.shape)
    ref = lin(x).data.copy()
    lin.weight.data[lin.weight.mask == 0] = 1e6        # poison the pruned raw slots
    assert np.array_equal(lin(x).data, ref), "pruned weight leaked into forward pass"


def test_grad_wrt_raw_weight_is_zero_at_masked_slots_and_matches_finite_diff():
    lin, x, y = _layer_and_data()
    lin.weight.mask = _random_mask(lin.weight.shape)
    _loss_grad(lin, x, y)
    g = lin.weight.grad
    assert np.all(g[lin.weight.mask == 0] == 0.0)
    W0 = lin.weight.data.copy()

    def f(W):
        lin.weight.data = W
        return float(softmax_cross_entropy(lin(x), y).data)
    num = numerical_grad(f, [W0.copy()])[0]
    lin.weight.data = W0
    assert np.max(np.abs(g - num)) < 1e-7


def test_dense_grad_is_exact_derivative_wrt_effective_weight_incl_pruned_slots():
    """dense_grad = dL/dW_eff; at pruned slots it is the signal 'if I were revived'."""
    lin, x, y = _layer_and_data()
    lin.weight.mask = _random_mask(lin.weight.shape)
    _loss_grad(lin, x, y)
    dg = lin.weight.dense_grad
    assert np.any(dg[lin.weight.mask == 0] != 0.0)     # informative where pruned

    def f(Weff):                                        # loss as a function of W_eff directly
        return float(softmax_cross_entropy(x @ Tensor(Weff) + lin.bias, y).data)
    num = numerical_grad(f, [(lin.weight.data * lin.weight.mask).copy()])[0]
    assert np.max(np.abs(dg - num)) < 1e-7


def test_masked_net_gradient_equals_explicitly_pruned_net():
    """Alive-slot gradients equal those of a net built with the zeros hard-coded."""
    lin, x, y = _layer_and_data()
    lin.weight.mask = _random_mask(lin.weight.shape)
    _loss_grad(lin, x, y)
    W_hard = Tensor(lin.weight.data * lin.weight.mask, requires_grad=True)   # plain tensor, no mask op
    softmax_cross_entropy(x @ W_hard + Tensor(lin.bias.data), y).backward()
    assert np.allclose(W_hard.grad * lin.weight.mask, lin.weight.grad)


# ------------------------------------------------------------------ optimizer state
def _train_steps(opt, lin, x, y, k):
    for _ in range(k):
        opt.zero_grad() if hasattr(opt, "zero_grad") else lin.weight.zero_grad()
        lin.weight.zero_grad(); lin.bias.zero_grad()
        _loss_grad(lin, x, y)
        opt.step()


def test_pruned_weight_stays_exactly_zero_despite_stale_momentum():
    for make in (lambda p: Adam(p, lr=1e-2), lambda p: SGD(p, lr=1e-2, momentum=0.9)):
        lin, x, y = _layer_and_data()
        opt = make(lin.parameters())
        _train_steps(opt, lin, x, y, 20)                       # build up momentum everywhere
        lin.weight.mask = _random_mask(lin.weight.shape, seed=5)   # prune mid-training
        dead = lin.weight.mask == 0
        for _ in range(30):
            _train_steps(opt, lin, x, y, 1)
            assert np.all(lin.weight.data[dead] == 0.0), "ghost momentum moved a pruned weight"


def test_naive_adam_violates_the_invariant():
    """Negative control: proves the test above can actually detect the bug."""
    lin, x, y = _layer_and_data()
    opt = NaiveAdam(lin.parameters(), lr=1e-2)
    _train_steps(opt, lin, x, y, 20)
    lin.weight.mask = _random_mask(lin.weight.shape, seed=5)
    dead = lin.weight.mask == 0
    _train_steps(opt, lin, x, y, 5)
    assert np.any(lin.weight.data[dead] != 0.0)               # raw weight drifts while "pruned"


def test_revived_connection_starts_with_clean_adam_state():
    lin, x, y = _layer_and_data()
    lr = 1e-2
    opt = Adam(lin.parameters(), lr=lr)
    _train_steps(opt, lin, x, y, 50)
    mask = _random_mask(lin.weight.shape, seed=7); lin.weight.mask = mask.copy()
    _train_steps(opt, lin, x, y, 10)
    st = opt.state[id(lin.weight)]
    dead = np.argwhere(mask == 0)[0]; i, j = dead
    assert st["m"][i, j] == 0 and st["v"][i, j] == 0 and st["t"][i, j] == 0
    # revive that slot
    mask2 = mask.copy(); mask2[i, j] = 1.0; lin.weight.mask = mask2
    _loss_grad(lin, x, y)
    g = lin.weight.grad[i, j]
    assert g != 0.0
    opt.step()
    # fresh slot, first Adam step: m_hat/sqrt(v_hat) = g/|g|  =>  |delta| = lr (bias-corrected)
    assert np.isclose(abs(lin.weight.data[i, j]), lr, rtol=1e-6)
    assert np.sign(lin.weight.data[i, j]) == -np.sign(g)
    assert st["t"][i, j] == 1


def test_naive_adam_revival_step_is_wrong():
    """Negative control: global t + stale/never-reset state => first step != lr."""
    lin, x, y = _layer_and_data()
    lr = 1e-2
    opt = NaiveAdam(lin.parameters(), lr=lr)
    _train_steps(opt, lin, x, y, 50)
    mask = _random_mask(lin.weight.shape, seed=7); lin.weight.mask = mask.copy()
    _train_steps(opt, lin, x, y, 10)
    i, j = np.argwhere(mask == 0)[0]
    mask2 = mask.copy(); mask2[i, j] = 1.0; lin.weight.mask = mask2
    before = lin.weight.data[i, j]
    _loss_grad(lin, x, y)
    opt.step()
    assert not np.isclose(abs(lin.weight.data[i, j] - before), lr, rtol=0.05)


def test_unmasked_slots_unaffected_by_mask_change_elsewhere():
    """Resetting state for changed slots must not touch the moments of unchanged slots."""
    lin, x, y = _layer_and_data()
    opt = Adam(lin.parameters(), lr=1e-2)
    _train_steps(opt, lin, x, y, 15)
    st = opt.state[id(lin.weight)]
    m_before = st["m"].copy()
    mask = np.ones(lin.weight.shape); mask[0, 0] = 0.0
    lin.weight.mask = mask
    opt._sync_mask(lin.weight, st)
    assert st["m"][0, 0] == 0.0 and np.array_equal(st["m"].ravel()[1:], m_before.ravel()[1:])


def test_unmasked_adam_matches_reference_formula():
    """Sanity: with no mask our Adam equals textbook Adam, step by step."""
    lin, x, y = _layer_and_data()
    lin.weight.mask = None
    ref_w = lin.weight.data.copy(); m = np.zeros_like(ref_w); v = np.zeros_like(ref_w)
    opt = Adam(lin.parameters(), lr=1e-2)
    for t in range(1, 6):
        _loss_grad(lin, x, y)
        g = lin.weight.grad.copy()
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        ref_w = ref_w - 1e-2 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        opt.step()
        assert np.allclose(lin.weight.data, ref_w, atol=1e-12)
