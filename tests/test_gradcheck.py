"""Part 1: finite-difference gradient checks for every implemented op."""
import numpy as np
from engine import Tensor, softmax_cross_entropy
from .gradcheck import check

R = np.random.default_rng(0)
A, B = R.normal(size=(4, 3)), R.normal(size=(4, 3))
POS = R.uniform(0.5, 2.0, size=(4, 3))
# keep ReLU inputs away from the kink at 0 where finite differences are invalid
RELU_IN = R.normal(size=(4, 3)); RELU_IN += np.sign(RELU_IN) * 0.2


def test_add():   check(lambda a, b: (a + b).sum(), A, B)
def test_sub():   check(lambda a, b: ((a - b) * (a - b)).sum(), A, B)
def test_mul():   check(lambda a, b: (a * b).sum(), A, B)
def test_div():   check(lambda a, b: (a / b).sum(), A, POS)
def test_neg():   check(lambda a: (-a * a).sum(), A)
def test_scalar_ops(): check(lambda a: ((2.0 - a) * 3.0 + 1.0 / (a * a + 1.0)).sum(), A)
def test_pow():   check(lambda a: (a ** 3).sum(), A)
def test_exp_log(): check(lambda a: (a.exp().log() + a.exp()).sum(), A)
def test_reuse_accumulates(): check(lambda a: (a * a * a).sum(), A)          # a used 3x
def test_matmul(): check(lambda a, b: (a @ b).sum() , A, R.normal(size=(3, 5)))
def test_matmul_weighted(): check(lambda a, b: ((a @ b) * (a @ b)).mean(), A, R.normal(size=(3, 5)))
def test_sum_axis(): check(lambda a: (a.sum(axis=0) * Tensor([1., 2., 3.])).sum(), A)
def test_sum_axis1_keepdims(): check(lambda a: (a.sum(axis=1, keepdims=True) * a).sum(), A)
def test_mean(): check(lambda a: (a.mean() * a.mean()), A)
def test_mean_axis(): check(lambda a: (a.mean(axis=1) ** 2).sum(), A)
def test_relu(): check(lambda a: (a.relu() * a.relu()).sum(), RELU_IN)
def test_tanh(): check(lambda a: a.tanh().sum(), A)
def test_sigmoid(): check(lambda a: a.sigmoid().sum(), A)
def test_sigmoid_stable():
    out = Tensor([-800.0, 0.0, 800.0]).sigmoid().data
    assert np.all(np.isfinite(out)) and out[0] == 0.0 and out[2] == 1.0
def test_gelu(): check(lambda a: a.gelu().sum(), A)


def test_broadcast_row_vector():
    """(N, D) + (D,) must backpropagate: grad of the (D,) operand sums over N."""
    check(lambda a, b: ((a + b) * (a + b)).sum(), A, R.normal(size=(3,)))
    a, b = Tensor(A, requires_grad=True), Tensor(np.zeros(3), requires_grad=True)
    (a + b).sum().backward()
    assert np.allclose(b.grad, 4.0) and b.grad.shape == (3,)


def test_broadcast_column_and_scalar():
    check(lambda a, c: (a * c).sum(), A, R.normal(size=(4, 1)))
    check(lambda a, c: (a / (c * c + 1.0)).sum(), A, np.array(0.7))
    check(lambda a, b: (a * b).sum(), A, R.normal(size=(1, 3)))


def test_softmax_ce():
    y = np.array([0, 2, 1, 2])
    check(lambda z: softmax_cross_entropy(z, y), R.normal(size=(4, 3)))


def test_softmax_ce_numerically_stable():
    z = Tensor([[1000.0, -1000.0, 0.0], [-1e4, 1e4, 3.0]], requires_grad=True)
    loss = softmax_cross_entropy(z, [0, 1])
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(z.grad)) and loss.data < 1e-6


def test_deep_graph_no_recursion_error():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert np.isclose(x.grad[0], 1.0001 ** 5000)


def test_mlp_end_to_end():
    from nn.layers import make_mlp
    rng = np.random.default_rng(1)
    net = make_mlp([3, 5, 4], rng, activation="gelu")
    x, y = rng.normal(size=(6, 3)), rng.integers(0, 4, 6)
    params = net.parameters()
    arrays = [p.data.copy() for p in params]

    def f(*ps):
        for p, a in zip(params, ps):
            p.data = a
        return float(softmax_cross_entropy(net(Tensor(x)), y).data)

    from .gradcheck import numerical_grad
    for p in params: p.zero_grad()
    for p, a in zip(params, arrays): p.data = a.copy()
    softmax_cross_entropy(net(Tensor(x)), y).backward()
    ana = [p.grad.copy() for p in params]
    num = numerical_grad(f, [a.copy() for a in arrays])
    for a, n in zip(ana, num):
        assert np.max(np.abs(a - n) / (1 + np.abs(n))) < 1e-6
