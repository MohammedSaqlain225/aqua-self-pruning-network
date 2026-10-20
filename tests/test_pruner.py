import numpy as np
from nn.layers import make_mlp
from nn.optim import Adam
from engine import Tensor, softmax_cross_entropy
from prune.pruner import SelfPruner, cubic_sparsity


def test_cubic_schedule_endpoints_and_monotone():
    vals = [cubic_sparsity(t, 100, 500, 0.9) for t in range(0, 700, 10)]
    assert vals[0] == 0.0 and abs(vals[-1] - 0.9) < 1e-12
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    # cubic ramp prunes fastest early (concave): first half reaches > 50% of target
    assert cubic_sparsity(300, 100, 500, 0.9) > 0.45


def _setup(regrow=0.0, crit="fisher", seed=0):
    rng = np.random.default_rng(seed)
    net = make_mlp([10, 32, 16, 4], rng)
    X, y = rng.normal(size=(64, 10)), rng.integers(0, 4, 64)
    ws = net.prunable_weights()
    p = SelfPruner(ws, 0.8, crit, t_start=0, t_end=100, every=1, regrow_frac=regrow)
    opt = Adam(net.parameters(), lr=1e-2)
    return net, X, y, p, opt


def _step(net, X, y, p, opt, t):
    opt.zero_grad()
    softmax_cross_entropy(net(Tensor(X)), y).backward()
    p.observe(); opt.step(); p.maybe_update(t)


def test_pruner_hits_target_exactly_and_masks_are_real_zeros():
    for crit in ("magnitude", "taylor", "fisher"):
        net, X, y, p, opt = _setup(crit=crit)
        for t in range(120):
            _step(net, X, y, p, opt, t)
        N = sum(w.data.size for w in p.ws)
        alive = sum(int(w.mask.sum()) for w in p.ws)
        assert alive == int(round(0.2 * N)), (crit, alive)
        for w in p.ws:
            assert np.all(w.data[w.mask == 0] == 0.0)


def test_sparsity_monotone_without_regrowth_and_no_revival():
    net, X, y, p, opt = _setup()
    prev = [w.mask.copy() for w in p.ws]
    for t in range(120):
        _step(net, X, y, p, opt, t)
        for w, pm in zip(p.ws, prev):
            assert np.all(w.mask <= pm), "a pruned slot came back without regrowth enabled"
        prev = [w.mask.copy() for w in p.ws]
    assert p.churn == 0


def test_regrowth_revives_with_zero_weight_and_keeps_budget():
    net, X, y, p, opt = _setup(regrow=0.3)
    for t in range(120):
        _step(net, X, y, p, opt, t)
    assert p.churn > 0
    N = sum(w.data.size for w in p.ws)
    assert sum(int(w.mask.sum()) for w in p.ws) == int(round(0.2 * N))


def test_every_layer_keeps_a_floor_of_connections():
    net, X, y, p, opt = _setup()
    for t in range(120):
        _step(net, X, y, p, opt, t)
    assert all(w.mask.sum() >= 1 for w in p.ws)
