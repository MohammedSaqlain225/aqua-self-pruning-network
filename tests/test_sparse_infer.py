import numpy as np
from nn.layers import make_mlp
from prune.sparse_infer import extract, dense_forward, SparseNet, compact, count_macs


def _pruned_net(seed=0, keep=0.1):
    rng = np.random.default_rng(seed)
    net = make_mlp([12, 20, 16, 5], rng)
    for l in net.linears():
        l.weight.mask = (rng.random(l.weight.shape) < keep).astype(float)
        l.bias.data = rng.normal(size=l.bias.shape)
    return net


def test_csc_forward_equals_dense_masked_forward():
    net = _pruned_net(); X = np.random.default_rng(1).normal(size=(7, 12))
    layers = extract(net)
    assert np.allclose(SparseNet(layers)(X), dense_forward(layers, X), atol=1e-12)


def test_csc_handles_empty_columns_and_empty_layer():
    net = _pruned_net(keep=0.02)
    X = np.random.default_rng(1).normal(size=(3, 12))
    layers = extract(net)
    assert np.allclose(SparseNet(layers)(X), dense_forward(layers, X), atol=1e-12)


def test_compaction_is_exact_and_smaller():
    net = _pruned_net(keep=0.08); X = np.random.default_rng(2).normal(size=(9, 12))
    layers = extract(net); comp = compact(layers)
    assert np.allclose(dense_forward(comp, X), dense_forward(layers, X), atol=1e-12)
    assert count_macs(comp) < count_macs(layers)
