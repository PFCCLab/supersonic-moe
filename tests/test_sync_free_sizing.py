"""Shape-bound ("sync-free") routing metadata sizing against exact sizing.

The ring dispatcher sizes the expert op's metadata from a shape bound
(``sync_free_sizing``) instead of host-side per-expert counts.  With dropped
lanes (expert id -1) the bound exceeds the real lane count, which must not
change any result: output and every gradient, including the router-score
gradient of the dropped lanes (zero), are compared bitwise against exact
sizing, and two runs must agree.  An int ``sync_free_sizing`` tightens the
bound to a caller-known lane count and must give the same results."""

import numpy as np
import paddle
import pytest

paddle.enable_compat(silent=True)

from sonicmoe import run_sonic_moe  # noqa: E402


def _routing(rng, T, K, E, drop):
    """[T, K] unique expert ids per row, each lane dropped (-1) w.p. ``drop``."""
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    ids[rng.rand(T, K) < drop] = -1
    ids[0, 0] = 0  # slot 0 is where a stray scatter of the padded tail lands
    return ids.astype(np.int64)


def _poison_allocator():
    """Leave NaN in freed allocator blocks, so a gradient slot the backward
    never writes shows up instead of reading as a lucky zero."""
    junk = [paddle.full([n], float("nan"), dtype=paddle.float32) for n in (1 << 12, 1 << 16, 1 << 20, 1 << 22)]
    paddle.device.synchronize()
    del junk


def _run(ids, sizing, T, H, I, E, K):
    _poison_allocator()
    paddle.seed(0)
    x = (paddle.randn([T, H]) * 0.5).astype(paddle.bfloat16)
    w1 = (paddle.randn([E, 2 * I, H]) * 0.05).astype(paddle.bfloat16)
    w2 = (paddle.randn([E, H, I]) * 0.05).astype(paddle.bfloat16)
    rng = np.random.RandomState(3)
    sc = np.where(ids >= 0, rng.dirichlet(np.ones(K), size=T), 0)
    sc = paddle.to_tensor(sc.astype(np.float32))
    for t in (x, w1, w2, sc):
        t.stop_gradient = False
    _poison_allocator()
    kwargs = {} if sizing is False else {"sync_free_sizing": sizing}
    o = run_sonic_moe(x, paddle.to_tensor(ids), sc, K, E, w1, w2, **kwargs)
    (o.astype(paddle.float32) * 0.01).sum().backward()
    paddle.device.synchronize()
    g = lambda t: t.astype(paddle.float32).numpy()  # noqa: E731
    return {"o": g(o), "x": g(x.grad), "w1": g(w1.grad), "w2": g(w2.grad), "sc": g(sc.grad)}


def _assert_same(a, b):
    for n in a:
        assert np.array_equal(a[n], b[n]), (n, np.abs(a[n] - b[n]).max())


@pytest.mark.parametrize("drop", [0.0, 0.3, 0.9])
def test_sync_free_sizing_matches_exact_sizing(drop):
    T, H, I, E, K = 1024, 256, 128, 16, 4
    ids = _routing(np.random.RandomState(1), T, K, E, drop)
    exact = _run(ids, False, T, H, I, E, K)
    for _ in range(2):
        _assert_same(exact, _run(ids, True, T, H, I, E, K))
    if drop:
        # dropped lanes carry no route, so their score gradient is zero
        assert not exact["sc"][ids < 0].any()


@pytest.mark.parametrize("slack", [0, 37])
def test_sync_free_sizing_lane_bound(slack):
    T, H, I, E, K = 1024, 256, 128, 16, 4
    ids = _routing(np.random.RandomState(2), T, K, E, 0.6)
    bound = int((ids >= 0).sum()) + slack
    _assert_same(_run(ids, False, T, H, I, E, K), _run(ids, bound, T, H, I, E, K))
