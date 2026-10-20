"""Grouped down projection (sonicmoe.functional.grouped_down_proj) against the
per-expert GEMM2 + combine.  Exactness is checked with small-integer data and
power-of-two weights (every product and partial sum exact in bf16/fp32): any
wrong row, lane, weight or expert segment shows up as a mismatch.  The
end-to-end check runs the MoE forward/backward with a plan installed."""

import itertools
from collections import Counter

import numpy as np
import paddle
import pytest

paddle.enable_compat(silent=True)

from quack.gemm_interface import gemm  # noqa: E402

from sonicmoe import row_group_planner as P  # noqa: E402
from sonicmoe import GroupList, Plan, grouped_down_proj, grouped_down_proj_plan, run_sonic_moe  # noqa: E402
from sonicmoe.functional.reduction_over_k_gather import (  # noqa: E402
    token_gather_and_sum_varlen_K_triton as gather,
)


def dev(a, dtype):
    return paddle.to_tensor(np.ascontiguousarray(a)).astype(dtype)


def frequent_groups(ids, sizes, n):
    out = []
    for c in sizes:
        cnt = Counter()
        for r in ids[:1024]:
            v = sorted(int(e) for e in r if e >= 0)
            cnt.update(itertools.combinations(v, c))
        out += [list(k) for k, _ in cnt.most_common(n // len(sizes))]
    np.random.RandomState(0).shuffle(out)
    return out


def lanes_of(ids, E):
    """Lanes token-major (dropped lanes removed), y1 rows expert-major and
    stable by lane; returns the per-expert row offsets and per-token offsets."""
    T, K = ids.shape
    tt, kk = np.nonzero(ids >= 0)
    lane_e = ids[tt, kk]
    off = np.zeros(T + 1, np.int32)
    off[1:] = np.cumsum((ids >= 0).sum(1))
    order = np.argsort(lane_e, kind="stable")
    row = np.empty(lane_e.size, np.int64)
    row[order] = np.arange(lane_e.size)
    efo = np.zeros(E + 1, np.int32)
    efo[1:] = np.cumsum(np.bincount(lane_e, minlength=E))
    return tt, kk, row, efo, off


def run_both(ids, w, y1_rows, w2, groups, err=None):
    T, K = ids.shape
    E, H, I = w2.shape
    tt, kk, row, efo, off = lanes_of(ids, E)
    y1 = np.zeros((row.size, I), np.float32)
    y1[row] = y1_rows[tt * K + kk]
    d_y1 = dev(y1, paddle.float32).astype(paddle.bfloat16)
    d_w2 = dev(w2, paddle.float32).astype(paddle.bfloat16)
    d_efo = dev(efo, paddle.int32)
    d_rows = dev(row, paddle.int32)
    d_w = dev(w[tt, kk], paddle.float32)
    d_off = dev(off, paddle.int32)
    # per-expert path
    y2p = paddle.empty([row.size, H], dtype=paddle.bfloat16)
    gemm(d_y1, d_w2.permute(0, 2, 1), out=y2p, cu_seqlens_m=d_efo, tuned=False)
    o_prod = paddle.empty([T, H], dtype=paddle.bfloat16)
    gather(y2p, d_w, o_prod, d_rows, d_off, T, K, H, True)
    # grouped path
    index = P.build_index(groups, E)
    out = P.alloc_out(T, K, index)
    nrow = P.plan(ids.astype(np.int16), index, P.Scratch(T, K, index, 8), out)
    err = err if err is not None else paddle.zeros([1], dtype=paddle.int32)
    o_g = grouped_down_proj(
        d_y1, d_w2, d_efo, d_rows, d_w, d_off, dev(ids, paddle.int32),
        Plan.from_host(out, nrow), GroupList(index), err,
    )
    paddle.device.synchronize()
    f = lambda x: x.astype(paddle.float32).numpy()  # noqa: E731
    return f(o_prod), f(o_g), int(err.numpy()[0]), nrow


CASES = {  # E, K, T, H, sizes, n groups, drop fraction
    "pairs": (64, 8, 2048, 256, (2,), 300, 0.0),
    "mixed": (64, 8, 2048, 256, (2, 3, 4), 400, 0.0),
    "mixed_k10": (48, 10, 2048, 256, (2, 3, 4), 400, 0.0),
    "dropped": (64, 8, 2048, 256, (2, 3, 4), 400, 0.2),
    "empty_list": (64, 8, 1024, 256, (2,), 0, 0.1),
}


def _routing(rng, E, K, T, drop):
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    ids[rng.rand(T, K) < drop] = -1
    return ids


@pytest.mark.parametrize("case", list(CASES))
def test_exact_on_integer_data(case):
    E, K, T, H, sizes, n, drop = CASES[case]
    rng = np.random.RandomState(7)
    ids = _routing(rng, E, K, T, drop)
    w = (2.0 ** -rng.randint(0, 2, (T, K))).astype(np.float32)
    y1 = (rng.randint(-1, 2, (T * K, 128)) * (rng.rand(T * K, 128) < 0.08)).astype(np.float32)
    w2 = rng.randint(-1, 2, (E, H, 128)).astype(np.float32)
    groups = frequent_groups(ids, sizes, n) if n else []
    o_prod, o_g, err, nrow = run_both(ids, w, y1, w2, groups)
    assert err == 0
    if n:
        assert nrow < int((ids >= 0).sum())
    assert np.array_equal(o_g, o_prod)


@pytest.mark.parametrize("I", [16, 32, 48, 96])
@pytest.mark.parametrize("case", ["pairs", "mixed"])
def test_exact_short_expert_k(case, I):
    """Per-expert K below 64 or not a power of 2 (e.g. I-sharded experts at
    high EP): the segmented GEMM uses a shorter k-tile, the pack kernel masks."""
    E, K, T, H, sizes, n, drop = CASES[case]
    rng = np.random.RandomState(I)
    ids = _routing(rng, E, K, T, drop)
    w = (2.0 ** -rng.randint(0, 2, (T, K))).astype(np.float32)
    y1 = (rng.randint(-1, 2, (T * K, I)) * (rng.rand(T * K, I) < 0.2)).astype(np.float32)
    w2 = rng.randint(-1, 2, (E, H, I)).astype(np.float32)
    o_prod, o_g, err, nrow = run_both(ids, w, y1, w2, frequent_groups(ids, sizes, n))
    assert err == 0 and nrow < int((ids >= 0).sum())
    assert np.array_equal(o_g, o_prod)


def test_mismatched_routing_sets_error():
    E, K, T, H = 64, 8, 512, 256
    rng = np.random.RandomState(3)
    ids = _routing(rng, E, K, T, 0.0)
    w = rng.rand(T, K).astype(np.float32)
    y1 = rng.randn(T * K, 128).astype(np.float32)
    w2 = (rng.randn(E, H, 128) * 0.05).astype(np.float32)
    tt, kk, row, efo, off = lanes_of(ids, E)
    bad = ids.copy()  # the plan's routing drops a lane the forward still has
    bad[5, 0] = -1
    index = P.build_index(frequent_groups(ids, (2,), 100), E)
    out = P.alloc_out(T, K, index)
    nrow = P.plan(bad.astype(np.int16), index, P.Scratch(T, K, index, 4), out)
    y1d = np.zeros((row.size, 128), np.float32)
    y1d[row] = y1[tt * K + kk]
    err = paddle.zeros([1], dtype=paddle.int32)
    grouped_down_proj(
        dev(y1d, paddle.float32).astype(paddle.bfloat16), dev(w2, paddle.float32).astype(paddle.bfloat16),
        dev(efo, paddle.int32), dev(row, paddle.int32), dev(w[tt, kk], paddle.float32), dev(off, paddle.int32),
        dev(bad, paddle.int32), Plan.from_host(out, nrow), GroupList(index), err,
    )
    paddle.device.synchronize()
    assert int(err.numpy()[0]) == 1


def test_moe_forward_backward_matches_default():
    """End to end: output within bf16 error of the default path, and every
    gradient bitwise identical (the backward recomputes y2 and reads neither
    y2 nor o)."""
    T, H, I, E, K = 2048, 512, 128, 32, 8
    rng = np.random.RandomState(9)
    ids = _routing(rng, E, K, T, 0.05)
    index = P.build_index(frequent_groups(ids, (2, 3), 200), E)
    out = P.alloc_out(T, K, index)
    nrow = P.plan(ids.astype(np.int16), index, P.Scratch(T, K, index, 8), out)
    plan, gl = Plan.from_host(out, nrow), GroupList(index)
    d_ids = dev(ids, paddle.int32)
    err = paddle.zeros([1], dtype=paddle.int32)

    def run(grouped):
        paddle.seed(0)
        x = (paddle.randn([T, H]) * 0.5).astype(paddle.bfloat16)
        w1 = (paddle.randn([E, 2 * I, H]) * 0.05).astype(paddle.bfloat16)
        w2 = (paddle.randn([E, H, I]) * 0.05).astype(paddle.bfloat16)
        sc = paddle.to_tensor(np.where(ids >= 0, rng.dirichlet(np.ones(K), size=T), 0).astype(np.float32))
        for t in (x, w1, w2, sc):
            t.stop_gradient = False
        ctx = grouped_down_proj_plan(gl, plan, d_ids, err) if grouped else _nullctx()
        with ctx:
            o = run_sonic_moe(x, paddle.to_tensor(ids.astype(np.int64)), sc, K, E, w1, w2)
        (o.astype(paddle.float32) * 0.01).sum().backward()
        paddle.device.synchronize()
        g = lambda t: t.astype(paddle.float32).numpy()  # noqa: E731
        return g(o), {n: g(t.grad) for n, t in (("x", x), ("w1", w1), ("w2", w2), ("sc", sc))}

    rng = np.random.RandomState(1)
    o0, g0 = run(False)
    rng = np.random.RandomState(1)
    o1, g1 = run(True)
    assert int(err.numpy()[0]) == 0
    rel = np.linalg.norm(o1 - o0) / np.linalg.norm(o0)
    assert np.isfinite(o1).all() and rel < 1e-2, rel
    for n in g0:
        assert np.array_equal(g0[n], g1[n]), (n, np.abs(g0[n] - g1[n]).max())


import contextlib  # noqa: E402


def _nullctx():
    return contextlib.nullcontext()
