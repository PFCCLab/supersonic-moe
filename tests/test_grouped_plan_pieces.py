"""Per-piece plan (grouped_plan_pieces) vs plan_gpu on the concatenated routing:
`cu` / `moff` must be identical and the merged down-projection output bitwise
equal (row order inside a segment may differ; the combine is order-free)."""

import numpy as np
import paddle
import pytest


paddle.enable_compat(silent=True)

from sonicmoe import GroupList  # noqa: E402
from sonicmoe import row_group_planner as P  # noqa: E402
from sonicmoe.functional.grouped_plan_gpu import GPUIndex, plan_gpu  # noqa: E402
from sonicmoe.functional.grouped_plan_pieces import PlanPieces, ncols  # noqa: E402

from .test_grouped_plan_gpu import dev, frequent_pairs, merged_o  # noqa: E402


# E, K, T, pieces, pairs in the list, dropped-lane rate
CASES = {
    "pairs": (64, 8, 2048, 4, 300, 0.0),
    "dropped": (64, 8, 2048, 4, 300, 0.2),
    "k10": (48, 10, 2048, 8, 200, 0.05),
    "empty": (64, 8, 1024, 2, 0, 0.1),
}


def _routing(E, K, T, drop, seed=7):
    rng = np.random.RandomState(seed)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    ids[rng.rand(T, K) < drop] = -1
    return rng, ids


def _pieces_src(pp, ids_d, w_d, x_first):
    """Per piece: local_select, then the batch as the ring would gather it."""
    TL = pp.TL
    rows = []
    for j in range(pp.P):
        sl = slice(j * TL, (j + 1) * TL)
        x = pp.local_select(ids_d[sl])
        cols = (
            [x, ids_d[sl].astype(paddle.float32), w_d[sl]]
            if x_first
            else [ids_d[sl].astype(paddle.float32), w_d[sl], x]
        )
        rows.append(paddle.concat(cols, axis=1))
    return paddle.concat(rows, axis=0)


@pytest.mark.parametrize("x_first", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_pieces_plan_matches_gpu_plan(case, x_first):
    E, K, T, NP, n, drop = CASES[case]
    rng, ids = _routing(E, K, T, drop)
    w = (2.0 ** -rng.randint(0, 2, (T, K))).astype(np.float32)
    y1 = (rng.randint(-1, 2, (T * K, 128)) * (rng.rand(T * K, 128) < 0.1)).astype(np.float32)
    w2 = rng.randint(-1, 2, (E, 256, 128)).astype(np.float32)
    index = P.build_index(frequent_pairs(ids, n) if n else [], E)
    gidx = GPUIndex(index)
    ids_d = dev(ids, paddle.int32)
    ref = plan_gpu(ids_d, gidx)
    paddle.device.synchronize()
    ref_moff = ref.moff.numpy().copy()
    used = np.zeros(gidx.cu_len, bool)  # cu entries the GEMMs read (the rest is unset)
    for _, s0, s1, at in gidx.gemms:
        used[at : at + s1 - s0 + 1] = True
    ref_cu = ref.cu.numpy()[used]
    gl = GroupList(index)
    o_ref, e_ref = merged_o(ids, w, y1, w2, ref, gl, E)

    pp = PlanPieces(gidx, T // NP, K, NP)
    src = _pieces_src(pp, ids_d, dev(w, paddle.float32), x_first)
    kw = dict(x_col=0, ids_col=ncols(K)) if x_first else {}
    for slot in (0, 1):  # a fresh slot and a reused one give the same plan
        plan = pp.merge(src, slot=slot % 2, **kw)
        paddle.device.synchronize()
        assert np.array_equal(plan.cu.numpy()[used], ref_cu)
        assert np.array_equal(plan.moff.numpy(), ref_moff)
        o, e = merged_o(ids, w, y1, w2, plan, gl, E)
        assert e == 0 and e_ref == 0
        assert np.array_equal(o, o_ref), np.abs(o - o_ref).max()


def test_fresh_plan_slot():
    """slot=None returns a new plan each call; earlier ones stay intact."""
    E, K, T, NP = 64, 8, 1024, 4
    _, ids = _routing(E, K, T, 0.1, seed=4)
    gidx = GPUIndex(P.build_index(frequent_pairs(ids, 100), E))
    pp = PlanPieces(gidx, T // NP, K, NP)
    ids_d = dev(ids, paddle.int32)
    src = _pieces_src(pp, ids_d, paddle.ones([T, K], dtype=paddle.float32), False)
    a = pp.merge(src, slot=None)
    a_moff = a.moff.numpy().copy()
    b = pp.merge(src, slot=None)
    paddle.device.synchronize()
    assert a.moff.data_ptr() != b.moff.data_ptr() and a.row_tok.data_ptr() != b.row_tok.data_ptr()
    assert np.array_equal(a.moff.numpy(), a_moff) and np.array_equal(b.moff.numpy(), a_moff)


def test_pieces_repeat_and_shape_checks():
    """Counters are re-zeroed between calls; bad shapes are rejected."""
    E, K, T, NP = 64, 8, 1024, 4
    _, ids = _routing(E, K, T, 0.1, seed=3)
    gidx = GPUIndex(P.build_index(frequent_pairs(ids, 100), E))
    pp = PlanPieces(gidx, T // NP, K, NP)
    a = pp.local_select(dev(ids[: T // NP], paddle.int32)).numpy()
    b = pp.local_select(dev(ids[: T // NP], paddle.int32)).numpy()
    R, S = K // 2, gidx.S
    # slots are atomic-ordered; picks, residual lanes + ranks, row offsets and
    # counts are not
    assert np.array_equal(a[:, :R], b[:, :R])
    ra, rb = a[:, 2 * R : 2 * R + K], b[:, 2 * R : 2 * R + K]
    assert np.array_equal(ra >= 0, rb >= 0)
    assert np.array_equal(np.where(ra >= 0, ra // 16384, 0), np.where(rb >= 0, rb // 16384, 0))
    assert np.array_equal(a[:, 2 * R + K], b[:, 2 * R + K])
    assert np.array_equal(a[:S, 2 * R + K + 1], b[:S, 2 * R + K + 1])
    with pytest.raises(ValueError):
        pp.local_select(dev(ids[: T // NP + 1], paddle.int32))
    with pytest.raises(ValueError):
        pp.merge(paddle.zeros([T - 1, 2 * K + ncols(K)], dtype=paddle.float32))
    with pytest.raises(ValueError):
        PlanPieces(gidx, 8, K, 2)  # fewer tokens per piece than segments


def test_pieces_batch_over_int16():
    """A batch of more than 65536 tokens (int32 token index): the merged down
    projection matches a dense reference, so no token index wraps."""
    E, K, NP, TL = 64, 8, 8, 9216
    T = NP * TL
    rng, ids = _routing(E, K, T, 0.05, seed=11)
    w = (2.0 ** -rng.randint(0, 2, (T, K))).astype(np.float32)
    y1 = (rng.randint(-1, 2, (T * K, 128)) * (rng.rand(T * K, 128) < 0.1)).astype(np.float32)
    w2 = rng.randint(-1, 2, (E, 256, 128)).astype(np.float32)
    index = P.build_index(frequent_pairs(ids, 300), E)
    gidx = GPUIndex(index)
    pp = PlanPieces(gidx, TL, K, NP)
    src = _pieces_src(pp, dev(ids, paddle.int32), dev(w, paddle.float32), False)
    plan = pp.merge(src, slot=None)
    assert plan.row_tok.dtype == paddle.int32
    o, e = merged_o(ids, w, y1, w2, plan, GroupList(index), E)
    assert e == 0
    ref = np.zeros((T, 256), np.float32)
    tt, kk = np.nonzero(ids >= 0)
    for x in range(E):
        s = ids[tt, kk] == x
        ref_part = (y1[tt[s] * K + kk[s]] * w[tt[s], kk[s], None]) @ w2[x].T
        np.add.at(ref, tt[s], ref_part)
    tol = np.maximum(np.abs(ref), 1.0) / 64
    assert np.all(np.abs(o - ref) <= tol), np.abs(o - ref).max()
    assert np.abs(o[65536:]).sum() > 0
