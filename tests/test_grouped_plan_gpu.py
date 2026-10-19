"""GPU planner (grouped_plan_gpu) vs the CPU planner: the merged down-projection
output must be identical when both plans are built from the same routing.  The
GPU plan's row order within a segment differs (atomic), but the combined output
is order-independent, so `o` matches bitwise on integer data."""

import itertools
from collections import Counter

import numpy as np
import paddle
import pytest

paddle.enable_compat(silent=True)

from sonicmoe import row_group_planner as P  # noqa: E402
from sonicmoe import GroupList, Plan, grouped_down_proj  # noqa: E402
from sonicmoe.functional.grouped_plan_gpu import GPUIndex, plan_gpu  # noqa: E402


def dev(a, dt):
    return paddle.to_tensor(np.ascontiguousarray(a)).astype(dt)


def frequent_pairs(ids, n):
    cnt = Counter()
    for r in ids[:1024]:
        v = sorted(int(e) for e in r if e >= 0)
        cnt.update(itertools.combinations(v, 2))
    out = [list(k) for k, _ in cnt.most_common(n)]
    np.random.RandomState(0).shuffle(out)
    return out


def lanes_of(ids, E):
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


def merged_o(ids, w, y1_rows, w2, plan, groups, E):
    T, K = ids.shape
    I = w2.shape[2]
    tt, kk, row, efo, off = lanes_of(ids, E)
    y1 = np.zeros((row.size, I), np.float32)
    y1[row] = y1_rows[tt * K + kk]
    err = paddle.zeros([1], dtype=paddle.int32)
    o = grouped_down_proj(
        dev(y1, paddle.float32).astype(paddle.bfloat16), dev(w2, paddle.float32).astype(paddle.bfloat16),
        dev(efo, paddle.int32), dev(row, paddle.int32), dev(w[tt, kk], paddle.float32),
        dev(off, paddle.int32), dev(ids, paddle.int32), plan, groups, err,
    )
    paddle.device.synchronize()
    return o.astype(paddle.float32).numpy(), int(err.numpy()[0])


CASES = {"pairs": (64, 8, 2048, 256, 300, 0.0), "dropped": (64, 8, 2048, 256, 300, 0.2),
         "k10": (48, 10, 2048, 256, 400, 0.05), "empty": (64, 8, 1024, 256, 0, 0.1)}


@pytest.mark.parametrize("case", list(CASES))
def test_gpu_plan_matches_cpu_plan(case):
    E, K, T, H, n, drop = CASES[case]
    rng = np.random.RandomState(7)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    ids[rng.rand(T, K) < drop] = -1
    w = (2.0 ** -rng.randint(0, 2, (T, K))).astype(np.float32)
    y1 = (rng.randint(-1, 2, (T * K, 128)) * (rng.rand(T * K, 128) < 0.1)).astype(np.float32)
    w2 = rng.randint(-1, 2, (E, H, 128)).astype(np.float32)
    groups = frequent_pairs(ids, n) if n else []
    index = P.build_index(groups, E)
    # CPU plan
    out = P.alloc_out(T, K, index)
    nrow = P.plan(ids.astype(np.int16), index, P.Scratch(T, K, index, 8), out)
    cpu_plan = Plan.from_host(out, nrow)
    gl = GroupList(index)
    o_cpu, e_cpu = merged_o(ids, w, y1, w2, cpu_plan, gl, E)
    # GPU plan
    gpu_plan = plan_gpu(dev(ids, paddle.int32), GPUIndex(index))
    paddle.device.synchronize()
    o_gpu, e_gpu = merged_o(ids, w, y1, w2, gpu_plan, gl, E)
    assert e_cpu == 0 and e_gpu == 0
    assert np.array_equal(o_gpu, o_cpu), np.abs(o_gpu - o_cpu).max()


def test_gpu_plan_covers_every_lane_once():
    """moff spans all rows and perm is a permutation of the used rows."""
    E, K, T = 64, 8, 1024
    rng = np.random.RandomState(3)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)])
    index = P.build_index(frequent_pairs(ids, 200), E)
    gpu_plan = plan_gpu(dev(ids, paddle.int32), GPUIndex(index))
    paddle.device.synchronize()
    moff = gpu_plan.moff.numpy()
    nrow = int(moff[-1])
    assert moff[0] == 0 and (np.diff(moff) >= 0).all()
    perm = gpu_plan.perm.numpy()[:nrow]
    assert len(np.unique(perm)) == nrow  # every row referenced once, contiguous [0, nrow)
    assert nrow < T * K  # pairs were merged (each merged pair saves one row)
