# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""On-device planner for the grouped down projection (pairs lists).

Produces the same `Plan` as `row_group_planner.plan` -- `cu`, `row_tok`,
`row_code` (4-bit sorted-rank lanes), `perm`, `moff` -- but on the GPU in
three triton launches with no host sync, so the per-batch assignment runs on
the device next to the GEMM instead of on a CPU process.

Three kernels (a counting sort):
  k1  per token: C(K,2) dense-table lookups, greedy by group priority over
      lane-disjoint pairs (<= K/2 rounds), atomicAdd per chosen segment and
      per residual expert; store the per-token selection and row count.
  k2  one program: exclusive scan of the segment counts -> seg_off, then the
      per-GEMM `cu` slices.
  k3  per token: scatter rows to their segment slots -> row_tok / row_code and
      the combine perm / moff.
Row order inside a segment is atomic-decided (not deterministic); the combine
reads by perm and the backward never reads the plan, so that is allowed.
Pairs lists only (every group size 2); mixed sizes still use the CPU planner.
"""

from __future__ import annotations

import numpy as np
import paddle
import triton
import triton.language as tl

from .grouped_down_proj import Plan

_NPP = 64  # padded C(K,2) (K <= 11 -> <= 55)


@triton.jit
def _k1(idx_ptr, tab_ptr, segof_ptr, li_ptr, lj_ptr, cnt_ptr, selseg_ptr, selslot_ptr,
        selcode_ptr, rslot_ptr, used_ptr, rank_ptr, nr_ptr, bsum_ptr, T, G, E: tl.constexpr,
        K: tl.constexpr, R: tl.constexpr, BT: tl.constexpr, NPP: tl.constexpr, NP: tl.constexpr):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    k = tl.arange(0, 16)
    km = k < K
    ids = tl.load(idx_ptr + t[:, None] * K + k[None, :], mask=tm[:, None] & km[None, :], other=-1)
    valid = (ids >= 0) & km[None, :]
    # sorted rank of every lane: # valid lanes with a smaller expert id
    lt = valid[:, :, None] & valid[:, None, :] & (ids[:, None, :] < ids[:, :, None])
    rank = tl.sum(lt.to(tl.int32), axis=2)  # [BT, 16]
    tl.store(rank_ptr + t[:, None] * K + k[None, :], rank, mask=tm[:, None] & km[None, :])
    p = tl.arange(0, NPP)
    pm = p < NP
    li = tl.load(li_ptr + p, mask=pm, other=0)
    lj = tl.load(lj_ptr + p, mask=pm, other=0)
    m2 = tm[:, None] & pm[None, :]
    a = tl.load(idx_ptr + t[:, None] * K + li[None, :], mask=m2, other=-1)
    b = tl.load(idx_ptr + t[:, None] * K + lj[None, :], mask=m2, other=-1)
    lo = tl.minimum(a, b)
    hi = tl.maximum(a, b)
    both = (a >= 0) & (b >= 0)
    g = tl.load(tab_ptr + lo * E + hi, mask=m2 & both, other=-1)
    INF: tl.constexpr = 2147483647
    g = tl.where((g >= 0) & both, g, INF)
    bits = (1 << li) | (1 << lj)
    used = tl.zeros([BT], dtype=tl.int32)
    ngrp = tl.zeros([BT], dtype=tl.int32)
    for _r in tl.static_range(R):
        free = (bits[None, :] & used[:, None]) == 0
        c = tl.where(free, g, INF)
        best = tl.min(c, axis=1)
        ok = (best < INF) & tm
        bp = tl.argmin(c, axis=1)
        sel = p[None, :] == bp[:, None]
        bl = tl.sum(tl.where(sel, bits[None, :], 0), axis=1)
        bi = tl.sum(tl.where(sel, li[None, :], 0), axis=1)
        bj = tl.sum(tl.where(sel, lj[None, :], 0), axis=1)
        # sorted-rank code: lower rank in nibble 0
        rk_i = tl.sum(tl.where(k[None, :] == bi[:, None], rank, 0), axis=1)
        rk_j = tl.sum(tl.where(k[None, :] == bj[:, None], rank, 0), axis=1)
        r0 = tl.minimum(rk_i, rk_j)
        r1 = tl.maximum(rk_i, rk_j)
        seg = tl.load(segof_ptr + best, mask=ok, other=0)
        slot = tl.atomic_add(cnt_ptr + seg, 1, mask=ok)
        tl.store(selseg_ptr + t * R + _r, tl.where(ok, seg, -1), mask=tm)
        tl.store(selslot_ptr + t * R + _r, slot, mask=ok)
        tl.store(selcode_ptr + t * R + _r, r0 + r1 * 16, mask=ok)
        used = tl.where(ok, used | bl, used)
        ngrp += ok.to(tl.int32)
    rmask = tm[:, None] & valid & (((used[:, None] >> k[None, :]) & 1) == 0)
    rseg = G + ids  # residual segment = G + expert id
    rs = tl.atomic_add(cnt_ptr + rseg, 1, mask=rmask)
    tl.store(rslot_ptr + t[:, None] * K + k[None, :], rs, mask=rmask)
    tl.store(used_ptr + t, used, mask=tm)
    nr = ngrp + tl.sum(rmask.to(tl.int32), axis=1)
    tl.store(nr_ptr + t, nr, mask=tm)
    tl.store(bsum_ptr + pid, tl.sum(tl.where(tm, nr, 0), axis=0))


@triton.jit
def _k2(cnt_ptr, bsum_ptr, segoff_ptr, boff_ptr, S, NB, SP: tl.constexpr, NBP: tl.constexpr):
    i = tl.arange(0, SP)
    m = i < S
    c = tl.load(cnt_ptr + i, mask=m, other=0)
    cs = tl.cumsum(c, axis=0)  # inclusive
    tl.store(segoff_ptr + 1 + i, cs, mask=m)
    tl.store(segoff_ptr + i, 0, mask=i == 0)
    q = tl.arange(0, NBP)
    qm = q < NB
    bs = tl.load(bsum_ptr + q, mask=qm, other=0)
    tl.store(boff_ptr + q, tl.cumsum(bs, axis=0) - bs, mask=qm)


@triton.jit
def _k3(idx_ptr, selseg_ptr, selslot_ptr, selcode_ptr, rslot_ptr, used_ptr, rank_ptr, nr_ptr,
        boff_ptr, segoff_ptr, cnt_ptr, rowtok_ptr, rowcode_ptr, perm_ptr, moff_ptr, S,
        T, G, K: tl.constexpr, R: tl.constexpr, BT: tl.constexpr, SP: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:  # reset counts so persistent buffers can be reused next call
        si = tl.arange(0, SP)
        tl.store(cnt_ptr + si, tl.zeros([SP], dtype=tl.int32), mask=si < S)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    nr = tl.load(nr_ptr + t, mask=tm, other=0)
    o = tl.cumsum(nr, axis=0) - nr + tl.load(boff_ptr + pid)
    tl.store(moff_ptr + t, o, mask=tm & (t == 0))
    tl.store(moff_ptr + 1 + t, o + nr, mask=tm)
    for r in tl.static_range(R):
        seg = tl.load(selseg_ptr + t * R + r, mask=tm, other=-1)
        ok = seg >= 0
        row = tl.load(segoff_ptr + seg, mask=ok, other=0) + tl.load(selslot_ptr + t * R + r, mask=ok, other=0)
        tl.store(rowtok_ptr + row, t.to(tl.int16), mask=ok)
        tl.store(rowcode_ptr + row, tl.load(selcode_ptr + t * R + r, mask=ok, other=0).to(tl.int16), mask=ok)
        tl.store(perm_ptr + o, row, mask=ok)
        o += ok.to(tl.int32)
    used = tl.load(used_ptr + t, mask=tm, other=0)
    for k in tl.static_range(K):
        rm = tm & (((used >> k) & 1) == 0) & (tl.load(idx_ptr + t * K + k, mask=tm, other=-1) >= 0)
        e = tl.load(idx_ptr + t * K + k, mask=rm, other=0)
        row = tl.load(segoff_ptr + G + e, mask=rm, other=0) + tl.load(rslot_ptr + t * K + k, mask=rm, other=0)
        tl.store(rowtok_ptr + row, t.to(tl.int16), mask=rm)
        tl.store(rowcode_ptr + row, tl.load(rank_ptr + t * K + k, mask=rm, other=0).to(tl.int16), mask=rm)
        tl.store(perm_ptr + o, row, mask=rm)
        o += rm.to(tl.int32)


class GPUIndex:
    """Device tables for a pairs-only list (``row_group_planner.build_index``)."""

    def __init__(self, index, BT=128):
        if not index.pairs_only:
            raise ValueError("grouped_plan_gpu supports pairs-only lists; use the CPU planner for mixed sizes")
        self.G, self.E, self.BT = index.G, index.E, BT
        self.S = index.G + index.E
        self.cu_len = index.cu_len
        self.gemms = tuple((c, s0, s1, at) for (c, s0, s1, at) in index.gemms)
        dev = lambda a: paddle.to_tensor(np.ascontiguousarray(a))  # noqa: E731
        self.tab = dev(index.pair_tab.astype(np.int32))
        self.seg_of = dev(index.seg_of.astype(np.int32))
        self.K = None  # set on first call (depends on ids), li/lj built then
        self._li = self._lj = None

    def _lane_pairs(self, K):
        if self.K == K:
            return
        li = [i for i in range(K) for j in range(i + 1, K)]
        lj = [j for i in range(K) for j in range(i + 1, K)]
        dev = lambda a: paddle.to_tensor(np.ascontiguousarray(np.int32(a)))  # noqa: E731
        self._li, self._lj, self.K = dev(li), dev(lj), K


def plan_gpu(ids, gidx):
    """Build one batch's `Plan` on the device from int32 routing ``ids`` [T, K]
    (-1 = dropped) and a :class:`GPUIndex`.  No host sync."""
    T, K = int(ids.shape[0]), int(ids.shape[1])
    G, E, S, BT = gidx.G, gidx.E, gidx.S, gidx.BT
    gidx._lane_pairs(K)
    NP = K * (K - 1) // 2
    R = K // 2
    NB = triton.cdiv(T, BT)
    TK = T * K
    i32 = lambda n: paddle.empty([n], dtype=paddle.int32)  # noqa: E731
    cnt = paddle.zeros([S], dtype=paddle.int32)
    selseg, selslot, selcode = i32(T * R), i32(T * R), i32(T * R)
    rslot, used, rank, nr, bsum = i32(TK), i32(T), i32(TK), i32(T), i32(NB)
    segoff, boff = i32(S + 1), i32(NB)
    row_tok = paddle.empty([TK], dtype=paddle.int16)
    row_code = paddle.empty([TK], dtype=paddle.int16)
    perm, moff, cu = i32(TK), i32(T + 1), i32(gidx.cu_len)
    _k1[(NB,)](ids, gidx.tab, gidx.seg_of, gidx._li, gidx._lj, cnt, selseg, selslot, selcode,
               rslot, used, rank, nr, bsum, T, G, E=E, K=K, R=R, BT=BT, NPP=_NPP, NP=NP, num_warps=4)
    _k2[(1,)](cnt, bsum, segoff, boff, S, NB, SP=triton.next_power_of_2(S),
              NBP=triton.next_power_of_2(NB), num_warps=8)
    for c, s0, s1, at in gidx.gemms:  # per-GEMM cu slice (device copy, no host sync)
        cu[at : at + s1 - s0 + 1] = segoff[s0 : s1 + 1]
    _k3[(NB,)](ids, selseg, selslot, selcode, rslot, used, rank, nr, boff, segoff, cnt,
               row_tok, row_code, perm, moff, S, T, G, K=K, R=R, BT=BT,
               SP=triton.next_power_of_2(S), num_warps=4)
    return Plan(cu=cu, row_tok=row_tok, row_code=row_code, perm=perm, moff=moff)
