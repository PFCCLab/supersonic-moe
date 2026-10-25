# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""On-device plan for the grouped down projection, built from per-piece parts.

When a batch is the concatenation of P pieces of T_l tokens that are produced
on different ranks (e.g. a ring round's batch gathered over the ranks of a
node), each token's group selection depends only on its own routing.  So the
plan splits into

  ``local_select``  once per piece, on the rank that owns the piece's tokens:
                    the greedy pair selection of `plan_gpu` plus each pick's
                    slot within the piece, packed into ``ncols(K)`` float32
                    columns that can travel with the routing;
  ``merge``         once per batch: one scan of the P x S piece counts and one
                    scatter into the batch's `Plan`.

`merge` returns the same `Plan` as `plan_gpu` on the concatenated routing
(identical ``cu`` / ``moff``; within a segment the row order may differ, which
the combine allows).  Pairs lists only, like `plan_gpu`.

Columns of a piece (float32 holding integers < 2**24; R = K // 2):
  [0, R)        segment << 8 | code of greedy pick r   (-1: none)
  [R, 2R)       slot of pick r within its segment, in the piece
  [2R, 2R+K)    slot | rank << 14 of lane k if it is a residual lane, else -1
  2R+K          the token's first row within the piece (exclusive prefix)
  2R+K+1        rows 0..S-1 only: the piece's row count of segment `row`
"""

from __future__ import annotations

import numpy as np
import paddle
import triton
import triton.language as tl

from .grouped_down_proj import Plan


def ncols(K):
    """Columns `local_select` produces per token for top-K routing."""
    return 2 * (K // 2) + K + 2


@triton.jit
def _local_k(
    idx_ptr,
    tab_ptr,
    segof_ptr,
    li_ptr,
    lj_ptr,
    cnt_ptr,
    x_ptr,
    nr_ptr,
    bsum_ptr,
    T,
    G,
    E: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    NX: tl.constexpr,
    BT: tl.constexpr,
    NPP: tl.constexpr,
    NP: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    k = tl.arange(0, 16)
    km = k < K
    ids = tl.load(idx_ptr + t[:, None] * K + k[None, :], mask=tm[:, None] & km[None, :], other=-1)
    valid = (ids >= 0) & km[None, :]
    lt = valid[:, :, None] & valid[:, None, :] & (ids[:, None, :] < ids[:, :, None])
    rank = tl.sum(lt.to(tl.int32), axis=2)
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
    key = tl.where((g >= 0) & both, (g << 6) | p[None, :], INF)
    bits = (1 << li) | (1 << lj)
    used = tl.zeros([BT], dtype=tl.int32)
    ngrp = tl.zeros([BT], dtype=tl.int32)
    xrow = x_ptr + t * NX
    for _r in tl.static_range(R):
        c = tl.where((bits[None, :] & used[:, None]) == 0, key, INF)
        best = tl.min(c, axis=1)
        ok = (best < INF) & tm
        bp = tl.where(ok, best & 63, 0)
        bi = tl.load(li_ptr + bp, mask=ok, other=0)
        bj = tl.load(lj_ptr + bp, mask=ok, other=0)
        rk_i = tl.sum(tl.where(k[None, :] == bi[:, None], rank, 0), axis=1)
        rk_j = tl.sum(tl.where(k[None, :] == bj[:, None], rank, 0), axis=1)
        code = tl.minimum(rk_i, rk_j) + tl.maximum(rk_i, rk_j) * 16
        seg = tl.load(segof_ptr + (best >> 6), mask=ok, other=0)
        slot = tl.atomic_add(cnt_ptr + seg, 1, mask=ok)
        tl.store(xrow + _r, tl.where(ok, seg * 256 + code, -1).to(tl.float32), mask=tm)
        tl.store(xrow + R + _r, tl.where(ok, slot, 0).to(tl.float32), mask=tm)
        used = tl.where(ok, used | (1 << bi) | (1 << bj), used)
        ngrp += ok.to(tl.int32)
    rmask = tm[:, None] & valid & (((used[:, None] >> k[None, :]) & 1) == 0)
    rs = tl.atomic_add(cnt_ptr + G + ids, 1, mask=rmask)
    tl.store(
        xrow[:, None] + 2 * R + k[None, :],
        tl.where(rmask, rs + rank * 16384, -1).to(tl.float32),
        mask=tm[:, None] & km[None, :],
    )
    nr = ngrp + tl.sum(rmask.to(tl.int32), axis=1)
    tl.store(nr_ptr + t, nr, mask=tm)
    tl.store(bsum_ptr + pid, tl.sum(tl.where(tm, nr, 0), axis=0))


@triton.jit
def _local_fin(
    cnt_ptr,
    nr_ptr,
    bsum_ptr,
    x_ptr,
    T,
    S,
    NB,
    K: tl.constexpr,
    R: tl.constexpr,
    NX: tl.constexpr,
    BT: tl.constexpr,
    NBP: tl.constexpr,
    SP: tl.constexpr,
):
    """Per block: the tokens' local row offsets.  Block 0 also stores the
    piece's per-segment counts (column 2R+K+1 of rows 0..S-1) and re-zeroes
    the counters for the next call."""
    pid = tl.program_id(0)
    q = tl.arange(0, NBP)
    bs = tl.load(bsum_ptr + q, mask=q < NB, other=0)
    base = tl.sum(tl.where(q < pid, bs, 0), axis=0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    nr = tl.load(nr_ptr + t, mask=tm, other=0)
    o = tl.cumsum(nr, axis=0) - nr + base
    tl.store(x_ptr + t * NX + 2 * R + K, o.to(tl.float32), mask=tm)
    if pid == 0:
        s = tl.arange(0, SP)
        sm = s < S
        c = tl.load(cnt_ptr + s, mask=sm, other=0)
        tl.store(x_ptr + s * NX + 2 * R + K + 1, c.to(tl.float32), mask=sm & (s < T))
        tl.store(cnt_ptr + s, tl.zeros([SP], dtype=tl.int32), mask=sm)


@triton.jit
def _merge_scan(
    x_ptr,
    segoff_ptr,
    poff_ptr,
    pbase_ptr,
    cu_ptr,
    cusrc_ptr,
    moff_ptr,
    P,
    TL,
    S,
    CUL,
    K: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    X0: tl.constexpr,
    PP: tl.constexpr,
    SP: tl.constexpr,
    CUP: tl.constexpr,
):
    """One program: piece counts [P, S] -> segment offsets, per-piece segment
    bases, per-piece row bases, and the per-GEMM `cu` slices."""
    pi = tl.arange(0, PP)
    s = tl.arange(0, SP)
    m = (pi[:, None] < P) & (s[None, :] < S)
    c = tl.load(x_ptr + (pi[:, None] * TL + s[None, :]) * W + X0 + 2 * R + K + 1, mask=m, other=0.0).to(tl.int32)
    tot = tl.sum(c, axis=0)  # [SP]
    so = tl.cumsum(tot, axis=0) - tot  # exclusive
    tl.store(segoff_ptr + s, so, mask=s < S)
    tl.store(segoff_ptr + S, tl.sum(tot, axis=0))
    tl.store(moff_ptr + P * TL, tl.sum(tot, axis=0))
    po = so[None, :] + tl.cumsum(c, axis=0) - c
    tl.store(poff_ptr + pi[:, None] * S + s[None, :], po, mask=m)
    pr = tl.sum(c, axis=1)
    tl.store(pbase_ptr + pi, tl.cumsum(pr, axis=0) - pr, mask=pi < P)
    ci = tl.arange(0, CUP)
    cm = ci < CUL
    src = tl.load(cusrc_ptr + ci, mask=cm, other=0)
    tl.debug_barrier()
    tl.store(cu_ptr + ci, tl.load(segoff_ptr + src, mask=cm, other=0), mask=cm)


@triton.jit
def _merge_rows(
    x_ptr,
    poff_ptr,
    pbase_ptr,
    rowtok_ptr,
    rowcode_ptr,
    perm_ptr,
    moff_ptr,
    T,
    TL,
    S,
    G,
    K: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    X0: tl.constexpr,
    IDC: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    pc = t // TL
    prow = x_ptr + t * W
    xrow = prow + X0
    o = tl.load(xrow + 2 * R + K, mask=tm, other=0.0).to(tl.int32) + tl.load(pbase_ptr + pc, mask=tm, other=0)
    tl.store(moff_ptr + t, o, mask=tm)
    pb = poff_ptr + pc * S
    for r in tl.static_range(R):
        v = tl.load(xrow + r, mask=tm, other=-1.0).to(tl.int32)
        ok = v >= 0
        row = tl.load(pb + (v >> 8), mask=ok, other=0) + tl.load(xrow + R + r, mask=ok, other=0.0).to(tl.int32)
        tl.store(rowtok_ptr + row, t, mask=ok)
        tl.store(rowcode_ptr + row, (v & 255).to(tl.int16), mask=ok)
        tl.store(perm_ptr + o, row, mask=ok)
        o += ok.to(tl.int32)
    for kk in tl.static_range(K):
        v = tl.load(xrow + 2 * R + kk, mask=tm, other=-1.0).to(tl.int32)
        ok = v >= 0
        e = tl.load(prow + IDC + kk, mask=ok, other=0.0).to(tl.int32)
        row = tl.load(pb + G + e, mask=ok, other=0) + (v & 16383)
        tl.store(rowtok_ptr + row, t, mask=ok)
        tl.store(rowcode_ptr + row, (v >> 14).to(tl.int16), mask=ok)
        tl.store(perm_ptr + o, row, mask=ok)
        o += ok.to(tl.int32)


class PlanPieces:
    """`local_select` / `merge` for one (tokens per piece, K, pieces) and a
    `GPUIndex`.  `merge` keeps its output buffers per ``slot``: a caller that
    holds several merged plans alive at once passes a distinct slot per plan,
    or ``slot=None`` for a freshly allocated plan it owns."""

    def __init__(self, gidx, tokens_per_piece, K, pieces, BT=64):
        TL, P = int(tokens_per_piece), int(pieces)
        if not 2 <= K <= 11:
            raise ValueError(f"K={K}: needs 2 <= K <= 11")
        # A residual lane's slot shares a column with its rank (slot | rank
        # << 14); the batch's token index is int32.
        if TL > 16384:
            raise ValueError(f"{P} pieces of {TL} tokens: needs <= 16384 tokens per piece")
        if gidx.S > TL:
            raise ValueError(f"{gidx.S} segments > {TL} tokens per piece (counts are stored one per row)")
        self.g, self.TL, self.K, self.P, self.BT = gidx, TL, K, P, BT
        self.R, self.NX = K // 2, ncols(K)
        self.NBL = triton.cdiv(TL, BT)
        self.cnt = paddle.zeros([gidx.S], dtype=paddle.int32)
        self.nr = paddle.empty([TL], dtype=paddle.int32)
        self.bsum = paddle.empty([self.NBL], dtype=paddle.int32)
        src = np.zeros([gidx.cu_len], np.int32)
        for _, s0, s1, at in gidx.gemms:
            src[at : at + s1 - s0 + 1] = np.arange(s0, s1 + 1)
        self.cusrc = paddle.to_tensor(src)
        gidx._lane_pairs(K)
        self._slots = {}

    def _new_plan(self):
        g, T = self.g, self.TL * self.P
        TK = T * self.K
        i32 = lambda n: paddle.empty([n], dtype=paddle.int32)  # noqa: E731
        return Plan(
            cu=i32(g.cu_len),
            row_tok=i32(TK),
            row_code=paddle.empty([TK], dtype=paddle.int16),
            perm=i32(TK),
            moff=i32(T + 1),
        )

    def _slot(self, slot):
        b = self._slots.get(slot)
        if b is None:
            g = self.g
            i32 = lambda n: paddle.empty([n], dtype=paddle.int32)  # noqa: E731
            b = self._slots[slot] = (
                i32(g.S + 1),
                i32(self.P * g.S),
                i32(self.P),
                None if slot is None else self._new_plan(),
            )
        if slot is None:  # a fresh plan; the merge's own work buffers are reused
            b = b[:3] + (self._new_plan(),)
        return b

    def local_select(self, ids):
        """ids [T_l, K] int32 (-1 = dropped) of one piece -> new [T_l, ncols(K)] float32."""
        g = self.g
        if list(ids.shape) != [self.TL, self.K]:
            raise ValueError(f"ids {list(ids.shape)} != [{self.TL}, {self.K}]")
        x = paddle.empty([self.TL, self.NX], dtype=paddle.float32)
        _local_k[(self.NBL,)](
            ids,
            g.tab,
            g.seg_of,
            g._li,
            g._lj,
            self.cnt,
            x,
            self.nr,
            self.bsum,
            self.TL,
            g.G,
            E=g.E,
            K=self.K,
            R=self.R,
            NX=self.NX,
            BT=self.BT,
            NPP=64,
            NP=self.K * (self.K - 1) // 2,
            num_warps=4,
        )
        _local_fin[(self.NBL,)](
            self.cnt,
            self.nr,
            self.bsum,
            x,
            self.TL,
            g.S,
            self.NBL,
            K=self.K,
            R=self.R,
            NX=self.NX,
            BT=self.BT,
            NBP=triton.next_power_of_2(self.NBL),
            SP=triton.next_power_of_2(g.S),
            num_warps=4,
        )
        return x

    def merge(self, src, slot=0, ids_col=0, x_col=None):
        """Plan of a batch from its P pieces, without host sync.

        src: [P * T_l, W] float32, row-contiguous; columns ``ids_col ..
        ids_col + K`` hold the routing ids, columns ``x_col .. x_col +
        ncols(K)`` the pieces' `local_select` output (default: the last
        ``ncols(K)`` columns)."""
        g = self.g
        T, W = self.TL * self.P, int(src.shape[1])
        x_col = W - self.NX if x_col is None else int(x_col)
        if int(src.shape[0]) != T or not 0 <= x_col <= W - self.NX or not 0 <= ids_col <= W - self.K:
            raise ValueError(f"src {list(src.shape)} does not hold {self.P} pieces of {self.TL} tokens")
        segoff, poff, pbase, p = self._slot(slot)
        _merge_scan[(1,)](
            src,
            segoff,
            poff,
            pbase,
            p.cu,
            self.cusrc,
            p.moff,
            self.P,
            self.TL,
            g.S,
            g.cu_len,
            K=self.K,
            R=self.R,
            W=W,
            X0=x_col,
            PP=triton.next_power_of_2(self.P),
            SP=triton.next_power_of_2(g.S),
            CUP=triton.next_power_of_2(g.cu_len),
            num_warps=8,
        )
        _merge_rows[(triton.cdiv(T, self.BT),)](
            src,
            poff,
            pbase,
            p.row_tok,
            p.row_code,
            p.perm,
            p.moff,
            T,
            self.TL,
            g.S,
            g.G,
            K=self.K,
            R=self.R,
            W=W,
            X0=x_col,
            IDC=int(ids_col),
            BT=self.BT,
            num_warps=4,
        )
        return p
