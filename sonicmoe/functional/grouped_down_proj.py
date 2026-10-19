#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Optional grouped down projection (GEMM2 + combine) for the MoE expert path.

Given a fixed list of expert groups and, per batch, an assignment of each
token's lanes to those groups (from ``row_group_planner``), the lanes of an
assigned group of size c become one row with a K axis of ``c * I``::

    a[row]  = [w_1 y1_1 | ... | w_c y1_c]
    y2[row] = a[row] @ [W2[e_1] | ... | W2[e_c]]^T = sum_j w_j y1_j @ W2[e_j]^T

computed as one segmented-B GEMM (quack ``B_idx``).  Lanes outside a group
stay one-expert rows pre-scaled by w, and the combine sums each token's rows.
Equivalent to the per-expert GEMM2 + combine up to bf16 rounding.  Opt-in and
forward only; the backward is unchanged and reads neither y2 nor o.

Install a plan for a call with ``grouped_down_proj_plan``; ``_DownProjection``
picks it up on the bf16 QuACK path without a bias.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field

import paddle
import triton
import triton.language as tl

from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm

from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton

_BR = 32  # rows per pack program
_BT = 32  # tokens per lane-table program


@triton.jit
def _lane_tables_kernel(
    rows_ptr, scores_ptr, off_ptr, efo_ptr, ids_ptr, row_out, w_out, err_ptr, T, E,
    K: tl.constexpr, KP: tl.constexpr, LOGE: tl.constexpr, VARLEN: tl.constexpr,
    CHECK: tl.constexpr, BT: tl.constexpr,
):
    """Per token: its lanes sorted by expert -> row_out / w_out[t*K + rank].
    The expert of a lane is the segment of efo holding its row.  CHECK: the
    sorted lane experts must equal the sorted valid (>= 0) routing ids, else
    *err_ptr is set."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    k = tl.arange(0, KP)
    tm = t < T
    if VARLEN:
        lo_t = tl.load(off_ptr + t, mask=tm, other=0)
        n = tl.load(off_ptr + t + 1, mask=tm, other=0) - lo_t
    else:
        lo_t = t * K
        n = tl.where(tm, K, 0)
    m = tm[:, None] & (k[None, :] < n[:, None])
    lane = lo_t[:, None] + k[None, :]
    row = tl.load(rows_ptr + lane, mask=m, other=0)
    lo = tl.zeros([BT, KP], dtype=tl.int32)
    hi = tl.full([BT, KP], E + 1, dtype=tl.int32)
    for _ in tl.static_range(LOGE):
        act = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(efo_ptr + mid, mask=m & act, other=0)
        go = act & (v <= row)
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(act & ~go, mid, hi)
    e = lo - 1
    key = tl.where(m, e * KP + k[None, :], 2147483647)
    key = tl.sort(key, dim=1)
    src = lo_t[:, None] + key % KP
    dst = t[:, None] * K + k[None, :]
    tl.store(row_out + dst, tl.load(rows_ptr + src, mask=m, other=0), mask=m)
    tl.store(w_out + dst, tl.load(scores_ptr + src, mask=m, other=0.0).to(tl.float32), mask=m)
    if CHECK:
        km = tm[:, None] & (k[None, :] < K)
        rid = tl.load(ids_ptr + t[:, None] * K + k[None, :], mask=km, other=-1)
        rid = tl.sort(tl.where(rid >= 0, rid, 2147483647), dim=1)
        bad = tl.sum(tl.where(tm[:, None] & (rid != tl.where(m, key // KP, 2147483647)), 1, 0))
        if bad > 0:
            tl.store(err_ptr, 1)


@triton.jit
def _pack_kernel(
    y1_ptr, tok_ptr, code_ptr, row_ptr, w_ptr, cu_ptr, dst_ptr, nseg,
    K: tl.constexpr, I: tl.constexpr, C: tl.constexpr, BR: tl.constexpr,
):
    """Rows [cu[0], cu[nseg]) of one GEMM:  dst[r, j*I:(j+1)*I] =
    w[t, l_j] * y1[row[t, l_j]],  l_j = nibble j of code[r] (sorted rank)."""
    lo = tl.load(cu_ptr)
    hi = tl.load(cu_ptr + nseg)
    r = lo + tl.program_id(0) * BR + tl.arange(0, BR)
    m = r < hi
    t = tl.load(tok_ptr + r, mask=m, other=0).to(tl.int32) & 0xFFFF
    code = tl.load(code_ptr + r, mask=m, other=0).to(tl.int32) & 0xFFFF
    c = tl.arange(0, I)
    for j in tl.static_range(C):
        lane = (code >> (4 * j)) & 15
        off = t * K + lane
        src = tl.load(row_ptr + off, mask=m, other=0).to(tl.int64)
        sc = tl.load(w_ptr + off, mask=m, other=0.0)
        v = tl.load(y1_ptr + src[:, None] * I + c[None, :], mask=m[:, None], other=0.0)
        v = v.to(tl.float32) * sc[:, None]
        tl.store(dst_ptr + (r[:, None].to(tl.int64) * C + j) * I + c[None, :], v.to(tl.bfloat16), mask=m[:, None])


def default_gemm_config():
    """Static (non-persistent) 128 x 256 tiles for the grouped GEMM (K = c*I)."""
    return GemmConfig(
        tile_m=128, tile_n=256, cluster_m=1, cluster_n=1, pingpong=False,
        is_dynamic_persistent=False, device_capacity=paddle.device.cuda.get_device_capability()[0],
    )


class GroupList:
    """A fixed group list on the device: per GEMM (one size bucket, then the
    residual one-expert rows) its group size, segment count, offset in
    ``Plan.cu`` and the expert table ``B_idx`` (None for the residual, where
    the segment index is the expert).  ``index`` is
    ``row_group_planner.build_index(groups, E)``."""

    def __init__(self, index):
        import numpy as np

        self.G, self.E = index.G, index.E
        self.gemms = []
        for c, s0, s1, at in index.gemms:
            bidx = None
            if c > 1:
                bidx = paddle.to_tensor(np.ascontiguousarray(index.seg_groups[s0:s1, :c], dtype=np.int32))
            self.gemms.append((c, s1 - s0, at, bidx))


@dataclass(eq=False)
class Plan:
    """One batch's plan on the device (``row_group_planner`` output).  Row
    arrays may be any prefix of length >= nrow; the buffers may be refilled in
    place."""

    cu: paddle.Tensor  # [index.cu_len] int32, per-GEMM row offsets
    row_tok: paddle.Tensor  # [>= nrow] int16
    row_code: paddle.Tensor  # [>= nrow] int16
    perm: paddle.Tensor  # [>= nrow] int32
    moff: paddle.Tensor  # [T + 1] int32
    _cu_views: dict = field(default_factory=dict, init=False, repr=False)

    def cu_slice(self, at, nseg):
        v = self._cu_views.get(at)
        if v is None:
            v = self._cu_views[at] = self.cu[at : at + nseg + 1]
        return v

    @classmethod
    def from_host(cls, out, nrow):
        import numpy as np

        n = max(int(nrow), 1)
        dev = lambda a: paddle.to_tensor(np.ascontiguousarray(a))  # noqa: E731
        return cls(dev(out["cu"]), dev(out["row_tok"][:n]), dev(out["row_code"][:n]), dev(out["perm"][:n]),
                   dev(out["moff"]))


def grouped_down_proj(
    y1, w2, expert_frequency_offset, lane_rows, lane_scores, token_lane_offset, ids, plan, groups,
    err=None, config=None, ready_event=None,
):
    """Grouped GEMM2 + combine of one batch; returns o [T, H] bf16.

    If ``ready_event`` is given (the plan is produced on another stream), the
    current stream waits on it first, so the plan can be built concurrently
    with the work before this call.
    """
    if ready_event is not None:
        paddle.device.cuda.current_stream().wait_event(ready_event)
    T, K = int(ids.shape[0]), int(ids.shape[1])
    I, E, H = int(y1.shape[1]), int(w2.shape[0]), int(w2.shape[1])
    assert E == groups.E
    config = config or default_gemm_config()
    if err is None:
        err = paddle.zeros([1], dtype=paddle.int32)
    TK = T * K
    row_s = paddle.empty([TK], dtype=paddle.int32)
    w_s = paddle.empty([TK], dtype=paddle.float32)
    varlen = token_lane_offset is not None
    off = token_lane_offset if varlen else lane_rows
    _lane_tables_kernel[(triton.cdiv(T, _BT),)](
        lane_rows, lane_scores, off, expert_frequency_offset, ids, row_s, w_s, err, T, E,
        K=K, KP=triton.next_power_of_2(K), LOGE=int(E + 1).bit_length(), VARLEN=varlen, CHECK=True,
        BT=_BT, num_warps=4,
    )
    a_buf = paddle.empty([TK * I], dtype=paddle.bfloat16)  # A of every bucket, reused in turn
    y2 = paddle.empty([TK, H], dtype=paddle.bfloat16)
    w2t = w2.permute(0, 2, 1)
    for c, nseg, at, bidx in groups.gemms:
        nmax = TK // c  # rows of this GEMM are < nmax
        a = a_buf[: nmax * c * I].view([nmax, c * I])
        cu = plan.cu_slice(at, nseg)
        _pack_kernel[(triton.cdiv(nmax, _BR),)](
            y1, plan.row_tok, plan.row_code, row_s, w_s, cu, a, nseg, K=K, I=I, C=c, BR=_BR, num_warps=4,
        )
        gemm(a, w2t, out=y2[:nmax], cu_seqlens_m=cu, B_idx=bidx, tuned=False, config=config)
    o = paddle.empty([T, H], dtype=paddle.bfloat16)
    token_gather_and_sum_varlen_K_triton(y2, None, o, plan.perm, plan.moff, T, K, H, True)
    return o


@dataclass
class _GroupedPlan:
    groups: GroupList
    plan: Plan
    ids: paddle.Tensor
    err: paddle.Tensor
    ready_event: object = None


_local = threading.local()


def current_grouped_plan():
    return getattr(_local, "cur", None)


@contextlib.contextmanager
def grouped_down_proj_plan(groups, plan, ids, err=None, ready_event=None):
    """Install (groups, plan, ids) for the next ``_DownProjection`` forward on
    this thread; None restores the default path.  ``ready_event``: an event the
    down projection waits on (the plan was produced on another stream).
    Nestable."""
    prev = getattr(_local, "cur", None)
    _local.cur = _GroupedPlan(
        groups, plan, ids, err if err is not None else paddle.zeros([1], dtype=paddle.int32), ready_event
    )
    try:
        yield _local.cur
    finally:
        _local.cur = prev
