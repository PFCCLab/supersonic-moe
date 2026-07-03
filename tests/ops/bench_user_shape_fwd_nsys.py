#!/usr/bin/env python3
"""Session 69 — reproduce user shape & profile sonic-meta routing region.

Shape: H=1024 I=1024 K=16 E_LOCAL=96 EP_SIZE=8 SEQ_LEN=16384
(matches tests/ops/bench_mlpnode_mem.py)

Generates routing identical in distribution to user's `make_inputs` but built
on-GPU in O(NK) instead of a Python loop.

Forward-only nsys profile so we can isolate the sonic-meta share of fwd time.
"""
import argparse
import math
import os
import sqlite3
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


def gpu_projection_us(sqlite_path: str, n_iters: int) -> float:
    conn = sqlite3.connect(sqlite_path)
    nv = conn.execute("SELECT start, end FROM NVTX_EVENTS WHERE text='BENCH'").fetchall()
    if not nv:
        rows = conn.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    else:
        s, e = nv[0]
        rows = conn.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start",
            (s, e),
        ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("no kernels")
    merged = []
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce)); cs, ce = s, e
    merged.append((cs, ce))
    return sum(e - s for s, e in merged) / 1000.0 / n_iters


def kernel_breakdown(sqlite_path: str, n_iters: int, top_n: int = 25):
    conn = sqlite3.connect(sqlite_path)
    nv = conn.execute("SELECT start, end FROM NVTX_EVENTS WHERE text='BENCH'").fetchone()
    s, e = nv
    rows = conn.execute(
        "SELECT k.shortName, COUNT(*), SUM(k.end-k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE k.start>=? AND k.end<=? "
        "GROUP BY k.shortName ORDER BY SUM(k.end-k.start) DESC LIMIT ?",
        (s, e, top_n),
    ).fetchall()
    ids = dict(conn.execute("SELECT id, value FROM StringIds").fetchall())
    conn.close()
    out = []
    for sid, cnt, ns in rows:
        out.append((ids.get(sid, "?"), cnt, ns / 1000.0 / n_iters))
    return out


def make_routing(N_global, n_experts, ep_size, topk, device, seed=42):
    """Build dispatched_indices + dispatched_probs matching user's distribution
    but in pure tensor ops (avoids 131k-iter Python loop)."""
    import torch
    torch.manual_seed(seed)
    E_global = ep_size * n_experts

    logits = torch.randn(N_global, E_global)
    probs = torch.softmax(logits, dim=-1)
    _, top_idx = torch.topk(probs, topk, dim=-1)
    perm = torch.argsort(torch.rand(N_global, topk), dim=-1)
    top_idx = torch.gather(top_idx, -1, perm)
    top_p = torch.gather(probs, -1, top_idx)

    keep = top_idx < n_experts
    has_any = keep.any(dim=-1)
    top_idx = top_idx[has_any]
    top_p = top_p[has_any]
    keep = keep[has_any]
    dispatched_indices = torch.where(keep, top_idx, torch.tensor(-1, dtype=top_idx.dtype))
    dispatched_probs = torch.where(keep, top_p, torch.tensor(0.0))

    dispatched_indices = dispatched_indices.to(device, dtype=torch.int32)
    dispatched_probs = dispatched_probs.to(device, dtype=torch.float32)

    tpe = [int((dispatched_indices == e).sum().item()) for e in range(n_experts)]
    return dispatched_indices, dispatched_probs, tpe


def run_forward_only(H, I, K, E, EP, SEQ, n_warmup, n_iters, seed=42):
    import paddle
    paddle.enable_compat()
    import torch

    from sonicmoe.ernie_compat import (
        SonicMoEMlpNode,
        flush_native_grads,
        invalidate_weight_caches,
    )
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True
    functional._ALIGNMENT_STREAK = 100

    class MockExpert:
        def __init__(self, h, i, s):
            paddle.seed(s)
            self.up_gate_proj = type("P", (), {
                "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
            })()
            self.down_proj = type("P", (), {
                "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
            })()
            self.up_gate_proj.weight.stop_gradient = False
            self.down_proj.weight.stop_gradient = False

    experts = [MockExpert(H, I, e) for e in range(E)]
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    dispatched_indices, dispatched_probs, tpe = make_routing(
        EP * SEQ, E, EP, K, "cuda", seed=seed,
    )
    N_recv = dispatched_indices.shape[0]
    print(f"  routing: N_recv={N_recv} tpe(min/max/sum)={min(tpe)}/{max(tpe)}/{sum(tpe)}")
    paddle.seed(0)
    x = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02

    print(f"warmup {n_warmup}...")
    for _ in range(n_warmup):
        out = node.forward(x, tpe, dispatched_indices=dispatched_indices, dispatched_probs=dispatched_probs)
        del out
    flush_native_grads()
    torch.cuda.synchronize()

    se = torch.cuda.Event(enable_timing=True); ee = torch.cuda.Event(enable_timing=True)
    print(f"bench fwd-only {n_iters}...")
    torch.cuda.nvtx.range_push("BENCH")
    se.record()
    for _ in range(n_iters):
        out = node.forward(x, tpe, dispatched_indices=dispatched_indices, dispatched_probs=dispatched_probs)
        del out
    ee.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    flush_native_grads()
    cuda_us = se.elapsed_time(ee) / n_iters * 1000
    print(f"\nFwd-only CUDA-event: {cuda_us:.1f} µs/iter (N_recv={N_recv})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--I", type=int, default=1024)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--E", type=int, default=96)
    p.add_argument("--EP", type=int, default=8)
    p.add_argument("--SEQ", type=int, default=16384)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--extract", type=str, default=None)
    p.add_argument("--breakdown", action="store_true")
    a = p.parse_args()
    if a.extract:
        us = gpu_projection_us(a.extract, a.iters)
        print(f"GPU-projection: {us:.1f} µs/iter ({a.iters} iters)")
        if a.breakdown:
            print("\nTop kernels:")
            for name, cnt, us_per in kernel_breakdown(a.extract, a.iters, 25):
                print(f"  {us_per:8.2f} µs/iter  cnt={cnt:5d}  {name[:80]}")
        return
    run_forward_only(a.H, a.I, a.K, a.E, a.EP, a.SEQ, a.warmup, a.iters)


if __name__ == "__main__":
    main()
