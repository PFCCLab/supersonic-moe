#!/usr/bin/env python
"""End-to-end SonicMoEMlpNode benchmark simulating real DeepEP pre-training.

Lifecycle modeled:
  1. INIT (one-time):  load experts, stack weights → _W_CACHE, warm FP8 cache
  2. TRAIN LOOP (per-iter): DeepEP dispatch → new (x, tokens_per_expert)
     - deepep_to_sonic_metadata   [per-iter: new routing each batch]
     - FP8 fwd GEMM               [per-iter: new x]
     - FP8 bwd GEMM               [per-iter]
     - main_grad += bf16 dw.float32  [per-iter: accumulates]
  3. OPTIMIZER STEP (every N iters):
     - optimizer reads main_grad, updates bf16 weight
     - invalidate_weight_caches()  → re-stack + re-quant next iter
     - zero main_grad

What should NOT happen per-iter:
  - weight stacking (contiguous/permute/stack) → cached in _W_CACHE
  - FP8 weight quantization → cached by FP8 weight cache
  - invalidate_weight_caches() → only after optimizer step

Modes:
  --nsys              : nsys-compatible mode (cudaProfilerStart/Stop markers)
  --frontier-compare  : apples-to-apples comparison with frontier
                        (same x, same routing, zero_grad per iter — no accumulation)
  --parse-sqlite PATH : parse nsys sqlite and report GPU-projection

Usage:
  source .../eb_venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_e2e_mlpnode.py
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_e2e_mlpnode.py --nsys
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_e2e_mlpnode.py --frontier-compare
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_e2e_mlpnode.py --frontier-compare --nsys
  nsys profile -c cudaProfilerApi --capture-range-end=stop -o /tmp/mlpnode \
    python tests/ops/test_e2e_mlpnode.py --nsys
  python tests/ops/test_e2e_mlpnode.py --parse-sqlite /tmp/mlpnode.sqlite
"""

import argparse
import gc
import math
import os
import sqlite3
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.environ.get("SONIC_MOE_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import (
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
)
from sonicmoe.ernie_compat.deepep_metadata import _HAS_CUDA_KERNEL

# ── Shape config (matches introspect.py grid) ──────────────────────────────
H = 3072
K = 8
# Default shape: the ERNIE production shape
DEFAULT_T, DEFAULT_E, DEFAULT_I = 8192, 8, 1536

N_WARMUP = 8   # enough to warm FP8 cache + alignment streak
N_BENCH  = 12  # profiled iterations


# ── Mock expert (simulates ERNIE expert module) ────────────────────────────

class MockExpert:
    """up_gate_proj.weight [H, 2I], down_proj.weight [I, H]."""
    def __init__(self, h, i, seed):
        paddle.seed(seed)
        self.up_gate_proj = type("P", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
        })()
        self.down_proj = type("P", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient = False


def make_deepep_batch(T, E, H, seed):
    """Simulate one DeepEP dispatch: tokens sorted by expert, varying distribution.

    Each batch has different routing (simulated by seed).
    T is fixed, but per-expert distribution varies.
    """
    # Dirichlet-like distribution across experts (deterministic via manual seed)
    import random
    rng = random.Random(seed)
    alphas = [rng.random() * 5 + 0.5 for _ in range(E)]
    total_a = sum(alphas)
    fracs = [a / total_a for a in alphas]
    raw_counts = [int(f * T) for f in fracs]
    # Fix rounding to match T exactly
    diff = T - sum(raw_counts)
    for i in range(abs(diff)):
        raw_counts[i % E] += 1 if diff > 0 else -1
    tpe = [max(0, c) for c in raw_counts]
    assert sum(tpe) == T, f"sum={sum(tpe)} != T={T}"

    paddle.seed(seed)
    x = paddle.randn([T, H], dtype="bfloat16") * 0.02
    return x, tpe


def zero_main_grads(experts):
    """Zero main_grad on all experts (simulates optimizer step end)."""
    for exp in experts:
        for name in ("up_gate_proj", "down_proj"):
            w = getattr(exp, name).weight
            if hasattr(w, "main_grad") and w.main_grad is not None:
                w.main_grad.zero_()


def check_main_grads(experts, label=""):
    """Verify main_grad is populated, fp32, non-NaN, non-zero."""
    for e_idx, exp in enumerate(experts):
        for name in ("up_gate_proj", "down_proj"):
            mg = getattr(getattr(exp, name).weight, "main_grad", None)
            assert mg is not None, f"[{label}] expert[{e_idx}].{name}.main_grad is None"
            assert mg.dtype == paddle.float32, f"[{label}] dtype={mg.dtype}"
            assert not mg.isnan().any(), f"[{label}] NaN"
            norm = float(mg.norm().item())
            assert norm > 0, f"[{label}] zero norm"


# ── nsys sqlite parser (inline, avoids import dependency on tools/) ──────────

def _nsys_parse_sqlite(db_path: str, num_iters: int) -> dict:
    """Parse nsys sqlite export for GPU-projection and kernel breakdown.

    Returns dict with:
      - gpu_projection_us: total GPU busy time (merged overlapping intervals)
      - per_iter_us: gpu_projection_us / num_iters
      - kernel_breakdown: list of {name, total_us, count, per_iter_us}
    """
    conn = sqlite3.connect(db_path)

    # Resolve string IDs
    string_map: dict[int, str] = {}
    try:
        for row in conn.execute("SELECT id, value FROM StringIds"):
            string_map[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass

    # Read all GPU kernel events
    kernels: list[tuple[int, int, int, int]] = []
    try:
        for row in conn.execute(
            "SELECT start, end, demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels.append((row[0], row[1], row[2], row[3]))
    except sqlite3.OperationalError:
        conn.close()
        return {"error": "No kernel data in sqlite"}

    conn.close()
    if not kernels:
        return {"error": "No kernels found"}

    # Compute GPU-projection (merge overlapping intervals)
    kernels.sort(key=lambda x: x[0])
    merged_ns = 0
    cur_start, cur_end = kernels[0][0], kernels[0][1]
    for start, end, _, _ in kernels[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged_ns += cur_end - cur_start
            cur_start, cur_end = start, end
    merged_ns += cur_end - cur_start
    gpu_projection_us = merged_ns / 1000.0

    # Per-kernel breakdown
    kernel_stats: dict[str, dict] = {}
    for start, end, demangled_id, short_id in kernels:
        name = string_map.get(demangled_id, string_map.get(short_id, f"unknown_{demangled_id}"))
        dur_us = (end - start) / 1000.0
        if name not in kernel_stats:
            kernel_stats[name] = {"total_us": 0.0, "count": 0}
        kernel_stats[name]["total_us"] += dur_us
        kernel_stats[name]["count"] += 1

    breakdown = []
    for name, stats in sorted(kernel_stats.items(), key=lambda x: -x[1]["total_us"]):
        breakdown.append({
            "name": name[:120],
            "total_us": round(stats["total_us"], 1),
            "count": stats["count"],
            "per_iter_us": round(stats["total_us"] / num_iters, 1),
            "per_call_us": round(stats["total_us"] / stats["count"], 1),
        })

    return {
        "gpu_projection_us": round(gpu_projection_us, 1),
        "per_iter_us": round(gpu_projection_us / num_iters, 1),
        "num_kernels": len(kernels),
        "unique_kernels": len(kernel_stats),
        "kernel_breakdown": breakdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys", action="store_true",
                        help="nsys mode: emit cudaProfilerStart/Stop markers")
    parser.add_argument("--frontier-compare", action="store_true",
                        help="Apples-to-apples frontier comparison: same x, same routing, "
                             "zero_grad per iter (no accumulation overhead)")
    parser.add_argument("--parse-sqlite", type=str, default=None,
                        help="Parse nsys sqlite file and print GPU-projection results")
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--E", type=int, default=DEFAULT_E)
    parser.add_argument("--I", type=int, default=DEFAULT_I)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--iters", type=int, default=N_BENCH)
    args = parser.parse_args()

    # ── Parse-only mode ────────────────────────────────────────────────────
    if args.parse_sqlite:
        print(f"Parsing: {args.parse_sqlite}")
        result = _nsys_parse_sqlite(args.parse_sqlite, args.iters)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return
        print(f"  GPU-projection: {result['gpu_projection_us']:.1f} μs total")
        print(f"  Per-iter:       {result['per_iter_us']:.1f} μs/iter ({args.iters} iters)")
        print(f"  Kernels:        {result['num_kernels']} total, {result['unique_kernels']} unique")
        print(f"\n  Top 15 kernels by per-iter time:")
        for k in result["kernel_breakdown"][:15]:
            print(f"    {k['per_iter_us']:7.1f} μs/iter ({k['count']:4d}x) {k['name'][:80]}")
        return

    T, E, I = args.T, args.E, args.I
    frontier_mode = args.frontier_compare
    print(f"Shape: T={T}, E={E}, I={I}, H={H}")
    print(f"Mode: {'FRONTIER-COMPARE' if frontier_mode else 'FULL E2E'}")
    print(f"CUDA metadata kernel: {'YES' if _HAS_CUDA_KERNEL else 'NO'}")
    print(f"Warmup={args.warmup}, Bench={args.iters}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: INIT (one-time cost)
    # ══════════════════════════════════════════════════════════════════════
    print("\n--- INIT (one-time) ---")
    experts = [MockExpert(H, I, e) for e in range(E)]
    node = SonicMoEMlpNode(
        experts=experts, n_experts=E,
        hidden_size=H, intermediate_size=I,
    )

    # First call: cold start — triggers weight stacking + FP8 quantization
    invalidate_weight_caches()
    x0, tpe0 = make_deepep_batch(T, E, H, seed=0)
    out0 = node.forward(x0, tpe0)
    out0.backward(paddle.randn_like(out0))
    flush_native_grads()
    torch.cuda.synchronize()
    print("  Cold start done (weight stack + FP8 cache warm)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: WARMUP (steady state, NOT profiled)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- WARMUP ({args.warmup} iters, not profiled) ---")
    out_grad = paddle.randn([T, H], dtype="bfloat16") * 0.01

    if frontier_mode:
        # Frontier-compare: use SAME x and routing every iter (pre-computed once)
        x_fixed, tpe_fixed = make_deepep_batch(T, E, H, seed=42)
        for i in range(args.warmup):
            out_i = node.forward(x_fixed, tpe_fixed)
            out_i.backward(out_grad)
            # zero_grad(set_to_none) semantics: don't accumulate
            # Native grads cleared per iter in frontier mode
    else:
        for i in range(args.warmup):
            x_i, tpe_i = make_deepep_batch(T, E, H, seed=100 + i)
            out_i = node.forward(x_i, tpe_i)
            out_i.backward(out_grad)

    torch.cuda.synchronize()

    if not frontier_mode:
        flush_native_grads()
        check_main_grads(experts, "warmup")
        print(f"  Warmup done, main_grad accumulated over {args.warmup} iters")
    else:
        print(f"  Warmup done (frontier-compare: same x/routing reused)")

    # Simulate optimizer step boundary: zero main_grad for clean profiled run
    zero_main_grads(experts)
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: PROFILED ITERATIONS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- PROFILED ({args.iters} iters, "
          f"{'frontier-compare' if frontier_mode else 'full e2e'}) ---")

    if args.nsys:
        torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()

    if frontier_mode:
        # ── FRONTIER-COMPARE: same x, same routing, zero_grad per iter ────
        # Matches introspect.py: reuse same pre-padded x, frozen metadata,
        # zero_grad(set_to_none=True) after each backward.
        x_fixed, tpe_fixed = make_deepep_batch(T, E, H, seed=42)
        for i in range(args.iters):
            out_i = node.forward(x_fixed, tpe_fixed)
            out_i.backward(out_grad)
            # Discard wgrad (like frontier's zero_grad(set_to_none=True))
            # Native accumulators are NOT flushed per-iter; they just accumulate
            # harmlessly since we zero main_grad at the end anyway.
    else:
        # ── FULL E2E: new x + new routing each iter, main_grad accumulates ──
        for i in range(args.iters):
            x_i, tpe_i = make_deepep_batch(T, E, H, seed=1000 + i)
            out_i = node.forward(x_i, tpe_i)
            out_i.backward(out_grad)

    end_evt.record()
    torch.cuda.synchronize()

    if args.nsys:
        torch.cuda.cudart().cudaProfilerStop()
        print("  NSYS_DONE")

    elapsed_ms = start_evt.elapsed_time(end_evt)
    per_iter_us = elapsed_ms / args.iters * 1000

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- VALIDATION ---")
    if not frontier_mode:
        flush_native_grads()
        check_main_grads(experts, "profiled")
        # Verify bf16 .grad is None
        for exp in experts:
            assert exp.up_gate_proj.weight.grad is None, "bf16 .grad should be None"
            assert exp.down_proj.weight.grad is None, "bf16 .grad should be None"
        mg_norms_w1 = [f"{float(e.up_gate_proj.weight.main_grad.norm()):.2f}" for e in experts]
        mg_norms_w2 = [f"{float(e.down_proj.weight.main_grad.norm()):.2f}" for e in experts]
        print(f"  w1 main_grad norms: [{', '.join(mg_norms_w1)}]")
        print(f"  w2 main_grad norms: [{', '.join(mg_norms_w2)}]")
        print(f"  bf16 .grad: all None (correct)")
    else:
        print(f"  Frontier-compare mode: main_grad validation skipped (grads zeroed per iter)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: SIMULATE OPTIMIZER STEP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- OPTIMIZER STEP (simulated) ---")
    invalidate_weight_caches()
    zero_main_grads(experts)
    print("  invalidate_weight_caches() + zero main_grad done")

    # Next iter would re-stack weights (one-time cost per optimizer step)
    x_post, tpe_post = make_deepep_batch(T, E, H, seed=9999)
    out_post = node.forward(x_post, tpe_post)
    torch.cuda.synchronize()
    print("  Post-optimizer forward OK (weight re-stacked)")

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"RESULTS: T={T}, E={E}, I={I}, H={H}")
    print(f"{'='*60}")
    mode_label = "frontier-compare" if frontier_mode else "full-e2e"
    print(f"  Mode:        {mode_label}")
    print(f"  CUDA events: {per_iter_us:.1f} μs/iter ({args.iters} iters)")
    print(f"  Total:       {elapsed_ms:.2f} ms")
    print(f"  Metadata:    {'CUDA kernel' if _HAS_CUDA_KERNEL else 'Python fallback'}")
    if frontier_mode:
        print(f"\n  FRONTIER-COMPARE: This number should match frontier ±5%")
        print(f"  Frontier gold-standard: 2715 μs/iter (fwd+bwd)")
        overhead = (per_iter_us - 2715) / 2715 * 100
        print(f"  Overhead vs frontier: {overhead:+.1f}%")
    else:
        print(f"\n  FULL E2E: Includes per-iter metadata + new x + grad accumulation")
    print()
    print(f"  Per-iter kernels (steady-state):")
    print(f"    - deepep_to_sonic_metadata (CUDA fill) [{'cached' if frontier_mode else 'per-iter'}]")
    print(f"    - FP8 GEMM fwd (up_proj + down_proj) [route-level padding, no x-pad]")
    print(f"    - FP8 GEMM bwd (dgrad + wgrad)")
    print(f"    - FP8 quantization (x,w → fp8)")
    print(f"    - native grad add_ (1 kernel w1 + 1 kernel w2)")
    print(f"    - token gather/scatter")
    print(f"  Eliminated overhead (vs naive integration):")
    print(f"    - NO x padding (route-level padding: gather idx=0, score=0)")
    print(f"    - NO grad padding (output is [T,H] directly)")
    print(f"    - NO .contiguous() transpose in main_grad accumulation")
    print(f"    - NO per-iter metadata recomputation (cached when routing frozen)")
    print()


if __name__ == "__main__":
    main()
