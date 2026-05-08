#!/usr/bin/env python3
"""Session 62 FP8 frontier comprehensive benchmark.

Collects CUDA-event timing (immediate) and optional nsys GPU-projection
(gold standard) for key shapes from the Session 53 README baseline.

Methodology (per env.md):
  - GPU must be idle (nvidia-smi util=0%)
  - Each shape in subprocess isolation (avoids JIT cache contamination)
  - 12+ measurement iters after 8 warmup iters
  - CUDA-event timing for immediate comparison
  - nsys --resolve-symbols=false to avoid download hang

Usage:
    source .runenv.sh
    # CUDA-event only (fast, ~2 min):
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_frontier_perf.py

    # With nsys GPU-projection (gold, ~10 min per shape):
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_frontier_perf.py --nsys

    # Specific shapes:
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_frontier_perf.py \
        --shapes 8192,8,1536 32768,8,3072
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Session 53 baseline from README.md (nsys GPU-projection, Target GPU)
BASELINE = {
    # (T, E, I): (bf16_us, fp8_us, speedup)
    (8192, 8, 1536): (3644, 2715, 1.34),
    (8192, 8, 3072): (8110, 4774, 1.70),
    (8192, 32, 1536): (3844, 2922, 1.32),
    (8192, 32, 3072): (8124, 5318, 1.53),
    (8192, 128, 1536): (5009, 3897, 1.29),
    (8192, 128, 3072): (10839, 7267, 1.49),
    (16384, 8, 1536): (7953, 5227, 1.52),
    (16384, 8, 3072): (16172, 10065, 1.61),
    (32768, 8, 1536): (16287, 10652, 1.53),
    (32768, 8, 3072): (33278, 20010, 1.66),
    (32768, 32, 3072): (33504, 19761, 1.70),
    (32768, 128, 3072): (35627, 22026, 1.62),
}

# Default 6-shape subset matching the most important baselines
DEFAULT_SHAPES = [
    (8192, 8, 1536),
    (8192, 8, 3072),
    (32768, 8, 1536),
    (32768, 8, 3072),
    (32768, 32, 3072),
    (32768, 128, 3072),
]

H = 3072
K = 8
N_WARMUP = 8
N_ITERS = 12


def benchmark_one_shape(T, E, I, topk=K, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Run FP8 topk MlpNode benchmark (fwd+bwd), return CUDA-event µs/iter."""
    import paddle
    paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)
    import torch
    from sonicmoe.enums import ActivationType
    from sonicmoe.ernie_compat import (
        SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches,
    )
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True

    device = "cuda"
    N_recv = T

    class MockExpert:
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

    experts = [MockExpert(H, I, e) for e in range(E)]
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Build deterministic topk dispatch
    torch.manual_seed(42)
    raw_scores = torch.randn(N_recv, E, device=device)
    _, top_experts = raw_scores.topk(topk, dim=-1)
    dispatched_indices = top_experts.int()
    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = (dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)).float()
    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    x = paddle.randn([N_recv, H], dtype="bfloat16")
    grad_out = paddle.randn([N_recv, H], dtype="bfloat16")

    # Warmup
    for _ in range(n_warmup):
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node.forward(xt, tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out)
    flush_native_grads()
    torch.cuda.synchronize()

    # Memory snapshot
    MiB = 1 << 20
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / MiB

    # Timed region
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(n_iters):
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node.forward(xt, tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out)
    end_ev.record()
    torch.cuda.synchronize()
    flush_native_grads()

    peak_mem = torch.cuda.max_memory_allocated() / MiB
    cuda_us = start_ev.elapsed_time(end_ev) / n_iters * 1000

    return {"cuda_us": cuda_us, "mem_peak_mib": peak_mem, "mem_before_mib": mem_before}


def run_nsys_one_shape(T, E, I, topk=K, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Run nsys GPU-projection for one shape via bench_mlpnode_topk_nsys.py."""
    output_dir = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/nsys"
    os.makedirs(output_dir, exist_ok=True)
    nsys_base = f"{output_dir}/s62_T{T}_E{E}_I{I}"

    # nsys profile with --resolve-symbols=false (env.md lesson: avoid hang)
    cmd = (
        f"nsys profile --trace=cuda,nvtx --sample=none --backtrace=none "
        f"--resolve-symbols=false --export=sqlite "
        f"--output={nsys_base} -f true "
        f"python tests/ops/bench_mlpnode_topk_nsys.py "
        f"--T {T} --E {E} --I {I} --topk {topk} --warmup {n_warmup} --iters {n_iters}"
    )
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"

    print(f"  nsys: {cmd}")
    result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [WARN] nsys failed: {result.stderr[-200:]}")
        return None

    # Extract GPU-projection
    sqlite_path = f"{nsys_base}.sqlite"
    if not os.path.exists(sqlite_path):
        print(f"  [WARN] sqlite not found: {sqlite_path}")
        return None

    sys.path.insert(0, os.path.join(_REPO, "tests", "ops"))
    from bench_mlpnode_topk_nsys import gpu_projection_us
    us = gpu_projection_us(sqlite_path, n_iters)
    return us


def main():
    parser = argparse.ArgumentParser(description="Session 62 FP8 frontier benchmark")
    parser.add_argument("--shapes", nargs="+", default=None,
                        help="Shapes as T,E,I (e.g. 8192,8,1536)")
    parser.add_argument("--nsys", action="store_true",
                        help="Also run nsys GPU-projection (slow but gold standard)")
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    args = parser.parse_args()

    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]

    print("=" * 90)
    print("SonicMoE Session 62 — FP8 Frontier Performance Benchmark")
    print("=" * 90)
    print(f"Methodology: CUDA events, {args.warmup} warmup + {args.iters} measured iters")
    if args.nsys:
        print("  + nsys GPU-projection (--resolve-symbols=false)")
    print()

    results = []
    for T, E, I in shapes:
        print(f"  T={T:>5d} E={E:>3d} I={I:>4d} K={K} H={H} ... ", end="", flush=True)
        r = benchmark_one_shape(T, E, I, K, args.warmup, args.iters)
        cuda_us = r["cuda_us"]

        nsys_us = None
        if args.nsys:
            print(f"cuda={cuda_us:.0f}µs, running nsys ... ", end="", flush=True)
            nsys_us = run_nsys_one_shape(T, E, I, K, args.warmup, args.iters)

        # Compare with baseline
        baseline = BASELINE.get((T, E, I))
        if baseline:
            bl_bf16, bl_fp8, bl_speed = baseline
            measure_us = nsys_us if nsys_us else cuda_us
            delta_pct = (measure_us - bl_fp8) / bl_fp8 * 100
            print(f"FP8={cuda_us:.0f}µs", end="")
            if nsys_us:
                print(f" (nsys={nsys_us:.0f}µs)", end="")
            sign = "+" if delta_pct >= 0 else ""
            print(f" | baseline FP8={bl_fp8}µs BF16={bl_bf16}µs "
                  f"| delta={sign}{delta_pct:.1f}% | peak={r['mem_peak_mib']:.0f}MiB")
        else:
            print(f"FP8={cuda_us:.0f}µs (no baseline) | peak={r['mem_peak_mib']:.0f}MiB")

        results.append({
            "shape": f"T{T}_E{E}_I{I}_K{K}",
            "T": T, "E": E, "I": I, "K": K, "H": H,
            "cuda_us": round(cuda_us, 1),
            "nsys_us": round(nsys_us, 1) if nsys_us else None,
            "mem_peak_mib": round(r["mem_peak_mib"], 1),
            "baseline_bf16_us": baseline[0] if baseline else None,
            "baseline_fp8_us": baseline[1] if baseline else None,
            "baseline_speedup": baseline[2] if baseline else None,
        })

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Shape':>25s}  {'FP8 µs':>8s}  {'BL FP8':>8s}  {'BL BF16':>8s}  "
          f"{'Δ%':>7s}  {'BL Spd':>7s}  {'Peak MiB':>8s}")
    print("-" * 90)
    for r in results:
        us = r["nsys_us"] or r["cuda_us"]
        bl_fp8 = r["baseline_fp8_us"]
        bl_bf16 = r["baseline_bf16_us"]
        bl_spd = r["baseline_speedup"]
        if bl_fp8:
            delta = (us - bl_fp8) / bl_fp8 * 100
            sign = "+" if delta >= 0 else ""
            print(f"{r['shape']:>25s}  {us:>8.0f}  {bl_fp8:>8d}  {bl_bf16:>8d}  "
                  f"{sign}{delta:>6.1f}%  {bl_spd:>6.2f}×  {r['mem_peak_mib']:>8.0f}")
        else:
            print(f"{r['shape']:>25s}  {us:>8.0f}  {'N/A':>8s}  {'N/A':>8s}  "
                  f"{'N/A':>7s}  {'N/A':>7s}  {r['mem_peak_mib']:>8.0f}")
    print("=" * 90)

    # Save JSON
    out_path = os.path.join(_REPO, "reports", "perf_session62", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"shapes": results, "metadata": {
            "warmup": args.warmup, "iters": args.iters,
            "nsys": args.nsys, "H": H, "K": K,
        }}, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
