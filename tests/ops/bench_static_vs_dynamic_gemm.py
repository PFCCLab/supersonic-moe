#!/usr/bin/env python3
"""A/B experiment: dynamic vs static compile_key for CuTe GEMM.

Hypothesis: The Session 61 fix (commit 126b532) removed dynamic token dimensions
from compile_key to avoid JIT recompilation on seqlen change. This forces CuTe to
generate a generic kernel via mark_layout_dynamic. For fixed-seqlen scenarios, the
specialized kernel (static compile_key) may be significantly faster.

This script monkey-patches the compile_key construction to add TK back, runs the
same benchmark, and compares per-call GEMM latency.

Usage:
    source .runenv.sh
    # A (dynamic, current): default
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_static_vs_dynamic_gemm.py --mode dynamic
    # B (static, restore TK in key):
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_static_vs_dynamic_gemm.py --mode static
    # Both:
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_static_vs_dynamic_gemm.py --mode both
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import (
    SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches,
)
import sonicmoe.functional as functional
functional._ALIGNMENT_ASSUMED = True

T, H, I, E, K = 8192, 3072, 1536, 8, 8
N_WARMUP = 8
N_ITERS = 12


def patch_static_compile_keys():
    """Monkey-patch compile_key construction to include TK (static compilation)."""
    import sonicmoe.quack_utils.gemm_sm100_fp8_zeromat as zeromat_mod
    import sonicmoe.quack_utils.blockscaled_fp8_gemm as bfp8_mod

    # Patch 1: gemm_gated_zeromat — add TK and A.shape to compile_key
    _orig_gated = zeromat_mod.gemm_gated_zeromat

    def _patched_gated(A, B, PostAct, cu_seqlens_m, A_idx, a_scales, b_scales,
                       activation="swiglu", out_dtype=torch.bfloat16,
                       postact_dtype=torch.bfloat16):
        # Clear cache to force recompile with static shape
        zeromat_mod._zeromat_compile_cache.clear()

        # Temporarily patch _make_cute to NOT mark dynamic
        orig_make_cute = zeromat_mod._make_cute

        def _static_make_cute(tensor, leading_dim):
            from cutlass.cute.runtime import from_dlpack
            _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
            if tensor.dtype in {torch.float8_e4m3fn, _E8M0}:
                storage = tensor.detach().view(torch.uint8)
                ct = from_dlpack(storage, assumed_align=16)
                ct.element_type = zeromat_mod._TORCH_TO_CUTLASS[tensor.dtype]
                return ct  # NO mark_layout_dynamic!
            return from_dlpack(tensor.detach(), assumed_align=16)  # NO dynamic!

        zeromat_mod._make_cute = _static_make_cute
        try:
            result = _orig_gated(A, B, PostAct, cu_seqlens_m, A_idx, a_scales, b_scales,
                                 activation, out_dtype, postact_dtype)
        finally:
            zeromat_mod._make_cute = orig_make_cute
        return result

    zeromat_mod.gemm_gated_zeromat = _patched_gated
    print("  [PATCH] gemm_gated_zeromat: static compile (no mark_layout_dynamic)")


def run_benchmark(mode):
    N_recv = T
    device = "cuda"

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

    torch.manual_seed(42)
    raw_scores = torch.randn(N_recv, E, device=device)
    _, top_experts = raw_scores.topk(K, dim=-1)
    dispatched_indices = top_experts.int()
    dispatched_probs = torch.rand(N_recv, K, device=device) * 0.5 + 0.5
    dispatched_probs = (dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)).float()
    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    x = paddle.randn([N_recv, H], dtype="bfloat16")
    grad_out = paddle.randn([N_recv, H], dtype="bfloat16")

    if mode == "static":
        patch_static_compile_keys()

    # Warmup
    print(f"  Warmup ({N_WARMUP} iters)...")
    for _ in range(N_WARMUP):
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node.forward(xt, tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out)
    flush_native_grads()
    torch.cuda.synchronize()

    # Timed region
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    print(f"  Benchmark ({N_ITERS} iters)...")
    start_ev.record()
    for _ in range(N_ITERS):
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node.forward(xt, tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out)
    end_ev.record()
    torch.cuda.synchronize()
    flush_native_grads()

    cuda_us = start_ev.elapsed_time(end_ev) / N_ITERS * 1000
    return cuda_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dynamic", "static", "both"], default="both")
    args = parser.parse_args()

    print("=" * 70)
    print("A/B Experiment: Dynamic vs Static CuTe compile_key")
    print(f"Shape: T={T} H={H} I={I} E={E} K={K}")
    print("=" * 70)

    if args.mode in ("dynamic", "both"):
        print("\n[A] DYNAMIC compile_key (current, no TK in key):")
        us_dyn = run_benchmark("dynamic")
        print(f"  → CUDA events: {us_dyn:.1f} µs/iter")

    if args.mode in ("static", "both"):
        print("\n[B] STATIC compile (no mark_layout_dynamic on fwd GEMM):")
        us_stat = run_benchmark("static")
        print(f"  → CUDA events: {us_stat:.1f} µs/iter")

    if args.mode == "both":
        delta = us_dyn - us_stat
        pct = delta / us_stat * 100
        print(f"\n{'='*70}")
        print(f"  Dynamic: {us_dyn:.1f} µs/iter")
        print(f"  Static:  {us_stat:.1f} µs/iter")
        print(f"  Delta:   {delta:+.1f} µs ({pct:+.1f}%)")
        print(f"  Conclusion: {'dynamic layout overhead confirmed' if pct > 5 else 'no significant difference'}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
