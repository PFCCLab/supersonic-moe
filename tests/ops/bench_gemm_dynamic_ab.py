#!/usr/bin/env python3
"""A/B experiment: compare per-call GEMM latency with/without mark_layout_dynamic.

Uses the known-working bench_mlpnode_topk_nsys.py infrastructure.
Patches _make_cute to skip mark_layout_dynamic for the "static" run.

Usage:
    source .runenv.sh
    # Dynamic (current behavior):
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_gemm_dynamic_ab.py --mode dynamic
    # Static (skip mark_layout_dynamic):
    CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_gemm_dynamic_ab.py --mode static
"""
import os, sys, math, argparse
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

T, H, I, E, K = 8192, 3072, 1536, 8, 8
N_WARMUP = 8
N_ITERS = 20  # more iters for lower variance


def apply_static_patch():
    """Remove mark_layout_dynamic from all CuTe GEMM tensor creation."""
    import sonicmoe.quack_utils.gemm_sm100_fp8_zeromat as zm
    import sonicmoe.quack_utils.blockscaled_fp8_gemm as bfp8

    # Patch the zeromat _make_cute
    from cutlass.cute.runtime import from_dlpack

    def _static_make_cute(tensor, leading_dim):
        _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
        if tensor.dtype in {torch.float8_e4m3fn, _E8M0}:
            storage = tensor.detach().view(torch.uint8)
            ct = from_dlpack(storage, assumed_align=16)
            ct.element_type = zm._TORCH_TO_CUTLASS[tensor.dtype]
            return ct  # static!
        return from_dlpack(tensor.detach(), assumed_align=16)  # static!

    zm._make_cute = _static_make_cute
    # Clear cache so new static compilation happens
    zm._zeromat_compile_cache.clear()

    # Patch blockscaled_fp8_gemm _make_cute_tensor_dynamic
    if hasattr(bfp8, '_make_cute_tensor_dynamic'):
        def _static_make(tensor, leading_dim):
            from cutlass.cute.runtime import from_dlpack
            _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
            if tensor.dtype in {torch.float8_e4m3fn, _E8M0}:
                storage = tensor.detach().view(torch.uint8)
                ct = from_dlpack(storage, assumed_align=16)
                ct.element_type = bfp8._TORCH_TO_CUTLASS[tensor.dtype]
                return ct
            return from_dlpack(tensor.detach(), assumed_align=16)
        bfp8._make_cute_tensor_dynamic = _static_make
        # Clear all compile caches
        for attr in dir(bfp8):
            obj = getattr(bfp8, attr)
            if hasattr(obj, 'clear') and 'cache' in attr.lower():
                obj.clear()

    print("  [PATCH] All mark_layout_dynamic removed. CuTe will compile with static shapes.")


def run(mode):
    from sonicmoe.enums import ActivationType
    from sonicmoe.ernie_compat import SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True

    if mode == "static":
        apply_static_patch()

    class MockExpert:
        def __init__(self, h, i, seed):
            paddle.seed(seed)
            self.up_gate_proj = type("P", (), {
                "weight": paddle.randn([h, 2*i], dtype="bfloat16") / math.sqrt(h)
            })()
            self.down_proj = type("P", (), {
                "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i)
            })()
            self.up_gate_proj.weight.stop_gradient = False
            self.down_proj.weight.stop_gradient = False

    experts = [MockExpert(H, I, e) for e in range(E)]
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    torch.manual_seed(42)
    raw = torch.randn(T, E, device="cuda")
    _, top_e = raw.topk(K, dim=-1)
    di = top_e.int()
    dp = torch.rand(T, K, device="cuda") * 0.5 + 0.5
    dp = (dp / dp.sum(dim=1, keepdim=True)).float()
    tpe = [int((di == e).sum().item()) for e in range(E)]

    x = paddle.randn([T, H], dtype="bfloat16")
    grad = paddle.randn([T, H], dtype="bfloat16")

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node.forward(xt, tpe, dispatched_indices=di, dispatched_probs=dp)
        out.backward(grad)
    flush_native_grads()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    print(f"  Bench ({N_ITERS})...")
    start_ev.record()
    for _ in range(N_ITERS):
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node.forward(xt, tpe, dispatched_indices=di, dispatched_probs=dp)
        out.backward(grad)
    end_ev.record()
    torch.cuda.synchronize()
    flush_native_grads()

    us = start_ev.elapsed_time(end_ev) / N_ITERS * 1000
    print(f"  → {mode}: {us:.1f} µs/iter")
    return us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dynamic", "static", "both"], default="both")
    args = p.parse_args()

    print("=" * 60)
    print(f"A/B: mark_layout_dynamic overhead (T={T} E={E} I={I})")
    print("=" * 60)

    results = {}
    for m in (["dynamic", "static"] if args.mode == "both" else [args.mode]):
        print(f"\n[{m.upper()}]")
        results[m] = run(m)

    if len(results) == 2:
        d, s = results["dynamic"], results["static"]
        print(f"\n{'='*60}")
        print(f"  Dynamic: {d:.1f} µs/iter")
        print(f"  Static:  {s:.1f} µs/iter")
        print(f"  Delta:   {d-s:+.1f} µs ({(d-s)/s*100:+.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
