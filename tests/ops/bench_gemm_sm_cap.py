"""Micro-benchmark: gemm_gated up-proj latency at several SM caps.

Demonstrates the compute/comm-overlap lever: capping SMs frees SMs for a
concurrent communication kernel on another stream, at the cost of a modest
per-GEMM slowdown.  Reports per-call latency for the up-proj gated GEMM at
gemm_num_sms in {None(all), 128, 112, 96, 64}.

Run: USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf with quack on PYTHONPATH.
"""
import os
import torch

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")

from sonicmoe.config import SonicMoEConfig, set_active_config
from sonicmoe.quack_utils.gemm_interface import gemm_gated
from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_and_pack_activation,
    precompute_weight_fp8_for_fused_gated,
)


def _setup(T=8192, H=3072, I=1536, E=8, K=8):
    TK = T * K // E
    total_M = TK * E
    cu = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.02
    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)
    return x_fp8, w_fp8, a_scales, b_scales, cu


def _bench(cap, x_fp8, w_fp8, a_scales, b_scales, cu, iters=50):
    set_active_config(SonicMoEConfig(gemm_num_sms=cap) if cap else None)
    try:
        def run():
            gemm_gated(x_fp8, w_fp8, activation="swiglu", cu_seqlens_m=cu,
                       a_scales=a_scales, b_scales=b_scales)
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1000.0  # us/call
    finally:
        set_active_config(None)


def main():
    x_fp8, w_fp8, a_scales, b_scales, cu = _setup()
    print(f"up-proj gated GEMM (ERNIE shape T=8192,H=3072,I=1536,E=8,K=8), fp8")
    print(f"{'cap':>8} {'us/call':>10}")
    base = None
    for cap in (None, 128, 112, 96, 64):
        us = _bench(cap, x_fp8, w_fp8, a_scales, b_scales, cu)
        if base is None:
            base = us
        label = "all" if cap is None else str(cap)
        print(f"{label:>8} {us:>10.2f}  ({us/base:.2f}x baseline)")


if __name__ == "__main__":
    main()
