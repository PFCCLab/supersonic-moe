#!/usr/bin/env python3
"""A/B benchmark: TMA reduce-add vs fused beta=1.0 for wgrad GEMM epilogue.

Hypothesis: The fused beta=1.0 epilogue (D = A@B + 1.0*C, C=D) inflates register
usage from ~50 to ~86 regs/thread, halving SM occupancy.  TMA reduce-add
(add_to_output=True) eliminates the C tensor load entirely — the hardware TMA
unit performs gmem[addr] += result atomically on store.

Usage:
    CUDA_VISIBLE_DEVICES=2 USE_QUACK_GEMM=1 \
      TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
      python tests/ops/bench_wgrad_epilogue.py
"""
import math
import os
import sys
import statistics

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

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    colwise_quantize_and_pack,
    _run_cutlass_blockscaled_gemm_varlen_k_accumulate,
    _run_cutlass_blockscaled_gemm_varlen_k_tma_add,
)

# ── Benchmark shapes (production H=3072, I=1536, K=8) ───────────────────────

SHAPES = [
    # name,     M,    N,    E,  TK_approx
    ("dw1_E8_T8k",  3072, 3072,  8, 65536),
    ("dw2_E8_T8k",  3072, 1536,  8, 65536),
    ("dw1_E32_T8k", 3072, 3072, 32, 65536),
    ("dw2_E8_T16k", 3072, 1536,  8, 131072),
]

N_WARMUP = 5
N_MEASURE = 20
PAD = 128  # cu_seqlens_k aligned to 128


def _align_up(x, align):
    return (x + align - 1) // align * align


def make_inputs(M, N, E, TK_approx, device):
    """Pre-quantize FP8 inputs and build cu_seqlens_k for E experts."""
    # Uniform token distribution, pad each expert's slice to PAD
    tokens_per_expert = _align_up(TK_approx // E, PAD)
    TK = tokens_per_expert * E

    # Build cu_seqlens_k
    boundaries = [tokens_per_expert * i for i in range(E + 1)]
    cu_seqlens_k = torch.tensor(boundaries, dtype=torch.int32, device=device)

    # Random bf16 sources — use paddle.randn so dtype is paddle.bfloat16
    # (required by check_tensor under paddle proxy)
    a_src = paddle.randn([TK, M], dtype="bfloat16")
    b_src = paddle.randn([TK, N], dtype="bfloat16")

    # Pre-quantize
    a_fp8, a_scales = colwise_quantize_and_pack(a_src, logical_rows=M, logical_cols=TK)
    b_fp8, b_scales = colwise_quantize_and_pack(b_src, logical_rows=N, logical_cols=TK)

    return a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k, TK


def _tensor_equal(a, b):
    """Bitwise compare two fp32 tensors (paddle proxy compatible)."""
    return bool((a == b).all())


def run_strategy(fn, a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k,
                 M, N, TK, E, device, accum_init):
    """Run a strategy (warmup + measure), return (median_us, times_us, deterministic, final_accum)."""
    # Warmup
    for _ in range(N_WARMUP):
        acc = accum_init.clone()
        fn(a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k,
           M, N, TK, E, device, acc)
    torch.cuda.synchronize()

    # Determinism check: run twice, compare bitwise
    acc_det1 = accum_init.clone()
    fn(a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k, M, N, TK, E, device, acc_det1)
    torch.cuda.synchronize()
    acc_det2 = accum_init.clone()
    fn(a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k, M, N, TK, E, device, acc_det2)
    torch.cuda.synchronize()
    deterministic = _tensor_equal(acc_det1, acc_det2)

    # Timed runs
    times_us = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(N_MEASURE):
        acc = accum_init.clone()
        start_event.record()
        fn(a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k,
           M, N, TK, E, device, acc)
        end_event.record()
        torch.cuda.synchronize()
        times_us.append(start_event.elapsed_time(end_event) * 1000.0)  # ms -> us

    median_us = statistics.median(times_us)
    stdev_us = statistics.stdev(times_us) if len(times_us) > 1 else 0.0
    return median_us, stdev_us, times_us, deterministic, acc_det1


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Warmup: {N_WARMUP}  Measure: {N_MEASURE}\n")

    for name, M, N, E, TK_approx in SHAPES:
        a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k, TK = make_inputs(
            M, N, E, TK_approx, device,
        )

        # Common accumulator initial state (fp32, use paddle for proxy compat)
        accum_init = 0.01 * paddle.randn([E, M, N], dtype="float32")

        print(f"Shape: {name} (M={M} N={N} E={E} TK={TK})")

        # Strategy A: TMA add
        med_a, std_a, _, det_a, result_a = run_strategy(
            _run_cutlass_blockscaled_gemm_varlen_k_tma_add,
            a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k,
            M, N, TK, E, device, accum_init,
        )

        # Strategy B: Fused beta=1.0
        med_b, std_b, _, det_b, result_b = run_strategy(
            _run_cutlass_blockscaled_gemm_varlen_k_accumulate,
            a_fp8, a_scales, b_fp8, b_scales, cu_seqlens_k,
            M, N, TK, E, device, accum_init,
        )

        # Correctness: compare A vs B
        bitwise = _tensor_equal(result_a, result_b)
        if bitwise:
            correctness = "PASS (bitwise identical)"
        else:
            close = torch.allclose(result_a, result_b, atol=1e-3, rtol=1e-3)
            max_diff = (result_a - result_b).abs().max().item()
            correctness = f"CLOSE (max_diff={max_diff:.2e})" if close else f"FAIL (max_diff={max_diff:.2e})"

        speedup = med_b / med_a if med_a > 0 else float("inf")

        print(f"  Strategy A (TMA add):    {med_a:7.0f} ± {std_a:3.0f} µs   deterministic={'OK' if det_a else 'FAIL'}")
        print(f"  Strategy B (fused beta): {med_b:7.0f} ± {std_b:3.0f} µs   deterministic={'OK' if det_b else 'FAIL'}")
        print(f"  Speedup: {speedup:.2f}x   correctness={correctness}")
        print()


if __name__ == "__main__":
    main()
