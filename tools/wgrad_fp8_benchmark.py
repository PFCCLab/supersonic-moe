#!/usr/bin/env python3
"""FP8 wgrad full-replacement benchmark.

Profiles BF16 vs FP8 weight-gradient computation at multiple shapes,
using CUDA events (not wallclock) for accurate kernel-level timing.
Covers both up-proj and down-proj wgrad paths.

Usage:
    CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python tools/wgrad_fp8_benchmark.py

Output:
    Per-kernel breakdown, total wgrad time for BF16 vs FP8, and the
    breakeven threshold.
"""
import json
import os
import sys
import torch
import math
import gc

os.environ.setdefault("USE_QUACK_GEMM", "1")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    colwise_quantize_and_pack,
    quantize_and_pack_activation,
    _run_cutlass_blockscaled_gemm_varlen_k,
)
from sonicmoe.quack_utils.cute_blockscaled_quant import colwise_quantize_cute
from sonicmoe.functional.utils import enable_quack_gemm

# QuACK GEMM for BF16 wgrad
from quack.gemm_interface import gemm


def cuda_timed(fn, *, warmup=5, repeats=20, sync=True):
    """Time a function using CUDA events. Returns median µs."""
    if sync:
        torch.cuda.synchronize()
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms → µs
    times.sort()
    n = len(times)
    return times[n // 2]  # median


def benchmark_wgrad(T, H, I, E, K, repeats=20):
    """Benchmark wgrad for a single shape config.

    Returns dict with per-kernel timings for both BF16 and FP8 paths.
    """
    TK = T * K  # total tokens after top-k expansion
    device = "cuda"
    dtype = torch.bfloat16

    # Simulate expert frequency offset (uniform, 128-aligned)
    tokens_per_expert = TK // E
    # Round down to nearest 128 for alignment
    tokens_per_expert = (tokens_per_expert // 128) * 128
    expert_freq = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        expert_freq[i] = i * tokens_per_expert
    expert_freq[E] = E * tokens_per_expert
    TK = expert_freq[E].item()  # Adjust TK to be exactly E * aligned_tokens

    # Create gather index (simulates top-K routing: TK→T mapping)
    x_gather_idx = torch.randint(0, T, (TK,), dtype=torch.int32, device=device)
    # Sort within each expert segment for realistic access patterns
    for i in range(E):
        start = expert_freq[i].item()
        end = expert_freq[i + 1].item()
        x_gather_idx[start:end] = x_gather_idx[start:end].sort()[0]

    # Inputs for up-proj wgrad: dz (TK, 2*I), x (T, H), gather_idx
    dz = torch.randn(TK, 2 * I, dtype=dtype, device=device)
    x = torch.randn(T, H, dtype=dtype, device=device)

    # Inputs for down-proj wgrad: dout (TK, H), y1s (TK, I)
    dout = torch.randn(TK, H, dtype=dtype, device=device)
    y1s = torch.randn(TK, I, dtype=dtype, device=device)

    # w1 shape: (2*I, H, E) — up-proj weights
    w1_shape = (2 * I, H, E)
    # w2 shape: (H, I, E) — down-proj weights
    w2_shape = (H, I, E)

    results = {"shape": {"T": T, "H": H, "I": I, "E": E, "K": K, "TK": TK}}
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # =============================================
    # UP-PROJ WGRAD: dw1 = dz.T @ x[gather_idx]
    # =============================================

    # --- BF16 path ---
    dw1_base = torch.empty((E, 2 * I, H), dtype=dtype, device=device)
    dw1 = dw1_base.permute(1, 2, 0)

    def bf16_upproj_wgrad():
        gemm(
            x.T,
            dz,
            out=dw1_base.permute(0, 2, 1),
            cu_seqlens_k=expert_freq,
            A_idx=x_gather_idx,
            batch_idx_permute=None,
            dynamic_scheduler=False,
        )

    t_bf16_up = cuda_timed(bf16_upproj_wgrad, repeats=repeats)
    results["upproj_bf16_wgrad_us"] = t_bf16_up
    del dw1_base, dw1

    # --- FP8 path: colwise quant dz + colwise quant x + FP8 GEMM ---
    # Step 1: dz colwise quant (CuTe DSL)
    def fp8_dz_colwise():
        return colwise_quantize_cute(
            dz, logical_rows=2 * I, logical_cols=TK, isa_pack=True,
        )

    t_dz_col = cuda_timed(fp8_dz_colwise, repeats=repeats)
    results["upproj_fp8_dz_colwise_us"] = t_dz_col

    # Step 2: x colwise quant (Triton, handles gather)
    def fp8_x_colwise():
        return colwise_quantize_and_pack(
            x, logical_rows=H, logical_cols=TK, gather_idx=x_gather_idx,
        )

    t_x_col = cuda_timed(fp8_x_colwise, repeats=repeats)
    results["upproj_fp8_x_colwise_us"] = t_x_col

    # Step 3: FP8 wgrad GEMM
    dz_col_fp8, dz_col_scales = colwise_quantize_cute(
        dz, logical_rows=2 * I, logical_cols=TK, isa_pack=True,
    )
    x_col_fp8, x_col_scales = colwise_quantize_and_pack(
        x, logical_rows=H, logical_cols=TK, gather_idx=x_gather_idx,
    )

    def fp8_upproj_wgrad_gemm():
        return _run_cutlass_blockscaled_gemm_varlen_k(
            dz_col_fp8, dz_col_scales,
            x_col_fp8, x_col_scales,
            expert_freq,
            M=2 * I, N=H, total_K=TK,
            num_experts=E, out_dtype=dtype, device=device,
        )

    t_fp8_gemm_up = cuda_timed(fp8_upproj_wgrad_gemm, repeats=repeats)
    results["upproj_fp8_wgrad_gemm_us"] = t_fp8_gemm_up
    results["upproj_fp8_total_us"] = t_dz_col + t_x_col + t_fp8_gemm_up
    results["upproj_speedup"] = t_bf16_up / (t_dz_col + t_x_col + t_fp8_gemm_up)

    del dz_col_fp8, dz_col_scales, x_col_fp8, x_col_scales
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # =============================================
    # DOWN-PROJ WGRAD: dw2 = dout.T @ y1s
    # =============================================

    # --- BF16 path ---
    dw2_base = torch.empty((E, H, I), dtype=dtype, device=device)
    dw2 = dw2_base.permute(1, 2, 0)

    def bf16_downproj_wgrad():
        gemm(
            dout.T,
            y1s,
            out=dw2.permute(2, 0, 1),
            cu_seqlens_k=expert_freq,
            A_idx=x_gather_idx,
            batch_idx_permute=None,
            dynamic_scheduler=False,
        )

    t_bf16_down = cuda_timed(bf16_downproj_wgrad, repeats=repeats)
    results["downproj_bf16_wgrad_us"] = t_bf16_down
    del dw2_base, dw2

    # --- FP8 path: colwise quant dout + colwise quant y1s + FP8 GEMM ---
    def fp8_dout_colwise():
        return colwise_quantize_cute(
            dout, logical_rows=H, logical_cols=TK, isa_pack=True,
        )

    t_dout_col = cuda_timed(fp8_dout_colwise, repeats=repeats)
    results["downproj_fp8_dout_colwise_us"] = t_dout_col

    def fp8_y1s_colwise():
        return colwise_quantize_cute(
            y1s, logical_rows=I, logical_cols=TK, isa_pack=True,
        )

    t_y1s_col = cuda_timed(fp8_y1s_colwise, repeats=repeats)
    results["downproj_fp8_y1s_colwise_us"] = t_y1s_col

    # FP8 GEMM for down-proj wgrad
    dout_col_fp8, dout_col_scales = colwise_quantize_cute(
        dout, logical_rows=H, logical_cols=TK, isa_pack=True,
    )
    y1s_col_fp8, y1s_col_scales = colwise_quantize_cute(
        y1s, logical_rows=I, logical_cols=TK, isa_pack=True,
    )

    def fp8_downproj_wgrad_gemm():
        return _run_cutlass_blockscaled_gemm_varlen_k(
            dout_col_fp8, dout_col_scales,
            y1s_col_fp8, y1s_col_scales,
            expert_freq,
            M=H, N=I, total_K=TK,
            num_experts=E, out_dtype=dtype, device=device,
        )

    t_fp8_gemm_down = cuda_timed(fp8_downproj_wgrad_gemm, repeats=repeats)
    results["downproj_fp8_wgrad_gemm_us"] = t_fp8_gemm_down
    results["downproj_fp8_total_us"] = t_dout_col + t_y1s_col + t_fp8_gemm_down
    results["downproj_speedup"] = t_bf16_down / (t_dout_col + t_y1s_col + t_fp8_gemm_down)

    del dout_col_fp8, dout_col_scales, y1s_col_fp8, y1s_col_scales
    del dz, x, dout, y1s

    # Combined totals
    bf16_total = t_bf16_up + t_bf16_down
    fp8_total = results["upproj_fp8_total_us"] + results["downproj_fp8_total_us"]
    results["combined_bf16_us"] = bf16_total
    results["combined_fp8_us"] = fp8_total
    results["combined_speedup"] = bf16_total / fp8_total

    # Memory analysis: FP8 wgrad frees dz_bf16 (TK*2I*2 bytes) earlier
    dz_bf16_bytes = TK * 2 * I * 2
    results["dz_bf16_freed_mib"] = dz_bf16_bytes / (1024 * 1024)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    shapes = [
        # (T, H, I, E, K)
        (8192, 3072, 1536, 8, 8),   # reference shape
        (8192, 3072, 2048, 8, 8),   # I=2048 shape
        (4096, 3072, 1536, 8, 8),   # Smaller T
        (8192, 4096, 2048, 8, 8),   # Larger H+I
    ]

    print("=" * 80)
    print("FP8 wgrad Full-Replacement Benchmark")
    print("=" * 80)
    print()

    all_results = []
    for T, H, I, E, K in shapes:
        print(f"Shape: T={T}, H={H}, I={I}, E={E}, K={K}, TK={T*K}")
        print("-" * 60)
        try:
            r = benchmark_wgrad(T, H, I, E, K)
            all_results.append(r)

            print(f"  UP-PROJ wgrad:")
            print(f"    BF16 GEMM:         {r['upproj_bf16_wgrad_us']:>8.1f} µs")
            print(f"    FP8 dz colwise:    {r['upproj_fp8_dz_colwise_us']:>8.1f} µs")
            print(f"    FP8 x  colwise:    {r['upproj_fp8_x_colwise_us']:>8.1f} µs")
            print(f"    FP8 GEMM:          {r['upproj_fp8_wgrad_gemm_us']:>8.1f} µs")
            print(f"    FP8 total:         {r['upproj_fp8_total_us']:>8.1f} µs")
            print(f"    Speedup:           {r['upproj_speedup']:>8.3f}×")

            print(f"  DOWN-PROJ wgrad:")
            print(f"    BF16 GEMM:         {r['downproj_bf16_wgrad_us']:>8.1f} µs")
            print(f"    FP8 dout colwise:  {r['downproj_fp8_dout_colwise_us']:>8.1f} µs")
            print(f"    FP8 y1s colwise:   {r['downproj_fp8_y1s_colwise_us']:>8.1f} µs")
            print(f"    FP8 GEMM:          {r['downproj_fp8_wgrad_gemm_us']:>8.1f} µs")
            print(f"    FP8 total:         {r['downproj_fp8_total_us']:>8.1f} µs")
            print(f"    Speedup:           {r['downproj_speedup']:>8.3f}×")

            print(f"  COMBINED wgrad:")
            print(f"    BF16 total:        {r['combined_bf16_us']:>8.1f} µs")
            print(f"    FP8 total:         {r['combined_fp8_us']:>8.1f} µs")
            print(f"    Speedup:           {r['combined_speedup']:>8.3f}×")
            print(f"    Memory freed:      {r['dz_bf16_freed_mib']:>8.1f} MiB (dz bf16)")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Save results
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "wgrad_fp8_benchmark_legacy.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    with enable_quack_gemm(True):
        main()
