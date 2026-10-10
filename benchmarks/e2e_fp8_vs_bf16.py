#!/usr/bin/env python3
"""End-to-end FP8 vs BF16 benchmark for SonicMoE.

Usage:
    # Standard benchmark (aligned segments, production-like shape)
    CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
        SONIC_MOE_FP8_SAVE_Z_FP8=1 \
        python benchmarks/e2e_fp8_vs_bf16.py

    # With zero-sync mode (production)
    SONIC_MOE_FP8_ASSUME_ALIGNED=1 ... python benchmarks/e2e_fp8_vs_bf16.py

    # With nsys profiling
    nsys profile -o fp8_vs_bf16 --force-overwrite true \
        python benchmarks/e2e_fp8_vs_bf16.py
"""
import os
import time

import torch

from sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm, get_default_fp8_protocol
from sonicmoe.enums import ActivationType


def benchmark_moe(
    moe: MoE,
    x: torch.Tensor,
    *,
    fp8: bool,
    warmup: int = 5,
    repeat: int = 20,
    label: str = "",
) -> dict:
    """Benchmark forward + backward, return timing dict."""
    protocol = get_default_fp8_protocol() if fp8 else None

    # Warmup
    for _ in range(warmup):
        x_in = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            out, _aux = moe(x_in, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
        out.sum().backward()
        torch.cuda.synchronize()

    # Timed runs
    fwd_times, bwd_times, total_times = [], [], []
    for _ in range(repeat):
        x_in = x.detach().clone().requires_grad_(True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with enable_quack_gemm(True):
            out, _aux = moe(x_in, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out.sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    # Median (robust to outliers)
    fwd_times.sort()
    bwd_times.sort()
    total_times.sort()
    mid = len(fwd_times) // 2

    result = {
        "label": label,
        "fwd_ms": fwd_times[mid],
        "bwd_ms": bwd_times[mid],
        "total_ms": total_times[mid],
        "fwd_min": fwd_times[0],
        "bwd_min": bwd_times[0],
        "total_min": total_times[0],
    }

    # Memory
    torch.cuda.reset_peak_memory_stats()
    x_in = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        out, _aux = moe(x_in, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
    out.sum().backward()
    torch.cuda.synchronize()
    result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    return result


def main():
    shapes = [
        # (num_experts, top_k, hidden, intermediate, tokens, label)
        (8, 1, 1024, 512, 1024, "small"),
        (8, 1, 4096, 1024, 4096, "medium"),
        (128, 8, 4096, 1024, 4096, "production"),
    ]

    print("=" * 80)
    print("SonicMoE E2E Benchmark: FP8 Blockscaled vs BF16")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"FP8 mode: {os.getenv('SONIC_MOE_FP8_MODE', 'off')}")
    print(f"Save z FP8: {os.getenv('SONIC_MOE_FP8_SAVE_Z_FP8', '0')}")
    print(f"Assume aligned: {os.getenv('SONIC_MOE_FP8_ASSUME_ALIGNED', '0')}")
    print()

    for num_experts, top_k, hidden, intermediate, tokens, label in shapes:
        print(f"--- Shape: {label} (E={num_experts}, K={top_k}, H={hidden}, I={intermediate}, T={tokens}) ---")

        moe = MoE(
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

        torch.manual_seed(42)
        x = (0.02 * torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)).detach()

        # BF16 baseline
        bf16_result = benchmark_moe(moe, x, fp8=False, label=f"BF16-{label}")

        # FP8
        fp8_result = benchmark_moe(moe, x, fp8=True, label=f"FP8-{label}")

        # Report
        speedup = bf16_result["total_ms"] / fp8_result["total_ms"]
        mem_saving = 1.0 - fp8_result["peak_mem_mb"] / bf16_result["peak_mem_mb"]

        print(f"  {'':20s} {'Forward':>10s} {'Backward':>10s} {'Total':>10s} {'Peak Mem':>10s}")
        print(f"  {'BF16':20s} {bf16_result['fwd_ms']:>9.2f}ms {bf16_result['bwd_ms']:>9.2f}ms {bf16_result['total_ms']:>9.2f}ms {bf16_result['peak_mem_mb']:>8.1f}MB")
        print(f"  {'FP8 blockscaled':20s} {fp8_result['fwd_ms']:>9.2f}ms {fp8_result['bwd_ms']:>9.2f}ms {fp8_result['total_ms']:>9.2f}ms {fp8_result['peak_mem_mb']:>8.1f}MB")
        print(f"  {'Speedup':20s} {bf16_result['fwd_ms']/fp8_result['fwd_ms']:>9.2f}x  {bf16_result['bwd_ms']/fp8_result['bwd_ms']:>9.2f}x  {speedup:>9.2f}x  {mem_saving:>8.1%} saved")
        print()

        del moe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
