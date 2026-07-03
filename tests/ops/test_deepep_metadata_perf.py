#!/usr/bin/env python
"""Test and benchmark deepep_metadata: CUDA kernel vs Python fallback.

Usage:
  python tests/ops/test_deepep_metadata_perf.py           # correctness + timing
  ncu --set full python tests/ops/test_deepep_metadata_perf.py --ncu   # NCU profile
"""

import os
import sys
import time
import argparse

os.environ.setdefault("USE_QUACK_GEMM", "1")

_REPO = os.environ.get("SONIC_MOE_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch

# Import both implementations explicitly
from sonicmoe.ernie_compat.deepep_metadata import (
    _deepep_to_sonic_metadata_python,
    _HAS_CUDA_KERNEL,
    deepep_to_sonic_metadata,
)

if _HAS_CUDA_KERNEL:
    from sonicmoe.ernie_compat.deepep_metadata import _deepep_to_sonic_metadata_cuda


def verify_correctness(tokens_per_expert, T, E, label=""):
    """Compare CUDA vs Python outputs element-by-element."""
    py_out = _deepep_to_sonic_metadata_python(tokens_per_expert, T, E, "cuda", 128)
    if not _HAS_CUDA_KERNEL:
        print(f"  [{label}] CUDA kernel not available, skipping comparison")
        return True

    cu_out = _deepep_to_sonic_metadata_cuda(tokens_per_expert, T, E, "cuda", 128)

    # Unpack: (efo, x_gather, s_scatter, s_rev_scatter, naept, scores, TK, pad_rows)
    names = ["efo", "x_gather_idx", "s_scatter", "s_rev_scatter",
             "naept_offset", "router_scores", "TK_padded", "total_pad_rows"]

    ok = True
    for i, name in enumerate(names):
        pv = py_out[i]
        cv = cu_out[i]
        if isinstance(pv, torch.Tensor):
            if pv.shape != cv.shape:
                print(f"  [{label}] MISMATCH {name}: shape py={pv.shape} cu={cv.shape}")
                ok = False
                continue
            diff_mask = (pv != cv)
            diff_count = int(diff_mask.sum().item())
            if diff_count > 0:
                print(f"  [{label}] MISMATCH {name}: {diff_count} elements differ")
                diff_idx = diff_mask.nonzero(as_tuple=True)[0][:5]
                for idx in diff_idx:
                    print(f"    idx={idx.item()}: py={pv[idx].item()} cu={cv[idx].item()}")
                ok = False
        else:
            if pv != cv:
                print(f"  [{label}] MISMATCH {name}: py={pv} cu={cv}")
                ok = False

    if ok:
        print(f"  [{label}] PASSED (T={T}, E={E}, TK_padded={py_out[6]})")
    return ok


def benchmark(tokens_per_expert, T, E, warmup=20, repeats=200, label=""):
    """Benchmark both implementations with CUDA events."""
    device = "cuda"

    # Python fallback timing
    torch.cuda.synchronize()
    for _ in range(warmup):
        _deepep_to_sonic_metadata_python(tokens_per_expert, T, E, device, 128)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _deepep_to_sonic_metadata_python(tokens_per_expert, T, E, device, 128)
    end.record()
    torch.cuda.synchronize()
    py_ms = start.elapsed_time(end) / repeats

    # CUDA kernel timing
    if _HAS_CUDA_KERNEL:
        torch.cuda.synchronize()
        for _ in range(warmup):
            _deepep_to_sonic_metadata_cuda(tokens_per_expert, T, E, device, 128)
        torch.cuda.synchronize()

        start.record()
        for _ in range(repeats):
            _deepep_to_sonic_metadata_cuda(tokens_per_expert, T, E, device, 128)
        end.record()
        torch.cuda.synchronize()
        cu_ms = start.elapsed_time(end) / repeats
        speedup = py_ms / cu_ms if cu_ms > 0 else float("inf")
        print(f"  [{label}] Python={py_ms*1000:.1f}us  CUDA={cu_ms*1000:.1f}us  speedup={speedup:.1f}x")
    else:
        print(f"  [{label}] Python={py_ms*1000:.1f}us  (CUDA not available)")


def ncu_target_kernel(tokens_per_expert, T, E):
    """Single invocation for NCU profiling."""
    if _HAS_CUDA_KERNEL:
        _deepep_to_sonic_metadata_cuda(tokens_per_expert, T, E, "cuda", 128)
    else:
        print("CUDA kernel not available for NCU profiling")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true", help="NCU mode: single kernel invocation")
    args = parser.parse_args()

    print(f"CUDA kernel available: {_HAS_CUDA_KERNEL}")

    # Test configurations
    configs = [
        # (label, tokens_per_expert)
        ("8exp_uniform", [512] * 8),
        ("8exp_skewed", [2048, 128, 64, 512, 1024, 256, 384, 100]),
        ("8exp_with_zeros", [1024, 0, 512, 0, 256, 0, 128, 0]),
        ("8exp_aligned", [128, 256, 384, 128, 256, 128, 384, 256]),
        ("8exp_large", [4096, 3200, 2800, 4000, 3500, 2900, 3100, 4200]),
        ("64exp_uniform", [256] * 64),
        ("128exp_uniform", [128] * 128),
        ("256exp_mixed", [100 + (i * 7) % 200 for i in range(256)]),
    ]

    if args.ncu:
        # NCU mode: just warm up and run one invocation of the largest config
        label, tpe = configs[-1]  # 256 experts
        T = sum(tpe)
        E = len(tpe)
        # Warm up JIT
        if _HAS_CUDA_KERNEL:
            _deepep_to_sonic_metadata_cuda(tpe, T, E, "cuda", 128)
            torch.cuda.synchronize()
        # Profiled invocation
        ncu_target_kernel(tpe, T, E)
        torch.cuda.synchronize()
        print("NCU target invocation complete")
        return

    # ── Correctness tests ──────────────────────────────────────────────
    print("\n=== Correctness Tests ===")
    all_ok = True
    for label, tpe in configs:
        T = sum(tpe)
        E = len(tpe)
        ok = verify_correctness(tpe, T, E, label)
        all_ok = all_ok and ok

    # Edge cases
    ok = verify_correctness([0] * 8, 0, 8, "all_zeros")
    all_ok = all_ok and ok
    ok = verify_correctness([1], 1, 1, "single_expert_1token")
    all_ok = all_ok and ok
    ok = verify_correctness([128], 128, 1, "single_expert_aligned")
    all_ok = all_ok and ok

    if all_ok:
        print("All correctness tests PASSED\n")
    else:
        print("SOME CORRECTNESS TESTS FAILED\n")
        sys.exit(1)

    # ── Benchmark ──────────────────────────────────────────────────────
    print("=== Benchmark (CUDA events, 200 iters) ===")
    for label, tpe in configs:
        T = sum(tpe)
        E = len(tpe)
        benchmark(tpe, T, E, label=label)

    print("\nDone.")


if __name__ == "__main__":
    main()
