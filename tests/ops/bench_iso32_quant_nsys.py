"""Perf microbench: iso32 vs 1×32 weight blockscaled-FP8 quant kernel.

Measures pure kernel time of `_quantize_weight_3d_triton` in both modes, for the
two reference expert-weight shapes that hit it in production (precompute_weight_fp8_*).
Output: NVTX-bracketed iters that nsys can resolve to per-iter GPU-projection.
"""

from __future__ import annotations

import argparse
import torch

from sonicmoe.quack_utils.blockscaled_fp8_gemm import _quantize_weight_3d_triton


SHAPES = [
    ("w1_E8",  8,  3072, 3072),  # (E, 2I, H)
    ("w2_E8",  8,  3072, 1536),  # (E, H, I)
    ("w1_E32", 32, 3072, 3072),
    ("w2_E32", 32, 3072, 1536),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    nvtx = torch.cuda.nvtx

    for label, E, N, K in SHAPES:
        w = torch.randn(E, N, K, device=device, dtype=torch.bfloat16).contiguous()
        for mode, iso in (("iso32", True), ("1x32", False)):
            for _ in range(args.warmup):
                _quantize_weight_3d_triton(w, isotropic=iso)
            torch.cuda.synchronize()
            nvtx.range_push(f"BENCH_{label}_{mode}")
            for i in range(args.iters):
                nvtx.range_push(f"ITER{i}")
                _quantize_weight_3d_triton(w, isotropic=iso)
                nvtx.range_pop()
            torch.cuda.synchronize()
            nvtx.range_pop()
            print(f"  done {label} {mode}")


if __name__ == "__main__":
    main()
