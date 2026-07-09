#!/usr/bin/env python
"""Isolated NCU profiling harness for deepep_topk_metadata CUDA kernels.

Usage:
  ncu --set full --launch-skip 10 --launch-count 2 -o /tmp/ncu_topk \
    python tests/ops/ncu_deepep_topk.py --config n16k_top8_e8

  ncu --set full --launch-skip 10 --launch-count 2 -o /tmp/ncu_topk_e256 \
    python tests/ops/ncu_deepep_topk.py --config n16k_top8_e256

Configs:
  n16k_top8_e8    : N=16384, topk=8, E=8   (typical production)
  n16k_top8_e256  : N=16384, topk=8, E=256 (large-E stress)
  n512_top4_e8    : N=512,   topk=4, E=8   (small batch)

Metrics to watch (from ``--set full``):
  - achieved_occupancy     : should be ~0.5-0.8 for memory-bound kernels
  - memory_throughput      : look for near-peak HBM bandwidth
  - compute_throughput     : usually low for metadata kernels
  - stall_barrier          : high = spin-wait on completion_flag
  - registers_per_thread   : check against occupancy limit
"""

import argparse
import os
import sys

_REPO = os.environ.get("SONIC_MOE_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch

from sonicmoe.ernie_compat.deepep_metadata import (
    _HAS_TOPK_CUDA_KERNEL,
    _HAS_TOPK_CUDA_SCALES_KERNEL,
)
from sonicmoe.ernie_compat.deepep_topk_metadata_cuda import (
    deepep_topk_metadata_cuda,
    deepep_topk_metadata_cuda_with_scales,
    deepep_topk_metadata_cuda_with_scales_rowpack,
    deepep_topk_metadata_cuda_with_scales_scatterpack,
)


CONFIGS = {
    "n16k_top8_e8":   {"N_recv": 16384, "topk": 8, "E": 8,   "block": 128, "cols": 7168},
    "n16k_top8_e256": {"N_recv": 16384, "topk": 8, "E": 256, "block": 128, "cols": 7168},
    "n512_top4_e8":   {"N_recv": 512,   "topk": 4, "E": 8,   "block": 128, "cols": 7168},
    "prod_a35b":      {"N_recv": 116000, "topk": 8, "E": 8,  "block": 128, "cols": 7168},
}

N_WARMUP = 10


def _make_data(N_recv, topk, E, block, device):
    """Generate deterministic mock dispatch data."""
    torch.manual_seed(42)

    # Each token routes to `topk` distinct experts
    avail = torch.arange(E, dtype=torch.int32, device=device)
    avail = avail.unsqueeze(0).expand(N_recv, E)
    # Shuffle per row and take topk
    perm = torch.argsort(torch.rand(N_recv, E, device=device), dim=1)
    dispatched_indices = torch.gather(avail, 1, perm[:, :topk])

    # Normalised random probs
    dispatched_probs = torch.rand(N_recv, topk, device=device, dtype=torch.float32)
    dispatched_probs = dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)

    # Derive tokens_per_expert from dispatched_indices
    flat = dispatched_indices.view(-1).long()
    counts = torch.bincount(flat, minlength=E).int()
    tokens_per_expert = counts  # [E] int32, already on device

    # Compute TK / TK_padded
    tpe_list = tokens_per_expert.tolist()
    TK = sum(tpe_list)
    TK_padded = 0
    for c in tpe_list:
        if c > 0:
            TK_padded += ((c + block - 1) // block) * block

    return dispatched_indices, dispatched_probs, tokens_per_expert, TK, TK_padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="n16k_top8_e8",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--with-scales", action="store_true",
                        help="Profile metadata plus raw-scale gather/ISA-pack in the same C++ launcher.")
    parser.add_argument("--scatter-pack", action="store_true",
                        help="Pack raw scales inside scatter/fixup instead of launching the follow-up pack kernel.")
    parser.add_argument("--row-pack", action="store_true",
                        help="Use row-major-load shared-transpose scale pack kernel.")
    parser.add_argument("--cols", type=int, default=None,
                        help="Activation hidden size for scale packing; defaults to config cols.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of measured launches inside one CUDA-event timing region.")
    args = parser.parse_args()

    if not _HAS_TOPK_CUDA_KERNEL:
        print("ERROR: topk CUDA kernel not compiled.")
        sys.exit(1)
    if args.with_scales and not _HAS_TOPK_CUDA_SCALES_KERNEL:
        print("ERROR: topk CUDA scale-packing kernel not compiled.")
        sys.exit(1)

    cfg = CONFIGS[args.config]
    N_recv = cfg["N_recv"]
    topk = cfg["topk"]
    E = cfg["E"]
    block = cfg["block"]
    cols = args.cols if args.cols is not None else cfg["cols"]
    device = "cuda"

    print(f"Config: {args.config}")
    print(
        f"  N_recv={N_recv}, topk={topk}, E={E}, block={block}, cols={cols}, "
        f"with_scales={args.with_scales}, scatter_pack={args.scatter_pack}, "
        f"row_pack={args.row_pack}"
    )

    dispatched_indices, dispatched_probs, tpe_dev, TK, TK_padded = \
        _make_data(N_recv, topk, E, block, device)
    scale_cols = (cols + 31) // 32
    raw_scales = (
        torch.arange(N_recv * scale_cols, dtype=torch.int32, device=device)
        .reshape(N_recv, scale_cols)
        % 251
    )

    stream_obj = torch.cuda.current_stream(device)
    stream = stream_obj.stream_base.raw_stream if hasattr(stream_obj, "stream_base") else stream_obj.cuda_stream

    def _launch():
        if args.with_scales:
            if args.scatter_pack and args.row_pack:
                raise ValueError("--scatter-pack and --row-pack are mutually exclusive")
            op = (
                deepep_topk_metadata_cuda_with_scales_scatterpack
                if args.scatter_pack
                else deepep_topk_metadata_cuda_with_scales_rowpack
                if args.row_pack
                else deepep_topk_metadata_cuda_with_scales
            )
            return op(
                dispatched_indices, dispatched_probs, tpe_dev,
                N_recv, E, topk, TK, TK_padded, block,
                raw_scales, cols, stream,
            )
        return deepep_topk_metadata_cuda(
            dispatched_indices, dispatched_probs, tpe_dev,
            N_recv, E, topk, TK, TK_padded, block, stream,
        )

    # Warmup
    for i in range(N_WARMUP):
        _launch()
        if i == 0:
            torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    last = None
    for _ in range(args.repeat):
        last = _launch()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / max(args.repeat, 1)
    print(f"  TK={TK}, TK_padded={TK_padded}, avg_gpu_ms={avg_ms:.6f}, repeats={args.repeat}")
    if args.with_scales and last is not None:
        print(f"  packed_scales_shape={tuple(last[-1].shape)}")
    print("  DONE")


if __name__ == "__main__":
    main()
