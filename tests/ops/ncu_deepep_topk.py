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
import sys

_REPO = os.environ.get("SONIC_MOE_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from sonicmoe.ernie_compat.deepep_metadata import _HAS_TOPK_CUDA_KERNEL
from sonicmoe.ernie_compat.deepep_topk_metadata_cuda import deepep_topk_metadata_cuda


CONFIGS = {
    "n16k_top8_e8":   {"N_recv": 16384, "topk": 8, "E": 8,   "block": 128},
    "n16k_top8_e256": {"N_recv": 16384, "topk": 8, "E": 256, "block": 128},
    "n512_top4_e8":   {"N_recv": 512,   "topk": 4, "E": 8,   "block": 128},
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
    args = parser.parse_args()

    if not _HAS_TOPK_CUDA_KERNEL:
        print("ERROR: topk CUDA kernel not compiled.")
        sys.exit(1)

    cfg = CONFIGS[args.config]
    N_recv = cfg["N_recv"]
    topk = cfg["topk"]
    E = cfg["E"]
    block = cfg["block"]
    device = "cuda"

    print(f"Config: {args.config}")
    print(f"  N_recv={N_recv}, topk={topk}, E={E}, block={block}")

    dispatched_indices, dispatched_probs, tpe_dev, TK, TK_padded = \
        _make_data(N_recv, topk, E, block, device)

    # Output buffers (reused across iterations)
    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    seg_starts = torch.empty(E, dtype=torch.int32, device=device)
    real_bases = torch.empty(E, dtype=torch.int32, device=device)
    x_gather_idx = torch.zeros(TK_padded, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK_padded, dtype=torch.int32, device=device)
    s_reverse = torch.empty(TK, dtype=torch.int32, device=device)
    topk_scores = torch.zeros(TK_padded, dtype=torch.float32, device=device)
    naept = torch.empty(N_recv + 1, dtype=torch.int32, device=device)

    num_blocks = (N_recv + 31) // 32
    cumsum_workspace = torch.empty(2 * num_blocks * E + 1, dtype=torch.int32, device=device)

    stream = torch.cuda.current_stream(device).stream_base.raw_stream

    def _launch():
        deepep_topk_metadata_cuda(
            dispatched_indices=dispatched_indices,
            dispatched_probs=dispatched_probs,
            tokens_per_expert=tpe_dev,
            expert_offsets=expert_offsets,
            seg_starts=seg_starts,
            real_bases=real_bases,
            x_gather_idx=x_gather_idx,
            s_scatter_idx=s_scatter_idx,
            s_reverse_scatter_idx=s_reverse,
            topk_scores=topk_scores,
            naept=naept,
            global_block_cumsum=cumsum_workspace,
            N_recv=N_recv,
            E=E,
            topk=topk,
            TK=TK,
            TK_padded=TK_padded,
            alignment=block,
            stream=stream,
        )

    # Warmup
    for i in range(N_WARMUP):
        _launch()
        if i == 0:
            torch.cuda.synchronize()

    # The profiled call (NCU captures this one)
    _launch()
    torch.cuda.synchronize()
    print("  DONE")


if __name__ == "__main__":
    main()
