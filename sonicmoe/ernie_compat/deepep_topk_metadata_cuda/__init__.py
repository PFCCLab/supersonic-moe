"""JIT-compiled CUDA kernel for DeepEP topk -> SonicMoE metadata (fused).

Single-pass fused kernel: eliminates argsort, all Python dispatch overhead,
and Host-to-Device transfers. Token ordering is stable and aligns with
Paddle's moe_permute (ascending token-ID within each expert segment).

Architecture:
  Kernel 1: compute_expert_offsets (1 thread, serial O(E) prefix sum on device)
  Kernel 2: topk_scatter (warp-ballot progressive cumsum, mirrors moe_permute)
  Kernel 3: naept_prefix_sum (serial scan, N_recv typically <= 32K)
  Kernel 4: fixup_token_major (resolve token-major positions after naept ready)

Total GPU time target: < 50us for N=16K, topk=8, E=8 (vs 1370us Python).
"""

from __future__ import annotations

import torch

from sonicmoe.enums import LIBRARY_NAME
from sonicmoe.jit import cpp_jit


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_topk_metadata_cuda",
    mutates_args={
        "expert_offsets",
        "seg_starts",
        "real_bases",
        "x_gather_idx",
        "s_scatter_idx",
        "s_reverse_scatter_idx",
        "topk_scores",
        "naept",
        "global_block_cumsum",
        "score_src_idx",
    },
)
@cpp_jit()
def deepep_topk_metadata_cuda(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    expert_offsets: torch.Tensor,
    seg_starts: torch.Tensor,
    real_bases: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    topk_scores: torch.Tensor,
    naept: torch.Tensor,
    global_block_cumsum: torch.Tensor,
    score_src_idx: torch.Tensor,
    N_recv: int,
    E: int,
    topk: int,
    TK: int,
    TK_padded: int,
    alignment: int,
    stream: int,
) -> None: ...
