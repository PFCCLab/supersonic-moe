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

from ...enums import LIBRARY_NAME
from ...jit import cpp_jit


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_topk_metadata_cuda",
    mutates_args=(),
)
@cpp_jit()
def deepep_topk_metadata_cuda(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    N_recv: int,
    E: int,
    topk: int,
    TK: int,
    TK_padded: int,
    alignment: int,
    stream: int,
) -> list[torch.Tensor]: ...


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_topk_metadata_cuda_with_scales",
    mutates_args=(),
)
@cpp_jit()
def deepep_topk_metadata_cuda_with_scales(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    N_recv: int,
    E: int,
    topk: int,
    TK: int,
    TK_padded: int,
    alignment: int,
    raw_scales: torch.Tensor,
    cols: int,
    stream: int,
) -> list[torch.Tensor]: ...


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_topk_metadata_cuda_with_scales_and_gated_outputs",
    mutates_args=(),
)
@cpp_jit()
def deepep_topk_metadata_cuda_with_scales_and_gated_outputs(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    N_recv: int,
    E: int,
    topk: int,
    TK: int,
    TK_padded: int,
    alignment: int,
    raw_scales: torch.Tensor,
    cols: int,
    gated_output_prototype: torch.Tensor,
    gated_n: int,
    gated_preact_bf16: bool,
    gated_allocate_z_scale: bool,
    stream: int,
) -> list[torch.Tensor]: ...


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_topk_metadata_cuda_with_scales_scatterpack",
    mutates_args=(),
)
@cpp_jit()
def deepep_topk_metadata_cuda_with_scales_scatterpack(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    N_recv: int,
    E: int,
    topk: int,
    TK: int,
    TK_padded: int,
    alignment: int,
    raw_scales: torch.Tensor,
    cols: int,
    stream: int,
) -> list[torch.Tensor]: ...


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_topk_metadata_cuda_with_scales_rowpack",
    mutates_args=(),
)
@cpp_jit()
def deepep_topk_metadata_cuda_with_scales_rowpack(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    N_recv: int,
    E: int,
    topk: int,
    TK: int,
    TK_padded: int,
    alignment: int,
    raw_scales: torch.Tensor,
    cols: int,
    stream: int,
) -> list[torch.Tensor]: ...
