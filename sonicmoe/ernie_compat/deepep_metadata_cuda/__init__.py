"""JIT-compiled CUDA kernel for DeepEP → SonicMoE metadata fill (V2).

Uses the same ``@cpp_jit()`` pattern as ``sonicmoe.count_cumsum``.
V2: host computes prefix-sum, kernel only does parallel segment fill.
"""

from __future__ import annotations

import torch

from sonicmoe.enums import LIBRARY_NAME
from sonicmoe.jit import cpp_jit


@torch.library.custom_op(
    f"{LIBRARY_NAME}::deepep_metadata_cuda",
    mutates_args={
        "x_gather_idx",
        "router_scores",
    },
)
@cpp_jit()
def deepep_metadata_cuda(
    expert_freq_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    router_scores: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_lens: torch.Tensor,
    real_counts: torch.Tensor,
    real_bases: torch.Tensor,
    pad_bases: torch.Tensor,
    E: int,
    stream: int,
) -> None: ...
