"""DeepEP → SonicMoE metadata conversion (zero argsort, zero sync).

DeepEP dispatch produces tokens already sorted by expert, so routing metadata
is trivially identity permutations + cumsum over ``tokens_per_expert``.

Padding strategy: **route-level padding** (matching the FP8 frontier design).
  - Padding rows use gather index 0 (arbitrary valid row) with score=0.
  - x is NOT modified — no sentinel rows appended.
  - This is identical to ``_pad_routing_metadata`` in functional/__init__.py.

Two implementations:
  1. **CUDA kernel** (V2): host computes prefix-sum (trivial on CPU for E ≤ 384),
     then launches 1-block-per-expert fill kernel with vectorized int4/float4
     stores.  No ``item()`` DtoH, no barriers, no over-allocation.
  2. **Python fallback**: pure PyTorch ops, used when CUDA kernel is not compiled.

``deepep_to_sonic_metadata`` dispatches to CUDA if available.

For the **topk** path (``deepep_topk_to_sonic_metadata``):
  1. **CUDA kernel** (fused): warp-ballot progressive cumsum (same as moe_permute)
     + 4 micro-kernels fused via stream ordering. Zero argsort. Stable ordering.
  2. **Python fallback**: argsort-based, ~1.3ms for typical shapes.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import torch

# Try to import the JIT-compiled CUDA kernels
_HAS_CUDA_KERNEL = False
try:
    from sonicmoe.ernie_compat.deepep_metadata_cuda import deepep_metadata_cuda
    _HAS_CUDA_KERNEL = True
except Exception:
    pass

_HAS_TOPK_CUDA_KERNEL = False
try:
    from sonicmoe.ernie_compat.deepep_topk_metadata_cuda import deepep_topk_metadata_cuda
    _HAS_TOPK_CUDA_KERNEL = True
except Exception:
    pass

_HAS_CUDA_ART = False
try:
    from cuda.bindings import runtime as cudart
    from collections import deque
    _HAS_CUDA_ART = True
    pin_memory_queue = deque()
except Exception:
    pass

# ── Identity-arange cache (per-device) ──────────────────────────────────────
# s_scatter_idx, s_reverse_scatter_idx, num_activated_expert_per_token_offset
# are pure identity sequences. They depend only on (TK_padded, device, dtype)
# and never on routing — caching them eliminates 3 arange allocations per iter.
_ARANGE_CACHE: dict[tuple, torch.Tensor] = {}


class _TopkOutputCache:
    """High-watermark tensor cache for the topk CUDA metadata path.

    Eliminates per-call allocations of ~10 output tensors + workspace.
    Each entry grows by 1.25x when re-allocated to amortize cost.
    """

    def __init__(self):
        self._bufs: dict[str, torch.Tensor] = {}

    def get_or_alloc(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a tensor of *exact* ``shape``.  Reuses cached buffer if large enough."""
        needed = math.prod(shape) if shape else 1
        t = self._bufs.get(name)

        if t is not None and t.dtype == dtype and str(t.device) == str(device) and t.numel() >= needed:
            view = t.view(-1)[:needed].view(shape)
            if zero:
                view.zero_()
            return view

        alloc = max(needed, int(t.numel() * 1.25) if t is not None else needed)
        t = torch.empty(alloc, dtype=dtype, device=device)
        if zero:
            t.zero_()
        self._bufs[name] = t
        return t[:needed].view(shape)

    def clear(self):
        self._bufs.clear()


# Global cache instance — cleared by invalidate_topk_cache().
_TOPK_CACHE = _TopkOutputCache()


def invalidate_topk_cache() -> None:
    """Clear the topk output tensor cache.  Called on optimizer step / shape change."""
    _TOPK_CACHE.clear()


def _cached_arange(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Return a cached identity arange of length n.

    The returned tensor is a read-only handle — callers must NOT mutate it.
    """
    key = (n, dtype, str(device))
    t = _ARANGE_CACHE.get(key)
    if t is None or t.shape[0] < n:
        t = torch.arange(max(n, 1), dtype=dtype, device=device)
        _ARANGE_CACHE[key] = t
    if t.shape[0] == n:
        return t
    return t[:n]


def deepep_topk_to_sonic_metadata(
    dispatched_indices: torch.Tensor,   # [N_recv, topk] int32, -1 = masked
    dispatched_probs: torch.Tensor,     # [N_recv, topk] float32
    tokens_per_expert: Sequence[int] | torch.Tensor,  # [E]
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Convert real DeepEP topk dispatch results to SonicMoE routing metadata.

    Unlike ``deepep_to_sonic_metadata`` which assumes tokens are already sorted
    by expert (identity layout, K=1), this function handles the real DeepEP
    dispatch where each received token can be routed to multiple local experts.

    Parameters
    ----------
    dispatched_indices : Tensor [N_recv, topk] int32
        Local expert indices from DeepEP dispatch. -1 means masked (not routed
        to any local expert in that slot).
    dispatched_probs : Tensor [N_recv, topk] float32
        Routing probabilities from DeepEP dispatch.
    tokens_per_expert : list[int] or Tensor [E]
        Per-expert token counts from DeepEP ``buffer.dispatch()``.
    E : int
        Number of local experts.
    device : str or torch.device
        Target device for output tensors.
    block : int
        Alignment block size (128 for FP8 GEMM).

    Returns
    -------
    expert_frequency_offset : Tensor [E+1] int32
    x_gather_idx : Tensor [TK_padded] int32
    s_scatter_idx : Tensor [TK_padded] int32
    s_reverse_scatter_idx : Tensor [TK] int32
    num_activated_expert_per_token_offset : Tensor [N_recv+1] int32
    topk_scores : Tensor [TK_padded] float32 (token-major scores in [0,TK); padding zeros in [TK,TK_padded))
    TK_padded : int
    total_pad_rows : int
    N_recv : int
        Number of original received tokens (T for _DownProjection).
    score_src_idx : Tensor [TK] int32 or None
        Token-major flat indices (row*topk + col) into dispatched_probs for
        differentiable score reconstruction.  None when the CUDA kernel is
        not available (Python fallback); callers must then rebuild via the
        Triton _build_score_src_idx_kernel.
    """
    N_recv = dispatched_indices.shape[0]
    topk = dispatched_indices.shape[1]

    if "int32" not in str(dispatched_indices.dtype):
        raise ValueError(f"dispatched_indices: expected int32, got {dispatched_indices.dtype}")
    if "float32" not in str(dispatched_probs.dtype):
        raise ValueError(f"dispatched_probs: expected float32, got {dispatched_probs.dtype}")
    if dispatched_indices.ndim != 2 or dispatched_probs.ndim != 2:
        raise ValueError("dispatched_indices and dispatched_probs must be 2D")
    if dispatched_probs.shape != dispatched_indices.shape:
        raise ValueError(f"shape mismatch: indices={dispatched_indices.shape} vs probs={dispatched_probs.shape}")

    # Dispatch to CUDA kernel if available
    if _HAS_TOPK_CUDA_KERNEL:
        return _deepep_topk_to_sonic_metadata_cuda(
            dispatched_indices, dispatched_probs, tokens_per_expert,
            E, device, block,
        )

    # ── Phase 1: Flatten and filter valid entries ───────────────────────
    tok_ids = torch.arange(N_recv, dtype=torch.int32, device=device) \
                   .unsqueeze(1).expand(N_recv, topk)
    valid = dispatched_indices >= 0

    tok_flat = tok_ids[valid]           # original token row for each valid entry
    exp_flat = dispatched_indices[valid].int()  # expert id for each valid entry
    scr_flat = dispatched_probs[valid].float()  # score for each valid entry
    TK = tok_flat.shape[0]              # total valid token-expert assignments

    if TK == 0:
        # Edge case: no tokens routed to any local expert
        efo = torch.zeros(E + 1, dtype=torch.int32, device=device)
        empty_i = torch.empty(0, dtype=torch.int32, device=device)
        empty_f = torch.empty(0, dtype=torch.float32, device=device)
        naept = torch.zeros(N_recv + 1, dtype=torch.int32, device=device)
        return efo, empty_i, empty_i, empty_i, naept, empty_f, 0, 0, N_recv, None

    # ── Phase 2: Sort by expert (argsort) ──────────────────────────────
    # sort_perm: expert-sorted position → token-major position
    sort_perm = exp_flat.long().argsort(stable=True)

    # x_gather_idx (unpadded): original token row at each expert-sorted position
    x_gather_unpadded = tok_flat[sort_perm].int()

    # s_reverse_scatter_idx (unpadded): token-major position → expert-sorted position
    s_reverse_unpadded = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_unpadded[sort_perm] = torch.arange(TK, dtype=torch.int32, device=device)

    # ── Phase 3: Host prefix-sum for padded offsets ────────────────────
    (offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases,
     TK_padded, total_pad_rows) = _host_prefix_sum(tokens_per_expert, E, TK, block)

    expert_frequency_offset = torch.tensor(offsets, dtype=torch.int32, device=device)

    # ── Phase 4: Apply route-level padding ─────────────────────────────
    if total_pad_rows == 0:
        # Already aligned — no padding needed
        x_gather_idx = x_gather_unpadded
        # s_scatter_idx: expert-sorted position → token-major position
        s_scatter_idx = sort_perm.int()
        s_reverse_scatter_idx = s_reverse_unpadded
        # topk_scores: token-major order
        topk_scores = scr_flat
    else:
        # Compute dst_idx: maps unpadded expert-sorted position → padded position
        # For each sorted position, determine which expert it belongs to and
        # its local offset within that expert's segment.
        real_bases_d = torch.tensor(real_bases, dtype=torch.long, device=device)
        seg_starts_d = torch.tensor(seg_starts, dtype=torch.long, device=device)

        sorted_experts = exp_flat[sort_perm].long()  # expert for each sorted pos
        local_pos = torch.arange(TK, dtype=torch.long, device=device) - real_bases_d[sorted_experts]
        dst_idx = (seg_starts_d[sorted_experts] + local_pos).long()

        # x_gather_idx: padded, padding rows gather from row 0 (score=0 nullifies)
        x_gather_idx = torch.zeros(TK_padded, dtype=torch.int32, device=device)
        x_gather_idx[dst_idx] = x_gather_unpadded

        # s_scatter_idx: maps padded expert-sorted position → token-major position
        # Real positions: point to their token-major index (via sort_perm)
        # Pad positions: point to virtual indices beyond TK (score=0, safe)
        N_pad = total_pad_rows
        s_scatter_idx = torch.empty(TK_padded, dtype=torch.int32, device=device)
        s_scatter_idx[dst_idx] = sort_perm.int()
        # Fill padding positions with virtual indices [TK, TK + N_pad)
        is_real = torch.zeros(TK_padded, dtype=torch.bool, device=device)
        is_real[dst_idx] = True
        pad_positions = torch.where(~is_real)[0]
        s_scatter_idx[pad_positions] = torch.arange(
            TK, TK + N_pad, dtype=torch.int32, device=device
        )

        # s_reverse_scatter_idx: token-major → padded expert-sorted position
        # Length TK (only real entries need reverse mapping)
        s_reverse_scatter_idx = dst_idx[s_reverse_unpadded.long()].int()

        # topk_scores: TK_padded-sized. Real entries at token-major positions [0, TK),
        # padding entries at [TK, TK_padded) = 0. The backward does
        # s = topk_scores[s_scatter_idx] where pad positions map to >= TK.
        topk_scores = torch.zeros(TK_padded, dtype=torch.float32, device=device)
        topk_scores[:TK] = scr_flat

    # ── Phase 5: Build num_activated_expert_per_token_offset ───────────
    # naept[i] = cumulative count of expert assignments for tokens 0..i-1
    per_token_counts = torch.bincount(tok_flat.long(), minlength=N_recv).int()
    naept = torch.zeros(N_recv + 1, dtype=torch.int32, device=device)
    naept[1:] = per_token_counts.cumsum(0).int()

    return (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        naept,
        topk_scores,
        TK_padded,
        total_pad_rows,
        N_recv,
        None,  # score_src_idx unavailable in Python fallback; caller will rebuild
    )


# ── CUDA topk implementation ─────────────────────────────────────────────────


def _normalize_device(device):
    """Normalize torch.device / paddle Place / str → 'cuda:N' string.

    Under paddle.compat torch-proxy, `torch.empty_like(..., device=...)`
    routes through paddle which only accepts strings or `core.Place`.
    Pre-normalizing here keeps callers (mlp_node_v2 passing `x.device`)
    proxy-agnostic.
    """
    if device is None or isinstance(device, str):
        return device
    idx = getattr(device, "index", None)
    if idx is None:
        try:
            idx = device.gpu_device_id()
        except Exception:
            idx = 0
    return f"cuda:{int(idx) if idx is not None else 0}"


def _copy_tpe_h2d_async(tpe_list, device):
    """Async pinned-memory H2D copy for tokens_per_expert.

    Replaces the implicit-sync `torch.tensor(list, device=device)` path with
    a true async H2D using cudaMemcpyAsync on a pinned source buffer.
    Pinned tensors are queued and recycled once their copy event completes.
    Falls back to the legacy sync path if cuda.bindings is unavailable.
    """
    device = _normalize_device(device)
    if not _HAS_CUDA_ART:
        return torch.tensor(tpe_list, dtype=torch.int32, device=device)

    if hasattr(torch, "CUDAPinnedPlace"):
        cpu_tensor = torch.to_tensor(
            tpe_list, dtype=torch.int32, place=torch.CUDAPinnedPlace())
    else:
        cpu_tensor = torch.tensor(tpe_list, dtype=torch.int32, pin_memory=True)
    gpu_tensor = torch.empty_like(cpu_tensor, device=device)
    current_stream = torch.cuda.current_stream(device)

    nbytes = (cpu_tensor.nbytes
              if hasattr(cpu_tensor, "nbytes")
              else cpu_tensor.numel() * cpu_tensor.element_size())
    raw_stream = (current_stream.stream_base.cuda_stream
                  if hasattr(current_stream, "stream_base")
                  else current_stream.cuda_stream)
    (err,) = cudart.cudaMemcpyAsync(
        gpu_tensor.data_ptr(),
        cpu_tensor.data_ptr(),
        nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        raw_stream,
    )
    assert err == cudart.cudaError_t.cudaSuccess, f"cudaMemcpyAsync failed: {err}"

    event = torch.cuda.Event()
    event.record()
    pin_memory_queue.append((cpu_tensor, event))

    while pin_memory_queue:
        _, ev = pin_memory_queue[0]
        if ev.query():
            pin_memory_queue.popleft()
        else:
            break

    return gpu_tensor


def _deepep_topk_to_sonic_metadata_cuda(
    dispatched_indices: torch.Tensor,
    dispatched_probs: torch.Tensor,
    tokens_per_expert: Sequence[int] | torch.Tensor,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """CUDA fused topk metadata conversion (warp-ballot, zero argsort).

    Uses Paddle moe_permute-style progressive cumsum for stable ordering.
    4 fused kernels via stream ordering, ~50us for N=16K, topk=8, E=8.
    """
    N_recv = dispatched_indices.shape[0]
    topk = dispatched_indices.shape[1]
    device = _normalize_device(device)

    # Optional contract check (debug only — sm-level scan, ~5µs):
    # dispatched_indices[i, :] must contain UNIQUE expert IDs. Real DeepEP
    # dispatch enforces this; duplicates would collapse two slots onto one
    # token-major position in s_reverse_scatter_idx and orphan padded slots,
    # eventually causing IMA in router_forward / wgrad gather. Enable via
    # SONIC_MOE_VALIDATE_DISPATCH=1 (off by default for production perf).
    if os.environ.get("SONIC_MOE_VALIDATE_DISPATCH") == "1" and N_recv > 0 and topk > 1:
        sorted_idx, _ = dispatched_indices.sort(dim=-1)
        # Adjacent-equal among non-negative (-1 = masked is allowed multi-occurrence).
        dup_mask = (sorted_idx[:, 1:] == sorted_idx[:, :-1]) & (sorted_idx[:, 1:] >= 0)
        if dup_mask.any():
            bad_row = int(dup_mask.any(dim=-1).nonzero()[0].item())
            raise ValueError(
                f"dispatched_indices contract violation: row {bad_row} has "
                f"duplicate expert IDs {dispatched_indices[bad_row].tolist()}. "
                "Each token's topk slots must be unique experts."
            )

    # Compute TK and TK_padded on host (need tokens_per_expert list)
    if isinstance(tokens_per_expert, torch.Tensor):
        tpe_list = tokens_per_expert.tolist()
        tpe_dev = _TOPK_CACHE.get_or_alloc(
            "tpe_dev", (E,), torch.int32, device, zero=False,
        )
        tpe_dev.copy_(tokens_per_expert.to(device=device, dtype=torch.int32))
    else:
        tpe_list = list(tokens_per_expert)
        tpe_dev = _copy_tpe_h2d_async(tokens_per_expert, device)

    TK = sum(tpe_list)
    # Compute TK_padded (padded sum)
    TK_padded = 0
    for count in tpe_list:
        if count > 0:
            TK_padded += ((count + block - 1) // block) * block
    total_pad_rows = TK_padded - TK

    if TK == 0:
        efo = torch.zeros(E + 1, dtype=torch.int32, device=device)
        empty_i = torch.empty(0, dtype=torch.int32, device=device)
        empty_f = torch.empty(0, dtype=torch.float32, device=device)
        naept = torch.zeros(N_recv + 1, dtype=torch.int32, device=device)
        return efo, empty_i, empty_i, empty_i, naept, empty_f, 0, 0, N_recv, None

    # NOTE: ``seg_starts`` / ``real_bases`` and the kernel scratch buffers are
    # allocated below alongside the device-typed outputs. An earlier draft
    # also pre-allocated ``expert_offsets`` etc. here without ``device=`` —
    # those duplicates were dead code immediately overwritten by the block
    # below, and under raw torch they would have been CPU tensors silently
    # smuggled into the CUDA launcher. Removed S76.

    # ── Output tensors (saved on autograd ctx) — MUST be per-call ────────────
    # Bug 2026-04: when two MoE layers' forwards run before any backward (true
    # in pipeline parallel and 1F1B-style schedules), reusing a global cache
    # for these outputs causes the second forward to overwrite the first
    # layer's saved metadata, silently corrupting its dw1/dw2/dx grads. The
    # extra alloc cost is ~5–10 µs total via the torch caching allocator,
    # negligible vs. the GEMM cost; correctness is non-negotiable.
    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    seg_starts = torch.empty(E, dtype=torch.int32, device=device)
    real_bases = torch.empty(E, dtype=torch.int32, device=device)
    x_gather_idx = torch.zeros(TK_padded, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK_padded, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    topk_scores = torch.zeros(TK_padded, dtype=torch.float32, device=device)
    naept = torch.empty(N_recv + 1, dtype=torch.int32, device=device)
    score_src_idx = torch.empty(TK, dtype=torch.int32, device=device)

    num_blocks = (N_recv + 31) // 32  # ROWS_PER_BLOCK = 32
    # Workspace layout (must match kernel.cu's launcher partition):
    #   block_hist[scatter_blocks*E] + block_offset[scatter_blocks*E]
    #   + block_naept_sum[scatter_blocks] + block_naept_base[scatter_blocks]
    # = 2*num_blocks*(E+1) int32 slots.
    # NOTE: an earlier (pre-session-71) layout used `2*num_blocks*E + 1`
    # (single int32 completion-flag tail). The session-71 kernel rewrite added
    # per-block naept arrays but did not update this caller, causing the
    # launcher to write `2*num_blocks` ints past the buffer end -> silent
    # cudaErrorIllegalAddress in downstream consumers when the OOB region
    # happens to sit on top of a live allocation. Keep this size exact.
    cumsum_workspace = torch.empty([2 * num_blocks * (E + 1)], dtype=torch.int32, device=device)

    _stream_obj = torch.cuda.current_stream(device)
    stream = _stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream

    deepep_topk_metadata_cuda(
        dispatched_indices=dispatched_indices.contiguous(),
        dispatched_probs=dispatched_probs.contiguous(),
        tokens_per_expert=tpe_dev,
        expert_offsets=expert_offsets,
        seg_starts=seg_starts,
        real_bases=real_bases,
        x_gather_idx=x_gather_idx,
        s_scatter_idx=s_scatter_idx,
        s_reverse_scatter_idx=s_reverse_scatter_idx,
        topk_scores=topk_scores,
        naept=naept,
        global_block_cumsum=cumsum_workspace,
        score_src_idx=score_src_idx,
        N_recv=N_recv,
        E=E,
        topk=topk,
        TK=TK,
        TK_padded=TK_padded,
        alignment=block,
        stream=stream,
    )

    # topk_scores is TK_padded-sized with zeros at pad positions.
    # For the topk path, _DownProjection uses T=N_recv and _router_forward
    # indexes scores via naept (token-major order within the first TK elements).
    # The backward indexes via s_scatter_idx which maps to token-major positions
    # for real entries and to >= TK for pad entries (where scores = 0).
    # Return topk_scores as full TK_padded buffer (compatible with backward).
    return (
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        naept,
        topk_scores,
        TK_padded,
        total_pad_rows,
        N_recv,
        score_src_idx,
    )


def deepep_to_sonic_metadata(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    T: int,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Zero-sync metadata conversion: DeepEP format → SonicMoE routing tensors.

    Uses **route-level padding** (frontier design): padding rows use gather
    index 0 with score=0.  x is NOT modified (no sentinel rows needed).

    Dispatches to CUDA kernel if available, otherwise falls back to Python.

    Parameters
    ----------
    tokens_per_expert : list[int] or Tensor of shape [E]
        Per-expert token counts from ``DeepEP buffer.dispatch()``.
    T : int
        Total number of real tokens (``sum(tokens_per_expert)``).
    E : int
        Number of local experts.
    device : str or torch.device
        Target device for output tensors.
    block : int
        Alignment block size (128 for FP8 GEMM).

    Returns
    -------
    expert_frequency_offset : Tensor [E+1] int32
    x_gather_idx : Tensor [TK_padded] int32
    s_scatter_idx : Tensor [TK_padded] int32
    s_reverse_scatter_idx : Tensor [TK_padded] int32
    num_activated_expert_per_token_offset : Tensor [T+1] int32
    router_scores : Tensor [TK_padded] float32
    TK_padded : int
    total_pad_rows : int
        Number of padding slots added (for reference; x is NOT padded).
    """
    if _HAS_CUDA_KERNEL:
        return _deepep_to_sonic_metadata_cuda(tokens_per_expert, T, E, device, block)
    return _deepep_to_sonic_metadata_python(tokens_per_expert, T, E, device, block)


# ── Host-side prefix-sum (shared by CUDA and Python paths) ──────────────────

def _host_prefix_sum(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    E: int,
    T: int,
    block: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], int, int]:
    """Compute all per-expert metadata on CPU.  O(E), trivial for E ≤ 384."""
    if isinstance(tokens_per_expert, torch.Tensor):
        tpe = tokens_per_expert.tolist()
    else:
        tpe = list(tokens_per_expert)

    # Per-expert arrays
    offsets = [0]       # [E+1] padded cumulative offsets
    seg_starts = []     # [E] = offsets[e]
    seg_lens = []       # [E] = offsets[e+1] - offsets[e]
    real_counts = []    # [E] = tpe[e]
    real_bases = []     # [E] cumulative real token offset
    pad_bases = []      # [E] = T + cumulative padding offset

    real_cum = 0
    pad_cum = 0

    for i in range(E):
        count = tpe[i]
        padded = ((count + block - 1) // block * block) if count > 0 else 0
        pad = padded - count

        seg_starts.append(offsets[-1])
        seg_lens.append(padded)
        real_counts.append(count)
        real_bases.append(real_cum)
        pad_bases.append(T + pad_cum)

        offsets.append(offsets[-1] + padded)
        real_cum += count
        pad_cum += pad

    TK_padded = offsets[-1]
    total_pad_rows = pad_cum

    return offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases, TK_padded, total_pad_rows


# ── CUDA implementation (V2: host prefix-sum + multi-block fill) ─────────────

def _deepep_to_sonic_metadata_cuda(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    T: int,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """CUDA V2: host prefix-sum → 1-block-per-expert fill kernel.

    Eliminates: item() DtoH sync, __syncthreads barrier, over-allocation.
    """
    (offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases,
     TK_padded, total_pad_rows) = _host_prefix_sum(tokens_per_expert, E, T, block)

    # Transfer pre-computed metadata to device in one fused tensor to minimize
    # HtoD launches.  Layout: [offsets(E+1) | seg_starts(E) | seg_lens(E) |
    #                           real_counts(E) | real_bases(E) | pad_bases(E)]
    # Total: 6E+1 int32s — fits in a single 128-byte cacheline for E≤20.
    fused_host = offsets + seg_starts + seg_lens + real_counts + real_bases + pad_bases
    fused_dev = torch.tensor(fused_host, dtype=torch.int32, device=device)

    # Slice views (no copy — shares storage)
    o = E + 1
    expert_freq_offset = fused_dev[:o]
    seg_starts_d  = fused_dev[o:o + E];          o += E
    seg_lens_d    = fused_dev[o:o + E];           o += E
    real_counts_d = fused_dev[o:o + E];           o += E
    real_bases_d  = fused_dev[o:o + E];           o += E
    pad_bases_d   = fused_dev[o:o + E]

    # Allocate exact-size output tensors
    x_gather_idx = torch.empty(TK_padded, dtype=torch.int32, device=device)
    router_scores = torch.empty(TK_padded, dtype=torch.float32, device=device)

    # Launch fill kernel: E blocks × 256 threads
    if TK_padded > 0:
        _stream_obj = torch.cuda.current_stream(device)
        stream = _stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream
        deepep_metadata_cuda(
            expert_freq_offset=expert_freq_offset,
            x_gather_idx=x_gather_idx,
            router_scores=router_scores,
            seg_starts=seg_starts_d,
            seg_lens=seg_lens_d,
            real_counts=real_counts_d,
            real_bases=real_bases_d,
            pad_bases=pad_bases_d,
            E=E,
            stream=stream,
        )

    # Identity scatter indices (cached — never depends on routing)
    s_scatter_idx = _cached_arange(TK_padded, torch.int32, device)
    s_reverse_scatter_idx = s_scatter_idx

    # num_activated_expert_per_token_offset: arange(T_padded+1)
    # Must cover T_padded positions (not T_orig) because _DownProjection
    # uses T_padded for the output tensor and indexes naept[0..T_padded].
    # Each padded position maps to exactly 1 expert (padding has score=0).
    T_for_naept = TK_padded  # T_padded = TK_padded in the DeepEP identity layout
    num_activated_expert_per_token_offset = _cached_arange(
        T_for_naept + 1, torch.int32, device,
    )

    return (
        expert_freq_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        router_scores,
        TK_padded,
        total_pad_rows,
    )


# ── Python fallback ──────────────────────────────────────────────────────────

def _deepep_to_sonic_metadata_python(
    tokens_per_expert: Sequence[int] | torch.Tensor,
    T: int,
    E: int,
    device: str | torch.device = "cuda",
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Pure-Python fallback using PyTorch ops."""
    (offsets, seg_starts, seg_lens, real_counts, real_bases, pad_bases,
     TK_padded, total_pad_rows) = _host_prefix_sum(tokens_per_expert, E, T, block)

    expert_frequency_offset = torch.tensor(offsets, dtype=torch.int32, device=device)

    if isinstance(tokens_per_expert, torch.Tensor):
        tpe = tokens_per_expert.tolist()
    else:
        tpe = list(tokens_per_expert)

    # Build x_gather_idx and router_scores
    gather_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []

    for i in range(E):
        rc = real_counts[i]
        pad_count = seg_lens[i] - rc
        if seg_lens[i] == 0:
            continue

        if rc > 0:
            gather_parts.append(
                torch.arange(real_bases[i], real_bases[i] + rc,
                             dtype=torch.int32, device=device)
            )
            score_parts.append(
                torch.ones(rc, dtype=torch.float32, device=device)
            )

        if pad_count > 0:
            # Route-level padding: padding rows gather from row 0 (score=0
            # nullifies contribution).  Matches frontier _pad_routing_metadata.
            gather_parts.append(
                torch.zeros(pad_count, dtype=torch.int32, device=device)
            )
            score_parts.append(
                torch.zeros(pad_count, dtype=torch.float32, device=device)
            )

    if TK_padded == 0:
        x_gather_idx = torch.empty(0, dtype=torch.int32, device=device)
        router_scores = torch.empty(0, dtype=torch.float32, device=device)
    else:
        x_gather_idx = torch.cat(gather_parts)
        router_scores = torch.cat(score_parts)

    s_scatter_idx = _cached_arange(TK_padded, torch.int32, device)
    s_reverse_scatter_idx = s_scatter_idx
    num_activated_expert_per_token_offset = _cached_arange(
        TK_padded + 1, torch.int32, device,
    )

    return (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        router_scores,
        TK_padded,
        total_pad_rows,
    )
