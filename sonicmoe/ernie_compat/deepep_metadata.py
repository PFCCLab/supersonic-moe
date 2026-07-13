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
    from .deepep_metadata_cuda import deepep_metadata_cuda
    _HAS_CUDA_KERNEL = True
except Exception:
    pass

_HAS_TOPK_CUDA_KERNEL = False
_HAS_TOPK_CUDA_SCALES_KERNEL = False
try:
    from .deepep_topk_metadata_cuda import (
        deepep_topk_metadata_cuda,
        deepep_topk_metadata_cuda_with_scales,
        deepep_topk_metadata_cuda_with_scales_and_gated_outputs,
        deepep_topk_metadata_cuda_with_scales_rowpack,
        deepep_topk_metadata_cuda_with_scales_scatterpack,
    )
    _HAS_TOPK_CUDA_KERNEL = True
    _HAS_TOPK_CUDA_SCALES_KERNEL = True
except Exception:
    try:
        from .deepep_topk_metadata_cuda import deepep_topk_metadata_cuda
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

    Fast path: when a call requests the SAME (shape, dtype, device) as the
    previous call for that ``name``, we return the *exact same tensor object*
    with zero Paddle dispatch (~0.05µs dict hit) — no ``view(-1)[:n].view()``
    round-trip (measured ~8.3µs, slower than a fresh ``torch.empty`` at 3.9µs).
    Only on a shape change do we recompute the sliced view.  This matters
    because these scratch tensors are re-requested with a stable shape across
    the many sub-batch calls within a step.

    IMPORTANT: entries handed out here alias a shared backing buffer and are
    only safe for **scratch** that does NOT survive into backward.  Never cache
    a tensor that is ctx-saved (x_gather_idx / s_scatter_idx /
    s_reverse_scatter_idx / naept / topk_scores / score_src_idx /
    expert_offsets), because interleaved forward/backward (pipeline parallel,
    1F1B) would let a later forward overwrite an earlier layer's saved metadata.
    """

    def __init__(self):
        self._bufs: dict[str, torch.Tensor] = {}
        # Per-name memo of the last exact view returned, keyed by (shape, dtype, device).
        self._views: dict[str, tuple[tuple, object, str, torch.Tensor]] = {}

    def get_or_alloc(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a tensor of *exact* ``shape``.  Reuses cached buffer if large enough."""
        shape = tuple(shape)
        dev_s = str(device)

        # Fast path: identical (shape, dtype, device) as last time → return the
        # memoized view object directly (no view/slice dispatch).
        memo = self._views.get(name)
        if memo is not None:
            m_shape, m_dtype, m_dev, m_view = memo
            if m_shape == shape and m_dtype == dtype and m_dev == dev_s:
                if zero:
                    m_view.zero_()
                return m_view

        needed = math.prod(shape) if shape else 1
        t = self._bufs.get(name)

        if t is not None and t.dtype == dtype and str(t.device) == dev_s and t.numel() >= needed:
            view = t.view(-1)[:needed].view(shape)
        else:
            alloc = max(needed, int(t.numel() * 1.25) if t is not None else needed)
            t = torch.empty(alloc, dtype=dtype, device=device)
            self._bufs[name] = t
            view = t[:needed].view(shape)

        self._views[name] = (shape, dtype, dev_s, view)
        if zero:
            view.zero_()
        return view

    def clear(self):
        self._bufs.clear()
        self._views.clear()


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
    device = _normalize_device(device)

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


def deepep_topk_to_sonic_metadata_with_scales(
    dispatched_indices: torch.Tensor,   # [N_recv, topk] int32, -1 = masked
    dispatched_probs: torch.Tensor,     # [N_recv, topk] float32
    tokens_per_expert: Sequence[int] | torch.Tensor,  # [E]
    E: int,
    raw_scales: torch.Tensor,           # raw bytes or compact int32 DeepEP carrier
    cols: int,
    device: str | torch.device = "cuda",
    block: int = 128,
    pack_scales_in_scatter: bool = False,
    pack_scales_rowmajor: bool = False,
    gated_output_prototype: torch.Tensor | None = None,
    gated_n: int | None = None,
    gated_preact_bf16: bool = False,
    gated_allocate_z_scale: bool = True,
):
    """Topk metadata conversion plus optional Sonic FP8 scale packing.

    Public ``deepep_topk_to_sonic_metadata`` intentionally keeps its 10-value
    return contract.  This helper is for the Sonic FP8 DeepEP path only: when
    the CUDA extension supports it, metadata and raw-scale gather/ISA-pack are
    launched from the same C++ op; otherwise the final value is ``None`` and
    callers fall back to the existing scale gather path.
    """
    N_recv = dispatched_indices.shape[0]
    device = _normalize_device(device)

    if "int32" not in str(dispatched_indices.dtype):
        raise ValueError(f"dispatched_indices: expected int32, got {dispatched_indices.dtype}")
    if "float32" not in str(dispatched_probs.dtype):
        raise ValueError(f"dispatched_probs: expected float32, got {dispatched_probs.dtype}")
    if dispatched_indices.ndim != 2 or dispatched_probs.ndim != 2:
        raise ValueError("dispatched_indices and dispatched_probs must be 2D")
    if dispatched_probs.shape != dispatched_indices.shape:
        raise ValueError(f"shape mismatch: indices={dispatched_indices.shape} vs probs={dispatched_probs.shape}")

    raw_scales_2d = raw_scales
    if raw_scales_2d.ndim == 3:
        if raw_scales_2d.shape[0] != 1:
            raise ValueError(f"raw_scales: expected batch=1, got shape {tuple(raw_scales_2d.shape)}")
        raw_scales_2d = raw_scales_2d.squeeze(0)
    if raw_scales_2d.ndim != 2:
        raise ValueError(f"raw_scales: expected rank 2, got shape {tuple(raw_scales_2d.shape)}")
    if raw_scales_2d.shape[0] != N_recv:
        raise ValueError(
            f"raw_scales row mismatch: expected {N_recv}, got {raw_scales_2d.shape[0]}"
        )
    expected_scale_cols = (int(cols) + 31) // 32
    compact_scale_cols = (
        expected_scale_cols // 4 if expected_scale_cols % 4 == 0 else None
    )
    is_compact_word_carrier = (
        "int32" in str(raw_scales_2d.dtype)
        and compact_scale_cols is not None
        and raw_scales_2d.shape[1] == compact_scale_cols
    )
    if (
        raw_scales_2d.shape[1] != expected_scale_cols
        and not is_compact_word_carrier
    ):
        compact_hint = (
            f" or compact int32 [{N_recv}, {compact_scale_cols}]"
            if compact_scale_cols is not None
            else ""
        )
        raise ValueError(
            f"raw_scales shape mismatch: expected [{N_recv}, "
            f"{expected_scale_cols}]{compact_hint} for cols={cols}, got "
            f"{tuple(raw_scales_2d.shape)}/{raw_scales_2d.dtype}"
        )
    if (gated_output_prototype is None) != (gated_n is None):
        raise ValueError(
            "gated_output_prototype and gated_n must be provided together"
        )
    if gated_output_prototype is not None:
        if "float8_e4m3" not in str(gated_output_prototype.dtype):
            raise ValueError(
                "gated_output_prototype must use float8_e4m3, got "
                f"{gated_output_prototype.dtype}"
            )
        if int(gated_n) <= 0 or int(gated_n) % 256 != 0:
            raise ValueError(
                f"gated_n must be positive and divisible by 256, got {gated_n}"
            )

    if _HAS_TOPK_CUDA_KERNEL and _HAS_TOPK_CUDA_SCALES_KERNEL:
        return _deepep_topk_to_sonic_metadata_cuda(
            dispatched_indices,
            dispatched_probs,
            tokens_per_expert,
            E,
            device,
            block,
            raw_scales=raw_scales_2d,
            cols=int(cols),
            pack_scales_in_scatter=pack_scales_in_scatter,
            pack_scales_rowmajor=pack_scales_rowmajor,
            gated_output_prototype=gated_output_prototype,
            gated_n=gated_n,
            gated_preact_bf16=bool(gated_preact_bf16),
            gated_allocate_z_scale=bool(gated_allocate_z_scale),
        )

    meta = deepep_topk_to_sonic_metadata(
        dispatched_indices,
        dispatched_probs,
        tokens_per_expert,
        E,
        device,
        block,
    )
    return (*meta, None)


# ── CUDA topk implementation ─────────────────────────────────────────────────

_INITIALIZED_PADDLE_DEVICES: set[int] = set()


def _selected_cuda_device() -> int:
    selected = os.environ.get("FLAGS_selected_gpus")
    if selected:
        return int(selected.split(",")[0])
    return int(os.environ.get("PADDLE_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))


def _ensure_paddle_device_context(device_id: int) -> None:
    if device_id in _INITIALIZED_PADDLE_DEVICES:
        return
    try:
        import paddle

        paddle.device.set_device(f"gpu:{device_id}")
        warm = paddle.empty([1], dtype="float32")
        del warm
        _INITIALIZED_PADDLE_DEVICES.add(device_id)
    except Exception:
        pass
    try:
        import torch as _torch

        _torch.cuda.set_device(device_id)
    except Exception:
        pass


def _normalize_device(device):
    """Normalize torch.device / paddle Place / str → 'cuda:N' string.

    Under paddle.compat torch-proxy, `torch.empty_like(..., device=...)`
    routes through paddle which only accepts strings or `core.Place`.
    Pre-normalizing here keeps callers (mlp_node_v2 passing `x.device`)
    proxy-agnostic.
    """
    if device is None:
        device_id = _selected_cuda_device()
        _ensure_paddle_device_context(device_id)
        return f"cuda:{device_id}"
    if isinstance(device, str):
        if device == "cuda":
            device_id = _selected_cuda_device()
            _ensure_paddle_device_context(device_id)
            return f"cuda:{device_id}"
        if device.startswith("cuda:"):
            _ensure_paddle_device_context(int(device.split(":", 1)[1]))
        return device
    idx = getattr(device, "index", None)
    if idx is None:
        try:
            idx = device.gpu_device_id()
        except Exception:
            idx = _selected_cuda_device()
    device_id = int(idx) if idx is not None else _selected_cuda_device()
    _ensure_paddle_device_context(device_id)
    return f"cuda:{device_id}"


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

    # CRITICAL: .size * .itemsize — NOT .numel() * .element_size()
    # The latter triggers cudaMemcpy D2H (GPU stream sync) under Paddle proxy.
    # Gated by tests/test_no_memcpy_sync.py. See PR#22.
    nbytes = (cpu_tensor.nbytes
              if hasattr(cpu_tensor, "nbytes")
              else cpu_tensor.size * cpu_tensor.itemsize)
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
    raw_scales: torch.Tensor | None = None,
    cols: int | None = None,
    pack_scales_in_scatter: bool = False,
    pack_scales_rowmajor: bool = False,
    gated_output_prototype: torch.Tensor | None = None,
    gated_n: int | None = None,
    gated_preact_bf16: bool = False,
    gated_allocate_z_scale: bool = True,
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

    # DeepEP already returns the counts on host, so use them only to size the
    # outputs. The CUDA kernels derive their own expert counts from block_hist;
    # no redundant pinned H2D copy is needed on the launch-critical path.
    if isinstance(tokens_per_expert, torch.Tensor):
        tpe_list = tokens_per_expert.tolist()
    else:
        tpe_list = list(tokens_per_expert)

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
        base = (efo, empty_i, empty_i, empty_i, naept, empty_f, 0, 0, N_recv, None)
        if raw_scales is not None:
            if gated_output_prototype is not None:
                return (*base, None, None, None, None, None)
            return (*base, None)
        return base

    # ── Output + scratch allocation now lives in C++ ─────────────────────────
    # The launcher allocates all output tensors (expert_offsets / x_gather_idx /
    # s_scatter_idx / s_reverse_scatter_idx / naept / topk_scores /
    # score_src_idx) plus its internal scratch (seg_starts / real_bases /
    # cumsum_workspace) via the caching allocator and returns the 7 outputs.
    # This removes ~9-11 Python empty/zeros dygraph dispatches per call.
    # Each call gets INDEPENDENT storage (not cross-call reuse), so ctx-saved
    # metadata stays correct under interleaved forward/backward (PP / 1F1B) —
    # semantically identical to the previous per-call Python allocation.
    _stream_obj = torch.cuda.current_stream(device)
    stream = _stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream

    if raw_scales is not None and _HAS_TOPK_CUDA_SCALES_KERNEL:
        if cols is None:
            raise ValueError("cols is required when raw_scales is provided")
        raw_scales_2d = raw_scales
        if raw_scales_2d.ndim == 3:
            if raw_scales_2d.shape[0] != 1:
                raise ValueError(f"raw_scales: expected batch=1, got shape {tuple(raw_scales_2d.shape)}")
            raw_scales_2d = raw_scales_2d.squeeze(0)
        if not raw_scales_2d.is_contiguous():
            raw_scales_2d = raw_scales_2d.contiguous()
        if gated_output_prototype is not None:
            if pack_scales_in_scatter or pack_scales_rowmajor:
                raise ValueError(
                    "preallocated gated outputs require the default scale packer"
                )
            metadata_with_scales = (
                deepep_topk_metadata_cuda_with_scales_and_gated_outputs
            )
        elif pack_scales_in_scatter:
            metadata_with_scales = deepep_topk_metadata_cuda_with_scales_scatterpack
        elif pack_scales_rowmajor:
            metadata_with_scales = deepep_topk_metadata_cuda_with_scales_rowpack
        else:
            metadata_with_scales = deepep_topk_metadata_cuda_with_scales
        indices_2d = (
            dispatched_indices
            if dispatched_indices.is_contiguous()
            else dispatched_indices.contiguous()
        )
        probs_2d = (
            dispatched_probs
            if dispatched_probs.is_contiguous()
            else dispatched_probs.contiguous()
        )
        args = [
            indices_2d,
            probs_2d,
            N_recv,
            E,
            topk,
            TK,
            TK_padded,
            block,
            raw_scales_2d,
            int(cols),
        ]
        if gated_output_prototype is not None:
            args.extend(
                [
                    gated_output_prototype,
                    int(gated_n),
                    bool(gated_preact_bf16),
                    bool(gated_allocate_z_scale),
                ]
            )
        args.append(stream)
        outputs = metadata_with_scales(*args)
    else:
        indices_2d = (
            dispatched_indices
            if dispatched_indices.is_contiguous()
            else dispatched_indices.contiguous()
        )
        probs_2d = (
            dispatched_probs
            if dispatched_probs.is_contiguous()
            else dispatched_probs.contiguous()
        )
        outputs = deepep_topk_metadata_cuda(
            indices_2d,
            probs_2d,
            N_recv,
            E,
            topk,
            TK,
            TK_padded,
            block,
            stream,
        )

    (
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        naept,
        topk_scores,
        score_src_idx,
    ) = outputs[:7]
    packed_scales = outputs[7] if len(outputs) > 7 else None
    gated_outputs = tuple(outputs[8:12]) if len(outputs) >= 12 else ()

    # topk_scores is TK_padded-sized with zeros at pad positions.
    # For the topk path, _DownProjection uses T=N_recv and _router_forward
    # indexes scores via naept (token-major order within the first TK elements).
    # The backward indexes via s_scatter_idx which maps to token-major positions
    # for real entries and to >= TK for pad entries (where scores = 0).
    # Return topk_scores as full TK_padded buffer (compatible with backward).
    base = (
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
    if raw_scales is not None:
        return (*base, packed_scales, *gated_outputs)
    return base


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
    device = _normalize_device(device)
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
