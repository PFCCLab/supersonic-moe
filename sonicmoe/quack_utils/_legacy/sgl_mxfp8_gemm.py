"""sgl-kernel MXFP8 blockscaled grouped GEMM for varlen MoE.

Uses ``es_sm100_mxfp8_blockscaled_grouped_mm`` from sgl-kernel for
hardware-native MXFP8 blockscaled GEMM on Blackwell SM100 GPUs.

Scale layout
------------
sgl-kernel expects E8M0 scale factors in **ScaleFactorTileLayout**
(128-row × 4-group swizzled tiles), not simple row-major.

The GEMM API has **separate** ``expert_offsets`` (for A / output data)
and ``blockscale_offsets`` (for scale-factor arrays).  Activations
only need 16-token alignment per expert, but scale-factor tiles need
128-row alignment.  We exploit this by:

  1. Padding activations to 16-aligned  (GEMM M_i alignment)
  2. Using our fast Triton quant      (row-major E8M0 output)
  3. Padding scales to 128-aligned     (ScaleFactorTileLayout tiles)
  4. Tiling with ``rowmajor_to_sgl_tiled``
  5. Passing separate ``expert_offsets`` / ``blockscale_offsets``

Weight scales are tiled during precomputation (one-time cached).

GEMM contract
-------------
- A : ``(total_M, K)``  float8_e4m3fn, row-major
- B : ``(E, K, N)``     float8_e4m3fn, stride[1]==1
- sfa : ``(total_scale_M, K//32)``  uint8, **ScaleFactorTileLayout**
- sfb : ``(E, N, K//32)``     uint8, **ScaleFactorTileLayout**
- K % 128 == 0, N % 128 == 0
- M_i % 16 == 0 per expert  (activation alignment)
- scale rows per expert: multiple of 128  (tile alignment)
"""

from __future__ import annotations

import torch

try:
    import sgl_kernel

    _HAS_SGL = True
except ImportError:
    sgl_kernel = None
    _HAS_SGL = False

from sonicmoe.functional.fp8_quant import quantize_activation_blockwise
from sonicmoe.functional.fp8_protocol import FP8Protocol, FP8ScaleGranularity
from sonicmoe.quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast

# ──────────────────────────────────────────────────────────────────────
# Weight FP8 cache (keyed by (data_ptr, shape) to survive ptr reuse)
# ──────────────────────────────────────────────────────────────────────
_SGL_WEIGHT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def clear_sgl_weight_cache():
    """Clear cached weight FP8 quantizations."""
    _SGL_WEIGHT_CACHE.clear()


def has_sgl_kernel() -> bool:
    """Check if sgl-kernel is available."""
    return _HAS_SGL


# ──────────────────────────────────────────────────────────────────────
# ScaleFactorTileLayout conversion
# ──────────────────────────────────────────────────────────────────────


def rowmajor_to_sgl_tiled(
    sc: torch.Tensor,
    n_experts: int,
    rows_per_expert: int,
    n_groups: int,
) -> torch.Tensor:
    """Convert row-major E8M0 scales to sgl's ScaleFactorTileLayout.

    Layout: ``Shape<Shape<_32, _4>, _4>, Stride<Stride<_16, _4>, _1>``
    — tiles of 128 rows × 4 groups.

    Parameters
    ----------
    sc : (n_experts * rows_per_expert, n_groups) uint8
    n_experts, rows_per_expert, n_groups : tile dimensions
        rows_per_expert must be a multiple of 128, n_groups a multiple of 4.

    Returns
    -------
    tiled : same shape as *sc*, elements reordered into ScaleFactorTileLayout.
    """
    assert rows_per_expert % 128 == 0, (
        f"rows_per_expert={rows_per_expert} must be a multiple of 128"
    )
    assert n_groups % 4 == 0, f"n_groups={n_groups} must be a multiple of 4"

    n_tr = rows_per_expert // 128
    n_tc = n_groups // 4
    x = sc.reshape(n_experts, n_tr, 128, n_tc, 4)
    x = x.reshape(n_experts, n_tr, 4, 32, n_tc, 4)  # 128 -> (r1=4, r0=32)
    x = x.permute(0, 1, 4, 3, 2, 5).contiguous()  # (E, tr, tc, r0, r1, g)
    return x.reshape(n_experts * rows_per_expert, n_groups)


# ──────────────────────────────────────────────────────────────────────
# Weight precomputation
# ──────────────────────────────────────────────────────────────────────


def precompute_weight_fp8_sgl(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for sgl-kernel MXFP8 GEMM (cached).

    Parameters
    ----------
    w : Tensor (dim0, dim1, E) bf16
        Original SonicMoE weight layout.
        Down-proj: ``(H, I, E)`` where H=output, I=input/reduction.
        Up-proj:   ``(I, H, E)`` where I=output, H=input/reduction.

    Returns
    -------
    w_fp8_gemm : ``(E, dim1, dim0)`` float8_e4m3fn
        stride[1]==1, ready for sgl GEMM B input.
    w_scales_tiled : ``(E, dim0, dim1//32)`` uint8
        ScaleFactorTileLayout applied per-expert.
    """
    key = (w.data_ptr(), tuple(w.shape))
    if key in _SGL_WEIGHT_CACHE:
        return _SGL_WEIGHT_CACHE[key]

    E = w.shape[-1]
    # (dim0, dim1, E) -> (E, dim0, dim1) contiguous
    w_ehi = w.permute(2, 0, 1).contiguous()
    dim0, dim1 = w_ehi.shape[1], w_ehi.shape[2]
    n_groups = dim1 // 32

    assert dim0 % 128 == 0, (
        f"Weight output dim={dim0} must be a multiple of 128 for ScaleFactorTileLayout"
    )
    assert n_groups % 4 == 0, (
        f"n_groups={n_groups} (dim1={dim1}//32) must be a multiple of 4"
    )

    # Quantize along dim1 (last dim / reduction dim) in blocks of 32
    proto = FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)
    w_fp8, w_scales_e8m0 = quantize_activation_blockwise(w_ehi, proto)
    w_scales_uint8 = w_scales_e8m0.view(torch.uint8).contiguous()

    # Apply ScaleFactorTileLayout per expert
    w_scales_flat = w_scales_uint8.reshape(E * dim0, n_groups)
    w_scales_tiled = rowmajor_to_sgl_tiled(w_scales_flat, E, dim0, n_groups)
    w_scales_tiled = w_scales_tiled.reshape(E, dim0, n_groups)

    # B layout for sgl GEMM: (E, dim1, dim0) with stride[1]==1
    w_fp8_gemm = w_fp8.permute(0, 2, 1)  # non-contiguous view, stride[1]==1

    result = (w_fp8_gemm, w_scales_tiled)
    _SGL_WEIGHT_CACHE[key] = result
    return result


# ──────────────────────────────────────────────────────────────────────
# Main GEMM entry point
# ──────────────────────────────────────────────────────────────────────


_GEMM_ALIGN = 16   # sgl-kernel GEMM requires M_i % 16 == 0 per expert
_TILE_ALIGN = 128  # ScaleFactorTileLayout requires 128-row tiles


def _build_pad_indices(
    cu_src: torch.Tensor,
    cu_dst: torch.Tensor,
    expert_tokens: torch.Tensor,
    total_M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised src->dst index mapping for per-expert padding.

    Returns (src_idx, dst_idx) such that ``buf[dst_idx] = a[src_idx]``
    copies each expert's tokens to padded positions.  O(total_M) work,
    no Python loop over experts.
    """
    # expert id for every token: [0]*n0, [1]*n1, ...
    expert_ids = torch.repeat_interleave(
        torch.arange(expert_tokens.numel(), device=cu_src.device, dtype=torch.int64),
        expert_tokens.long(),
    )
    src_idx = torch.arange(total_M, device=cu_src.device, dtype=torch.int64)
    # Position within each expert: token_pos - cu_src[expert_id]
    within = src_idx - cu_src[expert_ids].long()
    dst_idx = cu_dst[expert_ids].long() + within
    return src_idx, dst_idx


def sgl_mxfp8_gemm_varlen(
    a: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> torch.Tensor:
    """Blockscaled MXFP8 varlen GEMM using sgl-kernel.

    Computes  ``out[m, :] = Σ_k a[m, k] · w[:, k, expert(m)]``

    Handles arbitrary expert token counts via dual-alignment padding:
    - Activations padded to 16  (GEMM requirement)
    - Scale factors padded to 128  (ScaleFactorTileLayout tiles)

    All index computation is vectorised — no Python loops over experts.

    Parameters
    ----------
    a : (total_M, K) bf16 — activation in expert-sorted order.
    w : (N, K, E) bf16   — weight in original SonicMoE layout.
    cu_seqlens_m : (E+1,) int32 — expert boundaries with leading zero.

    Returns
    -------
    out : (total_M, N) bf16
    """
    assert _HAS_SGL, "sgl-kernel not installed"

    total_M, K = a.shape
    N, K_w, E = w.shape
    assert K == K_w, f"K mismatch: activation {K} vs weight {K_w}"
    assert K % 128 == 0, f"K={K} must be divisible by 128 for sgl MXFP8 GEMM"
    assert N % 128 == 0, f"N={N} must be divisible by 128 for sgl MXFP8 GEMM"
    n_groups = K // 32

    expert_tokens = (cu_seqlens_m[1:] - cu_seqlens_m[:-1]).to(torch.int32)

    # ── Step 1: Pad activations to 16 for GEMM M_i alignment ────────
    padded_tokens = (expert_tokens + _GEMM_ALIGN - 1) // _GEMM_ALIGN * _GEMM_ALIGN
    needs_act_padding = not torch.equal(expert_tokens, padded_tokens)

    if needs_act_padding:
        padded_cu = torch.zeros(E + 1, dtype=torch.int32, device=a.device)
        padded_cu[1:] = torch.cumsum(padded_tokens, 0)
        total_padded = padded_cu[-1].item()

        src_idx, dst_idx = _build_pad_indices(
            cu_seqlens_m, padded_cu, expert_tokens, total_M,
        )

        a_padded = torch.zeros(total_padded, K, dtype=a.dtype, device=a.device)
        a_padded[dst_idx] = a[src_idx]
        act_offsets = padded_cu[:-1].to(torch.int32).contiguous()
    else:
        a_padded = a
        total_padded = total_M
        act_offsets = cu_seqlens_m[:-1].to(torch.int32).contiguous()

    # ── Step 2: Quantize with our fast Triton kernel (row-major E8M0) ─
    a_fp8, a_scales_e8m0 = quantize_activation_blockscaled_fast(a_padded)
    a_scales_uint8 = a_scales_e8m0.view(torch.uint8).contiguous()

    # ── Step 3: Build 128-aligned tiled scale buffer ─────────────────
    scale_padded_tokens = (padded_tokens + _TILE_ALIGN - 1) // _TILE_ALIGN * _TILE_ALIGN
    max_sp = int(scale_padded_tokens.max().item()) if total_M > 0 else _TILE_ALIGN
    min_sp = int(scale_padded_tokens.min().item()) if total_M > 0 else _TILE_ALIGN

    # Fast path: all experts same size AND already 128-aligned (no scatter)
    use_fast_tile = (
        not needs_act_padding
        and total_M > 0
        and max_sp == min_sp
        and int(padded_tokens[0].item()) == max_sp  # TPE itself is 128-aligned
    )

    if use_fast_tile:
        # Scales are already contiguous per expert in cu_seqlens_m order;
        # each expert block is exactly max_sp rows -> tile in one call.
        sfa_tiled = rowmajor_to_sgl_tiled(a_scales_uint8, E, max_sp, n_groups)
        blockscale_offsets = act_offsets.clone()
    else:
        # General path: scatter into uniform (E * max_sp) buffer, then tile.
        scale_buf = torch.zeros(E * max_sp, n_groups, dtype=torch.uint8, device=a.device)

        if total_padded > 0:
            padded_expert_ids = torch.repeat_interleave(
                torch.arange(E, device=a.device, dtype=torch.int64),
                padded_tokens.long(),
            )
            padded_rows = torch.arange(total_padded, device=a.device, dtype=torch.int64)
            within_padded = padded_rows - act_offsets[padded_expert_ids].long()
            scale_dst = padded_expert_ids * max_sp + within_padded
            scale_buf[scale_dst] = a_scales_uint8

        sfa_tiled = rowmajor_to_sgl_tiled(scale_buf, E, max_sp, n_groups)
        blockscale_offsets = (
            torch.arange(E, dtype=torch.int32, device=a.device) * max_sp
        ).contiguous()

    # ── Step 4: Problem sizes + precomputed weights ──────────────────
    problem_sizes = torch.empty(E, 3, dtype=torch.int32, device=a.device)
    problem_sizes[:, 0] = padded_tokens
    problem_sizes[:, 1] = N
    problem_sizes[:, 2] = K

    w_fp8_gemm, sfb_tiled = precompute_weight_fp8_sgl(w)

    # ── Step 5: GEMM with separate act/scale offsets ─────────────────
    out_padded = torch.empty(total_padded, N, dtype=torch.bfloat16, device=a.device)
    sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_mm(
        out_padded, a_fp8, w_fp8_gemm, sfa_tiled, sfb_tiled,
        problem_sizes, act_offsets, blockscale_offsets,
    )

    # ── Step 6: Unpad output ─────────────────────────────────────────
    if needs_act_padding:
        out = torch.empty(total_M, N, dtype=torch.bfloat16, device=a.device)
        out[src_idx] = out_padded[dst_idx]
        return out
    return out_padded
