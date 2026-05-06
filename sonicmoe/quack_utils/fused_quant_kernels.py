"""API-level fusion: dual_quantize_varlen(dz) + colwise_quantize_and_pack(dout, gather).

Replaces two separate Python calls with a single function that pre-allocates
all 6 output buffers and launches the two original Triton kernels back-to-back.
Bit-exact with sequential execution. Saves ~47µs/iter wall-clock by eliminating
Python dispatch overhead between the two kernel launches.

NCU analysis (Session 61) showed:
- Single-kernel fusion inflates registers 128→142, dropping occupancy 23%→17%
  due to branch divergence between dual and colwise logic. Net: no kernel speedup.
- The original two kernels (dual: 128 regs, colwise: 130 regs) are already at
  optimal register budgets for their respective access patterns.
- The wall-clock savings come from eliminating Python-side overhead (tensor
  allocation, function dispatch, Paddle proxy) between the two calls.

S80 — ISO32 fast-path (env: SONIC_MOE_DZ_ISO32=1):
  Drops the dz_col_fp8 output entirely; dz is quantized with iso32 (single
  amax per 32×32 sub-block). The iso32 invariant guarantees the FP8 bytes
  are identical from row/col perspectives so ONE FP8 buffer serves both
  consumers; dz_row_scales and dz_col_scales contain the same byte values
  in their respective ISA layouts. Phase 0 audit
  (reports/iso32_dz_audit/audit.md) showed iso32 produces identical
  downstream-GEMM RRMSE to 1×32 (ratio = 1.000×) on real Ernie-shape dz.
"""

import os

import torch

from ..triton_utils import wrap_triton_kernel
from ._validate import check_tensor


def _iso32_enabled() -> bool:
    """iso32 dz quant default = ON.

    Phase 0 audit (S80a) proved iso32 produces identical downstream-GEMM
    RRMSE to 1×32 on real Ernie-shape dz (ratio 1.000× across 6 captures).
    Phase 1B microbench: −60 µs/iter (−20.7%) in dz quant stage; multi-layer
    nsys (reports/mfu_s80b/): 2754→2693 µs/iter, 44.91%→45.92% MFU.
    Determinism CI passes. Set SONIC_MOE_DZ_ISO32=0 to fall back to 1×32.
    """
    return os.environ.get("SONIC_MOE_DZ_ISO32", "1") == "1"


def fused_dual_colwise_quantize(
    dz: torch.Tensor,
    dout: torch.Tensor,
    gather_idx: torch.Tensor,
    TK: int,
    dz_dim: int,
    dout_dim: int,
) -> tuple[
    torch.Tensor, torch.Tensor,   # dz_row_fp8, dz_row_scales
    torch.Tensor, torch.Tensor,   # dz_col_fp8, dz_col_scales
    torch.Tensor, torch.Tensor,   # dout_col_fp8, dout_col_scales
]:
    """API-level fusion: dual_quantize_varlen(dz) + colwise_quantize_and_pack(dout, gather).

    Pre-allocates all output buffers, then launches two Triton kernels
    back-to-back with zero Python overhead between them.
    Bit-exact with sequential dual_quantize_varlen + colwise_quantize_and_pack.

    When SONIC_MOE_DZ_ISO32=1, the dz path uses iso32 (one amax per 32×32
    sub-block) and returns ONE FP8 buffer aliased to both dz_row_fp8 and
    dz_col_fp8 outputs (precision-validated by Phase 0 audit).
    """
    check_tensor(dz, "dz", dtype=torch.bfloat16, ndim=2, last_stride_1=True)
    check_tensor(dout, "dout", dtype=torch.bfloat16, ndim=2, last_stride_1=True)
    check_tensor(gather_idx, "gather_idx", ndim=1, stride0_1=True)
    if dz.shape[0] < TK or dz.shape[1] < dz_dim:
        raise ValueError(f"dz shape {dz.shape} too small for TK={TK} dz_dim={dz_dim}")
    if dout.shape[1] < dout_dim:
        raise ValueError(f"dout shape {dout.shape} too small for dout_dim={dout_dim}")
    from .blockscaled_fp8_gemm import (
        _dual_varlen_quantize_kernel,
        _dual_varlen_iso32_quantize_kernel,
        _colwise_quantize_and_pack_kernel,
        _div_up, _SF_TILE_K, _SF_TILE_M, _SF_TILE_STORAGE, _SF_VEC_SIZE,
        _storage_per_batch,
    )
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    device = dz.device
    GROUP_SIZE = _SF_VEC_SIZE   # 32
    BLOCK_DIM = 128

    use_iso32 = _iso32_enabled()

    # ── Pre-allocate dz outputs ──
    dz_row_fp8 = torch.empty(TK, dz_dim, dtype=torch.float8_e4m3fn, device=device)
    if use_iso32:
        # ISO32: one FP8 buffer serves both consumers (byte-identical).
        # Alias dz_col_fp8 = dz_row_fp8 so callers see the same data.
        dz_col_fp8 = dz_row_fp8
    else:
        dz_col_fp8 = torch.empty(TK, dz_dim, dtype=torch.float8_e4m3fn, device=device)
    dout_col_fp8 = torch.empty(TK, dout_dim, dtype=torch.float8_e4m3fn, device=device)

    dz_row_per = _storage_per_batch(TK, dz_dim)
    dz_col_per = _storage_per_batch(dz_dim, TK)
    dout_col_per = _storage_per_batch(dout_dim, TK)

    dz_row_al = (TK % _SF_TILE_M == 0 and dz_dim % _SF_TILE_K == 0)
    dz_col_al = (dz_dim % _SF_TILE_M == 0 and TK % _SF_TILE_K == 0)
    dout_col_al = (dout_dim % _SF_TILE_M == 0 and TK % _SF_TILE_K == 0)

    _alloc = lambda aligned, size: (
        torch.empty((1, size), dtype=torch.uint8, device=device) if aligned
        else torch.full((1, size), 127, dtype=torch.uint8, device=device)
    )
    dz_row_scales = _alloc(dz_row_al, dz_row_per)
    dz_col_scales = _alloc(dz_col_al, dz_col_per)
    dout_col_scales = _alloc(dout_col_al, dout_col_per)

    dz_row_k_tiles = _div_up(dz_dim, _SF_TILE_K)
    dz_col_k_tiles = _div_up(TK, _SF_TILE_K)
    dout_col_k_tiles = _div_up(TK, _SF_TILE_K)

    grid_dual = (_div_up(TK, GROUP_SIZE), _div_up(dz_dim, BLOCK_DIM))

    if use_iso32:
        # ── Kernel 1 (iso32): single FP8 + dual-layout iso32 scales ──
        _dual_varlen_iso32_quantize_kernel[grid_dual](
            dz,
            dz_row_fp8,                # single FP8 output
            dz_row_scales, dz_col_scales,
            TK, dz_dim,
            dz.stride(0), dz.stride(1),
            dz_row_k_tiles, dz_col_k_tiles,
            fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
            GROUP_SIZE=GROUP_SIZE, BLOCK_DIM=BLOCK_DIM,
            SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K,
            SF_TILE_STORAGE=_SF_TILE_STORAGE,
            num_warps=1,
        )
    else:
        # ── Kernel 1 (1×32 dual): row + col separate amaxes ──
        _dual_varlen_quantize_kernel[grid_dual](
            dz,
            dz_row_fp8, dz_row_scales,
            dz_col_fp8, dz_col_scales,
            TK, dz_dim,
            dz.stride(0), dz.stride(1),
            dz_row_k_tiles, dz_col_k_tiles,
            fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
            GROUP_SIZE=GROUP_SIZE, BLOCK_DIM=BLOCK_DIM,
            SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K,
            SF_TILE_STORAGE=_SF_TILE_STORAGE,
            num_warps=1,
        )

    # ── Kernel 2: colwise quant on dout with gather (130 regs, ~18% occ) ──
    grid_col = (_div_up(TK, GROUP_SIZE), _div_up(dout_dim, BLOCK_DIM))
    _colwise_quantize_and_pack_kernel[grid_col](
        dout, gather_idx, dout_col_fp8, dout_col_scales,
        TK, dout_dim,
        dout.stride(0), dout.stride(1),
        dout_col_fp8.stride(0), dout_col_fp8.stride(1),
        dout_col_k_tiles,
        HAS_GATHER=True,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        num_warps=1,
    )

    return (
        dz_row_fp8, dz_row_scales.view(_E8M0),
        dz_col_fp8, dz_col_scales.view(_E8M0),
        dout_col_fp8, dout_col_scales.view(_E8M0),
    )
