"""Blockscaled FP8 dequantization kernel for interleaved SwiGLU z tensors.

The interleaved layout stores gate/up pairs as z(TK, 2I) where:
  z[:, 0::2] = gate,  z[:, 1::2] = up

In the FP8 frontier path, SwiGLU forward/backward are fused into the CUTLASS
GemmGated/GemmDGated kernels (epilogue). This module only provides the
standalone dequantize utility needed by the BF16 fallback backward path.
"""

import torch

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)

import triton
import triton.language as tl
from ..triton_utils import wrap_triton_kernel
from ._validate import check_tensor, check_divisible

_GROUP_SIZE: tl.constexpr = 32  # 1×32 blockscaled granularity

# ISA tile layout constants (must match blockscaled_fp8_gemm.py)
_SF_VEC_SIZE = 32
_SF_TILE_M = 128
_SF_TILE_K = 128
_SF_TILE_STORAGE = _SF_TILE_M * (_SF_TILE_K // _SF_VEC_SIZE)  # 512


def _div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def _storage_per_batch(rows: int, cols: int) -> int:
    return _div_up(rows, _SF_TILE_M) * _div_up(cols, _SF_TILE_K) * _SF_TILE_STORAGE


# ===================================================================
# Dequantize: fp8(TK, D) + e8m0_scales(TK, D//32) -> bf16(TK, D)
# ===================================================================

@wrap_triton_kernel
@triton.jit
def _dequant_blockscaled_fp8_kernel(
    FP8_ptr, SCALES_ptr, OUT_ptr,
    rows,
    stride_fp8_row, stride_scale_row, stride_out_row,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Dequantize multiple rows of blockscaled FP8 data to bfloat16.

    Each group of GROUP_SIZE fp8 elements shares one e8m0 scale.
    Actual value = fp8_raw * 2^(e8m0 - 127).
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    # int64 row index to avoid row*stride overflow on multi-GB tensors (rows*D > 2^31).
    row_ids = (row_base + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    row_mask = row_ids < rows
    num_groups: tl.constexpr = D // GROUP_SIZE

    for g in range(num_groups):
        e8m0_vals = tl.load(SCALES_ptr + row_ids * stride_scale_row + g, mask=row_mask, other=0)
        scale_f32 = (e8m0_vals.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        col_offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = row_mask[:, None] & (col_offs[None, :] < D)
        fp8_vals = tl.load(FP8_ptr + row_ids[:, None] * stride_fp8_row + col_offs[None, :], mask=mask, other=0.0)
        bf16_vals = (fp8_vals.to(tl.float32) * scale_f32[:, None]).to(tl.bfloat16)
        tl.store(OUT_ptr + row_ids[:, None] * stride_out_row + col_offs[None, :], bf16_vals, mask=mask)


def dequantize_blockscaled_fp8(
    fp8_data: torch.Tensor,
    scales_uint8: torch.Tensor,
) -> torch.Tensor:
    """Dequantize blockscaled FP8 tensor to bfloat16.

    Args:
        fp8_data:     (TK, D) float8_e4m3fn — raw FP8 values.
        scales_uint8: (TK, D//32) uint8 — e8m0 scale per group of 32.

    Returns:
        (TK, D) bfloat16 — properly dequantized values.
    """
    check_tensor(fp8_data, "fp8_data", dtype=torch.float8_e4m3fn, ndim=2, last_stride_1=True)
    check_tensor(scales_uint8, "scales_uint8", dtype=torch.uint8, ndim=2, last_stride_1=True)
    TK, D = fp8_data.shape
    assert D % _GROUP_SIZE == 0, f"D={D} must be multiple of {_GROUP_SIZE}"
    out = torch.empty(TK, D, dtype=torch.bfloat16, device=fp8_data.device)
    BLOCK_ROWS = 16
    _dequant_blockscaled_fp8_kernel[(_div_up(TK, BLOCK_ROWS),)](
        fp8_data, scales_uint8, out,
        TK,
        fp8_data.stride(0), scales_uint8.stride(0), out.stride(0),
        D=D, GROUP_SIZE=_GROUP_SIZE, BLOCK_ROWS=BLOCK_ROWS,
    )
    return out
