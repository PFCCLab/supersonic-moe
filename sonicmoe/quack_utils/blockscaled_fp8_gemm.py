# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils.blockscaled_layout as _upstream_blockscaled_utils
import torch

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)

import triton
import triton.language as tl
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_interface import default_config
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase

from ..functional.fp8_protocol import FP8Protocol, FP8ScaleGranularity, validate_fp8_runtime_support
from ..functional.fp8_quant import quantize_activation_blockwise, round_scale_to_e8m0


# ---------------------------------------------------------------------------
# Rank-aware tile_atom_to_shape_SF with gather_A SFA override
# ---------------------------------------------------------------------------
# The upstream CUTLASS DSL function ``tile_atom_to_shape_SF`` hardcodes a
# 3-element order ``(2, 1, 3)`` which assumes rank-3 tensor shapes.  The
# varlen_m scheduler produces rank-2 activation tensors ``(total_M, K)``,
# causing a compile-time rank mismatch.
#
# We provide a rank-aware replacement that uses ``(2, 1)`` for rank-2 and
# ``(2, 1, 3)`` for rank-3 tensors.  This replacement is installed into the
# upstream module namespace so that ``GemmDefaultSm100.__call__`` (which
# references ``blockscaled_utils.tile_atom_to_shape_SF``) picks it up at
# kernel-tracing time.
#
# Additionally, when gather_A + blockscaled is active, the SFA M-dimension
# must be TK (not T). A thread-local override allows callers to temporarily
# replace the M-dimension in the Shape passed to this function.
# ---------------------------------------------------------------------------

import threading as _threading
from ..triton_utils import wrap_triton_kernel
from ._validate import check_tensor, check_divisible

_SFA_M_OVERRIDE = _threading.local()


def set_sfa_m_override(m_override: int | None) -> None:
    """Set a thread-local M-dimension override for SFA layout derivation.

    When set, the next call to tile_atom_to_shape_SF will use this value
    for the M-dimension instead of the one in Shape. Automatically cleared
    after one use.
    """
    _SFA_M_OVERRIDE.value = m_override
    _SFA_M_OVERRIDE.call_count = 0


@dsl_user_op
def _tile_atom_to_shape_SF_rank_aware(
    Shape: cute.Shape,
    sf_vec_size: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Rank-aware version of ``tile_atom_to_shape_SF``.

    Handles both rank-2 (varlen flat) and rank-3 (grouped/batched) shapes.
    """
    rank = cute.rank(Shape)
    if const_expr(rank == 2):
        # varlen: (total_M, K) -> ((Atom_MN, Rest_MN), (Atom_K, Rest_K))
        sf_layout = cute.tile_to_shape(
            BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1)
        )
    else:
        # grouped/batched: (M, K, L) -> ((Atom_MN, Rest_MN), (Atom_K, Rest_K), RestL)
        sf_layout = cute.tile_to_shape(
            BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1, 3)
        )
    return sf_layout


# Install into upstream module so GemmDefaultSm100 uses it at trace time.
_upstream_blockscaled_utils.tile_atom_to_shape_SF = _tile_atom_to_shape_SF_rank_aware


_SF_VEC_SIZE = 32
_SF_TILE_M = 128
_SF_TILE_K = 128
_SF_TILE_STORAGE = _SF_TILE_M * (_SF_TILE_K // _SF_VEC_SIZE)
_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

_INDEX_CACHE: dict[tuple[int, int, int | None], torch.Tensor] = {}
_WEIGHT_CACHE: dict[
    tuple[int, tuple[int, ...], tuple[int, ...], int | None, int, str, str, str, str],
    tuple[torch.Tensor, torch.Tensor],
] = {}
from sonicmoe.cache_manager import InstrumentedCompileCache as _ICC
_COMPILE_CACHE = _ICC("blockscaled_grouped")
_PAD_PLAN_CACHE: dict = {}       # content-key -> plan
# Fast-path cache: skip validation/tensor-info/compile-key on steady-state calls.
# Maps (total_M, K, H, E, out_dtype, w_shape, w_stride, a_sc_cols, w_sc_shape, dev)
#   -> (compiled_fn, scheduler_args, epi_args)
# Bounded: cleared when exceeding _MAX_FAST_PATH_ENTRIES to prevent memory leak
# from unbounded unique seqlen values in variable-length training.
_MAX_FAST_PATH_ENTRIES = 64
_GEMM_FAST_PATH: dict[tuple, tuple] = {}
_TORCH_TO_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    _E8M0_DTYPE: cutlass.Float8E8M0FNU,
    torch.uint8: cutlass.Uint8,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def _div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def _get_blockscaled_expert_capacity() -> int:
    value = os.getenv("SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY")
    if value is None:
        raise RuntimeError(
            "blockscaled_fp8_gemm requires SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY; "
            "the blockscaled path only accepts externally padded/aligned expert groups"
        )
    try:
        capacity = int(value)
    except ValueError as exc:
        raise RuntimeError("SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY must be an integer") from exc
    if capacity <= 0 or capacity % _SF_TILE_M != 0:
        raise RuntimeError(
            f"SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY must be a positive multiple of {_SF_TILE_M}, got {capacity}"
        )
    return capacity


def _validate_blockscaled_capacity(cu_seqlens_m: torch.Tensor, capacity: int) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    vals = _get_cu_seqlens_cpu(cu_seqlens_m)
    max_expert_tokens = max(vals[i + 1] - vals[i] for i in range(len(vals) - 1)) if len(vals) > 1 else 0
    if max_expert_tokens > capacity:
        raise RuntimeError(
            "SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY is smaller than the routed expert load: "
            f"capacity={capacity}, max_expert_tokens={max_expert_tokens}"
        )


def make_blockscaled_grouped_reverse_scatter_idx(
    flat_sorted_positions: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    capacity: Optional[int] = None,
    expert_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if flat_sorted_positions.ndim != 1:
        raise ValueError(f"expected 1D flat_sorted_positions, got shape {tuple(flat_sorted_positions.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be a 1D int32 tensor")
    if flat_sorted_positions.device != cu_seqlens_m.device:
        raise ValueError("flat_sorted_positions and cu_seqlens_m must be on the same device")

    if capacity is None:
        capacity = _get_blockscaled_expert_capacity()
    flat_sorted_positions_i64 = flat_sorted_positions.to(torch.int64)
    if expert_ids is None:
        expert_ids_i64 = torch.searchsorted(cu_seqlens_m[1:], flat_sorted_positions_i64, right=True)
    else:
        if expert_ids.ndim != 1:
            raise ValueError(f"expected 1D expert_ids, got shape {tuple(expert_ids.shape)}")
        if expert_ids.device != flat_sorted_positions.device:
            raise ValueError("expert_ids and flat_sorted_positions must be on the same device")
        if expert_ids.numel() != flat_sorted_positions.numel():
            raise ValueError("expert_ids must have the same number of elements as flat_sorted_positions")
        expert_ids_i64 = expert_ids.to(torch.int64)
    expert_starts = cu_seqlens_m.index_select(0, expert_ids_i64).to(torch.int64)
    grouped_positions = expert_ids_i64 * capacity + (flat_sorted_positions_i64 - expert_starts)
    return grouped_positions.to(torch.int32)


def _storage_per_batch(rows: int, cols: int) -> int:
    return _div_up(rows, _SF_TILE_M) * _div_up(cols, _SF_TILE_K) * _SF_TILE_STORAGE


def _scale_pack_index(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    key = (rows, cols, device.index)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    scale_cols = _div_up(cols, _SF_VEC_SIZE)
    k_tiles = _div_up(cols, _SF_TILE_K)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    scale_block_ids = torch.arange(scale_cols, device=device, dtype=torch.int64).unsqueeze(0)

    row_tiles = row_ids // _SF_TILE_M
    row_in_tile = row_ids % _SF_TILE_M
    k_tiles_idx = scale_block_ids // (_SF_TILE_K // _SF_VEC_SIZE)
    k_in_tile = scale_block_ids % (_SF_TILE_K // _SF_VEC_SIZE)

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * _SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    index = (tile_base + row_base + k_in_tile).reshape(-1).contiguous()
    _INDEX_CACHE[key] = index
    return index


def pack_blockscaled_1x32_scales(scales: torch.Tensor, cols: int) -> torch.Tensor:
    if scales.ndim not in (2, 3):
        raise ValueError(f"expected 2D or 3D scales, got shape {tuple(scales.shape)}")
    if scales.device.type != "cuda":
        raise ValueError("blockscaled scale packing requires CUDA tensors")

    if scales.ndim == 2:
        scales = scales.unsqueeze(0)

    batches, rows, _ = scales.shape
    per_batch_storage = _storage_per_batch(rows, cols)
    packed = torch.ones((batches, per_batch_storage), device=scales.device, dtype=scales.dtype)
    flat_index = _scale_pack_index(rows, cols, scales.device)
    batch_offsets = torch.arange(batches, device=scales.device, dtype=torch.int64).unsqueeze(1) * per_batch_storage
    packed.view(-1)[(batch_offsets + flat_index.unsqueeze(0)).reshape(-1)] = scales.reshape(-1)
    return packed


def _blockscaled_protocol(protocol: FP8Protocol) -> FP8Protocol:
    return replace(protocol, scale_granularity=FP8ScaleGranularity.BLOCK_1X32)


def _is_runtime_fp8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in {torch.float8_e4m3fn, _E8M0_DTYPE}


def _make_cute_tensor_dynamic(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    if _is_runtime_fp8_tensor(tensor):
        storage = tensor.detach().view(torch.uint8)
        cute_tensor = from_dlpack(storage, assumed_align=16)
        cute_tensor.element_type = _TORCH_TO_CUTLASS_DTYPE[tensor.dtype]
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def _tensor_version(t: torch.Tensor) -> int:
    """Get tensor in-place version, compatible with both PyTorch and Paddle proxy."""
    v = getattr(t, '_inplace_version', None)
    if v is not None:
        return v() if callable(v) else v
    return getattr(t, '_version', 0)


def _get_raw_cuda_stream(device=None) -> int:
    """Get raw CUDA stream pointer, compatible with both PyTorch and Paddle proxy."""
    s = torch.cuda.current_stream(device)
    if hasattr(s, 'stream_base'):
        return s.stream_base.raw_stream  # Paddle proxy
    return s.cuda_stream  # native PyTorch


def _weight_cache_key(
    weight: torch.Tensor,
    protocol: FP8Protocol,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], int | None, int, str, str, str, str]:
    return (
        weight.data_ptr(),
        id(weight),  # guards against CUDA memory reuse after del
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.device.index,
        _tensor_version(weight),
        protocol.activation_dtype.value,
        protocol.scale_encoding.value,
        protocol.scale_granularity.value,
        protocol.backend.value,
    )




def _quantize_w2_cached(
    w2: torch.Tensor,
    protocol: FP8Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Key on storage identity + version + view layout so that repeated
    # permute() calls on the same underlying Parameter hit cache.
    key = (
        w2.data_ptr(),
        _tensor_version(w2),
        tuple(w2.shape),
        tuple(w2.stride()),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    weight_ehi = w2.permute(2, 0, 1).contiguous()
    weight_fp8_ehi, packed_scales = _quantize_weight_3d_triton(weight_ehi)
    result = (weight_fp8_ehi, packed_scales)
    if len(_WEIGHT_CACHE) > 2:
        _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = result
    return result


def clear_blockscaled_fp8_weight_cache() -> None:
    _WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE.clear()
    _PAD_PLAN_CACHE.clear()


def clear_fused_weight_cache() -> None:
    """Clear the fused-gated/dgated weight cache to eagerly release GPU memory.

    In training the optimizer step invalidates these entries anyway
    (via ``w._inplace_version()``), so clearing between forward and backward is free.
    Saves ~74 MiB (w1 fused cache) during the down-projection and backward.
    """
    _FUSED_WEIGHT_CACHE.clear()


def evict_fp8_weight_cache_entry(w: torch.Tensor) -> None:
    """Remove cached FP8 data for *w* from all blockscaled weight caches.

    Call this right after a cached FP8 weight has been consumed to release
    GPU memory eagerly instead of waiting for the global cache clear at the
    next optimizer step.  In training the cache would miss anyway (because
    ``w._inplace_version()`` increments on each in-place update), so the eviction
    adds zero overhead.
    """
    key = (
        w.data_ptr(),
        _tensor_version(w),
        tuple(w.shape),
        tuple(w.stride()),
    )
    _FUSED_WEIGHT_CACHE.pop(key, None)
    _VARLEN_WEIGHT_CACHE.pop(key, None)
    _WEIGHT_CACHE.pop(key, None)


def _get_cu_seqlens_cpu(cu_seqlens: torch.Tensor) -> tuple:
    """Return cu_seqlens values as a Python tuple, cached on the tensor object.

    Exactly ONE D2H sync per tensor object lifetime.  All subsequent calls
    with the same tensor are pure Python attribute lookups — zero GPU sync.
    """
    cached = getattr(cu_seqlens, '_cached_cpu_tuple', None)
    if cached is not None:
        return cached
    cpu_tuple = tuple(cu_seqlens.tolist())
    cu_seqlens._cached_cpu_tuple = cpu_tuple
    return cpu_tuple


def _get_padding_plan(
    cu_seqlens_m: torch.Tensor,
    total_M: int,
) -> tuple[bool, torch.Tensor | None, int, torch.Tensor | None]:
    """Compute (and cache) CTA-tile padding plan for a cu_seqlens_m tensor.

    Uses tensor-attribute caching: zero D2H sync after first call per tensor.
    Even on cache miss, all decisions are made from CPU tuple — no `.item()`.
    """
    cpu_tuple = _get_cu_seqlens_cpu(cu_seqlens_m)
    cached = _PAD_PLAN_CACHE.get(cpu_tuple)
    if cached is not None:
        return cached

    # Pure Python arithmetic — zero GPU operations
    cpu_lens = [cpu_tuple[i + 1] - cpu_tuple[i] for i in range(len(cpu_tuple) - 1)]
    remainders = [s % _SF_TILE_M for s in cpu_lens]
    needs_pad = any(r > 0 for r in remainders)

    if needs_pad:
        padded_lens = [s + (_SF_TILE_M - r) % _SF_TILE_M for s, r in zip(cpu_lens, remainders)]
        padded_cu_list = [0]
        for pl in padded_lens:
            padded_cu_list.append(padded_cu_list[-1] + pl)
        padded_total = padded_cu_list[-1]

        # H2D transfer (async, no sync) + GPU scatter
        padded_cu = torch.tensor(padded_cu_list, dtype=torch.int32, device=cu_seqlens_m.device)
        token_idx = torch.arange(total_M, device=cu_seqlens_m.device, dtype=torch.int64)
        expert_ids = torch.searchsorted(cu_seqlens_m, token_idx, right=True) - 1
        local_off = token_idx - cu_seqlens_m[expert_ids].to(torch.int64)
        dst_idx = padded_cu[expert_ids].to(torch.int64) + local_off
        plan = (True, padded_cu, padded_total, dst_idx)
    else:
        plan = (False, None, 0, None)

    if len(_PAD_PLAN_CACHE) > 16:
        _PAD_PLAN_CACHE.clear()
    _PAD_PLAN_CACHE[cpu_tuple] = plan
    return plan


def prefetch_blockscaled_w2_fp8(
    w2: torch.Tensor,
    protocol: FP8Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    if w2.ndim != 3:
        raise ValueError(f"expected w2 with shape (H, I, E), got {tuple(w2.shape)}")
    if w2.device.type != "cuda":
        raise ValueError("blockscaled FP8 weight prefetch requires CUDA weights")

    blockscaled_protocol = validate_fp8_runtime_support(_blockscaled_protocol(protocol), device=w2.device)
    return _quantize_w2_cached(w2, blockscaled_protocol)


# ---------------------------------------------------------------------------
# Fast fused blockscaled quantization kernel for flat 2D activations
# ---------------------------------------------------------------------------
# Single-pass: read bf16 -> compute per-32 amax -> E8M0 scale -> quantize -> write fp8 + scales
# Replaces the Python quantize_activation_blockwise which does ~8 separate kernel launches.

@wrap_triton_kernel
@triton.jit
def _quantize_flat_blockscaled_kernel(
    src_ptr,
    dst_fp8_ptr,
    dst_scale_ptr,
    rows,
    cols,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    scale_stride_row,
    scale_stride_col,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """Quantize a flat (M, K) bf16 tensor to blockscaled FP8 with 1×GROUP_SIZE scales.

    Each program processes BLOCK_ROWS × GROUPS_PER_BLOCK groups.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    group_base = tl.program_id(1) * GROUPS_PER_BLOCK

    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    for g in range(GROUPS_PER_BLOCK):
        group_id = group_base + g
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        src_ptrs = src_ptr + row_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        block_amax = tl.max(tl.abs(values), axis=1)

        # Pure-integer E8M0 computation (matches CUTLASS/sgl convention):
        # E8M0 = biased_exponent(amax) - 8 + carry
        # where carry = 1 iff mantissa(amax) > mantissa(fp8_max=448=1.75*2^8)
        # This avoids all log2/ceil/exp2 float precision issues.
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        # fp8_max = 448 = 1.75 * 2^8; mantissa of 1.75 in IEEE754 = 0x600000
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

        # Quant scale = 2^(127 - e8m0) = 1 / dequant_scale
        # Construct exact power-of-2 float via bit manipulation.
        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)
        scale_ptrs = dst_scale_ptr + row_ids * scale_stride_row + group_id * scale_stride_col
        tl.store(scale_ptrs, e8m0_byte, mask=row_mask_1d)


@wrap_triton_kernel
@triton.jit
def _quantize_flat_v2_kernel(
    src_ptr, dst_fp8_ptr, dst_scale_ptr,
    rows, cols,
    src_stride_row, src_stride_col,
    dst_stride_row, dst_stride_col,
    scale_stride_row, scale_stride_col,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,       # 32
    TILE_ROWS: tl.constexpr,        # 128
    TILE_COLS: tl.constexpr,        # 256
):
    """High-BW blockscaled quantize: large tiles, vectorized, pipelined.

    Each CTA processes TILE_ROWS × TILE_COLS elements.
    Within each tile, processes TILE_COLS/GROUP_SIZE groups per row.
    Uses vectorized loads for full memory bus utilization.

    Grid: (ceil(rows/TILE_ROWS), ceil(cols/TILE_COLS))
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_base = pid_row * TILE_ROWS
    col_base = pid_col * TILE_COLS

    GROUPS_PER_TILE: tl.constexpr = TILE_COLS // GROUP_SIZE  # 256/32 = 8

    row_offs = row_base + tl.arange(0, TILE_ROWS)
    row_mask = row_offs < rows

    for g in tl.range(0, GROUPS_PER_TILE):
        col_offs = col_base + g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offs[None, :] < cols
        mask = row_mask[:, None] & col_mask

        # Vectorized load: (TILE_ROWS, GROUP_SIZE) bf16 tile
        src_ptrs = src_ptr + row_offs[:, None] * src_stride_row + col_offs[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # E8M0 per-row amax (reduce over GROUP_SIZE=32 cols)
        amax = tl.max(tl.abs(values), axis=1)  # (TILE_ROWS,)
        amax_i32 = amax.to(tl.int32, bitcast=True)
        bexp = (amax_i32 >> 23) & 0xFF
        mant = amax_i32 & 0x7FFFFF
        carry = tl.where(mant > 0x600000, 1, 0)
        e8m0 = tl.where(bexp > 0, bexp - 8 + carry, 0)
        e8m0_byte = tl.maximum(e8m0, 0).to(tl.uint8)

        # Quant scale
        qexp = tl.maximum(tl.minimum(254 - e8m0, 254), 1)
        qscale = (qexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * qscale[:, None]).to(tl.float8e4nv)

        # Vectorized store: fp8 data
        dst_ptrs = dst_fp8_ptr + row_offs[:, None] * dst_stride_row + col_offs[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # Store scale byte
        group_id = (col_base // GROUP_SIZE) + g
        scale_ptrs = dst_scale_ptr + row_offs * scale_stride_row + group_id * scale_stride_col
        tl.store(scale_ptrs, e8m0_byte, mask=row_mask)


def quantize_activation_blockscaled_fast(
    x: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast fused 1×group_size blockscaled quantization using a single Triton kernel.

    Returns (fp8_data, e8m0_scales).
    """
    check_tensor(x, "x", dtype=torch.bfloat16, ndim=2)
    assert x.is_contiguous(), "Input must be contiguous"
    M, K = x.shape
    num_groups = _div_up(K, group_size)

    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    scale_out = torch.empty(M, num_groups, dtype=torch.uint8, device=x.device)

    TILE_ROWS = 128  # Larger tile: fewer CTAs, better wave occupancy
    TILE_COLS = min(K, 256)
    grid = (_div_up(M, TILE_ROWS), _div_up(K, TILE_COLS))
    _quantize_flat_v2_kernel[grid](
        x,
        fp8_out,
        scale_out,
        M,
        K,
        x.stride(0),
        x.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        scale_out.stride(0),
        scale_out.stride(1),
        fp8_max=_FP8_E4M3_MAX,
        GROUP_SIZE=group_size,
        TILE_ROWS=TILE_ROWS,
        TILE_COLS=TILE_COLS,
    )
    return fp8_out, scale_out


# ---------------------------------------------------------------------------
# Fused z-save + y1-quant kernel
# ---------------------------------------------------------------------------
# After the forward GemmGated GEMM, we need to:
#   1. quantize z (TK, 2I) -> FP8 with raw E8M0 scales  (for backward z-save)
#   2. quantize y1 (TK, I) -> FP8 with ISA-packed scales (for FP8 down-proj)
# These share the same row dimension (TK) and are independent. Fusing them
# into a single kernel eliminates one launch overhead and improves SM
# utilization while both tensors are L2-hot from the GEMM output.
# ---------------------------------------------------------------------------

@wrap_triton_kernel
@triton.jit
def _fused_z_save_y1_quant_kernel(
    # --- z tensor (TK, 2I) -> raw scales ---
    z_src_ptr,
    z_dst_fp8_ptr,
    z_dst_scale_ptr,
    z_cols,                # 2I
    z_src_stride_row,
    z_src_stride_col,
    z_dst_stride_row,
    z_dst_stride_col,
    z_scale_stride_row,
    z_scale_stride_col,
    z_col_blocks,          # ceil(z_num_groups / Z_GROUPS_PER_BLOCK)
    # --- y1 tensor (TK, I) -> ISA-packed scales ---
    y1_src_ptr,
    y1_dst_fp8_ptr,
    y1_dst_packed_scale_ptr,
    y1_cols,               # I
    y1_src_stride_row,
    y1_src_stride_col,
    y1_dst_stride_row,
    y1_dst_stride_col,
    y1_k_tiles,
    # --- shared ---
    rows,                  # TK
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    Z_GROUPS_PER_BLOCK: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    Y1_GROUPS_PER_K_TILE: tl.constexpr = 4,
    SAFE_INT64: tl.constexpr = False,
):
    """Fused quantization of z (raw E8M0) and y1 (ISA-packed E8M0).

    2D grid: (row_blocks, z_col_blocks + y1_k_tiles).
    pid_1 < z_col_blocks  -> process Z_GROUPS_PER_BLOCK groups of z (raw scales)
    pid_1 >= z_col_blocks -> process 1 k-tile of y1 (ISA-packed scales)

    Ernie shape: grid=(2048, 8+12)=(2048, 20) = 40960 blocks
    vs old 1D: (2048,) = 2048 blocks -> 20× more parallelism.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    work_id = tl.program_id(1)
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows
    if SAFE_INT64:
        row_ids = row_ids.to(tl.int64)

    if work_id < z_col_blocks:
        # ---- Z path: quantize Z_GROUPS_PER_BLOCK groups with raw E8M0 scales ----
        # Scale store optimization: accumulate 4 consecutive group bytes into
        # uint32, then write as int32 into row-major scale buffer. This reduces
        # 12 byte stores to 3 uint32 stores per block, improving store throughput.
        group_base = work_id * Z_GROUPS_PER_BLOCK

        for g4 in range(Z_GROUPS_PER_BLOCK // 4):
            packed_z_scale_i32 = tl.zeros([BLOCK_ROWS], dtype=tl.int32)
            for g_inner in range(4):
                g = g4 * 4 + g_inner
                group_id = group_base + g
                col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
                col_mask = col_offsets[None, :] < z_cols
                mask = row_mask_1d[:, None] & col_mask

                src_ptrs = z_src_ptr + row_ids[:, None] * z_src_stride_row + col_offsets[None, :] * z_src_stride_col
                values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

                block_amax = tl.max(tl.abs(values), axis=1)
                amax_bits = block_amax.to(tl.int32, bitcast=True)
                biased_exp = (amax_bits >> 23) & 0xFF
                mantissa_bits = amax_bits & 0x7FFFFF
                carry = tl.where(mantissa_bits > 0x600000, 1, 0)
                e8m0_i32 = biased_exp - 8 + carry
                e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
                e8m0_byte = tl.maximum(e8m0_i32, 0)

                quant_biased_exp = 254 - e8m0_i32
                quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
                quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

                quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
                dst_ptrs = z_dst_fp8_ptr + row_ids[:, None] * z_dst_stride_row + col_offsets[None, :] * z_dst_stride_col
                tl.store(dst_ptrs, quantized, mask=mask)

                # Pack scale byte into uint32 (little-endian: byte 0 at bits[0:8])
                packed_z_scale_i32 = packed_z_scale_i32 | ((e8m0_byte & 0xFF) << (g_inner * 8))

            # Write 4 packed scale bytes as uint32 into row-major scale buffer.
            # Address = base + row * num_groups + (group_base + g4*4), cast to int32.
            z_scale_i32_ptr = z_dst_scale_ptr.to(tl.pointer_type(tl.int32))
            # z_scale_stride_row is num_groups in bytes; divide by 4 for int32 stride
            z_scale_i32_stride_row = z_scale_stride_row // 4
            scale_i32_offset = (group_base + g4 * 4) // 4
            tl.store(
                z_scale_i32_ptr + row_ids * z_scale_i32_stride_row + scale_i32_offset,
                packed_z_scale_i32,
                mask=row_mask_1d,
            )
    else:
        # ---- Y1 path: quantize 1 k-tile (Y1_GROUPS_PER_K_TILE groups) with ISA-packed scales ----
        k_tile_idx = work_id - z_col_blocks

        # Pre-compute row-dependent ISA layout
        row_tiles = row_ids // SF_TILE_M
        row_in_tile = row_ids % SF_TILE_M
        row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

        packed_scale_i32 = tl.zeros([BLOCK_ROWS], dtype=tl.int32)

        for g in range(Y1_GROUPS_PER_K_TILE):
            group_id = k_tile_idx * Y1_GROUPS_PER_K_TILE + g
            col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            col_mask = col_offsets[None, :] < y1_cols
            mask = row_mask_1d[:, None] & col_mask

            src_ptrs = y1_src_ptr + row_ids[:, None] * y1_src_stride_row + col_offsets[None, :] * y1_src_stride_col
            values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

            block_amax = tl.max(tl.abs(values), axis=1)
            amax_bits = block_amax.to(tl.int32, bitcast=True)
            biased_exp = (amax_bits >> 23) & 0xFF
            mantissa_bits = amax_bits & 0x7FFFFF
            carry = tl.where(mantissa_bits > 0x600000, 1, 0)
            e8m0_i32 = biased_exp - 8 + carry
            e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
            e8m0_clamped = tl.maximum(e8m0_i32, 0)

            quant_biased_exp = 254 - e8m0_i32
            quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
            quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

            dst_ptrs = y1_dst_fp8_ptr + row_ids[:, None] * y1_dst_stride_row + col_offsets[None, :] * y1_dst_stride_col
            tl.store(dst_ptrs, quantized, mask=mask)

            packed_scale_i32 = packed_scale_i32 | ((e8m0_clamped & 0xFF) << (g * 8))

        # Write 4 scale bytes as single uint32 (row_base_offset is always 4-byte aligned)
        tile_base = (row_tiles * y1_k_tiles + k_tile_idx) * SF_TILE_STORAGE
        packed_offset_i32 = (tile_base + row_base_offset) // 4
        scale_ptr_i32 = y1_dst_packed_scale_ptr.to(tl.pointer_type(tl.int32))
        tl.store(scale_ptr_i32 + packed_offset_i32, packed_scale_i32, mask=row_mask_1d)


def fused_z_save_y1_quant(
    z: torch.Tensor,
    y1: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused quantization of z (raw scales) and y1 (ISA-packed scales).

    Replaces two separate kernel launches:
      quantize_activation_blockscaled_fast(z)  -> z_fp8, z_raw_scales
      quantize_and_pack_activation(y1)          -> y1_fp8, y1_packed_scales

    Uses a 2D grid (row_blocks, z_col_blocks + y1_k_tiles) so that z and y1
    quantization run as independent column-parallel work units.  Each block
    checks pid_1 < z_col_blocks to decide whether it processes z or y1.

    Ernie shape: grid = (2048, 8+12) = 40960 blocks
    vs separate:  16384 + 24576 = 40960 blocks (same parallelism, 1 launch)
    vs old 1D:   (2048,) = 2048 blocks (20× less parallelism)

    Parameters
    ----------
    z  : Tensor (TK, 2I) bf16 — GemmGated pre-activation output.
    y1 : Tensor (TK, I) bf16 — SwiGLU output.

    Returns
    -------
    z_fp8           : Tensor (TK, 2I) float8_e4m3fn
    z_raw_scales    : Tensor (TK, num_groups_z) float8_e8m0fnu (raw layout)
    y1_fp8          : Tensor (TK, I) float8_e4m3fn
    y1_packed_scales: Tensor (1, packed_size) float8_e8m0fnu (ISA layout)
    """
    assert z.is_contiguous(), "z must be contiguous"
    assert y1.is_contiguous(), "y1 must be contiguous"

    TK = z.shape[0]
    z_K = z.shape[1]      # 2I
    y1_K = y1.shape[1]    # I
    assert y1.shape[0] == TK, "z and y1 must have the same number of rows"

    # --- z outputs (raw scales) ---
    z_num_groups = _div_up(z_K, group_size)
    Z_GROUPS_PER_BLOCK = min(z_num_groups, 12)
    assert z_num_groups % 4 == 0, f"z_num_groups={z_num_groups} must be divisible by 4 for packed int32 scale writes"
    assert Z_GROUPS_PER_BLOCK % 4 == 0, f"Z_GROUPS_PER_BLOCK={Z_GROUPS_PER_BLOCK} must be divisible by 4"
    z_col_blocks = _div_up(z_num_groups, Z_GROUPS_PER_BLOCK)
    z_fp8 = torch.empty(TK, z_K, dtype=torch.float8_e4m3fn, device=z.device)
    z_scale_out = torch.empty(TK, z_num_groups, dtype=torch.uint8, device=z.device)

    # --- y1 outputs (ISA-packed scales) ---
    y1_k_tiles = _div_up(y1_K, _SF_TILE_K)
    y1_fp8 = torch.empty(TK, y1_K, dtype=torch.float8_e4m3fn, device=y1.device)
    y1_per_batch_storage = _storage_per_batch(TK, y1_K)
    if TK % _SF_TILE_M == 0 and y1_K % _SF_TILE_K == 0:
        y1_packed_scales = torch.empty(
            (1, y1_per_batch_storage), dtype=torch.uint8, device=y1.device
        )
    else:
        y1_packed_scales = torch.full(
            (1, y1_per_batch_storage), 127, dtype=torch.uint8, device=y1.device
        )

    y1_groups_per_k_tile = _SF_TILE_K // group_size  # 4 for default

    BLOCK_ROWS = 32
    # INT32 overflow guard: dispatch to int64 branch when row_id * stride > 2^31
    _max_stride = max(z.stride(0), z_fp8.stride(0), y1.stride(0), y1_fp8.stride(0))
    _needs_int64 = int((TK - 1) * _max_stride > 2**31 - 1)

    # 2D grid: dim0=row_blocks, dim1=z_col_blocks + y1_k_tiles
    grid = (_div_up(TK, BLOCK_ROWS), z_col_blocks + y1_k_tiles)
    _fused_z_save_y1_quant_kernel[grid](
        # z pointers
        z,
        z_fp8,
        z_scale_out,
        z_K,
        z.stride(0),
        z.stride(1),
        z_fp8.stride(0),
        z_fp8.stride(1),
        z_scale_out.stride(0),
        z_scale_out.stride(1),
        z_col_blocks,
        # y1 pointers
        y1,
        y1_fp8,
        y1_packed_scales,
        y1_K,
        y1.stride(0),
        y1.stride(1),
        y1_fp8.stride(0),
        y1_fp8.stride(1),
        y1_k_tiles,
        # shared
        TK,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        Z_GROUPS_PER_BLOCK=Z_GROUPS_PER_BLOCK,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        Y1_GROUPS_PER_K_TILE=y1_groups_per_k_tile,
        SAFE_INT64=_needs_int64,
    )
    return (
        z_fp8,
        z_scale_out.view(_E8M0_DTYPE),
        y1_fp8,
        y1_packed_scales.view(_E8M0_DTYPE),
    )


@triton.jit
def _pack_quantize_expert_segments_kernel(
    src_ptr,
    dst_fp8_ptr,
    dst_scale_ptr,
    offsets_ptr,
    src_stride_row,
    src_stride_col,
    dst_stride_expert,
    dst_stride_row,
    dst_stride_col,
    dst_scale_stride_expert,
    dst_scale_stride_row,
    dst_scale_stride_col,
    cols,
    fp8_max,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    scale_col = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = scale_col * BLOCK_N + tl.arange(0, BLOCK_N)

    expert_start = tl.load(offsets_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(offsets_ptr + expert_id + 1).to(tl.int32)
    expert_len = expert_end - expert_start

    src_rows = expert_start + row_offsets[:, None]
    src_cols = col_offsets[None, :]
    row_mask = row_offsets[:, None] < expert_len
    col_mask = src_cols < cols
    mask = row_mask & col_mask

    src_ptrs = src_ptr + src_rows * src_stride_row + src_cols * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    block_amax = tl.max(tl.abs(values), axis=1)
    positive_scale = tl.where(block_amax > 0, block_amax / fp8_max, 1.0)
    exponent = tl.ceil(tl.log2(positive_scale))
    dequant_scale = tl.where(block_amax > 0, tl.exp2(exponent), 1.0)
    quant_scale = tl.where(block_amax > 0, tl.exp2(-exponent), 1.0)

    quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)
    dst_ptrs = (
        dst_fp8_ptr
        + expert_id * dst_stride_expert
        + row_offsets[:, None] * dst_stride_row
        + src_cols * dst_stride_col
    )
    tl.store(dst_ptrs, quantized, mask=mask)

    scale_ptrs = (
        dst_scale_ptr
        + expert_id * dst_scale_stride_expert
        + row_offsets * dst_scale_stride_row
        + scale_col * dst_scale_stride_col
    )
    tl.store(scale_ptrs, dequant_scale, mask=row_offsets < expert_len)


@wrap_triton_kernel
@triton.jit
def _pack_expert_segments_kernel(
    src_ptr,
    dst_ptr,
    offsets_ptr,
    src_stride_row,
    src_stride_col,
    dst_stride_expert,
    dst_stride_row,
    dst_stride_col,
    cols,
    capacity,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    expert_start = tl.load(offsets_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(offsets_ptr + expert_id + 1).to(tl.int32)
    expert_len = expert_end - expert_start

    src_rows = expert_start + row_offsets[:, None]
    src_cols = col_offsets[None, :]
    row_mask = row_offsets[:, None] < expert_len
    col_mask = src_cols < cols
    mask = row_mask & col_mask

    src_ptrs = src_ptr + src_rows * src_stride_row + src_cols * src_stride_col
    values = tl.load(src_ptrs, mask=mask, other=0.0)

    dst_rows = row_offsets[:, None]
    dst_ptrs = (
        dst_ptr
        + expert_id * dst_stride_expert
        + dst_rows * dst_stride_row
        + src_cols * dst_stride_col
    )
    store_mask = (dst_rows < capacity) & col_mask
    tl.store(dst_ptrs, values, mask=store_mask)


@wrap_triton_kernel
@triton.jit
def _unpack_expert_segments_kernel(
    src_ptr,
    dst_ptr,
    offsets_ptr,
    src_stride_expert,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    expert_start = tl.load(offsets_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(offsets_ptr + expert_id + 1).to(tl.int32)
    expert_len = expert_end - expert_start

    row_mask = row_offsets[:, None] < expert_len
    col_mask = col_offsets[None, :] < cols
    mask = row_mask & col_mask

    src_ptrs = (
        src_ptr
        + expert_id * src_stride_expert
        + row_offsets[:, None] * src_stride_row
        + col_offsets[None, :] * src_stride_col
    )
    values = tl.load(src_ptrs, mask=mask, other=0.0)

    dst_rows = expert_start + row_offsets[:, None]
    dst_ptrs = dst_ptr + dst_rows * dst_stride_row + col_offsets[None, :] * dst_stride_col
    tl.store(dst_ptrs, values, mask=mask)


# ---------------------------------------------------------------------------
# Fused transpose + quantize + ISA-pack kernel for weight-gradient data prep
# ---------------------------------------------------------------------------
# Single pass: read flat expert-sorted (TK, dim) bf16 ->
#   transpose+quantize -> write (E*dim, capacity) fp8 + ISA-packed scales.
# Optional gather_idx fuses the gather step too.
# Eliminates the separate pack_blockscaled_1x32_scales call entirely.

@wrap_triton_kernel
@triton.jit
def _fused_transpose_quantize_kernel(
    src_ptr,           # (TK, dim) bf16
    gather_idx_ptr,    # (TK,) int32 gather index, or unused if HAS_GATHER=False
    dst_fp8_ptr,       # (E*dim, capacity) fp8
    dst_packed_ptr,    # (E, per_batch_storage) u8 — ISA-packed scales
    dim,
    capacity,
    per_batch_storage,
    src_stride_row,
    src_stride_col,
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,      # 32
    BLOCK_DIM: tl.constexpr,       # 128
    # ISA layout constants
    SF_TILE_M: tl.constexpr,       # 128
    SF_TILE_K: tl.constexpr,       # 128
    SF_TILE_STORAGE: tl.constexpr, # 512
):
    """Each block processes one (expert, dim_block, quant_group) tile.

    Reads a (GROUP_SIZE, BLOCK_DIM) tile from the input, transposes to
    (BLOCK_DIM, GROUP_SIZE), quantizes with E8M0 blockscaling, and writes
    FP8 data + ISA-packed scales in a single fused pass.
    """
    pid_row = tl.program_id(0)       # expert_id * num_dim_blocks + dim_block
    pid_group = tl.program_id(1)     # quant group along capacity

    num_dim_blocks = tl.cdiv(dim, BLOCK_DIM)
    expert_id = pid_row // num_dim_blocks
    dim_block = pid_row % num_dim_blocks

    cap_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    # Compute source rows — optionally indirect through gather_idx
    flat_token_ids = expert_id * capacity + cap_offs
    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + flat_token_ids).to(tl.int64)
    else:
        src_rows = flat_token_ids.to(tl.int64)

    # Load (GROUP_SIZE, BLOCK_DIM) tile
    dim_mask = dim_offs[None, :] < dim
    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    values = tl.load(src_ptrs, mask=dim_mask, other=0.0).to(tl.float32)

    # Transpose in registers: (GROUP_SIZE, BLOCK_DIM) -> (BLOCK_DIM, GROUP_SIZE)
    values_t = tl.trans(values)

    # Blockscaled E8M0 quantization (per row of BLOCK_DIM, over GROUP_SIZE elements)
    block_amax = tl.max(tl.abs(values_t), axis=1)

    amax_bits = block_amax.to(tl.int32, bitcast=True)
    biased_exp = (amax_bits >> 23) & 0xFF
    mantissa_bits = amax_bits & 0x7FFFFF
    carry = tl.where(mantissa_bits > 0x600000, 1, 0)
    e8m0_i32 = biased_exp - 8 + carry
    e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
    e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

    quant_biased_exp = 254 - e8m0_i32
    quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
    quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

    quantized = (values_t * quant_scale[:, None]).to(tl.float8e4nv)

    # Write FP8 data: dst_fp8[expert_id*dim + dim_offs, group*GROUP_SIZE + j]
    out_rows = expert_id * dim + dim_offs
    out_cols = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    fp8_ptrs = dst_fp8_ptr + out_rows[:, None] * capacity + out_cols[None, :]
    out_mask = dim_offs[:, None] < dim
    tl.store(fp8_ptrs, quantized, mask=out_mask)

    # Write ISA-packed scales directly (inline index computation).
    # ISA layout: tile_base + row_base + k_in_tile
    # where:
    #   row = dim_offs (each element in BLOCK_DIM)
    #   scale_block = pid_group (which capacity group)
    #   row_tiles = row // SF_TILE_M
    #   row_in_tile = row % SF_TILE_M
    #   k_tiles_idx = scale_block // (SF_TILE_K // GROUP_SIZE)
    #   k_in_tile = scale_block % (SF_TILE_K // GROUP_SIZE)
    #   tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    #   row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    #   index = tile_base + row_base + k_in_tile
    k_tiles: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    groups_per_k_tile: tl.constexpr = SF_TILE_K // GROUP_SIZE  # 4

    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    k_tiles_idx = pid_group // groups_per_k_tile
    k_in_tile = pid_group % groups_per_k_tile

    tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    isa_index = tile_base + row_base + k_in_tile

    scale_ptrs = dst_packed_ptr + expert_id.to(tl.int64) * per_batch_storage + isa_index.to(tl.int64)
    tl.store(scale_ptrs, e8m0_byte, mask=dim_offs < dim)


@wrap_triton_kernel
@triton.jit
def _warp32x32_transpose_quant_kernel(
    src_ptr, gather_idx_ptr,
    dst_fp8_ptr, dst_scales_ptr,
    dim, capacity, per_batch_storage,
    src_stride_row, src_stride_col,
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_K: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """32x32 single-warp transpose+quant: minimal L1 pressure, no smem.

    Each block processes GROUPS_PER_BLOCK × (32 tokens × 32 dims) tiles.
    Only 32 output cache lines per store (vs 128 in original).
    """
    pid_row = tl.program_id(0)
    pid_grp_block = tl.program_id(1)
    num_dim_blocks = tl.cdiv(dim, BLOCK_DIM)
    expert_id = pid_row // num_dim_blocks
    dim_block = pid_row % num_dim_blocks
    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offs < dim

    k_tiles: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    groups_per_k: tl.constexpr = SF_TILE_K // GROUP_SIZE
    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    out_rows = expert_id * dim + dim_offs

    for g in tl.range(0, GROUPS_PER_BLOCK):
        pid_group = pid_grp_block * GROUPS_PER_BLOCK + g
        cap_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        flat_ids = expert_id * capacity + cap_offs
        if HAS_GATHER:
            src_rows = tl.load(gather_idx_ptr + flat_ids).to(tl.int64)
        else:
            src_rows = flat_ids.to(tl.int64)

        src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=dim_mask[None, :], other=0.0).to(tl.float32)
        values_t = tl.trans(values)

        block_amax = tl.max(tl.abs(values_t), axis=1)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa > 0x600000, 1, 0)
        e8m0 = biased_exp - 8 + carry
        e8m0 = tl.where(biased_exp > 0, e8m0, 0)
        e8m0_byte = tl.maximum(e8m0, 0).to(tl.uint8)
        qexp = tl.maximum(tl.minimum(254 - e8m0, 254), 1)
        qscale = (qexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values_t * qscale[:, None]).to(tl.float8e4nv)

        out_cols = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        fp8_ptrs = dst_fp8_ptr + out_rows[:, None].to(tl.int64) * capacity + out_cols[None, :].to(tl.int64)
        tl.store(fp8_ptrs, quantized, mask=dim_mask[:, None])

        k_idx = pid_group // groups_per_k
        k_in = pid_group % groups_per_k
        tile_base = (row_tiles * k_tiles + k_idx) * SF_TILE_STORAGE
        isa_idx = tile_base + row_base + k_in
        scale_ptrs = dst_scales_ptr + expert_id.to(tl.int64) * per_batch_storage + isa_idx.to(tl.int64)
        tl.store(scale_ptrs, e8m0_byte, mask=dim_mask)


def fused_transpose_quantize_for_wgrad(
    flat_sorted: torch.Tensor,
    num_experts: int,
    capacity: int,
    dim: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused transpose+quantize+ISA-pack for wgrad operands.

    Reads flat expert-sorted (TK, dim) bf16, produces:
      - fp8_out: (E, dim, capacity) fp8e4m3
      - packed_scales: ISA-packed scales for CUTLASS (directly, no separate pack step)

    Optionally fuses a gather step via *gather_idx*.

    Uses 32x32 single-warp kernel for minimal L1 cache pressure (32 output
    cache lines vs 128 in the original 32x128 kernel). GROUPS_PER_BLOCK=16
    amortizes block launch overhead.
    """
    check_tensor(flat_sorted, "flat_sorted", dtype=torch.bfloat16, ndim=2)
    check_divisible(capacity, 32, "capacity")
    device = flat_sorted.device
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    BLOCK_DIM = 32  # 32x32 tile for minimal L1 pressure
    # Heuristic: larger GPB reduces grid overhead but increases register pressure
    # NCU shows 72 regs/thread with GPB=16. Try GPB=8 for less reg -> more occupancy.
    GROUPS_PER_BLOCK = 8

    fp8_flat = torch.empty(num_experts * dim, capacity, dtype=torch.float8_e4m3fn, device=device)

    # Pre-allocate ISA-packed scales directly (fill with 1s = E8M0 exponent 127 = scale 1.0)
    per_batch_storage = _storage_per_batch(dim, capacity)
    packed_scales = torch.ones(num_experts, per_batch_storage, dtype=_E8M0_DTYPE, device=device)

    total_groups = capacity // GROUP_SIZE
    grid = (num_experts * _div_up(dim, BLOCK_DIM), _div_up(total_groups, GROUPS_PER_BLOCK))

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else flat_sorted  # dummy, unused

    _warp32x32_transpose_quant_kernel[grid](
        flat_sorted,
        gather_ptr,
        fp8_flat,
        packed_scales.view(torch.uint8),
        dim,
        capacity,
        per_batch_storage,
        flat_sorted.stride(0),
        flat_sorted.stride(1),
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        num_warps=1,
    )

    fp8_3d = fp8_flat.reshape(num_experts, dim, capacity)
    return fp8_3d, packed_scales


# ---------------------------------------------------------------------------
# Dual quantization: read bf16 once, produce row-major + col-major fp8
# ---------------------------------------------------------------------------

@wrap_triton_kernel
@triton.jit
def _dual_quantize_kernel(
    # Input
    src_ptr,            # (T, H) bf16
    gather_idx_ptr,     # (TK,) int32 — maps TK positions to T rows

    # Output 1: row-major (T, H) fp8 + ISA scales
    row_fp8_ptr,        # (T, H) fp8
    row_scales_ptr,     # (1, row_packed_size) uint8 ISA

    # Output 2: col-major (E*H, capacity) fp8 + ISA scales
    col_fp8_ptr,        # (E*H, capacity) fp8
    col_scales_ptr,     # (E, col_packed_size) uint8 ISA

    # Dimensions
    T,                  # source rows (may be < TK when topK > 1)
    H,                  # hidden dim
    capacity,           # padded per-expert token count
    col_per_batch_storage,  # ISA storage per expert for col output
    src_stride_row,
    src_stride_col,
    row_k_tiles,        # ceil(H / SF_TILE_K) for row ISA

    # Constexprs
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,      # 32
    BLOCK_DIM: tl.constexpr,       # 128
    SF_TILE_M: tl.constexpr,       # 128
    SF_TILE_K: tl.constexpr,       # 128
    SF_TILE_STORAGE: tl.constexpr, # 512
):
    """Read (32, 128) bf16 tile once, produce both row-major and col-major fp8.

    Grid: (E * ceil(H/BLOCK_DIM), ceil(capacity/GROUP_SIZE))
    Each block processes one (expert, dim_block, cap_group) tile.
    """
    pid_row = tl.program_id(0)       # expert_id * num_dim_blocks + dim_block
    pid_group = tl.program_id(1)     # capacity group index

    num_dim_blocks = tl.cdiv(H, BLOCK_DIM)
    expert_id = pid_row // num_dim_blocks
    dim_block = pid_row % num_dim_blocks

    cap_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)     # (32,)
    dim_offs = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)      # (128,)

    # Source row indices (with optional gather)
    flat_token_ids = expert_id * capacity + cap_offs
    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + flat_token_ids).to(tl.int64)
    else:
        src_rows = flat_token_ids.to(tl.int64)

    # Load (GROUP_SIZE=32, BLOCK_DIM=128) bf16 tile — THE SINGLE HBM READ
    dim_mask = dim_offs[None, :] < H
    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    values = tl.load(src_ptrs, mask=dim_mask, other=0.0).to(tl.float32)

    # ═══════════════════════════════════════════════════════════════════
    # OUTPUT 2: Col-major (transposed) quantization -> (E*H, capacity) fp8
    # Groups of 32 along capacity (the K dimension for wgrad GEMM)
    # ═══════════════════════════════════════════════════════════════════
    values_t = tl.trans(values)  # (128, 32)

    col_amax = tl.max(tl.abs(values_t), axis=1)  # (128,)
    col_amax_bits = col_amax.to(tl.int32, bitcast=True)
    col_biased_exp = (col_amax_bits >> 23) & 0xFF
    col_mantissa = col_amax_bits & 0x7FFFFF
    col_carry = tl.where(col_mantissa > 0x600000, 1, 0)
    col_e8m0 = col_biased_exp - 8 + col_carry
    col_e8m0 = tl.where(col_biased_exp > 0, col_e8m0, 0)
    col_e8m0_byte = tl.maximum(col_e8m0, 0).to(tl.uint8)
    col_qexp = 254 - col_e8m0
    col_qexp = tl.maximum(tl.minimum(col_qexp, 254), 1)
    col_qscale = (col_qexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    col_quantized = (values_t * col_qscale[:, None]).to(tl.float8e4nv)

    # Write col fp8
    col_out_rows = expert_id * H + dim_offs
    col_out_cols = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    col_fp8_ptrs = col_fp8_ptr + col_out_rows[:, None].to(tl.int64) * capacity + col_out_cols[None, :].to(tl.int64)
    tl.store(col_fp8_ptrs, col_quantized, mask=dim_offs[:, None] < H)

    # Write col ISA scales
    col_k_tiles: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    col_groups_per_k: tl.constexpr = SF_TILE_K // GROUP_SIZE
    col_row_tiles = dim_offs // SF_TILE_M
    col_row_in_tile = dim_offs % SF_TILE_M
    col_k_tiles_idx = pid_group // col_groups_per_k
    col_k_in_tile = pid_group % col_groups_per_k
    col_tile_base = (col_row_tiles * col_k_tiles + col_k_tiles_idx) * SF_TILE_STORAGE
    col_row_base = (col_row_in_tile % 32) * 16 + (col_row_in_tile // 32) * 4
    col_isa_idx = col_tile_base + col_row_base + col_k_in_tile
    col_scale_ptrs = col_scales_ptr + expert_id.to(tl.int64) * col_per_batch_storage + col_isa_idx.to(tl.int64)
    tl.store(col_scale_ptrs, col_e8m0_byte, mask=dim_offs < H)

    # ═══════════════════════════════════════════════════════════════════
    # OUTPUT 1: Row-major quantization -> (T, H) fp8
    # Groups of 32 along H (the K dimension for actgrad GEMM)
    # Process 4 groups within the 128-dim block, pack into uint32
    # Re-uses the ALREADY-LOADED values — but re-computes amax per group.
    # The HBM load was already done above; only register computation + store.
    # ═══════════════════════════════════════════════════════════════════
    GROUPS_PER_DIM_BLOCK: tl.constexpr = BLOCK_DIM // GROUP_SIZE  # 4

    # ISA layout for row output: rows indexed by src_rows (T-space)
    row_row_tiles = src_rows // SF_TILE_M
    row_row_in_tile = src_rows % SF_TILE_M
    row_row_base_offset = (row_row_in_tile % 32) * 16 + (row_row_in_tile // 32) * 4

    packed_row_scale_i32 = tl.zeros([GROUP_SIZE], dtype=tl.int32)

    # Re-load from L1/L2 cache (data was just loaded above, should be cache-hot)
    for g in tl.range(0, GROUPS_PER_DIM_BLOCK):
        g_col_offs = dim_block * BLOCK_DIM + g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        g_col_mask = g_col_offs[None, :] < H

        g_vals = tl.load(
            src_ptr + src_rows[:, None] * src_stride_row + g_col_offs[None, :] * src_stride_col,
            mask=g_col_mask,
            other=0.0,
        ).to(tl.float32)

        # E8M0 quant per source row (amax over 32 cols)
        row_amax = tl.max(tl.abs(g_vals), axis=1)  # (32,)
        row_amax_bits = row_amax.to(tl.int32, bitcast=True)
        row_biased_exp = (row_amax_bits >> 23) & 0xFF
        row_mantissa = row_amax_bits & 0x7FFFFF
        row_carry = tl.where(row_mantissa > 0x600000, 1, 0)
        row_e8m0 = row_biased_exp - 8 + row_carry
        row_e8m0 = tl.where(row_biased_exp > 0, row_e8m0, 0)
        row_e8m0_clamped = tl.maximum(row_e8m0, 0)
        row_qexp = 254 - row_e8m0
        row_qexp = tl.maximum(tl.minimum(row_qexp, 254), 1)
        row_qscale = (row_qexp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        row_quantized = (g_vals * row_qscale[:, None]).to(tl.float8e4nv)

        # Write row fp8 at (src_rows, g_col_offs)
        row_fp8_ptrs = row_fp8_ptr + src_rows[:, None] * H + g_col_offs[None, :]
        tl.store(row_fp8_ptrs, row_quantized, mask=g_col_mask)

        # Pack scale byte into uint32
        packed_row_scale_i32 = packed_row_scale_i32 | ((row_e8m0_clamped & 0xFF) << (g * 8))

    # Write row ISA packed scales as uint32
    # k_tile_idx for the row output corresponds to dim_block
    row_tile_base = (row_row_tiles * row_k_tiles + dim_block) * SF_TILE_STORAGE
    row_packed_offset_i32 = (row_tile_base + row_row_base_offset) // 4
    row_scale_ptr_i32 = row_scales_ptr.to(tl.pointer_type(tl.int32))
    tl.store(row_scale_ptr_i32 + row_packed_offset_i32, packed_row_scale_i32, mask=src_rows < T)


def dual_quantize_and_pack(
    src: torch.Tensor,
    num_experts: int,
    capacity: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read bf16 source once, produce both row-major and col-major fp8 + ISA scales.

    Parameters
    ----------
    src : (T, H) bf16 — source activation (e.g. dout).
    num_experts : number of experts E.
    capacity : padded per-expert token count.
    gather_idx : optional (TK,) int32 — maps TK positions to T source rows.

    Returns
    -------
    row_fp8 : (T, H) fp8 — row-major quantized (groups along H).
    row_scales : (1, packed_size) e8m0fnu ISA-packed scales.
    col_fp8 : (E, H, capacity) fp8 — col-major quantized (groups along capacity).
    col_scales : (E, col_packed_size) e8m0fnu ISA-packed scales.
    """
    check_tensor(src, "src", dtype=torch.bfloat16, ndim=2)
    check_divisible(capacity, 32, "capacity")
    src = src.contiguous()
    T, H = src.shape
    device = src.device

    GROUP_SIZE = _SF_VEC_SIZE       # 32
    BLOCK_DIM = 128

    # Row-major outputs
    row_fp8 = torch.empty(T, H, dtype=torch.float8_e4m3fn, device=device)
    row_per_batch = _storage_per_batch(T, H)
    if T % _SF_TILE_M == 0 and H % _SF_TILE_K == 0:
        row_scales = torch.empty((1, row_per_batch), dtype=torch.uint8, device=device)
    else:
        row_scales = torch.full((1, row_per_batch), 127, dtype=torch.uint8, device=device)

    # Col-major (transposed) outputs
    col_fp8_flat = torch.empty(num_experts * H, capacity, dtype=torch.float8_e4m3fn, device=device)
    col_per_batch = _storage_per_batch(H, capacity)
    col_scales = torch.ones(num_experts, col_per_batch, dtype=torch.uint8, device=device)

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else src  # dummy when unused

    row_k_tiles = _div_up(H, _SF_TILE_K)
    grid = (num_experts * _div_up(H, BLOCK_DIM), capacity // GROUP_SIZE)

    _dual_quantize_kernel[grid](
        src,
        gather_ptr,
        row_fp8,
        row_scales,
        col_fp8_flat,
        col_scales,
        T,
        H,
        capacity,
        col_per_batch,
        src.stride(0),
        src.stride(1),
        row_k_tiles,
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )

    col_fp8_3d = col_fp8_flat.reshape(num_experts, H, capacity)
    return (
        row_fp8,
        row_scales.view(_E8M0_DTYPE),
        col_fp8_3d,
        col_scales.view(_E8M0_DTYPE),
    )


# ---------------------------------------------------------------------------
# Column-wise blockscaled quantize + ISA-pack for wgrad (varlen_k approach)
# ---------------------------------------------------------------------------
# Groups of 32 along dim 0 of (TK, dim) tensor. ISA-packed scales for
# the logical (dim, TK) layout that CUTLASS sees after .T view.
# Optional gather_idx fuses scatter-read into the quantize pass.

@wrap_triton_kernel
@triton.jit
def _colwise_quantize_and_pack_kernel(
    src_ptr,
    gather_idx_ptr,
    dst_fp8_ptr,
    dst_packed_ptr,
    total_K,
    dim,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    HAS_GATHER: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_K: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr = 1,
):
    """Column-wise blockscaled quantize + ISA-pack.

    2D grid: (num_groups_along_TK // GROUPS_PER_BLOCK, num_blocks_along_dim).
    Quantization groups are along TK (axis 0). Each program processes
    GROUPS_PER_BLOCK consecutive K-groups, amortizing dim-related index math.
    ISA-packed scales target the LOGICAL (dim, TK) layout.
    """
    pid_group_blk = tl.program_id(0)
    pid_dim = tl.program_id(1)

    dim_offs = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offs < dim

    groups_per_k_tile: tl.constexpr = SF_TILE_K // GROUP_SIZE
    row_tiles = dim_offs // SF_TILE_M
    row_in_tile = dim_offs % SF_TILE_M
    row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for g_local in tl.static_range(0, GROUPS_PER_BLOCK):
        pid_group = pid_group_blk * GROUPS_PER_BLOCK + g_local
        k_offs = pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        k_mask = k_offs < total_K
        mask = k_mask[:, None] & dim_mask[None, :]

        if HAS_GATHER:
            src_rows = tl.load(gather_idx_ptr + k_offs, mask=k_mask, other=0).to(tl.int64)
        else:
            src_rows = k_offs.to(tl.int64)

        src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        block_amax = tl.max(tl.abs(values), axis=0)

        # Simplified E8M0 (bit-exact for bf16): see _quantize_and_pack_kernel.
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        carry = (amax_bits & 0x7FFFFF) > 0x600000
        e8m0_i32 = tl.maximum(biased_exp + carry.to(tl.int32) - 8, 0)
        e8m0_byte = e8m0_i32.to(tl.uint8)
        quant_scale = ((254 - e8m0_i32).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * quant_scale[None, :]).to(tl.float8e4nv)

        dst_ptrs = dst_fp8_ptr + k_offs[:, None] * dst_stride_row + dim_offs[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        k_tiles_idx = pid_group // groups_per_k_tile
        k_in_tile = pid_group % groups_per_k_tile

        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        isa_index = tile_base + row_base + k_in_tile

        scale_ptrs = dst_packed_ptr + isa_index.to(tl.int64)
        tl.store(scale_ptrs, e8m0_byte, mask=dim_mask)


# ---------------------------------------------------------------------------
# Fused transpose + rowwise quantize for wgrad operands
# ---------------------------------------------------------------------------
# Instead of colwise quant on (TK, dim) -> scattered access,
# we transpose + rowwise quant in a single kernel:
#   Read (TK, dim) bf16 -> SMEM transpose -> write (dim, TK) FP8 + ISA scales
# This converts the cache-hostile colwise pattern into cache-friendly rowwise.

@wrap_triton_kernel
@triton.jit
def _fused_transpose_rowquant_kernel(
    src_ptr,              # (TK, dim) bf16 source, row-major
    gather_idx_ptr,       # (TK,) int32 gather indices (or dummy)
    dst_fp8_ptr,          # (dim, TK) fp8 output, row-major
    dst_packed_scale_ptr, # ISA-packed scales for (dim, TK) layout
    TK,                   # total K dimension
    dim,                  # H or I dimension
    src_stride_row,       # stride of source along TK
    src_stride_col,       # stride of source along dim
    dst_stride_row,       # stride of output along dim (= TK)
    dst_stride_col,       # stride of output along TK (= 1)
    k_tiles,              # ceil(TK / SF_TILE_K) for ISA layout
    HAS_GATHER: tl.constexpr,
    TILE_TK: tl.constexpr,   # tile size along TK (e.g. 128)
    TILE_DIM: tl.constexpr,  # tile size along dim (e.g. 32)
    GROUP_SIZE: tl.constexpr, # 32 for blockscaled
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,  # TILE_TK // GROUP_SIZE (e.g. 4)
):
    """Fused transpose + rowwise blockscaled quantize.

    Grid: (dim_tiles, tk_tiles)
    Each block reads a (TILE_TK, TILE_DIM) tile from src,
    transposes to (TILE_DIM, TILE_TK), then rowwise-quantizes
    with groups of GROUP_SIZE along TK, writing FP8 + ISA-packed scales.

    The output is (dim, TK) row-major FP8 with ISA-packed scales
    for the (dim, TK) logical layout — exactly what wgrad CUTLASS needs.
    """
    pid_dim = tl.program_id(0)   # which dim tile
    pid_tk = tl.program_id(1)    # which TK tile

    dim_offs = pid_dim * TILE_DIM + tl.arange(0, TILE_DIM)
    tk_offs = pid_tk * TILE_TK + tl.arange(0, TILE_TK)
    dim_mask = dim_offs < dim
    tk_mask = tk_offs < TK

    # Load (TILE_TK, TILE_DIM) tile from src with optional gather
    if HAS_GATHER:
        src_rows = tl.load(gather_idx_ptr + tk_offs, mask=tk_mask, other=0).to(tl.int64)
    else:
        src_rows = tk_offs.to(tl.int64)

    src_ptrs = src_ptr + src_rows[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
    mask_2d = tk_mask[:, None] & dim_mask[None, :]
    values = tl.load(src_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    # values shape: (TILE_TK, TILE_DIM) in registers

    # Transpose: now process as (TILE_DIM, TILE_TK) — each "row" is one dim element
    # For rowwise quant, we need amax over groups of GROUP_SIZE along TK axis
    # Reshape values to (TILE_DIM, TILE_TK) by transposing
    # Triton: values[tk, d] -> transposed[d, tk]
    # We process group-by-group along TK within this tile

    # ISA layout for output (dim, TK): dim is M-axis, TK is K-axis
    dst_row_tiles = dim_offs // SF_TILE_M
    dst_row_in_tile = dim_offs % SF_TILE_M
    dst_row_base_offset = (dst_row_in_tile % 32) * 16 + (dst_row_in_tile // 32) * 4

    packed_scale_i32 = tl.zeros([TILE_DIM], dtype=tl.int32)

    for g in tl.range(0, GROUPS_PER_TILE):
        group_tk_start = g * GROUP_SIZE
        group_tk_offs = group_tk_start + tl.arange(0, GROUP_SIZE)
        group_mask = (pid_tk * TILE_TK + group_tk_offs) < TK

        # Extract the (GROUP_SIZE, TILE_DIM) sub-tile from values
        # values[group_tk_offs, :] -> (GROUP_SIZE, TILE_DIM)
        subtile = tl.load(
            src_ptr + (src_rows[group_tk_start:group_tk_start + GROUP_SIZE] if HAS_GATHER
                       else (pid_tk * TILE_TK + group_tk_offs).to(tl.int64))[:, None] * src_stride_row
            + dim_offs[None, :] * src_stride_col,
            mask=group_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Amax along GROUP_SIZE (axis=0) -> per-dim-element scale
        block_amax = tl.max(tl.abs(subtile), axis=0)  # (TILE_DIM,)

        # Pure-integer E8M0 computation
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_clamped = tl.maximum(e8m0_i32, 0)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        # Quantize: scale then cast
        quantized = (subtile * quant_scale[None, :]).to(tl.float8e4nv)
        # quantized shape: (GROUP_SIZE, TILE_DIM)

        # Write transposed: dst[dim, tk] = quantized^T
        # Output row = dim_offs, output col = actual_tk_offs
        actual_tk_offs = pid_tk * TILE_TK + group_tk_offs
        out_ptrs = dst_fp8_ptr + dim_offs[None, :] * dst_stride_row + actual_tk_offs[:, None] * dst_stride_col
        # But we need to write (GROUP_SIZE, TILE_DIM) transposed as (TILE_DIM, GROUP_SIZE)
        # Use swapped indexing:
        out_ptrs_t = dst_fp8_ptr + dim_offs[:, None] * dst_stride_row + actual_tk_offs[None, :] * dst_stride_col
        tl.store(out_ptrs_t, tl.trans(quantized), mask=dim_mask[:, None] & group_mask[None, :])

        # Pack scale bytes into uint32 (4 groups per TILE_TK=128)
        packed_scale_i32 = packed_scale_i32 | ((e8m0_clamped & 0xFF) << (g * 8))

    # Write packed uint32 ISA scales
    global_group_idx = pid_tk  # which k-tile (each TILE_TK = SF_TILE_K = 128)
    tile_base = (dst_row_tiles * k_tiles + global_group_idx) * SF_TILE_STORAGE
    packed_offset_i32 = (tile_base + dst_row_base_offset) // 4
    scale_ptr_i32 = dst_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(scale_ptr_i32 + packed_offset_i32, packed_scale_i32, mask=dim_mask)


def fused_transpose_quantize_and_pack(
    src: torch.Tensor,
    logical_rows: int,
    logical_cols: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused transpose + rowwise blockscaled quantize for wgrad operands.

    Replaces colwise_quantize_and_pack with ~3-5x better performance by
    converting the cache-hostile column-wise access pattern into row-wise.

    Input: (TK, dim) bf16 row-major
    Output: (dim, TK) fp8 row-major + ISA-packed scales for (dim, TK) layout

    Parameters are identical to colwise_quantize_and_pack.
    """
    H = logical_rows   # dim
    TK = logical_cols
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    TILE_TK = _SF_TILE_K       # 128 (= SF_TILE_K, matches ISA tile)
    TILE_DIM = 32              # process 32 dim rows per block

    # Output: (dim, TK) row-major FP8
    fp8_out = torch.empty(H, TK, dtype=torch.float8_e4m3fn, device=src.device)

    # ISA-packed scales for (dim, TK) layout
    per_batch_storage = _storage_per_batch(H, TK)
    if H % _SF_TILE_M == 0 and TK % _SF_TILE_K == 0:
        packed_scales = torch.empty((1, per_batch_storage), dtype=torch.uint8, device=src.device)
    else:
        packed_scales = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=src.device)

    k_tiles = _div_up(TK, _SF_TILE_K)
    grid = (_div_up(H, TILE_DIM), _div_up(TK, TILE_TK))
    groups_per_tile = TILE_TK // GROUP_SIZE  # 4

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else src

    _fused_transpose_rowquant_kernel[grid](
        src, gather_ptr, fp8_out, packed_scales,
        TK, H,
        src.stride(0), src.stride(1),
        fp8_out.stride(0), fp8_out.stride(1),
        k_tiles,
        HAS_GATHER=has_gather,
        TILE_TK=TILE_TK,
        TILE_DIM=TILE_DIM,
        GROUP_SIZE=GROUP_SIZE,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_TILE=groups_per_tile,
    )
    return fp8_out, packed_scales.view(_E8M0_DTYPE)


def colwise_quantize_and_pack(
    src: torch.Tensor,
    logical_rows: int,
    logical_cols: int,
    *,
    gather_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-wise blockscaled FP8 quantize + ISA-pack for wgrad operands.

    Quantizes (TK, dim) with groups of 32 along TK (dim 0).
    ISA-packed scales target the logical (dim, TK) layout.

    Parameters
    ----------
    src : (T_or_TK, dim) bf16 source.
    logical_rows : dim — rows of the logical (dim, TK) matrix.
    logical_cols : TK — cols (= total K for varlen_k).
    gather_idx : optional (TK,) int32 gather indices.

    Returns
    -------
    fp8_data : (TK, dim) float8_e4m3fn
    packed_scales : (1, packed_size) float8_e8m0fnu in ISA layout for (dim, TK).
    """
    check_tensor(src, "src", dtype=torch.bfloat16, ndim=2)
    H = logical_rows
    TK = logical_cols
    GROUP_SIZE = _SF_VEC_SIZE
    BLOCK_DIM = 128

    fp8_out = torch.empty(TK, H, dtype=torch.float8_e4m3fn, device=src.device)
    per_batch_storage = _storage_per_batch(H, TK)
    # When both dims are tile-aligned, every scale byte is written by the kernel.
    # Skip expensive torch.full fill kernel (~5µs launch) — use torch.empty instead.
    if H % _SF_TILE_M == 0 and TK % _SF_TILE_K == 0:
        packed_scales = torch.empty((1, per_batch_storage), dtype=torch.uint8, device=src.device)
    else:
        packed_scales = torch.full((1, per_batch_storage), 127, dtype=torch.uint8, device=src.device)

    num_groups = _div_up(TK, GROUP_SIZE)
    k_tiles = _div_up(TK, _SF_TILE_K)
    # NCU-tuned: GROUPS_PER_BLOCK=2 amortizes dim-related index math across 2
    # consecutive K-groups -> ~3-4% gain at TK=65536. Requires num_groups to be
    # a multiple of GROUPS_PER_BLOCK; fall back to 1 otherwise (preserves
    # bit-exactness for arbitrary shapes).
    GROUPS_PER_BLOCK = 2 if (num_groups % 2 == 0) else 1
    grid = (num_groups // GROUPS_PER_BLOCK, _div_up(H, BLOCK_DIM))

    has_gather = gather_idx is not None
    gather_ptr = gather_idx if has_gather else src

    # NCU-guided: num_warps=1 reduces register pressure per block, allowing
    # more blocks in-flight on each SM -> 2.3× speedup at TK=65536 (verified
    # bitwise identical to num_warps=4 across all shapes).
    _colwise_quantize_and_pack_kernel[grid](
        src, gather_ptr, fp8_out, packed_scales,
        TK, H,
        src.stride(0), src.stride(1),
        fp8_out.stride(0), fp8_out.stride(1),
        k_tiles,
        HAS_GATHER=has_gather,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_DIM=BLOCK_DIM,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        num_warps=1,
    )
    return fp8_out, packed_scales.view(_E8M0_DTYPE)



# ---------------------------------------------------------------------------
# Dual row+col quantize for fused actgrad+wgrad data prep
# ---------------------------------------------------------------------------

@wrap_triton_kernel
@triton.jit
def _dual_varlen_quantize_kernel(
    src_ptr,            # (TK, dim) bf16
    # Row output: (TK, dim) fp8 + ISA scales (groups along dim)
    row_fp8_ptr, row_scales_ptr,
    # Col output: (TK, dim) fp8 + ISA scales (groups along TK)
    col_fp8_ptr, col_scales_ptr,
    TK, dim,
    src_stride_row, src_stride_col,
    row_k_tiles,        # ceil(dim / SF_TILE_K)
    col_k_tiles,        # ceil(TK / SF_TILE_K)
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,       # 32
    BLOCK_DIM: tl.constexpr,        # 128
    SF_TILE_M: tl.constexpr,        # 128
    SF_TILE_K: tl.constexpr,        # 128
    SF_TILE_STORAGE: tl.constexpr,  # 512
):
    """Read (32, 128) bf16 tile once, produce both row-major and col-major fp8.

    Row: groups of 32 along dim (contiguous, for actgrad A×B where K=dim).
    Col: groups of 32 along TK (strided groups, for wgrad A^T×B where K=TK).

    Grid: (ceil(TK/GROUP_SIZE), ceil(dim/BLOCK_DIM))
    """
    pid_tk_group = tl.program_id(0)   # TK group index (col groups)
    pid_dim_block = tl.program_id(1)  # dim block index

    tk_offs = pid_tk_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)   # (32,)
    dim_offs = pid_dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)  # (128,)

    tk_mask = tk_offs < TK
    dim_mask = dim_offs < dim
    mask_2d = tk_mask[:, None] & dim_mask[None, :]

    # ── Single HBM read: (32, 128) bf16 tile ──
    src_ptrs = src_ptr + tk_offs[:, None].to(tl.int64) * src_stride_row + dim_offs[None, :].to(tl.int64) * src_stride_col
    values = tl.load(src_ptrs, mask=mask_2d, other=0.0).to(tl.float32)  # (32, 128)

    # ── Col-major quantize: groups of 32 along TK ──
    # For each of the 128 dim elements, find amax across 32 TK rows
    values_t = tl.trans(values)   # (128, 32)
    col_amax = tl.max(tl.abs(values_t), axis=1)  # (128,)
    col_bits = col_amax.to(tl.int32, bitcast=True)
    col_bexp = (col_bits >> 23) & 0xFF
    col_mant = col_bits & 0x7FFFFF
    col_carry = tl.where(col_mant > 0x600000, 1, 0)
    col_e8m0 = tl.where(col_bexp > 0, col_bexp - 8 + col_carry, 0)
    col_e8m0 = tl.maximum(col_e8m0, 0).to(tl.uint8)
    col_qexp = tl.maximum(tl.minimum(254 - col_e8m0.to(tl.int32), 254), 1)
    col_qscale = (col_qexp << 23).to(tl.float32, bitcast=True)
    col_fp8 = (values_t * col_qscale[:, None]).to(tl.float8e4nv)  # (128, 32)

    # Write col fp8: same physical (TK, dim) layout, just different quant
    col_fp8_t = tl.trans(col_fp8)  # (32, 128) — back to (TK, dim) order
    col_out_ptrs = col_fp8_ptr + tk_offs[:, None].to(tl.int64) * dim + dim_offs[None, :].to(tl.int64)
    tl.store(col_out_ptrs, col_fp8_t, mask=mask_2d)

    # Write col ISA scales (dim rows, TK cols in logical layout)
    GROUPS_PER_K: tl.constexpr = SF_TILE_K // GROUP_SIZE
    col_row_tiles = dim_offs // SF_TILE_M
    col_row_in_tile = dim_offs % SF_TILE_M
    col_k_tile_idx = pid_tk_group // GROUPS_PER_K
    col_k_in_tile = pid_tk_group % GROUPS_PER_K
    col_tile_base = (col_row_tiles * col_k_tiles + col_k_tile_idx) * SF_TILE_STORAGE
    col_row_base = (col_row_in_tile % 32) * 16 + (col_row_in_tile // 32) * 4
    col_isa_offs = col_tile_base + col_row_base + col_k_in_tile
    tl.store(col_scales_ptr + col_isa_offs.to(tl.int64), col_e8m0, mask=dim_mask)

    # ── Row-major quantize: groups of 32 along dim ──
    # Process 4 groups of 32 within the 128-dim block.
    # Re-reads from source (L2 cache hot from the initial load above).
    GROUPS_PER_DIM: tl.constexpr = BLOCK_DIM // GROUP_SIZE  # 4
    row_row_tiles = tk_offs // SF_TILE_M
    row_row_in_tile = tk_offs % SF_TILE_M
    row_row_base = (row_row_in_tile % 32) * 16 + (row_row_in_tile // 32) * 4

    packed_i32 = tl.zeros([GROUP_SIZE], dtype=tl.int32)

    for g in tl.range(0, GROUPS_PER_DIM):
        g_dim_offs = pid_dim_block * BLOCK_DIM + g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        g_mask = tk_mask[:, None] & (g_dim_offs[None, :] < dim)
        g_ptrs = src_ptr + tk_offs[:, None].to(tl.int64) * src_stride_row + g_dim_offs[None, :].to(tl.int64) * src_stride_col
        g_vals = tl.load(g_ptrs, mask=g_mask, other=0.0).to(tl.float32)

        # E8M0 quant per TK row (amax over 32 dim cols)
        row_amax = tl.max(tl.abs(g_vals), axis=1)  # (32,)
        row_bits = row_amax.to(tl.int32, bitcast=True)
        row_bexp = (row_bits >> 23) & 0xFF
        row_mant = row_bits & 0x7FFFFF
        row_carry = tl.where(row_mant > 0x600000, 1, 0)
        row_e8m0 = tl.where(row_bexp > 0, row_bexp - 8 + row_carry, 0)
        row_e8m0 = tl.maximum(row_e8m0, 0)
        row_qexp = tl.maximum(tl.minimum(254 - row_e8m0, 254), 1)
        row_qscale = (row_qexp << 23).to(tl.float32, bitcast=True)
        row_fp8_g = (g_vals * row_qscale[:, None]).to(tl.float8e4nv)

        # Write row fp8
        row_out_ptrs = row_fp8_ptr + tk_offs[:, None].to(tl.int64) * dim + g_dim_offs[None, :].to(tl.int64)
        tl.store(row_out_ptrs, row_fp8_g, mask=g_mask)

        # Pack ISA scale byte
        packed_i32 = packed_i32 | ((row_e8m0 & 0xFF) << (g * 8))

    # Write row ISA packed scales as uint32
    row_tile_base = (row_row_tiles * row_k_tiles + pid_dim_block) * SF_TILE_STORAGE
    row_packed_offs = (row_tile_base + row_row_base) // 4
    row_scale_i32 = row_scales_ptr.to(tl.pointer_type(tl.int32))
    tl.store(row_scale_i32 + row_packed_offs.to(tl.int64), packed_i32, mask=tk_mask)


def dual_quantize_varlen(
    src: torch.Tensor,
    TK: int,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read bf16 once, produce both row-major and col-major fp8 + ISA scales.

    For fused actgrad + wgrad data prep: one HBM read, two quantization outputs.

    Parameters
    ----------
    src : (TK, dim) bf16 contiguous.
    TK : total token count (rows).
    dim : hidden/intermediate dim (cols).

    Returns
    -------
    row_fp8 : (TK, dim) fp8 — row-quantized (groups along dim, for actgrad).
    row_scales : (1, packed) e8m0fnu ISA-packed.
    col_fp8 : (TK, dim) fp8 — col-quantized (groups along TK, for wgrad).
    col_scales : (1, packed) e8m0fnu ISA-packed for logical (dim, TK).
    """
    check_tensor(src, "src", dtype=torch.bfloat16, ndim=2)
    if src.shape != (TK, dim):
        raise ValueError(f"src shape {src.shape} != ({TK}, {dim})")
    src = src.contiguous()
    device = src.device
    GROUP_SIZE = _SF_VEC_SIZE
    BLOCK_DIM = 128

    # Row outputs
    row_fp8 = torch.empty(TK, dim, dtype=torch.float8_e4m3fn, device=device)
    row_per_batch = _storage_per_batch(TK, dim)
    if TK % _SF_TILE_M == 0 and dim % _SF_TILE_K == 0:
        row_scales = torch.empty((1, row_per_batch), dtype=torch.uint8, device=device)
    else:
        row_scales = torch.full((1, row_per_batch), 127, dtype=torch.uint8, device=device)

    # Col outputs (ISA scales for logical (dim, TK) layout)
    col_fp8 = torch.empty(TK, dim, dtype=torch.float8_e4m3fn, device=device)
    col_per_batch = _storage_per_batch(dim, TK)
    if dim % _SF_TILE_M == 0 and TK % _SF_TILE_K == 0:
        col_scales = torch.empty((1, col_per_batch), dtype=torch.uint8, device=device)
    else:
        col_scales = torch.full((1, col_per_batch), 127, dtype=torch.uint8, device=device)

    row_k_tiles = _div_up(dim, _SF_TILE_K)
    col_k_tiles = _div_up(TK, _SF_TILE_K)

    grid = (_div_up(TK, GROUP_SIZE), _div_up(dim, BLOCK_DIM))
    # NCU-guided: num_warps=1 gives 2× speedup (157µs vs 314µs at TK=65536).
    _dual_varlen_quantize_kernel[grid](
        src,
        row_fp8, row_scales,
        col_fp8, col_scales,
        TK, dim,
        src.stride(0), src.stride(1),
        row_k_tiles, col_k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=GROUP_SIZE, BLOCK_DIM=BLOCK_DIM,
        SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        num_warps=1,
    )
    return (
        row_fp8, row_scales.view(_E8M0_DTYPE),
        col_fp8, col_scales.view(_E8M0_DTYPE),
    )


# ---------------------------------------------------------------------------
# varlen_k blockscaled FP8 GEMM for weight gradients
# ---------------------------------------------------------------------------

_COMPILE_CACHE_VK = _ICC("varlen_k")
_GEMM_FAST_PATH_VK: dict = {}


def _run_cutlass_blockscaled_gemm_varlen_k(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    out_dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUTLASS blockscaled FP8 GEMM with varlen_k scheduling.

    D[e] = A[:, K_e] @ B[K_e, :] for each expert e.

    A is logical (M, total_K) M-major — physical a_fp8 is (total_K, M).
    B is logical (N, total_K) N-major — physical b_fp8 is (total_K, N).
    D is (num_experts, M, N).

    If *out* is provided it must be a contiguous (num_experts, M, N) tensor
    of the correct dtype.
    """
    a_logical = a_fp8.T
    b_logical = b_fp8.T
    if out is None:
        out = torch.empty(num_experts, M, N, dtype=out_dtype, device=device)

    fast_key = (
        "vk", M, N, total_K, num_experts, out_dtype,
        a_fp8.shape[0], a_fp8.shape[1],
        b_fp8.shape[0], b_fp8.shape[1],
        a_scales.size(1), b_scales.size(1),
        device.index if device.index is not None else -1,
    )
    cached = _GEMM_FAST_PATH_VK.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        d_permuted = out.permute(1, 2, 0)
        a_cute = _make_cute_tensor_dynamic(a_logical, leading_dim=0)
        b_cute = _make_cute_tensor_dynamic(b_logical, leading_dim=0)
        d_cute = _make_cute_tensor_dynamic(d_permuted, leading_dim=1)
        a_sc_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
        b_sc_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
        )
        stream = cuda.CUstream(_get_raw_cuda_stream())
        compiled(
            a_cute, b_cute, d_cute, None,
            epi_args, scheduler_args, varlen_args, stream,
            a_sc_cute, b_sc_cute,
        )
        return out

    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled varlen_k does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_logical),
        "B": GemmTensorInfo(b_logical),
        "D": GemmTensorInfo(out),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_k=True)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        Float32, tensor_infos["D"].dtype,
        tensor_infos["A"].major, tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for varlen_k")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None, batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
    )
    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)

    # compile_key must NOT contain dynamic token dimensions (total_K,
    # a_scales.size(1), b_scales.size(1)) — those are handled at runtime
    # via mark_layout_dynamic / CuTe sym_int64 symbols.  Including them
    # would trigger a CuTe recompile every time seqlen changes.
    compile_key = (
        "vk",
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn, cluster_shape_mnk,
        M, N,
        tensor_infos["A"].major, tensor_infos["B"].major,
        tensor_infos["D"].major,
        config.pingpong, _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE_VK.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32, tensor_infos["A"].dtype,
            tile_shape_mn, cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE, gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor, tensor_infos["C"].cute_tensor,
            epi_args, scheduler_args, varlen_args, current_stream,
            a_scale_cute, b_scale_cute,
        )
        _COMPILE_CACHE_VK[compile_key] = compiled

    if len(_GEMM_FAST_PATH_VK) > _MAX_FAST_PATH_ENTRIES:
        _GEMM_FAST_PATH_VK.clear()
    _GEMM_FAST_PATH_VK[fast_key] = (compiled, scheduler_args, epi_args)

    compiled(
        tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor, tensor_infos["C"].cute_tensor,
        epi_args, scheduler_args, varlen_args, current_stream,
        a_scale_cute, b_scale_cute,
    )
    return out


# ── Fused wgrad GEMM + fp32 accumulation (zero extra kernels) ─────────────────

_COMPILE_CACHE_VK_ACCUM = _ICC("varlen_k_accum")
_GEMM_FAST_PATH_VK_ACCUM: dict = {}


def _run_cutlass_blockscaled_gemm_varlen_k_accumulate(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device: torch.device,
    accumulator: torch.Tensor,
) -> None:
    """CUTLASS blockscaled FP8 GEMM with fused fp32 accumulation.

    Computes: accumulator += A @ B (per-expert varlen_k GEMM)

    Uses the CUTLASS epilogue `D = 1.0 * (A@B) + 1.0 * C` where C=D=accumulator.
    This eliminates the separate transpose+cast+add kernels (saving ~170μs/iter).

    accumulator must be a contiguous fp32 tensor of shape (num_experts, M, N).
    """
    assert accumulator.dtype == torch.float32
    assert accumulator.shape == (num_experts, M, N)
    assert accumulator.is_contiguous()

    a_logical = a_fp8.T
    b_logical = b_fp8.T

    fast_key = (
        "vk_accum", M, N, total_K, num_experts,
        a_fp8.shape[0], a_fp8.shape[1],
        b_fp8.shape[0], b_fp8.shape[1],
        a_scales.size(1), b_scales.size(1),
        device.index if device.index is not None else -1,
    )
    cached = _GEMM_FAST_PATH_VK_ACCUM.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        d_permuted = accumulator.permute(1, 2, 0)
        a_cute = _make_cute_tensor_dynamic(a_logical, leading_dim=0)
        b_cute = _make_cute_tensor_dynamic(b_logical, leading_dim=0)
        d_cute = _make_cute_tensor_dynamic(d_permuted, leading_dim=1)
        c_cute = _make_cute_tensor_dynamic(d_permuted, leading_dim=1)  # C = D (inplace)
        a_sc_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
        b_sc_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
        )
        stream = cuda.CUstream(_get_raw_cuda_stream())
        compiled(
            a_cute, b_cute, d_cute, c_cute,
            epi_args, scheduler_args, varlen_args, stream,
            a_sc_cute, b_sc_cute,
        )
        return

    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled varlen_k does not support swap_ab")

    # D and C are the same tensor (inplace accumulation)
    tensor_infos = {
        "A": GemmTensorInfo(a_logical),
        "B": GemmTensorInfo(b_logical),
        "D": GemmTensorInfo(accumulator),
        "C": GemmTensorInfo(accumulator),  # C = D for D += A@B
    }
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_k=True)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            info.cute_tensor = _make_cute_tensor_dynamic(
                info.tensor,
                leading_dim=1 if info.major == major_configs[name][1] else 0,
            )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        Float32, tensor_infos["D"].dtype,
        tensor_infos["A"].major, tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for varlen_k accumulate")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None, batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
    )
    # beta=1.0: D = 1.0 * (A@B) + 1.0 * C, where C=D=accumulator
    epi_args = GemmDefaultSm100.EpilogueArguments(beta=Float32(1.0))
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)

    # compile_key must NOT contain dynamic token dimensions — see "vk" above.
    compile_key = (
        "vk_accum",
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn, cluster_shape_mnk,
        M, N,
        tensor_infos["A"].major, tensor_infos["B"].major,
        tensor_infos["D"].major,
        config.pingpong, _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE_VK_ACCUM.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32, tensor_infos["A"].dtype,
            tile_shape_mn, cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE, gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor, tensor_infos["C"].cute_tensor,
            epi_args, scheduler_args, varlen_args, current_stream,
            a_scale_cute, b_scale_cute,
        )
        _COMPILE_CACHE_VK_ACCUM[compile_key] = compiled

    if len(_GEMM_FAST_PATH_VK_ACCUM) > _MAX_FAST_PATH_ENTRIES:
        _GEMM_FAST_PATH_VK_ACCUM.clear()
    _GEMM_FAST_PATH_VK_ACCUM[fast_key] = (compiled, scheduler_args, epi_args)

    compiled(
        tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor, tensor_infos["C"].cute_tensor,
        epi_args, scheduler_args, varlen_args, current_stream,
        a_scale_cute, b_scale_cute,
    )


# ── TMA Reduce-Add wgrad GEMM (hardware atomic add, ~50 regs) ────────────────

_COMPILE_CACHE_VK_TMA_ADD = _ICC("varlen_k_tma_add")
_GEMM_FAST_PATH_VK_TMA_ADD: dict = {}


def _run_cutlass_blockscaled_gemm_varlen_k_tma_add(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device: torch.device,
    accumulator: torch.Tensor,
) -> None:
    """CUTLASS blockscaled FP8 GEMM with TMA reduce-add accumulation.

    Computes: accumulator += A @ B (per-expert varlen_k GEMM)

    Uses TMA hardware reduce-add (add_to_output=True) instead of the CUTLASS
    epilogue `D = A@B + 1.0*C`.  This avoids loading C via TMA, eliminating
    the epilogue register overhead (86 -> ~50 regs/thread) and the extra
    smem staging (epi_c_stage=2 -> 0).

    accumulator must be a contiguous fp32 tensor of shape (num_experts, M, N).
    """
    assert accumulator.dtype == torch.float32
    assert accumulator.shape == (num_experts, M, N)
    assert accumulator.is_contiguous()

    a_logical = a_fp8.T
    b_logical = b_fp8.T

    fast_key = (
        "vk_tma_add", M, N, total_K, num_experts,
        a_fp8.shape[0], a_fp8.shape[1],
        b_fp8.shape[0], b_fp8.shape[1],
        a_scales.size(1), b_scales.size(1),
        device.index if device.index is not None else -1,
    )
    cached = _GEMM_FAST_PATH_VK_TMA_ADD.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        d_permuted = accumulator.permute(1, 2, 0)
        a_cute = _make_cute_tensor_dynamic(a_logical, leading_dim=0)
        b_cute = _make_cute_tensor_dynamic(b_logical, leading_dim=0)
        d_cute = _make_cute_tensor_dynamic(d_permuted, leading_dim=1)
        a_sc_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
        b_sc_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
        )
        stream = cuda.CUstream(_get_raw_cuda_stream())
        compiled(
            a_cute, b_cute, d_cute, None,
            epi_args, scheduler_args, varlen_args, stream,
            a_sc_cute, b_sc_cute,
        )
        return

    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled varlen_k does not support swap_ab")

    # D = accumulator (inplace via TMA reduce-add); C = None (no epilogue load)
    tensor_infos = {
        "A": GemmTensorInfo(a_logical),
        "B": GemmTensorInfo(b_logical),
        "D": GemmTensorInfo(accumulator),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_k=True)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            info.cute_tensor = _make_cute_tensor_dynamic(
                info.tensor,
                leading_dim=1 if info.major == major_configs[name][1] else 0,
            )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        Float32, tensor_infos["D"].dtype,
        tensor_infos["A"].major, tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for varlen_k tma_add")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None, batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=None,
    )
    # add_to_output=True: TMA store becomes CopyReduceBulkTensorTileS2GOp(ADD)
    # No beta needed — accumulation handled by hardware atomic add on TMA store
    epi_args = GemmDefaultSm100.EpilogueArguments(add_to_output=True)
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)

    compile_key = (
        "vk_tma_add",
        tensor_infos["A"].dtype, tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn, cluster_shape_mnk,
        M, N,
        tensor_infos["A"].major, tensor_infos["B"].major,
        tensor_infos["D"].major,
        config.pingpong, _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE_VK_TMA_ADD.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32, tensor_infos["A"].dtype,
            tile_shape_mn, cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE, gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor, None,
            epi_args, scheduler_args, varlen_args, current_stream,
            a_scale_cute, b_scale_cute,
        )
        _COMPILE_CACHE_VK_TMA_ADD[compile_key] = compiled

    if len(_GEMM_FAST_PATH_VK_TMA_ADD) > _MAX_FAST_PATH_ENTRIES:
        _GEMM_FAST_PATH_VK_TMA_ADD.clear()
    _GEMM_FAST_PATH_VK_TMA_ADD[fast_key] = (compiled, scheduler_args, epi_args)

    compiled(
        tensor_infos["A"].cute_tensor, tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor, None,
        epi_args, scheduler_args, varlen_args, current_stream,
        a_scale_cute, b_scale_cute,
    )


def blockscaled_fp8_wgrad_varlen_k(
    a_src: torch.Tensor,
    b_src: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    M: int,
    N: int,
    *,
    a_gather_idx: Optional[torch.Tensor] = None,
    b_gather_idx: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full FP8 weight-gradient: column-wise quant + varlen_k GEMM.

    Computes dW[e] = A_e^T @ B_e for each expert via:
    1. Column-wise quantize A and B (groups of 32 along TK).
    2. varlen_k CUTLASS GEMM with non-contiguous FP8 via TMA.

    Parameters
    ----------
    a_src : (T_or_TK, M) bf16 — e.g., dout, dz, or x.
    b_src : (T_or_TK, N) bf16 — e.g., y1s, dz, or x.
    cu_seqlens_k : (E+1,) int32 expert boundaries.
    M : output rows (e.g., H).
    N : output cols (e.g., I or 2*I).
    a_gather_idx : optional (TK,) int32 gather for A.
    b_gather_idx : optional (TK,) int32 gather for B.
    out_dtype : output dtype.
    out : optional pre-allocated (E, M, N) contiguous output tensor.

    Returns
    -------
    dW : (E, M, N) weight gradient.
    """
    a_rows = a_gather_idx.size(0) if a_gather_idx is not None else a_src.shape[0]
    b_rows = b_gather_idx.size(0) if b_gather_idx is not None else b_src.shape[0]
    if a_rows != b_rows:
        raise ValueError(
            "effective row count must match between operands, "
            f"got a={a_rows}, b={b_rows}"
        )
    TK = a_rows
    num_experts = cu_seqlens_k.shape[0] - 1
    device = b_src.device

    a_fp8, a_scales = colwise_quantize_and_pack(
        a_src, logical_rows=M, logical_cols=TK, gather_idx=a_gather_idx,
    )
    b_fp8, b_scales = colwise_quantize_and_pack(
        b_src, logical_rows=N, logical_cols=TK, gather_idx=b_gather_idx,
    )

    return _run_cutlass_blockscaled_gemm_varlen_k(
        a_fp8, a_scales, b_fp8, b_scales,
        cu_seqlens_k, M, N, TK, num_experts,
        out_dtype, device, out=out,
    )


def _pack_grouped_rows(tensor: torch.Tensor, cu_seqlens_m: torch.Tensor, groups: int, capacity: int) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D tensor to pack, got shape {tuple(tensor.shape)}")
    packed = torch.empty((groups, capacity, tensor.size(1)), dtype=tensor.dtype, device=tensor.device)
    grid = (groups, triton.cdiv(capacity, 16), triton.cdiv(tensor.size(1), 128))
    _pack_expert_segments_kernel[grid](
        tensor,
        packed,
        cu_seqlens_m,
        tensor.stride(0),
        tensor.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        tensor.size(1),
        capacity,
        BLOCK_M=16,
        BLOCK_N=128,
    )
    return packed


def _pack_quantize_grouped_rows(
    tensor: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    groups: int,
    capacity: int,
    protocol: FP8Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D tensor to pack, got shape {tuple(tensor.shape)}")
    if tensor.size(1) % protocol.group_size != 0:
        raise ValueError(f"expected width divisible by group size {protocol.group_size}, got {tensor.size(1)}")

    quantized = torch.empty((groups, capacity, tensor.size(1)), dtype=protocol.activation_torch_dtype, device=tensor.device)
    dequant_scale_fp32 = torch.ones(
        (groups, capacity, tensor.size(1) // protocol.group_size),
        dtype=torch.float32,
        device=tensor.device,
    )
    grid = (groups, triton.cdiv(capacity, 8), tensor.size(1) // protocol.group_size)
    _pack_quantize_expert_segments_kernel[grid](
        tensor,
        quantized,
        dequant_scale_fp32,
        cu_seqlens_m,
        tensor.stride(0),
        tensor.stride(1),
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        dequant_scale_fp32.stride(0),
        dequant_scale_fp32.stride(1),
        dequant_scale_fp32.stride(2),
        tensor.size(1),
        _FP8_E4M3_MAX,
        BLOCK_M=8,
        BLOCK_N=_SF_VEC_SIZE,
    )
    return quantized, round_scale_to_e8m0(dequant_scale_fp32, protocol)


def _unpack_grouped_rows(grouped: torch.Tensor, cu_seqlens_m: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if grouped.ndim != 3:
        raise ValueError(f"expected 3D grouped tensor, got shape {tuple(grouped.shape)}")
    total_rows = int(out.size(0) if out is not None else cu_seqlens_m[-1].item())
    if out is None:
        out = torch.empty(total_rows, grouped.size(2), dtype=grouped.dtype, device=grouped.device)
    grid = (grouped.size(0), triton.cdiv(grouped.size(1), 16), triton.cdiv(grouped.size(2), 128))
    _unpack_expert_segments_kernel[grid](
        grouped,
        out,
        cu_seqlens_m,
        grouped.stride(0),
        grouped.stride(1),
        grouped.stride(2),
        out.stride(0),
        out.stride(1),
        grouped.size(2),
        BLOCK_M=16,
        BLOCK_N=128,
    )
    return out


def blockscaled_fp8_gemm_grouped(
    a: torch.Tensor,
    w2: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol,
    out_dtype: Optional[torch.dtype] = None,
    capacity: Optional[int] = None,
) -> torch.Tensor:
    protocol = validate_fp8_runtime_support(_blockscaled_protocol(protocol), a.device, quack_enabled=True)
    if get_device_capacity(a.device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_gemm currently supports Blackwell only")

    if a.ndim != 2:
        raise ValueError(f"expected activation to be 2D, got shape {tuple(a.shape)}")
    if w2.ndim != 3:
        raise ValueError(f"expected w2 to be 3D, got shape {tuple(w2.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be a 1D int32 tensor")
    if a.size(1) % _SF_TILE_K != 0:
        raise RuntimeError(f"blockscaled_fp8_gemm requires aligned intermediate width multiple of {_SF_TILE_K}")
    if w2.size(0) % _SF_TILE_M != 0:
        raise RuntimeError(f"blockscaled_fp8_gemm requires aligned hidden width multiple of {_SF_TILE_M}")

    if capacity is not None:
        expert_capacity = capacity
    else:
        expert_capacity = _get_blockscaled_expert_capacity()
    _validate_blockscaled_capacity(cu_seqlens_m, expert_capacity)
    num_experts = w2.size(2)
    weight_fp8, weight_scales = _quantize_w2_cached(w2, protocol)
    a_fp8, a_scales = _pack_quantize_grouped_rows(a, cu_seqlens_m, num_experts, expert_capacity, protocol)
    packed_a_scales = pack_blockscaled_1x32_scales(a_scales, a.size(1))

    grouped_out = torch.empty(
        num_experts,
        expert_capacity,
        w2.size(0),
        dtype=(a.dtype if out_dtype is None else out_dtype),
        device=a.device,
    )

    config = default_config(a.device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_gemm does not support swap_ab configs")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(weight_fp8),
        "D": GemmTensorInfo(grouped_out),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported FP8 blockscaled type/major combination")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = None
    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(packed_a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(weight_scales, leading_dim=1)

    # compile_key must NOT contain dynamic token dimensions (a.size(0) =
    # total_K varies with seqlen / batch; packed_a_scales.size(0) follows).
    # Including them would trigger a CuTe recompile every iteration of
    # production training. Mirror the L2165 ``vk`` pattern: only static
    # dims (dtypes, tile/cluster shapes, num_experts, hidden / intermediate,
    # expert_capacity, major orders, config flags).
    compile_key = (
        "grouped",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        int(num_experts),
        int(expert_capacity),
        int(w2.size(0)),  # hidden (N)
        int(a.size(1)),   # intermediate (K)
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        tensor_infos["D"].major,
        config.pingpong,
        True,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE,
            gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
        _COMPILE_CACHE[compile_key] = compiled

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )
    return grouped_out


@wrap_triton_kernel
@triton.jit
def _quantize_and_pack_kernel(
    src_ptr,
    dst_fp8_ptr,
    dst_packed_scale_ptr,
    rows,
    cols,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_K_TILE: tl.constexpr = 4,
    SAFE_INT64: tl.constexpr = False,
):
    """Fused blockscaled quantize + ISA scale pack — 2D grid version.

    Each program handles BLOCK_ROWS rows × 1 k-tile (GROUPS_PER_K_TILE groups).
    2D grid: (row_blocks, k_tiles) for maximum SM occupancy.
    Packs GROUPS_PER_K_TILE (4) scale bytes into uint32 per k-tile.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    k_tile_idx = tl.program_id(1)
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows
    if SAFE_INT64:
        row_ids = row_ids.to(tl.int64)

    # Pre-compute row-dependent ISA layout (invariant across groups)
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    packed_scale_i32 = tl.zeros([BLOCK_ROWS], dtype=tl.int32)

    for g in tl.range(0, GROUPS_PER_K_TILE):
        group_id = k_tile_idx * GROUPS_PER_K_TILE + g
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        src_ptrs = src_ptr + row_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Pure-integer E8M0 computation (NCU-simplified: see notes below).
        # When biased_exp == 0 (zero/denormal input), mantissa <= 0x7FFFFF and
        # carry <= 1, so biased_exp + carry - 8 <= -7, which max(.,0) clamps to
        # 0 — same result as the original where(biased_exp>0,...,0). The outer
        # clamp(quant_biased_exp,1,254) is also dropped: bf16 max biased_exp =
        # 255 -> e8m0 <= 247 -> quant_biased_exp >= 7, and e8m0 >= 0 ->
        # quant_biased_exp <= 254. Bit-exact with the original on all bf16.
        block_amax = tl.max(tl.abs(values), axis=1)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        carry = (amax_bits & 0x7FFFFF) > 0x600000
        e8m0_clamped = tl.maximum(biased_exp + carry.to(tl.int32) - 8, 0)
        quant_scale = ((254 - e8m0_clamped).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # Pack scale byte into uint32 (little-endian: byte 0 at bits[0:8])
        packed_scale_i32 = packed_scale_i32 | ((e8m0_clamped & 0xFF) << (g * 8))

    # Write 4 scale bytes as single uint32 (row_base_offset is always 4-byte aligned)
    tile_base = (row_tiles * k_tiles + k_tile_idx) * SF_TILE_STORAGE
    packed_offset_i32 = (tile_base + row_base_offset) // 4
    scale_ptr_i32 = dst_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(scale_ptr_i32 + packed_offset_i32, packed_scale_i32, mask=row_mask_1d)


@wrap_triton_kernel
@triton.jit
def _gather_quantize_and_pack_kernel(
    src_ptr,          # original (T, K) tensor
    gather_idx_ptr,   # (TK,) int32/int64 gather indices
    dst_fp8_ptr,      # (TK, K) fp8 output
    dst_packed_scale_ptr,  # ISA-packed scales
    rows,             # TK
    cols,             # K
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    SAFE_INT64: tl.constexpr = False,
):
    """Fused gather + blockscaled quantize + ISA scale pack.

    1D grid: each block handles BLOCK_ROWS rows × ALL scale groups (loop).
    Reads from src[gather_idx[row], col] and writes fp8 + ISA-packed scales
    without materializing a bf16 gathered intermediate tensor.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    # Load gather indices (invariant across groups)
    gather_ids = tl.load(gather_idx_ptr + row_ids, mask=row_mask_1d, other=0)
    if SAFE_INT64:
        row_ids = row_ids.to(tl.int64)
        gather_ids = gather_ids.to(tl.int64)

    # Pre-compute row-dependent ISA layout
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        mask = row_mask_1d[:, None] & col_mask

        # Gather from original tensor using indices
        src_ptrs = src_ptr + gather_ids[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Pure-integer E8M0 computation (no transcendentals)
        block_amax = tl.max(tl.abs(values), axis=1)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile_val = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile_val

        tl.store(dst_packed_scale_ptr + packed_offset, e8m0_byte, mask=row_mask_1d)


def gather_quantize_and_pack_activation(
    x: torch.Tensor,
    gather_idx: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gather + blockscaled FP8 quantize + ISA scale pack.

    Eliminates the bf16 gathered intermediate tensor.
    Instead of: x_gathered = x[gather_idx]; fp8, scales = quantize_and_pack(x_gathered)
    Does:       fp8, scales = gather_quantize_and_pack(x, gather_idx)

    Parameters
    ----------
    x : Tensor (T, K) bf16 — original activation (NOT gathered).
    gather_idx : Tensor (TK,) int32/int64 — token gather indices.

    Returns
    -------
    fp8_data : Tensor (TK, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    TK = gather_idx.shape[0]
    K = x.shape[1]
    check_tensor(x, "x", dtype=(torch.bfloat16, torch.float16), ndim=2)
    check_tensor(gather_idx, "gather_idx", ndim=1, stride0_1=True)
    num_groups = _div_up(K, group_size)
    k_tiles = _div_up(K, _SF_TILE_K)

    fp8_out = torch.empty(TK, K, dtype=torch.float8_e4m3fn, device=x.device)
    per_batch_storage = _storage_per_batch(TK, K)
    if TK % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed_scales = torch.empty(
            (1, per_batch_storage), dtype=torch.uint8, device=x.device
        )
    else:
        packed_scales = torch.full(
            (1, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )

    BLOCK_ROWS = 32
    # INT32 overflow guard: gather_ids index into src (T rows), row_ids index dst (TK rows)
    _max_offset = max((x.shape[0] - 1) * x.stride(0), (TK - 1) * fp8_out.stride(0))
    _needs_int64 = int(_max_offset > 2**31 - 1)

    grid = (_div_up(TK, BLOCK_ROWS),)
    _gather_quantize_and_pack_kernel[grid](
        x,
        gather_idx,
        fp8_out,
        packed_scales,
        TK, K,
        x.stride(0), x.stride(1),
        fp8_out.stride(0), fp8_out.stride(1),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        SAFE_INT64=_needs_int64,
    )
    return fp8_out, packed_scales.view(_E8M0_DTYPE)


def quantize_and_pack_activation(
    x: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 activation to blockscaled FP8 with ISA-packed scales.

    Single fused Triton kernel: bf16 -> fp8 + ISA-layout packed E8M0 scales.
    Eliminates the intermediate raw_scales tensor and fancy-indexing scatter.

    Parameters
    ----------
    x : Tensor (M, K) bf16/fp16 — contiguous activation tensor.
    group_size : block size for scale computation (default 32).

    Returns
    -------
    fp8_data : Tensor (M, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    check_tensor(x, "x", dtype=(torch.bfloat16, torch.float16), ndim=2)
    x = x.contiguous()
    M, K = x.shape
    num_groups = _div_up(K, group_size)
    k_tiles = _div_up(K, _SF_TILE_K)

    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)

    per_batch_storage = _storage_per_batch(M, K)
    # When M and K are tile-aligned, every scale byte is written by the kernel.
    # Skip expensive torch.full fill kernel — use torch.empty instead.
    if M % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed_scales = torch.empty(
            (1, per_batch_storage), dtype=torch.uint8, device=x.device
        )
    else:
        packed_scales = torch.full(
            (1, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )

    # NCU-tuned: BLOCK_ROWS=64 halves grid -> waves/SM ~10 vs 20, ~2% gain at
    # H/I shapes; bit-exact with BLOCK_ROWS=32. Achieves ~93% of practical HBM
    # peak (5.84 / 6.26 TB/s memcpy) at TK=65536 K=3072.
    BLOCK_ROWS = 64
    groups_per_k_tile = _SF_TILE_K // group_size  # 4 for default SF_TILE_K=128, GROUP_SIZE=32
    # INT32 overflow guard: dispatch to int64 branch when row_id * stride > 2^31
    _max_stride = max(x.stride(0), fp8_out.stride(0))
    _needs_int64 = int((M - 1) * _max_stride > 2**31 - 1)

    grid = (_div_up(M, BLOCK_ROWS), k_tiles)
    _quantize_and_pack_kernel[grid](
        x,
        fp8_out,
        packed_scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_K_TILE=groups_per_k_tile,
        SAFE_INT64=_needs_int64,
    )
    return fp8_out, packed_scales.view(_E8M0_DTYPE)



# ---------------------------------------------------------------------------
# Three-step optimized gather: T-quant -> fp8_gather -> scale_gather
# ---------------------------------------------------------------------------
# Replaces the monolithic gather_quantize_and_pack_activation when the source
# tensor T is small enough to fit in L2 cache (e.g., T=4096..8192, K≤4096).
# 1. quantize_and_pack_activation(x) on (T, K) — ~2-8µs for small T
# 2. fp8 data gather via index_select — ~15-25µs (L2-resident reads)
# 3. ISA-packed scale gather — ~3-8µs (very small I/O)
# Total ~20-40µs vs ~96-99µs for gather_quantize_and_pack_activation.
# Numerically identical: scale for row r is computed from x[r, :] regardless.

@wrap_triton_kernel
@triton.jit
def _gather_isa_packed_scales_kernel(
    src_scale_ptr,     # ISA-packed scales for T rows (uint8)
    gather_idx_ptr,    # (TK,) int32/int64 gather indices
    dst_scale_ptr,     # ISA-packed scales for TK rows (uint8)
    TK,                # number of destination rows
    src_k_tiles: tl.constexpr,   # ceil(K / SF_TILE_K) for T layout
    dst_k_tiles: tl.constexpr,   # ceil(K / SF_TILE_K) for TK layout (same value)
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUPS_PER_K_TILE: tl.constexpr,
):
    """Gather ISA-packed scales from T-sized layout to TK-sized layout.

    2D grid: (row_blocks, k_tiles).
    Each block processes BLOCK_ROWS destination rows × 1 k-tile (4 scale groups).
    Reads 4 source scale bytes as uint32, writes 4 dest scale bytes as uint32.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    k_tile_idx = tl.program_id(1)
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < TK

    # Load gather indices
    gather_ids = tl.load(gather_idx_ptr + row_ids, mask=row_mask, other=0)

    # Source row ISA offsets (T-layout)
    src_row_tiles = gather_ids // SF_TILE_M
    src_row_in_tile = gather_ids % SF_TILE_M
    src_row_base_offset = (src_row_in_tile % 32) * 16 + (src_row_in_tile // 32) * 4
    src_tile_base = (src_row_tiles * src_k_tiles + k_tile_idx) * SF_TILE_STORAGE

    # Dest row ISA offsets (TK-layout)
    dst_row_tiles = row_ids // SF_TILE_M
    dst_row_in_tile = row_ids % SF_TILE_M
    dst_row_base_offset = (dst_row_in_tile % 32) * 16 + (dst_row_in_tile // 32) * 4
    dst_tile_base = (dst_row_tiles * dst_k_tiles + k_tile_idx) * SF_TILE_STORAGE

    # Read 4 scale bytes as uint32 from source, write to dest
    src_offset_i32 = (src_tile_base + src_row_base_offset) // 4
    dst_offset_i32 = (dst_tile_base + dst_row_base_offset) // 4

    src_ptr_i32 = src_scale_ptr.to(tl.pointer_type(tl.int32))
    dst_ptr_i32 = dst_scale_ptr.to(tl.pointer_type(tl.int32))

    packed_val = tl.load(src_ptr_i32 + src_offset_i32, mask=row_mask, other=0)
    tl.store(dst_ptr_i32 + dst_offset_i32, packed_val, mask=row_mask)


def fast_gather_quantize_and_pack_activation(
    x: torch.Tensor,
    gather_idx: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three-step optimized gather+quantize: T-quant -> fp8_gather -> scale_gather.

    Numerically identical to gather_quantize_and_pack_activation, but 3-5x
    faster when T << TK (common in MoE with high top-K).

    Step 1: quantize_and_pack_activation on small T-sized tensor (~2-8µs)
    Step 2: gather fp8 data via index_select (~15-25µs, L2-resident source)
    Step 3: gather ISA-packed scales via Triton kernel (~3-8µs)

    Parameters
    ----------
    x : Tensor (T, K) bf16 — original activation (NOT gathered).
    gather_idx : Tensor (TK,) int32/int64 — token gather indices.

    Returns
    -------
    fp8_data : Tensor (TK, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    check_tensor(x, "x", dtype=(torch.bfloat16, torch.float16), ndim=2)
    check_tensor(gather_idx, "gather_idx", ndim=1, stride0_1=True)
    TK = gather_idx.shape[0]
    T, K = x.shape
    k_tiles = _div_up(K, _SF_TILE_K)

    # Step 1: quantize small T-sized tensor
    x_fp8_t, scales_t = quantize_and_pack_activation(x, group_size)

    # Step 2: gather fp8 data (source is T-sized, fits in L2 for typical T)
    gather_idx_i64 = gather_idx.long()
    fp8_out = x_fp8_t.index_select(0, gather_idx_i64)

    # Step 3: gather ISA-packed scales
    per_batch_storage_tk = _storage_per_batch(TK, K)
    if TK % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed_scales_tk = torch.empty(
            (1, per_batch_storage_tk), dtype=torch.uint8, device=x.device
        )
    else:
        packed_scales_tk = torch.full(
            (1, per_batch_storage_tk), 127, dtype=torch.uint8, device=x.device
        )

    BLOCK_ROWS = 128  # Larger blocks: fewer CTAs for lightweight gather
    grid = (_div_up(TK, BLOCK_ROWS), k_tiles)
    groups_per_k_tile = _SF_TILE_K // group_size  # typically 4
    _gather_isa_packed_scales_kernel[grid](
        scales_t.view(torch.uint8),
        gather_idx,
        packed_scales_tk,
        TK,
        src_k_tiles=k_tiles,
        dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS,
        GROUPS_PER_K_TILE=groups_per_k_tile,
    )

    del x_fp8_t, scales_t
    return fp8_out, packed_scales_tk.view(_E8M0_DTYPE)


@wrap_triton_kernel
@triton.jit
def _pad_quantize_and_pack_kernel(
    src_ptr,               # original (total_M, K) bf16
    src_idx_ptr,           # (padded_total,) int64 — padded -> original row, -1 for pad
    dst_fp8_ptr,           # (padded_total, K) fp8 output
    dst_packed_scale_ptr,  # ISA-packed scales for padded layout
    rows,                  # padded_total
    cols,                  # K
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    dst_stride_col,
    k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
):
    """Fused pad + blockscaled quantize + ISA scale pack.

    1D grid: each block handles BLOCK_ROWS rows × ALL scale groups (loop).
    For data rows (src_idx >= 0): reads from src, quantizes, packs scales.
    For padding rows (src_idx == -1): writes fp8 zeros + e8m0 scale 0.
    Avoids materializing a bf16 padded intermediate buffer.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask_1d = row_ids < rows

    # Load source indices (invariant across groups)
    src_rows = tl.load(src_idx_ptr + row_ids, mask=row_mask_1d, other=-1)
    is_data = src_rows >= 0
    safe_src_rows = tl.where(is_data, src_rows, 0)

    # Pre-compute row-dependent ISA layout
    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    for group_id in tl.range(0, NUM_GROUPS):
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        data_mask = is_data[:, None] & col_mask

        # Read from original data (use row 0 for padding rows to avoid OOB)
        src_ptrs = src_ptr + safe_src_rows[:, None] * src_stride_row + col_offsets[None, :] * src_stride_col
        values = tl.load(src_ptrs, mask=data_mask, other=0.0).to(tl.float32)

        # Quantize (zeros for padding rows)
        block_amax = tl.max(tl.abs(values), axis=1)
        amax_bits = block_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_byte = tl.maximum(e8m0_i32, 0).to(tl.uint8)

        quant_biased_exp = 254 - e8m0_i32
        quant_biased_exp = tl.maximum(tl.minimum(quant_biased_exp, 254), 1)
        quant_scale = (quant_biased_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        quantized = (values * quant_scale[:, None]).to(tl.float8e4nv)

        # Write fp8 data
        mask = row_mask_1d[:, None] & col_mask
        dst_ptrs = dst_fp8_ptr + row_ids[:, None] * dst_stride_row + col_offsets[None, :] * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # Write ISA-packed scale
        k_tiles_idx = group_id // (SF_TILE_M // GROUP_SIZE)
        k_in_tile = group_id % (SF_TILE_M // GROUP_SIZE)
        tile_base = (row_tiles * k_tiles + k_tiles_idx) * SF_TILE_STORAGE
        packed_offset = tile_base + row_base_offset + k_in_tile

        tl.store(dst_packed_scale_ptr + packed_offset, e8m0_byte, mask=row_mask_1d)


_SRC_IDX_CACHE: dict[tuple, torch.Tensor] = {}


def pad_quantize_and_pack_activation(
    a: torch.Tensor,
    padded_total: int,
    dst_idx: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused pad + quantize + ISA-pack: avoids bf16 padded intermediate.

    Instead of allocating a (padded_total, K) bf16 buffer, scattering, then
    quantizing 25% more rows, this function:
    1. Computes the inverse mapping (padded_row -> original_row)
    2. Runs a single kernel that reads from original data and writes padded
       fp8 + ISA-packed scales directly

    Parameters
    ----------
    a : Tensor (total_M, K) bf16 — original (non-padded) activation.
    padded_total : int — total padded rows.
    dst_idx : Tensor (total_M,) int64 — original -> padded row mapping.

    Returns
    -------
    fp8_data : Tensor (padded_total, K) float8_e4m3fn
    packed_scales : Tensor (1, packed_size) float8_e8m0fnu in ISA layout
    """
    total_M, K = a.shape
    num_groups = _div_up(K, group_size)
    k_tiles = _div_up(K, _SF_TILE_K)

    # Compute inverse index: padded_row -> original_row (cached)
    cache_key = (total_M, padded_total, id(dst_idx))
    src_idx = _SRC_IDX_CACHE.get(cache_key)
    if src_idx is None:
        src_idx = torch.full((padded_total,), -1, device=a.device, dtype=torch.int64)
        src_idx[dst_idx] = torch.arange(total_M, device=a.device, dtype=torch.int64)
        if len(_SRC_IDX_CACHE) > 16:
            _SRC_IDX_CACHE.clear()
        _SRC_IDX_CACHE[cache_key] = src_idx

    fp8_out = torch.empty(padded_total, K, dtype=torch.float8_e4m3fn, device=a.device)
    per_batch_storage = _storage_per_batch(padded_total, K)
    packed_scales = torch.zeros((1, per_batch_storage), dtype=torch.uint8, device=a.device)

    BLOCK_ROWS = 32  # Larger block vs 16: fewer CTAs, better for padded path
    grid = (_div_up(padded_total, BLOCK_ROWS),)
    _pad_quantize_and_pack_kernel[grid](
        a,
        src_idx,
        fp8_out,
        packed_scales,
        padded_total,
        K,
        a.stride(0),
        a.stride(1),
        fp8_out.stride(0),
        fp8_out.stride(1),
        k_tiles,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        GROUP_SIZE=group_size,
        BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
    )
    return fp8_out, packed_scales.view(_E8M0_DTYPE)


def quantize_and_pack_activation_varlen(
    x: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    group_size: int = _SF_VEC_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize varlen activation to blockscaled FP8 with per-expert ISA-packed scales.

    Three-step pipeline:
    1. Quantize entire (TK, K) in one kernel -> fp8 + raw E8M0 scales (TK, num_groups)
    2. Reshape raw scales to per-expert (E, max_TPE, num_groups) with padding
    3. ISA-pack -> (E, per_expert_storage) packed scales

    For uniform TPE, step 2 is a zero-copy reshape.

    Parameters
    ----------
    x : Tensor (TK, K) bf16 — contiguous, expert-sorted activation.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    group_size : block size for scale computation (default 32).

    Returns
    -------
    fp8_data : Tensor (TK, K) float8_e4m3fn
    packed_scales : Tensor (E, per_expert_storage) float8_e8m0fnu in ISA layout
    """
    x = x.contiguous()
    TK, K = x.shape
    E = cu_seqlens_m.numel() - 1

    # Step 1: Quantize entire tensor in one kernel
    fp8_data, raw_scales = quantize_activation_blockscaled_fast(x, group_size)
    # raw_scales: (TK, num_groups) float8_e8m0fnu
    num_groups = raw_scales.size(1)

    # Step 2: Reshape raw scales to per-expert with padding
    cpu_tuple = _get_cu_seqlens_cpu(cu_seqlens_m)
    cpu_lens = [cpu_tuple[i + 1] - cpu_tuple[i] for i in range(len(cpu_tuple) - 1)]
    max_tpe = max(cpu_lens) if cpu_lens else 0

    if max_tpe == 0:
        per_batch_storage = max(_storage_per_batch(0, K), 1)
        empty_scales = torch.full(
            (E, per_batch_storage), 127, dtype=torch.uint8, device=x.device
        )
        return fp8_data, empty_scales.view(_E8M0_DTYPE)

    raw_uint8 = raw_scales.view(torch.uint8)
    min_tpe = min(cpu_lens)

    if min_tpe == max_tpe and TK == E * max_tpe:
        # Uniform TPE: zero-copy reshape
        padded_scales = raw_uint8.view(E, max_tpe, num_groups)
    else:
        # Non-uniform: vectorized scatter into padded tensor
        padded_scales = torch.full(
            (E, max_tpe, num_groups), 127, dtype=torch.uint8, device=x.device
        )
        row_indices = torch.arange(TK, device=x.device)
        # searchsorted with right=True on cu_seqlens_m[1:] gives expert_id for each row
        expert_ids = torch.searchsorted(cu_seqlens_m[1:].long(), row_indices.long(), right=True)
        local_rows = row_indices - cu_seqlens_m[expert_ids]
        padded_scales[expert_ids, local_rows] = raw_uint8

    # Step 3: ISA-pack the 3D scales
    packed = pack_blockscaled_1x32_scales(
        padded_scales.view(_E8M0_DTYPE), K
    )
    return fp8_data, packed


@wrap_triton_kernel
@triton.jit
def _quantize_and_pack_iso32_kernel(
    src_ptr, dst_fp8_ptr, dst_packed_scale_ptr,
    rows, cols, src_stride_row, src_stride_col,
    dst_stride_row, dst_stride_col, k_tiles,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,       # 32
    BLOCK_ROWS: tl.constexpr,       # 32
    NUM_GROUPS: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_K_TILE: tl.constexpr,
):
    """32×32 isotropic block quantize + ISA pack with row-broadcast scales.

    Same as _quantize_and_pack_kernel but computes amax over the entire
    32×32 tile (32 rows × 32 cols) instead of per-row (1×32).  All 32 rows
    in a block share the same E8M0 scale.

    This makes the quantized fp8 data suitable for BOTH row-major and
    col-major GEMM: after transpose, the 32-column groups map to 32-row
    groups with the same scale value.
    """
    row_base = tl.program_id(0) * BLOCK_ROWS
    k_tile_idx = tl.program_id(1)
    row_ids = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < rows

    # int64 row offsets to avoid overflow when rows * stride > 2^31
    row_ids_i64 = row_ids.to(tl.int64)

    row_tiles = row_ids // SF_TILE_M
    row_in_tile = row_ids % SF_TILE_M
    row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4

    packed_scale_i32 = tl.zeros([BLOCK_ROWS], dtype=tl.int32)

    src_stride_row_i64 = src_stride_row.to(tl.int64)
    dst_stride_row_i64 = dst_stride_row.to(tl.int64)

    for g in tl.range(0, GROUPS_PER_K_TILE):
        group_id = k_tile_idx * GROUPS_PER_K_TILE + g
        col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        col_mask = col_offsets[None, :] < cols
        mask = row_mask[:, None] & col_mask

        src_ptrs = src_ptr + row_ids_i64[:, None] * src_stride_row_i64 + col_offsets[None, :].to(tl.int64) * src_stride_col
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        # 32×32 isotropic amax: reduce over BOTH axes
        row_amax = tl.max(tl.abs(values), axis=1)        # (32,) per-row max
        tile_amax = tl.max(row_amax)                       # scalar: max over all 32 rows
        amax_bits = tile_amax.to(tl.int32, bitcast=True)
        biased_exp = (amax_bits >> 23) & 0xFF
        mantissa_bits = amax_bits & 0x7FFFFF
        carry = tl.where(mantissa_bits > 0x600000, 1, 0)
        e8m0_i32 = biased_exp - 8 + carry
        e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
        e8m0_clamped = tl.maximum(e8m0_i32, 0)

        quant_exp = 254 - e8m0_i32
        quant_exp = tl.maximum(tl.minimum(quant_exp, 254), 1)
        quant_scale = (quant_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        quantized = (values * quant_scale).to(tl.float8e4nv)

        dst_ptrs = dst_fp8_ptr + row_ids_i64[:, None] * dst_stride_row_i64 + col_offsets[None, :].to(tl.int64) * dst_stride_col
        tl.store(dst_ptrs, quantized, mask=mask)

        # Broadcast: all 32 rows get the SAME scale byte
        e8m0_broadcast = tl.full([BLOCK_ROWS], e8m0_clamped.to(tl.int32), dtype=tl.int32)
        packed_scale_i32 = packed_scale_i32 | ((e8m0_broadcast & 0xFF) << (g * 8))

    tile_base = (row_tiles.to(tl.int64) * k_tiles + k_tile_idx) * SF_TILE_STORAGE
    packed_offset_i32 = (tile_base + row_base_offset.to(tl.int64)) // 4
    scale_ptr_i32 = dst_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(scale_ptr_i32 + packed_offset_i32, packed_scale_i32, mask=row_mask)


def quantize_and_pack_weight_iso32(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """32×32 isotropic blockscaled quantize for weights.

    Same interface as quantize_and_pack_activation but uses 32×32 block amax
    (all 32 rows share the same scale).  This enables transpose without
    re-quantization: the transposed fp8 data has compatible scales.

    Parameters
    ----------
    w : (M, K) bf16 contiguous weight tensor (M and K should be multiples of 32).

    Returns
    -------
    fp8_data : (M, K) float8_e4m3fn
    packed_scales : (1, packed_size) float8_e8m0fnu ISA-packed
    """
    check_tensor(w, "w", dtype=torch.bfloat16, ndim=2)
    w = w.contiguous()
    M, K = w.shape
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    BLOCK_ROWS = 32
    k_tiles = _div_up(K, _SF_TILE_K)
    groups_per_k_tile = _SF_TILE_K // GROUP_SIZE
    num_groups = _div_up(K, GROUP_SIZE)

    fp8_out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=w.device)
    per_batch = _storage_per_batch(M, K)
    if M % _SF_TILE_M == 0 and K % _SF_TILE_K == 0:
        packed = torch.empty((1, per_batch), dtype=torch.uint8, device=w.device)
    else:
        packed = torch.full((1, per_batch), 127, dtype=torch.uint8, device=w.device)

    grid = (_div_up(M, BLOCK_ROWS), k_tiles)
    _quantize_and_pack_iso32_kernel[grid](
        w, fp8_out, packed,
        M, K, w.stride(0), w.stride(1),
        fp8_out.stride(0), fp8_out.stride(1), k_tiles,
        fp8_max=_FP8_E4M3_MAX,
        GROUP_SIZE=GROUP_SIZE, BLOCK_ROWS=BLOCK_ROWS,
        NUM_GROUPS=num_groups,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_K_TILE=groups_per_k_tile,
    )
    return fp8_out, packed.view(_E8M0_DTYPE)


def _quantize_weight_3d_triton(
    w_enk: torch.Tensor,
    *, isotropic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous 3D (E, N, K) weight tensor using the fast Triton kernel.

    Default is per-row 1×32 blockscaled FP8 (same format as activations).

    The legacy ``isotropic=True`` mode uses 32×32 block quantization where all 32
    rows in a tile share one E8M0 scale. That mode was originally intended for
    transpose-reuse (one quant for both forward and wgrad), but the current code
    paths re-quantize the transpose from BF16 anyway, so the property is unused.
    Iso32 is also strictly equal-or-worse in precision (E4M3 floating-point rounding
    means a different scale just shifts the precision window — see
    ``tests/ops/audit_iso32_numerics.py``) and offers no measurable speed advantage
    (see ``tests/ops/bench_iso32_quant_nsys.py``). Kept opt-in only for back-compat
    with any external caller that still relies on the byte-identical transpose
    invariant; **do not add new callers**.

    Exploits the fact that when N % SF_TILE_M == 0, the ISA scale tile boundaries
    align perfectly at expert boundaries. So (E, N, K) can be reshaped to (E*N, K),
    quantized as a single 2D tensor, and reshaped back — producing identical results
    to per-expert quantization but in a single kernel launch.

    Returns (w_fp8 (E, N, K), packed_scales) with ISA-packed E8M0 scales.
    """
    E, N, K = w_enk.shape
    assert w_enk.is_contiguous(), "Weight must be contiguous (E, N, K)"

    quant_fn = quantize_and_pack_weight_iso32 if isotropic else quantize_and_pack_activation

    if N % _SF_TILE_M == 0:
        w_2d = w_enk.reshape(E * N, K)
        fp8_2d, packed_scales = quant_fn(w_2d)
        return fp8_2d.reshape(E, N, K), packed_scales
    else:
        fp8_slices = []
        scale_slices = []
        for e in range(E):
            fp8_e, scales_e = quant_fn(w_enk[e])
            fp8_slices.append(fp8_e)
            scale_slices.append(scales_e)
        return torch.stack(fp8_slices), torch.cat(scale_slices, dim=-1)


def precompute_weight_fp8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize a 3D expert weight to blockscaled FP8 with ISA-packed scales (cached).

    Converts (dim0, dim1, E) bf16 weights to (E, dim0, dim1) fp8 + packed scales.
    Uses a dedicated cache (NOT shared with fused_dgated variants which quantize
    in transposed layout with scales along a different dimension).

    When used with MoE.refresh_fp8_shadow_weights(), the cache is pre-populated
    and this function returns instantly (zero quantize overhead).

    Parameters
    ----------
    w : Tensor (dim0, dim1, E) bf16 — expert weights in any layout.

    Returns
    -------
    w_fp8 : Tensor (E, dim0, dim1) float8_e4m3fn — contiguous row-major.
    w_scales : Tensor packed float8_e8m0fnu in ISA layout
    """
    key = (
        w.data_ptr(),
        _tensor_version(w),
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _VARLEN_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w_ehi = w.permute(2, 0, 1).contiguous()
    w_fp8, w_scales_packed = _quantize_weight_3d_triton(w_ehi)
    result = (w_fp8, w_scales_packed)
    if len(_VARLEN_WEIGHT_CACHE) > 8:
        _VARLEN_WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE[key] = result
    return result


_VARLEN_WEIGHT_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...]],
    tuple[torch.Tensor, torch.Tensor],
] = {}


_FUSED_WEIGHT_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...]],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def precompute_weight_fp8_for_fused_gated(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for fused gemm_gated blockscaled path (cached).

    gemm_gated_tuned expects B in (L, K, N) format and applies .mT internally
    to get (L, N, K).  Blockscaled 1x32 quantisation is done on the physical
    (contiguous) layout which must match what CUTLASS sees after .mT.

    We quantise in (E, N, K) contiguous layout (scales along K), then return
    the .mT view (E, K, N) so that gemm_gated_tuned's internal .mT recovers
    the original contiguous (E, N, K) layout with correctly-aligned scales.

    Parameters
    ----------
    w : Tensor (2I, H, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, K, N) float8_e4m3fn — .mT view of contiguous (E, N, K).
        gemm_gated_tuned's internal B.mT will recover the contiguous (E, N, K).
    w_scales : Tensor packed float8_e8m0fnu in ISA layout (scales along K-axis
        of the contiguous layout)
    """
    key = (
        w.data_ptr(),
        _tensor_version(w),
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # w is (2I, H, E) -> (E, 2I, H) contiguous = (E, N, K) physical layout
    w_enk = w.permute(2, 1, 0).mT.contiguous()  # (E, N=2I, K=H) contiguous
    w_fp8_enk, w_scales_packed = _quantize_weight_3d_triton(w_enk)
    # Return .mT view (E, K, N) so gemm_gated_tuned's B.mT recovers (E, N, K)
    w_fp8_ekn = w_fp8_enk.mT  # stride view — same physical memory
    result = (w_fp8_ekn, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result
    return result


def precompute_weight_fp8_for_fused_dgated(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for fused gemm_dgated blockscaled path (cached).

    gemm_dgated_tuned expects B in (L, K, N) format and applies .mT internally
    to get (L, N, K).  For w2 (H, I, E), the input to gemm_dgated_tuned is
    w2.permute(2, 0, 1) = (E, H, I) = (E, K, N).  After .mT: (E, I, H) = (E, N, K).

    We quantise in (E, N=I, K=H) contiguous layout (scales along K=H), then
    return the .mT view (E, H, I) so gemm_dgated_tuned's internal .mT recovers
    the original contiguous (E, I, H) layout with correctly-aligned scales.

    Parameters
    ----------
    w : Tensor (H, I, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, H, I) float8_e4m3fn — .mT view of contiguous (E, I, H).
    w_scales : Tensor packed float8_e8m0fnu in ISA layout (scales along K=H)
    """
    key = (
        w.data_ptr(),
        _tensor_version(w),
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # w is (H, I, E) -> (E, I, H) contiguous = (E, N, K) physical layout
    w_enk = w.permute(2, 1, 0).contiguous()  # (E, N=I, K=H) contiguous
    w_fp8_enk, w_scales_packed = _quantize_weight_3d_triton(w_enk)
    # Return .mT view (E, K=H, N=I) so gemm_dgated_tuned's B.mT recovers (E, N=I, K=H)
    w_fp8_ekn = w_fp8_enk.mT  # stride view — same physical memory
    result = (w_fp8_ekn, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result
    return result


def precompute_weight_fp8_for_direct_fused_dgated(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for direct low-level gemm_dgated blockscaled path.

    Reuses the fused_dgated cache — identical physical (E, N, K) contiguous layout.
    Returns the contiguous tensor (not .mT view) that the low-level kernel consumes
    directly, bypassing gemm_interface's B.mT.contiguous() copy.

    Parameters
    ----------
    w : Tensor (H, I, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, N=I, K=H) float8_e4m3fn — contiguous physical layout.
    w_scales : Tensor packed float8_e8m0fnu in ISA layout (scales along K=H).
    """
    key = (
        w.data_ptr(),
        _tensor_version(w),
        tuple(w.shape),
        tuple(w.stride()),
    )
    # Check unified fused cache first (shared with precompute_weight_fp8_for_fused_dgated)
    cached = _FUSED_WEIGHT_CACHE.get(key)
    if cached is not None:
        w_fp8_view, w_scales = cached
        # fused_dgated stores .mT view (E, H, I); we need contiguous (E, I, H)
        if not w_fp8_view.is_contiguous():
            return w_fp8_view.mT, w_scales  # .mT of .mT = contiguous original
        return w_fp8_view, w_scales

    w_enk = w.permute(2, 1, 0).contiguous()  # (E, N=I, K=H) contiguous
    w_fp8_enk, w_scales_packed = _quantize_weight_3d_triton(w_enk)
    # Store contiguous in fused cache; fused_dgated will create .mT view from it
    result_contiguous = (w_fp8_enk, w_scales_packed)
    result_view = (w_fp8_enk.mT, w_scales_packed)
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[key] = result_view  # Store .mT view (fused_dgated convention)
    return result_contiguous  # Return contiguous (direct convention)


# ---------------------------------------------------------------------------
# Fused pair quantize: read bf16 once, produce two transposed FP8 outputs.
# ---------------------------------------------------------------------------
# The four per-step prequantize calls do redundant work: each separately
# permute+contiguous'es the BF16 weight (a strided read + contig write of the
# full tensor) and then reads it back to compute amax+quantize.  When the same
# weight is needed in two transposed layouts (forward = scales-along-K,
# backward = scales-along-N), most of that DRAM traffic is wasted.
#
# This fused kernel reads the source BF16 ONCE (strided), and writes:
#   - output A: fp8 (E, N, K) row-major + ISA-packed scales-along-K
#   - output B: fp8 (E, K, N) row-major + ISA-packed scales-along-N
# Per (E, N=128, K=128) tile, both A's and B's ISA scale tile are produced
# in-flight from registers — no second BF16 read, no .contiguous() copy.
# ---------------------------------------------------------------------------


@wrap_triton_kernel
@triton.jit
def _quantize_pair_kernel(
    src_ptr,                    # bf16, any-stride (E, N, K) view
    A_fp8_ptr, A_packed_scale_ptr,
    B_fp8_ptr, B_packed_scale_ptr,
    E, N, K,
    src_stride_e, src_stride_n, src_stride_k,
    A_stride_e, A_stride_n, A_stride_k,
    B_stride_e, B_stride_k, B_stride_n,
    A_k_tiles_per_row, B_k_tiles_per_row,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SF_TILE_M: tl.constexpr,
    SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_TILE_K: tl.constexpr,   # BLOCK_K / GROUP_SIZE  (e.g. 4)
    GROUPS_PER_TILE_N: tl.constexpr,   # BLOCK_N / GROUP_SIZE  (e.g. 4)
):
    """Fuse two transposed blockscaled FP8 quantizations into one kernel.

    Per program (e, n_tile, k_tile):
      1. Load BLOCK_N × BLOCK_K bf16 tile from src (strided).
      2. For each of GROUPS_PER_TILE_K K-groups (size GROUP_SIZE=32):
           - per-row amax over the group → BLOCK_N E8M0 scales for A.
           - quantize and store fp8 to A.
           - pack 4 scale bytes per N-row into uint32; store to A's ISA tile.
      3. For each of GROUPS_PER_TILE_N N-groups (size GROUP_SIZE=32):
           - per-col amax over the group → BLOCK_K E8M0 scales for B.
           - quantize and store fp8 to B (transposed offsets).
           - pack 4 scale bytes per K-row into uint32; store to B's ISA tile.
    """
    e = tl.program_id(0)
    n_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    n_offs = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = n_offs < N
    k_mask = k_offs < K
    tile_mask = n_mask[:, None] & k_mask[None, :]

    # int64 base pointers to avoid stride*index overflow on multi-GB tensors
    src_e_base = src_ptr + e.to(tl.int64) * src_stride_e
    src_ptrs = (
        src_e_base
        + n_offs[:, None].to(tl.int64) * src_stride_n
        + k_offs[None, :].to(tl.int64) * src_stride_k
    )
    vals = tl.load(src_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
    abs_vals = tl.abs(vals)

    # ------------------------------------------------------------------
    # Output A: scales along K, layout (E, N, K) row-major
    # ------------------------------------------------------------------
    # Per-row amax for each of the 4 K-groups in this tile.
    abs_grouped_k = tl.reshape(abs_vals, (BLOCK_N, GROUPS_PER_TILE_K, GROUP_SIZE))
    row_amax = tl.max(abs_grouped_k, axis=2)              # (BLOCK_N, GROUPS_PER_TILE_K)

    amax_bits = row_amax.to(tl.int32, bitcast=True)
    biased_exp = (amax_bits >> 23) & 0xFF
    mantissa_bits = amax_bits & 0x7FFFFF
    carry = tl.where(mantissa_bits > 0x600000, 1, 0)
    e8m0_i32 = biased_exp - 8 + carry
    e8m0_i32 = tl.where(biased_exp > 0, e8m0_i32, 0)
    e8m0_clamped_a = tl.maximum(e8m0_i32, 0)
    quant_exp = 254 - e8m0_i32
    quant_exp = tl.maximum(tl.minimum(quant_exp, 254), 1)
    quant_scale_a = (quant_exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    # broadcast scale to (BLOCK_N, GROUPS_PER_TILE_K, GROUP_SIZE) → (BLOCK_N, BLOCK_K)
    quant_scale_a_full = tl.reshape(
        tl.broadcast_to(quant_scale_a[:, :, None], (BLOCK_N, GROUPS_PER_TILE_K, GROUP_SIZE)),
        (BLOCK_N, BLOCK_K),
    )
    fp8_a = (vals * quant_scale_a_full).to(tl.float8e4nv)

    A_e_base = A_fp8_ptr + e.to(tl.int64) * A_stride_e
    A_ptrs = (
        A_e_base
        + n_offs[:, None].to(tl.int64) * A_stride_n
        + k_offs[None, :].to(tl.int64) * A_stride_k
    )
    tl.store(A_ptrs, fp8_a, mask=tile_mask)

    # Pack 4 K-group scale bytes per N-row into one uint32 word.
    # e8m0_clamped_a is (BLOCK_N, GROUPS_PER_TILE_K). Build uint32 per row:
    g_idx = tl.arange(0, GROUPS_PER_TILE_K)
    packed_a = tl.sum((e8m0_clamped_a & 0xFF) << (g_idx[None, :] * 8), axis=1)

    # ISA layout: tile_base = (row_tile * k_tiles + k_tile_idx) * SF_TILE_STORAGE.
    # Match the layout of the existing kernel which reshapes (E, N, K) -> (E*N, K)
    # and uses a global 2D row_tile.  Since N % SF_TILE_M == 0, the per-expert
    # offset is exactly e * (N / SF_TILE_M).
    row_tile_a = e * (N // SF_TILE_M) + n_tile
    row_in_tile = tl.arange(0, BLOCK_N) % SF_TILE_M
    row_off_a = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    tile_base_a = (row_tile_a * A_k_tiles_per_row + k_tile).to(tl.int64) * SF_TILE_STORAGE
    A_scale_i32_ptr = A_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(A_scale_i32_ptr + (tile_base_a + row_off_a.to(tl.int64)) // 4, packed_a, mask=n_mask)

    # ------------------------------------------------------------------
    # Output B: scales along N (= rows of B's (E, K, N) layout), transposed
    # ------------------------------------------------------------------
    # Per-col amax for each of the 4 N-groups in this tile.
    abs_grouped_n = tl.reshape(abs_vals, (GROUPS_PER_TILE_N, GROUP_SIZE, BLOCK_K))
    col_amax = tl.max(abs_grouped_n, axis=1)              # (GROUPS_PER_TILE_N, BLOCK_K)

    amax_bits_b = col_amax.to(tl.int32, bitcast=True)
    biased_exp_b = (amax_bits_b >> 23) & 0xFF
    mantissa_bits_b = amax_bits_b & 0x7FFFFF
    carry_b = tl.where(mantissa_bits_b > 0x600000, 1, 0)
    e8m0_i32_b = biased_exp_b - 8 + carry_b
    e8m0_i32_b = tl.where(biased_exp_b > 0, e8m0_i32_b, 0)
    e8m0_clamped_b = tl.maximum(e8m0_i32_b, 0)
    quant_exp_b = 254 - e8m0_i32_b
    quant_exp_b = tl.maximum(tl.minimum(quant_exp_b, 254), 1)
    quant_scale_b = (quant_exp_b.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    quant_scale_b_full = tl.reshape(
        tl.broadcast_to(quant_scale_b[:, None, :], (GROUPS_PER_TILE_N, GROUP_SIZE, BLOCK_K)),
        (BLOCK_N, BLOCK_K),
    )
    fp8_b_NK = (vals * quant_scale_b_full).to(tl.float8e4nv)
    # B is laid out (E, K, N): we wrote N as outer, K as inner — store at
    # B[e, k_offs, n_offs] with strides (B_stride_k, B_stride_n).
    B_e_base = B_fp8_ptr + e.to(tl.int64) * B_stride_e
    B_ptrs = (
        B_e_base
        + n_offs[:, None].to(tl.int64) * B_stride_n      # N is the inner dim of B's row layout
        + k_offs[None, :].to(tl.int64) * B_stride_k      # K is the row dim
    )
    tl.store(B_ptrs, fp8_b_NK, mask=tile_mask)

    # Pack 4 N-group scale bytes per K-row into one uint32 word. e8m0_clamped_b
    # is (GROUPS_PER_TILE_N, BLOCK_K) — transpose semantically: each K-row
    # gets GROUPS_PER_TILE_N scale bytes (one per N-group).
    g2_idx = tl.arange(0, GROUPS_PER_TILE_N)
    packed_b = tl.sum((e8m0_clamped_b & 0xFF) << (g2_idx[:, None] * 8), axis=0)  # (BLOCK_K,)

    # B's M-axis is K (since B has shape (E, K, N) row-major). Match 2D-reshape
    # layout: row_tile_b = e * (K / SF_TILE_M) + k_tile.
    row_tile_b = e * (K // SF_TILE_M) + k_tile
    row_in_tile_b = tl.arange(0, BLOCK_K) % SF_TILE_M
    row_off_b = (row_in_tile_b % 32) * 16 + (row_in_tile_b // 32) * 4
    tile_base_b = (row_tile_b * B_k_tiles_per_row + n_tile).to(tl.int64) * SF_TILE_STORAGE
    B_scale_i32_ptr = B_packed_scale_ptr.to(tl.pointer_type(tl.int32))
    tl.store(B_scale_i32_ptr + (tile_base_b + row_off_b.to(tl.int64)) // 4, packed_b, mask=k_mask)


def _quantize_weight_pair_3d_triton(
    src_enk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pass dual quantize of an (E, N, K) bf16 tensor.

    Returns four tensors:
        (A_fp8 (E, N, K), A_scales packed,  B_fp8 (E, K, N), B_scales packed)

    A is the "scales along K" layout (forward GEMM convention).
    B is the "scales along N" layout (transposed; backward GEMM convention).
    """
    assert src_enk.dtype == torch.bfloat16, "pair quantize requires bf16 src"
    assert src_enk.ndim == 3, "expected (E, N, K) shape"
    E, N, K = src_enk.shape
    assert N % _SF_TILE_M == 0 and K % _SF_TILE_K == 0, (
        f"pair-quantize fast path requires N%{_SF_TILE_M}==0 and K%{_SF_TILE_K}==0, got N={N} K={K}"
    )

    BLOCK_N = _SF_TILE_M       # 128
    BLOCK_K = _SF_TILE_K       # 128
    GROUP_SIZE = _SF_VEC_SIZE  # 32
    groups_per_tile_k = BLOCK_K // GROUP_SIZE
    groups_per_tile_n = BLOCK_N // GROUP_SIZE

    A_fp8 = torch.empty((E, N, K), dtype=torch.float8_e4m3fn, device=src_enk.device)
    B_fp8 = torch.empty((E, K, N), dtype=torch.float8_e4m3fn, device=src_enk.device)

    A_per_batch = _storage_per_batch(N, K)
    B_per_batch = _storage_per_batch(K, N)
    A_scales = torch.empty((E, A_per_batch), dtype=torch.uint8, device=src_enk.device)
    B_scales = torch.empty((E, B_per_batch), dtype=torch.uint8, device=src_enk.device)

    A_k_tiles = K // _SF_TILE_K
    B_k_tiles = N // _SF_TILE_K  # B's K-axis is the original N

    grid = (E, _div_up(N, BLOCK_N), _div_up(K, BLOCK_K))
    _quantize_pair_kernel[grid](
        src_enk,
        A_fp8, A_scales,
        B_fp8, B_scales,
        E, N, K,
        src_enk.stride(0), src_enk.stride(1), src_enk.stride(2),
        A_fp8.stride(0), A_fp8.stride(1), A_fp8.stride(2),
        B_fp8.stride(0), B_fp8.stride(1), B_fp8.stride(2),
        A_k_tiles, B_k_tiles,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        GROUPS_PER_TILE_K=groups_per_tile_k,
        GROUPS_PER_TILE_N=groups_per_tile_n,
        num_warps=4,
    )
    # Per-expert ISA scales are concatenated along last dim by other helpers;
    # match _quantize_weight_3d_triton's behaviour: when N%SF_TILE_M==0 the
    # scales for the (E,N,K) tensor are a single contiguous (1, E*per_batch)
    # block.  Reshape (E, per_batch) → (1, E*per_batch) view-only.
    A_scales_flat = A_scales.view(1, E * A_per_batch)
    B_scales_flat = B_scales.view(1, E * B_per_batch)
    return A_fp8, A_scales_flat.view(_E8M0_DTYPE), B_fp8, B_scales_flat.view(_E8M0_DTYPE)


# ---------------------------------------------------------------------------
# Multi-stream warmup: launch the 4 weight prequantize kernels concurrently
# on separate CUDA streams.  Each call is HBM-bandwidth-bound but only uses a
# small fraction of available SMs, so concurrent execution overlaps cleanly.
# Eliminates ~50% of the per-step warmup wall-clock vs. sequential calls.
# ---------------------------------------------------------------------------
_WARMUP_STREAMS: list = []


def _get_warmup_streams(n: int = 4) -> list:
    global _WARMUP_STREAMS
    if len(_WARMUP_STREAMS) < n:
        _WARMUP_STREAMS = [torch.cuda.Stream() for _ in range(n)]
    return _WARMUP_STREAMS[:n]


def _cache_pair_w1(w1: torch.Tensor) -> None:
    """One-pass quantize w1 in both transposed layouts; populate both caches.

    w1 has logical shape (2I, H, E) bf16.  Two consumers want it as:
      A (forward, fused_gated):  contig (E, 2I, H), scales-along-K=H
      B (backward, w1T_varlen):  contig (E, H, 2I), scales-along-K=2I
    A and B are physical transposes of each other in (N, K).  We read the
    strided bf16 from w1 directly (no .contiguous() copies) and write both.
    """
    # Source view in (E, N=2I, K=H) order with original strides.  The four
    # dims of `_quantize_pair_kernel` are (e, n, k) and the kernel uses
    # the supplied strides — we don't need a copy.
    w1_enk = w1.permute(2, 0, 1)  # (E, 2I, H), strided
    A_fp8, A_scales, B_fp8, B_scales = _quantize_weight_pair_3d_triton(w1_enk)

    # A populates _FUSED_WEIGHT_CACHE (returned as .mT view per the existing
    # convention of precompute_weight_fp8_for_fused_gated).
    fused_key = (
        w1.data_ptr(),
        _tensor_version(w1),
        tuple(w1.shape),
        tuple(w1.stride()),
    )
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    _FUSED_WEIGHT_CACHE[fused_key] = (A_fp8.mT, A_scales)

    # B populates _VARLEN_WEIGHT_CACHE keyed under w1.permute(1, 0, 2),
    # matching what precompute_weight_fp8(w1.permute(1,0,2)) would key on.
    w1_perm = w1.permute(1, 0, 2)
    varlen_key = (
        w1_perm.data_ptr(),
        _tensor_version(w1_perm),
        tuple(w1_perm.shape),
        tuple(w1_perm.stride()),
    )
    if len(_VARLEN_WEIGHT_CACHE) > 8:
        _VARLEN_WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE[varlen_key] = (B_fp8, B_scales)


def _cache_pair_w2(w2: torch.Tensor) -> None:
    """One-pass quantize w2 in both transposed layouts; populate both caches.

    w2 has logical shape (H, I, E) bf16.  Two consumers want it as:
      A (forward, varlen):                 contig (E, H, I), scales-along-K=I
      B (backward, direct_fused_dgated):   contig (E, I, H), scales-along-K=H
    """
    w2_enk = w2.permute(2, 0, 1)  # (E, H, I), strided
    A_fp8, A_scales, B_fp8, B_scales = _quantize_weight_pair_3d_triton(w2_enk)

    # A populates _VARLEN_WEIGHT_CACHE keyed on w2.
    varlen_key = (
        w2.data_ptr(),
        _tensor_version(w2),
        tuple(w2.shape),
        tuple(w2.stride()),
    )
    if len(_VARLEN_WEIGHT_CACHE) > 8:
        _VARLEN_WEIGHT_CACHE.clear()
    _VARLEN_WEIGHT_CACHE[varlen_key] = (A_fp8, A_scales)

    # B populates _FUSED_WEIGHT_CACHE under w2's key, with the
    # ``direct_fused_dgated`` convention: stored as the contiguous (E, I, H)
    # tensor.  fused_dgated readers create the .mT view on demand.
    fused_key = (
        w2.data_ptr(),
        _tensor_version(w2),
        tuple(w2.shape),
        tuple(w2.stride()),
    )
    if len(_FUSED_WEIGHT_CACHE) > 8:
        _FUSED_WEIGHT_CACHE.clear()
    # precompute_weight_fp8_for_direct_fused_dgated returns the contig tensor
    # but stores the .mT view in cache so fused_dgated still works.  Mirror.
    _FUSED_WEIGHT_CACHE[fused_key] = (B_fp8.mT, B_scales)


def precompute_weight_fp8_warmup(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> None:
    """Warm both forward+backward FP8 weight caches for w1 and w2 concurrently.

    Replaces the four sequential calls::

        precompute_weight_fp8_for_fused_gated(w1)            # UpProj fwd
        precompute_weight_fp8(w2)                            # DownProj fwd varlen
        precompute_weight_fp8_for_direct_fused_dgated(w2)    # DownProj bwd dgated
        precompute_weight_fp8(w1.permute(1, 0, 2))           # UpProj bwd w1T_varlen

    Implementation: a single fused Triton pair-quantize kernel per weight
    reads the BF16 source ONCE (strided, no .contiguous() copy) and writes
    both transposed FP8 outputs + both ISA-packed scale tensors in one pass.
    The two per-weight kernels then run on separate streams so w1 and w2
    overlap.  Caches populated here are picked up transparently by the
    existing `precompute_weight_fp8*` lookup paths.

    Parameters
    ----------
    w1 : Tensor (2I, H, E) bf16 — UpProj expert weights.
    w2 : Tensor (H, I, E) bf16 — DownProj expert weights.
    """
    streams = _get_warmup_streams(2)
    cur = torch.cuda.current_stream()
    fork_ev = cur.record_event()
    for st in streams:
        st.wait_event(fork_ev)

    with torch.cuda.stream(streams[0]):
        _cache_pair_w1(w1)
    with torch.cuda.stream(streams[1]):
        _cache_pair_w2(w2)

    for st in streams:
        cur.wait_event(st.record_event())


def blockscaled_fp8_gemm_varlen(
    a: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol | None = None,
    out_dtype: Optional[torch.dtype] = None,
    a_scales: Optional[torch.Tensor] = None,
    w_fp8: Optional[torch.Tensor] = None,
    w_scales: Optional[torch.Tensor] = None,
    assume_aligned: bool = False,
) -> torch.Tensor:
    """Blockscaled FP8 GEMM using the varlen scheduler.

    Supports two calling conventions:

    **Pre-quantized (preferred)** — pass fp8 data + packed scales directly::

        a_fp8, a_scales = quantize_and_pack_activation(x_bf16)
        w_fp8, w_scales = precompute_weight_fp8(w_bf16)
        out = blockscaled_fp8_gemm_varlen(
            a_fp8, w_bf16, cu,             # w_bf16 only for shape/device
            a_scales=a_scales, w_fp8=w_fp8, w_scales=w_scales,
            out_dtype=torch.bfloat16,
        )

    **Legacy (backward compat)** — pass bf16 and let the function quantize::

        out = blockscaled_fp8_gemm_varlen(x_bf16, w_bf16, cu, protocol=protocol)

    Parameters
    ----------
    a : Tensor (total_M, K) — fp8 (pre-quantized) or bf16 (legacy).
    w : Tensor (H, I, E) bf16 — expert weights (used for shape; ignored
        if w_fp8/w_scales are provided).
    cu_seqlens_m : Tensor (E+1,) int32 — expert boundaries.
    protocol : FP8Protocol — required for legacy bf16 path.
    out_dtype : output dtype (defaults to bf16).
    a_scales : packed ISA-layout scales for ``a`` (pre-quantized path).
    w_fp8 : pre-quantized weight fp8 (E,H,I).
    w_scales : packed ISA-layout scales for weights.
    assume_aligned : if True, skip the CTA-tile alignment check (avoids
        D2H sync).  Caller must guarantee all expert segments in
        cu_seqlens_m are multiples of the CTA tile size (128).
    """

    # --- Hot path: pre-quantized + assume_aligned (production steady-state) ---
    if (
        assume_aligned
        and a.dtype == torch.float8_e4m3fn
        and a_scales is not None
        and w_fp8 is not None
        and w_scales is not None
    ):
        total_M, K = a.shape
        H = w.shape[0]
        num_experts = w.shape[2]
        d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        return _run_cutlass_blockscaled_gemm(
            a, a_scales, w_fp8, w_scales, cu_seqlens_m,
            total_M, K, H, num_experts, d_dtype, a.device,
        )

    # --- Full validation path (legacy / first few iterations) ---
    device = a.device
    if get_device_capacity(device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_gemm_varlen requires Blackwell (sm_100)")

    if a.ndim != 2:
        raise ValueError(f"expected activation 2D, got {tuple(a.shape)}")
    if w.ndim != 3:
        raise ValueError(f"expected w 3D (H, I, E), got {tuple(w.shape)}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be 1D int32")

    total_M, K = a.shape
    H, I, num_experts = w.shape
    assert K == I, f"K mismatch: a has {K}, w has I={I}"
    assert cu_seqlens_m.shape[0] == num_experts + 1

    if K % _SF_TILE_K != 0:
        raise RuntimeError(f"K must be multiple of {_SF_TILE_K}, got {K}")
    if H % _SF_TILE_M != 0:
        raise RuntimeError(f"H must be multiple of {_SF_TILE_M}, got {H}")

    # --- Determine if inputs are pre-quantized ---
    pre_quantized_act = (a.dtype == torch.float8_e4m3fn and a_scales is not None)
    pre_quantized_wt = (w_fp8 is not None and w_scales is not None)

    # --- CTA tile alignment padding ---
    if assume_aligned or torch.cuda.is_current_stream_capturing():
        needs_pad = False
    else:
        needs_pad, padded_cu, padded_total, dst_idx = _get_padding_plan(cu_seqlens_m, total_M)

    if needs_pad:
        # --- Weight quantization (shared by both paths) ---
        if pre_quantized_wt:
            weight_fp8, weight_scales = w_fp8, w_scales
        else:
            bs_proto = _blockscaled_protocol(protocol or FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32))
            weight_fp8, weight_scales = _quantize_w2_cached(w, bs_proto)

        if pre_quantized_act:
            # Pre-quantized with non-aligned segments: dequant is lossy.
            # This path should be avoided (see _all_segments_128_aligned guard
            # in functional/__init__.py). Fallback: raw cast + re-quantize.
            a_bf16 = a.to(torch.bfloat16)
        else:
            a_bf16 = a

        # Fused pad + quantize + ISA-pack: avoids the bf16 padded buffer.
        padded_fp8, padded_scales = pad_quantize_and_pack_activation(
            a_bf16, padded_total, dst_idx,
        )

        d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        padded_out = _run_cutlass_blockscaled_gemm(
            padded_fp8, padded_scales,
            weight_fp8, weight_scales,
            padded_cu,
            padded_total, K, H, num_experts,
            d_dtype, device,
        )
        return padded_out[dst_idx]

    # --- Weight quantization ---
    if pre_quantized_wt:
        weight_fp8, weight_scales = w_fp8, w_scales
    else:
        # Legacy path: ensure blockscaled protocol for weight quantization
        bs_proto = _blockscaled_protocol(protocol or FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32))
        weight_fp8, weight_scales = _quantize_w2_cached(w, bs_proto)

    # --- Activation quantization ---
    if pre_quantized_act:
        a_fp8_data, packed_a_scales = a, a_scales
    else:
        a_fp8_data, packed_a_scales = quantize_and_pack_activation(
            a.contiguous(), group_size=_SF_VEC_SIZE
        )

    # --- Run CUTLASS GEMM kernel ---
    d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    return _run_cutlass_blockscaled_gemm(
        a_fp8_data, packed_a_scales,
        weight_fp8, weight_scales,
        cu_seqlens_m,
        total_M, K, H, num_experts,
        d_dtype, device,
    )


def _run_cutlass_blockscaled_gemm(
    a_fp8: torch.Tensor,
    a_scales_packed: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales_packed: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    total_M: int,
    K: int,
    H: int,
    num_experts: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Pure CUTLASS blockscaled FP8 GEMM — all inputs pre-quantized.

    Uses a shape-pinned fast path to skip validation, GemmTensorInfo
    construction, major-order determination, compile-key hashing, and
    _COMPILE_CACHE lookup on steady-state calls.  Only cute-tensor
    creation (unavoidable — different data pointers) and the kernel
    launch remain on the hot path.
    """

    # --- Fast path: steady-state calls with identical problem shapes ---
    fast_key = (
        total_M, K, H, num_experts, out_dtype,
        w_fp8.shape[0], w_fp8.shape[1], w_fp8.shape[2],
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        a_scales_packed.size(1),
        w_scales_packed.shape[0], w_scales_packed.shape[1],
        device.index if device.index is not None else -1,
    )
    cached = _GEMM_FAST_PATH.get(fast_key)
    if cached is not None:
        compiled, scheduler_args, epi_args = cached
        out = torch.empty(total_M, H, dtype=out_dtype, device=device)
        # B permute for varlen_m: (H,K,E) -> (K,E,H)
        w_permuted = w_fp8.permute(1, 2, 0)
        # Cute tensors — all row-major (leading_dim=1) in production
        a_cute = _make_cute_tensor_dynamic(a_fp8, leading_dim=1)
        b_cute = _make_cute_tensor_dynamic(w_permuted, leading_dim=1)
        d_cute = _make_cute_tensor_dynamic(out, leading_dim=1)
        a_sc_cute = _make_cute_tensor_dynamic(a_scales_packed, leading_dim=1)
        b_sc_cute = _make_cute_tensor_dynamic(w_scales_packed, leading_dim=1)
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m, cu_seqlens_k=None, A_idx=None,
        )
        stream = cuda.CUstream(_get_raw_cuda_stream())
        compiled(
            a_cute, b_cute, d_cute, None,
            epi_args, scheduler_args, varlen_args, stream,
            a_sc_cute, b_sc_cute,
        )
        return out

    # --- Slow path (first call for this shape config) ---
    out = torch.empty(total_M, H, dtype=out_dtype, device=device)

    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_gemm_varlen does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(w_fp8),
        "D": GemmTensorInfo(out),
        "C": GemmTensorInfo(None),
    }

    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=True)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)

    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )

    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        cu_seqlens_k=None,
        A_idx=None,
    )

    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(a_scales_packed, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(w_scales_packed, leading_dim=1)

    # compile_key must NOT contain dynamic token dimensions (total_M,
    # a_scales_packed.size(1)) — those change with seqlen and are handled
    # at runtime via mark_layout_dynamic.  Including them would trigger a
    # CuTe recompile every time seqlen changes.
    compile_key = (
        "varlen",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        K,
        tuple(tensor_infos["B"].tensor.shape),
        tuple(tensor_infos["B"].tensor.stride()),
        H,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        config.pingpong,
        True,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE,
            gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
        _COMPILE_CACHE[compile_key] = compiled

    # Populate fast-path cache for subsequent calls
    if len(_GEMM_FAST_PATH) > _MAX_FAST_PATH_ENTRIES:
        _GEMM_FAST_PATH.clear()
    _GEMM_FAST_PATH[fast_key] = (compiled, scheduler_args, epi_args)

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )
    return out


def _auto_capacity(cu_seqlens_m: torch.Tensor) -> int:
    """Compute minimum expert capacity aligned to tile boundary from cu_seqlens_m."""
    max_tokens = int((cu_seqlens_m[1:] - cu_seqlens_m[:-1]).max().item())
    return int(math.ceil(max_tokens / _SF_TILE_M) * _SF_TILE_M)


def blockscaled_fp8_gemm(
    a: torch.Tensor,
    w2: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    protocol: FP8Protocol,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    capacity = _auto_capacity(cu_seqlens_m)
    grouped_out = blockscaled_fp8_gemm_grouped(
        a,
        w2,
        cu_seqlens_m,
        protocol=protocol,
        out_dtype=(a.dtype if out is None else out.dtype),
        capacity=capacity,
    )
    return _unpack_grouped_rows(grouped_out, cu_seqlens_m, out=out)


def blockscaled_fp8_weight_grad_gemm(
    a_flat: torch.Tensor,
    b_flat: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Blockscaled FP8 weight-gradient GEMM: D[e] = A_e^T @ B_e per expert.

    Both *a_flat* and *b_flat* are flat 2-D activations in expert-sorted
    token order (``TK, dim``).  ``cu_seqlens_m`` (int32, ``E+1``) marks
    expert boundaries.

    The function:

    1. Packs both operands into ``(E, capacity, dim)`` groups.
    2. Transposes to ``(E, dim, capacity)`` so that K=capacity is the
       last dimension — required for correct 1x32 blockscaled quantisation.
    3. Quantises to blockscaled FP8 (E8M0 scales).
    4. Runs a batched CUTLASS GEMM producing ``D(E, dim_A, dim_B)``.

    Parameters
    ----------
    a_flat : Tensor (TK, dim_A) bf16 — first operand (will be transposed).
    b_flat : Tensor (TK, dim_B) bf16 — second operand.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    out : optional pre-allocated Tensor (E, dim_A, dim_B).
    out_dtype : output element type (default bf16).

    Returns
    -------
    D : Tensor (E, dim_A, dim_B).
    """
    device = a_flat.device
    if get_device_capacity(device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm requires Blackwell (sm_100)")
    if a_flat.ndim != 2 or b_flat.ndim != 2:
        raise ValueError(
            f"expected 2D operands, got a={tuple(a_flat.shape)}, b={tuple(b_flat.shape)}"
        )
    if a_flat.size(0) != b_flat.size(0):
        raise ValueError("a_flat and b_flat must have the same number of rows (TK)")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise ValueError("cu_seqlens_m must be 1D int32")

    num_experts = cu_seqlens_m.shape[0] - 1
    dim_A = a_flat.size(1)
    dim_B = b_flat.size(1)
    capacity = _auto_capacity(cu_seqlens_m)

    if dim_A % _SF_TILE_M != 0:
        raise RuntimeError(
            f"dim_A ({dim_A}) must be a multiple of {_SF_TILE_M} for blockscaled tiling"
        )
    if dim_B % _SF_TILE_M != 0:
        raise RuntimeError(
            f"dim_B ({dim_B}) must be a multiple of {_SF_TILE_M} for blockscaled tiling"
        )
    # capacity is already tile-aligned by _auto_capacity

    # 1. Pack flat tokens into per-expert groups: (E, cap, dim)
    a_grouped = _pack_grouped_rows(a_flat, cu_seqlens_m, num_experts, capacity)
    b_grouped = _pack_grouped_rows(b_flat, cu_seqlens_m, num_experts, capacity)

    # 2. Transpose so K=capacity is the last dim for correct blockscaling
    a_t = a_grouped.transpose(1, 2).contiguous()   # (E, dim_A, cap)
    b_t = b_grouped.transpose(1, 2).contiguous()   # (E, dim_B, cap)
    del a_grouped, b_grouped

    # 3. Quantise to blockscaled FP8 — reshape to 2-D, quantise, reshape back
    a_2d = a_t.reshape(-1, capacity)                # (E*dim_A, cap)
    a_fp8_2d, a_scales_2d = quantize_activation_blockscaled_fast(a_2d)
    a_fp8 = a_fp8_2d.reshape(num_experts, dim_A, capacity)
    a_scales_3d = a_scales_2d.reshape(num_experts, dim_A, -1)
    del a_t, a_2d, a_fp8_2d, a_scales_2d

    b_2d = b_t.reshape(-1, capacity)                # (E*dim_B, cap)
    b_fp8_2d, b_scales_2d = quantize_activation_blockscaled_fast(b_2d)
    b_fp8 = b_fp8_2d.reshape(num_experts, dim_B, capacity)
    b_scales_3d = b_scales_2d.reshape(num_experts, dim_B, -1)
    del b_t, b_2d, b_fp8_2d, b_scales_2d

    # 4. Pack scales into ISA tile layout
    packed_a_scales = pack_blockscaled_1x32_scales(a_scales_3d, capacity)
    packed_b_scales = pack_blockscaled_1x32_scales(b_scales_3d, capacity)
    del a_scales_3d, b_scales_3d

    # 5. Allocate output — CUTLASS needs a contiguous buffer
    d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    out_is_alias = out is not None and not out.is_contiguous()
    if out is not None and out.is_contiguous():
        grouped_out = out
    else:
        grouped_out = torch.empty(num_experts, dim_A, dim_B, dtype=d_dtype, device=device)

    # 6. Run CUTLASS grouped blockscaled GEMM
    #    A_cutlass (E, M=dim_A, K=cap)  @  B_cutlass (E, N=dim_B, K=cap)^T
    #    -> D (E, M=dim_A, N=dim_B)
    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(b_fp8),
        "D": GemmTensorInfo(grouped_out),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for weight-grad GEMM")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = None
    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(packed_a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(packed_b_scales, leading_dim=1)

    # compile_key must NOT contain dynamic dims (capacity changes with routing).
    # All tensors use mark_layout_dynamic — the compiled kernel handles any capacity.
    # Only include static model dims (dim_A, dim_B) and device config.
    compile_key = (
        "weight_grad",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        tensor_infos["A"].tensor.shape[1],  # dim_A — STATIC
        tensor_infos["B"].tensor.shape[1],  # dim_B — STATIC
        num_experts,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        config.pingpong,
        True,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE,
            gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
        _COMPILE_CACHE[compile_key] = compiled

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )
    if out_is_alias:
        out.copy_(grouped_out)
        return out
    return grouped_out


def blockscaled_fp8_weight_grad_gemm_fast(
    a_flat: torch.Tensor,
    b_flat: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    a_gather_idx: Optional[torch.Tensor] = None,
    b_gather_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    pre_quantized_a: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    pre_quantized_b: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Optimized blockscaled FP8 weight-gradient GEMM with fused data prep.

    Same semantics as ``blockscaled_fp8_weight_grad_gemm`` but:
      - For aligned routing, skips separate pack (zero-copy reshape)
      - Fuses transpose + quantize + optional gather into one Triton kernel
      - ~2-3x faster data preparation than the decomposed pipeline

    Parameters
    ----------
    a_flat : Tensor (TK, dim_A) bf16 — first operand (will be transposed).
    b_flat : Tensor (TK, dim_B) bf16 — second operand.
    cu_seqlens_m : Tensor (E+1,) int32 — expert token boundaries.
    a_gather_idx : optional (TK,) int32 — gather index for a_flat.
    b_gather_idx : optional (TK,) int32 — gather index for b_flat.
    out : optional pre-allocated Tensor (E, dim_A, dim_B).
    out_dtype : output element type (default bf16).
    pre_quantized_a : optional (fp8_3d, packed_scales) — skip A transpose+quant.
    pre_quantized_b : optional (fp8_3d, packed_scales) — skip B transpose+quant.
    """
    device = a_flat.device
    if get_device_capacity(device)[0] != 10:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm_fast requires Blackwell (sm_100)")
    if a_flat.ndim != 2 or b_flat.ndim != 2:
        raise ValueError(f"expected 2D operands, got a={tuple(a_flat.shape)}, b={tuple(b_flat.shape)}")

    # When gather indices are provided, effective row count comes from the
    # gather index length (TK), not from a_flat/b_flat which may be shorter (T).
    TK = int(cu_seqlens_m[-1].item())
    a_rows = a_gather_idx.size(0) if a_gather_idx is not None else a_flat.size(0)
    b_rows = b_gather_idx.size(0) if b_gather_idx is not None else b_flat.size(0)
    if a_rows != TK or b_rows != TK:
        raise ValueError(
            f"effective row count must match cu_seqlens_m[-1]={TK}, "
            f"got a={a_rows}, b={b_rows}"
        )

    num_experts = cu_seqlens_m.shape[0] - 1
    dim_A = a_flat.size(1)
    dim_B = b_flat.size(1)
    capacity = _auto_capacity(cu_seqlens_m)

    if dim_A % _SF_TILE_M != 0:
        raise RuntimeError(f"dim_A ({dim_A}) must be a multiple of {_SF_TILE_M}")
    if dim_B % _SF_TILE_M != 0:
        raise RuntimeError(f"dim_B ({dim_B}) must be a multiple of {_SF_TILE_M}")

    # Fused transpose + quantize (+ optional gather) — skip if pre-quantized
    if pre_quantized_a is not None:
        a_fp8, packed_a_scales = pre_quantized_a
    else:
        a_fp8, packed_a_scales = fused_transpose_quantize_for_wgrad(
            a_flat, num_experts, capacity, dim_A, gather_idx=a_gather_idx,
        )
    if pre_quantized_b is not None:
        b_fp8, packed_b_scales = pre_quantized_b
    else:
        b_fp8, packed_b_scales = fused_transpose_quantize_for_wgrad(
            b_flat, num_experts, capacity, dim_B, gather_idx=b_gather_idx,
        )

    # Allocate output
    d_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    out_is_alias = out is not None and not out.is_contiguous()
    if out is not None and out.is_contiguous():
        grouped_out = out
    else:
        grouped_out = torch.empty(num_experts, dim_A, dim_B, dtype=d_dtype, device=device)

    # CUTLASS grouped blockscaled GEMM
    config = default_config(device)
    if config.swap_ab:
        raise RuntimeError("blockscaled_fp8_weight_grad_gemm_fast does not support swap_ab")

    tensor_infos = {
        "A": GemmTensorInfo(a_fp8),
        "B": GemmTensorInfo(b_fp8),
        "D": GemmTensorInfo(grouped_out),
        "C": GemmTensorInfo(None),
    }
    GemmWrapperBase.permute_tensors(tensor_infos)

    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            if name != "C":
                info.cute_tensor = _make_cute_tensor_dynamic(
                    info.tensor,
                    leading_dim=1 if info.major == major_configs[name][1] else 0,
                )

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported FP8 blockscaled type/major combination for weight-grad GEMM")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = None
    epi_args = GemmDefaultSm100.EpilogueArguments()
    current_stream = cuda.CUstream(_get_raw_cuda_stream())
    a_scale_cute = _make_cute_tensor_dynamic(packed_a_scales, leading_dim=1)
    b_scale_cute = _make_cute_tensor_dynamic(packed_b_scales, leading_dim=1)

    # compile_key must NOT contain dynamic dims — same pattern as "weight_grad".
    compile_key = (
        "weight_grad_fast",
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        tensor_infos["A"].tensor.shape[1],  # dim_A — STATIC
        tensor_infos["B"].tensor.shape[1],  # dim_B — STATIC
        num_experts,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        config.pingpong,
        True,
        _SF_VEC_SIZE,
    )
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            sf_vec_size=_SF_VEC_SIZE,
            gather_A=False,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
        _COMPILE_CACHE[compile_key] = compiled

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )
    if out_is_alias:
        out.copy_(grouped_out)
        return out
    return grouped_out
