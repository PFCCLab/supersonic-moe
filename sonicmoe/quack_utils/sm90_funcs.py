# ********************************************************************************
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ********************************************************************************

"""SM90 (Hopper) FP8 backend of ``sonicmoe.functional.fp8_kernels``.

Native WGMMA FP8 kernels (``gemm_sm90_scaled.py``) with E4M3 activations,
raw uint8 UE8M0 ``[rows, K/G]`` activation scales and GxG weight blocks
``[E, N/B, K/B]`` (``G == B`` in {32, 128}). Also hosts the SM90 UE8M0
activation / weight quantization helpers.

Every op exported here has the same name, signature and return arity as its
counterpart in ``sm100_funcs``; ops without an SM90 implementation raise
``NotImplementedError``.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..enums import ActivationType
from ..triton_utils import wrap_triton_kernel
from .activation_situ import is_situ_activation
from .blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
# Imported eagerly: sonicmoe is loaded under paddle's torch-compat guard, and a
# lazy import at call time would bind these modules to a different ``torch``.
from .gemm_sm90_scaled import (
    sm90_scaled_dgated,
    sm90_scaled_gated,
    sm90_scaled_gemm,
    sm90_scaled_recompute_z,
    sm90_scaled_wgrad_varlen_k,
)
from .swiglu_triton import _dequant_blockscaled_fp8_kernel, dequantize_blockscaled_fp8


# ---------------------------------------------------------------------------
# UE8M0 quantization helpers
# ---------------------------------------------------------------------------

_SM90_QDQ_GROUP_SIZES = (32, 128)


def is_sm90_device(device: torch.device | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9


def validate_sm90_qdq_profile(
    activation_group_size: int, weight_block_size: int
) -> tuple[int, int]:
    profile = (int(activation_group_size), int(weight_block_size))
    if profile not in ((32, 32), (128, 128)):
        raise ValueError(
            "SM90 FP8 QDQ profiles are fixed to activation 1x32 + weight "
            "32x32 or activation 1x128 + weight 128x128; got "
            f"activation 1x{profile[0]} + weight {profile[1]}x{profile[1]}"
        )
    return profile


def _decode_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    raw = scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales
    exponent = raw.to(torch.float32) - 127.0
    base = torch.full_like(exponent, 2.0)
    return torch.pow(base, exponent)


def _raw_activation_scales(
    scales: torch.Tensor, rows: int, groups: int
) -> torch.Tensor:
    if scales.dtype is torch.uint8 or scales.dtype == torch.uint8:
        raw = scales
    else:
        raw = scales.contiguous().view(torch.uint8)
    shape = tuple(raw.shape)
    if shape == (rows, groups):
        # Already the [rows, K/G] carrier: skip the reshape/slice views (~3 us
        # each under paddle compat).
        return raw
    raw_numel = math.prod(shape)  # host-side; numel() syncs under paddle compat
    if raw_numel < rows * groups:
        raise ValueError(
            "Invalid SM90 UE8M0 activation scale carrier: "
            f"need {rows * groups} bytes, got {raw_numel}"
        )
    return raw.reshape(rows, -1)[:, :groups]


def materialize_activation_sm90(
    data: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Materialize an SM90 rowwise payload as contiguous BF16."""
    if group_size not in _SM90_QDQ_GROUP_SIZES:
        raise ValueError(f"SM90 activation group size must be 32 or 128, got {group_size}")
    if data.ndim != 2 or data.shape[1] % group_size:
        raise ValueError(
            "SM90 activation materialization expects a rank-2 tensor whose hidden "
            f"dimension is divisible by {group_size}, got {tuple(data.shape)}"
        )
    rows, cols = (int(data.shape[0]), int(data.shape[1]))
    raw_scales = _raw_activation_scales(scales, rows, cols // group_size)
    data = data.contiguous()
    # Fused Triton dequant: fp8 * 2^(e8m0-127) -> bf16 without materializing
    # the two activation-sized fp32 temporaries of the eager formulation.
    out = torch.empty(rows, cols, dtype=torch.bfloat16, device=_alloc_device(data))
    if rows == 0:
        return out
    block_rows = 16
    _dequant_blockscaled_fp8_kernel[(triton.cdiv(rows, block_rows),)](
        data, raw_scales, out,
        rows,
        data.stride(0), raw_scales.stride(0), out.stride(0),
        D=cols, GROUP_SIZE=group_size, BLOCK_ROWS=block_rows,
    )
    return out


@wrap_triton_kernel
@triton.jit
def _sm90_dequant_rows_kernel(
    src_ptr, scale_ptr, idx_ptr, out_ptr,
    rows, src_stride, scale_stride, out_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """out[r] = dequant(src[idx[r]]): same fp8 * 2^(e-127) -> bf16 math as
    ``_dequant_blockscaled_fp8_kernel``, rows gathered on the fly."""
    rows_out = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows_out < rows
    rows_src = tl.load(idx_ptr + rows_out, mask=row_mask, other=0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    groups = tl.program_id(1) * (BLOCK_COLS // GROUP_SIZE) + tl.arange(0, BLOCK_COLS // GROUP_SIZE)
    q = tl.load(src_ptr + rows_src[:, None] * src_stride + cols[None, :], mask=row_mask[:, None], other=0.0)
    e = tl.load(scale_ptr + rows_src[:, None] * scale_stride + groups[None, :], mask=row_mask[:, None], other=0)
    scale = (e.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    v = tl.reshape(q.to(tl.float32), (BLOCK_ROWS, BLOCK_COLS // GROUP_SIZE, GROUP_SIZE)) * scale[:, :, None]
    tl.store(
        out_ptr + rows_out[:, None].to(tl.int64) * out_stride + cols[None, :],
        tl.reshape(v, (BLOCK_ROWS, BLOCK_COLS)).to(tl.bfloat16),
        mask=row_mask[:, None],
    )


def _wgrad_route_rows(k: int) -> bool:
    """Whether the BF16 up-proj wgrad should read a route-ordered bf16 x.

    The SM90 varlen-K wgrad then feeds A by TMA instead of the cp.async row
    gather: at H = 7168 (ERNIE-5.1) dw1 5006 -> 4325 us, bitwise. The gather
    is fused into the dequant that has to run anyway, which only writes
    (TK - T) more rows. At H <= 4096 the GEMM gain (-5..+5%) does not pay for
    the extra rows (Round 16).
    """
    return k >= 6144


def dequantize_activation_routed(
    data: torch.Tensor, scales: torch.Tensor, cfg, gather_idx: torch.Tensor
) -> torch.Tensor | None:
    """BF16 ``dequant(data)[gather_idx]`` for the up-proj wgrad, or ``None``
    when the gathered wgrad is faster (see ``_wgrad_route_rows``)."""
    group = cfg.activation_group_size
    if data.ndim != 2:
        return None
    cols = int(data.shape[1])
    if not _wgrad_route_rows(cols) or cols % 128:
        return None
    raw_scales = _raw_activation_scales(scales, int(data.shape[0]), cols // group)
    data = data.contiguous()
    rows = int(gather_idx.shape[0])
    out = torch.empty(rows, cols, dtype=torch.bfloat16, device=_alloc_device(data))
    if rows == 0:
        return out
    block_cols = next(c for c in (1024, 512, 256, 128) if cols % c == 0)
    block_rows = 4096 // block_cols
    _sm90_dequant_rows_kernel[(triton.cdiv(rows, block_rows), cols // block_cols)](
        data, raw_scales, gather_idx, out,
        rows, data.stride(0), raw_scales.stride(0), out.stride(0),
        BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols, GROUP_SIZE=group,
    )
    return out


@wrap_triton_kernel
@triton.jit
def _sm90_wgrad_colwise_quant_kernel(
    src_ptr,
    src_scales_ptr,
    gather_idx_ptr,
    offsets_ptr,
    packed_offsets_ptr,
    dst_ptr,
    scales_ptr,
    feature,
    total_k,
    packed_total_k,
    src_stride_row,
    src_stride_col,
    num_experts: tl.constexpr,
    has_gather: tl.constexpr,
    source_is_fp8: tl.constexpr,
    source_group: tl.constexpr,
    block_feature: tl.constexpr,
):
    expert = tl.program_id(0)
    feature_block = tl.program_id(1)
    group = tl.program_id(2)
    start = tl.load(offsets_ptr + expert)
    end = tl.load(offsets_ptr + expert + 1)
    packed_start = tl.load(packed_offsets_ptr + expert)
    segment_groups = (end - start + 31) // 32
    lane = tl.arange(0, 32)
    source_k = start + group * 32 + lane
    packed_k = packed_start + group * 32 + lane
    f = feature_block * block_feature + tl.arange(0, block_feature)
    valid_group = group < segment_groups
    valid_source = valid_group & (source_k < end)
    mask = valid_source[:, None] & (f[None, :] < feature)
    src_rows = tl.load(
        gather_idx_ptr + source_k,
        mask=valid_source,
        other=0,
    ) if has_gather else source_k
    values = tl.load(
        src_ptr + src_rows[:, None].to(tl.int64) * src_stride_row
        + f[None, :].to(tl.int64) * src_stride_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if source_is_fp8:
        source_groups = (feature + source_group - 1) // source_group
        source_scale = tl.load(
            src_scales_ptr
            + src_rows[:, None].to(tl.int64) * source_groups
            + (f[None, :] // source_group).to(tl.int64),
            mask=mask,
            other=127,
        )
        values *= tl.exp2(source_scale.to(tl.float32) - 127.0)
    amax = tl.max(tl.abs(values), axis=0)
    bits = amax.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    carry = (bits & 0x7FFFFF) > 0x600000
    raw = tl.maximum(biased_exp + carry.to(tl.int32) - 8, 0)
    qexp = tl.maximum(tl.minimum(254 - raw, 254), 1)
    qscale = (qexp << 23).to(tl.float32, bitcast=True)
    quantized = (values * qscale[None, :]).to(tl.float8e4nv)
    packed_mask = (
        valid_group
        & (packed_k[:, None] < packed_start + segment_groups * 32)
        & (f[None, :] < feature)
    )
    tl.store(
        dst_ptr
        + f[:, None].to(tl.int64) * packed_total_k
        + packed_k[None, :].to(tl.int64),
        tl.trans(quantized),
        mask=tl.trans(packed_mask),
    )
    scale_col = packed_start // 32 + group
    tl.store(
        scales_ptr + f.to(tl.int64) * (packed_total_k // 32) + scale_col,
        raw.to(tl.uint8),
        mask=valid_group & (f < feature),
    )


def _validated_colwise_offsets(
    expert_offsets: torch.Tensor, total_k: int
) -> list[int]:
    """Validate routed segments and return host offsets for packed construction."""
    if expert_offsets.numel() < 2:
        raise ValueError("expert_offsets must contain at least [0, total_K]")
    offsets = [int(value) for value in expert_offsets.detach().cpu().tolist()]
    if offsets[0] != 0:
        raise ValueError(f"expert_offsets must start at 0, got {offsets[0]}")
    if offsets[-1] != total_k:
        raise ValueError(
            f"expert_offsets must end at total_K={total_k}, got {offsets[-1]}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("expert_offsets must be monotonically nondecreasing")
    return offsets


def _packed_colwise_offsets(offsets: list[int]) -> list[int]:
    packed = [0]
    for left, right in zip(offsets, offsets[1:]):
        length = right - left
        packed.append(packed[-1] + ((length + 31) // 32) * 32)
    return packed


def quantize_wgrad_colwise_sm90(
    source: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    source_scales: torch.Tensor | None = None,
    source_group: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Hopper wgrad K-major E4M3 ``[feature,total_K]`` and raw scales.

    Scale ABI is contiguous uint8 ``[feature,packed_total_K/32]``. Each expert
    receives ``ceil(segment_length/32)`` private groups and a zero-filled tail,
    so no scale group crosses an expert boundary.
    """
    source_dtype = str(source.dtype)
    source_is_fp8 = source_dtype in (
        "torch.float8_e4m3fn", "paddle.float8_e4m3fn", "float8_e4m3fn"
    )
    if source.ndim != 2 or source_dtype not in (
        "torch.bfloat16", "paddle.bfloat16", "bfloat16",
        "torch.float8_e4m3fn", "paddle.float8_e4m3fn", "float8_e4m3fn",
    ) or not source.is_cuda:
        raise TypeError("SM90 wgrad colwise source must be rank-2 CUDA BF16/E4M3")
    if source_is_fp8:
        if source_scales is None:
            raise ValueError("E4M3 colwise source requires raw UE8M0 source_scales")
        expected = (int(source.shape[0]), (int(source.shape[1]) + source_group - 1) // source_group)
        if tuple(source_scales.shape) != expected:
            raise ValueError(f"E4M3 source scales must have shape {expected}")
    elif source_scales is not None:
        raise ValueError("source_scales are valid only for an E4M3 source")
    offsets_dtype = str(expert_offsets.dtype)
    if expert_offsets.ndim != 1 or offsets_dtype not in (
        "torch.int32", "paddle.int32", "int32"
    ):
        raise TypeError("expert_offsets must be rank-1 int32")
    total_k = int(gather_idx.numel()) if gather_idx is not None else int(source.shape[0])
    if gather_idx is not None and (
        gather_idx.ndim != 1
        or str(gather_idx.dtype) not in ("torch.int32", "paddle.int32", "int32")
    ):
        raise TypeError("gather_idx must be rank-1 int32")
    if expert_offsets.device != source.device:
        raise ValueError("expert_offsets must be on the same device as source")
    if gather_idx is not None and gather_idx.device != source.device:
        raise ValueError("gather_idx must be on the same device as source")
    offsets = _validated_colwise_offsets(expert_offsets, total_k)
    packed_offsets = _packed_colwise_offsets(offsets)
    packed_total_k = packed_offsets[-1]
    packed_offsets_tensor = torch.tensor(
        packed_offsets, dtype=torch.int32, device=expert_offsets.device
    )
    if gather_idx is not None and gather_idx.numel():
        gather_min = int(gather_idx.min().item())
        gather_max = int(gather_idx.max().item())
        if gather_min < 0 or gather_max >= int(source.shape[0]):
            raise ValueError(
                "gather_idx values must be in source row range "
                f"[0, {int(source.shape[0])}), got [{gather_min}, {gather_max}]"
            )
    feature = int(source.shape[1])
    output_place = getattr(source, "place", None)
    output_device = output_place if output_place is not None else str(source.device)
    data = torch.zeros(
        (feature, packed_total_k), dtype=torch.float8_e4m3fn, device=output_device
    )
    scales = torch.zeros(
        (feature, packed_total_k // 32), dtype=torch.uint8, device=output_device
    )
    num_experts = int(expert_offsets.numel()) - 1
    max_groups = max(
        (((right - left) + 31) // 32 for left, right in zip(offsets, offsets[1:])),
        default=0,
    )
    if packed_total_k == 0:
        return data, scales, packed_offsets_tensor
    dummy_gather = gather_idx if gather_idx is not None else expert_offsets
    dummy_scales = source_scales if source_scales is not None else expert_offsets
    _sm90_wgrad_colwise_quant_kernel[(num_experts, triton.cdiv(feature, 64), max_groups)](
        source, dummy_scales, dummy_gather, expert_offsets, packed_offsets_tensor,
        data, scales, feature, total_k, packed_total_k,
        source.stride(0), source.stride(1), num_experts=num_experts,
        has_gather=gather_idx is not None, source_is_fp8=source_is_fp8,
        source_group=source_group, block_feature=64, num_warps=4,
    )
    return data, scales, packed_offsets_tensor


@wrap_triton_kernel
@triton.jit
def _sm90_wgrad_colwise_quant128_kernel(
    src_ptr,
    src_scales_ptr,
    gather_idx_ptr,
    offsets_ptr,
    dst_ptr,
    scales_ptr,
    feature,
    total_k,
    src_stride_row,
    src_stride_col,
    src_scale_stride,
    num_experts: tl.constexpr,
    has_gather: tl.constexpr,
    source_is_fp8: tl.constexpr,
    source_group: tl.constexpr,
    block_feature: tl.constexpr,
):
    # One program = 128 K rows x block_feature features. Expert segments are
    # already 128-aligned, so a group never crosses an expert boundary. Rows at
    # or beyond offsets[E] (sync-free upper-bound sizing) are written as zero.
    feature_block = tl.program_id(0)
    group = tl.program_id(1)
    limit = tl.load(offsets_ptr + num_experts)
    k = group * 128 + tl.arange(0, 128)
    valid_k = k < limit
    f = feature_block * block_feature + tl.arange(0, block_feature)
    f_ok = f < feature
    mask = valid_k[:, None] & f_ok[None, :]
    src_rows = tl.load(gather_idx_ptr + k, mask=valid_k, other=0) if has_gather else k
    src_rows = src_rows.to(tl.int64)
    values = tl.load(
        src_ptr + src_rows[:, None] * src_stride_row
        + f[None, :].to(tl.int64) * src_stride_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if source_is_fp8:
        if source_group % block_feature == 0:
            # One source scale per gathered row for the whole feature block.
            row_scale = tl.load(
                src_scales_ptr + src_rows * src_scale_stride
                + (feature_block * block_feature) // source_group,
                mask=valid_k,
                other=127,
            )
            values *= tl.exp2(row_scale.to(tl.float32) - 127.0)[:, None]
        else:
            source_scale = tl.load(
                src_scales_ptr + src_rows[:, None] * src_scale_stride
                + (f[None, :] // source_group).to(tl.int64),
                mask=mask,
                other=127,
            )
            values *= tl.exp2(source_scale.to(tl.float32) - 127.0)
    amax = tl.max(tl.abs(values), axis=0)
    bits = amax.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    carry = (bits & 0x7FFFFF) > 0x600000
    raw = tl.maximum(biased_exp + carry.to(tl.int32) - 8, 0)
    qexp = tl.maximum(tl.minimum(254 - raw, 254), 1)
    qscale = (qexp << 23).to(tl.float32, bitcast=True)
    quantized = (values * qscale[None, :]).to(tl.float8e4nv)
    tl.store(
        dst_ptr + f[:, None].to(tl.int64) * total_k + k[None, :].to(tl.int64),
        tl.trans(quantized),
        mask=f_ok[:, None],
    )
    tl.store(
        scales_ptr + group.to(tl.int64) * feature + f,
        raw.to(tl.uint8),
        mask=f_ok,
    )


_SM90_WGRAD_QUANT128_BF = int(os.environ.get("SONIC_MOE_SM90_WGRAD_QUANT128_BF", "64"))
_SM90_WGRAD_QUANT128_WARPS = int(os.environ.get("SONIC_MOE_SM90_WGRAD_QUANT128_WARPS", "0"))  # 0 = auto


def _wgrad128_scale_rows(groups: int, feature: int) -> int:
    # One row for the GEMM's one-tile-ahead scale prefetch plus >= 256 bytes
    # for its unmasked phantom-row/column scale reads stay in bounds.
    return groups + 1 + (256 + feature - 1) // feature


def quantize_wgrad_colwise128_sm90(
    source: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    source_scales: torch.Tensor | None = None,
    source_group: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hopper 1x128 wgrad operand without host syncs.

    Returns ``(fp8 [feature,total_K], u8 [total_K/128 + pad, feature],
    expert_offsets)``. ``expert_offsets`` (device int32 ``[E+1]``) must be
    128-aligned (routing_block_size=128) and is used as-is as the packed
    offsets; rows ``>= offsets[E]`` are zero-filled.
    """
    total_k = int(gather_idx.shape[0]) if gather_idx is not None else int(source.shape[0])
    feature = int(source.shape[1])
    if total_k % 128 or feature % 8:
        raise ValueError(
            f"1x128 wgrad needs total_K%128==0 and feature%8==0, got {total_k}, {feature}"
        )
    source_is_fp8 = source_scales is not None
    device = _alloc_device(source)
    groups = total_k // 128
    data = torch.empty((feature, total_k), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(
        (_wgrad128_scale_rows(groups, feature), feature), dtype=torch.uint8, device=device
    )
    if groups == 0:
        return data, scales, expert_offsets
    dummy_gather = gather_idx if gather_idx is not None else expert_offsets
    dummy_scales = source_scales if source_is_fp8 else expert_offsets
    bf = _SM90_WGRAD_QUANT128_BF
    warps = _SM90_WGRAD_QUANT128_WARPS or (2 if source_is_fp8 else 4)
    _sm90_wgrad_colwise_quant128_kernel[(triton.cdiv(feature, bf), groups)](
        source, dummy_scales, dummy_gather, expert_offsets, data, scales,
        feature, total_k, source.stride(0), source.stride(1),
        source_scales.stride(0) if source_is_fp8 else 0,
        num_experts=int(expert_offsets.shape[0]) - 1,
        has_gather=gather_idx is not None, source_is_fp8=source_is_fp8,
        source_group=source_group, block_feature=bf, num_warps=warps,
    )
    return data, scales, expert_offsets


def _encode_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    exponent = torch.ceil(torch.log(safe_amax / 448.0) / 0.6931471805599453)
    exponent = torch.minimum(
        torch.maximum(exponent, torch.full_like(exponent, -126.0)),
        torch.full_like(exponent, 127.0),
    )
    encoded = (exponent.to(torch.int32) + 127).to(torch.uint8)
    return torch.where(amax > 0, encoded, torch.zeros_like(encoded))


@wrap_triton_kernel
@triton.jit
def _sm90_weight_quant_kernel(
    w_ptr,
    q_ptr,
    s_ptr,
    cols,
    BS: tl.constexpr,
    NB: tl.constexpr,
):
    """One program quantizes a [BS, NB*BS] strip of the flattened [E*R, C] weight."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BS + tl.arange(0, BS)
    cidx = pid_c * (NB * BS) + tl.arange(0, NB * BS)
    offs = rows[:, None].to(tl.int64) * cols + cidx[None, :]
    x = tl.reshape(tl.load(w_ptr + offs).to(tl.float32), (BS, NB, BS))
    amax = tl.max(tl.max(tl.abs(x), axis=2), axis=0)
    # Same fp32 op chain as _encode_scale_from_amax (x * (1/448), logf, * (1/ln2),
    # ceil): bitwise-identical scales for every bf16 amax.  The exact
    # bit-trick ceil(log2(amax/448)) differs at 23 amax == 1.75*2^k values.
    t = tl.where(amax > 0, amax, 1.0) * (1.0 / 448.0)
    ex = tl.ceil(libdevice.log(t) * (1.0 / 0.6931471805599453))
    enc = tl.minimum(tl.maximum(ex, -126.0), 127.0).to(tl.int32) + 127
    enc = tl.where(amax > 0, enc, 0)
    # x / 2^(enc-127) == x * 2^(127-enc) bit for bit: both are one correctly
    # rounded op on the same real value, and 2^(127-enc) is an exact normal fp32
    # (biased exponent 254-enc; a finite bf16 amax gives enc <= 247). Replaces
    # the multi-instruction div.rn.f32 (611 -> 525 us at 10x7168x7168).
    inv = tl.where(amax > 0, ((254 - enc) << 23).to(tl.float32, bitcast=True), 1.0)
    q = x * inv[None, :, None]
    tl.store(q_ptr + offs, tl.reshape(q, (BS, NB * BS)).to(tl.float8e4nv))
    scols = cols // BS
    tl.store(
        s_ptr + pid_r.to(tl.int64) * scols + pid_c * NB + tl.arange(0, NB),
        enc.to(tl.uint8),
    )


def quantize_weight_sm90(
    weight: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size not in _SM90_QDQ_GROUP_SIZES:
        raise ValueError(f"SM90 weight block size must be 32 or 128, got {block_size}")
    if weight.ndim != 3 or weight.dtype != torch.bfloat16:
        raise TypeError("SM90 weight quantization expects a rank-3 bfloat16 tensor")
    experts, rows, cols = weight.shape
    if rows % block_size or cols % block_size:
        raise ValueError(
            f"SM90 {block_size}x{block_size} weight quantization requires divisible "
            f"dimensions, got {tuple(weight.shape)}"
        )
    weight = weight.contiguous()
    device = _alloc_device(weight)
    data = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(
        (experts, rows // block_size, cols // block_size),
        dtype=torch.uint8,
        device=device,
    )
    if experts * rows * cols == 0:
        return data, scales
    if block_size == 32:
        nb = next(n for n in (8, 4, 2, 1) if (cols // 32) % n == 0)
    else:
        # Two 128x128 blocks per program: more bytes in flight per CTA
        # (w1 544 -> 525 us; 4 blocks is slower).
        nb = 2 if (cols // 128) % 2 == 0 else 1
    _sm90_weight_quant_kernel[(experts * rows // block_size, cols // (block_size * nb))](
        weight, data, scales, cols,
        BS=block_size, NB=nb,
        num_warps=8 if block_size == 128 or nb >= 4 else 4,
    )
    return data, scales


def _quantize_weight_sm90_torch(
    weight: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused reference of ``quantize_weight_sm90`` (kept for numerics checks)."""
    experts, rows, cols = weight.shape
    blocks = weight.contiguous().float().reshape(
        experts, rows // block_size, block_size, cols // block_size, block_size
    )
    amax = blocks.abs().permute(0, 1, 3, 2, 4).reshape(
        experts, rows // block_size, cols // block_size, block_size * block_size
    ).amax(dim=-1)
    scales = _encode_scale_from_amax(amax)
    decoded = _decode_ue8m0(scales).reshape(
        experts, rows // block_size, 1, cols // block_size, 1
    )
    divisor = torch.where(
        amax.reshape(experts, rows // block_size, 1, cols // block_size, 1) > 0,
        decoded,
        torch.ones_like(decoded),
    )
    return (blocks / divisor).reshape(weight.shape).to(torch.float8_e4m3fn), scales


def dequantize_weight_sm90(
    data: torch.Tensor, scales: torch.Tensor, block_size: int
) -> torch.Tensor:
    if block_size not in _SM90_QDQ_GROUP_SIZES:
        raise ValueError(f"SM90 weight block size must be 32 or 128, got {block_size}")
    experts, rows, cols = data.shape
    if rows % block_size or cols % block_size:
        raise ValueError(
            f"SM90 {block_size}x{block_size} weight dequantization requires divisible "
            f"dimensions, got {tuple(data.shape)}"
        )
    values = data.float().reshape(
        experts, rows // block_size, block_size, cols // block_size, block_size
    )
    decoded = _decode_ue8m0(scales).reshape(
        experts, rows // block_size, 1, cols // block_size, 1
    )
    return (values * decoded).reshape(experts, rows, cols).to(torch.bfloat16)



def quantize_sm90_weights(
    w1: torch.Tensor, w2: torch.Tensor, block_size: int
) -> dict[str, object]:
    return {
        "format": f"{block_size}x{block_size}",
        "w1": quantize_weight_sm90(w1, block_size),
        "w2": quantize_weight_sm90(w2, block_size),
    }


# ---------------------------------------------------------------------------
# Backend-private helpers
# ---------------------------------------------------------------------------


def _groups(cfg) -> tuple[int, int]:
    return validate_sm90_qdq_profile(cfg.activation_group_size, cfg.weight_block_size)


def _raw_scales_u8(scales: torch.Tensor, data: torch.Tensor, group: int) -> torch.Tensor:
    rows, cols = int(data.shape[0]), int(data.shape[1])
    raw = _raw_activation_scales(scales, rows, cols // group)
    return raw if raw.is_contiguous() else raw.contiguous()


def _alloc_device(tensor):
    """Allocation device of ``tensor``. Under paddle compat ``.place`` is ~6x
    cheaper than ``.device``, and ``torch.empty(device=place)`` ~4 us cheaper
    than with a converted ``torch.device``."""
    place = getattr(tensor, "place", None)
    return place if place is not None else tensor.device


def _placeholder(shape, dtype, device) -> torch.Tensor:
    """Zero-stride tensor standing in for an output that is not materialized."""
    return torch.empty(1, dtype=dtype, device=device).as_strided(tuple(shape), (0,) * len(shape))


def _is_situ_activation_type(activation_type) -> bool:
    return activation_type is ActivationType.SITU_GLU or is_situ_activation(
        activation_type
    )


def _sm90_activation_operand(x, x_fp8_pre, x_scales_pre, group):
    """T-sized rowwise E4M3 activation with raw u8 ``[T, K/G]`` scales."""
    if x_fp8_pre is not None:
        return x_fp8_pre, _raw_scales_u8(x_scales_pre, x_fp8_pre, group)
    if x is None:
        raise ValueError("BF16 x is required when no prequant activation payload is provided")
    return quantize_activation_blockscaled_fast(x, group_size=group)


def _unsupported(op: str):
    raise NotImplementedError(f"{op} has no SM90 FP8 implementation (SM100 only)")


# ---------------------------------------------------------------------------
# Config / layout
# ---------------------------------------------------------------------------


def validate_config(cfg, device) -> None:
    # SM90 supports every FP8 option of the gated flow; only the
    # (activation group, weight block) profile is restricted.
    if cfg.enabled:
        _groups(cfg)


def segments_aligned(expert_frequency_offset: torch.Tensor) -> bool:
    # The varlen kernels consume raw per-row scales: any segment length works.
    return True


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


def quantize_activation(x: torch.Tensor, cfg):
    return quantize_activation_blockscaled_fast(x, group_size=cfg.activation_group_size)


def quantize_activation_raw(x: torch.Tensor, cfg):
    return quantize_activation_blockscaled_fast(x, group_size=cfg.activation_group_size)


def dequantize_activation(data: torch.Tensor, scales: torch.Tensor, cfg) -> torch.Tensor:
    return materialize_activation_sm90(data, scales, cfg.activation_group_size)


def quantize_preact(z: torch.Tensor):
    return quantize_activation_blockscaled_fast(z)


def dequantize_preact(z_fp8: torch.Tensor, z_scales_u8: torch.Tensor) -> torch.Tensor:
    return dequantize_blockscaled_fp8(z_fp8, z_scales_u8)


def quantize_z_y1(z: torch.Tensor, y1: torch.Tensor, cfg):
    z_fp8, z_scales = quantize_activation_blockscaled_fast(z)
    return (z_fp8, z_scales, *quantize_activation(y1, cfg))


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


def precompute_weight_dgated(w2: torch.Tensor):
    _unsupported("precompute_weight_dgated")


# ---------------------------------------------------------------------------
# GEMMs
# ---------------------------------------------------------------------------


# Expert GEMM CTA tile: 192x128 (three MMA warpgroups) beats the 128x128
# default by 3-15% on gated/dgated/down/dx at all bench shapes (bitwise; the
# per-K=32 promotion of one warpgroup overlaps the other two's WGMMA). The
# BF16-output recompute keeps 128x128 at profile 32, where 192 is slower.
_SM90_TILE = (192, 128)


def _sm90_tile(group: int, kind: str = "gather", k: int = 0):
    """CTA tile per GEMM kind (all choices bitwise: the K order is unchanged).

    Profile 128 (one scale per 128-K tile, tensor-core accumulation):
      * cp.async gather kernels (gated / recompute_z / dgated) prefer 128x192
        at K >= 2048: the cp.async A gather of 192 rows costs more than the
        wider N saves. At K <= 1024 (ERNIELite 4B) gated is ~20% faster at
        192x128. The TMA gather path (``_sm90_gather_kw``) uses 192x128.
        dgated stays 128x192 there too: its colvec partial width follows
        tile N, so another tile changes the dprobs summation order.
      * non-gather down / dx: 192x128 (3 MMA warpgroups, block-aligned N) is
        4-13% faster than 128x192 at every bench shape (Round 16).
    """
    if group != 128:
        return _SM90_TILE
    if kind == "gemm" or (kind == "gated" and 0 < k <= 1024):
        return (192, 128)
    return (128, 192)


def _sm90_gather_kw(group: int, kind: str, x_fp8: torch.Tensor, n: int):
    """Kernel kwargs for gated / recompute_z, which gather A rows in-kernel.

    Profile 128 (K tile 128 = one 128B swizzle row): in-kernel per-row TMA
    gather (``tma_gather=(1, 4)``): each A row is fetched by one 2D TMA
    (indices staged in smem), the rows of a tile split over the 4 load warps
    (one lane each, so the issue rate is not bound by a single lane) and over
    a 1x2 cluster with multicast. Tile 192x128 at every K: it enables the fused
    1x128 y1 quant in gated (6-14% faster than 128x192 + separate quant at
    K > 2048). vs the cp.async-only gather it is 8-25% faster on gated /
    recompute_z at every bench shape (Round 22), bitwise. No out-of-kernel row
    permute: that copy costs time and a transient M x K buffer in the
    end-to-end step.
    """
    k = int(x_fp8.shape[1])
    kw = dict(activation_scale_order="source", tile_shape=_sm90_tile(group, kind, k))
    if group == 128 and k % 128 == 0 and x_fp8.dim() == 2 and x_fp8.is_contiguous() and n > 128:
        kw.update(tile_shape=(192, 128), cluster_shape=(1, 2, 1), tma_gather=(1, 4))
    return kw


def _sm90_cluster(group: int, k: int, kind: str = "gemm", n: int = 0):
    """Thread block cluster for the non-gather 192x128 down / dx kernels.
    Clusters are bitwise (same K order).

    A 1x2 cluster multicasts the 192-row A tile to two N-adjacent CTAs, which
    cuts more L2->smem traffic than the 2x1 B multicast of a 128-row B tile
    (Round 18, H800): down / dx are 0-4% faster at every bench shape (e51 dx
    -3.5% vs 2x1, e51 down -3.7%). The 1x2 cluster needs an even number of
    128-wide N tiles; otherwise long K falls back to the 2x1 cluster (Round 16).
    """
    if group != 128:
        return (1, 1, 1)
    if kind == "gemm" or k >= 6144:
        if n % 256 == 0:
            return (1, 2, 1)
        if k >= 6144:
            return (2, 1, 1)
    return (1, 1, 1)


# Store the BF16 y1 intermediate (re-quantized to FP8 for the down GEMM) with
# round-to-odd so the FP8 quant equals quantizing the fp32 y1 directly.  RN here
# double-rounds fp32->bf16->fp8 and biases y1 magnitudes low.
_SM90_Y1_ROUND_ODD = os.environ.get("SONIC_MOE_SM90_Y1_ROUND_ODD", "1") == "1"
# Fuse the 1x128 y1 quant into the gated epilogue (from the fp32 y1, single
# rounding; the two CTAs of the 1x2 cluster exchange row amax via DSMEM).
_SM90_FUSE_Y1_128 = os.environ.get("SONIC_MOE_SM90_FUSE_Y1_128", "1") == "1"


def gemm_gated(
    x,
    w1,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    x_fp8_pre,
    w1_fp8,
    cfg,
    current_stream,
    activation_type,
):
    if _is_situ_activation_type(activation_type):
        raise NotImplementedError("SM90 FP8 gated forward supports SwiGLU only")
    group, block = _groups(cfg)
    x_fp8, x_scales = _sm90_activation_operand(
        x,
        None if x_fp8_pre is None else x_fp8_pre[0],
        None if x_fp8_pre is None else x_fp8_pre[1],
        group,
    )
    w_fp8, w_scales = w1_fp8
    device = _alloc_device(x_fp8)
    total_m = int(x_gather_idx.shape[0])
    n = int(w_fp8.shape[1])

    store_z = not cfg.recompute_z
    z_epilogue_quant = store_z and cfg.epilogue_quant and cfg.save_z_fp8
    gather_kw = _sm90_gather_kw(group, "gated", x_fp8, n)
    # G == 32: raw 1x32 y1 quant in the epilogue.  G == 128: fused 1x128 y1
    # quant across the 1x2 cluster (tile 192x128 only); otherwise BF16 y1 +
    # a separate quant kernel.
    fuse_y1_128 = (
        cfg.fuse_y1_quant and _SM90_FUSE_Y1_128 and group == 128
        and gather_kw.get("tile_shape") == (192, 128)
        and gather_kw.get("cluster_shape") == (1, 2, 1)
        and n % 256 == 0
    )
    fuse_y1 = (cfg.fuse_y1_quant and group == 32) or fuse_y1_128
    if not store_z:
        z_out, z_scale_out = None, None
    elif z_epilogue_quant:
        z_out = torch.empty((total_m, n), dtype=torch.float8_e4m3fn, device=device)
        z_scale_out = torch.empty((total_m, n // 32), dtype=torch.uint8, device=device)
    else:
        z_out = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
        z_scale_out = None
    if fuse_y1:
        y1_out = torch.empty((total_m, n // 2), dtype=torch.float8_e4m3fn, device=device)
        y1_scale_out = torch.empty(
            (total_m, n // (2 * group)), dtype=torch.uint8, device=device
        )
    else:
        y1_out = torch.empty((total_m, n // 2), dtype=torch.bfloat16, device=device)
        y1_scale_out = None
    sm90_scaled_gated(
        x_fp8, w_fp8, z_out, y1_out, x_scales, w_scales,
        activation="swiglu",
        z_scale_out=z_scale_out,
        postact_scale_out=y1_scale_out,
        swiglu_clamp_value=cfg.swiglu_clamp_value,
        # bf16 truncation only mirrors the G == 32 reference; the fused 128
        # path must quantize the fp32 y1 directly.
        postact_bf16_trunc=fuse_y1 and group == 32 and cfg.fuse_y1_bf16_trunc,
        postact_round_odd=_SM90_Y1_ROUND_ODD and not fuse_y1,
        postact_group=group if fuse_y1 else 32,
        cu_seqlens_m=expert_frequency_offset,
        activation_group=group,
        weight_block=block,
        A_idx=x_gather_idx,
        **gather_kw,
    )

    z_fp8_payload = None
    if z_out is None or z_epilogue_quant:
        if z_epilogue_quant:
            z_fp8_payload = (z_out, z_scale_out)
        z = _placeholder((total_m, n), torch.bfloat16, device)
    else:
        z = z_out
    if fuse_y1:
        y1 = _placeholder((total_m, n // 2), torch.bfloat16, device)
        return z, y1, y1_out, y1_scale_out, z_fp8_payload
    y1_fp8, y1_scales = quantize_activation_blockscaled_fast(y1_out, group_size=group)
    return z, y1_out, y1_fp8, y1_scales, z_fp8_payload


def gemm_gated_padded(
    x,
    w1,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    activation_type,
):
    _unsupported("gemm_gated_padded")


def recompute_z(
    x,
    w1,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    x_fp8_pre=None,
    x_scales_pre=None,
    w1_fp8_pre=None,
    *,
    cfg,
    fp8: bool,
):
    if w1_fp8_pre is None:
        raise RuntimeError("Sonic FP8 recompute requires explicit w1_fused payload")
    if (x_fp8_pre is None) != (x_scales_pre is None):
        raise ValueError("x_fp8_pre and x_scales_pre must be provided together")
    group, block = _groups(cfg)
    x_fp8, x_scales = _sm90_activation_operand(x, x_fp8_pre, x_scales_pre, group)
    w_fp8, w_scales = w1_fp8_pre
    device = _alloc_device(x_fp8)
    total_m = int(x_gather_idx.shape[0])
    n = int(w_fp8.shape[1])
    kwargs = dict(activation_group=group, weight_block=block)
    if fp8:
        z_fp8 = torch.empty((total_m, n), dtype=torch.float8_e4m3fn, device=device)
        z_scales = torch.empty((total_m, n // 32), dtype=torch.uint8, device=device)
        sm90_scaled_recompute_z(
            x_fp8, w_fp8, z_fp8, x_scales, w_scales,
            expert_frequency_offset, x_gather_idx, z_scale_out=z_scales, **kwargs,
        )
        return z_fp8, z_scales
    z = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
    if group == 128:
        kwargs.update(_sm90_gather_kw(group, "gather", x_fp8, n))
    sm90_scaled_recompute_z(
        x_fp8, w_fp8, z, x_scales, w_scales,
        expert_frequency_offset, x_gather_idx, **kwargs,
    )
    return z


def gemm(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    *,
    out_features: int,
    num_experts: int,
    cfg,
    out: torch.Tensor | None = None,
    scatter_dst: torch.Tensor | None = None,
    scatter_out: torch.Tensor | None = None,
    scatter_w: torch.Tensor | None = None,
) -> torch.Tensor:
    group, block = _groups(cfg)
    if out is None:
        out = torch.empty(
            (int(a_fp8.shape[0]), out_features), dtype=torch.bfloat16, device=_alloc_device(a_fp8)
        )
    # Non-gather varlen-M GEMMs (down-proj, dx); optional fused single-route
    # token reduction (see sm90_scaled_gemm).
    sm90_scaled_gemm(
        a_fp8, w_fp8, out, _raw_scales_u8(a_scales, a_fp8, group), w_scales,
        scatter_dst=scatter_dst, scatter_out=scatter_out, scatter_w=scatter_w,
        cu_seqlens_m=expert_frequency_offset,
        activation_group=group,
        weight_block=block,
        tile_shape=_sm90_tile(group, "gemm"),
        cluster_shape=_sm90_cluster(group, int(a_fp8.shape[1]), "gemm", out_features),
    )
    return out


def gemm_padded(x: torch.Tensor, w2: torch.Tensor, expert_frequency_offset: torch.Tensor):
    _unsupported("gemm_padded")


def dgated_col_quant_supported(cfg, preact, total_m: int) -> bool:
    """Whether :func:`gemm_dgated` can fuse the 1x128 wgrad col quant
    (the same operand :func:`quantize_wgrad_operand` would return)."""
    group, _ = _groups(cfg)
    return (
        group == 128 and preact is not None and _sm90_tile(group)[0] == 128
        and _wgrad_group(total_m) == 128
    )


def gemm_dgated(
    dout_fp8: torch.Tensor,
    dout_scales: torch.Tensor,
    *,
    w2_fp8: torch.Tensor,
    w2_scales: torch.Tensor,
    preact,
    preact_fp8,
    preact_scales,
    colvec_scale: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    activation: str,
    cfg,
    quantize_dz: bool = False,
    dz_out: torch.Tensor | None = None,
    col_quant: tuple = (),
    skip_bf16: tuple = (),
    dz_col_inplace: bool = False,
):
    """``quantize_dz=True`` also returns the rowwise ``(fp8, u8 scales)`` of dz
    (byte-identical to :func:`quantize_activation`), fused into the epilogue
    when supported (profile 128, BF16 preact), else ``None``.

    ``dz_out`` may be the BF16 ``preact`` itself: every output tile reads its
    own preact tile (same rows/cols) before storing dz there, so dz can
    overwrite a preact that is dead after this call.

    ``col_quant`` (subset of ``("y1s", "dz")``) fuses their colwise 1x128 wgrad
    operands (as :func:`quantize_wgrad_colwise128_sm90` returns them) into the
    epilogue and appends ``{name: operand}`` to the return tuple.

    ``skip_bf16`` (``("y1s",)`` or ``("y1s", "dz")``, each also in ``col_quant``;
    ``"dz"`` is ignored unless the rowwise dz quant is fused too) drops those BF16
    stores: y1s is returned as ``None`` (no buffer is allocated) and dz keeps the
    stale ``dz_out`` contents.

    ``dz_col_inplace`` (``dz_out=preact``, both stores skipped, rowwise dz quant
    fused) writes the dz col operand tile-locally into ``preact`` and returns it
    as a :class:`TileLocalCol`; ``preact`` then holds no valid dz/preact values."""
    device = _alloc_device(dout_fp8)
    total_m = int(x_gather_idx.shape[0])
    n = int(w2_fp8.shape[-2])
    group, block = _groups(cfg)
    tile = _sm90_tile(group)
    tile_n = tile[1]
    skip_bf16 = tuple(sorted(skip_bf16))
    if skip_bf16 not in ((), ("y1s",), ("dz", "y1s")) or not set(skip_bf16) <= set(col_quant):
        raise ValueError(f"unsupported skip_bf16={skip_bf16!r} for col_quant={col_quant!r}")
    if dz_out is not None:
        if (dz_out.dtype != torch.bfloat16 or tuple(dz_out.shape) != (total_m, 2 * n)
                or not dz_out.is_contiguous()):
            raise ValueError(f"dz_out must be contiguous bf16 {(total_m, 2 * n)}")
        if preact is not None and dz_out is not preact and dz_out.data_ptr() == preact.data_ptr():
            raise ValueError("dz_out may alias preact only as the same tensor")
        dz = dz_out
    else:
        dz = torch.empty((total_m, 2 * n), dtype=torch.bfloat16, device=device)
    if "y1s" in skip_bf16:
        # Never written (its TMA store is dropped): only the gmem layout matters,
        # so it borrows dz's storage instead of allocating [M, I] bf16.
        y1s = dz.view(-1)[: total_m * n].view(total_m, n)
    else:
        y1s = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
    colvec_reduce_partial = torch.empty(
        (total_m, (n + tile_n - 1) // tile_n), dtype=torch.float32, device=device
    )
    dz_q = None
    fused_q = {}
    if quantize_dz and group == 128 and preact is not None and (2 * n) % 128 == 0 and tile_n % 64 == 0:
        dz_q = (
            torch.empty((total_m, 2 * n), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((total_m, 2 * n // 128), dtype=torch.uint8, device=device),
        )
        fused_q = dict(dz_fp8_out=dz_q[0], dz_scale_out=dz_q[1])
    cols = {}
    dz_col_inplace = dz_col_inplace and dz_q is not None  # else dz bf16 stays the source
    if dz_col_inplace and not (
            "dz" in col_quant and skip_bf16 == ("dz", "y1s")
            and preact is not None and dz is preact):
        raise ValueError("dz_col_inplace needs col_quant/skip_bf16 with dz and y1s "
                         "and dz_out=preact")
    if col_quant:
        if not dgated_col_quant_supported(cfg, preact, total_m):
            raise ValueError("fused col quant needs profile 128, BF16 preact, 1x128 wgrad")
        for name, feat in (("y1s", n), ("dz", 2 * n)):
            if name in col_quant:
                scale = torch.empty((_wgrad128_scale_rows(total_m // 128, feat), feat),
                                    dtype=torch.uint8, device=device)
                if name == "dz" and dz_col_inplace:
                    cols[name] = TileLocalCol(preact, scale, expert_frequency_offset)
                    fused_q["dz_col_inplace"] = True
                else:
                    cols[name] = (
                        torch.empty((feat, total_m), dtype=torch.float8_e4m3fn, device=device),
                        scale, expert_frequency_offset,
                    )
                fused_q[name + "_col_out"] = tuple(cols[name][:2])
    if skip_bf16:
        if "dz" in skip_bf16 and dz_q is None:
            skip_bf16 = ("y1s",)  # dz bf16 stays the only rowwise source
        fused_q["skip_stores"] = skip_bf16
    sm90_scaled_dgated(
        dout_fp8, w2_fp8.contiguous(), dz,
        preact if preact is not None else preact_fp8,
        y1s,
        _raw_scales_u8(dout_scales, dout_fp8, group),
        w2_scales.contiguous(),
        activation=activation,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce_partial,
        preact_fp8=preact_fp8,
        preact_scales=None if preact_scales is None else preact_scales.contiguous(),
        swiglu_clamp_value=cfg.swiglu_clamp_value,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        activation_group=group,
        weight_block=block,
        activation_scale_order="source",
        tile_shape=tile,
        **fused_q,
    )
    out = (dz, None if "y1s" in skip_bf16 else y1s, colvec_reduce_partial.sum(dim=-1))
    if quantize_dz:
        out = (*out, dz_q)
    return (*out, cols) if col_quant else out


# ---------------------------------------------------------------------------
# FP8 weight gradient. Wgrad operand: ``(fp8 [feature, P], u8 [feature, P/32],
# packed_offsets)``, each expert span zero-padded to a multiple of 32.
# ---------------------------------------------------------------------------


_SM90_WGRAD_GROUP = int(os.environ.get("SONIC_MOE_SM90_WGRAD_GROUP", "128"))


def _wgrad_group(total_k: int) -> int:
    """1x128 needs 128-aligned expert segments (routing_block_size=128)."""
    return 128 if _SM90_WGRAD_GROUP == 128 and total_k % 128 == 0 else 32


def quantize_wgrad_operand(
    src: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    cfg,
    *,
    gather_idx: torch.Tensor | None = None,
    src_scales: torch.Tensor | None = None,
):
    total_k = int(gather_idx.shape[0]) if gather_idx is not None else int(src.shape[0])
    quant = (
        quantize_wgrad_colwise128_sm90 if _wgrad_group(total_k) == 128
        else quantize_wgrad_colwise_sm90
    )
    if src_scales is None:
        return quant(src, expert_frequency_offset, gather_idx=gather_idx)
    group = cfg.activation_group_size
    return quant(
        src, expert_frequency_offset,
        gather_idx=gather_idx,
        source_scales=_raw_scales_u8(src_scales, src, group),
        source_group=group,
    )


def quantize_dz_dout(
    dz: torch.Tensor,
    dout: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    cfg,
    *,
    dout_fp8: torch.Tensor | None = None,
    dout_scales: torch.Tensor | None = None,
    dz_rowwise=None,
    defer_dz_col: bool = False,
):
    # The three quantizations are independent kernels, so ``defer_dz_col``
    # (caller quantizes dz_col later) does not change any result.
    dz_col = None if defer_dz_col else quantize_wgrad_operand(dz, expert_frequency_offset, cfg)
    if dout_fp8 is not None:
        dout_col = quantize_wgrad_operand(
            dout_fp8, expert_frequency_offset, cfg,
            gather_idx=x_gather_idx, src_scales=dout_scales,
        )
    else:
        dout_col = quantize_wgrad_operand(
            dout, expert_frequency_offset, cfg, gather_idx=x_gather_idx,
        )
    return (quantize_activation(dz, cfg) if dz_rowwise is None else dz_rowwise), dz_col, dout_col


class TileLocalCol(tuple):
    """``(z, scales, offsets)``: a 1x128 colwise wgrad operand that
    ``gemm_dgated(dz_col_inplace=True)`` wrote tile-locally into the bf16
    ``[total_K, F]`` buffer ``z``; ``wgrad_gemm_varlen_k`` reads it in place."""

    def __new__(cls, z, scales, offsets):
        return super().__new__(cls, (z, scales, offsets))


def wgrad_gemm_varlen_k(
    a_col,
    b_col,
    expert_frequency_offset: torch.Tensor,
    *,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device,
    accumulator: torch.Tensor | None = None,
    out_dtype=None,
):
    a_data, a_scales, packed_offsets = a_col
    b_data, b_scales, _ = b_col
    tile_local = isinstance(a_col, TileLocalCol)
    if tile_local:
        feat_a, k_a = int(a_data.shape[1]), int(a_data.shape[0])
    else:
        feat_a, k_a = int(a_data.shape[0]), int(a_data.shape[1])
    group = 128 if (
        k_a % 128 == 0
        and tuple(a_scales.shape) == (_wgrad128_scale_rows(k_a // 128, feat_a), feat_a)
    ) else 32
    if tile_local and group != 128:
        raise ValueError("tile-local wgrad operands use 1x128 groups")
    kw = {"a_tile_local": True} if tile_local else {}
    if accumulator is not None:
        sm90_scaled_wgrad_varlen_k(
            a_data, b_data, accumulator, a_scales, b_scales, packed_offsets,
            add_to_output=True, group=group, **kw,
        )
        return None
    # The kernel is validated with an fp32 output; cast afterwards.
    out = torch.empty((num_experts, M, N), dtype=torch.float32, device=_alloc_device(a_data))
    sm90_scaled_wgrad_varlen_k(
        a_data, b_data, out, a_scales, b_scales, packed_offsets, group=group, **kw,
    )
    return out.to(out_dtype)
