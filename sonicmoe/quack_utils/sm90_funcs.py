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

import torch
import triton
import triton.language as tl

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
from .swiglu_triton import dequantize_blockscaled_fp8


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
    if scales.dtype == torch.uint8:
        raw = scales
    elif str(scales.dtype) in ("torch.int32", "paddle.int32", "int32"):
        raw = scales.contiguous().view(torch.uint8)
    else:
        raw = scales.contiguous().view(torch.uint8)
    if raw.numel() < rows * groups:
        raise ValueError(
            "Invalid SM90 UE8M0 activation scale carrier: "
            f"need {rows * groups} bytes, got {raw.numel()}"
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
    restored = (
        data.float().reshape(rows, cols // group_size, group_size)
        * _decode_ue8m0(raw_scales).unsqueeze(-1)
    )
    return restored.reshape(rows, cols).to(torch.bfloat16).contiguous()


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


def _encode_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    exponent = torch.ceil(torch.log(safe_amax / 448.0) / 0.6931471805599453)
    exponent = torch.minimum(
        torch.maximum(exponent, torch.full_like(exponent, -126.0)),
        torch.full_like(exponent, 127.0),
    )
    encoded = (exponent.to(torch.int32) + 127).to(torch.uint8)
    return torch.where(amax > 0, encoded, torch.zeros_like(encoded))


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
    return _raw_activation_scales(scales, rows, cols // group).contiguous()


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
    device = x_fp8.device
    total_m = int(x_gather_idx.shape[0])
    n = int(w_fp8.shape[1])

    store_z = not cfg.recompute_z
    z_epilogue_quant = store_z and cfg.epilogue_quant and cfg.save_z_fp8
    # Epilogue scales are always raw 1x32; the down GEMM needs y1 in the
    # activation group, so the y1 quant can only be fused when G == 32.
    fuse_y1 = cfg.fuse_y1_quant and group == 32
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
        y1_scale_out = torch.empty((total_m, n // 64), dtype=torch.uint8, device=device)
    else:
        y1_out = torch.empty((total_m, n // 2), dtype=torch.bfloat16, device=device)
        y1_scale_out = None
    sm90_scaled_gated(
        x_fp8, w_fp8, z_out, y1_out, x_scales, w_scales,
        activation="swiglu",
        z_scale_out=z_scale_out,
        postact_scale_out=y1_scale_out,
        swiglu_clamp_value=cfg.swiglu_clamp_value,
        postact_bf16_trunc=fuse_y1 and cfg.fuse_y1_bf16_trunc,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        activation_group=group,
        weight_block=block,
        activation_scale_order="source",
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
    device = x_fp8.device
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
) -> torch.Tensor:
    group, block = _groups(cfg)
    out = torch.empty(
        (int(a_fp8.shape[0]), out_features), dtype=torch.bfloat16, device=a_fp8.device
    )
    sm90_scaled_gemm(
        a_fp8, w_fp8, out, _raw_scales_u8(a_scales, a_fp8, group), w_scales,
        cu_seqlens_m=expert_frequency_offset,
        activation_group=group,
        weight_block=block,
    )
    return out


def gemm_padded(x: torch.Tensor, w2: torch.Tensor, expert_frequency_offset: torch.Tensor):
    _unsupported("gemm_padded")


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
):
    device = dout_fp8.device
    total_m = int(x_gather_idx.shape[0])
    n = int(w2_fp8.shape[-2])
    y1s = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
    group, block = _groups(cfg)
    tile_n = 128  # sm90_scaled_* default tile_shape=(128, 128)
    dz = torch.empty((total_m, 2 * n), dtype=torch.bfloat16, device=device)
    colvec_reduce_partial = torch.empty(
        (total_m, (n + tile_n - 1) // tile_n), dtype=torch.float32, device=device
    )
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
    )
    return dz, y1s, colvec_reduce_partial.sum(dim=-1)


# ---------------------------------------------------------------------------
# FP8 weight gradient. Wgrad operand: ``(fp8 [feature, P], u8 [feature, P/32],
# packed_offsets)``, each expert span zero-padded to a multiple of 32.
# ---------------------------------------------------------------------------


def quantize_wgrad_operand(
    src: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    cfg,
    *,
    gather_idx: torch.Tensor | None = None,
    src_scales: torch.Tensor | None = None,
):
    if src_scales is None:
        return quantize_wgrad_colwise_sm90(
            src, expert_frequency_offset, gather_idx=gather_idx
        )
    group = cfg.activation_group_size
    return quantize_wgrad_colwise_sm90(
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
):
    dz_col = quantize_wgrad_operand(dz, expert_frequency_offset, cfg)
    if dout_fp8 is not None:
        dout_col = quantize_wgrad_operand(
            dout_fp8, expert_frequency_offset, cfg,
            gather_idx=x_gather_idx, src_scales=dout_scales,
        )
    else:
        dout_col = quantize_wgrad_operand(
            dout, expert_frequency_offset, cfg, gather_idx=x_gather_idx,
        )
    return quantize_activation(dz, cfg), dz_col, dout_col


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
    if accumulator is not None:
        sm90_scaled_wgrad_varlen_k(
            a_data, b_data, accumulator, a_scales, b_scales, packed_offsets,
            add_to_output=True,
        )
        return None
    # The kernel is validated with an fp32 output; cast afterwards.
    out = torch.empty((num_experts, M, N), dtype=torch.float32, device=a_data.device)
    sm90_scaled_wgrad_varlen_k(a_data, b_data, out, a_scales, b_scales, packed_offsets)
    return out.to(out_dtype)
