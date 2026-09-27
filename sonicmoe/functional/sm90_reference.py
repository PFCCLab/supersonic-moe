# ********************************************************************************
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ********************************************************************************

"""Vectorized UE8M0 FP8 QDQ utilities for exact-SM90 SonicMoE."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..triton_utils import wrap_triton_kernel
from ..quack_utils.sm90_fp8_runtime import (
    bind_sm90_payload,
    consume_sm90_payload,
    current_sm90_invocation,
    get_sm90_carrier,
    has_sm90_native_carriers,
    is_sm90_native_active,
    register_sm90_carrier,
    replace_with_sm90_anchor,
    resolve_sm90_metadata_tensor,
    unique_retained_sm90_invocation,
    use_sm90_invocation,
)
from ..quack_utils.blockscaled_fp8_gemm import quantize_activation_blockscaled_fast
from ..quack_utils import gemm_sm100_fp8_zeromat as _sm100_zeromat


def blockscaled_fp8_gemm_zeromat_quant(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    A_idx: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    z_fp8_out: torch.Tensor | None = None,
    z_scale_out: torch.Tensor | None = None,
):
    """Recompute z only, then materialize its fixed raw-UE8M0 1x32 ABI."""
    invocation = current_sm90_invocation() or unique_retained_sm90_invocation()
    if invocation is None:
        return _sm100_zeromat.blockscaled_fp8_gemm_zeromat_quant(
            A, B, cu_seqlens_m, A_idx, a_scales, b_scales,
            z_fp8_out=z_fp8_out, z_scale_out=z_scale_out,
        )
    from ..quack_utils.gemm_sm90_scaled import sm90_scaled_recompute_z

    rows, cols = int(A_idx.shape[0]), int(B.shape[-2])
    if cols % 32:
        raise ValueError(f"SM90 recomputed z width must be divisible by 32, got {cols}")
    data_shape = (rows, cols)
    scale_shape = (rows, cols // 32)
    if z_fp8_out is None:
        z_fp8_out = torch.empty(data_shape, dtype=torch.float8_e4m3fn, device=A.device)
    elif tuple(z_fp8_out.shape) != data_shape or z_fp8_out.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "z_fp8_out must be E4M3 with shape "
            f"{data_shape}, got {tuple(z_fp8_out.shape)} and {z_fp8_out.dtype}"
        )
    if z_scale_out is None:
        z_scale_out = torch.empty(scale_shape, dtype=torch.uint8, device=A.device)
    elif tuple(z_scale_out.shape) != scale_shape or z_scale_out.dtype != torch.uint8:
        raise TypeError(
            "z_scale_out must be raw uint8 UE8M0 with shape "
            f"{scale_shape}, got {tuple(z_scale_out.shape)} and {z_scale_out.dtype}"
        )
    for output, name in ((z_fp8_out, "z_fp8_out"), (z_scale_out, "z_scale_out")):
        if output.device != A.device or not output.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the same device as A")

    with use_sm90_invocation(invocation):
        sm90_scaled_recompute_z(
            A,
            B,
            z_fp8_out,
            a_scales,
            b_scales,
            cu_seqlens_m,
            A_idx,
            z_scale_out=z_scale_out,
            activation_group=invocation.activation_group_size,
            weight_block=invocation.weight_block_size,
        )
    return z_fp8_out, z_scale_out


def blockscaled_fp8_gemm_zeromat_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    A_idx: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    z_bf16_out: torch.Tensor | None = None,
):
    """Native SM90 recompute-only GEMM with a BF16 z output."""
    invocation = current_sm90_invocation() or unique_retained_sm90_invocation()
    if invocation is None:
        return _sm100_zeromat.blockscaled_fp8_gemm_zeromat_bf16(
            A, B, cu_seqlens_m, A_idx, a_scales, b_scales,
            z_bf16_out=z_bf16_out,
        )
    from ..quack_utils.gemm_sm90_scaled import sm90_scaled_gemm

    rows, cols = int(A_idx.shape[0]), int(B.shape[-2])
    if z_bf16_out is None:
        z_bf16_out = torch.empty(
            (rows, cols), dtype=torch.bfloat16, device=A.device
        )
    with use_sm90_invocation(invocation):
        sm90_scaled_gemm(
            A, B, z_bf16_out, a_scales, b_scales,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            activation_group=invocation.activation_group_size,
            weight_block=invocation.weight_block_size,
            z_mode="recompute_z_only",
        )
    return z_bf16_out


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


def register_sm90_activation_payload(
    carrier: torch.Tensor,
    data: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    role: str = "activation",
):
    """Register communicated/native bytes without dequantizing the payload."""
    if not is_sm90_native_active():
        return None
    existing = get_sm90_carrier(carrier)
    if existing is not None:
        if existing.data is data and existing.scales is scales:
            return existing
        return register_sm90_carrier(
            carrier,
            data=data,
            scales=scales,
            role=role,
            metadata={"group_size": int(group_size)},
            replace=True,
        )
    return register_sm90_carrier(
        carrier,
        data=data,
        scales=scales,
        role=role,
        metadata={"group_size": int(group_size)},
    )


def dequantize_activation_sm90(
    data: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    carrier: torch.Tensor | None = None,
    communication_payload: bool = False,
) -> torch.Tensor:
    if group_size not in _SM90_QDQ_GROUP_SIZES:
        raise ValueError(f"SM90 activation group size must be 32 or 128, got {group_size}")
    rows, cols = data.shape
    if cols % group_size:
        raise ValueError(
            f"SM90 activation dimension {cols} must be divisible by {group_size}"
        )
    # Native Hopper treats this API as the FP8 communication boundary. Existing
    # functional callers cannot pass ``communication_payload`` without changing
    # their signature, so an active native scope is sufficient to preserve bytes.
    if is_sm90_native_active():
        raw_scales = _raw_activation_scales(scales, rows, cols // group_size)
        if carrier is None:
            carrier = torch.empty(
                data.shape, dtype=torch.bfloat16, device=data.device
            ).as_strided(tuple(data.shape), (0,) * data.ndim)
        register_sm90_activation_payload(
            carrier, data, raw_scales, group_size, role="communication"
        )
        return replace_with_sm90_anchor(carrier)
    if communication_payload or has_sm90_native_carriers():
        invocation = current_sm90_invocation() or unique_retained_sm90_invocation()
        if invocation is not None:
            if carrier is None:
                carrier = torch.empty(
                    data.shape, dtype=torch.bfloat16, device=data.device
                ).as_strided(tuple(data.shape), (0,) * data.ndim)
            bind_sm90_payload(
                carrier, data, scales, group_size, invocation=invocation
            )
            return carrier
    groups = cols // group_size
    raw_scales = _raw_activation_scales(scales, rows, groups)
    values = data.float().reshape(rows, groups, group_size)
    restored = values * _decode_ue8m0(raw_scales).unsqueeze(-1)
    return restored.reshape(rows, cols).to(torch.bfloat16)


def quantize_activation_sm90(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size not in _SM90_QDQ_GROUP_SIZES:
        raise ValueError(f"SM90 activation group size must be 32 or 128, got {group_size}")
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError(
            "SM90 activation QDQ expects a rank-2 tensor whose hidden dimension "
            f"is divisible by {group_size}, got {tuple(x.shape)}"
        )
    existing = get_sm90_carrier(x)
    if existing is None:
        pending = consume_sm90_payload(
            x, invocation=current_sm90_invocation()
        )
        if pending is not None:
            data, scales, pending_group_size, role, pending_metadata = pending
            if int(pending_group_size) != int(group_size):
                raise RuntimeError("SM90 pending activation payload profile mismatch")
            invocation = current_sm90_invocation()
            if invocation is not None:
                existing = register_sm90_carrier(
                    x,
                    data=data,
                    scales=scales,
                    role=role,
                    metadata={
                        "group_size": int(group_size),
                        **pending_metadata,
                    },
                    invocation=invocation,
                )
    if existing is not None:
        if existing.data is not None:
            existing.metadata["quantize_hits"] = int(existing.metadata.get("quantize_hits", 0)) + 1
            return existing.data, existing.scales, x
        # Do not force an unrelated deferred recipe merely to attach wgrad
        # metadata; the dispatcher can materialize the independent colwise
        # recipe directly.
        if existing.recipe is None:
            existing.materialize()
            if existing.data is not None:
                return existing.data, existing.scales, x
    data, scales = quantize_activation_blockscaled_fast(
        x.contiguous(), group_size=group_size
    )
    invocation = current_sm90_invocation()
    if invocation is None:
        invocation = unique_retained_sm90_invocation()
    if invocation is not None:
        register_sm90_carrier(
            x,
            data=data,
            scales=scales,
            role="activation",
            metadata={"group_size": int(group_size), "quantize_hits": 1},
            invocation=invocation,
        )
        return data, scales, x
    if has_sm90_native_carriers():
        invocation = unique_retained_sm90_invocation()
        if invocation is not None:
            register_sm90_carrier(
                x,
                data=data,
                scales=scales,
                role="activation",
                orientation="rowwise",
                metadata={
                    "group_size": int(group_size),
                    "source": x,
                    "orientation": "rowwise",
                },
                invocation=invocation,
            )
            return data, scales, x
    return data, scales, dequantize_activation_sm90(data, scales, group_size)


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


def attach_sm90_wgrad_colwise_recipe(
    carrier: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    source: torch.Tensor | None = None,
    gather_idx: torch.Tensor | None = None,
    invocation=None,
):
    """Attach a lazy independent colwise payload recipe to an existing carrier."""
    native = get_sm90_carrier(carrier, invocation=invocation)
    if native is None:
        return None
    if invocation is None:
        invocation = native.metadata.get("invocation")
    source = carrier if source is None else source
    source_descriptor = native.metadata.get("source")
    if source_descriptor is None or source is not carrier:
        register_sm90_carrier(
            carrier,
            role=native.role,
            metadata={"source": source},
            invocation=invocation,
        )

    def recipe():
        recipe_source = resolve_sm90_metadata_tensor(native.metadata, "source")
        return quantize_wgrad_colwise_sm90(
            recipe_source, expert_offsets, gather_idx=gather_idx
        )

    native.metadata.update({
        "colwise_recipe": recipe,
        "colwise_data": None,
        "colwise_scales": None,
        "colwise_group_size": 32,
        "colwise_layout": "feature_total_k_k_major",
        "colwise_expert_offsets": expert_offsets,
        "colwise_gathered": gather_idx is not None,
    })
    return native


def ensure_sm90_wgrad_colwise_payload(
    carrier: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    source: torch.Tensor | None = None,
    gather_idx: torch.Tensor | None = None,
    invocation=None,
):
    """Attach the recipe when needed and return the materialized raw payload."""
    native = get_sm90_carrier(carrier, invocation=invocation)
    if native is None:
        return None
    offsets = native.metadata.get("colwise_expert_offsets")
    same_offsets = offsets is expert_offsets
    same_gather = bool(native.metadata.get("colwise_gathered")) == (gather_idx is not None)
    if native.metadata.get("colwise_data") is None and (
        native.metadata.get("colwise_recipe") is None or not same_offsets or not same_gather
    ):
        attach_sm90_wgrad_colwise_recipe(
            carrier,
            expert_offsets,
            source=source,
            gather_idx=gather_idx,
            invocation=invocation,
        )
    native.materialize_colwise()
    data = native.metadata.get("colwise_data")
    scales = native.metadata.get("colwise_scales")
    packed_offsets = native.metadata.get("colwise_packed_offsets")
    if data is None or scales is None:
        return None
    return (data, scales) if packed_offsets is None else (data, scales, packed_offsets)


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


def register_sm90_weight_carrier(
    weight: torch.Tensor,
    data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    *,
    orientation: str = "identity",
    source_version=None,
):
    """Attach a persistent FP8 payload to a per-invocation BF16 carrier."""
    invocation = current_sm90_invocation()
    if invocation is None:
        return None
    existing = get_sm90_carrier(weight, invocation=invocation)
    if existing is not None:
        return existing
    return register_sm90_carrier(
        weight,
        data=data,
        scales=scales,
        role="weight",
        orientation=orientation,
        metadata={
            "block_size": int(block_size),
            "source_version": source_version,
        },
        invocation=invocation,
    )


def register_sm90_weight_carriers(
    weight: torch.Tensor,
    data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    *,
    source_version=None,
) -> tuple[object, object] | None:
    """Register identity and transposed payload views without BF16 materialization."""
    invocation = current_sm90_invocation()
    if invocation is None or not is_sm90_native_active():
        return None
    identity = register_sm90_weight_carrier(
        weight,
        data,
        scales,
        block_size,
        orientation="identity",
        source_version=source_version,
    )
    weight_t = weight.permute(0, 2, 1)
    data_t = data.permute(0, 2, 1)
    scales_t = scales.permute(0, 2, 1) if scales.ndim == 3 else scales.mT
    transposed = register_sm90_carrier(
        weight_t,
        data=data_t,
        scales=scales_t,
        role="weight",
        orientation="transposed",
        metadata={
            "block_size": int(block_size),
            "source_version": source_version,
        },
        invocation=invocation,
    )
    return identity, transposed


def quantize_sm90_weights(
    w1: torch.Tensor, w2: torch.Tensor, block_size: int
) -> dict[str, object]:
    w1_payload = quantize_weight_sm90(w1, block_size)
    w2_payload = quantize_weight_sm90(w2, block_size)
    return {
        "format": f"{block_size}x{block_size}",
        "w1": w1_payload,
        "w2": w2_payload,
        # Compatibility fields intentionally carry the BF16 source tensors.
        # Native invocations resolve FP8 data/scales from the runtime registry;
        # the QDQ oracle remains available through dequantize_weight_sm90().
        "w1_dequantized": w1,
        "w2_dequantized": w2,
    }
