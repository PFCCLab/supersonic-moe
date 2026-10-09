# ********************************************************************************
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ********************************************************************************

"""Device-specific FP8 kernel facade for the aligned blockscaled MoE flow.

``functional/__init__.py`` runs one FP8 flow for every GPU; each op below picks
the kernel for the current architecture:

* SM100: CUTLASS blockscaled kernels, E4M3 + ISA-packed UE8M0 1x32 scales,
  per-row 1x32 weight scales.
* SM90: native WGMMA FP8 kernels (``quack_utils/gemm_sm90_scaled.py``),
  E4M3 + raw uint8 UE8M0 ``[rows, K/G]`` activation scales and GxG weight
  blocks ``[E, N/B, K/B]`` (``G == B`` in {32, 128}).

SM100 branches look up helpers on the ``functional`` module at call time so
that the existing implementation (and test patches on it) stays the single
source of truth.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import torch

# Imported eagerly: sonicmoe is loaded under paddle's torch-compat guard, and a
# lazy import at call time would bind these modules to a different ``torch``.
from ..quack_utils.gemm_sm90_scaled import (
    sm90_scaled_dgated,
    sm90_scaled_gated,
    sm90_scaled_gemm,
    sm90_scaled_recompute_z,
    sm90_scaled_wgrad_varlen_k,
)
from .sm90_reference import (
    _raw_activation_scales,
    is_sm90_device,
    materialize_activation_sm90,
    quantize_wgrad_colwise_sm90,
    validate_sm90_qdq_profile,
)


def _functional():
    return sys.modules[__package__]


@lru_cache(maxsize=None)
def _is_sm90(device_key: str) -> bool:
    # One GPU architecture per process: resolve against the current device.
    return is_sm90_device()


def is_sm90(device) -> bool:
    return _is_sm90(str(device))


def _groups(cfg) -> tuple[int, int]:
    return validate_sm90_qdq_profile(cfg.activation_group_size, cfg.weight_block_size)


def _raw_scales_u8(scales: torch.Tensor, data: torch.Tensor, group: int) -> torch.Tensor:
    rows, cols = int(data.shape[0]), int(data.shape[1])
    return _raw_activation_scales(scales, rows, cols // group).contiguous()


def _placeholder(shape, dtype, device) -> torch.Tensor:
    """Zero-stride tensor standing in for an output that is not materialized."""
    return torch.empty(1, dtype=dtype, device=device).as_strided(tuple(shape), (0,) * len(shape))


def _sm90_activation_operand(x, x_fp8_pre, x_scales_pre, group):
    """T-sized rowwise E4M3 activation with raw u8 ``[T, K/G]`` scales."""
    if x_fp8_pre is not None:
        return x_fp8_pre, _raw_scales_u8(x_scales_pre, x_fp8_pre, group)
    if x is None:
        raise ValueError("BF16 x is required when no prequant activation payload is provided")
    return _functional().quantize_activation_blockscaled_fast(x, group_size=group)


# ---------------------------------------------------------------------------
# Config / layout
# ---------------------------------------------------------------------------


def constrain_config(cfg, device):
    """Restrict ``cfg`` (in place, idempotent) to what the device supports.

    SM90 supports every FP8 option of the fused gated flow (``save_z_fp8``,
    ``epilogue_quant``, ``recompute_z``, ``fuse_y1_quant``, ``fp8_wgrad``).
    The split (``fused_gated=False``) path is SM100-only, so SM90 always uses
    the fused gated kernel.
    """
    if cfg.enabled and is_sm90(device):
        _groups(cfg)
        cfg.fused_gated = True
    return cfg


def segments_aligned(expert_frequency_offset: torch.Tensor) -> bool:
    """Whether every expert segment satisfies the FP8 kernels' M alignment.

    SM100 ISA-packed scales need 128-row segments; the SM90 varlen kernels
    consume raw per-row scales and accept arbitrary segment lengths.
    """
    if is_sm90(expert_frequency_offset.device):
        return True
    return _functional()._all_segments_128_aligned(expert_frequency_offset)


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


def quantize_activation(x: torch.Tensor, cfg):
    """Rowwise E4M3 quantization in the layout consumed by this device's GEMMs."""
    F = _functional()
    if is_sm90(x.device):
        return F.quantize_activation_blockscaled_fast(x, group_size=cfg.activation_group_size)
    return F.quantize_and_pack_activation(x)


def quantize_activation_raw(x: torch.Tensor, cfg):
    """Rowwise E4M3 quantization with raw (un-packed) UE8M0 scales."""
    F = _functional()
    if is_sm90(x.device):
        return F.quantize_activation_blockscaled_fast(x, group_size=cfg.activation_group_size)
    return F.quantize_activation_blockscaled_fast(x)


def dequantize_activation(data: torch.Tensor, scales: torch.Tensor, cfg) -> torch.Tensor:
    """BF16 view of a rowwise E4M3 payload with raw UE8M0 scales."""
    if is_sm90(data.device):
        return materialize_activation_sm90(data, scales, cfg.activation_group_size)
    F = _functional()
    return F.dequantize_blockscaled_fp8(
        data,
        F._raw_1x32_scale_bytes(scales, int(data.shape[0]), int(data.shape[1])),
    )


def quantize_dz_for_actgrad(dz: torch.Tensor, cfg):
    """Prepare the dz operand of the up-projection actgrad: ``(data, scales)``."""
    return quantize_activation(dz, cfg)


# ---------------------------------------------------------------------------
# GEMMs
# ---------------------------------------------------------------------------


def gated_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    x_fp8_pre,
    w1_fp8,
    cfg,
    current_stream,
    activation_type,
):
    """Up-projection ``z = x[gather] @ w1``, ``y1 = act(z)``.

    Returns ``(z, y1, y1_fp8, y1_scales)``; ``y1_fp8``/``y1_scales`` are the
    down-projection operand (``None`` if the caller must quantize ``y1``).
    With ``recompute_z`` z is a zero-stride placeholder; with
    ``epilogue_quant`` + ``save_z_fp8`` z is a placeholder and the E4M3 z with
    raw 1x32 scales is published as ``_PREQUANTIZED_SCALES["z_fp8"]``; with a
    fused y1 quant y1 is a placeholder.
    """
    F = _functional()
    if not is_sm90(expert_frequency_offset.device):
        return F._fused_blockscaled_gated_forward(
            x, w1, expert_frequency_offset, x_gather_idx,
            x_fp8_pre=x_fp8_pre,
            w1_fp8_pre=w1_fp8, store_z=not cfg.recompute_z,
            fp8_config=cfg,
            current_stream=current_stream,
            w1_pretransposed=True,
            activation_type=activation_type,
        )

    if F._is_situ_activation_type(activation_type):
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

    if z_out is None or z_epilogue_quant:
        if z_epilogue_quant:
            F._PREQUANTIZED_SCALES["z_fp8"] = (z_out, z_scale_out)
        z = _placeholder((total_m, n), torch.bfloat16, device)
    else:
        z = z_out
    if fuse_y1:
        y1 = _placeholder((total_m, n // 2), torch.bfloat16, device)
        return z, y1, y1_out, y1_scale_out
    y1_fp8, y1_scales = F.quantize_activation_blockscaled_fast(y1_out, group_size=group)
    return z, y1_out, y1_fp8, y1_scales


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
    """Re-run the up-projection GEMM for backward (``recompute_z``).

    Arguments after ``x_gather_idx`` are the tuple stashed by the up-projection
    forward. Returns ``(z_fp8, z_raw_1x32_scales)`` if ``fp8`` else BF16 z.
    """
    F = _functional()
    if not is_sm90(expert_frequency_offset.device):
        fn = F._recompute_z_fp8 if fp8 else F._recompute_z_bf16
        return fn(
            x, w1, expert_frequency_offset, x_gather_idx,
            x_fp8_pre, x_scales_pre, w1_fp8_pre,
        )

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


def quantize_z_y1_fused(z: torch.Tensor, y1: torch.Tensor, cfg):
    """``SONIC_MOE_FP8_FUSED_ZY1_QUANT``: z raw 1x32 + y1 in the down-GEMM layout.

    Returns ``(z_fp8, z_scales, y1_fp8, y1_scales)``.
    """
    F = _functional()
    if is_sm90(z.device):
        z_fp8, z_scales = F.quantize_activation_blockscaled_fast(z)
        return (z_fp8, z_scales, *quantize_activation(y1, cfg))
    return F.fused_z_save_y1_quant(z, y1)


def down_forward(
    y1_fp8: torch.Tensor,
    y1_scales: torch.Tensor,
    w2: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    *,
    w2_fp8,
    cfg,
) -> torch.Tensor:
    """Down-projection ``y2 = y1 @ w2`` (BF16 output, varlen over experts)."""
    F = _functional()
    w_fp8, w_scales = w2_fp8
    if not is_sm90(y1_fp8.device):
        return F.blockscaled_fp8_gemm_varlen(
            y1_fp8, w2, expert_frequency_offset,
            a_scales=y1_scales,
            w_fp8=w_fp8, w_scales=w_scales,
            out_dtype=torch.bfloat16,
            assume_aligned=True,
        )

    group, block = _groups(cfg)
    out = torch.empty(
        (int(y1_fp8.shape[0]), int(w_fp8.shape[1])),
        dtype=torch.bfloat16, device=y1_fp8.device,
    )
    sm90_scaled_gemm(
        y1_fp8, w_fp8, out, _raw_scales_u8(y1_scales, y1_fp8, group), w_scales,
        cu_seqlens_m=expert_frequency_offset,
        activation_group=group,
        weight_block=block,
    )
    return out


def dgated(
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
    """Fused down-projection backward + activation backward.

    ``dout_fp8`` is T-sized with raw UE8M0 scales (uint8 or an int32 carrier)
    and is gathered by ``x_gather_idx`` inside the kernel. ``w2_fp8`` is the
    ``[E, I, H]`` transposed weight. ``preact`` is BF16 z, or ``None`` when the
    FP8 ``preact_fp8``/``preact_scales`` pair is supplied.

    Returns ``(dz, y1s, ds)`` with ``ds`` the per-row router-score gradient.
    """
    F = _functional()
    device = dout_fp8.device
    total_m = int(x_gather_idx.shape[0])
    n = int(w2_fp8.shape[-2])
    y1s = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)

    if is_sm90(device):
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

    config = F.gemm_dgated.default_config(device, num_experts=w2_fp8.shape[0])
    dz = (
        preact
        if (
            preact is not None
            and F._can_reuse_bf16_preact_for_dz(preact, total_m, n, device)
        )
        else torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=device)
    )
    colvec_reduce_partial = torch.empty(
        (total_m, (n + config.tile_n - 1) // config.tile_n),
        dtype=torch.float32,
        device=device,
    )
    a_scales = F._gather_1x32_scales_to_isa(
        dout_scales, x_gather_idx, int(dout_fp8.shape[0]), int(dout_fp8.shape[1]),
    )
    F.gemm_dgated_kernel(
        dout_fp8,
        w2_fp8,
        dz,
        preact if preact is not None else dz,  # ignored when preact_fp8 is set
        y1s,
        None,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        persistent=True,
        max_swizzle_size=config.max_swizzle_size,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce_partial,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        a_scales=a_scales,
        b_scales=w2_scales,
        preact_fp8=preact_fp8,
        preact_scales=preact_scales,
        swiglu_clamp_value=cfg.swiglu_clamp_value,
    )
    return dz, y1s, colvec_reduce_partial.sum(dim=-1)


def actgrad(
    dz_fp8: torch.Tensor,
    dz_scales,
    w1T_fp8: torch.Tensor,
    w1T_scales: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    *,
    H: int,
    num_experts: int,
    cfg,
) -> torch.Tensor:
    """Up-projection input gradient ``dx_expanded = dz @ w1^T`` (BF16).

    ``dz_fp8``/``dz_scales`` come from :func:`quantize_dz_for_actgrad`;
    ``w1T_fp8`` is the ``[E, H, 2I]`` (K-major) transposed weight.
    """
    if not is_sm90(dz_fp8.device):
        return _functional()._run_cutlass_blockscaled_gemm(
            dz_fp8, dz_scales,
            w1T_fp8, w1T_scales,
            expert_frequency_offset,
            total_M=dz_fp8.shape[0],
            K=dz_fp8.shape[1],
            H=H,
            num_experts=num_experts,
            out_dtype=torch.bfloat16,
            device=dz_fp8.device,
        )

    group, block = _groups(cfg)
    dx_expanded = torch.empty(
        (int(dz_fp8.shape[0]), H), dtype=torch.bfloat16, device=dz_fp8.device
    )
    sm90_scaled_gemm(
        dz_fp8, w1T_fp8, dx_expanded,
        _raw_scales_u8(dz_scales, dz_fp8, group), w1T_scales,
        cu_seqlens_m=expert_frequency_offset,
        activation_group=group,
        weight_block=block,
    )
    return dx_expanded


# ---------------------------------------------------------------------------
# FP8 weight gradient (varlen-K over the routed token dimension)
#
# A "wgrad operand" is an opaque column-wise (token-dim) quantized tensor:
# SM100 ``(fp8, isa_scales)``; SM90 ``(fp8 [feature, P], u8 [feature, P/32],
# packed_offsets)`` where each expert span is zero-padded to a multiple of 32.
# ---------------------------------------------------------------------------


def wgrad_operand(
    src: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    cfg,
    *,
    gather_idx: torch.Tensor | None = None,
    src_scales: torch.Tensor | None = None,
):
    """Column-wise quantize ``src[gather_idx]`` (``[TK, feature]``) for wgrad.

    ``src`` is BF16, or rowwise E4M3 with raw UE8M0 ``src_scales``.
    """
    F = _functional()
    feature = int(src.shape[1])
    if is_sm90(src.device):
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

    total_k = int(gather_idx.shape[0]) if gather_idx is not None else int(src.shape[0])
    if src_scales is None:
        return F.colwise_quantize_and_pack(
            src, logical_rows=feature, logical_cols=total_k, gather_idx=gather_idx,
        )
    return F.dequant_colwise_quantize_and_pack_from_isa(
        src,
        F._ensure_isa_1x32_scales(src_scales, int(src.shape[0]), feature),
        logical_rows=feature, logical_cols=total_k,
        gather_idx=gather_idx,
    )


def down_wgrad_dz_dout(
    dz: torch.Tensor,
    dout: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    cfg,
    *,
    dout_fp8: torch.Tensor | None = None,
    dout_scales: torch.Tensor | None = None,
):
    """Quantize the down-projection backward wgrad operands.

    ``dout`` (T-sized, BF16) or its raw-scaled FP8 payload ``dout_fp8`` is
    gathered by ``x_gather_idx``. Returns ``(dz_actgrad, dz_col, dout_col)``
    where ``dz_actgrad`` is the ``(data, scales)`` pair for :func:`actgrad`.
    """
    F = _functional()
    total_k = int(x_gather_idx.shape[0])
    if is_sm90(dz.device):
        dz_col = wgrad_operand(dz, expert_frequency_offset, cfg)
        if dout_fp8 is not None:
            dout_col = wgrad_operand(
                dout_fp8, expert_frequency_offset, cfg,
                gather_idx=x_gather_idx, src_scales=dout_scales,
            )
        else:
            dout_col = wgrad_operand(
                dout, expert_frequency_offset, cfg, gather_idx=x_gather_idx,
            )
        return quantize_dz_for_actgrad(dz, cfg), dz_col, dout_col

    if dout_fp8 is not None:
        dz_fp8, dz_scales, dz_col_fp8, dz_col_scales = F.dual_quantize_varlen(
            dz, total_k, dz.shape[1]
        )
        dout_col = wgrad_operand(
            dout_fp8, expert_frequency_offset, cfg,
            gather_idx=x_gather_idx, src_scales=dout_scales,
        )
    else:
        dz_fp8, dz_scales, dz_col_fp8, dz_col_scales, dout_col_fp8, dout_col_scales = (
            F.fused_dual_colwise_quantize(
                dz, dout, x_gather_idx, total_k, dz.shape[1], dout.shape[1],
            )
        )
        dout_col = (dout_col_fp8, dout_col_scales)
    return (dz_fp8, dz_scales), (dz_col_fp8, dz_col_scales), dout_col


def wgrad(
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
    """Per-expert ``D[e] = A[:, seg_e] @ B[:, seg_e]^T`` of shape ``[E, M, N]``.

    With ``accumulator`` (fp32 ``[E, M, N]``) the result is added in place and
    ``None`` is returned; otherwise a fresh ``out_dtype`` tensor is returned.
    """
    if not is_sm90(device):
        F = _functional()
        if accumulator is not None:
            gemm = (
                F._run_cutlass_blockscaled_gemm_varlen_k_accumulate
                if F._use_wgrad_beta_accum()
                else F._run_cutlass_blockscaled_gemm_varlen_k_tma_add
            )
            gemm(
                a_col[0], a_col[1], b_col[0], b_col[1], expert_frequency_offset,
                M=M, N=N, total_K=total_K, num_experts=num_experts,
                device=device, accumulator=accumulator,
            )
            return None
        return F._run_cutlass_blockscaled_gemm_varlen_k(
            a_col[0], a_col[1], b_col[0], b_col[1], expert_frequency_offset,
            M=M, N=N, total_K=total_K, num_experts=num_experts,
            out_dtype=out_dtype, device=device,
        )

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
