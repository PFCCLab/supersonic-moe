# ********************************************************************************
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ********************************************************************************

"""SM100 (Blackwell) FP8 backend of ``sonicmoe.functional.fp8_kernels``.

CUTLASS blockscaled kernels with E4M3 data, ISA-packed UE8M0 1x32 activation
scales and per-row 1x32 weight scales; expert segments must be 128-aligned.
Exposes the same op names, signatures and return arity as ``sm90_funcs``.

Shared runtime state (``_PREQUANT_HIT_COUNT``, ``_STASHED_FP8_WEIGHTS``, the
alignment streak behind ``_all_segments_128_aligned``) and the activation-name
conversion point ``_gemm_activation_name`` stay in ``sonicmoe.functional``:
tests, ``moe.py`` and ``jit_warmup`` read and mutate them there.  This module
must not import that package at top level (``functional`` imports this module
while it is still initializing), so they are looked up through
``_functional()`` at call time.
"""

from __future__ import annotations

import importlib

import torch

from ..enums import ActivationType
from ..functional.fp8_config import _FP8Config, _get_fp8_config
from .blockscaled_fp8_gemm import (
    _SF_TILE_K,
    _SF_TILE_M,
    _SF_TILE_STORAGE,
    _SF_VEC_SIZE,
    _div_up,
    _gather_isa_packed_scales_kernel,
    _get_padding_plan,
    _run_cutlass_blockscaled_gemm,
    _run_cutlass_blockscaled_gemm_varlen_k,
    _run_cutlass_blockscaled_gemm_varlen_k_accumulate,
    _run_cutlass_blockscaled_gemm_varlen_k_tma_add,
    _storage_per_batch,
    blockscaled_fp8_gemm_varlen,
    colwise_quantize_and_pack,
    dequant_colwise_quantize_and_pack_from_isa,
    dual_quantize_varlen,
    fused_z_save_y1_quant,
    gather_raw_blockscaled_1x32_scales_to_isa,
    pack_blockscaled_1x32_scales_fast,
    precompute_weight_fp8,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    quantize_activation_blockscaled_fast,
    quantize_and_pack_activation,
)
from .fused_quant_kernels import fused_dual_colwise_quantize
from .gemm_dgated import gemm_dgated as gemm_dgated_kernel
from .gemm_gated import try_gemm_gated_tvm_ffi_sonic_prepared
# Aliased: this module defines ops with the same names (``gemm_gated`` /
# ``gemm_dgated``) that wrap these QuACK interfaces.
from .gemm_interface import gemm_dgated as quack_gemm_dgated
from .gemm_interface import gemm_gated as quack_gemm_gated
from .gemm_sm100_fp8_zeromat import (
    blockscaled_fp8_gemm_zeromat_bf16,
    blockscaled_fp8_gemm_zeromat_quant,
)
from .swiglu_triton import dequantize_blockscaled_fp8

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
_GATED_OUTPUT_PAYLOAD_ATTR = "_sonic_preallocated_gated_outputs"


def _functional():
    """``sonicmoe.functional`` (fully initialized by the time any op runs)."""
    return importlib.import_module("..functional", __package__)


# ---------------------------------------------------------------------------
# 1x32 scale layout helpers
# ---------------------------------------------------------------------------


def _is_raw_1x32_scale_layout(scales: torch.Tensor, rows: int, cols: int) -> bool:
    return scales.ndim == 2 and tuple(scales.shape) == (rows, _div_up(cols, _SF_VEC_SIZE))


def _is_compact_1x32_scale_carrier(
    scales: torch.Tensor, rows: int, cols: int
) -> bool:
    groups = _div_up(cols, _SF_VEC_SIZE)
    return (
        groups % 4 == 0
        and str(scales.dtype) in ("torch.int32", "paddle.int32", "int32")
        and scales.ndim == 2
        and tuple(scales.shape) == (rows, groups // 4)
    )


def _ensure_isa_1x32_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if _is_compact_1x32_scale_carrier(scales, rows, cols):
        scales = scales.view(torch.uint8).reshape(
            rows, _div_up(cols, _SF_VEC_SIZE)
        )
    if _is_raw_1x32_scale_layout(scales, rows, cols):
        return pack_blockscaled_1x32_scales_fast(scales, cols).view(_E8M0_DTYPE)
    return scales

def _gather_1x32_scales_to_isa(
    scales: torch.Tensor,
    gather_idx: torch.Tensor,
    rows: int,
    cols: int,
    *,
    fill_value: int = 127,
) -> torch.Tensor:
    if _is_compact_1x32_scale_carrier(scales, rows, cols):
        scales = scales.view(torch.uint8).reshape(
            rows, _div_up(cols, _SF_VEC_SIZE)
        )
    if _is_raw_1x32_scale_layout(scales, rows, cols):
        return gather_raw_blockscaled_1x32_scales_to_isa(
            scales, gather_idx, cols
        ).view(_E8M0_DTYPE)

    TK = gather_idx.shape[0]
    k_tiles = _div_up(cols, _SF_TILE_K)
    per_batch_tk = _storage_per_batch(TK, cols)
    out = (
        torch.empty((1, per_batch_tk), dtype=torch.uint8, device=scales.device)
        if (TK % _SF_TILE_M == 0 and cols % _SF_TILE_K == 0)
        else torch.full((1, per_batch_tk), fill_value, dtype=torch.uint8, device=scales.device)
    )
    block_rows = 128
    _gather_isa_packed_scales_kernel[(_div_up(TK, block_rows), k_tiles)](
        scales.view(torch.uint8), gather_idx, out, TK,
        src_k_tiles=k_tiles, dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M, SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=block_rows, GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    return out.view(_E8M0_DTYPE)

def _is_fp8_e4m3_dtype(dtype) -> bool:
    return dtype == torch.float8_e4m3fn or str(dtype) == "paddle.float8_e4m3fn"

def _raw_1x32_scale_bytes(
    scales: torch.Tensor, rows: int | None = None, cols: int | None = None
) -> torch.Tensor:
    if str(scales.dtype) in ("torch.uint8", "paddle.uint8", "uint8"):
        return scales
    if str(scales.dtype) in ("torch.int32", "paddle.int32", "int32"):
        if (
            rows is not None
            and cols is not None
            and _is_compact_1x32_scale_carrier(scales, rows, cols)
        ):
            return scales.view(torch.uint8).reshape(
                rows, _div_up(cols, _SF_VEC_SIZE)
            )
        return scales.to(torch.uint8)
    return scales.view(torch.uint8)

# ---------------------------------------------------------------------------
# Up-projection forward / z recompute
# ---------------------------------------------------------------------------


def _fused_blockscaled_gated_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    x_fp8_pre: tuple[torch.Tensor, ...] | None = None,
    w1_fp8_pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    store_z: bool = True,
    fp8_config: _FP8Config | None = None,
    current_stream=None,
    w1_pretransposed: bool = False,
    activation_type=ActivationType.SWIGLU,
) -> tuple:
    """Run blockscaled GEMM+SwiGLU with zero-materialization FP8.

    Returns ``(z, y1, y1_fp8, y1_scales, z_fp8_payload)``.  With epilogue
    quant + ``save_z_fp8``, ``z_fp8_payload`` is ``(z_fp8, z_scales_e8m0)``
    (the caller publishes it as ``_PREQUANTIZED_SCALES["z_fp8"]``); otherwise
    it is ``None``.

    Zero-materialization path (SonicMoE design principle):
    1. quantize_and_pack_activation(x) on T-sized tensor (~2-8µs)
    2. ISA-packed scale gather T->TK (~3-8µs, tiny I/O)
    3. Custom GemmGatedSm100ZeroMat kernel: T-FP8 + A_idx + TK-scales
    No TK-sized FP8 activation is materialized in HBM.

    Falls back to three-step pipeline if custom kernel fails.

    Parameters
    ----------
    w1_fp8_pre : optional pre-computed (w1_fp8, w1_scales) tuple.
        When provided, skips the global cache lookup (used in stash mode
        when the cache key may not match the modified parameter data_ptr).
    """

    if w1_fp8_pre is not None:
        w1_fp8, w1_scales = w1_fp8_pre
    else:
        raise RuntimeError("Sonic FP8 fused gated forward requires explicit w1_fused payload")

    # Step 1: Quantize at T-size (NOT TK)
    x_scales_tk_pre = None
    gated_output_payload = None
    if x_fp8_pre is not None:
        if len(x_fp8_pre) not in (2, 3, 7):
            raise ValueError(
                "prequant activation payload must contain 2, 3, or 7 tensors, "
                f"got {len(x_fp8_pre)}"
            )
        x_fp8, x_scales_t = x_fp8_pre[:2]
        if len(x_fp8_pre) >= 3:
            x_scales_tk_pre = x_fp8_pre[2]
        if len(x_fp8_pre) == 7:
            gated_output_payload = x_fp8_pre[3:7]
        elif x_scales_tk_pre is not None:
            gated_output_payload = getattr(
                x_scales_tk_pre, _GATED_OUTPUT_PAYLOAD_ATTR, None
            )
        _functional()._PREQUANT_HIT_COUNT["activation_fwd"] += 1
    else:
        x_fp8, x_scales_t = quantize_and_pack_activation(x)

    # Step 2: Gather scales T->TK in ISA layout (~3-8µs)
    TK = x_gather_idx.shape[0]
    K = x.shape[1]
    if x_scales_tk_pre is not None:
        x_scales_tk_e8m0 = (
            x_scales_tk_pre
            if x_scales_tk_pre.dtype == _E8M0_DTYPE
            else x_scales_tk_pre.view(_E8M0_DTYPE)
        )
    else:
        x_scales_tk_e8m0 = _gather_1x32_scales_to_isa(
            x_scales_t, x_gather_idx, int(x_fp8.shape[0]), K
        )
    del x_scales_t

    # Step 3: Zero-materialization GEMM via standard interface.
    # gemm_gated() with A_idx auto-selects GemmGatedSm100ZeroMat on SM100,
    # which gathers A rows inside the kernel (no TK FP8 materialization).
    # When epilogue quant is enabled, D output is fp8 directly (no bf16 round-trip).
    # The epilogue multiplies z by quant_scale in registers -> hardware fp8 saturating
    # cast writes z_fp8 to D. This eliminates the standalone z quant kernel
    # and halves D bandwidth (192MB fp8 vs 384MB bf16).
    cfg = fp8_config if fp8_config is not None else _get_fp8_config()
    epilogue_quant = cfg.epilogue_quant and cfg.save_z_fp8
    # ActivationType -> GEMM string (SiTU betas baked into the key here).
    gemm_activation = _functional()._gemm_activation_name(
        activation_type, cfg, fp8=True, site="_fused_blockscaled_gated_forward"
    )
    N = w1.shape[1]  # pretransposed weight is (E, 2I, H)
    if epilogue_quant:
        z_scale_out = (
            gated_output_payload[2]
            if gated_output_payload is not None
            # Paddle's Tensor.numel() returns a device scalar; use shape metadata
            # here so output selection cannot synchronize the launch stream.
            and all(dim != 0 for dim in gated_output_payload[2].shape)
            else torch.empty(TK, N // 32, dtype=torch.uint8, device=x.device)
        )
    else:
        z_scale_out = None

    # Fused y1 (postact) FP8 quant in the up-proj epilogue.  When z epilogue
    # quant (save_z_fp8) is ALSO on, both run in a single combined pass
    # (GemmGatedSm100ZeroMatQuantPostActQuant): z -> fp8 via mZScale AND y1 ->
    # fp8 via mPostActScaleIsa.  This is the precision-preferred combination —
    # z stays fp8 (save_z_fp8 path) so the backward takes the HEALTHY fp8-preact
    # dgated path, while y1 is fused (no standalone quant kernel, no bf16 y1 HBM
    # round-trip).  fuse_y1 with z bf16 (save_z_fp8 off) is the legacy y1-only
    # path; both are supported.
    fuse_y1 = cfg.fuse_y1_quant
    if fuse_y1:
        I_dim = w1.shape[1] // 2  # w1 is (2I, H, E) -> N=2I, postact y1 has I cols
        assert TK % 128 == 0 and I_dim % 128 == 0, (
            f"fuse_y1_quant requires TK ({TK}) and I ({I_dim}) be multiples of 128 "
            "for the ISA-packed scale layout"
        )
        postact_scale_out = torch.empty(
            (TK // 128, I_dim // 128, 512), dtype=torch.uint8, device=x.device
        ) if gated_output_payload is None else gated_output_payload[3]
        postact_dtype = torch.float8_e4m3fn
    else:
        postact_scale_out = None
        postact_dtype = torch.bfloat16

    # CUTLASS fp8 D output: writes z directly as fp8, epilogue computes
    # blockscaled e8m0 scales in registers.  Eliminates standalone z quant
    # kernel (~141µs) and halves D write bandwidth (192MB fp8 vs 384MB bf16).
    # The fp8 z is stored ONLY in the prequant cache — the autograd graph
    # sees a lightweight bf16 placeholder (storage freed) to avoid fp8-dtype
    # tensors in the autograd chain which cause illegal memory access in
    # backward at large shapes.
    z_out_dtype = torch.float8_e4m3fn if epilogue_quant else torch.bfloat16
    preact_out = None
    postact_out = None
    if gated_output_payload is not None:
        if not fuse_y1:
            raise ValueError(
                "preallocated gated outputs require fused y1 epilogue quantization"
            )
        preact_out, postact_out = gated_output_payload[:2]
        expected = (
            (preact_out, (TK, N), z_out_dtype, "preact"),
            (postact_out, (TK, I_dim), postact_dtype, "postact"),
            (
                gated_output_payload[2],
                (TK, N // 32) if epilogue_quant else (0,),
                torch.uint8,
                "z_scale",
            ),
            (
                postact_scale_out,
                (TK // 128, I_dim // 128, 512),
                torch.uint8,
                "postact_scale",
            ),
        )
        for tensor, shape, dtype, name in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(
                    f"invalid preallocated {name}: shape={tuple(tensor.shape)}, "
                    f"dtype={tensor.dtype}; expected shape={shape}, dtype={dtype}"
                )

    launched_prepared = (
        w1_pretransposed
        and preact_out is not None
        and postact_out is not None
        and try_gemm_gated_tvm_ffi_sonic_prepared(
            x_fp8,
            w1_fp8,
            preact_out,
            postact_out,
            activation=gemm_activation,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0,
            b_scales=w1_scales,
            z_scale_out=z_scale_out,
            postact_scale_out=postact_scale_out,
            swiglu_clamp_value=cfg.swiglu_clamp_value,
            postact_bf16_trunc=cfg.fuse_y1_bf16_trunc,
            current_stream=current_stream,
        )
    )
    if launched_prepared:
        z, y1 = preact_out, postact_out
    else:
        # The first call validates and compiles through the generic interface;
        # subsequent calls reuse only its pointer-stable launch plan.
        z, y1 = quack_gemm_gated(
            x_fp8,
            w1_fp8,
            activation=gemm_activation,
            preact_out=preact_out,
            postact_out=postact_out,
            out_dtype=z_out_dtype,
            postact_dtype=postact_dtype,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            a_scales=x_scales_tk_e8m0,
            b_scales=w1_scales,
            store_preact=True,
            dynamic_scheduler=False,
            tuned=False,
            z_scale_out=z_scale_out,
            postact_scale_out=postact_scale_out,
            swiglu_clamp_value=cfg.swiglu_clamp_value,
            postact_bf16_trunc=cfg.fuse_y1_bf16_trunc,
            current_stream=current_stream,
            b_pretransposed=w1_pretransposed,
        )
    del x_fp8, x_scales_tk_e8m0

    z_fp8_payload = None
    if epilogue_quant:
        # z is fp8 from CUTLASS.  Hand it back for the prequant cache (the
        # caller stores it for backward), then replace it with a lightweight
        # bf16 placeholder for autograd.
        z_fp8 = z
        z_scales = z_scale_out.view(_E8M0_DTYPE)
        z_fp8_payload = (z_fp8, z_scales)
        # Lightweight bf16 placeholder: 2 bytes of storage, broadcast to (TK, 2I)
        # via zero strides.  autograd only needs the tensor as a graph node;
        # _DownProjection.forward reads z.device/z.dtype for metadata and gets
        # actual fp8 data from the prequant cache.  This avoids a 384 MiB
        # momentary peak from allocating a full-size bf16 tensor.
        z = torch.empty(1, dtype=torch.bfloat16, device=z_fp8.device).as_strided(
            z_fp8.shape, (0, 0)
        )
    elif not store_z:
        z = torch.empty(1, dtype=torch.bfloat16, device=y1.device).as_strided(
            (TK, w1.shape[1]), (0, 0)
        )

    y1_fp8_fused = None
    y1_scales_fused = None
    if fuse_y1:
        # y1 is already FP8 from the epilogue; scales are ISA-packed.  Hand them
        # back so the caller skips the standalone quantize_and_pack_activation(y1).
        y1_fp8_fused = y1
        y1_scales_fused = postact_scale_out.reshape(1, -1).view(_E8M0_DTYPE)
        y1 = torch.empty(1, dtype=torch.bfloat16, device=y1_fp8_fused.device).as_strided(
            (TK, w1.shape[1] // 2), (0, 0)
        )

    return z, y1, y1_fp8_fused, y1_scales_fused, z_fp8_payload


def _recompute_z_fp8(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    x_fp8_pre: torch.Tensor | None = None,
    x_scales_pre: torch.Tensor | None = None,
    w1_fp8_pre: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-run up-proj to materialize only z_fp8 + scales for backward."""
    if w1_fp8_pre is None:
        raise RuntimeError("Sonic FP8 recompute requires explicit w1_fused payload")
    w1_fp8, w1_scales = w1_fp8_pre

    if x_fp8_pre is not None or x_scales_pre is not None:
        if x_fp8_pre is None or x_scales_pre is None:
            raise ValueError("x_fp8_pre and x_scales_pre must be provided together")
        x_fp8 = x_fp8_pre
        x_scales_t = x_scales_pre
        _functional()._PREQUANT_HIT_COUNT["activation_recompute"] += 1
    else:
        if x is None:
            raise ValueError("BF16 x is required when no prequant activation payload is provided")
        x_fp8, x_scales_t = quantize_and_pack_activation(x)
    TK = x_gather_idx.shape[0]
    K = x_fp8.shape[1]
    x_scales_tk_e8m0 = _gather_1x32_scales_to_isa(
        x_scales_t, x_gather_idx, int(x_fp8.shape[0]), K
    )
    del x_scales_t

    z_fp8, z_raw_scales = blockscaled_fp8_gemm_zeromat_quant(
        x_fp8,
        w1_fp8,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
    )
    return z_fp8, z_raw_scales.view(_E8M0_DTYPE)


def _recompute_z_bf16(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    x_fp8_pre: torch.Tensor | None = None,
    x_scales_pre: torch.Tensor | None = None,
    w1_fp8_pre: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Re-run up-proj to materialize z in BF16 for backward (no z quant).

    Precision-preserving counterpart of ``_recompute_z_fp8``: identical FP8 GEMM
    accumulation (``blockscaled_fp8_gemm_zeromat_bf16``) but a BF16 D output, so z
    is never quantized.  Used as the default recompute path when ``save_z_fp8`` is
    disabled (precision first).
    """

    if w1_fp8_pre is None:
        raise RuntimeError("Sonic FP8 recompute requires explicit w1_fused payload")
    w1_fp8, w1_scales = w1_fp8_pre

    if x_fp8_pre is not None or x_scales_pre is not None:
        if x_fp8_pre is None or x_scales_pre is None:
            raise ValueError("x_fp8_pre and x_scales_pre must be provided together")
        x_fp8 = x_fp8_pre
        x_scales_t = x_scales_pre
        _functional()._PREQUANT_HIT_COUNT["activation_recompute"] += 1
    else:
        if x is None:
            raise ValueError("BF16 x is required when no prequant activation payload is provided")
        x_fp8, x_scales_t = quantize_and_pack_activation(x)
    TK = x_gather_idx.shape[0]
    K = x_fp8.shape[1]
    x_scales_tk_e8m0 = _gather_1x32_scales_to_isa(
        x_scales_t, x_gather_idx, int(x_fp8.shape[0]), K
    )
    del x_scales_t

    z = blockscaled_fp8_gemm_zeromat_bf16(
        x_fp8,
        w1_fp8,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
    )
    return z


def _padded_blockscaled_gated_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    activation_type=ActivationType.SWIGLU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 up-proj with padding for non-128-aligned expert segments.

    Pads expert segment boundaries to 128, runs the zero-mat FP8 GEMM+SwiGLU,
    then unpads the results.  Avoids the full BF16 fallback while keeping the
    same E8M0 quantization as the aligned path.

    Padding overhead is ~5-25% extra GEMM rows (typical for MoE routing).
    """
    TK = x_gather_idx.shape[0]
    needs_pad, padded_cu, padded_total, dst_idx = _get_padding_plan(
        expert_frequency_offset, TK
    )
    if not needs_pad:
        z, y1, _y1_fp8, _y1_sc, _z_fp8 = _fused_blockscaled_gated_forward(
            x, w1, expert_frequency_offset, x_gather_idx,
            activation_type=activation_type,
        )
        return z, y1

    # Step 1: Quantize at T-size (same as aligned path — no padding here)
    x_fp8, x_scales_t = quantize_and_pack_activation(x)

    # Step 2: Pad gather indices (padding rows -> row 0, safe arbitrary data)
    padded_gather_idx = torch.zeros(
        padded_total, dtype=x_gather_idx.dtype, device=x_gather_idx.device
    )
    padded_gather_idx[dst_idx] = x_gather_idx

    # Step 3: Gather ISA-packed scales T->TK_padded
    K = x.shape[1]
    k_tiles = _div_up(K, _SF_TILE_K)
    per_batch_tk = _storage_per_batch(padded_total, K)
    # padded_total is 128-aligned by construction -> torch.empty is safe
    x_scales_tk = torch.empty(
        (1, per_batch_tk), dtype=torch.uint8, device=x.device
    )
    BLOCK_ROWS = 128
    _gather_isa_packed_scales_kernel[
        (_div_up(padded_total, BLOCK_ROWS), k_tiles)
    ](
        x_scales_t.view(torch.uint8),
        padded_gather_idx,
        x_scales_tk,
        padded_total,
        src_k_tiles=k_tiles,
        dst_k_tiles=k_tiles,
        SF_TILE_M=_SF_TILE_M,
        SF_TILE_STORAGE=_SF_TILE_STORAGE,
        BLOCK_ROWS=BLOCK_ROWS,
        GROUPS_PER_K_TILE=_SF_TILE_K // _SF_VEC_SIZE,
    )
    x_scales_tk_e8m0 = x_scales_tk.view(_E8M0_DTYPE)
    del x_scales_t

    # Step 4: Weight FP8 (cached — same as aligned path)
    w1_fp8, w1_scales = (
        _functional()._STASHED_FP8_WEIGHTS.get("w1_fused", None)
        or precompute_weight_fp8_for_fused_gated(w1)
    )

    # Step 5: Zero-mat GEMM+gated activation with padded 128-aligned boundaries
    _padded_cfg = _get_fp8_config()
    z_padded, y1_padded = quack_gemm_gated(
        x_fp8,
        w1_fp8,
        activation=_functional()._gemm_activation_name(
            activation_type, _padded_cfg, fp8=True,
            site="_padded_blockscaled_gated_forward",
        ),
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=padded_cu,
        A_idx=padded_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
        dynamic_scheduler=False,
        tuned=False,
        swiglu_clamp_value=_padded_cfg.swiglu_clamp_value,
    )
    del x_fp8, x_scales_tk_e8m0, padded_gather_idx

    # Step 6: Unpad results — discard padding rows
    z = z_padded[dst_idx]
    y1 = y1_padded[dst_idx]
    del z_padded, y1_padded

    return z, y1


def _can_reuse_bf16_preact_for_dz(z: torch.Tensor | None, total_m: int, n: int, device) -> bool:
    if z is None:
        return False
    if str(z.dtype) not in {"torch.bfloat16", "paddle.bfloat16", "bfloat16"}:
        return False
    if z.device != device:
        return False
    if tuple(z.shape) != (int(total_m), int(n) * 2):
        return False
    return z.is_contiguous() and z.stride(-1) == 1


# ---------------------------------------------------------------------------
# Config / layout
# ---------------------------------------------------------------------------


def validate_config(cfg, device) -> None:
    # Every FP8 option of the gated flow runs on SM100: nothing to reject.
    return None


def segments_aligned(expert_frequency_offset: torch.Tensor) -> bool:
    # ISA-packed scales need 128-row expert segments.  The check (and its
    # alignment-streak state) lives in functional.
    return _functional()._all_segments_128_aligned(expert_frequency_offset)


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


def quantize_activation(x: torch.Tensor, cfg):
    return quantize_and_pack_activation(x)


def quantize_activation_raw(x: torch.Tensor, cfg):
    return quantize_activation_blockscaled_fast(x)


def dequantize_activation(data: torch.Tensor, scales: torch.Tensor, cfg) -> torch.Tensor:
    return dequantize_blockscaled_fp8(
        data,
        _raw_1x32_scale_bytes(scales, int(data.shape[0]), int(data.shape[1])),
    )


def quantize_preact(z: torch.Tensor):
    return quantize_activation_blockscaled_fast(z)


def dequantize_preact(z_fp8: torch.Tensor, z_scales_u8: torch.Tensor) -> torch.Tensor:
    return dequantize_blockscaled_fp8(z_fp8, z_scales_u8)


def quantize_z_y1(z: torch.Tensor, y1: torch.Tensor, cfg):
    return fused_z_save_y1_quant(z, y1)


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


def precompute_weight_dgated(w2: torch.Tensor):
    return precompute_weight_fp8_for_direct_fused_dgated(w2)


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
    return _fused_blockscaled_gated_forward(
        x, w1, expert_frequency_offset, x_gather_idx,
        x_fp8_pre=x_fp8_pre,
        w1_fp8_pre=w1_fp8, store_z=not cfg.recompute_z,
        fp8_config=cfg,
        current_stream=current_stream,
        w1_pretransposed=True,
        activation_type=activation_type,
    )


def gemm_gated_padded(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    activation_type,
):
    return _padded_blockscaled_gated_forward(
        x, w1, expert_frequency_offset, x_gather_idx,
        activation_type=activation_type,
    )


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
    fn = _recompute_z_fp8 if fp8 else _recompute_z_bf16
    return fn(
        x, w1, expert_frequency_offset, x_gather_idx,
        x_fp8_pre, x_scales_pre, w1_fp8_pre,
    )


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
    # Same launch as the pre-quantized + assume_aligned hot path of
    # ``blockscaled_fp8_gemm_varlen``.
    return _run_cutlass_blockscaled_gemm(
        a_fp8, a_scales,
        w_fp8, w_scales,
        expert_frequency_offset,
        total_M=a_fp8.shape[0],
        K=a_fp8.shape[1],
        H=out_features,
        num_experts=num_experts,
        out_dtype=torch.bfloat16,
        device=a_fp8.device,
    )


def gemm_padded(x: torch.Tensor, w2: torch.Tensor, expert_frequency_offset: torch.Tensor):
    # Non-128-aligned segments: blockscaled_fp8_gemm_varlen pads internally.
    w2_fp8, w2_scales = (
        _functional()._STASHED_FP8_WEIGHTS.get("w2_varlen", None)
        or precompute_weight_fp8(w2)
    )
    return blockscaled_fp8_gemm_varlen(
        x, w2, expert_frequency_offset,
        w_fp8=w2_fp8, w_scales=w2_scales,
        out_dtype=torch.bfloat16,
        assume_aligned=False,
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
):
    device = dout_fp8.device
    total_m = int(x_gather_idx.shape[0])
    n = int(w2_fp8.shape[-2])
    y1s = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
    config = quack_gemm_dgated.default_config(device, num_experts=w2_fp8.shape[0])
    dz = (
        preact
        if (
            preact is not None
            and _can_reuse_bf16_preact_for_dz(preact, total_m, n, device)
        )
        else torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=device)
    )
    colvec_reduce_partial = torch.empty(
        (total_m, (n + config.tile_n - 1) // config.tile_n),
        dtype=torch.float32,
        device=device,
    )
    a_scales = _gather_1x32_scales_to_isa(
        dout_scales, x_gather_idx, int(dout_fp8.shape[0]), int(dout_fp8.shape[1]),
    )
    gemm_dgated_kernel(
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


# ---------------------------------------------------------------------------
# FP8 weight gradient. Wgrad operand: ``(fp8, isa_scales, None)``; the third
# slot (SM90 packed offsets) is unused here.
# ---------------------------------------------------------------------------


def quantize_wgrad_operand(
    src: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    cfg,
    *,
    gather_idx: torch.Tensor | None = None,
    src_scales: torch.Tensor | None = None,
):
    feature = int(src.shape[1])
    total_k = int(gather_idx.shape[0]) if gather_idx is not None else int(src.shape[0])
    if src_scales is None:
        col_fp8, col_scales = colwise_quantize_and_pack(
            src, logical_rows=feature, logical_cols=total_k, gather_idx=gather_idx,
        )
    else:
        col_fp8, col_scales = dequant_colwise_quantize_and_pack_from_isa(
            src,
            _ensure_isa_1x32_scales(src_scales, int(src.shape[0]), feature),
            logical_rows=feature, logical_cols=total_k,
            gather_idx=gather_idx,
        )
    return col_fp8, col_scales, None


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
    total_k = int(x_gather_idx.shape[0])
    if dout_fp8 is not None:
        dz_fp8, dz_scales, dz_col_fp8, dz_col_scales = dual_quantize_varlen(
            dz, total_k, dz.shape[1]
        )
        dout_col = quantize_wgrad_operand(
            dout_fp8, expert_frequency_offset, cfg,
            gather_idx=x_gather_idx, src_scales=dout_scales,
        )
    else:
        dz_fp8, dz_scales, dz_col_fp8, dz_col_scales, dout_col_fp8, dout_col_scales = (
            fused_dual_colwise_quantize(
                dz, dout, x_gather_idx, total_k, dz.shape[1], dout.shape[1],
            )
        )
        dout_col = (dout_col_fp8, dout_col_scales, None)
    return (dz_fp8, dz_scales), (dz_col_fp8, dz_col_scales, None), dout_col


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
    if accumulator is not None:
        run = (
            _run_cutlass_blockscaled_gemm_varlen_k_accumulate
            if _functional()._use_wgrad_beta_accum()
            else _run_cutlass_blockscaled_gemm_varlen_k_tma_add
        )
        run(
            a_col[0], a_col[1], b_col[0], b_col[1], expert_frequency_offset,
            M=M, N=N, total_K=total_K, num_experts=num_experts,
            device=device, accumulator=accumulator,
        )
        return None
    return _run_cutlass_blockscaled_gemm_varlen_k(
        a_col[0], a_col[1], b_col[0], b_col[1], expert_frequency_offset,
        M=M, N=N, total_K=total_K, num_experts=num_experts,
        out_dtype=out_dtype, device=device,
    )
