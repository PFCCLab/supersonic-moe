# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import collections
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ..config import get_active_config
from ..count_cumsum import count_cumsum
from ..enums import ActivationType, is_glu
from ..quack_utils import (
    bf16_wgrad_gemm_varlen_k,
    bf16_wgrad_gemm_varlen_k_accumulate,
    bf16_wgrad_gemm_varlen_k_tma_add,
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    blockscaled_fp8_gemm_varlen,
    clear_blockscaled_fp8_weight_cache,
    clear_raw_weight_cache,
    clear_sgl_weight_cache,
    fast_gather_quantize_and_pack_activation,
    gemm_dgated,
    gemm_gated,
    make_blockscaled_grouped_reverse_scatter_idx,
    precompute_weight_fp8,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    quantize_and_pack_activation,
)
from quack.gemm_interface import gemm
from ..quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
from ..quack_utils.fp8_quack_patch import apply_fp8_quack_patch

apply_fp8_quack_patch()


from .backward import (
    _softmax_topk_bwd,
    _token_broadcast_backward,
)
from .fp8_protocol import (
    FP8ActivationDType,
    FP8Backend,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    get_default_fp8_protocol,
    is_blackwell_device,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)
try:
    from .fp8_cutely_fused import apply_activation_fp8_protocol_cutely_fused
    from .fp8_cutely_fused import apply_preact_activation_fp8_protocol_cutely_fused
except ImportError:
    apply_activation_fp8_protocol_cutely_fused = None
    apply_preact_activation_fp8_protocol_cutely_fused = None
from .fp8_reference import (
    FP8Tensor,
    apply_activation_fp8_protocol,
    dequantize_activation_reference,
    quantize_activation_reference,
)
from .forward import _router_forward, _softmax_topk_fwd
from .triton_kernels import TC_topk_router_metadata_triton
from .utils import enable_fp8, enable_quack_gemm, is_fp8_active, is_using_quack_gemm

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)


def _is_raw_1x32_scale_layout(scales: torch.Tensor, rows: int, cols: int) -> bool:
    return scales.ndim == 2 and tuple(scales.shape) == (rows, _div_up(cols, _SF_VEC_SIZE))


def _raw_1x32_scale_bytes(scales: torch.Tensor) -> torch.Tensor:
    if str(scales.dtype) in ("torch.uint8", "paddle.uint8", "uint8"):
        return scales
    if str(scales.dtype) in ("torch.int32", "paddle.int32", "int32"):
        return scales.to(torch.uint8)
    return scales.view(torch.uint8)


def _ensure_isa_1x32_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
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


@triton.jit
def _gather_router_scores_i32_kernel(src, idx, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    src_idx = tl.load(idx + offs, mask=mask, other=0)
    vals = tl.load(src + src_idx, mask=mask, other=0.0)
    tl.store(dst + offs, vals, mask=mask)


@triton.jit
def _scatter_router_scores_i32_kernel(src, idx, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dst_idx = tl.load(idx + offs, mask=mask, other=0)
    vals = tl.load(src + offs, mask=mask, other=0.0)
    tl.store(dst + dst_idx, vals, mask=mask)


def _gather_router_scores_i32(scores: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    n = int(idx.shape[0])
    out = scores.new_empty((n,))
    if n > 0:
        block = 256
        _gather_router_scores_i32_kernel[(triton.cdiv(n, block),)](
            scores, idx, out, n, BLOCK=block,
        )
    return out


def _scatter_router_scores_i32(scores: torch.Tensor, idx: torch.Tensor, n_total: int) -> torch.Tensor:
    out = scores.new_empty((n_total,))
    n = int(idx.shape[0])
    if n < n_total:
        out.zero_()
    if n > 0:
        block = 256
        _scatter_router_scores_i32_kernel[(triton.cdiv(n, block),)](
            scores, idx, out, n, BLOCK=block,
        )
    return out


# ---------------------------------------------------------------------------
# Standalone SwiGLU forward/backward (for blockscaled split path)
# ---------------------------------------------------------------------------
# SonicMoE stores w1 interleaved: [gate_row0, up_row0, gate_row1, ...].
# The GEMM output z thus has interleaved layout: columns 0,2,4,...=gate,
# columns 1,3,5,...=up.

from ..quack_utils.swiglu_triton import dequantize_blockscaled_fp8
try:
    from ..quack_utils.swiglu_triton import (
        swiglu_backward_quant_pack_triton,
        swiglu_forward_quant_pack_zsave_triton,
    )
except ImportError:
    swiglu_backward_quant_pack_triton = None
    swiglu_forward_quant_pack_zsave_triton = None
from ..quack_utils.blockscaled_fp8_gemm import (
    _FUSED_WEIGHT_CACHE,
    _VARLEN_WEIGHT_CACHE,
    _div_up,
    _gather_isa_packed_scales_kernel,
    _get_padding_plan,
    _run_cutlass_blockscaled_gemm,
    _run_cutlass_blockscaled_gemm_varlen_k,
    _run_cutlass_blockscaled_gemm_varlen_k_accumulate,
    _run_cutlass_blockscaled_gemm_varlen_k_tma_add,
    _SF_TILE_K,
    _SF_TILE_M,
    _SF_TILE_STORAGE,
    _SF_VEC_SIZE,
    _storage_per_batch,
    colwise_quantize_and_pack,
    dequant_colwise_quantize_and_pack_from_isa,
    gather_raw_blockscaled_1x32_scales_to_isa,
    pack_blockscaled_1x32_scales,
    pack_blockscaled_1x32_scales_fast,
    dual_quantize_varlen,
    fused_z_save_y1_quant,
    iso32_dual_quantize_varlen,
    precompute_weight_fp8_for_fused_gated,
    quantize_activation_blockscaled_fast,
    quantize_and_pack_activation,
)
from ..quack_utils.fused_quant_kernels import fused_dual_colwise_quantize
from ..quack_utils.gemm_sm100_fp8_zeromat import (
    blockscaled_fp8_gemm_zeromat_bf16,
    blockscaled_fp8_gemm_zeromat_quant,
)


def _swiglu_forward_interleaved(z: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU on interleaved pre-activation z(TK, 2I) -> y1(TK, I)."""
    return swiglu_forward_triton(z)


def _swiglu_backward_interleaved(
    dy1: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward SwiGLU + router score weighting on interleaved layout."""
    return swiglu_backward_triton(dy1, z, s)


def _swiglu_forward_clamp_reference(z: torch.Tensor, clamp_value: float) -> torch.Tensor:
    gate = z[..., 0::2].float().clamp(max=clamp_value)
    up = z[..., 1::2].float().clamp(min=-clamp_value, max=clamp_value)
    return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


def _swiglu_backward_clamp_reference(
    dy1: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
    clamp_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = z[..., 0::2].float()
    up = z[..., 1::2].float()
    gate_c = gate.clamp(max=clamp_value)
    up_c = up.clamp(min=-clamp_value, max=clamp_value)
    sig = torch.sigmoid(gate_c)
    silu = gate_c * sig
    y1s = (silu * up_c * s.unsqueeze(-1).float()).to(torch.bfloat16)
    dy = dy1.float() * s.unsqueeze(-1).float()
    dgate = dy * up_c * sig * (1.0 + gate_c * (1.0 - sig))
    dup = dy * silu
    dgate = torch.where(gate <= clamp_value, dgate, torch.zeros_like(dgate))
    dup = torch.where(up.abs() <= clamp_value, dup, torch.zeros_like(dup))
    dz = torch.empty_like(z, dtype=torch.bfloat16)
    dz[..., 0::2] = dgate.to(torch.bfloat16)
    dz[..., 1::2] = dup.to(torch.bfloat16)
    ds = (dy1.float() * (silu * up_c)).sum(dim=-1)
    return dz, y1s, ds


def _fused_blockscaled_gated_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    *,
    w1_fp8_pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    x_fp8_pre: torch.Tensor | None = None,
    x_scales_pre: torch.Tensor | None = None,
    store_z: bool = True,
    fuse_y1_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run blockscaled GEMM+SwiGLU with zero-materialization FP8.

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
    if w1_fp8_pre is None:
        raise RuntimeError("Sonic FP8 fused gated forward requires explicit w1_fused payload")
    w1_fp8, w1_scales = w1_fp8_pre

    # Step 1: Quantize at T-size (NOT TK)
    if x_fp8_pre is not None or x_scales_pre is not None:
        if x_fp8_pre is None or x_scales_pre is None:
            raise ValueError("x_fp8_pre and x_scales_pre must be provided together")
        x_fp8 = x_fp8_pre
        x_scales_t = x_scales_pre
        _PREQUANT_HIT_COUNT["activation_fwd"] += 1
    else:
        x_fp8, x_scales_t = quantize_and_pack_activation(x)

    # Step 2: Gather scales T->TK in ISA layout (~3-8µs)
    TK = x_gather_idx.shape[0]
    K = x.shape[1]
    x_scales_tk_e8m0 = _gather_1x32_scales_to_isa(
        x_scales_t, x_gather_idx, int(x_fp8.shape[0]), K
    )
    del x_scales_t

    # Step 3: Zero-materialization GEMM via standard interface.
    # gemm_gated() with A_idx auto-selects GemmGatedSm100ZeroMat on SM100,
    # which gathers A rows inside the kernel (no TK FP8 materialization).
    # When epilogue quant is enabled, D output is fp8 directly (no bf16 round-trip).
    # In recompute_z mode, store_z=False skips the preact/D store entirely; y1 is
    # still produced for DownProjection, and z_fp8 is materialized later in backward.
    cfg = _get_fp8_config()
    epilogue_quant = store_z and cfg.epilogue_quant and cfg.save_z_fp8
    if epilogue_quant:
        N = w1.shape[0]  # (2I, H, E) -> w1.shape[0] = 2I
        z_scale_out = torch.empty(TK, N // 32, dtype=torch.uint8, device=x.device)
    else:
        z_scale_out = None

    # y1 postact-quant fusion: write y1 (=SwiGLU(z)) directly as FP8 + ISA scales
    # in the up-proj epilogue (GemmGatedSm100ZeroMatPostActQuant).  Mutually
    # exclusive with z epilogue quant.  Precision-first: z stays bf16.
    fuse_y1 = fuse_y1_quant and not epilogue_quant
    if fuse_y1_quant and epilogue_quant:
        raise RuntimeError(
            "SONIC_MOE_FUSE_Y1_QUANT (y1 postact epilogue quant) is mutually "
            "exclusive with z epilogue quant (save_z_fp8); disable one"
        )
    if fuse_y1:
        I_dim = w1.shape[0] // 2  # w1 is (2I, H, E) -> N=2I, postact y1 has I cols
        assert TK % 128 == 0 and I_dim % 128 == 0, (
            f"fuse_y1_quant requires TK ({TK}) and I ({I_dim}) be multiples of 128 "
            "for the ISA-packed scale layout"
        )
        postact_scale_out = torch.empty(
            (TK // 128, I_dim // 128, 512), dtype=torch.uint8, device=x.device
        )
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

    z, y1 = gemm_gated(
        x_fp8, w1_fp8,
        activation="swiglu",
        out_dtype=z_out_dtype,
        postact_dtype=postact_dtype,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
        store_preact=store_z,
        dynamic_scheduler=False,
        tuned=False,
        z_scale_out=z_scale_out,
        postact_scale_out=postact_scale_out,
        swiglu_clamp_value=cfg.swiglu_clamp_value,
    )
    del x_fp8, x_scales_tk_e8m0

    if epilogue_quant:
        # z is fp8 from CUTLASS.  Store in prequant cache for backward,
        # then replace with a lightweight bf16 placeholder for autograd.
        z_fp8 = z
        z_scales = z_scale_out.view(_E8M0_DTYPE)
        _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_scales)
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
            (TK, w1.shape[0]), (0, 0)
        )

    # y1 fusion: y1 is FP8 from the epilogue.  Hand back the fp8 data + ISA
    # scales explicitly, replace the graph node with a lightweight bf16
    # placeholder (freed storage); the bf16 y1 values are consumed ONLY by the
    # (now-eliminated) standalone quant — DownProjection backward recomputes y1
    # from z, never reading these values.
    y1_fp8_fused = None
    y1_scales_fused = None
    if fuse_y1:
        y1_fp8_fused = y1
        y1_scales_fused = postact_scale_out.reshape(1, -1).view(_E8M0_DTYPE)
        y1 = torch.empty(1, dtype=torch.bfloat16, device=y1_fp8_fused.device).as_strided(
            (TK, w1.shape[0] // 2), (0, 0)
        )

    return z, y1, y1_fp8_fused, y1_scales_fused


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
        _PREQUANT_HIT_COUNT["activation_recompute"] += 1
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
        w1_fp8.mT,
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
        _PREQUANT_HIT_COUNT["activation_recompute"] += 1
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
        w1_fp8.mT,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
    )
    return z


# ---------------------------------------------------------------------------
# Route-level padding: pad routing metadata once so the alignment check
# sees 128-aligned expert_frequency_offset → entire fwd+bwd runs the proven
# aligned fast path.  Zero GEMM code changes.
# ---------------------------------------------------------------------------

def _pad_routing_metadata(
    expert_frequency_offset: torch.Tensor,  # (E+1,) int32
    x_gather_idx: torch.Tensor,             # (TK,) int32
    s_scatter_idx: torch.Tensor,            # (TK,) int32
    s_reverse_scatter_idx: torch.Tensor,    # (TK,) int32
    topk_scores: torch.Tensor,              # (T*K,) float32 (already flattened)
    TK: int, T: int, E: int, K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    """Pad routing metadata to ensure 128-aligned expert segments for FP8.

    Padding rows use gather index 0 (arbitrary valid row — data doesn't matter
    because score=0 nullifies the contribution) and score=0, so they contribute
    nothing to output or gradients.  No sentinel row is appended to x.

    Returns:
        (padded_efo, padded_x_gather, padded_s_scatter,
         padded_s_reverse, padded_scores, padded_total, was_padded)
    """
    needs_pad, padded_cu, padded_total, dst_idx = _get_padding_plan(
        expert_frequency_offset, TK
    )
    if not needs_pad:
        return (expert_frequency_offset, x_gather_idx, s_scatter_idx,
                s_reverse_scatter_idx, topk_scores, TK, False)

    N_pad = padded_total - TK
    device = x_gather_idx.device

    # 1. expert_frequency_offset — directly from _get_padding_plan
    padded_efo = padded_cu

    # 2. x_gather_idx: padding positions → row 0 (arbitrary safe row;
    #    score=0 nullifies the contribution regardless of data)
    padded_x_gather = torch.zeros(
        padded_total, dtype=x_gather_idx.dtype, device=device
    )
    padded_x_gather[dst_idx] = x_gather_idx

    # 3. topk_scores: append zeros for padding positions
    padded_scores = torch.cat([
        topk_scores,
        torch.zeros(N_pad, dtype=topk_scores.dtype, device=device),
    ])

    # 4. s_scatter_idx: real tokens remapped to padded positions,
    #    padding positions → virtual flat-topk indices T*K .. T*K+N_pad-1
    padded_s_scatter = torch.empty(
        padded_total, dtype=s_scatter_idx.dtype, device=device
    )
    padded_s_scatter[dst_idx] = s_scatter_idx
    # Compute pad positions (positions in [0, padded_total) NOT in dst_idx)
    is_real = torch.zeros(padded_total, dtype=torch.bool, device=device)
    is_real[dst_idx] = True
    pad_positions = torch.where(~is_real)[0]
    # Padding positions get virtual scatter indices beyond T*K
    padded_s_scatter[pad_positions] = torch.arange(
        TK, TK + N_pad, dtype=s_scatter_idx.dtype, device=device
    )

    # 5. s_reverse_scatter_idx: stays (T*K,) — only real tokens need reverse mapping.
    #    Value remapping: original values pointed into [0, TK), now must point
    #    into padded positions via dst_idx[original_value].
    padded_s_reverse = dst_idx[s_reverse_scatter_idx.long()].to(
        s_reverse_scatter_idx.dtype
    )

    return (padded_efo, padded_x_gather, padded_s_scatter,
            padded_s_reverse, padded_scores, padded_total, True)


def _padded_blockscaled_gated_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    w1_fp8_pre: tuple[torch.Tensor, torch.Tensor],
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
        z, y1, _, _ = _fused_blockscaled_gated_forward(
            x, w1, expert_frequency_offset, x_gather_idx,
            w1_fp8_pre=w1_fp8_pre,
        )
        return z, y1

    # Step 1: Quantize at T-size (same as aligned path — no padding here)
    x_fp8, x_scales_t = quantize_and_pack_activation(x)

    # Step 2: Pad gather indices (padding rows -> row 0, safe arbitrary data)
    padded_gather_idx = torch.zeros(
        padded_total, dtype=x_gather_idx.dtype, device=x_gather_idx.device
    )
    padded_gather_idx[dst_idx] = x_gather_idx

    # Step 3: Gather scales T->TK_padded in ISA layout
    K = x.shape[1]
    x_scales_tk_e8m0 = _gather_1x32_scales_to_isa(
        x_scales_t, padded_gather_idx, int(x_fp8.shape[0]), K
    )
    del x_scales_t

    # Step 4: Weight FP8 from the owning layer lifecycle
    w1_fp8, w1_scales = w1_fp8_pre

    # Step 5: Zero-mat GEMM+SwiGLU with padded 128-aligned boundaries
    z_padded, y1_padded = gemm_gated(
        x_fp8,
        w1_fp8,
        activation="swiglu",
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=padded_cu,
        A_idx=padded_gather_idx,
        a_scales=x_scales_tk_e8m0,
        b_scales=w1_scales,
        dynamic_scheduler=False,
        tuned=False,
        swiglu_clamp_value=_get_fp8_config().swiglu_clamp_value,
    )
    del x_fp8, x_scales_tk_e8m0, padded_gather_idx

    # Step 6: Unpad results — discard padding rows
    z = z_padded[dst_idx]
    y1 = y1_padded[dst_idx]
    del z_padded, y1_padded

    return z, y1


def _use_epilogue_quant() -> bool:
    """Check if epilogue blockscaled quant of z is enabled (default: enabled).

    When enabled, the GemmGated epilogue computes blockscaled FP8 quantization
    of z in registers (integer+carry E8M0, matching Triton/Paddle reference)
    AND writes z directly as fp8 to D (out_dtype=float8_e4m3fn).

    Benefits:
    - Eliminates standalone _quantize_flat_v2_kernel (~141 µs)
    - Eliminates z.to(fp8) cast (~288 µs)
    - Never allocates bf16 z (saves 384 MiB allocation + write bandwidth)
    - Direct fp32->fp8 path is more precise than fp32->bf16->fp8 (one less rounding)
    """
    cfg = get_active_config()
    if cfg is not None and cfg.epilogue_quant is not None:
        return cfg.epilogue_quant
    return os.getenv("SONIC_MOE_FP8_EPILOGUE_QUANT", "1").lower() in {"1", "true", "yes", "on"}


def _use_fused_swiglu_quant() -> bool:
    """Check if fused SwiGLU+quantize kernels are enabled (default: enabled)."""
    cfg = get_active_config()
    if cfg is not None and cfg.fused_swiglu_quant is not None:
        return cfg.fused_swiglu_quant
    return os.getenv("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", "1").lower() in {"1", "true", "yes", "on"}


def _use_wgrad_beta_accum() -> bool:
    return os.getenv("SONIC_MOE_FP8_WGRAD_TMA_ADD", "").lower() not in {"1", "true", "yes", "on"}


def _use_fused_zy1_quant() -> bool:
    """Check if fused z+y1 quantization is enabled (default: disabled).

    When enabled, z (flat scales) and y1 (ISA-packed scales) are quantized
    in a single fused Triton kernel launch, saving ~3us launch overhead.
    Cost: +96 MiB forward peak (z_fp8 + y1_fp8 coexist during kernel).
    """
    cfg = get_active_config()
    if cfg is not None and cfg.fused_zy1_quant is not None:
        return cfg.fused_zy1_quant
    return os.getenv("SONIC_MOE_FP8_FUSED_ZY1_QUANT", "").lower() in {"1", "true", "yes", "on"}


def _use_fuse_y1_quant() -> bool:
    """Fuse y1 (=SwiGLU(z)) FP8 quant into the up-proj GEMM epilogue (default OFF).

    When enabled, the up-proj GemmGatedSm100ZeroMatPostActQuant kernel writes y1
    directly as FP8 + ISA-packed UE8M0 scales, eliminating the standalone
    quantize_and_pack_activation(y1) kernel (~138us) and the bf16 y1 HBM
    materialization (~814 MiB).  Mutually exclusive with z epilogue quant
    (save_z_fp8); precision-first — z stays bf16.
    """
    cfg = get_active_config()
    if cfg is not None and getattr(cfg, "fuse_y1_quant", None) is not None:
        return cfg.fuse_y1_quant
    # Default ON: nsys-projection A/B (TK>=131072, H2048 I1024) shows -5..-11%
    # step time, precision-neutral; eliminates the standalone y1 quant kernel
    # (~132us/step) + bf16 y1 HBM (~814 MiB/layer).  Unaligned shapes fall back
    # gracefully at the call site (alignment gate); z stays bf16 (save_z OFF).
    return os.getenv("SONIC_MOE_FUSE_Y1_QUANT", "1").lower() in {"1", "true", "yes", "on"}


def _use_fp8_wgrad() -> bool | None:
    """Check if FP8 weight gradients are enabled.

    Returns True/False if explicitly set (config or env var), None for auto-detect.
    When None (auto), ``_FP8Config.resolve_wgrad(I)`` applies the shape-based
    heuristic: ON for I >= 2048, OFF for I < 2048.
    """
    cfg = get_active_config()
    if cfg is not None and cfg.fp8_wgrad is not None:
        return cfg.fp8_wgrad
    val = os.getenv("SONIC_MOE_FP8_WGRAD", "").lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None  # auto-detect based on I


def _save_z_fp8() -> bool:
    """Check if z tensor should be stored in FP8 format to save memory (default: DISABLED).

    When enabled, z(TK, 2I) is quantized to blockscaled FP8 at end of forward
    and dequantized at start of backward, saving ~50% of z's memory footprint.
    Default is OFF (precision first): z stays BF16 end-to-end and is never
    quantized — the dgated backward kernel consumes z as a BF16 epilogue preact.
    """
    cfg = get_active_config()
    if cfg is not None and cfg.save_z_fp8 is not None:
        return cfg.save_z_fp8
    return os.getenv("SONIC_MOE_FP8_SAVE_Z_FP8", "0").lower() in {"1", "true", "yes", "on"}


def _recompute_z() -> bool:
    """Defer z_fp8 materialization to backward (saves ~213 MiB peak per layer).

    When enabled and ``save_z_fp8`` is also enabled, ``_UpProjection.forward``
    runs the gated GEMM but does NOT populate ``_PREQUANTIZED_SCALES['z_fp8']``;
    ``_DownProjection.forward`` records a recompute closure on ctx; and
    ``_DownProjection.backward`` re-runs a quant-only up-proj GEMM that emits
    ``z_fp8 + scales`` without recomputing SwiGLU/y1.
    """
    cfg = get_active_config()
    if cfg is not None and cfg.recompute_z is not None:
        return cfg.recompute_z
    return os.getenv("SONIC_MOE_FP8_RECOMPUTE_Z", "0").lower() in {"1", "true", "yes", "on"}


def _use_fused_blockscaled_gated() -> bool:
    """Check if fused gemm_gated + blockscaled FP8 is enabled (default: enabled).

    When enabled, the blockscaled FP8 path uses fused gemm_gated/gemm_dgated
    (single CUTLASS kernel: GEMM + SwiGLU + blockscaled descale) instead of
    separate blockscaled_fp8_gemm_varlen + standalone SwiGLU.  This is the
    best-performing FP8 up-proj path on SM100 and is enabled by default.
    """
    cfg = get_active_config()
    if cfg is not None and cfg.fused_gated is not None:
        return cfg.fused_gated
    return os.getenv("SONIC_MOE_FP8_FUSED_GATED", "1").lower() in {"1", "true", "yes", "on"}


# Transfer pre-packed blockscaled scales between autograd Function boundaries.
# Each entry maps a tag to (fp8_tensor, packed_scales) or
# (fp8_tensor, packed_scales, raw_scales_uint8).  The consumer checks
# that its input tensor shares the same storage/view metadata as the stored
# tensor before using the scales. Custom autograd boundaries may wrap the same
# storage in a fresh Tensor object, so object identity alone is too strict.
# "fwd": _UpProjection.forward -> _DownProjection.forward  (3-tuple: ref, fp8, scales)
# "bwd": _DownProjection.backward -> _UpProjection.backward (3-tuple: ref, fp8, scales)
_PREQUANTIZED_SCALES: dict[str, tuple] = {}

# Counter for pre-quantization hits (testing/diagnostics).
_PREQUANT_HIT_COUNT: dict[str, int] = collections.defaultdict(int)

def _matches_prequant_tensor(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> bool:
    if lhs is None or rhs is None:
        return False
    _offset = lambda t: t._offset() if hasattr(t, '_offset') else t.storage_offset()
    return (
        lhs.device == rhs.device
        and lhs.dtype == rhs.dtype
        and tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and _offset(lhs) == _offset(rhs)
        and lhs.data_ptr() == rhs.data_ptr()
    )


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


_ALIGNMENT_STREAK: int = 0
_ALIGNMENT_ASSUMED: bool = True  # route-level padding guarantees 128-alignment
_ALIGNMENT_STREAK_THRESHOLD: int = 3


def _is_alignment_assumed() -> bool:
    """Check if alignment is assumed via config, env var, or streak."""
    cfg = get_active_config()
    if cfg is not None and cfg.assume_aligned is not None:
        return cfg.assume_aligned
    return _ALIGNMENT_ASSUMED


def _all_segments_128_aligned(cu_seqlens: torch.Tensor) -> bool:
    """Return True if all expert segments are 128-aligned (no GEMM padding needed).

    Pre-quantized activation input to blockscaled_fp8_gemm_varlen is only
    beneficial when no padding is required, because the padding fallback must
    dequantize -> pad -> re-quantize which is very expensive.

    After ``_ALIGNMENT_STREAK_THRESHOLD`` consecutive aligned iterations, the
    check is skipped entirely (zero D2H sync).  ``SonicMoEConfig(assume_aligned=True)``
    or env var ``SONIC_MOE_FP8_ASSUME_ALIGNED=1`` forces immediate zero-sync mode.
    """
    global _ALIGNMENT_STREAK, _ALIGNMENT_ASSUMED
    if _is_alignment_assumed():
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    vals = _get_cu_seqlens_cpu(cu_seqlens)
    result = all((vals[i + 1] - vals[i]) % 128 == 0 for i in range(len(vals) - 1))
    if result:
        _ALIGNMENT_STREAK += 1
        if _ALIGNMENT_STREAK >= _ALIGNMENT_STREAK_THRESHOLD:
            _ALIGNMENT_ASSUMED = True
    else:
        _ALIGNMENT_STREAK = 0
    return result



def _parse_runtime_precision(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise RuntimeError(f"{name} must be one of {{{allowed_list}}}, but got {value!r}")
    return value


def _upproj_epilogue_precision() -> str:
    return _parse_runtime_precision(
        "SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION",
        default="fp8",
        allowed={"bf16", "fp8"},
    )


def _downproj_mainloop_precision() -> str:
    return _parse_runtime_precision(
        "SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION",
        default="bf16",
        allowed={"bf16", "fp8-blockscaled"},
    )


def _downproj_weight_precision() -> str:
    default = "fp8" if _downproj_mainloop_precision() == "fp8-blockscaled" else "bf16"
    return _parse_runtime_precision(
        "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION",
        default=default,
        allowed={"bf16", "fp8"},
    )


def _use_blockscaled_fp8_downproj() -> bool:
    return _downproj_mainloop_precision() == "fp8-blockscaled"



def _fp8_mode() -> str:
    """Return FP8 mode: 'off', 'perf' (cache+speed), or 'mem' (no-cache+savings).

    Priority: SonicMoEConfig > enable_fp8() context > SONIC_MOE_FP8_MODE env var.
    When is_fp8_active() returns False (e.g. enable_fp8(False)), this returns
    'off' regardless of the env var — the context manager takes precedence.
    """
    if not is_fp8_active():
        return "off"
    mode = os.getenv("SONIC_MOE_FP8_MODE", "").strip().lower()
    if mode == "mem":
        return "mem"
    return "perf"


def _fp8_enabled() -> bool:
    return _fp8_mode() != "off"


# ---------------------------------------------------------------------------
# FP8 runtime config — resolved once per forward, passed via ctx
# ---------------------------------------------------------------------------
class _FP8Config:
    """Snapshot of all FP8 flags, resolved once at forward entry.

    Resolves from SonicMoEConfig (if active), then env vars, then defaults.
    Instances are cheap (no tensors), picklable, stored on autograd ctx.
    """
    __slots__ = (
        "enabled", "fused_gated", "save_z_fp8", "recompute_z", "fused_swiglu_quant",
        "epilogue_quant", "fp8_wgrad", "_fp8_wgrad_setting", "alignment_assumed",
        "iso32_weight", "dz_iso32", "swiglu_clamp_value",
    )

    def __init__(self) -> None:
        self.enabled: bool = _fp8_enabled()
        self.fused_gated: bool = _use_fused_blockscaled_gated()
        self.save_z_fp8: bool = _save_z_fp8()
        self.recompute_z: bool = _recompute_z()
        self.fused_swiglu_quant: bool = _use_fused_swiglu_quant()
        self.epilogue_quant: bool = _use_epilogue_quant()
        self._fp8_wgrad_setting = _use_fp8_wgrad()  # True/False/None
        self.fp8_wgrad: bool = self._fp8_wgrad_setting or False  # resolved in resolve_wgrad
        self.alignment_assumed: bool = False
        active_cfg = get_active_config()
        if active_cfg is not None:
            self.iso32_weight = active_cfg.resolve_iso32_weight()
            self.dz_iso32 = active_cfg.resolve_dz_iso32()
            self.swiglu_clamp_value = active_cfg.resolve_swiglu_clamp_value()
        else:
            self.iso32_weight = os.environ.get("SONIC_MOE_FP8_ISO32_WEIGHT", "0") == "1"
            self.dz_iso32 = os.environ.get("SONIC_MOE_DZ_ISO32", "0") != "0"
            self.swiglu_clamp_value = 0.0

    # Threshold below which FP8 wgrad quant overhead exceeds GEMM savings.
    # Session 53 re-benchmarked after cache fix + stash:
    #   I=1536: 1.300× (aligned), I=2048: 1.35×, I=3072: 1.50×.
    # Wgrad ON is profitable at all I values when aligned.
    # For non-aligned shapes, wgrad is disabled (padding overhead + CUTLASS
    # compat issues make it unprofitable).
    _WGRAD_FP8_I_THRESHOLD = 0

    def resolve_wgrad(self, I: int) -> None:
        """Resolve FP8 wgrad based on explicit setting or shape-based heuristic.

        After the Session 53 VARLEN cache fix, FP8 wgrad is profitable at all
        tested I values (1536, 2048, 3072). Threshold is effectively 0.
        """
        if self._fp8_wgrad_setting is not None:
            self.fp8_wgrad = self._fp8_wgrad_setting
        else:
            self.fp8_wgrad = I >= self._WGRAD_FP8_I_THRESHOLD

    @staticmethod
    def disabled() -> "_FP8Config":
        """Return a config where everything is off (BF16 path)."""
        cfg = _FP8Config.__new__(_FP8Config)
        cfg.enabled = False
        cfg.fused_gated = False
        cfg.save_z_fp8 = False
        cfg.recompute_z = False
        cfg.fused_swiglu_quant = False
        cfg.epilogue_quant = False
        cfg.fp8_wgrad = False
        cfg._fp8_wgrad_setting = False
        cfg.alignment_assumed = False
        cfg.iso32_weight = False
        cfg.dz_iso32 = False
        cfg.swiglu_clamp_value = 0.0
        return cfg


# Module-level singleton, refreshed per forward call.
_fp8_cfg: _FP8Config = _FP8Config.disabled()


def _get_fp8_config() -> _FP8Config:
    """Return the current FP8 config (resolved at forward entry)."""
    return _fp8_cfg


def _refresh_fp8_config() -> _FP8Config:
    """Re-read all env vars and return a fresh config. Call at forward entry."""
    global _fp8_cfg
    _fp8_cfg = _FP8Config()
    return _fp8_cfg


def _get_blockscaled_protocol() -> FP8Protocol:
    """Return FP8Protocol with 1×32 blockscaling for SM100 hardware-native descaling."""
    return FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)


# ---------------------------------------------------------------------------
# FP8 weight helpers
# ---------------------------------------------------------------------------
_FP8_WEIGHT_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}

# Permuted + contiguous caches for gemm_gated / gemm_dgated custom kernels
_TAG_PERM = {
    "w1_ekh": (2, 1, 0),  # (2I,H,E) -> (E,H,2I) contiguous — gemm_gated
    "w2_ehi": (2, 0, 1),  # (H,I,E)  -> (E,H,I)  contiguous — gemm_dgated
}


def _make_fp8_weight(w: torch.Tensor, tag: str) -> torch.Tensor:
    """Create an fp8 copy of *w* with the permutation for *tag*.
    Single allocation: no intermediate bf16 contiguous copy."""
    perm = _TAG_PERM[tag]
    target_shape = tuple(w.shape[p] for p in perm)
    fp8_w = torch.empty(target_shape, dtype=torch.float8_e4m3fn, device=w.device)
    fp8_w.copy_(w.permute(*perm))
    return fp8_w


# Flag for one-shot lazy eviction when switching to blockscaled path.
_PER_TENSOR_EVICTED: bool = False


def _get_cached_fp8_weight(w: torch.Tensor, tag: str) -> torch.Tensor:
    """Return a cached fp8 copy of *w*. Always cached (essential for fused kernels)."""
    global _PER_TENSOR_EVICTED
    key = (w.data_ptr(), w._inplace_version(), tag)
    cached = _FP8_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    fp8_w = _make_fp8_weight(w, tag)
    if len(_FP8_WEIGHT_CACHE) >= 4:
        oldest = next(iter(_FP8_WEIGHT_CACHE))
        del _FP8_WEIGHT_CACHE[oldest]
    _FP8_WEIGHT_CACHE[key] = fp8_w
    # Per-tensor cache is being populated again; allow future eviction.
    _PER_TENSOR_EVICTED = False
    return fp8_w


# Original-layout fp8 cache for quack.gemm paths (permute views at call site)
_FP8_ORIG_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _get_fp8_weight_orig(w: torch.Tensor) -> torch.Tensor:
    """Return fp8 copy of *w* in original layout. Cached in perf mode."""
    global _PER_TENSOR_EVICTED
    if _fp8_mode() != "perf":
        return w.to(torch.float8_e4m3fn)
    key = (w.data_ptr(), w._inplace_version())
    cached = _FP8_ORIG_CACHE.get(key)
    if cached is not None:
        return cached
    fp8_w = w.to(torch.float8_e4m3fn)
    if len(_FP8_ORIG_CACHE) >= 4:
        oldest = next(iter(_FP8_ORIG_CACHE))
        del _FP8_ORIG_CACHE[oldest]
    _FP8_ORIG_CACHE[key] = fp8_w
    # Per-tensor cache is being populated again; allow future eviction.
    _PER_TENSOR_EVICTED = False
    return fp8_w


def clear_fp8_native_weight_cache() -> None:
    """Call between steps if weights change (e.g. optimizer step)."""
    global _PER_TENSOR_EVICTED
    _FP8_WEIGHT_CACHE.clear()
    _FP8_ORIG_CACHE.clear()
    _PER_TENSOR_EVICTED = False


def _evict_per_tensor_caches_once() -> None:
    """Clear per-tensor FP8 weight caches when transitioning to blockscaled path.

    Called once when the blockscaled path is first taken; subsequent calls are no-ops
    until the flag is reset (e.g. by clear_all_fp8_weight_caches).
    """
    global _PER_TENSOR_EVICTED
    if _PER_TENSOR_EVICTED:
        return
    _FP8_WEIGHT_CACHE.clear()
    _FP8_ORIG_CACHE.clear()
    _PER_TENSOR_EVICTED = True


def clear_all_fp8_weight_caches() -> None:
    """Clear every FP8 weight cache (per-tensor + blockscaled).

    Intended for MoE.clear_fp8_weight_cache() and optimizer-step boundaries.
    """
    global _PER_TENSOR_EVICTED
    _FP8_WEIGHT_CACHE.clear()
    _FP8_ORIG_CACHE.clear()
    _PER_TENSOR_EVICTED = False
    # Also clear the blockscaled weight cache in blockscaled_fp8_gemm.py
    clear_blockscaled_fp8_weight_cache()
    # Clear the Triton raw-scale weight cache
    clear_raw_weight_cache()
    # Clear the sgl-kernel weight cache
    clear_sgl_weight_cache()


def _get_fp8_weight_attr(
    weight: torch.Tensor,
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    attr = getattr(weight, key, None)
    if attr is None:
        raise RuntimeError(
            f"Sonic FP8 forward requires weight.{key} attribute; "
            "call quant_weight() before FP8 forward"
        )
    return attr


def _validate_runtime_precision_switches(fp8_protocol: FP8Protocol | None) -> None:
    upproj_precision = _upproj_epilogue_precision()
    downproj_mainloop_precision = _downproj_mainloop_precision()
    downproj_weight_precision = _downproj_weight_precision()

    if fp8_protocol is None:
        return

    if downproj_weight_precision == "fp8" and downproj_mainloop_precision != "fp8-blockscaled":
        raise RuntimeError(
            "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION=fp8 currently requires "
            "SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION=fp8-blockscaled"
        )


def _stage_memory_debug_enabled() -> bool:
    cfg = get_active_config()
    if cfg is not None and cfg.stagewise_memory is not None:
        return cfg.stagewise_memory
    return os.getenv("SONIC_MOE_STAGEWISE_MEMORY", "").lower() in {"1", "true", "yes", "on"}


def _reset_stage_memory_probe() -> None:
    if not _stage_memory_debug_enabled() or torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _log_stage_memory(stage: str) -> None:
    if not _stage_memory_debug_enabled() or torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.synchronize()
    mib = 1024**2
    print(
        f"[stage-memory] {stage}: "
        f"alloc_mib={torch.cuda.memory_allocated() / mib:.2f}, "
        f"reserved_mib={torch.cuda.memory_reserved() / mib:.2f}, "
        f"peak_alloc_mib={torch.cuda.max_memory_allocated() / mib:.2f}, "
        f"peak_reserved_mib={torch.cuda.max_memory_reserved() / mib:.2f}"
    )


def general_routing_router_metadata(
    router_scores_selected: torch.Tensor, sorted_selected_T: torch.Tensor, selected_E: torch.Tensor, T: int, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = router_scores_selected.device

    expert_frequency, expert_frequency_offset = count_cumsum(selected_E, E, do_cumsum=True)
    expert_frequency_offset = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), expert_frequency_offset])

    s_scatter_idx = selected_E.argsort().int()
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.size(0), device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    x_gather_idx = sorted_selected_T[s_scatter_idx]

    if T % 4 == 0 and T <= 50000:
        _, num_activated_expert_per_token_offset = count_cumsum(sorted_selected_T, T, do_cumsum=True)
    else:
        num_activated_expert_per_token_offset = torch.bincount(sorted_selected_T, minlength=T).cumsum(0).int()

    num_activated_expert_per_token_offset = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), num_activated_expert_per_token_offset]
    )

    return (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits: torch.Tensor, E: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        # change this to router_logits.dtype (bfloat16) increase another 5 tflops at fwd at the cost of numerical accuracy
        topk_router_score = torch.empty(T, K, dtype=torch.float32, device=router_logits.device)
        topk_router_indices = torch.empty(T, K, dtype=torch.int32, device=router_logits.device)

        _softmax_topk_fwd(router_logits, topk_router_score, topk_router_indices, E, K)

        ctx.save_for_backward(topk_router_score, topk_router_indices)
        ctx.E = E
        ctx.dtype = router_logits.dtype

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, K = dtopk_score.size()

        topk_router_score, topk_router_indices = ctx.saved_tensor()
        dlogits = torch.zeros(T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device)

        _softmax_topk_bwd(dlogits, None, dtopk_score, topk_router_score, topk_router_indices, K)

        return (dlogits,)


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        use_low_precision_postact_buffer: bool = False,
        prequant_activation_payload: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        use_quack_gemm = is_using_quack_gemm()

        if use_quack_gemm:
            # assert not torch.compiler.is_compiling()  # Paddle compat
            assert is_glu_activation, "QuACK GEMM does not support non GLU activation yet"
            cfg = _get_fp8_config()
            if cfg.enabled:
                cfg.resolve_wgrad(w1.shape[0] // 2)  # w1 is (2I, H, E), I = shape[0]/2
                global _ALIGNMENT_ASSUMED
                _evict_per_tensor_caches_once()
                aligned = _all_segments_128_aligned(expert_frequency_offset)
                _ALIGNMENT_ASSUMED = aligned
                cfg.alignment_assumed = aligned

                if aligned and cfg.fused_gated:
                    x_fp8_pre = x_scales_pre = None
                    if prequant_activation_payload is not None:
                        x_fp8_pre, x_scales_pre = prequant_activation_payload
                    w1_fused_payload = _get_fp8_weight_attr(w1, "fp8")
                    fuse_y1 = _use_fuse_y1_quant()
                    z, y1, y1_fp8_fused, y1_scales_fused = (
                        _fused_blockscaled_gated_forward(
                            x,
                            w1,
                            expert_frequency_offset,
                            x_gather_idx,
                            w1_fp8_pre=w1_fused_payload,
                            x_fp8_pre=x_fp8_pre,
                            x_scales_pre=x_scales_pre,
                            store_z=not cfg.recompute_z,
                            fuse_y1_quant=fuse_y1,
                        )
                    )
                    if cfg.recompute_z:
                        # Forward skips preact storage; z is materialized just-in-time
                        # in DownProj.backward from the recompute args stashed here.
                        # Precision-first: recompute z in BF16 (no z quant) by default;
                        # only emit FP8 when save_z_fp8 is explicitly opted in.
                        _PREQUANTIZED_SCALES.pop("z_fp8", None)
                        x_recompute = None if x_fp8_pre is not None else x
                        recompute_key = (
                            "z_fp8_recompute" if cfg.save_z_fp8 else "z_bf16_recompute"
                        )
                        _PREQUANTIZED_SCALES[recompute_key] = (
                            x_recompute, w1, expert_frequency_offset, x_gather_idx,
                            x_fp8_pre, x_scales_pre, w1_fused_payload,
                        )
                        if y1_fp8_fused is not None:
                            y1_fp8, y1_packed_scales = y1_fp8_fused, y1_scales_fused
                        else:
                            y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                    elif cfg.save_z_fp8 and "z_fp8" not in _PREQUANTIZED_SCALES:
                        if _use_fused_zy1_quant():
                            # Fused z+y1 quantization: single kernel launch, ~3µs
                            # less launch overhead, but +96 MiB peak (z_fp8 + y1_fp8
                            # coexist with z_bf16 + y1_bf16 during the kernel).
                            z_fp8, z_raw_scales, y1_fp8, y1_packed_scales = (
                                fused_z_save_y1_quant(z, y1)
                            )
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            # z.untyped_storage().resize_(0)
                        else:
                            # Split quantization: z first, free z bf16, then y1.
                            # This avoids z_bf16+y1_bf16+z_fp8+y1_fp8 all coexisting
                            # and reduces forward peak by ~96 MiB at Ernie shape.
                            z_fp8, z_raw_scales = quantize_activation_blockscaled_fast(z)
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            # z.untyped_storage().resize_(0)
                            y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                    else:
                        # z_fp8 already populated by epilogue quant inside
                        # _fused_blockscaled_gated_forward.  z is a bf16 placeholder
                        # with freed storage (for autograd graph only).
                        # No resize needed — storage is already 0.
                        if y1_fp8_fused is not None:
                            y1_fp8, y1_packed_scales = y1_fp8_fused, y1_scales_fused
                        else:
                            y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                    _PREQUANTIZED_SCALES["fwd"] = (y1, y1_fp8, y1_packed_scales)
                    if y1_fp8_fused is not None:
                        # Fused y1: bf16 y1 is a freed-storage placeholder.  The
                        # DownProjection MUST hit this prequant entry — quantizing
                        # the placeholder would yield garbage.  Mark it required.
                        _PREQUANTIZED_SCALES["fwd_required"] = True
                    # y1.untyped_storage().resize_(0)
                elif aligned:
                    w1_fp8, w1_scales = precompute_weight_fp8(w1)
                    # All segments 128-aligned: use fused gather+quantize
                    # and pre-quantized GEMM (no padding overhead).
                    x_fp8, x_scales = fast_gather_quantize_and_pack_activation(
                        x, x_gather_idx
                    )
                    z = blockscaled_fp8_gemm_varlen(
                        x_fp8, w1, expert_frequency_offset,
                        a_scales=x_scales,
                        w_fp8=w1_fp8, w_scales=w1_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=True,
                    )
                    del x_fp8, x_scales

                    # Fused SwiGLU+quant only when segments are aligned
                    if cfg.fused_swiglu_quant:
                        if cfg.swiglu_clamp_value > 0.0:
                            y1 = _swiglu_forward_clamp_reference(z, cfg.swiglu_clamp_value)
                            if cfg.save_z_fp8:
                                z_fp8, z_raw_scales = quantize_activation_blockscaled_fast(z)
                                _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                        elif cfg.save_z_fp8:
                            # Fused SwiGLU+y1_quant+z_save: read z ONCE
                            if swiglu_forward_quant_pack_zsave_triton is None:
                                raise RuntimeError("fused SwiGLU z-save quant helper is unavailable")
                            y1_fp8, y1_packed_scales, z_fp8, z_raw_scales = (
                                swiglu_forward_quant_pack_zsave_triton(z)
                            )
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            y1 = y1_fp8
                        else:
                            y1 = _swiglu_forward_interleaved(z)
                            y1_fp8, y1_packed_scales = quantize_and_pack_activation(y1)
                        _PREQUANTIZED_SCALES["fwd"] = (y1, y1_fp8, y1_packed_scales)
                        if cfg.swiglu_clamp_value <= 0.0 and cfg.save_z_fp8:
                            y1 = y1_fp8
                    else:
                        if cfg.swiglu_clamp_value > 0.0:
                            y1 = _swiglu_forward_clamp_reference(z, cfg.swiglu_clamp_value)
                        else:
                            y1 = _swiglu_forward_interleaved(z)
                else:
                    # Non-aligned: pad expert segments to 128, use FP8 zero-mat
                    # path.  Overhead is only the extra padded GEMM rows (~5-25%
                    # depending on routing), much cheaper than full BF16 fallback.
                    z, y1 = _padded_blockscaled_gated_forward(
                        x, w1, expert_frequency_offset, x_gather_idx,
                        _get_fp8_weight_attr(w1, "fp8"),
                    )
            else:
                z, y1 = gemm_gated(
                    x,
                    w1.permute(2, 1, 0),
                    activation="swiglu",
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    postact_dtype=(torch.float8_e4m3fn if use_low_precision_postact_buffer else None),
                    dynamic_scheduler=False,
                    tuned=False,
                    swiglu_clamp_value=cfg.swiglu_clamp_value,
                )
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_varlen_K = is_varlen_K
        ctx.is_glu_activation = is_glu_activation
        ctx.stream_id = stream_id
        ctx.use_quack_gemm = use_quack_gemm
        # Store FP8 config snapshot for backward (avoids os.getenv in backward).
        ctx._fp8_cfg = cfg if use_quack_gemm else _FP8Config.disabled()
        # Legacy compat: keep individual flags for code that reads them directly.
        ctx._fp8_enabled = ctx._fp8_cfg.enabled
        ctx._alignment_assumed = ctx._fp8_cfg.alignment_assumed
        # Track which optional tensor inputs were actually provided (for Paddle backward return count)
        ctx._has_b1 = b1 is not None
        ctx._has_num_activated = num_activated_expert_per_token_offset is not None
        ctx._prequant_activation_payload = prequant_activation_payload is not None

        # Weight decoupling: in FP8+aligned mode, backward doesn't need bf16 w1 data
        # (only uses fp8 cache + metadata). This enables stash_bf16_to_cpu() to
        # resize_(0) the bf16 param storage without breaking backward.
        _fp8_aligned = (use_quack_gemm and cfg.enabled and cfg.alignment_assumed)
        ctx._w1_decoupled = _fp8_aligned
        if _fp8_aligned and ctx._prequant_activation_payload and not cfg.fp8_wgrad:
            raise RuntimeError("prequant activation payload requires FP8 wgrad because BF16 x is not retained")
        if _fp8_aligned:
            # Store metadata needed for dw1 allocation
            ctx._w1_shape = w1.shape  # (2I, H, E)
            ctx._w1_dtype = w1.dtype
            ctx._w1_device = w1.device
            ctx._w1T_fp8, ctx._w1T_scales = _get_fp8_weight_attr(
                w1, "transposed_fp8"
            )
            if ctx._prequant_activation_payload:
                x_fp8_pre, x_scales_pre = prequant_activation_payload
                x_saved = torch.empty(1, dtype=x.dtype, device=x.device).as_strided((T, H), (0, 0))
                ctx.save_for_backward(
                    x_saved,
                    # w1 omitted — backward uses ctx._w1T_fp8 + metadata
                    b1,
                    expert_frequency_offset,
                    x_gather_idx,
                    None if use_quack_gemm else s_scatter_idx,
                    s_reverse_scatter_idx,
                    num_activated_expert_per_token_offset,
                    x_fp8_pre,
                    x_scales_pre,
                )
            else:
                ctx.save_for_backward(
                    x,
                    # w1 omitted — backward uses ctx._w1T_fp8 + metadata
                    b1,
                    expert_frequency_offset,
                    x_gather_idx,
                    None if use_quack_gemm else s_scatter_idx,
                    s_reverse_scatter_idx,
                    num_activated_expert_per_token_offset,
                )
        else:
            ctx.save_for_backward(
                x,
                w1,
                b1,
                expert_frequency_offset,
                x_gather_idx,
                None if use_quack_gemm else s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
            )

        ctx.mark_non_differentiable(y1)
        ctx.set_materialize_grads(False)

        # Keep w1 FP8 cache — backward hits cache (~112µs savings) at ~74MB memory cost.
        # The cache auto-invalidates via w._version when optimizer updates weights.

        return y1, z

    @staticmethod
    def backward(ctx, _: None, dz: torch.Tensor):
        is_compiling = False

        if not is_compiling:
            assert _ is None

        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_varlen_K = ctx.is_varlen_K
        stream_id = ctx.stream_id
        use_quack_gemm = ctx.use_quack_gemm

        x_fp8_pre = x_scales_pre = None
        if ctx._w1_decoupled:
            # FP8+aligned: w1 not in saved_tensors; use metadata + fp8 cache.
            saved = ctx.saved_tensor()
            if ctx._prequant_activation_payload:
                (
                    x,
                    b1,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    num_activated_expert_per_token_offset,
                    x_fp8_pre,
                    x_scales_pre,
                ) = saved
            else:
                (
                    x,
                    b1,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    num_activated_expert_per_token_offset,
                ) = saved
            w1_shape = ctx._w1_shape   # (2I, H, E)
            w1_dtype = ctx._w1_dtype
            w1_device = ctx._w1_device
        else:
            (
                x,
                w1,
                b1,
                expert_frequency_offset,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
            ) = ctx.saved_tensor()
            w1_shape = w1.shape
            w1_dtype = w1.dtype
            w1_device = w1.device

        # Defer dw1 allocation for FP8 wgrad path (blockscaled_fp8_wgrad_varlen_k
        # allocates its own output).  BF16 path allocates below.
        dw1_base = dw1 = None
        db1 = None if b1 is None else torch.empty_like(b1)
        _reset_stage_memory_probe()

        if use_quack_gemm:
            assert not is_compiling

            if ctx._fp8_enabled and ctx._alignment_assumed:
                # Blockscaled FP8 act-grad + weight-grad.
                # Memory-optimized: run wgrad first, free dz bf16 (~384 MiB),
                # then run actgrad using FP8 dz from prequant cache.
                # This serializes the two GEMMs but avoids dz_bf16 + dx_expanded
                # coexisting, reducing backward peak by ~384 MiB.
                dz_bf16 = dz if dz.dtype == torch.bfloat16 else dz.to(torch.bfloat16)

                # Prepare actgrad resources first (cache lookup, no alloc).
                if ctx._w1_decoupled:
                    # w1T fp8 was pre-looked-up in forward; use directly.
                    w1T_fp8 = ctx._w1T_fp8
                    w1T_scales = ctx._w1T_scales
                else:
                    w1T_fp8, w1T_scales = precompute_weight_fp8(w1.permute(1, 0, 2))
                prequant_dz = _PREQUANTIZED_SCALES.pop("bwd", None)
                if ctx._fp8_cfg.fp8_wgrad:
                    # FP8 wgrad: dz_bf16 was already freed in DownProj via dual-quant.
                    # Skip _matches_prequant_tensor (dz storage is 0).
                    has_prequant = prequant_dz is not None
                else:
                    has_prequant = (
                        prequant_dz is not None
                        and _matches_prequant_tensor(prequant_dz[0], dz)
                    )

                # Phase 1: Wgrad.
                if ctx._fp8_cfg.fp8_wgrad:
                    # FP8 wgrad with early dz_bf16 release.
                    # dz_col_fp8 was pre-computed in DownProj via dual_quantize_varlen
                    # (single HBM read of dz produced both row+col fp8).
                    bwd_col = _PREQUANTIZED_SCALES.pop("bwd_col", None)

                    # Sequential quant pipeline (all on default stream):
                    if ctx._prequant_activation_payload:
                        x_scales_pre_isa = _ensure_isa_1x32_scales(
                            x_scales_pre, int(x_fp8_pre.shape[0]), int(x_fp8_pre.shape[1])
                        )
                        x_col_fp8, x_col_scales = dequant_colwise_quantize_and_pack_from_isa(
                            x_fp8_pre, x_scales_pre_isa,
                            logical_rows=H, logical_cols=TK,
                            gather_idx=x_gather_idx,
                        )
                        del x_fp8_pre, x_scales_pre, x_scales_pre_isa
                    else:
                        x_col_fp8, x_col_scales = colwise_quantize_and_pack(
                            x, logical_rows=H, logical_cols=TK,
                            gather_idx=x_gather_idx,
                        )

                    if bwd_col is not None:
                        # Use pre-computed col-fp8 from dual quant (zero extra HBM read)
                        dz_col_fp8, dz_col_scales = bwd_col
                    else:
                        # Fallback: compute col-fp8 now (Triton nw=1)
                        dz_col_fp8, dz_col_scales = colwise_quantize_and_pack(
                            dz_bf16, logical_rows=w1_shape[0], logical_cols=TK,
                        )
                    # FREE dz_bf16 NOW (-384 MiB before wgrad GEMM!)
                    # dz.untyped_storage().resize_(0)
                    del dz_bf16

                    # CUTLASS wgrad GEMM
                    # If a fp32 wgrad accumulator is provided (ERNIE main_grad path),
                    # use TMA hardware reduce-add (default) or fused beta=1.0 epilogue.
                    _wgrad_accum = getattr(ctx, '_wgrad_w1_accumulator', None)
                    if _wgrad_accum is not None:
                        if _use_wgrad_beta_accum():
                            _run_cutlass_blockscaled_gemm_varlen_k_accumulate(
                                dz_col_fp8, dz_col_scales,
                                x_col_fp8, x_col_scales,
                                expert_frequency_offset,
                                M=w1_shape[0], N=H, total_K=TK,
                                num_experts=E, device=x.device,
                                accumulator=_wgrad_accum,
                            )
                        else:
                            _run_cutlass_blockscaled_gemm_varlen_k_tma_add(
                                dz_col_fp8, dz_col_scales,
                                x_col_fp8, x_col_scales,
                                expert_frequency_offset,
                                M=w1_shape[0], N=H, total_K=TK,
                                num_experts=E, device=x.device,
                                accumulator=_wgrad_accum,
                            )
                        dw1_base = None
                        dw1 = None
                    else:
                        dw1_base = _run_cutlass_blockscaled_gemm_varlen_k(
                            dz_col_fp8, dz_col_scales,
                            x_col_fp8, x_col_scales,
                            expert_frequency_offset,
                            M=w1_shape[0], N=H, total_K=TK,
                            num_experts=E, out_dtype=w1_dtype, device=x.device,
                        )
                        dw1 = dw1_base.permute(1, 2, 0)
                    del dz_col_fp8, dz_col_scales, x_col_fp8, x_col_scales
                else:
                    _wgrad_accum = getattr(ctx, '_wgrad_w1_accumulator', None)
                    if _wgrad_accum is not None:
                        # BF16 wgrad + fp32 accumulate.
                        # _wgrad_accum: [E, 2I, H] fp32. GEMM out: [E, H, 2I].
                        # permute(0,2,1) gives [E,H,2I] non-contiguous view —
                        # CuTe handles via stride.
                        accum_view = _wgrad_accum.permute(0, 2, 1)  # [E, H, 2I]
                        if _use_wgrad_beta_accum():
                            bf16_wgrad_gemm_varlen_k_accumulate(
                                x.T,
                                dz_bf16,
                                expert_frequency_offset,
                                x_gather_idx,
                                accumulator=accum_view,
                                M=H,
                                N=w1_shape[0],
                                total_K=TK,
                                num_experts=E,
                                device=x.device,
                            )
                        else:
                            bf16_wgrad_gemm_varlen_k_tma_add(
                                x.T,
                                dz_bf16,
                                expert_frequency_offset,
                                x_gather_idx,
                                accumulator=accum_view,
                                M=H,
                                N=w1_shape[0],
                                total_K=TK,
                                num_experts=E,
                                device=x.device,
                            )
                        dw1_base = None
                        dw1 = None
                    else:
                        dw1_base = torch.empty((E, w1_shape[0], w1_shape[1]), dtype=w1_dtype, device=w1_device)
                        dw1 = dw1_base.permute(1, 2, 0)
                        bf16_wgrad_gemm_varlen_k(
                            x.T,
                            dz_bf16,
                            expert_frequency_offset,
                            x_gather_idx,
                            out=dw1_base.permute(0, 2, 1),
                            M=H,
                            N=w1_shape[0],
                            total_K=TK,
                            num_experts=E,
                            device=x.device,
                        )

                # Phase 2: Free dz bf16 storage (~384 MiB at Ernie shape).
                # FP8 wgrad already freed it in step 2 above; BF16 path frees here.
                if not ctx._fp8_cfg.fp8_wgrad:
                    # dz.untyped_storage().resize_(0)
                    del dz_bf16

                # Phase 3: Actgrad using FP8 dz (avoids dz_bf16 + dx_expanded coexistence).
                if has_prequant:
                    _PREQUANT_HIT_COUNT["bwd"] += 1
                    _, dz_fp8, dz_packed_scales = prequant_dz
                    if ctx._w1_decoupled:
                        # w1 not in saved_tensors; call low-level GEMM directly
                        # with shape metadata (avoids needing a weight tensor).
                        dx_expanded = _run_cutlass_blockscaled_gemm(
                            dz_fp8, dz_packed_scales,
                            w1T_fp8, w1T_scales,
                            expert_frequency_offset,
                            total_M=dz_fp8.shape[0],
                            K=dz_fp8.shape[1],
                            H=w1_shape[1],       # w1 is (2I, H, E), H=shape[1]
                            num_experts=E,
                            out_dtype=torch.bfloat16,
                            device=dz_fp8.device,
                        )
                    else:
                        dx_expanded = blockscaled_fp8_gemm_varlen(
                            dz_fp8, w1.permute(1, 0, 2), expert_frequency_offset,
                            a_scales=dz_packed_scales,
                            w_fp8=w1T_fp8, w_scales=w1T_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                    del dz_fp8, dz_packed_scales
                    # Keep w1T FP8 cache (~74 MiB) — avoids 308µs permute+contiguous
                    # on next iter.  Cache auto-invalidates via w._version at optimizer step.
                    del w1T_fp8, w1T_scales
                else:
                    # No prequant: quantize dz inline (dz storage was freed;
                    # this path should not be reached with fused gated).
                    raise RuntimeError(
                        "dz storage freed but no bwd prequant — cannot quantize. "
                        "Ensure _DownProjection backward creates bwd prequant."
                    )
            else:
                # Non-FP8 BF16 path: actgrad + wgrad using CuTe DSL BF16 GEMMs.
                # Supports wgrad accumulator (MlpNode main_grad) and zero-mat gather.
                _wgrad_accum = getattr(ctx, '_wgrad_w1_accumulator', None)
                if _wgrad_accum is not None:
                    accum_view = _wgrad_accum.permute(0, 2, 1)  # [E,2I,H] → [E,H,2I]
                    if _use_wgrad_beta_accum():
                        bf16_wgrad_gemm_varlen_k_accumulate(
                            x.T,
                            dz,
                            expert_frequency_offset,
                            x_gather_idx,
                            accumulator=accum_view,
                            M=H,
                            N=w1_shape[0],
                            total_K=TK,
                            num_experts=E,
                            device=x.device,
                        )
                    else:
                        bf16_wgrad_gemm_varlen_k_tma_add(
                            x.T,
                            dz,
                            expert_frequency_offset,
                            x_gather_idx,
                            accumulator=accum_view,
                            M=H,
                            N=w1_shape[0],
                            total_K=TK,
                            num_experts=E,
                            device=x.device,
                        )
                    dw1 = None
                else:
                    dw1_base = torch.empty((E, w1_shape[0], w1_shape[1]), dtype=w1_dtype, device=w1_device)
                    dw1 = dw1_base.permute(1, 2, 0)
                    bf16_wgrad_gemm_varlen_k(
                        x.T,
                        dz,
                        expert_frequency_offset,
                        x_gather_idx,
                        out=dw1_base.permute(0, 2, 1),
                        M=H,
                        N=w1_shape[0],
                        total_K=TK,
                        num_experts=E,
                        device=x.device,
                    )
                dx_expanded = gemm(
                    dz, w1.permute(2, 0, 1),
                    cu_seqlens_m=expert_frequency_offset, dynamic_scheduler=False,
                    tuned=False,
                )
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        _log_stage_memory("backward:up-proj-core")
        _reset_stage_memory_probe()
        dx_reduced = torch.empty(T, H, dtype=dz.dtype, device=dz.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(K if K is not None else E),
            H=H,
            is_varlen_K=is_varlen_K,
        )
        _log_stage_memory("backward:token-reduce")

        # Paddle PyLayer: return grads only for tensor inputs (not int/bool/enum)
        grads = [dx_reduced, dw1]
        if ctx._has_b1:
            grads.append(db1)
        # expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
        grads.extend([None, None, None, None])
        if ctx._has_num_activated:
            grads.append(None)
        if ctx._prequant_activation_payload:
            grads.append((None, None))
        return tuple(grads)


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y1: torch.Tensor,
        z: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        fp8_protocol: FP8Protocol | None,
        fp8_combine_grad_handle=None,
        topk_scores_expert_order: torch.Tensor | None = None,
        topk_scores_token_order: torch.Tensor | None = None,
        score_src_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        TK = y1.size(0)
        H, I, E = w2.shape

        use_quack_gemm = is_using_quack_gemm()

        if use_quack_gemm:
            # assert not torch.compiler.is_compiling()  # Paddle compat

            assert b2 is None
            cfg = _get_fp8_config()
            if cfg.enabled and cfg.alignment_assumed:
                if cfg.fused_gated:
                    # Use pre-quantized y1 from _UpProjection if available
                    # (zero quant overhead — y1 was quantized while hot in L2).
                    # Format: 3-tuple (bf16_ref, fp8_data, packed_scales).
                    prequant = _PREQUANTIZED_SCALES.pop("fwd", None)
                    fwd_required = _PREQUANTIZED_SCALES.pop("fwd_required", False)
                    has_prequant = (
                        prequant is not None
                        and len(prequant) == 3
                        and _matches_prequant_tensor(prequant[0], y1)
                    )
                    if has_prequant:
                        _PREQUANT_HIT_COUNT["fwd"] += 1
                        w2_fp8, w2_scales = _get_fp8_weight_attr(w2, "fp8")
                        _, y1_fp8, y1_packed_scales = prequant
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1_fp8, w2, expert_frequency_offset,
                            a_scales=y1_packed_scales,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                        del y1_fp8, y1_packed_scales
                    else:
                        if fwd_required:
                            # Fused y1-quant mode: bf16 y1 is a freed-storage
                            # placeholder; quantizing it would yield garbage.
                            # A miss here is a hard bug, never a fallback.
                            raise RuntimeError(
                                "Fused y1-quant prequant MISS in DownProjection "
                                f"(fwd_required set): y1 meta dtype={y1.dtype} "
                                f"shape={tuple(y1.shape)} stride={tuple(y1.stride())} "
                                f"data_ptr={y1.data_ptr()}; prequant="
                                f"{'None' if prequant is None else 'present'}. "
                                "The up-proj epilogue y1_fp8 must be consumed here."
                            )
                        assert y1.stride() != (0, 0), (
                            "y1 has zero-stride (freed-storage placeholder) but no "
                            "prequant entry — refusing to quantize garbage"
                        )
                        # Fallback: inline FP8 quant (prequant cache miss)
                        w2_fp8, w2_scales = _get_fp8_weight_attr(w2, "fp8")
                        y1_fp8, y1_scales = quantize_and_pack_activation(y1)
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1_fp8, w2, expert_frequency_offset,
                            a_scales=y1_scales,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                        del y1_fp8, y1_scales
                else:
                    # Blockscaled FP8 down-proj: use pre-quantized y1 if available
                    # from fused SwiGLU+quant in _UpProjection.forward.
                    w2_fp8, w2_scales = _get_fp8_weight_attr(w2, "fp8")
                    prequant = _PREQUANTIZED_SCALES.pop("fwd", None)
                    has_prequant = (
                        prequant is not None
                        and len(prequant) == 3
                        and _matches_prequant_tensor(prequant[0], y1)
                    )
                    if has_prequant:
                        _PREQUANT_HIT_COUNT["fwd"] += 1
                        _, y1_fp8, y1_packed_scales = prequant
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1_fp8, w2, expert_frequency_offset,
                            a_scales=y1_packed_scales,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                    else:
                        y2 = blockscaled_fp8_gemm_varlen(
                            y1, w2, expert_frequency_offset,
                            w_fp8=w2_fp8, w_scales=w2_scales,
                            out_dtype=torch.bfloat16,
                            assume_aligned=True,
                        )
                # Keep w2 varlen cache — iso32 re-quant is expensive (~87µs/iter).
                # Cache auto-invalidates via w._version at optimizer step.
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
            elif cfg.enabled:
                # FP8 enabled but not aligned: use blockscaled_fp8_gemm_varlen
                # with assume_aligned=False — it pads internally.
                w2_fp8, w2_scales = precompute_weight_fp8(w2)
                y2 = blockscaled_fp8_gemm_varlen(
                    y1, w2, expert_frequency_offset,
                    w_fp8=w2_fp8, w_scales=w2_scales,
                    out_dtype=torch.bfloat16,
                    assume_aligned=False,
                )
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
            else:
                y2 = gemm(
                    y1,
                    w2.permute(2, 1, 0),
                    cu_seqlens_m=expert_frequency_offset,
                    tuned=False,
                )
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        # Output must always be bf16 (z may be fp8 when epilogue_quant is active).
        o = torch.empty(T, H, device=z.device, dtype=torch.bfloat16)
        ctx._topk_scores_shape = tuple(topk_scores.shape)
        topk_scores = topk_scores if topk_scores.ndim == 1 else topk_scores.flatten()
        router_scores_forward = topk_scores_token_order if topk_scores_token_order is not None else topk_scores

        _router_forward(
            y2=y2_for_router,
            o=o,
            topk_scores=router_scores_forward,
            s_reverse_scatter_idx=router_perm,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(K if K is not None else E),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type
        ctx.stream_id = stream_id
        ctx.use_quack_gemm = use_quack_gemm
        # Store FP8 config snapshot for backward.
        ctx._fp8_cfg = cfg if use_quack_gemm else _FP8Config.disabled()
        # Legacy compat aliases
        ctx._fp8_enabled_flag = ctx._fp8_cfg.enabled
        ctx._alignment_assumed_flag = ctx._fp8_cfg.alignment_assumed
        ctx._use_fused_blockscaled_gated_flag = ctx._fp8_cfg.fused_gated
        # Track which optional tensor inputs were actually provided (for Paddle backward return count)
        ctx._has_b2 = b2 is not None
        ctx._has_num_activated = num_activated_expert_per_token_offset is not None
        # Always compute ds (topk_scores gradient) — needed for router training.
        # NOTE: topk_scores.stop_gradient is unreliable inside .apply() because
        # Paddle's torch-proxy resets stop_gradient=True on inputs (mirroring
        # PyTorch Function.apply() detach behavior) without providing
        # ctx.needs_input_grad.  Defaulting to True is safe: if the caller truly
        # doesn't need ds, the autograd engine simply discards it.

        if not hasattr(topk_scores, "stop_gradient"):
            ctx._topk_scores_needs_grad = False
        else:
            ctx._topk_scores_needs_grad = not topk_scores.stop_gradient
        ctx._fp8_combine_grad_handle = fp8_combine_grad_handle
        ctx._has_topk_scores_expert_order = topk_scores_expert_order is not None
        ctx._has_topk_scores_token_order = topk_scores_token_order is not None
        ctx._has_score_src_idx = score_src_idx is not None
        ctx._score_src_idx = score_src_idx

        # Memory optimization: store z in FP8 to save ~50% of z's memory.
        # At Ernie shape (TK=65536, 2I=3072), z is 384MB BF16 -> ~213MB FP8 = ~171MB saved.
        # Accept fp8 z when prequant cache already holds the fp8+scales pair
        # (e.g. epilogue quant produced them), even if z.dtype is no longer bf16.
        z_has_prequant = "z_fp8" in _PREQUANTIZED_SCALES
        z_has_recompute = "z_fp8_recompute" in _PREQUANTIZED_SCALES
        # BF16 recompute (precision-first default): forward stored only a zero-storage
        # bf16 z placeholder; backward re-runs the up-proj GEMM with a BF16 D output so
        # z is never quantized.  Independent of save_z_fp8 / the fp8 recompute path.
        z_bf16_recompute_args = _PREQUANTIZED_SCALES.pop("z_bf16_recompute", None)
        ctx._needs_z_recompute_bf16 = z_bf16_recompute_args is not None
        ctx._z_bf16_recompute_args = z_bf16_recompute_args
        z_is_fp8 = (cfg.enabled and use_quack_gemm and cfg.save_z_fp8
                    and cfg.alignment_assumed
                    and (z.dtype == torch.bfloat16 or z_has_prequant or z_has_recompute))
        ctx._z_is_fp8 = z_is_fp8

        # w2 decoupling: in FP8+aligned+fused_gated mode, backward doesn't
        # read bf16 w2 data (uses fp8 dgated cache + metadata).  This enables
        # clear_param_storage("moe_expert") to release bf16 expert weight storage
        # without requiring recompute_z/save_z_fp8.
        _w2_decouple = (
            cfg.enabled
            and use_quack_gemm
            and cfg.alignment_assumed
            and cfg.fused_gated
        )
        ctx._w2_decoupled = _w2_decouple
        if _w2_decouple:
            ctx._w2_dgated_fp8, ctx._w2_dgated_scales = _get_fp8_weight_attr(
                w2, "transposed_fp8"
            )
            ctx._w2_shape = w2.shape  # (H, I, E)
            ctx._w2_dtype = w2.dtype
            ctx._w2_device = w2.device

        if z_is_fp8:
            recompute_args = _PREQUANTIZED_SCALES.pop("z_fp8_recompute", None)
            if recompute_args is not None:
                # Defer z_fp8 materialization to backward.  Save zero-storage
                # placeholders with correct shape/dtype/device so the existing
                # save_for_backward + ctx.saved_tensor() unpacking still works;
                # backward will re-bind them via the recompute helper.
                ctx._needs_z_recompute = True
                ctx._z_recompute_args = recompute_args
                TK_z, twoI_z = z.shape
                z_fp8 = torch.empty(
                    1, dtype=torch.float8_e4m3fn, device=z.device
                ).as_strided((TK_z, twoI_z), (0, 0))
                z_raw_scales = torch.empty(
                    1, dtype=_E8M0_DTYPE, device=z.device
                ).as_strided((TK_z, twoI_z // 32), (0, 0))
            else:
                ctx._needs_z_recompute = False
                ctx._z_recompute_args = None
                precomputed_z_fp8 = _PREQUANTIZED_SCALES.pop("z_fp8", None)
                if precomputed_z_fp8 is not None:
                    z_fp8, z_raw_scales = precomputed_z_fp8
                else:
                    assert z.nelement() > 0, (
                        "z storage was freed for memory optimization but prequant "
                        "cache miss — this should not happen"
                    )
                    assert z.dtype in (torch.bfloat16, torch.float8_e4m3fn), (
                        f"z_is_fp8=True but no prequant cache and z.dtype={z.dtype} "
                        f"(expected bf16 or fp8 for inline quantization)"
                    )
                    if z.dtype == torch.float8_e4m3fn:
                        # fp8 D output from CUTLASS but cache was cleared.
                        # This shouldn't happen in normal flow but handle gracefully.
                        z_fp8 = z
                        z_raw_scales = torch.ones(
                            z.shape[0], z.shape[1] // 32,
                            dtype=_E8M0_DTYPE, device=z.device
                        )
                    else:
                        z_fp8, z_raw_scales = quantize_activation_blockscaled_fast(z)
            if _w2_decouple:
                ctx.save_for_backward(
                    z_fp8,
                    z_raw_scales,
                    # w2 omitted — backward uses ctx._w2_dgated_fp8 + metadata
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                )
            else:
                ctx.save_for_backward(
                    z_fp8,
                    z_raw_scales,
                    w2,
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                )
        else:
            if _w2_decouple:
                ctx.save_for_backward(
                    z,
                    # w2 omitted — backward uses ctx._w2_dgated_fp8 + metadata
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                )
            else:
                ctx.save_for_backward(
                    z,
                    w2,
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                )

        # Keep w2 FP8 cache — backward hits cache (~38µs savings) at ~37MB memory cost.
        # The cache auto-invalidates via w._version when optimizer updates weights.

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        stream_id = ctx.stream_id
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type
        use_quack_gemm = ctx.use_quack_gemm
        fp8_combine_grad_handle = ctx._fp8_combine_grad_handle

        # Ensure dout is contiguous (expanded tensors from e.g. sum().backward()
        # have stride (0,0) which violates GEMM k-major assertions)
        if not dout.is_contiguous():
            dout = dout.contiguous()

        if ctx._z_is_fp8:
            if ctx._w2_decoupled:
                (
                    z_fp8,
                    z_raw_scales,
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                ) = ctx.saved_tensor()
                w2_shape = ctx._w2_shape   # (H, I, E)
                w2_dtype = ctx._w2_dtype
                w2_device = ctx._w2_device
            else:
                (
                    z_fp8,
                    z_raw_scales,
                    w2,
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                ) = ctx.saved_tensor()
                w2_shape = w2.shape
                w2_dtype = w2.dtype
                w2_device = w2.device
            if getattr(ctx, "_needs_z_recompute", False):
                # Replace zero-storage placeholders with real fp8 z + scales.
                z_fp8, z_raw_scales = _recompute_z_fp8(*ctx._z_recompute_args)
                ctx._z_recompute_args = None
            z_raw_scales_u8 = z_raw_scales.view(torch.uint8)
            # Defer dequantize: FP8 path uses fused kernel, others lazy-dequant
            z = None
        else:
            if ctx._w2_decoupled:
                (
                    z,
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                ) = ctx.saved_tensor()
                w2_shape = ctx._w2_shape
                w2_dtype = ctx._w2_dtype
                w2_device = ctx._w2_device
            else:
                (
                    z,
                    w2,
                    b2,
                    topk_scores,
                    topk_scores_expert_order,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                ) = ctx.saved_tensor()
                w2_shape = w2.shape
                w2_dtype = w2.dtype
                w2_device = w2.device
            z_fp8 = z_raw_scales_u8 = None
            if getattr(ctx, "_needs_z_recompute_bf16", False):
                # Replace the zero-storage bf16 placeholder with a freshly recomputed
                # BF16 z (no z quant); feeds the bf16-preact path of gemm_dgated.
                z = _recompute_z_bf16(*ctx._z_bf16_recompute_args)
                ctx._z_bf16_recompute_args = None

        # Defer dw2 allocation: in the fused_gated path, dw2 is not needed
        # until the wgrad GEMM (~384 MiB after dgated outputs dz+y1s).
        # Allocating here adds 72 MiB to the dgated peak unnecessarily.
        dw2_base = dw2 = None  # allocated just before wgrad in each path
        db2 = None if b2 is None else torch.empty_like(b2)
        _reset_stage_memory_probe()

        if use_quack_gemm:
            # assert not torch.compiler.is_compiling()  # Paddle compat
            assert is_glu(activation_type), "QuACK GEMM does not support non GLU activation yet"

            if ctx._has_topk_scores_expert_order:
                s = topk_scores_expert_order
            else:
                s = _gather_router_scores_i32(topk_scores, s_scatter_idx)
            if ctx._fp8_enabled_flag and ctx._alignment_assumed_flag:
                # ── Early eviction: free forward-only FP8 weight caches ──
                # Clears w2_varlen (~37 MiB freeable, forward down-proj only)
                # and w1_fused dict entries at backward entry.
                # At I=1536 this doesn't reduce peak (wgrad dominates), but
                # at larger shapes where dgated peak exceeds wgrad peak,
                # this saves ~37 MiB.
                # Backward-needed caches survive via ctx references:
                #   w2_dgated  -> ctx._w2_dgated_fp8 (DownProj dgated)
                #   w1T_varlen -> UpProj ctx._w1T_fp8 (actgrad)
                #
                # NOTE: keep _VARLEN_WEIGHT_CACHE alive — it is keyed by
                # (data_ptr, _version) so stale entries auto-invalidate at
                # optimizer step, and keeping it avoids ~300µs/iter weight
                # re-quantize overhead on the next forward pass.
                # Keep ALL weight caches alive by default (max performance).
                # Cache eviction is available via SonicMoEConfig(memopt=True)
                # for memory-constrained scenarios (saves ~1755 MiB at E=128,
                # costs ~500µs/iter re-quantization).

                # All segments aligned: use blockscaled FP8 path.
                if ctx._use_fused_blockscaled_gated_flag:
                    # Zero-materialization FP8 dgated: T-quant + scale_gather + A_idx
                    # --- Phase 3.1: FP8 PreAct eliminates z dequant + 384MB temp ---
                    # When z is fp8 (from ctx), pass directly to GemmDGated.
                    # The kernel loads fp8 z + scales in its epilogue via EpiOp LDG,
                    # avoiding the standalone dequant kernel and z_bf16 allocation.
                    use_fp8_preact = (z is None and z_fp8 is not None)

                    if not use_fp8_preact:
                        # Standalone dequant (when z is already bf16) — all on default stream.
                        s_float = s if str(s.dtype) in ("torch.float32", "paddle.float32", "float32") else s.float()
                        if z is None:
                            z = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                            del z_fp8, z_raw_scales_u8
                    else:
                        s_float = s if str(s.dtype) in ("torch.float32", "paddle.float32", "float32") else s.float()

                    if ctx._w2_decoupled:
                        w2_fp8_enk = ctx._w2_dgated_fp8
                        w2_scales = ctx._w2_dgated_scales
                    else:
                        w2_fp8_enk, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)
                    config = gemm_dgated.default_config(dout.device, num_experts=w2_fp8_enk.shape[0])
                    total_m = x_gather_idx.shape[0]  # TK
                    n = w2_fp8_enk.shape[-2]
                    dz = torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=dout.device)
                    use_dz_epi_fuse = os.environ.get("SONIC_MOE_DZ_EPI_FUSE", "0") == "1"
                    use_dz_epi_fuse = use_dz_epi_fuse and use_fp8_preact and ctx._fp8_cfg.fp8_wgrad
                    y1s_col_epi = os.environ.get("SONIC_MOE_DGATED_Y1S_COL_EPI", "0") == "1"
                    y1s_col_epi = y1s_col_epi and use_fp8_preact and ctx._fp8_cfg.fp8_wgrad and not use_dz_epi_fuse
                    y1s_dtype = torch.float8_e4m3fn if y1s_col_epi else torch.bfloat16
                    y1s = torch.empty((total_m, n), dtype=y1s_dtype, device=dout.device)
                    colvec_reduce_partial = torch.empty(
                        (total_m, (n + config.tile_n - 1) // config.tile_n),
                        dtype=torch.float32,
                        device=dout.device,
                    )

                    K_bwd = dout.shape[1]
                    combine_grad_data = (
                        fp8_combine_grad_handle.get("data")
                        if fp8_combine_grad_handle is not None else None
                    )
                    combine_grad_scale = (
                        fp8_combine_grad_handle.get("scale")
                        if fp8_combine_grad_handle is not None else None
                    )
                    dout_has_comm_fp8_payload = combine_grad_data is not None and combine_grad_scale is not None
                    if dout_has_comm_fp8_payload:
                        if str(combine_grad_scale.dtype) in ("torch.uint8", "paddle.uint8", "uint8"):
                            dout_raw_scales_t = combine_grad_scale
                        elif str(combine_grad_scale.dtype) in ("torch.int32", "paddle.int32", "int32"):
                            dout_raw_scales_t = combine_grad_scale
                        else:
                            dout_raw_scales_t = combine_grad_scale.to(device=combine_grad_scale.device, dtype=torch.uint8)
                        if not _is_fp8_e4m3_dtype(combine_grad_data.dtype):
                            raise RuntimeError("Sonic FP8 combine backward requires FP8 combine grad payload")
                        dout_comm_fp8 = combine_grad_data
                        dout_fp8 = dout_comm_fp8
                        if not ctx._fp8_cfg.fp8_wgrad:
                            dout_dequant_scales = _raw_1x32_scale_bytes(dout_raw_scales_t)
                            dout = dequantize_blockscaled_fp8(dout_comm_fp8, dout_dequant_scales)
                    else:
                        dout_fp8, dout_raw_scales_t = quantize_activation_blockscaled_fast(dout)
                    source_rows_bwd = int(dout_fp8.shape[0])
                    dout_scales = _gather_1x32_scales_to_isa(
                        dout_raw_scales_t,
                        x_gather_idx,
                        source_rows_bwd,
                        K_bwd,
                    )

                    dz_epi_fp8 = None
                    dz_epi_row_scales = None
                    dz_epi_col_scales = None
                    dz_epi_row_scales_3d = None
                    dz_epi_col_scales_3d = None
                    if use_dz_epi_fuse:
                        dz_dim = n * 2
                        dz_epi_fp8 = torch.empty(
                            (total_m, dz_dim), dtype=torch.float8_e4m3fn, device=dout.device,
                        )
                        dz_row_storage = _storage_per_batch(total_m, dz_dim)
                        dz_col_storage = _storage_per_batch(dz_dim, total_m)
                        dz_row_aligned = total_m % _SF_TILE_M == 0 and dz_dim % _SF_TILE_K == 0
                        dz_col_aligned = dz_dim % _SF_TILE_M == 0 and total_m % _SF_TILE_K == 0
                        dz_epi_row_scales = (
                            torch.empty((1, dz_row_storage), dtype=torch.uint8, device=dout.device)
                            if dz_row_aligned else
                            torch.full((1, dz_row_storage), 127, dtype=torch.uint8, device=dout.device)
                        )
                        dz_epi_col_scales = (
                            torch.empty((1, dz_col_storage), dtype=torch.uint8, device=dout.device)
                            if dz_col_aligned else
                            torch.full((1, dz_col_storage), 127, dtype=torch.uint8, device=dout.device)
                        )
                        dz_epi_row_scales_3d = dz_epi_row_scales.view(_div_up(total_m, 128), _div_up(dz_dim, 128), 512)
                        dz_epi_col_scales_3d = dz_epi_col_scales.view(_div_up(dz_dim, 128), _div_up(total_m, 128), 512)

                    # ── y1s colwise epilogue quant (TE cudnn quant_sfd_col pattern) ──
                    y1s_col_scales_epi = None
                    if y1s_col_epi:
                        num_n_tiles = _div_up(n, 128)
                        col_k_tiles = _div_up(total_m, 128)
                        y1s_col_scales_epi = torch.empty(
                            (num_n_tiles, col_k_tiles, 512), dtype=torch.uint8,
                            device=dout.device,
                        )

                    gemm_dgated_kernel(
                        dout_fp8,
                        w2_fp8_enk,
                        dz,
                        z if not use_fp8_preact else dz,
                        y1s,
                        None,
                        "swiglu",
                        config.tile_m,
                        config.tile_n,
                        config.cluster_m,
                        config.cluster_n,
                        config.pingpong,
                        persistent=True,
                        max_swizzle_size=config.max_swizzle_size,
                        colvec_scale=s_float,
                        colvec_reduce=colvec_reduce_partial,
                        cu_seqlens_m=expert_frequency_offset,
                        A_idx=x_gather_idx,
                        a_scales=dout_scales,
                        b_scales=w2_scales,
                        preact_fp8=z_fp8 if use_fp8_preact else None,
                        preact_scales=z_raw_scales_u8 if use_fp8_preact else None,
                        iso32_dz_fp8=dz_epi_fp8,
                        iso32_dz_row_scales=dz_epi_row_scales_3d,
                        iso32_dz_col_scales=dz_epi_col_scales_3d,
                        y1s_col_fp8=None,
                        y1s_col_scales=y1s_col_scales_epi,
                        swiglu_clamp_value=ctx._fp8_cfg.swiglu_clamp_value,
                    )
                    ds = colvec_reduce_partial.sum(dim=-1)
                    del dout_fp8, dout_scales, z, colvec_reduce_partial
                    # Release FP8 preact tensors from ctx (z_fp8 ~192 MiB + scales ~6 MiB).
                    # The dgated GEMM is done; these are no longer needed.
                    if use_fp8_preact:
                        del z_fp8, z_raw_scales_u8
                    del w2_fp8_enk, w2_scales

                    # Weight-grad: dw2 = dout.T @ y1s (per expert).
                    _log_stage_memory("backward:down-proj-dgated")
                    _reset_stage_memory_probe()
                    TK_wgrad = x_gather_idx.shape[0]
                    if ctx._fp8_cfg.fp8_wgrad:
                        # Memory-optimized wgrad pipeline (all main stream):
                        # Step 1: colwise(y1s) then del y1s to free 192 MiB
                        if y1s_col_epi:
                            y1s_col_fp8 = y1s
                            y1s_col_sc = y1s_col_scales_epi.reshape(1, -1).view(_E8M0_DTYPE)
                        else:
                            y1s_col_fp8, y1s_col_sc = colwise_quantize_and_pack(
                                y1s, logical_rows=y1s.shape[1], logical_cols=TK_wgrad,
                            )
                            del y1s

                        if use_dz_epi_fuse:
                            if dout_has_comm_fp8_payload:
                                dout_packed_scales_t = _ensure_isa_1x32_scales(
                                    dout_raw_scales_t,
                                    source_rows_bwd,
                                    K_bwd,
                                )
                                dout_col_fp8, dout_col_sc = dequant_colwise_quantize_and_pack_from_isa(
                                    dout_comm_fp8, dout_packed_scales_t,
                                    logical_rows=dout.shape[1], logical_cols=TK_wgrad,
                                    gather_idx=x_gather_idx,
                                )
                                del dout_packed_scales_t
                            else:
                                dout_col_fp8, dout_col_sc = colwise_quantize_and_pack(
                                    dout, logical_rows=dout.shape[1], logical_cols=TK_wgrad,
                                    gather_idx=x_gather_idx,
                                )
                            _PREQUANTIZED_SCALES["bwd"] = (
                                dz, dz_epi_fp8, dz_epi_row_scales.view(_E8M0_DTYPE),
                            )
                            _PREQUANTIZED_SCALES["bwd_col"] = (
                                dz_epi_fp8, dz_epi_col_scales.view(_E8M0_DTYPE),
                            )
                        else:
                            if dout_has_comm_fp8_payload:
                                if os.environ.get("SONIC_MOE_DZ_ISO32", "0") != "0":
                                    dz_fp8, dz_packed_scales, dz_col_scales = iso32_dual_quantize_varlen(
                                        dz, TK_wgrad, dz.shape[1]
                                    )
                                    dz_col_fp8 = dz_fp8
                                else:
                                    dz_fp8, dz_packed_scales, dz_col_fp8, dz_col_scales = dual_quantize_varlen(
                                        dz, TK_wgrad, dz.shape[1]
                                    )
                                dout_packed_scales_t = _ensure_isa_1x32_scales(
                                    dout_raw_scales_t,
                                    source_rows_bwd,
                                    K_bwd,
                                )
                                dout_col_fp8, dout_col_sc = dequant_colwise_quantize_and_pack_from_isa(
                                    dout_comm_fp8, dout_packed_scales_t,
                                    logical_rows=dout.shape[1], logical_cols=TK_wgrad,
                                    gather_idx=x_gather_idx,
                                )
                                del dout_packed_scales_t
                            else:
                                dz_fp8, dz_packed_scales, dz_col_fp8, dz_col_scales, \
                                    dout_col_fp8, dout_col_sc = fused_dual_colwise_quantize(
                                        dz, dout, x_gather_idx,
                                        TK_wgrad, dz.shape[1], dout.shape[1],
                                    )
                            _PREQUANTIZED_SCALES["bwd"] = (dz, dz_fp8, dz_packed_scales)
                            _PREQUANTIZED_SCALES["bwd_col"] = (dz_col_fp8, dz_col_scales)

                        # Fused wgrad accumulation (same as w1 path)
                        _wgrad_accum_w2 = getattr(ctx, '_wgrad_w2_accumulator', None)
                        if _wgrad_accum_w2 is not None:
                            if _use_wgrad_beta_accum():
                                _run_cutlass_blockscaled_gemm_varlen_k_accumulate(
                                    dout_col_fp8, dout_col_sc,
                                    y1s_col_fp8, y1s_col_sc,
                                    expert_frequency_offset,
                                    M=dout.shape[1], N=w2_shape[1],
                                    total_K=TK_wgrad, num_experts=w2_shape[2],
                                    device=dout.device,
                                    accumulator=_wgrad_accum_w2,
                                )
                            else:
                                _run_cutlass_blockscaled_gemm_varlen_k_tma_add(
                                    dout_col_fp8, dout_col_sc,
                                    y1s_col_fp8, y1s_col_sc,
                                    expert_frequency_offset,
                                    M=dout.shape[1], N=w2_shape[1],
                                    total_K=TK_wgrad, num_experts=w2_shape[2],
                                    device=dout.device,
                                    accumulator=_wgrad_accum_w2,
                                )
                            dw2_base = None
                            dw2 = None
                        else:
                            dw2_base = _run_cutlass_blockscaled_gemm_varlen_k(
                                dout_col_fp8, dout_col_sc,
                                y1s_col_fp8, y1s_col_sc,
                                expert_frequency_offset,
                                M=dout.shape[1], N=w2_shape[1],
                                total_K=TK_wgrad, num_experts=w2_shape[2],
                                out_dtype=w2_dtype, device=dout.device,
                            )
                            dw2 = dw2_base.permute(1, 2, 0)
                        del dout_col_fp8, dout_col_sc, y1s_col_fp8, y1s_col_sc
                    else:
                        _wgrad_accum_w2 = getattr(ctx, '_wgrad_w2_accumulator', None)
                        if _wgrad_accum_w2 is not None:
                            # BF16 wgrad + fp32 accumulate.
                            # _wgrad_accum_w2: [E, H, I] fp32.
                            # GEMM out: [E, H, I] — same layout, no permute needed.
                            y1s_wgrad = (
                                y1s if y1s.dtype == torch.bfloat16 else y1s.to(torch.bfloat16)
                            )
                            if _use_wgrad_beta_accum():
                                bf16_wgrad_gemm_varlen_k_accumulate(
                                    dout.T,
                                    y1s_wgrad,
                                    expert_frequency_offset,
                                    x_gather_idx,
                                    accumulator=_wgrad_accum_w2,
                                    M=dout.shape[1],
                                    N=w2_shape[1],
                                    total_K=TK_wgrad,
                                    num_experts=w2_shape[2],
                                    device=dout.device,
                                )
                            else:
                                bf16_wgrad_gemm_varlen_k_tma_add(
                                    dout.T,
                                    y1s_wgrad,
                                    expert_frequency_offset,
                                    x_gather_idx,
                                    accumulator=_wgrad_accum_w2,
                                    M=dout.shape[1],
                                    N=w2_shape[1],
                                    total_K=TK_wgrad,
                                    num_experts=w2_shape[2],
                                    device=dout.device,
                                )
                            del y1s_wgrad
                            del y1s
                            dw2_base = None
                            dw2 = None
                        else:
                            dw2_base = torch.empty((w2_shape[2], w2_shape[0], w2_shape[1]), dtype=w2_dtype, device=w2_device)
                            dw2 = dw2_base.permute(1, 2, 0)
                            y1s_wgrad = y1s if y1s.dtype == torch.bfloat16 else y1s.to(torch.bfloat16)
                            bf16_wgrad_gemm_varlen_k(
                                dout.T,
                                y1s_wgrad,
                                expert_frequency_offset,
                                x_gather_idx,
                                out=dw2.permute(2, 0, 1),
                                M=dout.shape[1],
                                N=w2_shape[1],
                                total_K=TK_wgrad,
                                num_experts=w2_shape[2],
                                device=dout.device,
                            )
                            del y1s_wgrad
                            del y1s
                    _log_stage_memory("backward:down-proj-weight")

                    # Pre-quantize dz for UpProj.backward (non-wgrad path only;
                    # wgrad path already did this above before dw2 allocation).
                    if not ctx._fp8_cfg.fp8_wgrad:
                        dz_fp8, dz_packed_scales = quantize_and_pack_activation(dz)
                        _PREQUANTIZED_SCALES["bwd"] = (dz, dz_fp8, dz_packed_scales)
                    ds = _gather_router_scores_i32(ds, s_reverse_scatter_idx)
                else:
                    w2_actgrad = w2.permute(1, 0, 2)  # (I, H, E)
                    w2_fp8, w2_scales = precompute_weight_fp8(w2_actgrad)

                    dout_fp8, dout_scales = fast_gather_quantize_and_pack_activation(
                        dout, x_gather_idx
                    )
                    dy1 = blockscaled_fp8_gemm_varlen(
                        dout_fp8, w2_actgrad, expert_frequency_offset,
                        a_scales=dout_scales,
                        w_fp8=w2_fp8, w_scales=w2_scales,
                        out_dtype=torch.bfloat16,
                        assume_aligned=ctx._alignment_assumed_flag,
                    )
                    del dout_fp8, dout_scales
                    # Eagerly release w2 FP8 cache (~37 MiB) — actgrad GEMM done.
                    del w2_fp8, w2_scales
                    # Keep w2 varlen cache — avoids re-quant on next iter.

                    # Step 3: SwiGLU backward
                    if z_fp8 is not None:
                        if ctx._fp8_cfg.fused_swiglu_quant:
                            # Decomposed path (faster than fully-fused):
                            # 1. Dequant z_fp8 -> z_bf16  (~0.046ms, BLOCK_ROWS=16)
                            # 2. dSwiGLU + quant + ISA-pack + dz_bf16  (~0.36ms, single kernel)
                            # Total ~0.41ms vs fused 0.47ms (12% faster)
                            if ctx._fp8_cfg.swiglu_clamp_value > 0.0:
                                z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                                dz, y1s, ds = _swiglu_backward_clamp_reference(
                                    dy1, z_bf16, s, ctx._fp8_cfg.swiglu_clamp_value
                                )
                                del z_bf16
                                dz_fp8, dz_packed_scales = quantize_and_pack_activation(dz)
                            else:
                                if swiglu_backward_quant_pack_triton is None:
                                    raise RuntimeError("fused SwiGLU backward quant helper is unavailable")
                                z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                                dz_fp8, dz_packed_scales, y1s, ds, dz = (
                                    swiglu_backward_quant_pack_triton(
                                        dy1, z_bf16, s, return_dz_bf16=True
                                    )
                                )
                                del z_bf16
                            _PREQUANTIZED_SCALES["bwd"] = (dz, dz_fp8, dz_packed_scales)
                        else:
                            # Fused: read fp8 z directly, skip bf16 materialization
                            if ctx._fp8_cfg.swiglu_clamp_value > 0.0:
                                z_bf16 = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                                dz, y1s, ds = _swiglu_backward_clamp_reference(
                                    dy1, z_bf16, s, ctx._fp8_cfg.swiglu_clamp_value
                                )
                                del z_bf16
                            else:
                                dz, y1s, ds = swiglu_backward_from_fp8_triton(
                                    dy1, z_fp8, z_raw_scales_u8, s
                                )
                        del z_fp8, z_raw_scales_u8
                    else:
                        if ctx._fp8_cfg.swiglu_clamp_value > 0.0:
                            dz, y1s, ds = _swiglu_backward_clamp_reference(
                                dy1, z, s, ctx._fp8_cfg.swiglu_clamp_value
                            )
                        else:
                            dz, y1s, ds = _swiglu_backward_interleaved(dy1, z, s)
                    del dy1

                    _log_stage_memory("backward:down-proj-dgated")
                    _reset_stage_memory_probe()

                    # Weight-grad: BF16 varlen GEMM
                    dw2_base = torch.empty((w2_shape[2], w2_shape[0], w2_shape[1]), dtype=w2_dtype, device=w2_device)
                    dw2 = dw2_base.permute(1, 2, 0)
                    y1s_wgrad = y1s if y1s.dtype == torch.bfloat16 else y1s.to(torch.bfloat16)
                    bf16_wgrad_gemm_varlen_k(
                        dout.T,
                        y1s_wgrad,
                        expert_frequency_offset,
                        x_gather_idx,
                        out=dw2.permute(2, 0, 1),
                        M=dout.shape[1],
                        N=w2_shape[1],
                        total_K=x_gather_idx.shape[0],
                        num_experts=w2_shape[2],
                        device=dout.device,
                    )
                    del y1s_wgrad
                    _log_stage_memory("backward:down-proj-weight")
                    ds = _gather_router_scores_i32(ds, s_reverse_scatter_idx)
            elif ctx._fp8_enabled_flag and not ctx._alignment_assumed_flag:
                # FP8 enabled but non-aligned: unreachable in production
                # (callers must use token rounding for 128-alignment).
                raise RuntimeError(
                    f"FP8 blockscaled backward requires 128-aligned expert segments. "
                    f"Got non-aligned cu_seqlens. Use token rounding in the router "
                    f"to ensure each expert receives a multiple of 128 tokens."
                )
            else:
                # BF16 path (fp8 disabled): standard gemm_dgated, no alignment req.
                if z is None:
                    z = dequantize_blockscaled_fp8(z_fp8, z_raw_scales_u8)
                    del z_fp8, z_raw_scales_u8
                dz = torch.empty_like(z)
                _, y1s, ds = gemm_dgated(
                    dout,
                    w2.permute(2, 0, 1),
                    PreAct=z,
                    activation="swiglu",
                    dx_out=dz,
                    colvec_scale=s.float(),
                    colvec_reduce=True,
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    dynamic_scheduler=False,
                    tuned=False,
                    swiglu_clamp_value=ctx._fp8_cfg.swiglu_clamp_value,
                )
                _log_stage_memory("backward:down-proj-dgated")
                _reset_stage_memory_probe()

                y1s_wgrad = y1s.to(torch.bfloat16) if y1s.dtype == torch.float8_e4m3fn else y1s
                _wgrad_accum_w2 = getattr(ctx, '_wgrad_w2_accumulator', None)
                if _wgrad_accum_w2 is not None:
                    if _use_wgrad_beta_accum():
                        bf16_wgrad_gemm_varlen_k_accumulate(
                            dout.T,
                            y1s_wgrad,
                            expert_frequency_offset,
                            x_gather_idx,
                            accumulator=_wgrad_accum_w2,
                            M=dout.shape[1],
                            N=w2.shape[1],
                            total_K=x_gather_idx.shape[0],
                            num_experts=w2.shape[2],
                            device=dout.device,
                        )
                    else:
                        bf16_wgrad_gemm_varlen_k_tma_add(
                            dout.T,
                            y1s_wgrad,
                            expert_frequency_offset,
                            x_gather_idx,
                            accumulator=_wgrad_accum_w2,
                            M=dout.shape[1],
                            N=w2.shape[1],
                            total_K=x_gather_idx.shape[0],
                            num_experts=w2.shape[2],
                            device=dout.device,
                        )
                    dw2 = None
                else:
                    dw2_base = torch.empty((w2.shape[2], w2.shape[0], w2.shape[1]), dtype=w2.dtype, device=w2.device)
                    dw2 = dw2_base.permute(1, 2, 0)
                    bf16_wgrad_gemm_varlen_k(
                        dout.T,
                        y1s_wgrad,
                        expert_frequency_offset,
                        x_gather_idx,
                        out=dw2.permute(2, 0, 1),
                        M=dout.shape[1],
                        N=w2.shape[1],
                        total_K=x_gather_idx.shape[0],
                        num_experts=w2.shape[2],
                        device=dout.device,
                    )
                _log_stage_memory("backward:down-proj-weight")
                ds = _gather_router_scores_i32(ds, s_reverse_scatter_idx)
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        _reset_stage_memory_probe()
        y1s = None  # may already be freed by fused dgated path
        _log_stage_memory("backward:down-proj-postact-release")
        # TC top-K routing
        # When route-level padding is active, topk_scores input was (T*K+N_pad,)
        # flat, but ds is (T*K,) after s_reverse_scatter_idx indexing.  Pad with
        # zeros so gradient shape matches input shape.
        if ctx._score_src_idx is not None:
            score_src_idx = ctx._score_src_idx
            n_scores = topk_scores.size if isinstance(topk_scores.size, int) else topk_scores.numel()
            ds = _scatter_router_scores_i32(ds, score_src_idx, n_scores)
            if len(ctx._topk_scores_shape) != 1:
                ds = ds.view(ctx._topk_scores_shape)
        else:
            N_scores = topk_scores.shape[0]
            if ds.shape[0] < N_scores:
                ds = torch.cat([ds, torch.zeros(
                    N_scores - ds.shape[0], dtype=ds.dtype, device=ds.device
                )])
            elif not is_varlen_K:
                ds = ds.view(T, K)

        # Paddle PyLayer: return grads only for tensor inputs (not int/bool/enum/None)
        # Tensor inputs: y1, z, w2, [b2], topk_scores, selected_experts, expert_frequency_offset,
        #   x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, [num_activated_expert_per_token_offset]
        if fp8_combine_grad_handle is not None:
            fp8_combine_grad_handle.pop("data", None)
            fp8_combine_grad_handle.pop("scale", None)
        grads = [None, dz, dw2]  # y1, z, w2
        if ctx._has_b2:
            grads.append(db2)
        grads.extend([ds if ctx._topk_scores_needs_grad else None, None, None])  # topk_scores, selected_experts, expert_frequency_offset
        grads.extend([None, None, None])  # x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
        if ctx._has_num_activated:
            grads.append(None)
        if ctx._has_topk_scores_expert_order:
            grads.append(None)
        if ctx._has_topk_scores_token_order:
            grads.append(None)
        if ctx._has_score_src_idx:
            grads.append(None)
        return tuple(grads)


def _moe_tc_softmax_topk_layer_quack_inference(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType,
    fp8_protocol: FP8Protocol | None,
    use_low_precision_postact_buffer: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E = router_w.size(0)
    T = x.size(0)
    H = w2.size(0)
    TK = T * K
    device = x.device

    with torch.no_grad():
        _reset_stage_memory_probe()
        router_logits = F.linear(x, router_w)
        topk_scores = torch.empty(T, K, dtype=torch.float32, device=device)
        topk_indices = torch.empty(T, K, dtype=torch.int32, device=device)
        _softmax_topk_fwd(router_logits, topk_scores, topk_indices, E, K)

        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
        )
        _log_stage_memory("forward:router-metadata")

        needs_preact = fp8_protocol is not None and _upproj_epilogue_precision() == "fp8"
        # Inference routing is independent of training: always do a real
        # alignment check instead of trusting _ALIGNMENT_ASSUMED.
        if _fp8_enabled() and _use_fused_blockscaled_gated():
            _vals = _get_cu_seqlens_cpu(expert_frequency_offset)
            aligned = all((_vals[i + 1] - _vals[i]) % 128 == 0 for i in range(len(_vals) - 1))
        else:
            aligned = False
        if _fp8_enabled() and _use_fused_blockscaled_gated() and aligned:
            w1_fp8, w1_scales = _get_fp8_weight_attr(w1, "fp8")
            x_fp8, x_scales = fast_gather_quantize_and_pack_activation(x, x_gather_idx)
            z, y1 = gemm_gated(
                x_fp8,
                w1_fp8,
                activation="swiglu",
                out_dtype=torch.bfloat16,
                postact_dtype=torch.bfloat16,
                cu_seqlens_m=expert_frequency_offset,
                dynamic_scheduler=False,
                a_scales=x_scales,
                b_scales=w1_scales,
                tuned=False,
                swiglu_clamp_value=_get_fp8_config().swiglu_clamp_value,
            )
            del x_fp8, x_scales
        elif _fp8_enabled():
            x_fp8 = x if x.dtype == torch.float8_e4m3fn else x.to(torch.float8_e4m3fn)
            w1_fp8 = _get_cached_fp8_weight(w1, "w1_ekh")
            z, y1 = gemm_gated(
                x_fp8,
                w1_fp8,
                activation="swiglu",
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                out_dtype=torch.bfloat16,
                postact_dtype=torch.float8_e4m3fn,
                store_preact=needs_preact,
                dynamic_scheduler=False,
                swiglu_clamp_value=_get_fp8_config().swiglu_clamp_value,
            )
        else:
            z, y1 = gemm_gated(
                x,
                w1.permute(2, 1, 0),
                activation="swiglu",
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                postact_dtype=(torch.float8_e4m3fn if use_low_precision_postact_buffer else None),
                store_preact=needs_preact,
                dynamic_scheduler=False,
                swiglu_clamp_value=_get_fp8_config().swiglu_clamp_value,
            )
        _log_stage_memory("forward:up-proj")

        # In full-pipeline FP8, y1 stays fp8 for down-proj.
        if _fp8_enabled() and not needs_preact:
            pass  # y1 stays fp8
        elif _fp8_enabled() and needs_preact:
            # Preact path with fp8 enabled: skip dequant round-trip
            if y1.dtype != x.dtype:
                y1 = y1.to(x.dtype)

        if needs_preact:
            _reset_stage_memory_probe()
            if _fp8_enabled():
                # y1 was computed via FP8 tensor cores; convert to bf16 and
                # skip the quant->dequant round-trip.
                if y1.dtype != x.dtype:
                    y1 = y1.to(x.dtype)
            else:
                restored_out = None
                if y1.size(-1) % fp8_protocol.group_size == 0:
                    if use_low_precision_postact_buffer:
                        restored_out = torch.empty(y1.shape, dtype=x.dtype, device=device)
                    else:
                        restored_out = y1
                y1, _ = apply_preact_activation_fp8_protocol_cutely_fused(
                    z,
                    None,
                    fp8_protocol,
                    quack_enabled=True,
                    return_scales=False,
                    use_ste=False,
                    restored_out=restored_out,
                    output_dtype=x.dtype,
                )
            _log_stage_memory("forward:fp8-boundary")

        del z
        _reset_stage_memory_probe()
        if fp8_protocol is not None and _use_blockscaled_fp8_downproj():
            y2 = blockscaled_fp8_gemm_grouped(
                y1,
                w2,
                expert_frequency_offset,
                protocol=fp8_protocol,
            )
            router_perm = make_blockscaled_grouped_reverse_scatter_idx(
                s_reverse_scatter_idx,
                expert_frequency_offset,
                expert_ids=topk_indices.reshape(-1),
            )
            y2_for_router = y2.view(-1, H)
        else:
            if _fp8_enabled() and _use_fused_blockscaled_gated() and aligned:
                y1_fp8, y1_scales = quantize_and_pack_activation(y1)
                w2_fp8, w2_scales = _get_fp8_weight_attr(w2, "fp8")
                y2 = blockscaled_fp8_gemm_varlen(
                    y1_fp8, w2, expert_frequency_offset,
                    a_scales=y1_scales,
                    w_fp8=w2_fp8, w_scales=w2_scales,
                    out_dtype=torch.bfloat16,
                    assume_aligned=True,
                )
            elif _fp8_enabled():
                y1_fp8 = y1 if y1.dtype == torch.float8_e4m3fn else y1.to(torch.float8_e4m3fn)
                w2_fp8 = _get_fp8_weight_orig(w2)
                y2 = gemm(y1_fp8, w2_fp8.permute(2, 1, 0),
                          cu_seqlens_m=expert_frequency_offset,
                          out_dtype=torch.bfloat16)
            else:
                y2 = gemm(y1, w2.permute(2, 1, 0), cu_seqlens_m=expert_frequency_offset)
            router_perm = s_reverse_scatter_idx
            y2_for_router = y2

        del y1
        o = torch.empty(T, H, device=device, dtype=y2_for_router.dtype)
        topk_scores = topk_scores.flatten()
        _router_forward(
            y2=y2_for_router,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=router_perm,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K,
            H=H,
            is_varlen_K=False,
        )
        _log_stage_memory("forward:down-proj-router")

    return o, router_logits, expert_frequency


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    fp8_protocol: FP8Protocol | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    _validate_runtime_precision_switches(fp8_protocol)
    # Resolve all FP8 flags once at entry — eliminates repeated os.getenv in hot path.
    _refresh_fp8_config()
    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    use_low_precision_postact_buffer = False
    if is_inference_mode_enabled and is_using_quack_gemm():
        return _moe_tc_softmax_topk_layer_quack_inference(
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            K,
            stream_id,
            activation_type,
            fp8_protocol,
            use_low_precision_postact_buffer,
        )

    E = router_w.size(0)
    _reset_stage_memory_probe()
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, E, K)

    T, K = topk_indices.size()
    TK = T * K
    device = topk_indices.device

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
    )
    _log_stage_memory("forward:router-metadata")

    T = x.size(0)

    # ── Route-level padding for FP8 non-aligned expert segments ──────────
    # Pad routing metadata once so _all_segments_128_aligned sees aligned
    # offsets → entire fwd+bwd runs the proven aligned fast path.
    # Padding rows gather from row 0 with score=0 → zero contribution.
    # x is NOT modified (no sentinel row).
    if _fp8_enabled():
        (expert_frequency_offset, x_gather_idx, s_scatter_idx,
         s_reverse_scatter_idx, topk_scores_flat, TK, _routing_padded
        ) = _pad_routing_metadata(
            expert_frequency_offset, x_gather_idx, s_scatter_idx,
            s_reverse_scatter_idx, topk_scores.flatten(), TK, T, E, K,
        )
        if _routing_padded:
            topk_scores = topk_scores_flat  # now (T*K+N_pad,) flat

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
        use_low_precision_postact_buffer,
        None,
    )
    _log_stage_memory("forward:up-proj")

    if fp8_protocol is not None and _upproj_epilogue_precision() == "fp8":
        _reset_stage_memory_probe()
        cfg = _get_fp8_config()
        if cfg.enabled and cfg.fused_gated and cfg.alignment_assumed and is_using_quack_gemm():
            # Blockscaled FP8 path: y1 was already quantized inside _UpProjection
            # (prequant cache holds fp8+scales).  Skip the adapter's quant->dequant
            # round-trip which costs ~250µs and is redundant.
            pass
        elif cfg.alignment_assumed and is_using_quack_gemm():
            # Aligned non-fused-gated path: cutify's fused SwiGLU+quant expects
            # z in stacked [gate|value] layout.  Both blockscaled_fp8_gemm_varlen
            # and fused_gated produce z compatible with this convention.
            restored_out = None
            if y1.size(-1) % fp8_protocol.group_size == 0:
                if use_low_precision_postact_buffer:
                    restored_out = torch.empty(y1.shape, dtype=z.dtype, device=z.device)
                else:
                    restored_out = y1
            with torch.no_grad():
                y1, _ = apply_preact_activation_fp8_protocol_cutely_fused(
                    z,
                    None,
                    fp8_protocol,
                    quack_enabled=True,
                    return_scales=False,
                    use_ste=False,
                    restored_out=restored_out,
                    output_dtype=z.dtype,
                )
        elif is_using_quack_gemm():
            # Unaligned QuACK path: up-proj used padded FP8 zero-mat, producing
            # z/y1 in interleaved layout.  Down-proj will use
            # blockscaled_fp8_gemm_varlen(assume_aligned=False) which handles
            # padding internally from bf16 y1.  No adapter quant needed.
            pass
        else:
            y1, _ = apply_activation_fp8_protocol(
                y1,
                fp8_protocol,
                quack_enabled=False,
                return_scales=False,
                use_ste=not is_inference_mode_enabled,
            )
        _log_stage_memory("forward:fp8-boundary")

    # ── Memory optimization: eagerly release forward transients ──────────
    # z bf16 and y1 bf16 storage was already freed inside _UpProjection
    # via untyped_storage().resize_(0).  Keep ALL weight caches (version-keyed,
    # auto-invalidate at optimizer step).  Session 53 analysis: clearing FUSED
    # cache saved ~74 MiB but cost ~980µs/iter at E=128 from weight re-quant.
    # if _get_fp8_config().enabled:
    #     clear_fused_weight_cache()

    _reset_stage_memory_probe()
    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        topk_scores,
        topk_indices,
        expert_frequency_offset,
        T,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_varlen_K
        activation_type,
        fp8_protocol,
        None,
    )
    _log_stage_memory("forward:down-proj-router")

    return o, router_logits, expert_frequency


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Weight format requirements:
# - w1_weight: Shape (2*I, H, E), stride order (2, 0, 1), must be interleaved [gate_row0, up_row0, gate_row1, up_row1, ...]
# - w2_weight: Shape (H, I, E), stride order (2, 0, 1)


# We assume token_indices is already SORTED ascendingly !!!
#   and len(token_indices) = len(expert_indices) = len(router_scores)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    _refresh_fp8_config()

    T = x.size(0)
    TK = router_scores.size(0)
    E = w2.size(-1)
    (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = general_routing_router_metadata(router_scores, token_indices, expert_indices, T, E)

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
        False,  # use_low_precision_postact_buffer
    )

    # ── Eagerly release forward transients (same as moe_TC_softmax_topk_layer) ──
    # z/y1 bf16 storage freed inside _UpProjection.
    # NOTE: keep FUSED weight cache alive (version-keyed, auto-invalidates at
    # optimizer step). Clearing it saved ~74 MiB but cost ~980µs/iter at E=128
    # from weight re-quantization.
    # if _fp8_enabled() and _ALIGNMENT_ASSUMED:
    #     clear_fused_weight_cache()

    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        router_scores,
        expert_indices,
        expert_frequency_offset,
        T,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_varlen_K
        activation_type,
        None,
    )

    return o, expert_frequency
