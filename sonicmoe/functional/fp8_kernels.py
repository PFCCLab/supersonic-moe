# ********************************************************************************
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# ********************************************************************************

"""Device-dispatching FP8 kernel interface for the blockscaled MoE flow.

``functional/__init__.py`` runs one FP8 flow for every GPU and calls only the
ops below.  Each op forwards to the backend chosen once, on first use, from
the current CUDA device:

* SM90 (``quack_utils/sm90_funcs.py``): native WGMMA FP8 kernels, E4M3 + raw
  uint8 UE8M0 ``[rows, K/G]`` activation scales, GxG weight blocks
  ``[E, N/B, K/B]`` (``G == B`` in {32, 128}).
* SM100 (``quack_utils/sm100_funcs.py``): CUTLASS blockscaled kernels, E4M3 +
  ISA-packed UE8M0 1x32 scales, per-row 1x32 weight scales.

Both backends expose the same names, signatures and return arity.  An op a
backend does not implement raises ``NotImplementedError``.
"""

from __future__ import annotations

import torch

from ..quack_utils import sm90_funcs, sm100_funcs

_BACKEND = None


def _backend():
    # One GPU architecture per process: resolve once against the current
    # device on first use (not at import, CUDA may not be initialized yet).
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = sm90_funcs if sm90_funcs.is_sm90_device() else sm100_funcs
    return _BACKEND


# ---------------------------------------------------------------------------
# Config / layout
# ---------------------------------------------------------------------------


def validate_config(cfg, device) -> None:
    """Reject FP8 configs the device kernels cannot run."""
    return _backend().validate_config(cfg, device)


def segments_aligned(expert_frequency_offset: torch.Tensor) -> bool:
    """Whether every expert segment satisfies the FP8 kernels' M alignment."""
    return _backend().segments_aligned(expert_frequency_offset)


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


def quantize_activation(x: torch.Tensor, cfg):
    """Rowwise E4M3 ``(data, scales)`` in the layout consumed by this device's GEMMs."""
    return _backend().quantize_activation(x, cfg)


def quantize_activation_raw(x: torch.Tensor, cfg):
    """Rowwise E4M3 ``(data, raw_ue8m0_scales)`` (un-packed scales)."""
    return _backend().quantize_activation_raw(x, cfg)


def dequantize_activation(data: torch.Tensor, scales: torch.Tensor, cfg) -> torch.Tensor:
    """BF16 view of a rowwise E4M3 payload with raw UE8M0 scales."""
    return _backend().dequantize_activation(data, scales, cfg)


def quantize_preact(z: torch.Tensor):
    """Quantize the pre-activation z for backward: ``(z_fp8, raw_1x32_scales)``."""
    return _backend().quantize_preact(z)


def dequantize_preact(z_fp8: torch.Tensor, z_scales_u8: torch.Tensor) -> torch.Tensor:
    """BF16 z from the :func:`quantize_preact` payload (u8 raw 1x32 scales)."""
    return _backend().dequantize_preact(z_fp8, z_scales_u8)


def quantize_z_y1(z: torch.Tensor, y1: torch.Tensor, cfg):
    """Fused z (raw 1x32) + y1 (down-GEMM layout) quant: ``(z_fp8, z_scales, y1_fp8, y1_scales)``."""
    return _backend().quantize_z_y1(z, y1, cfg)


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


def precompute_weight_dgated(w2: torch.Tensor):
    """FP8 ``(w2_fp8, w2_scales)`` for :func:`gemm_dgated` from the ``(H, I, E)`` w2 view."""
    return _backend().precompute_weight_dgated(w2)


# ---------------------------------------------------------------------------
# GEMMs
# ---------------------------------------------------------------------------


def gemm_gated(
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

    Returns ``(z, y1, y1_fp8, y1_scales, z_fp8_payload)``. ``y1_fp8``/``y1_scales``
    are the down-projection operand (``None`` if the caller must quantize
    ``y1``). With ``recompute_z`` z is a zero-stride placeholder; with
    ``epilogue_quant`` + ``save_z_fp8`` z is a placeholder and
    ``z_fp8_payload`` is the E4M3 z with raw 1x32 scales (else ``None``); with a
    fused y1 quant y1 is a placeholder.
    """
    return _backend().gemm_gated(
        x, w1, expert_frequency_offset, x_gather_idx,
        x_fp8_pre=x_fp8_pre,
        w1_fp8=w1_fp8,
        cfg=cfg,
        current_stream=current_stream,
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
    """Up-projection for non-aligned expert segments (pads, runs, unpads): ``(z, y1)``."""
    return _backend().gemm_gated_padded(
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
    """Re-run the up-projection GEMM for backward (``recompute_z``).

    Arguments after ``x_gather_idx`` are the tuple stashed by the up-projection
    forward. Returns ``(z_fp8, z_raw_1x32_scales)`` if ``fp8`` else BF16 z.
    """
    return _backend().recompute_z(
        x, w1, expert_frequency_offset, x_gather_idx,
        x_fp8_pre, x_scales_pre, w1_fp8_pre,
        cfg=cfg, fp8=fp8,
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
    """Varlen grouped ``out = a @ w`` with BF16 output ``[rows, out_features]``.

    ``a_fp8``/``a_scales`` come from :func:`quantize_activation`; ``w_fp8`` is
    the K-major ``[E, out_features, K]`` weight. Used for the down-projection
    forward and the up-projection input gradient.
    """
    return _backend().gemm(
        a_fp8, a_scales, w_fp8, w_scales, expert_frequency_offset,
        out_features=out_features, num_experts=num_experts, cfg=cfg,
    )


def gemm_padded(x: torch.Tensor, w2: torch.Tensor, expert_frequency_offset: torch.Tensor):
    """Down-projection for non-aligned expert segments from BF16 ``x`` and the ``(H, I, E)`` w2 view."""
    return _backend().gemm_padded(x, w2, expert_frequency_offset)


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
    """Fused down-projection backward + activation backward.

    ``dout_fp8`` is T-sized with raw UE8M0 scales and is gathered by
    ``x_gather_idx`` inside the kernel. ``preact`` is BF16 z, or ``None`` when
    the FP8 ``preact_fp8``/``preact_scales`` pair is supplied.

    Returns ``(dz, y1s, ds)`` with ``ds`` the per-row router-score gradient.
    """
    return _backend().gemm_dgated(
        dout_fp8, dout_scales,
        w2_fp8=w2_fp8, w2_scales=w2_scales,
        preact=preact, preact_fp8=preact_fp8, preact_scales=preact_scales,
        colvec_scale=colvec_scale,
        expert_frequency_offset=expert_frequency_offset,
        x_gather_idx=x_gather_idx,
        activation=activation,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# FP8 weight gradient (varlen-K over the routed token dimension).
#
# A "wgrad operand" is an opaque 3-tuple of column-wise (token-dim) quantized
# data: SM100 ``(fp8, isa_scales, None)``; SM90 ``(fp8 [feature, P],
# u8 [feature, P/32], packed_offsets)``.
# ---------------------------------------------------------------------------


def quantize_wgrad_operand(
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
    return _backend().quantize_wgrad_operand(
        src, expert_frequency_offset, cfg,
        gather_idx=gather_idx, src_scales=src_scales,
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
    """Quantize the down-projection backward operands.

    Returns ``(dz_rowwise, dz_col, dout_col)``: ``dz_rowwise`` is the
    ``(data, scales)`` pair for the up-projection :func:`gemm`; the other two
    are wgrad operands (``dout`` / ``dout_fp8`` gathered by ``x_gather_idx``).
    """
    return _backend().quantize_dz_dout(
        dz, dout, expert_frequency_offset, x_gather_idx, cfg,
        dout_fp8=dout_fp8, dout_scales=dout_scales,
    )


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
    """Per-expert ``D[e] = A[:, seg_e] @ B[:, seg_e]^T`` of shape ``[E, M, N]``.

    With ``accumulator`` (fp32 ``[E, M, N]``) the result is added in place and
    ``None`` is returned; otherwise a fresh ``out_dtype`` tensor is returned.
    """
    return _backend().wgrad_gemm_varlen_k(
        a_col, b_col, expert_frequency_offset,
        M=M, N=N, total_K=total_K, num_experts=num_experts,
        device=device, accumulator=accumulator, out_dtype=out_dtype,
    )
