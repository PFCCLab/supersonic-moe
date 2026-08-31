# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import torch

from ..enums import ActivationType
from .fp8_protocol import FP8Protocol, validate_fp8_runtime_support
from .fp8_quant import dequantize_activation_blockwise, quantize_activation_blockwise, round_scale_to_e8m0


def _reject_situ_activation(activation_type, site: str) -> None:
    """Refuse SiTU-GLU: this module's fused kernel is hard-wired to SwiGLU.

    ``cutify.fused_weighted_swiglu_act_quant_best`` computes SwiGLU internally,
    so there is no way to express ``situ(g, beta) * up_act(u, lb)`` here.
    Silently returning a SwiGLU postact would be a wrong-activation bug, so
    raise instead.  Deliberately a cheap prefix check rather than an import of
    ``quack_utils.activation_situ`` (which pulls in cutlass at import time).
    """
    if activation_type is None:
        return
    is_situ = activation_type is ActivationType.SITU_GLU or (
        isinstance(activation_type, str)
        and activation_type.split(":", 1)[0] == ActivationType.SITU_GLU.value
    )
    if is_situ:
        raise NotImplementedError(
            f"sonicmoe/functional/fp8_cutely_fused.py [{site}]: the fused "
            "cutify preact path is SwiGLU-only "
            "(fused_weighted_swiglu_act_quant_best) and cannot express "
            "situ_glu. Disable the fused SwiGLU quant adapter "
            "(SonicMoEConfig(fused_swiglu_quant=False) / "
            "SONIC_MOE_FP8_FUSED_SWIGLU_QUANT=0) or run the fused gated "
            "CUTLASS epilogue path (fused_gated=True), which computes the "
            "SiTU epilogue natively. Falling back to swiglu is deliberately "
            "not done."
        )


def _get_cutify():
    """Lazy import cutify — only needed for non-aligned preact path."""
    try:
        from cutify import fused_act_dequant_best, fused_weighted_swiglu_act_quant_best
    except ImportError:
        raise ImportError(
            "cutify is required for the non-aligned preact FP8 path. "
            "Install it or ensure all expert segments are 128-aligned "
            "(which bypasses cutify via the Triton-based quant kernels)."
        )
    return fused_weighted_swiglu_act_quant_best, fused_act_dequant_best


def _pad_postact_tensor(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    original_last_dim = x.size(-1)
    remainder = original_last_dim % group_size
    if remainder == 0:
        return x, original_last_dim

    padded_last_dim = original_last_dim + (group_size - remainder)
    padding = torch.zeros(*x.shape[:-1], padded_last_dim - original_last_dim, device=x.device, dtype=x.dtype)
    return torch.cat([x, padding], dim=-1), original_last_dim


def _pad_preact_for_group_size(preact: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    if preact.ndim != 2:
        raise ValueError(f"expected preact to be 2D, got shape {tuple(preact.shape)}")
    if preact.size(-1) % 2 != 0:
        raise ValueError(f"expected even preact width for SwiGLU, got {preact.size(-1)}")

    postact_width = preact.size(-1) // 2
    padded_postact_width = ((postact_width + group_size - 1) // group_size) * group_size
    if padded_postact_width == postact_width:
        return preact.contiguous(), postact_width

    padded_preact_width = padded_postact_width * 2
    padding = torch.zeros(
        preact.size(0),
        padded_preact_width - preact.size(-1),
        device=preact.device,
        dtype=preact.dtype,
    )
    return torch.cat([preact, padding], dim=-1).contiguous(), postact_width


def apply_activation_fp8_protocol_cutely_fused(
    x: torch.Tensor,
    protocol: FP8Protocol | None = None,
    quack_enabled: bool | None = None,
    return_scales: bool = True,
    use_ste: bool = True,
    scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    protocol = validate_fp8_runtime_support(protocol or FP8Protocol(), x.device, quack_enabled=quack_enabled)
    padded_x, original_last_dim = _pad_postact_tensor(x, protocol.group_size)
    data, scales = quantize_activation_blockwise(padded_x, protocol, scale_out=scale_out)
    dequantized = dequantize_activation_blockwise(data, scales, protocol, output_dtype=x.dtype)[..., :original_last_dim]

    restored_with_ste = x + (dequantized - x).detach() if use_ste else dequantized
    return (
        restored_with_ste,
        scales[..., : ((original_last_dim + protocol.group_size - 1) // protocol.group_size)] if return_scales else None,
    )


def apply_preact_activation_fp8_protocol_cutely_fused(
    preact: torch.Tensor,
    postact: torch.Tensor | None,
    protocol: FP8Protocol | None = None,
    quack_enabled: bool | None = None,
    return_scales: bool = True,
    use_ste: bool = True,
    scale_out: torch.Tensor | None = None,
    restored_out: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    activation_type=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # The fused branch below computes SwiGLU inside cutify; refuse situ_glu
    # before doing any work.  The group_size != 128 fallback is
    # activation-agnostic (it only quantizes an already-computed postact), but
    # this call cannot know which branch a caller wanted, so reject up front.
    _reject_situ_activation(
        activation_type, "apply_preact_activation_fp8_protocol_cutely_fused"
    )
    protocol = validate_fp8_runtime_support(protocol or FP8Protocol(), preact.device, quack_enabled=quack_enabled)
    if output_dtype is None:
        if postact is None:
            raise ValueError("output_dtype must be provided when postact is None")
        output_dtype = postact.dtype
    if protocol.group_size != 128:
        if postact is None:
            raise ValueError("postact is required when using the unfused fallback path")
        return apply_activation_fp8_protocol_cutely_fused(
            postact,
            protocol,
            quack_enabled=quack_enabled,
            return_scales=return_scales,
            use_ste=use_ste,
            scale_out=scale_out,
        )
    padded_preact, original_postact_width = _pad_preact_for_group_size(preact, protocol.group_size)
    fused_weighted_swiglu_act_quant_best, fused_act_dequant_best = _get_cutify()
    quantized, packed_scales = fused_weighted_swiglu_act_quant_best(
        padded_preact,
        prob=None,
        block_size=protocol.group_size,
        using_pow2_scaling=True,
        using_ue8m0_scale=False,
    )
    if restored_out is not None:
        expected_shape = tuple(quantized.shape)
        if tuple(restored_out.shape) != expected_shape:
            raise ValueError(f"restored_out must have shape {expected_shape}, got {tuple(restored_out.shape)}")
        if restored_out.device != preact.device:
            raise ValueError(f"restored_out must be on device {preact.device}, got {restored_out.device}")
        if restored_out.dtype != output_dtype:
            raise ValueError(f"restored_out must have dtype {output_dtype}, got {restored_out.dtype}")
    restored_full = fused_act_dequant_best(
        quantized,
        packed_scales,
        block_size=protocol.group_size,
        out=restored_out,
    )
    restored = restored_full if restored_full.size(-1) == original_postact_width else restored_full[..., :original_postact_width]
    if restored.dtype != output_dtype:
        restored = restored.to(dtype=output_dtype)

    if use_ste:
        if postact is None:
            raise ValueError("postact is required when use_ste=True")
        restored_with_ste = postact + (restored - postact).detach()
    else:
        restored_with_ste = restored
    if not return_scales:
        return restored_with_ste, None

    scale_cols = (original_postact_width + protocol.group_size - 1) // protocol.group_size
    decoded_scale_input = packed_scales[:, :scale_cols]
    decoded_scales = round_scale_to_e8m0(decoded_scale_input, protocol, out=scale_out)
    return restored_with_ste, decoded_scales
