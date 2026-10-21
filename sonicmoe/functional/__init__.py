# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from __future__ import annotations

import collections
import os
from functools import partial

import paddle
import torch
import torch.nn.functional as F
from ..count_cumsum import count_cumsum
from ..enums import ActivationType, is_glu
from ..quack_utils import (
    bf16_wgrad_gemm_varlen_k,
    bf16_wgrad_gemm_varlen_k_accumulate,
    bf16_wgrad_gemm_varlen_k_tma_add,
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    clear_raw_weight_cache,
    clear_sgl_weight_cache,
    fast_gather_quantize_and_pack_activation,
    gemm_dgated,
    gemm_gated,
    make_blockscaled_grouped_reverse_scatter_idx,
)
from quack.gemm_interface import gemm
from ..quack_utils.fp8_quack_patch import apply_fp8_quack_patch

apply_fp8_quack_patch()

from .backward import (
    _softmax_topk_bwd,
    _token_broadcast_backward,
)
from .fp8_protocol import (
    FP8ActivationDType,
    FP8Backend,
    FP8BackendResolution,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    get_default_fp8_protocol,
    is_blackwell_device,
    resolve_fp8_backend,
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
from .reduction_over_k_gather import prepare_skip_single
from . import fp8_kernels
from .triton_kernels import TC_topk_router_metadata_triton
from .utils import enable_fp8, enable_quack_gemm, is_fp8_active, is_using_quack_gemm
from ..quack_utils.blockscaled_fp8_gemm import (
    _get_padding_plan,
    _scatter_router_scores_i32,
    _gather_router_scores_i32,
)
from ..quack_utils import (
    clear_blockscaled_fp8_weight_cache as _clear_blockscaled_fp8_weight_cache,
)

from ..quack_utils.sm100_funcs import (
    _E8M0_DTYPE,
    _GATED_OUTPUT_PAYLOAD_ATTR,
    _is_fp8_e4m3_dtype,
)


# ---------------------------------------------------------------------------
# Activation representation boundary
# ---------------------------------------------------------------------------
# Two representations of "which activation" coexist, and this module is the
# only place they meet:
#
#   1. Upper layers (interface.run_sonic_moe, ernie_compat.mlp_node_v2,
#      moe.MoE, jit_warmup, moe_TC_softmax_topk_layer, and the
#      _UpProjection / _DownProjection signatures) carry an
#      ``ActivationType`` enum member plus, for SiTU, two separate beta
#      scalars living on ``_FP8Config`` (``cfg.situ_beta`` /
#      ``cfg.situ_linear_beta``).
#
#   2. The quack_utils GEMM kernels (``gemm_gated`` / ``gemm_dgated`` /
#      ``try_gemm_gated_tvm_ffi_sonic_prepared``) take a *string*, which is
#      also part of every JIT / autotune cache key.  For SiTU that string
#      must additionally encode the betas, because they are baked into the
#      traced kernel as Constexpr values:
#      ``"situ_glu:b=4.0:lb=25.0"``.
#
# ``_gemm_activation_name()`` below is the single conversion point:
# ``ActivationType`` (+ cfg) -> GEMM string, calling
# ``encode_situ_activation()`` for SiTU.  Nothing else in this file, and
# nothing above it, may build a GEMM activation string by hand.
#
# It is also the single place that refuses SiTU on the BF16 path: pass
# ``fp8=False`` and a SiTU activation raises NotImplementedError, so it is
# structurally impossible for a BF16 caller to silently fall back to SwiGLU.
from ..quack_utils.activation_situ import (
    encode_situ_activation,
    is_situ_activation,
    parse_situ_activation,
)

# ActivationType -> quack GEMM activation string.  Deliberately NOT
# ``activation_type.value``: quack's ``"swiglu"`` entry already is
# ``swiglu_precise``, and ``"swiglu_precise"`` is not a key in its
# gate_fn_map / dgate_fn_map.  SITU_GLU is absent on purpose — it needs the
# betas and is handled separately below.
_GEMM_ACTIVATION_NAMES = {
    ActivationType.SWIGLU: "swiglu",
    ActivationType.SWIGLU_PRECISE: "swiglu",
    ActivationType.GEGLU: "geglu",
    ActivationType.REGLU: "reglu",
}

# Raw strings the quack kernels accept directly (no ActivationType member).
_RAW_GEMM_ACTIVATIONS = frozenset({"swiglu", "swiglu_oai", "reglu", "geglu", "glu"})


def _gemm_activation_name(
    activation_type,
    cfg=None,
    *,
    fp8: bool,
    site: str,
) -> str:
    """Translate an ``ActivationType`` (or raw string) into a GEMM activation.

    Parameters
    ----------
    activation_type:
        ``ActivationType`` member, a raw quack activation string, or an
        already-encoded ``"situ_glu:..."`` string.
    cfg:
        ``_FP8Config`` snapshot supplying ``situ_beta`` / ``situ_linear_beta``.
        Only consulted for SiTU.
    fp8:
        Whether the call site is on the FP8 codepath.  SiTU on ``fp8=False``
        raises: BF16 SiTU is not implemented in SonicMoE today and must never
        silently degrade to SwiGLU.
    site:
        Human-readable call-site tag used in error messages.
    """
    # A bare ``"situ_glu"`` string is the enum's own ``.value``; normalise it so
    # it resolves its betas from ``cfg`` exactly like the enum member does,
    # instead of silently taking ``parse_situ_activation``'s built-in defaults.
    if activation_type == ActivationType.SITU_GLU.value:
        activation_type = ActivationType.SITU_GLU

    # Already-encoded SiTU string: validate and pass through, re-encoding to
    # normalise the key.
    if is_situ_activation(activation_type):
        if not fp8:
            raise NotImplementedError(
                f"sonicmoe/functional/__init__.py [{site}]: situ_glu is FP8-only "
                "on SonicMoE today; the BF16 (non-FP8) GEMM path has no SiTU "
                "epilogue. Enable FP8 (SonicMoEConfig(use_fp8=True) / "
                "SONIC_MOE_FP8_MODE=1) or use a SwiGLU-family activation. "
                "Falling back to swiglu is deliberately not done."
            )
        beta, linear_beta, precise = parse_situ_activation(activation_type)
        return encode_situ_activation(beta, linear_beta, precise)

    if activation_type is ActivationType.SITU_GLU:
        if not fp8:
            raise NotImplementedError(
                f"sonicmoe/functional/__init__.py [{site}]: situ_glu is FP8-only "
                "on SonicMoE today; the BF16 (non-FP8) GEMM path has no SiTU "
                "epilogue. Enable FP8 (SonicMoEConfig(use_fp8=True) / "
                "SONIC_MOE_FP8_MODE=1) or use a SwiGLU-family activation. "
                "Falling back to swiglu is deliberately not done."
            )
        if cfg is None:
            cfg = _get_fp8_config()
        return encode_situ_activation(
            getattr(cfg, "situ_beta", None),
            getattr(cfg, "situ_linear_beta", None),
        )

    name = _GEMM_ACTIVATION_NAMES.get(activation_type)
    if name is not None:
        return name
    if isinstance(activation_type, str) and activation_type in _RAW_GEMM_ACTIVATIONS:
        return activation_type
    raise NotImplementedError(
        f"sonicmoe/functional/__init__.py [{site}]: activation "
        f"{activation_type!r} has no QuACK gated-GEMM epilogue. Supported: "
        f"{sorted(_RAW_GEMM_ACTIVATIONS)} plus situ_glu (FP8 only)."
    )


def _coerce_activation_type(activation_type):
    """Normalise a user-supplied activation into an ``ActivationType``.

    Accepts ``ActivationType`` members, plain enum values (``"swiglu"``), and
    encoded SiTU strings (``"situ_glu:b=4.0:lb=25.0"``) — the latter cannot go
    through ``ActivationType(...)`` because the enum value is the bare
    ``"situ_glu"``.  Encoded SiTU strings are returned unchanged so the betas
    they carry are not lost; ``_gemm_activation_name`` handles both forms.
    """
    if isinstance(activation_type, ActivationType):
        return activation_type
    if isinstance(activation_type, str):
        if is_situ_activation(activation_type):
            # Validate now so a malformed string fails at the API boundary.
            parse_situ_activation(activation_type)
            return (
                ActivationType.SITU_GLU
                if activation_type == ActivationType.SITU_GLU.value
                else activation_type
            )
        return ActivationType(activation_type)
    return activation_type


def _is_situ_activation_type(activation_type) -> bool:
    """True for ``ActivationType.SITU_GLU`` or an encoded ``situ_glu:...``."""
    return activation_type is ActivationType.SITU_GLU or is_situ_activation(
        activation_type
    )


from ..quack_utils.swiglu_triton import (
    dequantize_blockscaled_fp8,
)
from ..config import get_active_config


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


def _get_transposed_fp8_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return (fp8, scales, lazy) for the backward (transposed) weight operand.

    SM90 does not keep a persistent ``transposed_fp8`` copy (WGMMA fp8 is
    K-major only, so it cannot be a view of ``fp8``).  In that case the
    forward ``fp8`` payload is returned with ``lazy=True`` and the caller
    must build the transposed copy via ``_materialize_transposed_fp8`` right
    before the backward GEMM and drop it afterwards.
    """
    attr = getattr(weight, "transposed_fp8", None)
    if attr is not None:
        return attr[0], attr[1], False
    fp8, scales = _get_fp8_weight_attr(weight, "fp8")
    return fp8, scales, True


def _materialize_transposed_fp8(
    fp8: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Same op as the eager SM90 quant_weight() copy -> bit-identical operand.
    return (
        fp8.permute(0, 2, 1).contiguous(),
        scales.permute(0, 2, 1).contiguous(),
    )


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


# _use_epilogue_quant, _use_fp8_wgrad, _save_z_fp8, _recompute_z are
# imported from .fp8_config (see the import block further below).


def _use_wgrad_beta_accum() -> bool:
    return os.getenv("SONIC_MOE_FP8_WGRAD_TMA_ADD", "1").lower() in {"1", "true", "yes", "on"}


# Optional hook letting the host framework take over the expert weight-grad
# GEMM instead of running it inline, so it can be replayed later to cover
# pipeline-parallel p2p transfers.  Installed by
# ``paddlefleet.transformer.dw_overlap``; ``None`` means "run inline".
#
#   hook(point: str, weight, run_wgrad: Callable[[], None]) -> bool
#
# Returning True means the hook owns ``run_wgrad`` and is also responsible for
# calling ``_apply_weight_backward_hook(weight)`` after it runs — the grad is
# not complete until then, and sharding uses that hook to start its reduce.
# The hook keeps ``run_wgrad`` alive, which keeps the quantized activations it
# closed over alive too; that is the memory cost of deferring.
_WGRAD_DEFERRAL_HOOK = None


def set_wgrad_deferral_hook(hook) -> None:
    """Install (or clear, with ``None``) the expert weight-grad deferral hook."""
    global _WGRAD_DEFERRAL_HOOK
    _WGRAD_DEFERRAL_HOOK = hook


def _defer_wgrad(point, weight, run_wgrad) -> bool:
    hook = _WGRAD_DEFERRAL_HOOK
    if hook is None:
        return False
    return bool(hook(point, weight, run_wgrad))


# EP-combine overlap for the up-proj (dw1) weight grad.
#
# The dx of _UpProjection feeds the EP combine (DeepEPDispatch.backward), and
# the layer's caller waits on that combine with nothing else to do.  dw1 does
# not feed the combine, so it can run while the combine is in flight:
#   DeepEPDispatch.forward  -> open_comm_wgrad_slot()     (a fresh list)
#   _UpProjection.forward   -> claims the slot into ctx  (one claim per open)
#   _UpProjection.backward  -> parks run_wgrad in the slot instead of running it
#   DeepEPDispatch.backward -> launches the combine async, run_comm_wgrad_slot(),
#                              then joins the combine.
# Results are unchanged (same GEMM, same inputs); only the launch order moves.
# Cost: dz (bf16) stays alive through the dx phase.
_COMM_WGRAD_SLOT = None


def _comm_wgrad_overlap_enabled() -> bool:
    cfg = get_active_config()
    v = getattr(cfg, "ep_combine_wgrad_overlap", None) if cfg is not None else None
    if v is not None:
        return bool(v)
    return os.getenv("SONIC_MOE_EP_COMBINE_WGRAD_OVERLAP", "1").lower() in {"1", "true", "yes", "on"}


def open_comm_wgrad_slot():
    """Open the slot the next _UpProjection.forward claims (None = disabled)."""
    global _COMM_WGRAD_SLOT
    _COMM_WGRAD_SLOT = [] if _comm_wgrad_overlap_enabled() else None
    return _COMM_WGRAD_SLOT


def _claim_comm_wgrad_slot():
    global _COMM_WGRAD_SLOT
    slot, _COMM_WGRAD_SLOT = _COMM_WGRAD_SLOT, None
    return slot


def _dequantize_x_for_wgrad(cfg, x_fp8_pre, x_scales_pre, x_gather_idx):
    """bf16 x for the dw1 GEMM -> (x_bf16, gather_idx for the GEMM)."""
    # Route-ordered bf16 x (gather fused into the dequant) lets the wgrad feed
    # A by TMA; None -> source-order x + gathered wgrad.
    x_bf16 = fp8_kernels.dequantize_activation_routed(x_fp8_pre, x_scales_pre, cfg, x_gather_idx)
    if x_bf16 is not None:
        return x_bf16, None
    return fp8_kernels.dequantize_activation(x_fp8_pre, x_scales_pre, cfg), x_gather_idx


def _run_w1_wgrad_bf16_accum(cfg, x, x_fp8_pre, x_scales_pre, x_gather_idx, dz_bf16,
                             expert_frequency_offset, accum, H, N, TK, E):
    """dw1 (bf16 operands) accumulated into the fp32 main_grad ``accum`` [E, 2I, H]."""
    if x_fp8_pre is not None:
        x_bf16, wgrad_x_idx = _dequantize_x_for_wgrad(cfg, x_fp8_pre, x_scales_pre, x_gather_idx)
    else:
        x_bf16, wgrad_x_idx = x, x_gather_idx
    _wgrad_gemm = (
        bf16_wgrad_gemm_varlen_k_accumulate
        if _use_wgrad_beta_accum()
        else bf16_wgrad_gemm_varlen_k_tma_add
    )
    # Runs while the DeepEP combine holds part of the SMs: the dynamic
    # scheduler only changes tile->CTA assignment (bitwise identical).
    _wgrad_gemm(
        x_bf16.T, dz_bf16, expert_frequency_offset, wgrad_x_idx,
        accumulator=accum.permute(0, 2, 1),
        M=H, N=N, total_K=TK, num_experts=E, device=dz_bf16.device,
        dynamic_scheduler=True,
    )


def _quantize_x_col_for_wgrad(cfg, x, x_fp8_pre, x_scales_pre, x_gather_idx, expert_frequency_offset):
    if x_fp8_pre is not None:
        return fp8_kernels.quantize_wgrad_operand(
            x_fp8_pre, expert_frequency_offset, cfg,
            gather_idx=x_gather_idx, src_scales=x_scales_pre,
        )
    return fp8_kernels.quantize_wgrad_operand(x, expert_frequency_offset, cfg, gather_idx=x_gather_idx)


def _dz_col_inplace_enabled(latent: int, z_shape) -> bool:
    """``SONIC_MOE_DGATED_DZ_COL_INPLACE`` = 0 / 1 / auto (default).

    auto (r35 ABBA) keeps the standalone dz_col quant for latent >= 1280 (T7:
    it hides under the combine and the heavier dgated epilogue lands on the
    critical path) and for dz_col below 4e8 fp8 bytes (T3 / T6: no bwd gain).
    """
    mode = os.environ.get("SONIC_MOE_DGATED_DZ_COL_INPLACE", "auto")
    if mode == "auto":
        return latent < 1280 and int(z_shape[0]) * int(z_shape[1]) >= 400_000_000
    return mode != "0"


def _run_w1_wgrad_fp8_accum(cfg, x, x_fp8_pre, x_scales_pre, x_gather_idx, dz_bf16, dz_col,
                            expert_frequency_offset, accum, H, N, TK, E):
    """dw1 (fp8 colwise operands) accumulated into the fp32 main_grad ``accum`` [E, 2I, H].

    Quantizes x_col (and dz_col when the caller has none) here, so neither is
    alive during the dx phase when this runs from the comm slot.
    """
    x_col = _quantize_x_col_for_wgrad(cfg, x, x_fp8_pre, x_scales_pre, x_gather_idx, expert_frequency_offset)
    if dz_col is None:
        dz_col = fp8_kernels.quantize_wgrad_operand(dz_bf16, expert_frequency_offset, cfg)
    fp8_kernels.wgrad_gemm_varlen_k(
        dz_col, x_col, expert_frequency_offset,
        M=N, N=H, total_K=TK, num_experts=E, device=dz_bf16.device,
        accumulator=accum,
    )


def run_comm_wgrad_slot(slot) -> None:
    """Run the parked weight grads (on the current stream) and fire their hooks."""
    while slot:
        run_wgrad, weight = slot.pop(0)
        run_wgrad()
        _apply_weight_backward_hook(weight)


def _main_grad_accumulator(weight, shape=None):
    if weight is None or not hasattr(weight, "main_grad"):
        return None
    if weight.main_grad is None:
        if shape is None:
            weight.main_grad = paddle.zeros_like(weight, dtype=paddle.float32)
        else:
            weight.main_grad = paddle.zeros(
                shape=shape, dtype=paddle.float32, device=weight.place
            )
    elif shape is not None and list(weight.main_grad.shape) != list(shape):
        weight.main_grad.reshape_(shape)
    return weight.main_grad


def _apply_weight_backward_hook(weight):
    if (
        weight is not None
        and hasattr(weight, "_apply_backward_hook")
        and not weight.stop_gradient
    ):
        weight._apply_backward_hook()


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


# Transfer pre-packed blockscaled scales between autograd Function boundaries.
# Each entry maps a tag to (fp8_tensor, packed_scales) or
# (fp8_tensor, packed_scales, raw_scales_uint8).  The consumer checks
# that its input tensor shares the same storage/view metadata as the stored
# tensor before using the scales. Custom autograd boundaries may wrap the same
# storage in a fresh Tensor object, so object identity alone is too strict.
# "fwd": _UpProjection.forward -> _DownProjection.forward  (3-tuple: ref, fp8, scales)
# "bwd": _DownProjection.backward -> _UpProjection.backward (3-tuple: ref, fp8, scales)
_PrequantKey = str | tuple[str, object]
_PREQUANTIZED_SCALES: dict[_PrequantKey, tuple] = {}

# Per-invocation addressing for the backward hand-off slots.
#
# The forward slot ("fwd") is safe with a constant key: _UpProjection.forward and
# _DownProjection.forward run back to back within one invocation. The backward
# order is not — autograd's reverse topological order plus a pipeline schedule
# can interleave two invocations as write(A), write(B), read, read, in which case
# the second write clobbers the first entry and the second reader pops nothing
# ("dz storage freed but no bwd prequant"). Any caller that invokes the expert
# more than once per micro batch can hit this.
#
# The token is created in _UpProjection.forward and stamped on the tensors it
# returns, using the same carrier trick as attach_preallocated_gated_outputs
# below. Forward keeps Python object identity, unlike gradients (see the note
# above on fresh Tensor objects wrapping the same storage). _DownProjection.forward
# copies it onto its own ctx, so both halves of a pair address the same entry.
_SM_INVOCATION_TOKEN_ATTR = "_sm_invocation_token"


def _new_invocation_token() -> object:
    return object()


def _get_invocation_token(value):
    return getattr(value, _SM_INVOCATION_TOKEN_ATTR, None)


def _require_invocation_token(ctx) -> object:
    token = _get_invocation_token(ctx)
    if token is None:
        raise RuntimeError(
            "FP8 backward requires an invocation token; "
            "the UpProjection/DownProjection hand-off was not preserved."
        )
    return token


def _set_invocation_token(value, token) -> None:
    setattr(value, _SM_INVOCATION_TOKEN_ATTR, token)


def _prequant_bwd_key(ctx, name: str) -> _PrequantKey:
    """Return the backward hand-off key for one invocation.

    Keep the legacy string key when no token is available for compatibility
    with callers that bypass _UpProjection.forward.
    """
    token = _get_invocation_token(ctx)
    return name if token is None else (name, token)

# Stashed FP8 weight references — populated by MoE.stash_bf16_to_cpu(),
# consumed by the SM100 FP8 backend (quack_utils/sm100_funcs.py, read via its
# lazy ``_functional()`` accessor) and _DownProjection.forward
# to bypass global cache lookups when bf16 param storage has been freed.
# Keys: "w1_fused", "w2_varlen", "w2_dgated", "w1T_varlen"
_STASHED_FP8_WEIGHTS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

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


def attach_preallocated_gated_outputs(
    packed_scales: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Carry per-call gated outputs without making them explicit PyLayer inputs."""
    if len(outputs) != 4:
        raise ValueError(f"expected four gated outputs, got {len(outputs)}")
    setattr(packed_scales, _GATED_OUTPUT_PAYLOAD_ATTR, tuple(outputs))
    return packed_scales


def _alias_preallocated_pylayer_output(tensor: torch.Tensor) -> torch.Tensor:
    """Return independent autograd metadata over the same temporary storage."""
    alias = torch.empty(1, dtype=tensor.dtype, device=tensor.device).as_strided(
        tensor.shape, tensor.stride()
    )
    tensor._share_buffer_to(alias)
    return alias


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



# Import from fp8_config to break circular import with quack_utils
from .fp8_config import (  # noqa: E402
    _FP8Config,
    _fp8_enabled,
    _fp8_mode,
    _get_fp8_config,
    _refresh_fp8_config,
    _recompute_z,
    _save_z_fp8,
    _use_epilogue_quant,
    _use_fp8_wgrad,
    _use_fuse_y1_quant,
)


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
    _clear_blockscaled_fp8_weight_cache()
    # Clear the Triton raw-scale weight cache
    clear_raw_weight_cache()
    # Clear the sgl-kernel weight cache
    clear_sgl_weight_cache()


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


_DGATED_DUMP_CALL_COUNTER = 0


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
        prequant_activation_payload: tuple[torch.Tensor, ...] | None = None,
        fp8_config: _FP8Config | None = None,
    ) -> torch.Tensor:
        # w1 = w1_original.permute([1, 2, 0])
        # w1.fp8 = getattr(w1_original, "fp8", None)
        # w1.transposed_fp8 = getattr(w1_original, "transposed_fp8", None)
        T, H = x.shape
        E, I, H = w1.shape
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        use_quack_gemm = is_using_quack_gemm()
        y1_fp8_fused = None

        if use_quack_gemm:
            # assert not torch.compiler.is_compiling()  # Paddle compat
            assert is_glu_activation, "QuACK GEMM does not support non GLU activation yet"
            cfg = fp8_config if fp8_config is not None else _get_fp8_config()
            if cfg.enabled:
                global _ALIGNMENT_ASSUMED
                fp8_kernels.validate_config(cfg, x.device)
                _evict_per_tensor_caches_once()
                aligned = fp8_kernels.segments_aligned(expert_frequency_offset)
                _ALIGNMENT_ASSUMED = aligned
                cfg.alignment_assumed = aligned

                if aligned:
                    w1_fp8 = _get_fp8_weight_attr(w1, "fp8")
                    z, y1, y1_fp8_fused, y1_scales_fused, z_fp8_payload = fp8_kernels.gemm_gated(
                        x, w1, expert_frequency_offset, x_gather_idx,
                        x_fp8_pre=prequant_activation_payload,
                        w1_fp8=w1_fp8,
                        cfg=cfg,
                        current_stream=stream_id,
                        activation_type=activation_type,
                    )
                    if z_fp8_payload is not None:
                        # Epilogue-quantized z (E4M3 + raw 1x32 scales).
                        _PREQUANTIZED_SCALES["z_fp8"] = z_fp8_payload
                    if cfg.recompute_z:
                        # Discard the z_fp8 just produced (epilogue quant or otherwise);
                        # we will recompute it just-in-time in DownProj.backward.
                        # Stash the args needed for that recompute so DownProj.forward
                        # can move them onto its ctx (avoids ctx-attr copy from
                        # UpProj.ctx — UpProj is a different autograd Function).
                        _PREQUANTIZED_SCALES.pop("z_fp8", None)
                        if prequant_activation_payload is not None:
                            _x_fp8_pre = prequant_activation_payload[0]
                            _x_scales_pre = prequant_activation_payload[1]
                        else:
                            _x_fp8_pre, _x_scales_pre = None, None
                        x_recompute = None if _x_fp8_pre is not None else x
                        recompute_key = (
                            "z_fp8_recompute" if cfg.save_z_fp8 else "z_bf16_recompute"
                        )
                        _PREQUANTIZED_SCALES[recompute_key] = (
                            x_recompute, w1, expert_frequency_offset, x_gather_idx,
                            _x_fp8_pre, _x_scales_pre, w1_fp8,
                        )
                        if y1_fp8_fused is not None:
                            y1_fp8, y1_packed_scales = y1_fp8_fused, y1_scales_fused
                        else:
                            y1_fp8, y1_packed_scales = fp8_kernels.quantize_activation(y1, cfg)
                    elif cfg.save_z_fp8 and "z_fp8" not in _PREQUANTIZED_SCALES:
                        if _use_fused_zy1_quant() and y1_fp8_fused is None:
                            # Fused z+y1 quantization: single kernel launch, ~3µs
                            # less launch overhead, but +96 MiB peak (z_fp8 + y1_fp8
                            # coexist with z_bf16 + y1_bf16 during the kernel).
                            z_fp8, z_raw_scales, y1_fp8, y1_packed_scales = (
                                fp8_kernels.quantize_z_y1(z, y1, cfg)
                            )
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            # z.untyped_storage().resize_(0)
                        else:
                            # Split quantization: z first, free z bf16, then y1.
                            # This avoids z_bf16+y1_bf16+z_fp8+y1_fp8 all coexisting
                            # and reduces forward peak by ~96 MiB at reference shape.
                            z_fp8, z_raw_scales = fp8_kernels.quantize_preact(z)
                            _PREQUANTIZED_SCALES["z_fp8"] = (z_fp8, z_raw_scales)
                            # z.untyped_storage().resize_(0)
                            if y1_fp8_fused is not None:
                                y1_fp8, y1_packed_scales = y1_fp8_fused, y1_scales_fused
                            else:
                                y1_fp8, y1_packed_scales = fp8_kernels.quantize_activation(y1, cfg)
                    else:
                        # z_fp8 already populated from the epilogue-quant payload
                        # of fp8_kernels.gemm_gated.  z is a bf16 placeholder
                        # with freed storage (for autograd graph only).
                        # No resize needed — storage is already 0.
                        if y1_fp8_fused is not None:
                            y1_fp8, y1_packed_scales = y1_fp8_fused, y1_scales_fused
                        else:
                            y1_fp8, y1_packed_scales = fp8_kernels.quantize_activation(y1, cfg)
                    if any(st != 0 for st in y1.stride()):
                        # y1 bf16 is dead once quantized: the down-proj consumes
                        # y1_fp8 and backward recomputes y1s in dgated.  Return a
                        # zero-stride placeholder (as the fused-quant path does) so
                        # the Up ctx's non_differentiable hold does not pin
                        # [TK, I] bf16 from forward to the end of backward.
                        y1 = torch.empty(1, dtype=y1.dtype, device=y1.device).as_strided(
                            tuple(y1.shape), (0,) * y1.ndim
                        )
                    _PREQUANTIZED_SCALES["fwd"] = (y1, y1_fp8, y1_packed_scales)
                    # y1.untyped_storage().resize_(0)
                else:
                    # Non-aligned: pad expert segments to 128, use FP8 zero-mat
                    # path.  Overhead is only the extra padded GEMM rows (~5-25%
                    # depending on routing), much cheaper than full BF16 fallback.
                    z, y1 = fp8_kernels.gemm_gated_padded(
                        x, w1.permute([1, 2, 0]), expert_frequency_offset, x_gather_idx,
                        activation_type=activation_type,
                    )
            else:
                z, y1 = gemm_gated(
                    x,
                    w1.permute(0, 2, 1),
                    # BF16 path: _gemm_activation_name(fp8=False) refuses situ_glu.
                    activation=_gemm_activation_name(
                        activation_type, cfg, fp8=False,
                        site="_UpProjection.forward bf16",
                    ),
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    postact_dtype=(torch.float8_e4m3fn if use_low_precision_postact_buffer else None),
                    dynamic_scheduler=False,
                    tuned=False,
                )
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        # _log_stage_memory(
        #     f"In forward, after up-gate gemm, up_gate_output shape:{z.shape}, dtype: {z.dtype}"
        # )
        swiglu_output = y1_fp8_fused if y1_fp8_fused is not None else y1
        # _log_stage_memory(
        #     f"In forward, after swiglu, swiglu_output:{swiglu_output.shape}, dtype: {swiglu_output.dtype}"
        # )

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
        ctx._fp8_enabled = ctx._fp8_cfg.enabled
        ctx._alignment_assumed = ctx._fp8_cfg.alignment_assumed
        # Paddle requires None for every frozen tensor input slot.  Keep direct
        # PyLayer gradients separate from main_grad side effects: a parameter
        # with a main_grad accumulator still needs its wgrad GEMM, but never a
        # returned tensor for that slot.
        ctx._x_input_needs_grad = not x.stop_gradient
        ctx._w1_input_needs_grad = not w1.stop_gradient
        ctx._b1_input_needs_grad = b1 is not None and not b1.stop_gradient
        # Track which optional tensor inputs were actually provided (for Paddle backward return count)
        ctx._has_b1 = b1 is not None
        ctx._has_num_activated = num_activated_expert_per_token_offset is not None
        ctx._prequant_activation_payload = prequant_activation_payload is not None
        ctx._prequant_activation_payload_len = (
            len(prequant_activation_payload)
            if prequant_activation_payload is not None
            else 0
        )
        ctx._w1_original = w1
        ctx._wgrad_w1_accumulator = _main_grad_accumulator(w1)
        ctx._comm_wgrad_slot = _claim_comm_wgrad_slot()
        ctx._w1_wgrad_needed = (
            ctx._w1_input_needs_grad or ctx._wgrad_w1_accumulator is not None
        )

        # Weight decoupling: in FP8+aligned mode, backward doesn't need bf16 w1 data
        # (only uses fp8 cache + metadata). This enables stash_bf16_to_cpu() to
        # resize_(0) the bf16 param storage without breaking backward.
        _fp8_aligned = (
            use_quack_gemm
            and cfg.enabled
            and cfg.alignment_assumed
        )
        ctx._w1_decoupled = _fp8_aligned
        # if _fp8_aligned and ctx._prequant_activation_payload and not cfg.fp8_wgrad:
        #     raise RuntimeError("prequant activation payload requires FP8 wgrad because BF16 x is not retained")
        if _fp8_aligned:
            # Store metadata needed for dw1 allocation. w1 is the (E, 2I, H) param;
            # backward operates in the legacy (2I, H, E) convention, so store that shape.
            ctx._w1_shape = w1.shape # (E, 2I, H)
            # ctx._w1_shape = (w1.shape[1], w1.shape[2], w1.shape[0])  # (2I, H, E)
            ctx._w1_dtype = w1.dtype
            ctx._w1_device = w1.device
            # Eagerly lookup w1T fp8 cache — will be used in backward actgrad.
            # This is a cache hit (zero compute) since forward already populated the fused cache.
            # _w1T_fp8, _w1T_scales = _STASHED_FP8_WEIGHTS.get("w1T_varlen", None) or precompute_weight_fp8(w1, permute=(1, 0, 2))
            # ctx._w1T_fp8 = _w1T_fp8
            # ctx._w1T_scales = _w1T_scales
            ctx._w1T_fp8, ctx._w1T_scales, ctx._w1T_lazy = (
                _get_transposed_fp8_weight(w1)
            )
            x_fp8_pre, x_scales_pre = None, None
            x_saved = x
            if prequant_activation_payload is not None:
                x_fp8_pre = prequant_activation_payload[0]
                x_scales_pre = prequant_activation_payload[1]
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
                w1,
                b1,
                expert_frequency_offset,
                x_gather_idx,
                None if use_quack_gemm else s_scatter_idx,
                s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
            )

        if (
            prequant_activation_payload is not None
            and len(prequant_activation_payload) == 7
        ):
            # Paddle forbids returning a differentiable PyLayer input directly.
            # Keep the temporary storage but return fresh autograd metadata.
            preallocated_z = prequant_activation_payload[3]
            if _matches_prequant_tensor(preallocated_z, z):
                z = _alias_preallocated_pylayer_output(preallocated_z)

        ctx.mark_non_differentiable(y1)
        ctx.set_materialize_grads(False)

        # Keep w1 FP8 cache — backward hits cache (~112µs savings) at ~74MB memory cost.
        # The cache auto-invalidates via w._version when optimizer updates weights.

        invocation_token = _new_invocation_token()
        _set_invocation_token(ctx, invocation_token)
        for carrier in (y1, z):
            if carrier is not None:
                try:
                    _set_invocation_token(carrier, invocation_token)
                except AttributeError:
                    pass
        return y1, z

    @staticmethod
    def backward(ctx, _: None, dz: torch.Tensor):
        return _UpProjection._backward_impl(ctx, _, dz)

    @staticmethod
    def _backward_impl(ctx, _: None, dz: torch.Tensor):
        is_compiling = False

        if not is_compiling:
            assert _ is None

        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        dx_skip_single = False
        is_glu_activation = ctx.is_glu_activation
        is_varlen_K = ctx.is_varlen_K
        stream_id = ctx.stream_id
        use_quack_gemm = ctx.use_quack_gemm

        if ctx._w1_decoupled:
            # FP8+aligned: w1 not in saved_tensors; use metadata + fp8 cache.
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
            ) = ctx.saved_tensor()
            w1_shape = ctx._w1_shape   # (E, 2I, H)
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
                    # w1T_fp8, w1T_scales = precompute_weight_fp8(w1, permute=(1, 0, 2))
                    raise NotImplementedError("Quant in backward not supported yet.")
                if ctx._fp8_cfg.fp8_wgrad:
                    _require_invocation_token(ctx)
                prequant_dz = _PREQUANTIZED_SCALES.pop(_prequant_bwd_key(ctx, "bwd"), None)
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
                _comm_slot = ctx._comm_wgrad_slot
                ctx._comm_wgrad_slot = None
                if (
                    ctx._fp8_cfg.fp8_wgrad
                    and _comm_slot is not None
                    and getattr(ctx, '_wgrad_w1_accumulator', None) is not None
                    and _WGRAD_DEFERRAL_HOOK is None
                ):
                    # FP8 dw1 parked until the EP combine of dx is in flight (see
                    # open_comm_wgrad_slot).  The x_col / dz_col quants move with
                    # it, so no colwise operand is alive during the dx phase.
                    _comm_slot.append((
                        partial(
                            _run_w1_wgrad_fp8_accum,
                            ctx._fp8_cfg, x, x_fp8_pre, x_scales_pre, x_gather_idx,
                            dz_bf16,
                            _PREQUANTIZED_SCALES.pop(_prequant_bwd_key(ctx, "bwd_col"), None),
                            expert_frequency_offset, ctx._wgrad_w1_accumulator,
                            H, w1_shape[1], TK, E,
                        ),
                        ctx._w1_original,
                    ))
                    ctx._wgrad_w1_deferred = True
                    del dz_bf16
                    dw1_base = dw1 = None
                elif ctx._fp8_cfg.fp8_wgrad:
                    # FP8 wgrad with early dz_bf16 release.
                    # dz_col_fp8 was pre-computed in DownProj via dual_quantize_varlen
                    # (single HBM read of dz produced both row+col fp8).
                    bwd_col = _PREQUANTIZED_SCALES.pop(_prequant_bwd_key(ctx, "bwd_col"), None)

                    # Sequential quant pipeline (all on default stream):
                    if ctx._prequant_activation_payload:
                        assert x_fp8_pre is not None and x_scales_pre is not None, "Pre-quantized input is None."
                        x_col = fp8_kernels.quantize_wgrad_operand(
                            x_fp8_pre, expert_frequency_offset, ctx._fp8_cfg,
                            gather_idx=x_gather_idx, src_scales=x_scales_pre,
                        )
                        del x_fp8_pre, x_scales_pre
                    else:
                        x_col = fp8_kernels.quantize_wgrad_operand(
                            x, expert_frequency_offset, ctx._fp8_cfg,
                            gather_idx=x_gather_idx,
                        )

                    if bwd_col is not None:
                        # Use pre-computed col-fp8 from dual quant (zero extra HBM read)
                        dz_col = bwd_col
                    else:
                        dz_col = fp8_kernels.quantize_wgrad_operand(
                            dz_bf16, expert_frequency_offset, ctx._fp8_cfg,
                        )
                    # FREE dz_bf16 NOW (-384 MiB before wgrad GEMM!)
                    # dz.untyped_storage().resize_(0)
                    del dz_bf16

                    # FP8 wgrad GEMM: dw1[e] = dz_col[:, seg] @ x_col[:, seg]^T -> [E, 2I, H].
                    # With a fp32 accumulator (ERNIE main_grad path) it adds in place.
                    _wgrad_accum = getattr(ctx, '_wgrad_w1_accumulator', None)
                    if _wgrad_accum is not None:
                        # partial() pins the quantized activations, so they outlive
                        # the ``del`` below when the host defers the GEMM.
                        _run_wgrad: partial[None] = partial(
                            fp8_kernels.wgrad_gemm_varlen_k,
                            dz_col, x_col,
                            expert_frequency_offset,
                            M=w1_shape[1], N=H, total_K=TK,
                            num_experts=E, device=x.device,
                            accumulator=_wgrad_accum,
                        )
                        ctx._wgrad_w1_deferred = _defer_wgrad(
                            "moe_sonic_expert_up_gate_proj",
                            ctx._w1_original,
                            _run_wgrad,
                        )
                        if ctx._wgrad_w1_deferred is False:
                            _run_wgrad()
                        # Drop our partial (a deferring hook holds its own), else
                        # it pins dz_col / x_col through the dx phase.
                        del _run_wgrad
                        dw1_base = None
                        dw1 = None
                    elif ctx._w1_input_needs_grad:
                        dw1_base = fp8_kernels.wgrad_gemm_varlen_k(
                            dz_col, x_col,
                            expert_frequency_offset,
                            M=w1_shape[1], N=H, total_K=TK,
                            num_experts=E, out_dtype=w1_dtype, device=x.device,
                        )
                        dw1 = dw1_base.permute(1, 2, 0)
                    else:
                        dw1_base = dw1 = None
                    del dz_col, x_col
                else:
                    bwd_col = _PREQUANTIZED_SCALES.pop(_prequant_bwd_key(ctx, "bwd_col"), None)
                    _wgrad_accum = getattr(ctx, '_wgrad_w1_accumulator', None)
                    if (
                        _comm_slot is not None
                        and _wgrad_accum is not None
                        and _WGRAD_DEFERRAL_HOOK is None
                    ):
                        # Parked until the EP combine of dx is in flight (see
                        # open_comm_wgrad_slot).  The dequant moves with it, so
                        # x_bf16 is not alive during the dx phase.
                        _comm_slot.append((
                            partial(
                                _run_w1_wgrad_bf16_accum,
                                ctx._fp8_cfg, x, x_fp8_pre, x_scales_pre,
                                x_gather_idx, dz_bf16, expert_frequency_offset,
                                _wgrad_accum, H, w1_shape[1], TK, E,
                            ),
                            ctx._w1_original,
                        ))
                        ctx._wgrad_w1_deferred = True
                        x_bf16 = None
                        _wgrad_accum = None
                        dw1_base = dw1 = None
                    elif ctx._prequant_activation_payload:
                        assert x_fp8_pre is not None and x_scales_pre is not None, "Pre-quantized input is None."
                        x_bf16, wgrad_x_idx = _dequantize_x_for_wgrad(
                            ctx._fp8_cfg, x_fp8_pre, x_scales_pre, x_gather_idx
                        )
                    else:
                        x_bf16, wgrad_x_idx = x, x_gather_idx
                    if _wgrad_accum is not None:
                        # BF16 wgrad + fp32 accumulate.
                        # _wgrad_accum: [E, 2I, H] fp32. GEMM out: [E, H, 2I].
                        # permute(0,2,1) gives [E,H,2I] non-contiguous view —
                        # CuTe handles via stride.
                        accum_view = _wgrad_accum.permute(0, 2, 1)  # [E, H, 2I]
                        _wgrad_gemm = (
                            bf16_wgrad_gemm_varlen_k_accumulate
                            if _use_wgrad_beta_accum()
                            else bf16_wgrad_gemm_varlen_k_tma_add
                        )
                        # partial() pins x_bf16 / dz_bf16, so they outlive the
                        # ``del dz_bf16`` below when the host defers the GEMM.
                        _run_wgrad = partial(
                            _wgrad_gemm,
                            x_bf16.T,
                            dz_bf16,
                            expert_frequency_offset,
                            wgrad_x_idx,
                            accumulator=accum_view,
                            M=H,
                            N=w1_shape[1],
                            total_K=TK,
                            num_experts=E,
                            device=x.device,
                        )
                        ctx._wgrad_w1_deferred = _defer_wgrad(
                            "moe_sonic_expert_up_gate_proj",
                            ctx._w1_original,
                            _run_wgrad,
                        )
                        if not ctx._wgrad_w1_deferred:
                            _run_wgrad()
                        # Drop our partial too, else it pins x_bf16 through dx.
                        del _run_wgrad
                        dw1_base = None
                        dw1 = None
                    elif ctx._w1_input_needs_grad and x_bf16 is not None:
                        dw1_base = torch.empty((E, w1_shape[1], w1_shape[2]), dtype=w1_dtype, device=w1_device) # (E, 2I, H)
                        dw1 = dw1_base.permute(1, 2, 0) # (2I, H, E)
                        bf16_wgrad_gemm_varlen_k(
                            x_bf16.T,
                            dz_bf16,
                            expert_frequency_offset,
                            wgrad_x_idx,
                            out=dw1_base.permute(0, 2, 1), # (E, H, 2I)
                            M=H,
                            N=w1_shape[1],
                            total_K=TK,
                            num_experts=E,
                            device=x.device,
                        )
                    elif x_bf16 is not None:
                        dw1_base = dw1 = None
                    # A deferred wgrad keeps its own reference; drop ours before
                    # the dx phase (w1T + dx_expanded) instead of at return.
                    del x_bf16

                # Phase 2: Free dz bf16 storage (~384 MiB at reference shape).
                # FP8 wgrad already freed it in step 2 above; BF16 path frees here.
                # _log_stage_memory("In backward, after w1 grad")

                if not ctx._fp8_cfg.fp8_wgrad:
                    # dz.untyped_storage().resize_(0)
                    del dz_bf16

                # Phase 3: Actgrad using FP8 dz (avoids dz_bf16 + dx_expanded coexistence).
                if has_prequant:
                    _PREQUANT_HIT_COUNT["bwd"] += 1
                    _, dz_fp8, dz_packed_scales = prequant_dz
                    if getattr(ctx, "_w1T_lazy", False):
                        # Built after wgrad/del dz_bf16, freed right after use.
                        w1T_fp8, w1T_scales = _materialize_transposed_fp8(
                            w1T_fp8, w1T_scales
                        )
                    if ctx._w1_decoupled:
                        route_scatter = {}
                        # Not with fp8 wgrad: there the dx phase is the bwd peak and
                        # dx_reduced would have to be live during the GEMM.
                        if (
                            not ctx._fp8_cfg.fp8_wgrad
                            and dz.dtype == torch.bfloat16
                            and x_gather_idx.shape[0] == dz_fp8.shape[0]
                            and fp8_kernels.route_scatter_supported(ctx._fp8_cfg, is_varlen_K, H)
                        ):
                            # Single-route tokens go straight from the epilogue to
                            # dx_reduced; the reduce only handles multi-route ones.
                            dx_skip_single = True
                            dx_expanded = torch.empty(int(dz_fp8.shape[0]), H, dtype=dz.dtype, device=dz.device)
                            dx_reduced = torch.empty(T, H, dtype=dz.dtype, device=dz.device)
                            dst, _ = prepare_skip_single(
                                dx_expanded, None, dx_reduced, x_gather_idx, s_reverse_scatter_idx,
                                num_activated_expert_per_token_offset, T, (K if K is not None else E), H,
                            )
                            route_scatter = dict(out=dx_expanded, scatter_dst=dst, scatter_out=dx_reduced)
                        # w1 not in saved_tensors; call low-level GEMM directly
                        # with shape metadata (avoids needing a weight tensor).
                        dx_expanded = fp8_kernels.gemm(
                            dz_fp8, dz_packed_scales,
                            w1T_fp8, w1T_scales,
                            expert_frequency_offset,
                            out_features=w1_shape[2],  # w1 is (E, 2I, H), H=shape[2]
                            num_experts=E,
                            cfg=ctx._fp8_cfg,
                            **route_scatter,
                        )
                        del route_scatter
                    else:
                        # Unreachable today (non-decoupled raises above).  Shapes
                        # follow the pre-quantized hot path of
                        # blockscaled_fp8_gemm_varlen on w1.permute(2, 1, 0).
                        _w1_view = w1.permute(2, 1, 0)
                        dx_expanded = fp8_kernels.gemm(
                            dz_fp8, dz_packed_scales,
                            w1T_fp8, w1T_scales,
                            expert_frequency_offset,
                            out_features=_w1_view.shape[1],
                            num_experts=_w1_view.shape[0],
                            cfg=ctx._fp8_cfg,
                        )
                    # prequant_dz also pins dz_fp8: drop it so the reduction below
                    # does not run with the [TK, 2I] fp8 dz still allocated.
                    del dz_fp8, dz_packed_scales
                    prequant_dz = None
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
                _wgrad_accum = getattr(ctx, '_wgrad_w1_accumulator', None)
                if _wgrad_accum is not None:
                    accum_view = _wgrad_accum.permute(0, 2, 1)  # [E,2I,H] → [E,H,2I]
                    _wgrad_gemm = (
                        bf16_wgrad_gemm_varlen_k_accumulate
                        if _use_wgrad_beta_accum()
                        else bf16_wgrad_gemm_varlen_k_tma_add
                    )
                    _run_wgrad = partial(
                        _wgrad_gemm,
                        x.T,
                        dz,
                        expert_frequency_offset,
                        x_gather_idx,
                        accumulator=accum_view,
                        M=H,
                        N=w1_shape[1],
                        total_K=TK,
                        num_experts=E,
                        device=x.device,
                    )
                    ctx._wgrad_w1_deferred = _defer_wgrad(
                        "moe_sonic_expert_up_gate_proj",
                        ctx._w1_original,
                        _run_wgrad,
                    )
                    if not ctx._wgrad_w1_deferred:
                        _run_wgrad()
                    dw1 = None
                elif ctx._w1_input_needs_grad:
                    dw1_base = torch.empty((E, w1_shape[1], w1_shape[2]), dtype=w1_dtype, device=w1_device)
                    dw1 = dw1_base.permute(1, 2, 0)
                    bf16_wgrad_gemm_varlen_k(
                        x.T,
                        dz,
                        expert_frequency_offset,
                        x_gather_idx,
                        out=dw1_base.permute(0, 2, 1),
                        M=H,
                        N=w1_shape[1],
                        total_K=TK,
                        num_experts=E,
                        device=x.device,
                    )
                else:
                    dw1_base = dw1 = None
                actgrad_weight = w1
                # Input-gradient GEMM dx = dz @ w1^T (grouped/varlen over experts).
                # The default SM90 grouped-GEMM config (cluster_m=2 + pingpong) has
                # a synchronization race in the shared-epilogue-smem / cluster
                # multicast path that makes this GEMM non-deterministic run-to-run
                # (relative diff ~1e-4 at the bf16-ULP scale). The input-gradient
                # signal here is tiny, so that noise breaks the strict 1e-8
                # fp8-dispatch-vs-fp8 comparison. Pin a deterministic cooperative
                # config (pingpong=False); disabling either cluster_m=2 or pingpong
                # removes the race.
                from quack.gemm_interface import gemm_tuned as _gemm_tuned
                from quack.gemm_config import GemmConfig as _GemmConfig
                dx_expanded = torch.empty(
                    dz.shape[0], actgrad_weight.shape[-1],
                    dtype=dz.dtype, device=dz.device,
                )
                _gemm_tuned.fn(
                    dz, actgrad_weight, dx_expanded,
                    cu_seqlens_m=expert_frequency_offset,
                    dynamic_scheduler=False,
                    config=_GemmConfig(
                        tile_m=256, tile_n=192, pingpong=False,
                        cluster_m=2, cluster_n=1,
                        is_dynamic_persistent=False,
                    ),
                )
                # _log_stage_memory("In backward, after w1 grad")
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        if not dx_skip_single:
            dx_reduced = torch.empty(T, H, dtype=dz.dtype, device=dz.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(K if K is not None else E),
            H=H,
            is_varlen_K=is_varlen_K,
            skip_single=dx_skip_single,
        )
        # _log_stage_memory(
        #     f"In backward, after up_gate gemm bwd, d_up_gate_input shape: {dx_reduced.shape}, dtype:{dx_reduced.dtype}"
        # )
        # _log_stage_memory("MoE expert compute backward end")

        # Paddle PyLayer: return grads only for tensor inputs (not int/bool/enum)
        grads = [dx_reduced if ctx._x_input_needs_grad else None]
        if ctx._w1_input_needs_grad and ctx._wgrad_w1_accumulator is None:
            grads.append(dw1.permute([2, 0, 1]))
        else:
            grads.append(None)
        if ctx._has_b1:
            grads.append(db1 if ctx._b1_input_needs_grad else None)
        # expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
        grads.extend([None, None, None, None])
        if ctx._has_num_activated:
            grads.append(None)
        if ctx._prequant_activation_payload:
            grads.append(tuple(None for _ in range(ctx._prequant_activation_payload_len)))
        if ctx._wgrad_w1_accumulator is not None and not getattr(
            ctx, "_wgrad_w1_deferred", False
        ):
            # When deferred, the host hook fires this after it runs the GEMM —
            # main_grad is not complete yet at this point.
            _apply_weight_backward_hook(ctx._w1_original)
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
        fp8_config: _FP8Config | None = None,
        router_score_source: torch.Tensor | None = None,
        router_score_src_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # w2 = w2_original.permute([1, 2, 0])
        # w2.fp8 = getattr(w2_original, "fp8", None)
        # w2.transposed_fp8 = getattr(w2_original, "transposed_fp8", None)
        TK = y1.size(0)
        E, H, I = w2.shape

        invocation_token = _get_invocation_token(z)
        if invocation_token is None:
            invocation_token = _get_invocation_token(y1)
        _set_invocation_token(ctx, invocation_token)

        if (router_score_source is None) != (router_score_src_idx is None):
            raise ValueError("router_score_source and router_score_src_idx must be provided together")
        has_router_score_source = router_score_source is not None
        router_score_source_needs_grad = (
            has_router_score_source
            and hasattr(router_score_source, "stop_gradient")
            and not router_score_source.stop_gradient
        )
        if has_router_score_source:
            if router_score_source.ndim != 2:
                raise ValueError("router_score_source must be rank-2, got " f"shape={router_score_source.shape}")
            if router_score_src_idx.ndim != 1 or str(router_score_src_idx.dtype) not in (
                "torch.int32",
                "paddle.int32",
                "int32",
            ):
                raise ValueError(
                    "router_score_src_idx must be rank-1 int32, got "
                    f"shape={router_score_src_idx.shape}, dtype={router_score_src_idx.dtype}"
                )
            if router_score_src_idx.shape[0] > topk_scores.shape[0]:
                raise ValueError("router_score_src_idx cannot exceed the metadata score buffer")

        use_quack_gemm = is_using_quack_gemm()
        fused_router_skip_single = False

        if use_quack_gemm:
            # assert not torch.compiler.is_compiling()  # Paddle compat

            assert b2 is None
            cfg = fp8_config if fp8_config is not None else _get_fp8_config()
            fp8_kernels.validate_config(cfg, y1.device)
            if cfg.enabled and cfg.alignment_assumed:
                # Use pre-quantized y1 from _UpProjection if available (quantized
                # while hot in L2); otherwise quantize inline.
                # Format: 3-tuple (bf16_ref, fp8_data, scales).
                prequant = _PREQUANTIZED_SCALES.pop("fwd", None)
                has_prequant = (
                    prequant is not None
                    and len(prequant) == 3
                    and _matches_prequant_tensor(prequant[0], y1)
                )
                if has_prequant:
                    _PREQUANT_HIT_COUNT["fwd"] += 1
                    _, y1_fp8, y1_scales = prequant
                else:
                    if y1.ndim == 2 and tuple(y1.stride()) == (0, 0) and y1.shape[0] > 1:
                        raise RuntimeError("y1 is an unmaterialized placeholder but its fp8 prequant was not found")
                    y1_fp8, y1_scales = fp8_kernels.quantize_activation(y1, cfg)
                w2_fp8, w2_scales = _get_fp8_weight_attr(w2, "fp8")
                route_scatter = {}
                if (x_gather_idx.shape[0] == TK
                        and fp8_kernels.route_scatter_supported(cfg, is_varlen_K, H)):
                    # Single-route tokens go straight from the epilogue to o;
                    # _router_forward only reduces the multi-route ones.
                    fused_router_skip_single = True
                    y2 = torch.empty(TK, H, device=y1.device, dtype=torch.bfloat16)
                    o = torch.empty(T, H, device=z.device, dtype=torch.bfloat16)
                    topk_scores = topk_scores if topk_scores.ndim == 1 else topk_scores.flatten()
                    dst, w_route = prepare_skip_single(
                        y2, topk_scores, o, x_gather_idx, s_reverse_scatter_idx,
                        num_activated_expert_per_token_offset, T, (K if K is not None else E), H,
                    )
                    route_scatter = dict(out=y2, scatter_dst=dst, scatter_out=o, scatter_w=w_route)
                # w2 is the (E, H, I) param: out_features=H, K=I.
                y2 = fp8_kernels.gemm(
                    y1_fp8, y1_scales, w2_fp8, w2_scales, expert_frequency_offset,
                    out_features=w2.shape[1], num_experts=w2.shape[0], cfg=cfg, **route_scatter,
                )
                del y1_fp8, y1_scales, w2_fp8, w2_scales, route_scatter
                # Keep w2 varlen cache — iso32 re-quant is expensive (~87µs/iter).
                # Cache auto-invalidates via w._version at optimizer step.
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
            elif cfg.enabled:
                # FP8 enabled but not aligned: the backend pads internally.
                y2 = fp8_kernels.gemm_padded(
                    y1, w2.permute([1, 2, 0]), expert_frequency_offset,
                )
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
            else:
                y2 = gemm(
                    y1,
                    w2.permute(0, 2, 1),
                    cu_seqlens_m=expert_frequency_offset,
                    tuned=False,
                )
                router_perm = s_reverse_scatter_idx
                y2_for_router = y2
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        # _log_stage_memory(
        #     f"In forward, after down gemm, down_output shape: {y2_for_router.shape}, dtype: {y2_for_router.dtype}"
        # )

        # Output must always be bf16 (z may be fp8 when epilogue_quant is active).
        if not fused_router_skip_single:
            o = torch.empty(T, H, device=z.device, dtype=torch.bfloat16)
            topk_scores = topk_scores if topk_scores.ndim == 1 else topk_scores.flatten()

        _router_forward(
            y2=y2_for_router,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=router_perm,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(K if K is not None else E),
            H=H,
            is_varlen_K=is_varlen_K,
            skip_single=fused_router_skip_single,
        )
        del y2_for_router
        del y2
        # _log_stage_memory(
        #     f"In forward, after local combine, expert_out shape:{o.shape}, dtype: {o.dtype}"
        # )

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
        # Paddle requires None for frozen tensor input slots. Keep this direct
        # return contract independent from optional main_grad accumulation.
        ctx._w2_input_needs_grad = not w2.stop_gradient
        ctx._b2_input_needs_grad = b2 is not None and not b2.stop_gradient
        # Track which optional tensor inputs were actually provided (for Paddle backward return count)
        ctx._has_b2 = b2 is not None
        ctx._has_num_activated = num_activated_expert_per_token_offset is not None
        ctx._fp8_combine_grad_handle = fp8_combine_grad_handle
        # An FP8 combine backward returns its E4M3 payload as our output grad
        # (the real operands travel in the handle). The FP8-wgrad backward only
        # reads dout's shape/device, so keep it E4M3 instead of letting the
        # engine cast the whole [T, H] grad to BF16.
        if (
            fp8_combine_grad_handle is not None
            and ctx._fp8_cfg.fp8_wgrad
            and ctx._fp8_enabled_flag
            and ctx._alignment_assumed_flag
        ):
            ctx.set_grad_in_dtype_consistent(False)
        ctx._has_router_score_source = has_router_score_source
        ctx._router_score_source_needs_grad = router_score_source_needs_grad
        if ctx._has_router_score_source:
            # The source tensor is an autograd edge only.  Forward keeps reading the
            # compact metadata scores; backward scatters ds to this original shape.
            ctx._router_score_source_shape = tuple(router_score_source.shape)
            ctx._router_score_source_numel = int(router_score_source.shape[0]) * int(router_score_source.shape[1])
        ctx._w2_original = w2
        # weight2 is Sonic-layout [E, H, I] here, while Fleet keeps its
        # main_grad in grouped layout [E, I, H].
        ctx._wgrad_w2_accumulator = _main_grad_accumulator(
            w2, shape=[w2.shape[0], w2.shape[2], w2.shape[1]]
        )
        ctx._w2_wgrad_needed = (
            ctx._w2_input_needs_grad or ctx._wgrad_w2_accumulator is not None
        )
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

        # Memory optimization: store z in FP8 to save ~50% of z's memory.
        # At reference shape (TK=65536, 2I=3072), z is 384MB BF16 -> ~213MB FP8 = ~171MB saved.
        # Accept fp8 z when prequant cache already holds the fp8+scales pair
        # (e.g. epilogue quant produced them), even if z.dtype is no longer bf16.
        z_has_prequant = "z_fp8" in _PREQUANTIZED_SCALES
        z_has_fp8_recompute = "z_fp8_recompute" in _PREQUANTIZED_SCALES
        # BF16 recompute (precision-first default): forward stored only a zero-storage
        # bf16 z placeholder; backward re-runs the up-proj GEMM with a BF16 D output so
        # z is never quantized.  Independent of save_z_fp8 / the fp8 recompute path.
        z_bf16_recompute_args = _PREQUANTIZED_SCALES.pop("z_bf16_recompute", None)
        ctx._needs_z_bf16_recompute = z_bf16_recompute_args is not None
        ctx._z_bf16_recompute_args = z_bf16_recompute_args
        z_is_fp8 = (cfg.enabled and use_quack_gemm and cfg.save_z_fp8
                    and cfg.alignment_assumed
                    and (z.dtype == torch.bfloat16 or z_has_prequant or z_has_fp8_recompute))
        ctx._z_is_fp8 = z_is_fp8

        # w2 decoupling: in FP8+aligned mode, backward doesn't
        # read bf16 w2 data (uses fp8 dgated cache + metadata).  This enables
        # stash_bf16_to_cpu() to resize_(0) the bf16 param storage.
        _w2_decouple = cfg.enabled and cfg.alignment_assumed
        ctx._w2_decoupled = _w2_decouple

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
                        z_fp8, z_raw_scales = fp8_kernels.quantize_preact(z)
            if _w2_decouple:
                # Eagerly look up w2 dgated fp8 cache for backward.
                # _w2_dgated_fp8, _w2_dgated_scales = _STASHED_FP8_WEIGHTS.get("w2_dgated", None) or precompute_weight_fp8_for_direct_fused_dgated(w2)
                _w2_dgated_fp8, _w2_dgated_scales, ctx._w2_dgated_lazy = (
                    _get_transposed_fp8_weight(w2)
                )
                ctx._w2_dgated_fp8 = _w2_dgated_fp8
                ctx._w2_dgated_scales = _w2_dgated_scales
                ctx._w2_shape = w2.shape  # (E, H, I)
                ctx._w2_dtype = w2.dtype
                ctx._w2_device = w2.device
                ctx.save_for_backward(
                    z_fp8,
                    z_raw_scales,
                    # w2 omitted — backward uses ctx._w2_dgated_fp8 + metadata
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
                )
            else:
                ctx.save_for_backward(
                    z_fp8,
                    z_raw_scales,
                    w2,
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
                )
        else:
            if _w2_decouple:
                _w2_dgated_fp8, _w2_dgated_scales, ctx._w2_dgated_lazy = (
                    _get_transposed_fp8_weight(w2)
                )
                ctx._w2_dgated_fp8 = _w2_dgated_fp8
                ctx._w2_dgated_scales = _w2_dgated_scales
                ctx._w2_shape = w2.shape
                ctx._w2_dtype = w2.dtype
                ctx._w2_device = w2.device
                ctx.save_for_backward(
                    z,
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
                )
            else:
                ctx.save_for_backward(
                    z,
                    w2,
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
                )

        # Keep w2 FP8 cache — backward hits cache (~38µs savings) at ~37MB memory cost.
        # The cache auto-invalidates via w._version when optimizer updates weights.

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        return _DownProjection._backward_impl(ctx, dout)

    @staticmethod
    def _backward_impl(ctx, dout: torch.Tensor):
        # _log_stage_memory("MoE expert compute backward start")
        T = ctx.T
        K = ctx.K
        stream_id = ctx.stream_id
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type
        use_quack_gemm = ctx.use_quack_gemm
        fp8_combine_grad_handle = ctx._fp8_combine_grad_handle

        # Ensure dout is contiguous (expanded tensors from e.g. sum().backward()
        # have stride (0,0) which violates GEMM k-major assertions)
        dout = dout.contiguous()

        if ctx._z_is_fp8:
            if ctx._w2_decoupled:
                (
                    z_fp8,
                    z_raw_scales,
                    b2,
                    topk_scores,
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
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
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
                ) = ctx.saved_tensor()
                w2_shape = w2.shape
                w2_dtype = w2.dtype
                w2_device = w2.device
            if getattr(ctx, "_needs_z_recompute", False):
                # Replace zero-storage placeholders with real fp8 z + scales
                # by re-running the up-proj GEMM (discards recomputed y1).
                z_fp8, z_raw_scales = fp8_kernels.recompute_z(
                    *ctx._z_recompute_args, cfg=ctx._fp8_cfg, fp8=True
                )
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
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
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
                    expert_frequency_offset,
                    x_gather_idx,
                    s_scatter_idx,
                    s_reverse_scatter_idx,
                    router_score_src_idx,
                ) = ctx.saved_tensor()
                w2_shape = w2.shape
                w2_dtype = w2.dtype
                w2_device = w2.device
            if getattr(ctx, "_needs_z_bf16_recompute", False):
                # Recompute z bf16 from saved args (re-run up-proj GEMM + SwiGLU)
                z = fp8_kernels.recompute_z(
                    *ctx._z_bf16_recompute_args, cfg=ctx._fp8_cfg, fp8=False
                )
                ctx._z_bf16_recompute_args = None
            z_fp8 = z_raw_scales_u8 = None

        # Defer dw2 allocation: dw2 is not needed
        # until the wgrad GEMM (~384 MiB after dgated outputs dz+y1s).
        # Allocating here adds 72 MiB to the dgated peak unnecessarily.
        dw2_base = dw2 = None  # allocated just before wgrad in each path
        db2 = None if b2 is None else torch.empty_like(b2)

        if use_quack_gemm:
            # assert not torch.compiler.is_compiling()  # Paddle compat
            assert is_glu(activation_type), "QuACK GEMM does not support non GLU activation yet"

            s = _gather_router_scores_i32(topk_scores, s_scatter_idx)
            if ctx._fp8_enabled_flag and ctx._alignment_assumed_flag:
                # All segments aligned: use blockscaled FP8 path.
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
                        z = fp8_kernels.dequantize_preact(z_fp8, z_raw_scales_u8)
                        del z_fp8, z_raw_scales_u8
                else:
                    s_float = s if str(s.dtype) in ("torch.float32", "paddle.float32", "float32") else s.float()

                if ctx._w2_decoupled:
                    w2_fp8_enk = ctx._w2_dgated_fp8
                    w2_scales = ctx._w2_dgated_scales
                else:
                    w2_fp8_enk, w2_scales = fp8_kernels.precompute_weight_dgated(w2.permute([1, 2, 0]))
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
                    # Raw UE8M0 scales arrive as uint8 or an int32 carrier; the kernel
                    # facade normalizes either form.
                    if str(combine_grad_scale.dtype) in ("torch.uint8", "paddle.uint8", "uint8", "torch.int32", "paddle.int32", "int32"):
                        dout_raw_scales_t = combine_grad_scale
                    else:
                        dout_raw_scales_t = combine_grad_scale.to(device=combine_grad_scale.device, dtype=torch.uint8)
                    if not _is_fp8_e4m3_dtype(combine_grad_data.dtype):
                        raise RuntimeError("Sonic FP8 combine backward requires FP8 combine grad payload")
                    dout_comm_fp8 = combine_grad_data
                    dout_fp8 = dout_comm_fp8
                    if not ctx._fp8_cfg.fp8_wgrad:
                        dout = fp8_kernels.dequantize_activation(
                            dout_fp8, dout_raw_scales_t, ctx._fp8_cfg
                        )
                else:
                    if _is_fp8_e4m3_dtype(dout.dtype):
                        raise RuntimeError("FP8 dout without the combine grad payload in its handle")
                    dout_fp8, dout_raw_scales_t = fp8_kernels.quantize_activation_raw(dout, ctx._fp8_cfg)

                if ctx._w2_decoupled and getattr(ctx, "_w2_dgated_lazy", False):
                    # Transient transposed copy, freed by ``del w2_fp8_enk`` below.
                    w2_fp8_enk, w2_scales = _materialize_transposed_fp8(
                        w2_fp8_enk, w2_scales
                    )

                # z (saved, recomputed or dequantized bf16) is dead after dgated:
                # write dz over it instead of allocating a second [TK, 2I] bf16.
                # Each epilogue tile reads its own preact tile before storing dz.
                dz_in_place = (
                    not use_fp8_preact
                    and z is not None
                    and z.dtype == torch.bfloat16
                    and z.ndim == 2
                    and z.is_contiguous()
                    and z.shape[0] == x_gather_idx.shape[0]
                    and os.environ.get("SONIC_MOE_DGATED_DZ_INPLACE", "1") != "0"
                )
                # FP8 wgrad: the dgated epilogue can also emit the 1x128 colwise
                # wgrad operands of y1s / dz (bitwise the separate quants); their
                # BF16 forms are then dead, so their stores are skipped as well
                # (y1s is not even allocated; dz is only consumed as fp8 below).
                col_quant = skip_bf16 = ()
                dz_col_inplace = False
                if ctx._fp8_cfg.fp8_wgrad and not use_fp8_preact:
                    col_quant = tuple(
                        c for c in os.environ.get("SONIC_MOE_DGATED_COL_QUANT", "y1s").split(",")
                        if c in ("y1s", "dz")
                    )
                    if "y1s" in col_quant and os.environ.get("SONIC_MOE_DGATED_SKIP_BF16", "1") != "0":
                        skip_bf16 = col_quant
                    # dz_col written tile-locally over z itself (z already lives until
                    # dw1 as its bf16 dz): no [2I, TK] fp8 buffer and no dz_col quant.
                    if dz_in_place and _dz_col_inplace_enabled(int(dout.shape[-1]), z.shape):
                        col_quant = skip_bf16 = ("dz", "y1s")
                        dz_col_inplace = True
                dgated_out = fp8_kernels.gemm_dgated(
                    dout_fp8, dout_raw_scales_t,
                    w2_fp8=w2_fp8_enk,
                    w2_scales=w2_scales,
                    preact=None if use_fp8_preact else z,
                    preact_fp8=z_fp8 if use_fp8_preact else None,
                    preact_scales=z_raw_scales_u8 if use_fp8_preact else None,
                    colvec_scale=s_float,
                    expert_frequency_offset=expert_frequency_offset,
                    x_gather_idx=x_gather_idx,
                    activation=_gemm_activation_name(
                        activation_type, ctx._fp8_cfg, fp8=True,
                        site="_DownProjection.backward fused dgated",
                    ),
                    cfg=ctx._fp8_cfg,
                    quantize_dz=True,
                    dz_out=z if dz_in_place else None,
                    col_quant=col_quant,
                    skip_bf16=skip_bf16,
                    dz_col_inplace=dz_col_inplace,
                )
                dz, y1s, ds, dz_rowwise = dgated_out[:4]
                fused_cols = dgated_out[4] if col_quant else {}
                del dgated_out
                del dout_fp8, z
                # Release FP8 preact tensors from ctx (z_fp8 ~192 MiB + scales ~6 MiB).
                # The dgated GEMM is done; these are no longer needed.
                if use_fp8_preact:
                    del z_fp8, z_raw_scales_u8
                del w2_fp8_enk, w2_scales

                # ── Eager release of w2_dgated (consumed by dgated GEMM above) ──
                # w1_fused and w2_varlen were already freed at backward entry
                # (early eviction). Only w2_dgated remains to clean up here.
                # NOTE: do NOT free ctx._w2_dgated_fp8 storage — it's shared
                # with fp8_weight_cache[fused]. Freeing it corrupts the cache and
                # causes "storage of size 0" errors on the next forward.
                # The cache is version-keyed and auto-invalidates at optimizer step.
                _w2d_stash = _STASHED_FP8_WEIGHTS.pop("w2_dgated", None)
                # Don't free stash entries either — they may alias cache tensors.

                # Weight-grad: dw2 = dout.T @ y1s (per expert).
                TK_wgrad = x_gather_idx.shape[0]
                if ctx._fp8_cfg.fp8_wgrad:
                    # Memory-optimized wgrad pipeline (all main stream):
                    # Step 1: colwise(y1s) (unless fused) then del y1s to free 192 MiB
                    y1s_col = fused_cols.pop("y1s", None)
                    if y1s_col is None:
                        y1s_col = fp8_kernels.quantize_wgrad_operand(
                            y1s, expert_frequency_offset, ctx._fp8_cfg,
                        )
                    del y1s

                    # Step 2: dz actgrad rows + colwise(dout, gather).  dz_col
                    # is deferred past the dw2 GEMM (where the backend allows)
                    # so it never coexists with y1s_col / dout_col.
                    fused_dz_col = fused_cols.pop("dz", None)
                    dz_actgrad, dz_col, dout_col = fp8_kernels.quantize_dz_dout(
                        dz, dout, expert_frequency_offset, x_gather_idx, ctx._fp8_cfg,
                        dout_fp8=dout_comm_fp8 if dout_has_comm_fp8_payload else None,
                        dout_scales=dout_raw_scales_t if dout_has_comm_fp8_payload else None,
                        dz_rowwise=dz_rowwise,
                        defer_dz_col=(fused_dz_col is not None
                                      or os.environ.get("SONIC_MOE_DEFER_DZ_COL", "1") != "0"),
                    )
                    if fused_dz_col is not None:
                        dz_col = fused_dz_col
                    del dz_rowwise, fused_dz_col, fused_cols
                    _PREQUANTIZED_SCALES[_prequant_bwd_key(ctx, "bwd")] = (dz, *dz_actgrad)
                    if dz_col is not None:
                        _PREQUANTIZED_SCALES[_prequant_bwd_key(ctx, "bwd_col")] = dz_col
                    del dz_actgrad

                    # FP8 wgrad GEMM: dw2[e] = y1s_col @ dout_col^T -> [E, I, H].
                    _wgrad_accum_w2 = getattr(ctx, '_wgrad_w2_accumulator', None)
                    if _wgrad_accum_w2 is not None:
                        # partial() pins the quantized activations, so they
                        # outlive the ``del`` below when the host defers.
                        _run_wgrad_w2 = partial(
                            fp8_kernels.wgrad_gemm_varlen_k,
                            y1s_col, dout_col,
                            expert_frequency_offset,
                            M=w2_shape[2], N=dout.shape[1],
                            total_K=TK_wgrad, num_experts=w2_shape[0],
                            device=dout.device,
                            accumulator=_wgrad_accum_w2,
                        )
                        ctx._wgrad_w2_deferred = _defer_wgrad(
                            "moe_sonic_expert_down_proj",
                            ctx._w2_original,
                            _run_wgrad_w2,
                        )
                        if ctx._wgrad_w2_deferred is False:
                            _run_wgrad_w2()
                        # Drop our partial (a deferring hook holds its own), else
                        # it pins y1s_col / dout_col until backward returns.
                        del _run_wgrad_w2
                        dw2_base = None
                        dw2 = None
                    elif ctx._w2_input_needs_grad:
                        dw2_base = fp8_kernels.wgrad_gemm_varlen_k(
                            dout_col, y1s_col,
                            expert_frequency_offset,
                            M=dout.shape[1], N=w2_shape[2],
                            total_K=TK_wgrad, num_experts=w2_shape[0],
                            out_dtype=w2_dtype, device=dout.device,
                        )
                        dw2 = dw2_base.permute(1, 2, 0)
                    else:
                        dw2_base = dw2 = None
                    del dout_col, y1s_col
                    # A deferred dz_col (dz_col is None) is quantized by the
                    # UpProjection dw1 consumer itself (inside the EP comm slot
                    # when that is open), so it is never alive during dx.
                    del dz_col
                else:
                    _wgrad_accum_w2 = getattr(ctx, '_wgrad_w2_accumulator', None)
                    if _wgrad_accum_w2 is not None:
                        # BF16 wgrad + fp32 accumulate.
                        # Fleet keeps w2 main_grad in grouped layout [E, I, H].
                        # Present a Sonic-layout [E, H, I] view to the wgrad kernel.
                        y1s_wgrad = (
                            y1s if y1s.dtype == torch.bfloat16 else y1s.to(torch.bfloat16)
                        )
                        _wgrad_gemm_w2 = (
                            bf16_wgrad_gemm_varlen_k_accumulate
                            if _use_wgrad_beta_accum()
                            else bf16_wgrad_gemm_varlen_k_tma_add
                        )
                        # partial() pins dout / y1s_wgrad, so they outlive
                        # the ``del`` below when the host defers the GEMM.
                        _run_wgrad_w2 = partial(
                            _wgrad_gemm_w2,
                            dout.T,
                            y1s_wgrad,
                            expert_frequency_offset,
                            x_gather_idx,
                            accumulator=_wgrad_accum_w2.permute(0, 2, 1),
                            M=dout.shape[1],
                            N=w2_shape[2],
                            total_K=TK_wgrad,
                            num_experts=w2_shape[0],
                            device=dout.device,
                        )
                        ctx._wgrad_w2_deferred = _defer_wgrad(
                            "moe_sonic_expert_down_proj",
                            ctx._w2_original,
                            _run_wgrad_w2,
                        )
                        if ctx._wgrad_w2_deferred is False:
                            _run_wgrad_w2()
                        # The partial pins dout / y1s_wgrad too: drop it with them.
                        del _run_wgrad_w2
                        del y1s_wgrad
                        del y1s
                        dw2_base = None
                        dw2 = None
                    elif ctx._w2_input_needs_grad:
                        dw2_base = torch.empty((w2_shape[0], w2_shape[1], w2_shape[2]), dtype=w2_dtype, device=w2_device)
                        dw2 = dw2_base.permute(1, 2, 0)
                        y1s_wgrad = y1s if y1s.dtype == torch.bfloat16 else y1s.to(torch.bfloat16)
                        bf16_wgrad_gemm_varlen_k(
                            dout.T,
                            y1s_wgrad,
                            expert_frequency_offset,
                            x_gather_idx,
                            out=dw2.permute(2, 0, 1),
                            M=dout.shape[1],
                            N=w2_shape[2],
                            total_K=TK_wgrad,
                            num_experts=w2_shape[0],
                            device=dout.device,
                        )
                        del y1s_wgrad
                        del y1s
                    else:
                        dw2_base = dw2 = None
                        del y1s

                # Pre-quantize dz for UpProj.backward (non-wgrad path only;
                # wgrad path already did this above before dw2 allocation).
                if not ctx._fp8_cfg.fp8_wgrad:
                    dz_fp8, dz_packed_scales = (
                        dz_rowwise if dz_rowwise is not None
                        else fp8_kernels.quantize_activation(dz, ctx._fp8_cfg)
                    )
                    del dz_rowwise
                    _PREQUANTIZED_SCALES[_prequant_bwd_key(ctx, "bwd")] = (dz, dz_fp8, dz_packed_scales)
                # _log_stage_memory("In backward, after w2 grad")
                ds = ds[s_reverse_scatter_idx]
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
                    w2,
                    PreAct=z,
                    # BF16 path: _gemm_activation_name(fp8=False) refuses situ_glu.
                    activation=_gemm_activation_name(
                        activation_type, ctx._fp8_cfg, fp8=False,
                        site="_DownProjection.backward bf16",
                    ),
                    dx_out=dz,
                    colvec_scale=s.float(),
                    colvec_reduce=True,
                    cu_seqlens_m=expert_frequency_offset,
                    A_idx=x_gather_idx,
                    dynamic_scheduler=False,
                    tuned=False,
                    swiglu_clamp_value=ctx._fp8_cfg.swiglu_clamp_value,
                )
                # _log_stage_memory(
                #     "In backward, after down gemm bwd + swiglu bwd, "
                #     f"d_swiglu_input shape:{dz.shape}, dtype:{dz.dtype}"
                # )

                y1s_wgrad = y1s.to(torch.bfloat16) if y1s.dtype == torch.float8_e4m3fn else y1s
                _wgrad_accum_w2 = getattr(ctx, '_wgrad_w2_accumulator', None)
                if _wgrad_accum_w2 is not None:
                    _wgrad_gemm_w2 = (
                        bf16_wgrad_gemm_varlen_k_accumulate
                        if _use_wgrad_beta_accum()
                        else bf16_wgrad_gemm_varlen_k_tma_add
                    )
                    _run_wgrad_w2 = partial(
                        _wgrad_gemm_w2,
                        dout.T,
                        y1s_wgrad,
                        expert_frequency_offset,
                        x_gather_idx,
                        accumulator=_wgrad_accum_w2.permute(0, 2, 1),
                        M=dout.shape[1],
                        N=w2.shape[2],
                        total_K=x_gather_idx.shape[0],
                        num_experts=w2.shape[0],
                        device=dout.device,
                    )
                    ctx._wgrad_w2_deferred = _defer_wgrad(
                        "moe_sonic_expert_down_proj",
                        ctx._w2_original,
                        _run_wgrad_w2,
                    )
                    if not ctx._wgrad_w2_deferred:
                        _run_wgrad_w2()
                    dw2 = None
                elif ctx._w2_input_needs_grad:
                    dw2_base = torch.empty((w2.shape[0], w2.shape[1], w2.shape[2]), dtype=w2.dtype, device=w2.device)
                    dw2 = dw2_base.permute(1, 2, 0)
                    bf16_wgrad_gemm_varlen_k(
                        dout.T,
                        y1s_wgrad,
                        expert_frequency_offset,
                        x_gather_idx,
                        out=dw2.permute(2, 0, 1),
                        M=dout.shape[1],
                        N=w2.shape[2],
                        total_K=x_gather_idx.shape[0],
                        num_experts=w2.shape[0],
                        device=dout.device,
                    )
                else:
                    dw2_base = dw2 = None
                # _log_stage_memory("In backward, after w2 grad")
                ds = ds[s_reverse_scatter_idx]
        else:
            raise RuntimeError(
                "Non-QuACK GEMM path is removed. Set USE_QUACK_GEMM=1."
            )

        y1s = None  # may already be freed by fused dgated path
        # TC top-K routing
        # When route-level padding is active, topk_scores input was (T*K+N_pad,)
        # flat, but ds is (T*K,) after s_reverse_scatter_idx indexing.  Pad with
        # zeros so gradient shape matches input shape.
        N_scores = topk_scores.shape[0]
        if ds.shape[0] < N_scores:
            ds = torch.cat([ds, torch.zeros(
                N_scores - ds.shape[0], dtype=ds.dtype, device=ds.device
            )])
        elif not is_varlen_K:
            ds = ds.view(T, K)

        router_score_source_grad = None
        if (
            ctx._has_router_score_source
            and ctx._router_score_source_needs_grad
        ):
            # score_src_idx covers only real routes; the metadata score buffer may
            # include aligned padding.  The scatter helper zero-fills source slots
            # not represented by a received route, matching the former carrier.
            router_score_source_grad = _scatter_router_scores_i32(
                ds.contiguous(),
                router_score_src_idx,
                ctx._router_score_source_numel,
            ).reshape(ctx._router_score_source_shape)

        if fp8_combine_grad_handle is not None:
            fp8_combine_grad_handle.pop("data", None)
            fp8_combine_grad_handle.pop("scale", None)
        # Paddle PyLayer: return grads only for tensor inputs (not int/bool/enum/None)
        # Tensor inputs: y1, z, w2, [b2], topk_scores, selected_experts, expert_frequency_offset,
        #   x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, [num_activated_expert_per_token_offset]
        grads = [None, dz]  # y1, z, w2
        if ctx._w2_input_needs_grad and ctx._wgrad_w2_accumulator is None:
            grads.append(dw2.permute([2, 0, 1]))
        else:
            grads.append(None)
        if ctx._has_b2:
            grads.append(db2 if ctx._b2_input_needs_grad else None)
        topk_scores_grad = None if ctx._has_router_score_source else ds if ctx._topk_scores_needs_grad else None
        grads.extend([topk_scores_grad, None, None])  # topk_scores, selected_experts, expert_frequency_offset
        grads.extend([None, None, None])  # x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
        if ctx._has_num_activated:
            grads.append(None)
        if ctx._has_router_score_source:
            # Optional tensor inputs are returned only when present; fp8_config is
            # non-tensor and therefore has no corresponding Paddle PyLayer slot.
            grads.extend([router_score_source_grad, None])
        if ctx._wgrad_w2_accumulator is not None and not getattr(
            ctx, "_wgrad_w2_deferred", False
        ):
            # When deferred, the host hook fires this after it runs the GEMM —
            # main_grad is not complete yet at this point.
            _apply_weight_backward_hook(ctx._w2_original)
        return tuple(grads)