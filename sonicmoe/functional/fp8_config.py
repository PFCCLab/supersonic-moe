# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

"""FP8 runtime configuration — extracted to break circular import.

This module contains _FP8Config and its helper functions, plus the
arch-neutral FP8 runtime state/helpers shared by functional/__init__.py and the
SM100 backend (``_gemm_activation_name``, ``_STASHED_FP8_WEIGHTS``,
``_PREQUANT_HIT_COUNT``, ``_use_wgrad_beta_accum``). It is imported by
functional/__init__.py, quack_utils/blockscaled_fp8_gemm.py and
quack_utils/sm100_funcs.py, so it must NOT import from functional/__init__.py
or quack_utils to avoid cycles.
"""

from __future__ import annotations

import collections
import os

import torch

from ..config import (
    DEFAULT_SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA,
    SonicMoEConfig,
    get_active_config,
)
from ..enums import ActivationType
from ..situ_params import (
    encode_situ_activation,
    is_situ_activation,
    parse_situ_activation,
)
from .utils import is_fp8_active


# ---------------------------------------------------------------------------
# FP8 mode helpers
# ---------------------------------------------------------------------------

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
# Individual FP8 sub-flag helpers
# ---------------------------------------------------------------------------

def _use_epilogue_quant() -> bool:
    """Check if epilogue blockscaled quant of z is enabled (default: enabled)."""
    cfg = get_active_config()
    if cfg is not None and cfg.epilogue_quant is not None:
        return cfg.epilogue_quant
    return os.getenv("SONIC_MOE_FP8_EPILOGUE_QUANT", "1").lower() in {"1", "true", "yes", "on"}


def _use_fp8_wgrad() -> bool | None:
    """Check if FP8 weight gradients are enabled.

    Returns True/False if explicitly set (config or env var), None for auto-detect.
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
    """Check if z tensor should be stored in FP8 format to save memory (default: enabled)."""
    cfg = get_active_config()
    if cfg is not None and cfg.save_z_fp8 is not None:
        return cfg.save_z_fp8
    return os.getenv("SONIC_MOE_FP8_SAVE_Z_FP8", "0").lower() in {"1", "true", "yes", "on"}


def _recompute_z() -> bool:
    """Defer z_fp8 materialization to backward (saves ~213 MiB peak per layer)."""
    cfg = get_active_config()
    if cfg is not None and cfg.recompute_z is not None:
        return cfg.recompute_z
    return os.getenv("SONIC_MOE_FP8_RECOMPUTE_Z", "0").lower() in {"1", "true", "yes", "on"}


def _use_fuse_y1_quant() -> bool:
    """Fuse y1 (=SwiGLU(z)) FP8 quant into the up-proj GEMM epilogue.

    This is intentionally Python-config only.  It defaults on through
    ``SonicMoEConfig`` and does not consult a standalone env flag.
    """
    cfg = get_active_config()
    if cfg is not None:
        return cfg.resolve_fuse_y1_quant()
    return SonicMoEConfig().resolve_fuse_y1_quant()


def _use_fuse_y1_bf16_trunc() -> bool:
    """RNE-truncate fused y1 postact to bf16 before FP8 quant.

    Python-config only.  Defaults to tracking ``fuse_y1_quant`` (on when the
    fused y1 quant is on, for standalone byte-parity) but can be overridden
    independently via ``SonicMoEConfig(fuse_y1_bf16_trunc=...)``.
    """
    cfg = get_active_config()
    if cfg is not None:
        return cfg.resolve_fuse_y1_bf16_trunc()
    return SonicMoEConfig().resolve_fuse_y1_bf16_trunc()


def _use_fused_weight_layout() -> bool:
    """Use fused expert weight layout conversion kernels.

    This is Python-config only and defaults on.  Fleet integration consults the
    same resolver before choosing the fused layout path.
    """
    cfg = get_active_config()
    if cfg is not None:
        return cfg.resolve_fused_weight_layout()
    return SonicMoEConfig().resolve_fused_weight_layout()


# ---------------------------------------------------------------------------
# FP8 runtime config — resolved once per forward, passed via ctx
# ---------------------------------------------------------------------------
class _FP8Config:
    """Snapshot of all FP8 flags, resolved once at forward entry.

    Resolves from SonicMoEConfig (if active), then env vars, then defaults.
    Instances are cheap (no tensors), picklable, stored on autograd ctx.
    """
    __slots__ = (
        "enabled", "save_z_fp8", "recompute_z",
        "epilogue_quant", "_fp8_wgrad", "_fp8_wgrad_setting", "alignment_assumed",
        "iso32_weight", "swiglu_clamp_value", "fuse_y1_quant", "fuse_y1_bf16_trunc",
        "fused_weight_layout", "situ_beta", "situ_linear_beta",
        "activation_group_size", "weight_block_size", "fp8_backend",
    )

    def __init__(self) -> None:
        self.enabled: bool = _fp8_enabled()
        self.save_z_fp8: bool = _save_z_fp8()
        self.recompute_z: bool = _recompute_z()
        self.epilogue_quant: bool = _use_epilogue_quant()
        self._fp8_wgrad_setting = _use_fp8_wgrad()  # True/False/None
        self._fp8_wgrad: bool = self._fp8_wgrad_setting or False  # resolved in resolve_wgrad
        self.alignment_assumed: bool = False
        self.iso32_weight: bool = os.environ.get("SONIC_MOE_FP8_ISO32_WEIGHT", "0") == "1"
        self.fuse_y1_quant: bool = _use_fuse_y1_quant()
        self.fuse_y1_bf16_trunc: bool = _use_fuse_y1_bf16_trunc()
        self.fused_weight_layout: bool = _use_fused_weight_layout()
        self.activation_group_size: int = 32
        self.weight_block_size: int = 32
        self.fp8_backend = None
        _active = get_active_config()
        self.swiglu_clamp_value: float = (
            _active.resolve_swiglu_clamp_value() if _active is not None else 0.0
        )
        # SiTU-GLU shape constants.  Always resolved (cheap) so that every GEMM
        # call site can build the encoded activation string from ``cfg`` alone;
        # they are ignored unless the activation is ActivationType.SITU_GLU.
        _situ_src = _active if _active is not None else SonicMoEConfig()
        self.situ_beta: float = _situ_src.resolve_situ_beta()
        self.situ_linear_beta = _situ_src.resolve_situ_linear_beta()

    @property
    def fp8_wgrad(self) -> bool:
        return self._fp8_wgrad

    @fp8_wgrad.setter
    def fp8_wgrad(self, value: bool) -> None:
        self._fp8_wgrad = value
        self._fp8_wgrad_setting = value

    # Threshold below which FP8 wgrad quant overhead exceeds GEMM savings.
    _WGRAD_FP8_I_THRESHOLD = 0

    def resolve_wgrad(self, I: int) -> None:
        """Resolve FP8 wgrad based on explicit setting or shape-based heuristic."""
        if self._fp8_wgrad_setting is not None:
            self._fp8_wgrad = self._fp8_wgrad_setting
        else:
            self._fp8_wgrad = I >= self._WGRAD_FP8_I_THRESHOLD

    @staticmethod
    def disabled() -> "_FP8Config":
        """Return a config where everything is off (BF16 path)."""
        cfg = _FP8Config.__new__(_FP8Config)
        cfg.enabled = False
        cfg.save_z_fp8 = False
        cfg.recompute_z = False
        cfg.epilogue_quant = False
        cfg.fp8_wgrad = False
        cfg._fp8_wgrad_setting = False
        cfg.alignment_assumed = False
        cfg.iso32_weight = False
        cfg.swiglu_clamp_value = 0.0
        cfg.fuse_y1_quant = False
        cfg.fuse_y1_bf16_trunc = False
        cfg.fused_weight_layout = False
        cfg.activation_group_size = 32
        cfg.weight_block_size = 32
        cfg.fp8_backend = None
        # BF16 path never reaches a SiTU epilogue (guarded in
        # ``_gemm_activation_name`` below), but keep the slots populated so
        # ``cfg.situ_beta`` is never an AttributeError.
        cfg.situ_beta = DEFAULT_SITU_BETA
        cfg.situ_linear_beta = DEFAULT_SITU_LINEAR_BETA
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


# Stashed FP8 weight references — populated by MoE.stash_bf16_to_cpu(),
# consumed by the SM100 FP8 backend (quack_utils/sm100_funcs.py) and
# functional._DownProjection.forward
# to bypass global cache lookups when bf16 param storage has been freed.
# Keys: "w1_fused", "w2_varlen", "w2_dgated", "w1T_varlen"
_STASHED_FP8_WEIGHTS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

# Counter for pre-quantization hits (testing/diagnostics).
_PREQUANT_HIT_COUNT: dict[str, int] = collections.defaultdict(int)


def _use_wgrad_beta_accum() -> bool:
    return os.getenv("SONIC_MOE_FP8_WGRAD_TMA_ADD", "1").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Activation representation boundary
# ---------------------------------------------------------------------------
# Two representations of "which activation" coexist, and this function is the
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
# ``encode_situ_activation()`` for SiTU.  Nothing else may build a GEMM
# activation string by hand.
#
# It is also the single place that refuses SiTU on the BF16 path: pass
# ``fp8=False`` and a SiTU activation raises NotImplementedError, so it is
# structurally impossible for a BF16 caller to silently fall back to SwiGLU.

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
                f"sonicmoe/functional/fp8_config.py [{site}]: situ_glu is FP8-only "
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
                f"sonicmoe/functional/fp8_config.py [{site}]: situ_glu is FP8-only "
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
        f"sonicmoe/functional/fp8_config.py [{site}]: activation "
        f"{activation_type!r} has no QuACK gated-GEMM epilogue. Supported: "
        f"{sorted(_RAW_GEMM_ACTIVATIONS)} plus situ_glu (FP8 only)."
    )
