# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

"""FP8 runtime configuration — extracted to break circular import.

This module contains _FP8Config and its helper functions. It is imported by
both functional/__init__.py and quack_utils/blockscaled_fp8_gemm.py, so it
must NOT import from functional/__init__.py or quack_utils to avoid cycles.
"""

from __future__ import annotations

import os

from ..config import (
    DEFAULT_SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA,
    SonicMoEConfig,
    get_active_config,
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


def _use_fused_swiglu_quant() -> bool:
    """Check if fused SwiGLU+quantize kernels are enabled (default: enabled)."""
    cfg = get_active_config()
    if cfg is not None and cfg.fused_swiglu_quant is not None:
        return cfg.fused_swiglu_quant
    return os.getenv("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", "1").lower() in {"1", "true", "yes", "on"}


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


def _use_fused_blockscaled_gated() -> bool:
    """Check if fused gemm_gated + blockscaled FP8 is enabled (default: enabled)."""
    cfg = get_active_config()
    if cfg is not None and cfg.fused_gated is not None:
        return cfg.fused_gated
    return os.getenv("SONIC_MOE_FP8_FUSED_GATED", "1").lower() in {"1", "true", "yes", "on"}


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
        "enabled", "fused_gated", "save_z_fp8", "recompute_z", "fused_swiglu_quant",
        "epilogue_quant", "_fp8_wgrad", "_fp8_wgrad_setting", "alignment_assumed",
        "iso32_weight", "swiglu_clamp_value", "fuse_y1_quant", "fuse_y1_bf16_trunc",
        "fused_weight_layout", "situ_beta", "situ_linear_beta",
    )

    def __init__(self) -> None:
        self.enabled: bool = _fp8_enabled()
        self.fused_gated: bool = _use_fused_blockscaled_gated()
        self.save_z_fp8: bool = _save_z_fp8()
        self.recompute_z: bool = _recompute_z()
        self.fused_swiglu_quant: bool = _use_fused_swiglu_quant()
        self.epilogue_quant: bool = _use_epilogue_quant()
        self._fp8_wgrad_setting = _use_fp8_wgrad()  # True/False/None
        self._fp8_wgrad: bool = self._fp8_wgrad_setting or False  # resolved in resolve_wgrad
        self.alignment_assumed: bool = False
        self.iso32_weight: bool = os.environ.get("SONIC_MOE_FP8_ISO32_WEIGHT", "0") == "1"
        self.fuse_y1_quant: bool = _use_fuse_y1_quant()
        self.fuse_y1_bf16_trunc: bool = _use_fuse_y1_bf16_trunc()
        self.fused_weight_layout: bool = _use_fused_weight_layout()
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
        cfg.fused_gated = False
        cfg.save_z_fp8 = False
        cfg.recompute_z = False
        cfg.fused_swiglu_quant = False
        cfg.epilogue_quant = False
        cfg.fp8_wgrad = False
        cfg._fp8_wgrad_setting = False
        cfg.alignment_assumed = False
        cfg.iso32_weight = False
        cfg.swiglu_clamp_value = 0.0
        cfg.fuse_y1_quant = False
        cfg.fuse_y1_bf16_trunc = False
        cfg.fused_weight_layout = False
        # BF16 path never reaches a SiTU epilogue (guarded in functional/
        # __init__.py::_gemm_activation_name), but keep the slots populated so
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
