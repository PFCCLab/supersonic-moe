# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

"""Pythonic configuration for SonicMoE FP8 and GEMM settings.

Replaces environment-variable-based configuration with a structured dataclass.
Environment variables are still respected as the lowest-priority fallback.

Priority order (highest -> lowest):
  1. Explicit field values in ``SonicMoEConfig``
  2. ``enable_fp8()`` / ``enable_quack_gemm()`` context managers
  3. Environment variables (``SONIC_MOE_FP8_MODE``, ``USE_QUACK_GEMM``, etc.)

Usage::

    from sonicmoe import MoE, SonicMoEConfig

    cfg = SonicMoEConfig(use_fp8=True, fp8_wgrad=True)
    moe = MoE(..., config=cfg)
    out, loss = moe(x)           # FP8 enabled via config, no env vars needed

    # Or use context manager for temporary override:
    cfg2 = SonicMoEConfig(use_fp8=True, save_z_fp8=False)
    with cfg2.activate():
        out, loss = moe(x)
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Optional


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    """Read a boolean from an environment variable. Returns None if unset."""
    val = os.getenv(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class SonicMoEConfig:
    """Configuration for SonicMoE FP8 and GEMM behavior.

    All fields default to ``None``, meaning "use env var or built-in default".
    Set a field explicitly to override the corresponding env var.

    Attributes:
        use_fp8: Enable the FP8 fast path. Env: ``SONIC_MOE_FP8_MODE``.
        use_quack_gemm: Enable QuACK GEMM backend. Env: ``USE_QUACK_GEMM``.
            Auto-enabled when ``use_fp8=True``.
        fp8_wgrad: Enable FP8 weight gradients. Env: ``SONIC_MOE_FP8_WGRAD``.
            Default: auto (ON at I>=2048, OFF at I<2048).
        fused_gated: Use fused gemm_gated/dgated CUTLASS kernels.
            Env: ``SONIC_MOE_FP8_FUSED_GATED``. Default: True.
        save_z_fp8: Save z tensor in FP8 to reduce memory.
            Env: ``SONIC_MOE_FP8_SAVE_Z_FP8``. Default: True.
        fused_swiglu_quant: Use fused SwiGLU+quantize kernels.
            Env: ``SONIC_MOE_FP8_FUSED_SWIGLU_QUANT``. Default: True.
        epilogue_quant: Enable epilogue blockscaled quant of z.
            Env: ``SONIC_MOE_FP8_EPILOGUE_QUANT``. Default: False.
        fused_zy1_quant: Enable fused z+y1 quantization.
            Env: ``SONIC_MOE_FP8_FUSED_ZY1_QUANT``. Default: False.
        fuse_y1_quant: Enable y1 postact FP8 quantization in the up-proj epilogue.
            Python-config only. Default: False.
        fuse_y1_bf16_trunc: RNE-truncate fused y1 postact to bf16 before FP8 quant.
            Python-config only. Default: tracks ``fuse_y1_quant`` (None ->
            follows fuse_y1_quant); set explicitly to decouple.
        assume_aligned: Force alignment assumption (skip D2H check).
            Env: ``SONIC_MOE_FP8_ASSUME_ALIGNED``. Default: False.
        stagewise_memory: Enable per-stage memory logging.
            Env: ``SONIC_MOE_STAGEWISE_MEMORY``. Default: False.
        swiglu_clamp_value: Optional user-controlled SwiGLU clamp value.
            Default: 0.0 (disabled).
        gemm_num_sms: Absolute SM-count cap for all frontier CUTLASS GEMMs
            (single knob, à la DeepGEMM ``set_num_sms``). None = use all SMs.
            Non-None auto-switches the scheduler off Blackwell CLC onto STATIC
            so the cap takes effect. Env: ``SONIC_MOE_GEMM_NUM_SMS``.
    """

    use_fp8: Optional[bool] = None
    use_quack_gemm: Optional[bool] = None
    fp8_wgrad: Optional[bool] = None
    fused_gated: Optional[bool] = None
    save_z_fp8: Optional[bool] = None
    recompute_z: Optional[bool] = None
    fused_swiglu_quant: Optional[bool] = None
    epilogue_quant: Optional[bool] = None
    fused_zy1_quant: Optional[bool] = None
    fuse_y1_quant: Optional[bool] = None
    fuse_y1_bf16_trunc: Optional[bool] = None
    assume_aligned: Optional[bool] = None
    stagewise_memory: Optional[bool] = None
    iso32_weight: Optional[bool] = None
    swiglu_clamp_value: Optional[float] = None
    gemm_num_sms: Optional[int] = None

    def __post_init__(self) -> None:
        # Auto-enable quack_gemm when fp8 is explicitly enabled.
        if self.use_fp8 is True and self.use_quack_gemm is None:
            self.use_quack_gemm = True

    # --- Resolution: config field > env var > built-in default ---------------

    def resolve_use_fp8(self) -> bool:
        if self.use_fp8 is not None:
            return self.use_fp8
        return _env_bool("SONIC_MOE_FP8_MODE", False) or False

    def resolve_use_quack_gemm(self) -> bool:
        if self.use_quack_gemm is not None:
            return self.use_quack_gemm
        return os.getenv("USE_QUACK_GEMM", "0") == "1"

    def resolve_fp8_wgrad(self) -> Optional[bool]:
        """Return True/False if explicitly set, None for auto-detect."""
        if self.fp8_wgrad is not None:
            return self.fp8_wgrad
        return _env_bool("SONIC_MOE_FP8_WGRAD")

    def resolve_fused_gated(self) -> bool:
        if self.fused_gated is not None:
            return self.fused_gated
        return _env_bool("SONIC_MOE_FP8_FUSED_GATED", True) or False

    def resolve_save_z_fp8(self) -> bool:
        if self.save_z_fp8 is not None:
            return self.save_z_fp8
        return _env_bool("SONIC_MOE_FP8_SAVE_Z_FP8", True) or False

    def resolve_recompute_z(self) -> bool:
        """Recompute z_fp8 in DownProj backward (skip storing in forward).

        When True, ``_UpProjection.forward`` does NOT populate the
        ``z_fp8`` prequant cache and ``_DownProjection.forward`` saves a
        recompute closure to ctx instead of the fp8 z tensor.  Backward
        re-runs the up-proj GEMM (gather-A + epilogue blockscaled fp8
        quant) just-in-time and discards the recomputed y1 (SwiGLU /
        PostAct write are wasted, ~5-15% of an up-proj fwd cost).
        Saves ~213 MiB peak (fp8 z) per active layer at ERNIE shape.
        Implies save_z_fp8=True semantically.
        """
        if self.recompute_z is not None:
            return self.recompute_z
        return _env_bool("SONIC_MOE_FP8_RECOMPUTE_Z", False) or False

    def resolve_fused_swiglu_quant(self) -> bool:
        if self.fused_swiglu_quant is not None:
            return self.fused_swiglu_quant
        return _env_bool("SONIC_MOE_FP8_FUSED_SWIGLU_QUANT", True) or False

    def resolve_epilogue_quant(self) -> bool:
        if self.epilogue_quant is not None:
            return self.epilogue_quant
        return _env_bool("SONIC_MOE_FP8_EPILOGUE_QUANT", False) or False

    def resolve_fused_zy1_quant(self) -> bool:
        if self.fused_zy1_quant is not None:
            return self.fused_zy1_quant
        return _env_bool("SONIC_MOE_FP8_FUSED_ZY1_QUANT", False) or False

    def resolve_fuse_y1_quant(self) -> bool:
        if self.fuse_y1_quant is not None:
            return self.fuse_y1_quant
        return False

    def resolve_fuse_y1_bf16_trunc(self) -> bool:
        # Defaults to tracking fuse_y1_quant: enabling the fused y1 quant turns
        # on bf16-trunc too (legacy/standalone byte-parity), but it can be set
        # independently to either value when an explicit override is desired.
        if self.fuse_y1_bf16_trunc is not None:
            return self.fuse_y1_bf16_trunc
        return self.resolve_fuse_y1_quant()

    def resolve_assume_aligned(self) -> bool:
        if self.assume_aligned is not None:
            return self.assume_aligned
        return _env_bool("SONIC_MOE_FP8_ASSUME_ALIGNED", False) or False

    def resolve_stagewise_memory(self) -> bool:
        if self.stagewise_memory is not None:
            return self.stagewise_memory
        return _env_bool("SONIC_MOE_STAGEWISE_MEMORY", False) or False

    def resolve_iso32_weight(self) -> bool:
        if self.iso32_weight is not None:
            return self.iso32_weight
        return _env_bool("SONIC_MOE_FP8_ISO32_WEIGHT", False) or False

    def resolve_swiglu_clamp_value(self) -> float:
        if self.swiglu_clamp_value is not None:
            return max(float(self.swiglu_clamp_value), 0.0)
        return 0.0

    def resolve_gemm_num_sms(self) -> Optional[int]:
        """Global SM-count cap for all frontier CUTLASS GEMMs.

        Returns an absolute upper bound on the number of SMs a persistent
        GEMM may occupy, or ``None`` for "use all SMs" (unchanged behavior).
        Intended for compute/communication multi-stream overlap: cap the
        GEMM so a DeepEP/HybridEP comm kernel on another stream can claim
        the freed SMs.  Mirrors DeepGEMM's ``set_num_sms`` single-knob API.

        When non-None, the GEMM wrappers additionally switch the persistent
        tile scheduler off the Blackwell CLC path (which ignores the SM cap)
        onto STATIC scheduling, where the cap actually shrinks the grid.
        Env fallback: ``SONIC_MOE_GEMM_NUM_SMS`` (int).
        """
        if self.gemm_num_sms is not None:
            return int(self.gemm_num_sms)
        raw = os.getenv("SONIC_MOE_GEMM_NUM_SMS", "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                return None
        return None

    # --- Context manager for temporary activation ----------------------------

    @contextmanager
    def activate(self):
        """Context manager: push this config as the active thread-local config.

        Usage::

            cfg = SonicMoEConfig(use_fp8=True, fp8_wgrad=True)
            with cfg.activate():
                out, loss = moe(x)
        """
        prev = get_active_config()
        set_active_config(self)
        try:
            yield self
        finally:
            set_active_config(prev)

    def replace(self, **kwargs) -> "SonicMoEConfig":
        """Return a new config with selected fields overridden."""
        from dataclasses import asdict
        d = asdict(self)
        d.update(kwargs)
        return SonicMoEConfig(**d)


# ---------------------------------------------------------------------------
# Thread-local active config — set by SonicMoEConfig.activate() or directly
# ---------------------------------------------------------------------------
_active_config = threading.local()


def get_active_config() -> Optional[SonicMoEConfig]:
    """Return the currently active SonicMoEConfig, or None."""
    return getattr(_active_config, "config", None)


def set_active_config(cfg: Optional[SonicMoEConfig]) -> None:
    """Set the active SonicMoEConfig for the current thread."""
    _active_config.config = cfg
