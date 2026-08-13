# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
from contextlib import contextmanager

from ..config import get_active_config


_IS_USING_QUACK_GEMM = os.getenv("USE_QUACK_GEMM", "0") == "1"


@contextmanager
def enable_quack_gemm(enable: bool = True):
    global _IS_USING_QUACK_GEMM

    previous_value = _IS_USING_QUACK_GEMM
    _IS_USING_QUACK_GEMM = enable

    yield

    _IS_USING_QUACK_GEMM = previous_value


def is_using_quack_gemm() -> bool:
    # Priority: active SonicMoEConfig > context manager > env var
    cfg = get_active_config()
    if cfg is not None and cfg.use_quack_gemm is not None:
        return cfg.use_quack_gemm
    return _IS_USING_QUACK_GEMM


# ---------------------------------------------------------------------------
# FP8 mode — single switch for the best-performing FP8 configuration.
#
# ``enable_fp8()`` activates the fork's blockscaled FP8 GEMM path with all
# optimal defaults (fused gated, fused SwiGLU+quant, FP8 z-save, BF16 wgrad).
# It also auto-enables QuACK GEMM which is required for the FP8 path.
#
# The env-var ``SONIC_MOE_FP8_MODE=perf`` is still respected for backward
# compatibility; ``enable_fp8()`` takes precedence.
#
# ``SonicMoEConfig`` (via ``get_active_config()``) takes highest priority.
# ---------------------------------------------------------------------------
_IS_FP8_ACTIVE = os.getenv("SONIC_MOE_FP8_MODE", "").strip().lower() in ("perf", "mem")


@contextmanager
def enable_fp8(enable: bool = True):
    """Context manager to enable/disable the FP8 fast path.

    When enabled, also activates QuACK GEMM (required for FP8).
    All optimal sub-flags are built into the defaults, so no extra
    environment variables are needed.
    """
    global _IS_FP8_ACTIVE, _IS_USING_QUACK_GEMM

    prev_fp8 = _IS_FP8_ACTIVE
    prev_quack = _IS_USING_QUACK_GEMM
    _IS_FP8_ACTIVE = enable
    if enable:
        _IS_USING_QUACK_GEMM = True

    yield

    _IS_FP8_ACTIVE = prev_fp8
    _IS_USING_QUACK_GEMM = prev_quack


def is_fp8_active() -> bool:
    # Priority: active SonicMoEConfig > context manager > env var
    cfg = get_active_config()
    if cfg is not None and cfg.use_fp8 is not None:
        return cfg.use_fp8
    return _IS_FP8_ACTIVE
