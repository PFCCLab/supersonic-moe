# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

"""Unified SM-count cap for all frontier CUTLASS-DSL persistent GEMMs.

Single knob ``SonicMoEConfig.gemm_num_sms`` (env ``SONIC_MOE_GEMM_NUM_SMS``)
caps the number of SMs each persistent GEMM may occupy, mirroring DeepGEMM's
``set_num_sms``.  Intended for compute/communication multi-stream overlap:
capping the GEMM leaves SMs free for a DeepEP/HybridEP comm kernel on another
stream.

Two mechanisms, applied together at every frontier GEMM call site:

  1. ``capped_max_active_clusters`` narrows the runtime-dynamic
     ``max_active_clusters`` (SM budget -> cluster budget) that drives the
     STATIC/DYNAMIC tile scheduler's persistent grid (grid.z).

  2. ``clc_persistence_default`` forces ``use_clc_persistence=False`` whenever
     the cap is set.  On Blackwell (SM100+) the default CLC scheduler path
     ignores ``max_active_clusters`` entirely (grid is always the full problem
     grid), so the cap is only effective under STATIC scheduling.  This is a
     compile-time constant, so the wrappers must also fold ``sm_cap_enabled()``
     into their compile keys.
"""

from __future__ import annotations

from typing import Optional

from quack.cute_dsl_utils import get_max_active_clusters


def resolve_gemm_num_sms() -> Optional[int]:
    """Active SM cap, or None (use all SMs).

    Priority: active ``SonicMoEConfig.gemm_num_sms`` > env
    ``SONIC_MOE_GEMM_NUM_SMS`` > None.  The ``..config`` import is deferred to
    call time to avoid an import-time cycle (config <-> quack_utils).
    """
    from ..config import get_active_config

    cfg = get_active_config()
    if cfg is not None:
        return cfg.resolve_gemm_num_sms()

    import os

    raw = os.getenv("SONIC_MOE_GEMM_NUM_SMS", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def sm_cap_enabled() -> bool:
    """True iff an SM cap is active (used as a compile-key discriminator)."""
    return resolve_gemm_num_sms() is not None


def capped_max_active_clusters(cluster_size: int, *, persistent: bool = True) -> int:
    """Persistent-grid cluster count, narrowed to the active SM cap.

    Mirrors ``get_max_active_clusters(cluster_size) if persistent else 0`` when
    no cap is set.  With a cap of ``n`` SMs, returns
    ``max(1, min(hw, n // max(1, cluster_size)))`` -- the SM budget converted
    to a cluster budget (floor division; at least one cluster).
    """
    if not persistent:
        return 0
    hw = get_max_active_clusters(cluster_size)
    n = resolve_gemm_num_sms()
    if n is None:
        return hw
    cs = max(1, int(cluster_size))
    return max(1, min(int(hw), int(n) // cs))


def clc_persistence_default(base: bool = True) -> bool:
    """CLC-persistence flag: forced off (STATIC) when the SM cap is active.

    ``base`` is the value the call site would otherwise use (default True for
    the SM100 frontier GEMMs).  When no cap is set, ``base`` is returned
    unchanged, so uncapped runs keep their original CLC behavior bit-for-bit.
    """
    if sm_cap_enabled():
        return False
    return base
