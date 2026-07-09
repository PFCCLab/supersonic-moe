# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

"""Unified SM-occupancy cap for the frontier CuTe-DSL persistent GEMMs.

Every persistent GEMM in the FP8/BF16 frontier launches with a
``max_active_clusters`` value that upper-bounds how many persistent clusters
(hence SMs, since one CTA == one SM) the grid may occupy.  By default that is
the hardware full-occupancy value from ``get_max_active_clusters(cluster_size)``.

To overlap compute and communication on separate CUDA streams we need to leave
some SMs free for the comm kernels.  ``capped_max_active_clusters`` shrinks the
per-launch ``max_active_clusters`` to honor a global SM budget
(``SonicMoEConfig.gemm_num_sms`` / env ``SONIC_MOE_GEMM_NUM_SMS``), mirroring
DeepGEMM's single ``set_num_sms`` knob.

The cap only lowers grid.z; it never changes the reduction order, so results
stay bit-identical to the uncapped run.  ``max_active_clusters`` is passed to
the kernel as a runtime dynamic ``Int32`` (via ``TileSchedulerOptions``), so
changing the cap does NOT force a recompile.

This module deliberately depends only on ``..config`` (single-directional, no
cycle) and ``quack.cute_dsl_utils``.
"""

from __future__ import annotations

from typing import Optional

from quack.cute_dsl_utils import get_max_active_clusters

from ..config import get_active_config


def _resolve_gemm_num_sms() -> Optional[int]:
    """Return the global SM budget for frontier GEMMs, or None for all SMs.

    Priority: active ``SonicMoEConfig`` > env ``SONIC_MOE_GEMM_NUM_SMS`` > None.
    Delegates entirely to ``SonicMoEConfig.resolve_gemm_num_sms`` (which also
    reads the env fallback) so there is exactly one place that knows the
    resolution order.
    """
    cfg = get_active_config()
    if cfg is not None:
        return cfg.resolve_gemm_num_sms()
    # No active config: still honor the env var for scripts / standalone use.
    import os

    raw = os.getenv("SONIC_MOE_GEMM_NUM_SMS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def capped_max_active_clusters(cluster_size: int, *, persistent: bool = True) -> int:
    """Hardware full-occupancy clusters, capped to the global SM budget.

    Args:
        cluster_size: number of CTAs per cluster (``cluster_M * cluster_N``).
        persistent: when False the caller uses a non-persistent scheduler and
            expects 0 (matching the historical ``... if persistent else 0``).

    Returns:
        The ``max_active_clusters`` to pass to the tile scheduler.  When a SM
        budget is set, this is ``min(hw, num_sms // cluster_size)`` clamped to
        at least 1 cluster; otherwise the hardware value.
    """
    if not persistent:
        return 0
    hw = get_max_active_clusters(cluster_size)
    n = _resolve_gemm_num_sms()
    if n is None:
        return hw
    cs = max(1, int(cluster_size))
    # SM budget -> cluster budget: each cluster occupies `cluster_size` SMs.
    budget_clusters = n // cs
    return max(1, min(hw, budget_clusters))
