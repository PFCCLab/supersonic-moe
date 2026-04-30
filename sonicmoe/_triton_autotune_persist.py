# ********************************************************************************
# Copyright (c) 2025 SonicMoE contributors
# ********************************************************************************
"""Persistent Triton autotune cache.

Triton 3.6 already implements a robust per-key disk cache for autotune
*results* (best config + per-config timings), but it is **disabled by
default** (`triton.knobs.autotuning.cache = False`). Without it, every
fresh process re-runs the empirical sweep for every Triton autotuned
kernel — for the production sonic-moe path (4 autotuned kernels, the
biggest of which is ``token_gather_sum_kernel`` at ~30 s wall per cold
process) this is the dominant cold-start cost on a multi-rank shared-GPFS
cluster.

Enabling it is a one-line knob flip *but* must happen **before any
``@triton.autotune`` decorator runs**, because ``Autotuner.__init__``
snapshots ``self.cache_results = (cache_results or knobs.autotuning.cache)``
at decoration time. We therefore set the env var at the very top of
``sonicmoe/__init__.py`` (which itself is the first sonic-moe import) so
that:

  * every sonic-moe Triton autotuned kernel honours it,
  * upstream ``quack`` Triton autotuners (decorated during sonicmoe's
    ``_quack_compat`` import) honour it too, for free.

The cache directory is governed by Triton's standard
``TRITON_CACHE_DIR`` knob; sonic-moe's CI orchestrator sets it to
``$SONIC_MOE_CACHE_DIR/triton_cache`` so cold/warm/reload measurements
all share the same disk store. On shared GPFS with N ranks, Triton's
own ``cache_manager`` does atomic ``rename`` for writes — multi-process
safe.

Opt-out for debugging:

    SONIC_MOE_NO_TRITON_AUTOTUNE_CACHE=1
"""

from __future__ import annotations

import os

_PATCHED = False


def _install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("SONIC_MOE_NO_TRITON_AUTOTUNE_CACHE", "0") == "1":
        return

    # If the caller has explicitly set the env var (either way), honour it.
    if "TRITON_CACHE_AUTOTUNING" not in os.environ:
        os.environ["TRITON_CACHE_AUTOTUNING"] = "1"

    # Some Triton builds read knobs lazily (env), some snapshot at import
    # time. Force the in-process knob too in case Triton was already imported
    # before us (defensive).
    try:
        import triton.knobs as _knobs

        try:
            _knobs.autotuning.cache = True
        except Exception:
            pass
    except Exception:
        # Triton not installed / not importable → nothing to do.
        return

    _PATCHED = True


_install()
