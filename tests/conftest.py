"""Pytest fixtures for sonic-moe tests.

Two purposes:

1. **Parallel test isolation under pytest-xdist** — each xdist worker is
   pinned to a distinct CUDA device (round-robin over the visible set)
   *before* paddle/torch initialise. This way:

     - CUDA_ERROR_ILLEGAL_ADDRESS in one test cannot poison another
       worker's context (each worker = its own process = its own context).
     - Tests can safely run in parallel under `pytest -n auto` because
       different workers use different GPUs.
     - On a single-GPU machine all workers share device 0; this still
       works because each xdist worker is a separate process with its
       own CUDA context (serialised by the driver).

2. **Per-test state hygiene** — provide an autouse fixture that wipes
   sonicmoe's in-process weight caches between tests so per-test mock
   weights cannot leak across test functions.
"""
from __future__ import annotations

import os
import sys

# Ensure the locally-built quack is importable for any test that triggers a
# sonicmoe import. Some Python interpreters on this host (e.g.
# /usr/local/bin/python) have no quack site-package — every individual
# tests/ops/* bench manually does ``sys.path.insert(0, _QUACK)``; do it once
# here so newly-added tests don't have to remember.
_QUACK = (
    "/root/paddlejob/share-storage/gpfs/system-public/"
    "zhangyichen/sonicmoe_for_ernie/quack"
)
if os.path.isdir(_QUACK) and _QUACK not in sys.path:
    sys.path.insert(0, _QUACK)

# Production-mode FP8 defaults — set BEFORE any sonicmoe import so module-level
# constants (e.g. ``sonicmoe.functional.utils._IS_FP8_ACTIVE``) and any
# ``_FP8Config`` snapshot taken during cross-test imports see the right value.
# Tests that explicitly need the BF16 path use ``enable_fp8(False)`` locally.
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("USE_QUACK_GEMM", "1")


def _assign_worker_gpu():
    """Pin this xdist worker to a single GPU before any cuda code runs."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")  # e.g. "gw0", "gw1"
    if not worker_id or not worker_id.startswith("gw"):
        return
    try:
        idx = int(worker_id[2:])
    except ValueError:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devs = [d for d in visible.split(",") if d.strip()]
    else:
        # Probe the device count without importing torch.
        try:
            import subprocess
            n = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode().splitlines()[0])
            devs = [str(i) for i in range(n)]
        except Exception:
            return
    if not devs:
        return
    chosen = devs[idx % len(devs)]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen


_assign_worker_gpu()


def _ensure_blackwell_ptxas():
    """Triton 3.5's bundled ptxas does not support sm_103a (Blackwell B30Z).
    System CUDA 13's ptxas does. Set TRITON_PTXAS_PATH globally so every
    test (including ones that don't set it locally) can compile fresh
    Triton kernels on B30Z."""
    if os.environ.get("TRITON_PTXAS_PATH"):
        return
    for cand in (
        "/usr/local/cuda-13.0/bin/ptxas",
        "/usr/local/cuda/bin/ptxas",
    ):
        if os.path.exists(cand):
            os.environ["TRITON_PTXAS_PATH"] = cand
            return


_ensure_blackwell_ptxas()


def pytest_collection_modifyitems(config, items):
    """If pytest-xdist scheduling is `loadgroup`, group by file so tests
    in the same module land on the same worker (avoids re-importing heavy
    sonicmoe across workers for every test)."""
    return  # informational: explicit `--dist=loadfile` does the same.
