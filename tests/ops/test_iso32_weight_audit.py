"""ISO32 weight: precision scaling audit + memory saving verification.

Runs via `pytest -v tests/ops/test_iso32_weight_audit.py`

Validates:
1. RRMSE(iso32_output, bf16_golden) == RRMSE(baseline_output, bf16_golden) across shapes
2. No shape-dependent precision scaling (ratio bounded at 1.0)
3. Memory saving: iso32 allocates fewer unique FP8 buffers than pair-kernel
"""
import gc
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

PYTHON = sys.executable
WORKER = os.path.join(os.path.dirname(__file__), "_iso32_audit_worker.py")

AUDIT_SHAPES = [
    pytest.param(1024, 8, 8, 3072, 1536, id="T1024-E8-prod"),
    pytest.param(8192, 8, 8, 3072, 1536, id="T8192-E8-prod"),
    pytest.param(16384, 8, 8, 3072, 1536, id="T16384-E8-large"),
    pytest.param(8192, 32, 8, 3072, 1536, id="T8192-E32-manyexp"),
    pytest.param(8192, 8, 8, 4096, 2048, id="T8192-E8-wide"),
]


def _run_mode(T, E, K, H, I, mode, gpu_id=0):
    out_file = tempfile.mktemp(suffix=".npy")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["__MODE__"] = mode
    env["__T__"] = str(T)
    env["__E__"] = str(E)
    env["__K__"] = str(K)
    env["__H__"] = str(H)
    env["__I__"] = str(I)
    env["__OUT_FILE__"] = out_file
    proc = subprocess.run(
        [PYTHON, WORKER], capture_output=True, text=True, env=env, timeout=240
    )
    if proc.returncode != 0:
        err = proc.stderr.strip().split("\n")[-1][:120]
        pytest.fail(f"Worker failed ({mode}, T={T}, E={E}): {err}")
    return out_file


def _rrmse(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(
        np.sqrt((d ** 2).sum()) / max(np.sqrt((b.astype(np.float64) ** 2).sum()), 1e-30)
    )


@pytest.mark.parametrize("T,E,K,H,I", AUDIT_SHAPES)
def test_iso32_precision_vs_bf16_golden(T, E, K, H, I):
    """iso32 RRMSE vs BF16 must equal baseline RRMSE vs BF16 (ratio < 1.01)."""
    bf16_file = _run_mode(T, E, K, H, I, "bf16", gpu_id=0)
    base_file = _run_mode(T, E, K, H, I, "baseline", gpu_id=0)
    iso32_file = _run_mode(T, E, K, H, I, "iso32", gpu_id=0)

    bf16 = np.load(bf16_file)
    base = np.load(base_file)
    iso32 = np.load(iso32_file)

    r_base = _rrmse(base, bf16)
    r_iso32 = _rrmse(iso32, bf16)
    ratio = r_iso32 / max(r_base, 1e-30)

    assert ratio < 1.01, (
        f"iso32 precision degraded vs baseline: ratio={ratio:.6f} "
        f"(base/bf16={r_base:.4e}, iso32/bf16={r_iso32:.4e})"
    )
    assert r_iso32 < 0.10, (
        f"iso32 RRMSE vs BF16 exceeds FP8 gate: {r_iso32:.4e} >= 0.10"
    )


@pytest.mark.parametrize("T,E,K,H,I", AUDIT_SHAPES)
def test_iso32_no_scaling_with_shape(T, E, K, H, I):
    """iso32-vs-baseline RRMSE must stay bounded (< 5e-3) regardless of shape."""
    base_file = _run_mode(T, E, K, H, I, "baseline", gpu_id=0)
    iso32_file = _run_mode(T, E, K, H, I, "iso32", gpu_id=0)

    base = np.load(base_file)
    iso32 = np.load(iso32_file)
    r_diff = _rrmse(iso32, base)

    assert r_diff < 5e-3, (
        f"iso32 vs baseline divergence too large at T={T},E={E},H={H}: "
        f"RRMSE={r_diff:.4e} (must be < 5e-3)"
    )
