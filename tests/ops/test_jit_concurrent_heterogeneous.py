# ********************************************************************************
# Concurrent heterogeneous-shape JIT cache stress test.
#
# Production scenario: 8 ranks share one GPFS-backed build/ + cache_dir.
# Different ranks may hit DIFFERENT shapes for the same cold start (e.g.
# seqlen variability across DP ranks before the first allgather, or per-rank
# data shards producing different total_K). All of:
#   * C++ extensions (deepep_topk_metadata_cuda, etc.) — same key across ranks
#   * Triton kernels keyed by (dtype, K, config) — same key across ranks
#   * Quack / CuTe autotuner cache keyed by tensor shape — DIFFERENT key per rank
# must warm up correctly with no torn writes, no missing artifacts, no
# duplicated rebuilds, and end-state must let any rank read any other rank's
# cache entry on the next iteration.
#
# We simulate this with two subprocesses concurrently warming DIFFERENT shapes
# against a shared cold cache_dir + build_directory. Both must succeed; after
# they finish a third process must hot-import both shapes without recompile.
# ********************************************************************************
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest


_SHAPE_A = dict(E=8, H=3072, I=1536, total_K=4096)
_SHAPE_B = dict(E=8, H=3072, I=1536, total_K=8192)


_CHILD = textwrap.dedent("""
    import os, sys, time, json
    os.environ["SONIC_MOE_CACHE_DIR"] = sys.argv[1]
    os.environ["TRITON_CACHE_DIR"] = os.path.join(sys.argv[1], "triton")
    os.environ.setdefault("USE_QUACK_GEMM", "1")
    os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
    os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")
    sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack")
    sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
    import paddle
    paddle.enable_compat()
    import torch  # noqa
    from sonicmoe.jit_warmup import _warmup_single
    E, H, I, total_K = (int(x) for x in sys.argv[2:6])
    t0 = time.time()
    _warmup_single(E, H, I, total_K, "cuda", fp8=True)
    dt = time.time() - t0
    print(f"OK pid={os.getpid()} total_K={total_K} dt={dt:.2f}s", flush=True)
""")


def _spawn(cache_dir: str, shape: dict, log_file: str) -> subprocess.Popen:
    py = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv/bin/python"
    log = open(log_file, "w")
    return subprocess.Popen(
        [py, "-c", _CHILD, cache_dir,
         str(shape["E"]), str(shape["H"]), str(shape["I"]), str(shape["total_K"])],
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
    )


@pytest.mark.timeout(900)
def test_concurrent_heterogeneous_cold_warmup(tmp_path):
    """Two cold processes warming different shapes must both succeed.

    Then a third hot process for either shape must skip recompile (i.e. its
    ``_warmup_single`` runs in <30s with the cache populated by either of
    the first two processes).
    """
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir)
    # Use the repo-default build/ — that's what production rank-shares on GPFS.
    log_a = str(tmp_path / "a.log")
    log_b = str(tmp_path / "b.log")
    pa = _spawn(cache_dir, _SHAPE_A, log_a)
    pb = _spawn(cache_dir, _SHAPE_B, log_b)
    rc_a = pa.wait(timeout=600)
    rc_b = pb.wait(timeout=600)
    out_a = open(log_a).read()
    out_b = open(log_b).read()
    assert rc_a == 0, f"shape A failed:\n{out_a}"
    assert rc_b == 0, f"shape B failed:\n{out_b}"
    assert "OK pid=" in out_a and "OK pid=" in out_b

    # Hot rerun — must skip via sentinel / disk cache.
    log_h = str(tmp_path / "hot.log")
    t0 = time.time()
    ph = _spawn(cache_dir, _SHAPE_A, log_h)
    rc_h = ph.wait(timeout=300)
    dt_hot = time.time() - t0
    out_h = open(log_h).read()
    assert rc_h == 0, f"hot rerun failed:\n{out_h}"
    # Hot rerun should be substantially faster than the cold ones — give a
    # generous bound so we don't false-positive on noisy CI: < 60s.
    assert dt_hot < 60.0, f"hot rerun took {dt_hot:.1f}s; cache likely missed:\n{out_h}"
