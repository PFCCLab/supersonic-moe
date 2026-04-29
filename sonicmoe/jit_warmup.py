"""JIT warmup: pre-compile all CuTe + Triton kernels before training begins.

Usage::

    from sonicmoe.jit_warmup import warmup_jit

    # Dynamic-dim mode (recommended): single warmup covers ALL seqlens.
    warmup_jit(E=8, H=3072, I=1536, device="cuda")

Environment variables:
    SONIC_MOE_CACHE_DIR : str
        Override cache root path.
    QUACK_COMPILE_WORKERS : int
        Number of parallel workers for CuTe compilation (default: CPU count).
"""

from __future__ import annotations

import logging
import math
import os
import time

import torch

_log = logging.getLogger("sonicmoe.jit")


def warmup_jit(
    E: int,
    H: int,
    I: int,
    device: torch.device | str | None = None,
    *,
    fp8: bool = True,
    total_K_list: list[int] | None = None,
    max_workers: int = 0,
    cache_dir: str | None = None,
    skip_if_warm: bool = True,
    force: bool = False,
) -> bool:
    """Pre-compile all CuTe + Triton kernels with dummy data.

    Returns ``True`` if compilation actually ran, ``False`` if it was skipped
    because the disk-cache sentinel matched.

    After the compile_key fix (dynamic dims removed from cache key),
    a single warmup shape covers ALL future seqlens.

    Parameters
    ----------
    E, H, I : int
        Model dimensions (num_experts, hidden_size, intermediate_size).
    device : str or torch.device, optional
        CUDA device to warmup on. Defaults to the current rank's device
        (``torch.device("cuda", torch.cuda.current_device())``); explicit
        rank-aware addressing avoids ``DeviceContextPool::Get`` failures in
        multi-rank / multi-machine setups under paddle-torch-compat.
    fp8 : bool
        Whether to warmup FP8 codepath (default True).
    total_K_list : list[int], optional
        Explicit total_K values.  Default: ``[E * 128]``.
    max_workers : int
        Parallel CuTe compile workers (0 = auto).
    cache_dir : str, optional
        Override SONIC_MOE_CACHE_DIR.
    skip_if_warm : bool
        If True (default), check the cache sentinel and skip compilation
        when it indicates a previous warmup for this (E,H,I,fp8,kernel-sig,
        git-hash) tuple has populated the disk caches. Ignored if ``force``.
    force : bool
        Run warmup even if the sentinel matches.
    """
    if device is None:
        # Resolve current device through paddle to avoid hitting the
        # torch-proxy lazy-import issue when sonicmoe is imported under
        # ``enable_torch_proxy(scope={"sonicmoe", ...})``.
        import paddle as _paddle
        try:
            place = _paddle.framework._current_expected_place()
            dev_id = getattr(place, "gpu_device_id", None)
            if callable(dev_id):
                dev_id = dev_id()
            if dev_id is None:
                dev_id = 0
        except Exception:
            dev_id = 0
        device = f"cuda:{int(dev_id)}"
    from sonicmoe.cache_manager import setup_cache, is_warm, mark_warm

    if cache_dir:
        setup_cache(cache_dir)
    else:
        setup_cache()

    if skip_if_warm and not force and is_warm(E, H, I, fp8):
        _log.info(
            f"[Warmup] sentinel hit (E={E}, H={H}, I={I}, fp8={fp8}); "
            f"skipping compile. Set force=True or `clear_warmup_sentinel()` "
            f"to override."
        )
        return False

    if max_workers == 0:
        max_workers = min(os.cpu_count() or 8, 16)
    os.environ["QUACK_COMPILE_WORKERS"] = str(max_workers)

    if total_K_list is None:
        total_K_list = [E * 128]

    _log.info(
        f"[Warmup] E={E}, H={H}, I={I}, fp8={fp8}, "
        f"shapes={len(total_K_list)}, workers={max_workers}"
    )
    t0 = time.perf_counter()

    for i, total_K in enumerate(total_K_list):
        t1 = time.perf_counter()
        _warmup_single(E, H, I, total_K, device, fp8)
        dt = time.perf_counter() - t1
        _log.info(
            f"[Warmup] shape {i + 1}/{len(total_K_list)} "
            f"(total_K={total_K}) done in {dt:.1f}s"
        )

    total = time.perf_counter() - t0
    _log.info(f"[Warmup] All done in {total:.1f}s")
    mark_warm(E, H, I, fp8)
    return True


def warmup_jit_parallel(
    E: int,
    H: int,
    I: int,
    *,
    fp8: bool = True,
    total_K_list: list[int] | None = None,
    workers: int = 2,
    cache_dir: str | None = None,
) -> float:
    """Multi-process cold warmup. Spawns ``workers`` subprocesses, each
    warming a partition of ``total_K_list``. All share the same disk cache
    (FileLock-protected). Returns total wall time.

    Use only for cold-start; warm reuse should always go through
    ``warmup_jit`` (in-process, sentinel-aware).
    """
    import subprocess
    import sys as _sys

    if total_K_list is None:
        total_K_list = [E * 128]
    if workers < 1:
        workers = 1
    if cache_dir:
        os.environ["SONIC_MOE_CACHE_DIR"] = cache_dir

    buckets: list[list[int]] = [[] for _ in range(workers)]
    for i, k in enumerate(total_K_list):
        buckets[i % workers].append(k)
    buckets = [b for b in buckets if b]

    snippet_template = """\
import os
os.environ.setdefault('SONIC_MOE_JIT_VERBOSE', '1')
import paddle
paddle.compat.enable_torch_proxy(scope={'sonicmoe','quack','triton'}, silent=True)
from sonicmoe.cache_manager import setup_cache; setup_cache()
from sonicmoe.jit_warmup import warmup_jit
warmup_jit(E=%d, H=%d, I=%d, fp8=%s, total_K_list=%s, force=True, skip_if_warm=False)
"""
    env_base = os.environ.copy()
    env_base.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    # Round-robin GPU assignment so concurrent workers don't all serialise
    # on one device. Respects an existing CUDA_VISIBLE_DEVICES whitelist.
    visible = env_base.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_pool = [d for d in visible.split(",") if d.strip()]
    else:
        try:
            import subprocess as _sp
            n = int(_sp.check_output(
                ["nvidia-smi", "--query-gpu=count",
                 "--format=csv,noheader,nounits"],
                stderr=_sp.DEVNULL).decode().splitlines()[0])
            gpu_pool = [str(i) for i in range(n)]
        except Exception:
            gpu_pool = ["0"]
    procs = []
    t0 = time.perf_counter()
    for w_idx, ks in enumerate(buckets):
        code = snippet_template % (E, H, I, repr(bool(fp8)), repr(list(ks)))
        env_w = env_base.copy()
        env_w["CUDA_VISIBLE_DEVICES"] = gpu_pool[w_idx % len(gpu_pool)]
        p = subprocess.Popen(
            [_sys.executable, "-c", code],
            env=env_w,
        )
        procs.append(p)
    rc_any = 0
    for p in procs:
        rc = p.wait()
        rc_any = rc_any or rc
    dt = time.perf_counter() - t0
    if rc_any != 0:
        raise RuntimeError(
            f"warmup_jit_parallel: at least one worker failed (rc={rc_any})"
        )
    _log.info(
        f"[Warmup-Parallel] {len(buckets)} workers × {len(total_K_list)} "
        f"shapes done in {dt:.1f}s"
    )
    return dt


def _warmup_single(E: int, H: int, I: int, total_K: int, device, fp8: bool):
    """One fwd+bwd through SonicMoEMlpNode to compile ALL production kernels.

    Runs under Paddle torch-proxy (same as production), compiles:
      - CuTe CUTLASS fwd GEMM (GatedFP8 zeromat)
      - CuTe CUTLASS bwd act GEMM (DGatedFP8)
      - CuTe CUTLASS bwd wgrad GEMM (varlen_k + varlen_k_accumulate)
      - Triton FP8 quantize kernels (row, colwise, dual, gather, iso32)
      - Triton SwiGLU fwd/bwd kernels
      - Triton metadata kernel (_build_score_src_idx_kernel)
      - CUDA topk metadata kernel
    """
    import paddle
    paddle.compat.enable_torch_proxy(
        scope={"sonicmoe", "quack", "triton"}, silent=True,
    )

    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.ernie_compat.mlp_node_v2 import (
        SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads,
    )
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True

    topk = min(E, 8)
    N_recv = total_K // topk  # approximate: each token routed to topk experts
    N_recv = max(N_recv, 128)  # at least 128 tokens for tile coverage

    # Build mock experts
    class _FL:
        def __init__(self, w): self.weight = w
    class _FE:
        def __init__(self, w1, w2):
            self.up_gate_proj = _FL(w1); self.down_proj = _FL(w2)

    paddle.seed(42)
    experts = []
    for _ in range(E):
        w1 = paddle.randn([H, 2 * I], dtype="bfloat16") * 0.001
        w2 = paddle.randn([I, H], dtype="bfloat16") * 0.001
        w1.stop_gradient = False; w2.stop_gradient = False
        experts.append(_FE(w1, w2))

    invalidate_weight_caches()
    clear_all_fp8_weight_caches()

    node = SonicMoEMlpNode(
        experts=experts, n_experts=E, hidden_size=H,
        intermediate_size=I, activation_type=ActivationType.SWIGLU,
    )

    # Build dispatched_indices and dispatched_probs
    di = paddle.zeros([N_recv, topk], dtype="int32")
    dp = paddle.full([N_recv, topk], 1.0 / topk, dtype="float32")
    for i in range(N_recv):
        di[i] = paddle.randperm(E)[:topk].cast("int32")
    tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()

    x = paddle.randn([N_recv, H], dtype="bfloat16")
    grad_out = paddle.randn([N_recv, H], dtype="bfloat16")

    # Forward + backward (compiles all kernels on the production path)
    with enable_fp8(fp8):
        _refresh_fp8_config()
        xt = paddle.randn_like(x); xt.stop_gradient = False
        out = node(xt, tpe, di, dp)
    out.backward(grad_out)
    flush_native_grads()

    # Second iter to compile the wgrad accumulate variant
    # (first iter had fresh _NATIVE_W*_GRAD, second triggers beta=1.0 path)
    with enable_fp8(fp8):
        _refresh_fp8_config()
        xt2 = paddle.randn_like(x); xt2.stop_gradient = False
        out2 = node(xt2, tpe, di, dp)
    out2.backward(grad_out)
    flush_native_grads()

    paddle.device.cuda.synchronize()
