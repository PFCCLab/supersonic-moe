#!/usr/bin/env python
"""Minimal MlpNode worker for nsys profiling. Run directly under nsys:

  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
       --stats=true -o /tmp/mlpnode_e2e -f true \
       python tests/ops/mlpnode_nsys_worker.py

Then parse:
  nsys stats /tmp/mlpnode_e2e.nsys-rep --report cuda_gpu_kern_sum
"""
import os, sys
os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.ernie_compat.mlp_node_v2 import (
    SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads)
import sonicmoe.ernie_compat.mlp_node_v2 as _m

# ── Config ───────────────────────────────────────────────────────────────
T, H, I, E, K = 8192, 3072, 1536, 8, 8   # Ernie shape
WARMUP = 5
ITERS = 12

class _FL:
    def __init__(self, w): self.weight = w
class _FE:
    def __init__(self, w1, w2):
        self.up_gate_proj = _FL(w1); self.down_proj = _FL(w2)

paddle.seed(42)

# ── Build ────────────────────────────────────────────────────────────────
experts = []
for _ in range(E):
    w1 = paddle.randn([H, 2*I], dtype="bfloat16") * 0.001
    w2 = paddle.randn([I, H], dtype="bfloat16") * 0.001
    w1.stop_gradient = False; w2.stop_gradient = False
    experts.append(_FE(w1, w2))

node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H,
                        intermediate_size=I, activation_type=ActivationType.SWIGLU)

x = paddle.randn([T, H], dtype="bfloat16")
out_grad = paddle.randn([T, H], dtype="bfloat16")

di = paddle.zeros([T, K], dtype="int32")
dp = paddle.full([T, K], 1.0/K, dtype="float32")
for i in range(T):
    di[i] = paddle.randperm(E)[:K].cast("int32")
tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()

# ── Warmup (forward-only, outside profiler) ──────────────────────────────
invalidate_weight_caches(); clear_all_fp8_weight_caches()
print(f"Warmup {WARMUP} forward-only iters...", flush=True)
for _ in range(WARMUP):
    xw = paddle.randn_like(x); xw.stop_gradient = False
    with enable_fp8(True):
        _refresh_fp8_config()
        _ = node(xw, tpe, di, dp)
paddle.device.cuda.synchronize()
print("Warmup done.", flush=True)

# ── Memory snapshot ──────────────────────────────────────────────────────
MiB = 1 << 20
mem_pre = paddle.device.cuda.memory_allocated() / MiB
print(f"Memory pre-profile: {mem_pre:.0f} MiB", flush=True)

# ── Profiled region (NO synchronize, NO cudaProfiler API) ────────────────
print(f"Profiling {ITERS} fwd+bwd iters...", flush=True)

for it in range(ITERS):
    xt = paddle.randn_like(x); xt.stop_gradient = False
    invalidate_weight_caches()
    with enable_fp8(True):
        _refresh_fp8_config()
        ot = node(xt, tpe, di, dp)
    ot.backward(out_grad)
    flush_native_grads()

paddle.device.cuda.synchronize()

peak = paddle.device.cuda.max_memory_allocated() / MiB
print(f"Memory peak: {peak:.0f} MiB", flush=True)
print(f"Done. {ITERS} iters captured for nsys GPU-projection.", flush=True)
