#!/usr/bin/env python3
"""Cold-start nsys benchmark: clears all caches, simulates first-time execution.

Produces a clean BENCH-region nsys profile for GPU-projection breakdown.
"""
import os, sys, math
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Nuke ALL on-disk JIT / Triton caches before import ──
import shutil
for d in [
    os.path.expanduser("~/.triton/cache"),
    os.path.join(_REPO, "__pycache__"),
]:
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"[COLD] Cleared {d}")

import paddle
paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.ernie_compat.mlp_node_v2 import (
    SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads,
)
import sonicmoe.ernie_compat.mlp_node_v2 as _m
import sonicmoe.functional as functional

# ── Nuke all in-memory caches ──
functional._ALIGNMENT_ASSUMED = True
functional.clear_all_fp8_weight_caches()
clear_all_fp8_weight_caches()
invalidate_weight_caches()
# Clear compile caches if available
try:
    from sonicmoe.cache_manager import InstrumentedCompileCache
    for attr_name in dir(sys.modules.get('sonicmoe.quack_utils.blockscaled_fp8_gemm', None) or type('', (), {})):
        obj = getattr(sys.modules.get('sonicmoe.quack_utils.blockscaled_fp8_gemm'), attr_name, None)
        if isinstance(obj, InstrumentedCompileCache):
            obj.clear()
except Exception:
    pass
print("[COLD] All in-memory compile/weight/grad caches cleared")

# ── Config ──
T, H, I, E, K = 8192, 3072, 1536, 8, 8
WARMUP = 8
ITERS = 12

# ── Build experts ──
class _FL:
    def __init__(self, w): self.weight = w
class _FE:
    def __init__(self, w1, w2):
        self.up_gate_proj = _FL(w1); self.down_proj = _FL(w2)

paddle.seed(42)
experts = []
for _ in range(E):
    w1 = paddle.randn([H, 2*I], dtype="bfloat16") * 0.001
    w2 = paddle.randn([I, H], dtype="bfloat16") * 0.001
    w1.stop_gradient = False; w2.stop_gradient = False
    experts.append(_FE(w1, w2))

node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H,
                        intermediate_size=I, activation_type=ActivationType.SWIGLU)

# ── Dispatch data ──
x = paddle.randn([T, H], dtype="bfloat16")
out_grad = paddle.randn([T, H], dtype="bfloat16")

di = paddle.zeros([T, K], dtype="int32")
dp = paddle.full([T, K], 1.0/K, dtype="float32")
for i in range(T):
    di[i] = paddle.randperm(E)[:K].cast("int32")
tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()

# ── Warmup (outside profiler, includes JIT compilation) ──
invalidate_weight_caches(); clear_all_fp8_weight_caches()
print(f"Warmup {WARMUP} fwd+bwd iters (includes JIT compilation)...")
for _ in range(WARMUP):
    xw = paddle.randn_like(x); xw.stop_gradient = False
    with enable_fp8(True):
        _refresh_fp8_config()
        ow = node(xw, tpe, di, dp)
    ow.backward(out_grad)
flush_native_grads()
paddle.device.cuda.synchronize()
print("Warmup done.")

# ── CUDA events for immediate feedback ──
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()

# ── Profiled region (steady-state: NO cache invalidation inside loop) ──
print(f"BENCH {ITERS} fwd+bwd iters...")
import time as _t
torch.cuda.nvtx.range_push("BENCH")
start_ev.record()
per_iter = []
import time as _t2
t_loop0 = _t2.perf_counter()
for it in range(ITERS):
    torch.cuda.nvtx.range_push(f"ITER{it}")
    xt = paddle.randn_like(x); xt.stop_gradient = False

    with enable_fp8(True):
        _refresh_fp8_config()
        ot = node(xt, tpe, di, dp)
    ot.backward(out_grad)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("FLUSH")
flush_native_grads()
torch.cuda.nvtx.range_pop()
end_ev.record()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

cuda_us = start_ev.elapsed_time(end_ev) / ITERS * 1000
peak = paddle.device.cuda.max_memory_allocated() / (1 << 20)
print(f"\nCUDA events: {cuda_us:.1f} µs/iter")
print(f"Peak memory: {peak:.0f} MiB")
print("Done.")
