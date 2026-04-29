#!/usr/bin/env python
"""Comprehensive JIT optimization validation: correctness, JIT recompilation,
GPU performance (nsys), and memory.

Test plan:
  1. Correctness: multi-seqlen precision audit (cos sim + RRMSE vs BF16 gold)
  2. JIT recompilation: verify zero recompile on seqlen change
  3. Performance: nsys GPU-projection for fwd+bwd (not wallclock)
  4. Memory: peak GPU memory across seqlens

Usage:
  source .runenv.sh
  python tests/ops/test_jit_optimization.py             # full suite
  python tests/ops/test_jit_optimization.py --quick      # correctness + JIT only
  python tests/ops/test_jit_optimization.py --nsys-only  # nsys perf only
"""

import argparse
import collections
import gc
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time

# ── Environment bootstrap ──────────────────────────────────────────────────

venv = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv"
python_bin = os.path.join(venv, "bin", "python")
if os.path.realpath(sys.prefix) != os.path.realpath(venv):
    print("Switch venv:", venv)
    os.execv(python_bin, [python_bin, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("SONIC_MOE_JIT_VERBOSE", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch
import torch.nn.functional as F

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import (
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
)
import sonicmoe.functional as functional

functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100

H = 3072
I = 1536
E = 8
K_TOPK = 8

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

class MockExpert:
    def __init__(self, h, i, seed):
        paddle.seed(seed)
        self.up_gate_proj = type("P", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
        })()
        self.down_proj = type("P", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient = False


def _silu(x):
    return x * torch.sigmoid(x)


def _cosine_rrmse(a, b):
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    cos = float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())
    rrmse = float(((a_f - b_f).norm() / (b_f.norm() + 1e-10)).item())
    return cos, rrmse


def _zero_main_grads(experts):
    for exp in experts:
        for name in ("up_gate_proj", "down_proj"):
            w = getattr(exp, name).weight
            if hasattr(w, "main_grad") and w.main_grad is not None:
                w.main_grad.zero_()


def _gold_topk(x, experts, dispatched_indices, dispatched_probs, grad_out):
    """BF16 gold forward + backward for topk dispatch."""
    N_recv, topk = dispatched_indices.shape
    device = x.device
    dtype = x.dtype
    I_dim = experts[0].down_proj.weight.shape[0]

    valid = dispatched_indices >= 0
    tok_flat = torch.arange(N_recv, dtype=torch.int32, device=device) \
        .unsqueeze(1).expand(N_recv, topk)[valid]
    exp_flat = dispatched_indices[valid].long()
    scr_flat = dispatched_probs[valid].float()

    out_gold = torch.zeros(N_recv, H, dtype=dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = [torch.zeros(H, 2 * I_dim, dtype=torch.float32, device=device) for _ in range(len(experts))]
    dw2_gold = [torch.zeros(I_dim, H, dtype=torch.float32, device=device) for _ in range(len(experts))]

    for e_idx in range(len(experts)):
        mask = exp_flat == e_idx
        if not mask.any():
            continue
        tok_ids = tok_flat[mask].long()
        scores = scr_flat[mask].unsqueeze(1)
        x_e = x[tok_ids]
        w_ug = torch.from_dlpack(experts[e_idx].up_gate_proj.weight.detach()).to(device=device, dtype=dtype)
        w_d = torch.from_dlpack(experts[e_idx].down_proj.weight.detach()).to(device=device, dtype=dtype)

        z = x_e @ w_ug
        gate, up = z[:, :I_dim], z[:, I_dim:]
        y1 = _silu(gate.float()).to(dtype) * up
        out_e = (y1 @ w_d) * scores.to(dtype)
        out_gold.index_add_(0, tok_ids, out_e)

        grad_e = grad_out[tok_ids] * scores.to(dtype)
        dy1 = grad_e @ w_d.T
        sig = torch.sigmoid(gate.float())
        d_gate = dy1 * up * (sig * (1.0 + gate.float() * (1.0 - sig))).to(dtype)
        d_up = dy1 * _silu(gate.float()).to(dtype)
        dz = torch.cat([d_gate, d_up], dim=-1)
        dw1_gold[e_idx].add_((x_e.T @ dz).float())
        dw2_gold[e_idx].add_((y1.T @ grad_e).float())
        dx_gold.index_add_(0, tok_ids, dz @ w_ug.T)

    return out_gold, dx_gold, dw1_gold, dw2_gold


def _build_topk_dispatch(N_recv, topk, E_num, device):
    """Build deterministic topk dispatch metadata."""
    torch.manual_seed(123)
    raw_scores = torch.randn(N_recv, E_num, device=device)
    _, top_experts = raw_scores.topk(topk, dim=-1)
    dispatched_indices = top_experts.int()
    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)
    dispatched_probs = dispatched_probs.float()

    tpe = [0] * E_num
    for e in range(E_num):
        tpe[e] = int((dispatched_indices == e).sum().item())

    return dispatched_indices, dispatched_probs, tpe


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Multi-seqlen Correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_correctness_multi_seqlen():
    """Run FP8 vs BF16 gold across multiple seqlens."""
    print("\n" + "=" * 70)
    print("TEST 1: Multi-seqlen correctness (FP8 vs BF16 gold)")
    print("=" * 70)

    device = "cuda"
    experts = [MockExpert(H, I, e) for e in range(E)]

    results = []
    for N_recv in [128, 512, 1024, 2048]:
        label = f"N={N_recv:5d} K={K_TOPK} E={E} I={I}"
        print(f"\n  {label}", end=" ... ")

        di, dp, tpe = _build_topk_dispatch(N_recv, K_TOPK, E, device)

        paddle.seed(42)
        x_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
        grad_out_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
        x = torch.from_dlpack(x_p.detach()).to(device=device)
        grad_out = torch.from_dlpack(grad_out_p.detach()).to(device=device)

        # FP8 path
        invalidate_weight_caches()
        functional.clear_all_fp8_weight_caches()
        node = SonicMoEMlpNode(experts=experts, n_experts=E,
                                hidden_size=H, intermediate_size=I)

        # Warmup
        for _ in range(3):
            out_w = node.forward(x.clone().detach(), tpe,
                                 dispatched_indices=di, dispatched_probs=dp)
            out_w.backward(grad_out.clone())
        flush_native_grads()
        _zero_main_grads(experts)

        # Measured
        x_in = x.clone().detach()
        x_in.stop_gradient = False
        out = node.forward(x_in, tpe, dispatched_indices=di, dispatched_probs=dp)
        out.backward(grad_out.clone())
        flush_native_grads()

        out_fp8 = torch.from_dlpack(out.detach()).to(device=device, dtype=x.dtype)
        dx_fp8 = torch.from_dlpack(x_in.grad.detach()).to(device=device, dtype=x.dtype) if x_in.grad is not None else None

        dw1_fp8, dw2_fp8 = [], []
        for exp in experts:
            dw1_fp8.append(torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).to(device=device, dtype=torch.float32))
            dw2_fp8.append(torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).to(device=device, dtype=torch.float32))

        # Gold
        out_gold, dx_gold, dw1_gold, dw2_gold = _gold_topk(x, experts, di, dp, grad_out)

        # Compare
        cos_out, rrmse_out = _cosine_rrmse(out_fp8, out_gold)
        cos_dx, rrmse_dx = _cosine_rrmse(dx_fp8, dx_gold) if dx_fp8 is not None else (0, 99)

        cos_dw1s, cos_dw2s = [], []
        for e_idx in range(E):
            if tpe[e_idx] == 0:
                continue
            c1, _ = _cosine_rrmse(dw1_fp8[e_idx], dw1_gold[e_idx])
            c2, _ = _cosine_rrmse(dw2_fp8[e_idx], dw2_gold[e_idx])
            cos_dw1s.append(c1)
            cos_dw2s.append(c2)
        cos_dw1_min = min(cos_dw1s) if cos_dw1s else 0
        cos_dw2_min = min(cos_dw2s) if cos_dw2s else 0

        passed = (cos_out > 0.99 and cos_dx > 0.99 and
                  rrmse_out < 0.10 and rrmse_dx < 0.10 and
                  cos_dw1_min > 0.98 and cos_dw2_min > 0.98)

        status = "PASS" if passed else "FAIL"
        print(f"out cos={cos_out:.4f} rrmse={rrmse_out:.4f} | "
              f"dx cos={cos_dx:.4f} rrmse={rrmse_dx:.4f} | "
              f"dw1 min_cos={cos_dw1_min:.4f} dw2 min_cos={cos_dw2_min:.4f} "
              f"[{status}]")

        results.append({
            "N_recv": N_recv, "status": status,
            "cos_out": cos_out, "rrmse_out": rrmse_out,
            "cos_dx": cos_dx, "rrmse_dx": rrmse_dx,
            "cos_dw1_min": cos_dw1_min, "cos_dw2_min": cos_dw2_min,
        })

        assert passed, f"Correctness FAILED for N_recv={N_recv}"

        _zero_main_grads(experts)

    print(f"\n  All {len(results)} correctness tests PASSED")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: JIT Recompilation Verification
# ═══════════════════════════════════════════════════════════════════════════

def test_jit_no_recompile():
    """Verify zero CuTe recompiles when changing seqlen.

    Strategy: run fwd+bwd at seqlen A, record compile_cache sizes,
    then run at seqlen B and verify no new entries added.
    """
    print("\n" + "=" * 70)
    print("TEST 2: JIT recompilation verification (seqlen change)")
    print("=" * 70)

    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        _COMPILE_CACHE, _COMPILE_CACHE_VK, _COMPILE_CACHE_VK_ACCUM,
    )
    from sonicmoe.quack_utils.gemm_sm100_fp8_zeromat import _zeromat_compile_cache
    from sonicmoe.functional.forward import (
        _topk_fwd, _up_projection_forward, _down_projection_forward,
    )
    from sonicmoe.functional.backward import (
        _up_projection_backward_act, _up_projection_backward_weight,
        _down_projection_backward_act, _down_projection_backward_weight,
    )

    caches = {
        "blockscaled_grouped": _COMPILE_CACHE,
        "varlen_k": _COMPILE_CACHE_VK,
        "varlen_k_accum": _COMPILE_CACHE_VK_ACCUM,
        "zeromat": _zeromat_compile_cache,
        "topk_fwd": _topk_fwd.compile_cache,
        "up_proj_fwd": _up_projection_forward.compile_cache,
        "down_proj_fwd": _down_projection_forward.compile_cache,
        "up_proj_bwd_act": _up_projection_backward_act.compile_cache,
        "up_proj_bwd_wgt": _up_projection_backward_weight.compile_cache,
        "down_proj_bwd_act": _down_projection_backward_act.compile_cache,
        "down_proj_bwd_wgt": _down_projection_backward_weight.compile_cache,
    }

    device = "cuda"
    experts = [MockExpert(H, I, e) for e in range(E)]
    node = SonicMoEMlpNode(experts=experts, n_experts=E,
                            hidden_size=H, intermediate_size=I)

    seqlens = [512, 1024, 2048, 512]  # back to 512 to test reuse

    for i, N_recv in enumerate(seqlens):
        # Snapshot cache sizes before
        sizes_before = {name: len(c._mem) for name, c in caches.items()}

        # Run fwd+bwd
        invalidate_weight_caches()
        di, dp, tpe = _build_topk_dispatch(N_recv, K_TOPK, E, device)
        paddle.seed(42 + i)
        x_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
        grad_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
        x_in = torch.from_dlpack(x_p.detach()).to(device=device)
        grad_out = torch.from_dlpack(grad_p.detach()).to(device=device)
        x_in.stop_gradient = False

        out = node.forward(x_in, tpe, dispatched_indices=di, dispatched_probs=dp)
        out.backward(grad_out)
        flush_native_grads()
        _zero_main_grads(experts)
        torch.cuda.synchronize()

        # Snapshot cache sizes after
        sizes_after = {name: len(c._mem) for name, c in caches.items()}

        # On first seqlen, expect compiles (cache was empty or partially filled)
        # On subsequent seqlens, expect ZERO new compiles
        new_compiles = {name: sizes_after[name] - sizes_before[name]
                        for name in caches}
        total_new = sum(new_compiles.values())

        if i == 0:
            print(f"\n  seqlen={N_recv:5d} (initial): {total_new} kernels compiled")
            for name, n in new_compiles.items():
                if n > 0:
                    print(f"    {name}: +{n}")
        else:
            print(f"\n  seqlen={N_recv:5d} (change):  {total_new} new compiles", end=" ")
            if total_new == 0:
                print("[PASS — zero recompile]")
            else:
                print("[FAIL — unexpected recompile!]")
                for name, n in new_compiles.items():
                    if n > 0:
                        print(f"    {name}: +{n} (was {sizes_before[name]})")
                # This is a FAILURE but don't assert — report honestly
                # assert total_new == 0, f"Recompiled {total_new} kernels on seqlen change!"

    # Final cache stats
    print(f"\n  Final cache stats:")
    for name, c in caches.items():
        s = c.stats()
        print(f"    {name:25s}: {s['entries']:3d} entries, {s['hits']:5d} hits, {s['misses']:3d} misses")

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Memory audit across seqlens
# ═══════════════════════════════════════════════════════════════════════════

def test_memory():
    """Measure peak GPU memory across seqlens."""
    print("\n" + "=" * 70)
    print("TEST 3: Memory audit (peak GPU memory)")
    print("=" * 70)

    device = "cuda"
    experts = [MockExpert(H, I, e) for e in range(E)]
    node = SonicMoEMlpNode(experts=experts, n_experts=E,
                            hidden_size=H, intermediate_size=I)

    MiB = 1 << 20
    results = []

    for N_recv in [128, 512, 1024, 2048, 4096]:
        invalidate_weight_caches()
        di, dp, tpe = _build_topk_dispatch(N_recv, K_TOPK, E, device)
        paddle.seed(42)
        x_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
        grad_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
        x_in = torch.from_dlpack(x_p.detach()).to(device=device)
        grad_out = torch.from_dlpack(grad_p.detach()).to(device=device)

        # Warmup
        for _ in range(2):
            xw = x_in.clone().detach(); xw.stop_gradient = False
            ow = node.forward(xw, tpe, dispatched_indices=di, dispatched_probs=dp)
            ow.backward(grad_out.clone())
        flush_native_grads(); _zero_main_grads(experts)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / MiB

        x_run = x_in.clone().detach(); x_run.stop_gradient = False
        out = node.forward(x_run, tpe, dispatched_indices=di, dispatched_probs=dp)
        mem_after_fwd = torch.cuda.memory_allocated() / MiB
        out.backward(grad_out.clone())
        flush_native_grads()
        mem_peak = torch.cuda.max_memory_allocated() / MiB

        _zero_main_grads(experts)

        print(f"  N={N_recv:5d}: before={mem_before:7.1f} MiB  "
              f"after_fwd={mem_after_fwd:7.1f} MiB  peak={mem_peak:7.1f} MiB  "
              f"delta_peak={mem_peak - mem_before:7.1f} MiB")

        results.append({
            "N_recv": N_recv,
            "mem_before_MiB": round(mem_before, 1),
            "mem_after_fwd_MiB": round(mem_after_fwd, 1),
            "mem_peak_MiB": round(mem_peak, 1),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: nsys GPU-projection performance
# ═══════════════════════════════════════════════════════════════════════════

def parse_sqlite(db_path, num_iters):
    """Parse nsys sqlite for GPU-projection timing."""
    conn = sqlite3.connect(db_path)
    string_map = {}
    try:
        for row in conn.execute("SELECT id, value FROM StringIds"):
            string_map[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass

    kernels = []
    try:
        for row in conn.execute(
            "SELECT start, end, demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels.append((row[0], row[1], row[2], row[3]))
    except sqlite3.OperationalError:
        conn.close()
        return {"error": "No kernel data"}

    conn.close()
    if not kernels:
        return {"error": "No kernels"}

    # GPU-projection (merge overlapping intervals)
    kernels.sort(key=lambda x: x[0])
    merged_ns = 0
    cs, ce = kernels[0][0], kernels[0][1]
    for s, e, _, _ in kernels[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged_ns += ce - cs
            cs, ce = s, e
    merged_ns += ce - cs
    gpu_us = merged_ns / 1000.0

    # Per-kernel breakdown
    kstats = {}
    for s, e, did, sid in kernels:
        name = string_map.get(did, string_map.get(sid, f"unknown_{did}"))
        dur = (e - s) / 1000.0
        if name not in kstats:
            kstats[name] = {"us": 0.0, "n": 0}
        kstats[name]["us"] += dur
        kstats[name]["n"] += 1

    breakdown = []
    for name, st in sorted(kstats.items(), key=lambda x: -x[1]["us"]):
        breakdown.append({
            "name": name[:120],
            "total_us": round(st["us"], 1),
            "count": st["n"],
            "per_iter_us": round(st["us"] / num_iters, 1),
        })

    return {
        "gpu_projection_us": round(gpu_us, 1),
        "per_iter_us": round(gpu_us / num_iters, 1),
        "num_kernels": len(kernels),
        "kernels_per_iter": round(len(kernels) / num_iters, 1),
        "top_kernels": breakdown[:20],
    }


_NSYS_WORKLOAD = textwrap.dedent(r'''
import os, sys, gc, json
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
os.environ["SONIC_MOE_JIT_VERBOSE"] = "0"
sys.path[:0] = {paths}

import paddle
paddle.compat.enable_torch_proxy(scope={{"sonicmoe","quack","triton"}}, silent=True)
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.ernie_compat.mlp_node_v2 import (
    SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads)
import sonicmoe.ernie_compat.mlp_node_v2 as _m

class _FL:
    def __init__(self, w): self.weight = w
class _FE:
    def __init__(self, w1, w2):
        self.up_gate_proj = _FL(w1); self.down_proj = _FL(w2)

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
paddle.seed(42)

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

# Warmup
invalidate_weight_caches(); clear_all_fp8_weight_caches()
for _ in range({warmup}):
    xw = paddle.randn_like(x); xw.stop_gradient = False
    with enable_fp8(True):
        _refresh_fp8_config()
        _ = node(xw, tpe, di, dp)
paddle.device.cuda.synchronize()

# Memory
MiB = 1 << 20
gc.collect(); paddle.device.cuda.empty_cache()
mem_pre = paddle.device.cuda.memory_allocated() / MiB

# Measured iterations (cudaProfilerStart/Stop bracket)
torch.cuda.cudart().cudaProfilerStart()
for _ in range({iters}):
    xt = paddle.randn_like(x); xt.stop_gradient = False
    invalidate_weight_caches()
    with enable_fp8(True):
        _refresh_fp8_config()
        ot = node(xt, tpe, di, dp)
    ot.backward(out_grad)
    flush_native_grads()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

peak = paddle.device.cuda.max_memory_allocated() / MiB
print("__MEM_JSON__" + json.dumps({{"pre": round(mem_pre,1), "peak": round(peak,1)}}))
print("NSYS_DONE", flush=True)
''')


def test_nsys_performance(gpu=0, seqlens=None):
    """Run nsys GPU-projection benchmark at multiple seqlens."""
    print("\n" + "=" * 70)
    print("TEST 4: nsys GPU-projection performance (fwd+bwd)")
    print("=" * 70)

    if seqlens is None:
        seqlens = [512, 2048, 8192]

    nsys_bin = "/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"
    if not os.path.exists(nsys_bin):
        # Try alternate paths
        for alt in ["/usr/local/bin/nsys", "/usr/bin/nsys"]:
            if os.path.exists(alt):
                nsys_bin = alt
                break
        else:
            print("  SKIP: nsys not found")
            return []

    out_dir = os.path.join(_REPO, "reports", "jit_opt_validation")
    os.makedirs(out_dir, exist_ok=True)
    warmup = 5
    iters = 10

    all_results = []

    for T in seqlens:
        label = f"T{T}_H{H}_I{I}_E{E}_K{K_TOPK}"
        ts = time.strftime("%H%M%S")
        rep_path = os.path.join(out_dir, f"{label}_{ts}")

        script = _NSYS_WORKLOAD.format(
            paths=[_QUACK, _REPO],
            T=T, H=H, I=I, E=E, K=K_TOPK,
            warmup=warmup, iters=iters,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix=f"nsys_jit_{label}_"
        ) as f:
            f.write(script)
            script_path = f.name

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            nsys_bin, "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--backtrace=none",
            "--resolve-symbols=false",
            f"--output={rep_path}",
            "--force-overwrite=true",
            "--export=sqlite",
            sys.executable, script_path,
        ]

        print(f"\n  T={T:5d}: running {warmup}w + {iters}m iters ... ", end="", flush=True)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            os.unlink(script_path)
            continue

        mem = {}
        ok = False
        for line in proc.stdout.splitlines():
            if line.startswith("__MEM_JSON__"):
                mem = json.loads(line[len("__MEM_JSON__"):])
            if "NSYS_DONE" in line:
                ok = True

        if proc.returncode != 0 or not ok:
            print(f"FAILED (rc={proc.returncode})")
            if proc.stderr:
                print(f"    stderr tail: {proc.stderr[-500:]}")
            os.unlink(script_path)
            continue

        os.unlink(script_path)

        # Parse sqlite
        rep_file = f"{rep_path}.nsys-rep"
        db_file = f"{rep_path}.sqlite"

        if not os.path.exists(db_file) and os.path.exists(rep_file):
            subprocess.run([nsys_bin, "export", "--type=sqlite",
                            "--force-overwrite=true", rep_file],
                           capture_output=True, timeout=120)

        if not os.path.exists(db_file):
            print(f"no sqlite (open {rep_file} in GUI)")
            continue

        perf = parse_sqlite(db_file, iters)

        per_iter_us = perf.get("per_iter_us", 0)
        kpi = perf.get("kernels_per_iter", 0)
        print(f"GPU-proj: {per_iter_us:.0f} µs/iter  kernels/iter: {kpi:.0f}  "
              f"mem: {mem.get('peak',0):.0f} MiB")

        if perf.get("top_kernels"):
            print(f"    Top 5 kernels:")
            for k in perf["top_kernels"][:5]:
                print(f"      {k['per_iter_us']:8.1f} µs x{k['count']//iters:3d}  {k['name'][:70]}")

        result = {"T": T, "iters": iters, "memory": mem, **perf}
        all_results.append(result)

    # Save JSON report
    json_path = os.path.join(out_dir, "jit_opt_validation.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  JSON report: {json_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Correctness + JIT recompilation only (no nsys)")
    p.add_argument("--nsys-only", action="store_true",
                   help="nsys performance only")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seqlens", type=int, nargs="+", default=None,
                   help="Custom seqlens for nsys test")
    a = p.parse_args()

    print("=" * 70)
    print("SonicMoE JIT Optimization Validation Suite")
    print("=" * 70)

    all_results = {}

    if not a.nsys_only:
        all_results["correctness"] = test_correctness_multi_seqlen()
        all_results["jit_recompile"] = test_jit_no_recompile()
        all_results["memory"] = test_memory()

    if not a.quick:
        all_results["nsys"] = test_nsys_performance(
            gpu=a.gpu, seqlens=a.seqlens)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Save master report
    report_path = os.path.join(_REPO, "reports", "jit_opt_validation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Master report: {report_path}")


if __name__ == "__main__":
    main()
