#!/usr/bin/env python3
"""Cross-Framework MoE Benchmark — 4-way: Paddle BF16 / Paddle FP8 / SonicMoE BF16 / SonicMoE FP8.

Compares precision, peak memory, and GPU-projection performance of four
MoE expert computation paths on **identical data** (numpy dump/load) using
the same Target GPU GPU.

Four paths:
  1. Paddle BF16  — BF16 matmul per expert (no FP8, ERNIE-core convention)
  2. Paddle FP8   — Fp8FusedMlpFunc.apply per expert (ERNIE-style handwritten bwd)
  3. SonicMoE BF16 — _UpProjection + _DownProjection (QuACK BF16 path)
  4. SonicMoE FP8  — same + enable_fp8(True) (blockscaled CUTLASS)

Subprocess isolation: Paddle (system python 3.10) and PyTorch (xfer python 3.13)
cannot coexist.  FP8 mode uses process-global flags.

Performance metric: **nsys GPU-projection ONLY** (sweep-line interval merge over
CUPTI_ACTIVITY_KIND_KERNEL).  No wall-clock timing.

Usage:
    # Full benchmark (all shapes, nsys, ~30 min, single GPU)
    CUDA_VISIBLE_DEVICES=0 python tools/cross_framework_bench.py

    # Quick smoke test
    CUDA_VISIBLE_DEVICES=0 python tools/cross_framework_bench.py --shapes smoke

    # Precision-only (skip nsys, ~5 min)
    CUDA_VISIBLE_DEVICES=0 python tools/cross_framework_bench.py --skip-nsys

    # Specific GPU
    python tools/cross_framework_bench.py --gpu 2
"""

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

REFERENCE_CORE_ROOT = os.environ.get("SONIC_MOE_REFERENCE_CORE_ROOT", "")
REFERENCE_EXPERT_NODE = os.environ.get("SONIC_MOE_REFERENCE_EXPERT_NODE", "")

SYSTEM_PYTHON = os.environ.get("SONIC_MOE_SYSTEM_PYTHON", sys.executable)
XFER_PYTHON = os.environ.get("SONIC_MOE_XFER_PYTHON", sys.executable)

NSYS_BIN = shutil.which("nsys") or "/usr/local/bin/nsys"

# ── Shape grid (from introspect.py) ───────────────────────────────────────────
# Default reference shape + selected grid shapes from introspect.py:
#   GRID_T = [8192, 16384, 32768], GRID_E = [8, 32, 128],
#   GRID_I = [1536, 2048, 3072], GRID_H = 3072, GRID_K = 8
# Avoid very large shapes to prevent reference OOM (single GPU, no model parallel)
SHAPES = {
    "T8192_H3072_I1536_E8_K8":    {"S": 8192,  "H": 3072, "I": 1536, "E": 8,   "K": 8},
    "T16384_H3072_I1536_E8_K8":   {"S": 16384, "H": 3072, "I": 1536, "E": 8,   "K": 8},
    "T8192_H3072_I1536_E32_K8":   {"S": 8192,  "H": 3072, "I": 1536, "E": 32,  "K": 8},
    "T8192_H3072_I2048_E8_K8":    {"S": 8192,  "H": 3072, "I": 2048, "E": 8,   "K": 8},
    "T8192_H3072_I3072_E8_K8":    {"S": 8192,  "H": 3072, "I": 3072, "E": 8,   "K": 8},
}

# nsys parameters
NSYS_WARMUP = 5
NSYS_ITERS = 20
NSYS_REPEATS = 3        # repeat nsys profiling, report median

# Precision
SEEDS = [42, 123, 777]  # repeated measurements for precision

# Target GPU hardware constants (for theoretical analysis)
TARGET_GPU_FP8_TFLOPS = 4500       # E4M3 tensor-core peak (TFLOPS)
TARGET_GPU_BF16_TFLOPS = 2250      # BF16 tensor-core peak (TFLOPS)
TARGET_GPU_HBM_BW_GBPS = 8000      # HBM3e bandwidth (GB/s)

# ── Path registry ─────────────────────────────────────────────────────────────
PATH_NAMES = ["paddle_bf16", "paddle_fp8", "sonic_bf16", "sonic_fp8"]
PATH_LABELS = {
    "paddle_bf16": "Paddle BF16",
    "paddle_fp8":  "Paddle FP8",
    "sonic_bf16":  "SonicMoE BF16",
    "sonic_fp8":   "SonicMoE FP8",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation (pure numpy, no framework deps)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(shape: dict, data_dir: str, seed: int = 42) -> None:
    """Generate shared numpy test data + float64 gold reference (fwd only)."""
    S, H, I, E, K = shape["S"], shape["H"], shape["I"], shape["E"], shape["K"]
    rng = np.random.RandomState(seed)
    os.makedirs(data_dir, exist_ok=True)

    x = (rng.randn(S, H) * 0.02).astype(np.float32)
    np.save(os.path.join(data_dir, "x.npy"), x)

    for e in range(E):
        w1_e = (rng.randn(H, 2 * I) * 0.02).astype(np.float32)
        w2_e = (rng.randn(I, H) * 0.02).astype(np.float32)
        np.save(os.path.join(data_dir, f"w1_e{e}.npy"), w1_e)
        np.save(os.path.join(data_dir, f"w2_e{e}.npy"), w2_e)

    topk_indices = np.zeros((S, K), dtype=np.int32)
    for s in range(S):
        for k in range(K):
            topk_indices[s, k] = (s * K + k) % E
    np.save(os.path.join(data_dir, "topk_indices.npy"), topk_indices)

    logits = rng.randn(S, E).astype(np.float64)
    gathered = np.array([logits[s, topk_indices[s]] for s in range(S)])
    exp_g = np.exp(gathered - gathered.max(axis=1, keepdims=True))
    topk_scores = (exp_g / exp_g.sum(axis=1, keepdims=True)).astype(np.float32)
    np.save(os.path.join(data_dir, "topk_scores.npy"), topk_scores)

    gold_reference, gold_sonic = _compute_gold_fp64(x, data_dir, S, H, I, E, K, topk_indices, topk_scores)
    np.save(os.path.join(data_dir, "gold_reference.npy"), gold_reference)
    np.save(os.path.join(data_dir, "gold_sonic.npy"), gold_sonic)

    # Grad output for backward (shared across all paths)
    grad_output = (rng.randn(S, H) * 0.01).astype(np.float32)
    np.save(os.path.join(data_dir, "grad_output.npy"), grad_output)


def _compute_gold_fp64(x, data_dir, S, H, I, E, K, topk_indices, topk_scores):
    """Gold MoE fwd in float64.

    Produces TWO gold references:
      - gold_reference: probs applied BEFORE down-proj (split-half convention)
      - gold_sonic: probs applied AFTER down-proj during scatter (SonicMoE convention)
    """
    x64 = x.astype(np.float64)
    TK = S * K
    flat_experts = topk_indices.reshape(-1)
    flat_tokens = np.repeat(np.arange(S), K)
    sorted_order = np.argsort(flat_experts, kind="stable")
    counts = np.bincount(flat_experts, minlength=E).astype(np.int32)
    cu = np.zeros(E + 1, dtype=np.int32)
    cu[1:] = np.cumsum(counts)
    xg = x64[flat_tokens[sorted_order]]
    s_rev = np.empty(TK, dtype=np.int32)
    for i, so in enumerate(sorted_order):
        s_rev[so] = i

    # Up-proj + SwiGLU
    y1 = np.zeros((TK, I), dtype=np.float64)
    for e in range(E):
        s, end = cu[e], cu[e + 1]
        if s >= end:
            continue
        w1 = np.load(os.path.join(data_dir, f"w1_e{e}.npy")).astype(np.float64)
        z = xg[s:end] @ w1
        gate, up = z[:, :I], z[:, I:]
        sig = 1.0 / (1.0 + np.exp(-gate))
        y1[s:end] = gate * sig * up

    # Compute probs in sorted order for split-half convention
    sc64 = topk_scores.astype(np.float64)
    probs_sorted = np.zeros(TK, dtype=np.float64)
    for i, so in enumerate(sorted_order):
        t, k = divmod(so, K)
        probs_sorted[i] = sc64[t, k]

    # ── reference gold: probs BEFORE down-proj ──
    y1_reference = y1.copy()
    for e in range(E):
        s, end = cu[e], cu[e + 1]
        y1_reference[s:end] *= probs_sorted[s:end, None]

    y2_reference = np.zeros((TK, H), dtype=np.float64)
    for e in range(E):
        s, end = cu[e], cu[e + 1]
        if s >= end:
            continue
        w2 = np.load(os.path.join(data_dir, f"w2_e{e}.npy")).astype(np.float64)
        y2_reference[s:end] = y1_reference[s:end] @ w2

    out_reference = np.zeros((S, H), dtype=np.float64)
    for t in range(S):
        for k in range(K):
            out_reference[t] += y2_reference[s_rev[t * K + k]]  # no probs here — already applied

    # ── SonicMoE gold: probs AFTER down-proj (at scatter) ──
    y2_sonic = np.zeros((TK, H), dtype=np.float64)
    for e in range(E):
        s, end = cu[e], cu[e + 1]
        if s >= end:
            continue
        w2 = np.load(os.path.join(data_dir, f"w2_e{e}.npy")).astype(np.float64)
        y2_sonic[s:end] = y1[s:end] @ w2

    out_sonic = np.zeros((S, H), dtype=np.float64)
    for t in range(S):
        for k in range(K):
            out_sonic[t] += y2_sonic[s_rev[t * K + k]] * sc64[t, k]

    return out_reference, out_sonic


# ═══════════════════════════════════════════════════════════════════════════════
# Precision Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def precision_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    d = a64 - b64
    return {
        "rrmse": float(np.sqrt(np.mean(d**2) / max(np.mean(b64**2), 1e-30))),
        "cosine": float(np.dot(a64, b64) / max(np.linalg.norm(a64) * np.linalg.norm(b64), 1e-30)),
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Theoretical Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_theory(shape: dict) -> dict:
    """Compute theoretical FLOPs, bytes, and roofline bounds for one MoE fwd."""
    S, H, I, E, K = shape["S"], shape["H"], shape["I"], shape["E"], shape["K"]
    TK = S * K  # total token-expert pairs

    # GEMM FLOPs (2*M*N*K per matmul)
    # Up-proj: (TK, H) × (H, 2I) → 2·TK·H·2I
    flops_up = 2 * TK * H * (2 * I)
    # Down-proj: (TK, I) × (I, H) → 2·TK·I·H
    flops_down = 2 * TK * I * H
    flops_total = flops_up + flops_down

    # Memory bytes (BF16 = 2B, FP8 = 1B)
    act_bytes_bf16 = TK * H * 2 + TK * 2 * I * 2 + TK * I * 2 + TK * H * 2
    weight_bytes_bf16 = E * (H * 2 * I * 2 + I * H * 2)
    total_bytes_bf16 = act_bytes_bf16 + weight_bytes_bf16
    total_bytes_fp8 = act_bytes_bf16 // 2 + weight_bytes_bf16 // 2  # rough

    # Roofline bounds
    bf16_compute_us = flops_total / (TARGET_GPU_BF16_TFLOPS * 1e6)
    fp8_compute_us = flops_total / (TARGET_GPU_FP8_TFLOPS * 1e6)
    bf16_mem_us = total_bytes_bf16 / (TARGET_GPU_HBM_BW_GBPS * 1e3)
    fp8_mem_us = total_bytes_fp8 / (TARGET_GPU_HBM_BW_GBPS * 1e3)

    return {
        "TK": TK,
        "flops_up": flops_up,
        "flops_down": flops_down,
        "flops_total": flops_total,
        "bf16_compute_bound_us": round(bf16_compute_us, 1),
        "fp8_compute_bound_us": round(fp8_compute_us, 1),
        "bf16_mem_bound_us": round(bf16_mem_us, 1),
        "fp8_mem_bound_us": round(fp8_mem_us, 1),
        "bf16_roofline_us": round(max(bf16_compute_us, bf16_mem_us), 1),
        "fp8_roofline_us": round(max(fp8_compute_us, fp8_mem_us), 1),
        "arithmetic_intensity_bf16": round(flops_total / total_bytes_bf16, 1),
        "arithmetic_intensity_fp8": round(flops_total / total_bytes_fp8, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess Scripts — 4 Paths
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Paddle paths: unified template using ExpertsGroupGemmContiguousNode ──────
# Both BF16 and FP8 use the configured reference expert node, which provides
# production-grade forward AND backward:
#   BF16: fp8=None → BF16 matmul per expert (fwd) + BF16 weight grad (bwd)
#   FP8:  fp8="e4m3" → FP8 GEMM (fwd) + FP8 wgrad + FP8 dact GEMM (bwd)
# Unzip/Zip: F.moe_permute / F.moe_unpermute (production fused kernels)
_PADDLE_SCRIPT_TEMPLATE = textwrap.dedent(r'''
import gc, json, os, sys
import numpy as np
REFERENCE_CORE_ROOT = "{reference_core_root}"
if REFERENCE_CORE_ROOT and REFERENCE_CORE_ROOT not in sys.path:
    sys.path.insert(0, REFERENCE_CORE_ROOT)
REFERENCE_EXPERT_NODE = "{reference_expert_node}"
if not REFERENCE_EXPERT_NODE:
    raise RuntimeError("Set SONIC_MOE_REFERENCE_EXPERT_NODE=module.path:ClassName")
os.environ["FLAGS_cudnn_deterministic"] = "True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import paddle, paddle.nn.functional as F

data_dir, S, H, I, E, K = "{data_dir}", {S}, {H}, {I}, {E}, {K}
FP8_ALIGN = 128
mode = "{mode}"
warmup, iters = {warmup}, {iters}
USE_FP8 = {use_fp8}  # True for FP8 path, False for BF16 path
OUTPUT_PREFIX = "{output_prefix}"

_mod_name, _cls_name = REFERENCE_EXPERT_NODE.split(":", 1)
_mod = __import__(_mod_name, fromlist=[_cls_name])
ExpertsGroupGemmContiguousNode = getattr(_mod, _cls_name)

# ── Mock custom_map for ExpertsGroupGemmContiguousNode ──
class _W:
    def __init__(self, t): self.weight = t
class _Expert:
    def __init__(self, w1, w2):
        self.up_gate_proj = _W(w1); self.down_proj = _W(w2)
class _Cfg:
    fp8_fused_ops_configs = {{"spaq": True, "stack_quant": True, "swiglu_probs_bwd": True, "transpose_split_quant": True}}
    # BF16 path: moe_grouped_gemm=False for the ERNIE BF16 implementation.
    # FP8 path: moe_grouped_gemm controlled by flag (default True = production training)
    moe_grouped_gemm = {use_grouped_gemm} if USE_FP8 else False
class _Map:
    def __init__(self, experts): self.experts = experts; self.config = _Cfg()

# ── Load data ──
x_all = paddle.to_tensor(np.load(os.path.join(data_dir, "x.npy")), dtype="bfloat16")
w1_list = [paddle.to_tensor(np.load(os.path.join(data_dir, f"w1_e{{e}}.npy")), dtype="bfloat16") for e in range(E)]
w2_list = [paddle.to_tensor(np.load(os.path.join(data_dir, f"w2_e{{e}}.npy")), dtype="bfloat16") for e in range(E)]
for w in w1_list + w2_list:
    w.stop_gradient = False
    w.main_grad = None
topk_indices = paddle.to_tensor(np.load(os.path.join(data_dir, "topk_indices.npy")), dtype="int32")
topk_probs = paddle.to_tensor(np.load(os.path.join(data_dir, "topk_scores.npy")), dtype="float32")

# ── Build node ──
experts = [_Expert(w1_list[e], w2_list[e]) for e in range(E)]
custom_map = _Map(experts)
node = ExpertsGroupGemmContiguousNode(
    custom_map,
    fp8="e4m3" if USE_FP8 else None,
    use_ue8m0=True if USE_FP8 else False,
)

# ── Precompute tokens_per_expert and ali_cnt (routing is deterministic) ──
flat_eid = topk_indices.reshape([-1])
tokens_per_expert = [int((flat_eid == e).sum()) for e in range(E)]
ali_cnt = [((c + FP8_ALIGN - 1) // FP8_ALIGN) * FP8_ALIGN if c > 0 else 0 for c in tokens_per_expert]

def do_permute():
    """F.moe_permute — called every iteration (matches production)."""
    with paddle.amp.auto_cast(False):
        ut, rowmap, up, _ = F.moe_permute(
            x_all, None, topk_indices, topk_probs,
            num_experts=E, tokens_per_expert=tokens_per_expert,
            padding_alignment=FP8_ALIGN, do_gather=True,
        )
    return ut, rowmap, up

def do_unpermute(o3, rowmap, up):
    """F.moe_unpermute — called every iteration (matches production)."""
    with paddle.amp.auto_cast(False):
        output, _ = F.moe_unpermute(o3, rowmap, topk_indices, up, total_zipped_tokens=S, num_experts=E)
    return output

def run_fwd():
    """Forward: permute → ExpertsGroupGemmContiguousNode.forward → unpermute."""
    ut, rowmap, up = do_permute()
    o3 = node.forward(ut, up, ali_cnt, tokens_per_expert)
    output = do_unpermute(o3, rowmap, up)
    return o3, output, up

def run_bwd(o3, grad_expert_out, up):
    """Backward: ExpertsGroupGemmContiguousNode.backward (production reference bwd)."""
    for w in w1_list + w2_list:
        w.main_grad = None
    dx, probs_grad = node.backward(grad_expert_out, up)
    return dx

def run_fwd_bwd():
    """Full forward + backward iteration (permute → fwd → unpermute → bwd)."""
    o3, output, up = run_fwd()
    grad_expert_out = paddle.ones_like(o3)
    dx = run_bwd(o3, grad_expert_out, up)
    return output, dx

if mode == "precision":
    # Warmup
    for _ in range(2):
        _ = run_fwd()
        node.reset_statue()
    paddle.device.synchronize(); gc.collect(); paddle.device.cuda.empty_cache()
    paddle.device.synchronize()

    # ── Staged memory measurement (every MiB accounted) ──
    paddle.device.cuda.reset_max_memory_allocated()
    mem_baseline = paddle.device.cuda.memory_allocated() / (1024**2)  # weights + routing tensors

    # Forward (includes saved tensors for backward)
    o3, output, up_measured = run_fwd()
    paddle.device.synchronize()
    mem_post_fwd = paddle.device.cuda.memory_allocated() / (1024**2)
    peak_fwd = paddle.device.cuda.max_memory_allocated() / (1024**2)

    # Backward (weight grads accumulated into main_grad)
    M = o3.shape[0]
    for w in w1_list + w2_list:
        w.main_grad = None
    grad_expert_out = paddle.ones([M, H], dtype="bfloat16")
    dx = node.backward(grad_expert_out, up_measured)
    paddle.device.synchronize()
    mem_post_bwd = paddle.device.cuda.memory_allocated() / (1024**2)
    peak_total = paddle.device.cuda.max_memory_allocated() / (1024**2)

    # Save output
    np.save(os.path.join(data_dir, f"{{OUTPUT_PREFIX}}_output.npy"), output.cast("float32").numpy())
    # Save gradients
    for e in range(E):
        if w1_list[e].main_grad is not None:
            np.save(os.path.join(data_dir, f"{{OUTPUT_PREFIX}}_dw1_e{{e}}.npy"), w1_list[e].main_grad.numpy())
        if w2_list[e].main_grad is not None:
            np.save(os.path.join(data_dir, f"{{OUTPUT_PREFIX}}_dw2_e{{e}}.npy"), w2_list[e].main_grad.numpy())
    print(json.dumps({{
        "status": "ok",
        "mem_baseline_mib": round(mem_baseline, 1),
        "mem_post_fwd_mib": round(mem_post_fwd, 1),
        "mem_post_bwd_mib": round(mem_post_bwd, 1),
        "peak_fwd_mib": round(peak_fwd, 1),
        "peak_total_mib": round(peak_total, 1),
    }}))

elif mode == "nsys":
    # Full fwd+bwd pipeline under nsys capture
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so")
    for _ in range(warmup):
        run_fwd_bwd()
        node.reset_statue()
    paddle.device.synchronize(); gc.collect()
    libcudart.cudaProfilerStart()
    for _ in range(iters):
        run_fwd_bwd()
        node.reset_statue()
    paddle.device.synchronize()
    libcudart.cudaProfilerStop()
    print("NSYS_DONE", flush=True)
''')

PADDLE_BF16_SCRIPT = _PADDLE_SCRIPT_TEMPLATE.replace("{use_fp8}", "False").replace("{output_prefix}", "paddle_bf16").replace("{use_grouped_gemm}", "True")
PADDLE_FP8_SCRIPT = _PADDLE_SCRIPT_TEMPLATE.replace("{use_fp8}", "True").replace("{output_prefix}", "paddle_fp8").replace("{use_grouped_gemm}", "True")

# ─── Shared SonicMoE script ──────────────────────────────────────────────────
# Precision: uses _UpProjection + _DownProjection with pre-determined routing
#            (must match gold reference routing exactly)
# nsys: uses full MoE class (matches introspect.py frontier path)
_SONIC_PREAMBLE = textwrap.dedent(r'''
import gc, json, os, sys
import numpy as np
os.environ["USE_QUACK_GEMM"] = "1"
{fp8_env}
sys.path.insert(0, "{project_root}")

import torch, torch.nn.functional as F

data_dir, S, H, I, E, K = "{data_dir}", {S}, {H}, {I}, {E}, {K}
mode = "{mode}"
warmup, iters = {warmup}, {iters}
device = torch.device("cuda:0")
use_fp8 = {use_fp8}

def split_to_interleaved(w):
    h = w.shape[0] // 2
    o = torch.empty_like(w); o[0::2] = w[:h]; o[1::2] = w[h:]
    return o

if mode == "precision":
    # ── Precision: use _UpProjection + _DownProjection with fixed routing ──
    from sonicmoe.functional import _UpProjection, _DownProjection, clear_all_fp8_weight_caches
    from sonicmoe.functional import _refresh_fp8_config
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    import sonicmoe.functional as functional

    clear_all_fp8_weight_caches()

    x = torch.from_numpy(np.load(os.path.join(data_dir, "x.npy"))).to(device=device, dtype=torch.bfloat16)
    topk_indices = torch.from_numpy(np.load(os.path.join(data_dir, "topk_indices.npy"))).to(device=device)
    topk_scores = torch.from_numpy(np.load(os.path.join(data_dir, "topk_scores.npy"))).to(device=device)

    w1l, w2l = [], []
    for e in range(E):
        w1_e = torch.from_numpy(np.load(os.path.join(data_dir, f"w1_e{{e}}.npy")))
        w2_e = torch.from_numpy(np.load(os.path.join(data_dir, f"w2_e{{e}}.npy")))
        w1l.append(split_to_interleaved(w1_e.T)); w2l.append(w2_e.T)
    w1_param = torch.stack(w1l).to(device=device, dtype=torch.bfloat16).contiguous()
    w2_param = torch.stack(w2l).to(device=device, dtype=torch.bfloat16).contiguous()
    w1_param.requires_grad_(True)
    w2_param.requires_grad_(True)
    del w1l, w2l

    TK = S * K
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    def run_sonic_fwd():
        w1f = w1_param.permute(1, 2, 0)
        w2f = w2_param.permute(1, 2, 0)
        if use_fp8:
            with enable_fp8(True):
                _refresh_fp8_config()
                try:
                    y1, z = _UpProjection.apply(
                        x, w1f, None, expert_frequency_offset, TK, K, 0,
                        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                        None, False, ActivationType.SWIGLU, False, False,
                    )
                    o = _DownProjection.apply(
                        y1, z, w2f, None, topk_scores, topk_indices,
                        expert_frequency_offset, S, K, 0, x_gather_idx,
                        s_scatter_idx, s_reverse_scatter_idx, None, False,
                        ActivationType.SWIGLU, None,
                    )
                finally:
                    clear_all_fp8_weight_caches()
        else:
            with enable_fp8(False):
                y1, z = _UpProjection.apply(
                    x, w1f, None, expert_frequency_offset, TK, K, 0,
                    x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                    None, False, ActivationType.SWIGLU, False, False,
                )
                o = _DownProjection.apply(
                    y1, z, w2f, None, topk_scores, topk_indices,
                    expert_frequency_offset, S, K, 0, x_gather_idx,
                    s_scatter_idx, s_reverse_scatter_idx, None, False,
                    ActivationType.SWIGLU, None,
                )
        return o

    for _ in range(2):
        _ = run_sonic_fwd()
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

    # ── Staged memory measurement ──
    torch.cuda.reset_peak_memory_stats()
    mem_baseline = torch.cuda.memory_allocated() / (1024**2)  # weights + routing tensors

    # Forward (autograd saves tensors for backward)
    x_input = x.detach().clone().requires_grad_(True)
    _orig_x = x
    x = x_input
    o = run_sonic_fwd()
    torch.cuda.synchronize()
    mem_post_fwd = torch.cuda.memory_allocated() / (1024**2)
    peak_fwd = torch.cuda.max_memory_allocated() / (1024**2)

    # Backward
    grad_out = torch.from_numpy(np.load(os.path.join(data_dir, "grad_output.npy"))).to(device=device, dtype=torch.bfloat16)
    o.backward(grad_out)
    torch.cuda.synchronize()
    mem_post_bwd = torch.cuda.memory_allocated() / (1024**2)
    peak_total = torch.cuda.max_memory_allocated() / (1024**2)

    x = _orig_x
    clear_all_fp8_weight_caches()
    np.save(os.path.join(data_dir, "{output_name}"), o.detach().float().cpu().numpy())
    # Save gradients
    pref = "{output_name}".replace("_output.npy", "")
    if x_input.grad is not None:
        np.save(os.path.join(data_dir, f"{{pref}}_dx.npy"), x_input.grad.float().cpu().numpy())
    if w1_param.grad is not None:
        np.save(os.path.join(data_dir, f"{{pref}}_dw1.npy"), w1_param.grad.float().cpu().numpy())
    if w2_param.grad is not None:
        np.save(os.path.join(data_dir, f"{{pref}}_dw2.npy"), w2_param.grad.float().cpu().numpy())
    print(json.dumps({{
        "status": "ok",
        "mem_baseline_mib": round(mem_baseline, 1),
        "mem_post_fwd_mib": round(mem_post_fwd, 1),
        "mem_post_bwd_mib": round(mem_post_bwd, 1),
        "peak_fwd_mib": round(peak_fwd, 1),
        "peak_total_mib": round(peak_total, 1),
    }}))

elif mode == "nsys":
    # ── nsys: matches introspect.py frontier path exactly ──
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    import sonicmoe.functional as functional

    functional.clear_all_fp8_weight_caches()
    functional._ALIGNMENT_ASSUMED = True

    moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
               intermediate_size=I, activation_function=ActivationType.SWIGLU,
               add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)

    if use_fp8:
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()

    x = torch.from_numpy(np.load(os.path.join(data_dir, "x.npy"))).to(device=device, dtype=torch.bfloat16)

    use_token_rounding = (E > 8)
    w1_p = moe.c_fc.weight.permute(1, 2, 0)
    w2_p = moe.c_proj.weight.permute(1, 2, 0)

    if use_token_rounding:
        # E>8: token rounding + moe_general_routing_inputs (same as introspect.py)
        from sonicmoe.functional import count_cumsum, moe_general_routing_inputs

        def run_iter():
            Mtile = 128
            xw = x.detach().clone().requires_grad_(True)
            with torch.no_grad():
                rl = F.linear(xw, moe.router.weight)
                sc = F.softmax(rl, dim=-1, dtype=torch.float32).to(torch.bfloat16)
                tv, ti = sc.topk(K, dim=-1)
                tv /= tv.sum(dim=-1, keepdim=True)
                sc.scatter_(-1, ti, tv)
                cb = sc.clone() - 1; cb.scatter_(1, ti, tv)
                si = cb.argsort(dim=0, descending=True).int()
                ef = count_cumsum(ti.view(-1), E, do_cumsum=True)[0]
                efr = (torch.ceil(ef / Mtile) * Mtile).int()
                mk = torch.arange(S, device=device, dtype=torch.int32)[:, None].expand(-1, E) < efr[None, :]
                tok = si[mk]; exp = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(S, -1)[mk]
                od = tok.argsort().int(); tok = tok[od]; exp = exp[od]
                rsc = sc[tok, exp].contiguous()
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        out, _ = moe_general_routing_inputs(
                            xw, rsc, tok, exp, w1_p, None, w2_p, None,
                            E, moe.stream_id, ActivationType.SWIGLU, False)
                else:
                    with enable_fp8(False):
                        out, _ = moe_general_routing_inputs(
                            xw, rsc, tok, exp, w1_p, None, w2_p, None,
                            E, moe.stream_id, ActivationType.SWIGLU, False)
            return xw, out
    else:
        # E<=8: direct moe(xw)
        def run_iter():
            xw = x.detach().clone().requires_grad_(True)
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        o, aux = moe(xw, use_fp8=True)
                else:
                    with enable_fp8(False):
                        o, aux = moe(xw)
            return xw, o

    for _ in range(warmup):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True)
        del xw, o
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True)
        del xw, o
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("NSYS_DONE", flush=True)
''')


def _build_sonic_script(fp8: bool) -> str:
    """Build SonicMoE runner script (BF16 or FP8)."""
    if fp8:
        fp8_env = 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"'
        output_name = "sonic_fp8_output.npy"
        use_fp8 = "True"
    else:
        fp8_env = 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
        output_name = "sonic_bf16_output.npy"
        use_fp8 = "False"

    return _SONIC_PREAMBLE.replace(
        "{fp8_env}", fp8_env
    ).replace(
        "{output_name}", output_name
    ).replace(
        "{use_fp8}", use_fp8
    )


SONIC_BF16_SCRIPT = _build_sonic_script(fp8=False)
SONIC_FP8_SCRIPT = _build_sonic_script(fp8=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess Execution
# ═══════════════════════════════════════════════════════════════════════════════

def _make_env(gpu: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run_runner(label: str, python_bin: str, script_tpl: str,
               fmt: dict, gpu: int, timeout: int = 600) -> dict:
    """Run precision+memory subprocess, parse JSON from stdout."""
    script = script_tpl.format(**fmt)
    env = _make_env(gpu)
    t0 = time.time()
    r = subprocess.run([python_bin, "-c", script],
                       capture_output=True, text=True, env=env, timeout=timeout)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"    [{label}] FAILED ({elapsed:.1f}s): {r.stderr[-800:]}")
        return {"status": "error", "error": r.stderr[-800:]}
    for line in reversed(r.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                d = json.loads(line.strip())
                d["elapsed_s"] = round(elapsed, 1)
                return d
            except json.JSONDecodeError:
                continue
    return {"status": "error", "error": "no JSON", "stdout": r.stdout[-500:]}


def run_nsys(label: str, python_bin: str, script_tpl: str,
             fmt: dict, prefix: str, gpu: int, timeout: int = 600) -> dict:
    """Run under nsys, parse sqlite for GPU-projection."""
    script = script_tpl.format(**fmt)
    sf = prefix + ".py"
    with open(sf, "w") as f:
        f.write(script)
    env = _make_env(gpu)
    cmd = [NSYS_BIN, "profile", "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop", f"--output={prefix}",
           "--export=sqlite", "--force-overwrite=true", python_bin, sf]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"    [{label}] nsys FAILED ({elapsed:.1f}s)")
        return {"error": "nsys failed"}
    db = prefix + ".sqlite"
    if not os.path.exists(db):
        return {"error": "no sqlite"}
    return _nsys_parse_sqlite(db, fmt.get("iters", NSYS_ITERS))


def _nsys_parse_sqlite(db_path: str, num_iters: int) -> dict:
    """Sweep-line interval merge for GPU-projection from nsys sqlite."""
    conn = sqlite3.connect(db_path)
    smap = {}
    try:
        for row in conn.execute("SELECT id, value FROM StringIds"):
            smap[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass
    kernels = []
    try:
        for row in conn.execute(
            "SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels.append((row[0], row[1], row[2], row[3]))
    except sqlite3.OperationalError:
        conn.close()
        return {"error": "no kernel data"}
    conn.close()
    if not kernels:
        return {"error": "0 kernels"}
    kernels.sort(key=lambda x: x[0])
    ns = 0
    cs, ce = kernels[0][0], kernels[0][1]
    for s, e, _, _ in kernels[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            ns += ce - cs; cs, ce = s, e
    ns += ce - cs
    gpu_us = ns / 1000.0
    # top kernels
    kstats = {}
    for s, e, did, sid in kernels:
        nm = smap.get(did, smap.get(sid, f"?{did}"))
        d = (e - s) / 1000.0
        if nm not in kstats:
            kstats[nm] = {"us": 0.0, "cnt": 0}
        kstats[nm]["us"] += d; kstats[nm]["cnt"] += 1
    bd = [{"name": n[:120], "us": round(s["us"], 1), "cnt": s["cnt"],
           "per_iter_us": round(s["us"] / num_iters, 1)}
          for n, s in sorted(kstats.items(), key=lambda x: -x[1]["us"])]
    return {
        "gpu_projection_us": round(gpu_us, 1),
        "per_iter_us": round(gpu_us / num_iters, 1),
        "num_iters": num_iters,
        "num_kernels": len(kernels),
        "top_kernels": bd[:20],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(results: dict, report_path: str) -> None:
    L = []
    L.append("# Cross-Framework MoE Benchmark Report (4-Way)")
    L.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  GPU: NVIDIA Target GPU (SM100)  |  Seeds: {SEEDS}")
    L.append(f"nsys: {NSYS_WARMUP} warmup + {NSYS_ITERS} measured × {NSYS_REPEATS} repeats (median)")
    L.append("")

    # ── 1. Setup ──
    L.append("## 1. Experimental Setup\n")
    L.append("| Path | Framework | Python | Compute | API |")
    L.append("|------|-----------|--------|---------|-----|")
    L.append("| Paddle BF16 | PaddlePaddle | 3.10 | BF16 matmul | `ExpertsGroupGemmContiguousNode(fp8=None)` fwd+bwd |")
    L.append("| Paddle FP8 | PaddlePaddle | 3.10 | FP8 (kitchen+deep_gemm) | `ExpertsGroupGemmContiguousNode(fp8='e4m3')` fwd+bwd |")
    L.append("| SonicMoE BF16 | PyTorch | 3.13 | BF16 QuACK GEMM | `MoE(...)` full module (same as introspect.py) |")
    L.append("| SonicMoE FP8 | PyTorch | 3.13 | FP8 blockscaled CUTLASS | `MoE(..., use_fp8=True)` + `enable_fp8(True)` |")
    L.append("\n### Shapes\n")
    L.append("| Label | S | H | I | E | K | TK | FLOPs (G) |")
    L.append("|-------|:---:|:---:|:---:|:---:|:---:|:---:|--------:|")
    for sl in results:
        sh = results[sl]["shape"]
        th = results[sl].get("theory", {})
        tk = sh["S"] * sh["K"]
        flops_g = th.get("flops_total", 0) / 1e9
        L.append(f"| {sl} | {sh['S']} | {sh['H']} | {sh['I']} | {sh['E']} | {sh['K']} | {tk} | {flops_g:.1f} |")
    L.append("")

    # ── 2. Precision ──
    L.append("## 2. Precision Analysis\n")
    L.append("### 2.1 Forward Output `output (S, H)` vs Float64 Gold\n")
    L.append("**Tensor**: final MoE output after down-proj + `F.moe_unpermute` (Paddle) / router scatter (SonicMoE).")
    L.append("")
    L.append("**Two gold references** (prob-scaling location differs):")
    L.append("- **Gold-reference**: `output = (swiglu(x@w1) * prob) @ w2`, accumulate at scatter (prob BEFORE down-proj)")
    L.append("- **Gold-SonicMoE**: `output = swiglu(x@w1) @ w2 * prob` (prob AFTER down-proj at scatter)")
    L.append("")
    L.append("Note: intermediate tensors (preact, postact, expert_out) are NOT comparable cross-framework due to")
    L.append("different padding/rounding strategies (`F.moe_permute` vs `TC_topk_router_metadata_triton`).")
    L.append("Only `output (S, H)` is permutation-invariant and comparable.\n")
    L.append(f"Metrics averaged over {len(SEEDS)} seeds.\n")
    for sl in results:
        prec = results[sl].get("precision", {})
        if not prec:
            continue
        L.append(f"### {sl}\n")
        L.append("| Path | RRMSE | Cosine | Max |err| | Mean |err| | PASS? |")
        L.append("|------|------:|-------:|--------:|--------:|:-----:|")
        for p in PATH_NAMES:
            m = prec.get(p, {})
            if not m:
                L.append(f"| {PATH_LABELS[p]} | — | — | — | — | — |")
                continue
            rr = m.get("rrmse_mean", float("nan"))
            cs = m.get("cosine_mean", float("nan"))
            mx = m.get("max_abs_mean", float("nan"))
            mn = m.get("mean_abs_mean", float("nan"))
            is_fp8 = "fp8" in p
            thr_rr = 0.10 if is_fp8 else 0.01
            thr_cs = 0.99 if is_fp8 else 0.999
            ok = "PASS" if rr < thr_rr and cs > thr_cs else "FAIL"
            L.append(f"| {PATH_LABELS[p]} | {rr:.6f} | {cs:.6f} | {mx:.4e} | {mn:.4e} | {ok} |")
        # Pairwise
        pw = results[sl].get("pairwise_precision", {})
        if pw:
            L.append(f"\n**Pairwise (seed-averaged):**\n")
            L.append("| A vs B | RRMSE | Cosine |")
            L.append("|--------|------:|-------:|")
            for k, m in pw.items():
                L.append(f"| {k} | {m['rrmse_mean']:.6f} | {m['cosine_mean']:.6f} |")
        L.append("")

    # ── 3. Memory ──
    L.append("## 3. Peak Memory (MiB) — fwd + bwd\n")
    L.append("Staged measurement: baseline (weights+routing) → post-fwd (+ saved tensors) → peak (fwd+bwd max).\n")
    L.append("| Shape | Path | Baseline | Post-Fwd | Peak (fwd) | Peak (fwd+bwd) | Fwd Delta | Bwd Delta |")
    L.append("|-------|------|--------:|--------:|--------:|--------:|--------:|--------:|")
    for sl in results:
        mem = results[sl].get("memory", {})
        for p in PATH_NAMES:
            m = mem.get(p, {})
            if not isinstance(m, dict):
                continue
            L.append(f"| {sl} | {PATH_LABELS[p]} | {m.get('baseline','—')} | {m.get('post_fwd','—')} | {m.get('peak_fwd','—')} | {m.get('peak_total','—')} | {round(float(m.get('post_fwd',0))-float(m.get('baseline',0)),1) if isinstance(m.get('post_fwd'),float) else '—'} | {round(float(m.get('post_bwd',0))-float(m.get('post_fwd',0)),1) if isinstance(m.get('post_bwd'),float) else '—'} |")
    L.append("")

    # ── 4. Performance (nsys GPU-projection ONLY) ──
    L.append("## 4. Performance — nsys GPU-Projection\n")
    L.append("**Metric**: merged GPU kernel busy time (sweep-line interval merge over `CUPTI_ACTIVITY_KIND_KERNEL`).")
    L.append(f"Warmup: {NSYS_WARMUP} iters (discarded).  Measured: {NSYS_ITERS} iters.  Repeats: {NSYS_REPEATS} (report median).")
    L.append("All 4 paths: **forward + backward** (permute → expert fwd → unpermute → expert bwd).\n")
    has_perf = any(
        results[sl].get("nsys", {}).get(p, {}).get("per_iter_us_median")
        for sl in results for p in PATH_NAMES
    )
    if has_perf:
        L.append("| Shape | " + " | ".join(f"{PATH_LABELS[p]} (us)" for p in PATH_NAMES) + " | FP8/BF16 (Paddle) | FP8/BF16 (Sonic) |")
        L.append("|-------" + "|----------:" * len(PATH_NAMES) + "|----------:|----------:|")
        for sl in results:
            nsys = results[sl].get("nsys", {})
            vals = []
            for p in PATH_NAMES:
                v = nsys.get(p, {}).get("per_iter_us_median")
                vals.append(f"{v:.1f}" if v else "—")
            pbf = nsys.get("paddle_bf16", {}).get("per_iter_us_median")
            pfp = nsys.get("paddle_fp8", {}).get("per_iter_us_median")
            sbf = nsys.get("sonic_bf16", {}).get("per_iter_us_median")
            sfp = nsys.get("sonic_fp8", {}).get("per_iter_us_median")
            sp_p = f"{pbf/pfp:.2f}×" if pbf and pfp else "—"
            sp_s = f"{sbf/sfp:.2f}×" if sbf and sfp else "—"
            L.append(f"| {sl} | " + " | ".join(vals) + f" | {sp_p} | {sp_s} |")
        L.append("")
    else:
        L.append("*nsys profiling skipped or failed.*\n")

    # ── 5. Theoretical Analysis ──
    L.append("## 5. Theoretical Analysis\n")
    L.append("### Roofline Model (Target GPU)\n")
    L.append(f"- BF16 tensor-core peak: {TARGET_GPU_BF16_TFLOPS} TFLOPS")
    L.append(f"- FP8 tensor-core peak: {TARGET_GPU_FP8_TFLOPS} TFLOPS")
    L.append(f"- HBM3e bandwidth: {TARGET_GPU_HBM_BW_GBPS} GB/s\n")
    L.append("| Shape | FLOPs (G) | AI (BF16) | AI (FP8) | Compute Bound BF16 (us) | Compute Bound FP8 (us) | Mem Bound BF16 (us) |")
    L.append("|-------|--------:|------:|-----:|----------:|----------:|----------:|")
    for sl in results:
        th = results[sl].get("theory", {})
        L.append(f"| {sl} | {th.get('flops_total',0)/1e9:.1f} | {th.get('arithmetic_intensity_bf16',0)} | {th.get('arithmetic_intensity_fp8',0)} | {th.get('bf16_compute_bound_us',0)} | {th.get('fp8_compute_bound_us',0)} | {th.get('bf16_mem_bound_us',0)} |")
    L.append("")
    L.append("### Efficiency Analysis\n")
    if has_perf:
        L.append("| Shape | Path | Measured (us) | Roofline (us) | Efficiency |")
        L.append("|-------|------|-------------:|--------------:|-----------:|")
        for sl in results:
            th = results[sl].get("theory", {})
            nsys = results[sl].get("nsys", {})
            for p in PATH_NAMES:
                v = nsys.get(p, {}).get("per_iter_us_median")
                if v is None:
                    continue
                roof = th.get("fp8_roofline_us") if "fp8" in p else th.get("bf16_roofline_us")
                eff = f"{roof/v*100:.1f}%" if roof and v > 0 else "—"
                L.append(f"| {sl} | {PATH_LABELS[p]} | {v:.1f} | {roof} | {eff} |")
        L.append("")

    L.append("### FP8 Quantization Error Model\n")
    L.append("E4M3 FP8: 3 mantissa bits → per-element relative error ε ≈ 2^{-4} ≈ 6.25%.")
    L.append("MoE pipeline has 2 quantized matmuls (up-proj, down-proj) plus SwiGLU.")
    L.append("Expected RRMSE ≈ √(ε_up² + ε_act² + ε_down²) ≈ 0.05–0.08.")
    L.append("Both reference and SonicMoE use blockscaled E4M3 with E8M0 power-of-2 scales,")
    L.append("so their FP8 errors should be comparable (differences from block size and fusion).\n")

    # ── 6. Conclusions ──
    L.append("## 6. Conclusions\n")
    for sl in results:
        prec = results[sl].get("precision", {})
        nsys = results[sl].get("nsys", {})
        L.append(f"### {sl}\n")
        for p in PATH_NAMES:
            pm = prec.get(p, {})
            nm = nsys.get(p, {})
            rr = pm.get("rrmse_mean")
            piu = nm.get("per_iter_us_median")
            if rr is not None:
                L.append(f"- **{PATH_LABELS[p]}**: RRMSE={rr:.4f}" + (f", GPU-proj={piu:.0f}us" if piu else ""))
        L.append("")

    L.append("---")
    L.append("*Benchmark: `tools/cross_framework_bench.py`*")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nReport: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(shape_labels: list[str], gpu: int, skip_nsys: bool,
                  num_gpus: int = 1) -> dict:
    all_results = {}
    base_tmp = tempfile.mkdtemp(prefix="moe_xfw_")
    print(f"Temp dir: {base_tmp}")

    # ── Phase 1: Precision (parallelized across GPUs) ──
    # Strategy: run all (shape, seed, path) combinations.
    # Each subprocess is pinned to a GPU. With 8 GPUs, we can run 8 concurrent jobs.
    # To avoid OOM: one subprocess per GPU at a time (each needs ~few GB).
    import concurrent.futures

    precision_jobs = []  # (shape_label, seed, path_name, python, script_tpl, data_dir)
    for sl in shape_labels:
        shape = SHAPES[sl]
        S, H, I, E, K = shape["S"], shape["H"], shape["I"], shape["E"], shape["K"]
        for seed in SEEDS:
            data_dir = os.path.join(base_tmp, sl, f"seed{seed}")
            generate_data(shape, data_dir, seed=seed)
            for pname, py, tpl in [
                ("paddle_bf16", SYSTEM_PYTHON, PADDLE_BF16_SCRIPT),
                ("paddle_fp8",  SYSTEM_PYTHON, PADDLE_FP8_SCRIPT),
                ("sonic_bf16",  XFER_PYTHON,   SONIC_BF16_SCRIPT),
                ("sonic_fp8",   XFER_PYTHON,   SONIC_FP8_SCRIPT),
            ]:
                fmt = dict(
                    data_dir=data_dir, reference_core_root=REFERENCE_CORE_ROOT,
                    reference_expert_node=REFERENCE_EXPERT_NODE, project_root=PROJECT_ROOT,
                    S=S, H=H, I=I, E=E, K=K, mode="precision",
                    warmup=NSYS_WARMUP, iters=NSYS_ITERS,
                )
                precision_jobs.append((sl, seed, pname, py, tpl, fmt, data_dir))

    print(f"\n[Phase 1] Precision: {len(precision_jobs)} jobs across {num_gpus} GPUs ...")

    gpu_counter = [0]  # mutable counter for round-robin GPU assignment
    def _run_precision_job(job):
        sl, seed, pname, py, tpl, fmt, data_dir = job
        g = gpu + (gpu_counter[0] % num_gpus)
        gpu_counter[0] += 1
        label = f"{PATH_LABELS[pname]}[{sl}/s={seed}]"
        return (sl, seed, pname, run_runner(label, py, tpl, fmt, g))

    # Run with bounded parallelism = num_gpus (one job per GPU)
    job_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_gpus, 8)) as ex:
        futures = [ex.submit(_run_precision_job, j) for j in precision_jobs]
        for f in concurrent.futures.as_completed(futures):
            job_results.append(f.result())

    # ── Aggregate precision results ──
    for sl in shape_labels:
        shape = SHAPES[sl]
        theory = compute_theory(shape)
        seed_metrics = {p: [] for p in PATH_NAMES}
        pairwise_metrics = {}
        memory = {}

        print(f"\n{'='*72}")
        print(f"Shape: {sl}  S={shape['S']} H={shape['H']} I={shape['I']} E={shape['E']} K={shape['K']}  (TK={shape['S']*shape['K']})")
        print(f"{'='*72}")

        for seed in SEEDS:
            data_dir = os.path.join(base_tmp, sl, f"seed{seed}")
            gold_reference = np.load(os.path.join(data_dir, "gold_reference.npy"))
            gold_sonic = np.load(os.path.join(data_dir, "gold_sonic.npy"))
            outputs = {}

            for res_sl, res_seed, res_pname, res_data in job_results:
                if res_sl != sl or res_seed != seed:
                    continue
                # Collect memory from first seed
                if seed == SEEDS[0] and res_data.get("status") == "ok":
                    memory[res_pname] = {
                        "baseline": res_data.get("mem_baseline_mib", "—"),
                        "post_fwd": res_data.get("mem_post_fwd_mib", "—"),
                        "post_bwd": res_data.get("mem_post_bwd_mib", "—"),
                        "peak_fwd": res_data.get("peak_fwd_mib", "—"),
                        "peak_total": res_data.get("peak_total_mib", "—"),
                    }
                # Load output
                npy = os.path.join(data_dir, f"{res_pname}_output.npy")
                if os.path.exists(npy):
                    outputs[res_pname] = np.load(npy)

            # Compare each path against its correct gold (forward output)
            for pname, arr in outputs.items():
                gold = gold_reference if "paddle" in pname else gold_sonic
                m = precision_metrics(arr, gold)
                seed_metrics[pname].append(m)

            # ── Gradient precision (within-framework BF16 vs FP8) ──
            # dx is (S, H) — comparable cross-framework
            # dw1, dw2 have different layouts — only compare within-framework
            if seed == SEEDS[0]:
                # Load gradients where available
                grads = {}
                for pname in PATH_NAMES:
                    pref = pname
                    dx_f = os.path.join(data_dir, f"{pref}_dx.npy")
                    if os.path.exists(dx_f):
                        grads[pname] = {"dx": np.load(dx_f)}
                    # Per-expert dw1/dw2 (Paddle) or stacked (SonicMoE)
                    dw1_f = os.path.join(data_dir, f"{pref}_dw1.npy")
                    dw2_f = os.path.join(data_dir, f"{pref}_dw2.npy")
                    if os.path.exists(dw1_f):
                        grads.setdefault(pname, {})["dw1"] = np.load(dw1_f)
                    if os.path.exists(dw2_f):
                        grads.setdefault(pname, {})["dw2"] = np.load(dw2_f)
                    # Paddle per-expert dw1/dw2 (concat)
                    dw1_parts = []
                    for e in range(shape["E"]):
                        f = os.path.join(data_dir, f"{pref}_dw1_e{e}.npy")
                        if os.path.exists(f):
                            dw1_parts.append(np.load(f))
                    if dw1_parts:
                        grads.setdefault(pname, {})["dw1_per_expert"] = dw1_parts
                    dw2_parts = []
                    for e in range(shape["E"]):
                        f = os.path.join(data_dir, f"{pref}_dw2_e{e}.npy")
                        if os.path.exists(f):
                            dw2_parts.append(np.load(f))
                    if dw2_parts:
                        grads.setdefault(pname, {})["dw2_per_expert"] = dw2_parts

                # Within-framework gradient comparisons
                grad_pairs = [
                    ("Paddle BF16→FP8 dx", "paddle_bf16", "paddle_fp8", "dx"),
                    ("SonicMoE BF16→FP8 dx", "sonic_bf16", "sonic_fp8", "dx"),
                ]
                for label, a, b, key in grad_pairs:
                    if a in grads and key in grads[a] and b in grads and key in grads[b]:
                        pairwise_metrics[label] = precision_metrics(grads[b][key], grads[a][key])

            if seed == SEEDS[0]:
                pairs = [
                    ("Paddle FP8 vs Paddle BF16", "paddle_fp8", "paddle_bf16"),
                    ("SonicMoE FP8 vs SonicMoE BF16", "sonic_fp8", "sonic_bf16"),
                    ("Paddle FP8 vs SonicMoE FP8", "paddle_fp8", "sonic_fp8"),
                    ("Paddle BF16 vs SonicMoE BF16", "paddle_bf16", "sonic_bf16"),
                    ("Gold reference vs Gold SonicMoE", None, None),  # special
                ]
                for label, a, b in pairs:
                    if label == "Gold reference vs Gold SonicMoE":
                        pairwise_metrics[label] = precision_metrics(gold_reference, gold_sonic)
                    elif a in outputs and b in outputs:
                        pairwise_metrics[label] = precision_metrics(outputs[a], outputs[b])

        precision_avg = {}
        for pname in PATH_NAMES:
            if not seed_metrics[pname]:
                continue
            avg = {}
            for key in seed_metrics[pname][0]:
                vals = [m[key] for m in seed_metrics[pname]]
                avg[f"{key}_mean"] = float(np.mean(vals))
                avg[f"{key}_std"] = float(np.std(vals))
            precision_avg[pname] = avg

        for pname in PATH_NAMES:
            m = precision_avg.get(pname)
            if m:
                print(f"    {PATH_LABELS[pname]:20s} RRMSE={m['rrmse_mean']:.6f}±{m['rrmse_std']:.6f}  cosine={m['cosine_mean']:.6f}")

        pairwise_avg = {}
        for label, m in pairwise_metrics.items():
            pairwise_avg[label] = {f"{k}_mean": v for k, v in m.items()}

        # ── Phase 2: nsys (parallel: 4 paths on 4 GPUs, each repeat sequential) ──
        nsys_results = {}
        if not skip_nsys and shutil.which("nsys"):
            # Persistent nsys directory structure:
            #   reports/nsys_xfw/T{S}_H{H}_I{I}_E{E}_K{K}/{path}/r{rep}.nsys-rep
            shape_dir = f"T{shape['S']}_H{shape['H']}_I{shape['I']}_E{shape['E']}_K{shape['K']}"
            nsys_base = os.path.join(PROJECT_ROOT, "reports", "nsys_xfw", shape_dir)
            os.makedirs(nsys_base, exist_ok=True)
            data_dir_nsys = os.path.join(base_tmp, sl, f"seed{SEEDS[0]}")
            fmt_nsys = dict(
                data_dir=data_dir_nsys, reference_core_root=REFERENCE_CORE_ROOT,
                reference_expert_node=REFERENCE_EXPERT_NODE, project_root=PROJECT_ROOT,
                S=shape["S"], H=shape["H"], I=shape["I"], E=shape["E"], K=shape["K"],
                mode="nsys", warmup=NSYS_WARMUP, iters=NSYS_ITERS,
            )

            path_configs_nsys = [
                ("paddle_bf16", SYSTEM_PYTHON, PADDLE_BF16_SCRIPT),
                ("paddle_fp8",  SYSTEM_PYTHON, PADDLE_FP8_SCRIPT),
                ("sonic_bf16",  XFER_PYTHON,   SONIC_BF16_SCRIPT),
                ("sonic_fp8",   XFER_PYTHON,   SONIC_FP8_SCRIPT),
            ]

            # Parallel across GPUs: each path gets a dedicated GPU
            print(f"\n  [nsys] {NSYS_REPEATS} repeats × 4 paths (parallel on GPUs {gpu}-{gpu+min(num_gpus,4)-1}) ...")

            def _run_nsys_path(args):
                pname, py, tpl, path_gpu = args
                # Per-path subdirectory: reports/nsys_xfw/{shape}/{path}/r{N}
                path_dir = os.path.join(nsys_base, pname)
                os.makedirs(path_dir, exist_ok=True)
                vals = []
                for rep in range(NSYS_REPEATS):
                    prefix = os.path.join(path_dir, f"r{rep}")
                    r = run_nsys(f"{PATH_LABELS[pname]}[rep={rep}]", py, tpl,
                                 fmt_nsys, prefix, path_gpu)
                    piu = r.get("per_iter_us")
                    if piu is not None:
                        vals.append(piu)
                return pname, vals

            nsys_tasks = [(pn, py, tpl, gpu + i % min(num_gpus, 4))
                          for i, (pn, py, tpl) in enumerate(path_configs_nsys)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_gpus, 4)) as ex:
                for pname, vals in ex.map(_run_nsys_path, nsys_tasks):
                    if vals:
                        med = float(np.median(vals))
                        nsys_results[pname] = {
                            "per_iter_us_median": round(med, 1),
                            "per_iter_us_all": vals,
                            "per_iter_us_std": round(float(np.std(vals)), 1),
                        }
                        print(f"    {PATH_LABELS[pname]:20s} GPU-proj={med:.1f} us (median of {vals})")
        else:
            print("\n  [nsys] Skipped.")

        all_results[sl] = {
            "shape": shape,
            "theory": theory,
            "precision": precision_avg,
            "pairwise_precision": pairwise_avg,
            "memory": memory,
            "nsys": nsys_results,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="4-Way Cross-Framework MoE Benchmark")
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES.keys()),
                        choices=list(SHAPES.keys()))
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (default: from CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for parallel precision tests (default: 1)")
    parser.add_argument("--paddle-split-gemm", action="store_true",
                        help="Use split_group_gemm instead of grouped GEMM contiguous (default: grouped)")
    parser.add_argument("--report", default=os.path.join(PROJECT_ROOT, "reports", "cross_framework_report.md"))
    args = parser.parse_args()

    gpu = args.gpu
    if gpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        gpu = int(cvd.split(",")[0]) if cvd else 0

    print("Cross-Framework MoE Benchmark (4-Way)")
    print(f"  Shapes: {args.shapes}")
    print(f"  GPU: {gpu}")
    print(f"  nsys: {'skip' if args.skip_nsys else f'{NSYS_REPEATS} repeats (exclusive GPU {gpu})'}")
    print(f"  Parallel GPUs: {args.num_gpus} (precision), GPU {gpu} (nsys exclusive)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Paths: {list(PATH_LABELS.values())}")
    print()

    # Apply --paddle-split-gemm flag: rebuild templates with split mode
    global PADDLE_BF16_SCRIPT, PADDLE_FP8_SCRIPT
    use_grouped = "False" if args.paddle_split_gemm else "True"
    PADDLE_BF16_SCRIPT = _PADDLE_SCRIPT_TEMPLATE.replace("{use_fp8}", "False").replace("{output_prefix}", "paddle_bf16").replace("{use_grouped_gemm}", use_grouped)
    PADDLE_FP8_SCRIPT = _PADDLE_SCRIPT_TEMPLATE.replace("{use_fp8}", "True").replace("{output_prefix}", "paddle_fp8").replace("{use_grouped_gemm}", use_grouped)

    gemm_mode = "split" if args.paddle_split_gemm else "grouped contiguous"
    print(f"  Paddle GEMM: {gemm_mode}")

    results = run_benchmark(args.shapes, gpu, args.skip_nsys, num_gpus=args.num_gpus)

    json_path = args.report.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON: {json_path}")

    generate_report(results, args.report)


if __name__ == "__main__":
    main()
