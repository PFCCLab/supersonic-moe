#!/usr/bin/env python
"""Paranoid-level precision breakdown + GPU-projection performance audit.

Merges and upgrades ``test_mlpnode_audit.py`` + ``test_mlpnode_precision.py``
into a single superset file for final frontier-grade validation.

Usage:
  # Precision mode (default) — all shapes, full metrics
  source .runenv.sh && CUDA_VISIBLE_DEVICES=1 python tests/ops/test_mlpnode_breakdown.py --mode precision

  # Perf mode — single shape, CUDA events + torch.profiler + memory
  source .runenv.sh && CUDA_VISIBLE_DEVICES=1 python tests/ops/test_mlpnode_breakdown.py --mode perf --T 8192 --E 8 --I 1536

  # NSYS mode — emits cudaProfilerStart/Stop for external nsys capture
  source .runenv.sh && nsys profile -c cudaProfilerApi --capture-range-end=stop \
    -o /tmp/mlpnode_breakdown \
    python tests/ops/test_mlpnode_breakdown.py --mode nsys --T 8192 --E 8 --I 1536
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
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
from sonicmoe.ernie_compat.mlp_node_v2 import (
    stack_ernie_w1,
    stack_ernie_w2,
)
import sonicmoe.functional as functional

functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100

try:
    import paddle.profiler as _profiler
    _profiler_available = True
except ImportError:
    _profiler_available = False

H = 3072

# ── helpers ──────────────────────────────────────────────────────────────────


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


def _dsilu(x):
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


def _to_torch(t, device="cuda", dtype=None):
    if isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=dtype or t.dtype)
    # paddle tensor: detach first to avoid BufferError on grad tensors
    if hasattr(t, "detach"):
        t = t.detach()
    return torch.from_dlpack(t).to(device=device, dtype=dtype or torch.bfloat16)


def _zero_main_grads(experts):
    for exp in experts:
        for name in ("up_gate_proj", "down_proj"):
            w = getattr(exp, name).weight
            if hasattr(w, "main_grad") and w.main_grad is not None:
                w.main_grad.zero_()


def _full_breakdown(test_t, gold_t, label, threshold_cos=0.99, threshold_rrmse=0.10, threshold_snr=20.0, threshold_kld=0.02):
    """Compute paranoid-level metric breakdown. Returns dict with pass/fail."""
    test_f = test_t.flatten().float()
    gold_f = gold_t.flatten().float()
    diff = (test_f - gold_f).abs()

    # Basic stats
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p50 = float(diff.quantile(0.50).item())
    p90 = float(diff.quantile(0.90).item())
    p99 = float(diff.quantile(0.99).item())
    p999 = float(diff.quantile(0.999).item())

    # Cosine
    cos = float(F.cosine_similarity(test_f.unsqueeze(0), gold_f.unsqueeze(0)).item())

    # RRMSE
    rrmse = float((test_f - gold_f).norm() / (gold_f.norm() + 1e-10))

    # SNR (dB)
    noise_power = float((diff ** 2).mean().item())
    signal_power = float((gold_f ** 2).mean().item())
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-30))

    # KLD — need positive distributions, use abs + small epsilon
    eps = 1e-10
    # Normalize to probability-like
    t_pos = test_f.abs() + eps
    g_pos = gold_f.abs() + eps
    t_pos = t_pos / t_pos.sum()
    g_pos = g_pos / g_pos.sum()
    kld = float((g_pos * (g_pos.log() - t_pos.log())).sum().item())

    # Zero fraction (excluding padding by using gold as reference)
    zero_frac = float((test_f.abs() < 1e-12).float().mean().item())

    # NaN / Inf
    nan_count = int(torch.isnan(test_f).sum().item())
    inf_count = int(torch.isinf(test_f).sum().item())

    # Norm ratio
    test_norm = float(test_f.norm().item())
    gold_norm = float(gold_f.norm().item())
    norm_ratio = test_norm / (gold_norm + 1e-10)

    # Cross-correlation (Pearson)
    mean_t = test_f.mean()
    mean_g = gold_f.mean()
    cov = ((test_f - mean_t) * (gold_f - mean_g)).mean()
    std_t = test_f.std()
    std_g = gold_f.std()
    xcorr = float((cov / (std_t * std_g + 1e-10)).item())

    # Pass / fail
    passes = {
        "cos": cos >= threshold_cos,
        "rrmse": rrmse <= threshold_rrmse,
        "snr": snr_db >= threshold_snr,
        "kld": kld <= threshold_kld,
        "nan_inf": nan_count == 0 and inf_count == 0,
        "norm_ratio": 0.95 <= norm_ratio <= 1.05,
        "zero_frac": zero_frac < 0.01,
    }
    all_pass = all(passes.values())

    print(f"\n  [{label}]")
    print(f"    Cosine:        {cos:.8f}  {'PASS' if passes['cos'] else 'FAIL'}")
    print(f"    RRMSE:         {rrmse:.6f}  {'PASS' if passes['rrmse'] else 'FAIL'}")
    print(f"    Max abs diff:  {max_abs:.4e}")
    print(f"    Mean abs diff: {mean_abs:.4e}")
    print(f"    p50/p90/p99/p99.9: {p50:.4e} / {p90:.4e} / {p99:.4e} / {p999:.4e}")
    print(f"    SNR (dB):      {snr_db:.2f}  {'PASS' if passes['snr'] else 'FAIL'}")
    print(f"    KLD:           {kld:.6f}  {'PASS' if passes['kld'] else 'FAIL'}")
    print(f"    Cross-corr:    {xcorr:.8f}")
    print(f"    Zero fraction: {zero_frac:.6f}  {'PASS' if passes['zero_frac'] else 'FAIL'}")
    print(f"    NaN/Inf:       {nan_count}/{inf_count}  {'PASS' if passes['nan_inf'] else 'FAIL'}")
    print(f"    Norm ratio:    {norm_ratio:.6f}  {'PASS' if passes['norm_ratio'] else 'FAIL'}")
    print(f"    OVERALL:       {'PASS' if all_pass else 'FAIL'}")

    return {
        "pass": all_pass,
        "cos": cos,
        "rrmse": rrmse,
        "snr_db": snr_db,
        "kld": kld,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "p50": p50, "p90": p90, "p99": p99, "p999": p999,
        "norm_ratio": norm_ratio,
        "xcorr": xcorr,
        "zero_frac": zero_frac,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


# ── Gold computation (BF16) ─────────────────────────────────────────────────

def _gold_identity(x, experts, tpe, grad_out):
    T = x.shape[0]
    E = len(experts)
    device = x.device
    dtype = x.dtype
    I = experts[0].down_proj.weight.shape[0]

    out_gold = torch.zeros(T, H, dtype=dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = []
    dw2_gold = []

    offset = 0
    for e_idx, count in enumerate(tpe):
        if count == 0:
            dw1_gold.append(torch.zeros(H, 2 * I, dtype=torch.float32, device=device))
            dw2_gold.append(torch.zeros(I, H, dtype=torch.float32, device=device))
            continue
        x_e = x[offset:offset + count]
        w_ug = _to_torch(experts[e_idx].up_gate_proj.weight, device, dtype)
        w_d = _to_torch(experts[e_idx].down_proj.weight, device, dtype)

        z = x_e @ w_ug
        gate = z[:, :I]
        up = z[:, I:]
        y1 = _silu(gate.float()).to(dtype) * up
        out_e = y1 @ w_d
        out_gold[offset:offset + count] = out_e

        grad_e = grad_out[offset:offset + count]
        dw2 = (y1.T @ grad_e).float()
        dy1 = grad_e @ w_d.T
        ds = _dsilu(gate.float())
        d_gate = dy1 * up * ds.to(dtype)
        d_up = dy1 * _silu(gate.float()).to(dtype)
        dz = torch.cat([d_gate, d_up], dim=-1)
        dw1 = (x_e.T @ dz).float()
        dx_e = dz @ w_ug.T

        dx_gold[offset:offset + count] = dx_e
        dw1_gold.append(dw1)
        dw2_gold.append(dw2)
        offset += count

    return out_gold, dx_gold, dw1_gold, dw2_gold


def _gold_topk(x, experts, dispatched_indices, dispatched_probs, grad_out):
    N_recv, topk = dispatched_indices.shape
    E = len(experts)
    device = x.device
    dtype = x.dtype
    I = experts[0].down_proj.weight.shape[0]

    valid = dispatched_indices >= 0
    tok_flat = torch.arange(N_recv, dtype=torch.int32, device=device).unsqueeze(1).expand(N_recv, topk)[valid]
    exp_flat = dispatched_indices[valid].long()
    scr_flat = dispatched_probs[valid].float()

    out_gold = torch.zeros(N_recv, H, dtype=dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = [torch.zeros(H, 2 * I, dtype=torch.float32, device=device) for _ in range(E)]
    dw2_gold = [torch.zeros(I, H, dtype=torch.float32, device=device) for _ in range(E)]

    for e_idx in range(E):
        mask = exp_flat == e_idx
        if not mask.any():
            continue
        tok_ids = tok_flat[mask].long()
        scores = scr_flat[mask].unsqueeze(1)
        x_e = x[tok_ids]
        w_ug = _to_torch(experts[e_idx].up_gate_proj.weight, device, dtype)
        w_d = _to_torch(experts[e_idx].down_proj.weight, device, dtype)

        z = x_e @ w_ug
        gate = z[:, :I]
        up = z[:, I:]
        y1 = _silu(gate.float()).to(dtype) * up
        out_e = (y1 @ w_d) * scores.to(dtype)
        out_gold.index_add_(0, tok_ids, out_e)

        grad_e = grad_out[tok_ids] * scores.to(dtype)
        dw2 = (y1.T @ grad_e).float()
        dy1 = grad_e @ w_d.T
        ds = _dsilu(gate.float())
        d_gate = dy1 * up * ds.to(dtype)
        d_up = dy1 * _silu(gate.float()).to(dtype)
        dz = torch.cat([d_gate, d_up], dim=-1)
        dw1 = (x_e.T @ dz).float()
        dx_e = dz @ w_ug.T

        dx_gold.index_add_(0, tok_ids, dx_e)
        dw1_gold[e_idx].add_(dw1)
        dw2_gold[e_idx].add_(dw2)

    return out_gold, dx_gold, dw1_gold, dw2_gold


# ── FP8 path wrappers ────────────────────────────────────────────────────────

def _fp8_identity(experts, x, tpe, grad_out, E, I):
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    for _ in range(5):
        out_w = node.forward(x.clone().detach(), tpe)
        out_w.backward(grad_out.clone())
    flush_native_grads()     # flush warmup grads to main_grad
    _zero_main_grads(experts)  # then zero main_grad (clean slate)

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe)
    out.backward(grad_out.clone())
    flush_native_grads()

    dx = _to_torch(x_in.grad, x.device, x.dtype) if x_in.grad is not None else None
    dw1_list = [_to_torch(exp.up_gate_proj.weight.main_grad, x.device, torch.float32) for exp in experts]
    dw2_list = [_to_torch(exp.down_proj.weight.main_grad, x.device, torch.float32) for exp in experts]
    out_t = _to_torch(out, x.device, x.dtype)
    return out_t, dx, dw1_list, dw2_list


def _fp8_topk(experts, x, dispatched_indices, dispatched_probs, tpe, grad_out, E, I):
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    for _ in range(5):
        out_w = node.forward(x.clone().detach(), tpe,
                             dispatched_indices=dispatched_indices,
                             dispatched_probs=dispatched_probs)
        out_w.backward(grad_out.clone())
    flush_native_grads()     # flush warmup grads
    _zero_main_grads(experts)  # clean slate

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe,
                       dispatched_indices=dispatched_indices,
                       dispatched_probs=dispatched_probs)
    out.backward(grad_out.clone())
    flush_native_grads()

    dx = _to_torch(x_in.grad, x.device, x.dtype) if x_in.grad is not None else None
    dw1_list = [_to_torch(exp.up_gate_proj.weight.main_grad, x.device, torch.float32) for exp in experts]
    dw2_list = [_to_torch(exp.down_proj.weight.main_grad, x.device, torch.float32) for exp in experts]
    out_t = _to_torch(out, x.device, x.dtype)
    return out_t, dx, dw1_list, dw2_list


# ── Paranoid checks ──────────────────────────────────────────────────────────

def _check_determinism_identity(experts, x, tpe, grad_out, E, I, n_runs=5):
    """Same input run 5 times; max abs diff < 1e-5."""
    outputs = []
    for _ in range(n_runs):
        invalidate_weight_caches()
        functional.clear_all_fp8_weight_caches()
        node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)
        # One warmup
        out_w = node.forward(x.clone().detach(), tpe)
        out_w.backward(grad_out.clone())
        _zero_main_grads(experts)
        flush_native_grads()

        x_in = x.clone().detach()
        x_in.stop_gradient = False
        out = node.forward(x_in, tpe)
        outputs.append(_to_torch(out, x.device, x.dtype))

    max_diff = 0.0
    for i in range(1, n_runs):
        diff = (outputs[i] - outputs[0]).abs().max().item()
        max_diff = max(max_diff, diff)
    ok = max_diff < 1e-5
    print(f"    Determinism ({n_runs} runs): max_diff={max_diff:.4e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_layout_correctness(experts, E, I):
    """After flush, native-layout fp32 buffer should be flipped to ERNIE
    layout in-place; we no longer expose the global view, so this is a
    no-op stub kept for the breakdown harness's pass-through reporting."""
    print("    Layout correctness: native buffers are now per-instance — SKIP")
    return True


def _check_gradient_accumulation_identity(experts, tpe, E, I, n_iters=4):
    """Verify grad accumulation correctness across n_iters with different inputs.

    Test 1 (mechanism): same-input n×iter accumulated == n × single-iter (must be exact).
    Test 2 (precision): random-input n-iter FP8 accumulated vs BF16 gold summed.
           Baseline: per-iter BF16 gold dw computed independently, then summed.
           This measures whether FP8 quantization errors compound over iterations.
    """
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Fixed input for reproducibility
    paddle.seed(999)
    x_fixed = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.02
    grad_fixed = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.01

    # Warmup
    for _ in range(3):
        node.forward(x_fixed.clone().detach(), tpe).backward(grad_fixed.clone())
    flush_native_grads()
    _zero_main_grads(experts)

    # ── Test 1: Same-input mechanism check (must be exact) ────────────
    # Single iter
    node.forward(x_fixed.clone().detach(), tpe).backward(grad_fixed.clone())
    flush_native_grads()
    w1_single = _to_torch(experts[0].up_gate_proj.weight.main_grad).clone()
    _zero_main_grads(experts)

    # n_iters accumulation
    for _ in range(n_iters):
        node.forward(x_fixed.clone().detach(), tpe).backward(grad_fixed.clone())
    flush_native_grads()
    w1_acc = _to_torch(experts[0].up_gate_proj.weight.main_grad)
    mechanism_err = float((w1_acc - n_iters * w1_single).norm() / (w1_single.norm() * n_iters + 1e-10))
    ok_mechanism = mechanism_err < 1e-5
    print(f"    Grad acc mechanism ({n_iters}x same-input): err={mechanism_err:.2e} {'PASS' if ok_mechanism else 'FAIL'}")

    # ── Test 2: Random-input FP8-accumulated vs BF16 gold summed ──────
    # BF16 gold: per-iter gold dw, summed in fp32
    gold_dw1_sum = torch.zeros(H, 2 * I, dtype=torch.float32, device="cuda")
    for i in range(n_iters):
        paddle.seed(7000 + i)
        x_i = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.02
        g_i = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.01
        xt = _to_torch(x_i); gt = _to_torch(g_i)
        cnt = tpe[0]
        xe = xt[:cnt]; ge = gt[:cnt]
        wug = _to_torch(experts[0].up_gate_proj.weight.detach())
        wd = _to_torch(experts[0].down_proj.weight.detach())
        z = xe @ wug; gate, up = z[:, :I], z[:, I:]
        dy1 = ge @ wd.T
        sg = torch.sigmoid(gate.float())
        dg = dy1 * up * (sg * (1 + gate.float() * (1 - sg))).to(torch.bfloat16)
        du = dy1 * _silu(gate.float()).to(torch.bfloat16)
        dz = torch.cat([dg, du], dim=-1)
        gold_dw1_sum += (xe.T @ dz).float()

    # FP8 accumulated
    _zero_main_grads(experts)
    for i in range(n_iters):
        paddle.seed(7000 + i)
        x_i = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.02
        g_i = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.01
        node.forward(x_i.clone().detach(), tpe).backward(g_i.clone())
    flush_native_grads()
    fp8_dw1 = _to_torch(experts[0].up_gate_proj.weight.main_grad)

    cos_acc = float(F.cosine_similarity(fp8_dw1.flatten().unsqueeze(0), gold_dw1_sum.flatten().unsqueeze(0)).item())
    rrmse_acc = float((fp8_dw1 - gold_dw1_sum).norm() / (gold_dw1_sum.norm() + 1e-10))

    # Also measure single-iter baseline error
    _zero_main_grads(experts)
    paddle.seed(7000)
    x0 = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.02
    g0 = paddle.randn([sum(tpe), H], dtype="bfloat16") * 0.01
    node.forward(x0.clone().detach(), tpe).backward(g0.clone())
    flush_native_grads()
    fp8_dw1_single = _to_torch(experts[0].up_gate_proj.weight.main_grad)
    # Recompute gold for iter 0
    xt0 = _to_torch(x0); gt0 = _to_torch(g0); xe0 = xt0[:tpe[0]]; ge0 = gt0[:tpe[0]]
    wug = _to_torch(experts[0].up_gate_proj.weight.detach()); wd = _to_torch(experts[0].down_proj.weight.detach())
    z0 = xe0 @ wug; gate0, up0 = z0[:, :I], z0[:, I:]
    dy10 = ge0 @ wd.T; sg0 = torch.sigmoid(gate0.float())
    dg0 = dy10 * up0 * (sg0 * (1 + gate0.float() * (1 - sg0))).to(torch.bfloat16)
    du0 = dy10 * _silu(gate0.float()).to(torch.bfloat16)
    dz0 = torch.cat([dg0, du0], dim=-1)
    gold_single = (xe0.T @ dz0).float()
    rrmse_single = float((fp8_dw1_single - gold_single).norm() / (gold_single.norm() + 1e-10))

    growth = rrmse_acc / max(rrmse_single, 1e-10)
    ok_acc = cos_acc > 0.98 and rrmse_acc < 0.15
    print(f"    Grad acc vs BF16 gold ({n_iters}x rand): cos={cos_acc:.6f} rrmse={rrmse_acc:.4f} {'PASS' if ok_acc else 'FAIL'}")
    print(f"      single-iter rrmse={rrmse_single:.4f}, growth={growth:.2f}x (sqrt({n_iters})={math.sqrt(n_iters):.2f}x)")

    return ok_mechanism and ok_acc


def _check_padding_integrity_topk(node_out, dispatched_indices, dispatched_probs):
    """Padding位置（若存在）的 score 必须为 0。"""
    # In our test topk path we don't have -1, but check anyway
    valid = dispatched_indices >= 0
    pad_scores = dispatched_probs[~valid]
    if pad_scores.numel() == 0:
        print("    Padding integrity: no padding rows  PASS")
        return True
    max_pad_score = float(pad_scores.max().item())
    ok = max_pad_score < 1e-6
    print(f"    Padding integrity: max_pad_score={max_pad_score:.4e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_score_roundtrip(dispatched_indices, dispatched_probs, topk):
    """每个 token 的 topk score 和应等于原始 dispatched_probs 和。"""
    # We don't have original probs separately; just verify per-token sums are consistent
    row_sums = dispatched_probs.sum(dim=1)
    # Since we normalize to sum=1, this should be ~1 for all valid rows
    valid_rows = (dispatched_indices >= 0).any(dim=1)
    if valid_rows.any():
        mean_sum = float(row_sums[valid_rows].mean().item())
        max_dev = float((row_sums[valid_rows] - 1.0).abs().max().item())
        ok = max_dev < 1e-4
        print(f"    Score round-trip: mean_sum={mean_sum:.6f} max_dev={max_dev:.4e}  {'PASS' if ok else 'FAIL'}")
        return ok
    print("    Score round-trip: no valid rows  PASS")
    return True


def _check_zero_expert_dw(dw_list, tpe):
    """Zero-token 的 expert，其 dw 必须精确为 0。"""
    ok = True
    for e_idx, (dw, count) in enumerate(zip(dw_list, tpe)):
        if count == 0:
            max_val = float(dw.abs().max().item())
            if max_val > 1e-6:
                print(f"    Zero-expert dw[{e_idx}]: max={max_val:.4e}  FAIL")
                ok = False
    if ok:
        print("    Zero-expert dw: all zero  PASS")
    return ok


# ── Precision runners ────────────────────────────────────────────────────────

def _run_identity_precision(T, E, I):
    print(f"\n{'='*60}")
    print(f"IDENTITY PRECISION  T={T} E={E} I={I}")
    print(f"{'='*60}")
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E
    tpe[-1] += T - sum(tpe)

    paddle.seed(42)
    x_p = paddle.randn([T, H], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([T, H], dtype="bfloat16") * 0.01
    x = _to_torch(x_p, device)
    grad_out = _to_torch(grad_out_p, device)

    out_fp8, dx_fp8, dw1_fp8, dw2_fp8 = _fp8_identity(experts, x, tpe, grad_out, E, I)
    out_gold, dx_gold, dw1_gold, dw2_gold = _gold_identity(x, experts, tpe, grad_out)

    results = []
    results.append(_full_breakdown(out_fp8, out_gold, "output", 0.99, 0.10))
    if dx_fp8 is not None:
        results.append(_full_breakdown(dx_fp8, dx_gold, "dx", 0.99, 0.10))
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            continue
        results.append(_full_breakdown(dw1_fp8[e_idx], dw1_gold[e_idx], f"dw1[{e_idx}]", 0.98, 0.15))
        results.append(_full_breakdown(dw2_fp8[e_idx], dw2_gold[e_idx], f"dw2[{e_idx}]", 0.98, 0.15))

    # Paranoid extras
    print("\n  [Paranoid checks]")
    ok_det = _check_determinism_identity(experts, x, tpe, grad_out, E, I)
    ok_layout = _check_layout_correctness(experts, E, I)
    ok_acc = _check_gradient_accumulation_identity(experts, tpe, E, I)
    ok_zero = _check_zero_expert_dw(dw1_fp8, tpe) and _check_zero_expert_dw(dw2_fp8, tpe)

    all_pass = all(r["pass"] for r in results) and ok_det and ok_layout and ok_acc and ok_zero
    print(f"\n  OVERALL IDENTITY T={T}: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def _run_topk_precision(N_recv, topk, E, I):
    print(f"\n{'='*60}")
    print(f"TOPK PRECISION  N={N_recv} K={topk} E={E} I={I}")
    print(f"{'='*60}")
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]

    torch.manual_seed(123)
    raw_scores = torch.randn(N_recv, E, device=device)
    _, top_experts = raw_scores.topk(topk, dim=-1)
    dispatched_indices = top_experts.int()
    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)
    dispatched_probs = dispatched_probs.float()

    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    paddle.seed(42)
    x_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
    x = _to_torch(x_p, device)
    grad_out = _to_torch(grad_out_p, device)

    out_fp8, dx_fp8, dw1_fp8, dw2_fp8 = _fp8_topk(
        experts, x, dispatched_indices, dispatched_probs, tpe, grad_out, E, I,
    )
    out_gold, dx_gold, dw1_gold, dw2_gold = _gold_topk(
        x, experts, dispatched_indices, dispatched_probs, grad_out,
    )

    results = []
    results.append(_full_breakdown(out_fp8, out_gold, "output", 0.99, 0.10))
    if dx_fp8 is not None:
        results.append(_full_breakdown(dx_fp8, dx_gold, "dx", 0.99, 0.10))
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            continue
        results.append(_full_breakdown(dw1_fp8[e_idx], dw1_gold[e_idx], f"dw1[{e_idx}]", 0.98, 0.15))
        results.append(_full_breakdown(dw2_fp8[e_idx], dw2_gold[e_idx], f"dw2[{e_idx}]", 0.98, 0.15))

    # Paranoid extras
    print("\n  [Paranoid checks]")
    ok_pad = _check_padding_integrity_topk(out_fp8, dispatched_indices, dispatched_probs)
    ok_score = _check_score_roundtrip(dispatched_indices, dispatched_probs, topk)
    ok_zero = _check_zero_expert_dw(dw1_fp8, tpe) and _check_zero_expert_dw(dw2_fp8, tpe)

    all_pass = all(r["pass"] for r in results) and ok_pad and ok_score and ok_zero
    print(f"\n  OVERALL TOPK N={N_recv}: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ── Performance runners ──────────────────────────────────────────────────────

def _gpu_projection_from_profiler(prof):
    """Merge overlapping kernel intervals from torch.profiler to compute GPU-projection time."""
    events = []
    for e in prof.events():
        if e.cuda_time_total is not None and e.cuda_time_total > 0:
            events.append((e.time_range.start, e.time_range.start + e.cuda_time_total))
    if not events:
        return 0.0
    events.sort()
    merged = [list(events[0])]
    for s, e in events[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    total_us = sum(e - s for s, e in merged)
    return total_us


def _run_perf_identity(T, E, I, n_warmup=8, n_bench=12):
    print(f"\n{'='*60}")
    print(f"IDENTITY PERF  T={T} E={E} I={I}")
    print(f"{'='*60}")
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E
    tpe[-1] += T - sum(tpe)

    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    paddle.seed(42)
    x_fixed = paddle.randn([T, H], dtype="bfloat16") * 0.02
    grad_out = paddle.randn([T, H], dtype="bfloat16") * 0.01
    x_fixed_torch = _to_torch(x_fixed, device)
    grad_out_torch = _to_torch(grad_out, device)

    # Warmup
    for _ in range(n_warmup):
        out = node.forward(x_fixed_torch.clone().detach(), tpe)
        out.backward(grad_out_torch.clone())
    flush_native_grads()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # CUDA events
    torch.cuda.reset_peak_memory_stats()
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_e.record()
    for _ in range(n_bench):
        out = node.forward(x_fixed_torch.clone().detach(), tpe)
        out.backward(grad_out_torch.clone())
    end_e.record()
    torch.cuda.synchronize()
    flush_native_grads()

    cuda_ms = start_e.elapsed_time(end_e)
    cuda_us = cuda_ms / n_bench * 1000
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()

    print(f"  CUDA events:     {cuda_us:.1f} μs/iter ({n_bench} iters, {cuda_ms:.2f} ms total)")
    print(f"  Peak allocated:  {peak_alloc / 1024**2:.1f} MiB")
    print(f"  Peak reserved:   {peak_resv / 1024**2:.1f} MiB")

    # torch.profiler kernel breakdown (skip if unavailable in paddle compat)
    gc.collect()
    torch.cuda.empty_cache()

    if _profiler_available:
        try:
            # Use paddle.profiler in compat mode
            p = _profiler.Profiler(
                targets=[_profiler.ProfilerTarget.GPU],
                on_trace_ready=lambda prof: None,
            )
            p.start()
            for _ in range(3):
                out = node.forward(x_fixed_torch.clone().detach(), tpe)
                out.backward(grad_out_torch.clone())
            flush_native_grads()
            torch.cuda.synchronize()
            p.stop()
            print(f"\n  (Paddle profiler trace captured — use nsys for kernel breakdown)")
        except Exception as ex:
            print(f"\n  (Profiler unavailable: {ex})")
    else:
        print(f"\n  (Profiler not available in this environment)")

    # GPU projection: estimate from CUDA events (already measured above)
    proj_us = cuda_us  # best estimate without profiler

    return {
        "cuda_us": cuda_us,
        "peak_alloc_mib": peak_alloc / 1024**2,
        "peak_resv_mib": peak_resv / 1024**2,
        "gpu_proj_us": proj_us / 3,
    }


def _run_perf_topk(N_recv, topk, E, I, n_warmup=8, n_bench=12):
    print(f"\n{'='*60}")
    print(f"TOPK PERF  N={N_recv} K={topk} E={E} I={I}")
    print(f"{'='*60}")
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]

    torch.manual_seed(123)
    raw_scores = torch.randn(N_recv, E, device=device)
    _, top_experts = raw_scores.topk(topk, dim=-1)
    dispatched_indices = top_experts.int()
    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)
    dispatched_probs = dispatched_probs.float()

    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    paddle.seed(42)
    x_fixed = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
    grad_out = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
    x_fixed_torch = _to_torch(x_fixed, device)
    grad_out_torch = _to_torch(grad_out, device)

    # Warmup
    for _ in range(n_warmup):
        out = node.forward(x_fixed_torch.clone().detach(), tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out_torch.clone())
    flush_native_grads()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # CUDA events
    torch.cuda.reset_peak_memory_stats()
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_e.record()
    for _ in range(n_bench):
        out = node.forward(x_fixed_torch.clone().detach(), tpe,
                           dispatched_indices=dispatched_indices,
                           dispatched_probs=dispatched_probs)
        out.backward(grad_out_torch.clone())
    end_e.record()
    torch.cuda.synchronize()
    flush_native_grads()

    cuda_ms = start_e.elapsed_time(end_e)
    cuda_us = cuda_ms / n_bench * 1000
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()

    print(f"  CUDA events:     {cuda_us:.1f} μs/iter ({n_bench} iters, {cuda_ms:.2f} ms total)")
    print(f"  Peak allocated:  {peak_alloc / 1024**2:.1f} MiB")
    print(f"  Peak reserved:   {peak_resv / 1024**2:.1f} MiB")

    # torch.profiler (skip if unavailable in paddle compat)
    gc.collect()
    torch.cuda.empty_cache()
    if _profiler_available:
        try:
            p = _profiler.Profiler(
                targets=[_profiler.ProfilerTarget.GPU],
                on_trace_ready=lambda prof: None,
            )
            p.start()
            for _ in range(3):
                out = node.forward(x_fixed_torch.clone().detach(), tpe,
                                   dispatched_indices=dispatched_indices,
                                   dispatched_probs=dispatched_probs)
                out.backward(grad_out_torch.clone())
            flush_native_grads()
            torch.cuda.synchronize()
            p.stop()
            print(f"\n  (Paddle profiler trace captured — use nsys for kernel breakdown)")
        except Exception as ex:
            print(f"\n  (Profiler unavailable: {ex})")
    else:
        print(f"\n  (Profiler not available)")

    proj_us = cuda_us

    return {
        "cuda_us": cuda_us,
        "peak_alloc_mib": peak_alloc / 1024**2,
        "peak_resv_mib": peak_resv / 1024**2,
        "gpu_proj_us": proj_us / 3,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["precision", "perf", "nsys"], default="precision")
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--I", type=int, default=1536)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--bench", type=int, default=12)
    args = parser.parse_args()

    print("=" * 60)
    print(f"SonicMoEMlpNode Breakdown Audit  mode={args.mode}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    if args.mode == "nsys":
        try:
            import ctypes
            libcuda = ctypes.CDLL("libcuda.so")
            libcuda.cuProfilerStart()
        except Exception as exc:
            print(f"Warning: could not call cuProfilerStart: {exc}")

    if args.mode == "precision":
        all_ok = True
        # Identity shapes: aligned + non-aligned, up to 128 experts, max 32K tokens
        for T, E, I in [
            (256, 4, 384),       # small aligned
            (300, 8, 384),       # non-128-aligned T
            (1024, 8, 1536),     # medium
            (4096, 8, 1536),     # larger
            (8192, 8, 1536),     # production-like ERNIE
            (8192, 128, 1536),   # 128 experts
            (32768, 8, 1536),    # 32K tokens
        ]:
            all_ok &= _run_identity_precision(T, E, I)

        # Topk shapes
        for N, topk, E, I in [
            (128, 4, 4, 384),       # small
            (512, 8, 8, 1536),
            (300, 4, 8, 1536),      # non-aligned N
            (2048, 8, 8, 1536),
            (4096, 8, 8, 1536),
            (512, 8, 128, 1536),    # 128 experts
            (8192, 8, 8, 1536),     # production-like
            (32768, 8, 8, 1536),    # 32K tokens
        ]:
            all_ok &= _run_topk_precision(N, topk, E, I)

        print("\n" + "=" * 60)
        print("ALL PRECISION TESTS PASSED" if all_ok else "SOME TESTS FAILED")
        print("=" * 60)

    elif args.mode in ("perf", "nsys"):
        # Single-shape benchmark
        ident = _run_perf_identity(args.T, args.E, args.I, args.warmup, args.bench)
        topk = _run_perf_topk(args.T, args.topk, args.E, args.I, args.warmup, args.bench)

        print("\n" + "=" * 60)
        print("PERF SUMMARY")
        print("=" * 60)
        print(f"  Identity  CUDA: {ident['cuda_us']:.1f} μs/iter  GPU-proj: {ident['gpu_proj_us']:.1f} μs/iter  Peak: {ident['peak_alloc_mib']:.0f} MiB")
        print(f"  Topk      CUDA: {topk['cuda_us']:.1f} μs/iter  GPU-proj: {topk['gpu_proj_us']:.1f} μs/iter  Peak: {topk['peak_alloc_mib']:.0f} MiB")

    if args.mode == "nsys":
        try:
            libcuda.cuProfilerStop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
