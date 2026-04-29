#!/usr/bin/env python3
"""Production cold-start E2E: cache-clear → warmup → multi-shape precision + perf.

Validates ALL outputs (out, dx, dw1, dw2) against BF16 gold reference for
every shape, whether warmed up or not.  Uses GPUs 2+ (0-1 may be freq-locked).

Phases:
  0. Nuke ALL compiled artifacts
  1. Import + CUDA extension build
  2. Warmup via fwd+bwd (JIT compilation)
  3. Multi-shape precision audit (expected + unexpected shapes)
  4. nsys GPU-projection for Ernie shape (optional, --nsys flag)

Usage:
    CUDA_VISIBLE_DEVICES=2 python tests/ops/test_cold_start_e2e.py
    CUDA_VISIBLE_DEVICES=2 python tests/ops/test_cold_start_e2e.py --nsys
"""
import math
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

E, H, I = 8, 3072, 1536

# Shapes: (N_recv, topk, description, is_warmup_shape)
SHAPES = [
    (1024,  8, "warmup",       True),
    (8192,  8, "ernie",        False),
    (4096,  8, "half-ernie",   False),
    (2048,  4, "small-topk4",  False),
    (512,   8, "tiny",         False),
    (16384, 8, "double-ernie", False),
]

COS_THRESHOLD = 0.99
RRMSE_THRESHOLD = 0.10  # 10%
CUTE_COMPILE_THRESHOLD_S = 5.0


def _silu(x):
    import torch
    return x * torch.sigmoid(x)


def _print(msg, indent=0):
    print(f"{'  ' * indent}{msg}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys", action="store_true", help="Also run nsys GPU-projection")
    args = parser.parse_args()

    t_total = time.perf_counter()
    _print("=" * 80)
    _print("SonicMoE Cold-Start E2E: Precision + Performance Validation")
    _print("=" * 80)

    # ── Phase 0: Nuke everything ──
    _print("\n[Phase 0] Clearing ALL compiled artifacts...")
    for d in [os.path.expanduser("~/.triton/cache"), os.path.join(_REPO, "build")]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            _print(f"Cleared {d}", 1)
    _print("Done.", 1)

    # ── Phase 1: Import ──
    _print("\n[Phase 1] Import...")
    t1 = time.perf_counter()

    import paddle
    paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)
    import torch
    import numpy as np

    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.ernie_compat.mlp_node_v2 import (
        SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads,
    )
    import sonicmoe.ernie_compat.mlp_node_v2 as _m
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True

    _print(f"Import: {time.perf_counter()-t1:.1f}s", 1)

    # ── Build experts ──
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

    invalidate_weight_caches(); clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # ── BF16 gold reference function ──
    def _gold_topk(x_t, di_t, dp_t, grad_t, topk):
        """BF16 gold forward + backward, returns (out, dx, dw1_list, dw2_list)."""
        N_recv = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        valid = di_t >= 0
        tok_flat = torch.arange(N_recv, dtype=torch.int32, device=device) \
                       .unsqueeze(1).expand(N_recv, topk)[valid]
        exp_flat = di_t[valid].long()
        scr_flat = dp_t[valid].float()

        out_gold = torch.zeros(N_recv, H, dtype=dtype, device=device)
        dx_gold = torch.zeros_like(x_t)
        ds_gold = torch.zeros(N_recv, topk, dtype=torch.float32, device=device)
        dw1_gold = [torch.zeros(H, 2*I, dtype=torch.float32, device=device) for _ in range(E)]
        dw2_gold = [torch.zeros(I, H, dtype=torch.float32, device=device) for _ in range(E)]

        for e_idx in range(E):
            mask = exp_flat == e_idx
            if not mask.any():
                continue
            tok_ids = tok_flat[mask].long()
            scores = scr_flat[mask].unsqueeze(1)
            x_e = x_t[tok_ids]
            w_ug = torch.from_dlpack(experts[e_idx].up_gate_proj.weight.detach()).to(dtype=dtype)
            w_d = torch.from_dlpack(experts[e_idx].down_proj.weight.detach()).to(dtype=dtype)

            z = x_e @ w_ug
            gate, up = z[:, :I], z[:, I:]
            y1 = _silu(gate.float()).to(dtype) * up
            out_e_unscaled = y1 @ w_d  # [n_tok, H] — output before score scaling
            out_e = out_e_unscaled * scores.to(dtype)
            out_gold.index_add_(0, tok_ids, out_e)

            # ds: d(loss)/d(score[t,k]) = dot(grad_out[t], out_e_unscaled[t])
            grad_at_tok = grad_t[tok_ids]  # [n_tok, H]
            ds_per_tok = (grad_at_tok * out_e_unscaled).sum(dim=1).float()  # [n_tok]
            # Map back to (N_recv, topk) layout — find which topk slot this expert was in
            for local_i in range(tok_ids.shape[0]):
                t_id = tok_ids[local_i].item()
                for k_slot in range(topk):
                    if di_t[t_id, k_slot].item() == e_idx:
                        ds_gold[t_id, k_slot] = ds_per_tok[local_i]
                        break

            grad_e = grad_at_tok * scores.to(dtype)
            dy1 = grad_e @ w_d.T
            sig = torch.sigmoid(gate.float())
            d_gate = dy1 * up * (sig * (1.0 + gate.float() * (1.0 - sig))).to(dtype)
            d_up = dy1 * _silu(gate.float()).to(dtype)
            dz = torch.cat([d_gate, d_up], dim=-1)
            dw1_gold[e_idx].add_((x_e.T @ dz).float())
            dw2_gold[e_idx].add_((y1.T @ grad_e).float())
            dx_gold.index_add_(0, tok_ids, dz @ w_ug.T)

        return out_gold, dx_gold, ds_gold, dw1_gold, dw2_gold

    def _cos_sim(a, b):
        a_f = a.float().flatten()
        b_f = b.float().flatten()
        return float(torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)))

    def _rrmse(a, b):
        diff = (a.float() - b.float()).norm()
        ref = b.float().norm()
        return float(diff / ref) if ref > 0 else 0.0

    def _make_dispatch(N, topk_val):
        di = paddle.zeros([N, topk_val], dtype="int32")
        dp = paddle.randn([N, topk_val], dtype="float32").abs() + 0.1
        dp = dp / dp.sum(axis=1, keepdim=True)
        dp.stop_gradient = False  # enable ds gradient
        for i in range(N):
            di[i] = paddle.randperm(E)[:topk_val].cast("int32")
        tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()
        return di, dp, tpe

    # ── Phase 2: Warmup ──
    _print(f"\n[Phase 2] Warmup (N=1024, 2 iters for JIT)...")
    di_w, dp_w, tpe_w = _make_dispatch(1024, 8)
    grad_w = paddle.randn([1024, H], dtype="bfloat16")

    t_w = time.perf_counter()
    for _ in range(2):
        xw = paddle.randn([1024, H], dtype="bfloat16"); xw.stop_gradient = False
        invalidate_weight_caches()
        with enable_fp8(True):
            _refresh_fp8_config()
            ow = node(xw, tpe_w, di_w, dp_w)
        ow.backward(grad_w)
        flush_native_grads()
    paddle.device.synchronize()
    _print(f"Warmup: {time.perf_counter()-t_w:.1f}s", 1)

    # ── Phase 3: Multi-shape precision audit ──
    _print(f"\n[Phase 3] Multi-shape precision audit ({len(SHAPES)} shapes)...")
    _print(f"{'N':>6s} {'K':>2s} {'Label':>15s} {'Time':>6s}  "
           f"{'out':>6s} {'dx':>6s} {'ds':>6s} {'dw1':>6s} {'dw2':>6s} {'Status':>6s}", 1)
    _print("-" * 85, 1)

    all_pass = True
    first_shape = True
    for N, topk_val, desc, is_warmup in SHAPES:
        torch.manual_seed(N + topk_val)
        paddle.seed(N + topk_val)
        di, dp, tpe = _make_dispatch(N, topk_val)

        # FP8 forward + backward
        x_fp8 = paddle.randn([N, H], dtype="bfloat16") * 0.02
        x_fp8.stop_gradient = False
        grad_out = paddle.randn([N, H], dtype="bfloat16") * 0.01

        # Zero native grad buffers + main_grad for isolated per-shape comparison
        for e in experts:
            if hasattr(e.up_gate_proj.weight, 'main_grad') and e.up_gate_proj.weight.main_grad is not None:
                e.up_gate_proj.weight.main_grad.zero_()
            if hasattr(e.down_proj.weight, 'main_grad') and e.down_proj.weight.main_grad is not None:
                e.down_proj.weight.main_grad.zero_()

        # Only invalidate on first shape; steady-state reuses cache
        if first_shape:
            invalidate_weight_caches()
            first_shape = False

        t_iter = time.perf_counter()
        with enable_fp8(True):
            _refresh_fp8_config()
            out_fp8 = node(x_fp8, tpe, di, dp)
        out_fp8.backward(grad_out)
        flush_native_grads()  # transpose native accum → main_grad
        paddle.device.synchronize()
        dt = time.perf_counter() - t_iter

        # Extract FP8 results — use from_dlpack for correct bf16 conversion
        def _to_torch(t):
            if t is None: return None
            return torch.from_dlpack(t.detach())

        dx_fp8 = _to_torch(x_fp8.grad)

        # Extract dw1/dw2 from main_grad
        dw1_fp8 = [_to_torch(e.up_gate_proj.weight.main_grad) for e in experts]
        dw2_fp8 = [_to_torch(e.down_proj.weight.main_grad) for e in experts]

        # BF16 gold — use from_dlpack for paddle→torch (preserves bf16 correctly)
        x_t = torch.from_dlpack(x_fp8.detach())
        di_t = torch.from_dlpack(di.detach()).to(torch.int32)
        dp_t = torch.from_dlpack(dp.detach()).float()
        grad_t = torch.from_dlpack(grad_out.detach())
        out_gold, dx_gold, ds_gold, dw1_gold, dw2_gold = _gold_topk(x_t, di_t, dp_t, grad_t, topk_val)

        # Compare
        out_t = _to_torch(out_fp8)
        out_cos = _cos_sim(out_t, out_gold)
        dx_cos = _cos_sim(dx_fp8, dx_gold) if dx_fp8 is not None else 0.0

        # ds: gradient w.r.t. dispatched_probs
        ds_fp8 = _to_torch(dp.grad) if dp.grad is not None else None
        ds_cos = _cos_sim(ds_fp8, ds_gold) if ds_fp8 is not None else 0.0
        dw1_min_cos = min(_cos_sim(dw1_fp8[e], dw1_gold[e]) for e in range(E)
                          if dw1_gold[e].norm() > 0) if any(g.norm() > 0 for g in dw1_gold) else 0.0
        dw2_min_cos = min(_cos_sim(dw2_fp8[e], dw2_gold[e]) for e in range(E)
                          if dw2_gold[e].norm() > 0) if any(g.norm() > 0 for g in dw2_gold) else 0.0

        ok = (out_cos >= COS_THRESHOLD and dx_cos >= COS_THRESHOLD
              and ds_cos >= COS_THRESHOLD
              and dw1_min_cos >= COS_THRESHOLD and dw2_min_cos >= COS_THRESHOLD
              and dt < CUTE_COMPILE_THRESHOLD_S)
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"

        _print(f"{N:>6d} {topk_val:>2d} {desc:>15s} {dt:>5.1f}s  "
               f"{out_cos:.4f} {dx_cos:.4f} {ds_cos:.4f} {dw1_min_cos:.4f} {dw2_min_cos:.4f} {status:>6s}", 1)

    # ── Phase 4: Steady-state timing ──
    _print(f"\n[Phase 4] Steady-state timing (N=8192, 10 iters)...")
    di_ss, dp_ss, tpe_ss = _make_dispatch(8192, 8)
    grad_ss = paddle.randn([8192, H], dtype="bfloat16")
    # 2 warmup
    for _ in range(2):
        xt = paddle.randn([8192, H], dtype="bfloat16"); xt.stop_gradient = False
        with enable_fp8(True):
            _refresh_fp8_config()
            _ = node(xt, tpe_ss, di_ss, dp_ss)
        _.backward(grad_ss)
    flush_native_grads()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    N_BENCH = 10
    start.record()
    for _ in range(N_BENCH):
        xt = paddle.randn([8192, H], dtype="bfloat16"); xt.stop_gradient = False
        with enable_fp8(True):
            _refresh_fp8_config()
            o = node(xt, tpe_ss, di_ss, dp_ss)
        o.backward(grad_ss)
    end.record()
    torch.cuda.synchronize()
    flush_native_grads()
    cuda_us = start.elapsed_time(end) / N_BENCH * 1000
    _print(f"CUDA events: {cuda_us:.0f} µs/iter", 1)

    # ── Phase 5: nsys GPU-projection (optional) ──
    nsys_us = None
    if args.nsys:
        _print(f"\n[Phase 5] nsys GPU-projection (N=8192, 12 iters)...")
        nsys_dir = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/nsys"
        os.makedirs(nsys_dir, exist_ok=True)
        nsys_out = f"{nsys_dir}/s62_coldstart_verified"
        env = os.environ.copy()
        cmd = (f"nsys profile --trace=cuda,nvtx --sample=none --backtrace=none "
               f"--resolve-symbols=false --export=sqlite "
               f"--output={nsys_out} -f true "
               f"python tests/ops/bench_mlpnode_topk_nsys.py "
               f"--T 8192 --E 8 --I 1536 --warmup 8 --iters 12")
        result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(f"{nsys_out}.sqlite"):
            sys.path.insert(0, os.path.join(_REPO, "tests", "ops"))
            from bench_mlpnode_topk_nsys import gpu_projection_us
            nsys_us = gpu_projection_us(f"{nsys_out}.sqlite", 12)
            _print(f"GPU-projection: {nsys_us:.0f} µs/iter", 1)
        else:
            _print(f"nsys failed: {result.stderr[-200:]}", 1)

    # ── Summary ──
    peak = paddle.device.cuda.max_memory_allocated() / (1 << 20)
    total_s = time.perf_counter() - t_total

    _print(f"\n{'='*80}")
    _print("RESULTS")
    _print(f"{'='*80}")
    _print(f"Shapes tested:    {len(SHAPES)} ({sum(1 for _,_,_,w in SHAPES if w)} warmup + {sum(1 for _,_,_,w in SHAPES if not w)} cold)")
    _print(f"Precision:        all cos>{COS_THRESHOLD}, all pass" if all_pass else "Precision: FAIL")
    _print(f"Steady-state:     {cuda_us:.0f} µs/iter (CUDA events)")
    if nsys_us:
        _print(f"GPU-projection:   {nsys_us:.0f} µs/iter (nsys)")
    _print(f"Peak memory:      {peak:.0f} MiB")
    _print(f"Total time:       {total_s:.1f}s")
    _print(f"{'='*80}")
    _print(f"STATUS: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
