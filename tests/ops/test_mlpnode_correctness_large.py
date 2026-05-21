#!/usr/bin/env python
"""Large-SEQ correctness audit for SonicMoEMlpNode FP8 frontier.

Verifies the just-fixed deepep_topk_metadata bug regime (SEQ ≥ 8192, TK ≥ 65536):
  - no hang (subprocess timeout watchdog),
  - no IMA / numerical corruption,
  - output / dx / ds / dw1 / dw2 all match BF16 reference within tolerance.

Each shape runs in a subprocess with a hard timeout to detect deadlocks or hangs.
The child process imports gold/fp8 helpers from test_mlpnode_precision.

Run: CUDA_VISIBLE_DEVICES=2 python tests/ops/test_mlpnode_correctness_large.py
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

# ── venv switch (same as test_mlpnode_precision) ─────────────────────────────
_VENV = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv"
_PY = f"{_VENV}/bin/python"
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    print(f"\033[33mSwitch venv: {_VENV}\033[0m")
    os.execv(_PY, [_PY, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Cases include >= 8192 SEQ_LEN and the bug-regime (TK > 65536 with topk).
# Each tuple: (label, N_recv, topk, E, I, distribution)
#   distribution ∈ {"uniform", "skew80", "extreme_one", "tpe0_holes"}
CASES = [
    # Anchor (also in test_mlpnode_precision, sanity).
    ("anchor_N1024_K8_E8",   1024,  8,  8, 1536, "uniform"),
    # SEQ_LEN ≥ 8192 — S53 baseline shape.
    ("seq8K_K8_E8",          8192,  8,  8, 1536, "uniform"),
    ("seq8K_K8_E32",         8192,  8, 32, 1536, "uniform"),
    # SEQ_LEN = 16384 — exercises the just-fixed grid-cap bug regime.
    # With K=8, total dispatched = N*K = 131072 > 65536 (the broken cap).
    # With E=8, ~16384 tokens/expert ⇒ scatter_blocks ≈ 16384/32 = 512 per expert,
    # aggregate well over the 65536 row threshold.
    ("seq16K_K8_E8",        16384,  8,  8, 1536, "uniform"),
    ("seq16K_K8_E32",       16384,  8, 32, 1536, "uniform"),
    # 80% tokens to E0 — production-like skew.
    ("skew80_seq8K_E8",      8192,  8,  8, 1536, "skew80"),
    # All tokens routed to E0..K-1 only — 0 tokens for other experts.
    ("extreme_seq8K_E32",    8192,  8, 32, 1536, "extreme_one"),
    # Forced 0-token holes scattered across experts.
    ("holes_seq8K_E32",      8192,  8, 32, 1536, "tpe0_holes"),
    # Small/edge: zero-padded tail.
    ("tiny_N128_K4_E8",       128,  4,  8,  384, "uniform"),
]

CHILD_TIMEOUT_S = 600  # 10 min per shape — generous; real runs ≪ 60s.


# ────────────────────────────────────────────────────────────────────────────
# Child entrypoint: runs a single case and prints metrics + PASS/FAIL.
# ────────────────────────────────────────────────────────────────────────────
def child_run(case_name: str, N_recv: int, topk: int, E: int, I: int, dist: str):
    """Single-case worker. Imports inside the function so subprocess startup is fast."""
    import paddle
    paddle.enable_compat()
    import torch
    import torch.nn.functional as F
    import numpy as np

    from sonicmoe.enums import ActivationType
    from sonicmoe.ernie_compat import (
        SonicMoEMlpNode,
        invalidate_weight_caches,
    )
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True
    functional._ALIGNMENT_STREAK = 100

    # Reuse gold/fp8 helpers from the existing precision test.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_mlpnode_precision import (
        H, MockExpert, _silu, _dsilu, _cosine_rrmse, _zero_main_grads,
    )

    torch.manual_seed(2026)
    np.random.seed(2026)
    device = "cuda"

    # ── Build experts ────────────────────────────────────────────────────────
    experts = [MockExpert(H, I, e) for e in range(E)]

    # ── Build dispatched_indices / probs per distribution ────────────────────
    if dist == "uniform":
        raw = torch.randn(N_recv, E, device=device)
        _, top_e = raw.topk(topk, dim=-1)
        dispatched_indices = top_e.int()
    elif dist == "skew80":
        # 80% of tokens route their top-1 to E0; rest random
        raw = torch.randn(N_recv, E, device=device)
        hot = torch.rand(N_recv, device=device) < 0.8
        raw[hot, 0] += 100.0
        _, top_e = raw.topk(topk, dim=-1)
        dispatched_indices = top_e.int()
    elif dist == "extreme_one":
        # All tokens routed to experts 0..topk-1 (E1..E_topk-1 must still be valid)
        idx = torch.arange(topk, device=device, dtype=torch.int32)
        dispatched_indices = idx.unsqueeze(0).expand(N_recv, topk).contiguous()
    elif dist == "tpe0_holes":
        # Use only every other expert (odd ones get 0 tokens)
        avail = torch.tensor([e for e in range(0, E, 2)], device=device, dtype=torch.int32)
        # Sample topk without replacement from avail per token
        # (perm-based for simplicity; fall back to fewer if avail<topk)
        if avail.numel() < topk:
            # Pad with first available to fill topk slots; still hits >=1 hole
            pad = avail[:topk - avail.numel()]
            avail = torch.cat([avail, pad])
        rows = []
        for _ in range(N_recv):
            perm = torch.randperm(avail.numel(), device=device)[:topk]
            rows.append(avail[perm])
        dispatched_indices = torch.stack(rows, dim=0).int()
    else:
        raise ValueError(f"unknown dist {dist}")

    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = (dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)).float()

    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]
    TK = sum(tpe)
    print(f"  N_recv={N_recv} K={topk} E={E} I={I} TK={TK} tpe(min/max)={min(tpe)}/{max(tpe)}",
          flush=True)

    # ── Inputs (moderate scale to avoid FP8 saturation noise) ───────────────
    paddle.seed(42)
    x_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
    x = torch.from_dlpack(x_p.detach()).to(device=device)
    grad_out = torch.from_dlpack(grad_out_p.detach()).to(device=device)

    # ── BF16 gold (extended: also computes ds_gold) ──────────────────────────
    out_gold = torch.zeros(N_recv, H, dtype=x.dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = [torch.zeros(H, 2 * I, dtype=torch.float32, device=device) for _ in range(E)]
    dw2_gold = [torch.zeros(I, H, dtype=torch.float32, device=device) for _ in range(E)]
    ds_gold = torch.zeros_like(dispatched_probs)  # [N_recv, topk] fp32

    valid = dispatched_indices >= 0
    tok_idx = torch.arange(N_recv, dtype=torch.int32, device=device).unsqueeze(1).expand(N_recv, topk)
    slot_idx = torch.arange(topk, dtype=torch.int32, device=device).unsqueeze(0).expand(N_recv, topk)
    tok_flat = tok_idx[valid]
    slot_flat = slot_idx[valid]
    exp_flat = dispatched_indices[valid].long()
    scr_flat = dispatched_probs[valid].float()

    for e_idx in range(E):
        mask = exp_flat == e_idx
        if not mask.any():
            continue
        tok_ids = tok_flat[mask].long()
        slots = slot_flat[mask].long()
        scores = scr_flat[mask].unsqueeze(1)  # [count, 1]
        x_e = x[tok_ids]
        w_ug = torch.from_dlpack(experts[e_idx].up_gate_proj.weight.detach()).to(device=device, dtype=x.dtype)
        w_d = torch.from_dlpack(experts[e_idx].down_proj.weight.detach()).to(device=device, dtype=x.dtype)

        z = x_e @ w_ug
        gate = z[:, :I]
        up = z[:, I:]
        y1 = _silu(gate.float()).to(x.dtype) * up
        y2_unscaled = y1 @ w_d  # [count, H]  pre-scaled output of expert
        out_e = y2_unscaled * scores.to(x.dtype)
        out_gold.index_add_(0, tok_ids, out_e)

        grad_e_full = grad_out[tok_ids]  # [count, H] unscaled grad
        grad_e_scaled = grad_e_full * scores.to(x.dtype)
        dw2 = (y1.T @ grad_e_scaled).float()
        dy1 = grad_e_scaled @ w_d.T
        ds = _dsilu(gate.float())
        d_gate = dy1 * up * ds.to(x.dtype)
        d_up = dy1 * _silu(gate.float()).to(x.dtype)
        dz = torch.cat([d_gate, d_up], dim=-1)
        dw1 = (x_e.T @ dz).float()
        dx_e = dz @ w_ug.T
        dx_gold.index_add_(0, tok_ids, dx_e)
        dw1_gold[e_idx].add_(dw1)
        dw2_gold[e_idx].add_(dw2)

        # ds[i,k] = (grad_out[tok_id_i] * y2_unscaled[i, :]).sum()
        ds_e = (grad_e_full.float() * y2_unscaled.float()).sum(dim=1)  # [count]
        ds_gold[tok_ids, slots] = ds_e

    # ── FP8 frontier path ────────────────────────────────────────────────────
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Convert to paddle tensors with autograd for x and probs
    def _to_paddle(t):
        return paddle.utils.dlpack.from_dlpack(t.contiguous().detach().clone())

    # Warmup
    for _ in range(3):
        x_w = _to_paddle(x)
        x_w.stop_gradient = True
        di = _to_paddle(dispatched_indices)
        dp = _to_paddle(dispatched_probs)
        di.stop_gradient = True
        dp.stop_gradient = True
        out_w = node.forward(x_w, tpe, dispatched_indices=di, dispatched_probs=dp)
        out_w.backward(_to_paddle(grad_out))
    node.flush_grads()
    _zero_main_grads(experts)

    # Measured run with autograd on x and dispatched_probs
    x_in = _to_paddle(x)
    x_in.stop_gradient = False
    dp_in = _to_paddle(dispatched_probs)
    dp_in.stop_gradient = False
    di_in = _to_paddle(dispatched_indices)
    di_in.stop_gradient = True

    out = node.forward(x_in, tpe, dispatched_indices=di_in, dispatched_probs=dp_in)
    out.backward(_to_paddle(grad_out))
    node.flush_grads()

    out_fp8 = torch.from_dlpack(out.detach()).to(device=device, dtype=x.dtype)
    dx_fp8 = torch.from_dlpack(x_in.grad.detach()).to(device=device, dtype=x.dtype) if x_in.grad is not None else None
    ds_fp8 = torch.from_dlpack(dp_in.grad.detach()).to(device=device, dtype=torch.float32) if dp_in.grad is not None else None
    dw1_fp8 = []
    dw2_fp8 = []
    for exp in experts:
        dw1_fp8.append(torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).to(device=device, dtype=torch.float32))
        dw2_fp8.append(torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).to(device=device, dtype=torch.float32))

    # ── Sanity: no NaN/Inf ───────────────────────────────────────────────────
    def _nan_check(t, name):
        if t is None:
            return  # ds may be None for zero-prob shapes; treated as failure below
        bad = torch.isnan(t).any().item() or torch.isinf(t).any().item()
        assert not bad, f"NaN/Inf in {name}"

    _nan_check(out_fp8, "out")
    _nan_check(dx_fp8, "dx")
    _nan_check(ds_fp8, "ds")
    for i, t in enumerate(dw1_fp8):
        _nan_check(t, f"dw1[{i}]")
    for i, t in enumerate(dw2_fp8):
        _nan_check(t, f"dw2[{i}]")

    # ── Tolerance checks ─────────────────────────────────────────────────────
    tol_out = (0.99, 0.10)
    tol_dx = (0.99, 0.10)
    tol_ds = (0.99, 0.10)
    tol_dw1 = (0.97, 0.20)  # large-N accumulates more FP8 quant noise
    tol_dw2 = (0.97, 0.20)

    metrics = {}

    cos, rrmse = _cosine_rrmse(out_fp8, out_gold)
    metrics["out"] = (cos, rrmse)
    assert cos > tol_out[0] and rrmse < tol_out[1], f"out cos={cos:.4f} rrmse={rrmse:.4f}"

    assert dx_fp8 is not None, "dx is None"
    cos, rrmse = _cosine_rrmse(dx_fp8, dx_gold)
    metrics["dx"] = (cos, rrmse)
    assert cos > tol_dx[0] and rrmse < tol_dx[1], f"dx cos={cos:.4f} rrmse={rrmse:.4f}"

    assert ds_fp8 is not None, "ds is None (dispatched_probs.grad missing)"
    # ds has many zero rows for slots not picked by any expert? In our build
    # all topk slots are valid, so all entries should be filled by gold.
    cos, rrmse = _cosine_rrmse(ds_fp8, ds_gold)
    metrics["ds"] = (cos, rrmse)
    assert cos > tol_ds[0] and rrmse < tol_ds[1], f"ds cos={cos:.4f} rrmse={rrmse:.4f}"

    dw1_cos_min, dw1_rrmse_max = 1.0, 0.0
    dw2_cos_min, dw2_rrmse_max = 1.0, 0.0
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            # main_grad rows for empty experts must remain exactly 0 (FP8 wgrad
            # path must short-circuit). Use scalar reduction to avoid paddle's
            # multi-element bool ambiguity in compat mode.
            assert float(dw1_fp8[e_idx].float().abs().sum().item()) == 0.0, \
                f"dw1[{e_idx}] non-zero for empty expert"
            assert float(dw2_fp8[e_idx].float().abs().sum().item()) == 0.0, \
                f"dw2[{e_idx}] non-zero for empty expert"
            continue
        cos, rrmse = _cosine_rrmse(dw1_fp8[e_idx], dw1_gold[e_idx])
        dw1_cos_min = min(dw1_cos_min, cos); dw1_rrmse_max = max(dw1_rrmse_max, rrmse)
        assert cos > tol_dw1[0] and rrmse < tol_dw1[1], f"dw1[{e_idx}] cos={cos:.4f} rrmse={rrmse:.4f}"
        cos, rrmse = _cosine_rrmse(dw2_fp8[e_idx], dw2_gold[e_idx])
        dw2_cos_min = min(dw2_cos_min, cos); dw2_rrmse_max = max(dw2_rrmse_max, rrmse)
        assert cos > tol_dw2[0] and rrmse < tol_dw2[1], f"dw2[{e_idx}] cos={cos:.4f} rrmse={rrmse:.4f}"
    metrics["dw1"] = (dw1_cos_min, dw1_rrmse_max)
    metrics["dw2"] = (dw2_cos_min, dw2_rrmse_max)

    print(f"  RESULT: out cos={metrics['out'][0]:.4f} rrmse={metrics['out'][1]:.4f}")
    print(f"          dx  cos={metrics['dx'][0]:.4f} rrmse={metrics['dx'][1]:.4f}")
    print(f"          ds  cos={metrics['ds'][0]:.4f} rrmse={metrics['ds'][1]:.4f}")
    print(f"          dw1 cos≥{metrics['dw1'][0]:.4f} rrmse≤{metrics['dw1'][1]:.4f}")
    print(f"          dw2 cos≥{metrics['dw2'][0]:.4f} rrmse≤{metrics['dw2'][1]:.4f}")
    print(f"  PASS")


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("CUDA_VISIBLE_DEVICES", "2").split(",")[0]))
    parser.add_argument("--cases", type=str, default="all", help="comma-list of case names")
    parser.add_argument("--timeout", type=int, default=CHILD_TIMEOUT_S)
    args = parser.parse_args()

    if args.child:
        # In-process run for one case.
        case_map = {c[0]: c for c in CASES}
        spec = case_map[args.case]
        _, N, K, E, I, dist = spec
        child_run(spec[0], N, K, E, I, dist)
        return

    # Driver: spawn one subprocess per case with timeout.
    selected = CASES if args.cases == "all" else [c for c in CASES if c[0] in set(args.cases.split(","))]
    print(f"\nRunning {len(selected)} cases on GPU {args.gpu}, timeout={args.timeout}s/case\n")
    results = []
    for spec in selected:
        case_name = spec[0]
        print("=" * 76)
        print(f"CASE: {case_name}  (N={spec[1]} K={spec[2]} E={spec[3]} I={spec[4]} dist={spec[5]})")
        print("=" * 76)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        cmd = [_PY, __file__, "--child", "--case", case_name]
        try:
            proc = subprocess.run(cmd, env=env, timeout=args.timeout)
            ok = proc.returncode == 0
            results.append((case_name, "PASS" if ok else f"FAIL(rc={proc.returncode})"))
        except subprocess.TimeoutExpired:
            results.append((case_name, f"HANG(>{args.timeout}s)"))
            print(f"\n*** {case_name} TIMED OUT — likely deadlock/hang ***\n")

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    failed = 0
    for name, status in results:
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {name:30s} {status}")
        if status != "PASS":
            failed += 1
    print()
    if failed:
        print(f"FAILED: {failed}/{len(results)}")
        sys.exit(1)
    print(f"ALL {len(results)} CASES PASSED")


if __name__ == "__main__":
    main()
