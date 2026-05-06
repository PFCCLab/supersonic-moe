#!/usr/bin/env python3
"""Phase 0.2: Audit iso32 vs 1×32 dz quantization on REAL gradient tensors.

For each captured ``dz_iter*.pt`` produced by ``tools/dump_real_dz.py``:

1. Direct quant fidelity (vs BF16 ref):
   - 1×32 row-wise blockscaled (production rowwise format)
   - 32×32 isotropic (the proposed iso32 epilogue output)
   metrics: cosine, RRMSE, max_abs, dyn-range bits-lost distribution

2. Downstream GEMM fidelity — these are the indicators that map to actual
   train-loss impact on the bwd path:
     dx_proxy  = dz @ w1ᵀ        (rowwise consumer)
     dw1_proxy = dzᵀ @ x          (colwise consumer)
   Computed three ways:
     (a) BF16 dz reference
     (b) 1×32 dq dz
     (c) 32×32 dq dz
   We then compare RRMSE of (b) and (c) vs (a). The 32×32/1×32 RRMSE ratio
   is the headline metric: ratio ≤ 2× means iso32 doesn't materially hurt
   gradient quality vs the production rowwise path.

Output: reports/iso32_dz_audit/audit.md and stdout summary.
"""

from __future__ import annotations

import glob
import math
import os
import sys

# Use stock torch directly — DO NOT enable the paddle torch-proxy here, since
# the audit relies on real torch ops (clamp_min, e4m3 cast, etc).
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from tests.ops.audit_iso32_numerics import (  # noqa: E402
    _quant_dequant_blockscaled, _per_row_dyn_range_loss,
)


def _metrics(ref: torch.Tensor, est: torch.Tensor) -> dict:
    ref_f = ref.float()
    est_f = est.float()
    d = ref_f - est_f
    cos = torch.nn.functional.cosine_similarity(
        ref_f.flatten().unsqueeze(0), est_f.flatten().unsqueeze(0)
    ).item()
    rrmse = (d.pow(2).sum().sqrt() /
             ref_f.pow(2).sum().sqrt().clamp_min(1e-30)).item()
    return {"cosine": cos, "rrmse": rrmse, "max_abs": d.abs().max().item()}


@torch.no_grad()
def audit_one(dz: torch.Tensor) -> dict:
    """All inputs/outputs on CUDA."""
    TK, twoI = dz.shape
    I = twoI // 2
    H = 3072  # Ernie hidden — used only for proxy GEMM downstream

    # 1) Direct quant fidelity
    dz_dq_1x32 = _quant_dequant_blockscaled(dz, row_tile=1)
    dz_dq_32x32 = _quant_dequant_blockscaled(dz, row_tile=32)
    direct_1 = _metrics(dz, dz_dq_1x32)
    direct_32 = _metrics(dz, dz_dq_32x32)
    dyn = _per_row_dyn_range_loss(dz)

    # 2) Downstream GEMM proxies
    #    dx ← dz @ w1ᵀ;   dz: (TK, 2I), w1: (2I, H) → dx: (TK, H)
    torch.manual_seed(0)
    w1 = (torch.randn(twoI, H, device=dz.device, dtype=torch.bfloat16) *
          (1.0 / math.sqrt(twoI))).contiguous()
    x = (torch.randn(TK, H, device=dz.device, dtype=torch.bfloat16) *
         0.02).contiguous()

    def _gemm_dx(d):
        return (d.float() @ w1.float())
    def _gemm_dw1(d):
        return (d.float().t() @ x.float())

    dx_ref = _gemm_dx(dz)
    dx_q1 = _gemm_dx(dz_dq_1x32)
    dx_q32 = _gemm_dx(dz_dq_32x32)
    dw1_ref = _gemm_dw1(dz)
    dw1_q1 = _gemm_dw1(dz_dq_1x32)
    dw1_q32 = _gemm_dw1(dz_dq_32x32)

    return {
        "shape": (TK, twoI),
        "direct_1x32": direct_1,
        "direct_32x32": direct_32,
        "dyn_range_loss": dyn,
        "dx_1x32_vs_ref": _metrics(dx_ref, dx_q1),
        "dx_32x32_vs_ref": _metrics(dx_ref, dx_q32),
        "dw1_1x32_vs_ref": _metrics(dw1_ref, dw1_q1),
        "dw1_32x32_vs_ref": _metrics(dw1_ref, dw1_q32),
    }


def _fmt_metrics(m: dict) -> str:
    return (f"cos={m['cosine']:.6f}  "
            f"RRMSE={m['rrmse']:.4e}  "
            f"max_abs={m['max_abs']:.4e}")


def _gate_summary(report: dict) -> tuple[bool, list[str]]:
    notes = []
    fails = []
    iso = report["direct_32x32"]
    one = report["direct_1x32"]
    if iso["cosine"] < 0.9995:
        fails.append(f"direct cos {iso['cosine']:.6f} < 0.9995")
    rrmse_ratio = iso["rrmse"] / max(one["rrmse"], 1e-30)
    if rrmse_ratio > 3.0:
        fails.append(f"direct RRMSE ratio {rrmse_ratio:.2f}× > 3×")
    notes.append(f"  direct RRMSE ratio (iso32/1x32) = {rrmse_ratio:.4f}×")
    dyn = report["dyn_range_loss"]
    if dyn:
        # Dyn-range loss is reported as **advisory** — for e4m3 the 3-bit
        # mantissa noise (~12.5% per value) dominates the per-tile amax
        # difference, so high frac>1b does NOT translate to downstream error.
        # The binding constraint is the downstream GEMM RRMSE ratio below.
        notes.append(f"  dyn-range frac>1b={dyn.get('frac>1b', 0):.4f} (advisory)")
    for label in ("dx", "dw1"):
        ratio = (report[f"{label}_32x32_vs_ref"]["rrmse"] /
                 max(report[f"{label}_1x32_vs_ref"]["rrmse"], 1e-30))
        notes.append(f"  {label} downstream RRMSE ratio (iso32/1x32) = {ratio:.4f}×")
        if ratio > 2.0:
            fails.append(f"{label} downstream RRMSE ratio {ratio:.2f}× > 2×")
    return (len(fails) == 0, fails + notes)


def main() -> None:
    out_dir = os.path.join(_REPO, "reports", "iso32_dz_audit")
    files = sorted(glob.glob(os.path.join(out_dir, "dz_*.npy")))
    if not files:
        print(f"ERROR: no dz captures in {out_dir}; run tools/dump_real_dz.py first")
        sys.exit(1)

    md = ["# iso32 dz precision audit (Phase 0.2)\n"]
    md.append("Pure-PyTorch quant→dequant comparison on **real** dz tensors "
              "captured from the bwd dGated path during Ernie-shape inference.\n")
    md.append("**Pass thresholds**: direct cos ≥ 0.9995; direct RRMSE iso32/1×32 ≤ 3×; "
              "downstream dx & dw1 RRMSE iso32/1×32 ≤ 2×.  Dyn-range bits-lost "
              "is reported as advisory only — for e4m3 the 3-bit mantissa noise "
              "dominates, so high frac>1b does **not** translate to downstream error.\n")

    all_pass = True
    for fp in files:
        print(f"\n=== {os.path.basename(fp)} ===")
        arr = np.load(fp)
        dz = torch.from_numpy(arr).to(device="cuda", dtype=torch.bfloat16).contiguous()

        rep = audit_one(dz)
        ok, lines = _gate_summary(rep)
        all_pass = all_pass and ok
        print(f"shape={rep['shape']}")
        print(f"  direct 1×32 :  {_fmt_metrics(rep['direct_1x32'])}")
        print(f"  direct 32×32:  {_fmt_metrics(rep['direct_32x32'])}")
        print(f"  dyn-range bits-lost: {rep['dyn_range_loss']}")
        print(f"  dx  1×32  vs ref:  {_fmt_metrics(rep['dx_1x32_vs_ref'])}")
        print(f"  dx  32×32 vs ref:  {_fmt_metrics(rep['dx_32x32_vs_ref'])}")
        print(f"  dw1 1×32  vs ref:  {_fmt_metrics(rep['dw1_1x32_vs_ref'])}")
        print(f"  dw1 32×32 vs ref:  {_fmt_metrics(rep['dw1_32x32_vs_ref'])}")
        for L in lines:
            print(L)
        print(f"  GATE: {'PASS' if ok else 'FAIL'}")

        md.append(f"\n## {os.path.basename(fp)}  shape={rep['shape']}\n")
        md.append(f"- direct 1×32 :  {_fmt_metrics(rep['direct_1x32'])}\n")
        md.append(f"- direct 32×32:  {_fmt_metrics(rep['direct_32x32'])}\n")
        md.append(f"- dyn-range bits-lost: {rep['dyn_range_loss']}\n")
        md.append(f"- dx  1×32  vs ref:  {_fmt_metrics(rep['dx_1x32_vs_ref'])}\n")
        md.append(f"- dx  32×32 vs ref:  {_fmt_metrics(rep['dx_32x32_vs_ref'])}\n")
        md.append(f"- dw1 1×32  vs ref:  {_fmt_metrics(rep['dw1_1x32_vs_ref'])}\n")
        md.append(f"- dw1 32×32 vs ref:  {_fmt_metrics(rep['dw1_32x32_vs_ref'])}\n")
        for L in lines:
            md.append(f"  - {L.strip()}\n")
        md.append(f"- **GATE**: {'✅ PASS' if ok else '❌ FAIL'}\n")

    md.append(f"\n# Overall: {'✅ PASS — proceed to Phase 1A' if all_pass else '❌ FAIL — re-evaluate'}\n")
    out_md = os.path.join(out_dir, "audit.md")
    with open(out_md, "w") as f:
        f.writelines(md)
    print(f"\nWrote {out_md}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
