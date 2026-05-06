#!/usr/bin/env python3
"""MFU theory model: fit a parametric model to real benchmark data.

Model: busy_time = t_gemm + t_overhead
  t_gemm = 18*TK*H*I / (peak * eta)          # eta = GEMM efficiency (shape-dependent)
  t_overhead = a*TK*max(H,2I) + b*T*H*K + c  # quant + combine + fixed per-iter

eta is modeled as: eta_max * (1 - d / sqrt(TK))  # tile utilization degrades at small TK

So: busy_us = 18*TK*H*I / (peak_tflops*1e6 * eta_max*(1-d/sqrt(TK)))
              + a*TK*max(H,2I)*1e-9 + b*T*H*K*1e-9 + c

Parameters to fit: eta_max, d, a, b, c
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import curve_fit

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "reports", "fresh_benchmark_ws1", "sweep.json")

PEAK_FP8 = 4500.0   # TFLOPS
PEAK_BF16 = 2250.0  # TFLOPS


def mfu_model_fp8(X, eta_max, a_quant, a_expert, c_fixed):
    """Predict busy_us for FP8 mode.

    X = (TK, H, I, T, K, E) stacked as columns

    Model:
      busy = t_gemm + t_quant + t_expert_overhead + c_fixed
      t_gemm = 18*TK*H*I / (peak * eta_max)
      t_quant = a_quant * TK * max(H, 2I)    [FP8 quant: bytes processed]
      t_expert_overhead = a_expert * E * TK   [per-expert routing/launch cost]
      c_fixed = launch overhead (µs)
    """
    TK, H, I, T, K, E = X
    t_gemm = 18.0 * TK * H * I / (PEAK_FP8 * 1e6 * eta_max)
    t_quant = a_quant * TK * np.maximum(H, 2*I) * 1e-9
    t_expert = a_expert * E * TK * 1e-6
    return t_gemm + t_quant + t_expert + c_fixed


def mfu_model_bf16(X, eta_max, a_quant, a_expert, c_fixed):
    """Same structure for BF16."""
    TK, H, I, T, K, E = X
    t_gemm = 18.0 * TK * H * I / (PEAK_BF16 * 1e6 * eta_max)
    # BF16 has no FP8 quant, but still has combine + routing
    t_quant = a_quant * TK * np.maximum(H, 2*I) * 1e-9
    t_expert = a_expert * E * TK * 1e-6
    return t_gemm + t_quant + t_expert + c_fixed


def main():
    with open(_DATA) as f:
        data = json.load(f)

    # Separate FP8 and BF16
    fp8_data = [r for r in data if r["mode"] == "fp8" and r["busy_us"] is not None]
    bf16_data = [r for r in data if r["mode"] == "bf16" and r["busy_us"] is not None]

    print(f"FP8 data points: {len(fp8_data)}, BF16 data points: {len(bf16_data)}")

    # ── FP8 model fit ──
    X_fp8 = np.array([[r["TK"], r["H"], r["I"], r["T"], r["K"],
                       r["E"]] for r in fp8_data]).T
    y_fp8 = np.array([r["busy_us"] for r in fp8_data])

    p0_fp8 = [0.55, 0.5, 5.0, 100.0]  # initial guess: eta, a_quant, a_expert, c
    bounds = ([0.3, 0, 0, 0], [1.0, 50, 500, 1000])
    popt_fp8, pcov_fp8 = curve_fit(mfu_model_fp8, X_fp8, y_fp8, p0=p0_fp8, bounds=bounds)
    eta_max, a_quant, a_expert, c_fixed = popt_fp8
    y_pred_fp8 = mfu_model_fp8(X_fp8, *popt_fp8)
    ss_res = np.sum((y_fp8 - y_pred_fp8) ** 2)
    ss_tot = np.sum((y_fp8 - np.mean(y_fp8)) ** 2)
    r2_fp8 = 1 - ss_res / ss_tot

    print(f"\n{'='*60}")
    print(f"FP8 Model Parameters:")
    print(f"  eta_max  = {eta_max:.4f} (peak GEMM efficiency, MFU ceiling)")
    print(f"  a_quant  = {a_quant:.4f} (quant overhead: ns per byte of TK*max(H,2I))")
    print(f"  a_expert = {a_expert:.4f} (per-expert overhead: µs per E*TK unit)")
    print(f"  c_fixed  = {c_fixed:.2f} µs (fixed per-iter overhead)")
    print(f"  R² = {r2_fp8:.6f}")
    print(f"\nPer-shape breakdown:")
    print(f"{'Shape':<35} {'Measured':<10} {'Predicted':<10} {'Error':<8} {'t_gemm':<10} {'t_quant':<8} {'t_expert':<8}")
    for i, r in enumerate(fp8_data):
        shape = f"T{r['T']}-H{r['H']}-I{r['I']}-E{r['E']}"
        TK = r['TK']
        t_gemm_i = 18.0 * TK * r['H'] * r['I'] / (PEAK_FP8 * 1e6 * eta_max)
        t_quant_i = a_quant * TK * max(r['H'], 2*r['I']) * 1e-9
        t_expert_i = a_expert * r['E'] * TK * 1e-6
        err = (y_pred_fp8[i] - y_fp8[i]) / y_fp8[i] * 100
        print(f"{shape:<35} {y_fp8[i]:<10.1f} {y_pred_fp8[i]:<10.1f} {err:+.1f}%   {t_gemm_i:<10.1f} {t_quant_i:<8.1f} {t_expert_i:<8.1f}")

    # ── BF16 model fit ──
    X_bf16 = np.array([[r["TK"], r["H"], r["I"], r["T"], r["K"],
                        r["E"]] for r in bf16_data]).T
    y_bf16 = np.array([r["busy_us"] for r in bf16_data])

    p0_bf16 = [0.85, 0.1, 2.0, 30.0]
    popt_bf16, _ = curve_fit(mfu_model_bf16, X_bf16, y_bf16, p0=p0_bf16, bounds=bounds)
    y_pred_bf16 = mfu_model_bf16(X_bf16, *popt_bf16)
    ss_res_b = np.sum((y_bf16 - y_pred_bf16) ** 2)
    ss_tot_b = np.sum((y_bf16 - np.mean(y_bf16)) ** 2)
    r2_bf16 = 1 - ss_res_b / ss_tot_b

    print(f"\n{'='*60}")
    print(f"BF16 Model Parameters:")
    print(f"  eta_max  = {popt_bf16[0]:.4f}")
    print(f"  a_quant  = {popt_bf16[1]:.4f}")
    print(f"  a_expert = {popt_bf16[2]:.4f}")
    print(f"  c_fixed  = {popt_bf16[3]:.2f} µs")
    print(f"  R² = {r2_bf16:.6f}")

    # ── MFU formula derivation ──
    print(f"\n{'='*60}")
    print(f"Final MFU Formula (FP8, derived from fit):")
    print(f"")
    print(f"  busy_us = 18*TK*H*I / (4500e6 * {eta_max:.4f})")
    print(f"          + {a_quant:.4f}e-9 * TK * max(H, 2I)")
    print(f"          + {a_expert:.4f}e-6 * E * TK")
    print(f"          + {c_fixed:.1f}")
    print(f"")
    print(f"  MFU = 18*TK*H*I / (busy_us * 4500e6)")
    print(f"")
    print(f"  where TK = T * K_effective (broadcast ratio)")
    print(f"  K_effective = K in single-node bench, < K in EP>1 production")
    print(f"")
    print(f"  MFU asymptote (TK→∞, fixed H,I,E):")
    print(f"  lim MFU = 1 / (1/{eta_max:.4f} + peak*a_quant*max(H,2I)/(18*H*I) + peak*a_expert*E/(18*H*I))")
    # Calculate for Ernie shape
    H_e, I_e, E_e = 3072, 1536, 8
    mfu_lim = 1.0 / (1.0/eta_max + PEAK_FP8*1e6*a_quant*1e-9*max(H_e,2*I_e)/(18*H_e*I_e)
                      + PEAK_FP8*1e6*a_expert*1e-6*E_e/(18*H_e*I_e))
    print(f"  Ernie (H=3072,I=1536,E=8): MFU_∞ = {mfu_lim*100:.1f}%")
    H_w, I_w, E_w = 6144, 2048, 8
    mfu_lim_w = 1.0 / (1.0/eta_max + PEAK_FP8*1e6*a_quant*1e-9*max(H_w,2*I_w)/(18*H_w*I_w)
                        + PEAK_FP8*1e6*a_expert*1e-6*E_w/(18*H_w*I_w))
    print(f"  Wide (H=6144,I=2048,E=8):  MFU_∞ = {mfu_lim_w*100:.1f}%")

    # ── Predictions for new shapes ──
    print(f"\n{'='*60}")
    print(f"Predictions (extrapolation):")
    extra_shapes = [
        (32768, 3072, 1536, 8, 8, 32768),
        (8192, 8192, 4096, 8, 8, 8192),
        (16384, 4096, 2048, 8, 8, 16384),
    ]
    for T, H, I, E, K, _ in extra_shapes:
        TK = T * K
        X_new = np.array([[TK], [H], [I], [T], [K], [E]])
        pred = mfu_model_fp8(X_new, *popt_fp8)[0]
        mfu_pred = 18.0 * TK * H * I / (pred * PEAK_FP8 * 1e6)
        print(f"  T={T}, H={H}, I={I}, E={E}, K={K}: busy={pred:.0f} µs, MFU={mfu_pred*100:.1f}%")

    # Save model params
    model_out = {
        "fp8": {"eta_max": float(eta_max), "a_quant": float(a_quant),
                "a_expert": float(a_expert), "c_fixed": float(c_fixed), "r2": float(r2_fp8)},
        "bf16": {"eta_max": float(popt_bf16[0]), "a_quant": float(popt_bf16[1]),
                 "a_expert": float(popt_bf16[2]), "c_fixed": float(popt_bf16[3]), "r2": float(r2_bf16)},
        "peak_fp8_tflops": PEAK_FP8,
        "peak_bf16_tflops": PEAK_BF16,
    }
    out_path = os.path.join(_REPO, "reports", "fresh_benchmark_ws1", "mfu_model.json")
    with open(out_path, "w") as f:
        json.dump(model_out, f, indent=2)
    print(f"\nModel saved to: {out_path}")


if __name__ == "__main__":
    main()
