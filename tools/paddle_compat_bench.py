#!/usr/bin/env python3
"""Paddle Compat Benchmark — SonicMoE under Paddle enable_compat().

Runs the same SonicMoE BF16/FP8 benchmark as cross_framework_bench.py but
under Paddle's torch-compat mode.  Compares results against the existing
reports/cross_framework_report.md (PyTorch native baseline).

Only runs SonicMoE paths (no ERNIE-core Paddle paths — those are unchanged).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/paddle_compat_bench.py
    CUDA_VISIBLE_DEVICES=0 python tools/paddle_compat_bench.py --skip-nsys   # precision+memory only
    CUDA_VISIBLE_DEVICES=0 python tools/paddle_compat_bench.py --shapes T8192_H3072_I1536_E8_K8
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "paddle_compat_bench"

# ── Environment ───────────────────────────────────────────────────────────────
EB_VENV_PYTHON = os.environ.get("SONIC_MOE_PADDLE_PYTHON", sys.executable)
QUACK_PATH = os.environ.get("SONIC_MOE_QUACK_PATH", "")
NSYS_BIN = shutil.which("nsys") or "/usr/local/bin/nsys"

SHAPES = {
    "T8192_H3072_I1536_E8_K8": {"S": 8192, "H": 3072, "I": 1536, "E": 8, "K": 8},
}

NSYS_WARMUP = 5
NSYS_ITERS = 20
NSYS_REPEATS = 3
SEEDS = [42, 123, 777]

# ── Reference data from cross_framework_report.md (PyTorch native) ────────────
PYTORCH_BASELINE = {
    "T8192_H3072_I1536_E8_K8": {
        "sonic_bf16": {"rrmse": 0.004691, "cosine": 0.999989, "peak_fwd_mib": 1753.3, "peak_bwd_mib": 2129.3},
        "sonic_fp8":  {"rrmse": 0.065298, "cosine": 0.997867, "peak_fwd_mib": 1696.4, "peak_bwd_mib": 2178.5},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation (same as cross_framework_bench.py)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(shape: dict, data_dir: str, seed: int = 42) -> None:
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    S, H, I, E, K = shape["S"], shape["H"], shape["I"], shape["E"], shape["K"]
    x = (rng.randn(S, H) * 0.02).astype(np.float32)
    np.save(os.path.join(data_dir, "x.npy"), x)
    for e in range(E):
        w1 = (rng.randn(H, 2 * I) * 0.02).astype(np.float32)
        w2 = (rng.randn(I, H) * 0.02).astype(np.float32)
        np.save(os.path.join(data_dir, f"w1_e{e}.npy"), w1)
        np.save(os.path.join(data_dir, f"w2_e{e}.npy"), w2)
    logits = rng.randn(S, E).astype(np.float32)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    topk_indices = np.argsort(-probs, axis=1)[:, :K].astype(np.int32)
    topk_logits = np.take_along_axis(logits, topk_indices, axis=1)
    exp_topk = np.exp(topk_logits - topk_logits.max(axis=1, keepdims=True))
    topk_scores = (exp_topk / exp_topk.sum(axis=1, keepdims=True)).astype(np.float32)
    np.save(os.path.join(data_dir, "topk_indices.npy"), topk_indices)
    np.save(os.path.join(data_dir, "topk_scores.npy"), topk_scores)


def compute_gold_fp64(data_dir: str, shape: dict) -> np.ndarray:
    S, H, I, E, K = shape["S"], shape["H"], shape["I"], shape["E"], shape["K"]
    x = np.load(os.path.join(data_dir, "x.npy")).astype(np.float64)
    ti = np.load(os.path.join(data_dir, "topk_indices.npy"))
    ts = np.load(os.path.join(data_dir, "topk_scores.npy")).astype(np.float64)
    w1s, w2s = [], []
    for e in range(E):
        w1s.append(np.load(os.path.join(data_dir, f"w1_e{e}.npy")).astype(np.float64))
        w2s.append(np.load(os.path.join(data_dir, f"w2_e{e}.npy")).astype(np.float64))

    output = np.zeros((S, H), dtype=np.float64)
    for t in range(S):
        for k in range(K):
            e = ti[t, k]
            s = ts[t, k]
            z = x[t] @ w1s[e]
            gate, up = z[:I], z[I:]
            silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
            y1 = silu_gate * up
            y2 = y1 @ w2s[e]
            output[t] += s * y2
    return output


def precision_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    diff = a64 - b64
    norm_b = np.linalg.norm(b64)
    return {
        "rrmse": float(np.linalg.norm(diff) / max(norm_b, 1e-30)),
        "cosine": float(np.dot(a64, b64) / max(np.linalg.norm(a64) * norm_b, 1e-30)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SonicMoE subprocess template (Paddle compat)
# ═══════════════════════════════════════════════════════════════════════════════

_SONIC_PADDLE_TEMPLATE = textwrap.dedent(r'''
import gc, json, os, sys
import numpy as np

# Paddle compat mode BEFORE any torch import
import paddle
paddle.enable_compat()

os.environ["USE_QUACK_GEMM"] = "1"
{fp8_env}
sys.path.insert(0, "{project_root}")

import torch, torch.nn.functional as F

data_dir, S, H, I, E, K = "{data_dir}", {S}, {H}, {I}, {E}, {K}
mode = "{mode}"
warmup, iters = {warmup}, {iters}
device = "cuda"
use_fp8 = {use_fp8}

def split_to_interleaved(w):
    h = w.shape[0] // 2
    o = torch.empty_like(w); o[0::2] = w[:h]; o[1::2] = w[h:]
    return o

if mode == "precision":
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

    output = o.float().cpu().numpy()
    output_name = "sonic_fp8_output.npy" if use_fp8 else "sonic_bf16_output.npy"
    np.save(os.path.join(data_dir, output_name), output)

    # Memory measurement
    MiB = 1048576
    _mem_dev = 0  # Paddle needs int device id
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(_mem_dev)
    baseline = torch.cuda.memory_allocated(_mem_dev) / MiB
    peak_fwd = torch.cuda.max_memory_allocated(_mem_dev) / MiB
    peak_bwd = peak_fwd  # single fwd, no bwd for precision run

    print("__RESULT__" + json.dumps({{
        "output_file": output_name,
        "baseline_mib": round(baseline, 1),
        "peak_fwd_mib": round(peak_fwd, 1),
    }}))

elif mode == "nsys":
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    import sonicmoe.functional as functional
    from sonicmoe.functional import moe_TC_softmax_topk_layer

    functional.clear_all_fp8_weight_caches()
    functional._ALIGNMENT_ASSUMED = False
    functional._ALIGNMENT_STREAK = 0

    paddle.seed(42)
    moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
              intermediate_size=I, activation_function=ActivationType.SWIGLU,
              add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)

    if use_fp8:
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()

    x = torch.from_numpy(np.load(os.path.join(data_dir, "x.npy"))).to(device=device, dtype=torch.bfloat16)
    w1_p = moe.c_fc.weight.permute(1, 2, 0)
    w2_p = moe.c_proj.weight.permute(1, 2, 0)

    def run_iter():
        xw = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True), enable_fp8(use_fp8):
            o, aux = moe(xw)
        return xw, o

    for _ in range(warmup):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True)
        del xw, o
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

    # Memory
    MiB = 1048576
    _mem_dev = 0
    torch.cuda.reset_peak_memory_stats(_mem_dev)
    xw, o = run_iter()
    torch.cuda.synchronize()
    peak_fwd = torch.cuda.max_memory_allocated(_mem_dev) / MiB
    torch.cuda.reset_peak_memory_stats(_mem_dev)
    o.sum().backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated(_mem_dev) / MiB
    moe.zero_grad(set_to_none=True); del xw, o
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # nsys capture
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True)
        del xw, o
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("__MEM__" + json.dumps({{
        "peak_fwd_mib": round(peak_fwd, 1),
        "peak_bwd_mib": round(peak_bwd, 1),
    }}))
    print("NSYS_DONE")
''')


# ═══════════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════════

def _make_env(gpu: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["USE_QUACK_GEMM"] = "1"
    env["PYTHONPATH"] = f"{QUACK_PATH}:{ROOT}"
    return env


def run_precision(label: str, shape: dict, seed: int, data_dir: str,
                  use_fp8: bool, gpu: int) -> dict:
    fp8_env = 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"' if use_fp8 else 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
    script = _SONIC_PADDLE_TEMPLATE.format(
        fp8_env=fp8_env, project_root=str(ROOT),
        data_dir=data_dir, mode="precision", warmup=0, iters=0,
        use_fp8=use_fp8, **shape,
    )
    env = _make_env(gpu)
    if use_fp8:
        env["SONIC_MOE_FP8_MODE"] = "perf"
    r = subprocess.run([EB_VENV_PYTHON, "-c", script],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        return {"error": r.stderr[-300:]}
    for line in r.stdout.split("\n"):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"error": "no __RESULT__ in stdout", "stdout": r.stdout[-200:]}


def nsys_gpu_projection(db_path: str, iters: int) -> float:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    conn.close()
    if not rows:
        return 0.0
    merged = 0
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged += ce - cs
            cs, ce = s, e
    merged += ce - cs
    return merged / 1000.0 / iters  # ns → µs/iter


def run_nsys(label: str, shape: dict, data_dir: str, use_fp8: bool, gpu: int) -> dict:
    fp8_env = 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"' if use_fp8 else 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
    script = _SONIC_PADDLE_TEMPLATE.format(
        fp8_env=fp8_env, project_root=str(ROOT),
        data_dir=data_dir, mode="nsys", warmup=NSYS_WARMUP, iters=NSYS_ITERS,
        use_fp8=use_fp8, **shape,
    )
    nsys_dir = REPORT_DIR / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H%M%S")
    prefix = nsys_dir / f"{label}_{ts}"
    sf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"pd_{label}_")
    sf.write(script); sf.close()
    env = _make_env(gpu)
    if use_fp8:
        env["SONIC_MOE_FP8_MODE"] = "perf"
    cmd = [NSYS_BIN, "profile", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
           f"--output={prefix}", "--export=sqlite", "--force-overwrite=true",
           EB_VENV_PYTHON, sf.name]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        os.unlink(sf.name)
        if p.returncode != 0:
            return {"error": p.stderr[-200:]}
        db = f"{prefix}.sqlite"
        if not os.path.exists(db):
            return {"error": "sqlite missing"}
        per_iter = nsys_gpu_projection(db, NSYS_ITERS)
        # Parse memory from stdout
        mem = {}
        for line in p.stdout.split("\n"):
            if line.startswith("__MEM__"):
                mem = json.loads(line[len("__MEM__"):])
        return {"per_iter_us": round(per_iter, 1), "nsys_rep": f"{prefix}.nsys-rep", **mem}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Paddle Compat Benchmark for SonicMoE")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES.keys()))
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--nsys-repeats", type=int, default=NSYS_REPEATS)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"paddle_compat_{ts}.json"

    print("=" * 70)
    print("  Paddle Compat Benchmark — SonicMoE BF16/FP8 under paddle.enable_compat()")
    print(f"  GPU: {args.gpu}  |  Shapes: {args.shapes}")
    print("=" * 70)

    results = {"timestamp": ts, "shapes": {}}
    base_tmp = tempfile.mkdtemp(prefix="pd_compat_")

    for sl in args.shapes:
        shape = SHAPES[sl]
        S = shape["S"]
        print(f"\n  ── {sl} ──")
        shape_result = {"shape": shape}

        # Phase 1: Precision
        print(f"  [precision] ", end="", flush=True)
        for seed in SEEDS:
            data_dir = os.path.join(base_tmp, sl, f"seed{seed}")
            generate_data(shape, data_dir, seed=seed)
            gold = compute_gold_fp64(data_dir, shape)

            for fp8, tag in [(False, "sonic_bf16_pd"), (True, "sonic_fp8_pd")]:
                r = run_precision(tag, shape, seed, data_dir, fp8, args.gpu)
                if "error" in r:
                    print(f"FAIL({tag}) ", end="", flush=True)
                    shape_result.setdefault(tag, {})["error"] = r["error"]
                    continue
                out_file = "sonic_fp8_output.npy" if fp8 else "sonic_bf16_output.npy"
                out = np.load(os.path.join(data_dir, out_file))
                m = precision_metrics(out, gold)
                shape_result.setdefault(tag, {}).setdefault("seeds", []).append(
                    {"seed": seed, **m, **r}
                )
                print(f"{tag[6:]}✓ ", end="", flush=True)
        print()

        # Aggregate precision
        for tag in ("sonic_bf16_pd", "sonic_fp8_pd"):
            seeds_data = shape_result.get(tag, {}).get("seeds", [])
            if seeds_data:
                avg_rrmse = sum(s["rrmse"] for s in seeds_data) / len(seeds_data)
                avg_cosine = sum(s["cosine"] for s in seeds_data) / len(seeds_data)
                shape_result[tag]["avg_rrmse"] = round(avg_rrmse, 6)
                shape_result[tag]["avg_cosine"] = round(avg_cosine, 6)

        # Phase 2: nsys performance + memory
        if not args.skip_nsys:
            for fp8, tag in [(False, "sonic_bf16_pd"), (True, "sonic_fp8_pd")]:
                data_dir = os.path.join(base_tmp, sl, "seed42")
                nsys_runs = []
                for rep in range(args.nsys_repeats):
                    print(f"  [nsys {tag} r{rep}] ", end="", flush=True)
                    nr = run_nsys(tag, shape, data_dir, fp8, args.gpu)
                    if "error" in nr:
                        print(f"FAIL ", flush=True)
                    else:
                        print(f"{nr['per_iter_us']:.0f}µs ", flush=True)
                        nsys_runs.append(nr)
                if nsys_runs:
                    median_us = sorted(r["per_iter_us"] for r in nsys_runs)[len(nsys_runs) // 2]
                    shape_result[tag]["nsys_median_us"] = round(median_us, 1)
                    shape_result[tag]["nsys_runs"] = nsys_runs
                    # Memory from last run
                    shape_result[tag]["peak_fwd_mib"] = nsys_runs[-1].get("peak_fwd_mib")
                    shape_result[tag]["peak_bwd_mib"] = nsys_runs[-1].get("peak_bwd_mib")

        results["shapes"][sl] = shape_result

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  Paddle Compat vs PyTorch Native (from cross_framework_report.md)")
    print("=" * 90)
    print(f"  {'Path':<22s} {'RRMSE':>8s} {'Cosine':>8s} {'nsys µs':>9s} {'Bwd MiB':>9s}  Ref RRMSE  Δ RRMSE")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*9} {'-'*9}  {'-'*9}  {'-'*8}")

    for sl in args.shapes:
        sr = results["shapes"].get(sl, {})
        baseline = PYTORCH_BASELINE.get(sl, {})
        for tag in ("sonic_bf16_pd", "sonic_fp8_pd"):
            ref_tag = tag.replace("_pd", "")
            ref = baseline.get(ref_tag, {})
            rr = sr.get(tag, {}).get("avg_rrmse", 0)
            co = sr.get(tag, {}).get("avg_cosine", 0)
            us = sr.get(tag, {}).get("nsys_median_us", 0)
            bwd = sr.get(tag, {}).get("peak_bwd_mib", 0)
            ref_rr = ref.get("rrmse", 0)
            delta_rr = rr - ref_rr if rr and ref_rr else 0
            print(f"  {tag:<22s} {rr:>7.6f}  {co:>7.6f}  {us:>8.1f}  {bwd or 0:>8.1f}"
                  f"  {ref_rr:>8.6f}  {delta_rr:>+7.6f}")

    # Write report
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Report: {report_path}")
    print("=" * 90)

    return results


if __name__ == "__main__":
    main()
