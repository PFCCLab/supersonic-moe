#!/usr/bin/env python3
"""27-Shape Grid Paddle Compat Benchmark — 8-GPU parallel.

Mirrors introspect.py grid mode: 3T × 3E × 3I = 27 shapes.
Runs SonicMoE BF16 + FP8 under Paddle compat (paddle.enable_compat()),
then compares against PyTorch native baseline from grid_session53.

Execution plan:
  Phase 1: Precision — 27 shapes × 2 modes × 1 seed = 54 jobs → 8 GPUs (~4 min)
  Phase 2: nsys — 27 shapes × 2 modes × 3 repeats = 162 jobs → 8 GPUs (~30 min)

Usage:
    python tools/paddle_compat_parallel.py            # full grid
    python tools/paddle_compat_parallel.py --quick     # 9 shapes (T=8192 only), 1 nsys repeat
"""
from __future__ import annotations
import concurrent.futures
import json, os, shutil, sqlite3, subprocess, sys, tempfile, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PADDLE_PY = os.environ.get("SONIC_MOE_PADDLE_PYTHON", sys.executable)
QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
NSYS = shutil.which("nsys") or "/usr/local/bin/nsys"
REPORT_DIR = ROOT / "reports" / "paddle_compat"
BASELINE_PATH = ROOT / "reports" / "grid_session53" / "session53_grid_full.json"

GRID_T = [8192, 16384, 32768]
GRID_E = [8, 32, 128]
GRID_I = [1536, 2048, 3072]
H, K = 3072, 8
NSYS_WARMUP, NSYS_ITERS = 5, 20

def _all_shapes():
    return [{"S": T, "H": H, "I": I, "E": E, "K": K}
            for T in GRID_T for E in GRID_E for I in GRID_I]

def _shape_key(s):
    return f"T{s['S']}_I{s['I']}_E{s['E']}K{s['K']}"

# ═══════════════════════════════════════════════════════════════════════════════
# Worker template
# ═══════════════════════════════════════════════════════════════════════════════
WORKER = textwrap.dedent(r'''
import gc, json, os, sys, time as _time
import paddle; paddle.enable_compat()
os.environ["USE_QUACK_GEMM"] = "1"
{fp8_env}
sys.path.insert(0, "{root}")
import torch
from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
import sonicmoe.functional as functional

S, H, I, E, K = {S}, {H}, {I}, {E}, {K}
use_fp8 = {use_fp8}
warmup, iters = {warmup}, {iters}
mode = "{mode}"

functional.clear_all_fp8_weight_caches()
functional._ALIGNMENT_ASSUMED = False
functional._ALIGNMENT_STREAK = 0
paddle.seed(42)
device = "cuda"
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
if use_fp8:
    moe.refresh_fp8_shadow_weights()
    moe.stash_bf16_to_cpu()
x = (0.02 * torch.randn(S, H, device=device, dtype=torch.bfloat16))

def run_iter():
    xw = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True), enable_fp8(use_fp8):
        o, aux = moe(xw)
    return xw, o

# Warmup
for _ in range(warmup):
    xw, o = run_iter()
    o.sum().backward()
    moe.zero_grad(set_to_none=True)
    del xw, o
torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

if mode == "precision":
    # Single fwd — check output norm (sanity)
    xw, o = run_iter()
    print("__PREC__" + json.dumps({{
        "o_norm": round(o.float().norm().item(), 6),
        "o_shape": list(o.shape),
    }}))

elif mode == "nsys":
    # Memory
    _md = 0
    torch.cuda.reset_peak_memory_stats(_md)
    xw, o = run_iter(); torch.cuda.synchronize()
    pf = torch.cuda.max_memory_allocated(_md) / 1048576
    torch.cuda.reset_peak_memory_stats(_md)
    o.sum().backward(); torch.cuda.synchronize()
    pb = torch.cuda.max_memory_allocated(_md) / 1048576
    moe.zero_grad(set_to_none=True); del xw, o
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    # Capture
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True); del xw, o
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("__MEM__" + json.dumps({{"pf": round(pf,1), "pb": round(pb,1)}}))
    print("NSYS_DONE")
''')

# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════

def _env(gpu):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["USE_QUACK_GEMM"] = "1"
    e["PYTHONPATH"] = os.pathsep.join(
        str(p) for p in (QUACK, ROOT, e.get("PYTHONPATH", "")) if p
    )
    return e

def run_prec(gpu, shape, fp8):
    tag = "fp8" if fp8 else "bf16"
    sk = _shape_key(shape)
    fp8_env = 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"' if fp8 else 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
    script = WORKER.format(fp8_env=fp8_env, root=str(ROOT), mode="precision",
                           use_fp8=fp8, warmup=2, iters=0, **shape)
    t0 = time.time()
    r = subprocess.run([PADDLE_PY, "-c", script], capture_output=True, text=True,
                       env=_env(gpu), timeout=600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return {"sk": sk, "tag": tag, "gpu": gpu, "error": r.stderr[-150:], "elapsed": round(elapsed,1)}
    for line in r.stdout.split("\n"):
        if line.startswith("__PREC__"):
            d = json.loads(line[8:])
            return {"sk": sk, "tag": tag, "gpu": gpu, **d, "elapsed": round(elapsed,1)}
    return {"sk": sk, "tag": tag, "gpu": gpu, "error": "no output", "elapsed": round(elapsed,1)}

def run_nsys_one(gpu, shape, fp8, rep):
    tag = "fp8" if fp8 else "bf16"
    sk = _shape_key(shape)
    fp8_env = 'os.environ["SONIC_MOE_FP8_MODE"] = "perf"' if fp8 else 'os.environ["SONIC_MOE_FP8_MODE"] = "off"'
    script = WORKER.format(fp8_env=fp8_env, root=str(ROOT), mode="nsys",
                           use_fp8=fp8, warmup=NSYS_WARMUP, iters=NSYS_ITERS, **shape)
    nsys_dir = REPORT_DIR / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(nsys_dir / f"{sk}_{tag}_r{rep}")
    sf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    sf.write(script); sf.close()
    t0 = time.time()
    try:
        cmd = [NSYS, "profile", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               f"--output={prefix}", "--export=sqlite", "--force-overwrite=true", PADDLE_PY, sf.name]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(gpu))
        elapsed = time.time() - t0
        os.unlink(sf.name)
        if p.returncode != 0:
            return {"sk": sk, "tag": tag, "rep": rep, "gpu": gpu, "error": p.stderr[-120:], "elapsed": round(elapsed,1)}
        db = f"{prefix}.sqlite"
        if not os.path.exists(db):
            return {"sk": sk, "tag": tag, "rep": rep, "gpu": gpu, "error": "no sqlite", "elapsed": round(elapsed,1)}
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
        conn.close()
        if not rows:
            return {"sk": sk, "tag": tag, "rep": rep, "gpu": gpu, "error": "no kernels", "elapsed": round(elapsed,1)}
        merged, cs, ce = 0, rows[0][0], rows[0][1]
        for s, e in rows[1:]:
            if s <= ce: ce = max(ce, e)
            else: merged += ce - cs; cs, ce = s, e
        merged += ce - cs
        us = merged / 1000.0 / NSYS_ITERS
        mem = {}
        for line in p.stdout.split("\n"):
            if line.startswith("__MEM__"): mem = json.loads(line[7:])
        return {"sk": sk, "tag": tag, "rep": rep, "gpu": gpu, "us": round(us,1),
                "pf": mem.get("pf"), "pb": mem.get("pb"), "elapsed": round(elapsed,1)}
    except subprocess.TimeoutExpired:
        return {"sk": sk, "tag": tag, "rep": rep, "gpu": gpu, "error": "timeout", "elapsed": 600}
    except Exception as ex:
        return {"sk": sk, "tag": tag, "rep": rep, "gpu": gpu, "error": str(ex)[:100]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="9 shapes (T=8192), 1 nsys repeat")
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--nsys-repeats", type=int, default=3)
    args = parser.parse_args()

    shapes = _all_shapes()
    if args.quick:
        shapes = [s for s in shapes if s["S"] == 8192]
        args.nsys_repeats = 1

    nsys_reps = args.nsys_repeats
    ts = time.strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline
    baseline = {}
    if BASELINE_PATH.exists():
        bl = json.loads(BASELINE_PATH.read_text())
        baseline = bl.get("shapes", {})

    n_shapes = len(shapes)
    n_prec = n_shapes * 2
    n_nsys = n_shapes * 2 * nsys_reps if not args.skip_nsys else 0
    print("=" * 90)
    print(f"  27-Shape Grid Paddle Compat Benchmark — {n_shapes} shapes × 2 modes")
    print(f"  Precision: {n_prec} jobs | nsys: {n_nsys} jobs | 8 GPUs")
    print(f"  Timestamp: {ts}")
    print("=" * 90)

    results = {"timestamp": ts, "n_shapes": n_shapes, "nsys_reps": nsys_reps, "shapes": {}}

    # ── Phase 1: Precision ────────────────────────────────────────────────
    print(f"\n  [Phase 1] Precision: {n_prec} jobs ...", flush=True)
    prec_jobs = []
    gpu_rr = 0
    for s in shapes:
        for fp8 in (False, True):
            prec_jobs.append((gpu_rr % 8, s, fp8))
            gpu_rr += 1

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(run_prec, g, s, f): (g, s, f) for g, s, f in prec_jobs}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            done += 1
            sk, tag = r["sk"], r["tag"]
            results["shapes"].setdefault(sk, {})
            if "error" in r:
                results["shapes"][sk][f"{tag}_prec"] = {"error": r["error"]}
                print(f"    [{done}/{n_prec}] FAIL {sk}/{tag}: {r['error'][:60]}", flush=True)
            else:
                results["shapes"][sk][f"{tag}_prec"] = r
                print(f"    [{done}/{n_prec}] OK   {sk}/{tag} norm={r.get('o_norm','?')} ({r['elapsed']:.0f}s)", flush=True)
    print(f"  Phase 1 done in {time.time()-t0:.0f}s", flush=True)

    # ── Phase 2: nsys ─────────────────────────────────────────────────────
    if not args.skip_nsys:
        print(f"\n  [Phase 2] nsys: {n_nsys} jobs ({NSYS_WARMUP}w+{NSYS_ITERS}m×{nsys_reps}r) ...", flush=True)
        nsys_jobs = []
        gpu_rr = 0
        for s in shapes:
            for fp8 in (False, True):
                for rep in range(nsys_reps):
                    nsys_jobs.append((gpu_rr % 8, s, fp8, rep))
                    gpu_rr += 1

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(run_nsys_one, g, s, f, rep): (g, s, f, rep) for g, s, f, rep in nsys_jobs}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                done += 1
                sk, tag, rep = r["sk"], r["tag"], r.get("rep", "?")
                results["shapes"].setdefault(sk, {}).setdefault(f"{tag}_nsys", []).append(r)
                if "error" in r:
                    print(f"    [{done}/{n_nsys}] FAIL {sk}/{tag}/r{rep}: {r['error'][:50]}", flush=True)
                else:
                    print(f"    [{done}/{n_nsys}] OK   {sk}/{tag}/r{rep} {r['us']:.0f}µs ({r['elapsed']:.0f}s)", flush=True)
        print(f"  Phase 2 done in {time.time()-t0:.0f}s", flush=True)

    # ── Aggregate & compare ───────────────────────────────────────────────
    summary_rows = []
    for sk, sv in sorted(results["shapes"].items()):
        row = {"shape": sk}
        for tag in ("bf16", "fp8"):
            # nsys median
            runs = [r for r in sv.get(f"{tag}_nsys", []) if "us" in r]
            if runs:
                vals = sorted(r["us"] for r in runs)
                row[f"pd_{tag}_us"] = vals[len(vals) // 2]
                row[f"pd_{tag}_pf"] = runs[-1].get("pf")
                row[f"pd_{tag}_pb"] = runs[-1].get("pb")
            # Prec
            pr = sv.get(f"{tag}_prec", {})
            if "o_norm" in pr:
                row[f"pd_{tag}_norm"] = pr["o_norm"]
            elif "error" in pr:
                row[f"pd_{tag}_err"] = pr["error"][:30]
            # Baseline
            bl = baseline.get(sk, {})
            row[f"pt_{tag}_us"] = bl.get(tag, {}).get("per_iter_us")
        # Speedup
        for tag in ("bf16", "fp8"):
            pd = row.get(f"pd_{tag}_us")
            pt = row.get(f"pt_{tag}_us")
            if pd and pt:
                row[f"{tag}_overhead_pct"] = round((pd - pt) / pt * 100, 1)
        summary_rows.append(row)

    results["summary"] = summary_rows

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  RESULTS: Paddle Compat vs PyTorch Native (nsys GPU-projection µs/iter)")
    print("=" * 110)
    print(f"  {'Shape':<28s} {'PD BF16':>8s} {'PT BF16':>8s} {'Δ%':>6s}  "
          f"{'PD FP8':>8s} {'PT FP8':>8s} {'Δ%':>6s}  "
          f"{'PD Bwd':>7s} {'PT Bwd':>7s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6}  {'-'*8} {'-'*8} {'-'*6}  {'-'*7} {'-'*7}")

    for row in sorted(summary_rows, key=lambda r: r["shape"]):
        sk = row["shape"]
        pd_bf = row.get("pd_bf16_us", 0)
        pt_bf = row.get("pt_bf16_us", 0)
        d_bf = row.get("bf16_overhead_pct", "")
        pd_fp = row.get("pd_fp8_us", 0)
        pt_fp = row.get("pt_fp8_us", 0)
        d_fp = row.get("fp8_overhead_pct", "")
        pd_bwd = row.get("pd_fp8_pb", 0) or 0
        # Get PT bwd from baseline
        bl = baseline.get(sk, {})
        pt_bwd_fp8 = bl.get("memory_fp8", {}).get("peak_bwd_mib", 0) if bl else 0

        bf_str = f"{d_bf:>+5.1f}%" if isinstance(d_bf, (int, float)) else f"{'N/A':>6s}"
        fp_str = f"{d_fp:>+5.1f}%" if isinstance(d_fp, (int, float)) else f"{'N/A':>6s}"
        err_bf = row.get("pd_bf16_err", "")
        err_fp = row.get("pd_fp8_err", "")
        pd_bf_s = f"{pd_bf:>7.0f}" if pd_bf else (f"{'ERR':>7s}" if err_bf else f"{'--':>7s}")
        pd_fp_s = f"{pd_fp:>7.0f}" if pd_fp else (f"{'ERR':>7s}" if err_fp else f"{'--':>7s}")
        pt_bf_s = f"{pt_bf:>7.0f}" if pt_bf else f"{'--':>7s}"
        pt_fp_s = f"{pt_fp:>7.0f}" if pt_fp else f"{'--':>7s}"

        print(f"  {sk:<28s} {pd_bf_s}  {pt_bf_s} {bf_str}  "
              f"{pd_fp_s}  {pt_fp_s} {fp_str}  "
              f"{pd_bwd:>6.0f}  {pt_bwd_fp8:>6.0f}")

    # Stats
    overheads_bf = [r["bf16_overhead_pct"] for r in summary_rows if "bf16_overhead_pct" in r]
    overheads_fp = [r["fp8_overhead_pct"] for r in summary_rows if "fp8_overhead_pct" in r]
    if overheads_bf:
        import numpy as np
        print(f"\n  BF16 overhead: mean={np.mean(overheads_bf):+.1f}%, "
              f"median={np.median(overheads_bf):+.1f}%, "
              f"range=[{min(overheads_bf):+.1f}%, {max(overheads_bf):+.1f}%]")
    if overheads_fp:
        import numpy as np
        print(f"  FP8  overhead: mean={np.mean(overheads_fp):+.1f}%, "
              f"median={np.median(overheads_fp):+.1f}%, "
              f"range=[{min(overheads_fp):+.1f}%, {max(overheads_fp):+.1f}%]")

    # Precision sanity
    prec_ok = sum(1 for r in summary_rows if "pd_bf16_norm" in r)
    prec_fail = sum(1 for r in summary_rows if "pd_bf16_err" in r)
    print(f"\n  Precision: {prec_ok} OK, {prec_fail} FAIL (out of {n_shapes})")

    # Write
    report_path = REPORT_DIR / f"grid_paddle_compat_{ts}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Report: {report_path}")
    print("=" * 110)

if __name__ == "__main__":
    main()
