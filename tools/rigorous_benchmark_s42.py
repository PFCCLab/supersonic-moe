#!/usr/bin/env python3
"""Rigorous benchmark: 3 modes × 3 seeds × 3 repeats, subprocess-isolated.

Measures: memory (peak alloc), precision (vs BF16), timing (CUDA events).
Each measurement in a fresh subprocess to avoid cross-contamination.

Usage:
    CUDA_VISIBLE_DEVICES=X python tools/rigorous_benchmark_s42.py --gpu 0
"""
import argparse, gc, json, os, subprocess, sys, tempfile, time
import numpy as np

PYTHON = sys.executable
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPE = dict(T=8192, H=3072, I=1536, E=8, K=8)
SEEDS = [42, 123, 777]
N_REPEATS = 3  # repeat each measurement
WARMUP = 5
TIMING_ITERS = 30

# ── Worker script (runs in subprocess) ─────────────────────────────────
WORKER = '''
import torch, gc, os, sys, json, numpy as np
sys.path.insert(0, os.environ["PROJECT"])
os.environ["USE_QUACK_GEMM"] = "1"
MODE = os.environ["MODE"]
if MODE != "bf16":
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
SEED = int(os.environ["SEED"])
TMPDIR = os.environ["TMPDIR"]
WARMUP = int(os.environ["WARMUP"])
TIMING_ITERS = int(os.environ["TIMING_ITERS"])
T, H, I, E, K = {T}, {H}, {I}, {E}, {K}

from sonicmoe import MoE
from sonicmoe.functional.utils import enable_quack_gemm, enable_fp8
from sonicmoe.enums import ActivationType

torch.manual_seed(SEED)
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).detach().requires_grad_()
dout = 0.02 * torch.randn_like(x)

use_fp8 = MODE != "bf16"
use_stash = MODE == "fp8_stash"

if use_fp8:
    moe.refresh_fp8_shadow_weights()
for _ in range(WARMUP):
    if use_stash:
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()
    ctx = enable_fp8() if use_fp8 else enable_quack_gemm(True)
    with ctx:
        out, _ = moe(x, use_fp8=use_fp8)
    out.backward(dout)
    if use_stash:
        moe.unstash_bf16()
    x.grad = None; moe.zero_grad(set_to_none=True)

if use_fp8:
    moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# ── Memory ──
if use_stash:
    moe.stash_bf16_to_cpu()
base = torch.cuda.memory_allocated() / (1024**2)
torch.cuda.reset_peak_memory_stats()
ctx = enable_fp8() if use_fp8 else enable_quack_gemm(True)
with ctx:
    out, _ = moe(x, use_fp8=use_fp8)
torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / (1024**2)
torch.cuda.reset_peak_memory_stats()
out.backward(dout); torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / (1024**2)
if use_stash:
    moe.unstash_bf16()
x.grad = None; moe.zero_grad(set_to_none=True)

# ── Precision ──
prec = {{}}
bf16_out_f = os.path.join(TMPDIR, f"bf16_out_{{SEED}}.npy")
bf16_dx_f = os.path.join(TMPDIR, f"bf16_dx_{{SEED}}.npy")
if MODE == "bf16":
    # Run a fresh forward+backward for precision reference
    with enable_quack_gemm(True):
        out_ref, _ = moe(x, use_fp8=False)
    out_ref.backward(dout)
    np.save(bf16_out_f, out_ref.detach().float().cpu().numpy())
    np.save(bf16_dx_f, x.grad.detach().float().cpu().numpy())
    x.grad = None; moe.zero_grad(set_to_none=True)
elif os.path.exists(bf16_out_f):
    if use_fp8:
        moe.refresh_fp8_shadow_weights()
    if use_stash:
        moe.stash_bf16_to_cpu()
    ctx = enable_fp8() if use_fp8 else enable_quack_gemm(True)
    with ctx:
        out_p, _ = moe(x, use_fp8=use_fp8)
    out_p.backward(dout)
    if use_stash:
        moe.unstash_bf16()
    ob = torch.from_numpy(np.load(bf16_out_f)).cuda()
    db = torch.from_numpy(np.load(bf16_dx_f)).cuda()
    rr = lambda a,b: ((a-b).norm()/b.norm().clamp(min=1e-8)).item()*100
    co = lambda a,b: torch.corrcoef(torch.stack([a.flatten(),b.flatten()]))[0,1].item()
    prec = dict(out_rrmse=round(rr(out_p.detach().float(),ob),2),
                out_corr=round(co(out_p.detach().float(),ob),4),
                dx_rrmse=round(rr(x.grad.detach().float(),db),2),
                dx_corr=round(co(x.grad.detach().float(),db),4))
    x.grad = None; moe.zero_grad(set_to_none=True)

# ── Timing ──
if use_fp8:
    moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
times = []
for _ in range(TIMING_ITERS):
    if use_stash:
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    ctx = enable_fp8() if use_fp8 else enable_quack_gemm(True)
    with ctx:
        out, _ = moe(x, use_fp8=use_fp8)
    out.backward(dout)
    e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
    if use_stash:
        moe.unstash_bf16()
    x.grad = None; moe.zero_grad(set_to_none=True)

# Trimmed mean (drop top/bottom 10%)
st = sorted(times)
lo, hi = int(len(st)*0.1), int(len(st)*0.9)
trimmed = st[lo:hi]

print("RESULT:" + json.dumps(dict(
    mode=MODE, seed=SEED,
    mem=dict(base=round(base,1), fwd_peak=round(fwd_peak,1), bwd_peak=round(bwd_peak,1)),
    timing=dict(trimmed_ms=round(sum(trimmed)/len(trimmed),3),
                min_ms=round(st[0],3), p50_ms=round(st[len(st)//2],3),
                max_ms=round(st[-1],3)),
    precision=prec,
    device=torch.cuda.get_device_name(0),
)))
'''.format(**SHAPE)


def run(mode, seed, tmpdir, gpu):
    env = {k: v for k, v in os.environ.items() if "FP8" not in k}
    env.update(PROJECT=PROJECT, MODE=mode, SEED=str(seed), TMPDIR=tmpdir,
               CUDA_VISIBLE_DEVICES=str(gpu), WARMUP=str(WARMUP),
               TIMING_ITERS=str(TIMING_ITERS))
    if mode != "bf16":
        env["SONIC_MOE_FP8_MODE"] = "perf"
    env["PYTHONPATH"] = PROJECT + ":" + env.get("PYTHONPATH", "")
    r = subprocess.run([PYTHON, "-c", WORKER], capture_output=True, text=True,
                       env=env, timeout=600)
    for line in r.stdout.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    raise RuntimeError(f"{mode}/s{seed} failed:\n{r.stderr[-500:]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="0")
    p.add_argument("--output", default=os.path.join(PROJECT, "benchmark_final.json"))
    args = p.parse_args()

    all_results = []
    tmp_root = os.environ.get("SONIC_MOE_PERSISTENT_TMP_ROOT")
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
        for repeat in range(N_REPEATS):
            print(f"\n{'='*70}")
            print(f"Repeat {repeat+1}/{N_REPEATS}")
            print(f"{'='*70}")
            for seed in SEEDS:
                for mode in ["bf16", "fp8", "fp8_stash"]:
                    t0 = time.time()
                    r = run(mode, seed, tmpdir, args.gpu)
                    r["repeat"] = repeat
                    all_results.append(r)
                    m = r["mem"]; t = r["timing"]
                    pr = r.get("precision", {})
                    prstr = f"  out={pr.get('out_rrmse','?')}% dx={pr.get('dx_rrmse','?')}%" if pr else ""
                    print(f"  {mode:12s} s{seed} fwd={m['fwd_peak']:>7.1f}M bwd={m['bwd_peak']:>7.1f}M "
                          f"t={t['trimmed_ms']:.2f}ms{prstr}  ({time.time()-t0:.0f}s)")

    report = dict(shape=SHAPE, seeds=SEEDS, n_repeats=N_REPEATS,
                  warmup=WARMUP, timing_iters=TIMING_ITERS,
                  results=all_results, device=all_results[0].get("device","?"))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.output}")

    # ── Summary ──
    print(f"\n{'='*90}")
    print("SUMMARY (mean ± std across 3 seeds × 3 repeats = 9 measurements)")
    print(f"{'='*90}")
    for mode in ["bf16", "fp8", "fp8_stash"]:
        entries = [r for r in all_results if r["mode"] == mode]
        fwd = [r["mem"]["fwd_peak"] for r in entries]
        bwd = [r["mem"]["bwd_peak"] for r in entries]
        base = [r["mem"]["base"] for r in entries]
        t = [r["timing"]["trimmed_ms"] for r in entries]
        print(f"\n  {mode}:")
        print(f"    base     = {np.mean(base):>7.1f} ± {np.std(base):.1f} MiB")
        print(f"    fwd_peak = {np.mean(fwd):>7.1f} ± {np.std(fwd):.1f} MiB")
        print(f"    bwd_peak = {np.mean(bwd):>7.1f} ± {np.std(bwd):.1f} MiB")
        print(f"    timing   = {np.mean(t):>7.3f} ± {np.std(t):.3f} ms (trimmed)")
        precs = [r["precision"] for r in entries if r.get("precision")]
        if precs:
            or_ = [p["out_rrmse"] for p in precs]
            dr_ = [p["dx_rrmse"] for p in precs]
            print(f"    out_rrmse= {np.mean(or_):>7.2f} ± {np.std(or_):.2f} %")
            print(f"    dx_rrmse = {np.mean(dr_):>7.2f} ± {np.std(dr_):.2f} %")


if __name__ == "__main__":
    main()
