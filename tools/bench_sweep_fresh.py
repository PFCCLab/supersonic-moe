#!/usr/bin/env python3
"""Fresh benchmark sweep: FP8 frontier + BF16 baseline (8-GPU parallel).

Runs tests/ops/bench_mlpnode_topk_nsys.py via nsys on multiple GPUs,
extracts GPU-projection µs/iter, outputs results table.

Usage:
    source .runenv.sh
    python tools/bench_sweep_fresh.py --gpus 0,1,2,3,4,5,6,7
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCH = os.path.join(_REPO, "tests", "ops", "bench_mlpnode_topk_nsys.py")
_OUTPUT_DIR = os.path.join(_REPO, "reports", "fresh_benchmark_ws1")
_NSYS = "/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys"
_PYTHON = sys.executable

PEAK_TFLOPS = 4500.0  # Target GPU FP8 boost-clock empirical peak
PEAK_BF16_TFLOPS = 2250.0  # BF16 = FP8/2

SHAPES = [
    # (T, H, I, E, K)
    (1024, 3072, 1536, 8, 8),
    (2048, 3072, 1536, 8, 8),
    (4096, 3072, 1536, 8, 8),
    (8192, 3072, 1536, 8, 8),
    (16384, 3072, 1536, 8, 8),
    (8192, 3072, 1536, 16, 8),
    (8192, 3072, 1536, 32, 8),
    (4096, 4096, 2048, 8, 8),
    (8192, 4096, 2048, 8, 8),
    (8192, 4096, 4096, 8, 8),
    (8192, 6144, 2048, 8, 8),
]


def run_one(T, H, I, E, K, mode, gpu, warmup, iters):
    """Run nsys profile + extract for one shape/mode. Returns (tag, busy_us)."""
    shape_str = f"T{T}-H{H}-I{I}-E{E}-K{K}"
    tag = f"{mode}_{shape_str}"
    nsys_path = os.path.join(tempfile.gettempdir(), f"bench_{tag}_gpu{gpu}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = f"/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64:/usr/bin:/bin:{env.get('PATH', '')}"
    if mode == "bf16":
        env["SONIC_MOE_FP8_MODE"] = ""
        env["SONIC_MOE_FP8_WGRAD"] = "0"
        env.pop("SONIC_MOE_FP8_ASSUME_ALIGNED", None)
    else:
        env["SONIC_MOE_FP8_MODE"] = "perf"
        env["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
    env["USE_QUACK_GEMM"] = "1"

    # nsys profile
    cmd_profile = [
        _NSYS, "profile",
        "--trace=cuda,nvtx", "--sample=none", "--backtrace=none",
        "--resolve-symbols=false", "--force-overwrite=true",
        f"--output={nsys_path}",
        _PYTHON, _BENCH,
        "--T", str(T), "--H", str(H), "--I", str(I),
        "--E", str(E), "--topk", str(K),
        "--warmup", str(warmup), "--iters", str(iters),
        "--mode", mode,
    ]
    r = subprocess.run(cmd_profile, env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return tag, None, f"PROFILE FAILED: {r.stderr[-500:]}"

    # export sqlite
    sqlite_path = nsys_path + ".sqlite"
    cmd_export = [
        _NSYS, "export", "--type=sqlite", "--force-overwrite", "true",
        f"--output={sqlite_path}", nsys_path + ".nsys-rep",
    ]
    r = subprocess.run(cmd_export, env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return tag, None, f"EXPORT FAILED: {r.stderr[-300:]}"

    # extract GPU-projection
    cmd_extract = [_PYTHON, _BENCH, "--extract", sqlite_path, "--iters", str(iters)]
    r = subprocess.run(cmd_extract, env=env, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return tag, None, f"EXTRACT FAILED: {r.stderr[-300:]}"

    # parse
    for line in r.stdout.splitlines():
        if "GPU-projection" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if "µs/iter" in p or "us/iter" in p:
                    return tag, float(parts[i - 1]), None
            try:
                return tag, float(parts[1]), None
            except (ValueError, IndexError):
                pass

    return tag, None, f"PARSE FAILED: {r.stdout[-300:]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    max_parallel = len(gpus)

    # Build job list: all shapes × both modes
    jobs = []
    for T, H, I, E, K in SHAPES:
        for mode in ["fp8", "bf16"]:
            jobs.append((T, H, I, E, K, mode))

    print(f"Running {len(jobs)} benchmarks on {max_parallel} GPUs...")
    results = []
    gpu_idx = 0

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for job in jobs:
            T, H, I, E, K, mode = job
            gpu = gpus[gpu_idx % max_parallel]
            gpu_idx += 1
            f = executor.submit(run_one, T, H, I, E, K, mode, gpu, args.warmup, args.iters)
            futures[f] = job

        for f in as_completed(futures):
            job = futures[f]
            T, H, I, E, K, mode = job
            tag, us, err = f.result()
            if err:
                print(f"  [{tag}] ERROR: {err[:100]}")
            else:
                print(f"  [{tag}] GPU-projection: {us:.1f} µs/iter")
            TK = T * K
            peak = PEAK_TFLOPS if mode == "fp8" else PEAK_BF16_TFLOPS
            matmul_flops = 18.0 * TK * H * I
            mfu = matmul_flops / (us * 1e-6 * peak * 1e12) if us else None
            results.append({
                "T": T, "H": H, "I": I, "E": E, "K": K,
                "mode": mode, "busy_us": us, "mfu": mfu, "TK": TK,
                "error": err,
            })

    # Sort for display
    results.sort(key=lambda r: (r["mode"], r["T"], r["H"], r["E"]))

    # Output
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(_OUTPUT_DIR, "sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results table
    print(f"\n{'='*90}")
    print(f"{'Shape':<35} {'Mode':<5} {'busy µs':<10} {'MFU':<8} {'TFLOPS':<8}")
    print(f"{'-'*90}")
    for r in results:
        shape = f"T{r['T']}-H{r['H']}-I{r['I']}-E{r['E']}-K{r['K']}"
        us_str = f"{r['busy_us']:.1f}" if r['busy_us'] else "FAIL"
        peak = PEAK_TFLOPS if r['mode'] == 'fp8' else PEAK_BF16_TFLOPS
        mfu_str = f"{r['mfu']*100:.2f}%" if r['mfu'] else "—"
        tflops = f"{r['mfu'] * peak:.0f}" if r['mfu'] else "—"
        print(f"{shape:<35} {r['mode']:<5} {us_str:<10} {mfu_str:<8} {tflops:<8}")

    # Speedup comparison
    print(f"\n{'='*90}")
    print(f"{'Shape':<35} {'BF16 µs':<10} {'FP8 µs':<10} {'Speedup':<8}")
    print(f"{'-'*90}")
    by_shape = {}
    for r in results:
        key = (r['T'], r['H'], r['I'], r['E'], r['K'])
        by_shape.setdefault(key, {})[r['mode']] = r['busy_us']
    for key in sorted(by_shape.keys()):
        shape = f"T{key[0]}-H{key[1]}-I{key[2]}-E{key[3]}-K{key[4]}"
        bf16 = by_shape[key].get('bf16')
        fp8 = by_shape[key].get('fp8')
        if bf16 and fp8:
            print(f"{shape:<35} {bf16:<10.1f} {fp8:<10.1f} {bf16/fp8:.2f}x")
        else:
            bf16_s = f"{bf16:.1f}" if bf16 else "FAIL"
            fp8_s = f"{fp8:.1f}" if fp8 else "FAIL"
            print(f"{shape:<35} {bf16_s:<10} {fp8_s:<10} —")

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
