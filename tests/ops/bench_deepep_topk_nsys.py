#!/usr/bin/env python
"""nsys GPU-projection benchmark for SonicMoEMlpNode E2E.

Follows the introspect.py gold-standard pattern:
  1. Write workload to tempfile
  2. nsys profile --capture-range=cudaProfilerApi --export=sqlite
  3. Parse sqlite for GPU-projection

Usage:
  python tests/ops/bench_deepep_topk_nsys.py [--T 8192] [--E 8] [--gpu 0]
"""
import argparse, collections, json, os, sqlite3, subprocess, sys, tempfile, time, textwrap

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"

# ═══════════════════════════════════════════════════════════════════════════
# Workload template (runs inside nsys subprocess)
# ═══════════════════════════════════════════════════════════════════════════

_WORKLOAD = textwrap.dedent(r'''
import os, sys, gc, json
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
sys.path[:0] = {paths}

import paddle
paddle.compat.enable_torch_proxy(scope={{"sonicmoe","quack","triton"}}, silent=True)
import torch

from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches, _refresh_fp8_config
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.ernie_compat.mlp_node_v2 import (
    SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads)
import sonicmoe.ernie_compat.mlp_node_v2 as _m

class _FL:
    def __init__(self, w): self.weight = w
class _FE:
    def __init__(self, w1, w2):
        self.up_gate_proj = _FL(w1); self.down_proj = _FL(w2)

T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
paddle.seed(42)

experts = []
for _ in range(E):
    w1 = paddle.randn([H, 2*I], dtype="bfloat16") * 0.001
    w2 = paddle.randn([I, H], dtype="bfloat16") * 0.001
    w1.stop_gradient = False; w2.stop_gradient = False
    experts.append(_FE(w1, w2))

node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H,
                        intermediate_size=I, activation_type=ActivationType.SWIGLU)

x = paddle.randn([T, H], dtype="bfloat16")
out_grad = paddle.randn([T, H], dtype="bfloat16")

di = paddle.zeros([T, K], dtype="int32")
dp = paddle.full([T, K], 1.0/K, dtype="float32")
for i in range(T):
    di[i] = paddle.randperm(E)[:K].cast("int32")
tpe = paddle.bincount(di.reshape([-1]).cast("int64"), minlength=E).tolist()

# Warmup (forward-only)
invalidate_weight_caches(); clear_all_fp8_weight_caches()
for _ in range({warmup}):
    xw = paddle.randn_like(x); xw.stop_gradient = False
    with enable_fp8(True):
        _refresh_fp8_config()
        _ = node(xw, tpe, di, dp)
paddle.device.cuda.synchronize()

# Memory
MiB = 1 << 20
gc.collect(); paddle.device.cuda.empty_cache()
mem_pre = paddle.device.cuda.memory_allocated() / MiB

# Measured iterations (cudaProfilerStart/Stop bracket)
torch.cuda.cudart().cudaProfilerStart()
for _ in range({iters}):
    xt = paddle.randn_like(x); xt.stop_gradient = False
    invalidate_weight_caches()
    with enable_fp8(True):
        _refresh_fp8_config()
        ot = node(xt, tpe, di, dp)
    ot.backward(out_grad)
    flush_native_grads()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

peak = paddle.device.cuda.max_memory_allocated() / MiB
print("__MEM_JSON__" + json.dumps({{"pre": round(mem_pre,1), "peak": round(peak,1)}}))
print("NSYS_DONE", flush=True)
''')


# ═══════════════════════════════════════════════════════════════════════════
# sqlite parser (identical to introspect.py _nsys_parse_sqlite)
# ═══════════════════════════════════════════════════════════════════════════

def parse_sqlite(db_path, num_iters):
    conn = sqlite3.connect(db_path)

    string_map = {}
    try:
        for row in conn.execute("SELECT id, value FROM StringIds"):
            string_map[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass

    kernels = []
    try:
        for row in conn.execute(
            "SELECT start, end, demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            kernels.append((row[0], row[1], row[2], row[3]))
    except sqlite3.OperationalError:
        conn.close()
        return {"error": "No kernel data"}

    conn.close()
    if not kernels:
        return {"error": "No kernels"}

    # GPU-projection (merge overlapping intervals)
    kernels.sort(key=lambda x: x[0])
    merged_ns = 0
    cs, ce = kernels[0][0], kernels[0][1]
    for s, e, _, _ in kernels[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged_ns += ce - cs
            cs, ce = s, e
    merged_ns += ce - cs
    gpu_us = merged_ns / 1000.0

    # Per-kernel breakdown
    kstats = {}
    for s, e, did, sid in kernels:
        name = string_map.get(did, string_map.get(sid, f"unknown_{did}"))
        dur = (e - s) / 1000.0
        if name not in kstats:
            kstats[name] = {"us": 0.0, "n": 0}
        kstats[name]["us"] += dur
        kstats[name]["n"] += 1

    breakdown = []
    for name, st in sorted(kstats.items(), key=lambda x: -x[1]["us"]):
        breakdown.append({
            "name": name[:120],
            "total_us": round(st["us"], 1),
            "count": st["n"],
            "per_iter_us": round(st["us"] / num_iters, 1),
        })

    return {
        "gpu_projection_us": round(gpu_us, 1),
        "per_iter_us": round(gpu_us / num_iters, 1),
        "num_kernels": len(kernels),
        "kernels_per_iter": round(len(kernels) / num_iters, 1),
        "top_kernels": breakdown[:20],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--H", type=int, default=3072)
    p.add_argument("--I", type=int, default=1536)
    p.add_argument("--E", type=int, default=8)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()

    label = f"T{a.T}_H{a.H}_I{a.I}_E{a.E}_K{a.K}"
    out_dir = os.path.join(_REPO, "reports", "mlpnode_nsys")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%H%M%S")
    rep_path = os.path.join(out_dir, f"{label}_{ts}")

    # Write workload to tempfile (introspect.py pattern)
    script = _WORKLOAD.format(
        paths=[_QUACK, _REPO],
        T=a.T, H=a.H, I=a.I, E=a.E, K=a.K,
        warmup=a.warmup, iters=a.iters,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"nsys_mlpnode_{label}_"
    ) as f:
        f.write(script)
        script_path = f.name

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    nsys_bin = "/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"
    cmd = [
        nsys_bin, "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--backtrace=none",
        "--resolve-symbols=false",
        f"--output={rep_path}",
        "--force-overwrite=true",
        "--export=sqlite",
        sys.executable, script_path,
    ]

    print(f"Shape: T={a.T} H={a.H} I={a.I} E={a.E} K={a.K}")
    print(f"Running {a.warmup}w + {a.iters}m iters on GPU {a.gpu}...")
    print(f"  cmd: {' '.join(cmd[:6])} ...")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

    # Extract memory
    mem = {}
    for line in proc.stdout.splitlines():
        if line.startswith("__MEM_JSON__"):
            mem = json.loads(line[len("__MEM_JSON__"):])
        if "NSYS_DONE" in line:
            print("  Worker finished OK.")

    if proc.returncode != 0:
        print(f"FAILED (rc={proc.returncode})")
        print(proc.stderr[-1000:] if proc.stderr else "")
        os.unlink(script_path)
        sys.exit(1)

    os.unlink(script_path)

    # Parse sqlite
    rep_file = f"{rep_path}.nsys-rep"
    db_file = f"{rep_path}.sqlite"

    if not os.path.exists(db_file) and os.path.exists(rep_file):
        print(f"  Exporting sqlite from {rep_file} ...")
        subprocess.run([nsys_bin, "export", "--type=sqlite",
                        "--force-overwrite=true", rep_file],
                       capture_output=True, timeout=120)

    if not os.path.exists(db_file):
        print(f"nsys-rep: {rep_file}")
        print(f"No sqlite found. Open the .nsys-rep in Nsight Systems GUI for timeline.")
        sys.exit(0)

    perf = parse_sqlite(db_file, a.iters)

    # Report
    report = {"shape": label, "iters": a.iters, "memory": mem, **perf}

    print(f"\n{'='*60}")
    print(f"  SonicMoEMlpNode E2E — T={a.T} H={a.H} I={a.I} E={a.E} K={a.K}")
    print(f"{'='*60}")
    print(f"  GPU-projection:  {perf['per_iter_us']:.0f} µs/iter")
    print(f"  Kernels/iter:    {perf['kernels_per_iter']:.0f}")
    if mem:
        print(f"  Memory: pre={mem.get('pre',0):.0f}  peak={mem.get('peak',0):.0f} MiB")
    print(f"\n  Top kernels (per iter):")
    for k in perf.get("top_kernels", [])[:10]:
        print(f"    {k['per_iter_us']:8.1f} µs  x{k['count']//a.iters:3d}  {k['name'][:80]}")

    json_path = os.path.join(out_dir, f"{label}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  nsys-rep: {rep_path}.nsys-rep")
    print(f"  JSON:     {json_path}")


if __name__ == "__main__":
    main()
