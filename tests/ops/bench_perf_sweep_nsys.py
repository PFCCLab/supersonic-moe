#!/usr/bin/env python3
"""8-GPU parallel nsys GPU-projection sweep for MFU model fitting.

Architecture:
  Main process: generates bench scripts + launches 8 GPU workers
  Each GPU worker: for each shape → warmup (in-process) → nsys profile (subprocess) → sqlite extract
  All 8 GPUs run truly in parallel (subprocess.Popen).

Uses NVTX BENCH range isolation for accurate GPU-projection.
"""
import os
import sys
import subprocess
import tempfile
import csv
import time
import json

_REPO = os.environ.get(
    "SONIC_MOE_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
_NSYS = os.environ.get("SONIC_MOE_NSYS", "nsys")
_PYTHON = os.environ.get("SONIC_MOE_PADDLE_PYTHON", sys.executable)

TOPK = 8
N_WARMUP = 12
N_BENCH_ITERS = 8

# (E, N_recv, H, I) — COMPLETE Cartesian grid gap-fill
# Sets: E={8,32,64,128}, HI={(3072,1536),(4096,2048),(4096,4096),(6144,3072)}, N={4K..512K}
# Only shapes NOT already in reports/perf_sweep_nsys_8gpu.csv
SWEEP_SHAPES = [
    # E=64 large TK (N=262144, 524288) × all HI
    (64, 262144, 3072, 1536), (64, 524288, 3072, 1536),
    (64, 262144, 4096, 2048), (64, 524288, 4096, 2048),
    (64, 262144, 4096, 4096), (64, 524288, 4096, 4096),
    # E=64 H=6144 I=3072 (all N)
    (64, 4096, 6144, 3072), (64, 8192, 6144, 3072),
    (64, 16384, 6144, 3072), (64, 32768, 6144, 3072),
    (64, 65536, 6144, 3072), (64, 131072, 6144, 3072),
    (64, 262144, 6144, 3072), (64, 524288, 6144, 3072),
    # E=128 large TK (N=262144, 524288) × H3072, H4096x2048
    (128, 262144, 3072, 1536), (128, 524288, 3072, 1536),
    (128, 262144, 4096, 2048), (128, 524288, 4096, 2048),
    # E=128 H=4096 I=4096 (all N — may OOM at large N)
    (128, 4096, 4096, 4096), (128, 8192, 4096, 4096),
    (128, 16384, 4096, 4096), (128, 32768, 4096, 4096),
    (128, 65536, 4096, 4096), (128, 131072, 4096, 4096),
    (128, 262144, 4096, 4096), (128, 524288, 4096, 4096),
    # E=128 H=6144 I=3072 (all N — may OOM at large N)
    (128, 4096, 6144, 3072), (128, 8192, 6144, 3072),
    (128, 16384, 6144, 3072), (128, 32768, 6144, 3072),
    (128, 65536, 6144, 3072), (128, 131072, 6144, 3072),
    (128, 262144, 6144, 3072), (128, 524288, 6144, 3072),
]

# INCREMENTAL MODE: append new results to existing CSV (never overwrite valid data)


def write_bench_script(path, E, N_recv_adj, H, I, tok_per_expert, gpu_id):
    with open(path, 'w') as f:
        f.write(f'''import os, sys, math
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-13.0/bin/ptxas"
sys.path.insert(0, "{_QUACK}")
sys.path.insert(0, "{_REPO}")
import torch
import paddle; paddle.enable_compat(); import torch.cuda
from sonicmoe.ernie_compat import SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads
import sonicmoe.functional as functional
functional._ALIGNMENT_ASSUMED = True; functional._ALIGNMENT_STREAK = 100
E, N_recv, H, I, TOPK = {E}, {N_recv_adj}, {H}, {I}, {TOPK}
tok_per_expert = {tok_per_expert}
class ME:
    def __init__(s,h,i,seed):
        paddle.seed(seed)
        s.up_gate_proj=type("P",(),dict(weight=paddle.randn([h,2*i],dtype="bfloat16")/math.sqrt(h)))()
        s.down_proj=type("P",(),dict(weight=paddle.randn([i,h],dtype="bfloat16")/math.sqrt(i)))()
        s.up_gate_proj.weight.stop_gradient=False
        s.down_proj.weight.stop_gradient=False
experts=[ME(H,I,e) for e in range(E)]
di=torch.zeros(N_recv,TOPK,dtype=torch.int32,device="cuda")
for i in range(N_recv):
    for k in range(TOPK): di[i,k]=(i*TOPK+k)%E
dp=torch.rand(N_recv,TOPK,device="cuda")*0.5+0.5
dp=(dp/dp.sum(dim=1,keepdim=True)).float()
tpe=[tok_per_expert]*E
paddle.seed(0)
x=torch.from_dlpack(paddle.randn([N_recv,H],dtype="bfloat16").detach()).to(device="cuda")*0.02
grad=torch.from_dlpack(paddle.randn([N_recv,H],dtype="bfloat16").detach()).to(device="cuda")*0.01
invalidate_weight_caches();functional.clear_all_fp8_weight_caches()
node=SonicMoEMlpNode(experts=experts,n_experts=E,hidden_size=H,intermediate_size=I)
dp_p=paddle.to_tensor(dp.cpu().numpy(),dtype="float32",place=paddle.CUDAPlace(0))
for _ in range(2):
    out=node.forward(x,tpe,dispatched_indices=di,dispatched_probs=dp_p);out.backward(grad)
flush_native_grads();torch.cuda.synchronize()
torch.cuda.nvtx.range_push("BENCH")
for _ in range({N_BENCH_ITERS}):
    out=node.forward(x,tpe,dispatched_indices=di,dispatched_probs=dp_p);out.backward(grad)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print("BENCH_DONE")
''')


def write_worker_script(path, gpu_id, tasks, output_dir):
    """Write a worker script that processes all shapes for one GPU."""
    with open(path, 'w') as f:
        f.write(f'''import os, sys, math, subprocess, json
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-13.0/bin/ptxas"
sys.path.insert(0, "{_QUACK}")
sys.path.insert(0, "{_REPO}")
sys.path.insert(0, os.path.join("{_REPO}", "tests", "ops"))

import torch
import paddle; paddle.enable_compat(); import torch.cuda
from sonicmoe.ernie_compat import SonicMoEMlpNode, invalidate_weight_caches, flush_native_grads
import sonicmoe.functional as functional
from bench_mlpnode_topk_nsys import gpu_projection_us
functional._ALIGNMENT_ASSUMED = True; functional._ALIGNMENT_STREAK = 100

TOPK = {TOPK}
tasks = {json.dumps(tasks)}
results = []

class ME:
    def __init__(s,h,i,seed):
        paddle.seed(seed)
        s.up_gate_proj=type("P",(),dict(weight=paddle.randn([h,2*i],dtype="bfloat16")/math.sqrt(h)))()
        s.down_proj=type("P",(),dict(weight=paddle.randn([i,h],dtype="bfloat16")/math.sqrt(i)))()
        s.up_gate_proj.weight.stop_gradient=False
        s.down_proj.weight.stop_gradient=False

for E, N_recv, H, I in tasks:
    tok_per_expert = ((N_recv * TOPK // E + 127) // 128) * 128
    TK = tok_per_expert * E
    N_recv_adj = TK // TOPK
    try:
        # Phase 1: warmup in THIS process (fills JIT cache)
        experts = [ME(H, I, e) for e in range(E)]
        di = torch.zeros(N_recv_adj, TOPK, dtype=torch.int32, device="cuda")
        for i in range(N_recv_adj):
            for k in range(TOPK): di[i, k] = (i * TOPK + k) % E
        dp = torch.rand(N_recv_adj, TOPK, device="cuda") * 0.5 + 0.5
        dp = (dp / dp.sum(dim=1, keepdim=True)).float()
        tpe = [tok_per_expert] * E
        paddle.seed(0)
        x = torch.from_dlpack(paddle.randn([N_recv_adj, H], dtype="bfloat16").detach()).to(device="cuda") * 0.02
        grad = torch.from_dlpack(paddle.randn([N_recv_adj, H], dtype="bfloat16").detach()).to(device="cuda") * 0.01
        invalidate_weight_caches(); functional.clear_all_fp8_weight_caches()
        node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)
        dp_p = paddle.to_tensor(dp.cpu().numpy(), dtype="float32", place=paddle.CUDAPlace(0))
        for _ in range({N_WARMUP}):
            out = node.forward(x, tpe, dispatched_indices=di, dispatched_probs=dp_p)
            out.backward(grad)
        flush_native_grads(); torch.cuda.synchronize()
        del experts, node, out, x, grad, di, dp, dp_p
        flush_native_grads(); torch.cuda.empty_cache()

        # Phase 2: nsys profile (subprocess, JIT cache warm)
        bench_path = os.path.join("{output_dir}", f"bench_E{{E}}_H{{H}}_I{{I}}_N{{N_recv}}_gpu{gpu_id}.py")
        nsys_out = os.path.join("{output_dir}", f"nsys_E{{E}}_H{{H}}_I{{I}}_N{{N_recv}}_gpu{gpu_id}")
        sqlite_path = nsys_out + ".sqlite"

        # bench script already generated by main process
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
        cmd = ["{_NSYS}", "profile", "--trace=cuda,nvtx", "--sample=none",
               "--backtrace=none", "--resolve-symbols=false", "-f", "true",
               f"--output={{nsys_out}}", "{_PYTHON}", bench_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if r.returncode != 0 or "BENCH_DONE" not in r.stdout:
            print(f"GPU{gpu_id} E={{E:>3}} H={{H:>4}} I={{I:>4}} TK={{TK:>10,}} NSYS_FAIL", flush=True)
            results.append(f"{{E}},{{TK}},{{N_recv_adj}},{{H}},{{I}},0,FAIL")
            continue

        # Export sqlite
        subprocess.run(["{_NSYS}", "export", "--type", "sqlite", "-f", "true",
                       "-o", sqlite_path, nsys_out + ".nsys-rep"],
                      capture_output=True, timeout=120, env=env)

        # Extract GPU-projection
        proj_us = gpu_projection_us(sqlite_path, {N_BENCH_ITERS})
        F = 18 * TK * H * I
        mfu = F / (proj_us * 1e-6) / 4500e12 * 100
        print(f"GPU{gpu_id} E={{E:>3}} H={{H:>4}} I={{I:>4}} TK={{TK:>10,}} proj={{proj_us:>10,.0f}}us MFU={{mfu:>5.1f}}%", flush=True)
        results.append(f"{{E}},{{TK}},{{N_recv_adj}},{{H}},{{I}},{{proj_us:.1f}},OK")

    except Exception as ex:
        print(f"GPU{gpu_id} E={{E:>3}} H={{H:>4}} I={{I:>4}} FAIL: {{str(ex)[:80]}}", flush=True)
        results.append(f"{{E}},0,0,{{H}},{{I}},0,FAIL")
        torch.cuda.empty_cache()

with open(os.path.join("{output_dir}", f"results_gpu{gpu_id}.csv"), "w") as f:
    f.write("E,TK,N_recv,H,I,gpu_projection_us,status\\n")
    for line in results:
        f.write(line + "\\n")
''')


def main():
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    output_dir = tempfile.mkdtemp(prefix="nsys_sweep_")
    n_gpus = 8

    print(f"nsys GPU-projection sweep: {len(SWEEP_SHAPES)} shapes, {n_gpus} GPUs")
    print(f"  Output dir: {output_dir}")

    # Distribute shapes round-robin
    gpu_tasks = [[] for _ in range(n_gpus)]
    for i, shape in enumerate(SWEEP_SHAPES):
        gpu_tasks[i % n_gpus].append(list(shape))

    # Phase 0: Generate ALL bench scripts from main process (no nesting issues)
    for i, (E, N_recv, H, I) in enumerate(SWEEP_SHAPES):
        gpu_id = i % n_gpus
        tok_per_expert = ((N_recv * TOPK // E + 127) // 128) * 128
        TK = tok_per_expert * E
        N_recv_adj = TK // TOPK
        bench_path = os.path.join(output_dir, f"bench_E{E}_H{H}_I{I}_N{N_recv}_gpu{gpu_id}.py")
        write_bench_script(bench_path, E, N_recv_adj, H, I, tok_per_expert, gpu_id)

    # Generate worker scripts
    procs = []
    for gpu_id in range(n_gpus):
        if not gpu_tasks[gpu_id]:
            continue
        worker_path = os.path.join(output_dir, f"worker_gpu{gpu_id}.py")
        write_worker_script(worker_path, gpu_id, gpu_tasks[gpu_id], output_dir)

        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PATH"] = "/opt/nsys/opt/nvidia/nsight-systems-cli/2026.2.1/bin:" + env.get("PATH", "")
        p = subprocess.Popen([_PYTHON, worker_path], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((gpu_id, p))
        print(f"  Launched GPU {gpu_id}: {len(gpu_tasks[gpu_id])} shapes")

    print(f"\nAll {len(procs)} workers launched. Waiting...")

    for gpu_id, p in procs:
        stdout, stderr = p.communicate(timeout=7200)
        for line in stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")

    # INCREMENTAL merge: read existing CSV, append new results, deduplicate
    csv_path = os.path.join(_REPO, "reports", "perf_sweep_nsys_8gpu.csv")
    existing_lines = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            header = next(f)
            for line in f:
                existing_lines.add(line.strip())
        print(f"  Existing data: {len(existing_lines)} rows")

    new_lines = set()
    for gpu_id in range(n_gpus):
        part = os.path.join(output_dir, f"results_gpu{gpu_id}.csv")
        if os.path.exists(part):
            with open(part) as f:
                next(f)  # skip header
                for line in f:
                    line = line.strip()
                    if line and ",OK" in line:
                        new_lines.add(line)

    all_lines = existing_lines | new_lines
    with open(csv_path, "w") as out_f:
        out_f.write("E,TK,N_recv,H,I,gpu_projection_us,status\n")
        for line in sorted(all_lines):
            out_f.write(line + "\n")
    print(f"  New: {len(new_lines)}, Total: {len(all_lines)}")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
