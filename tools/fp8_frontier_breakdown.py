#!/usr/bin/env python3
"""FP8 Frontier rigorous breakdown: memory, precision, performance.

Usage:
    # Full analysis (subprocess-isolated)
    python tools/fp8_frontier_breakdown.py

    # With NCU for per-kernel GPU time (requires idle GPU)
    ncu --clock-control=none --kernel-name "regex:.*" \
        --launch-skip 5 --launch-count 50 \
        --metrics gpu__time_duration.sum \
        python tools/fp8_frontier_breakdown.py --ncu-mode
"""
import argparse
import os
import sys
import subprocess
import json
import tempfile


def run_in_subprocess(mode: str, gpu: int = 0) -> dict:
    """Run a single measurement in an isolated subprocess."""
    script = f'''
import os, sys, json, torch
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "{mode}"
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"

torch.manual_seed(42)
torch.cuda.manual_seed(42)

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

# reference shape
T, H, I, E, K = 8192, 3072, 1536, 8, 8
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

use_fp8 = "{mode}" != "bf16"

if use_fp8:
    from sonicmoe.functional import enable_fp8
    # Refresh FP8 weight caches
    moe.refresh_fp8_shadow_weights()
    if hasattr(moe, "stash_bf16_to_cpu") and "{mode}" == "fp8_stash":
        moe.stash_bf16_to_cpu()

# Warmup
for _ in range(3):
    torch.cuda.reset_peak_memory_stats()
    if use_fp8:
        with enable_fp8():
            out, aux = moe(x, use_fp8=True)
    else:
        out, aux = moe(x, use_fp8=False)
    out.backward(dout)
    if hasattr(moe, "unstash_bf16") and "{mode}" == "fp8_stash":
        moe.unstash_bf16()
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()
    x.grad = None
    for p in moe.parameters():
        p.grad = None

# Measurement run
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

if use_fp8:
    if "{mode}" == "fp8_stash":
        moe.stash_bf16_to_cpu()
    with enable_fp8():
        out, aux = moe(x, use_fp8=True)
    fwd_peak = torch.cuda.max_memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    out.backward(dout)
    bwd_peak = torch.cuda.max_memory_allocated() / (1024**2)
    if hasattr(moe, "unstash_bf16") and "{mode}" == "fp8_stash":
        moe.unstash_bf16()
else:
    out, aux = moe(x, use_fp8=False)
    fwd_peak = torch.cuda.max_memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    out.backward(dout)
    bwd_peak = torch.cuda.max_memory_allocated() / (1024**2)

# Precision: save tensors for comparison
result = dict(
    mode="{mode}",
    fwd_peak_mib=fwd_peak,
    bwd_peak_mib=bwd_peak,
)

# Save output + grads for precision comparison
torch.save(dict(
    output=out.detach().cpu(),
    dx=x.grad.detach().cpu() if x.grad is not None else None,
    dw1=moe.c_fc.weight.grad.detach().cpu() if moe.c_fc.weight.grad is not None else None,
    dw2=moe.c_proj.weight.grad.detach().cpu() if moe.c_proj.weight.grad is not None else None,
), "/tmp/fp8_breakdown_{mode}.pt")

print(json.dumps(result))
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        f.flush()
        try:
            proc = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                print(f"SUBPROCESS [{mode}] FAILED:\n{proc.stderr[-2000:]}", file=sys.stderr)
                return {"mode": mode, "error": proc.stderr[-500:]}
            # Find last JSON line
            for line in reversed(proc.stdout.strip().split('\n')):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            return {"mode": mode, "error": "no JSON output"}
        finally:
            os.unlink(f.name)


def precision_analysis():
    """Compare per-tensor precision between BF16 and FP8."""
    import torch

    bf16 = torch.load("/tmp/fp8_breakdown_bf16.pt", weights_only=False)
    fp8 = torch.load("/tmp/fp8_breakdown_fp8.pt", weights_only=False)
    fp8s = torch.load("/tmp/fp8_breakdown_fp8_stash.pt", weights_only=False) if os.path.exists("/tmp/fp8_breakdown_fp8_stash.pt") else None

    print("\n" + "=" * 70)
    print("  Per-Tensor Precision Analysis (FP8 vs BF16 gold)")
    print("=" * 70)

    def rrmse(a, b):
        if a is None or b is None:
            return float('nan')
        diff = (a.float() - b.float())
        return (diff.pow(2).mean().sqrt() / (b.float().pow(2).mean().sqrt() + 1e-12)).item()

    def cosine(a, b):
        if a is None or b is None:
            return float('nan')
        a, b = a.float().flatten(), b.float().flatten()
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def max_abs_err(a, b):
        if a is None or b is None:
            return float('nan')
        return (a.float() - b.float()).abs().max().item()

    tensors = [
        ("output", "output"),
        ("dx (input grad)", "dx"),
        ("dw1 (w1 grad)", "dw1"),
        ("dw2 (w2 grad)", "dw2"),
    ]

    print(f"\n{'Tensor':<20s}  {'RRMSE':>8s}  {'Cosine':>8s}  {'MaxAbsE':>10s}  {'Stash RRMSE':>12s}")
    print("-" * 70)
    for name, key in tensors:
        r = rrmse(fp8.get(key), bf16.get(key))
        c = cosine(fp8.get(key), bf16.get(key))
        m = max_abs_err(fp8.get(key), bf16.get(key))
        rs = rrmse(fp8s.get(key), bf16.get(key)) if fp8s else float('nan')
        print(f"  {name:<18s}  {r*100:>7.3f}%  {c:>8.6f}  {m:>10.6f}  {rs*100:>11.3f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-mode", action="store_true", help="Run in NCU-compatible mode (single process)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.ncu_mode:
        print("NCU mode: running single forward+backward for kernel profiling.")
        # Just run the FP8 path once for NCU to capture
        os.environ["USE_QUACK_GEMM"] = "1"
        os.environ["SONIC_MOE_FP8_MODE"] = "perf"
        import torch
        from sonicmoe import MoE
        from sonicmoe.enums import ActivationType
        from sonicmoe.functional import enable_fp8

        torch.manual_seed(42)
        moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072,
                   intermediate_size=1536, activation_function=ActivationType.SWIGLU,
                   add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)

        moe.refresh_fp8_shadow_weights()
        # Warmup
        for _ in range(3):
            with enable_fp8():
                out, aux = moe(x, use_fp8=True)
            out.backward(dout)
            x.grad = None
            for p in moe.parameters(): p.grad = None

        # Profiled run
        torch.cuda.nvtx.range_push("fp8_frontier")
        with enable_fp8():
            out, aux = moe(x, use_fp8=True)
        out.backward(dout)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print("NCU profiling run complete.")
        return

    print("=" * 70)
    print("  FP8 Frontier Rigorous Breakdown (subprocess-isolated)")
    print("=" * 70)

    # Run all three modes in separate subprocesses
    modes = ["bf16", "fp8", "fp8_stash"]
    results = {}
    for mode in modes:
        print(f"\nRunning {mode}...")
        r = run_in_subprocess(mode, args.gpu)
        results[mode] = r
        if "error" in r:
            print(f"  ERROR: {r['error'][:200]}")
        else:
            print(f"  fwd_peak={r['fwd_peak_mib']:.1f} MiB, bwd_peak={r['bwd_peak_mib']:.1f} MiB")

    # Memory table
    print("\n" + "=" * 70)
    print("  Memory Breakdown (peak allocated MiB)")
    print("=" * 70)
    print(f"\n{'Mode':<15s}  {'Fwd Peak':>10s}  {'Bwd Peak':>10s}")
    print("-" * 40)
    for mode in modes:
        r = results.get(mode, {})
        if "error" not in r:
            print(f"  {mode:<13s}  {r['fwd_peak_mib']:>9.1f}  {r['bwd_peak_mib']:>9.1f}")

    # Savings
    if all("error" not in results.get(m, {}) for m in modes):
        bf16_fwd = results["bf16"]["fwd_peak_mib"]
        bf16_bwd = results["bf16"]["bwd_peak_mib"]
        fp8_fwd = results["fp8"]["fwd_peak_mib"]
        fp8_bwd = results["fp8"]["bwd_peak_mib"]
        stash_fwd = results["fp8_stash"]["fwd_peak_mib"]
        stash_bwd = results["fp8_stash"]["bwd_peak_mib"]
        print(f"\n  FP8 vs BF16:       fwd {fp8_fwd - bf16_fwd:+.1f} MiB, bwd {fp8_bwd - bf16_bwd:+.1f} MiB")
        print(f"  FP8+stash vs BF16: fwd {stash_fwd - bf16_fwd:+.1f} MiB ({(stash_fwd-bf16_fwd)/bf16_fwd*100:+.1f}%), bwd {stash_bwd - bf16_bwd:+.1f} MiB ({(stash_bwd-bf16_bwd)/bf16_bwd*100:+.1f}%)")

    # Precision
    precision_analysis()

    print("\n" + "=" * 70)
    print("  NOTE: wall-clock times are unreliable on busy GPUs.")
    print("  Use NCU for kernel-level GPU time:")
    print("    ncu --clock-control=none --kernel-name 'regex:.*' \\")
    print("        --launch-skip 5 --launch-count 50 \\")
    print("        --metrics gpu__time_duration.sum \\")
    print("        python tools/fp8_frontier_breakdown.py --ncu-mode")
    print("=" * 70)


if __name__ == "__main__":
    main()
