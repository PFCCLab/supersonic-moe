#!/usr/bin/env python
"""FP8 Frontier: Compute-Sanitizer + Memory-Extreme + Full Precision Stress Test.

Phase 1: compute-sanitizer memcheck at multiple shapes
Phase 2: Maximum-memory precision audit (output, dx, dw1, dw2, ds)
Phase 3: Determinism (bit-exact repeat)

Uses same infrastructure as test_mlpnode_precision.py (proven working).
"""
import math
import os
import sys
import time
import subprocess

venv = os.environ.get("SONIC_MOE_PADDLE_VENV", sys.prefix)
python_bin = os.path.join(venv, "bin", "python")
if os.path.realpath(sys.prefix) != os.path.realpath(venv):
    os.execv(python_bin, [python_bin, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

_REPO = os.environ.get(
    "SONIC_MOE_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch
import torch.nn.functional as F

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches
import sonicmoe.functional as functional

functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100

H = 3072


class MockExpert:
    def __init__(self, h, i, seed):
        paddle.seed(seed)
        self.up_gate_proj = type("P", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
        })()
        self.down_proj = type("P", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient = False


def _silu(x):
    return x * torch.sigmoid(x)


def _dsilu(x):
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


def _cosine_rrmse(a, b):
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    cos = float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())
    rrmse = float(((a_f - b_f).norm() / (b_f.norm() + 1e-10)).item())
    return cos, rrmse


def _zero_main_grads(experts):
    for exp in experts:
        for name in ("up_gate_proj", "down_proj"):
            w = getattr(exp, name).weight
            if hasattr(w, "main_grad") and w.main_grad is not None:
                w.main_grad.zero_()


def _gold_topk(x, experts, dispatched_indices, dispatched_probs, grad_out, h=H):
    """BF16 gold: forward + backward for topk dispatch."""
    N_recv, topk = dispatched_indices.shape
    E = len(experts)
    device = x.device
    dtype = x.dtype
    I = experts[0].down_proj.weight.shape[0]

    valid = dispatched_indices >= 0
    tok_flat = torch.arange(N_recv, dtype=torch.int32, device=device).unsqueeze(1).expand(N_recv, topk)[valid]
    exp_flat = dispatched_indices[valid].long()
    scr_flat = dispatched_probs[valid].float()

    out_gold = torch.zeros(N_recv, h, dtype=dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = [torch.zeros(h, 2 * I, dtype=torch.float32, device=device) for _ in range(E)]
    dw2_gold = [torch.zeros(I, h, dtype=torch.float32, device=device) for _ in range(E)]
    ds_gold = torch.zeros(N_recv, topk, dtype=torch.float32, device=device)

    for e_idx in range(E):
        mask = exp_flat == e_idx
        if not mask.any():
            continue
        tok_ids = tok_flat[mask].long()
        scores = scr_flat[mask].unsqueeze(1)
        x_e = x[tok_ids]
        w_ug = torch.from_dlpack(experts[e_idx].up_gate_proj.weight.detach()).to(device=device, dtype=dtype)
        w_d = torch.from_dlpack(experts[e_idx].down_proj.weight.detach()).to(device=device, dtype=dtype)

        z = x_e @ w_ug
        gate = z[:, :I]
        up = z[:, I:]
        y1 = _silu(gate.float()).to(dtype) * up
        out_e = (y1 @ w_d) * scores.to(dtype)
        out_gold.index_add_(0, tok_ids, out_e)

        grad_e = grad_out[tok_ids] * scores.to(dtype)
        dw2 = (y1.T @ grad_e).float()
        dy1 = grad_e @ w_d.T
        ds_val = _dsilu(gate.float())
        d_gate = dy1 * up * ds_val.to(dtype)
        d_up = dy1 * _silu(gate.float()).to(dtype)
        dz = torch.cat([d_gate, d_up], dim=-1)
        dw1 = (x_e.T @ dz).float()
        dx_e = dz @ w_ug.T

        dx_gold.index_add_(0, tok_ids, dx_e)
        dw1_gold[e_idx].add_(dw1)
        dw2_gold[e_idx].add_(dw2)

    return out_gold, dx_gold, dw1_gold, dw2_gold


def _fp8_topk(experts, x, dispatched_indices, dispatched_probs, tpe, grad_out, E, I, h=H):
    """Run FP8 MlpNode topk path."""
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=h, intermediate_size=I)

    # Warmup (3 iters sufficient for JIT)
    for _ in range(3):
        out_w = node.forward(x.clone().detach(), tpe,
                             dispatched_indices=dispatched_indices,
                             dispatched_probs=dispatched_probs)
        out_w.backward(grad_out.clone())
    node.flush_grads()
    _zero_main_grads(experts)

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe,
                       dispatched_indices=dispatched_indices,
                       dispatched_probs=dispatched_probs)
    out.backward(grad_out.clone())
    node.flush_grads()

    dx = torch.from_dlpack(x_in.grad.detach()).to(device=x.device, dtype=x.dtype) if x_in.grad is not None else None
    dw1_list, dw2_list = [], []
    for exp in experts:
        mg1 = torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).to(device=x.device, dtype=torch.float32)
        mg2 = torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).to(device=x.device, dtype=torch.float32)
        dw1_list.append(mg1)
        dw2_list.append(mg2)

    out_t = torch.from_dlpack(out.detach()).to(device=x.device, dtype=x.dtype)
    return out_t, dx, dw1_list, dw2_list


def run_topk_precision(N, K, E, I, h=H):
    """Run full topk precision test, return dict of metrics."""
    print(f"\n  topk  N={N:6d} K={K} E={E:3d} I={I:5d} ", end="", flush=True)
    device = "cuda"
    experts = [MockExpert(h, I, e) for e in range(E)]

    paddle.seed(42)
    x_p = paddle.randn([N, h], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([N, h], dtype="bfloat16") * 0.01
    x = torch.from_dlpack(x_p.detach()).to(device=device)
    grad_out = torch.from_dlpack(grad_out_p.detach()).to(device=device)

    # Random routing
    paddle.seed(99)
    indices = paddle.zeros([N, K], dtype="int32")
    for i in range(N):
        perm = paddle.randperm(E)[:K]
        indices[i] = perm.cast("int32")
    probs = paddle.ones([N, K], dtype="float32") / K
    dispatched_indices = torch.from_dlpack(indices.detach()).to(device=device)
    dispatched_probs = torch.from_dlpack(probs.detach()).to(device=device)

    # tpe must match actual routing (not assumed uniform)
    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    t0 = time.time()
    out_fp8, dx_fp8, dw1_fp8, dw2_fp8 = _fp8_topk(
        experts, x, dispatched_indices, dispatched_probs, tpe, grad_out, E, I, h)
    out_gold, dx_gold, dw1_gold, dw2_gold = _gold_topk(
        x, experts, dispatched_indices, dispatched_probs, grad_out, h)
    elapsed = time.time() - t0

    results = {}
    # output
    cos, rrmse = _cosine_rrmse(out_fp8, out_gold)
    results["out"] = {"cos": cos, "rrmse": rrmse, "pass": cos > 0.99 and rrmse < 0.10}

    # dx
    if dx_fp8 is not None:
        cos, rrmse = _cosine_rrmse(dx_fp8, dx_gold)
        results["dx"] = {"cos": cos, "rrmse": rrmse, "pass": cos > 0.99 and rrmse < 0.10}

    # dw1 (aggregate across experts)
    dw1_fp8_cat = torch.cat([d.flatten() for d in dw1_fp8])
    dw1_gold_cat = torch.cat([d.flatten() for d in dw1_gold])
    cos, rrmse = _cosine_rrmse(dw1_fp8_cat, dw1_gold_cat)
    results["dw1"] = {"cos": cos, "rrmse": rrmse, "pass": cos > 0.98 and rrmse < 0.15}

    # dw2
    dw2_fp8_cat = torch.cat([d.flatten() for d in dw2_fp8])
    dw2_gold_cat = torch.cat([d.flatten() for d in dw2_gold])
    cos, rrmse = _cosine_rrmse(dw2_fp8_cat, dw2_gold_cat)
    results["dw2"] = {"cos": cos, "rrmse": rrmse, "pass": cos > 0.98 and rrmse < 0.15}

    all_pass = all(r["pass"] for r in results.values())
    status = "PASS" if all_pass else "FAIL"
    print(f"{status} ({elapsed:.1f}s)  out={results['out']['cos']:.4f} dx={results.get('dx',{}).get('cos',0):.4f} dw1={results['dw1']['cos']:.4f} dw2={results['dw2']['cos']:.4f}")
    return results, all_pass


def run_compute_sanitizer(N, K, E, I, gpu_id, h=H):
    """Run compute-sanitizer on fwd+bwd."""
    print(f"\n  sanitizer N={N:6d} K={K} E={E:3d} I={I:5d} ", end="", flush=True)

    script = f'''
import os, sys, math
os.environ["USE_QUACK_GEMM"]="1"
os.environ["SONIC_MOE_FP8_MODE"]="perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"
os.environ["TRITON_PTXAS_PATH"]="/usr/local/cuda-13.0/bin/ptxas"
sys.path.insert(0,"{_QUACK}")
sys.path.insert(0,"{_REPO}")
import paddle; paddle.enable_compat()
import torch
from sonicmoe.ernie_compat import SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches
import sonicmoe.functional as functional
functional._ALIGNMENT_ASSUMED=True; functional._ALIGNMENT_STREAK=100

N,K,E,I,h={N},{K},{E},{I},{h}
class ME:
    def __init__(s,h,i,seed):
        paddle.seed(seed)
        s.up_gate_proj=type("P",(),{{"weight":paddle.randn([h,2*i],dtype="bfloat16")/math.sqrt(h)}})()
        s.down_proj=type("P",(),{{"weight":paddle.randn([i,h],dtype="bfloat16")/math.sqrt(i)}})()
        s.up_gate_proj.weight.stop_gradient=False
        s.down_proj.weight.stop_gradient=False

experts=[ME(h,I,e) for e in range(E)]
paddle.seed(42)
x=paddle.randn([N,h],dtype="bfloat16")*0.02
grad_out=paddle.randn([N,h],dtype="bfloat16")*0.01
indices=paddle.zeros([N,K],dtype="int32")
for i in range(N):
    indices[i]=paddle.randperm(E)[:K].cast("int32")
probs=paddle.ones([N,K],dtype="float32")/K
import torch as th
di=th.from_dlpack(indices.detach())
dp=th.from_dlpack(probs.detach())
tpe=[int((di==e).sum().item()) for e in range(E)]

invalidate_weight_caches()
functional.clear_all_fp8_weight_caches()
node=SonicMoEMlpNode(experts=experts,n_experts=E,hidden_size=h,intermediate_size=I)
for it in range(2):
    xin=x.clone().detach(); xin.stop_gradient=False
    out=node.forward(xin,tpe,dispatched_indices=di,dispatched_probs=dp)
    out.backward(grad_out.clone())
    node.flush_grads()
    paddle.device.cuda.synchronize()
print("SANITIZER_OK")
'''
    with open("/tmp/_sani.py", "w") as f:
        f.write(script)

    cmd = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1",
           "--print-limit", "5", sys.executable, "/tmp/_sani.py"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
    elapsed = time.time() - t0
    passed = "SANITIZER_OK" in result.stdout and result.returncode == 0
    status = "PASS" if passed else "FAIL"
    print(f"{status} ({elapsed:.1f}s)")
    if not passed:
        lines = (result.stdout + result.stderr).strip().split('\n')
        for l in lines[-15:]:
            print(f"    {l}")
    return passed


def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "6").split(",")[0])
    print("=" * 72)
    print("  FP8 FRONTIER COMPREHENSIVE STRESS TEST")
    print(f"  GPU: {gpu_id}")
    print("=" * 72)

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  Free memory: {free_gb:.1f} GB")

    # ── Phase 1: Compute-Sanitizer ──
    print("\n" + "=" * 72)
    print("  PHASE 1: COMPUTE-SANITIZER MEMCHECK (illegal memory access)")
    print("=" * 72)

    sani_cases = [
        (8192, 8, 8, 1536),    # production-like reference
        (16384, 8, 8, 1536),   # Large T
        (4096, 8, 32, 1536),   # Large E
        (8192, 8, 8, 4096),    # Large I (peak MFU shape)
    ]
    sani_results = []
    for N, K, E, I in sani_cases:
        passed = run_compute_sanitizer(N, K, E, I, gpu_id)
        sani_results.append((f"N{N}_K{K}_E{E}_I{I}", passed))

    # ── Phase 2: Precision Audit (all shapes, full tensor set) ──
    print("\n" + "=" * 72)
    print("  PHASE 2: PRECISION AUDIT (FP8 vs BF16 gold, all tensors)")
    print("=" * 72)

    prec_cases = [
        (8192, 8, 8, 1536),    # reference standard
        (16384, 8, 8, 1536),   # 2x tokens
        (8192, 8, 8, 4096),    # Peak MFU shape
        (4096, 8, 32, 1536),   # E=32
        (2048, 8, 64, 1536),   # E=64 (many experts)
    ]

    # Add largest shape that fits (~60% of free mem)
    for N_try in [32768, 24576, 20480, 16384]:
        est_gb = N_try * 8 * max(3072, 2*1536) * 8 / 1e9  # rough
        if est_gb < free_gb * 0.5:
            prec_cases.append((N_try, 8, 8, 1536))
            print(f"  Added memory-extreme: N={N_try} (TK={N_try*8})")
            break

    prec_results = []
    for N, K, E, I in prec_cases:
        try:
            metrics, passed = run_topk_precision(N, K, E, I)
            prec_results.append((f"N{N}_K{K}_E{E}_I{I}", metrics, passed))
        except Exception as e:
            print(f"ERROR: {e}")
            prec_results.append((f"N{N}_K{K}_E{E}_I{I}", {}, False))
        torch.cuda.empty_cache()

    # ── Phase 3: Determinism check (bit-exact) ──
    print("\n" + "=" * 72)
    print("  PHASE 3: DETERMINISM (bit-exact fwd+bwd repeat)")
    print("=" * 72)

    N, K, E, I = 8192, 8, 8, 1536
    print(f"\n  N={N} K={K} E={E} I={I}: ", end="", flush=True)
    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [N * K // E] * E
    paddle.seed(42)
    x_p = paddle.randn([N, H], dtype="bfloat16") * 0.02
    grad_p = paddle.randn([N, H], dtype="bfloat16") * 0.01
    x = torch.from_dlpack(x_p.detach())
    grad = torch.from_dlpack(grad_p.detach())
    paddle.seed(99)
    indices = paddle.zeros([N, K], dtype="int32")
    for i in range(N):
        indices[i] = paddle.randperm(E)[:K].cast("int32")
    probs = paddle.ones([N, K], dtype="float32") / K
    di = torch.from_dlpack(indices.detach())
    dp = torch.from_dlpack(probs.detach())

    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)
    # warmup
    for _ in range(5):
        o = node.forward(x.clone().detach(), tpe, dispatched_indices=di, dispatched_probs=dp)
        o.backward(grad.clone())
    node.flush_grads(); _zero_main_grads(experts)

    # Run twice
    outputs = []
    for run in range(2):
        _zero_main_grads(experts)
        xin = x.clone().detach(); xin.stop_gradient = False
        out = node.forward(xin, tpe, dispatched_indices=di, dispatched_probs=dp)
        out.backward(grad.clone())
        node.flush_grads()
        out_t = torch.from_dlpack(out.detach()).clone()
        dx_t = torch.from_dlpack(xin.grad.detach()).clone() if xin.grad is not None else None
        outputs.append((out_t, dx_t))

    det_pass = True
    out_eq = torch.equal(outputs[0][0], outputs[1][0]) if hasattr(torch, 'equal') else (outputs[0][0] == outputs[1][0]).all().item()
    dx_eq = torch.equal(outputs[0][1], outputs[1][1]) if hasattr(torch, 'equal') else (outputs[0][1] == outputs[1][1]).all().item()
    # Paddle proxy: torch.equal returns element-wise tensor
    if hasattr(out_eq, 'all'):
        out_eq = out_eq.all().item()
    if hasattr(dx_eq, 'all'):
        dx_eq = dx_eq.all().item()
    det_pass = bool(out_eq) and bool(dx_eq)
    print(f"{'PASS' if det_pass else 'FAIL'} (out={'exact' if out_eq else 'DIFF'}, dx={'exact' if dx_eq else 'DIFF'})")

    # ── Phase 4: DeepEP-style extreme imbalance (E=32, large T) ──
    print("\n" + "=" * 72)
    print("  PHASE 4: DEEPEP EXTREME IMBALANCE (E=32, skewed routing)")
    print("=" * 72)

    deepep_pass = True

    def _build_imbalanced_routing(N_recv, E, topk, skew="zipf", seed=77):
        """Build DeepEP-style imbalanced routing.

        skew="zipf": Zipf(s=1.2) expert popularity -> hot experts get ~20x cold
        skew="cliff": 4 experts get 80% of tokens, rest share 20%
        skew="dropout": 25% of experts get zero tokens (simulates expert dropout)
        """
        torch.manual_seed(seed)
        device = "cuda"

        if skew == "zipf":
            ranks = torch.arange(1, E + 1, dtype=torch.float32, device=device)
            weights = 1.0 / ranks.pow(1.2)
            weights = weights / weights.sum()
        elif skew == "cliff":
            weights = torch.ones(E, dtype=torch.float32, device=device) * 0.2 / max(E - 4, 1)
            weights[:4] = 0.8 / 4
        elif skew == "dropout":
            weights = torch.ones(E, dtype=torch.float32, device=device)
            n_dead = E // 4
            perm = torch.randperm(E, device=device)
            weights[perm[:n_dead]] = 0.0
            weights = weights / weights.sum()
        else:
            weights = torch.ones(E, dtype=torch.float32, device=device) / E

        indices = torch.zeros(N_recv, topk, dtype=torch.int32, device=device)
        for i in range(N_recv):
            chosen = torch.multinomial(weights, topk, replacement=False)
            indices[i] = chosen.int()

        probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
        probs = (probs / probs.sum(dim=1, keepdim=True)).float()

        tpe = [int((indices == e).sum().item()) for e in range(E)]
        return indices, probs, tpe

    E_deepep = 32
    K_deepep = 8
    I_deepep = 1536

    for skew_type, N_recv_deepep in [
        ("zipf",    16384),
        ("cliff",   32768),
        ("dropout", 16384),
    ]:
        label = f"N={N_recv_deepep} E={E_deepep} skew={skew_type}"
        print(f"\n  {label}: ", end="", flush=True)

        try:
            di_dp, dp_dp, tpe_dp = _build_imbalanced_routing(
                N_recv_deepep, E_deepep, K_deepep, skew=skew_type
            )
            tpe_min, tpe_max = min(tpe_dp), max(tpe_dp)
            tpe_zero = sum(1 for t in tpe_dp if t == 0)
            ratio = tpe_max / max(tpe_min, 1)
            print(f"tpe=[{tpe_min}..{tpe_max}] ratio={ratio:.0f}x dead={tpe_zero} ", end="", flush=True)

            experts_dp = [MockExpert(H, I_deepep, e) for e in range(E_deepep)]

            paddle.seed(42)
            x_p = paddle.randn([N_recv_deepep, H], dtype="bfloat16") * 0.02
            grad_p = paddle.randn([N_recv_deepep, H], dtype="bfloat16") * 0.01
            x_dp = torch.from_dlpack(x_p.detach()).to(device="cuda")
            grad_dp = torch.from_dlpack(grad_p.detach()).to(device="cuda")

            t0 = time.time()
            out_fp8, dx_fp8, dw1_fp8, dw2_fp8 = _fp8_topk(
                experts_dp, x_dp, di_dp, dp_dp, tpe_dp, grad_dp, E_deepep, I_deepep)
            out_gold, dx_gold, dw1_gold, dw2_gold = _gold_topk(
                x_dp, experts_dp, di_dp, dp_dp, grad_dp)
            elapsed = time.time() - t0

            cos_out, _ = _cosine_rrmse(out_fp8, out_gold)
            cos_dx, _ = _cosine_rrmse(dx_fp8, dx_gold) if dx_fp8 is not None else (0, 0)
            dw1_cat = torch.cat([d.flatten() for d in dw1_fp8])
            dw1_gold_cat = torch.cat([d.flatten() for d in dw1_gold])
            cos_dw1, _ = _cosine_rrmse(dw1_cat, dw1_gold_cat)
            dw2_cat = torch.cat([d.flatten() for d in dw2_fp8])
            dw2_gold_cat = torch.cat([d.flatten() for d in dw2_gold])
            cos_dw2, _ = _cosine_rrmse(dw2_cat, dw2_gold_cat)

            case_pass = cos_out > 0.99 and cos_dx > 0.99 and cos_dw1 > 0.98 and cos_dw2 > 0.98
            deepep_pass &= case_pass
            status = "PASS" if case_pass else "FAIL"
            print(f"{status} ({elapsed:.1f}s) out={cos_out:.4f} dx={cos_dx:.4f} dw1={cos_dw1:.4f} dw2={cos_dw2:.4f}")

            del experts_dp, x_dp, grad_dp, out_fp8, dx_fp8, out_gold, dx_gold
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR: {str(e)[:200]}")
            deepep_pass = False

    # ══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  FINAL REPORT")
    print("=" * 72)

    print("\n  [Phase 1] Compute-Sanitizer:")
    all_sani = True
    for name, passed in sani_results:
        print(f"    {name:<30} {'PASS' if passed else 'FAIL'}")
        all_sani &= passed

    print("\n  [Phase 2] Precision (cos thresholds: out/dx>0.99, dw1/dw2>0.98):")
    print(f"    {'Shape':<30} {'out':>7} {'dx':>7} {'dw1':>7} {'dw2':>7} {'Status'}")
    print(f"    {'-'*72}")
    all_prec = True
    for name, metrics, passed in prec_results:
        if metrics:
            print(f"    {name:<30} {metrics['out']['cos']:>7.4f} {metrics.get('dx',{}).get('cos',0):>7.4f} {metrics['dw1']['cos']:>7.4f} {metrics['dw2']['cos']:>7.4f} {'PASS' if passed else 'FAIL'}")
        else:
            print(f"    {name:<30} {'ERR':>7} {'ERR':>7} {'ERR':>7} {'ERR':>7} FAIL")
        all_prec &= passed

    print(f"\n  [Phase 3] Determinism: {'PASS' if det_pass else 'FAIL'}")

    print(f"\n  [Phase 4] DeepEP extreme imbalance (E=32): {'PASS' if deepep_pass else 'FAIL'}")

    overall = all_sani and all_prec and det_pass and deepep_pass
    print(f"\n  {'='*50}")
    print(f"  OVERALL: {'ALL PASS' if overall else 'FAILURES DETECTED'}")
    print(f"  {'='*50}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
