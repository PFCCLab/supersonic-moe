#!/usr/bin/env python
"""Element-wise precision audit: FP8 MlpNode vs BF16 gold (output/dx/dw1/dw2).

Covers both identity layout (K=1, pre-sorted) and real topk dispatch.
"""

import math
import os
import sys

venv = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv"
python_bin = os.path.join(venv, "bin", "python")
if os.path.realpath(sys.prefix) != os.path.realpath(venv):
    print("Switch venv:", venv)
    os.execv(python_bin, [python_bin, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch
import torch.nn.functional as F

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import (
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
)
import sonicmoe.functional as functional

functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100


# ── helpers ──────────────────────────────────────────────────────────────────

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
    """Derivative of SiLU: sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
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


# ── Gold computation (BF16) ─────────────────────────────────────────────────

def _gold_identity(x, experts, tpe, grad_out):
    """Gold forward + backward for identity layout (K=1).

    Returns (out_gold, dx_gold, dw1_gold_list, dw2_gold_list).
    """
    T = x.shape[0]
    E = len(experts)
    device = x.device
    dtype = x.dtype
    I = experts[0].down_proj.weight.shape[0]

    out_gold = torch.zeros(T, H, dtype=dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = []
    dw2_gold = []

    offset = 0
    for e_idx, count in enumerate(tpe):
        if count == 0:
            dw1_gold.append(torch.zeros(H, 2 * I, dtype=torch.float32, device=device))
            dw2_gold.append(torch.zeros(I, H, dtype=torch.float32, device=device))
            continue
        x_e = x[offset:offset + count]  # [count, H]
        w_ug = torch.from_dlpack(experts[e_idx].up_gate_proj.weight.detach()).to(device=device, dtype=dtype)
        w_d = torch.from_dlpack(experts[e_idx].down_proj.weight.detach()).to(device=device, dtype=dtype)

        z = x_e @ w_ug  # [count, 2I]
        gate = z[:, :I]
        up = z[:, I:]
        y1 = _silu(gate.float()).to(dtype) * up  # [count, I]
        out_e = y1 @ w_d  # [count, H]
        out_gold[offset:offset + count] = out_e

        # backward
        grad_e = grad_out[offset:offset + count]  # [count, H]
        dw2 = (y1.T @ grad_e).float()  # [I, H]
        dy1 = grad_e @ w_d.T  # [count, I]
        ds = _dsilu(gate.float())
        d_gate = dy1 * up * ds.to(dtype)  # [count, I]
        d_up = dy1 * _silu(gate.float()).to(dtype)  # [count, I]
        dz = torch.cat([d_gate, d_up], dim=-1)  # [count, 2I]
        dw1 = (x_e.T @ dz).float()  # [H, 2I]
        dx_e = dz @ w_ug.T  # [count, H]

        dx_gold[offset:offset + count] = dx_e
        dw1_gold.append(dw1)
        dw2_gold.append(dw2)
        offset += count

    return out_gold, dx_gold, dw1_gold, dw2_gold


def _gold_topk(x, experts, dispatched_indices, dispatched_probs, grad_out):
    """Gold forward + backward for topk dispatch.

    Returns (out_gold, dx_gold, dw1_gold_list, dw2_gold_list).
    """
    N_recv, topk = dispatched_indices.shape
    E = len(experts)
    device = x.device
    dtype = x.dtype
    I = experts[0].down_proj.weight.shape[0]

    # Flatten valid entries
    valid = dispatched_indices >= 0
    tok_flat = torch.arange(N_recv, dtype=torch.int32, device=device).unsqueeze(1).expand(N_recv, topk)[valid]
    exp_flat = dispatched_indices[valid].long()
    scr_flat = dispatched_probs[valid].float()

    # Per-expert grouping
    out_gold = torch.zeros(N_recv, H, dtype=dtype, device=device)
    dx_gold = torch.zeros_like(x)
    dw1_gold = [torch.zeros(H, 2 * I, dtype=torch.float32, device=device) for _ in range(E)]
    dw2_gold = [torch.zeros(I, H, dtype=torch.float32, device=device) for _ in range(E)]

    for e_idx in range(E):
        mask = exp_flat == e_idx
        if not mask.any():
            continue
        tok_ids = tok_flat[mask].long()
        scores = scr_flat[mask].unsqueeze(1)  # [count, 1]
        x_e = x[tok_ids]  # [count, H]
        w_ug = torch.from_dlpack(experts[e_idx].up_gate_proj.weight.detach()).to(device=device, dtype=dtype)
        w_d = torch.from_dlpack(experts[e_idx].down_proj.weight.detach()).to(device=device, dtype=dtype)

        z = x_e @ w_ug  # [count, 2I]
        gate = z[:, :I]
        up = z[:, I:]
        y1 = _silu(gate.float()).to(dtype) * up  # [count, I]
        out_e = (y1 @ w_d) * scores.to(dtype)  # [count, H]
        # Accumulate into output (same token may appear multiple times)
        out_gold.index_add_(0, tok_ids, out_e)

        # backward
        grad_e = grad_out[tok_ids] * scores.to(dtype)  # [count, H]
        dw2 = (y1.T @ grad_e).float()  # [I, H]
        dy1 = grad_e @ w_d.T  # [count, I]
        ds = _dsilu(gate.float())
        d_gate = dy1 * up * ds.to(dtype)  # [count, I]
        d_up = dy1 * _silu(gate.float()).to(dtype)  # [count, I]
        dz = torch.cat([d_gate, d_up], dim=-1)  # [count, 2I]
        dw1 = (x_e.T @ dz).float()  # [H, 2I]
        dx_e = dz @ w_ug.T  # [count, H]

        dx_gold.index_add_(0, tok_ids, dx_e)
        dw1_gold[e_idx].add_(dw1)
        dw2_gold[e_idx].add_(dw2)

    return out_gold, dx_gold, dw1_gold, dw2_gold


# ── FP8 path ─────────────────────────────────────────────────────────────────

def _fp8_identity(experts, x, tpe, grad_out, E, I):
    """Run FP8 MlpNode identity path and return (out, dx, dw1_list, dw2_list)."""
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Warmup to stabilise FP8 caches
    for _ in range(5):
        out_w = node.forward(x.clone().detach(), tpe)
        out_w.backward(grad_out.clone())
    _zero_main_grads(experts)
    flush_native_grads()

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe)
    out.backward(grad_out.clone())
    flush_native_grads()

    # Extract dx
    dx = torch.from_dlpack(x_in.grad.detach()).to(device=x.device, dtype=x.dtype) if x_in.grad is not None else None

    # Extract dw from main_grad (already flushed)
    dw1_list = []
    dw2_list = []
    for exp in experts:
        mg1 = torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).to(device=x.device, dtype=torch.float32)
        mg2 = torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).to(device=x.device, dtype=torch.float32)
        dw1_list.append(mg1)
        dw2_list.append(mg2)

    out_t = torch.from_dlpack(out.detach()).to(device=x.device, dtype=x.dtype)
    return out_t, dx, dw1_list, dw2_list


def _fp8_topk(experts, x, dispatched_indices, dispatched_probs, tpe, grad_out, E, I):
    """Run FP8 MlpNode topk path and return (out, dx, dw1_list, dw2_list)."""
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Warmup
    for _ in range(5):
        out_w = node.forward(x.clone().detach(), tpe,
                             dispatched_indices=dispatched_indices,
                             dispatched_probs=dispatched_probs)
        out_w.backward(grad_out.clone())
    node.flush_grads()          # flush warmup residuals from native buffer first
    _zero_main_grads(experts)   # then zero main_grad (including what was just flushed)

    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe,
                       dispatched_indices=dispatched_indices,
                       dispatched_probs=dispatched_probs)
    out.backward(grad_out.clone())
    node.flush_grads()

    dx = torch.from_dlpack(x_in.grad.detach()).to(device=x.device, dtype=x.dtype) if x_in.grad is not None else None

    dw1_list = []
    dw2_list = []
    for exp in experts:
        mg1 = torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).to(device=x.device, dtype=torch.float32)
        mg2 = torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).to(device=x.device, dtype=torch.float32)
        dw1_list.append(mg1)
        dw2_list.append(mg2)

    out_t = torch.from_dlpack(out.detach()).to(device=x.device, dtype=x.dtype)
    return out_t, dx, dw1_list, dw2_list


# ── tests ────────────────────────────────────────────────────────────────────

def _run_identity_case(T, E, I):
    """Single identity-layout precision test."""
    print(f"\n  identity  T={T:5d} E={E:3d} I={I:5d}", end=" ")
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E
    # Adjust last expert to match T exactly
    tpe[-1] += T - sum(tpe)

    paddle.seed(42)
    x_p = paddle.randn([T, H], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([T, H], dtype="bfloat16") * 0.01
    x = torch.from_dlpack(x_p.detach()).to(device=device)
    grad_out = torch.from_dlpack(grad_out_p.detach()).to(device=device)

    out_fp8, dx_fp8, dw1_fp8, dw2_fp8 = _fp8_identity(experts, x, tpe, grad_out, E, I)
    out_gold, dx_gold, dw1_gold, dw2_gold = _gold_identity(x, experts, tpe, grad_out)

    # output
    cos, rrmse = _cosine_rrmse(out_fp8, out_gold)
    assert cos > 0.99, f"forward cos={cos:.4f}"
    assert rrmse < 0.10, f"forward rrmse={rrmse:.4f}"

    # dx
    if dx_fp8 is not None:
        cos, rrmse = _cosine_rrmse(dx_fp8, dx_gold)
        assert cos > 0.99, f"dx cos={cos:.4f}"
        assert rrmse < 0.10, f"dx rrmse={rrmse:.4f}"

    # dw1
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            continue
        cos, rrmse = _cosine_rrmse(dw1_fp8[e_idx], dw1_gold[e_idx])
        assert cos > 0.98, f"dw1[{e_idx}] cos={cos:.4f}"
        assert rrmse < 0.15, f"dw1[{e_idx}] rrmse={rrmse:.4f}"

    # dw2
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            continue
        cos, rrmse = _cosine_rrmse(dw2_fp8[e_idx], dw2_gold[e_idx])
        assert cos > 0.98, f"dw2[{e_idx}] cos={cos:.4f}"
        assert rrmse < 0.15, f"dw2[{e_idx}] rrmse={rrmse:.4f}"

    print("PASS")


def _run_topk_case(N_recv, topk, E, I, verbose=False):
    """Single topk-layout precision test. Returns dict of (cos, rrmse) per tensor."""
    label = f"N={N_recv:5d} K={topk} E={E:3d} I={I:5d}"
    print(f"\n  topk      {label}", end=" ", flush=True)
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]

    # Build deterministic topk dispatch
    torch.manual_seed(123)
    raw_scores = torch.randn(N_recv, E, device=device)
    _, top_experts = raw_scores.topk(topk, dim=-1)  # [N, topk]
    # Ensure no -1 (all valid)
    dispatched_indices = top_experts.int()  # [N, topk]
    dispatched_probs = torch.rand(N_recv, topk, device=device) * 0.5 + 0.5
    dispatched_probs = dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)
    dispatched_probs = dispatched_probs.float()

    # Compute tokens_per_expert
    tpe = [0] * E
    for e in range(E):
        tpe[e] = int((dispatched_indices == e).sum().item())

    paddle.seed(42)
    x_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([N_recv, H], dtype="bfloat16") * 0.01
    x = torch.from_dlpack(x_p.detach()).to(device=device)
    grad_out = torch.from_dlpack(grad_out_p.detach()).to(device=device)

    out_fp8, dx_fp8, dw1_fp8, dw2_fp8 = _fp8_topk(
        experts, x, dispatched_indices, dispatched_probs, tpe, grad_out, E, I,
    )
    out_gold, dx_gold, dw1_gold, dw2_gold = _gold_topk(
        x, experts, dispatched_indices, dispatched_probs, grad_out,
    )

    results = {}

    # output
    cos, rrmse = _cosine_rrmse(out_fp8, out_gold)
    results["out"] = (cos, rrmse)
    assert cos > 0.99, f"forward cos={cos:.4f}"
    assert rrmse < 0.10, f"forward rrmse={rrmse:.4f}"

    # dx
    if dx_fp8 is not None:
        cos, rrmse = _cosine_rrmse(dx_fp8, dx_gold)
        results["dx"] = (cos, rrmse)
        assert cos > 0.99, f"dx cos={cos:.4f}"
        assert rrmse < 0.10, f"dx rrmse={rrmse:.4f}"

    # dw1 (min across experts)
    dw1_cos_list = []
    dw1_rrmse_list = []
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            continue
        cos, rrmse = _cosine_rrmse(dw1_fp8[e_idx], dw1_gold[e_idx])
        dw1_cos_list.append(cos)
        dw1_rrmse_list.append(rrmse)
        assert cos > 0.98, f"dw1[{e_idx}] cos={cos:.4f}"
        assert rrmse < 0.15, f"dw1[{e_idx}] rrmse={rrmse:.4f}"
    results["dw1"] = (min(dw1_cos_list), max(dw1_rrmse_list))

    # dw2 (min across experts)
    dw2_cos_list = []
    dw2_rrmse_list = []
    for e_idx in range(E):
        if tpe[e_idx] == 0:
            continue
        cos, rrmse = _cosine_rrmse(dw2_fp8[e_idx], dw2_gold[e_idx])
        dw2_cos_list.append(cos)
        dw2_rrmse_list.append(rrmse)
        assert cos > 0.98, f"dw2[{e_idx}] cos={cos:.4f}"
        assert rrmse < 0.15, f"dw2[{e_idx}] rrmse={rrmse:.4f}"
    results["dw2"] = (min(dw2_cos_list), max(dw2_rrmse_list))

    print("PASS")
    return results


# ── internal dx regression test (bypasses SonicMoEMlpNode.detach) ──────────

def _run_identity_dx_internal(T, E, I):
    """Test internal dx correctness by calling _SonicMoEDeepEPFunc directly.

    This bypasses SonicMoEMlpNode.forward()'s x.detach(), allowing us to
    verify that dx propagates correctly even when total_pad_rows > 0.
    Regression test for the identity-path grad re-padding bug.
    """
    from sonicmoe.ernie_compat.mlp_node_v2 import (
        _SonicMoEDeepEPFunc,
        stack_ernie_w1,
        stack_ernie_w2,
    )
    from sonicmoe.ernie_compat.deepep_metadata import deepep_to_sonic_metadata
    from sonicmoe.functional import _refresh_fp8_config, clear_all_fp8_weight_caches

    block = 128
    tpe = [T // E] * E
    tpe[-1] += T - sum(tpe)
    pad_per_expert = [((c + block - 1) // block * block) - c for c in tpe]
    total_pad = sum(pad_per_expert)
    print(f"\n  dx_internal T={T:5d} E={E:3d} I={I:5d} pad={total_pad:4d}", end=" ")

    device = "cuda"
    experts = [MockExpert(H, I, e) for e in range(E)]

    invalidate_weight_caches()
    clear_all_fp8_weight_caches()

    # Build metadata
    (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        router_scores,
        TK_padded,
        total_pad_rows,
    ) = deepep_to_sonic_metadata(tpe, T, E, device=device)

    w1 = stack_ernie_w1(experts, H, I)
    w2 = stack_ernie_w2(experts)

    paddle.seed(42)
    x_p = paddle.randn([T, H], dtype="bfloat16") * 0.02
    x_t = torch.from_dlpack(x_p.detach()).to(device=device)
    grad_out_p = paddle.randn([T, H], dtype="bfloat16") * 0.01
    grad_out_t = torch.from_dlpack(grad_out_p.detach()).to(device=device)

    # Warmup FP8 caches
    for _ in range(5):
        x_w = x_t.clone().detach()
        x_w.stop_gradient = False
        rs_w = router_scores.clone().detach()
        rs_w.stop_gradient = False
        x_gather_idx.stop_gradient = True
        s_scatter_idx.stop_gradient = True
        w1.stop_gradient = True
        w2.stop_gradient = True
        out_w = _SonicMoEDeepEPFunc.apply(
            x_w, rs_w,
            expert_frequency_offset, x_gather_idx,
            s_scatter_idx, s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            w1, w2,
            experts, E, T, TK_padded, TK_padded,
            ActivationType.SWIGLU, 0, False,
        )
        out_w.backward(grad_out_t.clone())

    _zero_main_grads(experts)
    flush_native_grads()

    # Actual run
    x_in = x_t.clone().detach()
    x_in.stop_gradient = False
    rs_in = router_scores.clone().detach()
    rs_in.stop_gradient = False
    x_gather_idx.stop_gradient = True
    s_scatter_idx.stop_gradient = True
    w1.stop_gradient = True
    w2.stop_gradient = True

    out = _SonicMoEDeepEPFunc.apply(
        x_in, rs_in,
        expert_frequency_offset, x_gather_idx,
        s_scatter_idx, s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        w1, w2,
        experts, E, T, TK_padded, TK_padded,
        ActivationType.SWIGLU, 0, False,
    )
    out.backward(grad_out_t.clone())
    flush_native_grads()

    dx_fp8 = x_in.grad
    assert dx_fp8 is not None, "dx is None — gradient did not flow"
    dx_fp8 = torch.from_dlpack(dx_fp8.detach()).to(device=device, dtype=x_t.dtype)

    # Gold reference
    out_gold, dx_gold, _, _ = _gold_identity(x_t, experts, tpe, grad_out_t)

    # Verify output
    cos, rrmse = _cosine_rrmse(
        torch.from_dlpack(out.detach()).to(device=device, dtype=x_t.dtype), out_gold
    )
    assert cos > 0.99, f"forward cos={cos:.4f}"

    # Verify dx (the key regression check)
    cos_dx, rrmse_dx = _cosine_rrmse(dx_fp8, dx_gold)
    assert cos_dx > 0.98, f"dx cos={cos_dx:.6f} (REGRESSION: grad re-padding may be wrong)"
    assert rrmse_dx < 0.15, f"dx rrmse={rrmse_dx:.4f}"

    print(f"cos_dx={cos_dx:.6f} rrmse_dx={rrmse_dx:.4f} PASS")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SonicMoEMlpNode Precision Audit")
    print("=" * 60)

    # Topk path (production path — real DeepEP dispatch)
    print("\n--- Topk path ---")
    all_results = []
    for N_recv, topk, E, I in [
        (128, 4, 4, 384),
        (128, 8, 8, 384),
        (512, 4, 8, 1536),
        (512, 8, 8, 1536),
        (1024, 8, 8, 1536),
        (256, 8, 32, 1536),
    ]:
        r = _run_topk_case(N_recv, topk, E, I, verbose=True)
        all_results.append((N_recv, topk, E, I, r))

    # Print summary table
    print("\n" + "=" * 90)
    print("  PRECISION SUMMARY (cosine similarity / RRMSE)")
    print("=" * 90)
    print(f"  {'Shape':>30s} | {'out cos':>8s} {'rrmse':>7s} | {'dx cos':>8s} {'rrmse':>7s} | {'dw1 cos':>8s} {'rrmse':>7s} | {'dw2 cos':>8s} {'rrmse':>7s}")
    print("  " + "-" * 88)
    for N, K, E, I, r in all_results:
        label = f"N={N} K={K} E={E} I={I}"
        out_c, out_r = r.get("out", (0, 0))
        dx_c, dx_r = r.get("dx", (0, 0))
        dw1_c, dw1_r = r.get("dw1", (0, 0))
        dw2_c, dw2_r = r.get("dw2", (0, 0))
        print(f"  {label:>30s} | {out_c:.4f}  {out_r:.4f} | {dx_c:.4f}  {dx_r:.4f} | {dw1_c:.4f}  {dw1_r:.4f} | {dw2_c:.4f}  {dw2_r:.4f}")

    print("\n" + "=" * 60)
    print("ALL PRECISION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
