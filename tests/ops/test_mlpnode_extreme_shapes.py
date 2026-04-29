#!/usr/bin/env python
"""
Extreme-shape stress tests for SonicMoEMlpNode (CI gating).

Covers production failure modes that have bitten us before:

    - T=0  total_K=0 and per-expert empty buckets
    - very-small T (≤ 16)
    - very-large T (32k tokens) at production H/I
    - extreme imbalance: 99% of tokens routed to one expert

Each case must run forward + backward + flush_grads without crashing,
produce finite outputs, and (for non-zero T) populate main_grad on every
expert that received tokens.

These tests intentionally don't compare to a gold reference — they are
robustness gates, not correctness regressions. Element-wise precision is
covered by tests/ops/test_mlpnode_precision.py.
"""

from __future__ import annotations

import os
import sys
import math

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

import pytest

import paddle
paddle.enable_compat()
import torch  # noqa: E402

import sonicmoe.functional as functional  # noqa: E402
from sonicmoe.ernie_compat import (  # noqa: E402
    SonicMoEMlpNode,
    invalidate_weight_caches,
)


functional._ALIGNMENT_ASSUMED = True
functional._ALIGNMENT_STREAK = 100


def _make_experts(H: int, I: int, E: int):
    """Mirror the MockExpert pattern from test_mlpnode_precision.py."""
    experts = []
    for e in range(E):
        paddle.seed(1000 + e)
        up = type("P", (), {
            "weight": paddle.randn([H, 2 * I], dtype="bfloat16") / math.sqrt(H),
        })()
        dn = type("P", (), {
            "weight": paddle.randn([I, H], dtype="bfloat16") / math.sqrt(I),
        })()
        up.weight.stop_gradient = False
        dn.weight.stop_gradient = False
        experts.append(type("Expert", (), {
            "up_gate_proj": up,
            "down_proj": dn,
        })())
    return experts


def _build_topk(T: int, E: int, topk: int, *, imbalance_expert: int | None = None,
                imbalance_frac: float = 0.99):
    """Construct (dispatched_indices, dispatched_probs, tpe) for given pattern.

    Contract: each row of ``dispatched_indices`` must contain **unique**
    expert IDs (matching what real DeepEP dispatch produces — duplicates
    within a row are illegal because two slots would collapse onto one
    token-major position in s_reverse_scatter_idx, leaving orphaned padded
    positions and causing IMA in the downstream router_forward kernel).
    """
    if T == 0:
        di = torch.zeros((0, topk), dtype=torch.int32, device="cuda")
        dp = torch.zeros((0, topk), dtype=torch.float32, device="cuda")
        tpe = [0] * E
        return di, dp, tpe

    if imbalance_expert is not None:
        assert topk <= E, "topk cannot exceed number of experts (uniqueness)"
        n_hot = int(T * imbalance_frac)
        # Hot rows: [imbalance_expert, then other distinct experts in order].
        other_pool = [e for e in range(E) if e != imbalance_expert]
        hot_row = [imbalance_expert] + other_pool[: topk - 1]
        idx = torch.tensor(hot_row, dtype=torch.int32, device="cuda") \
                   .repeat(T, 1).contiguous()
        # Cold rows: random unique-per-row top-k via argsort over noise.
        if n_hot < T:
            torch.manual_seed(0)
            noise = torch.randn(T - n_hot, E, device="cuda")
            cold = noise.topk(topk, dim=-1).indices.to(torch.int32)
            idx[n_hot:] = cold
        torch.manual_seed(0)
        prob = torch.rand(T, topk, device="cuda") * 0.5 + 0.5
        prob = (prob / prob.sum(dim=1, keepdim=True)).float()
        tpe = [int((idx == e).sum().item()) for e in range(E)]
        return idx, prob, tpe

    torch.manual_seed(0)
    raw = torch.randn(T, E, device="cuda")
    _, top = raw.topk(topk, dim=-1)
    di = top.int()
    dp = torch.rand(T, topk, device="cuda") * 0.5 + 0.5
    dp = (dp / dp.sum(dim=1, keepdim=True)).float()
    tpe = [int((di == e).sum().item()) for e in range(E)]
    return di, dp, tpe


def _run_one(T: int, E: int, H: int, I: int, *,
             topk: int = 2,
             imbalance_expert: int | None = None,
             imbalance_frac: float = 0.99):
    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    experts = _make_experts(H, I, E)

    di, dp, tpe = _build_topk(T, E, topk,
                              imbalance_expert=imbalance_expert,
                              imbalance_frac=imbalance_frac)

    paddle.seed(7)
    if T == 0:
        x_p = paddle.zeros([0, H], dtype="bfloat16")
        g_p = paddle.zeros([0, H], dtype="bfloat16")
    else:
        x_p = paddle.randn([T, H], dtype="bfloat16") * 0.02
        g_p = paddle.randn([T, H], dtype="bfloat16") * 0.01
    x = torch.from_dlpack(x_p.detach())
    g = torch.from_dlpack(g_p.detach())

    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H,
                           intermediate_size=I)
    x_in = x.clone().detach()
    x_in.stop_gradient = False
    out = node.forward(x_in, tpe, dispatched_indices=di, dispatched_probs=dp)
    out.backward(g.clone())
    node.flush_grads()

    # Output must be finite and right shape (even when T=0).
    assert out.shape[0] == T, f"expected T={T} rows, got {out.shape}"
    if T > 0:
        out_t = torch.from_dlpack(out.detach()).float()
        assert torch.isfinite(out_t).all(), "non-finite values in output"

    # Every expert that received tokens must have a non-zero main_grad.
    for e_idx, exp in enumerate(experts):
        if tpe[e_idx] == 0:
            continue
        mg1 = torch.from_dlpack(exp.up_gate_proj.weight.main_grad.detach()).float()
        mg2 = torch.from_dlpack(exp.down_proj.weight.main_grad.detach()).float()
        assert torch.isfinite(mg1).all(), f"expert {e_idx} dw1 has non-finite"
        assert torch.isfinite(mg2).all(), f"expert {e_idx} dw2 has non-finite"
        assert mg1.abs().sum() > 0, f"expert {e_idx} dw1 is all-zero"
        assert mg2.abs().sum() > 0, f"expert {e_idx} dw2 is all-zero"


# ── parameterized cases ──────────────────────────────────────────────────────

_E, _H, _I = 8, 3072, 1536


@pytest.mark.parametrize("T", [0])
def test_zero_total_tokens(T: int):
    """T=0 must not crash anywhere."""
    _run_one(T, _E, _H, _I)


@pytest.mark.parametrize("T", [8, 16])
def test_tiny_token_count(T: int):
    """Very small T stresses padding / capacity handling."""
    _run_one(T, _E, _H, _I)


def test_per_expert_empty_bucket():
    """One specific expert receives 0 tokens — common in early training."""
    # Force all tokens onto expert 0 → expert 1..7 receive 0.
    _run_one(T=2048, E=_E, H=_H, I=_I,
             imbalance_expert=0, imbalance_frac=1.0)


def test_extreme_imbalance_99():
    """99% of tokens routed to a single expert."""
    _run_one(T=4096, E=_E, H=_H, I=_I,
             imbalance_expert=3, imbalance_frac=0.99)


def test_strong_imbalance_85():
    """85% to a single expert — stresses but stays within CuTe limits."""
    _run_one(T=4096, E=_E, H=_H, I=_I,
             imbalance_expert=3, imbalance_frac=0.85)


@pytest.mark.parametrize("T", [16384, 32768])
def test_large_T(T: int):
    """Large T at production shapes — checks no OOM / shape overflow."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    free, _ = torch.cuda.mem_get_info()
    # Conservative skip — skip if <12 GiB free.
    if free < 12 * (1 << 30):
        pytest.skip(f"insufficient free GPU mem ({free/(1<<30):.1f} GiB)")
    _run_one(T=T, E=_E, H=_H, I=_I)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-s", "-v"]))
