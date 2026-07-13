"""Cross-framework MoE module precision test.

Tests the full MoE forward/backward pipeline:
  permute -> up-gate projection -> SwiGLU -> down projection -> unpermute

Validates SonicMoE BF16 and FP8 paths against a pure-torch float32 gold reference
that uses the ERNIE split-half SwiGLU convention.

Weight conversion between split-half and interleaved (SonicMoE) is verified
explicitly.
"""
import paddle
paddle.enable_compat()

import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

from tests.ops.conftest import (
    requires_blackwell,
    requires_quack,
    rrmse,
    cosine_sim,
    SEEDS,
)

pytestmark = [requires_blackwell, requires_quack]


def _assert_close(actual, expected, atol=1e-5, rtol=0):
    """Paddle-compatible replacement for _assert_close."""
    diff = (actual.float() - expected.float()).abs()
    limit = atol + rtol * expected.float().abs()
    if not (diff <= limit).all():
        max_diff = diff.max().item()
        raise AssertionError(
            f"Tensors not close: max abs diff={max_diff}, atol={atol}, rtol={rtol}"
        )

# ---------------------------------------------------------------------------
# Shapes: (T, H, I, E, K)
# ---------------------------------------------------------------------------
MOE_SHAPES = [
    pytest.param(256, 768, 384, 8, 2, id="small"),
    pytest.param(2048, 3072, 1536, 8, 8, id="production"),
]

# ---------------------------------------------------------------------------
# Precision thresholds
# ---------------------------------------------------------------------------
BF16_RRMSE = 0.01
BF16_COSINE = 0.999
BF16_DW_RRMSE = 0.02

FP8_RRMSE = 0.10
FP8_COSINE = 0.99
FP8_DW_RRMSE = 0.15


# ═══════════════════════════════════════════════════════════════════════════
# Weight conversion helpers
# ═══════════════════════════════════════════════════════════════════════════

def split_to_interleaved(w_split: torch.Tensor) -> torch.Tensor:
    """Convert split-half (2I, H) -> interleaved (2I, H).

    Split-half:              [gate_0..gate_{I-1}, up_0..up_{I-1}]
    Interleaved (SonicMoE):  [gate_0, up_0, gate_1, up_1, ...]
    """
    two_I = w_split.shape[0]
    I = two_I // 2
    w_out = torch.empty_like(w_split)
    w_out[0::2] = w_split[:I]   # gate rows -> even
    w_out[1::2] = w_split[I:]   # up rows   -> odd
    return w_out


def interleaved_to_split(w_interleaved: torch.Tensor) -> torch.Tensor:
    """Convert interleaved (2I, H) -> split-half (2I, H)."""
    two_I = w_interleaved.shape[0]
    I = two_I // 2
    w_out = torch.empty_like(w_interleaved)
    w_out[:I] = w_interleaved[0::2]   # even -> gate
    w_out[I:] = w_interleaved[1::2]   # odd  -> up
    return w_out


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic routing
# ═══════════════════════════════════════════════════════════════════════════

def _make_deterministic_routing(T: int, E: int, K: int, device: str = "cuda"):
    """Generate deterministic topk routing: round-robin expert assignment.

    Returns topk_indices (T, K) int32, topk_scores (T, K) float32.
    Each token t selects experts (t*K + k) % E for k in 0..K-1.
    Scores are softmax-normalized over each token's selected experts.
    """
    topk_indices = torch.zeros(T, K, dtype=torch.int32, device=device)
    for t in range(T):
        for k in range(K):
            topk_indices[t, k] = (t * K + k) % E

    # Generate random logits and pick scores from selected experts
    paddle.seed(42)  # fixed seed for routing scores
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    # Gather logits for selected experts
    gathered = logits.gather(1, topk_indices.long())
    topk_scores = F.softmax(gathered, dim=-1)
    return topk_indices, topk_scores


# ═══════════════════════════════════════════════════════════════════════════
# Gold reference (pure torch, float32)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_routing_metadata(topk_indices: torch.Tensor, E: int):
    """Compute routing metadata on CPU in int64 for correctness.

    Returns:
        cu_seqlens: (E+1,) int32 on cuda — expert_frequency_offset
        x_gather_idx: (TK,) int32 on cuda — which input token each sorted pos maps to
        s_reverse_scatter_idx: (TK,) int32 on cuda — for each flat (t*K+k), the sorted position
    """
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device

    # Flatten indices: for position p = t*K + k, expert = topk_indices[t, k]
    flat_experts = topk_indices.reshape(-1).cpu().long()  # (TK,)
    flat_tokens = torch.arange(T, device="cpu").unsqueeze(1).expand(T, K).reshape(-1).long()

    # Sort by expert (stable so within-expert order is deterministic)
    sorted_order = torch.argsort(flat_experts, stable=True)

    # cu_seqlens: count per expert
    counts = torch.zeros(E, dtype=torch.int32)
    for e in range(E):
        counts[e] = (flat_experts == e).sum().item()
    cu_seqlens = torch.zeros(E + 1, dtype=torch.int32)
    cu_seqlens[1:] = counts.cumsum(0)

    # x_gather_idx: for sorted position p, which token to gather from x
    x_gather_idx = flat_tokens[sorted_order].to(torch.int32)

    # s_reverse_scatter_idx: for each flat (t*K+k), the sorted position
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32)
    for i, so in enumerate(sorted_order):
        s_reverse_scatter_idx[so.item()] = i

    return (
        cu_seqlens.to(device),
        x_gather_idx.to(device),
        s_reverse_scatter_idx.to(device),
    )


def _torch_moe_gold(
    x: torch.Tensor,          # (T, H) bf16
    w1_split: torch.Tensor,   # (E, 2I, H) float32 — split-half convention
    w2: torch.Tensor,          # (E, H, I) float32
    topk_indices: torch.Tensor,  # (T, K) int32
    topk_scores: torch.Tensor,  # (T, K) float32
) -> torch.Tensor:
    """Full MoE forward in float32 using split-half SwiGLU (split-half convention).

    Pipeline:
      1. Compute routing metadata (cu_seqlens, x_gather_idx, s_reverse_scatter_idx)
      2. Gather: x_gathered[p] = x[x_gather_idx[p]]
      3. Per-expert up-proj: z[s:e] = x_gathered[s:e] @ w1_expert.T
      4. Split-half SwiGLU: y1 = silu(z[:, :I]) * z[:, I:]
      5. Per-expert down-proj: y2[s:e] = y1[s:e] @ w2_expert.T
      6. Scatter with scores: o[t] = sum_k y2[sorted_pos[t*K+k]] * score[t*K+k]
    """
    T, H = x.shape
    E = w1_split.shape[0]
    two_I = w1_split.shape[1]
    I = two_I // 2
    K = topk_indices.shape[1]
    TK = T * K

    x_f32 = x.float()

    cu_seqlens, x_gather_idx, s_reverse_scatter_idx = _compute_routing_metadata(
        topk_indices, E
    )
    cu_seqlens_cpu = cu_seqlens.cpu()

    # Gather
    x_gathered = x_f32[x_gather_idx.long()]  # (TK, H)

    # Up-proj + SwiGLU per expert
    y1 = torch.zeros(TK, I, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        # z_e = x_gathered[s:end] @ w1_split[e].T  -> (n, 2I)
        z_e = x_gathered[s:end] @ w1_split[e].T
        # Split-half SwiGLU
        gate = z_e[:, :I]
        up = z_e[:, I:]
        y1[s:end] = F.silu(gate) * up

    # Down-proj per expert
    y2 = torch.zeros(TK, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        y2[s:end] = y1[s:end] @ w2[e].T

    # Scatter: o[t] = sum_k y2[reverse_scatter_idx[t*K+k]] * scores[t, k]
    o = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    s_rev_cpu = s_reverse_scatter_idx.cpu()
    scores_f32 = topk_scores.float()
    for t in range(T):
        for k in range(K):
            flat_idx = t * K + k
            sorted_pos = s_rev_cpu[flat_idx].item()
            o[t] += y2[sorted_pos] * scores_f32[t, k].item()

    return o


def _torch_moe_gold_backward(
    x: torch.Tensor,          # (T, H) bf16
    w1_split: torch.Tensor,   # (E, 2I, H) float32
    w2: torch.Tensor,          # (E, H, I) float32
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    grad_output: torch.Tensor,  # (T, H) float32
):
    """Gold backward pass in float32 for full MoE.

    Returns dw1 (E, 2I, H), dw2 (E, H, I), dx (T, H) in float32.
    dw1 uses split-half convention matching w1_split.
    """
    T, H = x.shape
    E = w1_split.shape[0]
    two_I = w1_split.shape[1]
    I = two_I // 2
    K = topk_indices.shape[1]
    TK = T * K

    x_f32 = x.float()
    grad_output_f32 = grad_output.float()

    cu_seqlens, x_gather_idx, s_reverse_scatter_idx = _compute_routing_metadata(
        topk_indices, E
    )
    cu_seqlens_cpu = cu_seqlens.cpu()

    # ── Recompute forward intermediates ──
    x_gathered = x_f32[x_gather_idx.long()]  # (TK, H)

    z_all = torch.zeros(TK, two_I, dtype=torch.float32, device=x.device)
    y1_all = torch.zeros(TK, I, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        z_e = x_gathered[s:end] @ w1_split[e].T
        z_all[s:end] = z_e
        gate = z_e[:, :I]
        up = z_e[:, I:]
        y1_all[s:end] = F.silu(gate) * up

    y2_all = torch.zeros(TK, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        y2_all[s:end] = y1_all[s:end] @ w2[e].T

    # ── Backward: scatter grad ──
    # dy2[sorted_pos] = grad_output[t] * score[t, k]
    dy2 = torch.zeros(TK, H, dtype=torch.float32, device=x.device)
    s_rev_cpu = s_reverse_scatter_idx.cpu()
    scores_f32 = topk_scores.float()
    for t in range(T):
        for k in range(K):
            flat_idx = t * K + k
            sorted_pos = s_rev_cpu[flat_idx].item()
            dy2[sorted_pos] += grad_output_f32[t] * scores_f32[t, k].item()

    # ── Backward: down-proj ──
    # dy1[s:e] = dy2[s:e] @ w2[e]  (w2 is (H, I), so dy2 @ w2 gives (n, I))
    dy1 = torch.zeros(TK, I, dtype=torch.float32, device=x.device)
    dw2 = torch.zeros(E, H, I, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        dy1[s:end] = dy2[s:end] @ w2[e]  # (n, H) @ (H, I) -> (n, I)
        dw2[e] = dy2[s:end].T @ y1_all[s:end]  # (H, n) @ (n, I) -> (H, I)

    # ── Backward: SwiGLU ──
    # z = [gate, up] (split-half), y1 = silu(gate) * up
    # dy1 is (n, I)
    # dgate = dy1 * up * dsilu(gate)
    # dup = dy1 * silu(gate)
    dz = torch.zeros(TK, two_I, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        gate = z_all[s:end, :I]
        up = z_all[s:end, I:]
        sig_gate = torch.sigmoid(gate)
        silu_gate = gate * sig_gate  # silu(gate)

        # dsilu/dgate = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
        #             = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        dsilu = sig_gate * (1.0 + gate * (1.0 - sig_gate))

        dgate = dy1[s:end] * up * dsilu
        dup = dy1[s:end] * silu_gate

        dz[s:end, :I] = dgate
        dz[s:end, I:] = dup

    # ── Backward: up-proj ──
    dw1 = torch.zeros(E, two_I, H, dtype=torch.float32, device=x.device)
    dx_gathered = torch.zeros(TK, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        s = cu_seqlens_cpu[e].item()
        end = cu_seqlens_cpu[e + 1].item()
        if s >= end:
            continue
        dx_gathered[s:end] = dz[s:end] @ w1_split[e]  # (n, 2I) @ (2I, H) -> (n, H)
        dw1[e] = dz[s:end].T @ x_gathered[s:end]  # (2I, n) @ (n, H) -> (2I, H)

    # ── Backward: scatter dx_gathered to dx ──
    dx = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    x_gather_cpu = x_gather_idx.cpu().long()
    dx.index_add_(0, x_gather_cpu.to(x.device), dx_gathered)

    return dw1, dw2, dx


# ═══════════════════════════════════════════════════════════════════════════
# SonicMoE runner (calls moe_TC_softmax_topk_layer through _UpProjection/_DownProjection)
# ═══════════════════════════════════════════════════════════════════════════

def _run_sonicmoe_bf16(
    x: torch.Tensor,       # (T, H)
    w1_param: torch.Tensor,  # (E, 2I, H) bf16 — contiguous parameter, interleaved
    w2_param: torch.Tensor,  # (E, H, I) bf16 — contiguous parameter
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    E: int,
    K: int,
    need_grad: bool = False,
):
    """Run SonicMoE BF16 path using quack gemm_gated + gemm.

    Mimics moe_TC_softmax_topk_layer but with pre-computed routing.
    Weight permutation follows the real code: w1_param.permute(1,2,0) -> (2I,H,E).
    Returns output (T, H) and optionally (dw1, dw2) gradients.
    """
    from sonicmoe.functional import (
        _UpProjection,
        _DownProjection,
    )
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    from sonicmoe.enums import ActivationType

    T, H = x.shape
    TK = T * K
    device = x.device

    # Compute routing metadata via Triton kernel
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    if need_grad:
        x_input = x.detach().requires_grad_(True)
        w1_p = w1_param.detach().requires_grad_(True)
        w2_p = w2_param.detach().requires_grad_(True)
    else:
        x_input = x
        w1_p = w1_param
        w2_p = w2_param

    # Permute to functional layout: (2I, H, E) and (H, I, E) — non-contiguous views
    w1_func = w1_p.permute(1, 2, 0)
    w2_func = w2_p.permute(1, 2, 0)

    # Disable FP8 for BF16 path
    old_env = os.environ.get("SONIC_MOE_FP8_MODE", None)
    os.environ["SONIC_MOE_FP8_MODE"] = "off"
    # Also force the module-level flag off
    from sonicmoe.functional import utils as _fp8_utils
    old_fp8_active = _fp8_utils._IS_FP8_ACTIVE
    _fp8_utils._IS_FP8_ACTIVE = False
    try:
        y1, z = _UpProjection.apply(
            x_input,
            w1_func,
            None,  # b1
            expert_frequency_offset,
            TK,
            K,
            0,  # stream_id
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,  # num_activated_expert_per_token_offset
            False,  # is_varlen_K
            ActivationType.SWIGLU,
            False,  # is_inference_mode_enabled
            False,  # use_low_precision_postact_buffer
        )

        o = _DownProjection.apply(
            y1,
            z,
            w2_func,
            None,  # b2
            topk_scores,
            topk_indices,
            expert_frequency_offset,
            T,
            K,
            0,  # stream_id
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,  # num_activated_expert_per_token_offset
            False,  # is_varlen_K
            ActivationType.SWIGLU,
            None,  # fp8_protocol
        )
    finally:
        _fp8_utils._IS_FP8_ACTIVE = old_fp8_active
        if old_env is None:
            os.environ.pop("SONIC_MOE_FP8_MODE", None)
        else:
            os.environ["SONIC_MOE_FP8_MODE"] = old_env

    if need_grad:
        grad_out = torch.randn_like(o)
        o.backward(grad_out)
        return o.detach(), w1_p.grad.detach(), w2_p.grad.detach(), grad_out
    return o.detach(), None, None, None


def _run_sonicmoe_fp8(
    x: torch.Tensor,
    w1_param: torch.Tensor,  # (E, 2I, H) bf16 — contiguous parameter, interleaved
    w2_param: torch.Tensor,  # (E, H, I) bf16 — contiguous parameter
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    E: int,
    K: int,
    need_grad: bool = False,
):
    """Run SonicMoE FP8 path using blockscaled CUTLASS kernels.

    Returns output (T, H) and optionally (dw1, dw2) gradients.
    """
    from sonicmoe.functional import (
        _UpProjection,
        _DownProjection,
    )
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    from sonicmoe.enums import ActivationType

    T, H = x.shape
    TK = T * K
    device = x.device

    # Compute routing metadata
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    if need_grad:
        x_input = x.detach().requires_grad_(True)
        w1_p = w1_param.detach().requires_grad_(True)
        w2_p = w2_param.detach().requires_grad_(True)
    else:
        x_input = x
        w1_p = w1_param
        w2_p = w2_param

    # Permute to functional layout
    w1_func = w1_p.permute(1, 2, 0)
    w2_func = w2_p.permute(1, 2, 0)

    # Enable FP8 via context manager (the proper way)
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.functional import (
        _refresh_fp8_config,
        clear_all_fp8_weight_caches,
    )
    clear_all_fp8_weight_caches()
    with enable_fp8(True):
        _refresh_fp8_config()
        try:
            y1, z = _UpProjection.apply(
                x_input,
                w1_func,
                None,
                expert_frequency_offset,
                TK,
                K,
                0,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                None,
                False,
                ActivationType.SWIGLU,
                False,
                False,
            )

            o = _DownProjection.apply(
                y1,
                z,
                w2_func,
                None,
                topk_scores,
                topk_indices,
                expert_frequency_offset,
                T,
                K,
                0,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                None,
                False,
                ActivationType.SWIGLU,
                None,  # fp8_protocol
            )
        finally:
            clear_all_fp8_weight_caches()

    if need_grad:
        grad_out = torch.randn_like(o)
        o.backward(grad_out)
        return o.detach(), w1_p.grad.detach(), w2_p.grad.detach(), grad_out
    return o.detach(), None, None, None


# ═══════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════

def _make_test_data(T, H, I, E, K, seed, device="cuda"):
    """Generate test data for MoE module test.

    Returns:
        x: (T, H) bf16
        w1_split: (E, 2I, H) float32 — split-half (split-half convention)
        w2: (E, H, I) float32
        topk_indices: (T, K) int32
        topk_scores: (T, K) float32
    """
    paddle.seed(seed)

    x = torch.randn(T, H, device=device, dtype=torch.float32)
    x = (x * 0.02).to(torch.bfloat16)  # realistic activation scale

    # Weights in split-half (gold convention)
    w1_split = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32) * 0.02
    w2 = torch.randn(E, H, I, device=device, dtype=torch.float32) * 0.02

    topk_indices, topk_scores = _make_deterministic_routing(T, E, K, device)

    return x, w1_split, w2, topk_indices, topk_scores


def _convert_weights_for_sonicmoe(w1_split, w2_gold):
    """Convert gold weights to SonicMoE parameter format.

    Mimics the Experts class: parameter shape is (E, out_features, in_features),
    and the functional layer receives w.permute(1, 2, 0) as a non-contiguous view.

    Args:
        w1_split: (E, 2I, H) float32
        w2_gold:  (E, H, I) float32

    Returns:
        w1_param: (E, 2I, H) bf16 — contiguous parameter (interleaved rows)
        w2_param: (E, H, I) bf16 — contiguous parameter
    """
    E = w1_split.shape[0]
    # Convert per-expert from split to interleaved, keep (E, 2I, H) contiguous
    w1_interleaved_list = []
    for e in range(E):
        w1_interleaved_list.append(split_to_interleaved(w1_split[e]))
    w1_param = torch.stack(w1_interleaved_list).to(torch.bfloat16).contiguous()  # (E, 2I, H)
    w2_param = w2_gold.to(torch.bfloat16).contiguous()  # (E, H, I)
    return w1_param, w2_param


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Gold self-consistency
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, H, I, E, K", MOE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_gold_self_consistency(T, H, I, E, K, seed):
    """Gold forward matches manual per-element computation for a tiny case."""
    paddle.seed(seed)

    # Use a tiny shape so we can verify element-by-element
    T_tiny, H_tiny, I_tiny, E_tiny, K_tiny = 4, 32, 16, 2, 2
    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(
        T_tiny, H_tiny, I_tiny, E_tiny, K_tiny, seed
    )

    o = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)

    # Manual computation: same pipeline but explicit loops
    x_f32 = x.float()
    cu_seqlens, x_gather_idx, s_reverse_scatter_idx = _compute_routing_metadata(
        topk_indices, E_tiny
    )

    x_gathered = x_f32[x_gather_idx.long()]
    cu_cpu = cu_seqlens.cpu()

    # Up-proj + SwiGLU
    TK = T_tiny * K_tiny
    y1_manual = torch.zeros(TK, I_tiny, dtype=torch.float32, device="cuda")
    for e in range(E_tiny):
        s, end = cu_cpu[e].item(), cu_cpu[e + 1].item()
        if s >= end:
            continue
        z_e = x_gathered[s:end] @ w1_split[e].T
        y1_manual[s:end] = F.silu(z_e[:, :I_tiny]) * z_e[:, I_tiny:]

    # Down-proj
    y2_manual = torch.zeros(TK, H_tiny, dtype=torch.float32, device="cuda")
    for e in range(E_tiny):
        s, end = cu_cpu[e].item(), cu_cpu[e + 1].item()
        if s >= end:
            continue
        y2_manual[s:end] = y1_manual[s:end] @ w2[e].T

    # Scatter
    o_manual = torch.zeros(T_tiny, H_tiny, dtype=torch.float32, device="cuda")
    s_rev_cpu = s_reverse_scatter_idx.cpu()
    for t in range(T_tiny):
        for k in range(K_tiny):
            pos = s_rev_cpu[t * K_tiny + k].item()
            o_manual[t] += y2_manual[pos] * topk_scores[t, k].float().item()

    _assert_close(o, o_manual, atol=1e-5, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: SonicMoE BF16 vs Gold
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, H, I, E, K", MOE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_sonicmoe_bf16_vs_gold(T, H, I, E, K, seed):
    """SonicMoE BF16 output should closely match gold float32 reference."""
    paddle.seed(seed)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, seed)

    # Gold forward
    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)

    # Convert weights to SonicMoE format
    w1_sonic, w2_sonic = _convert_weights_for_sonicmoe(w1_split, w2)

    # SonicMoE BF16 forward
    o_bf16, _, _, _ = _run_sonicmoe_bf16(
        x, w1_sonic, w2_sonic, topk_indices, topk_scores, E, K
    )

    r = rrmse(o_bf16, o_gold.to(torch.bfloat16))
    c = cosine_sim(o_bf16, o_gold.to(torch.bfloat16))
    print(f"  BF16 vs Gold: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < BF16_RRMSE, f"RRMSE {r:.6f} >= {BF16_RRMSE}"
    assert c > BF16_COSINE, f"cosine {c:.6f} <= {BF16_COSINE}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: SonicMoE FP8 vs Gold (subprocess-isolated)
# ═══════════════════════════════════════════════════════════════════════════

_FP8_SUBPROCESS_TEMPLATE = textwrap.dedent("""\
    import os, sys, json
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    import paddle
    paddle.enable_compat()
    import torch
    sys.path.insert(0, {project_root!r})

    from tests.ops.test_moe_module import (
        _make_test_data, _convert_weights_for_sonicmoe,
        _torch_moe_gold, _run_sonicmoe_fp8,
        rrmse, cosine_sim,
    )

    T, H, I, E, K, seed = {T}, {H}, {I}, {E}, {K}, {seed}
    paddle.seed(seed)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, seed)
    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)

    w1_param, w2_param = _convert_weights_for_sonicmoe(w1_split, w2)
    o_fp8, _, _, _ = _run_sonicmoe_fp8(
        x, w1_param, w2_param, topk_indices, topk_scores, E, K
    )

    r = rrmse(o_fp8, o_gold.to(torch.bfloat16))
    c = cosine_sim(o_fp8, o_gold.to(torch.bfloat16))
    print(json.dumps({{"rrmse": r, "cosine": c}}))
""")


@pytest.mark.parametrize("T, H, I, E, K", MOE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_sonicmoe_fp8_vs_gold(T, H, I, E, K, seed):
    """SonicMoE FP8 output matches gold float32 reference within FP8 tolerance.

    Run in a subprocess to isolate FP8 varlen state.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _FP8_SUBPROCESS_TEMPLATE.format(
        project_root=project_root, T=T, H=H, I=I, E=E, K=K, seed=seed
    )
    python = sys.executable
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"

    result = subprocess.run(
        [python, "-c", script],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"FP8 subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    import json
    # Find the JSON line in output
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            metrics = json.loads(line)
            break
    else:
        pytest.fail(f"No JSON output from FP8 subprocess:\n{result.stdout}")

    r = metrics["rrmse"]
    c = metrics["cosine"]
    print(f"  FP8 vs Gold: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < FP8_RRMSE, f"RRMSE {r:.6f} >= {FP8_RRMSE}"
    assert c > FP8_COSINE, f"cosine {c:.6f} <= {FP8_COSINE}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: SonicMoE BF16 vs FP8 (subprocess-isolated)
# ═══════════════════════════════════════════════════════════════════════════

_BF16_VS_FP8_SUBPROCESS_TEMPLATE = textwrap.dedent("""\
    import os, sys, json
    os.environ["USE_QUACK_GEMM"] = "1"
    import paddle
    paddle.enable_compat()
    import torch
    sys.path.insert(0, {project_root!r})

    from tests.ops.test_moe_module import (
        _make_test_data, _convert_weights_for_sonicmoe,
        _run_sonicmoe_bf16, _run_sonicmoe_fp8,
        rrmse, cosine_sim,
    )

    T, H, I, E, K, seed = {T}, {H}, {I}, {E}, {K}, {seed}
    paddle.seed(seed)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, seed)
    w1_param, w2_param = _convert_weights_for_sonicmoe(w1_split, w2)

    o_bf16, _, _, _ = _run_sonicmoe_bf16(
        x, w1_param, w2_param, topk_indices, topk_scores, E, K
    )
    o_fp8, _, _, _ = _run_sonicmoe_fp8(
        x, w1_param, w2_param, topk_indices, topk_scores, E, K
    )

    r = rrmse(o_fp8, o_bf16)
    c = cosine_sim(o_fp8, o_bf16)
    print(json.dumps({{"rrmse": r, "cosine": c}}))
""")


@pytest.mark.parametrize("T, H, I, E, K", MOE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_sonicmoe_bf16_vs_fp8(T, H, I, E, K, seed):
    """Cross-check: BF16 vs FP8 module-level output within FP8 tolerance."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _BF16_VS_FP8_SUBPROCESS_TEMPLATE.format(
        project_root=project_root, T=T, H=H, I=I, E=E, K=K, seed=seed
    )
    python = sys.executable
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"

    result = subprocess.run(
        [python, "-c", script],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"BF16-vs-FP8 subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    import json
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            metrics = json.loads(line)
            break
    else:
        pytest.fail(f"No JSON output:\n{result.stdout}")

    r = metrics["rrmse"]
    c = metrics["cosine"]
    print(f"  BF16 vs FP8: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < FP8_RRMSE, f"RRMSE {r:.6f} >= {FP8_RRMSE}"
    assert c > FP8_COSINE, f"cosine {c:.6f} <= {FP8_COSINE}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: split-half SwiGLU vs Gold (validates weight conversion)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, H, I, E, K", MOE_SHAPES)
@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_split_half_vs_gold(T, H, I, E, K, seed):
    """Verify split-half SwiGLU torch implementation matches gold exactly.

    Also validates round-trip weight conversion:
      split -> interleaved -> split should recover original.
    """
    paddle.seed(seed)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, seed)

    # Gold forward (split-half)
    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)

    # Manual split-half implementation for cross-check
    x_f32 = x.float()
    cu_seqlens, x_gather_idx, s_reverse_scatter_idx = _compute_routing_metadata(
        topk_indices, E
    )
    cu_cpu = cu_seqlens.cpu()
    TK = T * K

    x_gathered = x_f32[x_gather_idx.long()]
    y1_check = torch.zeros(TK, I, dtype=torch.float32, device="cuda")
    for e in range(E):
        s, end = cu_cpu[e].item(), cu_cpu[e + 1].item()
        if s >= end:
            continue
        z_e = x_gathered[s:end] @ w1_split[e].T
        y1_check[s:end] = F.silu(z_e[:, :I]) * z_e[:, I:]

    y2_check = torch.zeros(TK, x.shape[1], dtype=torch.float32, device="cuda")
    for e in range(E):
        s, end = cu_cpu[e].item(), cu_cpu[e + 1].item()
        if s >= end:
            continue
        y2_check[s:end] = y1_check[s:end] @ w2[e].T

    o_check = torch.zeros(T, x.shape[1], dtype=torch.float32, device="cuda")
    s_rev_cpu = s_reverse_scatter_idx.cpu()
    for t in range(T):
        for k in range(K):
            pos = s_rev_cpu[t * K + k].item()
            o_check[t] += y2_check[pos] * topk_scores[t, k].float().item()

    # Must match gold exactly (both are float32 with same math)
    _assert_close(o_gold, o_check, atol=1e-5, rtol=0)

    # Also test round-trip weight conversion
    for e in range(E):
        w1_orig = w1_split[e]  # (2I, H)
        w1_inter = split_to_interleaved(w1_orig)
        w1_back = interleaved_to_split(w1_inter)
        _assert_close(w1_orig, w1_back, atol=0, rtol=0)

    # Verify interleaved version with SonicMoE's swiglu convention
    # x_gathered @ w1_interleaved.T gives z_interleaved
    # Then z[:, 0::2] = gate, z[:, 1::2] = up
    for e in range(E):
        w1_inter_e = split_to_interleaved(w1_split[e])
        s, end = cu_cpu[e].item(), cu_cpu[e + 1].item()
        if s >= end:
            continue
        z_inter = x_gathered[s:end] @ w1_inter_e.T
        gate_inter = z_inter[:, 0::2]
        up_inter = z_inter[:, 1::2]
        y1_inter = F.silu(gate_inter) * up_inter
        _assert_close(y1_check[s:end], y1_inter, atol=1e-5, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 (bonus): Gold backward self-consistency
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", SEEDS[:1], ids=[f"seed{SEEDS[0]}"])
def test_gold_backward_self_consistency(seed):
    """Gold backward matches torch.autograd.grad on the gold forward."""
    paddle.seed(seed)

    T_tiny, H_tiny, I_tiny, E_tiny, K_tiny = 4, 32, 16, 2, 2
    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(
        T_tiny, H_tiny, I_tiny, E_tiny, K_tiny, seed
    )

    # Make tensors require grad for autograd
    x_ag = x.float().detach().requires_grad_(True)
    w1_ag = w1_split.detach().requires_grad_(True)
    w2_ag = w2.detach().requires_grad_(True)

    # Forward with autograd
    o_ag = _torch_moe_gold(x_ag.to(torch.bfloat16), w1_ag, w2_ag, topk_indices, topk_scores)
    grad_out = torch.randn_like(o_ag)
    o_ag.backward(grad_out)
    dw1_ag = w1_ag.grad.clone()
    dw2_ag = w2_ag.grad.clone()

    # Our manual backward
    dw1_manual, dw2_manual, _ = _torch_moe_gold_backward(
        x, w1_split, w2, topk_indices, topk_scores, grad_out
    )

    # Compare
    r_dw1 = rrmse(dw1_manual, dw1_ag)
    r_dw2 = rrmse(dw2_manual, dw2_ag)
    print(f"  Gold backward: dw1 RRMSE={r_dw1:.6f}, dw2 RRMSE={r_dw2:.6f}")
    assert r_dw1 < 0.001, f"dw1 RRMSE {r_dw1:.6f} >= 0.001"
    assert r_dw2 < 0.001, f"dw2 RRMSE {r_dw2:.6f} >= 0.001"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: routing with specific active experts
# ═══════════════════════════════════════════════════════════════════════════

def _make_sparse_routing(T, E, K, active_experts, device="cuda"):
    """Generate routing where only `active_experts` receive tokens (rest get 0).

    Round-robins tokens across active_experts for uniform 128-aligned segments.
    Requires T*K % len(active_experts) == 0.
    """
    active = list(active_experts)
    n_active = len(active)
    topk_indices = torch.zeros(T, K, dtype=torch.int32, device=device)
    for t in range(T):
        for k in range(K):
            topk_indices[t, k] = active[(t * K + k) % n_active]
    paddle.seed(42)  # fixed seed for routing scores
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    gathered = logits.gather(1, topk_indices.long())
    topk_scores = F.softmax(gathered, dim=-1)
    return topk_indices, topk_scores


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Empty experts — BF16 forward + backward
# ═══════════════════════════════════════════════════════════════════════════

# Scenarios: (T, H, I, E, K, active_experts)
_EMPTY_EXPERT_SCENARIOS = [
    pytest.param(512, 768, 384, 8, 2, [0, 2, 5, 7], id="half-empty"),
    pytest.param(256, 768, 384, 8, 2, [0], id="single-expert-K2"),
    pytest.param(512, 768, 384, 8, 1, [3], id="single-expert-K1"),
    pytest.param(512, 768, 384, 8, 2, [0, 1], id="only-two-active"),
]


@pytest.mark.parametrize("T, H, I, E, K, active_experts", _EMPTY_EXPERT_SCENARIOS)
def test_empty_experts_bf16_forward(T, H, I, E, K, active_experts):
    """BF16 forward produces correct output when some experts receive 0 tokens."""
    paddle.seed(42)
    x = (torch.randn(T, H, device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16)
    w1_split = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float32) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float32) * 0.02

    topk_indices, topk_scores = _make_sparse_routing(T, E, K, active_experts)

    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)
    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)
    o_bf16, _, _, _ = _run_sonicmoe_bf16(x, w1_p, w2_p, topk_indices, topk_scores, E, K)

    r = rrmse(o_bf16, o_gold.to(torch.bfloat16))
    c = cosine_sim(o_bf16, o_gold.to(torch.bfloat16))
    print(f"  BF16 empty-expert fwd: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < BF16_RRMSE, f"RRMSE {r:.6f} >= {BF16_RRMSE}"
    assert c > BF16_COSINE, f"cosine {c:.6f} <= {BF16_COSINE}"


@pytest.mark.parametrize("T, H, I, E, K, active_experts", _EMPTY_EXPERT_SCENARIOS)
def test_empty_experts_bf16_backward(T, H, I, E, K, active_experts):
    """BF16 backward gives zero gradients for unused experts."""
    paddle.seed(42)
    x = (torch.randn(T, H, device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16)
    w1_split = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float32) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float32) * 0.02

    topk_indices, topk_scores = _make_sparse_routing(T, E, K, active_experts)
    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)

    o, dw1, dw2, _ = _run_sonicmoe_bf16(
        x, w1_p, w2_p, topk_indices, topk_scores, E, K, need_grad=True
    )
    assert o is not None, "Forward output is None"
    assert dw1 is not None, "dw1 is None"

    # Empty experts must have exactly zero gradient
    active_set = set(active_experts)
    for e in range(E):
        if e not in active_set:
            assert dw1[e].norm().item() == 0.0, f"dw1[{e}] should be 0 (empty expert)"
            assert dw2[e].norm().item() == 0.0, f"dw2[{e}] should be 0 (empty expert)"
        else:
            assert dw1[e].norm().item() > 0.0, f"dw1[{e}] should be nonzero (active expert)"


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Empty experts — FP8 forward + backward (subprocess)
# ═══════════════════════════════════════════════════════════════════════════

_FP8_EMPTY_EXPERT_TEMPLATE = textwrap.dedent("""\
    import os, sys, json
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    import paddle
    paddle.enable_compat()
    import torch, torch.nn.functional as F
    sys.path.insert(0, {project_root!r})

    from tests.ops.test_moe_module import (
        _convert_weights_for_sonicmoe, _run_sonicmoe_fp8,
        _torch_moe_gold, _make_sparse_routing,
        rrmse, cosine_sim,
    )

    T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
    active_experts = {active_experts!r}
    paddle.seed(42)
    x = (torch.randn(T, H, device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16)
    w1_split = torch.randn(E, 2*I, H, device="cuda", dtype=torch.float32) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float32) * 0.02

    topk_indices, topk_scores = _make_sparse_routing(T, E, K, active_experts)
    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)
    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)

    # Forward
    o_fp8, dw1, dw2, _ = _run_sonicmoe_fp8(
        x, w1_p, w2_p, topk_indices, topk_scores, E, K, need_grad={need_grad!r}
    )
    r = rrmse(o_fp8, o_gold.to(torch.bfloat16))
    c = cosine_sim(o_fp8, o_gold.to(torch.bfloat16))

    result = {{"rrmse": r, "cosine": c}}
    if {need_grad!r} and dw1 is not None:
        # Check empty expert grads
        empty_ok = all(
            dw1[e].norm().item() == 0.0
            for e in range(E) if e not in set(active_experts)
        )
        result["empty_grad_zero"] = empty_ok
    print(json.dumps(result))
""")

# Only test scenarios where segments ARE 128-aligned for FP8
_FP8_EMPTY_EXPERT_SCENARIOS = [
    pytest.param(512, 768, 384, 8, 2, [0, 2, 5, 7], False, id="fwd-half-empty"),
    pytest.param(512, 768, 384, 8, 2, [0, 2, 5, 7], True, id="bwd-half-empty"),
    pytest.param(512, 768, 384, 8, 2, [0, 1], False, id="fwd-two-active"),
    pytest.param(512, 768, 384, 8, 2, [0, 1], True, id="bwd-two-active"),
]


@pytest.mark.parametrize("T, H, I, E, K, active_experts, need_grad", _FP8_EMPTY_EXPERT_SCENARIOS)
def test_empty_experts_fp8(T, H, I, E, K, active_experts, need_grad):
    """FP8 forward/backward with empty experts: output correct, empty grads zero."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _FP8_EMPTY_EXPERT_TEMPLATE.format(
        project_root=project_root, T=T, H=H, I=I, E=E, K=K,
        active_experts=active_experts, need_grad=need_grad,
    )
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    import json
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            metrics = json.loads(line)
            break
    else:
        pytest.fail(f"No JSON output:\n{result.stdout}")

    r = metrics["rrmse"]
    c = metrics["cosine"]
    print(f"  FP8 empty-expert: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < FP8_RRMSE, f"RRMSE {r:.6f} >= {FP8_RRMSE}"
    assert c > FP8_COSINE, f"cosine {c:.6f} <= {FP8_COSINE}"

    if need_grad and "empty_grad_zero" in metrics:
        assert metrics["empty_grad_zero"], "Empty expert gradients should be zero"


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Deterministic — repeated forward gives identical output
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, H, I, E, K", MOE_SHAPES)
def test_deterministic_bf16(T, H, I, E, K):
    """BF16 forward is deterministic: same inputs -> same outputs."""
    paddle.seed(42)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, 42)
    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)

    o1, _, _, _ = _run_sonicmoe_bf16(x, w1_p, w2_p, topk_indices, topk_scores, E, K)
    o2, _, _, _ = _run_sonicmoe_bf16(x, w1_p, w2_p, topk_indices, topk_scores, E, K)

    _assert_close(o1, o2, atol=0, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Large tensor stress test
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, H, I, E, K", [
    pytest.param(4096, 3072, 1536, 8, 8, id="large-production"),
])
def test_large_tensor_bf16_vs_gold(T, H, I, E, K):
    """Large shape BF16 forward matches gold within tolerance."""
    paddle.seed(42)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, 42)
    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)

    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)
    o_bf16, _, _, _ = _run_sonicmoe_bf16(x, w1_p, w2_p, topk_indices, topk_scores, E, K)

    r = rrmse(o_bf16, o_gold.to(torch.bfloat16))
    c = cosine_sim(o_bf16, o_gold.to(torch.bfloat16))
    print(f"  Large BF16 vs Gold: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < BF16_RRMSE, f"RRMSE {r:.6f} >= {BF16_RRMSE}"
    assert c > BF16_COSINE, f"cosine {c:.6f} <= {BF16_COSINE}"


@pytest.mark.parametrize("T, H, I, E, K", [
    pytest.param(4096, 3072, 1536, 8, 8, id="large-production"),
])
def test_large_tensor_fp8_vs_gold(T, H, I, E, K):
    """Large shape FP8 forward matches gold within tolerance (subprocess)."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _FP8_SUBPROCESS_TEMPLATE.format(
        project_root=project_root, T=T, H=H, I=I, E=E, K=K, seed=42
    )
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"Large FP8 subprocess failed:\nstderr: {result.stderr[-500:]}")

    import json
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            metrics = json.loads(line)
            break
    else:
        pytest.fail(f"No JSON output:\n{result.stdout}")

    r = metrics["rrmse"]
    c = metrics["cosine"]
    print(f"  Large FP8 vs Gold: RRMSE={r:.6f}, cosine={c:.6f}")
    assert r < FP8_RRMSE, f"RRMSE {r:.6f} >= {FP8_RRMSE}"
    assert c > FP8_COSINE, f"cosine {c:.6f} <= {FP8_COSINE}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Weight conversion round-trip — split ↔ interleaved
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("I_val", [128, 384, 1536])
def test_weight_conversion_roundtrip(I_val):
    """split->interleaved->split and interleaved->split->interleaved are identity."""
    w = torch.randn(2 * I_val, 768, device="cuda", dtype=torch.float32)

    # Round-trip: split -> interleaved -> split
    w_inter = split_to_interleaved(w)
    w_back = interleaved_to_split(w_inter)
    _assert_close(w, w_back, atol=0, rtol=0)

    # Round-trip: interleaved -> split -> interleaved
    w2 = torch.randn(2 * I_val, 768, device="cuda", dtype=torch.float32)
    w2_split = interleaved_to_split(w2)
    w2_back = split_to_interleaved(w2_split)
    _assert_close(w2, w2_back, atol=0, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Routing metadata correctness — verify cu_seqlens and gather_idx
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, E, K", [
    pytest.param(16, 4, 2, id="tiny"),
    pytest.param(256, 8, 2, id="small"),
    pytest.param(1024, 8, 8, id="large-K"),
])
def test_routing_metadata_correctness(T, E, K):
    """TC_topk_router_metadata_triton produces correct cu_seqlens and gather indices."""
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

    topk_indices, _ = _make_deterministic_routing(T, E, K)
    TK = T * K
    device = "cuda"

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    ef = expert_frequency.cpu()
    efo = expert_frequency_offset.cpu()
    xgi = x_gather_idx.cpu()
    sri = s_reverse_scatter_idx.cpu()

    # cu_seqlens[0] == 0
    assert efo[0].item() == 0, f"cu_seqlens[0]={efo[0].item()} != 0"
    # cu_seqlens[-1] == TK
    assert efo[-1].item() == TK, f"cu_seqlens[-1]={efo[-1].item()} != {TK}"
    # Sum of expert_frequency == TK
    assert ef.sum().item() == TK, f"sum(expert_freq)={ef.sum().item()} != {TK}"
    # expert_frequency[e] == efo[e+1] - efo[e]
    for e in range(E):
        assert ef[e].item() == efo[e + 1].item() - efo[e].item()

    # x_gather_idx maps to valid token indices [0, T)
    assert xgi.min().item() >= 0
    assert xgi.max().item() < T

    # s_reverse_scatter_idx is a permutation of [0, TK)
    assert sorted(sri.tolist()) == list(range(TK))

    # Verify expert assignment: for sorted position p in expert e's segment,
    # the original token at x_gather_idx[p] must have expert e in its topk
    flat_experts = topk_indices.cpu().reshape(-1)
    for e in range(E):
        s = efo[e].item()
        end = efo[e + 1].item()
        for p in range(s, end):
            token_id = xgi[p].item()
            token_experts = topk_indices[token_id].cpu().tolist()
            assert e in token_experts, (
                f"sorted pos {p} in expert {e}'s segment, but token {token_id} "
                f"has experts {token_experts}"
            )


@pytest.mark.parametrize("T, E, K, active_experts", [
    pytest.param(32, 8, 2, [0, 3, 7], id="sparse"),
    pytest.param(32, 8, 1, [5], id="single-expert"),
])
def test_routing_metadata_empty_experts(T, E, K, active_experts):
    """Routing metadata correctly handles experts with 0 assigned tokens."""
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

    topk_indices, _ = _make_sparse_routing(T, E, K, active_experts)
    TK = T * K
    device = "cuda"

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    ef = expert_frequency.cpu()
    efo = expert_frequency_offset.cpu()

    # Empty experts must have frequency 0 and equal consecutive cu_seqlens
    active_set = set(active_experts)
    for e in range(E):
        if e not in active_set:
            assert ef[e].item() == 0, f"expert {e} should have 0 tokens, got {ef[e].item()}"
            assert efo[e].item() == efo[e + 1].item(), (
                f"expert {e} cu_seqlens not equal: {efo[e].item()} != {efo[e+1].item()}"
            )
        else:
            assert ef[e].item() > 0, f"active expert {e} should have >0 tokens"


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: BF16 backward gradient correctness vs gold
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T, H, I, E, K", [
    pytest.param(256, 768, 384, 8, 2, id="small"),
])
@pytest.mark.parametrize("seed", SEEDS[:1], ids=[f"seed{SEEDS[0]}"])
def test_sonicmoe_bf16_backward_vs_gold(T, H, I, E, K, seed):
    """SonicMoE BF16 weight gradients match gold backward within tolerance."""
    paddle.seed(seed)

    x, w1_split, w2, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, seed)
    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)

    o_bf16, dw1_bf16, dw2_bf16, grad_out = _run_sonicmoe_bf16(
        x, w1_p, w2_p, topk_indices, topk_scores, E, K, need_grad=True
    )

    # Gold backward (using same grad_out)
    dw1_gold, dw2_gold, _ = _torch_moe_gold_backward(
        x, w1_split, w2, topk_indices, topk_scores, grad_out.float()
    )

    # dw1_bf16 is (E, 2I, H) interleaved; dw1_gold is (E, 2I, H) split-half
    # Convert gold to interleaved for comparison
    dw1_gold_inter = torch.stack([
        split_to_interleaved(dw1_gold[e]) for e in range(E)
    ])

    r_dw1 = rrmse(dw1_bf16.float(), dw1_gold_inter)
    r_dw2 = rrmse(dw2_bf16.float(), dw2_gold.to(torch.bfloat16))
    c_dw1 = cosine_sim(dw1_bf16.float(), dw1_gold_inter)
    print(f"  BF16 backward: dw1 RRMSE={r_dw1:.6f}, dw2 RRMSE={r_dw2:.6f}, dw1 cosine={c_dw1:.6f}")
    assert r_dw1 < BF16_DW_RRMSE, f"dw1 RRMSE {r_dw1:.6f} >= {BF16_DW_RRMSE}"
    assert r_dw2 < BF16_DW_RRMSE, f"dw2 RRMSE {r_dw2:.6f} >= {BF16_DW_RRMSE}"


def test_down_projection_router_score_bridge_bit_exact():
    """The fused DeepEP score edge matches the standalone carrier exactly."""
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import _DownProjection, _UpProjection
    from sonicmoe.functional import utils as _fp8_utils
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import _gather_router_scores_i32, _scatter_router_scores_i32

    T, H, I, E, K = 256, 768, 384, 8, 2
    TK = T * K
    N_PAD = 128
    x, w1_split, w2, topk_indices, source_values = _make_test_data(T, H, I, E, K, seed=314159)
    w1_param, w2_param = _convert_weights_for_sonicmoe(w1_split, w2)
    # The current functional API receives expert-major Sonic weights directly.
    w1_func = w1_param
    w2_func = w2_param
    w2_func.stop_gradient = False

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device="cuda")
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device="cuda")
    expert_frequency = torch.empty(E, dtype=torch.int32, device="cuda")
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device="cuda")
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device="cuda")
    TC_topk_router_metadata_triton(
        topk_indices,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )
    token_offsets = torch.arange(0, TK + 1, K, dtype=torch.int32, device="cuda")

    old_env = os.environ.get("SONIC_MOE_FP8_MODE")
    old_fp8_active = _fp8_utils._IS_FP8_ACTIVE
    os.environ["SONIC_MOE_FP8_MODE"] = "off"
    _fp8_utils._IS_FP8_ACTIVE = False
    try:
        y1, z = _UpProjection.apply(
            x,
            w1_func,
            None,
            expert_frequency_offset,
            TK,
            K,
            0,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,
            ActivationType.SWIGLU,
            False,
            False,
        )
        y1 = y1.detach()
        z = z.detach()
        z.stop_gradient = False

        # DeepEP may reorder received rows.  Reverse order makes an accidental
        # identity mapping visible while keeping every source index unique.
        score_src_idx = torch.arange(TK - 1, -1, -1, dtype=torch.int32, device="cuda")
        metadata_values = _gather_router_scores_i32(source_values.detach().reshape([-1]), score_src_idx)
        # Real DeepEP metadata is block-aligned and can contain score padding.
        # Keep source indices route-sized while exercising a larger score buffer.
        metadata_values = torch.cat(
            [
                metadata_values,
                torch.zeros(N_PAD, dtype=metadata_values.dtype, device="cuda"),
            ]
        )
        grad_out = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")

        metadata_reference = metadata_values.detach().clone()
        metadata_reference.stop_gradient = False
        z_reference = z.detach().clone()
        z_reference.stop_gradient = False
        w2_reference = w2_func.detach().clone()
        w2_reference.stop_gradient = False
        out_reference = _DownProjection.apply(
            y1,
            z_reference,
            w2_reference,
            None,
            metadata_reference,
            s_scatter_idx,
            expert_frequency_offset,
            T,
            K,
            0,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            token_offsets,
            True,
            ActivationType.SWIGLU,
            None,
        )
        out_reference.backward(grad_out)
        expected_source_grad = _scatter_router_scores_i32(
            metadata_reference.grad.contiguous(), score_src_idx, TK
        ).reshape([T, K])
        expected_z_grad = z_reference.grad.detach().clone()
        expected_w2_grad = w2_reference.grad.detach().clone()

        source_fused = source_values.detach().clone()
        source_fused.stop_gradient = False
        metadata_fused = metadata_values.detach().clone()
        z_fused = z.detach().clone()
        z_fused.stop_gradient = False
        w2_fused = w2_func.detach().clone()
        w2_fused.stop_gradient = False
        out_fused = _DownProjection.apply(
            y1,
            z_fused,
            w2_fused,
            None,
            metadata_fused,
            s_scatter_idx,
            expert_frequency_offset,
            T,
            K,
            0,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            token_offsets,
            True,
            ActivationType.SWIGLU,
            None,
            None,
            None,
            source_fused,
            score_src_idx,
        )
        out_fused.backward(grad_out)

        assert bool(torch.all(out_fused.view(torch.uint8) == out_reference.view(torch.uint8)).item())
        assert bool(torch.all(source_fused.grad.view(torch.uint8) == expected_source_grad.view(torch.uint8)).item())
        assert bool(torch.all(z_fused.grad.view(torch.uint8) == expected_z_grad.view(torch.uint8)).item())
        assert bool(torch.all(w2_fused.grad.view(torch.uint8) == expected_w2_grad.view(torch.uint8)).item())
        assert metadata_fused.grad is None
    finally:
        _fp8_utils._IS_FP8_ACTIVE = old_fp8_active
        if old_env is None:
            os.environ.pop("SONIC_MOE_FP8_MODE", None)
        else:
            os.environ["SONIC_MOE_FP8_MODE"] = old_env


# ═══════════════════════════════════════════════════════════════════════════
# Test 14: Gold reference handles all-same-expert routing
# ═══════════════════════════════════════════════════════════════════════════

def test_gold_all_same_expert():
    """Gold forward works when all tokens are routed to the same expert."""
    T, H, I, E, K = 8, 64, 32, 4, 1
    x = (torch.randn(T, H, device="cuda") * 0.02).to(torch.bfloat16)
    w1 = torch.randn(E, 2 * I, H, device="cuda") * 0.02
    w2 = torch.randn(E, H, I, device="cuda") * 0.02

    topk_indices = torch.zeros(T, K, dtype=torch.int32, device="cuda")
    topk_scores = torch.ones(T, K, device="cuda")

    o = _torch_moe_gold(x, w1, w2, topk_indices, topk_scores)
    assert o.shape == (T, H)
    assert not torch.isnan(o).any(), "Output contains NaN"
    assert not torch.isinf(o).any(), "Output contains Inf"


# ═══════════════════════════════════════════════════════════════════════════
# Test 15: Numerical stability — large activation magnitudes
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_stability_activation_scale(scale):
    """BF16 forward remains stable with different activation scales."""
    T, H, I, E, K = 256, 768, 384, 8, 2

    x = (torch.randn(T, H, device="cuda") * scale).to(torch.bfloat16)
    w1_split = torch.randn(E, 2 * I, H, device="cuda") * 0.02
    w2 = torch.randn(E, H, I, device="cuda") * 0.02

    topk_indices, topk_scores = _make_deterministic_routing(T, E, K)
    o_gold = _torch_moe_gold(x, w1_split, w2, topk_indices, topk_scores)

    w1_p, w2_p = _convert_weights_for_sonicmoe(w1_split, w2)
    o_bf16, _, _, _ = _run_sonicmoe_bf16(x, w1_p, w2_p, topk_indices, topk_scores, E, K)

    assert not torch.isnan(o_bf16).any(), "BF16 output contains NaN"
    assert not torch.isinf(o_bf16).any(), "BF16 output contains Inf"
    r = rrmse(o_bf16, o_gold.to(torch.bfloat16))
    # Relax threshold for larger activations
    threshold = BF16_RRMSE * (1 + scale)
    assert r < threshold, f"RRMSE {r:.6f} >= {threshold:.6f} at scale={scale}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 16: Route-level padding — FP8 non-aligned fwd+bwd (subprocess)
# ═══════════════════════════════════════════════════════════════════════════

_FP8_ROUTE_PAD_TEMPLATE = textwrap.dedent("""\
    import os, sys, json
    os.environ["USE_QUACK_GEMM"] = "1"
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    import torch, torch.nn.functional as F
    sys.path.insert(0, {project_root!r})

    from tests.ops.test_moe_module import (
        _make_test_data, _convert_weights_for_sonicmoe,
        _torch_moe_gold, _torch_moe_gold_backward,
        rrmse, cosine_sim, split_to_interleaved,
    )
    from sonicmoe.functional import (
        moe_TC_softmax_topk_layer,
        clear_all_fp8_weight_caches,
        _refresh_fp8_config,
    )
    from sonicmoe.functional.utils import enable_fp8
    from sonicmoe.enums import ActivationType

    T, H, I, E, K, seed = {T}, {H}, {I}, {E}, {K}, {seed}
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x, w1_split, w2_gold, topk_indices, topk_scores = _make_test_data(T, H, I, E, K, seed)
    w1_param, w2_param = _convert_weights_for_sonicmoe(w1_split, w2_gold)

    # Router weights — routing will differ from deterministic, but that's OK
    router_w = torch.randn(E, H, device="cuda", dtype=torch.bfloat16) * 0.01

    x_input = x.detach().requires_grad_(True)
    w1_func = w1_param.detach().requires_grad_(True)
    w2_func = w2_param.detach().requires_grad_(True)

    # Functional layout: parameter (E, 2I, H) -> functional (2I, H, E)
    w1_f = w1_func.permute(1, 2, 0)
    w2_f = w2_func.permute(1, 2, 0)

    clear_all_fp8_weight_caches()
    with enable_fp8(True):
        _refresh_fp8_config()
        o, router_logits, expert_freq = moe_TC_softmax_topk_layer(
            x_input, router_w, w1_f, None, w2_f, None,
            K=K, stream_id=0, activation_type=ActivationType.SWIGLU,
        )

    assert o.shape == (T, H), f"Output shape mismatch: {{o.shape}} vs {{(T, H)}}"
    assert not torch.isnan(o).any(), "Output contains NaN"
    assert not torch.isinf(o).any(), "Output contains Inf"

    # Run backward
    grad_out = torch.randn_like(o)
    o.backward(grad_out)

    assert w1_func.grad is not None, "w1 gradient is None"
    assert w2_func.grad is not None, "w2 gradient is None"
    assert not torch.isnan(w1_func.grad).any(), "w1 gradient contains NaN"
    assert not torch.isnan(w2_func.grad).any(), "w2 gradient contains NaN"

    dw1_norm = w1_func.grad.float().norm().item()
    dw2_norm = w2_func.grad.float().norm().item()
    assert dw1_norm > 1e-8, f"dw1 norm too small: {{dw1_norm}}"
    assert dw2_norm > 1e-8, f"dw2 norm too small: {{dw2_norm}}"

    # Compare FP8 (route-padded) vs BF16 using same router_w (same routing)
    x_bf16 = x.detach().requires_grad_(True)
    w1_bf16 = w1_param.detach().requires_grad_(True)
    w2_bf16 = w2_param.detach().requires_grad_(True)
    w1_bf16_f = w1_bf16.permute(1, 2, 0)
    w2_bf16_f = w2_bf16.permute(1, 2, 0)

    o_bf16, _, _ = moe_TC_softmax_topk_layer(
        x_bf16, router_w, w1_bf16_f, None, w2_bf16_f, None,
        K=K, stream_id=0, activation_type=ActivationType.SWIGLU,
    )
    o_bf16.backward(grad_out.detach())

    r_out = rrmse(o.float(), o_bf16.float())
    c_out = cosine_sim(o.float(), o_bf16.float())
    r_dw1 = rrmse(w1_func.grad.float(), w1_bf16.grad.float())
    r_dw2 = rrmse(w2_func.grad.float(), w2_bf16.grad.float())

    print(json.dumps({{
        "rrmse_out": round(r_out, 6),
        "cosine_out": round(c_out, 6),
        "rrmse_dw1": round(r_dw1, 6),
        "rrmse_dw2": round(r_dw2, 6),
        "dw1_norm": round(dw1_norm, 6),
        "dw2_norm": round(dw2_norm, 6),
    }}))
""")


@pytest.mark.parametrize("T,H,I,E,K", [
    pytest.param(300, 768, 384, 32, 8, id="non-aligned-E32"),
    pytest.param(256, 768, 384, 8, 2, id="aligned-regression"),
])
def test_fp8_route_padded_fwd_bwd(T, H, I, E, K):
    """FP8 route-level padding: fwd+bwd produces valid output and gradients.

    E=32,K=8 with T=300 gives per_expert~75 tokens — non-128-aligned.
    Route-level padding pads metadata once so aligned fast path runs.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _FP8_ROUTE_PAD_TEMPLATE.format(
        project_root=project_root, T=T, H=H, I=I, E=E, K=K, seed=42
    )
    env = os.environ.copy()
    env["USE_QUACK_GEMM"] = "1"
    env["SONIC_MOE_FP8_MODE"] = "perf"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"FP8 route-pad subprocess failed:\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")

    import json
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            metrics = json.loads(line)
            break
    else:
        pytest.fail(f"No JSON output from route-pad subprocess:\n{result.stdout}")

    r_out = metrics["rrmse_out"]
    c_out = metrics["cosine_out"]
    r_dw1 = metrics["rrmse_dw1"]
    r_dw2 = metrics["rrmse_dw2"]
    print(f"  FP8 route-pad: out RRMSE={r_out:.6f}, cosine={c_out:.6f}")
    print(f"  FP8 route-pad: dw1 RRMSE={r_dw1:.6f}, dw2 RRMSE={r_dw2:.6f}")
    assert r_out < FP8_RRMSE, f"output RRMSE {r_out:.6f} >= {FP8_RRMSE}"
    assert c_out > FP8_COSINE, f"output cosine {c_out:.6f} <= {FP8_COSINE}"
    assert r_dw1 < FP8_DW_RRMSE, f"dw1 RRMSE {r_dw1:.6f} >= {FP8_DW_RRMSE}"
    assert r_dw2 < FP8_DW_RRMSE, f"dw2 RRMSE {r_dw2:.6f} >= {FP8_DW_RRMSE}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 17: Axiomatic _pad_routing_metadata correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_pad_routing_metadata_axiomatic():
    """Verify _pad_routing_metadata preserves every token — no drops, no misdirects.

    Axiom: for identical x, weights, routing, the MoE output must satisfy
        o_padded[t] == o_unpadded[t]  ∀ t ∈ [0, T)
    where o_unpadded is computed by a per-expert pure-Python gold reference
    and o_padded uses the padded routing metadata through the same per-expert
    gold computation (to isolate padding logic from FP8/CUTLASS).

    Uses a small shape (T=10, E=3, K=2) so every value can be enumerated.
    """
    from sonicmoe.functional import _pad_routing_metadata

    T, H, I, E, K = 10, 16, 8, 3, 2
    device = "cuda"
    paddle.seed(999)

    x = torch.randn(T, H, device=device, dtype=torch.float32) * 0.1
    # Split-half weights per expert: (E, 2I, H) and (E, H, I)
    w1 = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32) * 0.1
    w2 = torch.randn(E, H, I, device=device, dtype=torch.float32) * 0.1

    # Deterministic routing: token t → experts [(t*K+k) % E for k in 0..K-1]
    topk_indices = torch.zeros(T, K, dtype=torch.int32, device=device)
    for t in range(T):
        for k in range(K):
            topk_indices[t, k] = (t * K + k) % E
    topk_scores = torch.rand(T, K, device=device, dtype=torch.float32)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # normalize

    # ── Compute routing metadata (CPU, int64 for correctness) ──
    cu, x_gather, s_reverse = _compute_routing_metadata(topk_indices, E)
    cu_cpu = cu.cpu().tolist()
    TK = T * K

    # Build s_scatter_idx: for sorted position p, which flat-topk position?
    # s_scatter_idx[sorted_pos] = flat_topk_pos such that s_reverse[flat_topk_pos] = sorted_pos
    s_scatter = torch.empty(TK, dtype=torch.int32, device=device)
    for flat_pos in range(TK):
        sorted_pos = s_reverse[flat_pos].item()
        s_scatter[sorted_pos] = flat_pos
    efo = cu.to(torch.int32)

    # ── Gold: per-expert forward (pure Python, no padding) ──
    x_gathered = x[x_gather.long()]  # (TK, H)
    y1_all = torch.zeros(TK, I, dtype=torch.float32, device=device)
    y2_all = torch.zeros(TK, H, dtype=torch.float32, device=device)
    for e in range(E):
        s, end = cu_cpu[e], cu_cpu[e + 1]
        if s >= end:
            continue
        z_e = x_gathered[s:end] @ w1[e].T  # (n, 2I)
        gate, up = z_e[:, :I], z_e[:, I:]
        y1_all[s:end] = F.silu(gate) * up
        y2_all[s:end] = y1_all[s:end] @ w2[e].T  # (n, H)

    # Scatter with scores
    o_gold = torch.zeros(T, H, dtype=torch.float32, device=device)
    for t in range(T):
        for k in range(K):
            flat = t * K + k
            sp = s_reverse[flat].item()
            o_gold[t] += y2_all[sp] * topk_scores[t, k]

    # ── Pad routing metadata ──
    (pefo, pxg, pss, psr, pscores, pt, padded) = _pad_routing_metadata(
        efo, x_gather, s_scatter, s_reverse,
        topk_scores.flatten(), TK, T, E, K,
    )
    assert padded, f"Expected padding for T={T}, E={E} but got no padding"
    pefo_cpu = pefo.cpu().tolist()

    # ── Verify structural invariants ──
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import _get_padding_plan
    _, _, _, dst_idx = _get_padding_plan(efo, TK)

    # (A) All real tokens preserved at dst_idx positions
    assert (pxg[dst_idx] == x_gather).all().item(), "x_gather_idx: real tokens not preserved"

    # (B) Original scores preserved
    assert (pscores[:TK] == topk_scores.flatten()).all().item(), "scores: real scores not preserved"

    # (C) Padding scores are zero
    assert (pscores[TK:] == 0).all().item(), "scores: padding scores not zero"

    # (D) All padded segments are 128-aligned
    for e in range(E):
        seg_len = pefo_cpu[e + 1] - pefo_cpu[e]
        assert seg_len % 128 == 0, f"Expert {e}: padded segment {seg_len} not 128-aligned"

    # (E) s_reverse_scatter_idx stays (T*K,) — same number of real tokens
    assert psr.shape[0] == TK, f"s_reverse_scatter_idx resized: {psr.shape[0]} != {TK}"

    # ── Gold with padded metadata (same per-expert Python, on padded layout) ──
    x_gathered_pad = x[pxg.long()]  # (padded_total, H) — padding rows gather x[0]
    y1_pad = torch.zeros(pt, I, dtype=torch.float32, device=device)
    y2_pad = torch.zeros(pt, H, dtype=torch.float32, device=device)
    for e in range(E):
        s, end = pefo_cpu[e], pefo_cpu[e + 1]
        if s >= end:
            continue
        z_e = x_gathered_pad[s:end] @ w1[e].T
        gate, up = z_e[:, :I], z_e[:, I:]
        y1_pad[s:end] = F.silu(gate) * up
        y2_pad[s:end] = y1_pad[s:end] @ w2[e].T

    # Scatter padded y2 with padded scores → o_padded
    o_padded = torch.zeros(T, H, dtype=torch.float32, device=device)
    for t in range(T):
        for k in range(K):
            flat = t * K + k
            sp = psr[flat].item()          # remapped sorted position in padded space
            score = pscores[flat].item()   # original score (first T*K entries)
            o_padded[t] += y2_pad[sp] * score

    # ── The axiom: o_padded == o_gold for every token ──
    max_diff = (o_padded - o_gold).abs().max().item()
    per_token_diff = [(o_padded[t] - o_gold[t]).abs().max().item() for t in range(T)]
    all_match = all(d < 1e-5 for d in per_token_diff)

    print(f"  pad_routing axiom: max_diff={max_diff:.2e}, all_tokens_match={all_match}")
    print(f"  per_token_max_diff: {per_token_diff}")
    print(f"  padding: TK={TK} -> padded_total={pt}, N_pad={pt - TK}")

    assert all_match, (
        f"Padding changed output for some tokens! max_diff={max_diff:.2e}\n"
        f"per_token: {per_token_diff}"
    )
    # Extra: verify no token was silently zeroed out
    for t in range(T):
        assert o_gold[t].abs().max() > 1e-8, f"Gold output for token {t} is suspiciously zero"
        assert o_padded[t].abs().max() > 1e-8, f"Padded output for token {t} is suspiciously zero"
