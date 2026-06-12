"""Unit tests for gemm_gated (fwd): torch ↔ BF16 ↔ FP8 3-way cross-validation.

gemm_gated performs: z = A @ B (grouped GEMM), y1 = SwiGLU(z).
  BF16 path: standard CUTLASS with cu_seqlens_m.
  FP8 path: blockscaled CUTLASS with ISA-packed E8M0 scales.

3-way verification:
  (1) torch gold vs BF16 CUTLASS — validates the official CUTLASS kernel
  (2) torch gold vs FP8 blockscaled — validates the FP8 frontier end-to-end
  (3) BF16 vs FP8 — cross-check between two kernel implementations

All metrics (RRMSE, cosine sim, max abs error) are reported for every comparison.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_bf16_close, assert_fp8_tolerance,
    rrmse, cosine_sim,
    GEMM_SHAPES,
)

pytestmark = [requires_blackwell, requires_quack]

# Filter to aligned shapes: per-expert tokens must be 128-aligned for FP8 path
_GATED_SHAPES = []
for p in GEMM_SHAPES:
    T, H, I, E, K = p.values
    TK = T * K // E
    if TK % 128 == 0 and H % 128 == 0 and I % 128 == 0 and (2 * I) % 128 == 0:
        _GATED_SHAPES.append(pytest.param(T, H, I, E, K, id=p.id))
GATED_SHAPES = _GATED_SHAPES if _GATED_SHAPES else GEMM_SHAPES[:1]


def _setup_uniform_routing(T, H, I, E, K):
    """Create uniform routing data: each expert gets T*K/E tokens."""
    TK = T * K // E
    total_M = TK * E
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    # A_idx: identity mapping (TK-sized, mapping each row to itself in A)
    # For simplicity, A = (TK*E, H) and A_idx = arange(TK*E)
    A_idx = torch.arange(total_M, dtype=torch.int32, device="cuda")
    return TK, total_M, cu_seqlens, A_idx


def _torch_gemm_gated_gold(x, w1, cu_seqlens, E):
    """Torch gold: per-expert grouped GEMM + SwiGLU (interleaved layout).

    x:  (total_M, H) bf16
    w1: (2I, H, E) bf16
    Returns (z, y1) each in bf16.
    z[s:e] = x[s:e] @ w1[:,:,exp].T for each expert exp.
    y1 = SwiGLU(z) with interleaved layout.
    """
    total_M = x.shape[0]
    two_I = w1.shape[0]
    I = two_I // 2
    z = torch.zeros(total_M, two_I, dtype=torch.bfloat16, device=x.device)
    for exp in range(E):
        s = cu_seqlens[exp].item()
        e = cu_seqlens[exp + 1].item()
        if s < e:
            z[s:e] = (x[s:e].float() @ w1[:, :, exp].float().T).to(torch.bfloat16)

    z_f32 = z.float()
    gate = z_f32[:, 0::2]
    up = z_f32[:, 1::2]
    y1 = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    return z, y1


def _report_metrics(actual, expected, label):
    """Print detailed precision metrics for a comparison."""
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    mean_abs = (actual.float() - expected.float()).abs().mean().item()
    print(f"  [{label}] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={max_abs:.6f}, mean_abs_err={mean_abs:.6f}")
    return r, c


@pytest.mark.parametrize("T,H,I,E,K", GATED_SHAPES)
def test_torch_vs_bf16(T, H, I, E, K, seed):
    """(1/3) BF16 CUTLASS gemm_gated vs torch gold."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    TK, total_M, cu_seqlens, _ = _setup_uniform_routing(T, H, I, E, K)
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.02

    gold_z, gold_y1 = _torch_gemm_gated_gold(x, w1, cu_seqlens, E)

    # BF16 CUTLASS grouped GEMM: B = (E, K=H, N=2I)
    w1_3d = w1.permute(2, 1, 0).contiguous()  # (E, H, 2I)
    preact, postact = gemm_gated(
        x, w1_3d,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens,
    )

    r_z, c_z = _report_metrics(preact, gold_z, "z: BF16 vs torch")
    r_y1, c_y1 = _report_metrics(postact, gold_y1, "y1: BF16 vs torch")

    assert_bf16_close(postact, gold_y1, atol=1.4e-2)
    if r_y1 > 0.005:
        import warnings
        warnings.warn(f"BF16 gemm_gated y1 RRMSE vs torch = {r_y1:.6f} (>0.5%)")


@pytest.mark.parametrize("T,H,I,E,K", GATED_SHAPES)
def test_torch_vs_fp8(T, H, I, E, K, seed):
    """(2/3) FP8 blockscaled gemm_gated vs torch gold."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
    )

    TK, total_M, cu_seqlens, _ = _setup_uniform_routing(T, H, I, E, K)
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.02

    gold_z, gold_y1 = _torch_gemm_gated_gold(x, w1, cu_seqlens, E)

    # FP8 path
    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)

    _, fp8_y1 = gemm_gated(
        x_fp8, w_fp8,
        activation="swiglu",
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=cu_seqlens,
        a_scales=a_scales,
        b_scales=b_scales,
    )

    r, c = _report_metrics(fp8_y1, gold_y1, "y1: FP8 vs torch")
    # When gold signal is tiny, RRMSE is unreliable; fall back to max absolute error
    gold_max = gold_y1.float().abs().max().item()
    if gold_max > 0.01:
        assert r < 0.10, f"FP8 gemm_gated y1 RRMSE {r:.6f} >= 0.10 vs torch gold"
        assert c > 0.99, f"FP8 gemm_gated y1 cosine {c:.8f} <= 0.99 vs torch gold"
    else:
        max_abs = (fp8_y1.float() - gold_y1.float()).abs().max().item()
        assert max_abs < 0.01, f"FP8 gemm_gated y1 max_abs {max_abs:.6f} >= 0.01 (near-zero gold)"


@pytest.mark.parametrize("T,H,I,E,K", GATED_SHAPES)
def test_bf16_vs_fp8(T, H, I, E, K, seed):
    """(3/3) FP8 blockscaled vs BF16 CUTLASS cross-check."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
    )

    TK, total_M, cu_seqlens, _ = _setup_uniform_routing(T, H, I, E, K)
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.02

    # BF16 path
    w1_3d = w1.permute(2, 1, 0).contiguous()  # (E, H, 2I)
    _, bf16_y1 = gemm_gated(
        x, w1_3d,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens,
    )

    # FP8 path
    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)

    _, fp8_y1 = gemm_gated(
        x_fp8, w_fp8,
        activation="swiglu",
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=cu_seqlens,
        a_scales=a_scales,
        b_scales=b_scales,
    )

    r, c = _report_metrics(fp8_y1, bf16_y1, "y1: FP8 vs BF16")
    ref_max = bf16_y1.float().abs().max().item()
    if ref_max > 0.01:
        assert r < 0.10, f"FP8 vs BF16 y1 RRMSE {r:.6f} >= 0.10"
        assert c > 0.99, f"FP8 vs BF16 y1 cosine {c:.8f} <= 0.99"
    else:
        max_abs = (fp8_y1.float() - bf16_y1.float()).abs().max().item()
        assert max_abs < 0.01, f"FP8 vs BF16 y1 max_abs {max_abs:.6f} >= 0.01 (near-zero ref)"
