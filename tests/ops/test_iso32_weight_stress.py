"""ISO32 weight precision stress test.

Full forward+backward MlpNode pass with iso32 weights vs pair-kernel (1x32) baseline.
Parametrized over production shapes, seeds, and routing patterns.
"""

import os
import pytest
import torch

os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"

from tests.ops.conftest import (
    requires_blackwell,
    requires_quack,
    rrmse,
    cosine_sim,
    GEMM_SHAPES,
    SEEDS,
)

pytestmark = [requires_blackwell, requires_quack]

ROUTING_PATTERNS = [
    pytest.param("uniform", id="uniform"),
    pytest.param("skewed", id="skewed"),
    pytest.param("sparse", id="sparse"),
]


def _make_routing(T, E, K, pattern, device="cuda"):
    if pattern == "uniform":
        indices = torch.randint(0, E, (T, K), device=device, dtype=torch.int32)
    elif pattern == "skewed":
        probs = torch.zeros(E, device=device)
        probs[0] = 0.8
        probs[1:] = 0.2 / (E - 1)
        indices = torch.multinomial(probs.expand(T, -1), K, replacement=False).to(torch.int32)
    elif pattern == "sparse":
        indices = torch.randint(0, E, (T, K), device=device, dtype=torch.int32)
        mask = torch.rand(T, K, device=device) < 0.3
        indices[mask] = -1
    return indices


def _run_gemm_with_iso32_weight(w_enk, x, cu_seqlens, iso32=True):
    """Run a single blockscaled_fp8_gemm_varlen with iso32 or 1x32 weight."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        iso32_dual_quantize_weight_3d,
        _quantize_weight_3d_triton,
        quantize_and_pack_activation,
        blockscaled_fp8_gemm_varlen,
    )
    if iso32:
        w_fp8, w_scales, _ = iso32_dual_quantize_weight_3d(w_enk)
    else:
        w_fp8, w_scales = _quantize_weight_3d_triton(w_enk.contiguous())

    x_fp8, x_scales = quantize_and_pack_activation(x)
    out = blockscaled_fp8_gemm_varlen(
        x_fp8,
        w_enk.permute(1, 2, 0).to(torch.bfloat16),
        cu_seqlens,
        a_scales=x_scales,
        w_fp8=w_fp8,
        w_scales=w_scales,
        out_dtype=torch.bfloat16,
        assume_aligned=True,
    )
    return out


@pytest.mark.parametrize("T,H,I,E,K", GEMM_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_iso32_weight_fwd_precision(T, H, I, E, K, seed):
    """Forward GEMM with iso32 weight matches 1x32 baseline within FP8 tolerance."""
    torch.manual_seed(seed)
    device = "cuda"

    w = torch.randn(E, H, I, dtype=torch.bfloat16, device=device)
    x = torch.randn(T, I, dtype=torch.bfloat16, device=device)
    tokens_per_expert = torch.full((E,), T // E, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(E + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = tokens_per_expert.cumsum(0)

    out_iso32 = _run_gemm_with_iso32_weight(w, x, cu_seqlens, iso32=True)
    out_1x32 = _run_gemm_with_iso32_weight(w, x, cu_seqlens, iso32=False)

    err = rrmse(out_iso32, out_1x32)
    cos = cosine_sim(out_iso32, out_1x32)
    assert err < 0.10, f"ISO32 vs 1x32 fwd RRMSE: {err:.4e}"
    assert cos > 0.99, f"ISO32 vs 1x32 fwd cosine: {cos:.6f}"


@pytest.mark.parametrize("T,H,I,E,K", GEMM_SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_iso32_weight_cache_warmup_roundtrip(T, H, I, E, K, seed):
    """Warmup populates cache; consumer functions return shared-storage views."""
    torch.manual_seed(seed)
    device = "cuda"

    os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "1"
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_warmup,
        precompute_weight_fp8_for_fused_gated,
        precompute_weight_fp8,
        precompute_weight_fp8_for_direct_fused_dgated,
        clear_blockscaled_fp8_weight_cache,
    )

    clear_blockscaled_fp8_weight_cache()
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device=device)

    precompute_weight_fp8_warmup(w1, w2)

    w1_fwd_fp8, w1_fwd_sc = precompute_weight_fp8_for_fused_gated(w1)
    w1_bwd_fp8, w1_bwd_sc = precompute_weight_fp8(w1.permute(1, 0, 2))
    w2_fwd_fp8, w2_fwd_sc = precompute_weight_fp8(w2)
    w2_bwd_fp8, w2_bwd_sc = precompute_weight_fp8_for_direct_fused_dgated(w2)

    assert w1_fwd_fp8.data_ptr() == w1_bwd_fp8.data_ptr(), "w1 forward/backward must share storage"
    assert w2_fwd_fp8.data_ptr() == w2_bwd_fp8.data_ptr(), "w2 forward/backward must share storage"

    assert w1_fwd_fp8.shape[0] == E
    assert w2_fwd_fp8.shape == (E, H, I)
    assert w2_bwd_fp8.shape == (E, I, H)
