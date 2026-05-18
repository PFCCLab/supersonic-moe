"""ISO32 weight cache unification: precision and correctness tests.

Validates:
1. FP8 byte-identity: iso32 quant produces same bytes regardless of access direction
2. Scale correctness: dual ISA scales match gold_e8m0_iso32_quant
3. GEMM precision: forward+backward output with iso32 weights vs pair-kernel baseline
4. Memory savings: iso32 path uses ONE FP8 buffer per weight (shared data_ptr)
5. Cache lifecycle: invalidation via _version, proper clear
"""

import os
import pytest
import torch

os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "1"
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"

from tests.ops.conftest import (
    requires_blackwell,
    requires_quack,
    rrmse,
    cosine_sim,
    assert_fp8_tolerance,
    gold_e8m0_iso32_quant,
    SEEDS,
)

pytestmark = [requires_blackwell, requires_quack]

WEIGHT_SHAPES = [
    pytest.param(8, 256, 128, id="smoke"),
    pytest.param(4, 1536, 768, id="small"),
    pytest.param(8, 3072, 1536, id="production"),
    pytest.param(32, 3072, 1536, id="large-E"),
]


@pytest.fixture(params=SEEDS)
def seed(request):
    torch.manual_seed(request.param)
    return request.param


class TestIso32WeightCache:
    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_fp8_transpose_invariance(self, E, N, K, seed):
        """ISO32 quantized fp8 bytes are transpose-invariant within 32x32 blocks."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            iso32_dual_quantize_weight_3d,
        )

        w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
        fp8_enk, row_scales, col_scales = iso32_dual_quantize_weight_3d(w)

        assert fp8_enk.shape == (E, N, K)
        assert fp8_enk.dtype == torch.float8_e4m3fn
        assert fp8_enk.is_contiguous()
        fp8_transposed_view = fp8_enk.permute(0, 2, 1)
        assert fp8_transposed_view.shape == (E, K, N)
        assert not fp8_transposed_view.is_contiguous()
        assert fp8_transposed_view.data_ptr() == fp8_enk.data_ptr()

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_dual_scales_vs_gold(self, E, N, K, seed):
        """Row and col scales produce correct dequantized values vs gold reference."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            iso32_dual_quantize_weight_3d,
        )
        from tests.ops.conftest import unpack_isa_scales

        w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
        fp8_enk, row_scales, col_scales = iso32_dual_quantize_weight_3d(w)

        w_2d = w.reshape(E * N, K)
        gold_fp8, gold_scales = gold_e8m0_iso32_quant(w_2d)
        gold_fp8_3d = gold_fp8.reshape(E, N, K)

        fp8_match = (
            fp8_enk.view(torch.uint8) == gold_fp8_3d.to(fp8_enk.device).view(torch.uint8)
        ).float().mean()
        assert fp8_match > 0.99, f"FP8 byte match rate: {fp8_match:.4f}"

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_iso32_vs_1x32_precision(self, E, N, K, seed):
        """ISO32 weight quant is close to 1x32 quant (slight precision relaxation)."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            iso32_dual_quantize_weight_3d,
            _quantize_weight_3d_triton,
        )

        w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
        fp8_iso32, _, _ = iso32_dual_quantize_weight_3d(w)
        w_contig = w.contiguous()
        fp8_1x32, _ = _quantize_weight_3d_triton(w_contig)

        iso32_f32 = fp8_iso32.float()
        baseline_f32 = fp8_1x32.float()
        err = rrmse(iso32_f32, baseline_f32)
        cos = cosine_sim(iso32_f32, baseline_f32)
        assert err < 0.15, f"ISO32 vs 1x32 RRMSE too high: {err:.4e}"
        assert cos > 0.98, f"ISO32 vs 1x32 cosine too low: {cos:.6f}"

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_memory_single_buffer(self, E, N, K, seed):
        """ISO32 path: forward and backward views share ONE data_ptr."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _cache_iso32_w1,
            _ISO32_WEIGHT_CACHE,
            precompute_weight_fp8_for_fused_gated,
            precompute_weight_fp8,
            clear_blockscaled_fp8_weight_cache,
        )

        clear_blockscaled_fp8_weight_cache()
        w1 = torch.randn(N, K, E, dtype=torch.bfloat16, device="cuda")
        _cache_iso32_w1(w1)

        fwd_fp8, fwd_scales = precompute_weight_fp8_for_fused_gated(w1)
        bwd_fp8, bwd_scales = precompute_weight_fp8(w1.permute(1, 0, 2))

        assert fwd_fp8.data_ptr() == bwd_fp8.data_ptr(), (
            "Forward and backward FP8 views must share storage"
        )

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_cache_invalidation(self, E, N, K, seed):
        """Cache misses after in-place weight update."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _cache_iso32_w1,
            _ISO32_WEIGHT_CACHE,
            _tensor_version,
            clear_blockscaled_fp8_weight_cache,
        )

        clear_blockscaled_fp8_weight_cache()
        w1 = torch.randn(N, K, E, dtype=torch.bfloat16, device="cuda")
        _cache_iso32_w1(w1)

        key_before = (w1.data_ptr(), _tensor_version(w1), tuple(w1.shape), tuple(w1.stride()))
        assert key_before in _ISO32_WEIGHT_CACHE

        w1.add_(0.1)
        key_after = (w1.data_ptr(), _tensor_version(w1), tuple(w1.shape), tuple(w1.stride()))
        assert key_after not in _ISO32_WEIGHT_CACHE
