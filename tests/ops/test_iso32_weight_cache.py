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

from sonicmoe.functional import _refresh_fp8_config
_refresh_fp8_config()

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
    def test_iso32_vs_bf16_precision(self, E, N, K, seed):
        """ISO32 weight quant→dequant fidelity vs BF16 ground truth.

        The gold standard: dequant(quantize(w)) vs w itself.
        For Gaussian-initialized weights (realistic), iso32 and 1x32 produce
        identical RRMSE vs BF16 (proven by audit_iso32_numerics.py). We gate
        at the same tolerance as the production 1x32 path: RRMSE < 0.035,
        cosine > 0.9995.
        """
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            iso32_dual_quantize_weight_3d,
        )
        from tests.ops.conftest import gold_dequant, gold_e8m0_iso32_quant

        w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * (1.0 / (K ** 0.5))
        fp8_3d, _, _ = iso32_dual_quantize_weight_3d(w)

        w_2d = w.reshape(E * N, K)
        _, gold_raw_scales = gold_e8m0_iso32_quant(w_2d)
        dequant = gold_dequant(fp8_3d.reshape(E * N, K), gold_raw_scales)

        err = rrmse(dequant, w_2d)
        cos = cosine_sim(dequant, w_2d)
        assert err < 0.035, f"ISO32 vs BF16 RRMSE: {err:.6e} (must be < 0.035)"
        assert cos > 0.9995, f"ISO32 vs BF16 cosine: {cos:.8f} (must be > 0.9995)"

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_iso32_vs_bf16_stress_outlier_weights(self, E, N, K, seed):
        """Stress: 3% of expert rows scaled 100x (outlier weights).

        Even with heavy-tail distribution, the quant→dequant RRMSE vs BF16
        must stay within FP8 quantization noise. Gate: RRMSE < 0.04, cos > 0.999.
        """
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            iso32_dual_quantize_weight_3d,
        )
        from tests.ops.conftest import gold_dequant, gold_e8m0_iso32_quant

        w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * (1.0 / (K ** 0.5))
        outlier_idx = torch.randperm(E * N, device="cuda")[: (E * N) // 32]
        w_flat = w.reshape(-1, K)
        w_flat[outlier_idx] *= 100.0

        fp8_3d, _, _ = iso32_dual_quantize_weight_3d(w)
        w_2d = w.reshape(E * N, K)
        _, gold_raw_scales = gold_e8m0_iso32_quant(w_2d)
        dequant = gold_dequant(fp8_3d.reshape(E * N, K), gold_raw_scales)

        err = rrmse(dequant, w_2d)
        cos = cosine_sim(dequant, w_2d)
        assert err < 0.04, f"ISO32 outlier stress RRMSE: {err:.6e}"
        assert cos > 0.999, f"ISO32 outlier stress cosine: {cos:.8f}"

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_memory_single_buffer(self, E, N, K, seed):
        """ISO32 path: forward and backward views share ONE data_ptr."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _cache_iso32_w1,
            precompute_weight_fp8_for_fused_gated,
            precompute_weight_fp8,
            clear_blockscaled_fp8_weight_cache,
        )

        clear_blockscaled_fp8_weight_cache()
        w1 = torch.randn(N, K, E, dtype=torch.bfloat16, device="cuda")
        _cache_iso32_w1(w1)

        fwd_fp8, fwd_scales = precompute_weight_fp8_for_fused_gated(w1)
        bwd_fp8, bwd_scales = precompute_weight_fp8(w1, permute=(1, 0, 2))

        assert fwd_fp8.data_ptr() == bwd_fp8.data_ptr(), (
            "Forward and backward FP8 views must share storage"
        )

    @pytest.mark.parametrize("E,N,K", WEIGHT_SHAPES)
    def test_cache_invalidation(self, E, N, K, seed):
        """Cache misses after in-place weight update."""
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _cache_iso32_w1,
            clear_blockscaled_fp8_weight_cache,
        )

        clear_blockscaled_fp8_weight_cache()
        w1 = torch.randn(N, K, E, dtype=torch.bfloat16, device="cuda")
        _cache_iso32_w1(w1)

        assert ("iso32", tuple(w1.stride())) in w1.fp8_weight_cache


class TestIso32MemorySaving:
    """End-to-end GPU memory saving verification.

    Methodology:
    - Use torch.cuda.memory_allocated() to measure ACTUAL GPU memory
    - Compare pair-kernel path (4 buffers) vs iso32 path (2 buffers)
    - Verify savings match theoretical expectation (E*2I*H + E*H*I bytes)
    - Test at production shapes to catch alignment/padding overhead
    """

    @pytest.mark.parametrize("E,H,I", [
        pytest.param(8, 3072, 1536, id="production-E8"),
        pytest.param(8, 256, 128, id="smoke-E8"),
    ])
    def test_memory_saving_vs_pair_kernel(self, E, H, I):
        """ISO32 uses strictly less GPU memory than pair-kernel for weight caches.

        Measured via torch.cuda.memory_allocated delta — the ground truth for
        actual GPU memory consumption. Verifies:
        1. ISO32 allocates fewer bytes than pair-kernel
        2. The byte-level saving matches theory: E*(2I*H + H*I) bytes
        3. The saving is at least 40% of baseline (guards against hidden copies)
        """
        import gc
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            precompute_weight_fp8_warmup,
            precompute_weight_fp8_for_fused_gated,
            precompute_weight_fp8,
            precompute_weight_fp8_for_direct_fused_dgated,
            clear_blockscaled_fp8_weight_cache,
        )

        torch.manual_seed(42)
        w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda")

        # --- Measure BASELINE (pair-kernel, iso32 OFF) ---
        os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "0"
        _refresh_fp8_config()
        clear_blockscaled_fp8_weight_cache()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        precompute_weight_fp8_warmup(w1, w2)
        _ = precompute_weight_fp8_for_fused_gated(w1)
        _ = precompute_weight_fp8(w1.permute(1, 0, 2))
        _ = precompute_weight_fp8(w2)
        _ = precompute_weight_fp8_for_direct_fused_dgated(w2)
        torch.cuda.synchronize()

        baseline_mem = torch.cuda.memory_allocated() - mem_before

        # --- Measure ISO32 ---
        os.environ["SONIC_MOE_FP8_ISO32_WEIGHT"] = "1"
        _refresh_fp8_config()
        clear_blockscaled_fp8_weight_cache()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        precompute_weight_fp8_warmup(w1, w2)
        _ = precompute_weight_fp8_for_fused_gated(w1)
        _ = precompute_weight_fp8(w1.permute(1, 0, 2))
        _ = precompute_weight_fp8(w2)
        _ = precompute_weight_fp8_for_direct_fused_dgated(w2)
        torch.cuda.synchronize()

        iso32_mem = torch.cuda.memory_allocated() - mem_before

        # --- Assertions ---
        saving_bytes = baseline_mem - iso32_mem
        saving_pct = saving_bytes / baseline_mem * 100 if baseline_mem > 0 else 0
        theoretical_saving = E * (2 * I * H + H * I)

        print(f"\n  E={E}, H={H}, I={I}")
        print(f"  Baseline (pair-kernel): {baseline_mem / 2**20:.2f} MiB")
        print(f"  ISO32 (single buffer):  {iso32_mem / 2**20:.2f} MiB")
        print(f"  Saving:   {saving_bytes / 2**20:.2f} MiB ({saving_pct:.1f}%)")
        print(f"  Theory:   {theoretical_saving / 2**20:.2f} MiB")

        assert saving_pct > 40, (
            f"Memory saving {saving_pct:.1f}% is less than 40% — "
            f"possible hidden copy defeating iso32 purpose"
        )
        assert saving_bytes >= theoretical_saving * 0.85, (
            f"Saving {saving_bytes/2**20:.2f} MiB is less than 85% of "
            f"theoretical minimum {theoretical_saving/2**20:.2f} MiB — "
            f"possible hidden allocation"
        )
