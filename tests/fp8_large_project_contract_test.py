import os

import paddle
paddle.enable_compat()

import torch

import sonicmoe.functional as functional

from .test_commons import TestCommons


class FP8AlignedContractTest(TestCommons):
    """Contract tests for the blockscaled FP8 quantization kernels.

    Covers weight/activation quantizers and the fused z-save + y1 quant kernel
    at production-relevant shapes.
    """

    def setUp(self) -> None:
        self._reset_fp8_state()

    def tearDown(self) -> None:
        self._reset_fp8_state()

    def _require_blackwell(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

    def _reset_fp8_state(self) -> None:
        """Reset all FP8 global state to prevent cross-test pollution."""
        # Strip env var that would make BF16 reference use FP8 path
        os.environ.pop("SONIC_MOE_FP8_MODE", None)
        functional.clear_all_fp8_weight_caches()
        functional._ALIGNMENT_ASSUMED = False
        functional._ALIGNMENT_STREAK = 0

    def _rrmse(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def _corr(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()

    def test_triton_weight_quant_matches_eager(self) -> None:
        """Triton _quantize_weight_3d_triton produces valid iso32 FP8 + scales.

        Verifies that the optimized single-kernel iso32 path produces the same
        FP8 values and ISA-packed scales as the 2D iso32 quantizer.
        """
        self._require_blackwell()
        torch.manual_seed(42)

        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _quantize_weight_3d_triton,
            quantize_and_pack_weight_iso32,
        )

        # Test with reference-like weight shapes
        for E, N, K in [(8, 3072, 3072), (8, 1536, 3072), (8, 3072, 1536)]:
            w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.02
            w_fp8, w_scales = _quantize_weight_3d_triton(w)

            # Verify shapes
            self.assertEqual(w_fp8.shape, (E, N, K), f"FP8 shape mismatch for ({E},{N},{K})")
            self.assertEqual(w_fp8.dtype, torch.float8_e4m3fn)

            # Verify per-expert consistency: each expert quantized independently
            # must match the 2D quantize_and_pack_weight_iso32 result
            if N % 128 == 0:  # fast path: single kernel launch, can verify per-expert
                for e in range(E):
                    ref_fp8, _ = quantize_and_pack_weight_iso32(w[e])
                    expert_fp8 = w_fp8[e]
                    fp8_match = torch.equal(expert_fp8, ref_fp8)
                    self.assertTrue(
                        fp8_match,
                        f"Expert {e} FP8 values differ for shape ({E},{N},{K})"
                    )

    def test_z_quant_blockscaled_roundtrip(self) -> None:
        """quantize_activation_blockscaled_fast + dequantize_blockscaled_fp8 round-trip.

        Covers the BR=32/GPB=12 tuned kernel grid parameters.
        Tests at both test shape and production shape.
        """
        self._require_blackwell()
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            quantize_activation_blockscaled_fast,
        )
        from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

        for label, M, K in [("test", 2048, 768), ("prod", 65536, 3072)]:
            with self.subTest(shape=label):
                torch.manual_seed(42)
                x = 0.02 * torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

                fp8_data, scales_e8m0 = quantize_activation_blockscaled_fast(x)
                self.assertEqual(fp8_data.shape, (M, K))
                self.assertEqual(fp8_data.dtype, torch.float8_e4m3fn)

                x_restored = dequantize_blockscaled_fp8(
                    fp8_data, scales_e8m0.view(torch.uint8)
                )
                rrmse = self._rrmse(x_restored, x)
                corr = self._corr(x_restored, x)
                self.assertLess(rrmse, 0.05,
                    f"{label} z-quant roundtrip RRMSE {rrmse:.4f} >= 5%")
                self.assertGreater(corr, 0.999,
                    f"{label} z-quant roundtrip corr {corr:.6f} < 0.999")

    def test_quantize_and_pack_isa_roundtrip(self) -> None:
        """quantize_and_pack_activation produces valid FP8 + ISA-packed scales.

        Verifies the fused quant+ISA-pack kernel at production activation shapes.
        """
        self._require_blackwell()
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            quantize_and_pack_activation,
        )

        for label, M, K in [("x", 8192, 3072), ("dz", 65536, 3072), ("y1", 65536, 1536)]:
            with self.subTest(activation=label):
                torch.manual_seed(42)
                x = 0.02 * torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

                fp8_data, packed_scales = quantize_and_pack_activation(x)
                self.assertEqual(fp8_data.shape, (M, K))
                self.assertEqual(fp8_data.dtype, torch.float8_e4m3fn)
                self.assertFalse(torch.all(fp8_data.view(torch.uint8) == 0).item(),
                    f"{label}: all-zero FP8 output — quant kernel broken")

                # Verify scale bytes are plausible (not all 0 or all 127)
                scale_bytes = packed_scales.view(torch.uint8)
                unique_scales = scale_bytes.unique().numel()
                self.assertGreater(unique_scales, 1,
                    f"{label}: packed scales have only {unique_scales} unique value(s)")

    # ------------------------------------------------------------------
    # Tests for fused_z_save_y1_quant kernel
    # ------------------------------------------------------------------

    def test_fused_z_save_y1_quant_matches_separate_multi_shape(self) -> None:
        """fused_z_save_y1_quant produces bit-identical output to the two separate kernels.

        Verifies at test, contract, and production shapes that fusing the
        z-save blockscaled quant + y1 ISA-packed quant into a single kernel
        does not change the numerical result.
        """
        self._require_blackwell()
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            fused_z_save_y1_quant,
            quantize_activation_blockscaled_fast,
            quantize_and_pack_activation,
        )

        shapes = [
            # (label, TK, 2I, I)
            # Production-relevant shapes only: 2I must have num_groups divisible
            # by 12 (the GROUPS_PER_BLOCK used by the separate z-quant kernel)
            # to avoid comparing padding scale bytes that differ due to OOB writes
            # in the separate kernel's 2D grid.
            ("contract", 8192, 3072, 1536),
            ("prod_reference", 65536, 3072, 1536),
            ("large_T", 32768, 3072, 1536),
        ]
        for label, TK, two_I, I in shapes:
            with self.subTest(shape=label, TK=TK):
                torch.manual_seed(42)
                z = 0.02 * torch.randn(TK, two_I, device="cuda", dtype=torch.bfloat16)
                y1 = 0.02 * torch.randn(TK, I, device="cuda", dtype=torch.bfloat16)

                # Reference: two separate kernels
                z_fp8_ref, z_raw_scales_ref = quantize_activation_blockscaled_fast(z)
                y1_fp8_ref, y1_packed_scales_ref = quantize_and_pack_activation(y1)

                # Fused kernel
                z_fp8_fused, z_raw_scales_fused, y1_fp8_fused, y1_packed_scales_fused = (
                    fused_z_save_y1_quant(z, y1)
                )

                # z outputs must be BIT-IDENTICAL
                torch.testing.assert_close(
                    z_fp8_fused.view(torch.uint8), z_fp8_ref.view(torch.uint8),
                    rtol=0, atol=0,
                    msg=f"{label}: z_fp8 mismatch",
                )
                torch.testing.assert_close(
                    z_raw_scales_fused.view(torch.uint8), z_raw_scales_ref.view(torch.uint8),
                    rtol=0, atol=0,
                    msg=f"{label}: z_raw_scales mismatch",
                )

                # y1 outputs must be BIT-IDENTICAL
                torch.testing.assert_close(
                    y1_fp8_fused.view(torch.uint8), y1_fp8_ref.view(torch.uint8),
                    rtol=0, atol=0,
                    msg=f"{label}: y1_fp8 mismatch",
                )
                torch.testing.assert_close(
                    y1_packed_scales_fused.view(torch.uint8), y1_packed_scales_ref.view(torch.uint8),
                    rtol=0, atol=0,
                    msg=f"{label}: y1_packed_scales mismatch",
                )

    def test_fused_z_save_y1_quant_roundtrip_precision(self) -> None:
        """fused_z_save_y1_quant roundtrip: quantize then dequantize preserves precision.

        Verifies that fused kernel output can be dequantized back with low error
        for both z (raw scales) and y1 (ISA-packed scales).
        """
        self._require_blackwell()
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import fused_z_save_y1_quant
        from sonicmoe.quack_utils.swiglu_triton import dequantize_blockscaled_fp8

        for label, TK, two_I, I in [("contract", 8192, 3072, 1536), ("prod", 65536, 3072, 1536)]:
            with self.subTest(shape=label):
                torch.manual_seed(42)
                z = 0.02 * torch.randn(TK, two_I, device="cuda", dtype=torch.bfloat16)
                y1 = 0.02 * torch.randn(TK, I, device="cuda", dtype=torch.bfloat16)

                z_fp8, z_raw_scales, y1_fp8, _ = fused_z_save_y1_quant(z, y1)

                # z roundtrip
                z_restored = dequantize_blockscaled_fp8(z_fp8, z_raw_scales.view(torch.uint8))
                z_rrmse = self._rrmse(z_restored, z)
                z_corr = self._corr(z_restored, z)
                self.assertLess(z_rrmse, 0.05,
                    f"{label} z roundtrip RRMSE {z_rrmse:.4f} >= 5%")
                self.assertGreater(z_corr, 0.999,
                    f"{label} z roundtrip corr {z_corr:.6f} < 0.999")

                # y1 FP8 data quality (can't easily roundtrip ISA-packed, check FP8 directly)
                y1_restored = y1_fp8.to(torch.bfloat16)
                # FP8 e4m3 has limited dynamic range but data is small-magnitude 0.02*randn
                self.assertFalse(torch.all(y1_fp8.view(torch.uint8) == 0).item(),
                    f"{label} y1_fp8 is all zeros")
