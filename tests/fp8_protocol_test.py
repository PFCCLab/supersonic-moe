# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sonicmoe import (
    FP8ActivationDType,
    apply_activation_fp8_protocol_cutely_fused,
    apply_preact_activation_fp8_protocol_cutely_fused,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    apply_activation_fp8_protocol,
    dequantize_activation_reference,
    enable_quack_gemm,
    get_default_fp8_protocol,
    pack_blockscaled_1x32_scales,
    quantize_activation_reference,
    make_blockscaled_grouped_reverse_scatter_idx,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)
from sonicmoe.functional.fp8_quant import (
    dequantize_activation_blockwise,
    quantize_activation_blockwise,
    round_scale_to_e8m0,
)
from sonicmoe.functional.fp8_protocol import FP8Backend, resolve_fp8_backend
from sonicmoe.quack_utils import blockscaled_fp8_gemm, clear_blockscaled_fp8_weight_cache, prefetch_blockscaled_w2_fp8
from sonicmoe.quack_utils.gemm_interface import gemm_dgated
from sonicmoe.utils import convert_torch_tensor_to_cute_tensor

from .test_commons import TestCommons


_SEED = 42


class FP8ProtocolTest(TestCommons):
    def test_backend_defaults_follow_device(self) -> None:
        for major, expected in ((9, FP8Backend.HOPPER), (10, FP8Backend.BLACKWELL), (11, FP8Backend.BLACKWELL)):
            with self.subTest(major=major), patch(
                "torch.cuda.get_device_capability", return_value=(major, 0)
            ):
                resolution = resolve_fp8_backend(device=torch.device("cuda"))
                self.assertIs(resolution.backend, expected)
                self.assertEqual(resolution.is_hopper, major == 9)
                self.assertEqual(resolution.routing_block_size, 128)

    def test_backend_explicit_request_must_match_device(self) -> None:
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            resolution = resolve_fp8_backend("hopper", torch.device("cuda"))
            self.assertTrue(resolution.is_hopper)
            with self.assertRaisesRegex(RuntimeError, "only supports SM100"):
                resolve_fp8_backend(FP8Backend.BLACKWELL, torch.device("cuda"))
            with self.assertRaisesRegex(ValueError, "Unsupported FP8 backend"):
                resolve_fp8_backend("hopper_qdq", torch.device("cuda"))
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(RuntimeError, "requires an exact SM90"):
                resolve_fp8_backend(FP8Backend.HOPPER, torch.device("cuda"))

    def test_backend_rejects_unsupported_device(self) -> None:
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            with self.assertRaisesRegex(RuntimeError, "supports exact SM90 and SM100"):
                resolve_fp8_backend(device=torch.device("cuda"))
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            resolve_fp8_backend(device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "Unsupported FP8 backend"):
            resolve_fp8_backend("invalid", torch.device("cuda"))

    def test_blackwell_fp8_protocol_defaults(self) -> None:
        protocol = validate_fp8_protocol(get_default_fp8_protocol())

        self.assertEqual(protocol.activation_dtype, FP8ActivationDType.E4M3)
        self.assertEqual(protocol.scale_encoding, FP8ScaleEncoding.E8M0)
        self.assertEqual(protocol.group_size, 128)
        self.assertTrue(protocol.requires_quack_gemm)

    def test_blackwell_fp8_protocol_rejects_unsupported_dtypes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only e4m3 activations"):
            validate_fp8_protocol(replace(FP8Protocol(), activation_dtype=None))

        with self.assertRaisesRegex(ValueError, "Only e8m0 scales"):
            validate_fp8_protocol(replace(FP8Protocol(), scale_encoding=None))

    def test_blockscaled_1x32_scale_pack_uses_sm100_tiled_storage(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        scales = torch.ones(8, 8, device="cuda", dtype=torch.float8_e8m0fnu)
        packed = pack_blockscaled_1x32_scales(scales, cols=256)
        self.assertEqual(tuple(packed.shape), (1, 1024))
        self.assertEqual(packed.dtype, torch.float8_e8m0fnu)

        protocol = validate_fp8_protocol(replace(FP8Protocol(), scale_granularity=FP8ScaleGranularity.BLOCK_1X32))
        self.assertEqual(protocol.group_size, 32)

    def test_blockscaled_grouped_reverse_scatter_idx_matches_static_capacity_layout(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        flat_sorted_positions = torch.tensor([0, 1, 2, 3, 4, 5], device="cuda", dtype=torch.int32)
        cu_seqlens_m = torch.tensor([0, 2, 5, 6], device="cuda", dtype=torch.int32)
        grouped_positions = make_blockscaled_grouped_reverse_scatter_idx(
            flat_sorted_positions,
            cu_seqlens_m,
            capacity=128,
        )
        expected = torch.tensor([0, 1, 128, 129, 130, 256], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(grouped_positions, expected)

    def test_blackwell_fp8_protocol_runtime_and_reference_quant(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()

        with self.assertRaisesRegex(RuntimeError, "requires the QuACK GEMM path"):
            validate_fp8_runtime_support(protocol, torch.device("cuda"), quack_enabled=False)

        x = (0.25 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).requires_grad_(False)

        with enable_quack_gemm(True):
            validate_fp8_runtime_support(protocol, x.device)
            fp8_tensor = quantize_activation_reference(x, protocol)

        self.assertEqual(fp8_tensor.data.dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8_tensor.scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(fp8_tensor.scales.shape), (8, 2))
        self.assertFalse(torch.isnan(fp8_tensor.data.float()).any().item())

        restored = dequantize_activation_reference(fp8_tensor)

        # e8m0 stores power-of-two scales, so the decoded scales should stay exact powers of two.
        log2_scales = torch.log2(fp8_tensor.scales.float())
        rounded_log2_scales = torch.round(log2_scales)
        self.assertTrue(torch.allclose(log2_scales, rounded_log2_scales, atol=0.0, rtol=0.0))

        self.assertEqual(restored.dtype, torch.float32)
        self.assertLess((restored - x.float()).abs().max().item(), 0.35)

    def test_blockwise_quant_matches_divide_reference_after_e8m0_encoding(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        x = 0.25 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

        quantized, scales = quantize_activation_blockwise(x, protocol)

        grouped_x = x.reshape(8, 2, protocol.group_size)
        amax = grouped_x.abs().amax(dim=-1).float()
        raw_scale = amax / torch.finfo(protocol.activation_torch_dtype).max
        safe_scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
        encoded_scale = round_scale_to_e8m0(safe_scale, protocol)
        divide_reference = (
            grouped_x / encoded_scale.float().to(dtype=grouped_x.dtype).unsqueeze(-1)
        ).to(protocol.activation_torch_dtype).reshape_as(x)

        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(scales.float(), encoded_scale.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(quantized.float(), divide_reference.float(), atol=0.0, rtol=0.0)

    def test_blockwise_quant_can_reuse_preallocated_outputs(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        x = 0.25 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

        quantized_ref, scales_ref = quantize_activation_blockwise(x, protocol)
        quantized_out = torch.empty_like(quantized_ref)
        scales_out = torch.empty_like(scales_ref)
        quantized, scales = quantize_activation_blockwise(x, protocol, out=quantized_out, scale_out=scales_out)
        restored_ref = dequantize_activation_blockwise(quantized_ref, scales_ref, protocol, output_dtype=x.dtype)
        restored_out = torch.empty_like(restored_ref)
        restored = dequantize_activation_blockwise(quantized, scales, protocol, output_dtype=x.dtype, out=restored_out)

        self.assertIs(quantized, quantized_out)
        self.assertIs(scales, scales_out)
        self.assertIs(restored, restored_out)
        torch.testing.assert_close(quantized.float(), quantized_ref.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(scales.float(), scales_ref.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(restored.float(), restored_ref.float(), atol=0.0, rtol=0.0)

    def test_blockscaled_weight_prefetch_reuses_cache_and_invalidates_on_weight_update(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        w2 = 0.02 * torch.randn(768, 256, 128, device="cuda", dtype=torch.bfloat16)

        clear_blockscaled_fp8_weight_cache()
        with enable_quack_gemm(True):
            weight_fp8_first, weight_scales_first = prefetch_blockscaled_w2_fp8(w2, protocol)
            weight_fp8_second, weight_scales_second = prefetch_blockscaled_w2_fp8(w2, protocol)

        self.assertIs(weight_fp8_first, weight_fp8_second)
        self.assertIs(weight_scales_first, weight_scales_second)

        with torch.no_grad():
            w2.add_(0.01)

        with enable_quack_gemm(True):
            weight_fp8_third, weight_scales_third = prefetch_blockscaled_w2_fp8(w2, protocol)

        self.assertIsNot(weight_fp8_first, weight_fp8_third)
        self.assertIsNot(weight_scales_first, weight_scales_third)

    def test_blockscaled_downproj_rejects_insufficient_capacity(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        protocol = get_default_fp8_protocol()
        a = torch.randn(129, 256, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(256, 256, 2, device="cuda", dtype=torch.bfloat16)
        cu_seqlens_m = torch.tensor([0, 129, 129], device="cuda", dtype=torch.int32)

        with patch.dict(
            os.environ,
            {
                "SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ": "1",
                "SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY": "128",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "smaller than the routed expert load"):
                blockscaled_fp8_gemm(a, w2, cu_seqlens_m, protocol=protocol)

    def test_cutely_fused_adapter_matches_reference_contract(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        sample = 0.1 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

        with enable_quack_gemm(True):
            restored_ref, scales_ref = apply_activation_fp8_protocol(sample, protocol)
            restored_adapter, scales_adapter = apply_activation_fp8_protocol_cutely_fused(sample, protocol)

        self.assertEqual(restored_ref.shape, restored_adapter.shape)
        self.assertEqual(scales_ref.shape, scales_adapter.shape)
        self.assertEqual(restored_ref.dtype, restored_adapter.dtype)
        self.assertEqual(scales_ref.dtype, scales_adapter.dtype)
        torch.testing.assert_close(restored_ref.float(), restored_adapter.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(scales_ref.float(), scales_adapter.float(), atol=0.0, rtol=0.0)

    def test_preact_cutely_fused_path_matches_reference_boundary(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        preact = 0.1 * torch.randn(8, 260, device="cuda", dtype=torch.bfloat16)
        postact = (preact[..., 1::2] * F.silu(preact[..., ::2].float()).to(dtype=preact.dtype)).contiguous()

        with enable_quack_gemm(True):
            restored_ref, scales_ref = apply_activation_fp8_protocol(postact, protocol)
            restored_fused, scales_fused = apply_preact_activation_fp8_protocol_cutely_fused(preact, postact, protocol)

        self.assertEqual(restored_ref.shape, restored_fused.shape)
        self.assertEqual(scales_ref.shape, scales_fused.shape)
        self.assertEqual(restored_fused.dtype, torch.bfloat16)
        self.assertEqual(scales_fused.dtype, torch.float8_e8m0fnu)
        diff = (restored_ref.float() - restored_fused.float()).abs()
        self.assertLess(diff.max().item(), 3e-2)
        self.assertLess(torch.sqrt(torch.mean(diff.square())).item(), 6e-3)
        log2_scales = torch.log2(scales_fused.float())
        self.assertTrue(torch.allclose(log2_scales, torch.round(log2_scales), atol=0.0, rtol=0.0))

    def test_preact_cutely_fused_path_can_reconstruct_without_postact_tensor(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        preact = 0.1 * torch.randn(8, 260, device="cuda", dtype=torch.bfloat16)
        postact = (preact[..., 1::2] * F.silu(preact[..., ::2].float()).to(dtype=preact.dtype)).contiguous()

        with enable_quack_gemm(True):
            restored_ref, scales_ref = apply_preact_activation_fp8_protocol_cutely_fused(
                preact,
                postact,
                protocol,
                use_ste=False,
            )
            restored_no_postact, scales_no_postact = apply_preact_activation_fp8_protocol_cutely_fused(
                preact,
                None,
                protocol,
                use_ste=False,
                output_dtype=postact.dtype,
            )

        torch.testing.assert_close(restored_no_postact.float(), restored_ref.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(scales_no_postact.float(), scales_ref.float(), atol=0.0, rtol=0.0)

    def test_convert_torch_tensor_to_cute_tensor_accepts_runtime_fp8(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        self.set_seed(_SEED)
        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        cute_tensor = convert_torch_tensor_to_cute_tensor(x, (0, 1), 1, 16, 8)

        self.assertEqual(str(cute_tensor.element_type), "Float8E4M3FN")

    def test_quack_dgated_can_emit_fp8_postact_buffer(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        preact = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

        with enable_quack_gemm(True):
            dx, y1s, ds = gemm_dgated(
                a,
                b,
                PreAct=preact,
                activation="swiglu",
                out_dtype=torch.bfloat16,
                postact_dtype=torch.float8_e4m3fn,
                colvec_reduce=True,
                dynamic_scheduler=False,
                tuned=False,
            )

        self.assertEqual(dx.dtype, torch.bfloat16)
        self.assertEqual(y1s.dtype, torch.float8_e4m3fn)
        self.assertEqual(ds.dtype, torch.float32)
        self.assertEqual(tuple(dx.shape), (64, 256))
        self.assertEqual(tuple(y1s.shape), (64, 128))
        self.assertEqual(tuple(ds.shape), (64,))
        self.assertFalse(torch.isnan(y1s.float()).any().item())

    def test_preact_cutely_fused_path_can_reuse_scale_buffer(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        preact = 0.1 * torch.randn(8, 260, device="cuda", dtype=torch.bfloat16)
        postact = (preact[..., 1::2] * F.silu(preact[..., ::2].float()).to(dtype=preact.dtype)).contiguous()
        scale_out = torch.empty((preact.size(0), (postact.size(1) + protocol.group_size - 1) // protocol.group_size), device="cuda", dtype=torch.float8_e8m0fnu)

        with enable_quack_gemm(True):
            _, scales_ref = apply_preact_activation_fp8_protocol_cutely_fused(preact, postact, protocol)
            _, scales = apply_preact_activation_fp8_protocol_cutely_fused(preact, postact, protocol, scale_out=scale_out)

        self.assertIs(scales, scale_out)
        torch.testing.assert_close(scales.float(), scales_ref.float(), atol=0.0, rtol=0.0)

    def test_preact_cutely_fused_path_can_reuse_restored_output_buffer(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        preact = 0.1 * torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        postact = (preact[..., 1::2] * F.silu(preact[..., ::2].float()).to(dtype=preact.dtype)).contiguous()
        restored_out = torch.empty_like(postact)

        with enable_quack_gemm(True):
            restored_ref, _ = apply_preact_activation_fp8_protocol_cutely_fused(
                preact,
                postact,
                protocol,
                use_ste=False,
            )
            restored, _ = apply_preact_activation_fp8_protocol_cutely_fused(
                preact,
                postact,
                protocol,
                use_ste=False,
                restored_out=restored_out,
            )

        self.assertIs(restored, restored_out)
        torch.testing.assert_close(restored.float(), restored_ref.float(), atol=0.0, rtol=0.0)
