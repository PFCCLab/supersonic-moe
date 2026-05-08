# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sonicmoe.functional as functional
from sonicmoe import (
    FP8ActivationDType,
    KernelBackendMoE,
    MoE,
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
from sonicmoe.enums import ActivationType
from sonicmoe.functional.fp8_quant import (
    dequantize_activation_blockwise,
    quantize_activation_blockwise,
    round_scale_to_e8m0,
)
from sonicmoe.quack_utils import blockscaled_fp8_gemm, clear_blockscaled_fp8_weight_cache, prefetch_blockscaled_w2_fp8
from sonicmoe.quack_utils.gemm_interface import gemm_dgated
from sonicmoe.utils import convert_torch_tensor_to_cute_tensor

from .test_commons import TestCommons


_SEED = 42


class FP8ProtocolTest(TestCommons):
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

    def test_blackwell_fp8_protocol_boundary_keeps_finite_forward_backward(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()

        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with enable_quack_gemm(True):
            y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
            y.backward(dout)

        self.assertFalse(torch.isnan(y.float()).any().item())
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad.float()).any().item())

        sample = 0.1 * torch.randn(4, 130, device="cuda", dtype=torch.bfloat16)
        with enable_quack_gemm(True):
            restored, scales = apply_activation_fp8_protocol(sample, protocol)
        self.assertEqual(tuple(scales.shape), (4, 2))
        self.assertEqual(tuple(restored.shape), (4, 130))
        self.assertFalse(torch.isnan(restored.float()).any().item())

    def test_fp8_runtime_switch_can_disable_upproj_epilogue_quant(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with patch.dict(
            os.environ,
            {
                "SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION": "bf16",
                "SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION": "bf16",
                "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION": "bf16",
            },
            clear=False,
        ):
            with patch.object(functional, "apply_preact_activation_fp8_protocol_cutely_fused", wraps=functional.apply_preact_activation_fp8_protocol_cutely_fused) as patched:
                with enable_quack_gemm(True):
                    y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
                    y.backward(dout)

        patched.assert_not_called()
        self.assertFalse(torch.isnan(y.float()).any().item())
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad.float()).any().item())

    def test_fp8_runtime_switch_rejects_fp8_downproj_weights_without_fp8_mainloop(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()

        with patch.dict(
            os.environ,
            {
                "SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION": "bf16",
                "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION": "fp8",
            },
            clear=False,
        ):
            with enable_quack_gemm(True):
                with self.assertRaisesRegex(RuntimeError, "SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION=fp8"):
                    moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)

    def test_blockscaled_downproj_rejects_unaligned_static_capacity(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()

        with patch.dict(
            os.environ,
            {
                "SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ": "1",
                "SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY": "127",
            },
            clear=False,
        ):
            with enable_quack_gemm(True):
                with self.assertRaisesRegex(RuntimeError, "SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY"):
                    moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)

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

    def test_moe_can_prefetch_supported_fp8_weights(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

        moe.clear_fp8_weight_cache()
        with enable_quack_gemm(True):
            prefetched = moe.prefetch_fp8_weights(protocol)

        self.assertIn("downproj", prefetched)
        weight_fp8, weight_scales = prefetched["downproj"]
        self.assertEqual(weight_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(weight_scales.dtype, torch.float8_e8m0fnu)

    def test_blockscaled_downproj_boundary_keeps_finite_forward_backward_with_static_capacity(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()

        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)

        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with patch.dict(
            os.environ,
            {
                "SONIC_MOE_FP8_BLOCKSCALED_DOWNPROJ": "1",
                "SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY": "128",
            },
            clear=False,
        ):
            with enable_quack_gemm(True):
                y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
                y.backward(dout)

        self.assertFalse(torch.isnan(y.float()).any().item())
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad.float()).any().item())

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

    def test_blackwell_fp8_runtime_path_requests_low_precision_upproj_postact_buffer(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with patch.dict(os.environ, {"SONIC_MOE_FP8_DUMMY_POSTACT_BUFFER": "1"}, clear=False):
            with patch.object(functional, "gemm_gated", wraps=functional.gemm_gated) as patched_gemm_gated:
                with patch.object(
                    functional,
                    "apply_preact_activation_fp8_protocol_cutely_fused",
                    wraps=functional.apply_preact_activation_fp8_protocol_cutely_fused,
                ) as patched_fp8_boundary:
                    with enable_quack_gemm(True):
                        y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, fp8_protocol=protocol)
                        y.backward(dout)

        self.assertTrue(patched_gemm_gated.called)
        postact_dtypes = [call.kwargs.get("postact_dtype") for call in patched_gemm_gated.call_args_list]
        self.assertIn(torch.float8_e4m3fn, postact_dtypes)
        restored_out_buffers = [call.kwargs.get("restored_out") for call in patched_fp8_boundary.call_args_list]
        restored_out_buffers = [buffer for buffer in restored_out_buffers if buffer is not None]
        self.assertTrue(restored_out_buffers)
        self.assertTrue(all(buffer.dtype == torch.bfloat16 for buffer in restored_out_buffers))

    def test_quack_backward_reuses_forward_backend_snapshot(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).detach().requires_grad_()
        dout = 0.02 * torch.randn_like(x)

        with enable_quack_gemm(True):
            y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)

        with patch.object(
            functional,
            "_up_projection_backward_act",
            side_effect=AssertionError("backward should keep using the forward QuACK backend"),
        ), patch.object(
            functional,
            "_down_projection_backward_act",
            side_effect=AssertionError("backward should keep using the forward QuACK backend"),
        ):
            with enable_quack_gemm(False):
                y.backward(dout)

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad.float()).any().item())

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

    def test_quack_inference_fastpath_skips_fp8_boundary_and_returns_no_grad_output(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = (0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)).requires_grad_()

        with patch.object(functional, "gemm_gated", wraps=functional.gemm_gated) as patched_gemm_gated:
            with patch.object(functional, "_softmax_topk_fwd", wraps=functional._softmax_topk_fwd) as patched_topk_fwd:
                with patch.object(functional.TC_Softmax_Topk_Router_Function, "apply") as patched_topk_apply:
                    with patch.object(
                        functional,
                        "apply_preact_activation_fp8_protocol_cutely_fused",
                        wraps=functional.apply_preact_activation_fp8_protocol_cutely_fused,
                    ) as patched_fp8_boundary:
                        with enable_quack_gemm(True):
                            y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, is_inference_mode=True)

        self.assertTrue(patched_gemm_gated.called)
        self.assertTrue(patched_topk_fwd.called)
        patched_topk_apply.assert_not_called()
        store_preact_flags = [call.kwargs.get("store_preact") for call in patched_gemm_gated.call_args_list]
        self.assertTrue(store_preact_flags)
        self.assertTrue(all(flag is False for flag in store_preact_flags))
        patched_fp8_boundary.assert_not_called()
        self.assertFalse(y.requires_grad)
        self.assertFalse(torch.isnan(y.float()).any().item())

    def test_quack_inference_fastpath_keeps_fp8_boundary_and_matches_standard_path(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            self.skipTest("SM100-only test requires SM100+")

        self.set_seed(_SEED)
        protocol = get_default_fp8_protocol()
        moe = MoE(
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=768,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device="cuda", dtype=torch.bfloat16)
        x = 0.02 * torch.randn(256, 768, device="cuda", dtype=torch.bfloat16)

        with patch.object(functional, "gemm_gated", wraps=functional.gemm_gated) as patched_gemm_gated:
            with patch.object(
                functional,
                "apply_preact_activation_fp8_protocol_cutely_fused",
                wraps=functional.apply_preact_activation_fp8_protocol_cutely_fused,
            ) as patched_fp8_boundary:
                with enable_quack_gemm(True):
                    fast_o, fast_logits, fast_freq = functional.moe_TC_softmax_topk_layer(
                        x,
                        moe.router.weight,
                        moe.c_fc.weight.permute(1, 2, 0),
                        moe.c_fc.bias,
                        moe.c_proj.weight.permute(1, 2, 0),
                        moe.c_proj.bias,
                        moe.top_k,
                        moe.stream_id,
                        moe.activation_function,
                        True,
                        protocol,
                    )
                    ref_o, ref_logits, ref_freq = functional.moe_TC_softmax_topk_layer(
                        x,
                        moe.router.weight,
                        moe.c_fc.weight.permute(1, 2, 0),
                        moe.c_fc.bias,
                        moe.c_proj.weight.permute(1, 2, 0),
                        moe.c_proj.bias,
                        moe.top_k,
                        moe.stream_id,
                        moe.activation_function,
                        False,
                        protocol,
                    )

        store_preact_flags = [call.kwargs.get("store_preact") for call in patched_gemm_gated.call_args_list]
        self.assertIn(True, store_preact_flags)
        self.assertTrue(patched_fp8_boundary.called)
        torch.testing.assert_close(fast_o.float(), ref_o.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(fast_logits.float(), ref_logits.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(fast_freq.int(), ref_freq.int(), atol=0, rtol=0)

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
