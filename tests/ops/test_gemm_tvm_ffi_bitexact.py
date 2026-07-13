"""Bit-exact regression tests for the Sonic MoE TVM-FFI GEMM launch path."""

from contextlib import contextmanager, nullcontext
from importlib import import_module
import os

import numpy as np
import pytest
import torch

from tests.ops.conftest import requires_blackwell, requires_quack

pytestmark = [requires_blackwell, requires_quack]

_SHAPE = (128, 512, 512, 4, 8)  # T, H, I, E, topk; TK=256, all 128-aligned.


@contextmanager
def _force_executor(module_name: str, predicate_name: str):
    module = import_module(module_name)
    original = getattr(module, predicate_name)
    setattr(module, predicate_name, lambda *args, **kwargs: False)
    try:
        yield
    finally:
        setattr(module, predicate_name, original)


@contextmanager
def _disable_gated_tvm_ffi():
    name = "SONIC_MOE_DISABLE_GATED_TVM_FFI"
    original = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


@contextmanager
def _forbid_gated_executor_fallback():
    module = import_module("sonicmoe.quack_utils.gemm_interface")
    original = module.gemm_gated_sm90_sm100

    def fail(*args, **kwargs):
        raise AssertionError("native-layout TVM-FFI path fell back to executor")

    module.gemm_gated_sm90_sm100 = fail
    try:
        yield
    finally:
        module.gemm_gated_sm90_sm100 = original


def _assert_byte_exact(actual, expected, label):
    a = actual.contiguous().view(torch.uint8)
    b = expected.contiguous().view(torch.uint8)
    mismatches = (a != b).sum().item()
    assert mismatches == 0, f"{label}: {mismatches}/{a.numel()} bytes differ"


def _current_raw_stream():
    stream = torch.cuda.current_stream()
    if hasattr(stream, "stream_base"):
        return int(stream.stream_base.raw_stream)
    return int(stream.cuda_stream)


@pytest.mark.parametrize(
    "b_pretransposed,preallocated",
    [(False, False), (True, False), (True, True)],
    ids=["legacy_b", "native_enk", "native_enk_preallocated"],
)
@pytest.mark.parametrize(
    "epilogue_mode", ["postact", "combined"], ids=["postact", "combined"]
)
def test_gemm_gated_tvm_ffi_matches_executor_quantized_postact(
    b_pretransposed, preallocated, epilogue_mode
):
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_fused_gated,
        quantize_and_pack_activation,
    )
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    from sonicmoe.quack_utils.gemm_gated import prepare_gated_tvm_ffi_weight

    T, H, I, E, topk = _SHAPE
    TK = T * topk // E
    total_m = TK * E
    rng = np.random.RandomState(123)
    x = torch.from_numpy((rng.randn(total_m, H).astype(np.float32) * 0.02)).to(
        "cuda", dtype=torch.bfloat16
    )
    w1 = torch.from_numpy((rng.randn(2 * I, H, E).astype(np.float32) * 0.02)).to(
        "cuda", dtype=torch.bfloat16
    )
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    a_idx = torch.arange(total_m, dtype=torch.int32, device="cuda")
    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)
    b = w_fp8.mT if b_pretransposed else w_fp8
    if b_pretransposed:
        prepare_gated_tvm_ffi_weight(b)

    def run(enabled):
        if enabled and b_pretransposed:
            context = _forbid_gated_executor_fallback()
        elif enabled:
            context = nullcontext()
        else:
            context = _disable_gated_tvm_ffi()
        with context:
            z_scale = (
                torch.empty(
                    (total_m, 2 * I // 32),
                    dtype=torch.uint8,
                    device="cuda",
                )
                if epilogue_mode == "combined"
                else None
            )
            y1_scale = torch.empty((total_m // 128, I // 128, 512), dtype=torch.uint8, device="cuda")
            preact = (
                torch.empty(
                    (total_m, 2 * I),
                    dtype=(
                        torch.float8_e4m3fn
                        if epilogue_mode == "combined"
                        else torch.bfloat16
                    ),
                    device="cuda",
                )
                if preallocated
                else None
            )
            postact = (
                torch.empty(
                    (total_m, I),
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                )
                if preallocated
                else None
            )
            z, y1 = gemm_gated(
                x_fp8,
                b,
                activation="swiglu",
                preact_out=preact,
                postact_out=postact,
                out_dtype=(
                    torch.float8_e4m3fn
                    if epilogue_mode == "combined"
                    else torch.bfloat16
                ),
                postact_dtype=torch.float8_e4m3fn,
                cu_seqlens_m=cu_seqlens,
                A_idx=a_idx,
                a_scales=a_scales,
                b_scales=b_scales,
                z_scale_out=z_scale,
                postact_scale_out=y1_scale,
                tuned=False,
                current_stream=_current_raw_stream(),
                b_pretransposed=b_pretransposed,
            )
            torch.cuda.synchronize()
            outputs = [z, y1, y1_scale]
            if z_scale is not None:
                outputs.append(z_scale)
            return outputs

    executor = run(False)
    tvm_ffi = run(True)
    labels = ["z", "y1", "y1_scale"]
    if epilogue_mode == "combined":
        labels.append("z_scale")
    for label, lhs, rhs in zip(labels, tvm_ffi, executor):
        _assert_byte_exact(lhs, rhs, label)


@pytest.mark.parametrize("epilogue_mode", ["postact", "combined"], ids=["postact", "combined"])
def test_gemm_gated_prepared_plan_rebuilds_microbatch_pointers(epilogue_mode):
    """A prepared plan never retains activation, routing, scale, or output pointers."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_fused_gated,
        quantize_and_pack_activation,
    )
    from sonicmoe.quack_utils.gemm_gated import prepare_gated_tvm_ffi_weight, try_gemm_gated_tvm_ffi_sonic_prepared
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    T, H, I, E, topk = _SHAPE
    TK = T * topk // E
    total_m = TK * E
    rng = np.random.RandomState(314159)
    w1 = torch.from_numpy((rng.randn(2 * I, H, E).astype(np.float32) * 0.02)).to("cuda", dtype=torch.bfloat16)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)
    b = prepare_gated_tvm_ffi_weight(w_fp8.mT)

    def make_inputs(seed):
        local_rng = np.random.RandomState(seed)
        x = torch.from_numpy((local_rng.randn(total_m, H).astype(np.float32) * 0.02)).to("cuda", dtype=torch.bfloat16)
        x_fp8, a_scales = quantize_and_pack_activation(x)
        a_idx = torch.arange(total_m - 1, -1, -1, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
        return x_fp8, a_scales, a_idx, cu_seqlens

    def allocate_outputs():
        z_scale = (
            torch.empty((total_m, 2 * I // 32), dtype=torch.uint8, device="cuda")
            if epilogue_mode == "combined"
            else None
        )
        return (
            torch.empty(
                (total_m, 2 * I),
                dtype=(torch.float8_e4m3fn if epilogue_mode == "combined" else torch.bfloat16),
                device="cuda",
            ),
            torch.empty((total_m, I), dtype=torch.float8_e4m3fn, device="cuda"),
            z_scale,
            torch.empty(
                (total_m // 128, I // 128, 512),
                dtype=torch.uint8,
                device="cuda",
            ),
        )

    # Populate the static plan through the fully validated path.
    x1, scales1, idx1, offsets1 = make_inputs(1)
    z1, y1, z_scale1, y1_scale1 = allocate_outputs()
    gemm_gated(
        x1,
        b,
        activation="swiglu",
        preact_out=z1,
        postact_out=y1,
        out_dtype=z1.dtype,
        postact_dtype=y1.dtype,
        cu_seqlens_m=offsets1,
        A_idx=idx1,
        a_scales=scales1,
        b_scales=b_scales,
        z_scale_out=z_scale1,
        postact_scale_out=y1_scale1,
        tuned=False,
        current_stream=_current_raw_stream(),
        b_pretransposed=True,
    )
    torch.cuda.synchronize()

    # Use distinct tensors for every dynamic argument. The executor provides
    # the expected bytes; the prepared launch must write only the new outputs.
    x2, scales2, idx2, offsets2 = make_inputs(2)
    expected = allocate_outputs()
    with _disable_gated_tvm_ffi():
        gemm_gated(
            x2,
            b,
            activation="swiglu",
            preact_out=expected[0],
            postact_out=expected[1],
            out_dtype=expected[0].dtype,
            postact_dtype=expected[1].dtype,
            cu_seqlens_m=offsets2,
            A_idx=idx2,
            a_scales=scales2,
            b_scales=b_scales,
            z_scale_out=expected[2],
            postact_scale_out=expected[3],
            tuned=False,
            current_stream=_current_raw_stream(),
            b_pretransposed=True,
        )
    actual = allocate_outputs()
    launched = try_gemm_gated_tvm_ffi_sonic_prepared(
        x2,
        b,
        actual[0],
        actual[1],
        activation="swiglu",
        cu_seqlens_m=offsets2,
        A_idx=idx2,
        a_scales=scales2,
        b_scales=b_scales,
        z_scale_out=actual[2],
        postact_scale_out=actual[3],
        swiglu_clamp_value=0.0,
        postact_bf16_trunc=False,
        current_stream=_current_raw_stream(),
    )
    assert launched
    torch.cuda.synchronize()
    labels = ("z", "y1", "z_scale", "y1_scale")
    for label, lhs, rhs in zip(labels, actual, expected):
        if lhs is not None:
            _assert_byte_exact(lhs, rhs, label)


def test_gemm_dgated_tvm_ffi_matches_executor_fp8_preact_reduce():
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_direct_fused_dgated,
        quantize_activation_blockscaled_fast,
        quantize_and_pack_activation,
    )
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
    from sonicmoe.quack_utils.gemm_interface import default_config

    T, H, I, E, topk = _SHAPE
    TK = T * topk // E
    total_m = TK * E
    rng = np.random.RandomState(456)
    dout = torch.from_numpy((rng.randn(total_m, H).astype(np.float32) * 0.02)).to(
        "cuda", dtype=torch.bfloat16
    )
    w2 = torch.from_numpy((rng.randn(H, I, E).astype(np.float32) * 0.02)).to(
        "cuda", dtype=torch.bfloat16
    )
    z_bf16 = torch.from_numpy((rng.randn(total_m, 2 * I).astype(np.float32) * 0.02)).to(
        "cuda", dtype=torch.bfloat16
    )
    colvec_scale = torch.from_numpy((rng.rand(total_m).astype(np.float32) + 0.5)).to("cuda")
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    a_idx = torch.arange(total_m, dtype=torch.int32, device="cuda")
    dout_fp8, dout_scales = quantize_and_pack_activation(dout)
    w2_fp8, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)
    z_fp8, z_scales = quantize_activation_blockscaled_fast(z_bf16.contiguous())
    config = default_config(dout.device, num_experts=E)
    partial_shape = (total_m, (I + config.tile_n - 1) // config.tile_n)

    def run(enabled):
        context = (
            nullcontext()
            if enabled
            else _force_executor("sonicmoe.quack_utils.gemm_dgated", "_can_use_dgated_tvm_ffi")
        )
        with context:
            dz = torch.empty((total_m, 2 * I), dtype=torch.bfloat16, device="cuda")
            y1s = torch.empty((total_m, I), dtype=torch.bfloat16, device="cuda")
            partial = torch.zeros(partial_shape, dtype=torch.float32, device="cuda")
            gemm_dgated_kernel(
                dout_fp8,
                w2_fp8,
                dz,
                dz,
                y1s,
                None,
                "swiglu",
                config.tile_m,
                config.tile_n,
                config.cluster_m,
                config.cluster_n,
                config.pingpong,
                persistent=True,
                max_swizzle_size=config.max_swizzle_size,
                colvec_scale=colvec_scale,
                colvec_reduce=partial,
                cu_seqlens_m=cu_seqlens,
                A_idx=a_idx,
                a_scales=dout_scales,
                b_scales=w2_scales,
                preact_fp8=z_fp8,
                preact_scales=z_scales,
            )
            torch.cuda.synchronize()
            return dz, y1s, partial

    executor = run(False)
    tvm_ffi = run(True)
    for label, lhs, rhs in zip(("dz", "y1s", "colvec_reduce"), tvm_ffi, executor):
        _assert_byte_exact(lhs, rhs, label)
