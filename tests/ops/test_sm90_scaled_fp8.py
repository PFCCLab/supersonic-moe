from __future__ import annotations

import gc
import importlib
import itertools
import weakref

import pytest


@pytest.fixture(scope="module")
def torch():
    paddle = pytest.importorskip("paddle")
    paddle.enable_compat()
    return pytest.importorskip("torch")


def _sm90(torch):
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


def _dequantize_colwise(torch, data, scales):
    exponent = scales.to(torch.float32) - 127.0
    scale = torch.exp(exponent * 0.6931471805599453)
    return (data.float().reshape(data.shape[0], -1, 32) * scale.unsqueeze(-1)).flatten(1)


def _wgrad_reference(torch, a, b, sa, sb, packed_offsets):
    a32 = _dequantize_colwise(torch, a, sa)
    b32 = _dequantize_colwise(torch, b, sb)
    offsets = packed_offsets.cpu().tolist()
    return torch.stack(
        [a32[:, left:right] @ b32[:, left:right].mT for left, right in zip(offsets, offsets[1:])]
    )


def _assert_wgrad_close(torch, actual, expected):
    error = (actual - expected).abs()
    relative = error / expected.abs().clip(min=1e-2)
    assert float(error.mean().item()) < 0.4
    assert float(relative.mean().item()) < 0.15


def test_non_aligned_wgrad_segments_fresh_add_beta(torch):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    ref = importlib.import_module("sonicmoe.functional.sm90_reference")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")

    torch.manual_seed(9191)
    offsets = torch.tensor([0, 17, 49, 49, 113], dtype=torch.int32, device="cuda")
    packed_offsets = torch.tensor(
        [0, 32, 64, 64, 128], dtype=torch.int32, device="cuda"
    )
    rows, m, n = 113, 32, 48
    a_source = torch.randn((rows, m), dtype=torch.bfloat16, device="cuda")
    b_source = torch.randn((rows, n), dtype=torch.bfloat16, device="cuda")
    a, sa, a_offsets = ref.quantize_wgrad_colwise_sm90(a_source, offsets)
    b, sb, b_offsets = ref.quantize_wgrad_colwise_sm90(b_source, offsets)
    assert bool((a_offsets == packed_offsets).all().item())
    assert bool((b_offsets == packed_offsets).all().item())
    assert a.shape == (m, 128) and sa.shape == (m, 4)
    assert b.shape == (n, 128) and sb.shape == (n, 4)

    expected = _wgrad_reference(torch, a, b, sa, sb, packed_offsets)
    out = torch.empty((4, m, n), dtype=torch.float32, device="cuda")
    scaled.sm90_scaled_wgrad_varlen_k(a, b, out, sa, sb, packed_offsets)
    _assert_wgrad_close(torch, out, expected)
    assert out[2].abs().max() == 0

    prior = torch.randn_like(out)
    out.copy_(prior)
    scaled.sm90_scaled_wgrad_varlen_k(
        a, b, out, sa, sb, packed_offsets, add_to_output=True
    )
    _assert_wgrad_close(torch, out, expected + prior)
    scaled.sm90_scaled_wgrad_varlen_k(
        a, b, out, sa, sb, packed_offsets, C=prior, beta=0.25
    )
    _assert_wgrad_close(torch, out, expected + prior * 0.25)


@pytest.mark.parametrize("profile", [32, 128])
def test_scaled_profiles_empty_experts_fresh_accumulate_add(torch, profile):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    ref = importlib.import_module("sonicmoe.functional.sm90_reference")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")

    torch.manual_seed(9100 + profile)
    rows, hidden, output, experts = 256, 128, 128, 4
    x = (torch.randn(rows, hidden, device="cuda") * 0.02).to(torch.bfloat16)
    w = (torch.randn(experts, output, hidden, device="cuda") * 0.02).to(torch.bfloat16)
    xq, xs, _ = ref.quantize_activation_sm90(x, profile)
    wq, ws = ref.quantize_weight_sm90(w, profile)
    offsets = torch.tensor([0, 128, 128, 256, 256], dtype=torch.int32, device="cuda")
    gather = torch.arange(rows, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, output), dtype=torch.bfloat16, device="cuda")
    scaled.sm90_scaled_gemm(
        xq, wq, out, xs, ws, cu_seqlens_m=offsets, A_idx=gather,
        activation_group=profile, weight_block=profile,
    )
    assert torch.isfinite(out).all()
    assert out[128:256].abs().sum() > 0

    total_k = 256
    a = torch.randn(output, total_k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(hidden, total_k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.full((output, total_k // 32), 127, dtype=torch.uint8, device="cuda")
    sb = torch.full((hidden, total_k // 32), 127, dtype=torch.uint8, device="cuda")
    d = torch.empty((experts, output, hidden), dtype=torch.float32, device="cuda")
    scaled.sm90_scaled_wgrad_varlen_k(a, b, d, sa, sb, offsets)
    fresh = d.clone()
    expected = _wgrad_reference(torch, a, b, sa, sb, offsets)
    _assert_wgrad_close(torch, fresh, expected)
    assert fresh[1].abs().max() == 0 and fresh[3].abs().max() == 0
    c = torch.randn_like(d)
    scaled.sm90_scaled_wgrad_varlen_k(a, b, d, sa, sb, offsets, C=c, beta=0.25)
    _assert_wgrad_close(torch, d, expected + c * 0.25)
    d.copy_(c)
    scaled.sm90_scaled_wgrad_varlen_k(a, b, d, sa, sb, offsets, add_to_output=True)
    _assert_wgrad_close(torch, d, expected + c)


@pytest.mark.parametrize(
    "offsets,error",
    [
        ([0, 16, 64], "divisible by 32"),
        ([0, 64, 32], "span exactly"),
        ([0, 32], "length must equal"),
    ],
)
def test_wgrad_rejects_invalid_packed_offsets(torch, offsets, error):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")
    a = torch.ones((32, 64), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.ones((48, 64), dtype=torch.float8_e4m3fn, device="cuda")
    sa = torch.full((32, 2), 127, dtype=torch.uint8, device="cuda")
    sb = torch.full((48, 2), 127, dtype=torch.uint8, device="cuda")
    d = torch.empty((2, 32, 48), dtype=torch.float32, device="cuda")
    packed = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=error):
        scaled.sm90_scaled_wgrad_varlen_k(a, b, d, sa, sb, packed)


@pytest.mark.parametrize("profile", [32, 128])
@pytest.mark.parametrize("z_quant,y1_quant", [(False, False), (True, False), (False, True), (True, True)])
def test_sm90_raw_gated_epilogue_matrix(torch, profile, z_quant, y1_quant, monkeypatch):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")
    rows, hidden, width = 128, 128, 256
    torch.manual_seed(8000 + profile + z_quant * 3 + y1_quant)
    a = (torch.randn((rows, hidden), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn((1, width, hidden), device="cuda") * 0.1).to(torch.float8_e4m3fn)
    sa = torch.full((rows, hidden // profile), 127, dtype=torch.uint8, device="cuda")
    sb = torch.full((1, width // profile, hidden // profile), 127, dtype=torch.uint8, device="cuda")
    offsets = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    gather = torch.arange(rows, dtype=torch.int32, device="cuda")
    z_ref = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    y_ref = torch.empty((rows, width // 2), dtype=torch.bfloat16, device="cuda")
    scaled.sm90_scaled_gated(
        a, b, z_ref, y_ref, sa, sb, cu_seqlens_m=offsets, A_idx=gather,
        activation_group=profile, weight_block=profile,
    )
    z_expected, zs_expected = quant.quantize_activation_blockscaled_fast(z_ref, 32)
    y_expected, ys_expected = quant.quantize_activation_blockscaled_fast(y_ref, 32)
    z = torch.empty_like(z_expected) if z_quant else torch.empty_like(z_ref)
    y = torch.empty_like(y_expected) if y1_quant else torch.empty_like(y_ref)
    zs = torch.empty((rows, width // 32), dtype=torch.uint8, device="cuda") if z_quant else None
    ys = torch.empty((rows, width // 64), dtype=torch.uint8, device="cuda") if y1_quant else None
    monkeypatch.setattr(
        quant, "quantize_activation_blockscaled_fast",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("standalone quant called")),
    )
    scaled.sm90_scaled_gated(
        a, b, z, y, sa, sb, z_scale_out=zs, postact_scale_out=ys,
        postact_bf16_trunc=True, cu_seqlens_m=offsets, A_idx=gather,
        activation_group=profile, weight_block=profile,
    )
    if z_quant:
        assert int((zs != zs_expected.view(torch.uint8)).sum().item()) <= 8
        z_deq = z.float() * torch.exp((zs.float() - 127) * 0.6931471805599453).repeat_interleave(32, 1)
        z_ref_deq = z_expected.float() * torch.exp(
            (zs_expected.view(torch.uint8).float() - 127) * 0.6931471805599453
        ).repeat_interleave(32, 1)
        assert float((z_deq - z_ref_deq).abs().mean().item()) < 1e-3
    else:
        assert bool((z == z_ref).all().item())
    if y1_quant:
        y_deq = y.float() * torch.exp((ys.float() - 127) * 0.6931471805599453).repeat_interleave(32, 1)
        y_ref_deq = y_expected.float() * torch.exp(
            (ys_expected.view(torch.uint8).float() - 127) * 0.6931471805599453
        ).repeat_interleave(32, 1)
        assert float((y_deq - y_ref_deq).abs().mean().item()) < 1e-2
    else:
        assert float((y.float() - y_ref.float()).abs().mean().item()) < 1e-2


def test_carrier_deferred_retain_release_and_cleanup(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = 32
        weight_block_size = 32

    tensor = torch.zeros((2, 2))
    with runtime.sm90_invocation_scope(
        Config(), "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        runtime.register_sm90_carrier(tensor, data=tensor, role="test")
        wrapped = runtime.wrap_deferred_sm90_call(lambda: "done", tensor)
        carrier = runtime.get_sm90_carrier(tensor)
        assert carrier.deferred_wgrad_count == 1
    assert wrapped() == "done"
    assert carrier.deferred_wgrad_count == 0
    runtime.release_sm90_invocation(invocation)
    assert runtime.sm90_registry_stats() == {
        "carriers": 0, "keys": 0, "invocations": 0,
        "retained_invocations": 0, "invocation_owners": 0,
        "pending_payloads": 0,
    }


def test_registry_metadata_is_weak_and_100_forward_only_calls_cleanup(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = weight_block_size = 32

    for _ in range(100):
        source = torch.zeros((2, 2))
        source_ref = weakref.ref(source)
        with runtime.sm90_invocation_scope(
            Config(), "swiglu", backend="hopper_native", capability=(9, 0)
        ) as invocation:
            carrier = runtime.register_sm90_carrier(
                source, data=torch.ones((2, 2)), role="z"
            )
            assert runtime.resolve_sm90_metadata_tensor(
                carrier.metadata, "source"
            ) is source
        runtime.release_sm90_invocation(invocation)
        del carrier, source
        gc.collect()
        assert source_ref() is None
    assert not any(runtime.sm90_registry_stats().values())


def test_forward_exception_cleans_invocation(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = weight_block_size = 32

    with pytest.raises(RuntimeError, match="forward failed"):
        with runtime.sm90_invocation_scope(
            Config(), "swiglu", backend="hopper_native", capability=(9, 0)
        ):
            tensor = torch.zeros((2, 2))
            runtime.register_sm90_carrier(tensor, data=tensor, role="test")
            raise RuntimeError("forward failed")
    assert not any(runtime.sm90_registry_stats().values())


def test_two_live_microbatches_require_explicit_invocation(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = weight_block_size = 32

    shared = torch.zeros((2, 2))
    invocations = []
    for value in (1.0, 2.0):
        with runtime.sm90_invocation_scope(
            Config(), "swiglu", backend="hopper_native", capability=(9, 0)
        ) as invocation:
            runtime.register_sm90_carrier(
                shared, data=torch.full_like(shared, value), role="test"
            )
        invocations.append(invocation)
    assert runtime.get_sm90_carrier(shared) is None
    with pytest.raises(RuntimeError, match="multiple SM90 invocations"):
        runtime.get_sm90_carrier(shared, required=True)
    with runtime.use_sm90_invocation(invocations[1]):
        assert runtime.get_sm90_carrier(shared).data[0, 0] == 2
    assert runtime.get_sm90_carrier(
        shared, invocation=invocations[0]
    ).data[0, 0] == 1
    for invocation in invocations:
        runtime.release_sm90_invocation(invocation)


def test_wgrad_invocation_resolves_from_operands_with_two_live_microbatches(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")
    wgrad = importlib.import_module("sonicmoe.quack_utils.bf16_wgrad_gemm")

    class Config:
        activation_group_size = weight_block_size = 32

    pairs = []
    for value in (1.0, 2.0):
        a = torch.zeros((2, 3))
        b = torch.zeros((3, 4))
        with runtime.sm90_invocation_scope(
            Config(), "swiglu", backend="hopper_native", capability=(9, 0)
        ) as invocation:
            runtime.bind_sm90_payload(
                a, torch.full_like(a, value), None, 32, role="activation"
            )
            runtime.bind_sm90_payload(
                b, torch.full_like(b, value + 10), None, 32, role="dz"
            )
        pairs.append((invocation, a, b))

    assert runtime.unique_retained_sm90_invocation() is None
    invocation, a, b = pairs[1]
    a_carrier, b_carrier = wgrad._resolve_sm90_wgrad_pair(a.mT, b)
    assert a_carrier.invocation_id == invocation.invocation_id
    assert b_carrier.invocation_id == invocation.invocation_id
    assert a_carrier.data[0, 0] == 2
    assert b_carrier.data[0, 0] == 12

    for retained, _, _ in pairs:
        runtime.release_sm90_invocation(retained)


def test_wgrad_invocation_synthesizes_missing_peer_carrier(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")
    wgrad = importlib.import_module("sonicmoe.quack_utils.bf16_wgrad_gemm")

    class Config:
        activation_group_size = weight_block_size = 32

    for registered_operand in ("a", "b"):
        a = torch.zeros((2, 3))
        b = torch.zeros((3, 4))
        with runtime.sm90_invocation_scope(
            Config(), "swiglu", backend="hopper_native", capability=(9, 0)
        ) as invocation:
            tensor = a if registered_operand == "a" else b
            runtime.register_sm90_carrier(tensor, role=registered_operand)
        a_carrier, b_carrier = wgrad._resolve_sm90_wgrad_pair(a.mT, b)
        assert a_carrier.invocation_id == invocation.invocation_id
        assert b_carrier.invocation_id == invocation.invocation_id
        runtime.release_sm90_invocation(invocation)


def test_pending_payload_owner_and_invocation_cleanup(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = weight_block_size = 32

    with runtime.sm90_invocation_scope(
        Config(), "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        owner = torch.zeros((2, 2))
        runtime.bind_sm90_payload(owner, owner, None, 32)
    assert runtime.sm90_registry_stats()["pending_payloads"] == 1
    del owner
    gc.collect()
    assert runtime.sm90_registry_stats()["pending_payloads"] == 0
    runtime.release_sm90_invocation(invocation)


def test_output_owner_survives_frozen_input_until_completion(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = weight_block_size = 32

    frozen_hidden = torch.zeros((2, 2), requires_grad=False)
    with runtime.sm90_invocation_scope(
        Config(), "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        payload = torch.ones((2, 2))
        runtime.register_sm90_carrier(frozen_hidden, data=payload, role="test")
        output = torch.ones((2, 2), requires_grad=True)
        runtime.retain_sm90_invocation(invocation, owner=output)
    assert runtime.sm90_registry_stats()["invocation_owners"] == 1
    runtime.complete_sm90_invocation(invocation, output)
    del output, frozen_hidden, payload
    gc.collect()
    assert not any(runtime.sm90_registry_stats().values())


def test_deferred_call_executes_after_output_completion(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        activation_group_size = weight_block_size = 32

    tensor = torch.zeros((2, 2))
    payload = torch.ones((2, 2))
    with runtime.sm90_invocation_scope(
        Config(), "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        runtime.register_sm90_carrier(tensor, data=payload, role="test")
        output = torch.ones((1,), requires_grad=True)
        runtime.retain_sm90_invocation(invocation, owner=output)
        deferred = runtime.wrap_deferred_sm90_call(lambda: "late", tensor)
    runtime.complete_sm90_invocation(invocation, output)
    del output, tensor, payload
    gc.collect()
    assert runtime.sm90_registry_stats()["carriers"] == 1
    assert deferred() == "late"
    gc.collect()
    assert not any(runtime.sm90_registry_stats().values())


def test_runtime_snapshot_is_immutable_and_complete(torch):
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")
    fp8_config = importlib.import_module("sonicmoe.functional.fp8_config")
    cfg = fp8_config._FP8Config()
    cfg.enabled = cfg.fused_gated = cfg.fp8_wgrad = True
    cfg.save_z_fp8 = cfg.recompute_z = cfg.fuse_y1_quant = True
    cfg.epilogue_quant = False
    cfg.activation_group_size = cfg.weight_block_size = 128
    cfg.fp8_backend = "hopper_native"
    with runtime.sm90_invocation_scope(
        cfg, "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        cfg.save_z_fp8 = cfg.recompute_z = cfg.fuse_y1_quant = False
        snapshot = invocation.fp8_config
        assert snapshot.save_z_fp8 and snapshot.recompute_z and snapshot.fuse_y1_quant
        assert not snapshot.epilogue_quant
        assert invocation.profile == (128, 128)
    runtime.release_sm90_invocation(invocation)


@pytest.mark.parametrize(
    "save_z_fp8,recompute_z,fuse_y1_quant",
    list(itertools.product([False, True], repeat=3)),
)
def test_carrier_semantics_matrix(
    torch, save_z_fp8, recompute_z, fuse_y1_quant
):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")

    class Config:
        enabled = fused_gated = fp8_wgrad = True
        fuse_y1_bf16_trunc = False
        swiglu_clamp_value = 0.0
        activation_group_size = weight_block_size = 32
        fp8_backend = "hopper_native"
        epilogue_quant = True

    Config.save_z_fp8 = save_z_fp8
    Config.recompute_z = recompute_z
    Config.fuse_y1_quant = fuse_y1_quant
    calls = {"recipe": 0}

    def recipe():
        calls["recipe"] += 1
        return torch.zeros_like(z), None

    with runtime.sm90_invocation_scope(
        Config(), "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        z = torch.empty((128, 256), dtype=torch.bfloat16, device="cuda")
        y1 = torch.empty((128, 128), dtype=torch.bfloat16, device="cuda")
        z_carrier = runtime.register_sm90_carrier(
            z,
            role="z",
            recipe=recipe if recompute_z else None,
            metadata={
                "mode": "recipe" if recompute_z else ("fp8" if save_z_fp8 else "bf16")
            },
        )
        y1_carrier = runtime.register_sm90_carrier(
            y1,
            data=torch.empty_like(y1, dtype=torch.float8_e4m3fn) if fuse_y1_quant else None,
            scales=torch.empty((128, 4), dtype=torch.uint8, device="cuda") if fuse_y1_quant else None,
            role="y1",
            metadata={"mode": "fp8" if fuse_y1_quant else "bf16"},
        )
        assert z_carrier.metadata["mode"] == (
            "recipe" if recompute_z else ("fp8" if save_z_fp8 else "bf16")
        )
        assert y1_carrier.metadata["mode"] == ("fp8" if fuse_y1_quant else "bf16")
        if recompute_z:
            z_carrier.materialize()
            z_carrier.materialize()
            assert calls["recipe"] == 1
            assert z_carrier.recipe is None
    runtime.release_sm90_invocation(invocation)
    assert runtime.sm90_registry_stats()["carriers"] == 0


@pytest.mark.parametrize("profile", [32, 128])
def test_recompute_fp8_recipe_materializes_into_dgated(torch, profile):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    ref = importlib.import_module("sonicmoe.functional.sm90_reference")
    runtime = importlib.import_module("sonicmoe.quack_utils.sm90_fp8_runtime")
    dgated = importlib.import_module("sonicmoe.quack_utils.gemm_dgated")
    interface = importlib.import_module("sonicmoe.quack_utils.gemm_interface")

    class Config:
        enabled = fused_gated = fp8_wgrad = True
        save_z_fp8 = recompute_z = True
        fuse_y1_quant = epilogue_quant = False
        fuse_y1_bf16_trunc = False
        swiglu_clamp_value = 0.0
        activation_group_size = weight_block_size = profile
        fp8_backend = "hopper_native"

    torch.manual_seed(12000 + profile)
    rows = hidden = intermediate = 128
    experts = 2
    offsets = torch.tensor([0, 64, 128], dtype=torch.int32, device="cuda")
    gather = torch.arange(rows, dtype=torch.int32, device="cuda")
    x = (torch.randn(rows, hidden, device="cuda") * 0.05).to(torch.bfloat16)
    dout = (torch.randn(rows, hidden, device="cuda") * 0.05).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(experts, intermediate, hidden, device="cuda") * 0.05).to(torch.bfloat16)

    with runtime.sm90_invocation_scope(
        Config(), "swiglu", backend="hopper_native", capability=(9, 0)
    ) as invocation:
        xq, xs, _ = ref.quantize_activation_sm90(x, profile)
        dq, ds, _ = ref.quantize_activation_sm90(dout, profile)
        w1q, w1s = ref.quantize_weight_sm90(w1, profile)
        w2q, w2s = ref.quantize_weight_sm90(w2, profile)
        runtime.register_sm90_carrier(x, data=xq, scales=xs, role="activation", metadata={"group_size": profile})
        runtime.register_sm90_carrier(dout, data=dq, scales=ds, role="activation", metadata={"group_size": profile})
        runtime.register_sm90_carrier(w1, data=w1q, scales=w1s, role="weight", metadata={"block_size": profile})
        runtime.register_sm90_carrier(w2, data=w2q, scales=w2s, role="weight", metadata={"block_size": profile})
        z = torch.empty((rows, 2 * intermediate), dtype=torch.bfloat16, device="cuda")
        z_carrier = runtime.register_sm90_carrier(
            z, role="z", recipe=lambda: ref.blockscaled_fp8_gemm_zeromat_quant(
                xq, w1q, offsets, gather, xs, w1s
            ), metadata={"mode": "recipe", "group_size": 32, "scale_abi": "raw_ue8m0_1x32"}
        )
        preact, preact_scales = dgated._native_preact_payload(z)
        assert z_carrier.data is preact and z_carrier.scales is preact_scales
        assert z_carrier.metadata["consumed_by_dgated"]
        assert z_carrier.metadata["group_size"] == 32
        assert preact_scales.dtype == torch.uint8
        assert tuple(preact_scales.shape) == (rows, 2 * intermediate // 32)
        assert torch.isfinite(ref._decode_ue8m0(preact_scales)).all()
        assert int(preact_scales.to(torch.int32).max().item()) <= 254

        dequant_z = ref.dequantize_activation_sm90(preact, preact_scales, 32)
        expected_z = torch.empty_like(dequant_z)
        expert_gather = gather.long()
        for expert in range(experts):
            start, end = int(offsets[expert]), int(offsets[expert + 1])
            x_ref = ref.dequantize_activation_sm90(xq, xs, profile)[expert_gather[start:end]]
            w_ref = ref.dequantize_weight_sm90(w1q, w1s, profile)[expert]
            expected_z[start:end] = x_ref.float().matmul(w_ref.float().mT).to(torch.bfloat16)
        assert torch.allclose(dequant_z.float(), expected_z.float(), rtol=0.2, atol=0.08)

        dz = torch.empty((rows, 2 * intermediate), dtype=torch.bfloat16, device="cuda")
        y1s = torch.empty((rows, intermediate), dtype=torch.bfloat16, device="cuda")
        config = interface.default_config(dout.device)
        dgated.gemm_dgated(
            dout, w2, dz, z, y1s, None, "swiglu",
            config.tile_m, config.tile_n, config.cluster_m, config.cluster_n,
            config.pingpong, persistent=True,
            max_swizzle_size=config.max_swizzle_size,
            cu_seqlens_m=offsets, A_idx=gather,
        )
        assert torch.isfinite(dz).all() and torch.isfinite(y1s).all()
    runtime.release_sm90_invocation(invocation)


@pytest.mark.parametrize(
    "offsets,total_k,error",
    [
        ([32, 64], 64, "start at 0"),
        ([0, 64, 32], 32, "monotonically"),
        ([0, 32], 64, "end at total_K"),
    ],
)
def test_colwise_offsets_contract(torch, offsets, total_k, error):
    ref = importlib.import_module("sonicmoe.functional.sm90_reference")
    tensor = torch.tensor(offsets, dtype=torch.int32)
    with pytest.raises(ValueError, match=error):
        ref._validated_colwise_offsets(tensor, total_k)
