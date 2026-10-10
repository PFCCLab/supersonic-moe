from __future__ import annotations

import importlib

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


def _calc_diff(torch, actual, expected):
    actual = actual.float()
    expected = expected.float()
    denominator = (actual.square() + expected.square()).sum()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((1 - 2 * (actual * expected).sum() / denominator).item())


def _assert_tiled_diff(torch, actual, expected, *, experts=None, row_tile=64, col_tile=128):
    if experts is None:
        experts = [(0, int(actual.shape[-2]))]
    for expert, (start, end) in enumerate(experts):
        for row in range(start, end, row_tile):
            for col in range(0, int(actual.shape[-1]), col_tile):
                lhs = actual[row:min(row + row_tile, end), col:col + col_tile]
                rhs = expected[row:min(row + row_tile, end), col:col + col_tile]
                diff = _calc_diff(torch, lhs, rhs)
                assert diff < 5e-5, (
                    f"expert={expert} rows={row}:{min(row + row_tile, end)} "
                    f"cols={col}:{min(col + col_tile, actual.shape[-1])} diff={diff}"
                )


def _nonuniform_matrix(torch, rows, cols, *, phase):
    row = torch.arange(rows, device="cuda", dtype=torch.float32)[:, None]
    col = torch.arange(cols, device="cuda", dtype=torch.float32)[None, :]
    values = torch.sin((row + 1) * (col + phase) * 0.0017)
    values += 0.5 * torch.cos((row + phase) * (col + 1) * 0.0009)
    group_exp = ((row.to(torch.int32) + col.to(torch.int32) // 128) % 3 - 1).float()
    values *= torch.exp(group_exp * 0.6931471805599453) * 0.05
    return values.to(torch.bfloat16)


def test_non_aligned_wgrad_segments_fresh_add_beta(torch):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
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
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")

    torch.manual_seed(9100 + profile)
    rows, hidden, output, experts = 256, 128, 128, 4
    x = (torch.randn(rows, hidden, device="cuda") * 0.02).to(torch.bfloat16)
    w = (torch.randn(experts, output, hidden, device="cuda") * 0.02).to(torch.bfloat16)
    xq, xs = quant.quantize_activation_blockscaled_fast(x, group_size=profile)
    wq, ws = ref.quantize_weight_sm90(w, profile)
    offsets = torch.tensor([0, 128, 128, 256, 256], dtype=torch.int32, device="cuda")
    gather = torch.arange(rows, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, output), dtype=torch.bfloat16, device="cuda")
    scaled.sm90_scaled_gemm(
        xq, wq, out, xs, ws, cu_seqlens_m=offsets, A_idx=gather,
        activation_group=profile, weight_block=profile,
        activation_scale_order="source",
    )
    expected_out = torch.zeros_like(out)
    x_restored = ref.materialize_activation_sm90(xq, xs, profile)
    w_restored = ref.dequantize_weight_sm90(wq, ws, profile)
    for expert in range(experts):
        start = int(offsets[expert].item())
        end = int(offsets[expert + 1].item())
        if start < end:
            expected_out[start:end] = x_restored[start:end] @ w_restored[expert].T
    denominator = (out.float().square() + expected_out.float().square()).sum()
    diff = 1 - 2 * (out.float() * expected_out.float()).sum() / denominator
    assert float(diff.item()) < 5e-5
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


@pytest.mark.parametrize("source_rows", [271, 160], ids=["source-route-differ", "equal-nonidentity"])
@pytest.mark.parametrize("hidden", [128, 1024])
def test_sm90_profile128_down_production_oracle(torch, source_rows, hidden):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")

    route_rows, width, experts = 160, 384, 4
    offsets_host = [0, 35, 35, 112, route_rows]
    offsets = torch.tensor(offsets_host, dtype=torch.int32, device="cuda")
    route = ((torch.arange(route_rows, device="cuda") * 37 + 11) % source_rows).to(torch.int32)
    source = _nonuniform_matrix(torch, source_rows, hidden, phase=3.0)
    weight = torch.stack(
        [_nonuniform_matrix(torch, width, hidden, phase=17.0 + expert * 9) for expert in range(experts)]
    )
    data, source_scales = quant.quantize_activation_blockscaled_fast(source, group_size=128)
    weight_data, weight_scales = ref.quantize_weight_sm90(weight, 128)
    routed_scales = source_scales[route.long()].contiguous()
    assert int((route != torch.arange(route_rows, device="cuda", dtype=torch.int32)).sum().item()) > 0

    source_dequant = ref.materialize_activation_sm90(data, source_scales, 128)
    weight_dequant = ref.dequantize_weight_sm90(weight_data, weight_scales, 128)
    expected = torch.empty((route_rows, width), dtype=torch.bfloat16, device="cuda")
    for expert, (start, end) in enumerate(zip(offsets_host, offsets_host[1:])):
        if start != end:
            expected[start:end] = source_dequant[route[start:end].long()].float().matmul(
                weight_dequant[expert].float().mT
            ).to(torch.bfloat16)

    output = torch.empty_like(expected)
    scaled.sm90_scaled_gemm(
        data, weight_data, output, source_scales, weight_scales,
        cu_seqlens_m=offsets, A_idx=route,
        activation_group=128, weight_block=128,
        activation_scale_order="source",
    )
    _assert_tiled_diff(
        torch, output, expected,
        experts=list(zip(offsets_host, offsets_host[1:])), row_tile=64, col_tile=128,
    )
    routed_output = torch.empty_like(expected)
    scaled.sm90_scaled_gemm(
        data, weight_data, routed_output, routed_scales, weight_scales,
        cu_seqlens_m=offsets, A_idx=route,
        activation_group=128, weight_block=128,
        activation_scale_order="route",
    )
    assert _calc_diff(torch, routed_output, expected) < 5e-5


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
    reference = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    rows, hidden, width = 128, 128, 256
    torch.manual_seed(8000 + profile + z_quant * 3 + y1_quant)
    a_bf16 = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    b_bf16 = (
        torch.randn((1, width, hidden), device="cuda", dtype=torch.bfloat16)
        * 0.001
    )
    a, sa = quant.quantize_activation_blockscaled_fast(a_bf16, profile)
    b, sb = reference.quantize_weight_sm90(b_bf16, profile)
    offsets = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    gather = torch.arange(rows, dtype=torch.int32, device="cuda")
    z_ref = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    y_ref = torch.empty((rows, width // 2), dtype=torch.bfloat16, device="cuda")
    scaled.sm90_scaled_gated(
        a, b, z_ref, y_ref, sa, sb, cu_seqlens_m=offsets, A_idx=gather,
        activation_group=profile, weight_block=profile,
        activation_scale_order="source",
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
        activation_scale_order="source",
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
        assert bool((ys == ys_expected.view(torch.uint8)).all().item())
        y_deq = y.float() * torch.exp((ys.float() - 127) * 0.6931471805599453).repeat_interleave(32, 1)
        y_ref_deq = y_expected.float() * torch.exp(
            (ys_expected.view(torch.uint8).float() - 127) * 0.6931471805599453
        ).repeat_interleave(32, 1)
        denominator = (y_deq.square() + y_ref_deq.square()).sum()
        y_diff = 1 - 2 * (y_deq * y_ref_deq).sum() / denominator
        assert float(y_diff.item()) < 5e-3
        assert float((y_deq - y_ref_deq).abs().mean().item()) < 1e-2
    else:
        # Same fp32 -> SwiGLU -> BF16 path as the reference: must be bit exact
        # (an abs-mean check cannot see a column permutation of tiny y1).
        assert bool((y == y_ref).all().item())


@pytest.mark.parametrize("profile", [32, 128])
def test_sm90_dgated_is_independent_of_output_storage(torch, profile):
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")

    torch.manual_seed(8181 + profile)
    source_rows, rows, hidden, width, experts = 193, 160, 256, 384, 4
    offsets = torch.tensor(
        [0, 35, 35, 112, rows], dtype=torch.int32, device="cuda"
    )
    route = ((torch.arange(rows, device="cuda") * 37 + 11) % source_rows).int()
    activation = torch.randn(
        source_rows, hidden, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        experts, width, hidden, dtype=torch.bfloat16, device="cuda"
    )
    preact = torch.randn(rows, 2 * width, dtype=torch.bfloat16, device="cuda")
    activation_data, activation_scales = quant.quantize_activation_blockscaled_fast(
        activation, group_size=profile
    )
    weight_data, weight_scales = ref.quantize_weight_sm90(weight, profile)

    results = []
    for _ in range(2):
        dz = torch.empty(rows, 2 * width, dtype=torch.bfloat16, device="cuda")
        postact = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        scaled.sm90_scaled_dgated(
            activation_data,
            weight_data,
            dz,
            preact,
            postact,
            activation_scales,
            weight_scales,
            cu_seqlens_m=offsets,
            A_idx=route,
            activation_group=profile,
            weight_block=profile,
            activation_scale_order="source",
        )
        rowwise, rowwise_scales = quant.quantize_activation_blockscaled_fast(dz, group_size=32)
        colwise = ref.quantize_wgrad_colwise_sm90(
            rowwise,
            offsets,
            source_scales=rowwise_scales,
            source_group=32,
        )
        results.append((dz, postact, rowwise, rowwise_scales, colwise))

    for lhs, rhs in zip(results[0][:4], results[1][:4]):
        if lhs.dtype == torch.float8_e4m3fn:
            lhs, rhs = lhs.view(torch.uint8), rhs.view(torch.uint8)
        assert bool((lhs == rhs).all().item())
    for lhs, rhs in zip(results[0][4], results[1][4]):
        if lhs.dtype == torch.float8_e4m3fn:
            lhs, rhs = lhs.view(torch.uint8), rhs.view(torch.uint8)
        assert bool((lhs == rhs).all().item())


@pytest.mark.parametrize(
    "offsets,total_k,error",
    [
        ([32, 64], 64, "start at 0"),
        ([0, 64, 32], 32, "monotonically"),
        ([0, 32], 64, "end at total_K"),
    ],
)
def test_colwise_offsets_contract(torch, offsets, total_k, error):
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    tensor = torch.tensor(offsets, dtype=torch.int32)
    with pytest.raises(ValueError, match=error):
        ref._validated_colwise_offsets(tensor, total_k)


def test_dynamic_shapes_compile_once(torch, monkeypatch):
    """T, TK and packed wgrad K change every MoE step; each kernel compiles once."""
    if not _sm90(torch):
        pytest.skip("requires exact SM90")
    monkeypatch.delenv("SONIC_MOE_DISABLE_SM90_SCALED_TVM_FFI", raising=False)
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    scaled = importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled")
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")
    compiled = scaled._compile_sm90_scaled_tvm_ffi
    keys = lambda name: {key for key in compiled.cache if key[0] == name}
    gemm_keys, gated_keys, wgrad_keys = (
        keys(name) for name in ("Sm90ScaledGemm", "Sm90ScaledGated", "Sm90ScaledWgradVarlenK")
    )

    hidden, width, experts = 256, 256, 4
    weight = torch.stack(
        [_nonuniform_matrix(torch, width, hidden, phase=5.0 + e * 7) for e in range(experts)]
    )
    weight_data, weight_scales = ref.quantize_weight_sm90(weight, 32)
    weight_dequant = ref.dequantize_weight_sm90(weight_data, weight_scales, 32)
    for step, (tokens, counts) in enumerate(
        [(271, [35, 0, 77, 48]), (160, [1, 64, 3, 129]), (513, [200, 200, 0, 100])]
    ):
        offsets_host = [0]
        for count in counts:
            offsets_host.append(offsets_host[-1] + count)
        routed = offsets_host[-1]
        offsets = torch.tensor(offsets_host, dtype=torch.int32, device="cuda")
        route = ((torch.arange(routed, device="cuda") * 37 + step) % tokens).to(torch.int32)
        source = _nonuniform_matrix(torch, tokens, hidden, phase=3.0 + step)
        data, scales = quant.quantize_activation_blockscaled_fast(source, group_size=32)
        source_dequant = ref.materialize_activation_sm90(data, scales, 32)
        expected = torch.zeros((routed, width), dtype=torch.float32, device="cuda")
        for expert, (start, end) in enumerate(zip(offsets_host, offsets_host[1:])):
            if start != end:
                expected[start:end] = (
                    source_dequant[route[start:end].long()].float() @ weight_dequant[expert].float().mT
                )
        output = torch.empty((routed, width), dtype=torch.bfloat16, device="cuda")
        scaled.sm90_scaled_gemm(
            data, weight_data, output, scales, weight_scales, cu_seqlens_m=offsets, A_idx=route,
            activation_group=32, weight_block=32, activation_scale_order="source",
        )
        assert _calc_diff(torch, output, expected) < 5e-5

        z = torch.empty((routed, width), dtype=torch.bfloat16, device="cuda")
        y1 = torch.empty((routed, width // 2), dtype=torch.bfloat16, device="cuda")
        scaled.sm90_scaled_gated(
            data, weight_data, z, y1, scales, weight_scales, cu_seqlens_m=offsets, A_idx=route,
            activation_group=32, weight_block=32, activation_scale_order="source",
        )
        assert _calc_diff(torch, z, expected) < 5e-5

        a_source = _nonuniform_matrix(torch, routed, 64, phase=11.0 + step)
        b_source = _nonuniform_matrix(torch, routed, 96, phase=13.0 + step)
        a, sa, packed = ref.quantize_wgrad_colwise_sm90(a_source, offsets)
        b, sb, _ = ref.quantize_wgrad_colwise_sm90(b_source, offsets)
        d = torch.empty((experts, 64, 96), dtype=torch.float32, device="cuda")
        scaled.sm90_scaled_wgrad_varlen_k(a, b, d, sa, sb, packed)
        _assert_wgrad_close(torch, d, _wgrad_reference(torch, a, b, sa, sb, packed))

    for name, before in (
        ("Sm90ScaledGemm", gemm_keys), ("Sm90ScaledGated", gated_keys),
        ("Sm90ScaledWgradVarlenK", wgrad_keys),
    ):
        assert len(keys(name) - before) <= 1, f"{name} recompiled for new shapes"
        assert keys(name), f"{name} did not use the TVM-FFI compile cache"
