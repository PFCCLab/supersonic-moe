"""SM90 FP8 facade paths selected by the shared FP8 config flags.

Covers the options the SM90 flow shares with SM100: no-z / FP8-z / fused-y1
gated forward, z recompute (BF16 and FP8) and the FP8-preact dgated.
"""

from __future__ import annotations

import importlib
import types

import pytest


@pytest.fixture(scope="module")
def torch():
    paddle = pytest.importorskip("paddle")
    paddle.enable_compat()
    return pytest.importorskip("torch")


def _require_sm90(torch):
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)):
        pytest.skip("requires exact SM90")


def _cfg(profile, **flags):
    base = dict(
        activation_group_size=profile,
        weight_block_size=profile,
        recompute_z=False,
        save_z_fp8=False,
        epilogue_quant=False,
        fuse_y1_quant=False,
        fuse_y1_bf16_trunc=False,
        swiglu_clamp_value=0.0,
    )
    base.update(flags)
    return types.SimpleNamespace(**base)


def _calc_diff(torch, actual, expected):
    actual = actual.float()
    expected = expected.float()
    denominator = (actual.square() + expected.square()).sum()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((1 - 2 * (actual * expected).sum() / denominator).item())


def _problem(torch, profile):
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    torch.manual_seed(7300 + profile)
    source_rows, rows, hidden, width, experts = 193, 160, 256, 512, 4
    offsets = torch.tensor([0, 35, 35, 112, rows], dtype=torch.int32, device="cuda")
    route = ((torch.arange(rows, device="cuda") * 37 + 11) % source_rows).int()
    x = torch.randn(source_rows, hidden, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(experts, width, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
    x_fp8, x_scales = quant.quantize_activation_blockscaled_fast(x, group_size=profile)
    w1_fp8 = ref.quantize_weight_sm90(w1, profile)
    return dict(
        x=x, x_fp8=x_fp8, x_scales=x_scales, w1=w1, w1_fp8=w1_fp8,
        offsets=offsets, route=route, rows=rows, width=width, experts=experts,
        hidden=hidden,
    )


def _gated(torch, K, p, cfg):
    F = importlib.import_module("sonicmoe.functional")
    return K.gemm_gated(
        None, p["w1"], p["offsets"], p["route"],
        x_fp8_pre=(p["x_fp8"], p["x_scales"]),
        w1_fp8=p["w1_fp8"], cfg=cfg, current_stream=None,
        activation_type=F.ActivationType.SWIGLU,
    )


@pytest.fixture(scope="module")
def modules(torch):
    _require_sm90(torch)
    F = importlib.import_module("sonicmoe.functional")
    K = importlib.import_module("sonicmoe.functional.fp8_kernels")
    quant = importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm")
    ref = importlib.import_module("sonicmoe.quack_utils.sm90_funcs")
    return F, K, quant, ref


@pytest.mark.parametrize("profile", [32, 128])
def test_sm90_gated_forward_config_variants(torch, modules, profile):
    F, K, quant, ref = modules
    p = _problem(torch, profile)
    z_ref, y1_ref, y1q_ref, y1s_ref, z_payload = _gated(torch, K, p, _cfg(profile))
    assert tuple(z_ref.stride()) != (0, 0) and tuple(y1_ref.stride()) != (0, 0)
    assert z_payload is None
    z_expected, zs_expected = quant.quantize_activation_blockscaled_fast(z_ref)

    # recompute_z: z not materialized, y1 identical.
    z, y1, y1q, y1s, z_payload = _gated(torch, K, p, _cfg(profile, recompute_z=True, save_z_fp8=True,
                                                           epilogue_quant=True))
    assert tuple(z.stride()) == (0, 0) and tuple(z.shape) == tuple(z_ref.shape)
    assert z_payload is None
    assert bool((y1 == y1_ref).all().item())
    assert bool((y1q.view(torch.uint8) == y1q_ref.view(torch.uint8)).all().item())

    # epilogue_quant + save_z_fp8: E4M3 z + raw 1x32 scales returned.
    z, y1, _, _, z_payload = _gated(torch, K, p, _cfg(profile, save_z_fp8=True, epilogue_quant=True))
    assert tuple(z.stride()) == (0, 0)
    assert z_payload is not None
    z_fp8, z_scales = z_payload
    assert z_fp8.dtype == torch.float8_e4m3fn and tuple(z_scales.shape) == (p["rows"], p["width"] // 32)
    # Epilogue scales come from the fp32 accumulator, the reference from the
    # BF16-rounded z: only amax values sitting on a power-of-2 edge may differ.
    mismatched = int((z_scales.view(torch.uint8) != zs_expected.view(torch.uint8)).sum().item())
    assert mismatched <= z_scales.numel() // 100
    z_deq = ref.materialize_activation_sm90(z_fp8, z_scales, 32)
    assert _calc_diff(torch, z_deq, z_ref) < 1e-3
    assert bool((y1 == y1_ref).all().item())

    # fuse_y1_quant: fused only for G == 32 (down GEMM consumes y1 in the G layout).
    z, y1, y1q, y1s, _ = _gated(torch, K, p, _cfg(profile, fuse_y1_quant=True))
    assert bool((z == z_ref).all().item())
    if profile == 32:
        assert tuple(y1.stride()) == (0, 0)
    else:
        assert bool((y1 == y1_ref).all().item())
    assert tuple(y1s.shape) == tuple(y1s_ref.shape)
    y1_deq = ref.materialize_activation_sm90(y1q, y1s, profile)
    y1_ref_deq = ref.materialize_activation_sm90(y1q_ref, y1s_ref, profile)
    # The fused epilogue quantizes the fp32 y1, the reference the BF16 y1, so
    # a few E4M3 values round differently.
    assert _calc_diff(torch, y1_deq, y1_ref_deq) < 1e-3
    assert _calc_diff(torch, y1_deq, y1_ref) < 5e-3


@pytest.mark.parametrize("profile", [32, 128])
@pytest.mark.parametrize("prequant", [True, False])
def test_sm90_recompute_z_matches_forward(torch, modules, profile, prequant):
    F, K, quant, ref = modules
    p = _problem(torch, profile)
    cfg = _cfg(profile)
    z_ref, _, _, _, _ = _gated(torch, K, p, cfg)
    if prequant:
        args = (None, p["w1"], p["offsets"], p["route"], p["x_fp8"], p["x_scales"], p["w1_fp8"])
    else:
        args = (p["x"], p["w1"], p["offsets"], p["route"], None, None, p["w1_fp8"])
    z = K.recompute_z(*args, cfg=cfg, fp8=False)
    assert z.dtype == torch.bfloat16
    assert bool((z == z_ref).all().item())

    z_fp8, z_scales = K.recompute_z(*args, cfg=cfg, fp8=True)
    epi_fp8, epi_scales = _gated(
        torch, K, p, _cfg(profile, save_z_fp8=True, epilogue_quant=True)
    )[4]
    assert bool((z_scales == epi_scales).all().item())
    assert bool((z_fp8.view(torch.uint8) == epi_fp8.view(torch.uint8)).all().item())


@pytest.mark.parametrize("profile", [32, 128])
def test_sm90_dgated_fp8_preact_matches_dequantized_bf16(torch, modules, profile):
    F, K, quant, ref = modules
    p = _problem(torch, profile)
    torch.manual_seed(7400 + profile)
    rows, width, hidden = p["rows"], p["width"], p["hidden"]
    source_rows = int(p["x"].shape[0])
    dout = torch.randn(source_rows, hidden, dtype=torch.bfloat16, device="cuda")
    dout_fp8, dout_scales = quant.quantize_activation_blockscaled_fast(dout, group_size=profile)
    w2T = torch.randn(p["experts"], width // 2, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
    w2T_fp8, w2T_scales = ref.quantize_weight_sm90(w2T, profile)
    z = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    z_fp8, z_scales = quant.quantize_activation_blockscaled_fast(z)
    z_deq = ref.materialize_activation_sm90(z_fp8, z_scales, 32)
    score = torch.rand(rows, dtype=torch.float32, device="cuda")
    cfg = _cfg(profile)

    common = dict(
        w2_fp8=w2T_fp8, w2_scales=w2T_scales, colvec_scale=score,
        expert_frequency_offset=p["offsets"], x_gather_idx=p["route"],
        activation="swiglu", cfg=cfg,
    )
    dz_b, y1s_b, ds_b = K.gemm_dgated(
        dout_fp8, dout_scales, preact=z_deq, preact_fp8=None, preact_scales=None, **common
    )
    dz_f, y1s_f, ds_f = K.gemm_dgated(
        dout_fp8, dout_scales, preact=None, preact_fp8=z_fp8,
        preact_scales=z_scales.view(torch.uint8), **common,
    )
    assert _calc_diff(torch, dz_f, dz_b) < 1e-5
    assert _calc_diff(torch, y1s_f, y1s_b) < 1e-5
    assert _calc_diff(torch, ds_f, ds_b) < 1e-5
