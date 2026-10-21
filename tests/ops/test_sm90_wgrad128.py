"""SM90 1x128 fp8 wgrad: colwise quant, varlen-K GEMM and the dgated-fused quant."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def torch():
    paddle = pytest.importorskip("paddle")
    paddle.enable_compat()
    return pytest.importorskip("torch")


def _mods(torch):
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)):
        pytest.skip("requires exact SM90")
    return (importlib.import_module("sonicmoe.quack_utils.sm90_funcs"),
            importlib.import_module("sonicmoe.quack_utils.gemm_sm90_scaled"),
            importlib.import_module("sonicmoe.quack_utils.blockscaled_fp8_gemm"))


def _offsets(torch, counts):
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    return offs, torch.tensor(offs, dtype=torch.int32, device="cuda")


def _ref_col128(torch, src, end):
    """Integer+carry UE8M0 per (128-row group, feature); rows >= end are zero."""
    import numpy as np

    k, f = src.shape
    x = src.float().cpu().numpy().copy()
    x[end:] = 0
    x = x.reshape(k // 128, 128, f)
    bits = np.abs(x).max(axis=1).view(np.int32)
    biased = (bits >> 23) & 0xFF
    e8m0 = np.clip(biased - 8 + ((bits & 0x7FFFFF) > 0x600000), 0, None)
    e8m0 = np.where(biased > 0, e8m0, 0).astype(np.int32)
    qs = (np.clip(254 - e8m0, 1, 254) << 23).view(np.float32)
    data = torch.from_numpy(np.ascontiguousarray((x * qs[:, None, :]).reshape(k, f).T)).cuda()
    return data.to(torch.float8_e4m3fn), torch.from_numpy(e8m0.astype(np.uint8)).cuda()


def _same(torch, a, b):
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return bool((a == b).all().item())


@pytest.mark.parametrize("gather", [False, True])
def test_colwise128_matches_reference(torch, gather):
    ref, _, _ = _mods(torch)
    torch.manual_seed(29)
    offs, offsets = _offsets(torch, [256, 0, 384, 128])
    total_k, feature = offs[-1] + 128, 136  # tail group beyond offsets[E]
    src = torch.randn(total_k + 7, feature, device="cuda").to(torch.bfloat16)
    idx = None
    if gather:
        idx = torch.randint(0, src.shape[0], (total_k,), dtype=torch.int32, device="cuda")
    data, scales, packed = ref.quantize_wgrad_colwise128_sm90(
        src if gather else src[:total_k], offsets, gather_idx=idx)
    rows = src[idx.long()] if gather else src[:total_k]
    exp_data, exp_scales = _ref_col128(torch, rows, offs[-1])
    assert packed is offsets
    assert tuple(scales.shape) == (ref._wgrad128_scale_rows(total_k // 128, feature), feature)
    assert _same(torch, data, exp_data)
    assert _same(torch, scales[: total_k // 128], exp_scales)


def test_wgrad128_gemm_matches_dequant_reference(torch):
    ref, _, _ = _mods(torch)
    torch.manual_seed(30)
    offs, offsets = _offsets(torch, [256, 0, 128, 384])
    total_k, m, n = offs[-1], 256, 384
    a = ref.quantize_wgrad_colwise128_sm90(
        torch.randn(total_k, m, device="cuda").to(torch.bfloat16), offsets)
    b = ref.quantize_wgrad_colwise128_sm90(
        torch.randn(total_k, n, device="cuda").to(torch.bfloat16), offsets)

    def deq(op, f):
        s = torch.exp((op[1][: total_k // 128].float() - 127.0) * 0.6931471805599453)  # [G, F]
        return op[0].float() * s.t().repeat_interleave(128, dim=1)[:f]

    a32, b32 = deq(a, m), deq(b, n)
    expect = torch.stack([a32[:, lo:hi] @ b32[:, lo:hi].t() for lo, hi in zip(offs, offs[1:])])
    out = ref.wgrad_gemm_varlen_k(a, b, offsets, M=m, N=n, total_K=total_k,
                                  num_experts=len(offs) - 1, device="cuda",
                                  out_dtype=torch.float32)
    assert bool((out[1] == 0).all().item())  # empty expert
    err = (out - expect).abs().max().item()
    assert err <= 1e-3 * expect.abs().max().item(), err
    acc = torch.ones_like(out)
    ref.wgrad_gemm_varlen_k(a, b, offsets, M=m, N=n, total_K=total_k,
                            num_experts=len(offs) - 1, device="cuda", accumulator=acc)
    assert bool((acc == out + 1).all().item())


def _to_tile_local(torch, data, z):
    """Write fp8 ``data`` [F, K] into bf16 ``z`` [K, F] in the dgated tile-local layout."""
    f, k = data.shape
    t = data.view(torch.uint8).view(f // 128, 128, k // 128, 128).permute(2, 1, 0, 3)
    z.view(torch.uint8).view(k, f // 128, 256)[:, :, :128] = t.reshape(k, f // 128, 128)


@pytest.mark.parametrize("m", [1024, 1792, 4608])
def test_wgrad128_tile_local_a_is_bitwise(torch, m):
    ref, scaled, _ = _mods(torch)
    torch.manual_seed(32)
    offs, offsets = _offsets(torch, [256, 0, 384, 128])
    total_k, n = offs[-1] + 128, 256  # tail group beyond offsets[E]
    a = ref.quantize_wgrad_colwise128_sm90(
        torch.randn(total_k, m, device="cuda").to(torch.bfloat16), offsets)
    b = ref.quantize_wgrad_colwise128_sm90(
        torch.randn(total_k, n, device="cuda").to(torch.bfloat16), offsets)
    z = torch.randn(total_k, m, device="cuda").to(torch.bfloat16)  # stale other half
    _to_tile_local(torch, a[0], z)
    kw = dict(M=m, N=n, total_K=total_k, num_experts=len(offs) - 1, device="cuda",
              out_dtype=torch.float32)
    exp = ref.wgrad_gemm_varlen_k(a, b, offsets, **kw)
    got = ref.wgrad_gemm_varlen_k(ref.TileLocalCol(z, a[1], offsets), b, offsets, **kw)
    assert _same(torch, exp, got)
    acc0, acc1 = torch.ones_like(exp), torch.ones_like(exp)
    kw.pop("out_dtype")
    ref.wgrad_gemm_varlen_k(a, b, offsets, accumulator=acc0, **kw)
    ref.wgrad_gemm_varlen_k(ref.TileLocalCol(z, a[1], offsets), b, offsets,
                            accumulator=acc1, **kw)
    assert _same(torch, acc0, acc1)


@pytest.mark.parametrize("col,skip", [
    (("y1s",), ()), (("dz",), ()), (("y1s", "dz"), ()),
    (("y1s",), ("y1s",)), (("y1s", "dz"), ("y1s",)), (("y1s", "dz"), ("y1s", "dz")),
])
def test_dgated_fused_col_quant_is_bitwise(torch, col, skip):
    ref, scaled, quant = _mods(torch)
    torch.manual_seed(31)
    offs, offsets = _offsets(torch, [256, 0, 384, 128, 128])
    rows, tokens, hidden, inter = offs[-1] + 256, 700, 256, 384
    route = torch.randint(0, tokens, (rows,), dtype=torch.int32, device="cuda")
    a_data, a_scales = quant.quantize_activation_blockscaled_fast(
        (torch.randn(tokens, hidden, device="cuda") * 0.1).to(torch.bfloat16), group_size=128)
    w_data, w_scales = ref.quantize_weight_sm90(
        (torch.randn(len(offs) - 1, inter, hidden, device="cuda") * 0.05).to(torch.bfloat16), 128)
    preact = torch.randn(rows, 2 * inter, device="cuda").to(torch.bfloat16)
    colvec = torch.rand(rows, device="cuda")

    def run(col_quant=(), skip_stores=()):
        dz = torch.zeros(rows, 2 * inter, dtype=torch.bfloat16, device="cuda")
        y1s = torch.zeros(rows, inter, dtype=torch.bfloat16, device="cuda")
        dzq = (torch.empty(rows, 2 * inter, dtype=torch.float8_e4m3fn, device="cuda"),
               torch.empty(rows, 2 * inter // 128, dtype=torch.uint8, device="cuda"))
        outs, kw = {}, {}
        for name, feat in (("y1s", inter), ("dz", 2 * inter)):
            if name in col_quant:
                outs[name] = (
                    torch.empty(feat, rows, dtype=torch.float8_e4m3fn, device="cuda"),
                    torch.empty(ref._wgrad128_scale_rows(rows // 128, feat), feat,
                                dtype=torch.uint8, device="cuda"))
                kw[name + "_col_out"] = outs[name]
        if skip_stores:
            kw["skip_stores"] = skip_stores
        scaled.sm90_scaled_dgated(
            a_data, w_data, dz, preact, y1s, a_scales, w_scales, colvec_scale=colvec,
            cu_seqlens_m=offsets, A_idx=route, activation_group=128, weight_block=128,
            activation_scale_order="source", tile_shape=(128, 192),
            dz_fp8_out=dzq[0], dz_scale_out=dzq[1], **kw)
        return dz, y1s, dzq, outs

    end = offs[-1]
    dz0, y0, q0, _ = run()
    dz1, y1, q1, outs = run(col, skip)
    assert _same(torch, q0[0][:end], q1[0][:end]) and _same(torch, q0[1][:end], q1[1][:end])
    for name, bf16, got in (("y1s", y0, y1), ("dz", dz0, dz1)):
        if name in skip:
            assert bool((got == 0).all().item())  # BF16 store dropped
        else:
            assert _same(torch, bf16[:end], got[:end])
        if name in col:
            exp = ref.quantize_wgrad_colwise128_sm90(bf16, offsets)
            assert _same(torch, outs[name][0][:, :end], exp[0][:, :end])
            assert _same(torch, outs[name][1][: end // 128], exp[1][: end // 128])


@pytest.mark.parametrize("inter", [384, 512, 896, 2304])
def test_dgated_dz_col_inplace_is_bitwise(torch, inter):
    ref, scaled, quant = _mods(torch)
    torch.manual_seed(33)
    offs, offsets = _offsets(torch, [256, 0, 384, 128, 128])
    rows, tokens, hidden, feat = offs[-1] + 256, 700, 256, 2 * inter
    route = torch.randint(0, tokens, (rows,), dtype=torch.int32, device="cuda")
    a_data, a_scales = quant.quantize_activation_blockscaled_fast(
        (torch.randn(tokens, hidden, device="cuda") * 0.1).to(torch.bfloat16), group_size=128)
    w_data, w_scales = ref.quantize_weight_sm90(
        (torch.randn(len(offs) - 1, inter, hidden, device="cuda") * 0.05).to(torch.bfloat16), 128)
    preact = torch.randn(rows, feat, device="cuda").to(torch.bfloat16)
    colvec = torch.rand(rows, device="cuda")
    end = offs[-1]

    def run(inplace):
        z = preact.clone()  # dz_out = preact, as in Down.backward
        y1s = z.view(-1)[: rows * inter].view(rows, inter)
        dzq = (torch.empty(rows, feat, dtype=torch.float8_e4m3fn, device="cuda"),
               torch.empty(rows, feat // 128, dtype=torch.uint8, device="cuda"))
        scale_rows = ref._wgrad128_scale_rows(rows // 128, feat)
        y_col = (torch.empty(inter, rows, dtype=torch.float8_e4m3fn, device="cuda"),
                 torch.empty(ref._wgrad128_scale_rows(rows // 128, inter), inter,
                             dtype=torch.uint8, device="cuda"))
        dz_scale = torch.zeros(scale_rows, feat, dtype=torch.uint8, device="cuda")
        dz_col = (z if inplace else torch.empty(feat, rows, dtype=torch.float8_e4m3fn,
                                                device="cuda"), dz_scale)
        scaled.sm90_scaled_dgated(
            a_data, w_data, z, z, y1s, a_scales, w_scales, colvec_scale=colvec,
            cu_seqlens_m=offsets, A_idx=route, activation_group=128, weight_block=128,
            activation_scale_order="source", tile_shape=(128, 192),
            dz_fp8_out=dzq[0], dz_scale_out=dzq[1], y1s_col_out=y_col, dz_col_out=dz_col,
            skip_stores=("y1s", "dz"), dz_col_inplace=inplace)
        return z, dzq, y_col, dz_col

    _, q0, y0, d0 = run(False)
    z1, q1, y1, d1 = run(True)
    g = end // 128
    for a, b in ((q0[0][:end], q1[0][:end]), (q0[1][:end], q1[1][:end]),
                 (y0[0][:, :end], y1[0][:, :end]), (y0[1][:g], y1[1][:g]), (d0[1][:g], d1[1][:g])):
        assert _same(torch, a, b)
    head = preact[:end].clone()
    _to_tile_local(torch, d0[0][:, :end].contiguous(), head)
    exp = preact.clone()
    exp[:end] = head
    assert _same(torch, exp, z1)  # odd halves and tail rows untouched
    # dw1-style consumer: (dz_col)^T x (some col operand) over the routed rows.
    b = ref.quantize_wgrad_colwise128_sm90(
        torch.randn(rows, 256, device="cuda").to(torch.bfloat16), offsets)
    kw = dict(M=feat, N=256, total_K=rows, num_experts=len(offs) - 1, device="cuda",
              out_dtype=torch.float32)
    exp_w = ref.wgrad_gemm_varlen_k((d0[0], d0[1], offsets), b, offsets, **kw)
    got_w = ref.wgrad_gemm_varlen_k(ref.TileLocalCol(z1, d1[1], offsets), b, offsets, **kw)
    assert _same(torch, exp_w, got_w)
