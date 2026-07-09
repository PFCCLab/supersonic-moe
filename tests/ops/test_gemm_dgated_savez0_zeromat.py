"""SAVE_Z=0 regression for GemmDGatedSm100ZeroMat.

This test isolates the fused backward down-projection operator used when z is
saved as BF16: A/B are blockscaled FP8, A_idx enables zero-materialized gather_A,
and PreAct is BF16 z. The reference is the unfused torch gold from
``test_gemm_dgated.py``.
"""

import pytest
import torch

from tests.ops.conftest import requires_blackwell, requires_quack, rrmse, cosine_sim
from tests.ops.test_gemm_dgated import _setup, _torch_dgated_gold

pytestmark = [requires_blackwell, requires_quack]


def _report(actual, expected, label):
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    mean_abs = (actual.float() - expected.float()).abs().mean().item()
    print(
        f"  [{label}] RRMSE={r:.6f}, cosine={c:.8f}, "
        f"max_abs_err={max_abs:.6f}, mean_abs_err={mean_abs:.6f}"
    )
    return r, c


def _torch_swiglu_postact(z_preact, scale=None):
    z_f32 = z_preact.float()
    y = z_f32[:, 0::2] * torch.sigmoid(z_f32[:, 0::2]) * z_f32[:, 1::2]
    if scale is not None:
        y = y * scale[:, None]
    return y.to(torch.bfloat16)


def _run_zeromat_bf16_preact_case(*, inplace_dz: bool):
    from sonicmoe.functional import _gather_1x32_scales_to_isa
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_direct_fused_dgated,
        quantize_activation_blockscaled_fast,
    )
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
    from sonicmoe.quack_utils.gemm_interface import default_config

    T, H, I, E, K = 128, 128, 128, 8, 8
    TK, total_M, cu_seqlens = _setup(T, H, I, E, K)
    source_rows = T
    dout_source = (torch.randn(source_rows, H, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
    w2 = (torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
    z_preact = (torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()

    base_idx = torch.arange(source_rows, dtype=torch.int32, device="cuda")
    a_idx = base_idx.repeat_interleave(K).reshape(T, K).transpose(0, 1).contiguous().reshape(-1)
    scale = (torch.rand(total_M, dtype=torch.float32, device="cuda") * 0.9 + 0.1).contiguous()

    dout_fp8, dout_raw_scales = quantize_activation_blockscaled_fast(dout_source)
    dout_scales = _gather_1x32_scales_to_isa(dout_raw_scales, a_idx, source_rows, H)
    w2_fp8_enk, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)

    config = default_config(dout_source.device)
    z_for_kernel = z_preact.clone()
    dz = z_for_kernel if inplace_dz else torch.empty_like(z_for_kernel)
    y1s = torch.empty((total_M, I), dtype=torch.bfloat16, device="cuda")
    colvec_reduce = torch.zeros(
        (total_M, (I + config.tile_n - 1) // config.tile_n),
        dtype=torch.float32,
        device="cuda",
    )

    gemm_dgated_kernel(
        dout_fp8,
        w2_fp8_enk,
        dz,
        z_for_kernel,
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
        colvec_scale=scale,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=cu_seqlens,
        A_idx=a_idx,
        a_scales=dout_scales,
        b_scales=w2_scales,
    )
    torch.cuda.synchronize()

    return {
        "dz": dz.detach().clone(),
        "y1s": y1s.detach().clone(),
        "colvec_reduce": colvec_reduce.detach().clone(),
        "z_preact": z_preact,
        "aliased": dz.data_ptr() == z_for_kernel.data_ptr(),
    }


@pytest.mark.parametrize("T,H,I,E,K", [pytest.param(128, 128, 128, 8, 8, id="gather_repeated_aligned")])
def test_fp8_zeromat_bf16_preact_vs_torch(T, H, I, E, K, seed):
    """FP8 ZeroMat dGated with BF16 PreAct must handle real A_idx gather."""
    from sonicmoe.functional import _gather_1x32_scales_to_isa
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_direct_fused_dgated,
        quantize_activation_blockscaled_fast,
    )
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated as gemm_dgated_kernel
    from sonicmoe.quack_utils.gemm_interface import default_config

    TK, total_M, cu_seqlens = _setup(T, H, I, E, K)
    source_rows = T
    dout_source = (torch.randn(source_rows, H, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
    w2 = (torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.2).contiguous()
    z_preact = (torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()

    # Real zero-materialized routing: TK rows gather from a smaller T-row source.
    base_idx = torch.arange(source_rows, dtype=torch.int32, device="cuda")
    a_idx = base_idx.repeat_interleave(K).reshape(T, K).transpose(0, 1).contiguous().reshape(-1)
    scale = (torch.rand(total_M, dtype=torch.float32, device="cuda") * 0.9 + 0.1).contiguous()

    dout_gathered = dout_source[a_idx.long()].contiguous()
    gold_dz = (_torch_dgated_gold(dout_gathered, w2, z_preact, cu_seqlens, E).float() * scale[:, None]).to(torch.bfloat16)
    gold_y1 = _torch_swiglu_postact(z_preact, scale)

    dout_fp8, dout_raw_scales = quantize_activation_blockscaled_fast(dout_source)
    dout_scales = _gather_1x32_scales_to_isa(dout_raw_scales, a_idx, source_rows, H)
    w2_fp8_enk, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)

    config = default_config(dout_source.device)
    dz = torch.empty((total_M, 2 * I), dtype=torch.bfloat16, device="cuda")
    y1s = torch.empty((total_M, I), dtype=torch.bfloat16, device="cuda")
    colvec_reduce = torch.empty(
        (total_M, (I + config.tile_n - 1) // config.tile_n),
        dtype=torch.float32,
        device="cuda",
    )

    gemm_dgated_kernel(
        dout_fp8,
        w2_fp8_enk,
        dz,
        z_preact,
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
        colvec_scale=scale,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=cu_seqlens,
        A_idx=a_idx,
        a_scales=dout_scales,
        b_scales=w2_scales,
    )
    torch.cuda.synchronize()

    print(
        f"  [zeromat bf16 preact] source_rows={source_rows} total_M={total_M} "
        f"dz_absmax={dz.float().abs().max().item():.6e} "
        f"gold_dz_absmax={gold_dz.float().abs().max().item():.6e} "
        f"y1_absmax={y1s.float().abs().max().item():.6e} "
        f"reduce_absmax={colvec_reduce.float().abs().max().item():.6e}"
    )
    r_dz, c_dz = _report(dz, gold_dz, "dz: FP8 ZeroMat BF16-PreAct vs torch")
    r_y1, c_y1 = _report(y1s, gold_y1, "y1s: FP8 ZeroMat BF16-PreAct vs torch")

    assert dz.float().abs().max().item() > 0.0, "ZeroMat BF16-PreAct dz is all zero"
    assert torch.isfinite(dz).all()
    assert torch.isfinite(y1s).all()
    assert torch.isfinite(colvec_reduce).all()
    assert c_dz > 0.95, f"dz cosine too low: {c_dz:.8f}, RRMSE={r_dz:.6f}"
    assert c_y1 > 0.99, f"y1s cosine too low: {c_y1:.8f}, RRMSE={r_y1:.6f}"


def test_fp8_zeromat_bf16_preact_dz_can_alias_z(seed):
    """BF16 PreAct storage can be overwritten by dz after each tile is read."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out_of_place = _run_zeromat_bf16_preact_case(inplace_dz=False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    in_place = _run_zeromat_bf16_preact_case(inplace_dz=True)

    assert in_place["aliased"], "inplace run did not reuse z storage for dz"
    assert torch.equal(in_place["z_preact"], out_of_place["z_preact"])
    assert torch.equal(in_place["dz"], out_of_place["dz"])
    assert torch.equal(in_place["y1s"], out_of_place["y1s"])
    assert torch.equal(in_place["colvec_reduce"], out_of_place["colvec_reduce"])
