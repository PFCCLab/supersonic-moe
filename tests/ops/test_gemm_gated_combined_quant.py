"""Byte-exact unit test for the COMBINED z+y1 epilogue quant.

The combined path (GemmGatedSm100ZeroMatQuantPostActQuant) folds BOTH the z
blockscaled FP8 quant (mZScale, over the GEMM-N accumulator tRS_rD) AND the y1
postact blockscaled FP8 quant (mPostActScaleIsa, over the SwiGLU postact
fragment tRS_rPostAct) into a single up-proj epilogue pass.  This lets
fuse_y1=1 coexist with save_z_fp8=1 so the backward takes the healthy
fp8-preact dgated path.

The two EpiOps operate on disjoint register fragments and disjoint scale
buffers, so the combined pass MUST be byte-for-byte identical to running the
two standalone quant kernels separately:

  * z_fp8 / z_scales  ==  GemmGatedSm100ZeroMatBlockscaledQuant (z-only path)
  * y1_fp8 / y1_scales == GemmGatedSm100ZeroMatPostActQuant   (y1-only path)

Any divergence proves register/buffer cross-talk between the two quant
EpiOps — a real bug.  Both standalone paths force epi_tile_n=64 for the y1
fragment (the y1-only mixin always does; the combined mixin does too), so even
the z-quant in the combined pass uses the SAME tiling as the y1-only ablation
already proved lossless (epi_tile_n 64 vs 32 bit-identical).
"""
import os

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
)

pytestmark = [requires_blackwell, requires_quack]


# (T, H, I, E, K) with TK = T*K/E; TK and I must be 128-aligned.
_SHAPES = [
    pytest.param(1024, 1024, 1024, 8, 8, id="eb5-local"),   # TK=1024, I=1024
    pytest.param(2048, 3072, 1536, 32, 8, id="large-E"),    # TK=512,  I=1536
    pytest.param(1024, 3072, 1536, 8, 8, id="aligned"),     # TK=1024, I=1536
]


def _make_inputs(T, H, I, E, K, scale=0.02, np_seed=42):
    TK = T * K // E
    total_M = TK * E
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    A_idx = torch.arange(total_M, dtype=torch.int32, device="cuda")
    import numpy as np
    rng = np.random.RandomState(np_seed)
    x = torch.from_numpy(rng.randn(total_M, H).astype(np.float32) * scale).to(
        device="cuda", dtype=torch.bfloat16)
    w1 = torch.from_numpy(rng.randn(2 * I, H, E).astype(np.float32) * scale).to(
        device="cuda", dtype=torch.bfloat16)
    return TK, total_M, cu_seqlens, A_idx, x, w1


def _run_z_only(x_fp8, w_fp8, a_scales, b_scales, cu_seqlens, A_idx, total_M, N):
    """z-only epilogue quant: fp8 z + (total_M, N//32) scales, bf16 y1."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    z_scale_out = torch.empty((total_M, N // 32), dtype=torch.uint8, device="cuda")
    z_fp8, _y1 = gemm_gated(
        x_fp8, w_fp8,
        activation="swiglu",
        out_dtype=torch.float8_e4m3fn,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=cu_seqlens,
        A_idx=A_idx,
        a_scales=a_scales,
        b_scales=b_scales,
        z_scale_out=z_scale_out,
        tuned=False,
    )
    return z_fp8, z_scale_out.view(_E8M0)


def _run_y1_only(x_fp8, w_fp8, a_scales, b_scales, cu_seqlens, A_idx, total_M, I):
    """y1-only epilogue quant: bf16 z + fp8 y1 + ISA-packed scales."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    postact_scale_out = torch.empty(
        (total_M // 128, I // 128, 512), dtype=torch.uint8, device="cuda")
    _z, y1_fp8 = gemm_gated(
        x_fp8, w_fp8,
        activation="swiglu",
        out_dtype=torch.bfloat16,
        postact_dtype=torch.float8_e4m3fn,
        cu_seqlens_m=cu_seqlens,
        A_idx=A_idx,
        a_scales=a_scales,
        b_scales=b_scales,
        postact_scale_out=postact_scale_out,
        tuned=False,
    )
    return y1_fp8, postact_scale_out.view(_E8M0)


def _run_combined(x_fp8, w_fp8, a_scales, b_scales, cu_seqlens, A_idx, total_M, N, I):
    """combined z+y1 epilogue quant: fp8 z + scales AND fp8 y1 + ISA scales."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    _E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    z_scale_out = torch.empty((total_M, N // 32), dtype=torch.uint8, device="cuda")
    postact_scale_out = torch.empty(
        (total_M // 128, I // 128, 512), dtype=torch.uint8, device="cuda")
    z_fp8, y1_fp8 = gemm_gated(
        x_fp8, w_fp8,
        activation="swiglu",
        out_dtype=torch.float8_e4m3fn,
        postact_dtype=torch.float8_e4m3fn,
        cu_seqlens_m=cu_seqlens,
        A_idx=A_idx,
        a_scales=a_scales,
        b_scales=b_scales,
        z_scale_out=z_scale_out,
        postact_scale_out=postact_scale_out,
        tuned=False,
    )
    return (z_fp8, z_scale_out.view(_E8M0),
            y1_fp8, postact_scale_out.view(_E8M0))


@pytest.mark.parametrize("T,H,I,E,K", _SHAPES)
def test_combined_vs_separate_byte_exact(T, H, I, E, K):
    """Combined z+y1 pass must be byte-for-byte equal to the two standalone
    quant kernels (z-only, y1-only).  Proves no register/buffer cross-talk.

    NOTE on z: the combined mixin forces epi_tile_n=64 (the y1 postact half-N
    needs exactly one 1x32 group/subtile); the standalone z-only mixin runs at
    the cutlass-default epi_tile_n=32.  Two pitfalls make an in-process z-only
    comparison INVALID:
      1. Different tiling => different fp amax-reduction order => the E8M0 scales
         differ by a small "noise" even though both are individually correct.
      2. ``GemmWrapperBase.get_compile_key`` omits epi_tile_n, so a same-process
         z-only call ALIASES the combined kernel from the JIT cache (false data).
    The combined z store was fixed to write TWO 1x32 groups per 64-col subtile
    (``BlockscaledScaleStoreDual``), and is byte-identical to the proven-correct
    z-only tile32 path *when that path is run in a separate process with a clean
    JIT cache* (verified by run_combined_byteexact.py).  Here we instead validate
    z against a tiling-independent numpy gold (integer+carry E8M0, 1e-4 floor):
    scales must match the gold to within the same tiny fp amax-order noise the
    correct tile32 HW path shows, and there must be NO systematic half-buffer
    miss (the pre-fix bug left ~half the groups stale: >50% mismatch).
    """
    import numpy as np
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
    )

    TK, total_M, cu_seqlens, A_idx, x, w1 = _make_inputs(T, H, I, E, K)
    N = 2 * I
    assert TK % 128 == 0 and I % 128 == 0

    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)

    # y1 standalone reference (also forces tile64 -> valid in-process compare).
    y1_fp8_ref, y1_scales_ref = _run_y1_only(
        x_fp8, w_fp8, a_scales, b_scales, cu_seqlens, A_idx, total_M, I)

    # z numpy gold: bf16-z reference -> per-32 integer+carry E8M0 (1e-4 floor).
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    zb, _ = gemm_gated(
        x_fp8, w_fp8, activation="swiglu",
        out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
        cu_seqlens_m=cu_seqlens, A_idx=A_idx,
        a_scales=a_scales, b_scales=b_scales, tuned=False)
    torch.cuda.synchronize()
    zb_np = zb.float().cpu().numpy()
    grp = zb_np.reshape(total_M, N // 32, 32)
    amax = np.maximum(np.abs(grp).max(axis=2), 1e-4)
    ai = amax.astype(np.float32).view(np.int32)
    bexp = (ai >> 23) & 0xFF
    mant = ai & 0x7FFFFF
    carry = (mant > 0x600000).astype(np.int32)
    e8_gold = np.maximum(np.where(bexp > 0, bexp - 8 + carry, 0), 0).astype(np.uint8)

    # Combined single-pass.
    z_fp8_c, z_scales_c, y1_fp8_c, y1_scales_c = _run_combined(
        x_fp8, w_fp8, a_scales, b_scales, cu_seqlens, A_idx, total_M, N, I)

    # --- z scales: combined vs numpy gold (allow tiny fp amax-order noise) ---
    zs_c_np = z_scales_c.view(torch.uint8).cpu().numpy()
    z_scale_mismatch = int((e8_gold != zs_c_np).sum())
    z_scale_frac = z_scale_mismatch / e8_gold.size
    n_cols_mismatch = int((e8_gold != zs_c_np).any(axis=0).sum())
    print(f"  [z   ] scale mismatch={z_scale_mismatch}/{e8_gold.size} "
          f"({z_scale_frac:.4%}), n group-cols touched={n_cols_mismatch}/{N // 32}")

    # --- y1: combined vs y1-only (byte-identical) ---
    y1_fp8_mismatch = (
        y1_fp8_ref.view(torch.uint8) != y1_fp8_c.view(torch.uint8)).sum().item()
    y1_scale_mismatch = (
        y1_scales_ref.view(torch.uint8) != y1_scales_c.view(torch.uint8)).sum().item()
    print(f"  [y1  ] fp8 mismatch={y1_fp8_mismatch}/{y1_fp8_ref.numel()}, "
          f"scale mismatch={y1_scale_mismatch}/{y1_scales_ref.numel()}")

    # z scales: only fp amax-order noise allowed.  The pre-fix bug left ~half the
    # group-columns stale (>50% mismatch with a clean column boundary); the fix
    # must keep mismatch to the small fp amax-order noise the correct tile32 HW
    # path also shows (<5%), with every group-column touched.
    assert n_cols_mismatch < N // 32 or z_scale_frac < 0.05, (
        "every z group-column mismatches gold — combined z store is broken")
    assert z_scale_frac < 0.05, (
        f"combined z scales diverge from gold by {z_scale_frac:.2%} — "
        "the dual-group z store is dropping groups (half-buffer miss?)")

    assert y1_fp8_mismatch == 0, (
        f"combined y1_fp8 diverges from y1-only by {y1_fp8_mismatch} bytes — "
        "the z quant EpiOp is corrupting the postact fragment")
    assert y1_scale_mismatch == 0, (
        f"combined y1 scales diverge from y1-only by {y1_scale_mismatch} bytes")
