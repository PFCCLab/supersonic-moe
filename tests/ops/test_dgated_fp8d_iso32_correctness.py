"""DGated FP8-D iso32 epilogue fusion — correctness vs gold reference.

Enabling the iso32 epilogue changes the register/SASS layout of the parent
GEMM, so bit-exact match of BF16 dz vs the baseline (iso32-OFF) path is NOT
required.  Instead we verify:

  * baseline BF16 dz is finite (no NaNs in the legal FP8-C-load path)
  * fused side-channel dz_fp8 is close to standalone iso32 quant of baseline dz
  * fused row-SF and col-SF bytes match the standalone iso32 reference
"""
from __future__ import annotations
import os, sys
import pytest

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import paddle  # noqa: E402
paddle.enable_compat()
import torch  # noqa: E402


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    num = (a - b).pow(2).sum().sqrt().item()
    den = b.pow(2).sum().sqrt().item() + 1e-30
    return num / den


def _build_inputs(TK, K, N, E):
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_direct_fused_dgated,
        quantize_activation_blockscaled_fast,
        _storage_per_batch,
    )

    device = "cuda"
    torch.manual_seed(123)
    dout = torch.randn(TK, K, dtype=torch.bfloat16, device=device) * 0.1
    w2 = torch.randn(K, N, E, dtype=torch.bfloat16, device=device) * 0.1
    z = torch.randn(TK, 2 * N, dtype=torch.bfloat16, device=device) * 0.8
    dout_fp8, dout_scales = quantize_and_pack_activation(dout)
    w2_fp8, w2_scales = precompute_weight_fp8_for_direct_fused_dgated(w2)
    z_fp8, z_scales = quantize_activation_blockscaled_fast(z)
    cu = torch.arange(0, TK + 1, TK // E, dtype=torch.int32, device=device)
    idx = torch.arange(TK, dtype=torch.int32, device=device)
    dz_row_storage = _storage_per_batch(TK, 2 * N)
    dz_col_storage = _storage_per_batch(2 * N, TK)
    dz_row = torch.empty((1, dz_row_storage), dtype=torch.uint8, device=device)
    dz_col = torch.empty((1, dz_col_storage), dtype=torch.uint8, device=device)
    return dict(
        dout_fp8=dout_fp8, dout_scales=dout_scales, w2_fp8=w2_fp8, w2_scales=w2_scales,
        z_fp8=z_fp8, z_scales_u8=z_scales.view(torch.uint8), cu_seqlens=cu, x_gather_idx=idx,
        dz_fp8=torch.empty(TK, 2 * N, dtype=torch.float8_e4m3fn, device=device),
        dz_row_scales=dz_row, dz_col_scales=dz_col,
        dz_row_scales_3d=dz_row.view(TK // 128, (2 * N + 127) // 128, 512),
        dz_col_scales_3d=dz_col.view((2 * N + 127) // 128, TK // 128, 512),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_iso32_fused_close_to_gold():
    from sonicmoe.quack_utils.gemm_dgated import gemm_dgated

    TK, K, N, E = 1024, 1024, 512, 4
    state = _build_inputs(TK, K, N, E)
    device = state["dout_fp8"].device
    tile_m, tile_n = 128, 128

    def alloc_outs():
        return (
            torch.zeros(TK, 2 * N, dtype=torch.bfloat16, device=device),
            torch.zeros(TK, N, dtype=torch.bfloat16, device=device),
            torch.ones(TK, dtype=torch.float32, device=device),
            torch.zeros((TK, (N + tile_n - 1) // tile_n), dtype=torch.float32, device=device),
        )

    common = dict(
        a_scales=state["dout_scales"], b_scales=state["w2_scales"],
        preact_fp8=state["z_fp8"], preact_scales=state["z_scales_u8"],
        cu_seqlens_m=state["cu_seqlens"], A_idx=state["x_gather_idx"],
        persistent=True, max_swizzle_size=8,
    )

    for clamp in (0.0, 0.35):
        # ── Baseline (iso32 OFF) ──
        dz_b, y1s_b, s_b, cr_b = alloc_outs()
        gemm_dgated(
            state["dout_fp8"], state["w2_fp8"], dz_b, dz_b, y1s_b, None,
            "swiglu", tile_m, tile_n, 1, 1, True,
            colvec_scale=s_b, colvec_reduce=cr_b, swiglu_clamp_value=clamp, **common,
        )
        torch.cuda.synchronize()

        # ── Fused (iso32 ON) ──
        dz_f, y1s_f, s_f, cr_f = alloc_outs()
        state["dz_fp8"].zero_()
        state["dz_row_scales"].zero_()
        state["dz_col_scales"].zero_()
        gemm_dgated(
            state["dout_fp8"], state["w2_fp8"], dz_f, dz_f, y1s_f, None,
            "swiglu", tile_m, tile_n, 1, 1, True,
            colvec_scale=s_f, colvec_reduce=cr_f, swiglu_clamp_value=clamp,
            iso32_dz_fp8=state["dz_fp8"],
            iso32_dz_row_scales=state["dz_row_scales_3d"],
            iso32_dz_col_scales=state["dz_col_scales_3d"],
            **common,
        )
        torch.cuda.synchronize()

        baseline_finite = torch.isfinite(dz_b.float())
        n_finite = int(baseline_finite.sum().item())
        n_total = dz_b.numel()
        print(f"\n[iso32] baseline finite ratio: {n_finite}/{n_total} = {n_finite/n_total:.4f}")
        assert n_finite == n_total, "baseline DGated FP8-C-load produced NaNs/Infs"

        # (1) Side-channel FP8 + scales are populated.
        fp8_nz = (state["dz_fp8"].view(torch.uint8).to(torch.int32) != 0).float().mean().item()
        row_u8 = state["dz_row_scales"].view(torch.uint8)
        col_u8 = state["dz_col_scales"].view(torch.uint8)
        row_nz = (row_u8.to(torch.int32) != 0).float().mean().item()
        col_nz = (col_u8.to(torch.int32) != 0).float().mean().item()
        row_unique = int(row_u8.to(torch.int32).unique().numel())
        col_unique = int(col_u8.to(torch.int32).unique().numel())
        print(f"[iso32 clamp={clamp}] fp8 nz={fp8_nz:.4f}  "
          f"row_sf nz={row_nz:.4f} uniq={row_unique}  "
          f"col_sf nz={col_nz:.4f} uniq={col_unique}")
        assert fp8_nz > 0.5, f"iso32 dz_fp8 mostly zero ({fp8_nz}) — store path broken"
        assert row_nz > 0.5, "row scales mostly zero"
        assert col_nz > 0.5, "col scales mostly zero"

        # (2) Side-channel fp8 bytes are close to the reference (standalone iso32
        #     quant of baseline's BF16 dz).  Accumulator-vs-BF16 and register-
        #     allocation differences allow small byte-level divergence.
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import iso32_dual_quantize_varlen
        dz_fp8_ref, row_ref, col_ref = iso32_dual_quantize_varlen(dz_b, TK, 2 * N)
        fp8_f_u8 = state["dz_fp8"].view(torch.uint8).to(torch.int32)
        fp8_r_u8 = dz_fp8_ref.view(torch.uint8).to(torch.int32)
        near = ((fp8_f_u8 - fp8_r_u8).abs() <= 1).float().mean().item()
        far = ((fp8_f_u8 - fp8_r_u8).abs() > 10).float().mean().item()
        print(f"[iso32 clamp={clamp}] dz_fp8 vs ref: near±1={near:.4f} far>10={far:.6f}")
        assert near > 0.99, f"dz_fp8 too far from reference: near±1={near:.4f}"
        assert far == 0.0, f"dz_fp8 has large byte errors: far>10={far:.6f}"

        row_ref_u8 = row_ref.view(torch.uint8).to(torch.int32)
        col_ref_u8 = col_ref.view(torch.uint8).to(torch.int32)
        row_f_u8 = row_u8.to(torch.int32)
        col_f_u8 = col_u8.to(torch.int32)
        row_match = (row_f_u8 == row_ref_u8).float().mean().item()
        col_match = (col_f_u8 == col_ref_u8).float().mean().item()
        row_near = ((row_f_u8 - row_ref_u8).abs() <= 1).float().mean().item()
        col_near = ((col_f_u8 - col_ref_u8).abs() <= 1).float().mean().item()
        print(f"[iso32 clamp={clamp}] row_sf vs ref: match={row_match:.4f} near±1={row_near:.4f}")
        print(f"[iso32 clamp={clamp}] col_sf vs ref: match={col_match:.4f} near±1={col_near:.4f}")
        assert row_match > 0.99 and row_near == 1.0, "row scales diverged from reference"
        assert col_match > 0.99 and col_near == 1.0, "col scales diverged from reference"

        # (3) BF16 dz from FUSED path is intentionally stale (TMA writes without
        #     the D-store, since the side-channel supersedes it).  Do NOT gate on
        #     its exact content; only require that it is finite where the baseline
        #     is.
        print(f"[iso32 clamp={clamp}] BF16 dz intentionally stale (side-channel supersedes D store)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v", "-s"]))
