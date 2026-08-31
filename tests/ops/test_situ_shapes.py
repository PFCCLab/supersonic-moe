"""Shape coverage for the SiTU-GLU descriptor.

``test_situ_gemm.py`` proves the numerics; it does so on two shapes, neither of
which is the one production runs.  This file adds (a) the exact geometry the
target config lands on and (b) an independent sweep of every GEMM dimension.

Why this is affordable
----------------------
The comment in ``test_situ_gemm.py`` that "every extra shape is another JIT
compile" is wrong, and this file relies on that::

    _compile_gemm_gated_tvm_ffi(...)      # gemm_gated.py:200
        -> no m/n/k in the key; the fake tensors use cute.sym_int()
    @autotune(key=["activation", "dynamic_scheduler"])   # gemm_interface.py:129

Neither the compile cache nor the autotune cache is keyed on a problem shape, so
within one process the *first* descriptor pays for autotune and every subsequent
shape reuses the same compiled kernel.  What is expensive is a new *descriptor*
(and ``gemm_dgated``, whose cold compile is 200-270 s), which is why the sweep
below varies shapes freely but sticks to a single descriptor.

Production geometry
-------------------
From ``conf/online/ernielite_layer43_pretrain_mla_hca_pipeline.yaml`` with
allgather EP=8, traced through PaddleFleet:

===============  ========  ===========================================
quantity         value     source
===============  ========  ===========================================
``E``            512       every rank holds ALL experts under allgather
                           (``token_dispatcher.py:1660``)
``H`` (GEMM K)   2048      ``moe_latent_size``, *not* hidden 4096
                           (``moe_layer.py:277``)
``I``            256       ``moe_intermediate_size 2048 // EP 8``
                           (``moe_layer.py:406-422``)
gated ``N``      512       ``2*I`` (``moe_expert.py:236-240``)
``total_M``      655360    ``65536 tokens * topk 10``
per-expert ``M`` 1280      ``655360/512``; ``= 10*128`` so a balanced
                           router needs zero padding
===============  ========  ===========================================

Note the counter-intuitive part: allgather shards the *intermediate* dimension,
not the expert dimension.  A naive "EP=8 so E=512/8=64" gives 81920x2048, which
has the same FLOPs as the real 655360x256 but an 8x different M and N, i.e. a
completely different tile profile.  Getting this wrong is why the shape is
spelled out with citations.
"""
import os

import pytest
import torch

from tests.ops.conftest import requires_blackwell, requires_quack, rrmse
from tests.ops.situ_reference import (
    deinterleave,
    dsitu_glu_ref_fp64,
    situ_glu_ref_fp64,
)
from tests.ops.test_situ_gemm import (
    BETA,
    BWD_LIMITS,
    FWD_LIMITS,
    LINEAR_BETA,
    SEED,
    _fwd_inputs,
    _grouped_gemm_fp64,
    _report,
    _situ,
)

pytestmark = [requires_blackwell, requires_quack]

# --- production geometry ----------------------------------------------------
PROD_H = 2048
PROD_I = 256
PROD_E = 512
PROD_TK = 1280          # 655360 / 512, = 10 * 128 -> zero padding when balanced
PROD_TOTAL_M = PROD_TK * PROD_E     # 655360

# The full production forward allocates ~18 GB (bf16 activations + the float64
# oracle).  Opt out with SITU_SKIP_HEAVY=1 on a contended box.
_SKIP_HEAVY = os.getenv("SITU_SKIP_HEAVY", "0") == "1"
heavy = pytest.mark.skipif(_SKIP_HEAVY, reason="SITU_SKIP_HEAVY=1")


def _cu_seqlens(counts):
    """int32 prefix sum over per-expert token counts."""
    out = torch.zeros(len(counts) + 1, dtype=torch.int32, device="cuda")
    out[1:] = torch.tensor(counts, dtype=torch.int32, device="cuda").cumsum(0)
    return out


def _uniform(TK, E):
    """``(total_M, cu_seqlens)`` for a perfectly balanced router."""
    return TK * E, _cu_seqlens([TK] * E)


def _fwd(H, I, E, cu_seqlens, total_M, *, fp8=False, beta=BETA,
         linear_beta=LINEAR_BETA, precise=True, x_scale=1.0, label=""):
    """Forward ``gemm_gated`` on an arbitrary routing, vs the float64 oracle.

    Returns ``(rrmse, cosine)`` of the post-activation and prints the metrics.
    The gold path accumulates ``z`` in float64 from the *same* bf16 operands the
    kernel sees, so the comparison isolates kernel accumulation + the activation
    from the input quantisation both sides share.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    x, w1 = _fwd_inputs(total_M, H, I, E, x_scale=x_scale)
    z_gold = _grouped_gemm_fp64(x, w1, cu_seqlens, E, transpose_b=True)
    out_gold, _, _ = situ_glu_ref_fp64(z_gold, beta, linear_beta)
    w1_3d = w1.permute(2, 1, 0).contiguous()            # (E, H, 2I)
    activation = _situ(beta, linear_beta, precise)

    if fp8:
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            precompute_weight_fp8_for_fused_gated,
            quantize_and_pack_activation,
        )

        x_q, a_scales = quantize_and_pack_activation(x)
        w_q, b_scales = precompute_weight_fp8_for_fused_gated(w1)
        _, postact = gemm_gated(
            x_q, w_q, activation=activation, out_dtype=torch.bfloat16,
            postact_dtype=torch.bfloat16, cu_seqlens_m=cu_seqlens,
            a_scales=a_scales, b_scales=b_scales,
        )
    else:
        _, postact = gemm_gated(x, w1_3d, activation=activation,
                                cu_seqlens_m=cu_seqlens)

    kind = "fp8" if fp8 else "bf16"
    print(f"\n{label or kind} H={H} I={I} E={E} total_M={total_M} "
          f"N={2 * I} {activation}")
    assert torch.isfinite(postact.float()).all(), "postact contains NaN/Inf"
    r, c, _, _ = _report(postact, out_gold, f"postact {kind}")
    rrmse_max, cosine_min = FWD_LIMITS[kind]
    assert r < rrmse_max, f"{kind} postact RRMSE {r:.3e} >= {rrmse_max:.1e}"
    assert c > cosine_min, f"{kind} postact cosine {c:.8f} <= {cosine_min}"
    return r, c


# ---------------------------------------------------------------------------
# 1. The real production shape
# ---------------------------------------------------------------------------


@heavy
@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None], ids=["lb25", "lbnone"])
def test_prod_fwd_bf16(linear_beta):
    """E=512, H=2048, I=256, total_M=655360 — the shape the yaml actually runs."""
    total_M, cu = _uniform(PROD_TK, PROD_E)
    assert total_M == PROD_TOTAL_M
    _fwd(PROD_H, PROD_I, PROD_E, cu, total_M, linear_beta=linear_beta,
         label="production bf16")


@heavy
def test_prod_fwd_fp8():
    """Same, through the blockscaled FP8 path (what training uses: fp8=e4m3)."""
    total_M, cu = _uniform(PROD_TK, PROD_E)
    _fwd(PROD_H, PROD_I, PROD_E, cu, total_M, fp8=True, label="production fp8")


@heavy
def test_prod_ragged_fp8():
    """Realistic router imbalance at production geometry, including dead experts.

    FP8 needs every per-expert offset 128-aligned, which the dispatcher
    guarantees (``interface.py:217`` ``block = 128 if fp8 else 1``), so imbalance
    shows up as a varying number of 128-row blocks per expert — including zero.
    A zero-token expert is the case most likely to break a grouped kernel's
    tile-to-expert mapping, and it happens routinely in real training.
    """
    g = torch.Generator().manual_seed(SEED)
    # 128-row blocks per expert, mean 10 (= the balanced 1280), some empty.
    blocks = torch.randint(0, 21, (PROD_E,), generator=g).tolist()
    blocks[0] = 0
    blocks[-1] = 0
    counts = [b * 128 for b in blocks]
    total_M = sum(counts)
    empty = sum(1 for c in counts if c == 0)
    print(f"\nragged: total_M={total_M} empty_experts={empty} "
          f"min={min(counts)} max={max(counts)}")
    assert empty >= 2 and total_M % 128 == 0
    _fwd(PROD_H, PROD_I, PROD_E, _cu_seqlens(counts), total_M, fp8=True,
         label="production fp8 ragged")


# ---------------------------------------------------------------------------
# 2. One dimension at a time
#
# Each sweep holds the other three at the production value (or a cheap stand-in)
# and walks one axis.  The point is not that the numbers change -- they should
# not -- but that no single dimension has a value at which the descriptor stops
# being honoured.  All of these share one descriptor, hence one autotune.
# ---------------------------------------------------------------------------

# gated N = 2*I must be a multiple of 256 (interface.py:158), i.e. I % 128 == 0.
I_SWEEP = [128, 256, 384, 512, 1024, 1536]
# GEMM K.  1280 and 2560 are deliberately not powers of two.
H_SWEEP = [512, 1024, 1280, 2048, 2560, 3072, 4096]
# E=1 degenerates to a plain GEMM; E=512 is production.
E_SWEEP = [1, 2, 8, 64, 128, 512]
# Per-expert token counts (all 128-aligned so the same list works for FP8).
TK_SWEEP = [128, 256, 512, 1280, 4096]


@pytest.mark.parametrize("I", I_SWEEP)
def test_sweep_gated_n(I):
    """Walk the epilogue's N: this is the axis the interleaved gate/up split
    lives on, so a half-vs-interleave mix-up would be N-dependent."""
    total_M, cu = _uniform(256, 8)
    _fwd(PROD_H, I, 8, cu, total_M, label="sweep gated-N")


@pytest.mark.parametrize("H", H_SWEEP)
def test_sweep_k(H):
    """Walk the GEMM K.  The activation does not see K, but the accumulator
    error grows with it, so this is where a too-tight tolerance would show."""
    total_M, cu = _uniform(256, 8)
    _fwd(H, PROD_I, 8, cu, total_M, label="sweep K")


@pytest.mark.parametrize("E", E_SWEEP)
def test_sweep_experts(E):
    """Walk the expert count with the per-expert row count pinned at one tile
    (128), which is the hardest case for the grouped tile scheduler."""
    total_M, cu = _uniform(128, E)
    _fwd(PROD_H, PROD_I, E, cu, total_M, label="sweep experts")


@pytest.mark.parametrize("TK", TK_SWEEP)
def test_sweep_tokens(TK):
    """Walk M.  Production is 1280 rows/expert = 10 tiles."""
    total_M, cu = _uniform(TK, 8)
    _fwd(PROD_H, PROD_I, 8, cu, total_M, label="sweep tokens")


@pytest.mark.parametrize("TK", [200, 384, 1000])
def test_sweep_tokens_unaligned_bf16(TK):
    """BF16 tolerates per-expert counts that are not multiples of 128.

    FP8 does not (``gemm_gated.py:634`` ``total_m % 128``), and the dispatcher
    pads for it, so this is a bf16-only statement -- but it is the case that
    exercises the ragged tail of the last tile in each expert.
    """
    total_M, cu = _uniform(TK, 8)
    _fwd(PROD_H, PROD_I, 8, cu, total_M, label="sweep tokens unaligned")


@pytest.mark.parametrize("I,H,E,TK", [
    (128, 512, 8, 128),          # smallest legal N, small K
    (PROD_I, PROD_H, 8, 256),    # production N/K, few experts
    (1536, 3072, 32, 256),       # the widest shape in GEMM_SHAPES
])
def test_sweep_fp8(I, H, E, TK):
    """A few corners through the FP8 path as well.

    FP8 bypasses the autotuner entirely (``gemm_interface.py:451`` -- blockscaled
    runs the default config because config search can hit
    CUDA_ERROR_ILLEGAL_INSTRUCTION), so these do not pay for tuning; they check
    that the 1x32 e8m0 scaling factors line up at each shape.
    """
    total_M, cu = _uniform(TK, E)
    _fwd(H, I, E, cu, total_M, fp8=True, label="sweep fp8")


# ---------------------------------------------------------------------------
# 3. Backward at the production geometry
# ---------------------------------------------------------------------------


@heavy
def test_prod_bwd():
    """``gemm_dgated`` at E=512, H=2048, I=256, total_M=655360.

    Checks all three outputs in one call because ``gemm_dgated`` is the most
    expensive kernel here to compile (200-270 s cold) and there is no reason to
    pay for it three times: the recomputed post-activation, ``dgate``/``dup`` in
    their own interleaved slots, and the ``colvec_reduce`` router-score
    gradient.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    total_M, cu = _uniform(PROD_TK, PROD_E)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    dout = torch.randn(total_M, PROD_H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(PROD_H, PROD_I, PROD_E, dtype=torch.bfloat16,
                     device="cuda") * (PROD_H ** -0.5)
    z_preact = torch.randn(total_M, 2 * PROD_I, dtype=torch.bfloat16,
                           device="cuda")

    dy1_gold = _grouped_gemm_fp64(dout, w2, cu, PROD_E, transpose_b=False)
    dz_gold, postact_gold = dsitu_glu_ref_fp64(
        z_preact.double(), dy1_gold, BETA, LINEAR_BETA
    )
    colvec_gold = (postact_gold * dy1_gold).sum(dim=-1)

    dx, postact, colvec = gemm_dgated(
        dout, w2.permute(2, 0, 1).contiguous(), z_preact,
        activation=_situ(), colvec_reduce=True, cu_seqlens_m=cu,
    )

    print(f"\nproduction bwd E={PROD_E} H={PROD_H} I={PROD_I} "
          f"total_M={total_M}")
    assert torch.isfinite(dx.float()).all(), "dx contains NaN/Inf"
    checks = [("postact", postact, postact_gold)]
    dgate, dup = deinterleave(dx.float())
    dgate_gold, dup_gold = deinterleave(dz_gold)
    checks += [("dgate", dgate, dgate_gold), ("dup", dup, dup_gold),
               ("colvec", colvec, colvec_gold)]
    for name, act, exp in checks:
        r, c, _, _ = _report(act, exp, f"prod {name}")
        rrmse_max, cosine_min = BWD_LIMITS[name]
        assert r < rrmse_max, f"{name} RRMSE {r:.3e} >= {rrmse_max:.1e}"
        assert c > cosine_min, f"{name} cosine {c:.8f} <= {cosine_min}"

    swapped = rrmse(dgate, dup_gold.float())
    print(f"  [slot-swap control] RRMSE(dgate, dup_gold)={swapped:.3e}")
    assert swapped > 0.5, "dgate/dup too similar for the slot check to bite"


# ---------------------------------------------------------------------------
# 4. Control: the descriptor is still honoured at the production geometry
# ---------------------------------------------------------------------------


def test_prod_geometry_differs_from_swiglu():
    """Repeat the SwiGLU negative control on the production E/H/I.

    ``test_situ_gemm.py`` runs this on the smoke shape.  It is worth repeating
    here because the failure it guards against -- a descriptor that falls through
    to SwiGLU -- could plausibly be shape-dependent (the fast path in
    ``gemm_gated.py:588`` has its own alignment gate, so which path a shape takes
    is itself shape-dependent).  M is dropped to one tile per expert since only
    the descriptor's effect is under test.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M, cu = _uniform(128, PROD_E)
    x, w1 = _fwd_inputs(total_M, PROD_H, PROD_I, PROD_E)
    w1_3d = w1.permute(2, 1, 0).contiguous()
    z_gold = _grouped_gemm_fp64(x, w1, cu, PROD_E, transpose_b=True)
    out_gold, _, _ = situ_glu_ref_fp64(z_gold, BETA, LINEAR_BETA)

    _, situ_out = gemm_gated(x, w1_3d, activation=_situ(), cu_seqlens_m=cu)
    _, swiglu_out = gemm_gated(x, w1_3d, activation="swiglu", cu_seqlens_m=cu)

    print(f"\nproduction-geometry control E={PROD_E} H={PROD_H} I={PROD_I} "
          f"total_M={total_M}")
    r_pair, _, _, _ = _report(situ_out, swiglu_out, "situ vs swiglu")
    r_situ, _, _, _ = _report(situ_out, out_gold, "situ vs situ oracle")
    r_swiglu, _, _, _ = _report(swiglu_out, out_gold, "swiglu vs situ oracle")
    assert r_pair > 0.03, (
        f"situ and swiglu differ by only RRMSE {r_pair:.3e} at the production "
        "geometry; the descriptor is probably being ignored"
    )
    assert r_situ < r_swiglu / 10.0, (
        f"swiglu is nearly as close to the SiTU oracle ({r_swiglu:.3e}) as SiTU "
        f"itself ({r_situ:.3e}); this control cannot distinguish them"
    )
