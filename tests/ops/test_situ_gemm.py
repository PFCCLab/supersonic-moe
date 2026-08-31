"""GEMM-level tests for the SiTU-GLU activation string.

Mirrors ``tests/ops/test_gemm_gated.py`` / ``tests/ops/test_gemm_dgated.py`` but
drives the epilogue with an ``activation`` *descriptor* instead of a plain name::

    encode_situ_activation(4.0, 25.0) -> "situ_glu:b=4.0:lb=25.0"

What each test buys
-------------------
``test_gated_bf16_vs_oracle``
    Forward ``gemm_gated``: the fused epilogue result against the float64 oracle
    applied to an exactly-accumulated ``z``.  Catches a wrong activation, a
    swapped gate/up slot, or a silently-ignored descriptor.
``test_gated_situ_differs_from_swiglu``
    Negative control: the same inputs through ``activation="swiglu"`` must give a
    *different* answer.  Without this, a descriptor that silently fell back to
    SwiGLU could still pass a loose tolerance on small inputs.
``test_gated_fp8_vs_oracle``
    Same forward through the blockscaled FP8 path (RRMSE / cosine, FP8 budget).
``test_dgated_postact_vs_oracle``
    Backward kernel's recomputed post-activation.  This one is *activation
    isolated*: the kernel and the oracle read the identical bf16 ``PreAct``
    tensor, so there is no GEMM error in the comparison at all.
``test_dgated_dx_vs_oracle``
    ``dgate`` / ``dup`` in the interleaved ``(total_M, 2I)`` layout the kernel
    writes, checked slot-by-slot so a re-interleave bug cannot hide.
``test_dgated_colvec_reduce``
    Router-score gradient.  ``quack/gemm_dact.py`` accumulates
    ``postact * D`` *before* ``D`` is scaled by ``colvec_scale``, so the
    reference is ``sum_i(postact[:, i] * (A @ B)[:, i])``.
``test_cache_key_beta``, ``test_cache_key_linear_beta``, ``test_cache_key_precise``
    The betas are baked into the traced kernel as ``Constexpr`` values, so they
    must participate in every JIT / autotune cache key.  Each test runs two
    descriptors back to back in one process and asserts the results (a) differ
    and (b) each match *their own* oracle.  A stale cache silently returns the
    first kernel for the second call, which these catch.
``test_situ_rejects_clamp``, ``test_situ_rejects_bad_descriptor``
    Unsupported configurations must raise rather than quietly computing
    something else.

Run with ``-s`` — every comparison prints its measured metrics.
"""
import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell,
    requires_quack,
    rrmse,
    cosine_sim,
    GEMM_SHAPES,
)
from tests.ops.situ_reference import (
    deinterleave,
    dsitu_glu_ref_fp64,
    situ_glu_ref_fp64,
)

pytestmark = [requires_blackwell, requires_quack]

BETA = 4.0
LINEAR_BETA = 25.0


def _situ(beta=BETA, linear_beta=LINEAR_BETA, precise=True) -> str:
    from sonicmoe.quack_utils.activation_situ import encode_situ_activation

    return encode_situ_activation(beta=beta, linear_beta=linear_beta, precise=precise)


# Same alignment predicate as the SwiGLU GEMM tests (FP8 path needs 128-aligned
# per-expert token counts).  Trimmed to two shapes on purpose: this file is about
# the numerics, and shape coverage lives in test_situ_shapes.py (which also
# carries the real production geometry).  Note that adding a shape here is
# actually cheap -- neither the compile cache (gemm_gated.py:200, fake tensors use
# cute.sym_int()) nor the autotune key (gemm_interface.py:129,
# key=["activation", "dynamic_scheduler"]) contains m/n/k -- what costs is a new
# *descriptor*.
_WANTED_IDS = {"smoke", "large-E"}
_SITU_SHAPES = []
for _p in GEMM_SHAPES:
    _T, _H, _I, _E, _K = _p.values
    _TK = _T * _K // _E
    if _TK % 128 == 0 and _H % 128 == 0 and _I % 128 == 0 and (2 * _I) % 128 == 0:
        if _p.id in _WANTED_IDS:
            _SITU_SHAPES.append(pytest.param(_T, _H, _I, _E, _K, id=_p.id))
SITU_SHAPES = _SITU_SHAPES if _SITU_SHAPES else GEMM_SHAPES[:1]

SMOKE_SHAPE = SITU_SHAPES[0]
# The backward kernel is by far the most expensive thing here to JIT-compile
# (200-270 s cold), and each descriptor below needs its own compile whose
# MLIR/LLVM state stays resident for the life of the process.  Keeping the
# backward and cache-key tests on one shape is therefore about the *descriptor*
# count, not the shape count; the numerics do not depend on M either way.
BWD_SHAPES = [SMOKE_SHAPE]
SEED = 42


# ---------------------------------------------------------------------------
# Setup / gold helpers
# ---------------------------------------------------------------------------


def _setup(T, H, I, E, K):
    """Uniform routing: every expert gets exactly ``T*K/E`` tokens."""
    TK = T * K // E
    total_M = TK * E
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    return TK, total_M, cu_seqlens


def _fwd_inputs(total_M, H, I, E, seed=SEED, x_scale=1.0):
    """``(x, w1)`` scaled so the pre-activation ``z`` lands on ~N(0, x_scale).

    The SwiGLU tests use ``randn * 0.02`` for both operands, which makes
    ``z ~ N(0, 1e-2)``; there SiTU is indistinguishable from a linear function
    (``4*tanh(g/4)*sigmoid(g) -> g/2``) and the tanh/sigmoid saturation that
    actually stresses the kernel is never reached.  Scaling ``w1`` by
    ``1/sqrt(H)`` puts ``z`` in the range the model really sees.

    ``x_scale`` pushes ``z`` further out for the tests that need the
    ``linear_beta`` clamp to actually bite (see ``test_cache_key_linear_beta``).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * x_scale
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * (H ** -0.5)
    return x, w1


def _bwd_inputs(total_M, H, I, E, seed=SEED):
    """``(dout, w2, z_preact)`` with ``dout @ w2`` and ``z_preact`` on ~N(0, 1)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dout = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * (H ** -0.5)
    z_preact = torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda")
    return dout, w2, z_preact


def _grouped_gemm_fp64(a, b_per_expert, cu_seqlens, E, transpose_b):
    """Exact (float64) per-expert grouped GEMM of bf16 operands.

    bf16 -> float64 is lossless, so this isolates the kernel's *accumulation*
    error from the input quantization the kernel and the gold share.
    """
    n = b_per_expert.shape[0] if transpose_b else b_per_expert.shape[1]
    out = torch.zeros(a.shape[0], n, dtype=torch.float64, device=a.device)
    for exp in range(E):
        s = int(cu_seqlens[exp])
        e = int(cu_seqlens[exp + 1])
        if s >= e:
            continue
        w = b_per_expert[:, :, exp].double()
        out[s:e] = a[s:e].double() @ (w.T if transpose_b else w)
    return out


def _fwd_gold(x, w1, cu_seqlens, E, beta, linear_beta):
    """``(z_fp64, out_fp64)`` — exact ``z`` then the float64 SiTU-GLU oracle."""
    z = _grouped_gemm_fp64(x, w1, cu_seqlens, E, transpose_b=True)
    out, _, _ = situ_glu_ref_fp64(z, beta, linear_beta)
    return z, out


def _bwd_gold(dout, w2, z_preact, cu_seqlens, E, beta, linear_beta):
    """``(dy1_fp64, dz_fp64_interleaved, postact_fp64)``.

    ``dy1 = dout @ w2`` is the incoming gradient w.r.t. the post-activation;
    ``dz`` is the gradient w.r.t. the interleaved pre-activation.
    """
    dy1 = _grouped_gemm_fp64(dout, w2, cu_seqlens, E, transpose_b=False)
    dz, postact = dsitu_glu_ref_fp64(z_preact.double(), dy1, beta, linear_beta)
    return dy1, dz, postact


def _report(actual, expected, label):
    a = actual.float()
    e = expected.float()
    r = rrmse(a, e)
    c = cosine_sim(a, e)
    max_abs = (a - e).abs().max().item()
    scale = e.abs().max().item()
    # Error in units of a bf16 ulp at the tensor's own scale (bf16 has 8 mantissa
    # bits, so one ulp at the top of the range is scale * 2^-8).
    ulp = max_abs / (scale * 2.0 ** -8) if scale > 0 else 0.0
    print(f"  [{label}] RRMSE={r:.3e} cosine={c:.8f} max_abs={max_abs:.3e} "
          f"scale={scale:.3e} bf16_ulp={ulp:.2f}")
    return r, c, max_abs, scale


# ---------------------------------------------------------------------------
# One-shot call helpers (shared by the numerics and the cache-key tests)
# ---------------------------------------------------------------------------


def _fwd_call(T, H, I, E, K, beta, linear_beta, precise=True, x_scale=1.0):
    """Run forward ``gemm_gated`` with a SiTU descriptor.

    Returns ``(preact, postact, z_gold_fp64, out_gold_fp64)``.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    _, total_M, cu_seqlens = _setup(T, H, I, E, K)
    x, w1 = _fwd_inputs(total_M, H, I, E, x_scale=x_scale)
    z_gold, out_gold = _fwd_gold(x, w1, cu_seqlens, E, beta, linear_beta)
    w1_3d = w1.permute(2, 1, 0).contiguous()  # (E, H, 2I)
    preact, postact = gemm_gated(
        x,
        w1_3d,
        activation=_situ(beta, linear_beta, precise),
        cu_seqlens_m=cu_seqlens,
    )
    return preact, postact, z_gold, out_gold


def _bwd_call(T, H, I, E, K, beta, linear_beta, precise=True, colvec_reduce=False):
    """Run backward ``gemm_dgated`` with a SiTU descriptor.

    Returns ``(dx, postact, colvec, dz_gold_fp64, postact_gold_fp64,
    colvec_gold_fp64)``; ``colvec`` is ``None`` unless requested.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    _, total_M, cu_seqlens = _setup(T, H, I, E, K)
    dout, w2, z_preact = _bwd_inputs(total_M, H, I, E)
    dy1_gold, dz_gold, postact_gold = _bwd_gold(
        dout, w2, z_preact, cu_seqlens, E, beta, linear_beta
    )
    colvec_gold = (postact_gold * dy1_gold).sum(dim=-1)

    w2_3d = w2.permute(2, 0, 1).contiguous()  # (E, H, I)
    activation = _situ(beta, linear_beta, precise)
    if colvec_reduce:
        dx, postact, colvec = gemm_dgated(
            dout, w2_3d, z_preact,
            activation=activation,
            colvec_reduce=True,
            cu_seqlens_m=cu_seqlens,
        )
    else:
        dx, postact = gemm_dgated(
            dout, w2_3d, z_preact,
            activation=activation,
            cu_seqlens_m=cu_seqlens,
        )
        colvec = None
    return dx, postact, colvec, dz_gold, postact_gold, colvec_gold


# ---------------------------------------------------------------------------
# Thresholds.  Every number below was measured on sm_103 and then rounded up;
# the measured value is in the trailing comment.  ``bf16_ulp`` is the useful
# scale-free unit here: the outputs are bf16, so ~1 ulp is the floor and the
# error budget is dominated by how much the activation amplifies the bf16
# rounding of its own input, not by the f32 activation math (which is ~1e-6,
# see test_situ_activation.py).
# ---------------------------------------------------------------------------

FWD_LIMITS = {  # (rrmse_max, cosine_min)
    # Measured: bf16 1.65e-3 - 1.66e-3 (pure bf16 rounding of the postact, see
    # BWD_LIMITS), FP8 blockscaled 5.40e-2.
    "bf16": (5.0e-3, 0.99998),
    "fp8": (1.0e-1, 0.99),
}


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None], ids=["lb25", "lbnone"])
@pytest.mark.parametrize("T,H,I,E,K", SITU_SHAPES)
def test_gated_bf16_vs_oracle(T, H, I, E, K, linear_beta):
    """BF16 fused forward epilogue vs the float64 SiTU-GLU oracle."""
    preact, postact, z_gold, out_gold = _fwd_call(T, H, I, E, K, BETA, linear_beta)

    print(f"\n{_situ(BETA, linear_beta)} shape T={T} H={H} I={I} E={E} K={K}")
    _report(preact, z_gold, "preact z")
    r, c, _, _ = _report(postact, out_gold, "postact")

    assert torch.isfinite(postact.float()).all(), "postact contains NaN/Inf"
    rrmse_max, cosine_min = FWD_LIMITS["bf16"]
    assert r < rrmse_max, f"forward postact RRMSE {r:.3e} >= {rrmse_max:.1e}"
    assert c > cosine_min, f"forward postact cosine {c:.8f} <= {cosine_min}"


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_gated_situ_differs_from_swiglu(T, H, I, E, K):
    """Negative control: the descriptor must not silently fall back to SwiGLU.

    Both activations are smooth gated products, so on small inputs they are
    numerically close enough that a loose tolerance alone would not notice a
    fallback.  Comparing them directly is the statement with content.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    _, total_M, cu_seqlens = _setup(T, H, I, E, K)
    x, w1 = _fwd_inputs(total_M, H, I, E)
    w1_3d = w1.permute(2, 1, 0).contiguous()

    _, situ_out = gemm_gated(x, w1_3d, activation=_situ(),
                             cu_seqlens_m=cu_seqlens)
    _, swiglu_out = gemm_gated(x, w1_3d, activation="swiglu",
                               cu_seqlens_m=cu_seqlens)

    r, c, max_abs, scale = _report(situ_out, swiglu_out, "situ vs swiglu")
    # Measured 7.73e-2 on the smoke shape.  The two activations are genuinely
    # close in shape -- 4*tanh(g/4)*sigmoid(g) -> g*sigmoid(g) as g -> 0 -- so
    # this floor is deliberately modest; the discriminating assertion is the
    # oracle-distance ratio below, not this one.
    assert r > 0.03, (
        f"situ_glu and swiglu differ by only RRMSE {r:.3e}; the SiTU descriptor "
        "is probably being ignored and swiglu computed instead"
    )
    # And SiTU must still be the one that matches the SiTU oracle.
    _, _, _, out_gold = _fwd_call(T, H, I, E, K, BETA, LINEAR_BETA)
    r_situ, _, _, _ = _report(situ_out, out_gold, "situ vs situ oracle")
    r_swiglu, _, _, _ = _report(swiglu_out, out_gold, "swiglu vs situ oracle")
    assert r_situ < r_swiglu / 10.0, (
        f"swiglu is nearly as close to the SiTU oracle ({r_swiglu:.3e}) as SiTU "
        f"itself ({r_situ:.3e}); this control cannot distinguish them"
    )


@pytest.mark.parametrize("T,H,I,E,K", SITU_SHAPES)
def test_gated_fp8_vs_oracle(T, H, I, E, K):
    """Blockscaled FP8 forward path with a SiTU descriptor vs the oracle."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
    )

    _, total_M, cu_seqlens = _setup(T, H, I, E, K)
    x, w1 = _fwd_inputs(total_M, H, I, E)
    _, out_gold = _fwd_gold(x, w1, cu_seqlens, E, BETA, LINEAR_BETA)

    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)

    _, postact = gemm_gated(
        x_fp8, w_fp8,
        activation=_situ(),
        out_dtype=torch.bfloat16,
        postact_dtype=torch.bfloat16,
        cu_seqlens_m=cu_seqlens,
        a_scales=a_scales,
        b_scales=b_scales,
    )

    print(f"\nFP8 {_situ()} shape T={T} H={H} I={I} E={E} K={K}")
    r, c, _, _ = _report(postact, out_gold, "postact FP8 vs oracle")
    assert torch.isfinite(postact.float()).all(), "FP8 postact contains NaN/Inf"
    rrmse_max, cosine_min = FWD_LIMITS["fp8"]
    assert r < rrmse_max, f"FP8 forward postact RRMSE {r:.3e} >= {rrmse_max:.1e}"
    assert c > cosine_min, f"FP8 forward postact cosine {c:.8f} <= {cosine_min}"


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

BWD_LIMITS = {  # (rrmse_max, cosine_min)
    # bf16 outputs: measured 1.64e-3 - 1.67e-3, i.e. pure bf16 rounding of the
    # result (2^-9 / sqrt(3) = 1.13e-3 for a uniformly distributed round-off,
    # somewhat more once the activation's own sensitivity to the bf16 PreAct is
    # folded in).  There is no headroom left for an actual math error here.
    "postact": (5.0e-3, 0.99998),
    "dgate": (5.0e-3, 0.99997),
    "dup": (5.0e-3, 0.99998),
    # colvec_reduce comes back in float32, so it is ~3 orders tighter:
    # measured 7.71e-7.
    "colvec": (1.0e-5, 0.9999999),
}


@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None], ids=["lb25", "lbnone"])
@pytest.mark.parametrize("T,H,I,E,K", BWD_SHAPES)
def test_dgated_postact_vs_oracle(T, H, I, E, K, linear_beta):
    """Recomputed post-activation from the backward kernel — GEMM-error free.

    ``PreAct`` is handed in as a bf16 tensor and the oracle reads that exact same
    tensor, so the only difference is the f32-vs-f64 activation evaluation plus
    the bf16 store of the result.  This is the tightest GEMM-level statement
    available about the forward math inside the backward epilogue.
    """
    _, postact, _, _, postact_gold, _ = _bwd_call(T, H, I, E, K, BETA, linear_beta)

    print(f"\n{_situ(BETA, linear_beta)} shape T={T} H={H} I={I} E={E} K={K}")
    r, c, _, _ = _report(postact, postact_gold, "dgated postact")
    assert torch.isfinite(postact.float()).all(), "postact contains NaN/Inf"
    rrmse_max, cosine_min = BWD_LIMITS["postact"]
    assert r < rrmse_max, f"dgated postact RRMSE {r:.3e} >= {rrmse_max:.1e}"
    assert c > cosine_min, f"dgated postact cosine {c:.8f} <= {cosine_min}"


@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None], ids=["lb25", "lbnone"])
@pytest.mark.parametrize("T,H,I,E,K", BWD_SHAPES)
def test_dgated_dx_vs_oracle(T, H, I, E, K, linear_beta):
    """``dgate`` / ``dup`` checked in their own interleaved slots.

    Checking the two halves separately (rather than the whole ``(total_M, 2I)``
    buffer at once) is what makes a swapped-slot bug visible: a combined metric
    over a symmetric-looking buffer can stay small while gate and up are
    exchanged.
    """
    dx, _, _, dz_gold, _, _ = _bwd_call(T, H, I, E, K, BETA, linear_beta)

    dgate, dup = deinterleave(dx.float())
    dgate_gold, dup_gold = deinterleave(dz_gold)

    print(f"\n{_situ(BETA, linear_beta)} shape T={T} H={H} I={I} E={E} K={K}")
    assert torch.isfinite(dx.float()).all(), "dx contains NaN/Inf"
    for name, act, exp in (("dgate", dgate, dgate_gold), ("dup", dup, dup_gold)):
        r, c, _, _ = _report(act, exp, f"dgated {name}")
        rrmse_max, cosine_min = BWD_LIMITS[name]
        assert r < rrmse_max, f"dgated {name} RRMSE {r:.3e} >= {rrmse_max:.1e}"
        assert c > cosine_min, f"dgated {name} cosine {c:.8f} <= {cosine_min}"

    # Slot sanity: swapping the two halves must be measurably worse, otherwise
    # the assertions above would not detect an interleave bug on this data.
    swapped = rrmse(dgate.float(), dup_gold.float())
    print(f"  [dgated slot-swap control] RRMSE(dgate, dup_gold)={swapped:.3e}")
    assert swapped > 0.5, (
        f"dgate and dup are too similar (swap RRMSE {swapped:.3e}) for this test "
        "to detect an interleave bug"
    )


@pytest.mark.parametrize("T,H,I,E,K", BWD_SHAPES)
def test_dgated_colvec_reduce(T, H, I, E, K):
    """Router-score gradient: ``sum_i(postact[:, i] * (A @ B)[:, i])``.

    ``quack/gemm_dact.py`` accumulates ``postact * D`` *before* ``D`` is scaled by
    ``colvec_scale`` (comment at the ``colvec_reduce_accumulate`` call site), and
    ``gemm_dgated_tuned`` then sums the per-N-tile partials.  With
    ``colvec_scale=None`` the reference is therefore the plain unscaled product.
    """
    out = _bwd_call(T, H, I, E, K, BETA, LINEAR_BETA, colvec_reduce=True)
    _, _, colvec, _, _, colvec_gold = out
    assert colvec is not None, "colvec_reduce=True did not return a reduction"
    assert colvec.shape == colvec_gold.shape, (
        f"colvec shape {tuple(colvec.shape)} != expected {tuple(colvec_gold.shape)}"
    )

    print(f"\n{_situ()} shape T={T} H={H} I={I} E={E} K={K}")
    r, c, _, _ = _report(colvec, colvec_gold, "colvec_reduce")
    assert torch.isfinite(colvec.float()).all(), "colvec_reduce contains NaN/Inf"
    rrmse_max, cosine_min = BWD_LIMITS["colvec"]
    assert r < rrmse_max, f"colvec_reduce RRMSE {r:.3e} >= {rrmse_max:.1e}"
    assert c > cosine_min, f"colvec_reduce cosine {c:.8f} <= {cosine_min}"


# ---------------------------------------------------------------------------
# Cache-key regressions
#
# `beta` / `linear_beta` / `precise` are baked into the traced kernel as
# Constexpr values via functools.partial, so they must participate in every JIT
# and autotune cache key.  activation_situ.py encodes them into the activation
# *string* precisely because that string is already part of all of those keys.
# If any layer drops them, the second call in each test below silently reuses the
# first kernel: the two results become identical and the second stops matching
# its own oracle.  Asserting BOTH ("they differ" and "each matches its own
# reference") is what pins this down — "they differ" alone would also be
# satisfied by two equally-wrong kernels.
# ---------------------------------------------------------------------------

# Separation floor between two configurations, as RRMSE of one against the
# other.  Measured values are far above these; the point is only that the
# configurations are distinguishable at all.
_CACHE_SEPARATION = 0.05


def _assert_two_configs(label, res_a, gold_a, res_b, gold_b, min_separation):
    """``res_a`` matches ``gold_a``, ``res_b`` matches ``gold_b``, a != b."""
    a = res_a.float()
    b = res_b.float()
    sep = rrmse(a, b)
    r_a, c_a, _, _ = _report(a, gold_a, f"{label} A vs its own oracle")
    r_b, c_b, _, _ = _report(b, gold_b, f"{label} B vs its own oracle")
    print(f"  [{label} separation] RRMSE(A, B)={sep:.3e}")

    assert sep > min_separation, (
        f"{label}: the two configurations produced near-identical results "
        f"(RRMSE {sep:.3e} <= {min_separation}); the second call almost certainly "
        "reused the first compiled kernel, i.e. the SiTU parameters are missing "
        "from a JIT/autotune cache key"
    )
    rrmse_max = 5.0e-3
    assert r_a < rrmse_max, f"{label}: config A RRMSE {r_a:.3e} >= {rrmse_max:.1e}"
    assert r_b < rrmse_max, f"{label}: config B RRMSE {r_b:.3e} >= {rrmse_max:.1e}"
    # Cross-check: each result must be closer to its own oracle than to the
    # other's, which is the actual statement "the right kernel ran".
    assert rrmse(a, gold_b) > r_a * 10.0, (
        f"{label}: config A is nearly as close to B's oracle as to its own"
    )
    assert rrmse(b, gold_a) > r_b * 10.0, (
        f"{label}: config B is nearly as close to A's oracle as to its own"
    )


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_cache_key_beta_fwd(T, H, I, E, K):
    """beta=4.0 then beta=2.0 in one process, forward."""
    _, post_a, _, gold_a = _fwd_call(T, H, I, E, K, 4.0, LINEAR_BETA)
    _, post_b, _, gold_b = _fwd_call(T, H, I, E, K, 2.0, LINEAR_BETA)
    print("\nfwd beta 4.0 vs 2.0")
    _assert_two_configs("fwd beta", post_a, gold_a, post_b, gold_b, _CACHE_SEPARATION)


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_cache_key_beta_bwd(T, H, I, E, K):
    """beta=4.0 then beta=2.0 in one process, backward ``dx``."""
    dx_a, _, _, gold_a, _, _ = _bwd_call(T, H, I, E, K, 4.0, LINEAR_BETA)
    dx_b, _, _, gold_b, _, _ = _bwd_call(T, H, I, E, K, 2.0, LINEAR_BETA)
    print("\nbwd beta 4.0 vs 2.0")
    _assert_two_configs("bwd beta", dx_a, gold_a, dx_b, gold_b, _CACHE_SEPARATION)


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_cache_key_linear_beta(T, H, I, E, K):
    """lb=25.0 then lb=None — the up branch changes shape, not just a constant.

    ``x_scale`` matters here.  At ``z ~ N(0, 1)`` the clamp is almost inactive
    (``25*tanh(u/25) = u - u^3/1875 + ...``), so the two configurations differ by
    ~2e-3 — the same order as the bf16 rounding of the output, which would make
    the comparison meaningless.  Pushing ``z`` to ~N(0, 8) makes the clamp bite
    in the tails and separates the two by a measurable margin.
    """
    _, post_a, _, gold_a = _fwd_call(T, H, I, E, K, BETA, LINEAR_BETA, x_scale=8.0)
    _, post_b, _, gold_b = _fwd_call(T, H, I, E, K, BETA, None, x_scale=8.0)
    print("\nfwd lb 25.0 vs none (x_scale=8)")
    _assert_two_configs("fwd lb", post_a, gold_a, post_b, gold_b, 1.0e-2)


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_cache_key_precise(T, H, I, E, K):
    """precise=True then precise=False — same math, different tanh/sigmoid.

    The two differ only at the ~1e-4 relative level (measured in
    test_situ_activation.py: geometric-mean RRMSE ratio 29x, worst-case max
    relative error 2.5e-4 for approx), so this asserts the *separation* is
    non-zero and each matches the shared oracle to its own budget, not that the
    outputs are far apart.
    """
    _, post_p, _, gold = _fwd_call(T, H, I, E, K, BETA, LINEAR_BETA, precise=True)
    _, post_a, _, _ = _fwd_call(T, H, I, E, K, BETA, LINEAR_BETA, precise=False)

    print("\nfwd precise vs approx")
    r_p, _, _, _ = _report(post_p, gold, "precise vs oracle")
    r_a, _, _, _ = _report(post_a, gold, "approx  vs oracle")
    sep = rrmse(post_p.float(), post_a.float())
    print(f"  [precise/approx separation] RRMSE={sep:.3e}")

    assert sep > 0.0, (
        "precise and approx produced bit-identical output; the ':approx' suffix "
        "is being dropped from a cache key or from the descriptor parse"
    )
    for tag, r in (("precise", r_p), ("approx", r_a)):
        assert r < 5.0e-3, f"{tag} forward RRMSE {r:.3e} >= 5.0e-3"


# ---------------------------------------------------------------------------
# Rejection: unsupported configurations must raise, not silently change numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_situ_rejects_clamp_fwd(T, H, I, E, K):
    """SiTU-GLU has no clamped variant, so a clamp request must not be dropped."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    _, total_M, cu_seqlens = _setup(T, H, I, E, K)
    x, w1 = _fwd_inputs(total_M, H, I, E)
    w1_3d = w1.permute(2, 1, 0).contiguous()

    with pytest.raises(Exception) as exc:
        gemm_gated(x, w1_3d, activation=_situ(), cu_seqlens_m=cu_seqlens,
                   swiglu_clamp_value=7.0)
    msg = str(exc.value)
    print(f"\n  [fwd clamp rejection] {type(exc.value).__name__}: {msg.splitlines()[0]}")
    assert "swiglu_clamp_value" in msg, (
        f"expected a clamp-rejection error, got: {msg}"
    )


@pytest.mark.parametrize("T,H,I,E,K", [SMOKE_SHAPE])
def test_situ_rejects_clamp_bwd(T, H, I, E, K):
    """Same contract on the backward path."""
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    _, total_M, cu_seqlens = _setup(T, H, I, E, K)
    dout, w2, z_preact = _bwd_inputs(total_M, H, I, E)
    w2_3d = w2.permute(2, 0, 1).contiguous()

    with pytest.raises(Exception) as exc:
        gemm_dgated(dout, w2_3d, z_preact, activation=_situ(),
                    cu_seqlens_m=cu_seqlens, swiglu_clamp_value=7.0)
    msg = str(exc.value)
    print(f"\n  [bwd clamp rejection] {type(exc.value).__name__}: {msg.splitlines()[0]}")
    assert "swiglu_clamp_value" in msg, (
        f"expected a clamp-rejection error, got: {msg}"
    )


def test_situ_rejects_bad_descriptor():
    """Malformed / invalid descriptors must raise at parse time, not compute.

    A descriptor is a string, so a typo cannot be caught by the type system; the
    parser is the only line of defence between a bad model config and a kernel
    that quietly computes something else.
    """
    from sonicmoe.quack_utils.activation_situ import (
        encode_situ_activation,
        is_situ_activation,
        parse_situ_activation,
        resolve_gate_fn,
        resolve_dgate_fn,
    )
    from sonicmoe.quack_utils.gemm_gated import gate_fn_map
    from sonicmoe.quack_utils.gemm_dgated import dgate_fn_map

    # Non-positive / non-finite betas, at encode time and at parse time.
    for bad in (0.0, -1.0, float("nan"), float("inf"), None):
        with pytest.raises(ValueError):
            encode_situ_activation(beta=bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            encode_situ_activation(beta=BETA, linear_beta=bad)

    for bad_str in (
        "situ_glu:b=0.0:lb=25.0",
        "situ_glu:b=-4.0:lb=25.0",
        "situ_glu:b=4.0:lb=0.0",
        "situ_glu:b=4.0:lb=-25.0",
        "situ_glu:b=4.0:lb=25.0:bogus",
        "situ_glu:beta=4.0",
    ):
        with pytest.raises(ValueError):
            parse_situ_activation(bad_str)

    # A genuinely unsupported activation must not resolve to a SiTU fallback.
    assert not is_situ_activation("situ")
    assert not is_situ_activation("swiglu")
    for resolver, fn_map in ((resolve_gate_fn, gate_fn_map),
                             (resolve_dgate_fn, dgate_fn_map)):
        with pytest.raises(AssertionError):
            resolver("situ", fn_map)
        with pytest.raises(AssertionError):
            resolver("not_an_activation", fn_map)

    # Round trip of the valid forms.
    assert parse_situ_activation(_situ(4.0, 25.0, True)) == (4.0, 25.0, True)
    assert parse_situ_activation(_situ(4.0, None, True)) == (4.0, None, True)
    assert parse_situ_activation(_situ(2.0, 25.0, False)) == (2.0, 25.0, False)
    print("\n  [descriptor rejection] all malformed descriptors raised")
