"""Precision matrix and edge-case coverage for the SiTU-GLU activation.

This file is additive: ``test_situ_activation.py`` proves the element-wise
numerics *at beta=4.0 only*, ``test_situ_gemm.py`` proves the GEMM epilogue on
two shapes, ``test_situ_shapes.py`` sweeps shapes at beta=4.0.  Four gaps are
left, and this file closes them:

1. **beta is never varied numerically.**  Every numeric assertion in the suite
   uses ``BETA = HAND_BETA = 4.0``.  ``beta`` changes the shape of the curve,
   the location of the ``d(gate_act)/dg`` root, and the magnitude of every
   intermediate, so a beta-independent kernel bug is currently invisible.
   Group 1 sweeps beta in {0.5, 1, 2, 4, 8, 25} x linear_beta in {None, 25}
   x {scalar, packed} against the float64 oracle.
2. **Inf has never been fed to the kernel.**  ``test_saturation_no_nan_inf``
   asserts its own inputs are finite before launching.  Group 2 records the
   exact measured behaviour for +-Inf/NaN so it cannot change silently.
3. **``total_M == 0`` and zero-token experts have no regression test.**  Both
   happen routinely in real MoE training (a dropped rank, a dead expert).
   Group 3 covers them in bf16 and FP8, forward and backward.
4. **The output bound is never asserted.**  SiTU's whole FP8 argument is that
   ``|situ| <= beta`` and ``|up| <= linear_beta``, so the epilogue output is
   bounded by construction where SwiGLU's is not.  Group 4 asserts it.

Group 5 pins the layout contract: which non-contiguous inputs work, and which
unsupported ones are rejected cleanly instead of producing garbage.

Every threshold below is annotated with the measurement it came from.  All
measurements were taken on this GPU (SM 10.3) with the same helpers the tests
use; the "measured" number is the worst value observed over the whole sweep and
the limit carries the stated headroom.  Nothing here is a guess.

Cost
----
The element-wise groups (1, 2, 4a) need no GEMM and run in ~15 s.  The GEMM
groups pay one cold compile + autotune per *activation descriptor* (not per
shape), so they all share ``_situ()`` = ``"situ_glu:b=4.0:lb=25.0"``:
~470 s for ``gemm_gated`` and ~300 s for ``gemm_dgated``, once per process.
The ``gemm_dgated`` tests are therefore marked ``@heavy`` (``SITU_SKIP_HEAVY=1``).
The rejection tests in group 5 raise before any compile and cost nothing.
"""
import math

import pytest
import torch

from tests.ops.conftest import (
    cosine_sim,
    requires_blackwell,
    requires_quack,
    rrmse,
)
from tests.ops.situ_reference import (
    deinterleave,
    dsitu_glu_ref_fp64,
    interleave,
    situ_glu_ref_fp64,
)
from tests.ops.test_situ_activation import (
    Err,
    dsitu_glu_kernel,
    situ_glu_kernel,
)
from tests.ops.test_situ_gemm import (
    BETA,
    LINEAR_BETA,
    SEED,
    _fwd_inputs,
    _grouped_gemm_fp64,
    _situ,
)
from tests.ops.test_situ_shapes import _cu_seqlens, heavy

pytestmark = [requires_blackwell]

# ===========================================================================
# Group 1 -- beta sweep at the element-wise level
# ===========================================================================

# 0.5 and 25.0 bracket the useful range: at beta=0.5 the tanh saturates for
# almost every input (so the kernel spends its time in the flat region), at
# beta=25.0 sigmoid(g) for g ~ -100 lands in the f32 subnormal range.  4.0 is
# production.
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0, 25.0]
LBS = [pytest.param(None, id="lbnone"), pytest.param(25.0, id="lb25")]
PATHS = [pytest.param(False, id="scalar"), pytest.param(True, id="packed")]
REGIMES = ["scaled", "fixed", "knee"]

# max |d(gate_act)/dg| over all g, per beta, computed in float64 on a
# 4,000,001-point linspace over +-40*beta.  ``beta=4`` gives 0.9263897687, and
# 25.0 * 0.9263897687 = 23.1597 -- the "|dgate| <= 23.16*|dout|" figure quoted
# for the production configuration.  The general bound is
#     |dgate| <= |dout| * |up_act| * GATE_GRAD_MAX[beta]
# which is what group 4a asserts (|up_act| <= linear_beta when the clamp is on,
# and is just |u| when it is off).
GATE_GRAD_MAX = {
    0.5: 0.5299053304,
    1.0: 0.6035533906,
    2.0: 0.7580756164,
    4.0: 0.9263897687,
    8.0: 1.0339403040,
    25.0: 1.0912010581,
}

N_EW = 8192


def _ew_inputs(regime, beta):
    """Three complementary input regimes for one beta.

    ``scaled``  g ~ N(0, beta): the knee at |g| ~ beta is straddled at every
                beta, so the mix of saturated/linear elements is constant.
    ``fixed``   the *same* data for every beta (g ~ N(0,3), u ~ N(0,30)), so a
                small beta means deep saturation and a large beta means the
                whole batch sits in the near-linear region.  This is the regime
                that would catch a kernel that only works when the input is
                scaled to beta.
    ``knee``    a dense linspace through +-4*beta with dout=1: guarantees
                elements land arbitrarily close to the gate_grad root, which is
                where the relative error of dgate is worst.
    """
    gen = torch.Generator(device="cuda").manual_seed(0xBE7A + len(regime))

    def rn(s):
        return torch.randn(N_EW, generator=gen, device="cuda",
                           dtype=torch.float32) * s

    if regime == "scaled":
        return rn(beta), rn(8.0), rn(1.0)
    if regime == "fixed":
        return rn(3.0), rn(30.0), rn(1.0)
    if regime == "knee":
        g = torch.linspace(-4 * beta, 4 * beta, N_EW, device="cuda",
                           dtype=torch.float32)
        return g, rn(30.0), torch.ones(N_EW, device="cuda", dtype=torch.float32)
    raise AssertionError(regime)


def _ew_oracle(g, u, dout, beta, lb):
    """float64 (out, dgate, dup) via the shared reference.

    Note the interleave/deinterleave: the epilogue contract is
    ``z[:, 0::2] = gate, z[:, 1::2] = up``, and ``situ_glu_ref_fp64`` speaks that
    layout.  Feeding it a chunked ``cat([gate, up])`` produces a stable and
    completely wrong oracle, so the conversion is explicit here.
    """
    z = interleave(g.double().unsqueeze(0), u.double().unsqueeze(0))
    out, _, _ = situ_glu_ref_fp64(z, beta, lb)
    dz, _ = dsitu_glu_ref_fp64(z, dout.double().unsqueeze(0), beta, lb)
    dgate, dup = deinterleave(dz)
    return out[0], dgate[0], dup[0]


# Limits are (max_rel_sig, rrmse, ulp_norm) and are global over all
# (regime, beta, linear_beta, path) combinations.  Measured worst / limit:
#
#   quantity   max_rel_sig            rrmse                  ulp_norm
#   out,bout   4.611e-07 / 4.0e-06    1.027e-07 / 1.0e-06    2.22 / 10.0
#   dgate      1.733e-04 / 1.5e-03    3.841e-07 / 4.0e-06    6.28 / 30.0
#   dup        3.445e-05 / 3.0e-04    1.133e-07 / 1.0e-06    1.79 / 10.0
#
# i.e. ~8x headroom on the relative statistic, ~10x on RRMSE, ~4.5x on ULP.
#
# Why these three and not ``max_rel``: the unfiltered relative error is exactly
# 1.0 at beta=25 for every quantity, because sigmoid(g) for g ~ -100 is ~1e-43,
# the f32 kernel flushes it to a subnormal/zero and the float64 oracle does not.
# That is f32 underflow, not a kernel defect, and no relative statistic can see
# past it -- hence ``max_rel_sig`` (drops elements below 1e-3 of the reference
# scale, which also drops the gate_grad root) plus RRMSE plus a ULP bound.
EW_LIMITS = {
    "out":   (4.0e-6, 1.0e-6, 10.0),
    "bout":  (4.0e-6, 1.0e-6, 10.0),
    "dgate": (1.5e-3, 4.0e-6, 30.0),
    "dup":   (3.0e-4, 1.0e-6, 10.0),
}


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("lb", LBS)
@pytest.mark.parametrize("packed", PATHS)
def test_beta_sweep_vs_fp64(regime, beta, lb, packed):
    """Forward and backward at six betas against the float64 oracle.

    ``bout`` is the third value returned by the backward kernel -- the
    recomputed, unscaled ``gate_act * up_act`` that feeds ``ColVecReduce``.  It
    is checked against the same oracle as the forward output because the
    contract is that the two agree.
    """
    g, u, d = _ew_inputs(regime, beta)
    oref, dgref, duref = _ew_oracle(g, u, d, beta, lb)
    out = situ_glu_kernel(g, u, beta, lb, True, packed)
    dgate, dup, bout = dsitu_glu_kernel(g, u, d, beta, lb, True, packed)

    for qty, act, exp in (("out", out, oref), ("dgate", dgate, dgref),
                          ("dup", dup, duref), ("bout", bout, oref)):
        err = Err(act, exp)
        rel_p, rrmse_p, ulp_p = EW_LIMITS[qty]
        print(f"  {regime} beta={beta} lb={lb} "
              f"{'packed' if packed else 'scalar'} {qty}: "
              f"relsig={err.max_rel_sig:.3e} rrmse={err.rrmse:.3e} "
              f"ulp={err.ulp_norm:.2f} abs={err.max_abs:.3e} "
              f"maxrel={err.max_rel:.3e} nexcl={err.n_excluded}")
        # Finite inputs must never produce NaN/Inf: measured n_bad = 0 for all
        # 72 combinations.
        assert err.n_bad == 0, f"{qty}: {err.n_bad} non-finite outputs"
        assert err.max_rel_sig < rel_p, (
            f"{qty} max_rel_sig {err.max_rel_sig:.3e} >= {rel_p:.1e} "
            f"(regime={regime} beta={beta} lb={lb} packed={packed})")
        assert err.rrmse < rrmse_p, (
            f"{qty} RRMSE {err.rrmse:.3e} >= {rrmse_p:.1e}")
        assert err.ulp_norm < ulp_p, (
            f"{qty} ulp_norm {err.ulp_norm:.2f} >= {ulp_p}")


# Measured scalar-vs-packed disagreement over the same 36 (regime,beta,lb)
# points, worst over all: ulp_norm 0.734 (out/bout), 1.695 (dgate), 0.786 (dup);
# RRMSE <= 7.751e-08 everywhere.  ``out`` and ``bout`` were bit-identical at
# every beta <= 8.  Limits below carry ~2.4x (ULP) / ~6x (RRMSE) headroom.
#
# The two paths are not required to be bit-identical -- ``mul_packed_f32x2`` and
# friends are a different instruction sequence -- but they must not disagree by
# more than the last couple of bits, which is what this pins.
SP_ULP_MAX = 4.0
SP_RRMSE_MAX = 5.0e-7


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("lb", LBS)
def test_beta_sweep_scalar_vs_packed(regime, beta, lb):
    """The packed f32x2 path must track the scalar path at every beta."""
    g, u, d = _ew_inputs(regime, beta)
    s = (situ_glu_kernel(g, u, beta, lb, True, False),
         *dsitu_glu_kernel(g, u, d, beta, lb, True, False))
    p = (situ_glu_kernel(g, u, beta, lb, True, True),
         *dsitu_glu_kernel(g, u, d, beta, lb, True, True))
    for name, a, b in zip(("out", "dgate", "dup", "bout"), p, s):
        err = Err(a, b.double())
        print(f"  SP {regime} beta={beta} lb={lb} {name}: "
              f"ulp={err.ulp_norm:.3f} rrmse={err.rrmse:.3e} "
              f"nbitdiff={int((a != b).sum())}")
        assert err.ulp_norm < SP_ULP_MAX, (
            f"{name} scalar-vs-packed ulp {err.ulp_norm:.3f} >= {SP_ULP_MAX}")
        assert err.rrmse < SP_RRMSE_MAX, (
            f"{name} scalar-vs-packed RRMSE {err.rrmse:.3e} >= {SP_RRMSE_MAX:.1e}")


# ===========================================================================
# Group 4a -- the output bound, element-wise
# ===========================================================================

# Slack on the bound assertions.  The bounds are exact in exact arithmetic, so
# the only allowance needed is f32 rounding of the kernel's own product plus the
# 10-decimal truncation of GATE_GRAD_MAX.  Measured worst |value| / |bound| over
# the full 6-beta x 2-lb x 2-path x 3-regime sweep:
#     out   0.999983569069    bout  0.999983569069
#     dup   0.999983559482    dgate 1.000000113768   <- attained, +1.1e-07 over
# so 1e-6 of relative slack is ~9x the observed overshoot.  A real violation
# (a missing clamp, a wrong beta reaching the epilogue) is orders of magnitude
# larger than this, not a few ULP.
BOUND_SLACK = 1.0 + 1.0e-6


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("lb", LBS)
@pytest.mark.parametrize("packed", PATHS)
def test_output_bounds_elementwise(regime, beta, lb, packed):
    """``|out| <= beta*|up_act|``, ``|dgate| <= |dout|*|up_act|*max|ga'|``,
    ``|dup| <= beta*|dout|``.

    This is the property that makes SiTU-GLU FP8-safe and SwiGLU not: the gate
    branch is bounded by ``beta`` for every input, so the epilogue output cannot
    grow without limit and the e4m3 range is never the binding constraint.  With
    the production clamp ``linear_beta=25`` the concrete bounds are
    ``|out| <= 100``, ``|dgate| <= 23.16*|dout|``, ``|dup| <= 4*|dout|``.

    Asserted per element (not just on the max) so a single escaping element is
    caught, and derived from the float64 ``up_act`` so the bound itself is not
    limited by f32.
    """
    g, u, d = _ew_inputs(regime, beta)
    out = situ_glu_kernel(g, u, beta, lb, True, packed)
    dgate, dup, bout = dsitu_glu_kernel(g, u, d, beta, lb, True, packed)

    ud = u.double()
    up_act = (lb * torch.tanh(ud / lb)) if lb is not None else ud
    up_abs = up_act.abs()
    dabs = d.double().abs()
    bounds = {
        "out": beta * up_abs,
        "bout": beta * up_abs,
        "dgate": dabs * up_abs * GATE_GRAD_MAX[beta],
        "dup": dabs * beta,
    }
    for name, t in (("out", out), ("bout", bout), ("dgate", dgate), ("dup", dup)):
        b = bounds[name]
        ratio = t.double().abs() / b.clamp(min=1e-300)
        worst = ratio.max().item()
        print(f"  BOUND {regime} beta={beta} lb={lb} "
              f"{'packed' if packed else 'scalar'} {name}: "
              f"max|.|={t.abs().max().item():.6e} worst_ratio={worst:.9f}")
        assert worst <= BOUND_SLACK, (
            f"{name} exceeds its analytic bound by {worst - 1.0:.3e} "
            f"(regime={regime} beta={beta} lb={lb} packed={packed})")
    if lb is not None:
        # The global consequence, spelled out: nothing can leave the box.
        assert out.abs().max().item() <= beta * lb, "out escaped beta*linear_beta"
        assert dup.abs().max().item() <= beta * d.abs().max().item() * BOUND_SLACK


# ===========================================================================
# Group 2 -- +-Inf and NaN, as a recording test
# ===========================================================================

_INF = float("inf")
_NAN = float("nan")

# Measured on SM 10.3 at beta=4.0, dout=1.0, precise=True.  The scalar and
# packed paths were verified bit-identical for every one of these 24 rows, so
# only one table is needed.  Key: (linear_beta, gate, up) -> (out, dgate, dup,
# bout).  ``bout`` (the backward kernel's recomputed forward) equals ``out`` in
# every row.
#
# This is a *recording* test, not a specification: it exists so that a change in
# saturation handling shows up as a diff instead of as a silent numerical change
# in training.  Comparison is bit-exact (signed zeros included), except that a
# NaN expectation only requires a NaN, since NaN payloads are not contractual.
INF_TABLE = {
    (25.0, _INF, 1.0): (3.9978678226470947, 0.0, 3.9936068058013916, 3.9978678226470947),
    (25.0, _INF, _INF): (100.0, 0.0, 0.0, 100.0),
    (25.0, -_INF, _INF): (-0.0, 0.0, -0.0, -0.0),
    (25.0, 0.0, _INF): (0.0, 12.5, 0.0, 0.0),
    (25.0, _INF, -_INF): (-100.0, -0.0, 0.0, -100.0),
    (25.0, -_INF, -_INF): (0.0, -0.0, -0.0, 0.0),
    (25.0, 1.0, _INF): (17.904987335205078, 21.995540618896484, 0.0, 17.904987335205078),
    (25.0, _INF, 0.0): (0.0, 0.0, 4.0, 0.0),
    (25.0, _NAN, _INF): (_NAN, _NAN, _NAN, _NAN),
    (25.0, _INF, _NAN): (_NAN, _NAN, _NAN, _NAN),
    (25.0, -_INF, 1.0): (-0.0, 0.0, -0.0, -0.0),
    (25.0, 0.0, -_INF): (-0.0, -12.5, 0.0, -0.0),
    (None, _INF, 1.0): (4.0, 0.0, 4.0, 4.0),
    (None, _INF, _INF): (_INF, _NAN, 4.0, _INF),
    (None, -_INF, _INF): (_NAN, _NAN, -0.0, _NAN),
    (None, 0.0, _INF): (_NAN, _INF, 0.0, _NAN),
    (None, _INF, -_INF): (-_INF, _NAN, 4.0, -_INF),
    (None, -_INF, -_INF): (_NAN, _NAN, -0.0, _NAN),
    (None, 1.0, _INF): (_INF, _INF, 0.716199517250061, _INF),
    (None, _INF, 0.0): (0.0, 0.0, 4.0, 0.0),
    (None, _NAN, _INF): (_NAN, _NAN, _NAN, _NAN),
    (None, _INF, _NAN): (_NAN, _NAN, 4.0, _NAN),
    (None, -_INF, 1.0): (-0.0, 0.0, -0.0, -0.0),
    (None, 0.0, -_INF): (_NAN, -_INF, 0.0, _NAN),
}

_INF_CASES = [
    (_INF, 1.0), (_INF, _INF), (-_INF, _INF), (0.0, _INF), (_INF, -_INF),
    (-_INF, -_INF), (1.0, _INF), (_INF, 0.0), (_NAN, _INF), (_INF, _NAN),
    (-_INF, 1.0), (0.0, -_INF),
]


def _bits(t):
    return t.view(torch.int32).item()


def _assert_recorded(label, actual, expected):
    if isinstance(expected, float) and math.isnan(expected):
        assert bool(actual.isnan()), f"{label}: expected NaN, got {actual.item()!r}"
        return
    ref = torch.tensor([expected], dtype=torch.float32, device=actual.device)
    assert _bits(actual) == _bits(ref[0]), (
        f"{label}: recorded {expected!r} (bits {_bits(ref[0]):#010x}) but got "
        f"{actual.item()!r} (bits {_bits(actual):#010x})")


@pytest.mark.parametrize("lb", LBS)
@pytest.mark.parametrize("packed", PATHS)
def test_inf_nan_recorded(lb, packed):
    """Pin the exact measured behaviour for +-Inf / NaN inputs at beta=4.

    The finding worth stating: with the *production* ``linear_beta=25`` every
    single Inf row above produces a finite, plausible-looking output and no NaN
    at all, and ``dgate`` is exactly 0 in most of them.  Mathematically that is
    the correct limit (``situ`` saturates, so its derivative vanishes), but
    operationally it means an upstream Inf is laundered into a normal number and
    the gradient quietly becomes zero -- the opposite of the diagnosability that
    ``test_nan_input_propagates`` guarantees for NaN.  With ``linear_beta=None``
    the Inf does surface as inf/nan.  If someone ever "fixes" the clamp to
    propagate Inf, this test is where it shows up.
    """
    # Repeated so the packed path (which processes pairs and pads to
    # 2*NUM_THREADS) sees the same values at the same lane parity as the scalar
    # path; 12 cases x 32 = 384 elements.
    rep = 32
    g = torch.tensor([c[0] for c in _INF_CASES] * rep, device="cuda",
                     dtype=torch.float32)
    u = torch.tensor([c[1] for c in _INF_CASES] * rep, device="cuda",
                     dtype=torch.float32)
    d = torch.ones_like(g)

    out = situ_glu_kernel(g, u, BETA, lb, True, packed)
    dgate, dup, bout = dsitu_glu_kernel(g, u, d, BETA, lb, True, packed)

    n_nonfinite = 0
    n_inf_rows = 0
    for i, (gv, uv) in enumerate(_INF_CASES):
        exp = INF_TABLE[(lb, gv, uv)]
        for j, (name, t) in enumerate((("out", out), ("dgate", dgate),
                                       ("dup", dup), ("bout", bout))):
            _assert_recorded(
                f"lb={lb} {'packed' if packed else 'scalar'} "
                f"g={gv!r} u={uv!r} {name}", t[i], exp[j])
        # Count only the pure-Inf rows: a NaN input is *supposed* to give NaN
        # out (``test_nan_input_propagates`` owns that contract), so those rows
        # say nothing about Inf laundering.
        if not (math.isnan(gv) or math.isnan(uv)):
            n_inf_rows += 1
            if not math.isfinite(exp[0]):
                n_nonfinite += 1
        # The repeated copies must all agree with the first one.
        for name, t in (("out", out), ("dgate", dgate), ("dup", dup)):
            col = t[i::len(_INF_CASES)]
            same = (col == col[0]) | (col.isnan() & col[0].isnan())
            assert bool(same.all()), f"{name} not uniform across repeats at case {i}"

    # The documented asymmetry, asserted rather than only described.
    assert n_inf_rows == 10
    if lb is None:
        # Measured: 6 of the 10 pure-Inf rows produce inf or nan in ``out``.
        assert n_nonfinite >= 5, "lb=None used to surface Inf; it no longer does"
    else:
        assert n_nonfinite == 0, (
            "linear_beta=25 used to launder every Inf into a finite value; "
            "some row now produces inf/nan -- update the table and the note")
        # No +-Inf anywhere in the output for lb=25 (NaN only from NaN input).
        assert bool((torch.isfinite(out) | out.isnan()).all())
        assert bool((torch.isfinite(dgate) | dgate.isnan()).all())
        assert bool((torch.isfinite(dup) | dup.isnan()).all())


# ===========================================================================
# Group 3 -- real-GEMM degenerate routings
# ===========================================================================
#
# All GEMM tests below share the single activation descriptor ``_situ()`` =
# "situ_glu:b=4.0:lb=25.0".  The JIT cache key has no m/n/k in it and
# ``@autotune`` is keyed on ("activation", ...) only, so shapes are free but a
# second descriptor is a second cold compile (~470 s forward, ~300 s backward).
#
# Shape: H=512, I=128, E=4, 128 rows/expert.  Small on purpose -- the point is
# the routing structure, not the arithmetic, and the arithmetic is already
# covered at the production geometry by ``test_situ_shapes.py``.
# NB: intermediate sizes of 1 or 129 must never be built here; they fault the
# device (CUDA_ERROR_ILLEGAL_ADDRESS -> Xid 13 -> Xid 154, GPU reset required).
G_H, G_I, G_E, G_TK = 512, 128, 4, 128

# bf16 grouped-GEMM accuracy at this shape, measured against the float64 oracle:
# forward postact RRMSE 1.676e-03 / cosine 0.99999857; backward dx 1.661e-03,
# postact 1.659e-03, colvec 5.222e-07.  These match FWD_LIMITS["bf16"] =
# (5.0e-3, 0.99998), so the shared limits are reused rather than re-derived.
G_BF16_RRMSE_MAX = 5.0e-3
G_BF16_COS_MIN = 0.99998
# FP8 at this shape: forward RRMSE 5.532e-02 / cosine 0.99847305, backward dx
# RRMSE 3.752e-02, colvec 3.833e-02.  FWD_LIMITS["fp8"] = (1.0e-1, 0.99).
G_FP8_RRMSE_MAX = 1.0e-1
G_FP8_COS_MIN = 0.99

# Bit-exactness of a later expert's rows when an earlier group is empty:
# measured exactly 0 differing elements for the forward postact, backward dx and
# backward postact in every layout tried.  ``colvec`` is the exception -- see
# COLVEC_SHIFT_RRMSE_MAX.
COLVEC_SHIFT_RRMSE_MAX = 1.0e-6


def _mk_fwd(total_M, E, seed=SEED):
    return _fwd_inputs(total_M, G_H, G_I, E, seed=seed)


def _mk_bwd(total_M, E, seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dout = torch.randn(total_M, G_H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(G_H, G_I, E, dtype=torch.bfloat16, device="cuda") * (G_H ** -0.5)
    z = torch.randn(total_M, 2 * G_I, dtype=torch.bfloat16, device="cuda")
    return dout, w2, z


def _insert_dead(w_ref, empty_at, E, junk_shape, seed):
    """Widen a per-expert weight stack by inserting junk experts.

    ``w_ref`` is ``(..., E_real)``; the returned stack has ``E`` slices, with
    freshly random ones at the indices in ``empty_at`` and the original slices,
    in order, everywhere else.  Because the *real* experts keep both their
    weights and their row ranges, the kernel's output for those rows must be
    identical to the dense run -- which is exactly the "an empty group must not
    shift later groups" property.  The junk weights are random rather than zero
    so that a kernel which accidentally applied them would be obvious.
    """
    torch.manual_seed(seed)
    cols, real = [], 0
    for e in range(E):
        if e in empty_at:
            cols.append(torch.randn(*junk_shape, dtype=torch.bfloat16, device="cuda"))
        else:
            cols.append(w_ref[..., real:real + 1])
            real += 1
    assert real == w_ref.shape[-1]
    return torch.cat(cols, dim=-1)


# (id, empty_at, counts).  Each layout has three live experts of G_TK rows, so
# every one is compared against the same dense E=3 reference.
DEAD_LAYOUTS = [
    pytest.param({0}, [0, G_TK, G_TK, G_TK], id="lead-empty"),
    pytest.param({1}, [G_TK, 0, G_TK, G_TK], id="mid-empty"),
    pytest.param({3}, [G_TK, G_TK, G_TK, 0], id="tail-empty"),
    pytest.param({0, 2}, [0, G_TK, 0, G_TK, G_TK], id="two-empty"),
]


@requires_quack
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_gemm_total_m_zero_fwd(fp8):
    """``gemm_gated`` with every expert empty must return empty tensors, quietly.

    A rank whose dispatcher produced no tokens at all is a real occurrence (all
    tokens routed elsewhere, or a partially-filled last global batch).  Measured:
    both paths return ``preact (0, 2I)`` and ``postact (0, I)`` with no launch
    and no error.  Note the postact width is ``I``, not ``2I``: the epilogue has
    already consumed the interleaved gate/up pair.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    cu = _cu_seqlens([0] * G_E)
    x, w1 = _mk_fwd(0, G_E)
    if fp8:
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            precompute_weight_fp8_for_fused_gated,
            quantize_and_pack_activation,
        )

        x_q, a_scales = quantize_and_pack_activation(x)
        w_q, b_scales = precompute_weight_fp8_for_fused_gated(w1)
        preact, postact = gemm_gated(
            x_q, w_q, activation=_situ(), out_dtype=torch.bfloat16,
            postact_dtype=torch.bfloat16, cu_seqlens_m=cu,
            a_scales=a_scales, b_scales=b_scales,
        )
    else:
        preact, postact = gemm_gated(x, w1.permute(2, 1, 0).contiguous(),
                                     activation=_situ(), cu_seqlens_m=cu)
    assert tuple(preact.shape) == (0, 2 * G_I), preact.shape
    assert tuple(postact.shape) == (0, G_I), postact.shape


@heavy
@requires_quack
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_gemm_total_m_zero_bwd(fp8):
    """``gemm_dgated`` with ``colvec_reduce=True`` on an empty problem.

    ``colvec_reduce`` is the one output whose shape is ``(total_M,)`` rather than
    a matrix, so an empty problem is where a reduction over zero tiles would
    surface.  Measured: ``dx (0, 2I)``, ``postact (0, I)``, ``colvec (0,)``.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    cu = _cu_seqlens([0] * G_E)
    dout, w2, z = _mk_bwd(0, G_E)
    if fp8:
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            precompute_weight_fp8_for_fused_dgated,
            quantize_and_pack_activation,
        )

        d_q, a_scales = quantize_and_pack_activation(dout)
        w_q, b_scales = precompute_weight_fp8_for_fused_dgated(w2)
        dx, postact, colvec = gemm_dgated(
            d_q, w_q, z, activation=_situ(), colvec_reduce=True,
            cu_seqlens_m=cu, out_dtype=torch.bfloat16,
            a_scales=a_scales, b_scales=b_scales,
        )
    else:
        dx, postact, colvec = gemm_dgated(
            dout, w2.permute(2, 0, 1).contiguous(), z, activation=_situ(),
            colvec_reduce=True, cu_seqlens_m=cu,
        )
    assert tuple(dx.shape) == (0, 2 * G_I), dx.shape
    assert tuple(postact.shape) == (0, G_I), postact.shape
    assert tuple(colvec.shape) == (0,), colvec.shape


@requires_quack
@pytest.mark.parametrize("empty_at,counts", DEAD_LAYOUTS)
def test_gemm_dead_experts_fwd_bf16(empty_at, counts):
    """A zero-token expert must not shift the rows of the experts after it.

    Three live experts of 128 rows each are laid out with dead experts inserted
    at ``empty_at``; the live rows must come out **bit-for-bit** identical to a
    dense E=3 run.  Measured: 0 of 49152 elements differ, in all four layouts.
    Bit-exactness (rather than a tolerance) is the right standard here because
    the arithmetic is unchanged -- only the tile-to-expert mapping moves -- so
    any difference at all means the mapping leaked.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = 3 * G_TK
    x, w1_ref = _mk_fwd(total_M, 3, seed=7)
    _, ref = gemm_gated(x, w1_ref.permute(2, 1, 0).contiguous(),
                        activation=_situ(), cu_seqlens_m=_cu_seqlens([G_TK] * 3))

    # Sanity: the dense reference is itself correct, else "bit-exact vs the
    # reference" would be a vacuous statement about two identically wrong runs.
    z_gold = _grouped_gemm_fp64(x, w1_ref, _cu_seqlens([G_TK] * 3), 3, True)
    gold, _, _ = situ_glu_ref_fp64(z_gold, BETA, LINEAR_BETA)
    r_ref, c_ref = rrmse(ref, gold), cosine_sim(ref, gold)
    print(f"\n  dense E=3 reference: RRMSE={r_ref:.3e} cos={c_ref:.8f}")
    assert r_ref < G_BF16_RRMSE_MAX and c_ref > G_BF16_COS_MIN

    E = len(counts)
    w1 = _insert_dead(w1_ref, empty_at, E, (2 * G_I, G_H, 1), seed=99)
    _, got = gemm_gated(x, w1.permute(2, 1, 0).contiguous(),
                        activation=_situ(), cu_seqlens_m=_cu_seqlens(counts))
    ndiff = int((got != ref).sum())
    print(f"  counts={counts}: ndiff={ndiff}/{ref.numel()} "
          f"rrmse_vs_dense={rrmse(got, ref):.3e}")
    assert ndiff == 0, (
        f"{ndiff} elements moved when experts {sorted(empty_at)} were emptied; "
        "the dead group is shifting the rows of the live ones")


@requires_quack
@pytest.mark.parametrize("empty_at,counts", DEAD_LAYOUTS)
def test_gemm_dead_experts_fwd_fp8(empty_at, counts):
    """Same, through the blockscaled FP8 path.

    FP8 has an extra way to get this wrong: the 1x32 e8m0 scale factors are
    themselves laid out per expert, so a dead expert can desynchronise the scale
    stream from the data stream even if the data offsets are right.  Measured
    bit-exact for lead-empty and mid-empty; the activation quantisation is shared
    between the two runs so the only moving part is the weight/scale indexing.
    """
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_fused_gated,
        quantize_and_pack_activation,
    )
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = 3 * G_TK
    x, w1_ref = _mk_fwd(total_M, 3, seed=7)
    x_q, a_scales = quantize_and_pack_activation(x)

    def _run(w, cu):
        w_q, b_scales = precompute_weight_fp8_for_fused_gated(w)
        _, post = gemm_gated(
            x_q, w_q, activation=_situ(), out_dtype=torch.bfloat16,
            postact_dtype=torch.bfloat16, cu_seqlens_m=cu,
            a_scales=a_scales, b_scales=b_scales,
        )
        return post

    ref = _run(w1_ref, _cu_seqlens([G_TK] * 3))
    z_gold = _grouped_gemm_fp64(x, w1_ref, _cu_seqlens([G_TK] * 3), 3, True)
    gold, _, _ = situ_glu_ref_fp64(z_gold, BETA, LINEAR_BETA)
    r_ref, c_ref = rrmse(ref, gold), cosine_sim(ref, gold)
    print(f"\n  dense E=3 fp8 reference: RRMSE={r_ref:.3e} cos={c_ref:.8f}")
    assert r_ref < G_FP8_RRMSE_MAX and c_ref > G_FP8_COS_MIN

    E = len(counts)
    w1 = _insert_dead(w1_ref, empty_at, E, (2 * G_I, G_H, 1), seed=99)
    got = _run(w1, _cu_seqlens(counts))
    ndiff = int((got != ref).sum())
    print(f"  counts={counts}: ndiff={ndiff}/{ref.numel()}")
    assert ndiff == 0, f"fp8: {ndiff} elements moved with dead experts {sorted(empty_at)}"


@heavy
@requires_quack
@pytest.mark.parametrize("empty_at,counts", DEAD_LAYOUTS)
def test_gemm_dead_experts_bwd_bf16(empty_at, counts):
    """Dead experts in ``gemm_dgated``, including the ``colvec_reduce`` output.

    ``dx`` and the recomputed ``postact`` are required to be bit-exact (measured:
    0 differences).  ``colvec`` is *not*: it is a reduction along N, and changing
    the tile-to-expert mapping changes the order the partial sums are combined
    in.  Measured RRMSE 9.954e-08 for lead-empty and mid-empty, exactly 0 for
    two-empty -- so the standard for colvec is 1e-6 RRMSE (~10x the measurement),
    which is still three orders of magnitude tighter than
    BWD_LIMITS["colvec"] = 1e-5 against the oracle.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    total_M = 3 * G_TK
    dout, w2_ref, z = _mk_bwd(total_M, 3, seed=7)
    cu3 = _cu_seqlens([G_TK] * 3)
    dx_r, post_r, cv_r = gemm_dgated(
        dout, w2_ref.permute(2, 0, 1).contiguous(), z, activation=_situ(),
        colvec_reduce=True, cu_seqlens_m=cu3)

    dy1 = _grouped_gemm_fp64(dout, w2_ref, cu3, 3, False)
    dz_gold, post_gold = dsitu_glu_ref_fp64(z.double(), dy1, BETA, LINEAR_BETA)
    cv_gold = (post_gold * dy1).sum(dim=-1)
    print(f"\n  dense E=3 bwd reference: dx={rrmse(dx_r, dz_gold):.3e} "
          f"post={rrmse(post_r, post_gold):.3e} cv={rrmse(cv_r, cv_gold):.3e}")
    assert rrmse(dx_r, dz_gold) < G_BF16_RRMSE_MAX
    assert rrmse(cv_r, cv_gold) < 1.0e-5      # BWD_LIMITS["colvec"]

    E = len(counts)
    w2 = _insert_dead(w2_ref, empty_at, E, (G_H, G_I, 1), seed=98)
    dx, post, cv = gemm_dgated(
        dout, w2.permute(2, 0, 1).contiguous(), z, activation=_situ(),
        colvec_reduce=True, cu_seqlens_m=_cu_seqlens(counts))
    n_dx, n_po = int((dx != dx_r).sum()), int((post != post_r).sum())
    r_cv = rrmse(cv, cv_r)
    print(f"  counts={counts}: ndiff_dx={n_dx} ndiff_post={n_po} "
          f"cv_rrmse={r_cv:.3e} cv_bitexact={torch.equal(cv, cv_r)}")
    assert n_dx == 0, f"{n_dx} dx elements moved with dead experts {sorted(empty_at)}"
    assert n_po == 0, f"{n_po} postact elements moved"
    assert r_cv < COLVEC_SHIFT_RRMSE_MAX, (
        f"colvec RRMSE {r_cv:.3e} >= {COLVEC_SHIFT_RRMSE_MAX:.1e}; a dead expert "
        "is doing more than reordering the reduction")


@heavy
@requires_quack
@pytest.mark.parametrize("empty_at,counts", DEAD_LAYOUTS)
def test_gemm_dead_experts_bwd_fp8(empty_at, counts):
    """Dead experts in the FP8 backward, ``dx`` and ``colvec``.

    Measured bit-exact for both outputs in the lead-empty and mid-empty layouts,
    so the standard is bit-exactness for ``dx`` and the same 1e-6 RRMSE band for
    ``colvec`` as the bf16 case.
    """
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        precompute_weight_fp8_for_fused_dgated,
        quantize_and_pack_activation,
    )
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    total_M = 3 * G_TK
    dout, w2_ref, z = _mk_bwd(total_M, 3, seed=7)
    d_q, a_scales = quantize_and_pack_activation(dout)

    def _run(w, cu):
        w_q, b_scales = precompute_weight_fp8_for_fused_dgated(w)
        return gemm_dgated(d_q, w_q, z, activation=_situ(), colvec_reduce=True,
                           cu_seqlens_m=cu, out_dtype=torch.bfloat16,
                           a_scales=a_scales, b_scales=b_scales)

    cu3 = _cu_seqlens([G_TK] * 3)
    dx_r, post_r, cv_r = _run(w2_ref, cu3)
    dy1 = _grouped_gemm_fp64(dout, w2_ref, cu3, 3, False)
    dz_gold, _ = dsitu_glu_ref_fp64(z.double(), dy1, BETA, LINEAR_BETA)
    print(f"\n  dense E=3 fp8 bwd reference: dx={rrmse(dx_r, dz_gold):.3e}")
    assert rrmse(dx_r, dz_gold) < G_FP8_RRMSE_MAX

    E = len(counts)
    w2 = _insert_dead(w2_ref, empty_at, E, (G_H, G_I, 1), seed=98)
    dx, post, cv = _run(w2, _cu_seqlens(counts))
    n_dx = int((dx != dx_r).sum())
    r_cv = rrmse(cv, cv_r)
    print(f"  counts={counts}: ndiff_dx={n_dx} cv_rrmse={r_cv:.3e}")
    assert n_dx == 0, f"fp8 bwd: {n_dx} dx elements moved"
    assert r_cv < COLVEC_SHIFT_RRMSE_MAX, f"fp8 bwd colvec RRMSE {r_cv:.3e}"


# ===========================================================================
# Group 4b -- the output bound through the real epilogue
# ===========================================================================

# The forward bound is asserted globally, not per element: the kernel's ``z`` is
# an f32 accumulation and the oracle's is float64, so a per-element bound derived
# from the oracle would be limited by the accumulation difference, not by the
# activation.  ``beta*linear_beta = 100.0`` is exactly representable in bf16
# (1.5625 * 2^6) and the next bf16 above it is 100.5, so a true value <= 100
# cannot round above 100: the assertion needs no slack at all.
#
# Measured max|postact| at this descriptor:
#   H=512  I=128 E=4 TK=128 x_scale=1    9.062500e+00
#   H=512  I=128 E=4 TK=128 x_scale=30   1.000000e+02   <- bound attained exactly
#   H=1024 I=256 E=2 TK=256 x_scale=1    7.875000e+00
#   H=512  I=256 E=8 TK=128 x_scale=10   9.350000e+01
# The x_scale=30 row is the one that matters: the bound is reached and not
# crossed, so this is a tight test rather than a vacuous one.
BOUND_SHAPES = [
    pytest.param(512, 128, 4, 128, 1.0, id="base"),
    pytest.param(512, 128, 4, 128, 30.0, id="saturated"),
    pytest.param(1024, 256, 2, 256, 1.0, id="wide-K"),
    pytest.param(512, 256, 8, 128, 10.0, id="wide-N"),
]


@requires_quack
@pytest.mark.parametrize("H,I,E,TK,x_scale", BOUND_SHAPES)
def test_gemm_postact_bound(H, I, E, TK, x_scale):
    """``|postact| <= beta*linear_beta`` out of the forward epilogue, at 4 shapes.

    This is the FP8-safety property: whatever the GEMM accumulator produces, the
    activation cannot emit a value outside ``+-100``, so the e4m3 dynamic range of
    the *next* layer's input is never the binding constraint.  SwiGLU has no such
    bound, which is why it needs ``swiglu_clamp_value``.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = TK * E
    x, w1 = _fwd_inputs(total_M, H, I, E, seed=7, x_scale=x_scale)
    preact, postact = gemm_gated(x, w1.permute(2, 1, 0).contiguous(),
                                 activation=_situ(),
                                 cu_seqlens_m=_cu_seqlens([TK] * E))
    limit = BETA * LINEAR_BETA
    peak = postact.abs().max().item()
    print(f"\n  H={H} I={I} E={E} TK={TK} x_scale={x_scale}: "
          f"max|postact|={peak:.6e} limit={limit} "
          f"max|preact|={preact.abs().max().item():.6e}")
    assert torch.isfinite(postact.float()).all(), "postact has NaN/Inf"
    assert peak <= limit, f"postact peak {peak:.6e} > beta*linear_beta {limit}"


# Backward bounds.  ``postact`` here is recomputed from ``z_preact``, which is a
# kernel *input* and therefore exact, so this one can be asserted per element
# against the float64 ``up_act``.  Slack is 2^-7: ``postact`` is bf16, and the
# measured worst per-element ratio is 1.003513514 (~2^-8.15), so 2^-7 =
# 1.0078125 is ~2.2x the observed overshoot and still far below anything a real
# bound violation would produce.
BWD_POSTACT_SLACK = 1.0 + 2.0 ** -7
# ``dgate``/``dup`` are bounded by the (unavailable-in-f32) ``dy1``, so they are
# checked globally against max|dy1| from the oracle, with 1% for the f32-vs-f64
# accumulation difference.  Measured ratios: dgate 0.072 / 0.462 / 0.680 and dup
# 0.400 / 0.818 / 0.732 at z_scale 1 / 10 / 60 -- i.e. the bound is approached
# but not attained, so this assertion is a blow-up detector rather than a tight
# characterisation.
BWD_DX_SLACK = 1.01


@heavy
@requires_quack
@pytest.mark.parametrize("z_scale", [1.0, 10.0, 60.0], ids=["z1", "z10", "z60"])
def test_gemm_bwd_bounds(z_scale):
    """Bounds on all three ``gemm_dgated`` outputs.

    ``z_scale`` drives ``z_preact`` from the near-linear region into deep
    saturation (``|g| >> 10*beta``), which is where a missing clamp would let the
    gate branch escape.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    total_M = G_TK * G_E
    dout, w2, _ = _mk_bwd(total_M, G_E)
    z = (torch.randn(total_M, 2 * G_I, dtype=torch.bfloat16,
                     device="cuda") * z_scale).bfloat16()
    cu = _cu_seqlens([G_TK] * G_E)
    dx, postact, colvec = gemm_dgated(
        dout, w2.permute(2, 0, 1).contiguous(), z, activation=_situ(),
        colvec_reduce=True, cu_seqlens_m=cu)

    # Per-element: |ga*up| <= beta*|up_act(u)|, with up read from the odd
    # (interleaved) slots of the exact bf16 input.
    up_act = LINEAR_BETA * torch.tanh(z.double()[:, 1::2] / LINEAR_BETA)
    ratio = (postact.double().abs() / (BETA * up_act.abs()).clamp(min=1e-300))
    worst = ratio.max().item()

    dy1 = _grouped_gemm_fp64(dout, w2, cu, G_E, False)
    dy1_max = dy1.abs().max().item()
    dgate, dup = deinterleave(dx.double())
    r_dg = dgate.abs().max().item() / (dy1_max * LINEAR_BETA * GATE_GRAD_MAX[BETA])
    r_du = dup.abs().max().item() / (dy1_max * BETA)
    print(f"\n  z_scale={z_scale}: postact worst_ratio={worst:.9f} "
          f"max|postact|={postact.abs().max().item():.6e} "
          f"dgate_ratio={r_dg:.6f} dup_ratio={r_du:.6f}")

    assert torch.isfinite(dx.float()).all(), "dx has NaN/Inf"
    assert torch.isfinite(colvec.float()).all(), "colvec has NaN/Inf"
    assert postact.abs().max().item() <= BETA * LINEAR_BETA
    assert worst <= BWD_POSTACT_SLACK, (
        f"recomputed postact exceeds beta*|up_act| by {worst - 1.0:.3e}")
    assert r_dg <= BWD_DX_SLACK, (
        f"|dgate| exceeds max|dy1|*linear_beta*max|ga'| by {r_dg - 1.0:.3e}")
    assert r_du <= BWD_DX_SLACK, (
        f"|dup| exceeds max|dy1|*beta by {r_du - 1.0:.3e}")


# ===========================================================================
# Group 5 -- layout contract
# ===========================================================================


@requires_quack
def test_gemm_row_strided_a():
    """A row-strided (non-contiguous) A is supported and must be bit-exact.

    ``A`` only has to be k-major (``A.stride(-1) == 1``); the row stride is free.
    That matters because a dispatcher can hand the GEMM a view into a larger
    buffer rather than a compacted copy.  The interleaved decoy rows carry values
    100x larger than the real ones, so a kernel that read the wrong rows would
    not merely be inaccurate, it would be off by two orders of magnitude.
    Measured: bit-exact against the contiguous run (0 differing elements), RRMSE
    1.663e-03 / cosine 0.99999857 against the float64 oracle.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = G_TK * G_E
    cu = _cu_seqlens([G_TK] * G_E)
    x, w1 = _mk_fwd(total_M, G_E)
    w1_3d = w1.permute(2, 1, 0).contiguous()
    _, ref = gemm_gated(x, w1_3d, activation=_situ(), cu_seqlens_m=cu)

    big = torch.empty(2 * total_M, G_H, dtype=torch.bfloat16, device="cuda")
    big[::2] = x
    big[1::2] = torch.randn(total_M, G_H, dtype=torch.bfloat16,
                            device="cuda") * 100.0
    view = big[::2]
    assert not view.is_contiguous() and view.stride(-1) == 1
    _, got = gemm_gated(view, w1_3d, activation=_situ(), cu_seqlens_m=cu)

    z_gold = _grouped_gemm_fp64(x, w1, cu, G_E, True)
    gold, _, _ = situ_glu_ref_fp64(z_gold, BETA, LINEAR_BETA)
    r, c = rrmse(got, gold), cosine_sim(got, gold)
    print(f"\n  strided A: strides={view.stride()} ndiff={int((got != ref).sum())} "
          f"RRMSE={r:.4e} cos={c:.8f}")
    assert int((got != ref).sum()) == 0, "row-strided A changed the result"
    assert r < G_BF16_RRMSE_MAX and c > G_BF16_COS_MIN


@requires_quack
def test_gemm_b_k_major_supported():
    """B may be k-major instead of n-major, and must give the same answer.

    ``varlen_m`` constrains A and the outputs to be k-/n-major but B's major mode
    is a template parameter, so ``(E, H, 2I)`` with strides ``(HN, 1, H)`` is
    legal.  Measured: bit-exact against the n-major run.  Recorded here because
    the natural expectation -- "non-contiguous B is rejected" -- is wrong, and a
    future reader should not have to rediscover that.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = G_TK * G_E
    cu = _cu_seqlens([G_TK] * G_E)
    x, w1 = _mk_fwd(total_M, G_E)
    w1_3d = w1.permute(2, 1, 0).contiguous()
    _, ref = gemm_gated(x, w1_3d, activation=_situ(), cu_seqlens_m=cu)

    w_km = w1_3d.transpose(1, 2).contiguous().transpose(1, 2)
    assert w_km.stride(-1) != 1 and w_km.stride(1) == 1
    _, got = gemm_gated(x, w_km, activation=_situ(), cu_seqlens_m=cu)
    print(f"\n  k-major B: strides={w_km.stride()} "
          f"ndiff={int((got != ref).sum())}")
    assert int((got != ref).sum()) == 0, "k-major B changed the result"


# The unsupported combinations, with the exact exception each one produced when
# measured.  These are asserted as *clean rejections*: the failure mode being
# guarded against is not the exception, it is a future version that accepts one
# of these silently and returns garbage (or faults the device).  All of them
# raise during shape/stride/dtype validation, before any kernel is compiled, so
# these tests cost nothing.
REJECTIONS = [
    pytest.param("a_not_k_major", AssertionError,
                 "varlen_m requires A to be k-major", id="A-not-k-major"),
    pytest.param("a_inner_strided", AssertionError,
                 "varlen_m requires A to be k-major", id="A-inner-strided"),
    pytest.param("b_no_unit_stride", RuntimeError,
                 r"Expected strides\[leading_dim\] == 1", id="B-no-unit-stride"),
    pytest.param("float32", TypeError,
                 "unsupported combination of types and majors", id="float32"),
    pytest.param("dtype_mismatch", AssertionError,
                 "A and B must have the same dtype", id="dtype-mismatch"),
    pytest.param("short_cu_seqlens", AssertionError,
                 "cu_seqlens_m must have shape", id="short-cu-seqlens"),
    pytest.param("k_mismatch", AssertionError,
                 "K dimension mismatch", id="K-mismatch"),
]


@requires_quack
@pytest.mark.parametrize("case,exc,msg", REJECTIONS)
def test_gemm_rejects_unsupported_layout(case, exc, msg):
    """Unsupported operands must be rejected with a specific, readable error."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = G_TK * G_E
    cu = _cu_seqlens([G_TK] * G_E)
    x, w1 = _mk_fwd(total_M, G_E)
    w1_3d = w1.permute(2, 1, 0).contiguous()
    a, b, act, kw = x, w1_3d, _situ(), {}

    if case == "a_not_k_major":
        a = x.t().contiguous().t()              # strides (1, total_M)
    elif case == "a_inner_strided":
        # A view with neither stride equal to 1: strides (2H, 2).
        a = torch.randn(total_M, 2 * G_H, dtype=torch.bfloat16,
                        device="cuda")[:, ::2]
        kw["tuned"] = False                     # measured on the untuned path
    elif case == "b_no_unit_stride":
        b = torch.randn(G_E, G_H, 4 * G_I, dtype=torch.bfloat16,
                        device="cuda")[:, :, ::2]
        kw["tuned"] = False                     # measured on the untuned path
    elif case == "float32":
        a, b = x.float(), w1_3d.float()
    elif case == "dtype_mismatch":
        b = w1_3d.half()
    elif case == "short_cu_seqlens":
        cu = _cu_seqlens([G_TK] * (G_E - 1))
    elif case == "k_mismatch":
        b = w1_3d[:, : G_H // 2, :].contiguous()
    else:
        raise AssertionError(case)

    with pytest.raises(exc, match=msg):
        gemm_gated(a, b, activation=act, cu_seqlens_m=cu, **kw)


# A malformed descriptor must be rejected before it can become a cache key: an
# accepted-but-wrong beta would compile a kernel and then serve it from cache to
# every later call in the process.  ``situ_glu`` with no fields is *not* an error
# -- it resolves to the DEFAULT_SITU_BETA / DEFAULT_SITU_LINEAR_BETA pair -- which
# is recorded here so it is not mistaken for a bug.  (Full validation of the
# beta/linear_beta grammar lives in ``test_situ_beta_validation.py``; this test
# only asserts that the GEMM entry point honours it rather than routing around
# it.)
BAD_DESCRIPTORS = [
    pytest.param("situ_glu:b=0.0:lb=25.0", id="beta-zero"),
    pytest.param("situ_glu:b=-1.0:lb=25.0", id="beta-negative"),
]


@requires_quack
@pytest.mark.parametrize("descriptor", BAD_DESCRIPTORS)
def test_gemm_rejects_bad_descriptor(descriptor):
    """Measured: ValueError "... must be a positive finite value, but got ..."."""
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = G_TK * G_E
    x, w1 = _mk_fwd(total_M, G_E)
    with pytest.raises(ValueError, match="must be a positive finite value"):
        gemm_gated(x, w1.permute(2, 1, 0).contiguous(), activation=descriptor,
                   cu_seqlens_m=_cu_seqlens([G_TK] * G_E))


@heavy
@requires_quack
def test_gemm_bare_descriptor_is_the_default():
    """``"situ_glu"`` with no fields == ``"situ_glu:b=4.0:lb=25.0"``, bit-for-bit.

    Recorded because the bare form is accepted silently.  It is a second cache
    key, so it compiles a second kernel (hence ``@heavy``); if the two ever
    diverge, a config that spells the activation the short way would train a
    different function from one that spells it out.  Comparison is bit-exact.
    """
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    total_M = G_TK * G_E
    cu = _cu_seqlens([G_TK] * G_E)
    x, w1 = _mk_fwd(total_M, G_E)
    w1_3d = w1.permute(2, 1, 0).contiguous()
    _, explicit = gemm_gated(x, w1_3d, activation=_situ(), cu_seqlens_m=cu)
    _, bare = gemm_gated(x, w1_3d, activation="situ_glu", cu_seqlens_m=cu)
    print(f"\n  bare-vs-explicit descriptor: ndiff={int((bare != explicit).sum())}")
    assert torch.equal(bare, explicit), (
        '"situ_glu" no longer resolves to the same (beta, linear_beta) as '
        f'"{_situ()}"')







