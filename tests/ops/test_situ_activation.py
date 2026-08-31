"""Standalone elementwise CuTe DSL kernel test for ``situ_glu`` / ``dsitu_glu``.

This is the *primary* numerics proof for the SiTU-GLU activation.  It depends only
on ``sonicmoe/quack_utils/activation_situ.py`` — no GEMM, no FP8, no epilogue — so
a failure here is unambiguously an activation-math bug.

What it does
------------
Builds a minimal ``@cute.jit`` / ``@cute.kernel`` elementwise launch that reads
f32 gate/up/dout vectors, calls the device functions under test, and writes f32
results.  Two variants are compiled for every configuration:

* **scalar** path — ``x`` is a ``Float32``
* **packed** path — ``x`` is a ``Tuple[Float32, Float32]``, so
  ``cute.arch.{mul,fma,sub}_packed_f32x2`` are exercised

The packed path is the one production actually runs: the SonicMoE gated epilogues
branch on ``arch >= 100`` and these GPUs are sm_103 (Blackwell).  Both are tested,
and the summary table reports them side by side.

Every comparison is against the **float64** oracle in ``tests/ops/situ_reference.py``.

Assertion thresholds
--------------------
The per-sweep thresholds in ``LIMITS`` were set from measured values on sm_103
(B30Z, CUDA 13.2, cutlass-dsl 4.5.2) with roughly 4x headroom.  Every test prints
its measured numbers under ``-s`` so a human can re-audit them.

Run with::

    SONIC_MOE_QUACK_PATH=<...>/third_party/quack CUDA_VISIBLE_DEVICES=0 \
        python3 -m pytest tests/ops/test_situ_activation.py -x -q -s
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Optional, Tuple

import pytest
import torch

from tests.ops.conftest import requires_blackwell
from tests.ops.situ_reference import (
    HAND_BETA,
    HAND_LINEAR_BETA,
    _hand_check,
    deinterleave,
    dsitu_glu_ref_fp64,
    dsitu_glu_ref_paddlefleet,
    finite_diff_grads_fp64,
    interleave,
    paddle_available,
    paddlefleet_source,
    situ_glu_ref_fp64,
    situ_glu_ref_paddlefleet,
)

pytestmark = [requires_blackwell]

BETA = HAND_BETA               # 4.0  — production value
LINEAR_BETA = HAND_LINEAR_BETA  # 25.0 — production value

NUM_THREADS = 128


# ===========================================================================
# Minimal standalone elementwise CuTe kernels
# ===========================================================================

import cutlass                      # noqa: E402
import cutlass.cute as cute         # noqa: E402
from cutlass import Float32, Int32, const_expr  # noqa: E402
import cuda.bindings.driver as cuda  # noqa: E402

from quack.compile_utils import make_fake_tensor  # noqa: E402

from sonicmoe.quack_utils.activation_situ import (  # noqa: E402
    dsitu_glu,
    situ_glu,
)


class _SituFwdOp:
    """out[i] = situ_glu(x[i], y[i]) — one element (scalar) or two (packed) per thread."""

    def __init__(self, beta: float, linear_beta: Optional[float], precise: bool, packed: bool):
        self.beta = beta
        self.linear_beta = linear_beta
        self.precise = precise
        self.packed = packed

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, mOut: cute.Tensor, stream):
        N = mX.shape[0]
        per_block = const_expr(NUM_THREADS * (2 if self.packed else 1))
        self.kernel(mX, mY, mOut, N).launch(
            grid=[cute.ceil_div(N, per_block), 1, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mY: cute.Tensor, mOut: cute.Tensor, N: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * const_expr(NUM_THREADS) + tidx
        if const_expr(not self.packed):
            if gid < N:
                out = situ_glu(
                    Float32(mX[gid]), Float32(mY[gid]),
                    self.beta, self.linear_beta, self.precise,
                )
                mOut[gid] = out
        else:
            i = gid * Int32(2)
            if i + Int32(1) < N:
                x = (Float32(mX[i]), Float32(mX[i + 1]))
                y = (Float32(mY[i]), Float32(mY[i + 1]))
                out = situ_glu(x, y, self.beta, self.linear_beta, self.precise)
                mOut[i] = out[0]
                mOut[i + 1] = out[1]


class _SituBwdOp:
    """(dx, dy, out) = dsitu_glu(x, y, dout)."""

    def __init__(self, beta: float, linear_beta: Optional[float], precise: bool, packed: bool):
        self.beta = beta
        self.linear_beta = linear_beta
        self.precise = precise
        self.packed = packed

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mDout: cute.Tensor,
        mDx: cute.Tensor,
        mDy: cute.Tensor,
        mOut: cute.Tensor,
        stream,
    ):
        N = mX.shape[0]
        per_block = const_expr(NUM_THREADS * (2 if self.packed else 1))
        self.kernel(mX, mY, mDout, mDx, mDy, mOut, N).launch(
            grid=[cute.ceil_div(N, per_block), 1, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mDout: cute.Tensor,
        mDx: cute.Tensor,
        mDy: cute.Tensor,
        mOut: cute.Tensor,
        N: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gid = bidx * const_expr(NUM_THREADS) + tidx
        if const_expr(not self.packed):
            if gid < N:
                dx, dy, out = dsitu_glu(
                    Float32(mX[gid]), Float32(mY[gid]), Float32(mDout[gid]),
                    self.beta, self.linear_beta, self.precise,
                )
                mDx[gid] = dx
                mDy[gid] = dy
                mOut[gid] = out
        else:
            i = gid * Int32(2)
            if i + Int32(1) < N:
                x = (Float32(mX[i]), Float32(mX[i + 1]))
                y = (Float32(mY[i]), Float32(mY[i + 1]))
                d = (Float32(mDout[i]), Float32(mDout[i + 1]))
                dx, dy, out = dsitu_glu(
                    x, y, d, self.beta, self.linear_beta, self.precise
                )
                mDx[i] = dx[0]
                mDx[i + 1] = dx[1]
                mDy[i] = dy[0]
                mDy[i + 1] = dy[1]
                mOut[i] = out[0]
                mOut[i + 1] = out[1]


_COMPILE_CACHE: dict = {}


def _fake_f32_vec():
    # 1-D f32, dynamic length, 16B-aligned base pointer (torch guarantees far more).
    return make_fake_tensor(cute.Float32, (cute.sym_int(),), divisibility=4)


def _compiled(mode: str, beta, linear_beta, precise, packed):
    key = (mode, beta, linear_beta, precise, packed)
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if mode == "fwd":
        op = _SituFwdOp(beta, linear_beta, precise, packed)
        args = (_fake_f32_vec(), _fake_f32_vec(), _fake_f32_vec())
    else:
        op = _SituBwdOp(beta, linear_beta, precise, packed)
        args = tuple(_fake_f32_vec() for _ in range(6))
    fn = cute.compile(op, *args, stream, options="--enable-tvm-ffi")
    _COMPILE_CACHE[key] = fn
    return fn


def _pad_to(t: torch.Tensor, mult: int) -> torch.Tensor:
    n = t.numel()
    pad = (-n) % mult
    if pad == 0:
        return t
    return torch.cat([t, torch.zeros(pad, dtype=t.dtype, device=t.device)])


def situ_glu_kernel(
    g: torch.Tensor, u: torch.Tensor, beta: float,
    linear_beta: Optional[float], precise: bool, packed: bool,
) -> torch.Tensor:
    """Launch the standalone forward kernel. g/u are 1-D f32 cuda tensors."""
    assert g.dtype == torch.float32 and u.dtype == torch.float32
    assert g.is_cuda and g.shape == u.shape and g.dim() == 1
    n = g.numel()
    mult = 2 * NUM_THREADS if packed else NUM_THREADS
    gp, up = _pad_to(g.contiguous(), mult), _pad_to(u.contiguous(), mult)
    out = torch.zeros_like(gp)
    _compiled("fwd", beta, linear_beta, precise, packed)(gp, up, out)
    return out[:n]


def dsitu_glu_kernel(
    g: torch.Tensor, u: torch.Tensor, dout: torch.Tensor, beta: float,
    linear_beta: Optional[float], precise: bool, packed: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the standalone backward kernel. Returns (dgate, dup, out)."""
    assert g.dtype == torch.float32 and g.is_cuda and g.dim() == 1
    assert g.shape == u.shape == dout.shape
    n = g.numel()
    mult = 2 * NUM_THREADS if packed else NUM_THREADS
    gp = _pad_to(g.contiguous(), mult)
    up = _pad_to(u.contiguous(), mult)
    dp = _pad_to(dout.contiguous(), mult)
    dx = torch.zeros_like(gp)
    dy = torch.zeros_like(gp)
    out = torch.zeros_like(gp)
    _compiled("bwd", beta, linear_beta, precise, packed)(gp, up, dp, dx, dy, out)
    return dx[:n], dy[:n], out[:n]


# ===========================================================================
# Error metrics
# ===========================================================================

# Elements whose |reference| is below SIG_FRAC * max|reference| are excluded from
# the *relative* error statistic.  Reason: ``d(gate_act)/dg`` has a genuine root
# near g ~ -1.2 (the two terms ``(1-t^2)*s`` and ``gate_act*(1-s)`` cancel), so the
# relative error of dgate is unbounded there no matter how good the arithmetic is.
# The unfiltered max-rel is still reported so nothing is hidden.
SIG_FRAC = 1e-3


class Err:
    """Measured error of an f32 kernel result against the fp64 oracle."""

    __slots__ = ("max_abs", "max_rel", "max_rel_sig", "rrmse", "ulp_norm",
                 "n_excluded", "n_bad", "scale", "req_atol", "rtol")

    def __init__(self, actual: torch.Tensor, expected: torch.Tensor,
                 rtol: float = 1e-5):
        a = actual.double().flatten()
        e = expected.double().flatten()
        assert a.shape == e.shape
        self.rtol = rtol
        self.n_bad = int((~torch.isfinite(a)).sum().item())
        finite_e = torch.isfinite(e)
        d = (a - e).abs()
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))
        self.max_abs = d[finite_e].max().item() if finite_e.any() else float("nan")
        self.scale = e[finite_e].abs().max().item() if finite_e.any() else float("nan")
        # req_atol: the smallest atol for which
        #   |a - e| <= atol + rtol * |e|      (torch.testing.assert_close contract)
        # holds everywhere. This is the honest, auditable single number: it says
        # "beyond the allowed relative slack, how much absolute error is left".
        slack = (d - rtol * e.abs())[finite_e]
        self.req_atol = max(slack.max().item(), 0.0) if finite_e.any() else float("nan")
        nz = finite_e & (e != 0)
        rel = torch.zeros_like(d)
        rel[nz] = d[nz] / e[nz].abs()
        self.max_rel = rel[nz].max().item() if nz.any() else 0.0
        sig = nz & (e.abs() >= SIG_FRAC * max(self.scale, 1e-300))
        self.max_rel_sig = rel[sig].max().item() if sig.any() else 0.0
        self.n_excluded = int((nz & ~sig).sum().item())
        en = e[finite_e].norm().item()
        self.rrmse = (d[finite_e].norm().item() / en) if en > 0 else 0.0
        # max_abs normalised by the dynamic range of the answer: ~1.0 means
        # "one float32 ulp of the largest element", the best f32 can do.
        self.ulp_norm = (self.max_abs / (self.scale * 1.1920929e-7)
                         if self.scale > 0 else 0.0)

    def __str__(self):
        return (f"max_rel={self.max_rel:.3e} max_rel_sig={self.max_rel_sig:.3e} "
                f"max_abs={self.max_abs:.3e} req_atol(rtol={self.rtol:g})="
                f"{self.req_atol:.3e} rrmse={self.rrmse:.3e} ulp={self.ulp_norm:.2f}")


# ===========================================================================
# Input sweeps
# ===========================================================================

N_DEFAULT = 8192
# Values that matter: exact zero, the two betas (tanh(1) region), the tanh
# saturation ladder, denormal-adjacent tinies, and plain 1.
_SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 1e-6, -1e-6, 1e-7, -1e-7,
    BETA, -BETA, LINEAR_BETA, -LINEAR_BETA,
    40.0, -40.0, 200.0, -200.0, 1e4, -1e4,
    1e-30, -1e-30, 1e30, -1e30,
]


def _sweep(name: str, n: int = N_DEFAULT):
    """Return (g, u, dout) as 1-D float32 cuda tensors for a named sweep."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0xC0FFEE + len(name))

    def rn(scale):
        return torch.randn(n, generator=gen, device=dev, dtype=torch.float32) * scale

    if name == "randn1":
        return rn(1.0), rn(1.0), rn(1.0)
    if name == "randn3":
        return rn(3.0), rn(3.0), rn(1.0)
    if name == "near_zero":
        # Uniform in [-1e-6, 1e-6]: where tanh implementations differ most in
        # *relative* terms (tanh(x) ~ x, so any additive error dominates).
        g = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * 1e-6
        u = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * 1e-6
        return g.float(), u.float(), rn(1.0)
    if name == "saturate":
        # |g/beta| and |u/linear_beta| large enough that tanh == +-1 in f32.
        lad = torch.tensor([40.0, -40.0, 200.0, -200.0, 1e4, -1e4],
                           device=dev, dtype=torch.float32)
        idx = torch.randint(0, lad.numel(), (n,), generator=gen, device=dev)
        idx2 = torch.randint(0, lad.numel(), (n,), generator=gen, device=dev)
        return lad[idx], lad[idx2], rn(1.0)
    if name == "boundary":
        # Full cartesian product of the special-value list, dout = 1 and randn.
        sv = torch.tensor(_SPECIALS, device=dev, dtype=torch.float32)
        k = sv.numel()
        g = sv.repeat_interleave(k)
        u = sv.repeat(k)
        d = torch.ones(g.numel(), device=dev, dtype=torch.float32)
        # second half with random dout to catch dout-dependent mistakes
        g = torch.cat([g, g])
        u = torch.cat([u, u])
        d = torch.cat([d, torch.randn(d.numel(), generator=gen, device=dev)])
        return g, u, d
    if name == "mixed_scale":
        # gate in the interesting region, up spanning the linear_beta knee.
        return rn(1.5), rn(50.0), rn(1.0)
    if name == "gate_grad_root":
        # Dense sampling around g ~ -1.2 where d(gate_act)/dg crosses zero.
        g = torch.linspace(-2.0, -0.5, n, device=dev, dtype=torch.float32)
        u = rn(1.0)
        return g, u, torch.ones(n, device=dev, dtype=torch.float32)
    raise AssertionError(f"unknown sweep {name}")


SWEEPS = ["randn1", "randn3", "near_zero", "saturate", "boundary",
          "mixed_scale", "gate_grad_root"]


def _oracle(g, u, dout, linear_beta, beta=BETA):
    """fp64 oracle for 1-D inputs. Returns (out, dgate, dup) as fp64 (n,) tensors.

    The oracle consumes the *interleaved* (1, 2n) layout, so this wrapper is also
    the place where the interleave contract is exercised on every single sweep.
    """
    z = interleave(g.double().unsqueeze(0), u.double().unsqueeze(0))
    out, _, _ = situ_glu_ref_fp64(z, beta, linear_beta)
    dz, out2 = dsitu_glu_ref_fp64(z, dout.double().unsqueeze(0), beta, linear_beta)
    # the two forward paths of the oracle must agree bit-for-bit
    assert torch.equal(out, out2)
    dgate, dup = deinterleave(dz)
    return out[0], dgate[0], dup[0]


# Collected measurements: (sweep, lb, precise, path, quantity) -> Err
_TABLE: dict = {}


def _record(sweep, linear_beta, precise, path, quantity, err: Err):
    _TABLE[(sweep, linear_beta, precise, path, quantity)] = err


@pytest.fixture(scope="module", autouse=True)
def _print_table():
    yield
    if not _TABLE:
        return
    print("\n\n" + "=" * 118)
    print("situ_glu measured error: f32 CuTe kernel vs float64 oracle "
          f"(beta={BETA}, sm_{torch.cuda.get_device_capability()[0]}"
          f"{torch.cuda.get_device_capability()[1]})")
    print("=" * 118)
    hdr = (f"{'sweep':<16}{'lb':>6}{'variant':>9}{'path':>8}{'qty':>7}"
           f"{'max_rel':>12}{'max_rel_sig':>13}{'max_abs':>12}{'rrmse':>12}{'ulp':>8}")
    print(hdr)
    print("-" * 118)
    for key in sorted(_TABLE, key=lambda k: (k[0], str(k[1]), not k[2], k[3], k[4])):
        sweep, lb, precise, path, qty = key
        e = _TABLE[key]
        print(f"{sweep:<16}{str(lb):>6}{'precise' if precise else 'approx':>9}"
              f"{path:>8}{qty:>7}{e.max_rel:>12.3e}{e.max_rel_sig:>13.3e}"
              f"{e.max_abs:>12.3e}{e.rrmse:>12.3e}{e.ulp_norm:>8.2f}")
    print("=" * 118)
    print("max_rel_sig excludes elements with |ref| < 1e-3 * max|ref| (genuine "
          "cancellation roots, see SIG_FRAC).")
    print("ulp = max_abs / (max|ref| * 2^-23): 1.0 means one float32 ulp of the "
          "largest element.\n")


# ===========================================================================
# Assertion thresholds — MEASURED on sm_103 then given ~4x headroom.
# ===========================================================================
# Every number below is 4x the worst value observed across
# {scalar, packed} x {lb=25, lb=None} on B30Z / sm_103 / CUDA 13.2 /
# cutlass-dsl 4.5.2.  The raw measurements are reproduced in the summary table
# that this module prints at teardown, so a reviewer can re-derive them.
#
# A limit of ``None`` means "this statistic is meaningless for this cell and is
# only reported, not asserted"; the accompanying absolute bound carries the
# burden instead.  There is exactly one such cell, ``saturate``/``dgate`` — see
# the comment on that row.

_Lim = namedtuple("_Lim", "rel_p rrmse_p ulp_p rel_a rrmse_a ulp_a abs_")

# (sweep, quantity) -> _Lim.  ``bout`` (the forward value that dsitu_glu returns)
# is judged with the same limits as ``out``; they measured identical everywhere.
LIMITS = {
    # ---- ordinary training data: ~1-2 f32 ulp for precise ------------------
    ("randn1", "out"):   _Lim(1.6e-6, 4.0e-7, 6.0, 2.5e-4, 2.0e-5, 450.0, None),
    ("randn1", "dgate"): _Lim(3.5e-5, 3.0e-7, 4.0, 2.0e-3, 1.5e-5, 150.0, None),
    ("randn1", "dup"):   _Lim(1.5e-6, 3.5e-7, 4.0, 2.5e-4, 2.0e-5, 450.0, None),
    ("randn3", "out"):   _Lim(1.5e-6, 4.0e-7, 8.0, 4.0e-4, 2.5e-5, 300.0, None),
    ("randn3", "dgate"): _Lim(5.0e-5, 5.0e-7, 9.0, 2.5e-3, 2.5e-5, 300.0, None),
    ("randn3", "dup"):   _Lim(1.2e-6, 3.0e-7, 4.0, 4.0e-4, 2.0e-5, 200.0, None),
    # ---- |g|,|u| <= 1e-6: tanh(x) ~ x, so this is pure rounding ------------
    # Note the approx limits equal the precise ones: MUFU.TANH is *not* worse
    # here (see test_precise_and_approx_tie_near_zero).
    ("near_zero", "out"):   _Lim(1.1e-6, 3.0e-7, 7.0, 1.1e-6, 3.0e-7, 7.0, None),
    ("near_zero", "dgate"): _Lim(1.2e-6, 3.5e-7, 5.0, 1.2e-6, 3.5e-7, 5.0, None),
    ("near_zero", "dup"):   _Lim(8.0e-7, 2.5e-7, 4.0, 8.0e-7, 2.5e-7, 4.0, None),
    # ---- tanh saturated: |g/beta| >= 10 -----------------------------------
    ("saturate", "out"):   _Lim(3.0e-8, 2.0e-8, 0.3, 2.5e-5, 1.2e-5, 180.0, None),
    # dgate is judged by an ABSOLUTE bound only.  In this regime f32 rounds
    # ``t = tanhf(g/4)`` to exactly +-1, so ``1 - t*t`` is exactly 0 and the
    # kernel returns dgate = 0 while the fp64 oracle keeps ~4e-9.  Every element
    # of the sweep is in that regime, so max_rel = rrmse = 1.0 and ulp = 2^23 by
    # construction.  This is inherent to f32, not a kernel defect: the whole
    # answer is <= 2.6e-4 in absolute terms (2.5e-4 comes from lb=None, where
    # up_act = u = 1e4 multiplies the residual).  Bounding |dgate| itself is the
    # only statement with content.
    ("saturate", "dgate"): _Lim(None, None, None, None, None, None, 1.0e-3),
    ("saturate", "dup"):   _Lim(1.5e-6, 1.5e-6, 11.0, 2.5e-4, 2.5e-4, 2000.0, None),
    # ---- exact zeros, +-beta, +-linear_beta, 1e-30 .. 1e30 -----------------
    # Judged on ulp (scale-relative) rather than an absolute bound: with lb=None
    # the answer legitimately reaches 4e30, so any fixed atol is meaningless.
    ("boundary", "out"):   _Lim(6.0e-7, 1.0e-7, 2.0, 1.6e-4, 2.5e-5, 400.0, None),
    ("boundary", "dgate"): _Lim(9.0e-7, 2.5e-7, 4.0, 1.5e-4, 3.0e-5, 550.0, None),
    ("boundary", "dup"):   _Lim(1.5e-6, 1.5e-7, 2.0, 4.0e-4, 2.5e-5, 250.0, None),
    # ---- gate ~ N(0,1.5), up ~ N(0,50): straddles the linear_beta knee -----
    ("mixed_scale", "out"):   _Lim(1.5e-6, 3.5e-7, 6.0, 3.5e-4, 2.5e-5, 400.0, None),
    ("mixed_scale", "dgate"): _Lim(6.0e-5, 3.0e-7, 5.0, 3.0e-3, 2.0e-5, 250.0, None),
    ("mixed_scale", "dup"):   _Lim(1.0e-4, 4.5e-7, 5.0, 5.0e-4, 4.0e-5, 400.0, None),
    # ---- dense sampling through the root of d(gate_act)/dg (g ~ -1.22) -----
    # max_rel_sig is intrinsically loose for dgate here: even after the SIG_FRAC
    # filter, elements 1e-3 of the way to the peak sit right on the root.
    ("gate_grad_root", "out"):   _Lim(1.6e-6, 4.0e-7, 9.0, 1.5e-4, 3.5e-5, 750.0, None),
    ("gate_grad_root", "dgate"): _Lim(3.5e-4, 8.0e-7, 8.0, 1.6e-2, 4.0e-5, 320.0, None),
    ("gate_grad_root", "dup"):   _Lim(1.6e-6, 4.0e-7, 13.0, 1.5e-4, 3.5e-5, 950.0, None),
}


def _limit(sweep, qty, precise):
    """-> (max_rel_sig_limit, rrmse_limit, ulp_limit, abs_limit); None = not asserted."""
    lim = LIMITS[(sweep, "out" if qty == "bout" else qty)]
    if precise:
        return lim.rel_p, lim.rrmse_p, lim.ulp_p, lim.abs_
    return lim.rel_a, lim.rrmse_a, lim.ulp_a, lim.abs_


def _check(sweep, linear_beta, precise, path, qty, actual, expected):
    e = Err(actual, expected)
    _record(sweep, linear_beta, precise, path, qty, e)
    rel_lim, rrmse_lim, ulp_lim, abs_lim = _limit(sweep, qty, precise)
    tag = f"[{sweep} lb={linear_beta} {'precise' if precise else 'approx'} {path} {qty}]"
    print(f"  {tag:<62} {e}")
    assert e.n_bad == 0, f"{tag} produced {e.n_bad} non-finite values"
    if rel_lim is not None:
        assert e.max_rel_sig <= rel_lim, (
            f"{tag} max_rel_sig {e.max_rel_sig:.3e} > {rel_lim:.3e}")
    if rrmse_lim is not None:
        assert e.rrmse <= rrmse_lim, f"{tag} rrmse {e.rrmse:.3e} > {rrmse_lim:.3e}"
    if ulp_lim is not None:
        assert e.ulp_norm <= ulp_lim, (
            f"{tag} max_abs {e.max_abs:.3e} = {e.ulp_norm:.1f} f32-ulp of max|ref| "
            f"({e.scale:.3e}) > {ulp_lim}")
    if abs_lim is not None:
        assert e.max_abs <= abs_lim, (
            f"{tag} max_abs {e.max_abs:.3e} > {abs_lim:.3e} (absolute bound; "
            f"relative statistics are meaningless for this cell)")
    return e


# ===========================================================================
# 0. Oracle self-validation (no GPU kernel involved)
# ===========================================================================


def test_oracle_hand_check():
    """Hardcoded hand-computed values pin the oracle's layout, signs and scale."""
    _hand_check(verbose=True)


def test_oracle_vs_paddlefleet_random():
    """Independent second opinion: the real PaddleFleet eager functions."""
    ok, why = paddle_available()
    if not ok:
        pytest.skip(f"paddle/paddlefleet not importable: {why}")
    print(f"\n  paddlefleet oracle source: {paddlefleet_source()}")
    torch.manual_seed(7)
    g = torch.randn(512, dtype=torch.float64) * 2.0
    u = torch.randn(512, dtype=torch.float64) * 8.0
    d = torch.randn(512, dtype=torch.float64)
    z = interleave(g.unsqueeze(0), u.unsqueeze(0))
    dout = d.unsqueeze(0)

    for lb in (LINEAR_BETA, None):
        out64, _, _ = situ_glu_ref_fp64(z, BETA, lb)
        dz64, _ = dsitu_glu_ref_fp64(z, dout, BETA, lb)
        pf_out = situ_glu_ref_paddlefleet(z, BETA, lb)
        pf_dz, pf_o = dsitu_glu_ref_paddlefleet(z, dout, BETA, lb)
        # PaddleFleet computes in float32 internally -> f32-level agreement.
        for name, a, b in (
            ("fwd out", pf_out, out64),
            ("bwd out", pf_o, out64),
            ("dgate", pf_dz[:, 0::2], dz64[:, 0::2]),
            ("dup", pf_dz[:, 1::2], dz64[:, 1::2]),
        ):
            e = Err(a, b)
            print(f"  [paddlefleet lb={lb} {name}] {e}")
            assert e.n_bad == 0
            assert e.rrmse < 1e-6, f"paddlefleet {name} rrmse {e.rrmse:.3e}"
            assert e.max_rel_sig < 1e-5, f"paddlefleet {name} {e.max_rel_sig:.3e}"


# Finite-difference vs analytic limits: (max_rel_sig, rrmse), 4x measured worst.
# Both sides are fp64, so these bounds are about the *step size* and the
# conditioning of the sweep, not about float32 at all.
FD_LIMITS = {
    "randn1":         (1.0e-8, 1.0e-10),   # measured 3.2e-9 / 1.6e-11
    "randn3":         (5.0e-9, 1.0e-10),   # measured 9.1e-10 / 2.2e-11
    "near_zero":      (1.0e-11, 1.0e-11),  # measured 2.1e-12 / 2.1e-12
    # dgate at |g/beta|>=10 is ~1e-7 while the forward value is ~1e5, so the
    # central difference loses ~12 digits to cancellation even in fp64. The
    # agreement is still 1e-4-level, which is far tighter than any plausible
    # formula error (a wrong derivative would be off by O(1)).
    "saturate":       (3.0e-4, 2.0e-4),    # measured 6.6e-5 / 3.9e-5
    "boundary":       (1.0e-9, 1.0e-10),   # measured 1.7e-10 / 1.1e-11
    "mixed_scale":    (2.0e-8, 1.0e-10),   # measured 3.2e-9 / 2.7e-11
    "gate_grad_root": (1.0e-7, 1.0e-9),    # measured 2.6e-8 / 7.9e-11
}


@pytest.mark.parametrize("lb", [LINEAR_BETA, None], ids=["lb25", "lbNone"])
@pytest.mark.parametrize("sweep", SWEEPS)
def test_finite_difference_backward(sweep, lb):
    """Validate the derivative *formula* itself, not just its implementation.

    Central differences on the fp64 **forward** oracle must reproduce the fp64
    analytic backward.  If the analytic formula in ``activation_situ.py``'s
    docstring were wrong, this is the test that would catch it — the kernel
    comparisons alone could not, because they share the formula.
    """
    g, u, d = _sweep(sweep, n=2048)
    z = interleave(g.double().unsqueeze(0), u.double().unsqueeze(0))
    dout = d.double().unsqueeze(0)
    dz_an, _ = dsitu_glu_ref_fp64(z, dout, BETA, lb)
    dz_fd = finite_diff_grads_fp64(z, dout, BETA, lb)
    rel_lim, rrmse_lim = FD_LIMITS[sweep]

    for name, sl in (("dgate", slice(0, None, 2)), ("dup", slice(1, None, 2))):
        a, b = dz_fd[:, sl], dz_an[:, sl]
        e = Err(a, b)
        print(f"  [fd {sweep} lb={lb} {name}] {e}")
        assert e.n_bad == 0, f"fd {name} non-finite"
        # The relative statistic blows up at the genuine gate_grad root and
        # wherever the reference is a cancellation residue, hence max_rel_sig
        # (which drops elements below SIG_FRAC of the peak) plus rrmse.
        assert e.max_rel_sig < rel_lim, (
            f"finite-difference vs analytic {name} disagree: "
            f"max_rel_sig={e.max_rel_sig:.3e} > {rel_lim:.3e} — the derivative "
            f"FORMULA is suspect")
        assert e.rrmse < rrmse_lim, (
            f"fd {name} rrmse {e.rrmse:.3e} > {rrmse_lim:.3e}")


# ===========================================================================
# 1. Kernel vs fp64 oracle — the main matrix
# ===========================================================================

PATHS = [pytest.param(False, id="scalar"), pytest.param(True, id="packed")]
PRECISE = [pytest.param(True, id="precise"), pytest.param(False, id="approx")]


@pytest.mark.parametrize("packed", PATHS)
@pytest.mark.parametrize("precise", PRECISE)
@pytest.mark.parametrize("lb", [LINEAR_BETA, None], ids=["lb25", "lbNone"])
@pytest.mark.parametrize("sweep", SWEEPS)
def test_forward_kernel_vs_oracle(sweep, lb, precise, packed):
    """f32 ``situ_glu`` kernel output vs the float64 oracle."""
    g, u, d = _sweep(sweep)
    out_ref, _, _ = _oracle(g, u, d, lb)
    out = situ_glu_kernel(g, u, BETA, lb, precise, packed)
    path = "packed" if packed else "scalar"
    print()
    _check(sweep, lb, precise, path, "out", out, out_ref)


@pytest.mark.parametrize("packed", PATHS)
@pytest.mark.parametrize("precise", PRECISE)
@pytest.mark.parametrize("lb", [LINEAR_BETA, None], ids=["lb25", "lbNone"])
@pytest.mark.parametrize("sweep", SWEEPS)
def test_backward_kernel_vs_oracle(sweep, lb, precise, packed):
    """f32 ``dsitu_glu`` (dgate, dup, out) vs the float64 oracle."""
    g, u, d = _sweep(sweep)
    out_ref, dg_ref, du_ref = _oracle(g, u, d, lb)
    dg, du, out = dsitu_glu_kernel(g, u, d, BETA, lb, precise, packed)
    path = "packed" if packed else "scalar"
    print()
    _check(sweep, lb, precise, path, "dgate", dg, dg_ref)
    _check(sweep, lb, precise, path, "dup", du, du_ref)
    # ``out`` is consumed by the epilogue's ColVecReduce to build the
    # router-score gradient, so it matters as much as the gradients.
    _check(sweep, lb, precise, path, "bout", out, out_ref)


@pytest.mark.parametrize("packed", PATHS)
@pytest.mark.parametrize("precise", PRECISE)
@pytest.mark.parametrize("lb", [LINEAR_BETA, None], ids=["lb25", "lbNone"])
def test_fwd_bwd_out_agree(lb, precise, packed):
    """``dsitu_glu``'s returned ``out`` must match ``situ_glu``'s output.

    The docstring in activation_situ.py claims they are "bit-comparable".  They
    are computed by different expression trees, so verify how close they really
    are and report the gap.
    """
    g, u, d = _sweep("randn1")
    fwd = situ_glu_kernel(g, u, BETA, lb, precise, packed)
    _, _, bwd_out = dsitu_glu_kernel(g, u, d, BETA, lb, precise, packed)
    n_diff = int((fwd != bwd_out).sum().item())
    max_d = (fwd.double() - bwd_out.double()).abs().max().item()
    print(f"\n  [fwd-vs-bwd out lb={lb} {'precise' if precise else 'approx'} "
          f"{'packed' if packed else 'scalar'}] bitwise-differing="
          f"{n_diff}/{fwd.numel()}, max_abs_diff={max_d:.3e}")
    # Not asserted bit-exact (the epilogues never require it); assert they are
    # within a couple of f32 ulp so recompute-vs-save-z paths stay consistent.
    scale = max(fwd.double().abs().max().item(), 1e-30)
    assert max_d <= 8 * scale * 1.1920929e-7, (
        f"situ_glu and dsitu_glu disagree on the forward value by {max_d:.3e} "
        f"(scale {scale:.3e}) — the epilogue's saved-z and recompute-z paths "
        f"would diverge")


@pytest.mark.parametrize("lb", [LINEAR_BETA, None], ids=["lb25", "lbNone"])
@pytest.mark.parametrize("precise", PRECISE)
def test_scalar_vs_packed(lb, precise):
    """Scalar and packed-f32x2 paths must be numerically equivalent.

    Production runs the packed path (epilogues branch on ``arch >= 100`` and these
    are sm_103), while most human reasoning is done on the scalar path.  A
    divergence here means the packed rewrite is not the same function.
    """
    g, u, d = _sweep("randn3")
    o_s = situ_glu_kernel(g, u, BETA, lb, precise, False)
    o_p = situ_glu_kernel(g, u, BETA, lb, precise, True)
    dgs, dus, os_ = dsitu_glu_kernel(g, u, d, BETA, lb, precise, False)
    dgp, dup_, op_ = dsitu_glu_kernel(g, u, d, BETA, lb, precise, True)
    print()
    for name, a, b in (("out", o_p, o_s), ("dgate", dgp, dgs),
                       ("dup", dup_, dus), ("bout", op_, os_)):
        e = Err(a, b.double())
        nbit = int((a != b).sum().item())
        print(f"  [scalar-vs-packed lb={lb} {'precise' if precise else 'approx'} "
              f"{name}] bitwise-differing={nbit}/{a.numel()}  {e}")
        assert e.n_bad == 0
        assert e.rrmse < 1e-6, f"packed vs scalar {name} rrmse {e.rrmse:.3e}"
        assert e.ulp_norm <= 8.0, (
            f"packed vs scalar {name} differ by {e.ulp_norm:.1f} f32-ulp")


def test_saturation_no_nan_inf():
    """Extreme *finite* inputs must not produce NaN/Inf anywhere, in either path.

    The full cartesian product of the f32 extremes below is fed through both the
    forward and the backward kernel.  With ``linear_beta=25`` every output is
    mathematically bounded (``|gate_act| <= 4``, ``|up_act| <= 25``), so any
    non-finite value would be an intermediate overflow inside the activation —
    a real bug.  With ``linear_beta=None`` the up branch is the identity, so
    ``out = gate_act * u`` genuinely overflows f32 for ``|u| ~ 3.4e38``; there
    only NaN is treated as a defect.
    """
    dev = "cuda"
    big = torch.tensor(
        [40.0, -40.0, 200.0, -200.0, 1e4, -1e4, 1e30, -1e30, 3.4e38, -3.4e38,
         1e38, 0.0],
        device=dev, dtype=torch.float32)
    assert torch.isfinite(big).all(), "the extremes list must itself be finite"
    k = big.numel()
    g = big.repeat_interleave(k)
    u = big.repeat(k)
    d = torch.ones_like(g)
    print()
    for packed in (False, True):
        for precise in (True, False):
            for lb in (LINEAR_BETA, None):
                out = situ_glu_kernel(g, u, BETA, lb, precise, packed)
                dg, du, bo = dsitu_glu_kernel(g, u, d, BETA, lb, precise, packed)
                for nm, t in (("out", out), ("dgate", dg), ("dup", du), ("bout", bo)):
                    n_nan = int(torch.isnan(t).sum().item())
                    n_inf = int(torch.isinf(t).sum().item())
                    ctx = (f"lb={lb} precise={precise} packed={packed}")
                    assert n_nan == 0, f"{nm} has {n_nan} NaN for {ctx}"
                    if lb is None and nm in ("out", "bout", "dgate"):
                        # up_act = u is unbounded: 4 * 3.4e38 overflows f32.
                        continue
                    assert n_inf == 0, f"{nm} has {n_inf} Inf for {ctx}"
            print(f"  [saturation packed={packed} precise={precise}] "
                  f"no NaN; no Inf where the result is mathematically bounded")


def test_nan_input_propagates():
    """A NaN input must yield NaN on every output that mathematically depends on it.

    Not a correctness requirement of the model, but a diagnosability one: if the
    activation swallowed NaNs, a corrupted GEMM output upstream would surface as
    a slow loss drift instead of an immediate NaN loss.

    The dependency structure is not uniform, so the expectation is per output:

    ==========  ================================================================
    ``out``     ``gate_act(g) * up_act(u)``            -> depends on g and u
    ``dgate``   ``dout * up_act(u) * gate_grad(g)``    -> depends on g and u
    ``dup``     ``dout * gate_act(g) * up_grad(u)``    -> depends on g always,
                on u only when ``linear_beta`` is set; with ``linear_beta=None``
                ``up_grad == 1`` and ``d(out)/du = gate_act(g)`` is genuinely
                independent of u, so a NaN in ``u`` must NOT appear in ``dup``.
    ==========  ================================================================
    """
    dev = "cuda"
    nan = float("nan")
    g = torch.tensor([nan, 1.0, nan, 1.0] * 32, device=dev, dtype=torch.float32)
    u = torch.tensor([1.0, nan, nan, 1.0] * 32, device=dev, dtype=torch.float32)
    d = torch.ones_like(g)
    g_nan, u_nan = torch.isnan(g), torch.isnan(u)
    print()
    for packed in (False, True):
        for lb in (LINEAR_BETA, None):
            out = situ_glu_kernel(g, u, BETA, lb, True, packed)
            dg, du, bo = dsitu_glu_kernel(g, u, d, BETA, lb, True, packed)
            dup_expect = (g_nan | u_nan) if lb is not None else g_nan
            for nm, t, want in (("out", out, g_nan | u_nan),
                                ("dgate", dg, g_nan | u_nan),
                                ("dup", du, dup_expect),
                                ("bout", bo, g_nan | u_nan)):
                got = torch.isnan(t)
                assert torch.equal(got, want), (
                    f"{nm} NaN propagation wrong for lb={lb} packed={packed}: "
                    f"{int(got.sum())} NaN out vs {int(want.sum())} expected")
        print(f"  [nan-propagation packed={packed}] NaN in -> NaN out on every "
              f"output that depends on the NaN input")


# ===========================================================================
# 2. precise vs approx — is precise=True the right default?
# ===========================================================================


def test_precise_beats_approx():
    """Measure ``precise=True`` against ``precise=False`` and print the verdict.

    ``precise=True``  -> libdevice ``tanhf`` + expf-based sigmoid
    ``precise=False`` -> ``tanh.approx.f32`` (MUFU.TANH) + tanh-based sigmoid

    This test does not assume precise wins; it measures, prints a side-by-side
    table, and only then asserts.  Two things are checked:

    1. the aggregate (geometric-mean RRMSE ratio) favours precise, and
    2. on the three sweeps that resemble real activations (``randn1``,
       ``randn3``, ``mixed_scale``) precise wins by a large margin per cell.

    It deliberately does NOT require precise to win everywhere: measurement says
    the two variants tie on ``near_zero`` and on saturated ``dgate``, and that
    tie is asserted as a tie (see ``TIE_BAND``) rather than swept under the rug.
    """
    # A ratio inside [1/TIE_BAND, TIE_BAND] counts as "same accuracy".
    TIE_BAND = 1.5
    rows = []
    for sweep in SWEEPS:
        g, u, d = _sweep(sweep)
        for lb in (LINEAR_BETA, None):
            out_ref, dg_ref, du_ref = _oracle(g, u, d, lb)
            for packed in (False, True):
                res = {}
                for precise in (True, False):
                    o = situ_glu_kernel(g, u, BETA, lb, precise, packed)
                    dg, du, _ = dsitu_glu_kernel(g, u, d, BETA, lb, precise, packed)
                    res[precise] = (Err(o, out_ref), Err(dg, dg_ref), Err(du, du_ref))
                for qi, qn in enumerate(("out", "dgate", "dup")):
                    p, a = res[True][qi], res[False][qi]
                    ratio = a.rrmse / p.rrmse if p.rrmse > 0 else (
                        float("inf") if a.rrmse > 0 else 1.0)
                    rows.append((sweep, lb, "packed" if packed else "scalar",
                                 qn, p.rrmse, a.rrmse, p.max_rel_sig,
                                 a.max_rel_sig, ratio))

    print("\n\n" + "=" * 112)
    print("precise=True (libdevice tanhf + expf sigmoid)  VS  "
          "precise=False (tanh.approx.f32 + tanh sigmoid)")
    print("=" * 112)
    print(f"{'sweep':<16}{'lb':>6}{'path':>8}{'qty':>7}"
          f"{'rrmse(prec)':>14}{'rrmse(apx)':>13}{'relsig(prec)':>14}"
          f"{'relsig(apx)':>13}{'apx/prec':>11}")
    print("-" * 112)
    for r in rows:
        print(f"{r[0]:<16}{str(r[1]):>6}{r[2]:>8}{r[3]:>7}"
              f"{r[4]:>14.3e}{r[5]:>13.3e}{r[6]:>14.3e}{r[7]:>13.3e}{r[8]:>11.1f}x")
    print("-" * 112)
    ratios = [r[8] for r in rows if math.isfinite(r[8])]
    geo = math.exp(sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios))

    ties = [r for r in rows if 1.0 / TIE_BAND <= r[8] <= TIE_BAND]
    losses = [r for r in rows if r[8] < 1.0 / TIE_BAND]
    print(f"geometric-mean rrmse ratio approx/precise = {geo:.1f}x  "
          f"(>1 means precise is more accurate)")
    print(f"cells where the two variants TIE (within {TIE_BAND}x): "
          f"{len(ties)}/{len(rows)}")
    for r in ties:
        print(f"    TIE   {r[0]:<16} lb={str(r[1]):<5} {r[2]:<7} {r[3]:<6} "
              f"precise rrmse {r[4]:.3e} vs approx {r[5]:.3e} ({r[8]:.2f}x)")
    if losses:
        print(f"\n!!! approx was MORE accurate than precise (by > {TIE_BAND}x) "
              f"in {len(losses)} cells !!!")
        for r in losses:
            print(f"    LOSS  {r[0]:<16} lb={str(r[1]):<5} {r[2]:<7} {r[3]:<6} "
                  f"precise rrmse {r[4]:.3e} vs approx {r[5]:.3e} ({r[8]:.2f}x)")
    print("=" * 112 + "\n")

    assert geo > 5.0, (
        f"precise=True is only {geo:.2f}x better than precise=False in "
        f"geometric-mean RRMSE. If this is <= 1, precise=True should NOT be the "
        f"default and the extra expf/tanhf cost is being paid for nothing.")
    assert not losses, (
        f"precise=True was materially LESS accurate than precise=False in "
        f"{len(losses)} of {len(rows)} measurements: {losses[:3]}")
    # Per-cell margin on the sweeps that look like real activations.
    REALISTIC = {"randn1", "randn3", "mixed_scale"}
    thin = [r for r in rows if r[0] in REALISTIC and r[8] < 10.0]
    assert not thin, (
        f"precise=True wins by less than 10x on realistic data in "
        f"{len(thin)} cells: {thin[:3]}")


def test_precise_and_approx_tie_near_zero():
    """FINDING: the near-zero regime does NOT favour ``precise=True``.

    The intuition is that ``tanh(x) ~ x`` for tiny x, so an absolute error of
    ~2^-24 in MUFU.TANH would become a relative error of order 1.  Measurement
    says otherwise: for |g|,|u| <= 1e-6 both variants land at ~1 f32 ulp and
    ``approx`` is in fact marginally *better* (its tanh happens to round the
    identity branch slightly more favourably here).  The reason is that
    ``tanh.approx.f32`` is spec'd in *relative* terms near zero too, and the
    dominant error in this regime is the final f32 multiply, not the tanh.

    So the case for ``precise=True`` rests entirely on ordinary-magnitude data
    (see ``test_precise_beats_approx``), not on the near-zero tail.  This test
    pins the tie so a future regression in either direction is visible.
    """
    TIE_BAND = 2.0
    g, u, d = _sweep("near_zero")
    out_ref, dg_ref, du_ref = _oracle(g, u, d, LINEAR_BETA)
    print()
    summary = {}
    for precise in (True, False):
        o = situ_glu_kernel(g, u, BETA, LINEAR_BETA, precise, True)
        dg, du, _ = dsitu_glu_kernel(g, u, d, BETA, LINEAR_BETA, precise, True)
        eo, ed, eu = Err(o, out_ref), Err(dg, dg_ref), Err(du, du_ref)
        summary[precise] = (eo, ed, eu)
        lbl = "precise" if precise else "approx "
        print(f"  [near_zero packed {lbl}] out   {eo}")
        print(f"  [near_zero packed {lbl}] dgate {ed}")
        print(f"  [near_zero packed {lbl}] dup   {eu}")
    for qi, qn in enumerate(("out", "dgate", "dup")):
        p, a = summary[True][qi], summary[False][qi]
        ratio = a.rrmse / max(p.rrmse, 1e-30)
        print(f"  near-zero {qn:<5} RRMSE: precise={p.rrmse:.3e} "
              f"approx={a.rrmse:.3e} -> approx/precise = {ratio:.2f}x")
        assert 1.0 / TIE_BAND <= ratio <= TIE_BAND, (
            f"near_zero {qn}: precise and approx no longer tie "
            f"(approx/precise = {ratio:.2f}x, expected within {TIE_BAND}x). "
            f"Either a tanh implementation changed or the sweep drifted; "
            f"re-measure before adjusting this bound.")
        # Both must still be at the f32 noise floor: ~1 ulp of the peak.
        for nm, e in (("precise", p), ("approx", a)):
            assert e.ulp_norm <= 7.0, (
                f"near_zero {qn} {nm} is {e.ulp_norm:.1f} f32-ulp off the "
                f"oracle; the near-zero path is no longer exact-to-rounding")

