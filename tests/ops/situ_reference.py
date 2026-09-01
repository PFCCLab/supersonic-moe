"""Float64 oracle for the ``situ_glu`` activation used by the SonicMoE epilogues.

Three independent implementations live here so a layout or sign error cannot hide:

1. :func:`situ_glu_ref_fp64` / :func:`dsitu_glu_ref_fp64`
   Pure float64 torch, derived directly from the closed-form math.  This is the
   gold standard every kernel result is measured against.
2. :func:`situ_glu_ref_paddlefleet` / :func:`dsitu_glu_ref_paddlefleet`
   Calls the *actual* PaddleFleet eager functions
   (``paddlefleet.transformer.activations``) so we cross-validate against the
   code the model itself trains with.  Optional: paddle may not be importable.
3. :func:`_hand_check`
   Hardcoded constants for ``beta=4, linear_beta=25`` that pin down the absolute
   scale of the answer, so a systematically-wrong-but-self-consistent oracle
   (the classic interleaved/chunked mixup) is caught immediately.

LAYOUT
------
The SonicMoE gated epilogues hand the activation **interleaved** pairs::

    z[:, 0::2] = gate      z[:, 1::2] = up          shape (TK, 2I)

PaddleFleet's ``situ_glu`` instead splits with ``paddle.chunk(x, 2, axis=-1)``,
i.e. **chunked**::

    x[:, :I] = gate        x[:, I:] = up

Feeding the interleaved tensor straight into PaddleFleet's function yields a
stable, plausible, and completely wrong oracle.  Every conversion in this module
goes through :func:`deinterleave` / :func:`interleave` explicitly.

MATH
----
::

    t        = tanh(g / beta)
    s        = sigmoid(g)
    gate_act = beta * t * s
    up_act   = linear_beta * tanh(u / linear_beta)   if linear_beta is not None else u
    out      = gate_act * up_act

    d(gate_act)/dg = (1 - t^2) * s + gate_act * (1 - s)
    d(up_act)/du   = 1 - tanh(u / linear_beta)^2     if linear_beta is not None else 1
    dgate = dout * up_act   * d(gate_act)/dg
    dup   = dout * gate_act * d(up_act)/du
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

__all__ = [
    "deinterleave",
    "interleave",
    "interleaved_to_chunked",
    "chunked_to_interleaved",
    "situ_glu_ref_fp64",
    "dsitu_glu_ref_fp64",
    "situ_glu_ref_paddlefleet",
    "dsitu_glu_ref_paddlefleet",
    "paddle_available",
    "paddlefleet_source",
    "finite_diff_grads_fp64",
    "HAND_CASES",
]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def deinterleave(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(..., 2I) interleaved -> (gate, up), each (..., I)."""
    assert z.shape[-1] % 2 == 0, f"last dim must be even, got {z.shape[-1]}"
    return z[..., 0::2], z[..., 1::2]


def interleave(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """(gate, up) each (..., I) -> (..., 2I) interleaved."""
    assert gate.shape == up.shape
    out = torch.empty(
        *gate.shape[:-1], 2 * gate.shape[-1], dtype=gate.dtype, device=gate.device
    )
    out[..., 0::2] = gate
    out[..., 1::2] = up
    return out


def interleaved_to_chunked(z: torch.Tensor) -> torch.Tensor:
    """(..., 2I) interleaved -> (..., 2I) chunked ``[gate | up]``."""
    gate, up = deinterleave(z)
    return torch.cat([gate, up], dim=-1)


def chunked_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """(..., 2I) chunked ``[gate | up]`` -> (..., 2I) interleaved."""
    assert x.shape[-1] % 2 == 0
    half = x.shape[-1] // 2
    return interleave(x[..., :half], x[..., half:])


# ---------------------------------------------------------------------------
# 1. float64 gold standard
# ---------------------------------------------------------------------------


def _situ_parts_fp64(
    gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: Optional[float]
):
    """Shared float64 forward intermediates. Returns (t, s, gate_act, up_act, tu)."""
    assert gate.dtype == torch.float64 and up.dtype == torch.float64
    b = torch.tensor(float(beta), dtype=torch.float64, device=gate.device)
    t = torch.tanh(gate / b)
    s = torch.sigmoid(gate)
    gate_act = b * t * s
    if linear_beta is None:
        tu = None
        up_act = up
    else:
        lb = torch.tensor(float(linear_beta), dtype=torch.float64, device=up.device)
        tu = torch.tanh(up / lb)
        up_act = lb * tu
    return t, s, gate_act, up_act, tu


def situ_glu_ref_fp64(
    z: torch.Tensor, beta: float, linear_beta: Optional[float]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward oracle.

    Parameters
    ----------
    z : (..., 2I) **float64**, **interleaved** (``z[..., 0::2]`` = gate).

    Returns
    -------
    (out, gate_act, up_act) — each (..., I) float64.
    """
    assert z.dtype == torch.float64, f"oracle requires float64, got {z.dtype}"
    gate, up = deinterleave(z)
    _, _, gate_act, up_act, _ = _situ_parts_fp64(gate, up, beta, linear_beta)
    return gate_act * up_act, gate_act, up_act


def dsitu_glu_ref_fp64(
    z: torch.Tensor,
    dout: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward oracle.

    Parameters
    ----------
    z    : (..., 2I) float64 interleaved pre-activation.
    dout : (..., I)  float64 gradient w.r.t. the post-activation output.
           NOTE: the epilogue's ``act_bwd_fn`` receives ``dout`` **already**
           scaled by the router probs, so there is no ``probs`` factor here.

    Returns
    -------
    (dz, out) where ``dz`` is (..., 2I) float64 **re-interleaved** exactly like
    the kernel writes it (``dz[..., 0::2]`` = dgate), and ``out`` is (..., I).
    """
    assert z.dtype == torch.float64 and dout.dtype == torch.float64
    gate, up = deinterleave(z)
    assert dout.shape == gate.shape, f"dout {dout.shape} vs gate {gate.shape}"

    t, s, gate_act, up_act, tu = _situ_parts_fp64(gate, up, beta, linear_beta)

    # d(gate_act)/dg = (1 - t^2) * s + gate_act * (1 - s)
    gate_grad = (1.0 - t * t) * s + gate_act * (1.0 - s)
    if linear_beta is None:
        up_grad = torch.ones_like(up)
    else:
        up_grad = 1.0 - tu * tu

    dgate = dout * up_act * gate_grad
    dup = dout * gate_act * up_grad
    return interleave(dgate, dup), gate_act * up_act


# ---------------------------------------------------------------------------
# 2. PaddleFleet cross-validation (independent second opinion)
# ---------------------------------------------------------------------------

_PADDLE_ERR: Optional[str] = None
_pf_act = None

# The ground truth named by the task.  Prepended to sys.path so the in-tree
# PaddleFleet wins over any installed copy (they are usually identical, but the
# oracle must be pinned to a file we can point at).
_PADDLEFLEET_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "src")
)


def _load_paddlefleet():
    """Import ``paddlefleet.transformer.activations`` lazily; record failure."""
    global _pf_act, _PADDLE_ERR
    if _pf_act is not None or _PADDLE_ERR is not None:
        return _pf_act
    try:  # pragma: no cover - environment dependent
        if os.path.isdir(os.path.join(_PADDLEFLEET_SRC, "paddlefleet")):
            if _PADDLEFLEET_SRC not in sys.path:
                sys.path.insert(0, _PADDLEFLEET_SRC)
        import paddle  # noqa: F401
        from paddlefleet.transformer import activations as pf_act

        _pf_act = pf_act
    except Exception as exc:  # pragma: no cover
        _PADDLE_ERR = f"{type(exc).__name__}: {exc}"
    return _pf_act


def paddlefleet_source() -> str:
    """Path of the ``activations.py`` actually used as the second opinion."""
    mod = _load_paddlefleet()
    return getattr(mod, "__file__", "<unavailable>")


def paddle_available() -> Tuple[bool, str]:
    """(available, reason-if-not). Used to build a *visible* pytest skip."""
    mod = _load_paddlefleet()
    return (mod is not None), (_PADDLE_ERR or "")


def _torch_to_paddle(t: torch.Tensor):
    import paddle
    import numpy as np

    return paddle.to_tensor(np.ascontiguousarray(t.detach().cpu().numpy()))


def _paddle_to_torch(p, device, dtype=torch.float64) -> torch.Tensor:
    return torch.from_numpy(p.numpy()).to(device=device, dtype=dtype)


def situ_glu_ref_paddlefleet(
    z: torch.Tensor, beta: float, linear_beta: Optional[float]
) -> torch.Tensor:
    """Forward via the real ``paddlefleet...situ_glu``.

    ``z`` is (..., 2I) **interleaved**; we convert to chunked for PaddleFleet and
    hand back its (..., I) output.  PaddleFleet computes in float32 internally
    (``x.astype("float32")``) regardless of the input dtype, so the result is a
    float32-accuracy answer promoted back to float64 — compare with a float32-ish
    tolerance, not bit-exactly against the fp64 oracle.
    """
    mod = _load_paddlefleet()
    assert mod is not None, f"paddle unavailable: {_PADDLE_ERR}"
    chunked = interleaved_to_chunked(z.double())
    p_out = mod.situ_glu(_torch_to_paddle(chunked), beta=beta, linear_beta=linear_beta)
    return _paddle_to_torch(p_out, z.device)


def dsitu_glu_ref_paddlefleet(
    z: torch.Tensor,
    dout: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward via the real ``paddlefleet...situ_glu_scale_backward`` with probs=1.

    ``situ_glu_scale_backward`` folds a router ``probs`` factor into the gradient
    (``out_grad * probs``); the epilogue's ``act_bwd_fn`` does not, because
    ``dout`` arrives pre-scaled.  Passing ``probs = 1`` makes the two contracts
    identical.

    Returns ``(dz_interleaved, out)`` in the kernel's layout.
    """
    mod = _load_paddlefleet()
    assert mod is not None, f"paddle unavailable: {_PADDLE_ERR}"
    z2 = z.double().reshape(-1, z.shape[-1])
    d2 = dout.double().reshape(-1, dout.shape[-1])
    chunked = interleaved_to_chunked(z2)
    probs = torch.ones(z2.shape[0], 1, dtype=torch.float64, device=z.device)

    dz_chunked, out, _probs_grad = mod.situ_glu_scale_backward(
        _torch_to_paddle(chunked),
        _torch_to_paddle(probs),
        _torch_to_paddle(d2),
        beta=beta,
        linear_beta=linear_beta,
    )
    dz = chunked_to_interleaved(_paddle_to_torch(dz_chunked, z.device))
    out_t = _paddle_to_torch(out, z.device)
    return dz.reshape(z.shape), out_t.reshape(dout.shape)


# ---------------------------------------------------------------------------
# Finite-difference gradient check (validates the derivative *formula*)
# ---------------------------------------------------------------------------


def finite_diff_grads_fp64(
    z: torch.Tensor,
    dout: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    eps: Optional[float] = None,
) -> torch.Tensor:
    """Central-difference ``dz`` from the fp64 *forward* oracle only.

    Returns (..., 2I) interleaved, same layout as :func:`dsitu_glu_ref_fp64`.

    Step size: a relative step ``h = eps * max(|v|, 1)`` keeps the truncation
    error (O(h^2 * f''')) and the cancellation error (O(macheps * |f| / h))
    balanced near ``h ~ macheps^(1/3) ~ 6e-6`` for float64.  We use 1e-5.
    """
    assert z.dtype == torch.float64 and dout.dtype == torch.float64
    if eps is None:
        eps = 1e-5
    gate, up = deinterleave(z)

    def fwd(g, u):
        _, _, ga, ua, _ = _situ_parts_fp64(g, u, beta, linear_beta)
        return ga * ua

    hg = eps * gate.abs().clamp(min=1.0)
    hu = eps * up.abs().clamp(min=1.0)
    dgate = dout * (fwd(gate + hg, up) - fwd(gate - hg, up)) / (2.0 * hg)
    dup = dout * (fwd(gate, up + hu) - fwd(gate, up - hu)) / (2.0 * hu)
    return interleave(dgate, dup)


# ---------------------------------------------------------------------------
# 3. Hand-computed self test
# ---------------------------------------------------------------------------

# beta = 4.0, linear_beta = 25.0, dout = 1.0.
#
#   t        = tanh(g/4)
#   s        = 1/(1+exp(-g))
#   gate_act = 4 * t * s
#   up_act   = 25 * tanh(u/25)
#   out      = gate_act * up_act
#   dgate    = up_act   * ((1 - t^2)*s + gate_act*(1 - s))
#   dup      = gate_act * (1 - tanh(u/25)^2)
#
# Case A: g = 0.5, u = 1.0
#   t        = tanh(0.125)   = (e^0.25 - 1)/(e^0.25 + 1) = 0.1243530017715962
#   s        = 1/(1+e^-0.5)                             = 0.6224593312018546
#   gate_act = 4 * 0.1243530017715962 * 0.6224593312018546
#            = 0.30961874526276323
#   up_act   = 25 * tanh(0.04)                          = 0.9994670077790893
#   out      = 0.30945372088009004
#   dgate/dg = (1 - t^2)*s + gate_act*(1 - s)           = 0.7297274942662559
#   dup /du  = 1 - tanh(0.04)^2                         = 0.9984017051205778
#   dgate    = 0.9994670077790893 * 0.7297274942662559  = 0.7293385551884274
#   dup      = 0.30961874526276323 * 0.9984017051205778 = 0.3091238832076366
#
# Case B: g = -2.0, u = 30.0  (crosses zero, saturates the up branch)
#   t        = tanh(-0.5)                               = -0.46211715726000974
#   s        = 1/(1+e^2)                                = 0.11920292202211755
#   gate_act = -0.22034286184779028
#   up_act   = 25 * tanh(1.2)                           = 20.84136517530388
#   out      = -4.59224604754133
#   dgate/dg = -0.10033048108160934
#   dup /du  = 1 - tanh(1.2)^2                          = 0.30501999620740905
#   dgate    = -2.0910241944357377
#   dup      = -0.06720897888514264

HAND_CASES = [
    # (g, u, dout, out, dgate, dup)
    (0.5, 1.0, 1.0, 0.30945372088009004, 0.7293385551884274, 0.3091238832076366),
    (-2.0, 30.0, 1.0, -4.59224604754133, -2.0910241944357377, -0.06720897888514264),
]

HAND_BETA = 4.0
HAND_LINEAR_BETA = 25.0


def _hand_check(verbose: bool = True) -> None:
    """Assert fp64 oracle, finite differences and PaddleFleet all agree with the
    hand-computed constants above.  Catches layout / sign / beta-placement bugs."""
    dev = "cpu"
    g = torch.tensor([c[0] for c in HAND_CASES], dtype=torch.float64, device=dev)
    u = torch.tensor([c[1] for c in HAND_CASES], dtype=torch.float64, device=dev)
    d = torch.tensor([c[2] for c in HAND_CASES], dtype=torch.float64, device=dev)
    want_out = torch.tensor([c[3] for c in HAND_CASES], dtype=torch.float64)
    want_dg = torch.tensor([c[4] for c in HAND_CASES], dtype=torch.float64)
    want_du = torch.tensor([c[5] for c in HAND_CASES], dtype=torch.float64)

    z = interleave(g.unsqueeze(0), u.unsqueeze(0))  # (1, 4) interleaved
    dout = d.unsqueeze(0)

    # --- the interleave itself, spelled out ---
    assert z.shape == (1, 4)
    assert z[0, 0].item() == HAND_CASES[0][0], "z[0] must be gate of case 0"
    assert z[0, 1].item() == HAND_CASES[0][1], "z[1] must be up of case 0"
    assert z[0, 2].item() == HAND_CASES[1][0], "z[2] must be gate of case 1"
    assert z[0, 3].item() == HAND_CASES[1][1], "z[3] must be up of case 1"

    # --- fp64 oracle forward ---
    out, gate_act, up_act = situ_glu_ref_fp64(z, HAND_BETA, HAND_LINEAR_BETA)
    torch.testing.assert_close(out[0], want_out, rtol=1e-12, atol=0.0)

    # --- fp64 oracle backward, checking the interleaved output slots ---
    dz, out2 = dsitu_glu_ref_fp64(z, dout, HAND_BETA, HAND_LINEAR_BETA)
    torch.testing.assert_close(out2[0], want_out, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(dz[0, 0::2], want_dg, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(dz[0, 1::2], want_du, rtol=1e-12, atol=0.0)

    # --- finite differences on the forward oracle ---
    dz_fd = finite_diff_grads_fp64(z, dout, HAND_BETA, HAND_LINEAR_BETA)
    torch.testing.assert_close(dz_fd[0, 0::2], want_dg, rtol=2e-9, atol=1e-11)
    torch.testing.assert_close(dz_fd[0, 1::2], want_du, rtol=2e-9, atol=1e-11)

    # --- linear_beta=None variant (identity up branch) ---
    out_n, ga_n, ua_n = situ_glu_ref_fp64(z, HAND_BETA, None)
    torch.testing.assert_close(ua_n[0], u, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out_n[0], ga_n[0] * u, rtol=0.0, atol=0.0)
    dz_n, _ = dsitu_glu_ref_fp64(z, dout, HAND_BETA, None)
    torch.testing.assert_close(dz_n[0, 1::2], ga_n[0] * d, rtol=1e-13, atol=0.0)
    dz_n_fd = finite_diff_grads_fp64(z, dout, HAND_BETA, None)
    torch.testing.assert_close(dz_n[0], dz_n_fd[0], rtol=2e-9, atol=1e-11)

    # --- PaddleFleet second opinion (float32 internals -> looser tol) ---
    ok, why = paddle_available()
    if ok:
        pf_out = situ_glu_ref_paddlefleet(z, HAND_BETA, HAND_LINEAR_BETA)
        torch.testing.assert_close(pf_out[0], want_out, rtol=2e-6, atol=1e-7)
        pf_dz, pf_o = dsitu_glu_ref_paddlefleet(z, dout, HAND_BETA, HAND_LINEAR_BETA)
        torch.testing.assert_close(pf_dz[0, 0::2], want_dg, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(pf_dz[0, 1::2], want_du, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(pf_o[0], want_out, rtol=2e-6, atol=1e-7)
        # And the trap itself: feeding the *interleaved* tensor to PaddleFleet
        # (no re-permute) must give a visibly different answer, proving the
        # de-interleave is load-bearing and not a no-op.
        import numpy as np
        import paddle

        mod = _load_paddlefleet()
        wrong = mod.situ_glu(
            paddle.to_tensor(np.ascontiguousarray(z.numpy())),
            beta=HAND_BETA,
            linear_beta=HAND_LINEAR_BETA,
        )
        wrong_t = _paddle_to_torch(wrong, z.device)
        assert not torch.allclose(wrong_t[0], want_out, rtol=1e-3), (
            "interleaved-fed PaddleFleet accidentally matched the oracle; the "
            "hand case is too symmetric to detect a layout bug"
        )
    if verbose:
        print("[situ_reference] _hand_check OK "
              f"(paddlefleet cross-check: {'yes @ ' + paddlefleet_source() if ok else 'SKIPPED - ' + why})")


if __name__ == "__main__":  # pragma: no cover
    _hand_check()
