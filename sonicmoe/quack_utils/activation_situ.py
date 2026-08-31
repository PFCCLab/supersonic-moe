"""SiTU-GLU activation for the SonicMoE FP8 gated GEMM epilogues.

Math (matches PaddleFleet ``transformer/activations.py::situ_glu``)::

    situ(g, beta)  = beta * tanh(g / beta) * sigmoid(g)
    up_act(u, lb)  = lb * tanh(u / lb)          if lb is not None else u
    situ_glu       = situ(g, beta) * up_act(u, lb)

Backward::

    t  = tanh(g / beta);  s = sigmoid(g);  ga = beta * t * s
    d(ga)/dg = (1 - t^2) * s + ga * (1 - s)
    d(up_act)/du = 1 - tanh(u / lb)^2           if lb is not None else 1
    dgate = dout * up_act * d(ga)/dg
    dup   = dout * ga      * d(up_act)/du

Layout note: the SonicMoE epilogues hand us **interleaved** gate/up pairs
(``z[:, 0::2] = gate``, ``z[:, 1::2] = up``), while PaddleFleet's eager
reference splits with ``paddle.chunk``.  The element-wise math below is
layout agnostic; any reference oracle must do the de-interleave itself.

Cache-key contract
------------------
``beta`` / ``linear_beta`` are baked into the traced kernel as ``Constexpr``
values via ``functools.partial``, so they *must* participate in every JIT /
autotune cache key.  Rather than threading two new fields through six cache
layers, they are encoded into the ``activation`` string itself, which is
already part of all of them::

    "situ_glu:b=4.0:lb=25.0"          precise (expf / libdevice tanhf)
    "situ_glu:b=4.0:lb=none"          no up-projection clamp
    "situ_glu:b=4.0:lb=25.0:approx"   MUFU.TANH + tanh-based sigmoid

Use :func:`encode_situ_activation` to build the string and
:func:`resolve_gate_fn` / :func:`resolve_dgate_fn` to turn it back into the
epilogue callables.
"""

from functools import partial
from typing import Callable, Optional, Tuple

import math

import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op

import quack.activation as _qact

F32_or_F32x2 = Float32 | Tuple[Float32, Float32]

# a - b, elementwise on a packed f32x2 pair.
_sub_packed = _qact.sub_packed_f32x2

_ONE_X2 = (1.0, 1.0)


# ---------------------------------------------------------------------------
# tanh / sigmoid primitives
# ---------------------------------------------------------------------------


@dsl_user_op
def tanh_precise(x: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE-ish ``tanhf`` (libdevice), i.e. NOT ``tanh.approx.f32``/MUFU.TANH."""
    return cute.math.tanh(Float32(x), fastmath=False, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


@dsl_user_op
def situ_glu(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    precise: bool = True,
    *,
    loc=None,
    ip=None,
) -> F32_or_F32x2:
    """``situ(x, beta) * up_act(y, linear_beta)``.

    ``beta`` / ``linear_beta`` / ``precise`` are trace-time Python constants.
    """
    tanh_fn = tanh_precise if const_expr(precise) else _qact.tanh
    sigmoid_fn = _qact.sigmoid_precise if const_expr(precise) else _qact.sigmoid
    inv_beta = 1.0 / beta

    if const_expr(not isinstance(x, tuple)):
        t = tanh_fn(x * inv_beta)
        s = sigmoid_fn(x)
        gate_act = (beta * t) * s
        if const_expr(linear_beta is None):
            up_act = y
        else:
            up_act = linear_beta * tanh_fn(y * (1.0 / linear_beta))
        return gate_act * up_act
    else:
        t0 = tanh_fn(x[0] * inv_beta)
        t1 = tanh_fn(x[1] * inv_beta)
        s = sigmoid_fn(x)
        # gate_act = beta * t * s
        gate_act = cute.arch.mul_packed_f32x2(
            cute.arch.mul_packed_f32x2((t0, t1), s), (beta, beta)
        )
        if const_expr(linear_beta is None):
            up_act = y
        else:
            inv_lb = 1.0 / linear_beta
            yb = cute.arch.mul_packed_f32x2(y, (inv_lb, inv_lb))
            up_act = cute.arch.mul_packed_f32x2(
                (tanh_fn(yb[0]), tanh_fn(yb[1])), (linear_beta, linear_beta)
            )
        return cute.arch.mul_packed_f32x2(gate_act, up_act)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@dsl_user_op
def dsitu_glu(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    dout: F32_or_F32x2,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    precise: bool = True,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    """Returns ``(dx, dy, out)`` — the contract ``act_bwd_fn`` must satisfy.

    ``out`` is the forward activation, consumed by the epilogue's
    ``ColVecReduce`` to build the router-score gradient, so it must be
    bit-comparable to what :func:`situ_glu` produces.
    """
    tanh_fn = tanh_precise if const_expr(precise) else _qact.tanh
    sigmoid_fn = _qact.sigmoid_precise if const_expr(precise) else _qact.sigmoid
    inv_beta = 1.0 / beta

    if const_expr(not isinstance(x, tuple)):
        t = tanh_fn(x * inv_beta)
        s = sigmoid_fn(x)
        gate_act = (beta * t) * s
        # d(gate_act)/dx = (1 - t^2) * s + gate_act * (1 - s)
        gate_grad = (1.0 - t * t) * s + gate_act * (1.0 - s)
        if const_expr(linear_beta is None):
            up_act = y
            dy = dout * gate_act
        else:
            tu = tanh_fn(y * (1.0 / linear_beta))
            up_act = linear_beta * tu
            dy = (dout * gate_act) * (1.0 - tu * tu)
        dx = (dout * up_act) * gate_grad
        return dx, dy, gate_act * up_act
    else:
        t = (tanh_fn(x[0] * inv_beta), tanh_fn(x[1] * inv_beta))
        s = sigmoid_fn(x)
        t_s = cute.arch.mul_packed_f32x2(t, s)
        gate_act = cute.arch.mul_packed_f32x2(t_s, (beta, beta))
        one_minus_t2 = _sub_packed(_ONE_X2, cute.arch.mul_packed_f32x2(t, t))
        one_minus_s = _sub_packed(_ONE_X2, s)
        # gate_grad = (1 - t^2) * s + gate_act * (1 - s)
        gate_grad = cute.arch.fma_packed_f32x2(
            gate_act, one_minus_s, cute.arch.mul_packed_f32x2(one_minus_t2, s)
        )
        dout_gate = cute.arch.mul_packed_f32x2(dout, gate_act)
        if const_expr(linear_beta is None):
            up_act = y
            dy = dout_gate
        else:
            inv_lb = 1.0 / linear_beta
            yb = cute.arch.mul_packed_f32x2(y, (inv_lb, inv_lb))
            tu = (tanh_fn(yb[0]), tanh_fn(yb[1]))
            up_act = cute.arch.mul_packed_f32x2(tu, (linear_beta, linear_beta))
            up_grad = _sub_packed(_ONE_X2, cute.arch.mul_packed_f32x2(tu, tu))
            dy = cute.arch.mul_packed_f32x2(dout_gate, up_grad)
        dx = cute.arch.mul_packed_f32x2(
            cute.arch.mul_packed_f32x2(dout, up_act), gate_grad
        )
        out = cute.arch.mul_packed_f32x2(gate_act, up_act)
        return dx, dy, out


# ---------------------------------------------------------------------------
# activation-string encoding / resolution
# ---------------------------------------------------------------------------

SITU_GLU_PREFIX = "situ_glu"

# Defaults from the ernielite layer43 model_config.
DEFAULT_SITU_BETA = 4.0
DEFAULT_SITU_LINEAR_BETA = 25.0


def _fmt(v: Optional[float]) -> str:
    # repr() of a float round-trips exactly, so the key is lossless.
    return "none" if v is None else repr(float(v))


def _check_beta(name: str, value) -> None:
    # Same contract as PaddleFleet transformer/activations.py::situ_glu, so an
    # invalid model config fails identically on both paths.
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(
            f"situ_glu {name} must be a positive finite value, but got {value!r}."
        )


def encode_situ_activation(
    beta: float = DEFAULT_SITU_BETA,
    linear_beta: Optional[float] = DEFAULT_SITU_LINEAR_BETA,
    precise: bool = True,
) -> str:
    """Build the cache-safe activation string for a SiTU-GLU configuration."""
    _check_beta("beta", beta)
    if linear_beta is not None:
        _check_beta("linear_beta", linear_beta)
    key = f"{SITU_GLU_PREFIX}:b={_fmt(beta)}:lb={_fmt(linear_beta)}"
    if not precise:
        key += ":approx"
    return key


def is_situ_activation(activation) -> bool:
    return isinstance(activation, str) and (
        activation == SITU_GLU_PREFIX or activation.startswith(SITU_GLU_PREFIX + ":")
    )


def parse_situ_activation(activation) -> Optional[Tuple[float, Optional[float], bool]]:
    """``"situ_glu:b=4.0:lb=25.0"`` -> ``(4.0, 25.0, True)``; ``None`` if not SiTU."""
    if not is_situ_activation(activation):
        return None
    beta = DEFAULT_SITU_BETA
    linear_beta = DEFAULT_SITU_LINEAR_BETA
    precise = True
    for field in activation.split(":")[1:]:
        if field == "approx":
            precise = False
        elif field == "precise":
            precise = True
        elif field.startswith("b="):
            beta = float(field[2:])
        elif field.startswith("lb="):
            raw = field[3:]
            linear_beta = None if raw == "none" else float(raw)
        else:
            raise ValueError(f"unrecognized situ_glu activation field {field!r} in {activation!r}")
    if beta is None or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"situ_glu beta must be positive and finite, got {activation!r}")
    if linear_beta is not None and (not math.isfinite(linear_beta) or linear_beta <= 0.0):
        raise ValueError(f"situ_glu linear_beta must be positive and finite, got {activation!r}")
    return beta, linear_beta, precise


def situ_gate_fn(activation: str) -> Callable:
    """Forward epilogue callable ``act_fn(gate, up) -> postact``."""
    beta, linear_beta, precise = parse_situ_activation(activation)
    return partial(situ_glu, beta=beta, linear_beta=linear_beta, precise=precise)


def situ_dgate_fn(activation: str) -> Callable:
    """Backward epilogue callable ``act_bwd_fn(gate, up, dout) -> (dx, dy, out)``."""
    beta, linear_beta, precise = parse_situ_activation(activation)
    return partial(dsitu_glu, beta=beta, linear_beta=linear_beta, precise=precise)


def resolve_gate_fn(activation, gate_fn_map: dict) -> Callable:
    """Look up an activation string in ``gate_fn_map``, honouring ``situ_glu:...``."""
    if is_situ_activation(activation):
        return situ_gate_fn(activation)
    if activation not in gate_fn_map:
        raise AssertionError(f"Unsupported activation {activation}")
    return gate_fn_map[activation]


def resolve_dgate_fn(activation, dgate_fn_map: dict) -> Callable:
    """Look up an activation string in ``dgate_fn_map``, honouring ``situ_glu:...``."""
    if is_situ_activation(activation):
        return situ_dgate_fn(activation)
    if activation not in dgate_fn_map:
        raise AssertionError(f"Unsupported activation {activation}")
    return dgate_fn_map[activation]


def is_supported_activation(activation, fn_map: dict) -> bool:
    return is_situ_activation(activation) or activation in fn_map
