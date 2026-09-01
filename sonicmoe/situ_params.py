"""SiTU-GLU scalar parameters and their validation. Pure Python, no imports.

This module exists because three layers need the same beta contract but cannot
share an import:

* ``quack_utils/activation_situ.py`` builds/parses the descriptor, and pulls in
  cutlass + quack at import time;
* ``config.py`` resolves ``situ_beta`` / ``situ_linear_beta`` and is imported by
  ``functional/fp8_config.py``, which must stay free of that dependency;
* ``cache_manager.py`` normalises warmup signatures and likewise avoids CuTe.

Before this module the defaults were duplicated in ``config.py`` under a "Keep
in sync" comment, and the range check lived only in ``activation_situ`` — so a
bad ``situ_beta`` set on the config object was not diagnosed until the first
``encode_situ_activation()`` call, far from the line that set it.

Why the range is narrower than "positive and finite"
---------------------------------------------------
``beta`` reaches the kernel as two ``Constexpr`` float32 values -- ``beta``
itself and ``1.0 / beta`` (``activation_situ.situ_glu``). Validating in Python's
float64 therefore accepts values that overflow or denormalise once traced, and
the kernel then computes with ``inf`` / ``0`` and returns silently wrong numbers
rather than raising. Measured on sm_103:

    beta=3.5e38, 1e39     -> out = nan / inf at g=0 and g=1
    beta=1e-39, 5e-324    -> out = nan / 0
    linear_beta=1e39      -> forward nan / inf (dup still looks plausible)

``g=0`` is the important one: it is what zero-padded rows carry, so a misconfigured
beta corrupts real production rows, not just pathological inputs. Requiring both
``float32(value)`` and ``float32(1 / value)`` to be *normal* floats turns all of
the above into a config-time ``ValueError``.
"""

from __future__ import annotations

# Defaults from the ernielite layer43 model_config.
DEFAULT_SITU_BETA = 4.0
DEFAULT_SITU_LINEAR_BETA = 25.0

# IEEE-754 binary32.
F32_MAX = 3.4028234663852886e38
F32_MIN_NORMAL = 1.1754943508222875e-38

# Requiring ``value`` and ``1 / value`` to both be normal float32 gives
# ``[F32_MIN_NORMAL, 1 / F32_MIN_NORMAL]``; the reciprocal is the binding
# constraint at both ends (1 / F32_MAX = 2.94e-39 is looser than F32_MIN_NORMAL).
BETA_MIN = F32_MIN_NORMAL           # 1.1754943508222875e-38
BETA_MAX = 1.0 / F32_MIN_NORMAL     # 8.507059173023462e+37

# Spellings that disable the up-projection clamp, accepted identically by the
# config field, the environment variable and the ``lb=`` descriptor field.
LINEAR_BETA_DISABLED = ("none", "null", "off")


def check_beta(name: str, value, *, where: str = "") -> float:
    """Validate one beta and return it as a ``float``.

    ``where`` is appended to the message so the caller can say which layer
    rejected it (config field, env var, or descriptor).
    """
    ctx = f" ({where})" if where else ""
    # bool is an int subclass, so ``float(True)`` would silently become 1.0 --
    # a plausible-looking beta that the user never asked for. ``False`` is
    # accepted only for ``linear_beta``, and is handled before this call.
    if isinstance(value, bool):
        raise ValueError(
            f"situ_glu {name}{ctx} must be a number, but got the boolean "
            f"{value!r}. To disable the up-projection clamp write "
            f"situ_linear_beta='none'; there is no boolean spelling for beta."
        )
    if value is None:
        raise ValueError(f"situ_glu {name}{ctx} must be a positive finite value, but got None.")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"situ_glu {name}{ctx} must be a number, but got {value!r}."
        ) from None
    # NaN fails every comparison below, so test it explicitly for a clear message.
    if v != v:
        raise ValueError(f"situ_glu {name}{ctx} must be a positive finite value, but got nan.")
    if v <= 0.0:
        hint = (
            " To disable the up-projection clamp write 'none', not 0."
            if name == "linear_beta"
            else ""
        )
        raise ValueError(
            f"situ_glu {name}{ctx} must be a positive finite value, but got {value!r}.{hint}"
        )
    if not (BETA_MIN <= v <= BETA_MAX):
        raise ValueError(
            f"situ_glu {name}{ctx} = {value!r} is outside the range this kernel can "
            f"represent, [{BETA_MIN!r}, {BETA_MAX!r}]. Both {name} and 1/{name} are "
            f"traced into the kernel as float32 constants, so a value outside this "
            f"range becomes inf or a denormal there and the kernel silently "
            f"returns nan/inf/0 -- including at g=0, i.e. on zero-padded rows."
        )
    return v


def is_linear_beta_disabled(value) -> bool:
    """``False`` / ``"none"`` / ``"null"`` / ``"off"`` (any case) disable the clamp.

    ``None`` is *not* included: every config field in ``SonicMoEConfig`` follows
    the "``None`` means unset" convention, so ``situ_linear_beta=None`` falls
    through to the built-in default.
    """
    if value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in LINEAR_BETA_DISABLED
