"""The betas must be rejected in the domain the kernel computes in.

``beta`` and ``1.0 / beta`` are both traced into the SiTU epilogue as float32
``Constexpr`` values (``quack_utils/activation_situ.situ_glu``).  Validating them
in Python's float64 -- "positive and finite" -- therefore accepts values that
become ``inf`` or a denormal once traced, and the kernel then computes with them
and returns silently wrong numbers instead of raising.  Measured on sm_103 with
the old check in place:

    beta=3.5e38, 1e39     -> out = nan at g=0, inf at g=1
    beta=1e-39, 5e-324    -> out = nan at g=0, 0 at g=1
    linear_beta=1e39      -> forward nan/inf while ``dup`` still looks plausible

``g=0`` is what matters: zero-padded rows carry it, so a misconfigured beta
corrupts real production rows rather than only pathological inputs.

``situ_params.check_beta`` closes this by requiring both ``float32(value)`` and
``float32(1 / value)`` to be *normal*, and every entry point -- config field, env
var, ``lb=`` descriptor field -- now routes through it, so one bad value gets one
error message naming the layer that set it.  No GPU is needed for any of this.
"""

import struct

import pytest

from sonicmoe.config import SonicMoEConfig
from sonicmoe.quack_utils.activation_situ import (
    encode_situ_activation,
    parse_situ_activation,
)
from sonicmoe.situ_params import (
    BETA_MAX,
    BETA_MIN,
    DEFAULT_SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA,
    F32_MAX,
    F32_MIN_NORMAL,
    check_beta,
    is_linear_beta_disabled,
)


def _f32(x: float) -> float:
    """Round-trip through IEEE-754 binary32, exactly as the tracer does."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _is_normal_f32(x: float) -> bool:
    y = _f32(x)
    return y == y and y not in (float("inf"), float("-inf")) and abs(y) >= F32_MIN_NORMAL


# Values that the old float64 check let through and the kernel then mis-computed.
SILENTLY_BAD = [3.5e38, 1e39, 1e-39, 5e-324, F32_MAX]
# 1e38 is representable but its reciprocal (1e-38) is denormal, so it is out too.
DENORMAL_RECIPROCAL = [1e38, 2e38]
NOT_POSITIVE = [0.0, -0.0, -1.0, -4.0, float("nan"), float("inf"), float("-inf")]
GOOD = [DEFAULT_SITU_BETA, DEFAULT_SITU_LINEAR_BETA, 1.0, 2.0, 8.0, 0.5, 1e-3, 1e3]


# ---------------------------------------------------------------------------
# 1. The bound is the float32 arithmetic domain, not float64 finiteness
# ---------------------------------------------------------------------------


def test_bounds_are_exactly_the_normal_f32_reciprocal_window():
    """``BETA_MIN``/``BETA_MAX`` are derived, not hand-picked constants."""
    assert BETA_MIN == F32_MIN_NORMAL
    assert BETA_MAX == 1.0 / F32_MIN_NORMAL
    # The reciprocal is the binding constraint at both ends: 1/F32_MAX is looser
    # than F32_MIN_NORMAL, which is why the lower bound is not 1/F32_MAX.
    assert 1.0 / F32_MAX < F32_MIN_NORMAL
    for edge in (BETA_MIN, BETA_MAX):
        assert _is_normal_f32(edge)
        assert _is_normal_f32(1.0 / edge)


@pytest.mark.parametrize("name", ["beta", "linear_beta"])
@pytest.mark.parametrize("bad", SILENTLY_BAD + DENORMAL_RECIPROCAL)
def test_rejects_values_the_kernel_cannot_represent(name, bad):
    # The premise: at least one of the two f32 constants is not normal.
    assert not (_is_normal_f32(bad) and _is_normal_f32(1.0 / bad))
    with pytest.raises(ValueError, match="outside the range this kernel can represent"):
        check_beta(name, bad)


@pytest.mark.parametrize("name", ["beta", "linear_beta"])
@pytest.mark.parametrize("bad", NOT_POSITIVE)
def test_rejects_non_positive_and_nan(name, bad):
    with pytest.raises(ValueError):
        check_beta(name, bad)


@pytest.mark.parametrize("good", GOOD + [BETA_MIN, BETA_MAX])
def test_accepts_the_representable_window(good):
    assert check_beta("beta", good) == float(good)
    assert _is_normal_f32(good) and _is_normal_f32(1.0 / good)


def test_zero_hint_only_mentions_none_for_linear_beta():
    """0 is a plausible way to *mean* "no clamp"; say so, but only where true."""
    with pytest.raises(ValueError, match="write 'none', not 0"):
        check_beta("linear_beta", 0)
    with pytest.raises(ValueError) as ei:
        check_beta("beta", 0)
    assert "none" not in str(ei.value)


def test_bool_is_not_a_number():
    """``float(True) == 1.0`` would be a clamp nobody asked for."""
    for name in ("beta", "linear_beta"):
        with pytest.raises(ValueError, match="boolean"):
            check_beta(name, True)


def test_where_names_the_offending_layer():
    with pytest.raises(ValueError, match=r"\(SonicMoEConfig\.situ_beta\)"):
        check_beta("beta", 0.0, where="SonicMoEConfig.situ_beta")


# ---------------------------------------------------------------------------
# 2. Every entry point shares the one check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", SILENTLY_BAD + NOT_POSITIVE)
def test_config_field_rejects_at_the_config_layer(bad):
    """Not deferred to the first ``encode_situ_activation()`` call."""
    with pytest.raises(ValueError, match=r"SonicMoEConfig\.situ_beta"):
        SonicMoEConfig(situ_beta=bad).resolve_situ_beta()
    with pytest.raises(ValueError, match=r"SonicMoEConfig\.situ_linear_beta"):
        SonicMoEConfig(situ_linear_beta=bad).resolve_situ_linear_beta()


@pytest.mark.parametrize("bad", ["1e39", "0", "-1", "nan", "abc", ""])
def test_env_var_rejects_with_its_own_name(bad, monkeypatch):
    monkeypatch.setenv("SONIC_MOE_SITU_BETA", bad)
    if bad == "":
        # An empty env var is "unset", per the file-wide convention.
        assert SonicMoEConfig().resolve_situ_beta() == DEFAULT_SITU_BETA
        return
    with pytest.raises(ValueError, match=r"\$SONIC_MOE_SITU_BETA"):
        SonicMoEConfig().resolve_situ_beta()


def test_env_zero_says_how_to_disable_the_clamp(monkeypatch):
    """The review's complaint: ``=0`` gave no hint that ``none`` exists."""
    monkeypatch.setenv("SONIC_MOE_SITU_LINEAR_BETA", "0")
    with pytest.raises(ValueError, match="write 'none', not 0"):
        SonicMoEConfig().resolve_situ_linear_beta()


@pytest.mark.parametrize("bad", SILENTLY_BAD)
def test_descriptor_rejects_and_quotes_itself(bad):
    desc = f"situ_glu:b={bad!r}:lb=25.0"
    with pytest.raises(ValueError, match="outside the range"):
        parse_situ_activation(desc)
    with pytest.raises(ValueError, match="outside the range"):
        parse_situ_activation(f"situ_glu:b=4.0:lb={bad!r}")
    # encode_situ_activation is the other half of the round trip.
    with pytest.raises(ValueError):
        encode_situ_activation(bad, 25.0)
    with pytest.raises(ValueError):
        encode_situ_activation(4.0, bad)


@pytest.mark.parametrize("junk", ["abc", "0x10", "", "1e400", "inf", "-0.0"])
def test_descriptor_junk_is_a_clear_error(junk):
    with pytest.raises(ValueError):
        parse_situ_activation(f"situ_glu:b={junk}")


# ---------------------------------------------------------------------------
# 3. "no clamp" is spelled one way at every layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["none", "None", "NONE", "NoNe", "null", "off", "OFF", " none "])
def test_disabled_spellings_agree_across_layers(spelling, monkeypatch):
    """The review found config accepted ``"NoNe"`` while ``lb=`` wanted lowercase."""
    assert is_linear_beta_disabled(spelling)
    assert SonicMoEConfig(situ_linear_beta=spelling).resolve_situ_linear_beta() is None
    monkeypatch.setenv("SONIC_MOE_SITU_LINEAR_BETA", spelling)
    assert SonicMoEConfig().resolve_situ_linear_beta() is None
    monkeypatch.delenv("SONIC_MOE_SITU_LINEAR_BETA")
    beta, linear_beta, precise = parse_situ_activation(
        f"situ_glu:b=4.0:lb={spelling.strip()}"
    )
    assert linear_beta is None and beta == 4.0 and precise


def test_false_disables_but_true_is_refused():
    assert is_linear_beta_disabled(False)
    assert SonicMoEConfig(situ_linear_beta=False).resolve_situ_linear_beta() is None
    assert not is_linear_beta_disabled(True)
    with pytest.raises(ValueError, match="boolean"):
        SonicMoEConfig(situ_linear_beta=True).resolve_situ_linear_beta()


def test_none_means_unset_not_disabled():
    """``None`` is the dataclass "unset" sentinel, so it takes the default."""
    assert not is_linear_beta_disabled(None)
    cfg = SonicMoEConfig(situ_linear_beta=None)
    assert cfg.resolve_situ_linear_beta() == DEFAULT_SITU_LINEAR_BETA


def test_alternate_spellings_still_canonicalise_to_one_cache_key():
    """A descriptor spelled ``lb=OFF`` must not fork the cache key."""
    canonical = encode_situ_activation(4.0, None)
    for spelling in ("none", "NONE", "off", "null"):
        beta, linear_beta, precise = parse_situ_activation(f"situ_glu:b=4.0:lb={spelling}")
        assert encode_situ_activation(beta, linear_beta, precise) == canonical


# ---------------------------------------------------------------------------
# 4. Round trip stays lossless over the whole accepted window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", GOOD + [BETA_MIN, BETA_MAX])
@pytest.mark.parametrize("linear_beta", [None, 25.0, 1.0])
def test_round_trip_is_exact(beta, linear_beta):
    desc = encode_situ_activation(beta, linear_beta)
    got_beta, got_lb, precise = parse_situ_activation(desc)
    assert got_beta == float(beta)
    assert got_lb == (None if linear_beta is None else float(linear_beta))
    assert precise
    assert encode_situ_activation(got_beta, got_lb, precise) == desc
