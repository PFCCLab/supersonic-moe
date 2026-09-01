"""The SiTU betas must survive the warmup boundary.

``beta`` / ``linear_beta`` are traced into the kernel as compile-time constants,
so a different beta is a different kernel.  Everything downstream of
``_gemm_activation_name`` already keys on the encoded descriptor
(``"situ_glu:b=4.0:lb=25.0"``), but the *warmup* path has two places where a
bare ``ActivationType.SITU_GLU`` used to lose them:

1. ``cache_manager.warmup_signature`` only sees what the caller hands it, so a
   bare enum recorded the sentinel as plain ``"situ_glu"`` — a beta=4 warmup then
   reported warm for a beta=8 run and warmup was skipped, leaving the first real
   step to compile on the critical path.
2. ``jit_warmup.warmup_jit_parallel`` forks subprocesses that inherit the
   environment but *not* an in-process ``SonicMoEConfig``, so a bare enum made
   the child compile the default betas while the parent marked its own warm —
   a genuinely wrong cache, not just a bad message.

``jit_warmup.resolve_warmup_activation`` closes both by resolving the bare form
at the warmup boundary; ``cache_manager._normalize_activation`` rejects the bare
form as a backstop.  No GPU is needed for any of this.
"""
import json

import pytest

from sonicmoe.cache_manager import (
    _normalize_activation,
    _warmup_sentinel_path,
    is_warm,
    mark_warm,
    setup_cache,
    warmup_signature,
)
from sonicmoe.config import SonicMoEConfig
from sonicmoe.enums import ActivationType
from sonicmoe.jit_warmup import resolve_warmup_activation
from sonicmoe.quack_utils.activation_situ import encode_situ_activation

B4 = encode_situ_activation(4.0, 25.0)
B8 = encode_situ_activation(8.0, 25.0)

E, H, I = 8, 2048, 256


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """An isolated cache root that ``is_warm``'s file-count guard accepts."""
    monkeypatch.setenv("SONIC_MOE_CACHE_DIR", str(tmp_path))
    setup_cache(str(tmp_path))
    # is_warm() requires min_triton_files=1 as a guard against a sentinel
    # surviving `rm -rf .jit_cache/triton`.
    (tmp_path / "triton").mkdir(parents=True, exist_ok=True)
    (tmp_path / "triton" / "kernel.cubin").write_bytes(b"")
    return tmp_path


# ---------------------------------------------------------------------------
# 1. The bare form is refused rather than silently normalised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bare", ["situ_glu", ActivationType.SITU_GLU])
def test_normalize_rejects_bare_situ(bare):
    with pytest.raises(ValueError, match="carries no beta"):
        _normalize_activation(bare)


def test_normalize_keeps_encoded_verbatim():
    assert _normalize_activation(B4) == B4
    assert _normalize_activation(None) == "swiglu"
    assert _normalize_activation(ActivationType.SWIGLU) == "swiglu"


# ---------------------------------------------------------------------------
# 2. resolve_warmup_activation fills the betas in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bare", ["situ_glu", ActivationType.SITU_GLU])
def test_resolve_expands_bare_from_active_config(bare):
    with SonicMoEConfig(situ_beta=8.0, situ_linear_beta=25.0).activate():
        assert resolve_warmup_activation(bare) == B8
    with SonicMoEConfig(situ_beta=4.0, situ_linear_beta=25.0).activate():
        assert resolve_warmup_activation(bare) == B4


def test_resolve_reads_env_when_no_active_config(monkeypatch):
    monkeypatch.setenv("SONIC_MOE_SITU_BETA", "8.0")
    monkeypatch.setenv("SONIC_MOE_SITU_LINEAR_BETA", "25.0")
    assert resolve_warmup_activation(ActivationType.SITU_GLU) == B8


def test_resolve_canonicalises_encoded_and_passes_others_through():
    # Already encoded: idempotent, and re-encoded so ``lb=none`` etc. is
    # spelled one way only.
    assert resolve_warmup_activation(B4) == B4
    assert resolve_warmup_activation("situ_glu:b=4.0:lb=none") == encode_situ_activation(
        4.0, None
    )
    assert resolve_warmup_activation(ActivationType.SWIGLU) is ActivationType.SWIGLU
    assert resolve_warmup_activation("swiglu") == "swiglu"
    assert resolve_warmup_activation(None) is None


def test_resolve_rejects_malformed_descriptor():
    with pytest.raises(ValueError):
        resolve_warmup_activation("situ_glu:b=0.0:lb=25.0")
    with pytest.raises(ValueError):
        resolve_warmup_activation("situ_glu:bogus=1")


# ---------------------------------------------------------------------------
# 3. The sentinel actually discriminates on beta
# ---------------------------------------------------------------------------


def test_signature_discriminates_beta():
    s4 = warmup_signature(E, H, I, True, B4)
    s8 = warmup_signature(E, H, I, True, B8)
    assert s4["activation"] == B4 and s8["activation"] == B8
    assert s4 != s8
    # and is stable for the same descriptor
    assert warmup_signature(E, H, I, True, B4) == s4


def test_marking_beta4_warm_does_not_warm_beta8(cache_root):
    mark_warm(E, H, I, True, B4)
    assert json.loads(_warmup_sentinel_path().read_text())["activation"] == B4
    assert is_warm(E, H, I, True, B4)
    assert not is_warm(E, H, I, True, B8)
    # ... nor SwiGLU, nor the other way round
    assert not is_warm(E, H, I, True, ActivationType.SWIGLU)
    assert not is_warm(E, H, I, True, None)


def test_bare_enum_would_have_aliased_without_the_fix(cache_root):
    """The regression this file exists for.

    Both betas resolve through ``resolve_warmup_activation`` before touching the
    sentinel, so beta=8 must not see a beta=4 warmup as warm.  Were the bare
    enum passed straight through (the old behaviour) both calls would normalise
    to ``"situ_glu"`` and this would pass warm.
    """
    with SonicMoEConfig(situ_beta=4.0, situ_linear_beta=25.0).activate():
        act4 = resolve_warmup_activation(ActivationType.SITU_GLU)
        mark_warm(E, H, I, True, act4)
        assert is_warm(E, H, I, True, act4)
    with SonicMoEConfig(situ_beta=8.0, situ_linear_beta=25.0).activate():
        act8 = resolve_warmup_activation(ActivationType.SITU_GLU)
        assert act8 != act4
        assert not is_warm(E, H, I, True, act8)


def test_linear_beta_none_is_a_distinct_signature(cache_root):
    lb_none = encode_situ_activation(4.0, None)
    mark_warm(E, H, I, True, B4)
    assert not is_warm(E, H, I, True, lb_none)


def test_approx_is_a_distinct_signature(cache_root):
    approx = encode_situ_activation(4.0, 25.0, precise=False)
    mark_warm(E, H, I, True, B4)
    assert approx != B4
    assert not is_warm(E, H, I, True, approx)


# ---------------------------------------------------------------------------
# 4. What the parallel-warmup subprocess would be handed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bare", ["situ_glu", ActivationType.SITU_GLU])
def test_subprocess_arg_carries_betas(bare):
    """Mirrors the ``_act_arg`` line in ``warmup_jit_parallel``.

    The child re-coerces a plain string, so what matters is that the betas are
    in the string at all: the child inherits ``os.environ`` but not the active
    ``SonicMoEConfig``, so a bare ``"situ_glu"`` there would silently compile
    ``DEFAULT_SITU_BETA``.
    """
    with SonicMoEConfig(situ_beta=8.0, situ_linear_beta=25.0).activate():
        resolved = resolve_warmup_activation(bare)
        act_arg = repr(getattr(resolved, "value", resolved))
    assert act_arg == repr(B8)
    assert "b=8.0" in act_arg
