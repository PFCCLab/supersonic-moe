from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(scope="module")
def proxy_env():
    pytest.importorskip("paddle")
    env = os.environ.copy()
    env["SONIC_MOE_DISPATCH_TEST_FILE"] = __file__
    return env


def _run_proxy_case(proxy_env, case: str) -> dict:
    code = textwrap.dedent(
        """
        import importlib.util
        import json
        import paddle

        paddle.enable_compat()
        spec = importlib.util.spec_from_file_location(
            "dispatch_isolation_case", __import__("os").environ["SONIC_MOE_DISPATCH_TEST_FILE"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._PROXY_CASES[__import__("os").environ["SONIC_MOE_DISPATCH_CASE"]]()
        print(json.dumps({"ok": True}))
        """
    )
    env = proxy_env.copy()
    env["SONIC_MOE_DISPATCH_CASE"] = case
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=120
    )
    if result.returncode != 0:
        pytest.fail(f"proxy subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _fake_native_carrier():
    class Invocation:
        invocation_id = 1
        backend = "hopper_native"

    carrier = type("Carrier", (), {})()
    carrier.invocation_id = 1
    carrier.metadata = {
        "invocation": Invocation(),
        "group_size": 32,
        "block_size": 32,
    }
    carrier.data = object()
    carrier.scales = object()
    carrier.materialize = lambda: carrier
    return carrier


def _case_sm100():
    import importlib

    importlib.import_module("torch")
    wgrad = importlib.import_module("sonicmoe.quack_utils.bf16_wgrad_gemm")
    calls = []
    original = lambda *args, **kwargs: calls.append((args, kwargs)) or "compiled"
    args = (type("Tensor", (), {"device": "cuda"})(), object(), object())
    kwargs = {"dynamic_scheduler": True}

    wgrad.get_device_capacity = lambda _device: (10, 0)
    wgrad.get_sm90_carrier = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("carrier registry accessed")
    )
    wgrad.unique_retained_sm90_invocation = lambda: (_ for _ in ()).throw(
        AssertionError("carrier invocation registry accessed")
    )

    qgi = importlib.import_module("quack.gemm_interface")
    result = qgi._run_gemm_pre_dispatch("gemm_out", original, *args, **kwargs)
    assert result == "compiled"
    assert calls == [(args, kwargs)]


def _case_gated():
    import importlib

    gated = importlib.import_module("sonicmoe.quack_utils.gemm_gated")
    a_carrier, b_carrier = _fake_native_carrier(), _fake_native_carrier()
    a_carrier.materialize = lambda: (_ for _ in ()).throw(
        AssertionError("capability checked too late")
    )
    carriers = iter((a_carrier, b_carrier))
    gated.get_sm90_carrier = lambda *_a, **_k: next(carriers)
    gated.is_sm90_native_active = lambda: True

    try:
        gated._try_sm90_native_gated(
            object(), object(), object(), None, object(), activation="geglu",
            cu_seqlens_m=None, A_idx=None, rowvec_bias=None, colvec_bias=None,
            z_scale_out=None, postact_scale_out=None, swiglu_clamp_value=0.0,
            postact_bf16_trunc=False,
        )
    except NotImplementedError as exc:
        assert "supports only swiglu" in str(exc)
    else:
        raise AssertionError("unsupported activation was accepted")


def _case_dgated():
    import importlib

    dgated = importlib.import_module("sonicmoe.quack_utils.gemm_dgated")
    a_carrier, b_carrier = _fake_native_carrier(), _fake_native_carrier()
    a_carrier.materialize = lambda: (_ for _ in ()).throw(
        AssertionError("capability checked too late")
    )
    carriers = iter((a_carrier, b_carrier))
    dgated.get_sm90_carrier = lambda *_a, **_k: next(carriers)
    dgated.is_sm90_native_active = lambda: True

    try:
        dgated._try_sm90_native_dgated(
            object(), object(), object(), object(), object(), activation="swiglu",
            cu_seqlens_m=None, A_idx=None, colvec_scale=None, colvec_reduce=None,
            swiglu_clamp_value=1.0,
        )
    except NotImplementedError as exc:
        assert "does not support swiglu_clamp_value" in str(exc)
    else:
        raise AssertionError("unsupported clamp was accepted")


_PROXY_CASES = {"sm100": _case_sm100, "gated": _case_gated, "dgated": _case_dgated}


def test_sm100_quack_hook_bypasses_registry_and_preserves_original_call(proxy_env):
    assert _run_proxy_case(proxy_env, "sm100") == {"ok": True}


def test_sm90_gated_rejects_unsupported_before_materialization(proxy_env):
    assert _run_proxy_case(proxy_env, "gated") == {"ok": True}


def test_sm90_dgated_rejects_clamp_before_materialization(proxy_env):
    assert _run_proxy_case(proxy_env, "dgated") == {"ok": True}
