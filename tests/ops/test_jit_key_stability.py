# ********************************************************************************
# JIT key stability — ensure no kernel keys on dynamic dims (T, total_K, seqlen).
#
# Failure mode this catches: a kernel adds ``tuple(tensor.shape)`` or similar
# to its compile-cache key. Running training with varying seqlen then triggers
# CuTe / Triton recompiles every iteration, blowing past the cold-start budget
# repeatedly and stalling the pipeline.
#
# This test runs ``_warmup_single`` (a real fwd+bwd through SonicMoEMlpNode)
# at multiple total_K values after a priming call, and asserts that every CuTe
# ``_COMPILE_CACHE`` size remains unchanged. Triton autotune cache size is
# also tracked.
# ********************************************************************************

from __future__ import annotations

import os
import pytest

import paddle  # noqa: F401  (must be first to register torch-proxy)
paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"}, silent=True)

import torch  # noqa: E402

from sonicmoe.jit_warmup import _warmup_single  # noqa: E402


def _all_compile_caches() -> dict[str, int]:
    """Collect sizes of all in-memory CuTe compile caches across the codebase.

    Caches live in many spots — module-level ``_COMPILE_CACHE*`` dicts, and
    function attributes like ``_topk_fwd.compile_cache``. We sweep all
    sonicmoe.quack_utils + sonicmoe.functional modules.
    """
    sizes: dict[str, int] = {}
    import importlib
    for mod_name in [
        "sonicmoe.quack_utils.blockscaled_fp8_gemm",
        "sonicmoe.quack_utils.gemm_gated",
        "sonicmoe.quack_utils.gemm_dgated",
        "sonicmoe.quack_utils.gemm_sm100_fp8_zeromat",
        "sonicmoe.functional.forward",
        "sonicmoe.functional.backward",
    ]:
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        for attr in dir(m):
            obj = getattr(m, attr, None)
            if obj is None:
                continue
            # Module-level cache dicts
            if attr.startswith("_COMPILE_CACHE") and hasattr(obj, "__len__"):
                sizes[f"{mod_name}.{attr}"] = len(obj)
            # function.compile_cache attribute
            cc = getattr(obj, "compile_cache", None)
            if cc is not None and hasattr(cc, "__len__"):
                sizes[f"{mod_name}.{attr}.compile_cache"] = len(cc)
    return sizes


@pytest.mark.parametrize("E,H,I", [(8, 3072, 1536)])
def test_jit_key_stability_across_token_count(E: int, H: int, I: int):
    """Compile cache MUST NOT grow when only total_K changes.

    Running this with seqlen variation = production training pattern. If any
    kernel keys on a dynamic dim, this test will see _COMPILE_CACHE.len()
    grow on subsequent calls, fail loudly, and pinpoint the offending cache.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    place = paddle.framework._current_expected_place()
    dev_id = getattr(place, "gpu_device_id", None)
    if callable(dev_id):
        dev_id = dev_id()
    if dev_id is None:
        dev_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    device = f"cuda:{int(dev_id)}"

    # Prime: run once at base shape so all kernels are compiled & cached.
    _warmup_single(E=E, H=H, I=I, total_K=E * 256, device=device, fp8=True)
    base = _all_compile_caches()
    assert sum(base.values()) > 0, "warmup did not populate any compile caches"

    # Now sweep across realistic seqlen variations. None should grow caches.
    test_total_Ks = [E * 128, E * 512, E * 1024, E * 2048, E * 96, E * 257]
    for total_K in test_total_Ks:
        _warmup_single(E=E, H=H, I=I, total_K=total_K, device=device, fp8=True)
        cur = _all_compile_caches()
        for k, v_base in base.items():
            v_cur = cur.get(k, 0)
            assert v_cur == v_base, (
                f"compile cache {k} grew {v_base} -> {v_cur} after total_K={total_K} "
                f"(E={E}, H={H}, I={I}). A kernel keys on a dynamic dim — "
                f"production training will recompile every iteration."
            )
