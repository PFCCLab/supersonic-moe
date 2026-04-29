"""Quack ↔ Paddle distributed compatibility shims.

Two issues are patched here, both auto-installed at ``import sonicmoe``:

1. ``quack.autotuner._gpu_warmup`` calls
   ``torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)`` on every
   autotune cache miss. Under ``paddle.compat.enable_torch_proxy()`` this is
   translated to ``paddle.randn(... device=CUDAPlace(0))``. In multi-rank /
   multi-machine training each process only initialises the
   ``DeviceContext`` for its own rank's GPU. If the ambiguous ``"cuda"``
   string resolves to ``CUDAPlace(0)`` on a non-rank-0 process, the
   ``DeviceContextPool::Get`` lookup fails with::

       set the correct device id if you use Executor.

   The crash is observed only on autotune cache miss. Production typically
   ships with a pre-warmed disk cache, but a single missed shape (e.g. an
   uneven token distribution at end of epoch) is enough to take down the
   whole job. Quack itself does not need a thermal warmup for our
   workloads — autotune is run once per shape and the result cached. We
   therefore neutralise ``_gpu_warmup`` to a true no-op.

2. ``Autotuner.benchmark`` body invokes ``_gpu_warmup`` *before* it knows
   which device the inputs live on. To be defensive against any future
   utility that allocates with ambiguous ``"cuda"``, we also wrap the
   tuned-call entrypoint to ensure ``paddle.device.set_device(...)`` matches
   the input tensor's place at call time.

Opt-out via ``SONIC_MOE_NO_QUACK_COMPAT_PATCH=1``.
"""
from __future__ import annotations

import os


def _install_gpu_warmup_noop() -> bool:
    """Override ``quack.autotuner._gpu_warmup`` to a no-op."""
    try:
        from quack import autotuner as _qa  # type: ignore
    except Exception:
        return False

    if getattr(_qa, "_sonic_moe_warmup_patched", False):
        return True

    def _noop(*_args, **_kwargs):
        return None

    _qa._sonic_moe_orig_gpu_warmup = _qa._gpu_warmup
    _qa._gpu_warmup = _noop
    _qa._sonic_moe_warmup_patched = True
    return True


def _install_paddle_torch_proxy_blockers() -> bool:
    """Tell paddle's torch-compat proxy NOT to intercept ``torch.utils.hipify``,
    AND eagerly import it so the ``from .hipify import hipify_python`` lookup
    inside ``torch.utils.cpp_extension._jit_compile`` always hits ``sys.modules``.

    Bug reproducer (paddle ≤ 3.x compat-proxy):

        $ python -c "import paddle; from torch.utils.hipify import hipify_python"
        ModuleNotFoundError: No module named 'paddle.utils.hipify'

    ``torch.utils.cpp_extension._jit_compile`` always does
    ``from .hipify import hipify_python`` which Python resolves as
    ``torch.utils.hipify``. Paddle's ``TorchProxyMetaFinder`` is on
    ``sys.meta_path`` and tries to redirect every ``torch.*`` import to the
    matching ``paddle.*`` name; ``paddle.utils.hipify`` doesn't exist, so any
    sonicmoe op that goes through the cpp-extension JIT path crashes —
    silently masking real CI signal as SKIP.

    Two layers of defense:
      1. ``extend_torch_proxy_blocked_modules({"torch.utils.hipify"})`` —
         the proxy will fall back to the real torch resolver for that name
         (and its submodules).
      2. Eager import — pre-populate ``sys.modules['torch.utils.hipify']``
         and ``sys.modules['torch.utils.hipify.hipify_python']`` so that
         downstream ``import`` statements never even reach the meta_path
         finders.
    """
    import sys

    ok = False
    try:
        import paddle.compat as _pc  # type: ignore
        extender = getattr(_pc, "extend_torch_proxy_blocked_modules", None)
        if extender is not None:
            # Block both the parent and the leaf module — paddle's TorchProxy
            # intercepts dotted lookups recursively so we must list every
            # name that might appear in `from torch.utils.hipify import X`.
            extender({
                "torch.utils.hipify",
                "torch.utils.hipify.hipify_python",
            })
            ok = True
    except Exception:
        pass

    # Eager-import so sys.modules has them before any subsequent lookup.
    for mod_name in (
        "torch.utils.hipify",
        "torch.utils.hipify.hipify_python",
    ):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
                ok = True
            except Exception:
                pass

    return ok


def install_quack_paddle_compat() -> bool:
    """Install all defensive patches. Returns True if at least one applied.

    Safe to call multiple times — each helper checks-and-installs idempotently.
    Callers that depend on the patches being live (e.g. JIT cpp_jit) should
    re-invoke this immediately before any paddle ``cpp_extension.load`` to
    cover the case where ``paddle.enable_compat()`` was called *after* the
    initial ``import sonicmoe``.
    """
    if os.environ.get("SONIC_MOE_NO_QUACK_COMPAT_PATCH"):
        return False
    a = _install_gpu_warmup_noop()
    b = _install_paddle_torch_proxy_blockers()
    return a or b


_PATCHED = install_quack_paddle_compat()
