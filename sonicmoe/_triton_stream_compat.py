"""Triton ↔ Paddle stream compatibility shim.

Triton's ``GPUDriver.get_current_stream`` is bound at import time to
``torch._C._cuda_getCurrentRawStream`` (see ``triton/backends/driver.py``).
That C-level call always returns *PyTorch's* current CUDA stream, which under
``paddle-torch-compat`` is the legacy default (NULL) stream — **not** Paddle's
compute stream.

If left unpatched, every Triton kernel (``_quantize_and_pack_kernel``,
``token_gather_sum_kernel``, …) launches on the NULL stream while Paddle's
GEMMs / cuBLAS / cuDNN ops run on Paddle's own stream. NULL-stream launches
have implicit cross-stream synchronization semantics, which both serializes
work AND creates correctness hazards: producer/consumer kernels on different
streams may execute in the wrong order with respect to the data they share.

This module overrides ``triton.runtime.driver.driver.active.get_current_stream``
to return Paddle's actual current compute-stream pointer when running under
Paddle. The patch is idempotent and opt-out via ``SONIC_MOE_NO_TRITON_STREAM_PATCH=1``.
"""
from __future__ import annotations

import os


def install_paddle_stream_compat() -> bool:
    """Install the Triton stream-compat patch. Returns True if installed."""
    if os.environ.get("SONIC_MOE_NO_TRITON_STREAM_PATCH"):
        return False

    try:
        from triton.runtime.driver import driver  # type: ignore
    except Exception:
        return False
    try:
        import paddle  # type: ignore
    except Exception:
        return False

    active = driver.active
    if getattr(active, "_sonic_moe_paddle_patched", False):
        return True

    _orig = active.get_current_stream

    def _paddle_aware_stream(device_idx: int = 0) -> int:
        try:
            s = paddle.device.current_stream()
            return s.stream_base.raw_stream
        except Exception:
            return _orig(device_idx)

    active.get_current_stream = _paddle_aware_stream
    active._sonic_moe_paddle_patched = True
    active._sonic_moe_paddle_orig_stream_fn = _orig
    return True


_PATCHED = install_paddle_stream_compat()
