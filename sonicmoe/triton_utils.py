import os
import sys

import paddle


def deterministic_autotune_enabled() -> bool:
    """Whether Triton autotuning must be replaced by a fixed config choice.

    ``@triton.autotune`` keeps the config with the lowest *measured* time, and
    for the reduction kernels the winning margin is often well below timing
    noise on a busy GPU. Different tile shapes give a different float32
    accumulation order, hence different output bits, so two processes running
    the same math can disagree in the low bits. That breaks bit-exact
    reproducibility across runs (e.g. the sharding-reshard CI case, which
    compares losses digit for digit).

    Opt in explicitly with ``SONIC_MOE_DETERMINISTIC_AUTOTUNE=1``; otherwise
    follow Paddle's global determinism switch.
    """
    value = os.environ.get("SONIC_MOE_DETERMINISTIC_AUTOTUNE")
    if value is not None:
        return value.lower() not in ("", "0", "false")
    return os.environ.get("FLAGS_cudnn_deterministic", "0").lower() in ("1", "true")

original_paddle_empty = paddle.empty


def torch_compat_empty(*args, **kwargs):
    if "device" in kwargs and kwargs["device"] == "cuda":
        del kwargs["device"]
    return original_paddle_empty(*args, **kwargs)


def swap_torch_guard(fn):
    def wrapped_fn(*args, **kwargs):
        if "torch" not in sys.modules:
            return fn(*args, **kwargs)
        torch_module = sys.modules["torch"]
        original_paddle_empty = paddle.empty
        sys.modules["torch"] = paddle
        paddle.empty = torch_compat_empty
        try:
            return fn(*args, **kwargs)
        finally:
            sys.modules["torch"] = torch_module
            paddle.empty = original_paddle_empty

    return wrapped_fn


def wrap_triton_kernel(triton_kernel):
    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            return swap_torch_guard(self.kernel[index])

        def __getattr__(self, name):
            return getattr(self.kernel, name)

    return WrappedTritonKernel(triton_kernel)
