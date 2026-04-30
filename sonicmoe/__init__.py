# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import paddle
import inspect

# ── Persistent Triton autotune cache ──────────────────────────────────────────
# MUST run before any sonicmoe submodule (or upstream quack) executes its
# `@triton.autotune` decorators, because Autotuner.__init__ snapshots the
# `cache_results` flag at decoration time. Without this, every cold process
# re-runs ~30 s of autotune sweep for token_gather_sum_kernel alone.
from sonicmoe import _triton_autotune_persist as _triton_autotune_persist  # noqa: F401

# ── Triton ↔ Paddle stream compat ────────────────────────────────────────────
# Must run before any Triton kernel is launched. Triton binds
# ``get_current_stream`` to torch._C internals at import time, so without this
# patch every sonicmoe Triton kernel launches on the legacy NULL stream while
# Paddle's GEMMs run on Paddle's own compute stream — causing implicit
# cross-stream sync (perf tax) and producer/consumer race hazards.
from sonicmoe import _triton_stream_compat as _triton_stream_compat  # noqa: F401
from sonicmoe import _quack_compat as _quack_compat  # noqa: F401

# ── Paddle compat shims for missing torch.cuda internals ─────────────────────
# torch.random.manual_seed() calls torch.cuda._is_in_bad_fork() internally.
# Paddle compat doesn't implement it — safe to stub as always-False (never forked).
if not hasattr(paddle.cuda, '_is_in_bad_fork'):
    paddle.cuda._is_in_bad_fork = lambda: False

if not (hasattr(paddle.library.CustomOpDef, "__call__") and inspect.isfunction(paddle.library.CustomOpDef.__call__)):
    def __call__(self, *args, **kwargs):
        return getattr(getattr(paddle.ops, self._namespace), self._name)(*args, **kwargs)
    paddle.library.CustomOpDef.__call__ = __call__


def torch_compat_empty(*args, **kwargs):
    if "device" in kwargs and kwargs["device"] == "cuda":
        del kwargs["device"]
    return paddle.empty(*args, **kwargs)


def torch_compat_corrcoef(input):
    """Pearson correlation coefficient matrix — matches torch.corrcoef."""
    x = input.cast("float32") if input.dtype != paddle.float32 else input
    x = x - x.mean(axis=-1, keepdim=True)
    cov = x @ x.T / (x.shape[-1] - 1)
    stddev = cov.diag().sqrt()
    outer = stddev.unsqueeze(1) * stddev.unsqueeze(0)
    return cov / outer.clip(min=1e-12)


paddle.corrcoef = torch_compat_corrcoef


paddle.compat.proxy._extend_torch_proxy_overrides(
    {
        "torch.empty": paddle.compat.proxy.RawOverriddenAttribute(torch_compat_empty),
        "torch.corrcoef": paddle.compat.proxy.RawOverriddenAttribute(torch_compat_corrcoef),
    }
)


__version__ = "0.1.1"

from .count_cumsum import count_cumsum
from .enums import KernelBackendMoE
from .functional import (
    FP8ActivationDType,
    FP8Backend,
    apply_activation_fp8_protocol_cutely_fused,
    apply_preact_activation_fp8_protocol_cutely_fused,
    FP8Protocol,
    FP8ScaleEncoding,
    FP8ScaleGranularity,
    FP8Tensor,
    apply_activation_fp8_protocol,
    dequantize_activation_reference,
    enable_fp8,
    enable_quack_gemm,
    get_default_fp8_protocol,
    is_blackwell_device,
    moe_general_routing_inputs,
    moe_TC_softmax_topk_layer,
    quantize_activation_reference,
    validate_fp8_protocol,
    validate_fp8_runtime_support,
)
from .config import SonicMoEConfig, get_active_config, set_active_config
from .moe import MoE
from .quack_utils import make_blockscaled_grouped_reverse_scatter_idx, pack_blockscaled_1x32_scales
