# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

# ---------------------------------------------------------------------------
# Workaround for quack memory leak (quack <= 0.3.7):
#   quack.gemm_interface calls `get_device_capacity(A)` where A is a tensor.
#   `get_device_capacity` is @lru_cache, so each unique tensor object creates
#   a permanent cache entry that prevents GC.  We wrap the function to
#   normalise tensor args to `torch.device` *before* the cache lookup.
# ---------------------------------------------------------------------------
import torch as _torch
import quack.cute_dsl_utils as _cdu
import quack.gemm_interface as _qgi

_orig_gdc = _cdu.get_device_capacity          # the @lru_cache-wrapped fn


def _safe_get_device_capacity(device=None):
    if isinstance(device, _torch.Tensor):
        device = device.device
    return _orig_gdc(device)


_safe_get_device_capacity.cache_info = getattr(_orig_gdc, "cache_info", None)
_safe_get_device_capacity.cache_clear = getattr(_orig_gdc, "cache_clear", None)

_qgi.get_device_capacity = _safe_get_device_capacity
# ---------------------------------------------------------------------------

from .bf16_wgrad_gemm import (
    bf16_wgrad_gemm_varlen_k,
    bf16_wgrad_gemm_varlen_k_accumulate,
    bf16_wgrad_gemm_varlen_k_tma_add,
)
from .blockscaled_fp8_gemm import (
    blockscaled_fp8_gemm,
    blockscaled_fp8_gemm_grouped,
    blockscaled_fp8_gemm_varlen,
    blockscaled_fp8_weight_grad_gemm,
    blockscaled_fp8_weight_grad_gemm_fast,
    blockscaled_fp8_wgrad_varlen_k,
    clear_blockscaled_fp8_weight_cache,
    evict_fp8_weight_cache_entry,
    fast_gather_quantize_and_pack_activation,
    make_blockscaled_grouped_reverse_scatter_idx,
    pack_blockscaled_1x32_scales,
    prefetch_blockscaled_w2_fp8,
    precompute_weight_fp8,
    quantize_native_fp8_weights,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_for_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    precompute_weight_fp8_warmup,
    quantize_and_pack_activation,
    quantize_activation_blockscaled_fast,
)
from .gemm_interface import gemm_dgated, gemm_gated
# Importing bf16_wgrad_gemm above installs the exact-SM90 down/actgrad
# QuACK pre-dispatch handler.
from .sm90_fp8_runtime import (
    alias_sm90_carrier,
    attach_sm90_carrier,
    bind_sm90_payload,
    claim_pending_sm90_carrier,
    consume_sm90_payload,
    current_sm90_invocation,
    get_sm90_carrier,
    get_sm90_storage_carriers,
    has_sm90_native_carriers,
    is_sm90_native_active,
    is_sm90_native_carrier,
    register_sm90_carrier,
    release_deferred_wgrad,
    release_sm90_carrier,
    release_sm90_invocation,
    replace_with_sm90_anchor,
    retain_for_deferred_wgrad,
    retain_sm90_carrier,
    sm90_invocation_scope,
    sm90_registry_stats,
    use_sm90_invocation,
    wrap_deferred_sm90_call,
)
from ._legacy.sgl_mxfp8_gemm import clear_sgl_weight_cache
from .triton_blockscaled_gemm import clear_raw_weight_cache
