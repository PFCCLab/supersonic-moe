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
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_for_fused_dgated,
    precompute_weight_fp8_for_fused_gated,
    precompute_weight_fp8_warmup,
    quantize_and_pack_activation,
)
from .gemm_interface import gemm_dgated, gemm_gated
from ._legacy.sgl_mxfp8_gemm import clear_sgl_weight_cache
from .triton_blockscaled_gemm import clear_raw_weight_cache
