# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import partial
from typing import Literal, Optional, Tuple

import paddle
import torch
from quack.autotuner import AutotuneConfig, autotune
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import GemmConfig, get_all_configs
from quack.gemm_interface import default_config as quack_default_config, prune_invalid_gemm_configs as prune_invalid_gemm_configs_base
from torch import Tensor


def _custom_op_or_plain(name, **kwargs):
    """torch.library.custom_op that falls back to plain function if already registered."""
    def decorator(fn):
        try:
            return torch.library.custom_op(name, **kwargs)(fn)
        except ValueError:
            return fn
    return decorator


def _register_fake_safe(name):
    """torch.library.register_fake that silently skips if registration fails."""
    def decorator(fn):
        try:
            return torch.library.register_fake(name)(fn)
        except (ValueError, RuntimeError):
            return fn
    return decorator

from .gemm_dgated import gemm_dgated as gemm_dgated_sm90_sm100
from .gemm_gated import gemm_gated as gemm_gated_sm90_sm100


default_device_capacity = get_device_capacity(paddle.device("cuda"))


# SM103 (B300) backward dgated blockscaled GEMM: above this expert count we
# keep the conservative 1-CTA tile. cluster_m=2 gives no measured speedup at
# E=128 small-hidden (+1.4% regression) and E>64 is exactly the regime the
# original stability override targeted ("avoid 2CTA-M varlen gather on SM103";
# illegal-instruction fault under skewed expert load), which has NOT been
# re-qualified for skew at E>64 on the current toolchain.
_DGATED_2CTA_MAX_EXPERTS = 64


def default_config(device, num_experts: int = 1) -> GemmConfig:
    config = quack_default_config(device)
    cap = get_device_capacity(device)
    if cap[0] == 10 and cap[1] >= 3:
        # SM103 (B300) backward dgated blockscaled GEMM. cluster_m=2 enables
        # 2-CTA cooperative tiling (TE-style 2cta) which the prior cluster_m=1
        # override disabled.  Empirically (nsys GPU-projection, fresh proc/cfg):
        # cluster_m=2 is ~6-14% faster on the dgated kernel at production shapes
        # (E<=64, e.g. Qwen3-30B-A3B H2048 I1024 E64: 459->421us, -8.4%) and
        # bit-identical (rrmse=0; tile/cluster partition does not change the
        # K-reduction order) with zero extra memory. Verified finite+bit-
        # identical under 80%-skewed and extreme single-expert routing at E=64.
        # E>64 stays 1-CTA: no measured win there and it is the historical
        # skew-fault regime (see _DGATED_2CTA_MAX_EXPERTS above).
        cluster_m = 2 if num_experts <= _DGATED_2CTA_MAX_EXPERTS else 1
        # tile_m=256 + cluster_m=2 => true 2-CTA tcgen05 MMA (use_2cta_instrs).
        # Session-38 ncu (B300, E64 H2048 I1024 TK262144): -4.7% on the dgated
        # kernel vs tile_m=128 (1,111,392->1,059,136 ns base-clock; tensor-pipe
        # 38.77->40.74%), BIT-IDENTICAL dz/colvec/y1s (max_abs_diff=0; M-tiling
        # does not change the K-reduction) and compute-sanitizer memcheck-clean
        # (0 errors) under 80%-skew + single-dominant routing at E=64.  Gated to
        # the real dgated expert range [2, 64]: num_experts==1 is the weight-grad
        # caller (it cannot see the real E, so it stays tile_m=128 to avoid the
        # unvalidated E>64 true-2CTA regime), and E>64 stays tile_m=128 cm=1.
        tile_m = 256 if 2 <= num_experts <= _DGATED_2CTA_MAX_EXPERTS else 128
        return GemmConfig(
            tile_m=tile_m,
            tile_n=128,
            cluster_m=cluster_m,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=config.is_dynamic_persistent,
            max_swizzle_size=config.max_swizzle_size,
            device_capacity=config.device_capacity,
        )
    return config


def _uses_blockscaled_runtime(a_scales: Optional[Tensor], b_scales: Optional[Tensor]) -> bool:
    return a_scales is not None and b_scales is not None


def prune_invalid_gated_configs(configs, named_args: dict, **kwargs):
    kwargs = named_args | kwargs
    configs = prune_invalid_gemm_configs_base(configs, named_args, **kwargs)
    if kwargs.get("A_idx", None) is not None:
        configs = [conf for conf in configs if conf.kwargs["config"].cluster_n == 1]
    return configs


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs(default_device_capacity[0], "gated")],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gated_configs},
)
def gemm_gated_tuned(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    # (M, N) or (L, M, N) or (total_M, N) if varlen_m - None if not storing preact
    preact_out: Optional[Tensor],
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
    a_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for A
    b_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for B
    z_scale_out: Optional[Tensor] = None,  # epilogue quant scale output
    postact_scale_out: Optional[Tensor] = None,  # ISA-packed UE8M0 scales for postact (y1) quant
    swiglu_clamp_value: float = 0.0,
) -> None:
    if config is None:
        config = quack_default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    tile_count_semaphore = torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    gemm_gated_sm90_sm100(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        persistent=True,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        a_scales=a_scales,
        b_scales=b_scales,
        z_scale_out=z_scale_out,
        postact_scale_out=postact_scale_out,
        swiglu_clamp_value=swiglu_clamp_value,
    )


def prune_invalid_gemm_dgated_configs(configs, named_args: dict, **kwargs):
    kwargs = named_args | kwargs
    configs = prune_invalid_gated_configs(configs, named_args, **kwargs)
    # if there's colvec_scale or colvec_reduce, don't swap_AB
    if kwargs.get("colvec_scale", None) is not None or kwargs.get("colvec_reduce", False):
        configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    return configs


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs(default_device_capacity[0], "dgated")],
    key=["activation", "colvec_reduce", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_dgated_configs},
)
def gemm_dgated_tuned(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    dx_out: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    # whether to do colvec reduction, returning (M,) or (L, M) or (total_M) if varlen_m
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    config: Optional[GemmConfig] = None,
    a_scales: Optional[Tensor] = None,
    b_scales: Optional[Tensor] = None,
    swiglu_clamp_value: float = 0.0,
) -> Optional[Tensor]:
    if config is None:
        # B is (K, N) or (L, K, N); L == num_experts for the grouped path.
        config = default_config(A.device, num_experts=(B.shape[0] if B.ndim == 3 else 1))
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    og_ndim_2 = A.ndim == 2 and not varlen_m
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
    if PreAct.ndim == 2 and not varlen_m:
        PreAct = PreAct.unsqueeze(0)  # (1, M, 2*N)
    if dx_out.ndim == 2 and not varlen_m:
        D = dx_out.unsqueeze(0)
    else:
        D = dx_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if colvec_scale is not None and colvec_scale.ndim == 1 and not varlen_m:
        colvec_scale = colvec_scale.unsqueeze(0)  # (L, N)
    if colvec_scale is not None:
        assert not config.swap_ab, "colvec_scale not supported with swap_ab"
    if colvec_reduce:
        tile_n = config.tile_n
        shape_n = (B.shape[-2] + tile_n - 1) // tile_n
        if varlen_m:
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            colvec_shape = (total_m, shape_n)
        else:
            colvec_shape = (A.shape[0], A.shape[-2], shape_n)
        colvec_reduce_partial = torch.empty(colvec_shape, dtype=torch.float32, device=A.device)
    else:
        colvec_reduce_partial = None
    tile_count_semaphore = torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    gemm_dgated_sm90_sm100(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        D if not config.swap_ab else D.mT,
        PreAct if not config.swap_ab else PreAct.mT,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        persistent=True,
        max_swizzle_size=config.max_swizzle_size,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce_partial,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        a_scales=a_scales,
        b_scales=b_scales,
        swiglu_clamp_value=swiglu_clamp_value,
    )
    if colvec_reduce:
        colvec_reduce_final = colvec_reduce_partial.sum(dim=-1)
        if og_ndim_2:
            colvec_reduce_final = colvec_reduce_final.squeeze(0)
    else:
        colvec_reduce_final = None
    return colvec_reduce_final


def gemm_gated(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    preact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Optional[Tensor] = None,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    store_preact: bool = True,
    dynamic_scheduler: bool = False,
    a_scales: Optional[Tensor] = None,
    b_scales: Optional[Tensor] = None,
    tuned: bool = True,
    z_scale_out: Optional[Tensor] = None,
    postact_scale_out: Optional[Tensor] = None,
    swiglu_clamp_value: float = 0.0,
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with gated activation and optional output tensors."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    varlen_m = cu_seqlens_m is not None
    # Determine output shape based on gather_A
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1])
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2)
    if preact_out is None and store_preact:
        preact_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    if z_scale_out is not None or postact_scale_out is not None:
        # Epilogue quant (z or postact/y1): bypass custom_op, call tuned fn directly (untuned for blockscaled).
        fn = partial(gemm_gated_tuned.fn, config=None)
        fn(A, B, preact_out, postact_out, C, bias, activation, cu_seqlens_m, A_idx,
           dynamic_scheduler, a_scales=a_scales, b_scales=b_scales, z_scale_out=z_scale_out,
           postact_scale_out=postact_scale_out, swiglu_clamp_value=swiglu_clamp_value)
    else:
        gemm_gated_out(
            A, B, preact_out, postact_out, C, bias, activation,
            cu_seqlens_m, A_idx, dynamic_scheduler, tuned, a_scales, b_scales, swiglu_clamp_value,
        )
    return preact_out, postact_out


@_custom_op_or_plain(
    "quack::gemm_gated_out",
    mutates_args=("preact_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!)? preact_out, Tensor(a3!) postact_out, Tensor? C=None, Tensor? bias=None, str activation='swiglu', Tensor? cu_seqlens_m=None, Tensor? A_idx=None, bool dynamic_scheduler=False, bool tuned=True, Tensor? a_scales=None, Tensor? b_scales=None, float swiglu_clamp_value=0.0) -> ()",
)
def gemm_gated_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    a_scales: Optional[Tensor] = None,
    b_scales: Optional[Tensor] = None,
    swiglu_clamp_value: float = 0.0,
) -> None:
    """GEMM with gated activation and pre-allocated output tensors."""
    # Blockscaled fused gated kernels run correctly with the default config on SM100,
    # but the autotuner still explores illegal configs and can crash with
    # CUDA_ERROR_ILLEGAL_INSTRUCTION. Keep the tuned API surface, but bypass the
    # autotuner until config search is blockscaled-safe.
    safe_tuned = tuned and not _uses_blockscaled_runtime(a_scales, b_scales)
    fn = gemm_gated_tuned if safe_tuned else partial(gemm_gated_tuned.fn, config=None)
    fn(A, B, preact_out, postact_out, C, bias, activation, cu_seqlens_m, A_idx, dynamic_scheduler,
       a_scales=a_scales, b_scales=b_scales, swiglu_clamp_value=swiglu_clamp_value)


def gemm_dgated(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    dx_out: Optional[Tensor] = None,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    a_scales: Optional[Tensor] = None,
    b_scales: Optional[Tensor] = None,
    tuned: bool = True,
    swiglu_clamp_value: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """GEMM with gated activation gradient and optional output tensors."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    varlen_m = cu_seqlens_m is not None
    # Determine output shape based on gather_A
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1] * 2)
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1] * 2)
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1] * 2)
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2)
    if dx_out is None:
        dx_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    colvec_reduce_final = gemm_dgated_out(
        A,
        B,
        PreAct,
        dx_out,
        postact_out,
        colvec_scale,
        activation,
        colvec_reduce,
        cu_seqlens_m,
        A_idx,
        dynamic_scheduler,
        tuned,
        a_scales,
        b_scales,
        swiglu_clamp_value,
    )
    if not colvec_reduce:
        return dx_out, postact_out
    else:
        return dx_out, postact_out, colvec_reduce_final


gemm_dgated.default_config = default_config


@_custom_op_or_plain(
    "quack::gemm_dgated_out",
    mutates_args=("dx_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor PreAct, Tensor(a3!) dx_out, Tensor(a4!) postact_out, Tensor? colvec_scale=None, str activation='swiglu', bool colvec_reduce=False, Tensor? cu_seqlens_m=None, Tensor? A_idx=None, bool dynamic_scheduler=True, bool tuned=True, Tensor? a_scales=None, Tensor? b_scales=None, float swiglu_clamp_value=0.0) -> Tensor?",
)
def gemm_dgated_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    dx_out: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    a_scales: Optional[Tensor] = None,
    b_scales: Optional[Tensor] = None,
    swiglu_clamp_value: float = 0.0,
) -> Optional[Tensor]:
    """GEMM with gated activation gradient and pre-allocated output tensors."""
    safe_tuned = tuned and not _uses_blockscaled_runtime(a_scales, b_scales)
    fn = gemm_dgated_tuned if safe_tuned else partial(gemm_dgated_tuned.fn, config=None)
    return fn(
        A,
        B,
        PreAct,
        dx_out,
        postact_out,
        colvec_scale,
        activation,
        colvec_reduce,
        cu_seqlens_m,
        A_idx,
        dynamic_scheduler,
        a_scales=a_scales,
        b_scales=b_scales,
        swiglu_clamp_value=swiglu_clamp_value,
    )


@_register_fake_safe("quack::gemm_dgated_out")
def gemm_dgated_out_fake(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    dx_out: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: Literal["swiglu", "swiglu_oai", "reglu", "geglu", "glu"] = "swiglu",
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    a_scales: Optional[Tensor] = None,
    b_scales: Optional[Tensor] = None,
    swiglu_clamp_value: float = 0.0,
) -> Optional[Tensor]:
    if not colvec_reduce:
        return None
    else:
        if cu_seqlens_m is not None:
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            out_shape = (total_m,)
        elif A.ndim == 2:
            out_shape = (A.shape[0],)
        else:
            out_shape = (A.shape[0], A.shape[-2])
        return torch.empty(out_shape, dtype=torch.float32, device=A.device)
