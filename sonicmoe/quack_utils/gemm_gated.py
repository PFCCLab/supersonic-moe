# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils.blackwell_helpers as sm100_utils
import quack.activation
import quack.sm90_utils as sm90_utils
import torch
from cutlass import const_expr, Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, mlir_namedtuple
from quack.epi_ops import TileStore, EpiOp, assume_stride_divisibility
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase
from quack.layout_utils import permute_gated_Cregs_b16

from torch import Tensor

from ._gated_epilogues import (
    _TORCH_TO_CUTLASS_DTYPE,
    _is_runtime_fp8_tensor,
    _make_cute_tensor_dynamic,
    _halve_epi_tile,
    GemmGatedMixin,
    _f32_as_i32,
    _i32_as_f32,
    BlockscaledScaleStore,
    GemmGatedBlockscaledQuantMixin,
    BlockscaledQuantOnlyMixin,
)

from .gemm_sm100_fp8_zeromat import (
    GemmGatedSm100ZeroMat,
    GemmGatedSm100ZeroMatBlockscaledQuant,
    GemmGatedSm100ZeroMatPostActQuant,
    GemmGatedSm100ZeroMatQuantPostActQuant,
    GemmSm100ZeroMatBlockscaledQuant,
)

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)


class GemmGatedSm90(GemmGatedMixin, GemmSm90):
    pass


class GemmGatedSm100(GemmGatedMixin, GemmSm100):
    pass


class GemmGatedBlockscaledQuantSm100(GemmGatedBlockscaledQuantMixin, GemmSm100):
    pass


class BlockscaledQuantOnlySm100(BlockscaledQuantOnlyMixin, GemmSm100):
    """SM100 GemmDefault + epilogue blockscaled FP8 quant of D, no activation."""
    pass


gate_fn_map = {
    "swiglu": quack.activation.swiglu_precise,
    "swiglu_oai": quack.activation.swiglu_oai,
    "reglu": quack.activation.reglu,
    "geglu": quack.activation.geglu,
    "glu": quack.activation.glu,
}


def gemm_gated(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n//2) or (total_m, n//2) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    a_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for A
    b_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for B
    z_scale_out: Optional[Tensor] = None,  # (total_m, N//32) uint8 — epilogue quant scale output
    postact_scale_out: Optional[Tensor] = None,  # ISA-packed UE8M0 scales for postact (y1) quant
    swiglu_clamp_value: float = 0.0,
) -> None:
    if activation != "swiglu":
        swiglu_clamp_value = 0.0
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in gate_fn_map, f"Unsupported activation {activation}"

    # Special validation for PostAct shape
    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )

    # PostAct shape validation depends on varlen_m
    if cu_seqlens_m is not None:
        # varlen_m case: PostAct is 2D (total_m, n//2)
        assert PostAct.dim() == 2 and PostAct.is_cuda, "PostAct must be a 2D CUDA tensor for varlen_m"
        assert PostAct.shape == (
            M,
            N // 2,
        ), f"PostAct must have shape {(M, N // 2)}, got {PostAct.shape}"
    else:
        # Normal case: PostAct is 3D (l, m, n//2)
        assert PostAct.dim() == 3 and PostAct.is_cuda, "PostAct must be a 3D CUDA tensor"
        assert PostAct.shape == (
            L,
            M,
            N // 2,
        ), f"PostAct must have shape {(L, M, N // 2)}, got {PostAct.shape}"

    tensor_infos["PostAct"] = GemmTensorInfo(PostAct)
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),  # PostAct has shape (m, n//2, l) after permute
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for info in tensor_infos.values():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    # Use zero-materialization kernel when gather_A + blockscaled (FP8 with A_idx)
    blockscaled_runtime = a_scales is not None and b_scales is not None
    epilogue_quant = z_scale_out is not None
    postact_quant = postact_scale_out is not None
    # z-quant and y1-quant CAN coexist (combined epilogue): the two EpiOps operate
    # on disjoint register fragments (tRS_rD vs tRS_rPostAct) and disjoint scale
    # buffers (mZScale vs mPostActScaleIsa).  Running both lets fuse_y1=1 coexist
    # with save_z_fp8=1 so the backward takes the healthy fp8-preact dgated path.
    combined_quant = epilogue_quant and postact_quant
    if epilogue_quant:
        assert device_capacity[0] > 9, "Epilogue quant only supported on SM100+"
    if postact_quant:
        assert device_capacity[0] > 9, "Postact quant only supported on SM100+"
        assert gather_A and blockscaled_runtime, (
            "Postact (y1) quant only supported on the gather_A+blockscaled zeromat path"
        )
    if device_capacity[0] > 9 and gather_A and blockscaled_runtime:
        if combined_quant:
            GemmCls = GemmGatedSm100ZeroMatQuantPostActQuant
        elif epilogue_quant:
            GemmCls = GemmGatedSm100ZeroMatBlockscaledQuant
        elif postact_quant:
            GemmCls = GemmGatedSm100ZeroMatPostActQuant
        else:
            GemmCls = GemmGatedSm100ZeroMat
    elif device_capacity[0] > 9:
        GemmCls = GemmGatedBlockscaledQuantSm100 if epilogue_quant else GemmGatedSm100
    else:
        GemmCls = GemmGatedSm90

    acc_dtype = cutlass.Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            leading_dim = 1 if info.major == major_configs[name][1] else 0
            info.cute_tensor = _make_cute_tensor_dynamic(info.tensor, leading_dim)
    act_fn = gate_fn_map[activation]
    epi_kwargs = {}
    if epilogue_quant:
        epi_kwargs["mZScale"] = _make_cute_tensor_dynamic(z_scale_out, leading_dim=1)
    if postact_quant:
        epi_kwargs["mPostActScaleIsa"] = _make_cute_tensor_dynamic(postact_scale_out, leading_dim=2)
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        swiglu_clamp_value=float(swiglu_clamp_value),
        mRowVecBroadcast=(
            from_dlpack(rowvec_bias.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
            if rowvec_bias is not None
            else None
        ),
        mColVecBroadcast=(
            from_dlpack(colvec_bias.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=1 if cu_seqlens_m is None else 0
            )
            if colvec_bias is not None
            else None
        ),
        **epi_kwargs,
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore,
        max_swizzle_size=max_swizzle_size,
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
    )

    _stream_obj = torch.cuda.current_stream()
    current_stream = cuda.CUstream(_stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream)

    blockscaled = a_scales is not None and b_scales is not None
    sf_vec_size = 32 if blockscaled else None

    # Prepare blockscaled scale cute tensors if provided
    if blockscaled:
        a_scale_cute = _make_cute_tensor_dynamic(a_scales, leading_dim=1)
        b_scale_cute = _make_cute_tensor_dynamic(b_scales, leading_dim=1)
    else:
        a_scale_cute = None
        b_scale_cute = None

    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        rowvec_bias.dtype if rowvec_bias is not None else None,
        colvec_bias.dtype if colvec_bias is not None else None,
        cu_seqlens_m is not None,
        A_idx is not None,
        blockscaled,
        epilogue_quant,
        postact_quant,
        float(swiglu_clamp_value),
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_gated.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
            sf_vec_size=sf_vec_size,
        )
        cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            a_scale_cute,
            b_scale_cute,
        )
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )


from ..cache_manager import InstrumentedCompileCache as _ICC
gemm_gated.compile_cache = _ICC("gated")
