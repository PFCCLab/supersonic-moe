# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import partial
from typing import Callable, NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import quack.activation
import quack.layout_utils as layout_utils
import quack.utils as utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import (
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epi_ops import ColVecReduce, TileStore, EpiOp, assume_stride_divisibility
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_wrapper_utils import GemmWrapperBase
from torch import Tensor

from ._gated_epilogues import (
    _TORCH_TO_CUTLASS_DTYPE,
    _is_runtime_fp8_tensor,
    _make_cute_tensor_dynamic,
    GemmDGatedMixin,
    _fp8e4m3_to_f32,
    _f32_as_i32,
    _i32_as_f32,
    FP8PreActLoad,
    GemmDGatedFP8PreActMixin,
    GemmDGatedFP8CLoadMixin,
)

from .gemm_sm100_fp8_zeromat import (
    GemmDGatedSm100ZeroMat,
    GemmDGatedFP8CLoadSm100ZeroMat,
)

from .sm_limit import capped_max_active_clusters

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)


class GemmDGatedSm90(GemmDGatedMixin, GemmSm90):
    pass


class GemmDGatedSm100(GemmDGatedMixin, GemmSm100):
    pass


class GemmDGatedFP8CLoadSm100(GemmDGatedFP8CLoadMixin, GemmSm100):
    pass


dgate_fn_map = {
    "swiglu": quack.activation.dswiglu_precise,
    "swiglu_oai": quack.activation.dswiglu_oai,
    "reglu": quack.activation.dreglu,
    "geglu": quack.activation.dgeglu,
    "glu": quack.activation.dglu,
}


def gemm_dgated(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, 2*n) if n_major or (l, 2*m, n) if m_major, or (total_m, 2*n) if varlen_m
    PreAct: Tensor,  # (l, m, 2*n) if n_major or (l, 2*m, n) if m_major, or (total_m, 2*n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    # (l, m, ceildiv(n, tile_n)), or (total_m, ceildiv(n, tile_n)) if varlen_m
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    a_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for A
    b_scales: Optional[Tensor] = None,  # ISA-packed blockscaled scales for B
    preact_fp8: Optional[Tensor] = None,  # (total_m, 2n) fp8 — replaces PreAct when provided
    preact_scales: Optional[Tensor] = None,  # (total_m, 2n//32) uint8 — blockscaled scales for preact_fp8
    swiglu_clamp_value: float = 0.0,
) -> None:
    """If tile_count_semaphore is provided, it must already be zero'ed out."""
    if activation != "swiglu":
        swiglu_clamp_value = 0.0
    fp8_preact_mode = preact_fp8 is not None and preact_scales is not None
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        if not fp8_preact_mode:
            assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in dgate_fn_map, f"Unsupported activation {activation}"

    # Special handling for Out and PreAct
    AB_swapped = not Out.stride(-1) == 1
    if fp8_preact_mode:
        # FP8 PreAct: View (TK, 2I) fp8 as (TK, I) Int16 to match D's shape (TK, I) f32.
        # Each Int16 = 2 packed fp8 values (gate+up), mirroring f32 = 2 packed bf16.
        # This avoids changing the epi_tile (shared by kernel for both C and D).
        implicit_dtype = cutlass.BFloat16  # for D output packing
        assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
        if cu_seqlens_m is not None or not AB_swapped:
            Out = Out.view(torch.float32)
        else:
            Out = Out.mT.view(torch.float32).mT
        # View fp8 (TK, 2I) as int16 (TK, I) — 2 fp8 per int16
        PreAct = preact_fp8.view(torch.int16)  # (TK, 2I) fp8 -> (TK, I) int16
    else:
        assert Out.dtype == PreAct.dtype
        implicit_dtype = torch2cute_dtype_map[Out.dtype]
        assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
        assert PreAct.element_size() == 2, "Preact dtype must be fp16 or bf16"
        if cu_seqlens_m is not None or not AB_swapped:
            Out = Out.view(torch.float32)
            PreAct = PreAct.view(torch.float32)
        else:
            Out = Out.mT.view(torch.float32).mT
            PreAct = PreAct.mT.view(torch.float32).mT

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A,
        B,
        Out,
        PreAct,  # Int16 (TK,I) for fp8_preact_mode, or f32 (TK,I) for standard
        additional_tensors={"PostAct": PostAct},
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    # Use zero-materialization kernel when gather_A + blockscaled (FP8 with A_idx)
    blockscaled_runtime = a_scales is not None and b_scales is not None
    if fp8_preact_mode:
        assert device_capacity[0] > 9, "FP8 PreAct only supported on SM100+"
        if gather_A and blockscaled_runtime:
            GemmCls = GemmDGatedFP8CLoadSm100ZeroMat
        else:
            GemmCls = GemmDGatedFP8CLoadSm100
    elif device_capacity[0] > 9 and gather_A and blockscaled_runtime:
        GemmCls = GemmDGatedSm100ZeroMat
    elif device_capacity[0] > 9:
        GemmCls = GemmDGatedSm100
    else:
        GemmCls = GemmDGatedSm90

    acc_dtype = Float32
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

    max_active_clusters = capped_max_active_clusters(cluster_M * cluster_N, persistent=persistent)
    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            info.cute_tensor = _make_cute_tensor_dynamic(
                info.tensor,
                leading_dim=1 if info.major == major_configs[name][1] else 0,
            )
    act_fn = dgate_fn_map[activation]
    epi_kwargs = {}
    if fp8_preact_mode:
        epi_kwargs["mFP8PreAct_fp8"] = _make_cute_tensor_dynamic(preact_fp8, leading_dim=1)
        epi_kwargs["mFP8PreAct_scales"] = _make_cute_tensor_dynamic(preact_scales, leading_dim=1)
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        implicit_dtype=implicit_dtype,
        swiglu_clamp_value=float(swiglu_clamp_value),
        mColVecBroadcast=(
            from_dlpack(colvec_scale.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=1 if cu_seqlens_m is None else 0
            )
            if colvec_scale is not None
            else None
        ),
        mColVecReduce=(
            from_dlpack(colvec_reduce.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=2 if cu_seqlens_m is None else 1
            )
            if colvec_reduce is not None
            else None
        ),
        **epi_kwargs,
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(max_active_clusters, tile_count_semaphore)

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
        colvec_scale.dtype if colvec_scale is not None else None,
        colvec_reduce.dtype if colvec_reduce is not None else None,
        cu_seqlens_m is not None,
        A_idx is not None,
        blockscaled,
        fp8_preact_mode,
        float(swiglu_clamp_value),
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_dgated.compile_cache
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
            tensor_infos["D"].cute_tensor,  # Out
            tensor_infos["C"].cute_tensor,  # PreAct
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
        tensor_infos["D"].cute_tensor,  # Out
        tensor_infos["C"].cute_tensor,  # PreAct
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        a_scale_cute,
        b_scale_cute,
    )


from ..cache_manager import InstrumentedCompileCache as _ICC
gemm_dgated.compile_cache = _ICC("dgated")
