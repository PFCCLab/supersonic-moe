# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import partial
import os
from pathlib import Path
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
from quack.cache_utils import EXTRA_SOURCE_DIRS, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
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
from quack.gemm_tvm_ffi_utils import (
    div_for_dtype,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
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

from .activation_situ import (
    is_situ_activation,
    is_supported_activation,
    resolve_dgate_fn,
)

from .gemm_sm100_fp8_zeromat import (
    GemmDGatedSm100ZeroMat,
    GemmDGatedFP8CLoadSm100ZeroMat,
)

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
_SONIC_QUACK_UTILS_DIR = Path(__file__).resolve().parent
if _SONIC_QUACK_UTILS_DIR not in EXTRA_SOURCE_DIRS:
    EXTRA_SOURCE_DIRS.append(_SONIC_QUACK_UTILS_DIR)


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
# `activation` may also be a SiTU-GLU descriptor, e.g. "situ_glu:b=4.0:lb=25.0";
# see `resolve_dgate_fn`.


_DGATED_FFI_CLASSES = {
    "GemmDGatedSm100ZeroMat": GemmDGatedSm100ZeroMat,
    "GemmDGatedFP8CLoadSm100ZeroMat": GemmDGatedFP8CLoadSm100ZeroMat,
}


def _fake_tensor_rank(dtype, ndim: int, leading_dim: int):
    if dtype is None or ndim <= 0:
        return None
    shape = tuple(cute.sym_int() for _ in range(ndim))
    return fake_tensor(dtype, shape, leading_dim=leading_dim, divisibility=div_for_dtype(dtype))


@jit_cache
def _compile_gemm_dgated_tvm_ffi(
    gemm_cls_name,
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    implicit_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    activation,
    device_capacity,
    max_swizzle_size,
    varlen_m,
    gather_A,
    blockscaled,
    fp8_preact_mode,
    swiglu_clamp_value,
    colvec_scale_dtype,
    colvec_scale_ndim,
    colvec_reduce_dtype,
    colvec_reduce_ndim,
    a_scale_dtype,
    a_scale_ndim,
    b_scale_dtype,
    b_scale_ndim,
    preact_fp8_dtype,
    preact_fp8_ndim,
    preact_scales_dtype,
    preact_scales_ndim,
):
    GemmCls = _DGATED_FFI_CLASSES[gemm_cls_name]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
    )
    postact_shape = (m, n) if varlen_m else (m, n, l)
    mPostAct = fake_tensor(
        postact_dtype,
        postact_shape,
        leading_dim=1 if postact_major == "n" else 0,
        divisibility=div_for_dtype(postact_dtype),
    )
    mColVec = None
    if colvec_scale_ndim == 2:
        mColVec = fake_tensor(colvec_scale_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_scale_ndim == 1:
        mColVec = fake_tensor(colvec_scale_dtype, (m,), leading_dim=0, divisibility=4)

    mColVecReduce = None
    n_tiles = cute.sym_int()
    if colvec_reduce_ndim == 3:
        mColVecReduce = fake_tensor(
            colvec_reduce_dtype, (l, m, n_tiles), leading_dim=2, divisibility=1
        )
    elif colvec_reduce_ndim == 2:
        mColVecReduce = fake_tensor(
            colvec_reduce_dtype, (m, n_tiles), leading_dim=1, divisibility=1
        )

    epi_kwargs = {}
    if fp8_preact_mode:
        epi_kwargs["mFP8PreAct_fp8"] = _fake_tensor_rank(
            preact_fp8_dtype, preact_fp8_ndim, leading_dim=1
        )
        epi_kwargs["mFP8PreAct_scales"] = _fake_tensor_rank(
            preact_scales_dtype, preact_scales_ndim, leading_dim=1
        )
    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        resolve_dgate_fn(activation, dgate_fn_map),
        swiglu_clamp_value=float(swiglu_clamp_value),
        mColVecBroadcast=mColVec,
        mColVecReduce=mColVecReduce,
        **epi_kwargs,
    )
    scheduler_args = make_fake_scheduler_args(False, False, l)
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)
    mSFA = _fake_tensor_rank(a_scale_dtype, a_scale_ndim, leading_dim=1)
    mSFB = _fake_tensor_rank(b_scale_dtype, b_scale_ndim, leading_dim=1)

    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        gather_A=gather_A,
        sf_vec_size=32 if blockscaled else None,
    )
    gemm_obj.implicit_dtype = implicit_dtype
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        stream,
        mSFA,
        mSFB,
        options="--enable-tvm-ffi",
    )


def _can_use_dgated_tvm_ffi(
    GemmCls,
    device_capacity,
    gather_A,
    blockscaled,
    tile_count_semaphore,
    tensor_infos,
) -> bool:
    if os.getenv("SONIC_MOE_DISABLE_DGATED_TVM_FFI", "0").lower() in {"1", "true", "yes", "on"}:
        return False
    return (
        device_capacity[0] > 9
        and GemmCls.__name__ in _DGATED_FFI_CLASSES
        and gather_A
        and blockscaled
        and tile_count_semaphore is None
        and tensor_infos["D"].tensor is not None
        and tensor_infos["C"].tensor is not None
    )


def _run_gemm_dgated_tvm_ffi(
    GemmCls,
    tensor_infos,
    activation,
    implicit_dtype,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    max_swizzle_size,
    device_capacity,
    cu_seqlens_m,
    A_idx,
    a_scales,
    b_scales,
    colvec_scale,
    colvec_reduce,
    fp8_preact_mode,
    preact_fp8,
    preact_scales,
    swiglu_clamp_value,
):
    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    blockscaled = a_scales is not None and b_scales is not None
    compiled_fn = _compile_gemm_dgated_tvm_ffi(
        GemmCls.__name__,
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tensor_infos["C"].dtype,
        tensor_infos["PostAct"].dtype,
        implicit_dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        tensor_infos["D"].major,
        tensor_infos["C"].major,
        tensor_infos["PostAct"].major,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        activation,
        device_capacity,
        max_swizzle_size,
        varlen_m,
        gather_A,
        blockscaled,
        fp8_preact_mode,
        float(swiglu_clamp_value),
        _TORCH_TO_CUTLASS_DTYPE[colvec_scale.dtype] if colvec_scale is not None else None,
        colvec_scale.ndim if colvec_scale is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[colvec_reduce.dtype] if colvec_reduce is not None else None,
        colvec_reduce.ndim if colvec_reduce is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[a_scales.dtype] if a_scales is not None else None,
        a_scales.ndim if a_scales is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[b_scales.dtype] if b_scales is not None else None,
        b_scales.ndim if b_scales is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[preact_fp8.dtype] if preact_fp8 is not None else None,
        preact_fp8.ndim if preact_fp8 is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[preact_scales.dtype] if preact_scales is not None else None,
        preact_scales.ndim if preact_scales is not None else 0,
    )
    from quack.cache_utils import COMPILE_ONLY as _COMPILE_ONLY
    if _COMPILE_ONLY:
        return

    epi_kwargs = {}
    if fp8_preact_mode:
        epi_kwargs["mFP8PreAct_fp8"] = preact_fp8
        epi_kwargs["mFP8PreAct_scales"] = preact_scales
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].tensor,
        None,
        swiglu_clamp_value=None,
        mColVecBroadcast=colvec_scale,
        mColVecReduce=colvec_reduce,
        rounding_mode=None,
        sr_seed=None,
        **epi_kwargs,
    )
    max_active_clusters = get_max_active_clusters(cluster_shape_mnk[0] * cluster_shape_mnk[1]) if persistent else 0
    scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle_size, None)
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)
    compiled_fn(
        tensor_infos["A"].tensor,
        tensor_infos["B"].tensor,
        tensor_infos["D"].tensor,
        tensor_infos["C"].tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        a_scales,
        b_scales,
    )


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
        # SiTU-GLU has no clamped variant: raise rather than silently changing
        # the numerics the caller asked for. Legacy non-swiglu activations keep
        # their historical silent-zero behaviour.
        if is_situ_activation(activation) and float(swiglu_clamp_value) > 0.0:
            raise ValueError(
                f"swiglu_clamp_value={swiglu_clamp_value} is not supported for activation "
                f"{activation!r}: SiTU-GLU has no clamped variant. Pass swiglu_clamp_value=0.0."
            )
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
    assert is_supported_activation(activation, dgate_fn_map), f"Unsupported activation {activation}"

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

    blockscaled = a_scales is not None and b_scales is not None
    if _can_use_dgated_tvm_ffi(
        GemmCls,
        device_capacity,
        gather_A,
        blockscaled,
        tile_count_semaphore,
        tensor_infos,
    ):
        _run_gemm_dgated_tvm_ffi(
            GemmCls,
            tensor_infos,
            activation,
            implicit_dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong,
            persistent,
            max_swizzle_size,
            device_capacity,
            cu_seqlens_m,
            A_idx,
            a_scales,
            b_scales,
            colvec_scale,
            colvec_reduce,
            fp8_preact_mode,
            preact_fp8,
            preact_scales,
            swiglu_clamp_value,
        )
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            info.cute_tensor = _make_cute_tensor_dynamic(
                info.tensor,
                leading_dim=1 if info.major == major_configs[name][1] else 0,
            )
    act_fn = resolve_dgate_fn(activation, dgate_fn_map)
    epi_kwargs = {}
    if fp8_preact_mode:
        epi_kwargs["mFP8PreAct_fp8"] = _make_cute_tensor_dynamic(preact_fp8, leading_dim=1)
        epi_kwargs["mFP8PreAct_scales"] = _make_cute_tensor_dynamic(preact_scales, leading_dim=1)
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
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
        gemm_obj.implicit_dtype = implicit_dtype
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
