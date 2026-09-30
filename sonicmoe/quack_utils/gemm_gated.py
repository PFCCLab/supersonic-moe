# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
import sys
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
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
from quack.cache_utils import EXTRA_SOURCE_DIRS, jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, mlir_namedtuple
from quack.epi_ops import TileStore, EpiOp, assume_stride_divisibility
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

from .activation_situ import (
    is_situ_activation,
    is_supported_activation,
    resolve_gate_fn,
)

from .sm90_fp8_runtime import (
    get_sm90_carrier,
    is_sm90_native_active,
    is_sm90_native_carrier,
    register_sm90_carrier,
    replace_with_sm90_anchor,
    use_sm90_invocation,
)
from . import gemm_sm90_scaled as _sm90_scaled
from .gemm_sm90_scaled import sm90_scaled_gated
from .gemm_sm100_fp8_zeromat import (
    GemmGatedSm100ZeroMat,
    GemmGatedSm100ZeroMatBlockscaledQuant,
    GemmGatedSm100ZeroMatPostActQuant,
    GemmGatedSm100ZeroMatQuantPostActQuant,
    GemmSm100ZeroMatBlockscaledQuant,
)

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
_SONIC_QUACK_UTILS_DIR = Path(__file__).resolve().parent
if _SONIC_QUACK_UTILS_DIR not in EXTRA_SOURCE_DIRS:
    EXTRA_SOURCE_DIRS.append(_SONIC_QUACK_UTILS_DIR)


def _sm90_reference_module():
    package_root = __package__.split(".quack_utils", 1)[0]
    for module_name in (
        f"{package_root}.functional.sm90_reference",
        "paddlefleet_ops.sonicmoe.functional.sm90_reference",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            return module
    raise RuntimeError("SM90 reference helpers were not loaded by sonicmoe.functional")


def _raw_stream_id(stream) -> Optional[int]:
    if stream is None:
        return None
    if isinstance(stream, int):
        return stream
    if hasattr(stream, "stream_base"):
        return int(stream.stream_base.raw_stream)
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    try:
        return int(stream)
    except (TypeError, ValueError):
        return None


def _current_raw_stream_id() -> int:
    stream = torch.cuda.current_stream()
    if hasattr(stream, "stream_base"):
        return int(stream.stream_base.raw_stream)
    return int(stream.cuda_stream)


def _stream_matches_current(stream) -> bool:
    stream_id = _raw_stream_id(stream)
    return stream_id is None or stream_id == _current_raw_stream_id()


_GATED_FFI_B_VIEW_ATTR = "_sonic_gated_tvm_ffi_b_view"
_GATED_FFI_LAST_PLAN_ATTR = "_sonic_gated_tvm_ffi_last_plan"


class _PreparedGatedTvmFfiPlan(NamedTuple):
    """Pointer-stable weight state for the Sonic gated FFI launch."""

    b_view: Tensor
    b_data_ptr: int
    b_view_data_ptr: int
    compiled_fn: Callable
    gemm_cls: type
    scheduler_args: object
    activation: str
    epilogue_quant: bool
    postact_quant: bool
    swiglu_clamp_value: float
    postact_bf16_trunc: bool


def prepare_gated_tvm_ffi_weight(B: Tensor) -> Tensor:
    """Attach the storage-free TVM-FFI B view to one quantized weight."""
    if B.ndim == 3 and B.is_cuda and B.is_contiguous():
        setattr(B, _GATED_FFI_B_VIEW_ATTR, B.permute(1, 2, 0))
    return B


def _prepared_gated_tvm_ffi_weight(B: Tensor) -> Optional[Tensor]:
    view = getattr(B, _GATED_FFI_B_VIEW_ATTR, None)
    if view is None or B.ndim != 3:
        return None
    expected_shape = (B.shape[1], B.shape[2], B.shape[0])
    expected_stride = (B.stride(1), B.stride(2), B.stride(0))
    if (
        view.dtype != B.dtype
        or view.device != B.device
        or tuple(view.shape) != expected_shape
        or tuple(view.stride()) != expected_stride
        or view.data_ptr() != B.data_ptr()
    ):
        return None
    return view


class GemmGatedSm90(GemmGatedMixin, GemmSm90):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GemmSm100 defines this flag, while current GemmSm90 does not.
        # Shared gated epilogue hooks use it to select the non-blockscaled path.
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        """Match the one-argument setup hook used by current QuACK Hopper."""
        return GemmSm90._setup_attributes(self, epilogue_args)


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
# `activation` may also be a SiTU-GLU descriptor, e.g. "situ_glu:b=4.0:lb=25.0".
# Those are resolved by `resolve_gate_fn` instead of a dict lookup, because the
# betas are baked into the traced kernel as Constexpr values.


def _reject_situ_clamp(activation, swiglu_clamp_value) -> None:
    """SiTU-GLU has no clamped variant, so never silently drop a requested clamp."""
    if is_situ_activation(activation) and float(swiglu_clamp_value) > 0.0:
        raise ValueError(
            f"swiglu_clamp_value={swiglu_clamp_value} is not supported for activation "
            f"{activation!r}: SiTU-GLU has no clamped variant. Pass swiglu_clamp_value=0.0."
        )


_GATED_FFI_CLASSES = {
    "GemmGatedSm100ZeroMat": GemmGatedSm100ZeroMat,
    "GemmGatedSm100ZeroMatBlockscaledQuant": GemmGatedSm100ZeroMatBlockscaledQuant,
    "GemmGatedSm100ZeroMatPostActQuant": GemmGatedSm100ZeroMatPostActQuant,
    "GemmGatedSm100ZeroMatQuantPostActQuant": GemmGatedSm100ZeroMatQuantPostActQuant,
}


def _fake_tensor_rank(dtype, ndim: int, leading_dim: int):
    if dtype is None or ndim <= 0:
        return None
    shape = tuple(cute.sym_int() for _ in range(ndim))
    return fake_tensor(dtype, shape, leading_dim=leading_dim, divisibility=div_for_dtype(dtype))


@lru_cache(maxsize=32)
@jit_cache
def _compile_gemm_gated_tvm_ffi(
    gemm_cls_name,
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
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
    epilogue_quant,
    postact_quant,
    swiglu_clamp_value,
    postact_bf16_trunc,
    a_scale_dtype,
    a_scale_ndim,
    b_scale_dtype,
    b_scale_ndim,
    z_scale_dtype,
    z_scale_ndim,
    postact_scale_dtype,
    postact_scale_ndim,
):
    GemmCls = _GATED_FFI_CLASSES[gemm_cls_name]
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
    postact_shape = (m, cute.sym_int()) if varlen_m else (m, cute.sym_int(), l)
    mPostAct = fake_tensor(
        postact_dtype,
        postact_shape,
        leading_dim=1 if postact_major == "n" else 0,
        divisibility=div_for_dtype(postact_dtype),
    )
    epi_kwargs = {}
    if epilogue_quant:
        epi_kwargs["mZScale"] = _fake_tensor_rank(z_scale_dtype, z_scale_ndim, leading_dim=1)
    if postact_quant:
        epi_kwargs["mPostActScaleIsa"] = _fake_tensor_rank(
            postact_scale_dtype, postact_scale_ndim, leading_dim=2
        )
        epi_kwargs["postact_bf16_trunc"] = bool(postact_bf16_trunc)
    epi_args = GemmCls.EpilogueArguments(
        mPostAct,
        resolve_gate_fn(activation, gate_fn_map),
        swiglu_clamp_value=float(swiglu_clamp_value),
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
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


def _can_use_gated_tvm_ffi(
    GemmCls,
    device_capacity,
    gather_A,
    blockscaled,
    tile_count_semaphore,
    rowvec_bias,
    colvec_bias,
    current_stream,
    tensor_infos,
) -> bool:
    if os.getenv("SONIC_MOE_DISABLE_GATED_TVM_FFI", "0").lower() in {"1", "true", "yes", "on"}:
        return False
    return (
        device_capacity[0] > 9
        and GemmCls.__name__ in _GATED_FFI_CLASSES
        and gather_A
        and blockscaled
        and tile_count_semaphore is None
        and rowvec_bias is None
        and colvec_bias is None
        and _stream_matches_current(current_stream)
        and tensor_infos["D"].tensor is not None
    )


def _run_gemm_gated_tvm_ffi(
    GemmCls,
    tensor_infos,
    activation,
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
    z_scale_out,
    postact_scale_out,
    swiglu_clamp_value,
    postact_bf16_trunc,
):
    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    blockscaled = a_scales is not None and b_scales is not None
    epilogue_quant = z_scale_out is not None
    postact_quant = postact_scale_out is not None
    compiled_fn = _compile_gemm_gated_tvm_ffi(
        GemmCls.__name__,
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tensor_infos["C"].dtype,
        tensor_infos["PostAct"].dtype,
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
        epilogue_quant,
        postact_quant,
        float(swiglu_clamp_value),
        bool(postact_bf16_trunc),
        _TORCH_TO_CUTLASS_DTYPE[a_scales.dtype] if a_scales is not None else None,
        a_scales.ndim if a_scales is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[b_scales.dtype] if b_scales is not None else None,
        b_scales.ndim if b_scales is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[z_scale_out.dtype] if z_scale_out is not None else None,
        z_scale_out.ndim if z_scale_out is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[postact_scale_out.dtype] if postact_scale_out is not None else None,
        postact_scale_out.ndim if postact_scale_out is not None else 0,
    )
    from quack.cache_utils import COMPILE_ONLY as _COMPILE_ONLY
    if _COMPILE_ONLY:
        return

    epi_kwargs = {}
    if epilogue_quant:
        epi_kwargs["mZScale"] = z_scale_out
    if postact_quant:
        epi_kwargs["mPostActScaleIsa"] = postact_scale_out
        epi_kwargs["postact_bf16_trunc"] = None
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].tensor,
        None,
        swiglu_clamp_value=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
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


@lru_cache(maxsize=16)
def _static_gated_scheduler_args(
    cluster_m: int,
    cluster_n: int,
    max_swizzle_size: int,
    device_index: int,
):
    """Cache scheduler metadata that never contains a tensor pointer."""
    del device_index  # Part of the cache key because cluster capacity is device-local.
    max_active_clusters = get_max_active_clusters(cluster_m * cluster_n)
    return make_scheduler_args(max_active_clusters, max_swizzle_size, None)


def _launch_prepared_gated_tvm_ffi(
    plan: _PreparedGatedTvmFfiPlan,
    A: Tensor,
    D: Tensor,
    PostAct: Tensor,
    cu_seqlens_m: Tensor,
    A_idx: Tensor,
    a_scales: Tensor,
    b_scales: Tensor,
    z_scale_out: Optional[Tensor],
    postact_scale_out: Tensor,
) -> None:
    """Build only pointer-bearing arguments, then issue one prepared FFI call."""
    epi_kwargs = {"mPostActScaleIsa": postact_scale_out}
    if plan.epilogue_quant:
        epi_kwargs["mZScale"] = z_scale_out
    epilogue_args = plan.gemm_cls.EpilogueArguments(
        PostAct,
        None,
        swiglu_clamp_value=None,
        postact_bf16_trunc=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        rounding_mode=None,
        sr_seed=None,
        **epi_kwargs,
    )
    # These structures contain current-microbatch pointers and must not be
    # cached across PP/VPP contexts.
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)
    plan.compiled_fn(
        A,
        plan.b_view,
        D,
        None,
        epilogue_args,
        plan.scheduler_args,
        varlen_args,
        a_scales,
        b_scales,
    )


def try_gemm_gated_tvm_ffi_sonic_prepared(
    A: Tensor,
    B: Tensor,
    D: Optional[Tensor],
    PostAct: Optional[Tensor],
    *,
    activation: str,
    cu_seqlens_m: Optional[Tensor],
    A_idx: Optional[Tensor],
    a_scales: Optional[Tensor],
    b_scales: Optional[Tensor],
    z_scale_out: Optional[Tensor],
    postact_scale_out: Optional[Tensor],
    swiglu_clamp_value: float,
    postact_bf16_trunc: bool,
    current_stream=None,
) -> bool:
    """Issue a previously validated plan from Sonic's preallocated fast path.

    The caller has already validated dynamic output shapes and dtypes. The plan
    owns only the persistent quantized-weight view and pointer-free compiled
    state; all microbatch-dependent pointer structures are rebuilt above.
    """
    if os.getenv("SONIC_MOE_DISABLE_GATED_TVM_FFI", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    plan = getattr(B, _GATED_FFI_LAST_PLAN_ATTR, None)
    if not isinstance(plan, _PreparedGatedTvmFfiPlan):
        return False
    epilogue_quant = z_scale_out is not None
    if (
        D is None
        or PostAct is None
        or cu_seqlens_m is None
        or A_idx is None
        or a_scales is None
        or b_scales is None
        or postact_scale_out is None
        or plan.activation != activation
        or plan.epilogue_quant != epilogue_quant
        or not plan.postact_quant
        or plan.swiglu_clamp_value != float(swiglu_clamp_value)
        or plan.postact_bf16_trunc != bool(postact_bf16_trunc)
        or not _stream_matches_current(current_stream)
    ):
        return False
    # Quantization normally replaces B wholesale. Detect uncommon in-place
    # storage rebinding so a stale prepared view can never launch.
    if int(B.data_ptr()) != plan.b_data_ptr or int(plan.b_view.data_ptr()) != plan.b_view_data_ptr:
        setattr(B, _GATED_FFI_LAST_PLAN_ATTR, None)
        return False
    _launch_prepared_gated_tvm_ffi(
        plan,
        A,
        D,
        PostAct,
        cu_seqlens_m,
        A_idx,
        a_scales,
        b_scales,
        z_scale_out,
        postact_scale_out,
    )
    return True


def try_gemm_gated_tvm_ffi_sonic_fast(
    A: Tensor,
    B: Tensor,
    D: Optional[Tensor],
    C: Optional[Tensor],
    PostAct: Optional[Tensor],
    *,
    activation: str,
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool,
    max_swizzle_size: int,
    bias: Optional[Tensor],
    cu_seqlens_m: Optional[Tensor],
    A_idx: Optional[Tensor],
    dynamic_scheduler: bool,
    a_scales: Optional[Tensor],
    b_scales: Optional[Tensor],
    z_scale_out: Optional[Tensor],
    postact_scale_out: Optional[Tensor],
    swiglu_clamp_value: float,
    postact_bf16_trunc: bool,
    current_stream=None,
) -> bool:
    """Launch the exact Sonic forward contract without generic CuTe setup.

    Only pointer-free scheduler and compilation state are cached. Epilogue and
    varlen arguments are reconstructed for every call so PP/VPP microbatches
    cannot observe stale tensor pointers.
    """
    if os.getenv("SONIC_MOE_DISABLE_GATED_TVM_FFI", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    # Nothing below is swiglu-specific: `activation` is only forwarded to the
    # compile cache key and to `resolve_gate_fn`, so SiTU descriptors are fine.
    _reject_situ_clamp(activation, swiglu_clamp_value)
    if (
        not (activation == "swiglu" or is_situ_activation(activation))
        or D is None
        or PostAct is None
        or C is not None
        or bias is not None
        or dynamic_scheduler
        or cu_seqlens_m is None
        or A_idx is None
        or a_scales is None
        or b_scales is None
        or postact_scale_out is None
        or cluster_N != 1
        or not _stream_matches_current(current_stream)
    ):
        return False
    if (
        A.ndim != 2
        or B.ndim != 3
        or D.ndim != 2
        or PostAct.ndim != 2
        or cu_seqlens_m.ndim != 1
        or A_idx.ndim != 1
        or not all(
            tensor.is_cuda
            for tensor in (A, B, D, PostAct, cu_seqlens_m, A_idx)
        )
    ):
        return False

    E, N, K = B.shape
    total_m = A_idx.shape[0]
    epilogue_quant = z_scale_out is not None
    postact_quant = postact_scale_out is not None
    expected_d_dtype = (
        torch.float8_e4m3fn if epilogue_quant else torch.bfloat16
    )
    if (
        A.shape[1] != K
        or tuple(D.shape) != (total_m, N)
        or tuple(PostAct.shape) != (total_m, N // 2)
        or tuple(cu_seqlens_m.shape) != (E + 1,)
        or N % 128 != 0
        or total_m % 128 != 0
        or A.stride(-1) != 1
        or D.stride(-1) != 1
        or PostAct.stride(-1) != 1
        or A.dtype != torch.float8_e4m3fn
        or B.dtype != torch.float8_e4m3fn
        or D.dtype != expected_d_dtype
        or PostAct.dtype != torch.float8_e4m3fn
        or cu_seqlens_m.dtype != torch.int32
        or A_idx.dtype != torch.int32
        or postact_scale_out.dtype != torch.uint8
        or tuple(postact_scale_out.shape)
        != (total_m // 128, (N // 2) // 128, 512)
    ):
        return False
    if epilogue_quant and (
        z_scale_out.dtype != torch.uint8
        or tuple(z_scale_out.shape) != (total_m, N // 32)
    ):
        return False

    B_ffi = _prepared_gated_tvm_ffi_weight(B)
    if B_ffi is None:
        return False
    device_capacity = get_device_capacity(A.device)
    if device_capacity[0] <= 9:
        return False
    scale_tensors = tuple(
        tensor
        for tensor in (a_scales, b_scales, z_scale_out, postact_scale_out)
        if tensor is not None
    )
    if any(tensor.dtype not in _TORCH_TO_CUTLASS_DTYPE for tensor in scale_tensors):
        return False

    GemmCls = (
        GemmGatedSm100ZeroMatQuantPostActQuant
        if epilogue_quant
        else GemmGatedSm100ZeroMatPostActQuant
    )
    compiled_fn = _compile_gemm_gated_tvm_ffi(
        GemmCls.__name__,
        _TORCH_TO_CUTLASS_DTYPE[A.dtype],
        _TORCH_TO_CUTLASS_DTYPE[B_ffi.dtype],
        _TORCH_TO_CUTLASS_DTYPE[D.dtype],
        None,
        _TORCH_TO_CUTLASS_DTYPE[PostAct.dtype],
        "k",
        "k",
        "n",
        None,
        "n",
        (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        True,
        activation,
        device_capacity,
        max_swizzle_size,
        True,
        True,
        True,
        epilogue_quant,
        True,
        float(swiglu_clamp_value),
        bool(postact_bf16_trunc),
        _TORCH_TO_CUTLASS_DTYPE[a_scales.dtype],
        a_scales.ndim,
        _TORCH_TO_CUTLASS_DTYPE[b_scales.dtype],
        b_scales.ndim,
        (
            _TORCH_TO_CUTLASS_DTYPE[z_scale_out.dtype]
            if z_scale_out is not None
            else None
        ),
        z_scale_out.ndim if z_scale_out is not None else 0,
        _TORCH_TO_CUTLASS_DTYPE[postact_scale_out.dtype],
        postact_scale_out.ndim,
    )
    from quack.cache_utils import COMPILE_ONLY as _COMPILE_ONLY

    if _COMPILE_ONLY:
        return True

    scheduler_args = _static_gated_scheduler_args(
        cluster_M,
        cluster_N,
        max_swizzle_size,
        int(torch.cuda.current_device()),
    )
    plan = _PreparedGatedTvmFfiPlan(
        b_view=B_ffi,
        b_data_ptr=int(B.data_ptr()),
        b_view_data_ptr=int(B_ffi.data_ptr()),
        compiled_fn=compiled_fn,
        gemm_cls=GemmCls,
        scheduler_args=scheduler_args,
        activation=activation,
        epilogue_quant=epilogue_quant,
        postact_quant=postact_quant,
        swiglu_clamp_value=float(swiglu_clamp_value),
        postact_bf16_trunc=bool(postact_bf16_trunc),
    )
    setattr(B, _GATED_FFI_LAST_PLAN_ATTR, plan)
    _launch_prepared_gated_tvm_ffi(
        plan,
        A,
        D,
        PostAct,
        cu_seqlens_m,
        A_idx,
        a_scales,
        b_scales,
        z_scale_out,
        postact_scale_out,
    )
    return True


def _carrier_view_for_kernel(carrier, tensor):
    del tensor
    carrier.materialize()
    return carrier.data


def _recompute_z_payload(
    output, save_fp8, a_carrier, b_carrier, A, B, cu_seqlens_m, A_idx,
    group_size, block_size,
):
    if save_fp8:
        from ..functional.sm90_reference import blockscaled_fp8_gemm_zeromat_quant

        z_fp8 = torch.empty(
            output.shape, dtype=torch.float8_e4m3fn, device=output.device
        )
        z_scales = torch.empty(
            (output.shape[0], output.shape[1] // 32),
            dtype=torch.uint8,
            device=output.device,
        )
        return blockscaled_fp8_gemm_zeromat_quant(
            _carrier_view_for_kernel(a_carrier, A),
            _carrier_view_for_kernel(b_carrier, B),
            cu_seqlens_m,
            A_idx,
            a_carrier.scales,
            b_carrier.scales,
            z_fp8_out=z_fp8,
            z_scale_out=z_scales,
        )
    z = torch.empty(output.shape, dtype=torch.bfloat16, device=output.device)
    _sm90_scaled.sm90_scaled_recompute_z(
        _carrier_view_for_kernel(a_carrier, A),
        _carrier_view_for_kernel(b_carrier, B),
        z,
        a_carrier.scales,
        b_carrier.scales,
        cu_seqlens_m,
        A_idx,
        activation_group=group_size,
        weight_block=block_size,
        activation_scale_order="source",
    )
    return z, None


def _try_sm90_native_gated(
    A, B, D, C, PostAct, *, activation, cu_seqlens_m, A_idx,
    rowvec_bias, colvec_bias, z_scale_out, postact_scale_out,
    swiglu_clamp_value, postact_bf16_trunc,
):
    a_carrier = get_sm90_carrier(A)
    b_carrier = get_sm90_carrier(B)
    if a_carrier is not None and b_carrier is None:
        invocation = a_carrier.metadata.get("invocation")
        if invocation is not None:
            b_carrier = get_sm90_carrier(B, invocation=invocation)
    if b_carrier is not None and a_carrier is None:
        invocation = b_carrier.metadata.get("invocation")
        if invocation is not None:
            a_carrier = get_sm90_carrier(A, invocation=invocation)
    if a_carrier is None or b_carrier is None:
        return False
    if not (
        is_sm90_native_active()
        or (is_sm90_native_carrier(a_carrier) and is_sm90_native_carrier(b_carrier))
    ):
        return False
    if rowvec_bias is not None or colvec_bias is not None or D is None:
        return False
    if C is not None:
        return False
    if activation != "swiglu":
        raise NotImplementedError(
            "native SM90 FP8 carrier gated GEMM supports only swiglu; "
            f"got activation={activation!r}. Refusing to read compatibility anchors."
        )
    if float(swiglu_clamp_value) != 0.0:
        raise NotImplementedError(
            "native SM90 FP8 carrier gated GEMM does not support "
            f"swiglu_clamp_value={swiglu_clamp_value}. Refusing to read compatibility anchors."
        )
    a_carrier.materialize()
    b_carrier.materialize()
    invocation = a_carrier.metadata["invocation"]
    group_size = invocation.activation_group_size
    block_size = invocation.weight_block_size
    cfg = invocation.fp8_config
    recompute_z = cfg.recompute_z
    save_z_fp8 = cfg.save_z_fp8 and not recompute_z
    fuse_y1 = cfg.fuse_y1_quant and group_size == 32
    epilogue_quant = cfg.epilogue_quant
    if recompute_z:
        pass
    elif save_z_fp8 and epilogue_quant:
        pass
    elif D.dtype != torch.bfloat16:
        return False
    if fuse_y1 and epilogue_quant:
        pass
    elif PostAct.dtype != torch.bfloat16:
        return False
    original_D = D
    original_postact = PostAct
    kernel_D = D
    kernel_postact = PostAct
    z_scales = z_scale_out
    y1_scales = postact_scale_out
    if recompute_z:
        kernel_D = None
        z_scales = None
    elif save_z_fp8 and epilogue_quant:
        kernel_D = torch.empty(D.shape, dtype=torch.float8_e4m3fn, device=D.device)
        z_scales = torch.empty(
            (D.shape[0], D.shape[1] // 32), dtype=torch.uint8, device=D.device
        )
    if fuse_y1 and epilogue_quant:
        kernel_postact = torch.empty(
            PostAct.shape, dtype=torch.float8_e4m3fn, device=PostAct.device
        )
        y1_scales = torch.empty(
            (PostAct.shape[0], PostAct.shape[1] // 32),
            dtype=torch.uint8,
            device=PostAct.device,
        )
    with use_sm90_invocation(invocation):
        sm90_scaled_gated(
            _carrier_view_for_kernel(a_carrier, A),
            _carrier_view_for_kernel(b_carrier, B),
            kernel_D,
            kernel_postact,
            a_carrier.scales,
            b_carrier.scales,
            activation=activation,
            z_scale_out=z_scales,
            postact_scale_out=y1_scales,
            swiglu_clamp_value=swiglu_clamp_value,
            postact_bf16_trunc=postact_bf16_trunc,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            activation_group=group_size,
            weight_block=block_size,
            activation_scale_order="source",
        )
        if recompute_z:
            carrier = register_sm90_carrier(
                original_D,
                role="z",
                recipe=lambda: _recompute_z_payload(
                    original_D,
                    cfg.save_z_fp8,
                    a_carrier,
                    b_carrier,
                    A,
                    B,
                    cu_seqlens_m,
                    A_idx,
                    group_size,
                    block_size,
                ),
                metadata={
                    "group_size": 32 if cfg.save_z_fp8 else group_size,
                    "mode": "recipe",
                    "recompute_output": "fp8" if cfg.save_z_fp8 else "bf16",
                    "scale_abi": "raw_ue8m0_1x32" if cfg.save_z_fp8 else None,
                },
                replace=True,
            )
            replace_with_sm90_anchor(original_D)
        elif save_z_fp8 and epilogue_quant:
            register_sm90_carrier(
                original_D,
                data=kernel_D,
                scales=z_scales,
                role="z",
                metadata={
                    "group_size": group_size,
                    "mode": "fp8",
                    "quant_source": "epilogue",
                },
                replace=True,
            )
            replace_with_sm90_anchor(original_D)
        elif save_z_fp8:
            sm90_reference = _sm90_reference_module()

            data, scales = sm90_reference.quantize_activation_blockscaled_fast(
                original_D.contiguous(), group_size=group_size
            )
            register_sm90_carrier(
                original_D, data=data, scales=scales, role="z",
                metadata={
                    "group_size": group_size,
                    "mode": "fp8",
                    "quant_source": "standalone_native",
                },
                replace=True,
            )
            replace_with_sm90_anchor(original_D)
        elif D is not None:
            if D.dtype == torch.float8_e4m3fn:
                if z_scale_out is None:
                    raise RuntimeError("native SM90 FP8 z output requires scales")
                register_sm90_carrier(
                    original_D, data=kernel_D, scales=z_scales, role="z",
                    metadata={"group_size": group_size, "mode": "fp8", "quant_source": "epilogue"},
                    replace=True,
                )
                replace_with_sm90_anchor(original_D)
            else:
                register_sm90_carrier(
                    original_D, role="z",
                    metadata={"group_size": group_size, "mode": "bf16"}, replace=True,
                )
        if fuse_y1:
            register_sm90_carrier(
                original_postact,
                data=kernel_postact,
                scales=y1_scales,
                role="y1",
                metadata={
                    "group_size": 32,
                    "mode": "fp8",
                    "quant_source": "epilogue",
                    "epilogue_fused": True,
                    "scale_abi": "raw_ue8m0_1x32",
                },
                replace=True,
            )
            replace_with_sm90_anchor(original_postact)
        else:
            sm90_reference = _sm90_reference_module()
            data, scales = sm90_reference.quantize_activation_blockscaled_fast(
                original_postact.contiguous(), group_size=32
            )
            register_sm90_carrier(
                original_postact,
                data=data,
                scales=scales,
                role="y1",
                metadata={
                    "group_size": 32,
                    "mode": "fp8",
                    "quant_source": "standalone_native",
                },
                replace=True,
            )
            replace_with_sm90_anchor(original_postact)
    # BF16 outputs remain materialized until quantize_activation_sm90 binds
    # their independent FP8 payload. Anchoring them here would make that
    # idempotent quantizer read zeros. FP8 compatibility tensors are themselves
    # payload storage and likewise must not be rebound.
    return True


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
    postact_bf16_trunc: bool = False,
    current_stream=None,
) -> None:
    if activation != "swiglu":
        # SiTU-GLU has no clamped variant: raise rather than silently changing
        # the numerics the caller asked for. Legacy non-swiglu activations keep
        # their historical silent-zero behaviour.
        _reject_situ_clamp(activation, swiglu_clamp_value)
        swiglu_clamp_value = 0.0
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        if not (get_sm90_carrier(A) is not None and get_sm90_carrier(B) is not None):
            assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert is_supported_activation(activation, gate_fn_map), f"Unsupported activation {activation}"

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
    if device_capacity[0] == 9 and _try_sm90_native_gated(
        A,
        B,
        D,
        C,
        PostAct,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        rowvec_bias=rowvec_bias,
        colvec_bias=colvec_bias,
        z_scale_out=z_scale_out,
        postact_scale_out=postact_scale_out,
        swiglu_clamp_value=swiglu_clamp_value,
        postact_bf16_trunc=postact_bf16_trunc,
    ):
        return
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

    blockscaled = a_scales is not None and b_scales is not None
    if _can_use_gated_tvm_ffi(
        GemmCls,
        device_capacity,
        gather_A,
        blockscaled,
        tile_count_semaphore,
        rowvec_bias,
        colvec_bias,
        current_stream,
        tensor_infos,
    ):
        _run_gemm_gated_tvm_ffi(
            GemmCls,
            tensor_infos,
            activation,
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
            z_scale_out,
            postact_scale_out,
            swiglu_clamp_value,
            postact_bf16_trunc,
        )
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            leading_dim = 1 if info.major == major_configs[name][1] else 0
            info.cute_tensor = _make_cute_tensor_dynamic(info.tensor, leading_dim)
    act_fn = resolve_gate_fn(activation, gate_fn_map)
    epi_kwargs = {}
    if epilogue_quant:
        epi_kwargs["mZScale"] = _make_cute_tensor_dynamic(z_scale_out, leading_dim=1)
    if postact_quant:
        epi_kwargs["mPostActScaleIsa"] = _make_cute_tensor_dynamic(postact_scale_out, leading_dim=2)
        epi_kwargs["postact_bf16_trunc"] = bool(postact_bf16_trunc)
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

    _stream_obj = (
        current_stream
        if current_stream is not None
        else torch.cuda.current_stream()
    )
    _stream_raw = _raw_stream_id(_stream_obj)
    if _stream_raw is None:
        _stream_raw = _current_raw_stream_id()
    current_stream = cuda.CUstream(_stream_raw)

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
        bool(postact_bf16_trunc),
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_gated.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_kwargs = {"gather_A": gather_A}
        if device_capacity[0] >= 10:
            gemm_kwargs["sf_vec_size"] = sf_vec_size
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            **gemm_kwargs,
        )
        compile_args = (
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )
        if device_capacity[0] >= 10:
            compile_args += (a_scale_cute, b_scale_cute)
        cache[compile_key] = cute.compile(*compile_args)
    call_args = (
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )
    if device_capacity[0] >= 10:
        call_args += (a_scale_cute, b_scale_cute)
    cache[compile_key](*call_args)


from ..cache_manager import InstrumentedCompileCache as _ICC
gemm_gated.compile_cache = _ICC("gated")
