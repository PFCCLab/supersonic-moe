from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_config import GemmConfig
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase

from ..cache_manager import InstrumentedCompileCache as _ICC


_MAX_FAST_PATH_ENTRIES = 64
_TORCH_TO_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.uint8: cutlass.Uint8,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def _make_cute_tensor_dynamic(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim,
    )


def _validate_16b_dynamic_alignment(name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is None:
        return
    elem_size = tensor.element_size()
    if tensor.data_ptr() % 16 != 0:
        raise RuntimeError(
            f"BF16 wgrad {name} data_ptr must be 16-byte aligned, "
            f"got data_ptr % 16 = {tensor.data_ptr() % 16}"
        )
    for dim, stride in enumerate(tensor.stride()):
        # Unit stride is the vectorized contiguous dimension; tile starts are
        # multiples of 128/256 elements.  Non-unit dynamic strides can shift the
        # base address of each copied row/expert and must preserve 16B alignment.
        if stride not in (0, 1) and (stride * elem_size) % 16 != 0:
            raise RuntimeError(
                f"BF16 wgrad {name}.stride({dim}) must preserve 16-byte alignment, "
                f"got stride={stride}, element_size={elem_size}, "
                f"byte_stride % 16 = {(stride * elem_size) % 16}"
            )


def _get_raw_cuda_stream(device=None) -> int:
    stream = torch.cuda.current_stream(device)
    if hasattr(stream, "stream_base"):
        return stream.stream_base.raw_stream
    return stream.cuda_stream


_COMPILE_CACHE_BF16_VK = _ICC("bf16_varlen_k")
_COMPILE_CACHE_BF16_VK_ACCUM = _ICC("bf16_varlen_k_accum")
_COMPILE_CACHE_BF16_VK_TMA_ADD = _ICC("bf16_varlen_k_tma_add")
_GEMM_FAST_PATH_BF16_VK: dict = {}
_GEMM_FAST_PATH_BF16_VK_ACCUM: dict = {}
_GEMM_FAST_PATH_BF16_VK_TMA_ADD: dict = {}


def _bf16_wgrad_default_config(
    device: torch.device, M: Optional[int] = None, N: Optional[int] = None,
) -> GemmConfig:
    cap = get_device_capacity(device)
    if cap[0] < 10:
        raise RuntimeError("bf16 wgrad direct varlen_k path requires SM100+")
    # A35B BF16 wgrad shapes are large enough to benefit from Blackwell 2CTA UMMA
    # along M.  gather_A requires cluster_n=1 in GemmSm100, so cluster_m is the
    # safe cluster dimension for this direct, non-materialized path.  Smaller or
    # unqualified shapes fall back to the previous conservative 1CTA tile.
    if M is not None and N is not None and M >= 4096 and N >= 2048:
        return GemmConfig(
            tile_m=256,
            tile_n=256,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            swap_ab=False,
            is_dynamic_persistent=True,
            max_swizzle_size=1,
            device_capacity=cap[0],
        )
    return GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        swap_ab=False,
        is_dynamic_persistent=True,
        max_swizzle_size=8,
        device_capacity=cap[0],
    )


def _prepare_tensors(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: Optional[torch.Tensor],
) -> dict[str, GemmTensorInfo]:
    _validate_16b_dynamic_alignment("A", A)
    _validate_16b_dynamic_alignment("B", B)
    _validate_16b_dynamic_alignment("D", D)
    _validate_16b_dynamic_alignment("C", C)
    tensor_infos = {
        "A": GemmTensorInfo(A),
        "B": GemmTensorInfo(B.mT),
        "D": GemmTensorInfo(D),
        "C": GemmTensorInfo(C),
    }
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_k=True)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for name, info in tensor_infos.items():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS_DTYPE[info.tensor.dtype]
            info.cute_tensor = _make_cute_tensor_dynamic(
                info.tensor,
                leading_dim=1 if info.major == major_configs[name][1] else 0,
            )
    return tensor_infos


def _run_bf16_wgrad_varlen_k(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    A_idx: torch.Tensor,
    *,
    out: torch.Tensor,
    C: Optional[torch.Tensor],
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device: torch.device,
    variant: str,
    compile_cache,
    fast_cache: dict,
    epi_args,
    config: Optional[GemmConfig] = None,
) -> torch.Tensor:
    expected_out_shape = (int(num_experts), int(M), int(N))
    if tuple(out.shape) != expected_out_shape:
        raise ValueError(
            "BF16 varlen-K wgrad output shape mismatch: expected "
            f"{expected_out_shape}, got {tuple(out.shape)}. "
            "A non-contiguous output view is supported, but its logical shape "
            "must remain [num_experts, M, N]."
        )
    if C is not None and tuple(C.shape) != expected_out_shape:
        raise ValueError(
            "BF16 varlen-K wgrad accumulator shape mismatch: expected "
            f"{expected_out_shape}, got {tuple(C.shape)}."
        )
    if config is None:
        config = _bf16_wgrad_default_config(device, M=M, N=N)
    fast_key = (
        variant,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.is_dynamic_persistent,
        config.max_swizzle_size,
        M,
        N,
        total_K,
        num_experts,
        A.dtype,
        B.dtype,
        out.dtype,
        C.dtype if C is not None else None,
        tuple(A.shape),
        tuple(B.shape),
        tuple(out.shape),
        tuple(C.shape) if C is not None else None,
        tuple(A.stride()),
        tuple(B.stride()),
        tuple(out.stride()),
        tuple(C.stride()) if C is not None else None,
        device.index if device.index is not None else -1,
    )
    cached = fast_cache.get(fast_key)
    tensor_infos = _prepare_tensors(A, B, out, C)

    if cached is not None:
        compiled, scheduler_args, cached_epi_args = cached
        varlen_args = GemmWrapperBase.create_varlen_args(
            cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx,
        )
        stream = cuda.CUstream(_get_raw_cuda_stream())
        compiled(
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            cached_epi_args,
            scheduler_args,
            varlen_args,
            stream,
            None,
            None,
        )
        return out

    tile_shape_mn = (config.tile_m, config.tile_n)
    cluster_shape_mnk = (config.cluster_m, config.cluster_n, 1)
    if not GemmDefaultSm100.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        Float32,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Unsupported BF16 wgrad type/major combination for varlen_k")

    max_active_clusters = get_max_active_clusters(config.cluster_m * config.cluster_n)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore=None,
        batch_idx_permute=None,
        max_swizzle_size=config.max_swizzle_size,
    )
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m=None, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx,
    )
    current_stream = cuda.CUstream(_get_raw_cuda_stream())

    compile_key = (
        variant,
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        tensor_infos["D"].dtype,
        tensor_infos["C"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
        tensor_infos["D"].major,
        tensor_infos["C"].major,
        tile_shape_mn,
        cluster_shape_mnk,
        M,
        N,
        config.pingpong,
        True,
        config.is_dynamic_persistent,
        config.device_capacity,
    )
    compiled = compile_cache.get(compile_key)
    if compiled is None:
        gemm_obj = GemmDefaultSm100(
            Float32,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=True,
            use_clc_persistence=config.is_dynamic_persistent,
        )
        compiled = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
            None,
            None,
        )
        compile_cache[compile_key] = compiled

    if len(fast_cache) > _MAX_FAST_PATH_ENTRIES:
        fast_cache.clear()
    fast_cache[fast_key] = (compiled, scheduler_args, epi_args)

    compiled(
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
        None,
        None,
    )
    return out


def bf16_wgrad_gemm_varlen_k(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    A_idx: torch.Tensor,
    *,
    out: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    return _run_bf16_wgrad_varlen_k(
        A,
        B,
        cu_seqlens_k,
        A_idx,
        out=out,
        C=None,
        M=M,
        N=N,
        total_K=total_K,
        num_experts=num_experts,
        device=device,
        variant="bf16_vk",
        compile_cache=_COMPILE_CACHE_BF16_VK,
        fast_cache=_GEMM_FAST_PATH_BF16_VK,
        epi_args=GemmDefaultSm100.EpilogueArguments(),
    )


def bf16_wgrad_gemm_varlen_k_accumulate(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    A_idx: torch.Tensor,
    *,
    accumulator: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device: torch.device,
) -> None:
    _run_bf16_wgrad_varlen_k(
        A,
        B,
        cu_seqlens_k,
        A_idx,
        out=accumulator,
        C=accumulator,
        M=M,
        N=N,
        total_K=total_K,
        num_experts=num_experts,
        device=device,
        variant="bf16_vk_accum",
        compile_cache=_COMPILE_CACHE_BF16_VK_ACCUM,
        fast_cache=_GEMM_FAST_PATH_BF16_VK_ACCUM,
        epi_args=GemmDefaultSm100.EpilogueArguments(beta=Float32(1.0)),
    )


def bf16_wgrad_gemm_varlen_k_tma_add(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    A_idx: torch.Tensor,
    *,
    accumulator: torch.Tensor,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
    device: torch.device,
) -> None:
    _run_bf16_wgrad_varlen_k(
        A,
        B,
        cu_seqlens_k,
        A_idx,
        out=accumulator,
        C=None,
        M=M,
        N=N,
        total_K=total_K,
        num_experts=num_experts,
        device=device,
        variant="bf16_vk_tma_add",
        compile_cache=_COMPILE_CACHE_BF16_VK_TMA_ADD,
        fast_cache=_GEMM_FAST_PATH_BF16_VK_TMA_ADD,
        epi_args=GemmDefaultSm100.EpilogueArguments(add_to_output=True),
    )
