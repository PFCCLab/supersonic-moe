from __future__ import annotations

from typing import Optional
import sys

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_config import GemmConfig
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_interface import gemm, gemm_add
from quack.gemm_interface import register_gemm_pre_dispatch_hook
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase

from .gemm_sm90_scaled import sm90_scaled_wgrad_varlen_k
from .sm90_fp8_runtime import (
    current_sm90_invocation,
    find_sm90_invocation_payload,
    find_sm90_operand_invocations,
    find_sm90_storage_carriers,
    get_pending_sm90_payload,
    get_sm90_carrier,
    is_sm90_native_active,
    is_sm90_native_carrier,
    pop_pending_sm90_payload,
    register_sm90_carrier,
    resolve_sm90_metadata_tensor,
    retained_sm90_invocations,
    unique_retained_sm90_invocation,
    use_sm90_invocation,
    wrap_deferred_sm90_call,
)


def _sm90_colwise_operand(
    tensor: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    source: torch.Tensor | None = None,
    gather_idx: torch.Tensor | None = None,
    invocation=None,
):
    carrier_tensor = tensor
    carrier = get_sm90_carrier(carrier_tensor, invocation=invocation)
    if carrier is None and source is not None:
        carrier_tensor = source
        carrier = get_sm90_carrier(carrier_tensor, invocation=invocation)
    functional_package = next(
        name
        for name in (
            __package__.split(".quack_utils", 1)[0] + ".functional",
            "paddlefleet_ops.sonicmoe.functional",
        )
        if name in sys.modules
    )
    functional = sys.modules[functional_package]
    source = tensor if source is None else source
    return functional.sm90_reference.quantize_wgrad_colwise_sm90(
        source, cu_seqlens_k, gather_idx=gather_idx
    )

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


def _validate_problem(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    A_idx: torch.Tensor,
    out: torch.Tensor,
    C: Optional[torch.Tensor],
    *,
    M: int,
    N: int,
    total_K: int,
    num_experts: int,
) -> None:
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
    if A.ndim != 2 or A.shape[0] != int(M):
        raise ValueError(
            f"BF16 varlen-K wgrad A must have shape [M, T] with M={int(M)}, "
            f"got {tuple(A.shape)}"
        )
    if B.ndim != 2 or tuple(B.shape) != (int(total_K), int(N)):
        raise ValueError(
            "BF16 varlen-K wgrad B must have shape [total_K, N], expected "
            f"{(int(total_K), int(N))}, got {tuple(B.shape)}"
        )
    if A_idx.ndim != 1 or A_idx.numel() != int(total_K):
        raise ValueError(
            "BF16 varlen-K wgrad A_idx must be route/total_K-sized [total_K], "
            f"expected {int(total_K)} elements, got shape {tuple(A_idx.shape)}"
        )
    if cu_seqlens_k.ndim != 1 or cu_seqlens_k.numel() != int(num_experts) + 1:
        raise ValueError(
            "BF16 varlen-K wgrad cu_seqlens_k must have num_experts + 1 elements, "
            f"got shape {tuple(cu_seqlens_k.shape)}"
        )
    if A_idx.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("BF16 varlen-K wgrad A_idx and cu_seqlens_k must be int32")


from .gemm_sm90_scaled import sm90_scaled_gemm


def _sm90_gemm_pre_dispatch(operation, original, args, kwargs):
    """Route exact-SM90 down/actgrad calls carrying native FP8 payloads."""
    if operation not in {"gemm_out", "gemm_add_out"} or len(args) < 3:
        return None
    A, B, out = args[:3]
    # The hook is process-global, but the carrier registry is Hopper-only.
    # Keep Blackwell's normal QuACK compile/launch path fully isolated from it.
    if get_device_capacity(A.device)[0] != 9:
        return None
    a_carrier = get_sm90_carrier(A)
    if a_carrier is None:
        invocation = unique_retained_sm90_invocation()
        if invocation is not None:
            a_carrier = get_sm90_carrier(A, invocation=invocation)
    b_carrier = get_sm90_carrier(B)
    if a_carrier is not None and b_carrier is None:
        invocation = a_carrier.metadata.get("invocation")
        if invocation is not None:
            b_carrier = get_sm90_carrier(B, invocation=invocation)
            if b_carrier is None:
                a_carrier.materialize()
                if a_carrier.data is None:
                    return None
                a_k = int(a_carrier.data.shape[-1])
                candidates = [
                    candidate
                    for candidate in find_sm90_storage_carriers(B)
                    if candidate.invocation_id == a_carrier.invocation_id
                    and candidate.data is not None
                    and int(candidate.data.shape[-1]) == a_k
                ]
                if len(candidates) == 1:
                    b_carrier = candidates[0]
    if b_carrier is not None and a_carrier is None:
        invocation = b_carrier.metadata.get("invocation")
        if invocation is not None:
            a_carrier = get_sm90_carrier(A, invocation=invocation)
    if a_carrier is None or b_carrier is None:
        return None
    if not (
        is_sm90_native_active()
        or (is_sm90_native_carrier(a_carrier) and is_sm90_native_carrier(b_carrier))
    ):
        return None
    if kwargs.get("cu_seqlens_k") is not None:
        return None
    a_carrier.materialize()
    b_carrier.materialize()
    if a_carrier.data is None or b_carrier.data is None:
        # Wgrad-only carriers own independent colwise payloads and are not valid
        # operands for the rowwise actgrad pre-dispatch hook.
        return None
    if A.ndim != 2 or B.ndim != 3:
        return None
    a_k = int(a_carrier.data.shape[-1])
    candidates = [
        candidate
        for candidate in find_sm90_storage_carriers(B)
        if candidate.invocation_id == a_carrier.invocation_id
        and candidate.data is not None
        and int(candidate.data.shape[-1]) == a_k
    ]
    if int(b_carrier.data.shape[-1]) != a_k:
        if len(candidates) != 1:
            return None
        b_carrier = candidates[0]
    C = kwargs.get("C") if operation == "gemm_out" else args[3]
    beta = kwargs.get("beta", 1.0)
    add_to_output = bool(kwargs.get("add_to_output", False))
    if kwargs.get("bias") is not None or kwargs.get("alpha", 1.0) != 1.0:
        return None
    invocation = a_carrier.metadata["invocation"]
    activation_group = int(a_carrier.metadata.get("group_size", 32))
    weight_block = int(b_carrier.metadata.get("block_size", 32))
    if activation_group != weight_block:
        a_carrier.materialize()
        functional_package = next(
            name
            for name in (
                __package__.split(".quack_utils", 1)[0] + ".functional",
                "paddlefleet_ops.sonicmoe.functional",
            )
            if name in sys.modules
        )
        a_source = resolve_sm90_metadata_tensor(
            a_carrier.metadata, "source", required=False
        )
        if a_source is None:
            a_source = A
        a_data, a_scales, _ = sys.modules[
            functional_package
        ].sm90_reference.quantize_activation_sm90(a_source, weight_block)
        activation_group = weight_block
    else:
        a_data, a_scales = a_carrier.data, a_carrier.scales
    kernel_out = out
    copy_back = not out.is_contiguous()
    if copy_back:
        if add_to_output:
            kernel_out = out.contiguous()
        else:
            kernel_out = out.contiguous()
    kernel_c = C
    if C is not None and not C.is_contiguous():
        kernel_c = C.contiguous()
    with use_sm90_invocation(invocation):
        sm90_scaled_gemm(
            a_data,
            b_carrier.data,
            kernel_out,
            a_scales,
            b_carrier.scales,
            C=kernel_c,
            beta=beta,
            add_to_output=add_to_output,
            cu_seqlens_m=kwargs.get("cu_seqlens_m"),
            A_idx=kwargs.get("A_idx"),
            activation_group=activation_group,
            weight_block=weight_block,
        )
    if copy_back:
        out.copy_(kernel_out)
    return out


register_gemm_pre_dispatch_hook(_sm90_gemm_pre_dispatch)


def _carrier_from_pending(tensor: torch.Tensor, invocation):
    pending = pop_pending_sm90_payload(
        tensor, invocation=invocation, required=True
    )
    if pending is None:
        return None
    data, scales, group_size, role, metadata = pending
    source = resolve_sm90_metadata_tensor(metadata, "source", required=False)
    if source is None:
        source = tensor
    orientation = "transposed" if tuple(tensor.shape) == tuple(source.mT.shape) else "identity"
    return register_sm90_carrier(
        tensor,
        data=data,
        scales=scales,
        role=role,
        orientation=orientation,
        metadata={"group_size": group_size, **metadata},
        invocation=invocation,
    )


def _resolve_sm90_wgrad_invocation(A: torch.Tensor, B: torch.Tensor):
    active = current_sm90_invocation()
    if active is not None:
        return active
    operand_invocations = []
    for tensor in (A, getattr(A, "mT", None), B):
        if tensor is None:
            continue
        invocations = find_sm90_operand_invocations(tensor)
        if invocations:
            operand_invocations.append(invocations)
    candidate_ids = {
        invocation.invocation_id
        for invocations in operand_invocations
        for invocation in invocations
    }
    if len(candidate_ids) == 1:
        target = next(iter(candidate_ids))
        return next(
            invocation
            for invocations in operand_invocations
            for invocation in invocations
            if invocation.invocation_id == target
        )
    if candidate_ids:
        return None
    retained = retained_sm90_invocations()
    compatible = tuple(
        invocation
        for invocation in retained
        if int(invocation.activation_group_size) in (32, 128)
    )
    return compatible[-1] if compatible else None


def _resolve_sm90_wgrad_pair(A: torch.Tensor, B: torch.Tensor):
    A_t = getattr(A, "mT", None)
    invocation = _resolve_sm90_wgrad_invocation(A, B)
    if invocation is not None:
        a_carrier = get_sm90_carrier(A, invocation=invocation)
        if a_carrier is None and A_t is not None:
            a_carrier = get_sm90_carrier(A_t, invocation=invocation)
        b_carrier = get_sm90_carrier(B, invocation=invocation)
        pending_a = get_pending_sm90_payload(A, invocation=invocation, required=True)
        pending_a_tensor = A
        if pending_a is None and A_t is not None:
            pending_a = get_pending_sm90_payload(
                A_t, invocation=invocation, required=True
            )
            pending_a_tensor = A_t
        pending_b = get_pending_sm90_payload(B, invocation=invocation, required=True)
        if a_carrier is None and pending_a is not None:
            a_carrier = _carrier_from_pending(pending_a_tensor, invocation)
        if a_carrier is None:
            a_source = A_t if A_t is not None else A
            a_carrier = register_sm90_carrier(
                A,
                role="wgrad_activation",
                orientation="identity",
                metadata={"source": A.mT, "group_size": 32},
                invocation=invocation,
            )
        if b_carrier is None and pending_b is not None:
            b_carrier = _carrier_from_pending(B, invocation)
        if b_carrier is None:
            b_carrier = register_sm90_carrier(
                B,
                role="wgrad_gradient",
                metadata={"source": B, "group_size": 32},
                invocation=invocation,
            )
        return a_carrier, b_carrier
    a_carrier = get_sm90_carrier(A)
    if a_carrier is None and A_t is not None:
        a_carrier = get_sm90_carrier(A_t)
    b_carrier = get_sm90_carrier(B)
    return a_carrier, b_carrier


def _run_bf16_wgrad_sm90(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    A_idx: torch.Tensor,
    *,
    out: torch.Tensor,
    C: Optional[torch.Tensor],
    add_to_output: bool,
    beta: float = 1.0,
) -> torch.Tensor:
    a_carrier, b_carrier = _resolve_sm90_wgrad_pair(A, B)
    if a_carrier is None or b_carrier is None:
        raise RuntimeError(
            "SM90 varlen-K wgrad lost its invocation-scoped native FP8 "
            "payload; BF16/QuACK fallback is forbidden"
        )
    if not (
        is_sm90_native_active()
        or (is_sm90_native_carrier(a_carrier) and is_sm90_native_carrier(b_carrier))
    ):
        raise RuntimeError("SM90 varlen-K wgrad requires native FP8 carriers")
    invocation = a_carrier.metadata["invocation"]
    a_source = A.mT
    b_source = B
    a_payload = None
    if int(a_source.stride(0)) == 0 and int(a_source.stride(1)) == 0:
        expected_a_shape = (int(A.shape[1]), int(A.shape[0]))
        native_a = (
            a_carrier
            if a_carrier.data is not None
            and tuple(a_carrier.data.shape) == expected_a_shape
            else find_sm90_invocation_payload(
                invocation,
                expected_a_shape,
                ("activation", "communication"),
            )
        )
        if native_a is None:
            raise RuntimeError("SM90 native wgrad lost its A activation payload")
        functional_package = next(
            name
            for name in (
                __package__.split(".quack_utils", 1)[0] + ".functional",
                "paddlefleet_ops.sonicmoe.functional",
            )
            if name in sys.modules
        )
        functional = sys.modules[functional_package]
        a_payload = functional.sm90_reference.quantize_wgrad_colwise_sm90(
            native_a.data,
            cu_seqlens_k,
            gather_idx=A_idx,
            source_scales=native_a.scales,
            source_group=int(native_a.metadata.get("group_size", 32)),
        )
    b_payload = None
    if int(b_source.stride(0)) == 0 and int(b_source.stride(1)) == 0:
        expected_b_shape = tuple(map(int, B.shape))
        native_b = (
            b_carrier
            if b_carrier.data is not None
            and tuple(b_carrier.data.shape) == expected_b_shape
            else find_sm90_invocation_payload(
                invocation,
                expected_b_shape,
                ("dgated_output", "y1", "communication"),
            )
        )
        if native_b is None:
            raise RuntimeError("SM90 native wgrad lost its B activation payload")
        functional_package = next(
            name
            for name in (
                __package__.split(".quack_utils", 1)[0] + ".functional",
                "paddlefleet_ops.sonicmoe.functional",
            )
            if name in sys.modules
        )
        functional = sys.modules[functional_package]
        b_payload = functional.sm90_reference.quantize_wgrad_colwise_sm90(
            native_b.data,
            cu_seqlens_k,
            source_scales=native_b.scales,
            source_group=int(native_b.metadata.get("group_size", 32)),
        )
    if a_payload is None:
        a_payload = _sm90_colwise_operand(
        A.mT,
        cu_seqlens_k,
        source=a_source,
        gather_idx=A_idx,
        invocation=invocation,
    )
    if b_payload is None:
        b_payload = _sm90_colwise_operand(
            B, cu_seqlens_k, source=b_source, invocation=invocation
        )
    if a_payload is None or b_payload is None:
        raise RuntimeError(
            "SM90 varlen-K wgrad requires independent colwise carrier payloads; "
            "BF16 fallback is forbidden"
        )
    a_data, a_scales, a_packed_offsets = a_payload
    b_data, b_scales, b_packed_offsets = b_payload
    if not bool((a_packed_offsets == b_packed_offsets).all().item()):
        raise RuntimeError("SM90 wgrad operands produced different packed expert offsets")
    kernel_out = out
    copy_back = not out.is_contiguous()
    if copy_back:
        # Paddle's ``contiguous()`` tensor does not preserve aliasing with a
        # permuted view, and ``view.copy_(...)`` may rebind that view instead of
        # scattering into its base storage. The SM90 direct-main-grad callers
        # pass a swap of the last two dimensions, so reconstruct the base from
        # that view before copying the contiguous kernel result back.
        if out.ndim != 3:
            raise RuntimeError(
                "SM90 wgrad non-contiguous output must be a rank-3 transpose view"
            )
        base_out = out.permute(0, 2, 1)
        kernel_out = out.contiguous()
    else:
        base_out = None
    kernel_c = C
    if C is not None and not C.is_contiguous():
        kernel_c = C.contiguous()
    with use_sm90_invocation(invocation):
        sm90_scaled_wgrad_varlen_k(
            a_data,
            b_data,
            kernel_out,
            a_scales,
            b_scales,
            a_packed_offsets,
            C=kernel_c,
            beta=None if add_to_output else (float(beta) if C is not None else 0.0),
            add_to_output=add_to_output,
            activation_group=32,
            weight_block=32,
        )
    if copy_back:
        base_out.copy_(kernel_out.permute(0, 2, 1).contiguous())
    return base_out if copy_back else out


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
    add_to_output: bool = False,
    config: Optional[GemmConfig] = None,
) -> torch.Tensor:
    _validate_problem(
        A,
        B,
        cu_seqlens_k,
        A_idx,
        out,
        C,
        M=M,
        N=N,
        total_K=total_K,
        num_experts=num_experts,
    )
    device_capacity = get_device_capacity(device)
    if device_capacity[0] == 9:
        invocation = _resolve_sm90_wgrad_invocation(A, B)
        a_carrier, b_carrier = _resolve_sm90_wgrad_pair(A, B)
        native_expected = a_carrier is not None or b_carrier is not None
        if native_expected:
            if invocation is None:
                raise RuntimeError(
                    "SM90 native wgrad operands span multiple live invocations; "
                    "BF16/QuACK fallback is forbidden"
                )
            if a_carrier is None or b_carrier is None:
                raise RuntimeError(
                    "SM90 varlen-K wgrad lost its invocation-scoped native FP8 "
                    "payload; BF16/QuACK fallback is forbidden"
                )
            return _run_bf16_wgrad_sm90(
                A,
                B,
                cu_seqlens_k,
                A_idx,
                out=out,
                C=None if add_to_output else C,
                add_to_output=add_to_output,
                beta=1.0 if C is not None else 0.0,
            )
        call = gemm_add if add_to_output else gemm
        kwargs = dict(
            out=out,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            dynamic_scheduler=False,
            tuned=False,
        )
        if C is not None or add_to_output:
            kwargs.update(C=out if C is None else C, beta=1.0)
        return call(A, B, **kwargs)
    if device_capacity[0] < 10:
        raise RuntimeError("BF16 varlen-K wgrad requires SM90+")
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


def wrap_sm90_wgrad_call(callback, *tensors: torch.Tensor):
    """Keep native payloads alive when a host executes wgrad after backward."""
    carriers = []
    for tensor in tensors:
        carriers.extend(find_sm90_storage_carriers(tensor))
    unique = tuple({id(carrier): carrier for carrier in carriers}.values())
    invocation_ids = {
        invocation.invocation_id
        for tensor in tensors
        for invocation in find_sm90_operand_invocations(tensor)
    }
    if len(invocation_ids) > 1:
        raise RuntimeError(
            "deferred SM90 wgrad operands span multiple microbatch invocations"
        )
    return wrap_deferred_sm90_call(callback, *unique)


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
        add_to_output=True,
    )
