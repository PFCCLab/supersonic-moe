# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing import Optional

import torch

from .utils import is_using_quack_gemm


class FP8ActivationDType(Enum):
    E4M3 = "e4m3"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn


class FP8ScaleEncoding(Enum):
    E8M0 = "e8m0"

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, "float8_e8m0fnu", torch.uint8)


class FP8ScaleGranularity(Enum):
    BLOCK_1X32 = "1x32"
    BLOCK_1X128 = "1x128"

    @property
    def group_size(self) -> int:
        if self is FP8ScaleGranularity.BLOCK_1X32:
            return 32
        return 128


class FP8Backend(Enum):
    BLACKWELL = "blackwell"
    HOPPER_NATIVE = "hopper_native"


@dataclass(frozen=True)
class FP8Protocol:
    activation_dtype: FP8ActivationDType = FP8ActivationDType.E4M3
    scale_encoding: FP8ScaleEncoding = FP8ScaleEncoding.E8M0
    scale_granularity: FP8ScaleGranularity = FP8ScaleGranularity.BLOCK_1X128
    backend: FP8Backend = FP8Backend.BLACKWELL
    requires_quack_gemm: bool = True

    @property
    def activation_torch_dtype(self) -> torch.dtype:
        return self.activation_dtype.torch_dtype

    @property
    def scale_torch_dtype(self) -> torch.dtype:
        return self.scale_encoding.torch_dtype

    @property
    def group_size(self) -> int:
        return self.scale_granularity.group_size


@dataclass(frozen=True)
class FP8BackendResolution:
    """The single architecture/backend decision used by FP8 callers."""

    backend: FP8Backend
    capability_major: int

    @property
    def is_hopper(self) -> bool:
        return self.capability_major == 9

    @property
    def is_native_hopper(self) -> bool:
        return self.backend is FP8Backend.HOPPER_NATIVE

    @property
    def is_blackwell(self) -> bool:
        return self.capability_major >= 10

    @property
    def routing_block_size(self) -> int:
        # Both native Hopper profiles and the Blackwell blockscaled path use
        # the 128-row metadata contract.  The activation/weight profile remains
        # in the FP8 config and is not a routing decision.
        return 128


def _coerce_fp8_backend(value) -> FP8Backend | None:
    if value is None:
        return None
    if isinstance(value, FP8Backend):
        return value
    try:
        return FP8Backend(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported FP8 backend: {value!r}") from exc


def resolve_fp8_backend(
    backend=None,
    device: Optional[torch.device] = None,
) -> FP8BackendResolution:
    """Resolve an optional backend request against the actual CUDA device.

    Exact SM90 resolves to native Hopper and SM100+ defaults to Blackwell.
    Other architectures are rejected rather than silently entering a
    mismatched kernel path.
    """
    if backend is not None and hasattr(backend, "fp8_backend"):
        backend = getattr(backend, "fp8_backend")
    requested = _coerce_fp8_backend(backend)

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("FP8 backend resolution requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(f"FP8 backend resolution requires a CUDA device, got {device!r}")

    major, _ = torch.cuda.get_device_capability(device)
    if major == 9:
        default = FP8Backend.HOPPER_NATIVE
    elif major >= 10:
        default = FP8Backend.BLACKWELL
    else:
        raise RuntimeError(
            f"SonicMoE FP8 supports exact SM90 and SM100+ GPUs, got SM{major}"
        )

    resolved = default if requested is None else requested
    if resolved is FP8Backend.BLACKWELL and major < 10:
        raise RuntimeError("The Blackwell FP8 backend only supports SM100+ GPUs")
    if resolved is FP8Backend.HOPPER_NATIVE and major != 9:
        raise RuntimeError(
            f"The Hopper FP8 native backend requires an exact SM90 GPU, got SM{major}"
        )
    return FP8BackendResolution(resolved, major)


def get_default_fp8_protocol() -> FP8Protocol:
    return FP8Protocol()


def is_blackwell_device(device: Optional[torch.device] = None) -> bool:
    if device is None:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda", torch.cuda.current_device())

    if device.type != "cuda":
        return False

    major, _ = torch.cuda.get_device_capability(device)
    return major >= 10


def validate_fp8_protocol(protocol: FP8Protocol) -> FP8Protocol:
    if protocol.activation_dtype is not FP8ActivationDType.E4M3:
        raise ValueError("Only e4m3 activations are supported in the current FP8 protocol")

    if protocol.scale_encoding is not FP8ScaleEncoding.E8M0:
        raise ValueError("Only e8m0 scales are supported in the current FP8 protocol")

    if protocol.scale_granularity not in {
        FP8ScaleGranularity.BLOCK_1X32,
        FP8ScaleGranularity.BLOCK_1X128,
    }:
        raise ValueError("Only 1x32 and 1x128 scale granularities are supported in the current FP8 protocol")

    if protocol.backend not in {
        FP8Backend.BLACKWELL,
        FP8Backend.HOPPER_NATIVE,
    }:
        raise ValueError("Only Blackwell and Hopper native FP8 backends are supported")

    return protocol


def validate_fp8_runtime_support(
    protocol: FP8Protocol,
    device: Optional[torch.device] = None,
    quack_enabled: bool | None = None,
) -> FP8Protocol:
    validate_fp8_protocol(protocol)

    if not torch.cuda.is_available():
        raise RuntimeError("FP8 protocol requires CUDA")

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    resolve_fp8_backend(protocol.backend, device)

    if quack_enabled is None:
        quack_enabled = is_using_quack_gemm()

    if protocol.requires_quack_gemm and not quack_enabled:
        raise RuntimeError("The current FP8 protocol requires the QuACK GEMM path")

    return protocol
