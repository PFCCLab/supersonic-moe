"""Custom Triton blockscaled FP8 varlen GEMM for down-proj forward.

Replaces the slow CUTLASS ``GemmDefaultSm100`` path (6.8× slower than BF16)
with a Triton kernel that uses ``tl.dot_scaled`` — the hardware-native MXFP8
path on SM100 tensor cores.

Key design choices
------------------
* **Hardware-accelerated scaling** via ``tl.dot_scaled(a, a_scale, "e4m3",
  b, b_scale, "e4m3", acc=acc)`` — the SM100 WGMMA instruction handles
  per-32-element E8M0 block-scale dequantisation in hardware, zero software
  overhead.

* **Per-expert capacity scheduling**: Each expert gets ``max_m_tiles`` M-tile
  slots in the 1-D program grid.  Programs whose M-tile exceeds the expert's
  actual token count exit early.  For balanced MoE routing the waste is <5 %.

* **Raw E8M0 scales**: Scales are kept in their natural ``(M, K//32)`` uint8
  layout — no ISA repacking.

GEMM: C[m, n] = Σ_k A[m, k] * B[expert, n, k]
------
* A: ``(total_M, K)`` float8_e4m3fn  — activation in expert-sorted order
* A_scale: ``(total_M, K//32)`` uint8 (E8M0)
* B: ``(E, N, K)`` float8_e4m3fn  — weight, row-major contiguous
* B_scale: ``(E, N, K//32)`` uint8 (E8M0)
* C: ``(total_M, N)`` bfloat16
* cu_seqlens_m: ``(E+1,)`` int32  — expert boundaries in A's M dimension
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from ..functional.fp8_protocol import FP8Protocol, FP8ScaleGranularity
from ..functional.fp8_quant import quantize_activation_blockwise
from ..triton_utils import wrap_triton_kernel

_VEC_SIZE = 32  # blockscale granularity (E8M0 group size)


# ──────────────────────────────────────────────────────────────────────
# Triton kernel — hardware-native tl.dot_scaled
# ──────────────────────────────────────────────────────────────────────

def _prune_blockscaled_configs(configs, nargs, **kwargs):
    """Drop autotune configs where BLOCK_K > K (would read out of bounds)."""
    K = nargs["K"]
    return [c for c in configs if c.kwargs.get("BLOCK_K", 32) <= K]


@wrap_triton_kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    ],
    key=["N", "K"],
    prune_configs_by={"early_config_prune": _prune_blockscaled_configs},
)
@triton.jit
def _blockscaled_fp8_varlen_gemm_kernel(
    # Data pointers
    A_ptr,                  # (total_M, K) fp8
    B_ptr,                  # (E, N, K) fp8  — contiguous
    C_ptr,                  # (total_M, N) bf16 output
    # Scale pointers — raw E8M0 uint8
    A_scale_ptr,            # (total_M, K // 32) uint8
    B_scale_ptr,            # (E, N, K // 32) uint8
    # Varlen
    cu_seqlens_ptr,         # (E + 1,) int32
    # Dimensions
    total_M,
    N: tl.constexpr,
    K: tl.constexpr,
    num_experts,
    # Strides — A (total_M, K)
    stride_am, stride_ak,
    # Strides — B (E, N, K)
    stride_be, stride_bn, stride_bk,
    # Strides — C (total_M, N)
    stride_cm, stride_cn,
    # Strides — A_scale (total_M, K//32)
    stride_as_m, stride_as_g,
    # Strides — B_scale (E, N, K//32)
    stride_bs_e, stride_bs_n, stride_bs_g,
    # Scheduling
    max_expert_tokens,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    max_m_tiles = tl.cdiv(max_expert_tokens, BLOCK_M)
    tiles_per_expert = max_m_tiles * num_n_tiles

    expert_id = pid // tiles_per_expert
    if expert_id >= num_experts:
        return
    local_pid = pid % tiles_per_expert
    local_m_tile = local_pid // num_n_tiles
    local_n_tile = local_pid % num_n_tiles

    # Expert boundaries
    expert_start = tl.load(cu_seqlens_ptr + expert_id).to(tl.int64)
    expert_end = tl.load(cu_seqlens_ptr + expert_id + 1).to(tl.int64)

    global_m = expert_start + local_m_tile * BLOCK_M
    if global_m >= expert_end:
        return

    offs_m = global_m + tl.arange(0, BLOCK_M)
    offs_n = local_n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < expert_end
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    GROUPS_PER_BK: tl.constexpr = BLOCK_K // 32
    NUM_K_GROUPS: tl.constexpr = K // 32

    for k_start in tl.static_range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_offsets < K

        # Load A: (BLOCK_M, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=m_mask[:, None] & k_valid[None, :],
            other=0.0,
        )

        # Load B as (BLOCK_K, BLOCK_N): B stored as (E, N, K), read transposed
        b = tl.load(
            B_ptr + expert_id * stride_be
            + k_offsets[:, None] * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=k_valid[:, None] & n_mask[None, :],
            other=0.0,
        )

        # Load A scales: (BLOCK_M, GROUPS_PER_BK)
        g_start = k_start // 32
        g_offsets = g_start + tl.arange(0, GROUPS_PER_BK)
        g_valid = g_offsets < NUM_K_GROUPS
        a_scale = tl.load(
            A_scale_ptr + offs_m[:, None] * stride_as_m + g_offsets[None, :] * stride_as_g,
            mask=m_mask[:, None] & g_valid[None, :],
            other=0,
        )

        # Load B scales: (BLOCK_N, GROUPS_PER_BK)
        b_scale = tl.load(
            B_scale_ptr + expert_id * stride_bs_e
            + offs_n[:, None] * stride_bs_n
            + g_offsets[None, :] * stride_bs_g,
            mask=n_mask[:, None] & g_valid[None, :],
            other=0,
        )

        # Hardware-native MXFP8 scaled dot product
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc=acc)

    # Store result as bf16
    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptr + c_offs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ──────────────────────────────────────────────────────────────────────
# Weight pre-quantization (raw E8M0 scales)
# ──────────────────────────────────────────────────────────────────────

_RAW_WEIGHT_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...]],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def precompute_weight_fp8_raw_scales(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize expert weight for Triton blockscaled GEMM (raw E8M0 scales).

    Parameters
    ----------
    w : Tensor (H, I, E) bf16 — expert weights.

    Returns
    -------
    w_fp8 : Tensor (E, H, I) float8_e4m3fn — contiguous, quantized along I.
    w_scales : Tensor (E, H, I // 32) uint8 — raw E8M0 blockscales along I.
    """
    key = (
        w.data_ptr(),
        w._inplace_version(),
        tuple(w.shape),
        tuple(w.stride()),
    )
    cached = _RAW_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # (H, I, E) -> (E, H, I) contiguous.  Blockscales along last dim = I = K.
    w_ehi = w.permute(2, 0, 1).contiguous()
    proto = FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)
    w_fp8, w_scales_e8m0 = quantize_activation_blockwise(w_ehi, proto)
    # e8m0fnu -> uint8 view (same bits)
    w_scales_raw = w_scales_e8m0.view(torch.uint8).contiguous()

    result = (w_fp8, w_scales_raw)
    if len(_RAW_WEIGHT_CACHE) > 8:
        _RAW_WEIGHT_CACHE.clear()
    _RAW_WEIGHT_CACHE[key] = result
    return result


def clear_raw_weight_cache() -> None:
    """Clear the raw-scale weight cache (called on optimizer step)."""
    _RAW_WEIGHT_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Activation quantization (raw E8M0 scales)
# ──────────────────────────────────────────────────────────────────────

def quantize_activation_raw(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation to blockscaled FP8 with raw E8M0 scales.

    Parameters
    ----------
    x : Tensor (M, K) bf16 — contiguous activation.

    Returns
    -------
    x_fp8 : Tensor (M, K) float8_e4m3fn
    x_scales : Tensor (M, K // 32) uint8 — raw E8M0 blockscales.
    """
    proto = FP8Protocol(scale_granularity=FP8ScaleGranularity.BLOCK_1X32)
    x_fp8, x_scales_e8m0 = quantize_activation_blockwise(x.contiguous(), proto)
    return x_fp8, x_scales_e8m0.view(torch.uint8).contiguous()


# ──────────────────────────────────────────────────────────────────────
# Python entry point
# ──────────────────────────────────────────────────────────────────────

def blockscaled_fp8_gemm_varlen_triton(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockscaled FP8 varlen GEMM using custom Triton kernel.

    Computes  C[m, n] = Σ_k A[m, k] * B[expert(m), n, k]
    with per-32-element E8M0 blockscales along K for both A and B.

    Parameters
    ----------
    a_fp8 : (total_M, K) float8_e4m3fn — activation, expert-sorted.
    a_scales : (total_M, K // 32) uint8 — raw E8M0 activation scales.
    w_fp8 : (E, N, K) float8_e4m3fn — weight, contiguous.
    w_scales : (E, N, K // 32) uint8 — raw E8M0 weight scales.
    cu_seqlens_m : (E + 1,) int32 — expert boundaries.
    out_dtype : output dtype (default bfloat16).

    Returns
    -------
    C : (total_M, N) ``out_dtype``
    """
    assert a_fp8.ndim == 2 and a_fp8.dtype == torch.float8_e4m3fn
    assert w_fp8.ndim == 3 and w_fp8.dtype == torch.float8_e4m3fn
    assert a_scales.ndim == 2 and a_scales.dtype == torch.uint8
    assert w_scales.ndim == 3 and w_scales.dtype == torch.uint8

    total_M, K = a_fp8.shape
    E, N, K_w = w_fp8.shape
    assert K == K_w, f"K mismatch: a has {K}, w has {K_w}"
    assert K % _VEC_SIZE == 0, f"K must be multiple of {_VEC_SIZE}"

    num_experts = E
    assert cu_seqlens_m.shape[0] == num_experts + 1

    # Output
    C = torch.empty(total_M, N, dtype=out_dtype, device=a_fp8.device)

    if total_M == 0:
        return C

    # Determine max expert tokens for scheduling
    if not torch.cuda.is_current_stream_capturing():
        expert_tokens = cu_seqlens_m[1:] - cu_seqlens_m[:-1]
        max_expert_tokens = int(expert_tokens.max().item())
    else:
        max_expert_tokens = total_M

    if max_expert_tokens == 0:
        return C

    # Grid: each expert gets ceil(max_expert_tokens/BLOCK_M) * ceil(N/BLOCK_N) slots
    def grid(META):
        bm = META["BLOCK_M"]
        bn = META["BLOCK_N"]
        mmt = math.ceil(max_expert_tokens / bm)
        nnt = math.ceil(N / bn)
        return (num_experts * mmt * nnt,)

    _blockscaled_fp8_varlen_gemm_kernel[grid](
        a_fp8, w_fp8, C,
        a_scales, w_scales,
        cu_seqlens_m,
        total_M, N, K, num_experts,
        # A strides
        a_fp8.stride(0), a_fp8.stride(1),
        # B strides
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        # C strides
        C.stride(0), C.stride(1),
        # A_scale strides
        a_scales.stride(0), a_scales.stride(1),
        # B_scale strides
        w_scales.stride(0), w_scales.stride(1), w_scales.stride(2),
        # Scheduling
        max_expert_tokens=max_expert_tokens,
    )

    return C
