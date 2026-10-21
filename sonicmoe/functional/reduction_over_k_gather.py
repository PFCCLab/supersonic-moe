# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils import get_powers_of_2
from ..triton_utils import deterministic_autotune_enabled, wrap_triton_kernel


### This triton impl is equivalent as the cute-dsl impl shown above,
# and also achieves similar memory bandwidth on H100 for large K and H.
# However, for small K and H, this impl is better by autotuning so we use it as the default.
def _get_triton_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_H in get_powers_of_2(256, 4096):
        for BLOCK_K in get_powers_of_2(1, 128):
            for num_warps in [4, 8]:
                if BLOCK_K * BLOCK_H <= 32768:
                    configs.append(
                        triton.Config({"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K}, num_warps=num_warps, num_stages=4)
                    )
    return configs


def _deterministic_config(configs: list[triton.Config]) -> triton.Config:
    """Pick one config without benchmarking, so the result is bit-reproducible.

    BLOCK_H is maximised because ``x`` is re-read once per h-tile, and BLOCK_K
    is minimised so K is accumulated strictly sequentially. On B300 with the
    production shapes (T=4096, MAX_K=10, H in {1024, 2048}) this lands within
    8% of the autotuned winner.
    """
    return min(
        configs,
        key=lambda c: (-c.kwargs["BLOCK_H"], c.kwargs["BLOCK_K"], c.num_warps, c.num_stages),
    )


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        BLOCK_H = c.kwargs["BLOCK_H"]
        BLOCK_K = c.kwargs["BLOCK_K"]
        H = kw["H"]
        MAX_K = kw["MAX_K"]
        # A tile spanning the whole row is never "too small": for H < 1024 the
        # tile-size floor alone would drop BLOCK_K=1, which is the fastest
        # choice for low average K (EP: ~1.3 local experts per token). With it
        # gone the tuner picks BLOCK_K>=2, whose cross-warp K reduction runs
        # ~4.7x slower on H800 at H=512.
        full_row = BLOCK_H == min(triton.next_power_of_2(H), 4096)
        if (
            BLOCK_H <= triton.next_power_of_2(H)
            and BLOCK_K <= triton.next_power_of_2(MAX_K)
            and (full_row or min(H * MAX_K, 1024) <= (BLOCK_H * BLOCK_K))
        ):
            pruned_configs.append(c)

    if len(pruned_configs) == 0:
        pruned_configs = configs

    if deterministic_autotune_enabled():
        return [_deterministic_config(pruned_configs)]
    return pruned_configs


@wrap_triton_kernel
@triton.autotune(
    configs=_get_triton_autotune_configs(),
    # SKIP_SINGLE is deliberately not in the key: the fused path must reuse the
    # unfused config (same BLOCK_K -> same multi-route summation order).
    key=["H", "MAX_K", "w_is_None", "is_varlen_K"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def token_gather_sum_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    M_perm_ptr,  # (Mtotal,) int32
    M_offset_ptr,  # (T+1,)   int32
    out_ptr,  # (T, H)
    T,
    H: tl.constexpr,
    MAX_K: tl.constexpr,
    # strides
    stride_xM: tl.constexpr,
    stride_xH: tl.constexpr,
    stride_outT: tl.constexpr,
    stride_outH: tl.constexpr,
    # tile sizes
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,
    is_varlen_K: tl.constexpr,
    SAFE_INT64: tl.constexpr = False,
    SKIP_SINGLE: tl.constexpr = False,
):
    # 1D tiling over T only
    pid_t = tl.program_id(axis=0)
    if SAFE_INT64:
        t_idx = pid_t.to(tl.int64)
    else:
        t_idx = pid_t.to(tl.uint32)

    # Load segment starts and ends for this token
    if is_varlen_K:
        if SAFE_INT64:
            Ms = tl.load(M_offset_ptr + t_idx).to(tl.int64)
            Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.int64)
        else:
            Ms = tl.load(M_offset_ptr + t_idx).to(tl.uint32)
            Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.uint32)
        K_this_token = Me - Ms  # actual K for this token
        if SKIP_SINGLE:
            # Single-route tokens were already written by the GEMM epilogue
            # (route scatter); K == 0 tokens still get their zeros here.
            single = K_this_token == 1
            K_this_token = tl.where(single, 0, K_this_token)
    else:
        Ms = MAX_K * t_idx
        K_this_token: tl.constexpr = MAX_K

    # Outer loop over H tiles
    for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
        if SAFE_INT64:
            h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
        else:
            h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.uint32)  # [BLOCK_H]
        m_h = h_idx < H

        # Initialize accumulator for this H tile
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)  # [BLOCK_H]

        # Inner loop over K tiles
        for k_tile in tl.range(tl.cdiv(K_this_token, BLOCK_K)):
            k_offset = k_tile * BLOCK_K

            if SAFE_INT64:
                k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.int64)
            else:
                k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.uint32)  # [BLOCK_K]

            # Mask for valid K indices
            m_k = k_idx < K_this_token  # [BLOCK_K]

            # Absolute positions into M_perm and w
            m_abs = Ms + k_idx  # [BLOCK_K]

            # Gather permuted indices
            if SAFE_INT64:
                perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(tl.int64)
            else:
                perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(tl.uint32)  # [BLOCK_K]

            # Load x values: [BLOCK_K, BLOCK_H]
            x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
            x_mask = m_k[:, None] & m_h[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            # Reduce along K dimension and add to accumulator
            if w_is_None:
                acc += tl.sum(x_vals, axis=0)  # [BLOCK_H]
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=m_k, other=0.0).to(tl.float32)  # [BLOCK_K]
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)  # [BLOCK_H]

        # Store final result for this H tile (only once!)
        out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
        if SKIP_SINGLE:
            tl.store(out_ptrs, acc, mask=m_h & ~single)
        else:
            tl.store(out_ptrs, acc, mask=m_h)


@triton.jit
def token_gather_sum_small_h_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    M_perm_ptr,  # (Mtotal,) int32
    M_offset_ptr,  # (T+1,) int32
    out_ptr,  # (T, H)
    T,
    stride_xM,
    stride_outT,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    w_is_None: tl.constexpr,
    SKIP_SINGLE: tl.constexpr,
):
    # BLOCK_T tokens x one H tile per program (the 1-token / 4-warp kernel is
    # launch- and latency-bound at H <= 512). Each token still adds its routes
    # one at a time in k order, so the result is bitwise equal to BLOCK_K=1.
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    m_t = t < T
    Ms = tl.load(M_offset_ptr + t, mask=m_t, other=0)
    Me = tl.load(M_offset_ptr + t + 1, mask=m_t, other=0)
    kt = Me - Ms
    if SKIP_SINGLE:
        single = kt == 1
        kt = tl.where(single, 0, kt)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m_h = h < H
    acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
    for k in tl.range(0, tl.max(kt, axis=0)):
        m_k = k < kt
        p = tl.load(M_perm_ptr + Ms + k, mask=m_k, other=0).to(tl.int64)
        v = tl.load(x_ptr + p[:, None] * stride_xM + h[None, :], mask=m_k[:, None] & m_h[None, :],
                    other=0.0).to(tl.float32)
        if not w_is_None:
            v = v * tl.load(w_ptr + Ms + k, mask=m_k, other=0.0).to(tl.float32)[:, None]
        acc = tl.where(m_k[:, None], acc + v, acc)
    m_o = m_t[:, None] & m_h[None, :]
    if SKIP_SINGLE:
        m_o = m_o & ~single[:, None]
    tl.store(out_ptr + t[:, None].to(tl.int64) * stride_outT + h[None, :], acc, mask=m_o)


def _small_h_config(H: int):
    """(BLOCK_T, BLOCK_H, num_warps) of the small-H kernel, or None.

    H800 microbench (T=414K / 105K, avg K ~1.9, bitwise vs BLOCK_K=1):
    H=256 351->230 us, H=384 390->330 us, H=512 128->112 us; H=768 only -3%.
    """
    if os.environ.get("SONIC_MOE_TGS_SMALL_H", "1") != "1" or H > 512:
        return None
    if H <= 256:
        return 2, triton.next_power_of_2(H), 1
    return 1, triton.next_power_of_2(H), 2


@triton.jit
def route_scatter_dst_kernel(
    gather_ptr,  # (rows,) int32: route row -> token (any value on pad rows)
    M_perm_ptr,  # (Mtotal,) int32: token-major position -> route row
    M_offset_ptr,  # (T+1,) int32
    w_ptr,  # (Mtotal,) fp32 token-major scores (unused if w_is_None)
    dst_ptr,  # (rows,) int32
    w_out_ptr,  # (rows,) fp32 route-order scores of single-route tokens
    rows,
    T,
    BLOCK: tl.constexpr,
    w_is_None: tl.constexpr,
):
    # One lane per route row, so every row is written (no -1 prefill launch).
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m_p = p < rows
    t = tl.load(gather_ptr + p, mask=m_p, other=-1)
    m_t = m_p & (t >= 0) & (t < T)
    Ms = tl.load(M_offset_ptr + t, mask=m_t, other=0)
    Me = tl.load(M_offset_ptr + t + 1, mask=m_t, other=0)
    single = m_t & (Me - Ms == 1)
    # Pad rows may alias a single-route token; only its own route row matches.
    single = single & (tl.load(M_perm_ptr + Ms, mask=single, other=-1) == p)
    tl.store(dst_ptr + p, tl.where(single, t, -1).to(tl.int32), mask=m_p)
    if not w_is_None:
        tl.store(w_out_ptr + p, tl.load(w_ptr + Ms, mask=single, other=0.0).to(tl.float32),
                 mask=m_p)


def route_scatter_dst(
    gather_idx: torch.Tensor,
    M_perm: torch.Tensor,
    M_offset: torch.Tensor,
    rows: int,
    T: int,
    w: Optional[torch.Tensor] = None,
):
    """Route-row destinations of the fused single-route token reduction.

    ``dst[p] = t`` if route row ``p`` is the only local route of token ``t``,
    else ``-1`` (multi-route tokens and alignment pads). With ``w`` (token-major
    scores) also returns ``w_route[p] = w[M_offset[t]]`` for those rows (0
    elsewhere). ``gather_idx`` (``>= rows`` entries) maps route rows to tokens.
    """
    dst = torch.empty((rows,), dtype=torch.int32, device=M_perm.device)
    w_route = None if w is None else torch.empty_like(dst, dtype=torch.float32)
    BLOCK = 1024
    route_scatter_dst_kernel[((rows + BLOCK - 1) // BLOCK,)](
        gather_idx, M_perm, M_offset, dst if w is None else w, dst,
        dst if w is None else w_route, rows, T, BLOCK=BLOCK, w_is_None=w is None,
    )
    return dst, w_route


_SKIP_SINGLE_TUNED = set()


def prepare_skip_single(
    x: torch.Tensor,  # (rows, H) route-row GEMM output, may be uninitialized
    w: Optional[torch.Tensor],
    out: torch.Tensor,  # (T, H)
    gather_idx: torch.Tensor,  # (rows,) int32 route row -> token
    M_perm: torch.Tensor,
    M_offset: torch.Tensor,
    T: int,
    MAX_K: int,
    H: int,
):
    """Set up the fused single-route reduction; call before the scatter GEMM.

    Returns :func:`route_scatter_dst`. On the first call per shape it also runs
    the unfused reduce once so the autotuner picks its config on the unfused
    workload; the later ``skip_single`` reduce hits the same cache entry and is
    bitwise equal to the unfused path. The garbage this writes to ``out`` is
    fully overwritten (single-route tokens by the GEMM, the rest by the reduce).
    """
    # x / out are bf16 and w fp32 on every caller: the dtypes stay out of the key.
    key = (H, MAX_K, w is None)
    if key not in _SKIP_SINGLE_TUNED and not deterministic_autotune_enabled() and _small_h_config(H) is None:
        token_gather_and_sum_varlen_K_triton(x, w, out, M_perm, M_offset, T, MAX_K, H, True)
        _SKIP_SINGLE_TUNED.add(key)
    return route_scatter_dst(gather_idx, M_perm, M_offset, x.shape[0], T, w)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,  # (Mtotal, H)
    w: Optional[torch.Tensor],  # (Mtotal,)
    out: torch.Tensor,  # (T, H)
    M_perm: torch.Tensor,  # (Mtotal,) int32
    M_offset: torch.Tensor,  # (T+1,)   int32, variable K per token
    T: int,
    MAX_K: int,  # maximum K across all tokens
    H: int,
    is_varlen_K: bool,
    skip_single: bool = False,
):
    """
    1D parallelization over T, with iterative accumulation over K tiles and H tiles.
    Supports variable K per token.

    out[i, :] = sum_{j=0..K[i]-1}  x[M_perm[M_offset[i] + j], :] * w[M_offset[i] + j]

    where K[i] = M_offset[i+1] - M_offset[i] can vary per token.
    ``skip_single``: leave tokens with K[i] == 1 untouched (written by the
    route-scatter GEMM epilogue); needs ``is_varlen_K``.
    """
    assert is_varlen_K or not skip_single
    small = _small_h_config(H) if is_varlen_K and x.stride(1) == 1 and out.stride(1) == 1 else None
    if small is not None:
        BLOCK_T, BLOCK_H, num_warps = small
        token_gather_sum_small_h_kernel[(triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))](
            x, x if w is None else w, M_perm, M_offset, out, T, x.stride(0), out.stride(0),
            H=H, BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, w_is_None=w is None,
            SKIP_SINGLE=bool(skip_single), num_warps=num_warps,
        )
        return
    # INT32/UINT32 overflow guard: perm_idx * stride_xM or t_idx * stride_outT
    _max_x_offset = (x.shape[0] - 1) * x.stride(0)
    _max_out_offset = (T - 1) * out.stride(0)
    _needs_int64 = int(max(_max_x_offset, _max_out_offset) > 2**32 - 1)

    # 1D grid over T only
    token_gather_sum_kernel[(T,)](
        x,
        w,
        M_perm,
        M_offset,
        out,
        T=T,
        H=H,
        MAX_K=MAX_K,
        stride_xM=x.stride(0),
        stride_xH=x.stride(1),
        stride_outT=out.stride(0),
        stride_outH=out.stride(1),
        w_is_None=(w is None),
        is_varlen_K=is_varlen_K,
        SAFE_INT64=_needs_int64,
        SKIP_SINGLE=bool(skip_single),
    )
