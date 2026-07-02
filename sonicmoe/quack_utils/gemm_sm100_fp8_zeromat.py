# ********************************************************************************
# Zero-materialization FP8 blockscaled GemmGated / GemmDGated for SM100 architecture.
#
# Self-contained subclass of GemmSm100 that fixes SFA layout derivation for
# gather_A + blockscaled. All other behavior inherited unchanged.
#
# The fix: when gather_A=True, use mSFA.shape (TK-rows) instead of mA.shape
# (T-rows) for SFA layout. This ensures TMA offsets (cu_seqlens_m) stay
# within the scale tensor bounds.
#
# Usage:
#   1. quantize_and_pack_activation(x) -> x_fp8 (T,K) + x_scales_t (T-ISA)
#   2. gather_isa_packed_scales(x_scales_t, A_idx) -> x_scales_tk (TK-ISA)
#   3. Call gemm with A=x_fp8, A_idx=gather_idx, a_scales=x_scales_tk
#      using GemmGatedSm100ZeroMat instead of GemmGatedSm100
#
# No TK-sized FP8 activation is materialized. Only TK-sized scales (~3% of
# FP8 data size) are gathered.
# ********************************************************************************

from __future__ import annotations

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import quack.activation
import quack.copy_utils as copy_utils
import cuda.bindings.driver as cuda
from cutlass import Float32, Int32, const_expr
from quack.gemm_sm100 import GemmSm100
from quack.gemm_default_epi import GemmDefaultSm100
from quack.gemm_wrapper_utils import GemmWrapperBase
from quack.layout_utils import permute_gated_Cregs_b16
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments, VarlenManager

# Re-use the rank-aware tile_atom_to_shape_SF from our monkey-patch
from .blockscaled_fp8_gemm import _tile_atom_to_shape_SF_rank_aware

# Import mixins for GemmGated/GemmDGated from the leaf epilogues module
from ._gated_epilogues import (
    GemmGatedMixin,
    GemmGatedBlockscaledQuantMixin,
    GemmGatedPostActIsaQuantMixin,
    GemmGatedBlockscaledQuantPostActMixin,
    BlockscaledQuantOnlyMixin,
    GemmDGatedMixin,
    GemmDGatedFP8CLoadMixin,
)
from .sm_limit import capped_max_active_clusters, clc_persistence_default, sm_cap_enabled

from cutlass.utils import LayoutEnum


class _GemmSm100ZeroMatMixin:
    """Mixin that overrides __call__ to fix SFA layout for gather_A + blockscaled.

    When gather_A=True and blockscaled=True, the upstream GemmSm100 uses
    mA.shape (T-rows) for SFA layout derivation. But cu_seqlens_m offsets
    reach TK rows. This mixin uses mSFA.shape (TK-rows) instead, which
    correctly matches the pre-gathered scale tensor.

    This is the ONLY change needed — all other kernel logic (data gathering
    via cp.async, SFA TMA loading, MMA mainloop) works correctly when SFA
    has the right layout.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: TileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
    ):
        """GemmSm100.__call__ with fixed SFA layout for gather_A + blockscaled.

        Identical to upstream except line marked with # ZERO-MAT FIX.
        """
        if const_expr(self.blockscaled):
            assert mSFA is not None and mSFB is not None
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.sf_dtype: Optional[Type[cutlass.Numeric]] = (
            mSFA.element_type if mSFA is not None else None
        )
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None
        self.a_major_mode = LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(mB).mma_major_mode()

        # Check if input data types are compatible with MMA instruction
        if const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert (varlen_args.mAIdx is not None) == self.gather_A
        varlen_m = varlen_args.mCuSeqlensM is not None
        varlen_k = varlen_args.mCuSeqlensK is not None

        # Assume all strides are divisible by 128 bits except the last stride
        def new_stride(t: cute.Tensor):
            return tuple(
                cute.assume(s, divby=128 // t.element_type.width) if not cute.is_static(s) else s
                for s in t.stride
            )

        mA, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mA, mD)
        ]

        # Setup attributes that depend on gemm inputs
        self._setup_attributes(epilogue_args, varlen_args)

        if const_expr(self.blockscaled):
            # ============================================================
            # ZERO-MAT FIX: derive SFA layout from logical (TK, K) shape.
            #
            # When gather_A=True, mA is (T, K) but the GEMM logically
            # operates on TK rows via the gather index.  The pre-gathered
            # SFA scales span TK rows, so the ISA tiled layout must be
            # computed from (TK, K).  TK comes from mD (the output tensor
            # which always has the correct M-dimension).
            #
            # Bug history: the previous version passed mSFA.shape which is
            # the flat ISA storage shape (1, packed_bytes) — giving M=1
            # and producing a completely wrong tiled layout (7-20× output
            # attenuation with zero correlation to BF16).
            # ============================================================
            if const_expr(self.gather_A):
                # mA is (T, K) but GEMM logically has TK output rows.
                # Use D when present, otherwise mPostAct; both carry total_M.
                total_m = mD.shape[0] if const_expr(mD is not None) else epilogue_args.mPostAct.shape[0]
                sfa_logical_shape = (total_m, mA.shape[1])
                sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    sfa_logical_shape, self.sf_vec_size
                )
            else:
                sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(mA.shape, self.sf_vec_size)
            mSFA = cute.make_tensor(mSFA.iterator, sfa_layout)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(mB.shape, self.sf_vec_size)
            mSFB = cute.make_tensor(mSFB.iterator, sfb_layout)

        atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)

        # Setup TMA load for A & B
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mnk, self.tiled_mma.thr_id
            )
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=1, ptr_shift=False)
                if varlen_k and not self.gather_A
                else mA,
                a_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=(cutlass.TFloat32 if mA.element_type is Float32 else None),
            )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma.thr_id
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            copy_utils.create_ragged_tensor_for_tma(mB, ragged_dim=1) if varlen_k else mB,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if mB.element_type is Float32 else None),
        )

        tma_atom_sfa, tma_tensor_sfa = None, None
        tma_atom_sfb, tma_tensor_sfb = None, None
        if const_expr(self.blockscaled):
            # SFA TMA: uses mSFA which now has correct TK-row layout
            sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mnk, self.tiled_mma.thr_id
            )
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mSFA,
                sfa_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # SFB TMA: unchanged
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mnk, self.tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                mSFB,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                self.tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        if const_expr(self.blockscaled):
            sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            self.num_tma_load_bytes += sfa_copy_size + sfb_copy_size
        self.num_tma_load_bytes *= atom_thr_size

        # Setup TMA store for D
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mD, ragged_dim=0, ptr_shift=True)
                if varlen_m
                else mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store"
                if not (hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output)
                else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.epi_c_smem_layout_staged, self.epi_tile, op_type="load"
            )

        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)

        TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_m)
        tile_sched_args = self.get_scheduler_arguments(
            mA, mB, mD, scheduler_args, varlen_args, epilogue_args
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )

        self.buffer_align_bytes = 1024

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0
        sf_dtype = self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU
        sfa_smem_size = (
            cute.cosize(self.sfa_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        sfb_smem_size = (
            cute.cosize(self.sfb_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        a_idx_smem_size = 0
        if const_expr(self.gather_A):
            a_idx_smem_size = self.a_prefetch_stage * (
                self.cta_tile_shape_mnk[0] if varlen_m else self.cta_tile_shape_mnk[2]
            )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            acc_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            a_prefetch_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.a_prefetch_stage * 2
            ]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 12]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: Int32
            sAIdx: cute.struct.Align[cute.struct.MemRange[Int32, a_idx_smem_size], 16]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfa_smem_size],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfb_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            self.tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.a_smem_load_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        return


# ---------------------------------------------------------------------------
# Concrete kernel classes: GemmGated + GemmDGated with zero-mat fix
# ---------------------------------------------------------------------------

class GemmGatedSm100ZeroMat(GemmGatedMixin, _GemmSm100ZeroMatMixin, GemmSm100):
    """SM100 GemmGated with zero-materialization FP8 SFA fix."""
    pass


class GemmGatedSm100ZeroMatBlockscaledQuant(GemmGatedBlockscaledQuantMixin, _GemmSm100ZeroMatMixin, GemmSm100):
    """SM100 GemmGated + epilogue blockscaled FP8 quant + zero-materialization SFA fix."""
    pass


class GemmGatedSm100ZeroMatPostActQuant(GemmGatedPostActIsaQuantMixin, _GemmSm100ZeroMatMixin, GemmSm100):
    """SM100 GemmGated + epilogue blockscaled FP8 quant of the postact (y1) + zero-mat SFA fix.

    Fuses the standalone quantize_and_pack_activation(y1) into the up-proj
    epilogue: emits y1 directly as FP8 (mPostAct fp8) + ISA-packed UE8M0
    scales (mPostActScaleIsa), eliminating the bf16 y1 materialization and the
    standalone y1 quant kernel.  z (mD) stays bf16.  Requires epi_tile_n(z)=64
    (default override) for 1:1 subtile->postact-group mapping.
    """
    pass


class GemmGatedSm100ZeroMatQuantPostActQuant(
    GemmGatedBlockscaledQuantPostActMixin, _GemmSm100ZeroMatMixin, GemmSm100
):
    """SM100 GemmGated + epilogue blockscaled FP8 quant of BOTH z AND postact (y1) + zero-mat SFA fix.

    Single up-proj pass that emits z as fp8 (mD fp8 + mZScale) AND y1 as fp8
    (mPostAct fp8 + mPostActScaleIsa).  Lets fuse_y1=1 coexist with save_z_fp8=1
    so the backward takes the healthy fp8-preact dgated path.  Requires
    epi_tile_n(z)=64 for the 1:1 subtile->postact-group mapping.
    """
    pass


class GemmSm100ZeroMatBlockscaledQuant(BlockscaledQuantOnlyMixin, _GemmSm100ZeroMatMixin, GemmSm100):
    """SM100 GemmDefault + epilogue blockscaled FP8 quant + zero-materialization SFA fix.

    No swiglu, no PostAct.  Used by recompute_z (Option B): backward-only path
    that re-runs the up-projection GEMM to materialize z_fp8 + scales without
    paying for y1 (the original y1 was already saved in forward).

    Bit-exactness contract: for the same inputs, this kernel's D and mZScale
    outputs must match GemmGatedSm100ZeroMatBlockscaledQuant byte-for-byte.
    """
    pass


class GemmDGatedSm100ZeroMat(GemmDGatedMixin, _GemmSm100ZeroMatMixin, GemmSm100):
    """SM100 GemmDGated with zero-materialization FP8 SFA fix."""
    pass


class GemmDGatedFP8CLoadSm100ZeroMat(GemmDGatedFP8CLoadMixin, _GemmSm100ZeroMatMixin, GemmSm100):
    """SM100 GemmDGated + TMA-based FP8 C load + zero-materialization SFA fix.

    Combines FP8CLoadMixin (TMA epilogue loading fp8 z as Int16 C) with
    _GemmSm100ZeroMatMixin (__call__ override that derives SFA layout from
    mD.shape instead of mA.shape when gather_A=True).

    Without ZeroMat, the SFA layout uses mA.shape = (T, K) but the
    pre-gathered scales have TK rows.  cu_seqlens_m offsets then map
    incorrectly — expert 0 works (offset 0) but experts 1-7 get wrong
    scale factors producing garbage dz output.
    """
    pass


# ---------------------------------------------------------------------------
# High-level wrapper: gemm_gated with zero-materialization
# ---------------------------------------------------------------------------
# Mirrors gemm_gated.py:gemm_gated() but uses GemmGatedSm100ZeroMat.
# Accepts T-sized FP8 A with A_idx + TK-sized pre-gathered scales.

from functools import partial
from torch import Tensor
import torch

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)

from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters

_TORCH_TO_CUTLASS = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    _E8M0_DTYPE: cutlass.Float8E8M0FNU,
    torch.uint8: cutlass.Uint8,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}

_gate_fn_map = {
    "swiglu": quack.activation.swiglu,
}

from ..cache_manager import InstrumentedCompileCache as _ICC
_zeromat_compile_cache = _ICC("zeromat")


def _make_cute(tensor: Tensor, leading_dim: int) -> cute.Tensor:
    if tensor.dtype in {torch.float8_e4m3fn, _E8M0_DTYPE}:
        storage = tensor.detach().view(torch.uint8)
        ct = from_dlpack(storage, assumed_align=16)
        ct.element_type = _TORCH_TO_CUTLASS[tensor.dtype]
        return ct.mark_layout_dynamic(leading_dim=leading_dim)
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def gemm_gated_zeromat(
    A: Tensor,              # (T, K) FP8 — T-sized, NOT gathered
    B: Tensor,              # (E, N, K) — contiguous expert weights (FP8)
    PostAct: Tensor,        # (TK, N//2) — output
    cu_seqlens_m: Tensor,   # (E+1,) cumulative expert offsets, sums to TK
    A_idx: Tensor,          # (TK,) gather indices into A
    a_scales: Tensor,       # TK-sized ISA-packed scales (pre-gathered)
    b_scales: Tensor,       # Expert weight scales
    activation: str = "swiglu",
    out_dtype: torch.dtype = torch.bfloat16,
    postact_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Optional[Tensor], Tensor]:
    """Zero-materialization FP8 blockscaled GemmGated.

    A is (T, K) FP8 — kernel gathers rows via A_idx.
    a_scales is TK-sized — pre-gathered ISA-packed scales.
    No TK-sized FP8 activation is materialized.
    """
    assert activation in _gate_fn_map
    TK = A_idx.shape[0]
    N = B.shape[-2]
    K = A.shape[-1]

    # Output tensors
    preact_out = torch.empty((TK, N), dtype=out_dtype, device=A.device)
    if PostAct is None:
        PostAct = torch.empty((TK, N // 2), dtype=postact_dtype, device=A.device)

    # validate_and_prepare_tensors equivalent
    L, M, _, _, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, preact_out, None, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    tensor_infos["PostAct"] = type(tensor_infos["A"])(PostAct)
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=True)
    major_configs = {
        "A": ("m", "k", "l"), "B": ("n", "k", "l"),
        "D": ("m", "n", "l"), "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for info in tensor_infos.values():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS[info.tensor.dtype]

    device_cap = get_device_capacity(A.device)
    assert device_cap[0] == 10, "Zero-mat kernel requires SM100"
    GemmCls = GemmGatedSm100ZeroMat

    tile_M, tile_N = 128, 128
    cluster_M, cluster_N = 1, 1
    max_active_clusters = capped_max_active_clusters(cluster_M * cluster_N)

    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            leading_dim = 1 if info.major == major_configs[name][1] else 0
            info.cute_tensor = _make_cute(info.tensor, leading_dim)

    act_fn = _gate_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(tensor_infos["PostAct"].cute_tensor, act_fn)
    scheduler_args = GemmWrapperBase.create_scheduler_args(max_active_clusters, None, max_swizzle_size=8)
    varlen_args = GemmWrapperBase.create_varlen_args(cu_seqlens_m, None, A_idx)
    _stream_obj = torch.cuda.current_stream()
    current_stream = cuda.CUstream(_stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream)

    a_scale_cute = _make_cute(a_scales, leading_dim=1)
    b_scale_cute = _make_cute(b_scales, leading_dim=1)

    # compile_key must NOT contain dynamic token dimensions (A.shape[0],
    # preact_out.shape[0], PostAct.shape[0], TK) — those change with seqlen
    # and are handled at runtime via mark_layout_dynamic.
    compile_key = (
        "zeromat_gated",
        A.shape[-1], A.dtype,           # K only — static (model dim)
        tuple(B.shape), B.dtype,        # (E,N,K) — static
        preact_out.dtype,
        PostAct.dtype,
        activation,
        True,  # blockscaled
        sm_cap_enabled(),  # STATIC vs CLC scheduler (compile-time)
    )

    cache = _zeromat_compile_cache
    if compile_key not in cache:
        gemm_obj = GemmCls(
            cutlass.Float32,
            _TORCH_TO_CUTLASS[A.dtype],
            (tile_M, tile_N),
            (cluster_M, cluster_N, 1),
            gather_A=True,
            sf_vec_size=32,
            use_clc_persistence=clc_persistence_default(True),
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
    return preact_out, PostAct


# ---------------------------------------------------------------------------
# Non-gated blockscaled FP8 quant wrapper (Option B for recompute_z)
# ---------------------------------------------------------------------------
# Mirrors gemm_gated_zeromat() but uses GemmSm100ZeroMatBlockscaledQuant —
# no swiglu, no PostAct allocation, no postact write.  Only writes z_fp8 (D)
# and a per-(M, N//32) UE8M0 scale tensor.

def blockscaled_fp8_gemm_zeromat_quant(
    A: Tensor,              # (T, K) FP8 — T-sized, NOT gathered
    B: Tensor,              # (E, N, K) — contiguous expert weights (FP8)
    cu_seqlens_m: Tensor,   # (E+1,) cumulative expert offsets, sums to TK
    A_idx: Tensor,          # (TK,) gather indices into A
    a_scales: Tensor,       # TK-sized ISA-packed scales (pre-gathered)
    b_scales: Tensor,       # Expert weight scales
    z_fp8_out: Optional[Tensor] = None,    # (TK, N) FP8 — z output (allocated if None)
    z_scale_out: Optional[Tensor] = None,  # (TK, N//32) uint8 — UE8M0 scale output (allocated if None)
) -> tuple[Tensor, Tensor]:
    """Recompute-only kernel: blockscaled FP8 GEMM that ONLY emits z_fp8 + scales.

    Used by `_recompute_z_fp8` (Option B): backward materializes z without
    paying for swiglu activation, postact register convert, R2S/S2G smem copy,
    or the y1 TMA store.

    Bit-exactness: the z_fp8_out and z_scale_out bytes produced by this kernel
    are byte-equal to those produced by `gemm_gated()` with `epilogue_quant=True`
    on the same (A, B, A_idx, a_scales, b_scales).  Validated by
    tests/ops/test_recompute_z_optionB.py.
    """
    TK = A_idx.shape[0]
    N = B.shape[-2]
    K = A.shape[-1]
    assert N % 32 == 0, f"N must be divisible by 32 for blockscaled scale output, got {N}"

    if z_fp8_out is None:
        z_fp8_out = torch.empty((TK, N), dtype=torch.float8_e4m3fn, device=A.device)
    if z_scale_out is None:
        z_scale_out = torch.empty((TK, N // 32), dtype=torch.uint8, device=A.device)

    L, M, _, _, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, z_fp8_out, None, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=True)
    major_configs = {
        "A": ("m", "k", "l"), "B": ("n", "k", "l"),
        "D": ("m", "n", "l"), "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for info in tensor_infos.values():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS[info.tensor.dtype]

    device_cap = get_device_capacity(A.device)
    assert device_cap[0] == 10, "Zero-mat quant-only kernel requires SM100"
    GemmCls = GemmSm100ZeroMatBlockscaledQuant

    tile_M, tile_N = 128, 128
    cluster_M, cluster_N = 1, 1
    max_active_clusters = capped_max_active_clusters(cluster_M * cluster_N)

    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            leading_dim = 1 if info.major == major_configs[name][1] else 0
            info.cute_tensor = _make_cute(info.tensor, leading_dim)

    z_scale_cute = _make_cute(z_scale_out, leading_dim=1)
    epi_args = GemmCls.EpilogueArguments(mZScale=z_scale_cute)
    scheduler_args = GemmWrapperBase.create_scheduler_args(max_active_clusters, None, max_swizzle_size=8)
    varlen_args = GemmWrapperBase.create_varlen_args(cu_seqlens_m, None, A_idx)
    _stream_obj = torch.cuda.current_stream()
    current_stream = cuda.CUstream(
        _stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream
    )

    a_scale_cute = _make_cute(a_scales, leading_dim=1)
    b_scale_cute = _make_cute(b_scales, leading_dim=1)

    compile_key = (
        "zeromat_quant_only",
        A.shape[-1], A.dtype,
        tuple(B.shape), B.dtype,
        z_fp8_out.dtype,
        True,  # blockscaled
        sm_cap_enabled(),  # STATIC vs CLC scheduler (compile-time)
    )

    cache = _zeromat_compile_cache
    if compile_key not in cache:
        gemm_obj = GemmCls(
            cutlass.Float32,
            _TORCH_TO_CUTLASS[A.dtype],
            (tile_M, tile_N),
            (cluster_M, cluster_N, 1),
            gather_A=True,
            sf_vec_size=32,
            use_clc_persistence=clc_persistence_default(True),
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
    return z_fp8_out, z_scale_out


# ---------------------------------------------------------------------------
# BF16 D wrapper (no quant epi) — used as a clean reference for iso32 byte-exact
# validation against `_dual_varlen_iso32_quantize_kernel` on the same GEMM input.
# ---------------------------------------------------------------------------


class GemmSm100ZeroMatBf16(_GemmSm100ZeroMatMixin, GemmDefaultSm100):
    """Bare zeromat blockscaled FP8 GEMM with BF16 D output (no quant epi).

    Reference path for iso32 epi validation: produces the same GEMM
    accumulator as `blockscaled_fp8_gemm_zeromat_iso32_quant` but writes
    raw BF16, allowing the production `_dual_varlen_iso32_quantize_kernel`
    to be run on the result for byte-exact comparison.
    """
    pass


_zeromat_bf16_compile_cache = {}


def blockscaled_fp8_gemm_zeromat_bf16(
    A: Tensor,
    B: Tensor,
    cu_seqlens_m: Tensor,
    A_idx: Tensor,
    a_scales: Tensor,
    b_scales: Tensor,
    z_bf16_out: Optional[Tensor] = None,
) -> Tensor:
    """Blockscaled FP8 GEMM with BF16 D output (no quant epi).

    Identical accumulation path to `blockscaled_fp8_gemm_zeromat_iso32_quant`;
    differs only in the final D dtype (BF16 vs FP8) and skips the iso32 epi
    quant step.  Used as the *ground-truth source tensor* for byte-exact
    validation of the iso32 epi against `_dual_varlen_iso32_quantize_kernel`.
    """
    TK = A_idx.shape[0]
    N = B.shape[-2]
    if z_bf16_out is None:
        z_bf16_out = torch.empty((TK, N), dtype=torch.bfloat16, device=A.device)

    L, M, _, _, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, z_bf16_out, None, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=True)
    major_configs = {
        "A": ("m", "k", "l"), "B": ("n", "k", "l"),
        "D": ("m", "n", "l"), "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)
    for info in tensor_infos.values():
        if info.tensor is not None:
            info.dtype = _TORCH_TO_CUTLASS[info.tensor.dtype]

    device_cap = get_device_capacity(A.device)
    assert device_cap[0] == 10
    GemmCls = GemmSm100ZeroMatBf16

    tile_M, tile_N, cluster_M, cluster_N = 128, 128, 1, 1
    max_active_clusters = capped_max_active_clusters(cluster_M * cluster_N)

    for name, info in tensor_infos.items():
        if info.tensor is not None and name in major_configs:
            leading_dim = 1 if info.major == major_configs[name][1] else 0
            info.cute_tensor = _make_cute(info.tensor, leading_dim)

    epi_args = GemmCls.EpilogueArguments()
    scheduler_args = GemmWrapperBase.create_scheduler_args(max_active_clusters, None, max_swizzle_size=8)
    varlen_args = GemmWrapperBase.create_varlen_args(cu_seqlens_m, None, A_idx)
    _stream_obj = torch.cuda.current_stream()
    current_stream = cuda.CUstream(
        _stream_obj.stream_base.raw_stream if hasattr(_stream_obj, "stream_base") else _stream_obj.cuda_stream
    )
    a_scale_cute = _make_cute(a_scales, leading_dim=1)
    b_scale_cute = _make_cute(b_scales, leading_dim=1)

    compile_key = (
        "zeromat_bf16",
        A.shape[-1], A.dtype,
        tuple(B.shape), B.dtype,
        z_bf16_out.dtype,
        True,
        tile_M, tile_N, cluster_M, cluster_N,
        sm_cap_enabled(),  # STATIC vs CLC scheduler (compile-time)
    )

    cache = _zeromat_bf16_compile_cache
    if compile_key not in cache:
        gemm_obj = GemmCls(
            cutlass.Float32,
            _TORCH_TO_CUTLASS[A.dtype],
            (tile_M, tile_N),
            (cluster_M, cluster_N, 1),
            gather_A=True,
            sf_vec_size=32,
            use_clc_persistence=clc_persistence_default(True),
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
    return z_bf16_out


