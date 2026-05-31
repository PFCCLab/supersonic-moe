# ********************************************************************************
# Leaf module: gated/dgated epilogue Mixins extracted from gemm_gated.py and
# gemm_dgated.py to break circular imports.
#
# IMPORTANT: This module must NOT import from .gemm_gated, .gemm_dgated,
# .gemm_sm100_fp8_zeromat, or .blockscaled_fp8_gemm.
# ********************************************************************************

from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import quack.layout_utils as layout_utils
import quack.utils as utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as _math
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import ParamsBase, mlir_namedtuple, torch2cute_dtype_map
from quack.epi_ops import ColVecReduce, TileStore, EpiOp, assume_stride_divisibility
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm100 import GemmSm100
from quack.layout_utils import permute_gated_Cregs_b16
from torch import Tensor

# ---------------------------------------------------------------------------
# Shared constants and utilities
# ---------------------------------------------------------------------------

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    _E8M0_DTYPE: cutlass.Float8E8M0FNU,
    torch.uint8: cutlass.Uint8,
    torch.int16: cutlass.Int16,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _is_runtime_fp8_tensor(tensor: Tensor) -> bool:
    return tensor.dtype in {torch.float8_e4m3fn, _E8M0_DTYPE}


def _make_cute_tensor_dynamic(tensor: Tensor, leading_dim: int) -> cute.Tensor:
    if _is_runtime_fp8_tensor(tensor):
        storage = tensor.detach().view(torch.uint8)
        cute_tensor = from_dlpack(storage, assumed_align=16)
        cute_tensor.element_type = _TORCH_TO_CUTLASS_DTYPE[tensor.dtype]
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def _halve_epi_tile(gemm, epi_tile):
    """Halve the N-dimension of the epilogue tile for gated activations."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(2, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // 2)


# ---------------------------------------------------------------------------
# GemmGatedMixin (from gemm_gated.py)
# ---------------------------------------------------------------------------

class GemmGatedMixin(GemmActMixin):
    _epi_ops = (*GemmActMixin._epi_ops[:-1], TileStore("mPostAct", epi_tile_fn=_halve_epi_tile))

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Override: force CopyUniversalOp for postact R2S when blockscaled.

        NOTE: The fused GemmGated + blockscaled FP8 path crashes due to a
        deeper issue in epi_visit_subtile's accumulator recast. This override
        alone is insufficient — the decomposed path is used instead.
        Kept as documentation of the attempted fix.
        """
        if const_expr(self.blockscaled):
            sPostAct = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
            copy_atom_postact_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.postact_dtype
            )
            tiled_copy_postact_r2s = cute.make_tiled_copy_S(
                copy_atom_postact_r2s, tiled_copy_r2s
            )
            tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
            batch_idx = tile_coord_mnkl[3]
            copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
                params.tma_atom_mPostAct,
                varlen_manager.offset_batch_epi(params.mPostAct, batch_idx),
                self.cta_tile_shape_postact_mn,
                params.epi_tile_mPostAct,
                sPostAct,
                tile_coord_mnkl,
            )
            return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact
        else:
            return GemmActMixin.epi_setup_postact(
                self, params, epi_smem_tensors, tiled_copy_r2s,
                tiled_copy_t2r, tile_coord_mnkl, varlen_manager, tidx,
            )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        assert self.postact_dtype.width in {8, 16}, "GemmGated only supports 8bit or 16bit postact for now"
        assert self.d_layout is None or self.d_layout.is_n_major_c()
        assert self.postact_layout.is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, "GemmGatedSm90 requires tileN to be divisible by 32"
        self.cta_tile_shape_postact_mn = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2)
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]), (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3])
                )
        return tRS_rPostAct

    @cute.jit
    def epi_convert_postact(self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx):
        tRS_rPostAct_out = GemmActMixin.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )
        if const_expr(self.arch == 90):
            permute_gated_Cregs_b16(tRS_rPostAct_out)
        return tRS_rPostAct_out


# ---------------------------------------------------------------------------
# DSL bitcast helpers (shared by gated and dgated)
# ---------------------------------------------------------------------------

@dsl_user_op
def _f32_as_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    """Bitcast float32 to int32 (reinterpret bits, no conversion)."""
    return Int32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _i32_as_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    """Bitcast int32 to float32 (reinterpret bits, no conversion)."""
    return Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


# ---------------------------------------------------------------------------
# BlockscaledScaleStore EpiOp (from gemm_gated.py)
# ---------------------------------------------------------------------------

class BlockscaledScaleStore(EpiOp):
    """EpiOp: writes UE8M0 scale bytes to gmem from epi_visit_subtile.

    Scale output layout: (total_M, N//32) uint8, row-major.
    begin(): computes absolute M row and N-group base for this thread.
    begin_loop(): returns (param, m_abs, n_group_abs, m_limit, n_limit) with bounds.
    The mixin's epi_visit_subtile writes the scale byte after bounds check.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        if tensor is not None:
            return {self.name: assume_stride_divisibility(tensor)}
        return {self.name: None}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]
            # Thread-to-M-row: SM100 Ld32x32bOp maps tidx -> M-row within tile
            m_in_tile = ctx.tidx % tile_M
            if const_expr(ctx.varlen_manager.varlen_m):
                batch_start = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
                m_abs = batch_start + ctx.tile_coord_mnkl[0] * tile_M + m_in_tile
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3] + Int32(1)]
            else:
                m_abs = ctx.tile_coord_mnkl[0] * tile_M + m_in_tile
                m_limit = param.shape[0]  # total_M
            n_base = ctx.tile_coord_mnkl[1] * (tile_N // 32)
            n_limit = param.shape[1]  # N//32
            return (param, m_abs, n_base, m_limit, n_limit)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            param, m_abs, n_base, m_limit, n_limit = state
            if const_expr(isinstance(epi_coord, tuple)):
                n_sub = epi_coord[1] if len(epi_coord) > 1 else epi_coord[0]
            else:
                n_sub = epi_coord
            return (param, m_abs, n_base + n_sub, m_limit, n_limit)
        return None


# ---------------------------------------------------------------------------
# GemmGatedBlockscaledQuantMixin (from gemm_gated.py)
# ---------------------------------------------------------------------------

class GemmGatedBlockscaledQuantMixin(GemmGatedMixin):
    """GemmGated + epilogue blockscaled FP8 quant of z.

    Integer+carry E8M0 algorithm matching Triton/Paddle reference.
    Precision: 0 byte mismatch across all shapes (verified).
    """
    _epi_ops = (
        *GemmGatedMixin._epi_ops,
        BlockscaledScaleStore("mZScale"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScale: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tRS_rPostAct = GemmGatedMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        _z_scale_active = epi_loop_tensors["mZScale"]
        if const_expr(_z_scale_active is not None):
            num_z = cute.size(tRS_rD)

            # Step 1: amax
            amax = Float32(0.0)
            for i in cutlass.range(num_z, unroll_full=True):
                val = tRS_rD[i]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax = cute.arch.fmax(amax, abs_val)
            amax = cute.arch.fmax(amax, Float32(1e-4))

            # Step 2: integer+carry E8M0
            amax_bits = _f32_as_i32(amax)
            biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
            mantissa_bits = amax_bits & Int32(0x7FFFFF)
            has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
            carry = Int32(1) if has_carry else Int32(0)
            e8m0 = biased_exp - Int32(8) + carry
            is_normal = cutlass.Boolean(biased_exp > Int32(0))
            e8m0 = e8m0 if is_normal else Int32(0)
            is_pos = cutlass.Boolean(e8m0 > Int32(0))
            e8m0 = e8m0 if is_pos else Int32(0)

            # Step 3: quant_scale = 2^(254 - e8m0)
            qexp = Int32(254) - e8m0
            qexp_hi = cutlass.Boolean(qexp > Int32(254))
            qexp = Int32(254) if qexp_hi else qexp
            qexp_lo = cutlass.Boolean(qexp < Int32(1))
            qexp = Int32(1) if qexp_lo else qexp
            quant_scale = _i32_as_f32(qexp << Int32(23))

            # Step 4: z *= quant_scale
            for i in cutlass.range(num_z, unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * quant_scale

            # Step 5: store UE8M0 scale to gmem (bounds-checked)
            z_scale_info = epi_loop_tensors["mZScale"]
            if const_expr(z_scale_info is not None):
                scale_tensor, m_abs, n_group_abs, m_limit, n_limit = z_scale_info
                in_bounds = cutlass.Boolean(m_abs < m_limit) & cutlass.Boolean(n_group_abs < n_limit)
                if in_bounds:
                    scale_tensor[m_abs, n_group_abs] = cutlass.Int8(e8m0)

        return tRS_rPostAct


# ---------------------------------------------------------------------------
# BlockscaledQuantOnlyMixin (from gemm_gated.py)
# ---------------------------------------------------------------------------

class BlockscaledQuantOnlyMixin(GemmDefaultEpiMixin):
    """GemmDefault + epilogue blockscaled FP8 quant of D, no activation/postact."""

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        BlockscaledScaleStore("mZScale"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScale: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # Default epi visit: alpha/beta/rowvec/colvec (returns None).
        GemmDefaultEpiMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        _z_scale_active = epi_loop_tensors["mZScale"]
        if const_expr(_z_scale_active is not None):
            num_z = cute.size(tRS_rD)

            # Step 1: amax over the register tile.
            amax = Float32(0.0)
            for i in cutlass.range(num_z, unroll_full=True):
                val = tRS_rD[i]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax = cute.arch.fmax(amax, abs_val)
            amax = cute.arch.fmax(amax, Float32(1e-4))

            # Step 2: integer+carry E8M0 (matches Triton/Paddle reference).
            amax_bits = _f32_as_i32(amax)
            biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
            mantissa_bits = amax_bits & Int32(0x7FFFFF)
            has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
            carry = Int32(1) if has_carry else Int32(0)
            e8m0 = biased_exp - Int32(8) + carry
            is_normal = cutlass.Boolean(biased_exp > Int32(0))
            e8m0 = e8m0 if is_normal else Int32(0)
            is_pos = cutlass.Boolean(e8m0 > Int32(0))
            e8m0 = e8m0 if is_pos else Int32(0)

            # Step 3: quant_scale = 2^(254 - e8m0) (clamped to [1, 254]).
            qexp = Int32(254) - e8m0
            qexp_hi = cutlass.Boolean(qexp > Int32(254))
            qexp = Int32(254) if qexp_hi else qexp
            qexp_lo = cutlass.Boolean(qexp < Int32(1))
            qexp = Int32(1) if qexp_lo else qexp
            quant_scale = _i32_as_f32(qexp << Int32(23))

            # Step 4: scale registers in place; saturating cast to fp8 happens
            # at TMA store time when the D tensor is fp8_e4m3fn.
            for i in cutlass.range(num_z, unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * quant_scale

            # Step 5: store UE8M0 scale to gmem (bounds-checked).
            z_scale_info = epi_loop_tensors["mZScale"]
            if const_expr(z_scale_info is not None):
                scale_tensor, m_abs, n_group_abs, m_limit, n_limit = z_scale_info
                in_bounds = cutlass.Boolean(m_abs < m_limit) & cutlass.Boolean(n_group_abs < n_limit)
                if in_bounds:
                    scale_tensor[m_abs, n_group_abs] = cutlass.Int8(e8m0)

        # No postact -> return None.
        return None


# ---------------------------------------------------------------------------
# BlockscaledIsaRowScaleStore EpiOp (Session 1A foundation block)
# ---------------------------------------------------------------------------
#
# Writes UE8M0 scale bytes directly into the ISA-pack layout used downstream
# by blockscaled FP8 GEMMs (and produced standalone today by
# `_quantize_and_pack_kernel`).
#
# Layout (3D uint8 buffer `(num_m_tiles, k_tiles, 512)`):
#   SF_TILE_M       = 128
#   SF_TILE_K       = 128
#   SF_TILE_STORAGE = 512  (== SF_TILE_M * SF_TILE_K // SF_VEC_SIZE)
#   SF_VEC_SIZE     = 32
#
# Offset of one E8M0 byte for absolute (m_abs, n_group_abs):
#   m_tile          = m_abs // 128
#   row_in_tile     = m_abs %  128
#   k_tile_idx      = n_group_abs // 4
#   k_in_tile       = n_group_abs %  4
#   row_base_offset = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
#   byte_at         = scale[m_tile, k_tile_idx, row_base_offset + k_in_tile]
#
# Single-byte stores (NOT the uint32 4-byte pack the Triton kernel uses) —
# the cute epi loop visits one group at a time, so packing 4 groups into
# uint32 would require cross-iteration buffering.  Single-byte stores still
# coalesce within a warp (consecutive lanes write adjacent bytes when
# k_in_tile sweeps 0..3).
# ---------------------------------------------------------------------------

class BlockscaledIsaRowScaleStore(EpiOp):
    """EpiOp: writes UE8M0 scale bytes into ISA-pack layout.

    Scale buffer shape: `(num_m_tiles, k_tiles, 512)` uint8, byte-equivalent
    to the 1D packed scale buffer produced by `_quantize_and_pack_kernel`.

    begin(): computes absolute M row and N-group base for this thread.
    begin_loop(): returns (param, m_abs, n_group_abs, m_limit, n_group_limit).
    The mixin's epi_visit_subtile writes the scale byte after bounds check.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        if tensor is not None:
            return {self.name: assume_stride_divisibility(tensor)}
        return {self.name: None}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]
            # Thread-to-M-row: SM100 Ld32x32bOp maps tidx -> M-row within tile.
            m_in_tile = ctx.tidx % tile_M
            if const_expr(ctx.varlen_manager.varlen_m):
                batch_start = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
                m_abs = batch_start + ctx.tile_coord_mnkl[0] * tile_M + m_in_tile
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3] + Int32(1)]
            else:
                m_abs = ctx.tile_coord_mnkl[0] * tile_M + m_in_tile
                # m_limit derived from buffer shape: num_m_tiles * SF_TILE_M.
                m_limit = param.shape[0] * Int32(128)
            n_base = ctx.tile_coord_mnkl[1] * (tile_N // 32)
            # n_group_limit derived from buffer shape: k_tiles * 4.
            n_group_limit = param.shape[1] * Int32(4)
            return (param, m_abs, n_base, m_limit, n_group_limit)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            param, m_abs, n_base, m_limit, n_group_limit = state
            if const_expr(isinstance(epi_coord, tuple)):
                n_sub = epi_coord[1] if len(epi_coord) > 1 else epi_coord[0]
            else:
                n_sub = epi_coord
            return (param, m_abs, n_base + n_sub, m_limit, n_group_limit)
        return None


# ---------------------------------------------------------------------------
# BlockscaledIsaQuantOnlyMixin (Session 1A foundation block)
# ---------------------------------------------------------------------------

class BlockscaledIsaQuantOnlyMixin(GemmDefaultEpiMixin):
    """GemmDefault + epi blockscaled FP8 quant of D with ISA-pack scale store.

    Parallel to `BlockscaledQuantOnlyMixin` but writes scales in ISA-pack
    layout (shape `(num_m_tiles, k_tiles, 512)` uint8) instead of flat
    `(M, N//32)`.  Same amax / E8M0 / quant_scale arithmetic — only the
    final scale store differs.
    """

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        BlockscaledIsaRowScaleStore("mZScaleIsa"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScaleIsa: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        _z_scale_active = epi_loop_tensors["mZScaleIsa"]
        if const_expr(_z_scale_active is not None):
            num_z = cute.size(tRS_rD)

            # Step 1: amax over the register tile.
            amax = Float32(0.0)
            for i in cutlass.range(num_z, unroll_full=True):
                val = tRS_rD[i]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax = cute.arch.fmax(amax, abs_val)
            amax = cute.arch.fmax(amax, Float32(1e-4))

            # Step 2: integer+carry E8M0 (matches Triton reference).
            amax_bits = _f32_as_i32(amax)
            biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
            mantissa_bits = amax_bits & Int32(0x7FFFFF)
            has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
            carry = Int32(1) if has_carry else Int32(0)
            e8m0 = biased_exp - Int32(8) + carry
            is_normal = cutlass.Boolean(biased_exp > Int32(0))
            e8m0 = e8m0 if is_normal else Int32(0)
            is_pos = cutlass.Boolean(e8m0 > Int32(0))
            e8m0 = e8m0 if is_pos else Int32(0)

            # Step 3: quant_scale = 2^(254 - e8m0) (clamped to [1, 254]).
            qexp = Int32(254) - e8m0
            qexp_hi = cutlass.Boolean(qexp > Int32(254))
            qexp = Int32(254) if qexp_hi else qexp
            qexp_lo = cutlass.Boolean(qexp < Int32(1))
            qexp = Int32(1) if qexp_lo else qexp
            quant_scale = _i32_as_f32(qexp << Int32(23))

            # Step 4: scale registers; saturating cast to fp8 at TMA store time.
            for i in cutlass.range(num_z, unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * quant_scale

            # Step 5: store UE8M0 byte at ISA-pack offset (bounds-checked).
            z_scale_info = epi_loop_tensors["mZScaleIsa"]
            if const_expr(z_scale_info is not None):
                scale_tensor, m_abs, n_group_abs, m_limit, n_group_limit = z_scale_info
                in_bounds = (
                    cutlass.Boolean(m_abs < m_limit)
                    & cutlass.Boolean(n_group_abs < n_group_limit)
                )
                if in_bounds:
                    # ISA-pack offset math (mirrors _quantize_and_pack_kernel).
                    m_tile = m_abs // Int32(128)
                    row_in_tile = m_abs % Int32(128)
                    k_tile_idx = n_group_abs // Int32(4)
                    k_in_tile = n_group_abs % Int32(4)
                    row_base = (row_in_tile % Int32(32)) * Int32(16) + (
                        row_in_tile // Int32(32)
                    ) * Int32(4)
                    inner_off = row_base + k_in_tile
                    scale_tensor[m_tile, k_tile_idx, inner_off] = cutlass.Int8(e8m0)

        return None


# ---------------------------------------------------------------------------
# BlockscaledIsaColScaleStore EpiOp (Session 1A-ext / iso32 col-axis)
# ---------------------------------------------------------------------------
#
# Companion to BlockscaledIsaRowScaleStore: writes UE8M0 bytes into the
# *column*-axis ISA-pack layout used by colwise blockscaled FP8 GEMMs (the
# layout produced today by `_colwise_quantize_and_pack_kernel` and by the
# col-SF half of `_dual_varlen_iso32_quantize_kernel`).
#
# Buffer shape `(num_n_tiles, col_k_tiles, 512)` uint8, where:
#   num_n_tiles  = ceil(N / SF_TILE_M)         # N=feature-dim becomes "M"
#   col_k_tiles  = ceil(M / SF_TILE_K)         # M=token-dim becomes "K"
#
# Offset for one E8M0 byte at absolute (n_abs, m_group_abs):
#   col_n_tile        = n_abs // 128
#   col_row_in_tile   = n_abs %  128
#   col_k_tile_idx    = m_group_abs // 4
#   col_k_in_tile     = m_group_abs %  4
#   col_row_base      = (col_row_in_tile % 32) * 16 + (col_row_in_tile // 32) * 4
#   byte_at           = scale[col_n_tile, col_k_tile_idx, col_row_base + col_k_in_tile]
#
# Caller (mixin) supplies (n_abs_per_lane, m_group_abs) at store time: lane
# k of each warp writes the col-SF byte for n_abs = warp_n_base + k.
# ---------------------------------------------------------------------------

class BlockscaledIsaColScaleStore(EpiOp):
    """EpiOp: writes UE8M0 scale bytes into col-axis ISA-pack layout."""

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        if tensor is not None:
            return {self.name: assume_stride_divisibility(tensor)}
        return {self.name: None}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]
            # Warp-base m_abs: all lanes in a warp share the same m_group
            # (since SM100 Ld32x32bOp maps consecutive tidx -> consecutive
            # m_in_tile within each 32-lane warp).
            m_in_tile = ctx.tidx % tile_M
            lane_id = ctx.tidx % Int32(32)
            warp_m_base_in_tile = m_in_tile - lane_id
            if const_expr(ctx.varlen_manager.varlen_m):
                batch_start = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
                m_warp_base = batch_start + ctx.tile_coord_mnkl[0] * tile_M + warp_m_base_in_tile
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3] + Int32(1)]
            else:
                m_warp_base = ctx.tile_coord_mnkl[0] * tile_M + warp_m_base_in_tile
                # The col-SF buffer's k_tile dim encodes num_m_groups (rounded up).
                # col_k_tiles * 4 (groups_per_k_tile) * 32 (rows_per_group) = M_upper.
                m_limit = param.shape[1] * Int32(128)
            # n_base = N-group start for this tile (each subtile bumps by 1
            # n_group via begin_loop).  Lane k writes col-SF byte for
            # n_abs = n_base_groups*32 + k.
            n_tile_base = ctx.tile_coord_mnkl[1] * tile_N
            # n_limit = num_n_tiles * SF_TILE_M (= 128)
            n_limit = param.shape[0] * Int32(128)
            return (param, m_warp_base, n_tile_base, lane_id, m_limit, n_limit)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            param, m_warp_base, n_tile_base, lane_id, m_limit, n_limit = state
            if const_expr(isinstance(epi_coord, tuple)):
                n_sub = epi_coord[1] if len(epi_coord) > 1 else epi_coord[0]
            else:
                n_sub = epi_coord
            # n_sub is in units of N-groups (one subtile == 32 N-cols).
            n_warp_base = n_tile_base + n_sub * Int32(32)
            return (param, m_warp_base, n_warp_base, lane_id, m_limit, n_limit)
        return None


# ---------------------------------------------------------------------------
# BlockscaledIso32QuantOnlyMixin (Session 1A-ext)
# ---------------------------------------------------------------------------
#
# Block-amax (iso32) quantization fused into the GEMM epilogue.  Produces
# THREE outputs from a single MMA:
#   - z_fp8       (e4m3, MxN) saturating cast using block scale
#   - z_row_isa   row-axis ISA-pack SF buffer (1A layout)
#   - z_col_isa   col-axis ISA-pack SF buffer (this file)
#
# iso32 invariant: the e8m0 byte at (m_group, n_group) is the SAME byte in
# both row-SF and col-SF layouts (just different offsets), because the amax
# is reduced over the full 32x32 block (row + col together).  This is what
# `_dual_varlen_iso32_quantize_kernel` exploits; we replicate it in epi.
#
# Warp layout assumption (same as 1A, validated): SM100 Ld32x32bOp maps
# tidx -> m_in_tile such that consecutive tidx values in a warp form 32
# contiguous M-rows of a single 32-row m_group, all sharing one 32-col
# n_group.  warp_redux_sync(MAX) across these 32 lanes yields the block
# amax (since each lane already reduced its 32 N-vals).
# ---------------------------------------------------------------------------


class BlockscaledIso32QuantOnlyMixin(GemmDefaultEpiMixin):
    """GemmDefault + epi iso32 (block-amax) blockscaled FP8 quant of D with
    dual ISA-pack scale stores (row + col).
    """

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        BlockscaledIsaRowScaleStore("mZScaleIsaRow"),
        BlockscaledIsaColScaleStore("mZScaleIsaCol"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScaleIsaRow: Optional[cute.Tensor] = None
        mZScaleIsaCol: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        row_info = epi_loop_tensors["mZScaleIsaRow"]
        col_info = epi_loop_tensors["mZScaleIsaCol"]
        any_active = (row_info is not None) or (col_info is not None)
        if const_expr(any_active):
            num_z = cute.size(tRS_rD)

            # Step 1: per-thread amax over the register tile (32 N-vals).
            amax = Float32(0.0)
            for i in cutlass.range(num_z, unroll_full=True):
                val = tRS_rD[i]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax = cute.arch.fmax(amax, abs_val)

            # Step 2: WARP-LEVEL REDUCE across 32 lanes -> block amax.
            # All 32 lanes in this warp share the same n_group AND span 32
            # consecutive M-rows of one m_group, so the reduced max is the
            # iso32 block amax for the (m_group, n_group) 32x32 block.
            amax = cute.arch.warp_redux_sync(amax, "max")
            amax = cute.arch.fmax(amax, Float32(1e-4))

            # Step 3: integer+carry E8M0 (matches Triton reference).
            amax_bits = _f32_as_i32(amax)
            biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
            mantissa_bits = amax_bits & Int32(0x7FFFFF)
            has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
            carry = Int32(1) if has_carry else Int32(0)
            e8m0 = biased_exp - Int32(8) + carry
            is_normal = cutlass.Boolean(biased_exp > Int32(0))
            e8m0 = e8m0 if is_normal else Int32(0)
            is_pos = cutlass.Boolean(e8m0 > Int32(0))
            e8m0 = e8m0 if is_pos else Int32(0)

            # Step 4: quant_scale = 2^(254 - e8m0).
            qexp = Int32(254) - e8m0
            qexp_hi = cutlass.Boolean(qexp > Int32(254))
            qexp = Int32(254) if qexp_hi else qexp
            qexp_lo = cutlass.Boolean(qexp < Int32(1))
            qexp = Int32(1) if qexp_lo else qexp
            quant_scale = _i32_as_f32(qexp << Int32(23))

            # Step 5: scale registers; saturating cast to fp8 at TMA store.
            for i in cutlass.range(num_z, unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * quant_scale

            # Step 6a: ROW-SF store (per-lane, at (m_abs, n_group_abs)).
            if const_expr(row_info is not None):
                row_tensor, m_abs, n_group_abs, m_limit, n_group_limit = row_info
                in_bounds = (
                    cutlass.Boolean(m_abs < m_limit)
                    & cutlass.Boolean(n_group_abs < n_group_limit)
                )
                if in_bounds:
                    m_tile = m_abs // Int32(128)
                    row_in_tile = m_abs % Int32(128)
                    k_tile_idx = n_group_abs // Int32(4)
                    k_in_tile = n_group_abs % Int32(4)
                    row_base = (row_in_tile % Int32(32)) * Int32(16) + (
                        row_in_tile // Int32(32)
                    ) * Int32(4)
                    inner_off = row_base + k_in_tile
                    row_tensor[m_tile, k_tile_idx, inner_off] = cutlass.Int8(e8m0)

            # Step 6b: COL-SF store (one lane per N-col within this warp's
            # n_group; lane k writes col-SF byte for n_abs = warp_n_base + k).
            if const_expr(col_info is not None):
                col_tensor, m_warp_base, n_warp_base, lane_id, m_limit_c, n_limit_c = col_info
                n_abs_lane = n_warp_base + lane_id
                # m_group_abs for col-SF = warp's m-base // 32 (uniform within warp).
                m_group_abs = m_warp_base // Int32(32)
                in_bounds_c = (
                    cutlass.Boolean(n_abs_lane < n_limit_c)
                    & cutlass.Boolean(m_warp_base < m_limit_c)
                )
                if in_bounds_c:
                    col_n_tile = n_abs_lane // Int32(128)
                    col_row_in_tile = n_abs_lane % Int32(128)
                    col_k_tile_idx = m_group_abs // Int32(4)
                    col_k_in_tile = m_group_abs % Int32(4)
                    col_row_base = (col_row_in_tile % Int32(32)) * Int32(16) + (
                        col_row_in_tile // Int32(32)
                    ) * Int32(4)
                    col_inner_off = col_row_base + col_k_in_tile
                    col_tensor[col_n_tile, col_k_tile_idx, col_inner_off] = cutlass.Int8(e8m0)

        return None


# ---------------------------------------------------------------------------
# BlockscaledColQuantOnlyMixin (Session 1C — pure colwise quant)
# ---------------------------------------------------------------------------
#
# Per-column-block-amax quantization fused into the GEMM epilogue.  Replaces
# the standalone `_colwise_quantize_and_pack_kernel` (237 us at production
# shape) by computing per-(32-rows-in-M × 1-col-in-N) amax via 32 successive
# `warp_redux_sync` calls (one per N-col covered by the warp's 32-wide
# fragment) and writing TWO outputs from a single MMA:
#   - z_fp8       (e4m3, MxN) saturating cast per col scale
#   - z_col_isa   col-axis ISA-pack SF buffer (one byte per (32-rows, 1-col))
#
# Layout invariant (same as 1A-ext col): lane k of each warp stores the
# col-SF byte for n_abs = warp_n_base + k, m_group_abs = warp_m_base // 32.
#
# Reference: TE cudnn `grouped_gemm_quant.quant_sfd_col` — per-col warp
# redux with e8m0 cast-roundtrip for scale quantization fidelity.
# ---------------------------------------------------------------------------


class BlockscaledColQuantOnlyMixin(GemmDefaultEpiMixin):
    """GemmDefault + epi pure colwise blockscaled FP8 quant of D with
    ISA-pack col-SF store.  Equivalent to `colwise_quantize_and_pack` but
    fused into the GEMM epi (zero standalone kernel)."""

    _epi_ops = (
        *GemmDefaultEpiMixin._epi_ops,
        BlockscaledIsaColScaleStore("mZScaleIsaCol"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScaleIsaCol: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        col_info = epi_loop_tensors["mZScaleIsaCol"]
        if const_expr(col_info is not None):
            col_tensor, m_warp_base, n_warp_base, lane_id, m_limit_c, n_limit_c = col_info
            num_z = cute.size(tRS_rD)

            # Per-col amax via per-col warp redux.  lane_id holds row
            # m_warp_base + lane_id of one m_group; tRS_rD[k] is the value at
            # N-col warp_n_base + k.  Reducing |tRS_rD[k]| MAX across 32
            # lanes gives the block (32-rows × col k) amax.  The reduced
            # value is returned uniformly to all lanes — lane k captures
            # *its* col's e8m0 for the col-SF byte store after the loop.
            #
            # Each lane also keeps the 32 quant scales (one per col) to
            # scale its 32 tRS_rD elements before the saturating fp8 cast.

            my_col_e8m0 = Int32(0)

            for k in cutlass.range(num_z, unroll_full=True):
                val = tRS_rD[k]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax_k = cute.arch.warp_redux_sync(abs_val, "max")
                amax_k = cute.arch.fmax(amax_k, Float32(1e-4))

                # E8M0 integer+carry (matches Triton reference; same as iso32).
                amax_bits = _f32_as_i32(amax_k)
                biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
                mantissa_bits = amax_bits & Int32(0x7FFFFF)
                has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
                carry = Int32(1) if has_carry else Int32(0)
                e8m0_k = biased_exp - Int32(8) + carry
                is_normal = cutlass.Boolean(biased_exp > Int32(0))
                e8m0_k = e8m0_k if is_normal else Int32(0)
                is_pos = cutlass.Boolean(e8m0_k > Int32(0))
                e8m0_k = e8m0_k if is_pos else Int32(0)

                # Quant scale = 2^(254 - e8m0_k).
                qexp = Int32(254) - e8m0_k
                qexp_hi = cutlass.Boolean(qexp > Int32(254))
                qexp = Int32(254) if qexp_hi else qexp
                qexp_lo = cutlass.Boolean(qexp < Int32(1))
                qexp = Int32(1) if qexp_lo else qexp
                quant_scale_k = _i32_as_f32(qexp << Int32(23))

                # Scale this col's value in every lane.
                tRS_rD[k] = tRS_rD[k] * quant_scale_k

                # Lane k captures e8m0 for col k = warp_n_base + k.
                if lane_id == Int32(k):
                    my_col_e8m0 = e8m0_k

            # Col-SF store: lane k writes byte for n_abs = warp_n_base + k.
            n_abs_lane = n_warp_base + lane_id
            m_group_abs = m_warp_base // Int32(32)
            in_bounds_c = (
                cutlass.Boolean(n_abs_lane < n_limit_c)
                & cutlass.Boolean(m_warp_base < m_limit_c)
            )
            if in_bounds_c:
                col_n_tile = n_abs_lane // Int32(128)
                col_row_in_tile = n_abs_lane % Int32(128)
                col_k_tile_idx = m_group_abs // Int32(4)
                col_k_in_tile = m_group_abs % Int32(4)
                col_row_base = (col_row_in_tile % Int32(32)) * Int32(16) + (
                    col_row_in_tile // Int32(32)
                ) * Int32(4)
                col_inner_off = col_row_base + col_k_in_tile
                col_tensor[col_n_tile, col_k_tile_idx, col_inner_off] = cutlass.Int8(my_col_e8m0)

        return None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GemmDGatedMixin (from gemm_dgated.py)
# ---------------------------------------------------------------------------

class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileStore("mPostAct"), ColVecReduce("mColVecReduce"))
    _extra_param_fields = (
        ("act_bwd_fn", cutlass.Constexpr, None),
        ("implicit_dtype", cutlass.Constexpr, None),
    )

    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Override: force CopyUniversalOp for postact R2S when blockscaled.

        Same fix as GemmGatedMixin — avoids StMatrix/smem layout mismatch
        in blockscaled mode.
        """
        if const_expr(self.blockscaled):
            sPostAct = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
            copy_atom_postact_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.postact_dtype
            )
            tiled_copy_postact_r2s = cute.make_tiled_copy_S(
                copy_atom_postact_r2s, tiled_copy_r2s
            )
            tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
            batch_idx = tile_coord_mnkl[3]
            copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
                params.tma_atom_mPostAct,
                varlen_manager.offset_batch_epi(params.mPostAct, batch_idx),
                self.cta_tile_shape_postact_mn,
                params.epi_tile_mPostAct,
                sPostAct,
                tile_coord_mnkl,
            )
            return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact
        else:
            return GemmActMixin.epi_setup_postact(
                self, params, epi_smem_tensors, tiled_copy_r2s,
                tiled_copy_t2r, tile_coord_mnkl, varlen_manager, tidx,
            )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0  # RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = getattr(args, "rounding_mode", 0)
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        assert args.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]
        assert tRS_rC is not None
        implicit_dtype = params.implicit_dtype
        assert implicit_dtype.width == 16, "GemmDGatedMixin only supports 16bit for now"
        tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
        tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):  # Scale D by colvec
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            # Need to multiply before D is scaled by colvec_scale
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1])
                    )
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            row_sum,
                        )
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]

        if const_expr(tDrColVec is not None):  # Scale Out by colvec
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        # Write dXY (packed f16x2 as f32) back to D
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, implicit_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        # Return PostAct in acc_dtype; conversion happens in epi_convert_postact
        return tRS_rOut


# ---------------------------------------------------------------------------
# _fp8e4m3_to_f32 DSL function (from gemm_dgated.py)
# ---------------------------------------------------------------------------

@dsl_user_op
def _fp8e4m3_to_f32(x, *, loc=None, ip=None) -> Float32:
    """Convert scalar f8E4M3FN to f32 via PTX: fp8 -> f16 -> f32."""
    from cutlass._mlir.dialects import arith as _arith
    x_i8 = llvm.bitcast(T.i8(), x.ir_value(loc=loc, ip=ip) if hasattr(x, 'ir_value') else x,
                         loc=loc, ip=ip)
    x_i16 = llvm.zext(T.i16(), x_i8, loc=loc, ip=ip)
    f16_val = llvm.inline_asm(
        T.f16(), [x_i16],
        "{ .reg .b8 s; mov.b16 {s, _}, $1; cvt.rn.f16.e4m3 $0, s; }",
        "=h,h",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    f32_val = _arith.extf(T.f32(), f16_val, loc=loc, ip=ip)
    return Float32(f32_val)


# ---------------------------------------------------------------------------
# Hardware-accelerated quant helpers (TE cudnn parity optimizations)
# ---------------------------------------------------------------------------
# These replace multi-instruction ALU sequences with native SM100 PTX
# instructions, saving 6-10 registers per quant path.
# ---------------------------------------------------------------------------

@dsl_user_op
def _hardware_f32_to_e8m0(x: Float32, *, loc=None, ip=None) -> Float32:
    """PTX f32→ue8m0→f32 roundtrip using native BX8 instructions.

    Replaces the manual integer+carry e8m0 computation (~10 ALU ops) with
    a single PTX inline asm (2 native instructions).  Returns the e8m0
    scale as a bf16→f32 value (approximately 2^e8m0).

    Reference: TE cudnn ``cvt_f32_to_f8_to_f32``.
    """
    src = x.ir_value(loc=loc, ip=ip) if hasattr(x, 'ir_value') else x
    asm_tmpl = (
        "{\n"
        "  .reg .b16 bf_lo;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 bf_lo, 0f00000000, $1;\n"
        "  cvt.rn.bf16x2.ue8m0x2  $0, bf_lo;\n"
        "}"
    )
    result = llvm.inline_asm(
        T.f32(), [src],
        asm_tmpl,
        "=f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Float32(result)


@dsl_user_op
def _hardware_f32x4_to_f8x4_i32(fp32x4, fp8_dtype, *, loc=None, ip=None):
    """PTX vec4 f32→fp8 conversion, packed as int32.

    Uses ``cvt.rn.satfinite.e4m3x2.f32`` (x2 PTX) to convert 4 f32 values
    into 4 packed fp8 bytes in one int32.  Replaces the DSL generic
    ``r4.load().to(Float8E4M3FN)`` path which requires 4×4-element rmem
    allocation + cast + recast.
    """
    from cutlass._mlir.dialects import vector as _vector

    # Extract individual f32 values from the 4-element rmem tensor
    f32_vals = fp32x4.load()  # Vec4f32

    # Each PTX cvt.rn.satfinite.e4m3x2.f32 handles 2 f32 → 2 fp8
    asm_tmpl = (
        "{\n"
        "  .reg .b16 lo, hi;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
        "  mov.b32 $0, {lo, hi};\n"
        "}"
    )
    # Get IR values from the 4 elements
    def _ir(v, idx):
        if hasattr(f32_vals, 'ir_value'):
            vec_ir = f32_vals.ir_value(loc=loc, ip=ip)
            return Float32(_vector.extract(vec_ir, [], [idx])).ir_value(loc=loc, ip=ip)
        # Fallback: direct element access
        return Float32(fp32x4[idx]).ir_value(loc=loc, ip=ip)

    src0, src1, src2, src3 = _ir(f32_vals, 0), _ir(f32_vals, 1), _ir(f32_vals, 2), _ir(f32_vals, 3)
    packed = llvm.inline_asm(
        T.i32(),
        [src0, src1, src2, src3],
        asm_tmpl,
        "=r,f,f,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return packed


# Pre-computed constant: POW_2_127 = 2^127 ≈ 1.7e38.
# Builder pattern: _i32_as_f32(Int32(254) << Int32(23)).
# Used with hardware e8m0: quant_scale = POW_2_127 * rcp_approx(e8m0_float).


@dsl_user_op
def dswiglu_te_exp2(
    x,
    y,
    dout,
    *,
    loc=None,
    ip=None,
):
    """TE cudnn dswiglu formula using exp2 + rcp_approx.

    This is a local Sonic copy of TE's vectorized dswiglu inner loop:
      sig = rcp_approx(1 + exp2(-x * log2(e)))
      swish = x * sig
      dy = dout * swish
      dx = dout * y * sig * (1 + x * (1 - sig))
      out = swish * y

    It replaces quack.activation.dswiglu's tanh.approx path and keeps the
    same return contract (dx, dy, swiglu_out).
    """
    LOG2_E = Float32(1.4426950408889634)
    if const_expr(not isinstance(x, tuple)):
        sig_rcp = cute.math.exp2(Float32(0.0) - x * LOG2_E, fastmath=True) + Float32(1.0)
        sig = cute.arch.rcp_approx(sig_rcp)
        swish = x * sig
        dy = dout * swish
        dsig = x * (Float32(1.0) - sig)
        dx = dout * y * sig * (Float32(1.0) + dsig)
        out = swish * y
        return dx, dy, out
    else:
        neg_log2e = (-LOG2_E, -LOG2_E)
        sig_rcp = cute.arch.mul_packed_f32x2(x, neg_log2e, rnd="rn", ftz=False)
        sig_rcp = cute.arch.add_packed_f32x2(
            (
                cute.math.exp2(sig_rcp[0], fastmath=True),
                cute.math.exp2(sig_rcp[1], fastmath=True),
            ),
            (Float32(1.0), Float32(1.0)),
            rnd="rn",
            ftz=False,
        )
        sig = (cute.arch.rcp_approx(sig_rcp[0]), cute.arch.rcp_approx(sig_rcp[1]))
        swish = cute.arch.mul_packed_f32x2(x, sig, rnd="rn", ftz=False)
        dy = cute.arch.mul_packed_f32x2(dout, swish, rnd="rn", ftz=False)
        dx = cute.arch.mul_packed_f32x2(dout, y, rnd="rn", ftz=False)
        dx = cute.arch.mul_packed_f32x2(dx, sig, rnd="rn", ftz=False)
        dsig = cute.arch.mul_packed_f32x2(x, (Float32(1.0) - sig[0], Float32(1.0) - sig[1]), rnd="rn", ftz=False)
        dsig = cute.arch.add_packed_f32x2(dsig, (Float32(1.0), Float32(1.0)), rnd="rn", ftz=False)
        dx = cute.arch.mul_packed_f32x2(dx, dsig, rnd="rn", ftz=False)
        out = cute.arch.mul_packed_f32x2(swish, y, rnd="rn", ftz=False)
        return dx, dy, out


# ---------------------------------------------------------------------------
# FP8PreActLoad EpiOp (from gemm_dgated.py)
# ---------------------------------------------------------------------------

class FP8PreActLoad(EpiOp):
    """EpiOp: loads fp8 z + UE8M0 scales from gmem, dequants in registers.

    Param is a tuple (z_fp8_tensor, z_scales_tensor) passed as a single field.
    begin(): unpacks and captures coordinates.
    begin_loop(): computes subtile coordinates.
    The mixin's epi_visit_subtile loads fp8 bytes + scales and dequants.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        return 0

    def to_params(self, gemm, args):
        fp8 = getattr(args, self.name + "_fp8", None)
        scales = getattr(args, self.name + "_scales", None)
        if fp8 is not None and scales is not None:
            return {self.name: (assume_stride_divisibility(fp8), assume_stride_divisibility(scales))}
        return {self.name: None}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            fp8_tensor, scales_tensor = param
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]

            # Compute varlen M offset
            if const_expr(ctx.varlen_manager.varlen_m):
                m_offset = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
            else:
                m_offset = Int32(0)
            m_base = ctx.tile_coord_mnkl[0] * tile_M

            # Identity tensor partitioned for this thread's epilogue elements
            # This gives the exact (row, col) for each register position
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((tile_M, tile_N))
            )

            # N base in fp8 logical coordinates (tile_N f32 = 2*tile_N bf16 = 2*tile_N fp8)
            n_base_logical = ctx.tile_coord_mnkl[1] * tile_N * 2

            return (fp8_tensor, scales_tensor, tDcD, m_offset, m_base, n_base_logical)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            fp8_tensor, scales_tensor, tDcD, m_offset, m_base, n_base = state
            # Extract this subtile's identity coordinates
            tDcD_sub = cute.group_modes(tDcD, 3, cute.rank(tDcD))[None, None, None, epi_coord]
            return (fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base)
        return None


# ---------------------------------------------------------------------------
# GemmDGatedFP8PreActMixin (from gemm_dgated.py)
# ---------------------------------------------------------------------------

class GemmDGatedFP8PreActMixin(GemmDGatedMixin):
    """GemmDGated with fp8 PreAct: loads z_fp8 + scales in epilogue, no C tensor.

    When mFP8PreAct_fp8 is provided, tRS_rC is ignored (can be None).
    The epilogue loads fp8 z bytes + UE8M0 scale bytes via LDG, dequants
    in registers, and constructs tRS_rXY_f32x2 for dSwiGLU computation.

    Memory saving: eliminates 384MB z_bf16 temporary buffer.
    """
    _epi_ops = (
        *GemmDGatedMixin._epi_ops,
        FP8PreActLoad("mFP8PreAct"),
    )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = getattr(args, "rounding_mode", 0)
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        fp8_mode = getattr(args, "mFP8PreAct_fp8", None) is not None
        if not fp8_mode:
            assert args.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
            assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        return self.EpilogueParams(**d)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]

        fp8_preact_info = epi_loop_tensors["mFP8PreAct"]

        if const_expr(fp8_preact_info is not None):
            # ── FP8 PreAct path: use identity tensor for correct coordinates ──
            fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base = fp8_preact_info

            # tDcD_sub[i] gives (row_in_tile, col_in_tile) for each D register element
            # col is C's physical N (f32 = bf16x2). Each f32 maps to 2 fp8 bytes.
            num_d = cute.size(tDcD_sub)

            # Allocate fp8 register tensor (2x D elements for gate+up pairs)
            tRS_rXY_bf16_layout = cute.recast_tensor(tRS_rD, cutlass.BFloat16).layout
            tRS_rFP8 = cute.make_rmem_tensor(tRS_rXY_bf16_layout.shape, cutlass.Float8E4M3FN)

            # Load fp8 bytes using identity-derived coordinates
            # For each D[i] at (row, col), load fp8[row, col*2] and fp8[row, col*2+1]
            for i in cutlass.range(num_d, unroll_full=True):
                coord = tDcD_sub[i]
                row = coord[0]
                col = coord[1]
                m_abs = m_offset + m_base + row
                n0 = n_base + col * 2
                tRS_rFP8[2 * i] = fp8_tensor[m_abs, n0]
                tRS_rFP8[2 * i + 1] = fp8_tensor[m_abs, n0 + Int32(1)]

            # Vectorized fp8->f32 (DSL auto-packs vec4 -> nvgpu.cvt_fpext)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_bf16_layout.shape, Float32)
            tRS_rXY_f32x2.store(tRS_rFP8.load().to(Float32))

            # Blockscaled dequant using identity coordinates for correct group index
            for i in cutlass.range(num_d, unroll_full=True):
                coord = tDcD_sub[i]
                row = coord[0]
                col = coord[1]
                m_abs = m_offset + m_base + row
                n0 = n_base + col * 2
                group_0 = n0 >> Int32(5)
                group_1 = (n0 + Int32(1)) >> Int32(5)
                scale_0 = _i32_as_f32(Int32(scales_tensor[m_abs, group_0]) << Int32(23))
                scale_1 = _i32_as_f32(Int32(scales_tensor[m_abs, group_1]) << Int32(23))
                tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            # ── Standard bf16 PreAct path ──
            assert tRS_rC is not None
            implicit_dtype = params.implicit_dtype
            tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
            tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))

        # ── dSwiGLU + colvec scale/reduce (unchanged from parent) ──
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1])
                    )
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            row_sum,
                        )
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]

        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )

        # Write dXY (packed f16x2 as f32) back to D
        if const_expr(fp8_preact_info is not None):
            pack_dtype = cutlass.BFloat16
        else:
            pack_dtype = params.implicit_dtype
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, pack_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(pack_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut


# ---------------------------------------------------------------------------
# GemmDGatedFP8CLoadMixin (from gemm_dgated.py)
# ---------------------------------------------------------------------------

class GemmDGatedFP8CLoadMixin(GemmDGatedMixin):
    """GemmDGated with TMA-based fp8 C load.

    Loads fp8 z (PreAct) via TMA to smem, then fp8->f32 conversion in registers.
    Eliminates 384MB z_bf16 temporary buffer.

    Key insight: View z_fp8 (TK, 2I) fp8 as (TK, I) Int16 to match D's shape.
    Each Int16 = 2 packed fp8 values (gate + up), mirroring D's f32 = 2 packed bf16.
    This avoids changing the epi_tile (shared by kernel for both C and D).

    C tensor: z_fp8.view(int16) -> (TK, I) Int16
    Scale tensor: z_scales (TK, 2I/32) uint8 — loaded via EpiOp LDG

    Key overrides:
    - _setup_attributes: create Int16 smem layout for C (same epi_tile, different dtype)
    - epilog_smem_load_and_partition: double the register layout (Int16 -> 2 fp8 elements)
    - epi_visit_subtile: unpack Int16 -> 2 fp8 -> 2 f32 + blockscaled dequant + dSwiGLU
    - epi_to_underlying_arguments: handle Int16 c_dtype
    """

    _epi_ops = (
        *GemmDGatedMixin._epi_ops,
        FP8PreActLoad("mFP8PreAct"),
    )

    # No _make_tma_epi_atoms_and_tensors override needed:
    # Int16 C has the same shape as D (TK, I), so the standard epi_tile works.
    # The parent's staticmethod handles TMA atom creation correctly.


    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = getattr(args, "rounding_mode", 0)
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        # Int16 C: c_dtype is Int16 (2 packed fp8), allow it
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        return self.EpilogueParams(**d)

    def _setup_attributes(self, epilogue_args, varlen_args):
        """Override: create Int16 smem layout for fp8 C.

        View z_fp8 (TK, 2I) fp8 as (TK, I) Int16 to match D's shape (TK, I) f32.
        Each Int16 = 2 packed fp8 values (gate + up), just as each f32 = 2 packed bf16.
        This way C and D share the same epi_tile, avoiding kernel-level changes.
        """
        super()._setup_attributes(epilogue_args, varlen_args)
        if const_expr(self.c_dtype is not None and self.c_dtype == cutlass.Int16):
            import cutlass.utils.blackwell_helpers as sm100_utils
            # Int16 C: same shape as D, but 16-bit element -> different smem swizzle
            self.epi_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.epi_c_stage
            )

    def epilog_smem_load_and_partition(
        self, tiled_copy_t2r, c_layout, dtype, sC, tRS_rD_layout, tidx
    ):
        """Override: for Int16 C, keep register layout same as D (N elements).

        Int16 C has N elements (same as D's N f32). No doubling needed here.
        In epi_visit_subtile, we recast N Int16 -> 2N fp8 -> 2N f32.
        """
        if const_expr(dtype == cutlass.Int16):
            # Same register shape as D (N elements), just Int16 dtype.
            # The parent's default handles this correctly.
            return GemmSm100.epilog_smem_load_and_partition(
                self, tiled_copy_t2r, c_layout, dtype, sC, tRS_rD_layout, tidx
            )
        else:
            return GemmSm100.epilog_smem_load_and_partition(
                self, tiled_copy_t2r, c_layout, dtype, sC, tRS_rD_layout, tidx
            )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]

        if const_expr(self.c_dtype == cutlass.Int16):
            # ── Int16 C path: tRS_rC has N Int16 elements from TMA ──
            # Each Int16 = 2 packed fp8 values (gate + up)
            # Recast N Int16 -> 2N fp8, then convert to 2N f32
            tRS_rC_fp8 = cute.recast_tensor(tRS_rC, cutlass.Float8E4M3FN)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rC_fp8.layout.shape, Float32)
            tRS_rXY_f32x2.store(tRS_rC_fp8.load().to(Float32))

            # Blockscaled dequant: multiply by 2^(e8m0 << 23) per group
            # Scale info from EpiOp
            fp8_preact_info = epi_loop_tensors["mFP8PreAct"]
            if const_expr(fp8_preact_info is not None):
                # Scales loaded via EpiOp (small data, LDG is fine)
                fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base = fp8_preact_info
                num_d = cute.size(tDcD_sub)
                for i in cutlass.range(num_d, unroll_full=True):
                    coord = tDcD_sub[i]
                    row, col = coord[0], coord[1]
                    m_abs = m_offset + m_base + row
                    n0 = n_base + col * 2
                    group_0 = n0 >> Int32(5)
                    group_1 = (n0 + Int32(1)) >> Int32(5)
                    scale_0 = _i32_as_f32(Int32(scales_tensor[m_abs, group_0]) << Int32(23))
                    scale_1 = _i32_as_f32(Int32(scales_tensor[m_abs, group_1]) << Int32(23))
                    tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                    tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            # ── Standard bf16 C path ──
            assert tRS_rC is not None
            implicit_dtype = params.implicit_dtype
            tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
            tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))

        # ── dSwiGLU + colvec (shared between both paths) ──
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rD_scaled_mn[m, 2*n], tRS_rD_scaled_mn[m, 2*n+1] = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2*n], tRS_rD_mn[m, 2*n+1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                tRS_rdXY_f32x2[2*i], tRS_rdXY_f32x2[2*i+1], tRS_rOut[i] = params.act_bwd_fn(
                    tRS_rXY_f32x2[2*i], tRS_rXY_f32x2[2*i+1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (tRS_rdXY_f32x2[4*i], tRS_rdXY_f32x2[4*i+2]), \
                (tRS_rdXY_f32x2[4*i+1], tRS_rdXY_f32x2[4*i+3]), \
                (tRS_rOut[2*i], tRS_rOut[2*i+1]) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4*i], tRS_rXY_f32x2[4*i+2]),
                    (tRS_rXY_f32x2[4*i+1], tRS_rXY_f32x2[4*i+3]),
                    (tRS_rD_scaled[2*i], tRS_rD_scaled[2*i+1]),
                )
        if const_expr(tDrColVecReduce is not None):
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1]))
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2*n], tRS_rD_mn[m, 2*n+1]),
                            (tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1]), row_sum)
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        # Write dXY back to D
        if const_expr(self.c_dtype == cutlass.Int16):
            pack_dtype = cutlass.BFloat16
        else:
            pack_dtype = params.implicit_dtype
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, pack_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(pack_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut


# ---------------------------------------------------------------------------
# Iso32DXYStore EpiOp (NEW — side-channel FP8 dXY + dual ISA SF for DGated)
# ---------------------------------------------------------------------------
#
# Captures per-element (m_abs, n_dXY_abs) coordinates so the mixin can:
#   * scatter-store FP8 dXY bytes to gmem  mDZFp8 [TK, 2I]  uint8
#   * scatter-store row-axis ISA-pack SF bytes  mDZScaleIsaRow  uint8
#   * scatter-store col-axis ISA-pack SF bytes  mDZScaleIsaCol  uint8
#
# Three optional kwargs (any subset can be None for A/B verification):
#   mDZFp8Iso32_fp8   : (total_TK, 2I) uint8/Float8E4M3FN
#   mDZFp8Iso32_row   : (num_m_tiles, k_tiles, 512) uint8  (row-ISA-pack, dXY-N domain)
#   mDZFp8Iso32_col   : (num_n_tiles, col_k_tiles, 512) uint8  (col-ISA-pack)
#
# The single 32x32 iso32 amax (over the dXY-tensor) is shared between the
# FP8 byte and BOTH scale bytes — this is the invariant that makes the
# fusion correct vs `_dual_varlen_iso32_quantize_kernel`.
# ---------------------------------------------------------------------------


class Iso32DXYStore(EpiOp):
    """EpiOp: captures coords for per-element scatter store of fp8 dXY +
    dual ISA-pack SF bytes.  Mirrors :class:`FP8PreActLoad` but in the
    write direction; passes a single tuple-payload through to the mixin's
    ``epi_visit_subtile``.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        return 0

    def to_params(self, gemm, args):
        fp8 = getattr(args, self.name + "_fp8", None)
        row = getattr(args, self.name + "_row", None)
        col = getattr(args, self.name + "_col", None)
        if fp8 is None and row is None and col is None:
            return {self.name: None}
        return {
            self.name: (
                assume_stride_divisibility(fp8) if fp8 is not None else None,
                assume_stride_divisibility(row) if row is not None else None,
                assume_stride_divisibility(col) if col is not None else None,
            )
        }

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            fp8_t, row_t, col_t = param
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]
            if const_expr(ctx.varlen_manager.varlen_m):
                m_offset = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[
                    ctx.tile_coord_mnkl[3] + Int32(1)
                ]
            else:
                m_offset = Int32(0)
                # Non-varlen: use a sentinel so the bounds check is effectively
                # always true (avoids Python `if` on a runtime value below).
                m_limit = Int32(2_000_000_000)
            m_base = ctx.tile_coord_mnkl[0] * tile_M
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((tile_M, tile_N))
            )
            # dXY-N base in dXY logical coords (tile_N D-cols == 2*tile_N dXY-cols).
            n_base_dxy = ctx.tile_coord_mnkl[1] * tile_N * 2
            lane_id = ctx.tidx % Int32(32)
            return (fp8_t, row_t, col_t, tDcD, m_offset, m_base, n_base_dxy, m_limit, lane_id)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            fp8_t, row_t, col_t, tDcD, m_offset, m_base, n_base_dxy, m_limit, lane_id = state
            tDcD_sub = cute.group_modes(tDcD, 3, cute.rank(tDcD))[None, None, None, epi_coord]
            return (fp8_t, row_t, col_t, tDcD_sub, m_offset, m_base, n_base_dxy, m_limit, lane_id)
        return None


# ---------------------------------------------------------------------------
# GemmDGatedFP8CLoadIso32QuantMixin
# ---------------------------------------------------------------------------
#
# Side-channel iso32 FP8-D quant for DGated FP8-C-load.  Keeps BF16 D path
# fully intact; ADDITIONALLY writes:
#   - dz_fp8   (Float8E4M3FN, varlen TK x 2I) via per-byte scatter
#   - dz_sf_row  ISA-pack row-axis SF (in dXY-N domain)
#   - dz_sf_col  ISA-pack col-axis SF
#
# Replaces the standalone `_dual_varlen_iso32_quantize_kernel(dz_bf16)`
# (~102 us at production T=8192 E=8) once flag-wired in functional/__init__.py.
#
# Reference: TE `grouped_gemm_dswiglu_quant` epi pattern + 1A-ext mixin
# invariants (block constancy / row=col SF equality at 32x32 blocks).
# ---------------------------------------------------------------------------


class GemmDGatedFP8CLoadIso32QuantMixin(GemmDGatedFP8CLoadMixin):
    """GemmDGatedFP8CLoad + side-channel iso32 FP8 dXY quant (additive)."""

    _epi_ops = (
        *GemmDGatedFP8CLoadMixin._epi_ops,
        Iso32DXYStore("mDZFp8Iso32"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None
        mDZFp8Iso32_fp8: Optional[cute.Tensor] = None
        mDZFp8Iso32_row: Optional[cute.Tensor] = None
        mDZFp8Iso32_col: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]

        if const_expr(self.c_dtype == cutlass.Int16):
            tRS_rC_fp8 = cute.recast_tensor(tRS_rC, cutlass.Float8E4M3FN)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rC_fp8.layout.shape, Float32)
            tRS_rXY_f32x2.store(tRS_rC_fp8.load().to(Float32))
            fp8_preact_info = epi_loop_tensors["mFP8PreAct"]
            if const_expr(fp8_preact_info is not None):
                fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base = fp8_preact_info
                num_d = cute.size(tDcD_sub)
                for i in cutlass.range(num_d, unroll_full=True):
                    coord = tDcD_sub[i]
                    row, col = coord[0], coord[1]
                    m_abs = m_offset + m_base + row
                    n0 = n_base + col * 2
                    group_0 = n0 >> Int32(5)
                    group_1 = (n0 + Int32(1)) >> Int32(5)
                    scale_0 = _i32_as_f32(Int32(scales_tensor[m_abs, group_0]) << Int32(23))
                    scale_1 = _i32_as_f32(Int32(scales_tensor[m_abs, group_1]) << Int32(23))
                    tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                    tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            assert tRS_rC is not None
            implicit_dtype = params.implicit_dtype
            tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
            tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))

        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(tRS_rD_scaled, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rD_scaled_mn[m, 2*n], tRS_rD_scaled_mn[m, 2*n+1] = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2*n], tRS_rD_mn[m, 2*n+1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                tRS_rdXY_f32x2[2*i], tRS_rdXY_f32x2[2*i+1], tRS_rOut[i] = params.act_bwd_fn(
                    tRS_rXY_f32x2[2*i], tRS_rXY_f32x2[2*i+1], tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (tRS_rdXY_f32x2[4*i], tRS_rdXY_f32x2[4*i+2]), \
                (tRS_rdXY_f32x2[4*i+1], tRS_rdXY_f32x2[4*i+3]), \
                (tRS_rOut[2*i], tRS_rOut[2*i+1]) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4*i], tRS_rXY_f32x2[4*i+2]),
                    (tRS_rXY_f32x2[4*i+1], tRS_rXY_f32x2[4*i+3]),
                    (tRS_rD_scaled[2*i], tRS_rD_scaled[2*i+1]),
                )
        if const_expr(tDrColVecReduce is not None):
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = layout_utils.convert_layout_zero_stride(tDrColVecReduce, tDrColVecReduce.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVecReduce.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVecReduce.layout)
                for m in cutlass.range(cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True):
                    row_sum = cute.arch.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]), (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1]))
                    for n in cutlass.range(1, cute.size(tDrColVecReduce_mn, mode=[1]) // 2, unroll_full=True):
                        row_sum = cute.arch.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2*n], tRS_rD_mn[m, 2*n+1]),
                            (tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1]), row_sum)
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]
        if const_expr(tDrColVec is not None):
            if const_expr(self.arch < 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True):
                        tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1] = cute.arch.mul_packed_f32x2(
                            (tRS_rOut_mn[m, 2*n], tRS_rOut_mn[m, 2*n+1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )

        # ── NEW: iso32 side-channel FP8 dXY quant ──
        #
        # tRS_rdXY_f32x2 holds 2*num_d f32 dXY values per lane, laid out so
        # indices [4*i, 4*i+1, 4*i+2, 4*i+3] map to consecutive dXY-cols
        # (2*col_D, 2*col_D+1, 2*(col_D+1), 2*(col_D+1)+1) for D-row coord
        # (row_2i, row_2i+1).  We need iso32 32×32 blocks in the dXY-N domain:
        # since each warp's 32 lanes span 32 contiguous M-rows AND share the
        # same N-cols, the 32 lanes form a 32x(num_dxy_per_lane) iso32 block(s).
        iso32_info = epi_loop_tensors["mDZFp8Iso32"]
        if const_expr(iso32_info is not None):
            fp8_t, row_t, col_t, tDcD_sub, m_offset, m_base, n_base_dxy, m_limit, lane_id = iso32_info
            num_d = cute.size(tDcD_sub)
            # num_dxy per lane = 2 * num_d (each D slot expands to 2 dXY cols).
            # Group structure: 32 dXY cols == 1 iso32 group.  num_d must be
            # a constexpr multiple of 16 so num_dxy_per_lane (= 2*num_d) is
            # a multiple of 32.  Asserted at JIT time.
            num_dxy = const_expr(2 * num_d)
            assert num_dxy % 32 == 0, "Iso32 fusion requires 2*num_d_per_lane to be multiple of 32"
            num_groups = const_expr(num_dxy // 32)

            for g in cutlass.range(num_groups, unroll_full=True):
                # Step 1: per-thread amax over 32 contiguous dXY vals.
                amax = Float32(0.0)
                for k in cutlass.range(32, unroll_full=True):
                    v = tRS_rdXY_f32x2[g * 32 + k]
                    neg = Float32(0.0) - v
                    av = cute.arch.fmax(v, neg)
                    amax = cute.arch.fmax(amax, av)
                # Step 2: warp_redux across 32 lanes -> block (32x32) amax.
                amax = cute.arch.warp_redux_sync(amax, "max")
                amax = cute.arch.fmax(amax, Float32(1e-4))
                # Step 3: integer+carry E8M0 (matches Triton reference).
                amax_bits = _f32_as_i32(amax)
                biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
                mantissa_bits = amax_bits & Int32(0x7FFFFF)
                has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
                carry = Int32(1) if has_carry else Int32(0)
                e8m0 = biased_exp - Int32(8) + carry
                is_normal = cutlass.Boolean(biased_exp > Int32(0))
                e8m0 = e8m0 if is_normal else Int32(0)
                is_pos = cutlass.Boolean(e8m0 > Int32(0))
                e8m0 = e8m0 if is_pos else Int32(0)
                # Step 4: quant_scale = 2^(254 - e8m0).
                qexp = Int32(254) - e8m0
                qexp_hi = cutlass.Boolean(qexp > Int32(254))
                qexp = Int32(254) if qexp_hi else qexp
                qexp_lo = cutlass.Boolean(qexp < Int32(1))
                qexp = Int32(1) if qexp_lo else qexp
                quant_scale = _i32_as_f32(qexp << Int32(23))

                # Step 5: scale into 32-element f32 buffer, then vector cast
                # to fp8 (DSL auto-packs to vec4 cvt_fptrunc).  Original
                # tRS_rdXY_f32x2 is left untouched so the BF16 pack-back at
                # the end of this function preserves the parent's contract.
                src32 = cute.make_rmem_tensor(cute.make_layout(32), Float32)
                dst32 = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                if const_expr(True):
                    for k in cutlass.range(32, unroll_full=True):
                        src32[k] = tRS_rdXY_f32x2[g * 32 + k] * quant_scale
                    dst32.store(src32.load().to(cutlass.Float8E4M3FN))

                # Step 6a: per-byte FP8 scatter store of dXY (via Uint8 recast
                # to avoid per-element fp8 scatter store quirks).
                if const_expr(fp8_t is not None):
                    fp8_t_u8 = cute.recast_tensor(fp8_t, cutlass.Uint8)
                    dst32_u8 = cute.recast_tensor(dst32, cutlass.Uint8)
                    for k in cutlass.range(32, unroll_full=True):
                        dxy_idx = g * 32 + k
                        d_idx = dxy_idx // 2
                        subbit = dxy_idx % 2
                        coord = tDcD_sub[d_idx]
                        row = coord[0]
                        col = coord[1]
                        m_abs_e = m_offset + m_base + row
                        n_dxy_abs = n_base_dxy + col * 2 + subbit
                        ok_m = cutlass.Boolean(m_abs_e < m_limit)
                        if ok_m:
                            fp8_t_u8[m_abs_e, n_dxy_abs] = dst32_u8[k]
                # Step 6b: row-ISA-pack SF store.
                if const_expr(row_t is not None):
                    d_idx0 = (g * 32) // 2
                    coord0 = tDcD_sub[d_idx0]
                    row0 = coord0[0]
                    col0 = coord0[1]
                    m_abs_r = m_offset + m_base + row0
                    n_group_abs = (n_base_dxy + col0 * 2 + g * 32) >> Int32(5)
                    n_group_limit = row_t.shape[1] * Int32(4)
                    ok_r = (
                        cutlass.Boolean(m_abs_r < m_limit)
                        & cutlass.Boolean(n_group_abs < n_group_limit)
                    )
                    if ok_r:
                        m_tile = m_abs_r // Int32(128)
                        row_in_tile = m_abs_r % Int32(128)
                        k_tile_idx = n_group_abs // Int32(4)
                        k_in_tile = n_group_abs % Int32(4)
                        row_base = (row_in_tile % Int32(32)) * Int32(16) + (row_in_tile // Int32(32)) * Int32(4)
                        inner_off = row_base + k_in_tile
                        row_t[m_tile, k_tile_idx, inner_off] = cutlass.Int8(e8m0)
                # Step 6c: col-ISA-pack SF store.
                if const_expr(col_t is not None):
                    d_idx0 = (g * 32) // 2
                    coord0 = tDcD_sub[d_idx0]
                    row_lane = coord0[0]
                    col_lane = coord0[1]
                    m_abs_lane = m_offset + m_base + row_lane
                    m_warp_base = m_abs_lane - lane_id
                    warp_n_dxy_base_g = n_base_dxy + col_lane * 2 + g * 32
                    n_abs_lane = warp_n_dxy_base_g + lane_id
                    m_group_abs = m_warp_base // Int32(32)
                    n_limit_c = col_t.shape[0] * Int32(128)
                    ok_c = (
                        cutlass.Boolean(n_abs_lane < n_limit_c)
                        & cutlass.Boolean(m_warp_base < m_limit)
                    )
                    if ok_c:
                        col_n_tile = n_abs_lane // Int32(128)
                        col_row_in_tile = n_abs_lane % Int32(128)
                        col_k_tile_idx = m_group_abs // Int32(4)
                        col_k_in_tile = m_group_abs % Int32(4)
                        col_row_base = (col_row_in_tile % Int32(32)) * Int32(16) + (col_row_in_tile // Int32(32)) * Int32(4)
                        col_inner_off = col_row_base + col_k_in_tile
                        col_t[col_n_tile, col_k_tile_idx, col_inner_off] = cutlass.Int8(e8m0)

        # ── Write dXY back to BF16 D (only when iso32 is NOT active) ──
        if const_expr(iso32_info is None):
            if const_expr(self.c_dtype == cutlass.Int16):
                pack_dtype = cutlass.BFloat16
            else:
                pack_dtype = params.implicit_dtype
            tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, pack_dtype)
            tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(pack_dtype))
            tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut


# ---------------------------------------------------------------------------
# Y1sColQuantStore EpiOp — side-channel FP8 y1s + col-ISA SF for DGated
# ---------------------------------------------------------------------------
#
# Captures per-element (m_abs, n_abs) coordinates for y1s so the mixin can:
#   * scatter-store FP8 y1s bytes to gmem  mY1sFp8  [TK, I]  uint8
#   * scatter-store col-axis ISA-pack SF bytes  mY1sScaleIsaCol  uint8
#
# Reference: TE cudnn `quant_sfd_col` — per-col warp redux with e8m0
# cast-roundtrip.  Single warp_redux_sync per N-col, same invariant as
# BlockscaledColQuantOnlyMixin but applied to DGated postact (tRS_rOut),
# NOT to D output.
# ---------------------------------------------------------------------------


class Y1sColQuantStore(EpiOp):
    """EpiOp: captures coords for per-element scatter store of fp8 y1s +
    col-ISA SF bytes.  Mirrors :class:`FP8PreActLoad` but in the write
    direction; passes a single tuple-payload through to the mixin's
    ``epi_visit_subtile``.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        return 0

    def to_params(self, gemm, args):
        fp8 = getattr(args, self.name + "_fp8", None)
        col = getattr(args, self.name + "_col", None)
        if fp8 is None and col is None:
            return {self.name: None}
        return {
            self.name: (
                assume_stride_divisibility(fp8) if fp8 is not None else None,
                assume_stride_divisibility(col) if col is not None else None,
            )
        }

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            fp8_t, col_t = param
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]
            if const_expr(ctx.varlen_manager.varlen_m):
                m_offset = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[
                    ctx.tile_coord_mnkl[3] + Int32(1)
                ]
            else:
                m_offset = Int32(0)
                m_limit = Int32(2_000_000_000)
            m_base = ctx.tile_coord_mnkl[0] * tile_M
            n_base = ctx.tile_coord_mnkl[1] * tile_N
            n_limit = col_t.shape[0] * Int32(128)  # total N for bounds check
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((tile_M, tile_N))
            )
            lane_id = ctx.tidx % Int32(32)
            return (fp8_t, col_t, tDcD, m_offset, m_base, n_base, n_limit, m_limit, lane_id)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            fp8_t, col_t, tDcD, m_offset, m_base, n_base, n_limit, m_limit, lane_id = state
            tDcD_sub = cute.group_modes(tDcD, 3, cute.rank(tDcD))[None, None, None, epi_coord]
            return (fp8_t, col_t, tDcD_sub, m_offset, m_base, n_base, n_limit, m_limit, lane_id)
        return None


# ---------------------------------------------------------------------------
# GemmDGatedFP8CLoadY1sColQuantMixin
# ---------------------------------------------------------------------------
#
# Side-channel FP8 y1s + col-ISA SF quant for DGated FP8-C-load.
# Keeps BF16 y1s path fully intact; ADDITIONALLY writes:
#   - y1s_fp8     (Float8E4M3FN, varlen TK x I) via per-byte scatter
#   - y1s_sf_col   ISA-pack col-axis SF
#
# Replaces the standalone `_colwise_quantize_and_pack_kernel(y1s)`
# (~79 us at production T=8192 E=8) once flag-wired.
#
# Reference: TE cudnn `quant_sfd_col` — per-col warp_redux_sync with
# e8m0 roundtrip, same math as BlockscaledColQuantOnlyMixin but applied
# to DGated postact (tRS_rOut) instead of D output (tRS_rD).
# ---------------------------------------------------------------------------


class GemmDGatedFP8CLoadY1sColQuantMixin(GemmDGatedFP8CLoadMixin):
    """GemmDGatedFP8CLoad + side-channel y1s FP8 + col-SF quant (additive)."""

    _epi_ops = (
        *GemmDGatedFP8CLoadMixin._epi_ops,
        Y1sColQuantStore("mY1sColQuant"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None
        mY1sColQuant_fp8: Optional[cute.Tensor] = None
        mY1sColQuant_col: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # Run the parent epi_visit_subtile (FP8CLoad) to get tRS_rOut.
        tRS_rOut = GemmDGatedFP8CLoadMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        # ── Side-channel: y1s colwise FP8 quant ──
        y1s_info = epi_loop_tensors["mY1sColQuant"]
        if const_expr(y1s_info is not None):
            fp8_t, col_t, tDcD_sub, m_offset, m_base, n_base, n_limit, m_limit, lane_id = y1s_info
            num_d = cute.size(tDcD_sub)

            fp8_t_u8 = cute.recast_tensor(fp8_t, cutlass.Uint8) if const_expr(fp8_t is not None) else None

            # Per-col amax + quant: group four N columns so the expensive FP8
            # cast uses one vec4 conversion, matching TE's quant_sfd_col shape.
            for j4 in cutlass.range(num_d // 4, unroll_full=True):
                if const_expr(fp8_t is not None):
                    qvals = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                e8m0s = cute.make_rmem_tensor(cute.make_layout(4), Int32)

                for jj in cutlass.range(4, unroll_full=True):
                    j = j4 * 4 + jj
                    val = tRS_rOut[j]
                    neg = Float32(0.0) - val
                    abs_val = cute.arch.fmax(val, neg)
                    amax_j = cute.arch.warp_redux_sync(abs_val, "max")
                    amax_j = cute.arch.fmax(amax_j, Float32(1e-4))

                    # Integer e8m0 for uint8 ISA SF store (must match consumer encoding).
                    amax_bits = _f32_as_i32(amax_j)
                    biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
                    mantissa_bits = amax_bits & Int32(0x7FFFFF)
                    has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
                    carry = Int32(1) if has_carry else Int32(0)
                    e8m0 = biased_exp - Int32(8) + carry
                    e8m0 = e8m0 if cutlass.Boolean(biased_exp > Int32(0)) else Int32(0)
                    e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
                    e8m0s[jj] = e8m0

                    # Hardware-accelerated quant_scale (TE cudnn parity).
                    # e8m0_float ≈ 2^actual_exp; scale = 256 * rcp = 2^(8-actual_exp).
                    if const_expr(fp8_t is not None or self.postact_dtype == cutlass.Float8E4M3FN):
                        qexp = Int32(254) - e8m0
                        qexp_hi = cutlass.Boolean(qexp > Int32(254))
                        qexp = Int32(254) if qexp_hi else qexp
                        qexp_lo = cutlass.Boolean(qexp < Int32(1))
                        qexp = Int32(1) if qexp_lo else qexp
                        quant_scale = _i32_as_f32(qexp << Int32(23))
                        if const_expr(fp8_t is not None):
                            qvals[jj] = val * quant_scale
                        else:
                            tRS_rOut[j] = val * quant_scale

                if const_expr(fp8_t is not None):
                    qvals_fp8 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float8E4M3FN)
                    qvals_fp8.store(qvals.load().to(cutlass.Float8E4M3FN))
                    qvals_u8 = cute.recast_tensor(qvals_fp8, cutlass.Uint8)

                for jj in cutlass.range(4, unroll_full=True):
                    j = j4 * 4 + jj
                    coord = tDcD_sub[j]
                    row = coord[0]
                    col = coord[1]
                    n_abs = n_base + col
                    m_abs = m_offset + m_base + row
                    m_warp_base = m_abs - lane_id
                    m_group_abs = m_warp_base // Int32(32)
                    if const_expr(col_t is not None):
                        col_n_tile = n_abs // Int32(128)
                        col_row_in_tile = n_abs % Int32(128)
                        col_k_tile_idx = m_group_abs // Int32(4)
                        col_k_in_tile = m_group_abs % Int32(4)
                        col_row_base = (col_row_in_tile % Int32(32)) * Int32(16) + (
                            col_row_in_tile // Int32(32)
                        ) * Int32(4)
                        col_inner_off = col_row_base + col_k_in_tile
                        ok_c = (
                            cutlass.Boolean(n_abs < n_limit)
                            & cutlass.Boolean(m_warp_base < m_limit)
                            & cutlass.Boolean(lane_id == Int32(0))
                        )
                        if ok_c:
                            col_t[col_n_tile, col_k_tile_idx, col_inner_off] = cutlass.Int8(e8m0s[jj])

                    if const_expr(fp8_t is not None):
                        ok_m = cutlass.Boolean(m_abs < m_limit)
                        if ok_m:
                            fp8_t_u8[m_abs, n_abs] = qvals_u8[jj]

            for j_tail in cutlass.range(num_d - (num_d % 4), num_d, unroll_full=True):
                val = tRS_rOut[j_tail]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax_j = cute.arch.warp_redux_sync(abs_val, "max")
                amax_j = cute.arch.fmax(amax_j, Float32(1e-4))
                amax_bits = _f32_as_i32(amax_j)
                biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
                mantissa_bits = amax_bits & Int32(0x7FFFFF)
                has_carry = cutlass.Boolean(mantissa_bits > Int32(0x600000))
                carry = Int32(1) if has_carry else Int32(0)
                e8m0 = biased_exp - Int32(8) + carry
                e8m0 = e8m0 if cutlass.Boolean(biased_exp > Int32(0)) else Int32(0)
                e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
                val_scaled = val
                if const_expr(fp8_t is not None or self.postact_dtype == cutlass.Float8E4M3FN):
                    qexp = Int32(254) - e8m0
                    qexp_hi = cutlass.Boolean(qexp > Int32(254))
                    qexp = Int32(254) if qexp_hi else qexp
                    qexp_lo = cutlass.Boolean(qexp < Int32(1))
                    qexp = Int32(1) if qexp_lo else qexp
                    quant_scale = _i32_as_f32(qexp << Int32(23))
                    val_scaled = val * quant_scale
                    if const_expr(fp8_t is None):
                        tRS_rOut[j_tail] = val_scaled
                coord = tDcD_sub[j_tail]
                row = coord[0]
                col = coord[1]
                n_abs = n_base + col
                m_abs = m_offset + m_base + row
                m_warp_base = m_abs - lane_id
                m_group_abs = m_warp_base // Int32(32)
                if const_expr(col_t is not None):
                    col_n_tile = n_abs // Int32(128)
                    col_row_in_tile = n_abs % Int32(128)
                    col_k_tile_idx = m_group_abs // Int32(4)
                    col_k_in_tile = m_group_abs % Int32(4)
                    col_row_base = (col_row_in_tile % Int32(32)) * Int32(16) + (
                        col_row_in_tile // Int32(32)
                    ) * Int32(4)
                    col_inner_off = col_row_base + col_k_in_tile
                    ok_c = (
                        cutlass.Boolean(n_abs < n_limit)
                        & cutlass.Boolean(m_warp_base < m_limit)
                        & cutlass.Boolean(lane_id == Int32(0))
                    )
                    if ok_c:
                        col_t[col_n_tile, col_k_tile_idx, col_inner_off] = cutlass.Int8(e8m0)
                if const_expr(fp8_t is not None):
                    ok_m = cutlass.Boolean(m_abs < m_limit)
                    if ok_m:
                        r4 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                        r4[0] = val_scaled
                        r4_fp8 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float8E4M3FN)
                        r4_fp8.store(r4.load().to(cutlass.Float8E4M3FN))
                        r4_u8 = cute.recast_tensor(r4_fp8, cutlass.Uint8)
                        fp8_t_u8[m_abs, n_abs] = r4_u8[0]

        return tRS_rOut
