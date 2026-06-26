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


def _make_postact_r2s_tiled_copy(gemm, copy_atom_postact_r2s, tiled_copy_r2s, params):
    """Build the register->smem tiled copy for the half-N gated postact SMEM.

    This is QuACK's stock ``make_tiled_copy_S(aux_atom, tiled_copy_r2s)``: that
    copy inherits D's full-N tiler and over-emits 2x into the half-N postact
    smem, but for the (4,1) epilogue-warp shape the over-emission has warp-N
    stride 0 (a harmless self-overwrite), so it is correct.

    The silent corruption of Dao-AILab/sonic-moe issue #63 / quack PR #133 only
    appears under the (2,2) tmem epilogue-warp shape, which QuACK produces only
    when ``cta_tile_m == 64 and use_2cta_instrs``. Our ``default_config`` pins
    the dgated/gated tile to ``tile_m=128`` (route-level padding is only
    guaranteed 128-aligned), so ``use_2cta_instrs`` is never set (it requires
    ``mma_tiler_m == 256``) and ``cta_tile_m`` stays 128 with the (4,1) warp
    shape. The (2,2) path is therefore unreachable by construction and no
    epilogue-copy override is needed. Kept as a single seam so the PR#133 TV
    copy can be slotted back in if a future config ever enables 256-tile 2-CTA.
    """
    return cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)


# ---------------------------------------------------------------------------
# GemmGatedMixin (from gemm_gated.py)
# ---------------------------------------------------------------------------


@dsl_user_op
@cute.jit
def _swiglu_clamp_pair(gate, up, clamp_value: cutlass.Constexpr[float], *, loc=None, ip=None):
    cv = Float32(clamp_value)
    neg_cv = Float32(0.0) - cv
    if const_expr(not isinstance(gate, tuple)):
        gate_c = utils.fmin(gate, cv)
        up_c = cute.arch.fmax(utils.fmin(up, cv), neg_cv)
        return gate_c, up_c
    else:
        gate_c = (utils.fmin(gate[0], cv), utils.fmin(gate[1], cv))
        up_c = (
            cute.arch.fmax(utils.fmin(up[0], cv), neg_cv),
            cute.arch.fmax(utils.fmin(up[1], cv), neg_cv),
        )
        return gate_c, up_c


@dsl_user_op
@cute.jit
def _swiglu_clamp_bwd_grads(dx, dy, gate, up, clamp_value: cutlass.Constexpr[float], *, loc=None, ip=None):
    cv = Float32(clamp_value)
    zero = Float32(0.0)
    if const_expr(not isinstance(gate, tuple)):
        gate_ok = cutlass.Boolean(gate <= cv)
        up_abs = cute.arch.fmax(up, zero - up)
        up_ok = cutlass.Boolean(up_abs <= cv)
        return (dx if gate_ok else zero), (dy if up_ok else zero)
    else:
        gate_ok0 = cutlass.Boolean(gate[0] <= cv)
        gate_ok1 = cutlass.Boolean(gate[1] <= cv)
        up_abs0 = cute.arch.fmax(up[0], zero - up[0])
        up_abs1 = cute.arch.fmax(up[1], zero - up[1])
        up_ok0 = cutlass.Boolean(up_abs0 <= cv)
        up_ok1 = cutlass.Boolean(up_abs1 <= cv)
        dx_c = (dx[0] if gate_ok0 else zero, dx[1] if gate_ok1 else zero)
        dy_c = (dy[0] if up_ok0 else zero, dy[1] if up_ok1 else zero)
        return dx_c, dy_c


class GemmGatedMixin(GemmActMixin):
    _epi_ops = (*GemmActMixin._epi_ops[:-1], TileStore("mPostAct", epi_tile_fn=_halve_epi_tile))
    _extra_param_fields = (
        ("act_fn", cutlass.Constexpr, None),
        ("swiglu_clamp_value", cutlass.Constexpr, 0.0),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None

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
            tiled_copy_postact_r2s = _make_postact_r2s_tiled_copy(
                self, copy_atom_postact_r2s, tiled_copy_r2s, params
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
        self.swiglu_clamp_value = args.swiglu_clamp_value
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        d["swiglu_clamp_value"] = args.swiglu_clamp_value
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                gate = tRS_rD[2 * i]
                up = tRS_rD[2 * i + 1]
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate, up = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                tRS_rPostAct[i] = params.act_fn(gate, up)
        else:
            for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                gate = (tRS_rD[4 * i], tRS_rD[4 * i + 2])
                up = (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3])
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate, up = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(gate, up)
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
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
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
# GemmDGatedMixin (from gemm_dgated.py)
# ---------------------------------------------------------------------------

class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileStore("mPostAct"), ColVecReduce("mColVecReduce"))
    _extra_param_fields = (
        ("act_bwd_fn", cutlass.Constexpr, None),
        ("implicit_dtype", cutlass.Constexpr, None),
        ("swiglu_clamp_value", cutlass.Constexpr, 0.0),
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
            tiled_copy_postact_r2s = _make_postact_r2s_tiled_copy(
                self, copy_atom_postact_r2s, tiled_copy_r2s, params
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
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
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
        self.swiglu_clamp_value = args.swiglu_clamp_value
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["implicit_dtype"] = args.implicit_dtype
        d["swiglu_clamp_value"] = args.swiglu_clamp_value
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
                gate = tRS_rXY_f32x2[2 * i]
                up = tRS_rXY_f32x2[2 * i + 1]
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate_c, up_c = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                    dx, dy, out = params.act_bwd_fn(gate_c, up_c, tRS_rD_scaled[i])
                    dx, dy = _swiglu_clamp_bwd_grads(dx, dy, gate, up, self.swiglu_clamp_value)
                    tRS_rdXY_f32x2[2 * i], tRS_rdXY_f32x2[2 * i + 1], tRS_rOut[i] = dx, dy, out
                else:
                    tRS_rdXY_f32x2[2 * i], tRS_rdXY_f32x2[2 * i + 1], tRS_rOut[i] = params.act_bwd_fn(
                        gate, up, tRS_rD_scaled[i]
                    )
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                gate = (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2])
                up = (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3])
                dout = (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1])
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate_c, up_c = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                    dx, dy, out = params.act_bwd_fn(gate_c, up_c, dout)
                    dx, dy = _swiglu_clamp_bwd_grads(dx, dy, gate, up, self.swiglu_clamp_value)
                else:
                    dx, dy, out = params.act_bwd_fn(gate, up, dout)
                tRS_rdXY_f32x2[4 * i] = dx[0]
                tRS_rdXY_f32x2[4 * i + 2] = dx[1]
                tRS_rdXY_f32x2[4 * i + 1] = dy[0]
                tRS_rdXY_f32x2[4 * i + 3] = dy[1]
                tRS_rOut[2 * i] = out[0]
                tRS_rOut[2 * i + 1] = out[1]
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
        d["swiglu_clamp_value"] = args.swiglu_clamp_value
        self.swiglu_clamp_value = args.swiglu_clamp_value
        return self.EpilogueParams(**d)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
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
                gate = tRS_rXY_f32x2[2 * i]
                up = tRS_rXY_f32x2[2 * i + 1]
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate_c, up_c = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                    dx, dy, out = params.act_bwd_fn(gate_c, up_c, tRS_rD_scaled[i])
                    dx, dy = _swiglu_clamp_bwd_grads(dx, dy, gate, up, self.swiglu_clamp_value)
                    tRS_rdXY_f32x2[2 * i], tRS_rdXY_f32x2[2 * i + 1], tRS_rOut[i] = dx, dy, out
                else:
                    tRS_rdXY_f32x2[2 * i], tRS_rdXY_f32x2[2 * i + 1], tRS_rOut[i] = params.act_bwd_fn(
                        gate, up, tRS_rD_scaled[i]
                    )
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                gate = (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2])
                up = (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3])
                dout = (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1])
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate_c, up_c = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                    dx, dy, out = params.act_bwd_fn(gate_c, up_c, dout)
                    dx, dy = _swiglu_clamp_bwd_grads(dx, dy, gate, up, self.swiglu_clamp_value)
                else:
                    dx, dy, out = params.act_bwd_fn(gate, up, dout)
                tRS_rdXY_f32x2[4 * i] = dx[0]
                tRS_rdXY_f32x2[4 * i + 2] = dx[1]
                tRS_rdXY_f32x2[4 * i + 1] = dy[0]
                tRS_rdXY_f32x2[4 * i + 3] = dy[1]
                tRS_rOut[2 * i] = out[0]
                tRS_rOut[2 * i + 1] = out[1]
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
        d["swiglu_clamp_value"] = args.swiglu_clamp_value
        self.swiglu_clamp_value = args.swiglu_clamp_value
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
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
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
                gate = tRS_rXY_f32x2[2*i]
                up = tRS_rXY_f32x2[2*i+1]
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate_c, up_c = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                    dx, dy, out = params.act_bwd_fn(gate_c, up_c, tRS_rD_scaled[i])
                    dx, dy = _swiglu_clamp_bwd_grads(dx, dy, gate, up, self.swiglu_clamp_value)
                    tRS_rdXY_f32x2[2*i], tRS_rdXY_f32x2[2*i+1], tRS_rOut[i] = dx, dy, out
                else:
                    tRS_rdXY_f32x2[2*i], tRS_rdXY_f32x2[2*i+1], tRS_rOut[i] = params.act_bwd_fn(
                        gate, up, tRS_rD_scaled[i])
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                gate = (tRS_rXY_f32x2[4*i], tRS_rXY_f32x2[4*i+2])
                up = (tRS_rXY_f32x2[4*i+1], tRS_rXY_f32x2[4*i+3])
                dout = (tRS_rD_scaled[2*i], tRS_rD_scaled[2*i+1])
                if const_expr(self.swiglu_clamp_value > 0.0):
                    gate_c, up_c = _swiglu_clamp_pair(gate, up, self.swiglu_clamp_value)
                    dx, dy, out = params.act_bwd_fn(gate_c, up_c, dout)
                    dx, dy = _swiglu_clamp_bwd_grads(dx, dy, gate, up, self.swiglu_clamp_value)
                else:
                    dx, dy, out = params.act_bwd_fn(gate, up, dout)
                tRS_rdXY_f32x2[4*i] = dx[0]
                tRS_rdXY_f32x2[4*i+2] = dx[1]
                tRS_rdXY_f32x2[4*i+1] = dy[0]
                tRS_rdXY_f32x2[4*i+3] = dy[1]
                tRS_rOut[2*i] = out[0]
                tRS_rOut[2*i+1] = out[1]
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
