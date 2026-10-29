# ********************************************************************************
# Leaf module: gated/dgated epilogue Mixins extracted from gemm_gated.py and
# gemm_dgated.py to break circular imports.
#
# IMPORTANT: This module must NOT import from .gemm_gated, .gemm_dgated,
# .gemm_sm100_fp8_zeromat, or .blockscaled_fp8_gemm.
# ********************************************************************************

import os
from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import quack.layout_utils as layout_utils
import quack.utils as utils
from quack.epi_ops import _get_lane_warp_layouts
from quack.sm90_utils import partition_for_epilogue
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


def _gated_epi_tile_n_target() -> int:
    """Target epilogue subtile N for the forward gated up-proj GEMM.

    The cutlass default (compute_epilogue_tile_shape) picks epi_tile_n=32 for the
    bf16-D grouped up-proj.  Raising it to 64 makes the gated postact (half-N)
    subtile a full 32-element 1x32 scale group, enabling fused postact (y1) FP8
    quant in the epilogue.  ncu-validated bit-identical (z/y1 max_abs_diff=0) vs
    n=32, and -5.8% kernel time.  Set to 0 to disable (use cutlass default).

    DEFAULT 0 (cutlass default tiling).  The forced epi_tile_n=64 is only needed
    by the fuse_y1 path, so the fused-postact mixin requests it explicitly (see
    ``GemmGatedPostActIsaQuantMixin``); the plain GemmGatedMixin honours this env
    only when the user opts in.  Keeping the default at 0 makes FUSE_Y1=0 a
    strict no-op for the production gemm_gated path (the committed-baseline
    tiling), so this epilogue refactor cannot perturb non-fuse_y1 training.
    """
    return int(os.getenv("SONIC_MOE_GATED_EPI_TILE_N", "0"))



def _make_forced_epi_tile_fn(orig_fn, target_n: int):
    """Wrap compute_epilogue_tile_shape to raise epi_tile_n to ``target_n``.

    Only raises (never lowers) the N subtile, caps at cta_n, and preserves the
    cutlass warp-layout stride structure so smem/TMA partitioning is unchanged.
    """

    def _forced(cta_tile_shape, use_2cta_instrs, layout_d, elem_ty_d,
                *, layout_c=None, elem_ty_c=None, loc=None, ip=None):
        tile_m_layout, tile_n_layout = orig_fn(
            cta_tile_shape, use_2cta_instrs, layout_d, elem_ty_d,
            layout_c=layout_c, elem_ty_c=elem_ty_c, loc=loc, ip=ip,
        )
        cta_m, cta_n = cta_tile_shape[:2]
        (warp_m, warp_n) = (2, 2) if (cta_m == 64 and use_2cta_instrs) else (4, 1)
        cur_n = cute.size(tile_n_layout)
        new_n = min(int(cta_n), max(int(cur_n), int(target_n)))
        # keep N a multiple of warp_n and even (gated halves the N tile)
        if new_n <= cur_n or (new_n % warp_n) != 0 or (new_n % 2) != 0:
            return (tile_m_layout, tile_n_layout)
        tile_n_layout_new = cute.make_layout(
            (new_n // warp_n, warp_n), stride=(1, cta_n // warp_n), loc=loc, ip=ip
        )
        return (tile_m_layout, cute.coalesce(tile_n_layout_new, loc=loc, ip=ip))

    return _forced


def _make_postact_r2s_tiled_copy(gemm, copy_atom_postact_r2s, tiled_copy_r2s, params):
    """Build the register-to-smem tiled copy for gated postact output."""
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

    def _setup_attributes(self, epilogue_args, varlen_args):
        """Raise the epilogue subtile N for the (compute-bound) gated up-proj.

        ncu-validated: epi_tile_n=64 is the optimum for the bf16-D grouped up-proj
        (vs cutlass-default 32), -5.8% kernel time, bit-identical output.  We
        temporarily wrap compute_epilogue_tile_shape during the base setup (epi_tile
        is consumed by _compute_stages inside it, so post-fixing is too late).
        Forward gated only — dgated uses a different mixin and is unaffected.

        Default behaviour (env unset -> target_n=0) is a strict no-op: the plain
        GemmGatedMixin keeps the cutlass-default tiling, so non-fuse_y1 fp8
        training is byte-identical to the committed baseline.  The fuse_y1 mixin
        overrides ``_forced_epi_tile_n`` to 64 because its postact fragment must
        be exactly one 1x32 group.
        """
        target_n = self._forced_epi_tile_n()
        if target_n <= 0:
            super()._setup_attributes(epilogue_args, varlen_args)
            return
        orig_fn = sm100_utils.compute_epilogue_tile_shape
        sm100_utils.compute_epilogue_tile_shape = _make_forced_epi_tile_fn(orig_fn, target_n)
        try:
            super()._setup_attributes(epilogue_args, varlen_args)
        finally:
            sm100_utils.compute_epilogue_tile_shape = orig_fn

    def _forced_epi_tile_n(self) -> int:
        """Epilogue subtile-N override for this mixin (0 = cutlass default).

        Base gated path: env-gated, default OFF (committed-baseline tiling).
        Overridden to a hard 64 by the fuse_y1 postact-quant mixin, which
        structurally requires it.
        """
        return _gated_epi_tile_n_target()


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
        if hasattr(args, "postact_bf16_trunc"):
            d["postact_bf16_trunc"] = args.postact_bf16_trunc
        if hasattr(args, "mPostActOutput"):
            d["mPostActOutput"] = args.mPostActOutput
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rPostAct_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rPostAct = cute.make_rmem_tensor(tRS_rPostAct_layout.shape, self.acc_dtype)
        if const_expr(
            self.arch < 100 and not self.swiglu_clamp_value > 0.0
            and _is_quack_act(params.act_fn, "swiglu_precise")
        ):
            _swiglu_precise_frag(tRS_rD, tRS_rPostAct)
        elif const_expr(self.arch < 100):
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


@dsl_user_op
def _bf16_trunc_rne(x: Float32, *, loc=None, ip=None) -> Float32:
    """Round-to-nearest-even fp32 -> bf16 -> fp32, matching an HBM bf16 store.

    Reproduces the standalone (carbide-kernel) precision: the standalone path
    writes y1 as bf16 to HBM (RNE truncation of the 16 low mantissa bits) before
    the triton quantize reads it back.  This helper performs that exact RNE
    truncation in-register so the FUSED epilogue can be made byte-identical to
    the standalone path when requested by Python FP8 config.

    NOT stochastic — deterministic RNE, identical to torch ``.to(bfloat16)`` and
    the numpy ``bf16_trunc`` model verified bit-exact (0 mismatch / 100k draws).
    bits' = ((bits>>16) + roundup) << 16,
      roundup = (low>0x8000) | (low==0x8000 & lsb_of_kept==1).
    """
    bits = _f32_as_i32(x)
    lower = bits & Int32(0xFFFF)
    kept_lsb = (bits >> Int32(16)) & Int32(1)
    # RNE roundup as PURE integer arithmetic — no dynamic Boolean (the DSL cannot
    # fold a `cutlass.Boolean` into a Python `if` inside a @dsl_user_op).
    #   roundup = (lower + kept_lsb + 0x7FFF) >> 16
    # Verified exact round-to-nearest-even over all 2^17 (lower,kept) patterns:
    #   lower>0x8000 -> always 1; lower==0x8000 -> 1 iff kept_lsb==1 (tie-to-even);
    #   lower<0x8000 -> 0.  Matches torch .to(bfloat16) bit-for-bit.
    roundup = (lower + kept_lsb + Int32(0x7FFF)) >> Int32(16)
    out_bits = ((bits >> Int32(16)) + roundup) << Int32(16)
    return _i32_as_f32(out_bits)


@dsl_user_op
def _bf16_round_odd(x: Float32, *, loc=None, ip=None) -> Float32:
    """Round-to-odd fp32 -> bf16 (returned as an exactly bf16-representable fp32).

    Truncate the 16 low mantissa bits and set the kept LSB if any of them was
    non-zero.  A bf16 intermediate rounded this way makes a later RN rounding to
    a narrower format (e4m3) identical to rounding the original fp32 directly,
    removing the biased fp32->bf16->fp8 double rounding.  Inf/NaN are preserved.
    """
    bits = _f32_as_i32(x)
    sticky = ((bits & Int32(0xFFFF)) + Int32(0xFFFF)) >> Int32(16)
    return _i32_as_f32(((bits >> Int32(16)) | sticky) << Int32(16))


# ---------------------------------------------------------------------------
# Exact sigmoid with the IEEE-division guard hoisted out of the element loop
# ---------------------------------------------------------------------------
# quack's *_precise activations compute sigmoid as ``1.0f / (1.0f + expf(-x))``.
# The correctly rounded division compiles to MUFU.RCP + one Newton step behind a
# per-element exponent check, BSSY/BRA/BSYNC and a slow-path CALL that is only
# needed when the denominator's biased exponent is outside [1, 252]. Since
# ``d = 1 + expf(-x) >= 1``, that only happens for d >= 2**126 (x < -87.3), inf
# or NaN. The helpers below evaluate the fast path for a whole fragment, track
# max(bits(d)) and redo the exact division only when some d leaves the range,
# giving bit-identical results with ~6 fewer instructions per element.
_RCP_FAST_MAX_BITS = 0x7E800000  # bits of 2**126; d >= 1 (or canonical NaN) so the sign bit is 0


@dsl_user_op
def _rcp_rn_fast(d: Float32, *, loc=None, ip=None) -> Float32:
    """Fast path of the compiler's ``1.0f / d`` (bitwise equal for 1 <= d < 2**126)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(d).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 r, e; rcp.approx.ftz.f32 r, $1; neg.f32 e, $1; "
            "fma.rn.f32 e, e, r, 0f3F800000; fma.rn.f32 $0, r, e, r; }",
            "=f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@cute.jit
def _sigmoid_precise_frag(tX, tSig, stride: cutlass.Constexpr[int]):
    """tSig[i] = sigmoid_precise(tX[stride * i]) for the whole fragment, bitwise."""
    n = cute.size(tSig)
    big = Int32(0)
    for i in cutlass.range(n, unroll_full=True):
        den = Float32(1.0) + cute.math.exp(-tX[stride * i], fastmath=False)
        tSig[i] = den
        big = _b32_op("max.s32", big, _f32_as_i32(den))
    if big >= Int32(_RCP_FAST_MAX_BITS):
        for i in cutlass.range(n, unroll_full=True):
            tSig[i] = Float32(1.0) / tSig[i]
    else:
        for i in cutlass.range(n, unroll_full=True):
            tSig[i] = _rcp_rn_fast(tSig[i])


def _is_quack_act(fn, name: str) -> bool:
    import quack.activation as _qa
    return fn is getattr(_qa, name, None)


@cute.jit
def _swiglu_precise_frag(tXY, tOut):
    """tOut[i] = swiglu_precise(tXY[2i], tXY[2i+1]), bitwise."""
    tSig = cute.make_rmem_tensor_like(tOut, Float32)
    _sigmoid_precise_frag(tXY, tSig, 2)
    for i in cutlass.range(cute.size(tOut), unroll_full=True):
        tOut[i] = (tXY[2 * i] * tSig[i]) * tXY[2 * i + 1]


@cute.jit
def _dswiglu_precise_frag(tXY, tDout, tdXY, tOut):
    """Fragment version of quack's dswiglu_precise (same expression order), bitwise."""
    tSig = cute.make_rmem_tensor_like(tDout, Float32)
    _sigmoid_precise_frag(tXY, tSig, 2)
    for i in cutlass.range(cute.size(tDout), unroll_full=True):
        x = tXY[2 * i]
        y = tXY[2 * i + 1]
        dout = tDout[i]
        sig = tSig[i]
        silu_x = x * sig
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sig - silu_x * sig) * dout + silu_x_dout
        tdXY[2 * i] = d_silu_x_dout * y
        tdXY[2 * i + 1] = silu_x_dout
        tOut[i] = silu_x * y


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


class BlockscaledScaleStoreDual(BlockscaledScaleStore):
    """z-scale store for the COMBINED z+y1 path (epi_tile_n forced to 64).

    The plain BlockscaledScaleStore assumes 1 scale group (32 cols) per epi
    subtile and collapses ``n_group_abs = n_base + n_sub``.  At epi_tile_n=64
    each subtile spans TWO 1x32 groups, so this variant returns the per-CTA-tile
    group BASE for the FIRST of the subtile's two groups, ``n_base + n_sub*2``;
    the combined mixin writes both groups at ``+0`` and ``+1``.  Identical
    bounds (m_limit, n_limit = N//32).
    """

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            param, m_abs, n_base, m_limit, n_limit = state
            if const_expr(isinstance(epi_coord, tuple)):
                n_sub = epi_coord[1] if len(epi_coord) > 1 else epi_coord[0]
            else:
                n_sub = epi_coord
            return (param, m_abs, n_base + n_sub * Int32(2), m_limit, n_limit)
        return None


class Sm90RawScaleStore(EpiOp):
    """SM90 raw 1x32 UE8M0 store using the actual R2S source coordinates."""

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        state = None
        if const_expr(param is not None):
            coord = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            state = (
                param,
                coord,
                ctx.tile_coord_mnkl,
                ctx.varlen_manager,
                ctx.tidx,
                ctx.tiled_copy_r2s,
                ctx.epi_tile,
                ctx.tile_M,
                ctx.tile_N,
            )
        return state

    def begin_loop(self, gemm, state, epi_coord):
        result = None
        if const_expr(state is not None):
            param, coord, tile_coord, varlen, tidx, tiled_copy, epi_tile, tile_m, tile_n = state
            coord_cur = cute.group_modes(coord, 3, cute.rank(coord))[
                None, None, None, epi_coord
            ]
            result = (
                param, coord_cur, tile_coord, varlen, tidx, tiled_copy,
                epi_tile, tile_m, tile_n,
            )
        return result


class Sm90RawGatedQuantMixin(GemmGatedMixin):
    """Hopper-only raw 1x32 z/y1 quantization in the GEMM epilogue."""

    _epi_ops = (
        *GemmGatedMixin._epi_ops,
        Sm90RawScaleStore("mZScale"),
        Sm90RawScaleStore("mPostActScaleRaw"),
    )
    _extra_param_fields = (
        *GemmGatedMixin._extra_param_fields,
        ("postact_bf16_trunc", cutlass.Constexpr, False),
        ("mPostActOutput", object, None),
    )

    def _forced_epi_tile_n(self):
        return 64

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
        postact_bf16_trunc: cutlass.Constexpr[bool] = False
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScale: Optional[cute.Tensor] = None
        mPostActScaleRaw: Optional[cute.Tensor] = None
        mPostActOutput: Optional[cute.Tensor] = None

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
        if const_expr(self.postact_dtype.width == 8):
            return None
        return GemmActMixin.epi_setup_postact(
            self,
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

    @staticmethod
    @cute.jit
    def _quantize_fragment(values, info, half_n: cutlass.Constexpr[bool]):
        if const_expr(info is not None):
            param, coords, tile_coord, varlen, tidx, tiled_copy, epi_tile, tile_m, tile_n = info
            lane_layout, warp_layout = _get_lane_warp_layouts(tiled_copy, True)
            lanes_n = cute.size(lane_layout, mode=[1])
            assert cute.size(warp_layout, mode=[1]) == 1
            batch = tile_coord[3]
            batch_start = varlen.params.cu_seqlens_m[batch] if varlen.varlen_m else Int32(0)
            m_limit = (
                varlen.params.cu_seqlens_m[batch + Int32(1)]
                if varlen.varlen_m else param.shape[0]
            )
            values_for_quant = cute.make_rmem_tensor(values.shape, Float32)
            values_for_quant.store(values.load())
            if const_expr(half_n and cute.size(values) == 16 and lanes_n == 4):
                # PostAct of a 64-column z subtile: value i is z element 2*i, so
                # its row half is i % 2 and its PostAct column is
                # lane_n + 4 * ((i // 2) % 2) + 8 * (i // 4).  Each parity class is
                # therefore 8 values of one row inside one 1x32 group, and the 4
                # N lanes complete it.  Reduce the two classes directly instead of
                # the generic all-pairs scan (identical amax: fmax is exact and
                # the generic scan also folds in 0.0 for out-of-group values).
                group_amax = [Float32(0.0), Float32(0.0)]
                for i in cutlass.range_constexpr(16):
                    group_amax[i % 2] = cute.arch.fmax(
                        group_amax[i % 2],
                        cute.arch.fmax(values_for_quant[i], Float32(0.0) - values_for_quant[i]),
                    )
                group_qscale = []
                group_e8m0 = []
                for g in cutlass.range_constexpr(2):
                    h_amax = cute.arch.warp_reduction(
                        group_amax[g], cute.arch.fmax, threads_in_group=lanes_n
                    )
                    h_amax_bits = _f32_as_i32(h_amax)
                    h_biased_exp = (h_amax_bits >> Int32(23)) & Int32(0xFF)
                    h_mantissa = h_amax_bits & Int32(0x7FFFFF)
                    h_carry = Int32(1) if cutlass.Boolean(h_mantissa > Int32(0x600000)) else Int32(0)
                    h_e8m0 = h_biased_exp - Int32(8) + h_carry
                    h_e8m0 = h_e8m0 if cutlass.Boolean(h_biased_exp > Int32(0)) else Int32(0)
                    h_e8m0 = h_e8m0 if cutlass.Boolean(h_e8m0 > Int32(0)) else Int32(0)
                    h_qexp = Int32(254) - h_e8m0
                    h_qexp = Int32(254) if cutlass.Boolean(h_qexp > Int32(254)) else h_qexp
                    h_qexp = Int32(1) if cutlass.Boolean(h_qexp < Int32(1)) else h_qexp
                    group_e8m0.append(h_e8m0)
                    group_qscale.append(_i32_as_f32(h_qexp << Int32(23)))
                h_output_tile_n = tile_n // Int32(2)
                for i in cutlass.range_constexpr(16):
                    values[i] = values_for_quant[i] * group_qscale[i % 2]
                    h_coord = coords[2 * i]
                    h_m_abs = batch_start + tile_coord[0] * tile_m + h_coord[0]
                    h_n_abs = tile_coord[1] * h_output_tile_n + h_coord[1] // Int32(2)
                    if (
                        cutlass.Boolean(h_m_abs < m_limit)
                        & cutlass.Boolean(h_n_abs < param.shape[1] * Int32(32))
                        & cutlass.Boolean((h_n_abs % Int32(32)) == Int32(0))
                    ):
                        param[h_m_abs, h_n_abs // Int32(32)] = cutlass.Int8(group_e8m0[i % 2])
            else:
                for i in cutlass.range(cute.size(values), unroll_full=True):
                    coord = coords[2 * i] if half_n else coords[i]
                    m_local = coord[0]
                    n_local = coord[1]
                    n_out = n_local // Int32(2) if half_n else n_local
                    # R2S coordinates show four adjacent N lanes per row; each lane
                    # owns eight register values, covering one complete 1x32 group.
                    amax = cute.arch.fmax(
                        values_for_quant[i], Float32(0.0) - values_for_quant[i]
                    )
                    for j in cutlass.range(cute.size(values_for_quant), unroll_full=True):
                        other = coords[2 * j] if half_n else coords[j]
                        other_n = other[1] // Int32(2) if half_n else other[1]
                        same_group = (
                            cutlass.Boolean(other[0] == m_local)
                            & cutlass.Boolean((other_n // Int32(32)) == (n_out // Int32(32)))
                        )
                        other_abs = cute.arch.fmax(
                            values_for_quant[j], Float32(0.0) - values_for_quant[j]
                        )
                        amax = cute.arch.fmax(amax, other_abs if same_group else Float32(0.0))
                    amax = cute.arch.warp_reduction(
                        amax, cute.arch.fmax, threads_in_group=lanes_n
                    )
                    amax_bits = _f32_as_i32(amax)
                    biased_exp = (amax_bits >> Int32(23)) & Int32(0xFF)
                    mantissa = amax_bits & Int32(0x7FFFFF)
                    carry = Int32(1) if cutlass.Boolean(mantissa > Int32(0x600000)) else Int32(0)
                    e8m0 = biased_exp - Int32(8) + carry
                    e8m0 = e8m0 if cutlass.Boolean(biased_exp > Int32(0)) else Int32(0)
                    e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
                    qexp = Int32(254) - e8m0
                    qexp = Int32(254) if cutlass.Boolean(qexp > Int32(254)) else qexp
                    qexp = Int32(1) if cutlass.Boolean(qexp < Int32(1)) else qexp
                    values[i] = values_for_quant[i] * _i32_as_f32(qexp << Int32(23))
                    m_abs = batch_start + tile_coord[0] * tile_m + m_local
                    output_tile_n = tile_n // Int32(2) if half_n else tile_n
                    n_abs = tile_coord[1] * output_tile_n + n_out
                    if (
                        cutlass.Boolean(m_abs < m_limit)
                        & cutlass.Boolean(n_abs < param.shape[1] * Int32(32))
                        & cutlass.Boolean((n_abs % Int32(32)) == Int32(0))
                    ):
                        param[m_abs, n_abs // Int32(32)] = cutlass.Int8(e8m0)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        postact = GemmGatedMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )
        if const_expr(epi_loop_tensors["mPostActScaleRaw"] is not None):
            if const_expr(params.postact_bf16_trunc):
                for i in cutlass.range(cute.size(postact), unroll_full=True):
                    postact[i] = _bf16_trunc_rne(postact[i])
            self._quantize_fragment(postact, epi_loop_tensors["mPostActScaleRaw"], True)
            info = epi_loop_tensors["mPostActScaleRaw"]
            param, coords, tile_coord, varlen, _, _, _, tile_m, tile_n = info
            batch = tile_coord[3]
            batch_start = varlen.params.cu_seqlens_m[batch] if varlen.varlen_m else Int32(0)
            m_limit = (
                varlen.params.cu_seqlens_m[batch + Int32(1)]
                if varlen.varlen_m else param.shape[0]
            )
            output = params.mPostActOutput
            for i in cutlass.range(cute.size(postact), unroll_full=True):
                coord = coords[2 * i]
                m_abs = batch_start + tile_coord[0] * tile_m + coord[0]
                n_abs = tile_coord[1] * (tile_n // Int32(2)) + coord[1] // Int32(2)
                if cutlass.Boolean(m_abs < m_limit) & cutlass.Boolean(n_abs < output.shape[1]):
                    output[m_abs, n_abs] = postact[i].to(self.postact_dtype)
        self._quantize_fragment(tRS_rD, epi_loop_tensors["mZScale"], False)
        return postact

    @cute.jit
    def epi_convert_postact(self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx):
        # Only reached for a 16-bit PostAct (stored through STSM), which needs
        # the gated register permute like the plain gated epilogue.
        return GemmGatedMixin.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )


class Sm90ClusterAmaxXchg(EpiOp):
    """Smem + register state of the fused 1x128 y1 quant (1x2 cluster).

    One Int64 smem range: words 0-1 are two mbarriers (double buffered by tile
    parity), the rest an f32 ``[2, tile_M]`` buffer the peer CTA fills with its
    per-row amax through ``st.async``.  ``begin`` also allocates the register
    buffer carrying subtile 0's y1 to subtile 1, where the group is quantized.
    """

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        # Fixed per CTA, but reported per epilogue stage (see ``_compute_stages``).
        epi_stage = 4 if epi_tile[1] <= 16 else 2
        return -(-(2 + cta_tile_shape_mnk[0]) * 8 // epi_stage)

    def smem_struct_field(self, gemm, params):
        n_words = 2 + gemm.cta_tile_shape_mnk[0]
        return (f"s_{self.name}",
                cute.struct.Align[cute.struct.MemRange[cutlass.Int64, n_words], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        return getattr(storage_epi, f"s_{self.name}").data_ptr()

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        coord = ctx.partition_for_epilogue_fn(
            cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
        )
        coord = cute.group_modes(coord, 3, cute.rank(coord))
        n_vals = cute.size(coord[None, None, None, 0]) // 2  # PostAct values per subtile
        buf = cute.make_rmem_tensor(cute.make_layout(n_vals), Float32)
        return (smem_tensor, coord, buf, ctx.tile_coord_mnkl, ctx.varlen_manager, ctx.tidx)

    def begin_loop(self, gemm, state, epi_coord):
        smem, coord, buf, tile_coord, varlen, tidx = state
        n_sub = epi_coord[1] if isinstance(epi_coord, tuple) else epi_coord
        assert isinstance(n_sub, int), "epilogue subtile loop must be unrolled"
        return (smem, coord[None, None, None, epi_coord], buf, tile_coord, varlen, tidx, n_sub)


@dsl_user_op
def _st_global_v2(ptr, a: Int32, b: Int32, *, loc=None, ip=None) -> None:
    """8B ``st.global.v2.b32`` at an 8B-aligned gmem pointer."""
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


class Sm90RawGatedQuant128Mixin(Sm90RawGatedQuantMixin):
    """Fused 1x128 y1 quant straight from the fp32 PostAct (single rounding).

    Needs a 192x128 tile in a 1x2 cluster: a CTA covers 64 y1 columns in two
    32-column subtiles, so a 1x128 group spans the two N-adjacent CTAs of the
    cluster.  At subtile 1 each quad reduces its row amax over 64 columns,
    sends it to the peer CTA's smem (``st.async`` + mbarrier complete_tx),
    waits for the peer's, and quantizes both subtiles with the shared scale.
    Byte-identical to RTO-bf16 y1 + ``quantize_activation_blockscaled_fast(., 128)``.
    """

    _epi_ops = (*Sm90RawGatedQuantMixin._epi_ops, Sm90ClusterAmaxXchg("y1_xchg"))

    def epilogue(self, *args, **kwargs):
        # The exchange buffer / mbarrier phase follow the per-CTA tile count,
        # which the two CTAs of the cluster advance in lockstep.
        sched = kwargs["tile_scheduler"] if "tile_scheduler" in kwargs else args[21]
        self._y1x_tile_idx = sched.num_tiles_executed
        return super().epilogue(*args, **kwargs)

    @cute.jit
    def epi_cluster_init(self, params, storage):
        mbar = getattr(storage.epi, "s_y1_xchg").data_ptr()
        if cute.arch.thread_idx()[0] == Int32(0):
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        postact = GemmGatedMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )
        assert self.cta_tile_shape_mnk[1] == 128 and self.epi_tile[1] == 64
        assert self.epi_tile[0] == self.cta_tile_shape_mnk[0]
        assert self.cluster_shape_mnk[0] == 1 and self.cluster_shape_mnk[1] == 2
        smem, coords, buf, tile_coord, varlen, tidx, n_sub = epi_loop_tensors["y1_xchg"]
        assert cute.size(postact) == 16
        if const_expr(params.postact_bf16_trunc):
            for i in cutlass.range_constexpr(16):
                postact[i] = _bf16_trunc_rne(postact[i])
        if const_expr(n_sub == 0):
            for i in cutlass.range_constexpr(16):
                buf[i] = postact[i]
        else:
            # Value i: row half r = i % 2, y1 column ln + 4 * (i // 2) of the subtile.
            ln = cute.arch.lane_idx() & Int32(3)
            amax = [Float32(0.0), Float32(0.0)]
            for i in cutlass.range_constexpr(16):
                a = cute.arch.fmax(
                    cute.arch.fmax(buf[i], Float32(0.0) - buf[i]),
                    cute.arch.fmax(postact[i], Float32(0.0) - postact[i]),
                )
                amax[i % 2] = cute.arch.fmax(amax[i % 2], a)
            for r in cutlass.range_constexpr(2):
                amax[r] = cute.arch.warp_reduction(amax[r], cute.arch.fmax, threads_in_group=4)
            tile_m = self.cta_tile_shape_mnk[0]
            t = self._y1x_tile_idx
            stage = t % Int32(2)
            phase = (t // Int32(2)) % Int32(2)
            mbar = smem + stage
            slot = cute.recast_ptr(smem + 2, dtype=Float32) + stage * Int32(tile_m)
            rows = [coords[0][0], coords[2][0]]
            if tidx == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, tile_m * 4)
            if ln == Int32(0):
                peer = cute.arch.block_idx_in_cluster() ^ Int32(1)
                for r in cutlass.range_constexpr(2):
                    utils.store_shared_remote(
                        amax[r], slot + rows[r], mbar, peer_cta_rank_in_cluster=peer
                    )
            cute.arch.mbarrier_wait(mbar, phase)
            for r in cutlass.range_constexpr(2):
                peer_amax = cute.make_tensor(slot + rows[r], cute.make_layout(1))[0]
                amax[r] = cute.arch.fmax(amax[r], peer_amax)
            scale_param = epi_loop_tensors["mPostActScaleRaw"][0]
            output = params.mPostActOutput
            batch = tile_coord[3]
            batch_start = varlen.params.cu_seqlens_m[batch] if varlen.varlen_m else Int32(0)
            m_limit = (
                varlen.params.cu_seqlens_m[batch + Int32(1)]
                if varlen.varlen_m else output.shape[0]
            )
            n_abs = tile_coord[1] * Int32(64) + Int32(16) * ln
            n_ok = cutlass.Boolean(n_abs < output.shape[1])
            write_scale = cutlass.Boolean(ln == Int32(0)) & cutlass.Boolean(
                (tile_coord[1] % Int32(2)) == Int32(0)
            )
            for r in cutlass.range_constexpr(2):
                e8m0, qs = _ue8m0_from_amax(amax[r])
                # Word q packs y1 columns ln + 16q + 4b (b = byte) of the CTA's 64.
                words = []
                for q in cutlass.range_constexpr(4):
                    v = []
                    for b in cutlass.range_constexpr(4):
                        k = 4 * q + b
                        src = buf[2 * k + r] if const_expr(k < 8) else postact[2 * (k - 8) + r]
                        v.append(src * qs)
                    words.append(_cvt_e4m3x4(*v))
                # After the lane transpose, word c holds columns c + 16ln + 4b;
                # a 4x4 byte transpose gives words of columns 16ln + 4j + (0..3).
                w = _xpose4_lanes(words, ln)
                lo01 = _prmt2(w[0], w[1], Int32(0x5140))
                lo23 = _prmt2(w[2], w[3], Int32(0x5140))
                hi01 = _prmt2(w[0], w[1], Int32(0x7362))
                hi23 = _prmt2(w[2], w[3], Int32(0x7362))
                out = [
                    _prmt2(lo01, lo23, Int32(0x5410)), _prmt2(lo01, lo23, Int32(0x7632)),
                    _prmt2(hi01, hi23, Int32(0x5410)), _prmt2(hi01, hi23, Int32(0x7632)),
                ]
                m_abs = batch_start + tile_coord[0] * Int32(tile_m) + rows[r]
                if cutlass.Boolean(m_abs < m_limit) & n_ok:
                    _st_global_v4(cute.domain_offset((m_abs, n_abs), output).iterator, out)
                    if write_scale:
                        scale_param[m_abs, tile_coord[1] // Int32(2)] = cutlass.Int8(e8m0)
        self._quantize_fragment(tRS_rD, epi_loop_tensors["mZScale"], False)
        return postact


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
# BlockscaledIsaRowScaleStore / BlockscaledIsaPostActScaleStore EpiOps
# ---------------------------------------------------------------------------
#
# Writes UE8M0 scale bytes directly into the ISA-pack layout used downstream
# by blockscaled FP8 GEMMs (produced standalone today by
# `_quantize_and_pack_kernel`).
#
# Layout (3D uint8 buffer `(num_m_tiles, k_tiles, 512)`):
#   SF_TILE_M = 128, SF_TILE_K = 128, SF_TILE_STORAGE = 512, SF_VEC_SIZE = 32
# Offset of one E8M0 byte for absolute (m_abs, n_group_abs):
#   m_tile = m_abs // 128 ; row_in_tile = m_abs % 128
#   k_tile_idx = n_group_abs // 4 ; k_in_tile = n_group_abs % 4
#   row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
#   byte_at = scale[m_tile, k_tile_idx, row_base + k_in_tile]
# Single-byte bounds-checked stores (mirrors _quantize_and_pack_kernel exactly).
# ---------------------------------------------------------------------------

class BlockscaledIsaRowScaleStore(EpiOp):
    """EpiOp: writes UE8M0 scale bytes into ISA-pack layout.

    Scale buffer shape: `(num_m_tiles, k_tiles, 512)` uint8, byte-equivalent
    to the 1D packed scale buffer produced by `_quantize_and_pack_kernel`.
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


class BlockscaledIsaPostActScaleStore(BlockscaledIsaRowScaleStore):
    """EpiOp: writes UE8M0 postact (y1) scale bytes into ISA-pack layout.

    Identical ISA-pack offset math to `BlockscaledIsaRowScaleStore`, but the
    N-group base is computed over the *postact* feature axis (N//2, the SwiGLU
    output width) rather than the full GEMM-N (z) axis.  With epi_tile_n(z)=64
    each epi subtile produces exactly one 1x32 postact group, so the z-subtile
    index (epi_coord N-sub) maps 1:1 to a postact group.
    """

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(param is not None):
            tile_M = gemm.cta_tile_shape_mnk[0]
            tile_N = gemm.cta_tile_shape_mnk[1]
            postact_tile_N = tile_N // 2
            m_in_tile = ctx.tidx % tile_M
            if const_expr(ctx.varlen_manager.varlen_m):
                batch_start = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3]]
                m_abs = batch_start + ctx.tile_coord_mnkl[0] * tile_M + m_in_tile
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3] + Int32(1)]
            else:
                m_abs = ctx.tile_coord_mnkl[0] * tile_M + m_in_tile
                m_limit = param.shape[0] * Int32(128)
            n_base = ctx.tile_coord_mnkl[1] * (postact_tile_N // 32)
            n_group_limit = param.shape[1] * Int32(4)
            return (param, m_abs, n_base, m_limit, n_group_limit)
        return None


# ---------------------------------------------------------------------------
# GemmGatedPostActIsaQuantMixin (y1 postact-quant fusion)
# ---------------------------------------------------------------------------

class GemmGatedPostActIsaQuantMixin(GemmGatedMixin):
    """GemmGated + epilogue blockscaled FP8 quant of the *postact* (y1=SwiGLU(z)).

    Fuses the standalone `quantize_and_pack_activation(y1)` into the up-proj
    epilogue: after computing tRS_rPostAct (SwiGLU of z), it amax-reduces the
    postact register fragment, derives the integer+carry E8M0 scale (identical
    arithmetic to the proven z-quant path), multiplies the postact registers,
    and writes the UE8M0 byte into the ISA-pack scale buffer.  The fp32->fp8
    saturating cast happens in epi_convert_postact at TMA store time.

    REQUIRES epi_tile_n(z)=64 so that each epi subtile's postact fragment is
    exactly one 1x32 group.  GemmGatedMixin._setup_attributes forces it.
    """
    _epi_ops = (
        *GemmGatedMixin._epi_ops,
        BlockscaledIsaPostActScaleStore("mPostActScaleIsa"),
    )
    _extra_param_fields = (
        *GemmGatedMixin._extra_param_fields,
        ("postact_bf16_trunc", cutlass.Constexpr, False),
    )

    def _forced_epi_tile_n(self) -> int:
        """Fused postact quant requires epi_tile_n=64 (one 1x32 group per half-N
        subtile).  Honour an explicit env override if >0, else force 64 — the
        fusion is structurally incorrect at the cutlass-default 32.
        """
        env_n = _gated_epi_tile_n_target()
        return env_n if env_n > 0 else 64


    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
        postact_bf16_trunc: cutlass.Constexpr[bool] = False
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mPostActScaleIsa: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tRS_rPostAct = GemmGatedMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        _pa_scale_active = epi_loop_tensors["mPostActScaleIsa"]
        if const_expr(_pa_scale_active is not None):
            num_pa = cute.size(tRS_rPostAct)

            # Optional config-controlled emulation of the standalone bf16 HBM
            # round-trip: when enabled, BOTH amax and quantized value derive from
            # the RNE-truncated bf16 postact.  Default OFF keeps fp32->fp8 direct.
            if const_expr(params.postact_bf16_trunc):
                for i in cutlass.range(num_pa, unroll_full=True):
                    tRS_rPostAct[i] = _bf16_trunc_rne(tRS_rPostAct[i])

            # Step 1: amax over the postact register fragment (one 1x32 group).
            # NOTE: no 1e-4 floor here — the standalone y1 quant kernel
            # (_quantize_and_pack_kernel) does not clamp amax, so omitting it
            # keeps the fused output byte-identical to the reference.
            amax = Float32(0.0)
            for i in cutlass.range(num_pa, unroll_full=True):
                val = tRS_rPostAct[i]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax = cute.arch.fmax(amax, abs_val)

            # Step 2: integer+carry E8M0 (matches Triton reference / z-quant).
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

            # Step 4: scale postact registers; fp8 cast at TMA store time.
            for i in cutlass.range(num_pa, unroll_full=True):
                tRS_rPostAct[i] = tRS_rPostAct[i] * quant_scale

            # Step 5: store UE8M0 byte at ISA-pack offset (bounds-checked).
            pa_scale_info = epi_loop_tensors["mPostActScaleIsa"]
            if const_expr(pa_scale_info is not None):
                scale_tensor, m_abs, n_group_abs, m_limit, n_group_limit = pa_scale_info
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
                    scale_tensor[m_tile, k_tile_idx, inner_off] = cutlass.Int8(e8m0)

        return tRS_rPostAct


class GemmGatedBlockscaledQuantPostActMixin(GemmGatedMixin):
    """GemmGated + epilogue blockscaled FP8 quant of BOTH z and the postact (y1).

    Combines the two orthogonal epilogue-quant EpiOps in a single up-proj pass:
      * z-quant  (``BlockscaledScaleStore("mZScale")``) — amax over the GEMM-N
        accumulator fragment ``tRS_rD`` (full z width), scales z to fp8, writes
        a row-major ``(total_M, N//32)`` UE8M0 scale buffer.
      * y1-quant (``BlockscaledIsaPostActScaleStore("mPostActScaleIsa")``) — amax
        over the SwiGLU postact fragment ``tRS_rPostAct`` (half-N), scales the
        postact to fp8, writes a 3D ISA-pack UE8M0 scale buffer.

    The two are provably non-interfering: ``GemmGatedMixin.epi_visit_subtile``
    materialises ``tRS_rPostAct`` as a FRESH rmem tensor (a copy of SwiGLU(z))
    BEFORE any in-place z scaling, so the y1 amax/quant reads un-perturbed values
    and the subsequent ``tRS_rD[i] *= z_quant_scale`` touches a disjoint fragment.
    The two scale buffers are disjoint. Each branch is identical arithmetic to the
    standalone z-quant / y1-quant mixins, so the combined output is byte-for-byte
    equal to running them separately (verified by the byte-exact unit test).

    Letting fuse_y1=1 coexist with save_z_fp8=1 routes the backward through the
    HEALTHY fp8-preact dgated path (``use_fp8_preact = z is None and z_fp8 is not
    None``), avoiding the broken bf16-preact dgated path that the y1-only mutex
    otherwise forced (SAVE_Z=0).

    REQUIRES epi_tile_n(z)=64 (same structural constraint as the y1-only mixin):
    each epi subtile's postact fragment must be exactly one 1x32 group.
    """
    _epi_ops = (
        *GemmGatedMixin._epi_ops,
        BlockscaledScaleStoreDual("mZScale"),
        BlockscaledIsaPostActScaleStore("mPostActScaleIsa"),
    )
    _extra_param_fields = (
        *GemmGatedMixin._extra_param_fields,
        ("postact_bf16_trunc", cutlass.Constexpr, False),
    )

    def _forced_epi_tile_n(self) -> int:
        """Same as the y1-only mixin: postact quant structurally requires 64."""
        env_n = _gated_epi_tile_n_target()
        return env_n if env_n > 0 else 64

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
        postact_bf16_trunc: cutlass.Constexpr[bool] = False
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mZScale: Optional[cute.Tensor] = None
        mPostActScaleIsa: Optional[cute.Tensor] = None

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # 1) SwiGLU into a FRESH tRS_rPostAct (copy of z), BEFORE any z scaling.
        tRS_rPostAct = GemmGatedMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        # 2) y1 (postact) quant — amax/scale read the un-perturbed postact copy.
        _pa_scale_active = epi_loop_tensors["mPostActScaleIsa"]
        if const_expr(_pa_scale_active is not None):
            num_pa = cute.size(tRS_rPostAct)
            # Optional config-controlled bf16 round-trip emulation (default OFF).
            if const_expr(params.postact_bf16_trunc):
                for i in cutlass.range(num_pa, unroll_full=True):
                    tRS_rPostAct[i] = _bf16_trunc_rne(tRS_rPostAct[i])
            # amax over the postact register fragment (one 1x32 group).  No 1e-4
            # floor — matches the standalone y1 quant kernel byte-for-byte.
            amax = Float32(0.0)
            for i in cutlass.range(num_pa, unroll_full=True):
                val = tRS_rPostAct[i]
                neg = Float32(0.0) - val
                abs_val = cute.arch.fmax(val, neg)
                amax = cute.arch.fmax(amax, abs_val)

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

            qexp = Int32(254) - e8m0
            qexp_hi = cutlass.Boolean(qexp > Int32(254))
            qexp = Int32(254) if qexp_hi else qexp
            qexp_lo = cutlass.Boolean(qexp < Int32(1))
            qexp = Int32(1) if qexp_lo else qexp
            quant_scale = _i32_as_f32(qexp << Int32(23))

            for i in cutlass.range(num_pa, unroll_full=True):
                tRS_rPostAct[i] = tRS_rPostAct[i] * quant_scale

            pa_scale_info = epi_loop_tensors["mPostActScaleIsa"]
            if const_expr(pa_scale_info is not None):
                scale_tensor, m_abs, n_group_abs, m_limit, n_group_limit = pa_scale_info
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
                    scale_tensor[m_tile, k_tile_idx, inner_off] = cutlass.Int8(e8m0)

        # 3) z quant — amax/scale over tRS_rD, then scale tRS_rD in place.  This
        # MUST run after the postact copy/quant above (tRS_rPostAct already holds
        # SwiGLU(z) computed from the un-scaled z).
        #
        # CRITICAL — TWO scale groups per subtile at epi_tile_n=64.  The z scale
        # buffer has one UE8M0 byte per 1x32 column group, but the combined mixin
        # forces epi_tile_n(z)=64 (the y1 postact half-N needs exactly one 1x32
        # group/subtile).  So each z subtile's tRS_rD spans 64 GEMM-N columns =
        # TWO 1x32 groups.  The accumulator fragment is N-major contiguous (tmem
        # Ld load, (4,1) epilogue-warp shape), so group g occupies reg indices
        # [g*32, g*32+32).  We MUST compute a per-group amax (with the 1e-4 floor)
        # and store TWO scale bytes — at n_group_abs = n_base + n_sub*2 + g — or
        # exactly half the z scale groups are left stale (the bug a single
        # amax/byte per subtile produced: ref-vs-tile64 = 352970/524288).
        _z_scale_active = epi_loop_tensors["mZScale"]
        if const_expr(_z_scale_active is not None):
            num_z = cute.size(tRS_rD)
            groups_z = const_expr(num_z // 32)
            scale_tensor, m_abs, n_group_base, m_limit, n_limit = epi_loop_tensors["mZScale"]
            for g in cutlass.range(groups_z, unroll_full=True):
                base = g * 32
                amax = Float32(0.0)
                for j in cutlass.range(32, unroll_full=True):
                    val = tRS_rD[base + j]
                    neg = Float32(0.0) - val
                    abs_val = cute.arch.fmax(val, neg)
                    amax = cute.arch.fmax(amax, abs_val)
                amax = cute.arch.fmax(amax, Float32(1e-4))

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

                qexp = Int32(254) - e8m0
                qexp_hi = cutlass.Boolean(qexp > Int32(254))
                qexp = Int32(254) if qexp_hi else qexp
                qexp_lo = cutlass.Boolean(qexp < Int32(1))
                qexp = Int32(1) if qexp_lo else qexp
                quant_scale = _i32_as_f32(qexp << Int32(23))

                for j in cutlass.range(32, unroll_full=True):
                    tRS_rD[base + j] = tRS_rD[base + j] * quant_scale

                # BlockscaledScaleStoreDual.begin_loop already mapped the subtile
                # index to the FIRST group of this 64-col subtile
                # (n_group_base = n_base + n_sub*2).  The two groups of this
                # subtile are therefore n_group_base + 0 and n_group_base + 1.
                n_group_abs = n_group_base + g
                in_bounds = (
                    cutlass.Boolean(m_abs < m_limit)
                    & cutlass.Boolean(n_group_abs < n_limit)
                )
                if in_bounds:
                    scale_tensor[m_abs, n_group_abs] = cutlass.Int8(e8m0)

        return tRS_rPostAct


# ---------------------------------------------------------------------------
# GemmDGatedMixin (from gemm_dgated.py)
# ---------------------------------------------------------------------------

class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileStore("mPostAct"), ColVecReduce("mColVecReduce"))
    _extra_param_fields = (
        ("act_bwd_fn", cutlass.Constexpr, None),
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
        assert self.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        self.swiglu_clamp_value = args.swiglu_clamp_value
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["swiglu_clamp_value"] = args.swiglu_clamp_value
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]
        assert tRS_rC is not None
        implicit_dtype = self.implicit_dtype
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
        if const_expr(
            self.arch < 100 and not self.swiglu_clamp_value > 0.0
            and _is_quack_act(params.act_bwd_fn, "dswiglu_precise")
        ):
            _dswiglu_precise_frag(tRS_rXY_f32x2, tRS_rD_scaled, tRS_rdXY_f32x2, tRS_rOut)
        elif const_expr(self.arch < 100):
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


@dsl_user_op
def _cvt_e4m3x4(a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None) -> Int32:
    """Four f32 -> four E4M3 bytes packed little-endian (a in the low byte)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
            "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
            "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _b32_op(ptx: str, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """Binary packed-16-bit PTX op on two b32 words (``$0 = op($1, $2)``)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            ptx + " $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


class Sm90DzQuantStore(EpiOp):
    """State for the fused 1x128 dz quant of the SM90 dGated epilogue.

    ``param`` is the E4M3 dz output viewed as int32 words ``[M, 2N/4]``.  A
    1x128 dz group is 64 packed (bf16x2) D columns = two 32-column epilogue
    subtiles, so ``begin`` allocates a register buffer that carries the even
    subtile's dz values to the odd one, where the group is quantized.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        return {self.name: None if tensor is None else assume_stride_divisibility(tensor)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        state = None
        if const_expr(param is not None):
            coord = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            coord = cute.group_modes(coord, 3, cute.rank(coord))
            n_vals = cute.size(coord[None, None, None, 0])
            buf = cute.make_rmem_tensor(cute.make_layout(n_vals), Int32)
            state = (param, coord, buf, ctx.tile_coord_mnkl, ctx.varlen_manager, ctx.tiled_copy_r2s)
        return state

    def begin_loop(self, gemm, state, epi_coord):
        result = None
        if const_expr(state is not None):
            param, coord, buf, tile_coord, varlen, tiled_copy = state
            n_sub = epi_coord[1] if isinstance(epi_coord, tuple) else epi_coord
            assert isinstance(n_sub, int), "epilogue subtile loop must be unrolled"
            result = (param, coord[None, None, None, epi_coord], buf, tile_coord, varlen,
                      tiled_copy, n_sub)
        return result


class Sm90DGatedDzQuantMixin(GemmDGatedMixin):
    """dGated that also writes dz as E4M3 with raw 1x128 UE8M0 scales.

    Byte-identical to ``quantize_activation_blockscaled_fast(dz_bf16, 128)``:
    the group amax is taken over the bf16-rounded dz that is stored, with the
    same integer+carry E8M0 rule and RN-satfinite E4M3 conversion.
    """

    _epi_ops = (*GemmDGatedMixin._epi_ops, Sm90DzQuantStore("mDzFp8"))
    _extra_param_fields = (*GemmDGatedMixin._extra_param_fields, ("mDzScale", object, None))

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0  # RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mDzFp8: Optional[cute.Tensor] = None
        mDzScale: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        assert self.cta_tile_shape_mnk[1] % 64 == 0, (
            "fused 1x128 dz quant needs tile N divisible by 64 (two 32-column subtiles)"
        )
        params = GemmDGatedMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
        params.mDzScale = args.mDzScale
        return params

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tRS_rOut = GemmDGatedMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        info = epi_loop_tensors["mDzFp8"]
        if const_expr(info is not None):
            param, coords, buf, tile_coord, varlen, tiled_copy, n_sub = info
            assert self.epi_tile[1] == 32, "fused dz quant needs 32-column subtiles"
            lane_layout, warp_layout = _get_lane_warp_layouts(tiled_copy, True)
            assert cute.size(lane_layout, mode=[1]) == 4 and cute.size(warp_layout, mode=[1]) == 1
            n_vals = cute.size(tRS_rD)
            # tRS_rD now holds the stored dz as bf16x2 words (low half = even column).
            if const_expr(n_sub % 2 == 0):
                for i in cutlass.range_constexpr(n_vals):
                    buf[i] = _f32_as_i32(tRS_rD[i])
            else:
                # SM90 accumulator fragment of a 32-column subtile (16 values):
                # value i sits in row half (i // 2) % 2 and column
                # 2 * lane_n + i % 2 + 8 * (i // 4); the 4 N lanes of a row
                # together cover the 64 packed columns (128 dz values) of a group.
                # |bf16| bit patterns order like their values -> u16x2 integer max.
                assert n_vals == 16
                umax = "max.u16x2"
                acc = [None, None]
                for i in cutlass.range_constexpr(n_vals):
                    ab = _b32_op(umax, buf[i] & Int32(0x7FFF7FFF),
                                 _f32_as_i32(tRS_rD[i]) & Int32(0x7FFF7FFF))
                    r = (i // 2) % 2
                    acc[r] = ab if const_expr(acc[r] is None) else _b32_op(umax, acc[r], ab)
                e8m0, qscale = [], []
                for r in cutlass.range_constexpr(2):
                    h = _b32_op(umax, acc[r], acc[r] >> Int32(16)) & Int32(0xFFFF)
                    e, q = _ue8m0_from_amax(
                        cute.arch.warp_reduction(_i32_as_f32(h << Int32(16)), cute.arch.fmax,
                                                 threads_in_group=4)
                    )
                    e8m0.append(e)
                    qscale.append(q)
                batch = tile_coord[3]
                batch_start = varlen.params.cu_seqlens_m[batch] if varlen.varlen_m else Int32(0)
                m_limit = (
                    varlen.params.cu_seqlens_m[batch + Int32(1)]
                    if varlen.varlen_m else param.shape[0]
                )
                n_limit = param.shape[1] * Int32(2)  # packed D columns
                tile_m, tile_n = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
                m_base = batch_start + tile_coord[0] * Int32(tile_m)
                n_lane = tile_coord[1] * Int32(tile_n) + coords[0][1]  # column of value 0
                w_prev = (n_lane >> Int32(1)) - Int32(16)  # int32 word of value 0, even subtile
                hi_mask = Int32(-65536)  # 0xFFFF0000
                # n_limit and the group start are multiples of 64 packed columns,
                # so a 1x128 group is either fully inside or fully outside N.
                n_ok = cutlass.Boolean(n_lane < n_limit)
                ln = cute.arch.lane_idx() & Int32(3)  # == lane_n of the WGMMA fragment
                w_lane = w_prev + Int32(3) * ln  # group word base + 4 * ln
                for r in cutlass.range_constexpr(2):
                    qs = qscale[r]
                    prev, cur = [], []
                    for k in cutlass.range_constexpr(4):  # 8-column blocks
                        i = 4 * k + 2 * r  # words i, i+1: packed columns (n, n+1)
                        words = []
                        for src in (buf[i], buf[i + 1], _f32_as_i32(tRS_rD[i]),
                                    _f32_as_i32(tRS_rD[i + 1])):
                            words.append((_i32_as_f32(src << Int32(16)) * qs,
                                          _i32_as_f32(src & hi_mask) * qs))
                        prev.append(_cvt_e4m3x4(*words[0], *words[1]))
                        cur.append(_cvt_e4m3x4(*words[2], *words[3]))
                    # Lane ln holds word 4k + ln of block k; a 4x4 transpose over the
                    # 4 N lanes gives each lane 4 consecutive words -> 16B stores.
                    prev = _xpose4_lanes(prev, ln)
                    cur = _xpose4_lanes(cur, ln)
                    m_abs = m_base + coords[2 * r][0]
                    if cutlass.Boolean(m_abs < m_limit) & n_ok:
                        ptr = cute.domain_offset((m_abs, w_lane), param).iterator
                        _st_global_v4(ptr, prev)
                        _st_global_v4(ptr + 16, cur)
                        if cutlass.Boolean(ln == Int32(0)):
                            params.mDzScale[m_abs, n_lane >> Int32(6)] = cutlass.Int8(e8m0[r])
        return tRS_rOut


def _xpose4_lanes(v, ln):
    """4x4 transpose of ``v[k]`` (lane ln, block k) across the 4 adjacent lanes."""
    v = list(v)
    for bit, pairs in ((2, ((0, 2), (1, 3))), (1, ((0, 1), (2, 3)))):
        m = Int32(0) - ((ln >> Int32(bit // 2)) & Int32(1))  # all-ones on "hi" lanes
        nm = ~m
        for lo_k, hi_k in pairs:
            send = (v[lo_k] & m) | (v[hi_k] & nm)
            recv = cute.arch.shuffle_sync_bfly(send, offset=bit)
            v[lo_k] = (recv & m) | (v[lo_k] & nm)
            v[hi_k] = (v[hi_k] & m) | (recv & nm)
    return v


@dsl_user_op
def _st_global_v4(ptr, vals, *, loc=None, ip=None) -> None:
    """16B ``st.global.v4.b32`` of four Int32 words at a 16B-aligned gmem pointer."""
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
        + [Int32(v).ir_value(loc=loc, ip=ip) for v in vals],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _st_global_v4_addr(addr, vals, *, loc=None, ip=None) -> None:
    """``_st_global_v4`` at a raw Int64 gmem address."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(v).ir_value(loc=loc, ip=ip) for v in vals],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _prmt2(a: Int32, b: Int32, sel: Int32, *, loc=None, ip=None) -> Int32:
    """``prmt.b32 d, a, b, sel``: bytes 0-3 from ``a``, 4-7 from ``b``."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
             Int32(sel).ir_value(loc=loc, ip=ip)],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _absmax_f32(a: Float32, v: Float32, *, loc=None, ip=None) -> Float32:
    """``max(a, |v|)`` (ptxas folds the abs into one FMNMX)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 t; abs.f32 t, $2; max.f32 $0, $1, t; }",
            "=f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_bf16_as_f32(smem_addr, *, loc=None, ip=None) -> Float32:
    """bf16 at ``smem_addr`` widened to f32 (exact: bf16 is the f32 high half)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 t; ld.shared.u16 t, [$1]; shl.b32 $0, t, 16; }",
            "=f,r",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


def _colq128_group(vals, q):
    """Quantize one feature's 128-row group held by 8 lanes (lane bits 2-4 = q).

    ``vals[j]`` is row ``16 * q + (j ^ q)`` of the group (the XOR stagger makes
    the smem column reads bank-conflict free).  Returns ``(e8m0, words)`` where
    ``words`` are the E4M3 bytes of rows ``16q .. 16q+15`` in order.
    """
    a = Float32(0.0)
    for v in vals:
        a = _absmax_f32(a, v)
    for off in (4, 8, 16):
        a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=off))
    e8m0, qs = _ue8m0_from_amax(a)
    w = [_cvt_e4m3x4(vals[4 * k] * qs, vals[4 * k + 1] * qs, vals[4 * k + 2] * qs,
                     vals[4 * k + 3] * qs) for k in range(4)]
    # Slot (k, b) holds row offset 4 * (k ^ q>>2) + (b ^ q&3): output word K takes
    # byte B from slot (K ^ q>>2, B ^ q&3), i.e. one prmt over (w[K], w[K ^ 1]).
    sel = ((Int32(0x3210) ^ ((q & Int32(3)) * Int32(0x1111)))
           + ((q >> Int32(2)) & Int32(1)) * Int32(0x4444))
    return e8m0, [_prmt2(w[k], w[k ^ 1], sel) for k in range(4)]


class Sm90DGatedColQuantMixin(Sm90DGatedDzQuantMixin):
    """dGated that also emits the colwise 1x128 wgrad operands of y1s and/or dz.

    Byte-identical to ``quantize_wgrad_colwise128_sm90(y1s|dz, offsets)`` on
    rows ``< offsets[E]``: fp8 ``[F, total_K]`` + u8 scales ``[total_K/128+pad, F]``.
    Each 128-row CTA tile is one group (expert segments must be 128-aligned).
    After a subtile is staged in smem for its TMA store, each of the 256
    epilogue threads reads 16 rows of one column unit (a y1s column / a packed
    dz word = 2 features) and 8 lanes reduce the group amax by shuffles.
    """

    dz_col_inplace = False  # dz col written tile-locally into C (see the dz branch)

    _extra_param_fields = (
        *Sm90DGatedDzQuantMixin._extra_param_fields,
        ("mY1sCol", object, None), ("mY1sColScale", object, None),
        ("mDzCol", object, None), ("mDzColScale", object, None),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        swiglu_clamp_value: cutlass.Constexpr[float] = 0.0
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0  # RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mDzFp8: Optional[cute.Tensor] = None
        mDzScale: Optional[cute.Tensor] = None
        mY1sCol: Optional[cute.Tensor] = None
        mY1sColScale: Optional[cute.Tensor] = None
        mDzCol: Optional[cute.Tensor] = None
        mDzColScale: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        params = Sm90DGatedDzQuantMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
        for name in ("mY1sCol", "mY1sColScale", "mDzCol", "mDzColScale"):
            setattr(params, name, getattr(args, name))
        return params

    @cute.jit
    def epi_after_smem_store(self, params, epi_smem_tensors, sD, epi_buffer, gmem_coord,
                             tile_coord, varlen, tidx, epi_idx: cutlass.Constexpr[int],
                             epi_tile_num: cutlass.Constexpr[int]):
        # Runs on odd subtiles only and covers both staged buffers with ONE rolled
        # loop over (subtile, operand/half) jobs, so the group-quant body is emitted
        # once: unrolled copies per subtile overflow the icache (dz fusion was 2x slower).
        # smem is read by raw address with the swizzle folded into 8 per-thread row
        # offsets (rows j and j + 8 differ by a constant), not per-element crd2idx.
        has_y = params.mY1sCol is not None
        has_d = params.mDzCol is not None
        if const_expr((has_y or has_d) and epi_idx % 2 == 1):
            assert self.cta_tile_shape_mnk[0] == 128 and tuple(self.epi_tile) == (128, 32)
            assert self.mma_warp_groups == 2 and not self.pingpong
            assert self.epi_stage == 2 and epi_tile_num % 2 == 0
            lane = tidx & Int32(31)
            q = lane >> Int32(2)
            u = ((tidx >> Int32(5)) & Int32(7)) * Int32(4) + (lane & Int32(3))
            batch = tile_coord[3]
            m_base = varlen.params.cu_seqlens_m[batch] + tile_coord[0] * Int32(128)
            group = m_base >> Int32(7)
            k_word = (m_base + q * Int32(16)) >> Int32(2)
            # Subtile epi_idx - 1 sits in the other buffer; epi tile M == CTA tile M.
            col0 = (tile_coord[1] * Int32(self.cta_tile_shape_mnk[1])
                    + Int32((gmem_coord[1] - 1) * 32) + u)
            xs = [Int32(j) ^ q for j in range(8)]  # row 16q + x (+8 for j >= 8)
            n_y = 1 if const_expr(has_y) else 0
            n_k = n_y + (2 if const_expr(has_d) else 0)
            if const_expr(has_y):
                # sPostAct: bf16 (128, 32) rows of 64 B, 64B swizzle (S<2,4,3> on bytes).
                sP = epi_smem_tensors[self._epi_smem_map["mPostAct"]]
                assert cute.cosize(sP.layout) == 8192 and sP.element_type == cutlass.BFloat16
                pbase = sP.iterator.toint() + q * Int32(1024) + u * Int32(2)
                p_off = [pbase + x * Int32(64) ^ ((x << Int32(3)) & Int32(0x30)) for x in xs]
            if const_expr(has_d):
                # sD: f32 words (128, 32) rows of 128 B, 128B swizzle (S<3,4,3> on bytes).
                assert cute.cosize(sD.layout) == 8192 and sD.element_type == cutlass.Float32
                dbase = sD.iterator.toint() + q * Int32(2048) + u * Int32(4)
                d_off = [dbase + x * Int32(128) ^ (x << Int32(4)) for x in xs]
            for s in cutlass.range(2, unroll=1):
                b = (epi_buffer + Int32(1) + s) & Int32(1)
                col = col0 + s * Int32(32)
                if const_expr(has_y):
                    pa = [o + b * Int32(8192) for o in p_off]
                if const_expr(has_d):
                    da = [o + b * Int32(16384) for o in d_off]
                # k < n_y: y1s column; else dz feature 2 col + h (h = bf16 half of the word).
                for k in cutlass.range(n_k, unroll=1):
                    h = k - Int32(n_y)
                    vals = [Float32(0.0)] * 16
                    if const_expr(has_y and has_d):
                        if k < Int32(n_y):
                            vals = [_ld_shared_bf16_as_f32(pa[j & 7] + Int32(512 * (j >> 3)))
                                    for j in range(16)]
                        else:
                            sh = (Int32(1) - h) * Int32(16)
                            vals = [_i32_as_f32((_ld_shared_b32(da[j & 7] + Int32(1024 * (j >> 3)))
                                                 << sh) & Int32(-65536)) for j in range(16)]
                    elif const_expr(has_y):
                        vals = [_ld_shared_bf16_as_f32(pa[j & 7] + Int32(512 * (j >> 3)))
                                for j in range(16)]
                    else:
                        sh = (Int32(1) - h) * Int32(16)
                        vals = [_i32_as_f32((_ld_shared_b32(da[j & 7] + Int32(1024 * (j >> 3)))
                                             << sh) & Int32(-65536)) for j in range(16)]
                    e8m0, w = _colq128_group(vals, q)
                    if const_expr(has_y):
                        if k < Int32(n_y):
                            if col < params.mY1sCol.shape[0]:
                                _st_global_v4(cute.domain_offset((col, k_word),
                                                                 params.mY1sCol).iterator, w)
                                if q == Int32(0):
                                    params.mY1sColScale[group, col] = cutlass.Int8(e8m0)
                    if const_expr(has_d):
                        f = col * Int32(2) + h
                        if k >= Int32(n_y):
                            if const_expr(self.dz_col_inplace):
                                # mDzCol = z as int32 [M, F/2]: block (group, f>>7) lands in
                                # rows m_base + (f&127), bytes 256 (f>>7) + [0, 128), i.e. the
                                # even subtile's own preact, already consumed by this CTA.
                                if f < params.mDzCol.shape[1] * Int32(2):
                                    _st_global_v4(cute.domain_offset(
                                        (m_base + (f & Int32(127)),
                                         (f >> Int32(7)) * Int32(64) + q * Int32(4)),
                                        params.mDzCol).iterator, w)
                                    if q == Int32(0):
                                        params.mDzColScale[group, f] = cutlass.Int8(e8m0)
                            elif f < params.mDzCol.shape[0]:
                                _st_global_v4(cute.domain_offset((f, k_word),
                                                                 params.mDzCol).iterator, w)
                                if q == Int32(0):
                                    params.mDzColScale[group, f] = cutlass.Int8(e8m0)

@dsl_user_op
def _cp_async_4(smem_addr, gmem_addr, src_bytes, *, loc=None, ip=None) -> None:
    """4-byte ``cp.async.ca`` (``src_bytes`` = 0 zero-fills without reading gmem)."""
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip),
         Int32(src_bytes).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4, $2;",
        "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(smem_addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _cvt_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """RN f32 pair -> bf16x2 word (``lo`` in the low half), as the D store's cvt."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _route_reduce_f32(x: Float32, w: Float32, *, loc=None, ip=None) -> Float32:
    """``0.0f + x * w`` as token_gather_sum computes it for one route (one fma.rn)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(w).ir_value(loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, 0f00000000;",
            "=f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


class Sm90RouteScatterStore(EpiOp):
    """Per-row direct store of a bf16 varlen-M GEMM output (replaces the TMA D store).

    ``param`` is D ``[M, N]`` (route order).  The row ``p`` of a token with a
    single local route (``mScatterDst[p] = token >= 0``) is reduced in place,
    i.e. written to ``mScatterOut[token]`` as ``bf16(0 + bf16(acc) * w[p])``
    (``w`` = ``mScatterW`` or 1 with the weightless ``0 + x``), exactly what
    ``token_gather_sum`` would produce.  Every other row goes to ``D[p]`` for
    the reduction of the multi-route tokens.  ``begin`` allocates per-thread
    row state (2 rows of the WGMMA fragment), filled on the first subtile.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        return {self.name: None if tensor is None else assume_stride_divisibility(tensor)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        state = None
        if const_expr(param is not None):
            coord = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            coord = cute.group_modes(coord, 3, cute.rank(coord))
            state = (param, coord, ctx.tile_coord_mnkl, ctx.varlen_manager, ctx.tiled_copy_r2s)
        return state

    def begin_loop(self, gemm, state, epi_coord):
        result = None
        if const_expr(state is not None):
            param, coord, tile_coord, varlen, tiled_copy = state
            result = (param, coord[None, None, None, epi_coord], tile_coord, varlen, tiled_copy)
        return result


@dsl_user_op
def _stmatrix_x4(smem_addr, vals, *, loc=None, ip=None) -> None:
    """``stmatrix.x4``: word ``vals[j]`` is lane l's (row l // 4, cols 2 * (l % 4) + {0, 1})
    of 8x8 bf16 matrix ``j``; lane l gives the smem address of row l % 8 of matrix l // 8."""
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)]
        + [Int32(v).ir_value(loc=loc, ip=ip) for v in vals],
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _ld_shared_v4(smem_addr, *, loc=None, ip=None):
    """16B ``ld.shared.v4.b32`` -> 4 Int32 words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [j], loc=loc, ip=ip)) for j in range(4)]


@dsl_user_op
def _add_zero_bf16x2(word: Int32, *, loc=None, ip=None) -> Int32:
    """``0 + x`` on both bf16 halves: ``bf16(0.0f + x)`` for a bf16 ``x`` (-0 -> +0)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 z; mov.b32 z, 0; add.rn.bf16x2 $0, $1, z; }",
            "=r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


class Sm90ScaledGemmRouteScatterMixin(GemmDefaultEpiMixin):
    """Default epilogue with the single-route token reduction fused in.

    Fwd down-proj (``mScatterW`` = router scores) and bwd dx (no weight): the
    rows of tokens with one local route are written straight to the
    ``[T, N]`` token output, so ``token_gather_sum`` only reduces the
    multi-route tokens (``skip_single=True``).  Bitwise identical to the
    unfused GEMM + ``token_gather_sum``.
    """

    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, Sm90RouteScatterStore("mScatterD"))
    _extra_param_fields = (
        ("mScatterDst", object, None), ("mScatterOut", object, None), ("mScatterW", object, None),
    )
    _epi_direct_store = True
    # Per-warp smem for the fragment -> row transpose: 16 rows x 32 bf16, rows
    # padded to 80 B so the stmatrix row writes are bank-conflict free.
    _XPOSE_PITCH = 80
    _XPOSE_WARP_BYTES = 16 * 80

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = 0  # RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mScatterD: Optional[cute.Tensor] = None
        mScatterDst: Optional[cute.Tensor] = None
        mScatterOut: Optional[cute.Tensor] = None
        mScatterW: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        assert args.rounding_mode == 0 and not args.add_to_output
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        d.update(mScatterDst=args.mScatterDst, mScatterOut=args.mScatterOut,
                 mScatterW=args.mScatterW)
        return self.EpilogueParams(**d)

    @cute.jit
    def _epi_tile_rows(self, tiled_copy_r2s, tile_coord, varlen, tidx):
        """(m_abs of this thread's 2 fragment rows, segment end) of the tile."""
        tile_m, tile_n = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        coord = partition_for_epilogue(
            cute.make_identity_tensor((tile_m, tile_n)), epi_tile=self.epi_tile,
            tiled_copy=tiled_copy_r2s, tidx=tidx, reference_src=True,
        )
        coord = cute.group_modes(coord, 3, cute.rank(coord))[None, None, None, 0]
        batch = tile_coord[3]
        m_base = varlen.params.cu_seqlens_m[batch] + tile_coord[0] * Int32(tile_m)
        m_limit = varlen.params.cu_seqlens_m[batch + Int32(1)]
        return m_base, (m_base + coord[0][0], m_base + coord[2][0]), m_limit

    @cute.jit
    def epi_tile_prefetch(self, params, tiled_copy_r2s, sD, tile_coord, varlen, tidx):
        """Start loading the route destination / weight of this thread's 2 fragment rows.

        Called before the mainloop: 4-byte ``cp.async`` into a private 16 B slot
        at the start of the D staging smem (unused by the direct-store epilogue),
        so the gmem latency hides behind the MMA without holding registers.
        The per-warp transpose buffers follow the slots.  Returns the slot address.
        """
        assert not self.pingpong, "route scatter slots live in the D smem shared by both WGs"
        stage_bytes = cute.size(self.epi_tile) * sD.element_type.width // 8
        assert stage_bytes * self.epi_stage >= (
            16 * self.num_epi_warps * 32 + self._XPOSE_WARP_BYTES * self.num_epi_warps
        )
        m_base, m_abs, m_limit = self._epi_tile_rows(tiled_copy_r2s, tile_coord, varlen, tidx)
        has_w = const_expr(params.mScatterW is not None)
        slot = sD.iterator.toint() + tidx * Int32(16)
        for r in cutlass.range_constexpr(2):
            ok = m_abs[r] < m_limit
            m_safe = cutlass.select_(ok, m_abs[r], m_base)  # m_base is a valid row if any is
            nbytes = cutlass.select_(ok, Int32(4), Int32(0))  # 0: zero-fill, no gmem read
            _cp_async_4(slot + Int32(4 * r),
                        cute.domain_offset((m_safe,), params.mScatterDst).iterator.toint(), nbytes)
            if const_expr(has_w):
                _cp_async_4(slot + Int32(8 + 4 * r),
                            cute.domain_offset((m_safe,), params.mScatterW).iterator.toint(),
                            nbytes)
        cute.arch.cp_async_commit_group()
        return slot

    @cute.jit
    def epi_tile_prefetch_load(self, params, slot, tiled_copy_r2s, tile_coord, varlen, tidx):
        """Per-tile row state, once before the subtile loop: for each of the 2 rows
        (gmem row base address, weight, in-segment), plus the stmatrix / ld.shared
        addresses of the warp's transpose buffer.  Single-route rows go to
        ``mScatterOut[dst]`` with their router weight; the others to ``D`` with
        weight 1, i.e. ``bf16(0 + x)`` = x up to the sign of zero, which the
        +0-seeded fp32 sum of ``token_gather_sum`` cannot observe.
        """
        _, m_abs, m_limit = self._epi_tile_rows(tiled_copy_r2s, tile_coord, varlen, tidx)
        param = params.mScatterD
        cute.arch.cp_async_wait_group(0)
        rows = []
        for r in cutlass.range_constexpr(2):
            d = _ld_shared_b32(slot + Int32(4 * r))
            single = cutlass.Boolean(d >= Int32(0))
            addr = cute.domain_offset((m_abs[r], 0), param).iterator.toint()
            if single:
                addr = cute.domain_offset((d, 0), params.mScatterOut).iterator.toint()
            w = Float32(1.0)
            if const_expr(params.mScatterW is not None):
                w = cutlass.select_(single, _i32_as_f32(_ld_shared_b32(slot + Int32(8 + 4 * r))),
                                    Float32(1.0))
            rows.append((addr, w, cutlass.Boolean(m_abs[r] < m_limit)))
        lane = tidx & Int32(31)
        buf = (slot - tidx * Int32(16) + Int32(16 * self.num_epi_warps * 32)
               + (tidx >> Int32(5)) * Int32(self._XPOSE_WARP_BYTES))
        pitch = Int32(self._XPOSE_PITCH)
        st_addr = buf + (lane & Int32(7)) * pitch + (lane >> Int32(3)) * Int32(16)
        ld_addr = buf + (lane >> Int32(2)) * pitch + (lane & Int32(3)) * Int32(16)
        return rows, st_addr, ld_addr

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        param, coords, tile_coord, varlen, tiled_copy = epi_loop_tensors["mScatterD"]
        rows, st_addr, ld_addr = epi_loop_tensors["epi_pre"]
        tile_n = self.cta_tile_shape_mnk[1]
        assert self.epi_tile[1] == 32 and self.epi_tile[0] == self.cta_tile_shape_mnk[0], (
            "route scatter needs full-M, 32-column epilogue subtiles"
        )
        lane_layout, warp_layout = _get_lane_warp_layouts(tiled_copy, True)
        assert cute.size(lane_layout, mode=[1]) == 4 and cute.size(warp_layout, mode=[1]) == 1
        assert cute.size(tRS_rD) == 16
        has_w = const_expr(params.mScatterW is not None)
        ln = cute.arch.lane_idx() & Int32(3)  # == lane_n of the WGMMA fragment
        # Value 0 of lane ln is at column 2 * ln of the subtile; after the smem
        # transpose lane ln holds columns 8 * ln .. 8 * ln + 7.
        col = tile_coord[1] * Int32(tile_n) + coords[0][1] + Int32(6) * ln
        n_ok = cutlass.Boolean(col < param.shape[1])  # N % 8 == 0: 8-col blocks never straddle
        col_bytes = cutlass.Int64(col) * cutlass.Int64(param.element_type.width // 8)
        hi_mask = Int32(-65536)  # 0xFFFF0000
        half = Int32(8 * self._XPOSE_PITCH)
        cute.arch.sync_warp()  # previous subtile's ld.shared done before the buffer is rewritten
        for r in cutlass.range_constexpr(2):
            wv = rows[r][1]
            words = []
            for k in cutlass.range_constexpr(4):  # 8-column blocks
                i = 4 * k + 2 * r  # values i, i + 1: columns (c, c + 1) of row half r
                word = _cvt_bf16x2(tRS_rD[i], tRS_rD[i + 1])
                if const_expr(has_w):
                    lo = _route_reduce_f32(_i32_as_f32(word << Int32(16)), wv)
                    hi = _route_reduce_f32(_i32_as_f32(word & hi_mask), wv)
                    word = _cvt_bf16x2(lo, hi)
                else:
                    word = _add_zero_bf16x2(word)
                words.append(word)
            # Matrix k of row half r = 8-column block k: the WGMMA fragment layout.
            _stmatrix_x4(st_addr + Int32(r) * half, words)
        cute.arch.sync_warp()
        for r in cutlass.range_constexpr(2):
            addr, _, ok = rows[r]
            words = _ld_shared_v4(ld_addr + Int32(r) * half)
            if ok & n_ok:
                _st_global_v4_addr(addr + col_bytes, words)
        return None


@dsl_user_op
@cute.jit
def _ue8m0_from_amax(amax: Float32, *, loc=None, ip=None):
    """Integer+carry UE8M0 rule of ``_quantize_flat_v2_kernel``: (e8m0, 2**(127 - e8m0))."""
    bits = _f32_as_i32(amax)
    biased_exp = (bits >> Int32(23)) & Int32(0xFF)
    carry = Int32(1) if cutlass.Boolean((bits & Int32(0x7FFFFF)) > Int32(0x600000)) else Int32(0)
    e8m0 = biased_exp - Int32(8) + carry
    e8m0 = e8m0 if cutlass.Boolean(biased_exp > Int32(0)) else Int32(0)
    e8m0 = e8m0 if cutlass.Boolean(e8m0 > Int32(0)) else Int32(0)
    qexp = Int32(254) - e8m0
    qexp = Int32(254) if cutlass.Boolean(qexp > Int32(254)) else qexp
    qexp = Int32(1) if cutlass.Boolean(qexp < Int32(1)) else qexp
    return e8m0, _i32_as_f32(qexp << Int32(23))


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
                # Upper bound on valid absolute M rows for this batch/expert. The
                # fp8-preact reads below (fp8_tensor[m_abs], scales_tensor[m_abs])
                # are manual LDGs with no TMA predication, unlike D. Under a 2-CTA
                # cluster (cluster_m=2) the scheduler can hand a CTA a phantom
                # upper-half M-tile whose rows extend past this expert's segment
                # (and, for the last expert, past total_M), so m_abs must be
                # bounds-checked or the LDG reads past the allocation (observed
                # CUDA-700 illegal global read). cluster_m=1 never produces such a
                # phantom tile, so this guard is a no-op there.
                m_limit = ctx.varlen_manager.params.cu_seqlens_m[ctx.tile_coord_mnkl[3] + Int32(1)]
            else:
                m_offset = Int32(0)
                m_limit = fp8_tensor.shape[0]  # total_M
            m_base = ctx.tile_coord_mnkl[0] * tile_M

            # Identity tensor partitioned for this thread's epilogue elements
            # This gives the exact (row, col) for each register position
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((tile_M, tile_N))
            )

            # N base in fp8 logical coordinates (tile_N f32 = 2*tile_N bf16 = 2*tile_N fp8)
            n_base_logical = ctx.tile_coord_mnkl[1] * tile_N * 2

            return (fp8_tensor, scales_tensor, tDcD, m_offset, m_base, n_base_logical, m_limit)
        return None

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        if const_expr(state is not None):
            fp8_tensor, scales_tensor, tDcD, m_offset, m_base, n_base, m_limit = state
            # Extract this subtile's identity coordinates
            tDcD_sub = cute.group_modes(tDcD, 3, cute.rank(tDcD))[None, None, None, epi_coord]
            return (fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base, m_limit)
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
            assert self.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
            assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        d["swiglu_clamp_value"] = args.swiglu_clamp_value
        self.swiglu_clamp_value = args.swiglu_clamp_value
        return self.EpilogueParams(**d)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
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
            fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base, m_limit = fp8_preact_info

            # tDcD_sub[i] gives (row_in_tile, col_in_tile) for each D register element
            # col is C's physical N (f32 = bf16x2). Each f32 maps to 2 fp8 bytes.
            num_d = cute.size(tDcD_sub)

            # Allocate fp8 register tensor (2x D elements for gate+up pairs)
            tRS_rXY_bf16_layout = cute.recast_tensor(tRS_rD, cutlass.BFloat16).layout
            tRS_rFP8 = cute.make_rmem_tensor(tRS_rXY_bf16_layout.shape, cutlass.Float8E4M3FN)

            # Load fp8 bytes using identity-derived coordinates
            # For each D[i] at (row, col), load fp8[row, col*2] and fp8[row, col*2+1]
            # Columns past N (partial last N tile) are clamped like rows: their
            # garbage bytes/scales could be NaN/inf and NaN * (zero D) poisons ds.
            n_lim = Int32(fp8_tensor.shape[1])
            for i in cutlass.range(num_d, unroll_full=True):
                coord = tDcD_sub[i]
                row = coord[0]
                col = coord[1]
                m_abs = m_offset + m_base + row
                # Clamp to last valid row: under cluster_m=2 a phantom upper tile
                # gives rows past this expert's segment (and total_M for the last
                # expert). These manual LDGs have no TMA predication, so clamp the
                # address; the value is dropped by D's predication downstream.
                m_safe = m_abs if m_abs < m_limit else (m_limit - Int32(1))
                n0 = n_base + col * 2
                n0 = n0 if n0 < n_lim else (n_lim - Int32(2))
                tRS_rFP8[2 * i] = fp8_tensor[m_safe, n0]
                tRS_rFP8[2 * i + 1] = fp8_tensor[m_safe, n0 + Int32(1)]

            # Vectorized fp8->f32 (DSL auto-packs vec4 -> nvgpu.cvt_fpext)
            tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_bf16_layout.shape, Float32)
            tRS_rXY_f32x2.store(tRS_rFP8.load().to(Float32))

            # Blockscaled dequant using identity coordinates for correct group index
            for i in cutlass.range(num_d, unroll_full=True):
                coord = tDcD_sub[i]
                row = coord[0]
                col = coord[1]
                m_abs = m_offset + m_base + row
                m_safe = m_abs if m_abs < m_limit else (m_limit - Int32(1))
                n0 = n_base + col * 2
                n0 = n0 if n0 < n_lim else (n_lim - Int32(2))
                group_0 = n0 >> Int32(5)
                group_1 = (n0 + Int32(1)) >> Int32(5)
                scale_0 = _i32_as_f32(Int32(scales_tensor[m_safe, group_0]) << Int32(23))
                scale_1 = _i32_as_f32(Int32(scales_tensor[m_safe, group_1]) << Int32(23))
                tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            # ── Standard bf16 PreAct path ──
            assert tRS_rC is not None
            implicit_dtype = self.implicit_dtype
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
        if const_expr(
            self.arch < 100 and not self.swiglu_clamp_value > 0.0
            and _is_quack_act(params.act_bwd_fn, "dswiglu_precise")
        ):
            _dswiglu_precise_frag(tRS_rXY_f32x2, tRS_rD_scaled, tRS_rdXY_f32x2, tRS_rOut)
        elif const_expr(self.arch < 100):
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
            pack_dtype = self.implicit_dtype
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
                fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base, m_limit = fp8_preact_info
                num_d = cute.size(tDcD_sub)
                for i in cutlass.range(num_d, unroll_full=True):
                    coord = tDcD_sub[i]
                    row, col = coord[0], coord[1]
                    m_abs = m_offset + m_base + row
                    # See FP8PreActLoad guard: clamp to last valid row so the
                    # cluster_m=2 phantom-tile rows can't read past the scales
                    # buffer (the value is dropped by D predication anyway).
                    m_safe = m_abs if m_abs < m_limit else (m_limit - Int32(1))
                    n0 = n_base + col * 2
                    # Same for the N tail: a garbage scale (255 -> inf) times the
                    # zero D of an out-of-range column would make ds NaN.
                    n_lim = Int32(fp8_tensor.shape[1])
                    n0 = n0 if n0 < n_lim else (n_lim - Int32(2))
                    group_0 = n0 >> Int32(5)
                    group_1 = (n0 + Int32(1)) >> Int32(5)
                    scale_0 = _i32_as_f32(Int32(scales_tensor[m_safe, group_0]) << Int32(23))
                    scale_1 = _i32_as_f32(Int32(scales_tensor[m_safe, group_1]) << Int32(23))
                    tRS_rXY_f32x2[2 * i] = tRS_rXY_f32x2[2 * i] * scale_0
                    tRS_rXY_f32x2[2 * i + 1] = tRS_rXY_f32x2[2 * i + 1] * scale_1
        else:
            # ── Standard bf16 C path ──
            assert tRS_rC is not None
            implicit_dtype = self.implicit_dtype
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
        if const_expr(
            self.arch < 100 and not self.swiglu_clamp_value > 0.0
            and _is_quack_act(params.act_bwd_fn, "dswiglu_precise")
        ):
            _dswiglu_precise_frag(tRS_rXY_f32x2, tRS_rD_scaled, tRS_rdXY_f32x2, tRS_rOut)
        elif const_expr(self.arch < 100):
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
            pack_dtype = self.implicit_dtype
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, pack_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(pack_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut
