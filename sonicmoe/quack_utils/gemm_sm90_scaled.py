# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py

import enum
import os
import threading
import time
from pathlib import Path
from typing import Tuple, Type, Callable, Optional, Union, Literal, NamedTuple
from functools import partial
import math


import cuda.bindings.driver as cuda
import torch
import triton
import triton.language as tl
from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Int64, Int16, Float32, Float16, Boolean, const_expr
from cutlass.utils import LayoutEnum


from dataclasses import dataclass

import quack.cache_utils as cache_utils
from quack.cache_utils import EXTRA_SOURCE_DIRS, jit_cache
from quack.cute_dsl_utils import ParamsBase
from quack.gemm_tvm_ffi_utils import make_fake_scheduler_args, make_scheduler_args, make_varlen_args
from quack.tile_scheduler import (
    TileSchedulerOptions,
    TileSchedulerArguments,
    TileScheduler,
    VarlenMTileSchedulerArguments,
    VarlenMTileScheduler,
    PersistenceMode,
)
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_act import GemmActMixin
from quack.gemm_wrapper_utils import GemmTensorInfo, GemmWrapperBase
from quack.cute_dsl_utils import get_max_active_clusters, torch2cute_dtype_map
from quack.epi_ops import TileStore
from quack.cute_dsl_utils import mlir_namedtuple
from quack.varlen_utils import VarlenArguments, VarlenManager

# return PipelineStateWAdvance instead of PipelineState
from quack.pipeline import make_pipeline_state, PipelineTmaCpAsync
import quack.copy_utils as copy_utils
import quack.sm90_utils as quack_sm90_utils
from quack.rounding import RoundingMode
import quack.activation
from ._gated_epilogues import (
    GemmGatedMixin,
    GemmDGatedMixin,
    GemmGatedBlockscaledQuantMixin,
    GemmGatedPostActIsaQuantMixin,
    GemmGatedBlockscaledQuantPostActMixin,
    Sm90RawGatedQuantMixin,
    Sm90RawGatedQuant128Mixin,
    GemmDGatedFP8PreActMixin,
    Sm90DGatedDzQuantMixin,
    Sm90DGatedColQuantMixin,
    Sm90ScaledGemmRouteScatterMixin,
    _bf16_round_odd,
)


class InstrumentedCompileCache:
    """Standalone, thread-safe compile cache with lightweight telemetry."""

    def __init__(self, name: str):
        self.name = name
        self._mem = {}
        self._lock = threading.RLock()
        self._hits = self._misses = self._compiles = 0
        self._compile_seconds = 0.0

    def get(self, key, default=None):
        with self._lock:
            if key in self._mem:
                self._hits += 1
                return self._mem[key]
            self._misses += 1
            return default

    def get_or_compile(self, key, compiler):
        value = self.get(key)
        if value is not None:
            return value
        with self._lock:
            value = self._mem.get(key)
            if value is not None:
                self._hits += 1
                return value
            start = time.perf_counter()
            value = compiler()
            self._mem[key] = value
            self._compiles += 1
            self._compile_seconds += time.perf_counter() - start
            return value

    def __contains__(self, key):
        with self._lock:
            hit = key in self._mem
            self._hits += int(hit)
            self._misses += int(not hit)
            return hit

    def __getitem__(self, key):
        with self._lock:
            return self._mem[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._mem[key] = value

    def __len__(self):
        with self._lock:
            return len(self._mem)

    def clear(self):
        with self._lock:
            self._mem.clear()

    def stats(self):
        with self._lock:
            return {"name": self.name, "entries": len(self._mem), "hits": self._hits,
                    "misses": self._misses, "compiles": self._compiles,
                    "compile_seconds": self._compile_seconds}

    def __repr__(self):
        return f"InstrumentedCompileCache({self.name!r}, entries={len(self)})"

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()


class _GemmSm90RawScaledBase:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = _GemmSm90RawScaledBase(
        ...     acc_dtype=Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    arch = 90
    a_tile_local = False  # see Sm90ScaledWgradVarlenKTileLocal

    @dataclass
    class EpilogueArguments:
        pass

    EpilogueParams = ParamsBase

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        fp8_fast_accum: bool = False,
        gather_A: bool = False,
        use_clc_persistence: bool = False,
        activation_group: int = 32,
        weight_block: int = 32,
        tma_gather_A: bool = False,
        tma_gather_cp_div: int = 0,
        tma_gather_warps: int = 1,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = use_clc_persistence
        # Native Hopper raw-UE8M0 mode is configured by the scaled subclasses.
        # It always preserves E4M3 operands and scales K=32 WGMMA partials into
        # a separate FP32 master accumulator.
        self.raw_ue8m0_scaled = True
        # Software-pipelined profile-32 mainloop (see mma_scaled_pipelined).
        # Disabled: the extra 64-reg accumulator spills at tile 128x128 (4-12% slower).
        self.pipe_mma = False
        # gather_A with source-row A scales: the MMA warps read scales through
        # A_idx (set by _new_kernel; requires gather_A and not varlen-K).
        self.gather_a_scales = False
        # CTA K tile override (0 = activation_group); set by _new_kernel.
        self.k_tile = 0
        self.activation_group = int(activation_group)
        self.weight_block = int(weight_block)
        if (self.activation_group, self.weight_block) not in ((32, 32), (128, 128)):
            raise ValueError("supported SM90 profiles are 1x32+32x32 and 1x128+128x128")
        if self.use_clc_persistence:
            assert self.arch == 100
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        self.fp8_slow_accum = not fp8_fast_accum and a_dtype.width == 8
        self.gather_A = gather_A
        # TMA row gather (varlen-M only): one 2D TMA per A row (box 1 x tile_K)
        # lands in the swizzled stage, so gathered A can be multicast across
        # the cluster N dimension like a plain TMA A tile.
        self.tma_gather_A = bool(tma_gather_A) and gather_A
        if gather_A and not self.tma_gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "

        self.cluster_shape_mnk = cluster_shape_mnk
        # K dimension is deferred in _setup_attributes
        self.cta_tile_shape_mnk = (*tile_shape_mn, 1)
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        # Hybrid TMA row gather: the last tile_M // tma_gather_cp_div rows of each
        # tile go through cp.async (LSU) in 3 producer warps, the rest through
        # per-row TMA in the TMA warp, so neither path bounds the mainloop.
        self.hybrid_cp_rows = (
            tile_M // tma_gather_cp_div if self.tma_gather_A and tma_gather_cp_div else 0
        )
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")

        if not self.pingpong:
            if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
                atom_layout_m, atom_layout_n = 1, 2
            elif tile_M == 192:
                if tile_N <= 128:
                    atom_layout_m, atom_layout_n = 3, 1
                else:
                    atom_layout_m, atom_layout_n = 1, 2
            else:
                atom_layout_m = (
                    self.cta_tile_shape_mnk[0] // 64 if self.cta_tile_shape_mnk[0] < 256 else 2
                )
                atom_layout_n = 1
            assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        else:
            atom_layout_m, atom_layout_n = 1, 1
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        if self.gather_A and not self.tma_gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        if self.pingpong:
            assert self.mma_warp_groups == 2
        assert self.mma_warp_groups in [1, 2, 3]
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        self.num_epi_warps = (self.mma_warp_groups if not self.pingpong else 1) * 4
        self.cpasync_gather_A = self.gather_A and (
            not self.tma_gather_A or self.hybrid_cp_rows > 0
        )
        # TMA row issue is bound by one lane's issue rate: split the TMA rows over
        # ``tma_gather_warps`` load warps (warp 0 also loads B and arms the barrier).
        self.tma_gather_warps = int(tma_gather_warps) if self.tma_gather_A else 1
        # multi_load: several load roles per stage (load_AB_hybrid, PipelineTmaCpAsync)
        self.multi_load = self.tma_gather_A and (
            self.hybrid_cp_rows > 0 or self.tma_gather_warps > 1
        )
        self.num_ab_load_warps = 4 if self.cpasync_gather_A or self.multi_load else 1
        if self.hybrid_cp_rows > 0:
            assert self.tma_gather_warps < 4
            self.num_cp_threads = (4 - self.tma_gather_warps) * 32
        else:
            self.num_cp_threads = 128 if self.cpasync_gather_A else 0
        self.ab_load_warp_id = self.mma_warp_groups * 4

        regs_per_thread = math.prod(self.cta_tile_shape_mnk[:2]) // (
            math.prod(self.atom_layout_mnk) * self.num_threads_per_warp_group
        )
        if self.fp8_slow_accum:
            regs_per_thread *= 2
        # The raw-scaled path always keeps an independent FP32 partial and
        # master accumulator, matching the register footprint of slow FP8.
        if not self.cpasync_gather_A:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 32, 160
            else:
                heavy_register_pressure = regs_per_thread >= 208
                self.num_regs_load, self.num_regs_mma = (
                    (40, 232) if not heavy_register_pressure else (24, 240)
                )
        else:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 56, 152
            else:
                self.num_regs_load, self.num_regs_mma = (56, 224)

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self, epilogue_args: EpilogueArguments):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, self.cta_tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            self.tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(self.tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        # CTA K tile: one scale group by default; profile-32 may use a K=128
        # tile (4 scale columns per stage) to amortize pipeline and scale
        # addressing overhead. Promotion order per K=32 partial is unchanged.
        k_tile = self.k_tile if self.k_tile else self.activation_group
        assert k_tile in (32, 128) and k_tile % self.activation_group == 0
        self.cta_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1],
            k_tile,
        )
        self.scale_partials_per_tile = k_tile // 32
        self.partials_per_scale = self.activation_group // 32
        self.scale_cols_per_tile = k_tile // self.activation_group
        if self.raw_ue8m0_scaled:
            assert self.a_dtype == cutlass.Float8E4M3FN and self.b_dtype == cutlass.Float8E4M3FN
            assert mma_inst_shape_k == 32, "Hopper scaled E4M3 requires WGMMA K=32 partials"
            assert self.activation_group in (32, 128)
            assert self.weight_block == self.activation_group

        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.cta_tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )

        smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")
        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.c_dtype,
            epilogue_args,
            smem_capacity,
            self.occupancy,
        )
        self.sched_stage = 2 if self.pingpong else 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )

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
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None
        if const_expr(self.raw_ue8m0_scaled):
            assert mSFA is not None and mSFB is not None
            assert mSFA.element_type.width == 8 and mSFB.element_type.width == 8
            # Raw scale tensors are deliberately not transformed into the SM100
            # ISA-packed layout. Kernel scale views are derived directly below.
            self.sf_dtype = mSFA.element_type

        if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

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

        mA, mB, mD = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mA, mB, mD)
        ]

        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
            mA_tma = mA
            if const_expr(self.a_tile_local):
                # A = fp8 [F, total_K] stored tile-locally in a bf16 [total_K, F]
                # buffer (see Sm90DGatedColQuantMixin): element (f, k) at byte
                # (k//128*128 + f%128) * 2F + f//128 * 256 + k%128.  The host
                # passes the first F*total_K bytes as a flat [F, total_K] view.
                fb, kb = mA.shape[0] // 128, mA.shape[1] // 128
                mA_tma = cute.make_tensor(mA.iterator, cute.make_layout(
                    ((128, fb), (128, kb)), stride=((fb * 256, 256), (1, fb * 32768))))
            # Packed varlen-K wgrad owns a physical contiguous K range per expert.
            # Keep the rank-2 descriptor: VarlenManager applies the packed offset
            # with domain_offset, avoiding the ragged descriptor's wraparound stride.
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                mA_tma,
                a_smem_layout,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self.cluster_shape_mnk[1],
            )
        elif const_expr(self.tma_gather_A):
            assert varlen_m, "TMA row gather supports varlen-M only"
            tma_atom_a = self._make_tma_row_atom(mA, a_smem_layout)
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            b_smem_layout,
            (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.cpasync_gather_A):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        elif const_expr(self.hybrid_cp_rows > 0):
            self.num_tma_load_bytes += (
                (self.cta_tile_shape_mnk[0] - self.hybrid_cp_rows)
                * self.cta_tile_shape_mnk[2] * self.a_dtype.width // 8
            )

        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(
                    mD,
                    ragged_dim=0,
                    ptr_shift=True,
                )
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

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            a_idx: cute.struct.MemRange[
                Int32, self.cta_tile_shape_mnk[0] if self.tma_gather_A else 1
            ]
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
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
            mSFA,
            mSFB,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        mSFA_raw: Optional[cute.Tensor],
        mSFB_raw: Optional[cute.Tensor],
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cluster_layout_mnk: CTA layout
        :type cluster_layout_mnk: cute.Layout
        :param a_smem_layout: Shared memory layout for A
        :type a_smem_layout: cute.ComposedLayout
        :param b_smem_layout: Shared memory layout for B
        :type b_smem_layout: cute.ComposedLayout
        :param epi_smem_layout: Shared memory layout for epilogue
        :type epi_smem_layout: cute.ComposedLayout
        """

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        assert not (varlen_m and varlen_k)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)
        assert mSFA_raw is not None and mSFB_raw is not None

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch Tma desc
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        if const_expr(has_C):
            epi_pipeline = self.make_epi_pipeline(
                c_smem_layout=cute.slice_(epi_c_smem_layout, (None, None, 0)),
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=varlen_k,
            )
            sched_data = storage.sched_data.get_tensor((4, self.sched_stage))
        if const_expr(hasattr(self, "epi_cluster_init")):
            self.epi_cluster_init(epilogue_params, storage)

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # Generate smem tensor A/B
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sAIdx = None
        if const_expr(self.tma_gather_A):
            sAIdx = storage.a_idx.get_tensor(cute.make_layout(self.cta_tile_shape_mnk[0]))
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = None
        if const_expr(has_D):
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            # Only used if not varlen_m
            len_m_static=Int32(
                cute.size(mA_mkl, mode=[0])
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                is_tma_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                # Get mcast mask
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    len_m = varlen_manager.len_m(batch_idx)
                    len_k = varlen_manager.len_k(batch_idx)
                    # Local_tile partition global tensors
                    if const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                    else:
                        mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
                        if const_expr(varlen_m):
                            gAIdx = cute.local_tile(
                                mAIdx_mk, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0],)
                            )
                            # (M, K)
                            mA_mk = mA_mkl
                        else:
                            assert varlen_k
                            # (tile_K, RestK)
                            gAIdx = cute.flat_divide(mAIdx_mk, (self.cta_tile_shape_mnk[2],))
                            # (tile_M, K)
                            mA_mk = cute.local_tile(
                                mA_mkl, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0], None)
                            )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    len_m = varlen_manager.len_m(batch_idx)
                    len_k = varlen_manager.len_k(batch_idx)
                    #  TMA load A partition_S/D
                    copy_A = None
                    if const_expr(not self.gather_A):
                        copy_A, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_a,
                            cta_coord=block_in_cluster_coord_mnk[1],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                            ),
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            mcast_mask=a_mcast_mask,
                        )
                    elif const_expr(self.multi_load):
                        pass  # built per producer role in load_AB_hybrid
                    elif const_expr(self.tma_gather_A):
                        copy_A = self.tma_row_gather_get_copy_fn(
                            tma_atom_a,
                            sA.iterator,
                            sAIdx,
                            gAIdx,
                            len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                            block_in_cluster_coord_mnk[1],
                            a_mcast_mask,
                        )
                    else:
                        tiled_copy_A = self._make_gmem_tiled_copy_A(
                            mA_mkl.element_type, self.a_layout, self.num_ab_load_warps * 32
                        )
                        tidx = (
                            cute.arch.thread_idx()[0] - cute.arch.WARP_SIZE * self.ab_load_warp_id
                        )
                        thr_copy_A = tiled_copy_A.get_slice(tidx)
                        copy_A, prefetch_A = None, None
                        if const_expr(varlen_m):
                            copy_A = copy_utils.gather_m_get_copy_fn(
                                thr_copy_A,
                                mA_mk,
                                sA,
                                gAIdx,
                                limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                                limit_k=len_k,
                            )
                        else:
                            copy_A, prefetch_A = copy_utils.gather_k_get_copy_fn(
                                thr_copy_A,
                                mA_mk,
                                sA,
                                gAIdx,
                                limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                                limit_k=len_k,
                            )
                    # TMA load B partition_S/D
                    copy_B, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=cute.make_layout(
                            cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                        ),
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        mcast_mask=b_mcast_mask,
                    )
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    if const_expr(not self.cpasync_gather_A and not self.multi_load):
                        ab_producer_state = self.load_AB(
                            ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt
                        )
                    elif const_expr(self.multi_load):
                        ab_producer_state = self.load_AB_hybrid(
                            ab_pipeline,
                            ab_producer_state,
                            tma_atom_a,
                            mA_mk,
                            sA,
                            sAIdx,
                            gAIdx,
                            len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                            len_k,
                            block_in_cluster_coord_mnk[1],
                            a_mcast_mask,
                            copy_B,
                            k_tile_cnt,
                        )
                    else:
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            prefetch_A,
                            copy_B,
                            k_tile_cnt,
                            varlen_m=varlen_m,
                        )
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                if const_expr(self.multi_load and self.hybrid_cp_rows == 0):
                    # Pure multi-warp TMA: the full barrier expects one producer
                    # arrive, so only the lead warp may run the tail acquire.
                    if warp_idx == self.ab_load_warp_id:
                        ab_pipeline.producer_tail(ab_producer_state)
                else:
                    ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            # Partition global tensor for TiledMMA_A/B/C
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if const_expr(not self.pingpong) else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            # Make fragments
            acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )
            acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
            # Profile-32 non-varlen-K: double-buffered partial accumulator so the
            # WGMMA of K tile k overlaps the FP32 promotion of tile k-1.
            use_pipe_mma = const_expr(
                not varlen_k and self.scale_partials_per_tile == 1 and self.pipe_mma
                and not self.gather_a_scales
            )
            acc2 = cute.make_rmem_tensor(acc.shape, self.acc_dtype) if use_pipe_mma else None
            # Differential control: this is the vendored native FP8 mainloop
            # callable used to prove that merely carrying scale tensors is safe.
            mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]),
                                              self.cta_tile_shape_mnk[2])
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                    # TODO: do we need to check if work_tile is valid?
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                safe_batch_idx = batch_idx if k_tile_cnt > 0 else Int32(0)
                scale_m_offset = Int32(0)
                scale_k_offset = Int32(0)
                if const_expr(varlen_m):
                    scale_m_offset = varlen_manager.params.cu_seqlens_m[batch_idx]
                if const_expr(varlen_k):
                    # Packed expert starts are multiples of the wgrad token group.
                    scale_k_offset = (
                        varlen_manager.params.cu_seqlens_k[safe_batch_idx]
                        // self.activation_group
                    )
                # Early C prefetch (non-pingpong): issue the first epi_c_stage C
                # subtile TMA loads before the mainloop so their DRAM latency is
                # hidden behind the MMA instead of stalling the epilogue start.
                early_c = const_expr(has_C and not self.pingpong)
                copy_C = None
                if const_expr(early_c):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                    epi_producer_state = self.epi_prefetch_c(
                        copy_C, epi_pipeline, epi_producer_state, is_tma_warp
                    )
                epilogue_barrier = pipeline.NamedBarrier(
                    barrier_id=int(NamedBarrierGemm.Epilogue),
                    num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
                )
                # Per-row epilogue metadata loaded before the mainloop (latency hidden by MMA).
                epi_pre = None
                if const_expr(hasattr(self, "epi_tile_prefetch")):
                    tiled_copy_r2s_pre, _, _ = self.epilog_smem_store_and_partition(
                        tiled_mma, self.d_layout,
                        self.d_dtype if self.d_dtype is not None else cutlass.BFloat16, sD, tidx,
                    )
                    epi_pre = self.epi_tile_prefetch(
                        epilogue_params, tiled_copy_r2s_pre, sD, tile_coord_mnkl, varlen_manager,
                        tidx,
                    )
                if k_tile_cnt == 0:
                    acc.fill(0.0)
                elif const_expr(use_pipe_mma):
                    ab_read_state = self.mma_scaled_pipelined(
                        ab_pipeline, ab_read_state, tiled_mma, acc, acc2, acc_slow,
                        tCrA, tCrB, k_tile_cnt, warp_group_idx,
                        mSFA_raw, mSFB_raw, tile_coord_mnkl, safe_batch_idx,
                        scale_m_offset, scale_k_offset,
                        tidx % self.num_threads_per_warp_group,
                    )
                else:
                    acc.fill(0.0)
                    ab_read_state = self.mma_scaled(
                        ab_pipeline, ab_read_state, tiled_mma, acc, acc_slow,
                        tCrA, tCrB, k_tile_cnt, warp_group_idx,
                        mSFA_raw, mSFB_raw, tile_coord_mnkl, safe_batch_idx,
                        scale_m_offset, scale_k_offset, tidx, varlen_k,
                        tidx % self.num_threads_per_warp_group,
                        varlen_manager.params.mAIdx
                        if const_expr(self.gather_a_scales) else None,
                    )

                # EPILOGUE
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")

                copy_D = None
                # _epi_direct_store: the epilogue writes D from registers itself
                # (mD stays bound for the varlen-M scheduler's total_m).
                if const_expr(has_D and not getattr(self, "_epi_direct_store", False)):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )
                if const_expr(has_C and not early_c):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                # (R2S, R2S_M, R2S_N, num_epi)
                tRS_rAcc = cute.flat_divide(acc, tRS_rD.layout)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

                epi_read_state, epi_producer_state = self.epilogue(
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    self.epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tRS_rC,
                    None,  # tiled_copy_t2r, for Sm100 only
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    copy_C,
                    tile_coord_mnkl,
                    varlen_manager,
                    epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                    c_prefetched=early_c,
                    epi_pre=epi_pre,
                    sD_full=sD,
                )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # Update starting mainloop pipeline state for the next tile
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            ab_read_state.advance_iters(k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

    @cute.jit
    def load_AB(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        # These are for Sm100 blockscaled gemm
        copy_SFA: Optional[Callable] = None,
        copy_SFB: Optional[Callable] = None,
    ) -> cutlass.pipeline.PipelineState:
        blockscaled = const_expr(copy_SFA is not None)
        if const_expr(blockscaled):
            assert copy_SFB is not None
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # TMA load
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            if const_expr(copy_A is not None):
                copy_A(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            if const_expr(blockscaled):
                copy_SFA(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
                copy_SFB(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    @cute.jit
    def load_AB_gather_A(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Callable,
        prefetch_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        varlen_m: bool = True,
    ) -> cutlass.pipeline.PipelineState:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # TMA load on B and cp.async on A
        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile),)
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            # A tiny bit faster to rotate the warp that does TMA
            is_tma_warp = warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps)
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            smem_idx = ab_producer_state.index
            # A bit faster to load B first while we calculate the indices for A
            if is_tma_warp:
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out)
            # This tells mbarrier to track the completion of cp.async
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # bound checking in the K dimension on the last k_tile
        if 0 < k_tile_cnt:
            k_tile = k_tile_cnt - 1
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, pred=True),)
            is_tma_warp = warp_idx == self.ab_load_warp_id + (
                (k_tile % self.num_ab_load_warps) if const_expr(varlen_m) else 0
            )
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            smem_idx = ab_producer_state.index
            if is_tma_warp:
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out, pred=True)
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
        return ab_producer_state

    @cute.jit
    def load_AB_hybrid(
        self,
        ab_pipeline,
        ab_producer_state,
        tma_atom_a,
        mA_mk: cute.Tensor,
        sA: cute.Tensor,
        sAIdx: cute.Tensor,
        gAIdx: cute.Tensor,
        limit_m: Int32,
        len_k: Int32,
        cta_n: Int32,
        a_mcast_mask,
        copy_B: Callable,
        k_tile_cnt: Int32,
    ):
        """Load warps [0, tma_gather_warps): per-row TMA of the first rows of each A
        tile (multicast over the N cluster), warp 0 also loads B and arms the full
        barrier. The remaining load warps: the last ``hybrid_cp_rows`` rows via cp.async."""
        tile_m, tile_k = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]
        cp_rows = self.hybrid_cp_rows
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        if warp_idx < self.ab_load_warp_id + self.tma_gather_warps:
            tma_warp = warp_idx - self.ab_load_warp_id
            is_lead = tma_warp == 0
            copy_A = self.tma_row_gather_get_copy_fn(
                tma_atom_a, sA.iterator, sAIdx, gAIdx, limit_m, cta_n, a_mcast_mask, tma_warp
            )
            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_lead)
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                smem_idx = ab_producer_state.index
                if is_lead:
                    copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
                copy_A(k_tile, smem_idx, tma_bar_ptr)
                ab_producer_state.advance()
                peek_ab_empty_status = Boolean(True)
                if k_tile + 1 < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        elif const_expr(cp_rows > 0):
            n_blk = tile_m // cp_rows
            tiled_copy_A = self._make_gmem_tiled_copy_A(
                mA_mk.element_type, self.a_layout, self.num_cp_threads
            )
            tidx = cute.arch.thread_idx()[0] - cute.arch.WARP_SIZE * (
                self.ab_load_warp_id + self.tma_gather_warps
            )
            sA_cp = cute.flat_divide(sA, (cp_rows, tile_k))[None, None, n_blk - 1, 0, None]
            gAIdx_cp = cute.flat_divide(gAIdx, (cp_rows,))[None, n_blk - 1]
            copy_A = copy_utils.gather_m_get_copy_fn(
                tiled_copy_A.get_slice(tidx), mA_mk, sA_cp, gAIdx_cp,
                limit_m=limit_m - (tile_m - cp_rows), limit_k=len_k,
            )
            for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
                ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, False)
                copy_A(k_tile, ab_producer_state.index)
                ab_pipeline.producer_cpasync_commit(ab_producer_state)
                ab_producer_state.advance()
                peek_ab_empty_status = Boolean(True)
                if k_tile + 1 < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            if 0 < k_tile_cnt:
                ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, False)
                copy_A(k_tile_cnt - 1, ab_producer_state.index, pred=True)
                ab_pipeline.producer_cpasync_commit(ab_producer_state)
                ab_producer_state.advance()
        return ab_producer_state

    @cute.jit
    def mma_unscaled(
        self,
        ab_pipeline,
        ab_read_state,
        mma_fn,
        acc,
        acc_slow,
        k_tile_cnt,
        warp_group_idx,
    ):
        """Vendored GemmSm90 FP8 slow-accum mainloop used for differential checks."""
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        if const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        peek = Boolean(True)
        if 0 < k_tile_cnt:
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        zero_init = Boolean(True)
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            zero_init = Boolean(False)
            ab_read_state.advance()
            peek = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        warpgroup.wait_group(0)
        acc_slow.store(acc.load())
        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            zero_init = Boolean(True)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            warpgroup.wait_group(0)
            acc_slow.store(acc_slow.load() + acc.load())
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        for _ in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        acc.store(acc_slow.load())
        return ab_read_state

    @cute.jit
    def mma_scaled(
        self,
        ab_pipeline,
        ab_read_state,
        tiled_mma,
        acc,
        acc_slow,
        tCrA,
        tCrB,
        k_tile_cnt,
        warp_group_idx,
        mSFA_raw,
        mSFB_raw,
        tile_coord_mnkl,
        batch_idx,
        scale_m_offset,
        scale_k_offset,
        tidx,
        varlen_k,
        mma_group_tidx,
        a_scale_idx=None,
    ):
        """Accumulate native E4M3 WGMMA K=32 partials in master FP32.

        No operand conversion is performed. A profile-128 tile executes four
        K=32 WGMMA operations. Raw UE8M0 values are bit-shifted into FP32
        exponents and multiplied only after each partial has completed.
        """
        if const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        acc_slow.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek = Boolean(True)
        if 0 < k_tile_cnt:
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        # Accumulator layout per thread: ((2, 2, n_wg / 8), m_reps) with one
        # 64 x n_wg WGMMA atom per M repetition. Element i = mr * frag + j:
        #   row = wg_m_off + mr * 64 + warp * 16 + ((j // 2) % 2) * 8 + lane // 4
        #   col = chunk_col(j // 4) + (lane % 4) * 2 + j % 2
        # where chunk_col(ch) = 8 * ch (atom_n == 1) or 8 * (wg_n + atom_n * ch)
        # under the N permutation of an N-split atom layout. An 8-column chunk
        # never straddles a weight block (weight_block >= 32).
        num_acc = const_expr(cute.size(acc))
        atom_m = const_expr(self.atom_layout_mnk[0])
        atom_n = const_expr(self.atom_layout_mnk[1])
        n_wg = const_expr(self.cta_tile_shape_mnk[1] // atom_n)
        frag = const_expr(n_wg // 2)
        m_reps = const_expr(num_acc // frag)
        num_chunks = const_expr(frag // 4)
        num_a_scales = const_expr(2 * m_reps)
        wb = const_expr(self.weight_block)
        tile_n_start = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        wg_m_off = Int32(0)
        wg_n = Int32(0)
        if const_expr(not self.pingpong):
            wg_m_off = (warp_group_idx % atom_m) * (64 * m_reps)
            wg_n = warp_group_idx // atom_m
        # B scale modes:
        #   0: tile N is a multiple of weight_block (atom_n == 1): static block
        #      index per chunk, ceil(n_wg / wb) loads.
        #   1: atom_n == 1, tile N not block aligned (e.g. 192 vs 128): the tile
        #      spans a few consecutive blocks; chunk ch uses block
        #      (tile_n_start % wb + 8 * ch) // wb, resolved by compare/select.
        #   2: atom_n > 1: one load per chunk (runtime block per chunk).
        b_mode = const_expr(
            0 if atom_n == 1 and self.cta_tile_shape_mnk[1] % wb == 0
            else (1 if atom_n == 1 else 2)
        )
        num_b_scales = const_expr(
            cute.ceil_div(n_wg, wb) if b_mode == 0
            else (
                cute.ceil_div(
                    wb - math.gcd(self.cta_tile_shape_mnk[1], wb) + self.cta_tile_shape_mnk[1], wb
                )
                if b_mode == 1 else num_chunks
            )
        )
        b_off = Int32(0)
        if const_expr(b_mode == 1):
            b_off = tile_n_start % wb
        # Source-ordered A scales under gather_A: resolve this thread's source
        # rows once per tile through A_idx instead of a host-side scale gather.
        # Rows past the route total read A_idx[0], matching the old gathered
        # carrier's safe row 0; their D elements are discarded by the epilogue.
        a_rows = []
        if const_expr(a_scale_idx is not None):
            for r_pre in cutlass.range_constexpr(num_a_scales):
                m_pre = (
                    scale_m_offset
                    + tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
                    + wg_m_off
                    + (mma_group_tidx // 32) * 16
                    + (r_pre // 2) * 64
                    + (mma_group_tidx % 32) // 4
                    + (r_pre % 2) * 8
                )
                a_rows.append(a_scale_idx[m_pre if m_pre < a_scale_idx.shape[0] else Int32(0)])
        # Profile-128 (one scale group per K=128 tile): like DeepGEMM, the four
        # K=32 WGMMAs accumulate in the tensor core and the tile is promoted
        # once, instead of a wait + 64-FFMA promotion after every partial.
        tc_accum = const_expr(
            self.scale_cols_per_tile == 1
            and self.partials_per_scale == self.scale_partials_per_tile
            and self.scale_partials_per_tile > 1
        )
        # Varlen-K wgrad, 1x128 token groups: both operands carry one scale per
        # output row / column and 128-token group, stored k-group-major
        # ([groups(+pad), feature] u8). Each thread reads the A scales of its 2
        # rows (per M repetition) and the B scales of its 2 * num_chunks columns
        # (2 adjacent columns per 8-column chunk: one 16-bit load).
        vk_a_base = None
        vk_b_base = None
        if const_expr(varlen_k and tc_accum):
            assert atom_n == 1, "varlen-K 1x128 wgrad promotion assumes atom_n == 1"
            vk_lane = mma_group_tidx % 32
            vk_m0 = (
                tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
                + wg_m_off
                + (mma_group_tidx // 32) * 16
                + vk_lane // 4
            )
            vk_n0 = tile_n_start + (vk_lane % 4) * 2
            vk_a_base = (
                mSFA_raw.iterator.toint()
                + Int64(scale_k_offset) * Int64(mSFA_raw.stride[0])
                + Int64(vk_m0)
            )
            vk_b_base = (
                mSFB_raw.iterator.toint()
                + Int64(scale_k_offset) * Int64(mSFB_raw.stride[0])
                + Int64(vk_n0)
            )
            # Scales are prefetched one K tile ahead (the LDG latency would
            # otherwise sit on the wait -> promote critical path). The read of
            # the tile past the expert end lands in the next expert's rows or
            # in the host-guaranteed pad rows.
            vk_a_pre = cute.make_rmem_tensor((num_a_scales,), Float32)
            vk_b_pre = cute.make_rmem_tensor((num_chunks,), Int32)
            for r in cutlass.range_constexpr(num_a_scales):
                vk_a_pre[r] = self.load_ue8m0(vk_a_base + ((r // 2) * 64 + (r % 2) * 8))
            for ch in cutlass.range_constexpr(num_chunks):
                vk_b_pre[ch] = self.load_scale_u16(vk_b_base + 8 * ch)
        # Outside varlen-K the scale rows/columns this thread reads are fixed for
        # the whole tile; resolve their (clamped) indices once per tile.
        # Manual scale LDGs are not TMA-predicated. Phantom rows or columns in
        # skewed edge tiles therefore use a safe scale; the corresponding D
        # element is discarded by the epilogue.
        a_scale_rows = []
        b_scale_rows = []
        if const_expr(not varlen_k):
            for r in cutlass.range_constexpr(num_a_scales):
                if const_expr(a_scale_idx is not None):
                    m_row_safe = a_rows[r]
                else:
                    m_row = (
                        scale_m_offset
                        + tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
                        + wg_m_off
                        + (mma_group_tidx // 32) * 16
                        + (r // 2) * 64
                        + (mma_group_tidx % 32) // 4
                        + (r % 2) * 8
                    )
                    m_row_safe = m_row if m_row < mSFA_raw.shape[0] else Int32(0)
                a_scale_rows.append(m_row_safe)
            for c in cutlass.range_constexpr(num_b_scales):
                if const_expr(b_mode == 2):
                    b_col = (tile_n_start + 8 * (wg_n + atom_n * c)) // wb
                else:
                    b_col = tile_n_start // wb + c
                b_col_safe = b_col if b_col < mSFB_raw.shape[0] else Int32(0)
                b_scale_rows.append(b_col_safe)
            if const_expr(self.scale_cols_per_tile == 4 or tc_accum):
                # Packed scales: a K=128 tile's 4 UE8M0 columns are 4 consecutive
                # bytes of each row (host guarantees 4B-aligned rows and a
                # < 2**31 byte span), so one 32-bit load per row replaces 4 byte
                # loads. Rows are kept as Int32 byte offsets from the tensor base:
                # under the 152-160 register MMA budget ptxas rematerializes
                # hoisted 64-bit row pointers every K tile otherwise.
                # Profile 128 (tc_accum, host guarantees a < 2**31 byte span)
                # uses the same Int32 offsets with one byte per row and tile.
                a_scale_base = mSFA_raw.iterator.toint()
                b_scale_base = mSFB_raw.iterator.toint()
                for r in cutlass.range_constexpr(num_a_scales):
                    a_row_off = a_scale_rows[r] * mSFA_raw.stride[0]
                    if const_expr(cute.rank(mSFA_raw) == 3):
                        a_row_off = a_row_off + batch_idx * mSFA_raw.stride[2]
                    a_scale_rows[r] = Int32(a_row_off)
                for c in cutlass.range_constexpr(num_b_scales):
                    b_row_off = b_scale_rows[c] * mSFB_raw.stride[0]
                    if const_expr(cute.rank(mSFB_raw) == 3):
                        b_row_off = b_row_off + batch_idx * mSFB_raw.stride[2]
                    b_scale_rows[c] = Int32(b_row_off)
                if const_expr(self.scale_cols_per_tile == 4):
                    # Scale words are prefetched one K tile ahead so the LDG
                    # latency hides behind a whole tile of WGMMAs (+6 registers).
                    scale_words = cute.make_rmem_tensor((num_a_scales + num_b_scales,), Int32)
                    for r in cutlass.range_constexpr(num_a_scales):
                        scale_words[r] = self.load_scale_word(a_scale_base + Int64(a_scale_rows[r]))
                    for c in cutlass.range_constexpr(num_b_scales):
                        scale_words[num_a_scales + c] = self.load_scale_word(
                            b_scale_base + Int64(b_scale_rows[c])
                        )
            else:
                # Per-row 1D views: the row offset is computed once per tile
                # instead of a 64-bit row*stride per scale load.
                for r in cutlass.range_constexpr(num_a_scales):
                    if const_expr(cute.rank(mSFA_raw) == 3):
                        a_scale_rows[r] = mSFA_raw[a_scale_rows[r], None, batch_idx]
                    else:
                        a_scale_rows[r] = mSFA_raw[a_scale_rows[r], None]
                for c in cutlass.range_constexpr(num_b_scales):
                    if const_expr(cute.rank(mSFB_raw) == 3):
                        b_scale_rows[c] = mSFB_raw[b_scale_rows[c], None, batch_idx]
                    else:
                        b_scale_rows[c] = mSFB_raw[b_scale_rows[c], None]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            if const_expr(tc_accum):
                warpgroup.fence()
                for inst_tc in cutlass.range_constexpr(self.scale_partials_per_tile):
                    mma_atom_tc = cute.make_mma_atom(tiled_mma.op)
                    mma_atom_tc.set(warpgroup.Field.ACCUMULATE, inst_tc != 0)
                    cute.gemm(
                        mma_atom_tc,
                        acc,
                        tCrA[None, None, inst_tc, ab_read_state.index],
                        tCrB[None, None, inst_tc, ab_read_state.index],
                        acc,
                    )
                warpgroup.commit_group()
            if const_expr(tc_accum and varlen_k):
                # This tile's scales were loaded one tile ahead; issue the next
                # tile's loads behind the in-flight WGMMAs.
                vk_a = [vk_a_pre[r] for r in range(num_a_scales)]
                vk_b_words = [vk_b_pre[ch] for ch in range(num_chunks)]
                vk_a_k = vk_a_base + Int64(k_tile + 1) * Int64(mSFA_raw.stride[0])
                vk_b_k = vk_b_base + Int64(k_tile + 1) * Int64(mSFB_raw.stride[0])
                for r in cutlass.range_constexpr(num_a_scales):
                    vk_a_pre[r] = self.load_ue8m0(vk_a_k + ((r // 2) * 64 + (r % 2) * 8))
                for ch in cutlass.range_constexpr(num_chunks):
                    vk_b_pre[ch] = self.load_scale_u16(vk_b_k + 8 * ch)
                warpgroup.wait_group(0)
                ab_pipeline.consumer_release(ab_release_state)
                vk_b = [
                    [self.decode_ue8m0_byte(vk_b_words[ch], 0),
                     self.decode_ue8m0_byte(vk_b_words[ch], 1)]
                    for ch in range(num_chunks)
                ]
                for mr in cutlass.range_constexpr(m_reps):
                    for ch in cutlass.range_constexpr(num_chunks):
                        for e in cutlass.range_constexpr(4):
                            i = mr * frag + ch * 4 + e
                            acc_slow[i] = acc_slow[i] + acc[i] * (
                                vk_a[(e // 2) + mr * 2] * vk_b[ch][e % 2]
                            )
            if const_expr(tc_accum and not varlen_k):
                a_k_off = k_tile * mSFA_raw.stride[1]
                b_k_off = k_tile * mSFB_raw.stride[1]
                a_scales_tc = [
                    self.load_ue8m0(a_scale_base + Int64(a_scale_rows[r] + a_k_off))
                    for r in range(num_a_scales)
                ]
                b_scales_tc = [
                    self.load_ue8m0(b_scale_base + Int64(b_scale_rows[c] + b_k_off))
                    for c in range(num_b_scales)
                ]
                warpgroup.wait_group(0)
                # The smem stage is no longer read: hand it back to the TMA
                # producer before the promotion FFMAs (DeepGEMM order).
                ab_pipeline.consumer_release(ab_release_state)
                # a*b products once per tile (DeepGEMM scale_0_0/0_1/...), then
                # one FFMA per element. For tiles straddling weight blocks the
                # runtime block offset dispatches to a compile-time chunk->block
                # map (DeepGEMM dispatch_num_former_iters) instead of per-chunk
                # compare/select. Same operands and order: bit-identical.
                prods = [
                    [a_scales_tc[a] * b_scales_tc[c] for c in range(num_b_scales)]
                    for a in range(num_a_scales)
                ]
                if const_expr(b_mode == 1):
                    g_off = const_expr(math.gcd(self.cta_tile_shape_mnk[1], wb))
                    for v_i in cutlass.range_constexpr(wb // g_off):
                        if b_off == v_i * g_off:
                            self._tc_promote(
                                acc_slow, acc, prods, m_reps, frag, num_chunks,
                                [(v_i * g_off + ch * 8) // wb for ch in range(num_chunks)],
                            )
                else:
                    self._tc_promote(
                        acc_slow, acc, prods, m_reps, frag, num_chunks,
                        [((ch * 8) // wb if b_mode == 0 else ch) for ch in range(num_chunks)],
                    )
            for inst in cutlass.range_constexpr(0 if tc_accum else self.scale_partials_per_tile):
                # ACCUMULATE=False ignores the old accumulator; no zero fill
                # (it materializes as 64 MOVs per partial).
                warpgroup.fence()
                mma_atom = cute.make_mma_atom(tiled_mma.op)
                mma_atom.set(warpgroup.Field.ACCUMULATE, False)
                cute.gemm(
                    mma_atom,
                    acc,
                    tCrA[None, None, inst, ab_read_state.index],
                    tCrB[None, None, inst, ab_read_state.index],
                    acc,
                )
                warpgroup.commit_group()
                if const_expr(varlen_k):
                    # Wgrad: every K=32 partial has its own scale column
                    # (scale_k_offset is expert_start/32), and A/B scales are
                    # per output row/column, so each element reads its own pair.
                    warpgroup.wait_group(0)
                    scale_a_k = scale_k_offset + k_tile * self.scale_partials_per_tile + inst
                    scale_a_k = (
                        scale_a_k if scale_a_k < mSFA_raw.shape[1] else mSFA_raw.shape[1] - 1
                    )
                    scale_b_k = (
                        scale_a_k if scale_a_k < mSFB_raw.shape[1] else mSFB_raw.shape[1] - 1
                    )
                    for i_hi in cutlass.range_constexpr(num_acc // 8):
                        for i_lo in cutlass.range_constexpr(8):
                            i = i_hi * 8 + i_lo
                            row_in_atom = (
                                (i // 64) * 8 + (mma_group_tidx % 32) // 4 + ((i // 2) % 2) * 8
                            )
                            col_in_atom = (
                                (i % 2) + ((i // 4) % 2) * 8 + (mma_group_tidx % 4) * 2
                            )
                            m_abs = (
                                scale_m_offset
                                + tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
                                + warp_group_idx * (64 // self.atom_layout_mnk[1])
                                + (mma_group_tidx // 32) * 16
                                + row_in_atom
                            )
                            m_safe = m_abs if m_abs < mSFA_raw.shape[0] else Int32(0)
                            # The second WGMMA operand is the logical output-N
                            # scale; its accumulator coordinate follows the
                            # transposed MMA value layout.
                            b_coord = (
                                tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
                                + col_in_atom
                                + (i // 8) * 16
                            )
                            b_safe = b_coord if b_coord < mSFB_raw.shape[0] else Int32(0)
                            scale = self.decode_ue8m0(
                                mSFA_raw[m_safe, scale_a_k]
                            ) * self.decode_ue8m0(mSFB_raw[b_safe, scale_b_k])
                            acc_slow[i] = acc_slow[i] + acc[i] * scale
                else:
                    # The K=32 partials of one scale group (all 4 for profile-128)
                    # share a scale column; a K tile spans scale_cols_per_tile
                    # columns.
                    # No clamp: outside varlen-K, K % k_tile == 0 and both scale
                    # tensors span >= K / group columns (host-checked in
                    # _run_scaled), so every column index is in range.
                    if const_expr(self.scale_cols_per_tile == 4):
                        if const_expr(inst == 0):
                            a_words = [scale_words[r] for r in range(num_a_scales)]
                            b_words = [scale_words[num_a_scales + c] for c in range(num_b_scales)]
                            # The last tile re-reads its own word: no read past the span.
                            k_next = k_tile + 1 if k_tile + 1 < k_tile_cnt else k_tile
                            k_byte = k_next * 4
                            for r in cutlass.range_constexpr(num_a_scales):
                                scale_words[r] = self.load_scale_word(
                                    a_scale_base + Int64(a_scale_rows[r] + k_byte)
                                )
                            for c in cutlass.range_constexpr(num_b_scales):
                                scale_words[num_a_scales + c] = self.load_scale_word(
                                    b_scale_base + Int64(b_scale_rows[c] + k_byte)
                                )
                        a_scales = [self.decode_ue8m0_byte(a_words[r], inst) for r in range(num_a_scales)]
                        b_scales = [self.decode_ue8m0_byte(b_words[c], inst) for c in range(num_b_scales)]
                    elif const_expr(inst % self.partials_per_scale == 0):
                        scale_k = k_tile * self.scale_cols_per_tile + inst // self.partials_per_scale
                        a_scales = [self.decode_ue8m0(a_scale_rows[r][scale_k]) for r in range(num_a_scales)]
                        b_scales = [self.decode_ue8m0(b_scale_rows[c][scale_k]) for c in range(num_b_scales)]
                    warpgroup.wait_group(0)
                    if const_expr(inst == self.scale_partials_per_tile - 1):
                        # Last partial done: release the smem stage before its
                        # promotion so the producer can refill it earlier.
                        ab_pipeline.consumer_release(ab_release_state)
                    # Nested static loops: a flat 64-iteration range_constexpr triggers
                    # DSLOptimizationWarning, and cutlass.range(..., unroll_full=True)
                    # measured slower here.
                    for mr in cutlass.range_constexpr(m_reps):
                        for ch in cutlass.range_constexpr(num_chunks):
                            if const_expr(b_mode == 0):
                                b_scale = b_scales[(ch * 8) // wb]
                            elif const_expr(b_mode == 1):
                                b_scale = b_scales[0]
                                for c in cutlass.range_constexpr(1, num_b_scales):
                                    b_scale = b_scales[c] if b_off + ch * 8 >= c * wb else b_scale
                            else:
                                b_scale = b_scales[ch]
                            for e in cutlass.range_constexpr(4):
                                i = mr * frag + ch * 4 + e
                                a_scale = a_scales[(e // 2) + mr * 2]
                                acc_slow[i] = acc_slow[i] + acc[i] * (a_scale * b_scale)
            if const_expr(varlen_k and not tc_accum):
                ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        acc.store(acc_slow.load())
        return ab_read_state

    @staticmethod
    def _tc_promote(acc_slow, acc, prods, m_reps, frag, num_chunks, chunk_b):
        """acc_slow += acc * prods[a][chunk_b[ch]] (trace-time unrolled, no control flow)."""
        for mr in range(m_reps):
            for ch in range(num_chunks):
                for e in range(4):
                    i = mr * frag + ch * 4 + e
                    acc_slow[i] = acc_slow[i] + acc[i] * prods[(e // 2) + mr * 2][chunk_b[ch]]

    def _pipe_scale_counts(self, num_acc):
        b_block_aligned = (
            self.atom_layout_mnk[1] == 1 and self.cta_tile_shape_mnk[1] % self.weight_block == 0
        )
        num_a = 2 * ((num_acc + 63) // 64)
        num_b = (
            ((num_acc // 8) * 16 + self.weight_block - 1) // self.weight_block
            if b_block_aligned
            else num_acc // 8
        )
        return b_block_aligned, num_a, num_b

    @cute.jit
    def _pipe_issue(self, tiled_mma, acc, tCrA, tCrB, stage_idx):
        """Issue one K=32 WGMMA partial into ``acc`` (overwrite, no zero-fill needed)."""
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, False)
        cute.gemm(mma_atom, acc, tCrA[None, None, 0, stage_idx], tCrB[None, None, 0, stage_idx], acc)
        warpgroup.commit_group()

    @cute.jit
    def _pipe_load_scales(
        self, sc, k_tile, mSFA_raw, mSFB_raw, tile_coord_mnkl, batch_idx,
        scale_m_offset, scale_k_offset, warp_group_idx, mma_group_tidx, num_acc: cutlass.Constexpr,
    ):
        """Decode the A/B UE8M0 scales of K tile ``k_tile`` into ``sc`` (a then b)."""
        b_block_aligned, num_a_scales, num_b_scales = self._pipe_scale_counts(num_acc)
        scale_a_k = scale_k_offset + k_tile
        scale_a_k = scale_a_k if scale_a_k < mSFA_raw.shape[1] else mSFA_raw.shape[1] - 1
        scale_b_k = scale_a_k if scale_a_k < mSFB_raw.shape[1] else mSFB_raw.shape[1] - 1
        for r in cutlass.range_constexpr(num_a_scales):
            m_abs = (
                scale_m_offset
                + tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
                + warp_group_idx * (64 // self.atom_layout_mnk[1])
                + (mma_group_tidx // 32) * 16
                + (r // 2) * 8
                + (mma_group_tidx % 32) // 4
                + (r % 2) * 8
            )
            m_safe = m_abs if m_abs < mSFA_raw.shape[0] else Int32(0)
            if const_expr(cute.rank(mSFA_raw) == 3):
                sc[r] = self.decode_ue8m0(mSFA_raw[m_safe, scale_a_k, batch_idx])
            else:
                sc[r] = self.decode_ue8m0(mSFA_raw[m_safe, scale_a_k])
        for c in cutlass.range_constexpr(num_b_scales):
            if const_expr(b_block_aligned):
                b_coord = (tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]) // self.weight_block + c
            else:
                b_coord = (
                    tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
                    + warp_group_idx * (64 if self.atom_layout_mnk[1] > 1 else 0)
                    + (mma_group_tidx % 4) * 2
                    + c * 16
                ) // self.weight_block
            b_safe = b_coord if b_coord < mSFB_raw.shape[0] else Int32(0)
            if const_expr(cute.rank(mSFB_raw) == 3):
                sc[num_a_scales + c] = self.decode_ue8m0(mSFB_raw[b_safe, scale_b_k, batch_idx])
            else:
                sc[num_a_scales + c] = self.decode_ue8m0(mSFB_raw[b_safe, scale_b_k])

    @cute.jit
    def _pipe_promote(self, acc_slow, acc, sc, num_acc: cutlass.Constexpr):
        """acc_slow += acc * (a_scale * b_scale), bit-identical to ``mma_scaled``."""
        b_block_aligned, num_a_scales, num_b_scales = self._pipe_scale_counts(num_acc)
        for i_hi in cutlass.range_constexpr(num_acc // 8):
            for i_lo in cutlass.range_constexpr(8):
                i = i_hi * 8 + i_lo
                a_scale = sc[(i // 2) % 2 + (i // 64) * 2]
                if const_expr(b_block_aligned):
                    b_scale = sc[num_a_scales + (i // 8) * 16 // self.weight_block]
                else:
                    b_scale = sc[num_a_scales + i // 8]
                acc_slow[i] = acc_slow[i] + acc[i] * (a_scale * b_scale)

    @cute.jit
    def mma_scaled_pipelined(
        self, ab_pipeline, ab_read_state, tiled_mma, acc, acc2, acc_slow, tCrA, tCrB,
        k_tile_cnt, warp_group_idx, mSFA_raw, mSFB_raw, tile_coord_mnkl, batch_idx,
        scale_m_offset, scale_k_offset, mma_group_tidx,
    ):
        """Software-pipelined profile-32 mainloop (varlen-M / dense, K tile == one K=32 partial).

        Two partial accumulators alternate: WGMMA of K tile k is in flight while the
        FP32 promotion of tile k-1 runs on the FMA pipe, so the tensor core is no longer
        idle during scaling. Scale LDGs for tile k are issued behind its WGMMA and
        consumed one step later. Arithmetic (order and operands) equals ``mma_scaled``.
        Requires k_tile_cnt >= 1.
        """
        num_acc = const_expr(cute.size(acc))
        _, num_a_scales, num_b_scales = self._pipe_scale_counts(num_acc)
        if const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        acc_slow.fill(0.0)
        sc0 = cute.make_rmem_tensor((num_a_scales + num_b_scales,), Float32)
        sc1 = cute.make_rmem_tensor((num_a_scales + num_b_scales,), Float32)
        ab_release_state = ab_read_state.clone()
        sargs = (mSFA_raw, mSFB_raw, tile_coord_mnkl, batch_idx, scale_m_offset,
                 scale_k_offset, warp_group_idx, mma_group_tidx, num_acc)
        peek = ab_pipeline.consumer_try_wait(ab_read_state)
        # Prologue: K tile 0 -> acc / sc0.
        ab_pipeline.consumer_wait(ab_read_state, peek)
        self._pipe_issue(tiled_mma, acc, tCrA, tCrB, ab_read_state.index)
        self._pipe_load_scales(sc0, Int32(0), *sargs)
        ab_read_state.advance()
        peek = Boolean(True)
        if 1 < k_tile_cnt:
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        num_pairs = (k_tile_cnt - 1) // 2
        for kp in cutlass.range(num_pairs, unroll=1):
            k = 1 + 2 * kp
            # K tile k -> acc2 / sc1 while promoting tile k-1 from acc / sc0.
            ab_pipeline.consumer_wait(ab_read_state, peek)
            self._pipe_issue(tiled_mma, acc2, tCrA, tCrB, ab_read_state.index)
            self._pipe_load_scales(sc1, k, *sargs)
            ab_read_state.advance()
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
            warpgroup.wait_group(1)
            self._pipe_promote(acc_slow, acc, sc0, num_acc)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            # K tile k+1 -> acc / sc0 while promoting tile k from acc2 / sc1.
            ab_pipeline.consumer_wait(ab_read_state, peek)
            self._pipe_issue(tiled_mma, acc, tCrA, tCrB, ab_read_state.index)
            self._pipe_load_scales(sc0, k + 1, *sargs)
            ab_read_state.advance()
            peek = Boolean(True)
            if k + 2 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
            warpgroup.wait_group(1)
            self._pipe_promote(acc_slow, acc2, sc1, num_acc)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        # Even k_tile_cnt: one K tile left (k_tile_cnt-1) -> acc2, then drain both.
        rem = k_tile_cnt - 1 - 2 * num_pairs
        for _ in cutlass.range(rem, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek)
            self._pipe_issue(tiled_mma, acc2, tCrA, tCrB, ab_read_state.index)
            self._pipe_load_scales(sc1, k_tile_cnt - 1, *sargs)
            ab_read_state.advance()
            warpgroup.wait_group(1)
            self._pipe_promote(acc_slow, acc, sc0, num_acc)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            warpgroup.wait_group(0)
            self._pipe_promote(acc_slow, acc2, sc1, num_acc)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        # Odd k_tile_cnt: last tile is in acc / sc0.
        for _ in cutlass.range(1 - rem, unroll=1):
            warpgroup.wait_group(0)
            self._pipe_promote(acc_slow, acc, sc0, num_acc)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        acc.store(acc_slow.load())
        return ab_read_state

    @staticmethod
    @cute.jit
    def decode_ue8m0(raw):
        """Decode raw UE8M0 as exact FP32 2**(raw-127)."""
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T
        raw_i8 = llvm.bitcast(T.i8(), raw.ir_value())
        raw_i32 = llvm.zext(T.i32(), raw_i8)
        bits = llvm.inline_asm(
            T.i32(), [raw_i32], "shl.b32 $0, $1, 23;", "=r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return Float32(llvm.bitcast(T.f32(), bits))

    @staticmethod
    @cute.jit
    def load_scale_word(addr):
        """One 32-bit read-only load of 4 consecutive UE8M0 scale bytes (4B-aligned)."""
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T
        return Int32(llvm.inline_asm(
            T.i32(), [Int64(addr).ir_value()], "ld.global.nc.b32 $0, [$1];", "=r,l",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))

    @staticmethod
    @cute.jit
    def load_scale_u16(addr):
        """16-bit read-only load of 2 consecutive UE8M0 scale bytes (2B-aligned), zero-extended."""
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T
        return Int32(llvm.inline_asm(
            T.i32(), [Int64(addr).ir_value()],
            "{\n\t.reg .u16 h;\n\tld.global.nc.u16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))

    @staticmethod
    @cute.jit
    def load_ue8m0(addr):
        """Read-only byte load of one UE8M0 scale, decoded like decode_ue8m0 (raw << 23)."""
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T
        bits = llvm.inline_asm(
            T.i32(), [Int64(addr).ir_value()],
            "{\n\t.reg .u32 r;\n\tld.global.nc.u8 r, [$1];\n\tshl.b32 $0, r, 23;\n\t}",
            "=r,l", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return Float32(llvm.bitcast(T.f32(), bits))

    @staticmethod
    @cute.jit
    def decode_ue8m0_byte(word, byte: cutlass.Constexpr[int]):
        """decode_ue8m0 of byte ``byte`` of a packed scale word (same bits: raw << 23)."""
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T
        shift = 23 - 8 * byte
        op = f"shl.b32 $0, $1, {shift};" if shift >= 0 else f"shr.b32 $0, $1, {-shift};"
        bits = llvm.inline_asm(
            T.i32(), [Int32(word).ir_value()], op + "\n\tand.b32 $0, $0, 0x7f800000;",
            "=r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return Float32(llvm.bitcast(T.f32(), bits))

    @cute.jit
    def epi_prefetch_c(self, copy_C, epi_pipeline, epi_producer_state, is_tma_warp):
        """Issue the TMA loads of the first epi_c_stage C subtiles (advances the state in place)."""
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), self.epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        for epi_idx in cutlass.range(min(cute.size(epi_tile_shape), self.epi_c_stage), unroll=1):
            gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
            if is_tma_warp:
                epi_pipeline.producer_acquire(epi_producer_state)
                copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                epi_pipeline.producer_commit(epi_producer_state)
            epi_producer_state.advance()
        return epi_producer_state

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
        c_prefetched: cutlass.Constexpr[bool] = False,
        epi_pre=None,
        sD_full=None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)

        # Setup postact output (returns None for default epilogue, context tuple for Act)
        postact_ctx = self.epi_setup_postact(
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
        )

        if const_expr(copy_C is not None and not c_prefetched):
            epi_producer_state = self.epi_prefetch_c(
                copy_C, epi_pipeline, epi_producer_state, is_tma_warp
            )

        if const_expr(epi_pre is not None):
            # Per-tile state of the prefetched rows, hoisted out of the subtile loop.
            epi_pre = self.epi_tile_prefetch_load(
                params, epi_pre, tiled_copy_r2s, tile_coord_mnkl, varlen_manager, tidx
            )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # The global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_idx)
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, gmem_coord)
            if const_expr(epi_pre is not None):
                epi_loop_tensors["epi_pre"] = epi_pre
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                gmem_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord_C, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()
            tRS_rPostAct = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
            # Convert and store postact if this epilogue produces one
            if const_expr(postact_ctx is not None):
                tRS_rPostAct_out = self.epi_convert_postact(
                    tRS_rPostAct,
                    epi_loop_tensors["sr_seed"],
                    tidx,
                    tile_coord_mnkl,
                    num_prev_subtiles,
                    epi_idx,
                )
            # Direct-store epilogue without postact: nothing goes through smem/TMA.
            smem_store = const_expr(
                not getattr(self, "_epi_direct_store", False) or has_D
                or postact_ctx is not None
            )
            if const_expr(smem_store):
                if is_tma_warp:
                    epi_store_pipeline.producer_acquire()
                epilogue_barrier.arrive_and_wait()
            # Copy from D registers to shared memory
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if const_expr(has_D):
                if const_expr(
                    self.rounding_mode == RoundingMode.RS
                    and self.acc_dtype == cutlass.Float32
                    and self.d_dtype == cutlass.BFloat16
                ):
                    seed = epi_loop_tensors["sr_seed"] + (
                        tile_coord_mnkl[0] * 65537
                        + tile_coord_mnkl[1] * 257
                        + tile_coord_mnkl[3] * 17
                        + (num_prev_subtiles + epi_idx) * 7
                    )
                    copy_utils.sr_cvt_copy(
                        tiled_copy_r2s,
                        tRS_rD,
                        tRS_sD[None, None, None, epi_buffer],
                        seed,
                        tidx,
                    )
                else:
                    copy_utils.cvt_copy(
                        tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer]
                    )
            # Copy postact from registers to shared memory
            if const_expr(postact_ctx is not None):
                tiled_copy_postact_r2s, tRS_sPostAct, copy_postact = postact_ctx
                cute.copy(
                    tiled_copy_postact_r2s,
                    tiled_copy_postact_r2s.retile(tRS_rPostAct_out),
                    tRS_sPostAct[None, None, None, epi_buffer],
                )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            if const_expr(smem_store):
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()
                # Copy from shared memory to global memory
                if is_tma_warp:
                    if const_expr(has_D and not getattr(self, "skip_d_store", False)):
                        copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
                    if const_expr(postact_ctx is not None
                                  and not getattr(self, "skip_postact_store", False)):
                        copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
                    epi_store_pipeline.producer_commit()
                if const_expr(hasattr(self, "epi_after_smem_store")):
                    # The staged subtile stays valid until the next subtile's barrier.
                    self.epi_after_smem_store(
                        params, epi_smem_tensors, sD_full, epi_buffer, gmem_coord,
                        tile_coord_mnkl, varlen_manager, tidx, epi_idx, epi_tile_num,
                    )

        self.epi_end(
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        return epi_read_state, epi_producer_state

    def get_scheduler_class(self, varlen_m: bool = False):
        """Return the scheduler class to use. Override in subclasses for custom schedulers."""
        return TileScheduler if not varlen_m else VarlenMTileScheduler

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        scheduler_args,
        varlen_args,
        epilogue_args,
    ):
        """Create scheduler arguments. Override in subclasses for custom schedulers."""
        if const_expr(not self.is_persistent):
            persistence_mode = PersistenceMode.NONE
        else:
            if const_expr(self.arch >= 100 and self.use_clc_persistence):
                persistence_mode = PersistenceMode.CLC
            elif const_expr(scheduler_args.tile_count_semaphore is not None):
                persistence_mode = PersistenceMode.DYNAMIC
            else:
                persistence_mode = PersistenceMode.STATIC
        if const_expr(varlen_args.mCuSeqlensM is None):
            num_problems = (
                mD.shape[2]
                if mD is not None
                else (
                    mB.shape[2]
                    if varlen_args.mCuSeqlensK is None
                    else varlen_args.mCuSeqlensK.shape[0] - 1
                )
            )
            problem_shape_ntile_mnl = (
                cute.ceil_div(mA.shape[0], self.cta_tile_shape_mnk[0]),
                cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
                num_problems,
            )
            tile_sched_args = TileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                batch_idx_permute=scheduler_args.batch_idx_permute,
                persistence_mode=persistence_mode,
            )
        else:
            assert (mD is not None) or (epilogue_args.mPostAct is not None) or (not self.gather_A)
            problem_shape_ntile_mnl = (
                None,
                cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
                varlen_args.mCuSeqlensM.shape[0] - 1,
            )
            tile_sched_args = VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                total_m=mD.shape[0] if mD is not None else varlen_args.mAIdx.shape[0],
                cu_seqlens_m=varlen_args.mCuSeqlensM,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                tile_shape_mn=self.cta_tile_shape_mnk[:2],
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                persistence_mode=persistence_mode,
            )
        return tile_sched_args

    @cute.jit
    def epi_load_acc_subtile(self, tRS_rAcc: cute.Tensor, tRS_rD: cute.Tensor, epi_idx: int):
        cute.autovec_copy(tRS_rAcc[None, None, None, epi_idx], tRS_rD)

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_begin_loop(
        self, params: EpilogueParams, epi_tensors: Tuple[cute.Tensor, ...], epi_coord: cute.Coord
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        return None

    def epi_visit_acc(
        self,
        params: EpilogueParams,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> None:
        pass

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager,
        tidx,
    ) -> None:
        pass

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        return self.EpilogueParams()

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        """Subclasses can override this"""
        return []

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: Optional[EpilogueArguments],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        return 0

    def epi_get_smem_struct(self, params: EpilogueParams):
        return cute.struct.MemRange[Int32, 0]  # Dummy struct

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        return tuple()

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: Literal["mma", "epi"]):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: Literal["mma", "epi"]):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def epilog_smem_copy_atom(self, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
        copy_atom_C = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                self.d_layout.is_m_major_c() if self.d_layout is not None else False,
                num_matrices=4 if self.epi_tile[1] % 16 == 0 else 2,
            ),
            Float16,  # this is just to get the right source layout
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        return tiled_copy_C_atom

    def epilog_smem_store_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        d_layout: Optional[LayoutEnum],
        dtype: Type[cutlass.Numeric],
        sD: Optional[cute.Tensor],
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        if d_layout is None:
            d_layout = LayoutEnum.ROW_MAJOR
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # Doesn't work with tile_N % 8 == 0 but tile_n % 16 != since this always
        # get st.matrix with num_matrices=4
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            d_layout, elem_ty_d=dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        sD_shape = sD.shape[:2] if sD is not None else self.epi_tile
        tRS_rD_shape = thr_copy_r2s.partition_S(cute.make_identity_tensor(sD_shape)).shape
        tRS_rD = cute.make_rmem_tensor(tRS_rD_shape, self.acc_dtype)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        c_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        copy_atom_s2r = copy_utils.sm90_get_smem_load_op(c_layout, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_C_atom)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_rmem_tensor(tRS_rD_layout, dtype)
        tSR_rC = thr_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    def epilog_gmem_copy_and_partition(
        self,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # (bM, bN)
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        is_s2g = isinstance(
            atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp)
        )
        src_tensor, dst_tensor = (
            (sD, tDgD_for_tma_partition) if is_s2g else (tDgD_for_tma_partition, sD)
        )
        return copy_utils.tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=src_tensor,
            dst_tensor=dst_tensor,
        )

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
    ):
        # Threads/warps participating in this pipeline
        producer_cnt = 1 + self.num_cp_threads
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        pipeline_cls = (
            PipelineTmaCpAsync if self.cpasync_gather_A or self.multi_load
            else pipeline.PipelineTmaAsync
        )
        return pipeline_cls.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_epi_pipeline(
        self, c_smem_layout: cute.Layout | cute.ComposedLayout, epi_pipeline_mbar_ptr: cute.Pointer
    ):
        # Threads/warps participating in this pipeline
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will contribute 1 to the arrive count
        consumer_arrive_cnt = self.num_epi_warps
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        tma_copy_c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=epi_pipeline_mbar_ptr,
            num_stages=self.epi_c_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            tx_count=tma_copy_c_bytes,
            defer_sync=True,
        )

    def make_epi_store_pipeline(self):
        # Threads/warps participating in tma store pipeline
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_epi_threads)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_store_producer_group
        )

    def make_sched_pipeline(
        self, cluster_layout_mnk: cute.Layout, sched_pipeline_mbar_ptr: cute.Pointer, varlen_k: bool
    ):
        # Threads/warps participating in this pipeline
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp will contribute 1 to the arrive count
        # If pingpong and varlen_k, then all 8 mma warps will participate in the scheduler barrier
        # at each round. If pingpong and not varlen_k, then only 4 mma warp will participate.
        consumer_arrive_cnt = (
            (self.mma_warp_groups if not (self.pingpong and not varlen_k) else 1) * 4
            + self.num_ab_load_warps
        ) * cluster_size
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
        )

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        epilogue_args: EpilogueArguments,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: Tuple[int, int]
        """

        epi_stage = 4 if epi_tile[1] <= 16 else 2
        d_bytes_per_stage = cute.size(epi_tile) * d_dtype.width // 8 if d_dtype is not None else 0
        epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
            epilogue_args, cta_tile_shape_mnk, epi_tile
        )
        epi_bytes = epi_bytes_per_stage * epi_stage
        epi_c_stage = 0 if c_dtype is None else (4 if epi_tile[1] <= 16 else 2)
        if c_dtype is not None:
            epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        # Keep one KiB below the architectural dynamic-smem ceiling. The copied
        # scheduler storage is 1024-byte aligned and is not fully represented by
        # the upstream stage-count heuristic.
        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes - 1024
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if epi_bytes_per_stage > 0:
            epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // epi_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if cta_tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        elif cta_tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = 64 if element_type is not None and element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: LayoutEnum,
        ab_stage: int,
        d_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: LayoutEnum,
        epi_stage: int,
        c_dtype: Optional[Type[cutlass.Numeric]],
        c_layout: Optional[LayoutEnum],
        epi_c_stage: int,
    ) -> Tuple[
        cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, Optional[cute.ComposedLayout]
    ]:
        """Create shared memory layouts for A, B, and C tensors.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param d_dtype: Data type for output matrix D
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum for the output matrix C
        :type d_layout: LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        a_major_mode_size = cta_tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))

        b_major_mode_size = cta_tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        epi_smem_layout_staged = None
        if d_dtype is not None:
            epi_smem_layout_staged = quack_sm90_utils.make_smem_layout_epi(
                d_dtype, d_layout, epi_tile, epi_stage
            )

        epi_c_smem_layout_staged = None
        if c_dtype is not None:
            assert c_layout is not None
            epi_c_smem_layout_staged = quack_sm90_utils.make_smem_layout_epi(
                c_dtype, c_layout, epi_tile, epi_c_stage
            )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_c_smem_layout_staged,
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C.

        :param tensor_d: Output tensor D
        :type tensor_d: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        assert op_type in ["load", "store", "add"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if op_type == "load"
            else cpasync.CopyBulkTensorTileS2GOp()
            if op_type == "store"
            else cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD)
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout: Shared memory layout for the tensor
        :type smem_layout: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    def _make_tma_row_atom(self, mA: cute.Tensor, a_smem_layout: cute.ComposedLayout):
        """TMA atom for one K-major A row (box 1 x tile_K) of the source tensor.

        tile_K bytes == the 128B swizzle width, so each row of a stage is a
        contiguous 128B line and the hardware swizzle (from smem address bits)
        matches the stage layout when a row is written at ``base + row * 128``.
        """
        tile_m, tile_k = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]
        assert self.a_dtype.width == 8 and tile_k == 128
        assert self.a_layout == LayoutEnum.ROW_MAJOR, "TMA row gather needs K-major A"
        tma_rows = tile_m - self.hybrid_cp_rows
        split = self.num_mcast_ctas_a * self.tma_gather_warps
        assert tma_rows % split == 0 and (tma_rows // split) % 8 == 0, (tma_rows, split)
        assert cute.cosize(self.a_smem_layout_staged) == tile_m * tile_k * self.ab_stage
        row_layout = cute.make_composed_layout(
            a_smem_layout.inner, 0, cute.make_layout((1, tile_k), stride=(tile_k, 1))
        )
        tma_atom, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mA, row_layout, (1, tile_k)
        )
        return tma_atom

    @staticmethod
    @cute.jit
    def tma_load_row(desc_addr, dst_addr, mbar_addr, col, row, mcast_mask):
        """2D tile TMA of one A row; multicast to ``mcast_mask`` CTAs when given."""
        from cutlass._mlir.dialects import llvm
        args = [Int32(dst_addr).ir_value(), Int64(desc_addr).ir_value(),
                Int32(col).ir_value(), Int32(row).ir_value(), Int32(mbar_addr).ir_value()]
        if const_expr(mcast_mask is None):
            ptx = ("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                   " [$0], [$1, {$2, $3}], [$4];")
            cons = "r,l,r,r,r"
        else:
            ptx = ("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                   ".multicast::cluster [$0], [$1, {$2, $3}], [$4], $5;")
            cons = "r,l,r,r,r,h"
            args.append(Int16(mcast_mask).ir_value())
        llvm.inline_asm(None, args, ptx, cons, has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT)

    @cute.jit
    def tma_row_gather_get_copy_fn(
        self, tma_atom, sA_base, sAIdx, gAIdx, limit_m: Int32, cta_n: Int32, mcast_mask,
        tma_warp=None,
    ) -> Callable:
        """Per-row TMAs for one tile: the N-cluster CTAs split the tile rows and
        multicast them; rows past ``limit_m`` re-read source row 0 (their outputs
        are masked, as on the cp.async path). With ``tma_gather_warps`` > 1 each
        TMA warp (``tma_warp``) issues a contiguous slice of this CTA's rows.

        UTMALDG takes uniform operands, so per-lane coordinates would make ptxas
        emit a waterfall loop per TMA. The lanes stage this warp's row indices in
        smem once per tile and a single lane issues all of its row TMAs."""
        tile_m, tile_k = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]
        n_split = self.num_mcast_ctas_a
        rows_cta = (tile_m - self.hybrid_cp_rows) // n_split
        rows_w = rows_cta // self.tma_gather_warps
        # first row (within the tile) of this warp's slice
        w_off = cta_n * rows_cta
        if const_expr(tma_warp is not None):
            w_off = w_off + tma_warp * rows_w
        lane = cute.arch.lane_idx()
        cute.arch.sync_warp()
        for j in cutlass.range_constexpr((rows_w + 31) // 32):
            if const_expr(32 * (j + 1) <= rows_w) or lane + 32 * j < rows_w:
                idx = Int32(0)
                if w_off + lane + 32 * j < limit_m:
                    idx = gAIdx[w_off + lane + 32 * j]
                sAIdx[w_off - cta_n * rows_cta + lane + 32 * j] = idx
        cute.arch.sync_warp()
        desc_addr = copy_utils.get_tma_desc_addr(tma_atom).toint()
        dst_w = sA_base.toint() + w_off * tile_k
        i_off = w_off - cta_n * rows_cta
        stage_bytes = tile_m * tile_k
        mask = mcast_mask if const_expr(n_split > 1) else None
        batch = 16 if rows_w % 16 == 0 else 8

        def copy_fn(k_tile, smem_idx, tma_bar_ptr: cute.Pointer):
            dst = dst_w + smem_idx * stage_bytes
            bar = tma_bar_ptr.toint()
            col = k_tile * tile_k
            # The TMA asm is a memory side effect, so smem index loads cannot be
            # hoisted across it by the compiler: load each batch of indices first.
            for r0 in cutlass.range_constexpr(0, rows_w, batch):
                rows = [sAIdx[i_off + r0 + i] for i in range(batch)]
                rows = [cute.arch.make_warp_uniform(v) for v in rows]
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(batch):
                        self.tma_load_row(desc_addr, dst + (r0 + i) * tile_k, bar, col, rows[i], mask)

        return copy_fn

    def _make_gmem_tiled_copy_A(self, dtype, major_mode, num_threads, copy_bits=128):
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        copy_elems = copy_bits // dtype.width
        loads_per_cache_line = 128 * 8 // copy_bits  # 128 bytes per cache line
        shape_dim_1 = cute.size(self.cta_tile_shape_mnk[2]) // copy_elems
        if shape_dim_1 > loads_per_cache_line:
            shape_dim_1 = math.gcd(shape_dim_1, loads_per_cache_line)
        # thread layout for copy
        thread_layout = cute.make_layout(
            (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.cta_tile_shape_mnk[0]) // copy_elems
            if shape_dim_0 > loads_per_cache_line:
                shape_dim_0 = math.gcd(shape_dim_0, loads_per_cache_line)
            thread_layout = cute.make_layout(
                (shape_dim_0, num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if a_dtype not in {Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            is_valid = False
        if acc_dtype not in {Float32, Float16}:
            is_valid = False
        # tested d_dtype
        if d_dtype not in {
            None,
            Float32,
            Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False
        return is_valid


class Sm90ScaledGemm(GemmDefaultEpiMixin, _GemmSm90RawScaledBase):
    """Raw-UE8M0 scaled E4M3 GEMM for Hopper."""


class Sm90ScaledGemmRouteScatter(Sm90ScaledGemmRouteScatterMixin, _GemmSm90RawScaledBase):
    """Scaled varlen-M GEMM whose single-route rows land in the token output."""


class Sm90ScaledGated(GemmGatedMixin, _GemmSm90RawScaledBase):
    """Scaled Hopper GEMM with gated forward epilogue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledGatedRTO(Sm90ScaledGated):
    """Gated forward storing BF16 y1 with round-to-odd.

    The BF16 y1 is only an intermediate for the following 1x128 FP8 quantize.
    Round-to-odd makes that FP8 RN quantize bit-identical to quantizing the
    fp32 y1 directly (no biased double rounding).
    """

    @cute.jit
    def epi_convert_postact(self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx):
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            tRS_rPostAct[i] = _bf16_round_odd(tRS_rPostAct[i])
        return Sm90ScaledGated.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )


class Sm90ScaledDGated(GemmDGatedMixin, _GemmSm90RawScaledBase):
    """Scaled Hopper GEMM with gated backward epilogue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledDGatedDzQuant(Sm90DGatedDzQuantMixin, _GemmSm90RawScaledBase):
    """Scaled dGated that also emits dz as E4M3 + raw 1x128 UE8M0 scales."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledDGatedColQuant(Sm90DGatedColQuantMixin, _GemmSm90RawScaledBase):
    """Scaled dGated that also emits colwise 1x128 wgrad operands of y1s / dz."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledDGatedColQuantNoY1s(Sm90ScaledDGatedColQuant):
    """... without the BF16 y1s TMA store (only its col operand is consumed)."""

    skip_postact_store = True


class Sm90ScaledDGatedColQuantNoStore(Sm90ScaledDGatedColQuant):
    """... without the BF16 y1s / dz TMA stores (only fp8 operands consumed)."""

    skip_postact_store = True
    skip_d_store = True


class Sm90ScaledDGatedColQuantNoStoreInplace(Sm90ScaledDGatedColQuantNoStore):
    """... with the dz col operand written tile-locally into the BF16 preact."""

    dz_col_inplace = True


class Sm90ScaledGatedZQuant(GemmGatedBlockscaledQuantMixin, _GemmSm90RawScaledBase):
    """Gated forward with FP8 z quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledGatedY1Quant(GemmGatedPostActIsaQuantMixin, _GemmSm90RawScaledBase):
    """Gated forward with FP8 y1 quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledGatedZY1Quant(GemmGatedBlockscaledQuantPostActMixin, _GemmSm90RawScaledBase):
    """Gated forward with independent FP8 z and y1 quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledGatedRawQuant(Sm90RawGatedQuantMixin, _GemmSm90RawScaledBase):
    """Gated forward with Hopper-native raw 1x32 z and/or y1 quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        cta_tile_shape_mnk,
        atom_layout_mnk,
        element_type=None,
        epi_tile_override=None,
    ):
        tile_m, _ = _GemmSm90RawScaledBase._sm90_compute_tile_shape_or_override(
            cta_tile_shape_mnk,
            atom_layout_mnk,
            element_type,
            epi_tile_override,
        )
        assert cta_tile_shape_mnk[1] >= 64
        return (tile_m, 64)

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledGatedRawQuantRTO(Sm90ScaledGatedRawQuant):
    """Raw-quant gated forward (FP8 z) storing BF16 y1 with round-to-odd.

    Same rationale as ``Sm90ScaledGatedRTO``; only valid for a BF16 y1, which is
    stored through ``epi_convert_postact``.
    """

    @cute.jit
    def epi_convert_postact(self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx):
        for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
            tRS_rPostAct[i] = _bf16_round_odd(tRS_rPostAct[i])
        return Sm90ScaledGatedRawQuant.epi_convert_postact(
            self, tRS_rPostAct, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )


class Sm90ScaledGatedRawQuant128(Sm90RawGatedQuant128Mixin, Sm90ScaledGatedRawQuant):
    """Raw-quant gated forward with a fused 1x128 FP8 y1 quantized from fp32.

    A 1x128 y1 group spans the two N-adjacent CTAs of a (1, 2) cluster; their
    per-row amax is exchanged through DSMEM (see ``Sm90RawGatedQuant128Mixin``).
    """


class Sm90ScaledDGatedFP8PreAct(GemmDGatedFP8PreActMixin, _GemmSm90RawScaledBase):
    """Scaled dGated using carrier FP8 preact and raw UE8M0 scales."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blockscaled = False

    def _setup_attributes(self, epilogue_args):
        return _GemmSm90RawScaledBase._setup_attributes(self, epilogue_args)


class Sm90ScaledWgradVarlenK(Sm90ScaledGemm):
    """Scaled varlen-K wgrad; inherited kernel zeros empty experts."""


class Sm90ScaledWgradVarlenKTileLocal(Sm90ScaledWgradVarlenK):
    """Varlen-K wgrad whose A is the tile-local dz_col written into z by dgated."""

    a_tile_local = True


class Sm90ScaledRecomputeZ(Sm90ScaledGemm):
    """Scaled gather/varlen-M recompute kernel."""


_SONIC_QUACK_UTILS_DIR = Path(__file__).resolve().parent
if _SONIC_QUACK_UTILS_DIR not in EXTRA_SOURCE_DIRS:
    # The kernels live here, so their sources must be in the jit_cache fingerprint.
    EXTRA_SOURCE_DIRS.append(_SONIC_QUACK_UTILS_DIR)

# Compile cache of the from_dlpack fallback (SONIC_MOE_DISABLE_SM90_SCALED_TVM_FFI=1).
# The default TVM-FFI path is cached by ``_compile_sm90_scaled_tvm_ffi``.
_SM90_SCALED_CACHE = InstrumentedCompileCache("sm90_raw_ue8m0_scaled")

_SM90_SCALED_CLASSES = {
    cls.__name__: cls
    for cls in (
        Sm90ScaledGemm, Sm90ScaledGated, Sm90ScaledGatedRTO, Sm90ScaledDGated, Sm90ScaledGatedZQuant,
        Sm90ScaledGatedY1Quant, Sm90ScaledGatedZY1Quant, Sm90ScaledGatedRawQuant,
        Sm90ScaledGatedRawQuantRTO, Sm90ScaledGatedRawQuant128, Sm90ScaledDGatedFP8PreAct,
        Sm90ScaledWgradVarlenK, Sm90ScaledRecomputeZ,
        Sm90ScaledDGatedDzQuant, Sm90ScaledGemmRouteScatter, Sm90ScaledDGatedColQuant,
        Sm90ScaledDGatedColQuantNoY1s, Sm90ScaledDGatedColQuantNoStore,
        Sm90ScaledDGatedColQuantNoStoreInplace, Sm90ScaledWgradVarlenKTileLocal,
    )
}

# Constexpr epilogue callables are passed by name so compile keys stay picklable.
_EPI_FUNCS = {
    "swiglu_precise": quack.activation.swiglu_precise,
    "dswiglu_precise": quack.activation.dswiglu_precise,
}
_EPI_FUNC_FIELDS = ("act_fn", "act_bwd_fn")
# Runtime (non-Constexpr) Float32 epilogue scalars.
_EPI_SCALAR_FIELDS = ("alpha", "beta")

_CUTE_DTYPES = {
    "float8_e4m3fn": cutlass.Float8E4M3FN,
    "bfloat16": cutlass.BFloat16,
    "float16": cutlass.Float16,
    "float32": cutlass.Float32,
    "uint8": cutlass.Uint8,
    "int32": cutlass.Int32,
}

_GEMM_DIMS = {
    "A": ("m", "k", "l"), "B": ("n", "k", "l"),
    "D": ("m", "n", "l"), "C": ("m", "n", "l"),
}


# Host launch path: under paddle compat `A.device` (~2 us), `dtype ==` (~0.7 us),
# `permute` (~3 us) and cutlass scalar construction (~1 us) dominate the Python
# overhead of a launch, so the wrapper below caches everything that only depends
# on the call configuration and compares dtypes by identity first.
_FP8 = torch.float8_e4m3fn
if _FP8 not in torch2cute_dtype_map:
    torch2cute_dtype_map[_FP8] = cutlass.Float8E4M3FN
_PROFILES = ((32, 32), (128, 128))
_DTYPE_NAMES = {}


def _same_dtype(dtype, expected):
    return dtype is expected or dtype == expected


def _dtype_name(dtype):
    # str() covers torch and paddle-compat dtypes ("torch.bfloat16", "paddle.bfloat16").
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        name = _DTYPE_NAMES[dtype] = str(dtype).rsplit(".", 1)[-1]
    return name


_TVM_FFI_ENV = {}


def _tvm_ffi_enabled():
    raw = os.environ.get("SONIC_MOE_DISABLE_SM90_SCALED_TVM_FFI", "0")
    enabled = _TVM_FFI_ENV.get(raw)
    if enabled is None:
        enabled = _TVM_FFI_ENV[raw] = raw.lower() not in {"1", "true", "yes", "on"}
    return enabled


_DEVICE_CAPABILITY = {}
_UNIFORM_CAPABILITY = []


def _device_capability(device):
    key = str(device)
    cap = _DEVICE_CAPABILITY.get(key)
    if cap is None:
        cap = tuple(torch.cuda.get_device_capability(device)) if torch.cuda.is_available() else None
        _DEVICE_CAPABILITY[key] = cap
    return cap


def _tensor_capability(tensor):
    """Capability of the tensor's GPU without touching ``tensor.device`` when
    every visible GPU has the same capability (the normal training setup)."""
    if _UNIFORM_CAPABILITY:
        return _UNIFORM_CAPABILITY[0]
    caps = set()
    if torch.cuda.is_available():
        caps = {tuple(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())}
    if len(caps) == 1:
        _UNIFORM_CAPABILITY.append(caps.pop())
        return _UNIFORM_CAPABILITY[0]
    return _device_capability(tensor.device)


class _EpiTensor(NamedTuple):
    """Epilogue tensor in kernel order (rank-3 permuted to (*, *, L))."""

    tensor: object
    leading_dim: int
    packed_scales: bool = False  # raw scales readable as aligned 32-bit words


def _epi_operand(tensor, *, varlen_m=False, leading_dim=None):
    if tensor is None:
        return None
    if tensor.ndim == 3 and not varlen_m:
        tensor = tensor.permute(1, 2, 0)
    if leading_dim is None:
        unit = [index for index, stride in enumerate(tensor.stride()) if int(stride) == 1]
        if not unit:
            raise ValueError("epilogue tensors require a unit-stride dimension")
        leading_dim = unit[-1]
    return _EpiTensor(tensor.detach(), leading_dim)


def _scale_operand(tensor):
    # Raw carriers are produced as (batch, row, k_group), while the kernel
    # indexes rank-3 scales as (row, k_group, batch), matching A/B tensors after
    # the (L, *, *) -> (*, *, L) operand permute.  Rank-2 scales remain row-major,
    # so their unit-stride K-group dimension is leading dimension 1.
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    stride = tensor.stride()
    leading_dim = next((i for i in range(len(stride) - 1, -1, -1) if stride[i] == 1), None)
    if leading_dim is None:
        tensor = tensor.contiguous()
        leading_dim = tensor.ndim - 1
        stride = tensor.stride()
    return _EpiTensor(tensor.detach(), leading_dim, _packed_scales_ok(tensor, stride, leading_dim))


def _prepare_fp8_operand(tensor, *, varlen_m=False, varlen_k=False):
    """Return physical E4M3 storage whose logical K dimension is contiguous."""
    if not _same_dtype(tensor.dtype, _FP8):
        return tensor
    k_dim = 1 if tensor.ndim == 2 or (varlen_k and not varlen_m) else 2
    return tensor if tensor.stride(k_dim) == 1 else tensor.contiguous()


def _validate_profile(group_size, weight_block):
    profile = (int(group_size), int(weight_block))
    if profile not in _PROFILES:
        raise ValueError(f"unsupported Hopper scaled profile {profile}")
    return profile


def _validate_public(A, B, D, a_scales, b_scales, group_size, weight_block):
    if (group_size, weight_block) not in _PROFILES:
        _validate_profile(group_size, weight_block)
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors")
    cap = _tensor_capability(A)
    if cap is None or cap[0] != 9:
        raise RuntimeError("native SM90 scaled GEMM requires a Hopper CUDA device")
    if not _same_dtype(A.dtype, _FP8) or not _same_dtype(B.dtype, _FP8):
        raise TypeError("A and B must remain torch.float8_e4m3fn operands")
    if A.shape[-1] % group_size or (
        B.ndim == 3 and (B.shape[-2] % weight_block or B.shape[-1] % weight_block)
    ):
        raise ValueError(
            f"profile 1x{group_size}+{weight_block}x{weight_block} requires divisible "
            f"operand dimensions, got A={tuple(A.shape)}, B={tuple(B.shape)}"
        )
    if D is not None and not D.is_cuda:
        raise ValueError("D must be a CUDA output tensor when provided")
    for scale, name in ((a_scales, "a_scales"), (b_scales, "b_scales")):
        if scale is None or not scale.is_cuda or scale.element_size() != 1:
            raise TypeError(f"{name} must be a raw one-byte UE8M0 CUDA tensor")


def _tensor_spec(operand):
    """Shape-free compile spec of an _EpiTensor: (dtype name, rank, leading dim)."""
    if operand is None:
        return None
    return (_dtype_name(operand.tensor.dtype), operand.tensor.ndim, operand.leading_dim)


def _epi_spec(epi_fields):
    """Shape-free compile spec of the epilogue fields (tensors, scalars, Constexprs)."""
    spec, seen = [], {}
    for name, value in epi_fields.items():
        if isinstance(value, _EpiTensor):
            if id(value) in seen:  # the same tensor bound to two fields
                spec.append((name, "alias", seen[id(value)]))
            else:
                seen[id(value)] = name
                spec.append((name, "tensor", _tensor_spec(value)))
        elif name in _EPI_SCALAR_FIELDS:
            spec.append((name, "scalar", value is not None))
        else:
            spec.append((name, "const", value))
    return tuple(spec)


def _fake_tensor(spec, align):
    """Fake tensor with fully dynamic shape/strides, matching ``mark_layout_dynamic``."""
    if spec is None:
        return None
    dtype_name, ndim, leading_dim = spec
    dtype = _CUTE_DTYPES[dtype_name]
    shape = tuple(cute.sym_int() for _ in range(ndim))
    stride = tuple(1 if i == leading_dim else cute.sym_int64() for i in range(ndim))
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)


def _fake_varlen_args(varlen_m, varlen_k, gather):
    if not varlen_m and not varlen_k:
        return None
    fake = lambda: _fake_tensor(("int32", 1, 0), 4)
    return VarlenArguments(
        mCuSeqlensM=fake() if varlen_m else None,
        mCuSeqlensK=fake() if varlen_k else None,
        mAIdx=fake() if gather else None,
    )


def _build_epilogue_args(GemmCls, epi_spec, make_tensor, make_scalar, resolve_const,
                         runtime=False):
    # TVM-FFI runtime calls pass None for every field that is not a tensor or scalar,
    # including Constexpr fields left at their defaults (e.g. rounding_mode).
    fields = dict.fromkeys(GemmCls.EpilogueArguments._fields) if runtime else {}
    tensors = {}
    for name, kind, value in epi_spec:
        if kind == "tensor":
            tensors[name] = fields[name] = make_tensor(name, value)
        elif kind == "alias":
            fields[name] = tensors[value]
        elif kind == "scalar":
            fields[name] = make_scalar(name) if value else None
        else:
            fields[name] = resolve_const(name, value)
    return GemmCls.EpilogueArguments(**fields)


def _new_kernel(gemm_cls_name, a_dtype, tile_shape, cluster_shape, pingpong, persistent,
                gather, activation_group, weight_block, implicit_dtype, k_tile=0):
    tg = gather - 2 if gather >= 3 else 0  # (tma_gather_warps - 1) * 16 + cp divisor
    kernel = _SM90_SCALED_CLASSES[gemm_cls_name](
        Float32, a_dtype, tile_shape, cluster_shape, pingpong=pingpong,
        is_persistent=persistent, gather_A=bool(gather),
        activation_group=activation_group, weight_block=weight_block,
        tma_gather_A=tg > 0, tma_gather_cp_div=tg % 16 if tg % 16 > 1 else 0,
        tma_gather_warps=tg // 16 + 1,
    )
    # gather == 2: source-ordered A scales are gathered in-kernel via A_idx.
    # gather >= 3: same, with A rows loaded by per-row TMA (see tma_gather_A);
    # (gather - 2) % 16 = d >= 2: hybrid, the last tile_M // d rows of a tile via
    # cp.async; (gather - 2) // 16 + 1 load warps issue the row TMAs.
    kernel.gather_a_scales = gather >= 2
    kernel.k_tile = int(k_tile)
    if implicit_dtype is not None:
        kernel.implicit_dtype = _CUTE_DTYPES[implicit_dtype]
    return kernel


def _resolve_compile_const(name, value):
    return _EPI_FUNCS[value] if name in _EPI_FUNC_FIELDS and value is not None else value


@jit_cache
def _compile_sm90_scaled_tvm_ffi(
    gemm_cls_name, a_spec, b_spec, d_spec, c_spec, sa_spec, sb_spec, epi_spec,
    varlen_m, varlen_k, gather, tile_shape, cluster_shape, pingpong, persistent,
    max_swizzle_size, activation_group, weight_block, implicit_dtype, k_tile, device_capability,
):
    """Compile once per layout/config; shapes and strides stay symbolic.

    ``max_swizzle_size`` and ``device_capability`` are not read here but keep
    compiled kernels apart in the in-memory and disk caches.
    """
    GemmCls = _SM90_SCALED_CLASSES[gemm_cls_name]
    mA, mB, mD, mC = (_fake_tensor(spec, 16) for spec in (a_spec, b_spec, d_spec, c_spec))
    mSFA, mSFB = _fake_tensor(sa_spec, 1), _fake_tensor(sb_spec, 1)
    epi_args = _build_epilogue_args(
        GemmCls, epi_spec,
        make_tensor=lambda name, spec: _fake_tensor(spec, 16),
        make_scalar=lambda name: Float32(1.0),
        resolve_const=_resolve_compile_const,
    )
    scheduler_args = make_fake_scheduler_args(False, False, None)
    varlen_args = _fake_varlen_args(varlen_m, varlen_k, gather)
    kernel = _new_kernel(gemm_cls_name, mA.element_type, tile_shape, cluster_shape, pingpong,
                         persistent, gather, activation_group, weight_block, implicit_dtype,
                         k_tile)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel, mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB,
        options="--enable-tvm-ffi",
    )


def _raw_stream():
    stream = torch.cuda.current_stream()
    return cuda.CUstream(
        stream.stream_base.raw_stream if hasattr(stream, "stream_base") else stream.cuda_stream
    )


def _runtime_cute_tensor(tensor, leading_dim, align):
    storage = tensor.detach()
    is_fp8 = tensor.dtype == torch.float8_e4m3fn
    if is_fp8:
        storage = storage.view(torch.uint8)
    result = from_dlpack(storage, assumed_align=align)
    if is_fp8:
        result.element_type = cutlass.Float8E4M3FN
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _launch_dlpack(
    gemm_cls_name, key, operands, scales, epi_spec, epi_fields, scheduler, varlen,
    kernel_args,
):
    """from_dlpack fallback; the key has no shapes since all layouts are dynamic."""
    GemmCls = _SM90_SCALED_CLASSES[gemm_cls_name]
    mA, mB, mD, mC = (
        None if op is None else _runtime_cute_tensor(op.tensor, op.leading_dim, 16)
        for op in operands
    )
    mSFA, mSFB = (
        from_dlpack(op.tensor, assumed_align=1).mark_layout_dynamic(leading_dim=op.leading_dim)
        for op in scales
    )
    epi_args = _build_epilogue_args(
        GemmCls, epi_spec,
        make_tensor=lambda name, spec: _runtime_cute_tensor(
            epi_fields[name].tensor, epi_fields[name].leading_dim, 16
        ),
        make_scalar=lambda name: float(epi_fields[name]),
        resolve_const=_resolve_compile_const,
    )
    max_active_clusters, max_swizzle_size = scheduler
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, max_swizzle_size=max_swizzle_size
    )
    varlen_args = GemmWrapperBase.create_varlen_args(*varlen)
    stream = _raw_stream()
    compiled = _SM90_SCALED_CACHE.get_or_compile(
        key,
        lambda: cute.compile(
            _new_kernel(gemm_cls_name, mA.element_type, *kernel_args), mA, mB, mD, mC, epi_args,
            scheduler_args, varlen_args, stream, mSFA, mSFB,
        ),
    )
    compiled(mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB)


_SM90_K_TILE = int(os.environ.get("SONIC_SM90_K_TILE", "128"))
# 1x128 varlen-K wgrad launch config (SONIC_MOE_SM90_WGRAD128_TILE="128x128",
# SONIC_MOE_SM90_WGRAD128_CLUSTER="1x1"); SONIC_MOE_SM90_WGRAD_VALIDATE=1 adds
# a host check (one sync) of the packed offsets.


def _parse_mn(raw, default):
    try:
        m, n = (int(v) for v in raw.lower().split("x"))
        return m, n
    except (AttributeError, ValueError):
        return default


_SM90_WGRAD128_TILE = _parse_mn(os.environ.get("SONIC_MOE_SM90_WGRAD128_TILE"), (128, 128))
_SM90_WGRAD128_CLUSTER = (
    *_parse_mn(os.environ.get("SONIC_MOE_SM90_WGRAD128_CLUSTER"), (1, 1)), 1
)
_SM90_WGRAD_VALIDATE = os.environ.get("SONIC_MOE_SM90_WGRAD_VALIDATE", "0") == "1"


def _packed_scales_ok(t, stride, leading_dim):
    """Kernel-order scale rows start 4B-aligned, K groups are unit-stride, and
    every byte offset fits the kernel's Int32 row offsets."""
    if leading_dim != 1 or t.data_ptr() & 3:
        return False
    shape = t.shape
    span = (shape[0] - 1) * stride[0] + shape[1]
    if stride[0] & 3 and shape[0] != 1:
        return False
    if len(shape) == 3:
        if stride[2] & 3 and shape[2] != 1:
            return False
        span += (shape[2] - 1) * stride[2]
    return span < 2**31


def _select_k_tile(activation_group, varlen_k, K):
    """Profile-32 uses a K=128 CTA tile when K is a multiple of 128.

    varlen-K (expert K boundaries are only 32-aligned) keeps K=32.
    SONIC_SM90_K_TILE=32 restores the one-group tile.
    """
    if activation_group == 32 and not varlen_k and _SM90_K_TILE == 128 and int(K) % 128 == 0:
        return 128
    return int(activation_group)


def _gemm_shapes(A, B, D, C, cu_seqlens_m, cu_seqlens_k, A_idx):
    """(L, M, K, N); the checks of GemmWrapperBase.validate_and_prepare_tensors."""
    if A_idx is not None:
        assert cu_seqlens_m is not None or cu_seqlens_k is not None, (
            "gather_A requires either varlen_m or varlen_k"
        )
        assert A_idx.ndim == 1, f"A_idx must be 1D, got {A_idx.ndim}D"
    a_shape, b_shape = tuple(A.shape), tuple(B.shape)
    if cu_seqlens_m is not None:
        assert len(a_shape) == 2, f"A must be 2D when using varlen_m, got {len(a_shape)}D"
        assert len(b_shape) == 3, f"B must be 3D with varlen_m, got {len(b_shape)}D"
        M = int(A_idx.shape[0]) if A_idx is not None else a_shape[0]
        K = a_shape[1]
        L, N, K_B = b_shape
        cu_shape = tuple(cu_seqlens_m.shape)
        dc_shape = (M, N)
    elif cu_seqlens_k is not None:
        assert len(a_shape) == 2, f"A must be 2D when using varlen_k, got {len(a_shape)}D"
        assert len(b_shape) == 2, f"B must be 2D with varlen_k, got {len(b_shape)}D"
        M = a_shape[0]
        K = int(A_idx.shape[0]) if A_idx is not None else a_shape[1]
        N, K_B = b_shape
        cu_shape = tuple(cu_seqlens_k.shape)
        L = cu_shape[0] - 1 if len(cu_shape) == 1 else -1
        dc_shape = (L, M, N)
    else:
        assert len(a_shape) == 3 and len(b_shape) == 3, "A and B must be 3D tensors"
        L, M, K = a_shape
        _, N, K_B = b_shape
        assert b_shape == (L, N, K), f"B must have shape {(L, N, K)}, got {b_shape}"
        cu_shape = (L + 1,)
        dc_shape = (L, M, N)
    assert K == K_B, f"K dimension mismatch: A has {K}, B has {K_B}"
    assert cu_shape == (L + 1,), f"cu_seqlens must have shape ({L + 1},), got {cu_shape}"
    for tensor, name in ((D, "D"), (C, "C")):
        if tensor is not None:
            assert tuple(tensor.shape) == dc_shape, (
                f"{name} shape {tuple(tensor.shape)} doesn't match expected {dc_shape}"
            )
    return L, M, K, N


def _gemm_operand(tensor, permute):
    """Kernel-order operand: rank-3 (L, *, *) -> (*, *, L); leading dim 1 if stride(1)==1."""
    if tensor is None:
        return None
    if permute and tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    return _EpiTensor(tensor.detach(), 1 if tensor.stride(1) == 1 else 0)


_SCHEDULER_ARGS = {}


def _scheduler_args(cluster_shape, persistent, max_swizzle_size):
    """(max_active_clusters, runtime TileSchedulerOptions), built once per config."""
    key = (tuple(cluster_shape), bool(persistent), max_swizzle_size)
    cached = _SCHEDULER_ARGS.get(key)
    if cached is None:
        max_active_clusters = (
            get_max_active_clusters(cluster_shape[0] * cluster_shape[1]) if persistent else 0
        )
        cached = _SCHEDULER_ARGS[key] = (
            max_active_clusters, make_scheduler_args(max_active_clusters, max_swizzle_size, None),
        )
    return cached


_EPI_RUNTIME_PLANS = {}


def _runtime_epilogue_args(GemmCls, epi_spec, epi_fields):
    """Runtime EpilogueArguments: tensors/scalars bound, every other field None."""
    plan_key = (GemmCls, epi_spec)
    plan = _EPI_RUNTIME_PLANS.get(plan_key)
    if plan is None:
        fields = GemmCls.EpilogueArguments._fields
        plan = _EPI_RUNTIME_PLANS[plan_key] = (len(fields), tuple(
            (fields.index(name), kind == "scalar", name)
            for name, kind, value in epi_spec
            if kind in ("tensor", "alias") or (kind == "scalar" and value)
        ))
    num_fields, bound = plan
    values = [None] * num_fields
    for index, is_scalar, name in bound:
        values[index] = Float32(epi_fields[name]) if is_scalar else epi_fields[name].tensor
    return GemmCls.EpilogueArguments(*values)


# In-process TVM-FFI kernels by compile key (jit_cache adds disk lookup and logging).
_TVM_FFI_COMPILED = {}

def _run_scaled(
    GemmCls, A, B, D, a_scales, b_scales, *, C=None, epi_fields=None,
    cu_seqlens_m=None, cu_seqlens_k=None, A_idx=None, activation_group=32,
    weight_block=32, tile_shape=(128, 128), cluster_shape=(1, 1, 1),
    pingpong=False, persistent=True, implicit_dtype=None, max_swizzle_size=8,
    activation_scale_order=None, tma_gather=False,
):
    _validate_public(A, B, D, a_scales, b_scales, activation_group, weight_block)
    if D is None and not issubclass(GemmCls, GemmGatedMixin):
        raise ValueError("D may be omitted only by a gated no-z variant")
    if cu_seqlens_m is not None and cu_seqlens_k is not None:
        raise ValueError("only one varlen dimension may be selected")
    if (cu_seqlens_m is not None or cu_seqlens_k is not None) and not persistent:
        raise ValueError("varlen requires persistent=True")
    if A_idx is not None and _dtype_name(A_idx.dtype) != "int32":
        raise TypeError("A_idx must be int32")
    if cu_seqlens_k is not None and _dtype_name(cu_seqlens_k.dtype) != "int32":
        raise TypeError("cu_seqlens_k must be int32")
    gather = A_idx is not None
    varlen_m, varlen_k = cu_seqlens_m is not None, cu_seqlens_k is not None
    A = _prepare_fp8_operand(A, varlen_m=varlen_m, varlen_k=varlen_k)
    B = _prepare_fp8_operand(B, varlen_m=varlen_m, varlen_k=varlen_k)
    if varlen_k:
        # Dedicated Hopper wgrad ABI: both operands are already physical and
        # logical K-major matrices [M,total_K] and [N,total_K]. No transpose,
        # gather, or BF16 operand is accepted by this path.
        if gather:
            raise ValueError("SM90 scaled varlen-K consumes pre-gathered K-major payloads")
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
            raise ValueError("SM90 scaled varlen-K expects A=[M,total_K], B=[N,total_K]")
        if not A.is_contiguous() or not B.is_contiguous():
            raise ValueError("SM90 scaled varlen-K operands must be contiguous K-major")
    if gather and not varlen_k:
        if activation_scale_order not in ("source", "route"):
            raise ValueError(
                "gathered SM90 scaled GEMM requires activation_scale_order='source' or 'route'"
            )
        expected_rows = A.shape[0] if activation_scale_order == "source" else int(A_idx.shape[0])
        if a_scales.shape[0] != expected_rows:
            raise ValueError(
                f"{activation_scale_order}-ordered A scales require {expected_rows} rows, "
                f"got {a_scales.shape[0]}"
            )
        if activation_scale_order == "source":
            # A is read through A_idx, and so are the source-row A scales: the
            # MMA warps resolve A_idx per tile (gather mode 2), no host gather.
            gather = 2
    L, M, K, N = _gemm_shapes(A, B, D, C, cu_seqlens_m, cu_seqlens_k, A_idx)
    # The non-varlen-K mainloop indexes scale columns without a clamp.
    if not varlen_k and min(a_scales.shape[-1], b_scales.shape[-1]) * activation_group < K:
        raise ValueError("a_scales/b_scales must span K / activation_group columns")
    # (L, *, *) -> (*, *, L): every rank-3 operand without varlen, B under
    # varlen-M, D/C under varlen-K.
    operands = (
        _gemm_operand(A, permute=not varlen_m and not varlen_k),
        _gemm_operand(B, permute=not varlen_k),
        _gemm_operand(D, permute=not varlen_m),
        _gemm_operand(C, permute=not varlen_m),
    )
    if varlen_k:
        # Raw scale ABI of the wgrad (independent of the forward profile):
        #   1x32:  contiguous [feature, total_K/32], rebased by expert_start/32.
        #   1x128: contiguous k-group-major [>= total_K/128 + pad, feature],
        #          rebased by expert_start/128; the pad rows (one prefetch row
        #          + >= 256 bytes) absorb the one-tile-ahead scale prefetch and
        #          the unmasked scale reads of phantom rows/columns of edge tiles.
        if (activation_group, weight_block) == (32, 32):
            groups = K // 32
            expected = ((M, groups), (N, groups))
            ok = tuple(a_scales.shape) == expected[0] and tuple(b_scales.shape) == expected[1]
        elif (activation_group, weight_block) == (128, 128):
            groups = K // 128
            expected = (f"[>={groups}+pad,{M}]", f"[>={groups}+pad,{N}]")
            ok = all(
                s.ndim == 2 and int(s.shape[1]) == f and f % 8 == 0
                and int(s.shape[0]) * f >= (groups + 1) * f + 256
                for s, f in ((a_scales, M), (b_scales, N))
            )
        else:
            raise ValueError("SM90 wgrad operands and scales use colwise 1x32 or 1x128 groups")
        if not ok:
            raise ValueError(
                f"SM90 varlen-K raw scale ABI requires a_scales={expected[0]}, "
                f"b_scales={expected[1]}; got {tuple(a_scales.shape)} and {tuple(b_scales.shape)}"
            )
        if not a_scales.is_contiguous() or not b_scales.is_contiguous():
            raise ValueError("SM90 varlen-K raw scales must be contiguous")
    scales = (_scale_operand(a_scales), _scale_operand(b_scales))
    epi_fields = epi_fields or {}
    epi_spec = _epi_spec(epi_fields)
    implicit_dtype = None if implicit_dtype is None else _dtype_name(implicit_dtype)
    k_tile = _select_k_tile(activation_group, varlen_k, K)
    packed = scales[0].packed_scales and scales[1].packed_scales
    if k_tile == 128 and activation_group == 32 and not packed:
        k_tile = 32  # the K=128 tile reads 4 scale columns as one aligned 32-bit word
    if tma_gather and gather == 2:
        # Per-row TMA: 128B K tile (= swizzle width), 16B-aligned K-major rows.
        if (k_tile != 128 or A.ndim != 2 or A.stride(-1) != 1 or A.stride(0) % 16
                or A.data_ptr() % 16 or pingpong):
            raise ValueError("tma_gather needs a 2D K-major A with 16B rows and a K=128 tile")
        # tma_gather=True: all rows by TMA; tma_gather=d >= 2: the last
        # tile_M // d rows by cp.async (hybrid); tma_gather=(d, w): w warps
        # issue the row TMAs (d = 1: no cp.async rows).
        d, w = (tma_gather if isinstance(tma_gather, tuple)
                else (1 if tma_gather is True else int(tma_gather), 1))
        assert d >= 1 and tile_shape[0] % d == 0 and 1 <= w <= 4 - (d > 1) and d < 16, tma_gather
        gather = 2 + d + 16 * (w - 1)
    if cluster_shape[1] > 1 and gather and gather < 3:
        raise ValueError("cp.async gather A cannot use a cluster along N")
    if activation_group == 128 and not varlen_k:
        for s in scales:
            t = s.tensor
            if sum((n - 1) * st for n, st in zip(t.shape, t.stride())) >= 2**31:
                raise ValueError("SM90 profile-128 scales must span < 2**31 bytes (Int32 offsets)")
    key = (
        GemmCls.__name__, *(_tensor_spec(op) for op in operands + scales), epi_spec,
        varlen_m, varlen_k, gather, tuple(tile_shape), tuple(cluster_shape), bool(pingpong),
        bool(persistent), int(max_swizzle_size), int(activation_group), int(weight_block),
        implicit_dtype, k_tile, _tensor_capability(A),
    )
    max_active_clusters, scheduler_args = _scheduler_args(
        cluster_shape, persistent, max_swizzle_size
    )
    varlen = (cu_seqlens_m, cu_seqlens_k, A_idx)
    if not _tvm_ffi_enabled():
        kernel_args = (tile_shape, cluster_shape, pingpong, persistent, gather,
                       activation_group, weight_block, implicit_dtype, k_tile)
        _launch_dlpack(
            GemmCls.__name__, key, operands, scales, epi_spec, epi_fields,
            (max_active_clusters, max_swizzle_size), varlen, kernel_args,
        )
        return D
    if cache_utils.COMPILE_ONLY:
        _compile_sm90_scaled_tvm_ffi(*key)
        return D
    compiled = _TVM_FFI_COMPILED.get(key)
    if compiled is None:
        compiled = _TVM_FFI_COMPILED[key] = _compile_sm90_scaled_tvm_ffi(*key)
    # Runtime call: tensors bind to the symbolic layouts, Constexpr fields are None.
    compiled(
        *(None if op is None else op.tensor for op in operands),
        _runtime_epilogue_args(GemmCls, epi_spec, epi_fields),
        scheduler_args, make_varlen_args(*varlen), scales[0].tensor, scales[1].tensor,
    )
    return D


def sm90_scaled_gemm(
    A, B, D, a_scales, b_scales, *, C=None, beta=1.0,
    add_to_output=False, scatter_dst=None, scatter_out=None, scatter_w=None, **kwargs,
):
    """Public native scaled GEMM wrapper.

    ``scatter_dst`` (int32 ``[M]``) / ``scatter_out`` (bf16 ``[T, N]``): fused
    single-route token reduction.  Row ``p`` with ``scatter_dst[p] = t >= 0``
    is written to ``scatter_out[t]`` as ``bf16(0 + bf16(acc) * w)`` (``w`` =
    ``scatter_w[p]`` fp32, or the weightless ``0 + x``) instead of ``D[p]``;
    rows with ``scatter_dst[p] < 0`` go to ``D`` unchanged.
    """
    if add_to_output and C is not None:
        raise ValueError("add_to_output cannot also take C")
    if scatter_dst is not None:
        if C is not None or add_to_output or kwargs.get("cu_seqlens_m") is None \
                or kwargs.get("A_idx") is not None:
            raise ValueError("route scatter needs a plain varlen-M GEMM without C / A_idx")
        N = D.shape[1]
        for t, name in ((D, "D"), (scatter_out, "scatter_out")):
            if (not _same_dtype(t.dtype, torch.bfloat16) or t.ndim != 2 or t.shape[1] != N
                    or t.stride(1) != 1 or t.stride(0) % 8 or t.data_ptr() % 16):
                raise ValueError(f"route scatter {name} must be 16B-aligned bf16 [*, {N}] rows")
        if N % 8:
            raise ValueError("route scatter needs N % 8 == 0")
        for t, name, dt in ((scatter_dst, "scatter_dst", "int32"),
                            (scatter_w, "scatter_w", "float32")):
            if t is not None and (_dtype_name(t.dtype) != dt or t.ndim != 1 or t.stride(0) != 1
                                  or t.shape[0] != D.shape[0] or t.data_ptr() % 16):
                raise ValueError(f"{name} must be a contiguous 16B-aligned {dt} [M] tensor")
        # Validated row-major 2D / 1D above: unit stride is the last dim.
        epi = dict(
            mScatterD=_EpiTensor(D.detach(), 1), mScatterDst=_EpiTensor(scatter_dst.detach(), 0),
            mScatterOut=_EpiTensor(scatter_out.detach(), 1),
            mScatterW=None if scatter_w is None else _EpiTensor(scatter_w.detach(), 0),
        )
        return _run_scaled(
            Sm90ScaledGemmRouteScatter, A, B, D, a_scales, b_scales, epi_fields=epi, **kwargs,
        )
    epi = dict(beta=float(beta) if C is not None else None, add_to_output=bool(add_to_output))
    return _run_scaled(
        Sm90ScaledGemm, A, B, D, a_scales, b_scales,
        C=None if add_to_output else C, epi_fields=epi, **kwargs,
    )


def sm90_scaled_gated(
    A, B, D, PostAct, a_scales, b_scales, *, activation="swiglu",
    z_scale_out=None, postact_scale_out=None, swiglu_clamp_value=0.0,
    postact_bf16_trunc=False, postact_round_odd=False, postact_group=32, **kwargs,
):
    """Gated forward supporting BF16/FP8/no-z and BF16/FP8 y1 outputs.

    ``postact_round_odd`` stores BF16 y1 with round-to-odd (for y1 that is only
    re-quantized to FP8 afterwards); ignored for an FP8 y1.

    ``postact_group`` is the FP8 y1 group (32 or 128). 128 quantizes y1 from
    the fp32 epilogue registers, byte-identical to
    ``quantize_activation_blockscaled_fast(y1_bf16_round_odd, 128)``; it needs
    tile (192, 128), cluster (1, 2, 1), no pingpong and ``2I % 256 == 0``, with
    ``postact_scale_out`` of shape ``[M, I/128]``.
    """
    if activation != "swiglu":
        raise ValueError(f"native SM90 scaled gated does not support {activation!r}")
    if PostAct is None:
        raise ValueError("PostAct output is required")
    z_fp8 = D is not None and D.dtype == torch.float8_e4m3fn
    y1_fp8 = PostAct.dtype == torch.float8_e4m3fn
    if D is not None and D.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("gated z output must be BF16, E4M3, or None")
    if PostAct.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("gated y1 output must be BF16 or E4M3")
    if z_fp8 != (z_scale_out is not None):
        raise ValueError("FP8 z requires z_scale_out, and BF16/no-z forbids it")
    if y1_fp8 != (postact_scale_out is not None):
        raise ValueError("FP8 y1 requires postact_scale_out, and BF16 y1 forbids it")
    postact = _epi_operand(PostAct, varlen_m=kwargs.get("cu_seqlens_m") is not None)
    epi = dict(mPostAct=postact, act_fn="swiglu_precise",
               swiglu_clamp_value=float(swiglu_clamp_value))
    if z_fp8 or y1_fp8:
        if y1_fp8 and postact_group == 128:
            if (tuple(kwargs.get("tile_shape", ())) != (192, 128)
                    or tuple(kwargs.get("cluster_shape", ())) != (1, 2, 1)
                    or kwargs.get("pingpong", False) or PostAct.shape[-1] % 128 != 0):
                raise ValueError("fused 1x128 y1 needs tile (192,128), cluster (1,2,1), "
                                 "no pingpong and 2I % 256 == 0")
            if postact_scale_out.shape[-1] * 128 != PostAct.shape[-1]:
                raise ValueError("fused 1x128 y1 scale must be [M, I/128]")
            GemmCls = Sm90ScaledGatedRawQuant128
        elif postact_group != 32 and y1_fp8:
            raise ValueError(f"unsupported FP8 y1 group {postact_group}")
        else:
            GemmCls = (Sm90ScaledGatedRawQuantRTO if postact_round_odd and not y1_fp8
                       else Sm90ScaledGatedRawQuant)
        epi.update(
            postact_bf16_trunc=bool(postact_bf16_trunc),
            mZScale=_epi_operand(z_scale_out, varlen_m=True),
            mPostActScaleRaw=_epi_operand(postact_scale_out, varlen_m=True),
            mPostActOutput=postact,
        )
    else:
        GemmCls = Sm90ScaledGatedRTO if postact_round_odd else Sm90ScaledGated
    return _run_scaled(GemmCls, A, B, D, a_scales, b_scales, epi_fields=epi, **kwargs)


def sm90_scaled_dgated(
    A, B, D, PreAct, PostAct, a_scales, b_scales, *,
    activation="swiglu", colvec_scale=None, colvec_reduce=None,
    preact_fp8=None, preact_scales=None, swiglu_clamp_value=0.0,
    dz_fp8_out=None, dz_scale_out=None, y1s_col_out=None, dz_col_out=None, skip_stores=(),
    dz_col_inplace=False, **kwargs,
):
    """dGated with BF16 preact or carrier FP8 preact plus UE8M0 scales.

    ``dz_fp8_out``/``dz_scale_out`` additionally emit dz as E4M3 ``[M, 2I]`` with
    raw 1x128 UE8M0 scales ``[M, 2I/128]``, byte-identical to
    ``quantize_activation_blockscaled_fast(dz, 128)`` (BF16 preact path only).

    ``y1s_col_out``/``dz_col_out`` = ``(fp8 [F, M], u8 [>= M/128, F])`` additionally
    emit the colwise 1x128 wgrad operand of y1s / dz, byte-identical to
    ``quantize_wgrad_colwise128_sm90(y1s|dz, cu_seqlens_m)`` on rows
    ``< cu_seqlens_m[E]`` (tile M 128, 128-aligned ``cu_seqlens_m``; rows beyond
    ``cu_seqlens_m[E]`` are left untouched).

    ``skip_stores`` (``("y1s",)`` or ``("y1s", "dz")``, with the matching fused
    fp8 outputs) drops those BF16 TMA stores: ``PostAct`` / ``D`` keep their
    previous contents (for the gmem layout only).

    ``dz_col_inplace`` (with ``dz_col_out=(PreAct, scales)`` and both stores
    skipped) writes the dz col data tile-locally into ``PreAct`` instead: the
    128x128 block (token block t, feature group g) goes to rows ``128 t + f % 128``,
    bytes ``256 g + [0, 128)``, which is the preact half this CTA has already read.
    ``sm90_scaled_wgrad_varlen_k(a_tile_local=True)`` consumes that layout.
    """
    if activation != "swiglu":
        raise ValueError(f"native SM90 scaled dGated does not support {activation!r}")
    fp8_mode = preact_fp8 is not None or preact_scales is not None
    if fp8_mode and (preact_fp8 is None or preact_scales is None):
        raise ValueError("FP8 preact requires both preact_fp8 and preact_scales")
    varlen_m = kwargs.get("cu_seqlens_m") is not None
    if D.dtype == torch.float32:
        D_packed = D
        output_dtype = torch.bfloat16
    elif D.dtype in (torch.bfloat16, torch.float16):
        D_packed = D.view(torch.float32)
        output_dtype = D.dtype
    else:
        raise TypeError("dGated dz output must be packed FP32 or BF16/FP16")
    if PostAct.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("dGated y1s output must be BF16/FP16")
    epi = dict(
        mPostAct=_epi_operand(PostAct, varlen_m=varlen_m), act_bwd_fn="dswiglu_precise",
        swiglu_clamp_value=float(swiglu_clamp_value),
        mColVecBroadcast=_epi_operand(colvec_scale, varlen_m=varlen_m, leading_dim=0),
        mColVecReduce=_epi_operand(colvec_reduce, varlen_m=varlen_m),
    )
    if fp8_mode:
        if preact_fp8.dtype != torch.float8_e4m3fn or preact_scales.dtype != torch.uint8:
            raise TypeError("carrier preact must be E4M3 with raw uint8 UE8M0 scales")
        if preact_fp8.ndim != 2 or int(preact_fp8.shape[1]) % 32:
            raise ValueError("carrier preact must be rank-2 with 32-aligned width")
        expected_scales = (int(preact_fp8.shape[0]), int(preact_fp8.shape[1]) // 32)
        if tuple(preact_scales.shape) != expected_scales or not preact_scales.is_contiguous():
            raise ValueError(
                "carrier preact scales always use fixed raw 1x32 ABI; expected "
                f"{expected_scales}, got {tuple(preact_scales.shape)}"
            )
        GemmCls = Sm90ScaledDGatedFP8PreAct
        C = None
        epi.update(
            mFP8PreAct_fp8=_epi_operand(preact_fp8, varlen_m=True),
            mFP8PreAct_scales=_epi_operand(preact_scales, varlen_m=True),
        )
    else:
        if PreAct is None or PreAct.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("BF16/FP16 preact must be provided")
        GemmCls = Sm90ScaledDGated
        C = PreAct.contiguous().view(torch.float32)
    if (dz_fp8_out is None) != (dz_scale_out is None):
        raise ValueError("fused dz quant requires both dz_fp8_out and dz_scale_out")
    if dz_fp8_out is not None:
        if fp8_mode:
            raise ValueError("fused dz quant supports only the BF16 preact path")
        rows, width = int(D_packed.shape[0]), 2 * int(D_packed.shape[1])
        if D_packed.ndim != 2 or width % 128:
            raise ValueError("fused dz quant needs rank-2 dz with width divisible by 128")
        if int(tuple(kwargs.get("tile_shape", (128, 128)))[1]) % 64:
            raise ValueError("fused dz quant needs tile N divisible by 64")
        if (dz_fp8_out.dtype != torch.float8_e4m3fn or tuple(dz_fp8_out.shape) != (rows, width)
                or not dz_fp8_out.is_contiguous()):
            raise ValueError(f"dz_fp8_out must be contiguous E4M3 {(rows, width)}")
        if (dz_scale_out.dtype != torch.uint8 or tuple(dz_scale_out.shape) != (rows, width // 128)
                or not dz_scale_out.is_contiguous()):
            raise ValueError(f"dz_scale_out must be contiguous uint8 {(rows, width // 128)}")
        GemmCls = Sm90ScaledDGatedDzQuant
        epi.update(
            mDzFp8=_epi_operand(dz_fp8_out.view(torch.int32), varlen_m=True),
            mDzScale=_epi_operand(dz_scale_out, varlen_m=True),
        )
    if y1s_col_out is not None or dz_col_out is not None:
        if fp8_mode or not varlen_m or kwargs.get("pingpong"):
            raise ValueError("fused col quant needs the BF16 preact varlen-M non-pingpong path")
        if int(tuple(kwargs.get("tile_shape", (128, 128)))[0]) != 128:
            raise ValueError("fused col quant needs tile M == 128 (one 1x128 group per tile)")
        rows = int(D_packed.shape[0])
        for name, out, feat in (("y1s_col_out", y1s_col_out, int(PostAct.shape[1])),
                                ("dz_col_out", dz_col_out, 2 * int(D_packed.shape[1]))):
            if out is None:
                continue
            data, scale = out
            if name == "dz_col_out" and dz_col_inplace:
                # Tile-local into the preact itself (see TileLocalCol): no [F, M] buffer.
                if (data is not PreAct or tuple(data.shape) != (rows, feat)
                        or not data.is_contiguous() or data.dtype != torch.bfloat16):
                    raise ValueError(f"in-place dz_col_out data must be the contiguous "
                                     f"bf16 preact [{rows}, {feat}]")
            elif (data.dtype != torch.float8_e4m3fn or tuple(data.shape) != (feat, rows)
                    or not data.is_contiguous()):
                raise ValueError(f"{name} must be (E4M3 [{feat}, {rows}], u8 [>={rows // 128}, {feat}])")
            if (rows % 128 or scale.dtype != torch.uint8
                    or scale.ndim != 2 or int(scale.shape[1]) != feat
                    or int(scale.shape[0]) < rows // 128 or not scale.is_contiguous()):
                raise ValueError(f"{name} must be (E4M3 [{feat}, {rows}], u8 [>={rows // 128}, {feat}])")
        GemmCls = Sm90ScaledDGatedColQuant
        for key, out in (("mY1sCol", y1s_col_out), ("mDzCol", dz_col_out)):
            if out is not None:
                epi[key] = _epi_operand(out[0].view(torch.int32))
                epi[key + "Scale"] = _epi_operand(out[1])
        skip = tuple(sorted(skip_stores))
        if dz_col_inplace and (skip != ("dz", "y1s") or dz_col_out is None):
            # The dropped dz TMA store is what keeps the even subtile's preact free.
            raise ValueError("dz_col_inplace needs dz_col_out and skip_stores=('y1s', 'dz')")
        if skip == ("y1s",) and y1s_col_out is not None:
            GemmCls = Sm90ScaledDGatedColQuantNoY1s
        elif (skip == ("dz", "y1s") and y1s_col_out is not None and dz_col_out is not None
              and dz_fp8_out is not None):
            GemmCls = (Sm90ScaledDGatedColQuantNoStoreInplace if dz_col_inplace
                       else Sm90ScaledDGatedColQuantNoStore)
        elif skip:
            raise ValueError(f"unsupported skip_stores={skip_stores!r}")
    elif skip_stores or dz_col_inplace:
        raise ValueError("skip_stores / dz_col_inplace need the fused col quant")
    _run_scaled(
        GemmCls, A, B, D_packed, a_scales, b_scales, C=C, epi_fields=epi,
        implicit_dtype=output_dtype, **kwargs,
    )
    return D


def _run_scaled_wgrad_triton(A, B, D, a_scales, b_scales, cu_seqlens_k, C, beta, add):
    """Single-launch SM90 varlen-K wgrad fallback over raw K-major payloads."""
    grid = (D.shape[0], triton.cdiv(D.shape[1], 16), triton.cdiv(D.shape[2], 16))
    _sm90_raw_wgrad_kernel[grid](
        A, B, D, a_scales, b_scales, cu_seqlens_k,
        C if C is not None else D,
        D.shape[1], D.shape[2], A.shape[1],
        D.stride(0), D.stride(1), D.stride(2),
        C.stride(0) if C is not None else D.stride(0),
        C.stride(1) if C is not None else D.stride(1),
        C.stride(2) if C is not None else D.stride(2),
        float(beta or 0.0),
        has_c=C is not None,
        add_to_output=bool(add),
        block_m=16,
        block_n=16,
        block_k=32,
        num_warps=4,
    )
    return D


@triton.jit
def _sm90_raw_wgrad_kernel(
    a_ptr, b_ptr, d_ptr, sa_ptr, sb_ptr, offsets_ptr, c_ptr,
    M, N, total_k, d_stride_e, d_stride_m, d_stride_n,
    c_stride_e, c_stride_m, c_stride_n, beta,
    has_c: tl.constexpr, add_to_output: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    expert = tl.program_id(0)
    m = tl.program_id(1) * block_m + tl.arange(0, block_m)
    n = tl.program_id(2) * block_n + tl.arange(0, block_n)
    start = tl.load(offsets_ptr + expert)
    end = tl.load(offsets_ptr + expert + 1)
    acc = tl.zeros((block_m, block_n), tl.float32)
    for k0 in range(start, end, block_k):
        k = k0 + tl.arange(0, block_k)
        a = tl.load(a_ptr + m[:, None] * total_k + k[None, :],
                    mask=(m[:, None] < M) & (k[None, :] < end), other=0.0)
        b = tl.load(b_ptr + n[:, None] * total_k + k[None, :],
                    mask=(n[:, None] < N) & (k[None, :] < end), other=0.0)
        group = k0 // 32
        sa = tl.load(sa_ptr + m * (total_k // 32) + group, mask=m < M, other=127)
        sb = tl.load(sb_ptr + n * (total_k // 32) + group, mask=n < N, other=127)
        scale_a = tl.exp2(sa.to(tl.float32) - 127.0)
        scale_b = tl.exp2(sb.to(tl.float32) - 127.0)
        acc += tl.dot(a, b.trans()) * scale_a[:, None] * scale_b[None, :]
    d_offsets = expert * d_stride_e + m[:, None] * d_stride_m + n[None, :] * d_stride_n
    mask = (m[:, None] < M) & (n[None, :] < N)
    if has_c:
        c_offsets = expert * c_stride_e + m[:, None] * c_stride_m + n[None, :] * c_stride_n
        acc += tl.load(c_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32) * beta
    if add_to_output:
        acc += tl.load(d_ptr + d_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(d_ptr + d_offsets, acc, mask=mask)


def _validate_packed_wgrad_inputs(A, B, D, a_scales, b_scales, cu_seqlens_k, C, group=32):
    """Validate the direct rank-2 TMA ABI before compiling or launching wgrad."""
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("SM90 scaled varlen-K expects A=[M,total_packed_K], B=[N,total_packed_K]")
    if not A.is_contiguous() or not B.is_contiguous() or A.stride(1) != 1 or B.stride(1) != 1:
        raise ValueError("SM90 scaled varlen-K operands must be contiguous physical K-major matrices")
    if D.ndim != 3 or tuple(D.shape[1:]) != (A.shape[0], B.shape[0]):
        raise ValueError(f"D must have shape [experts,{A.shape[0]},{B.shape[0]}]")
    if not D.is_contiguous():
        raise ValueError("SM90 scaled varlen-K D must be contiguous")
    if C is not None and (tuple(C.shape) != tuple(D.shape) or not C.is_contiguous()):
        raise ValueError("wgrad C must match D shape and be contiguous")
    if cu_seqlens_k.ndim != 1 or str(cu_seqlens_k.dtype) not in (
        "torch.int32", "paddle.int32", "int32",
    ):
        raise TypeError("cu_seqlens_k must be rank-1 int32")
    if cu_seqlens_k.device != A.device or B.device != A.device or D.device != A.device:
        raise ValueError("wgrad operands, output, and packed offsets must share one CUDA device")
    if C is not None and C.device != A.device:
        raise ValueError("wgrad C must be on the operand CUDA device")
    if group == 128:
        # 1x128: segments come 128-aligned from the routing metadata
        # (routing_block_size = 128); offsets stay on the device (no sync).
        # Scale shapes are checked by _run_scaled.
        if int(A.shape[1]) % 128 or int(cu_seqlens_k.shape[0]) != int(D.shape[0]) + 1:
            raise ValueError("1x128 wgrad needs total_K % 128 == 0 and len(offsets) == E + 1")
        if _SM90_WGRAD_VALIDATE:
            offsets = [int(v) for v in cu_seqlens_k.detach().cpu().tolist()]
            if (offsets[0] != 0 or offsets[-1] > int(A.shape[1])
                    or any(r < l or (r - l) % 128 for l, r in zip(offsets, offsets[1:]))):
                raise ValueError(f"1x128 wgrad offsets must be 128-aligned in [0,{A.shape[1]}]")
        return
    offsets = [int(value) for value in cu_seqlens_k.detach().cpu().tolist()]
    total_k = int(A.shape[1])
    if len(offsets) != int(D.shape[0]) + 1:
        raise ValueError("packed offsets length must equal D.shape[0] + 1")
    if not offsets or offsets[0] != 0 or offsets[-1] != total_k:
        raise ValueError(f"packed offsets must span exactly [0,{total_k}]")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("packed offsets must be monotonically nondecreasing")
    if any((right - left) % 32 for left, right in zip(offsets, offsets[1:])):
        raise ValueError("every packed expert K span must be divisible by 32")
    groups = total_k // 32
    expected_a_scales = (int(A.shape[0]), groups)
    expected_b_scales = (int(B.shape[0]), groups)
    if tuple(a_scales.shape) != expected_a_scales or tuple(b_scales.shape) != expected_b_scales:
        raise ValueError(
            f"raw scales must have shapes {expected_a_scales} and {expected_b_scales}"
        )
    if not a_scales.is_contiguous() or not b_scales.is_contiguous():
        raise ValueError("SM90 scaled varlen-K raw scales must be contiguous")


def sm90_scaled_wgrad_varlen_k(
    A, B, D, a_scales, b_scales, cu_seqlens_k, *, C=None, beta=None,
    add_to_output=False, group=32, a_tile_local=False, **kwargs,
):
    """Varlen-K Hopper wgrad over contiguous K-major raw operands.

    ``A`` and ``B`` are E4M3 ``[M,total_K]`` and ``[N,total_K]``.
    ``group=32``: raw UE8M0 scales are uint8 ``[M,total_K/32]`` and
    ``[N,total_K/32]``; scale column zero for expert ``e`` is
    ``cu_seqlens_k[e] / 32``.
    ``group=128``: scales are k-group-major uint8 ``[total_K/128 + pad, M]``
    and ``[total_K/128 + pad, N]`` (pad >= 256 bytes); expert segments are
    128-aligned; one tensor-core promotion per 128 K.
    No group crosses an expert boundary. ``A_idx`` is intentionally
    unsupported because gather and transpose happen in the colwise quantizer.
    ``a_tile_local=True`` (group 128): ``A`` is the bf16 ``[total_K, M]``
    buffer holding the tile-local E4M3 operand written by
    ``sm90_scaled_dgated(dz_col_inplace=True)``.
    """
    if group not in (32, 128):
        raise ValueError(f"unsupported SM90 wgrad group {group}")
    if a_tile_local:
        k, m = int(A.shape[0]), int(A.shape[1])
        if (group != 128 or A.ndim != 2 or not _same_dtype(A.dtype, torch.bfloat16)
                or not A.is_contiguous() or k % 128 or m % 128):
            raise ValueError("a_tile_local needs group 128 and a contiguous bf16 "
                             "[total_K % 128 == 0, M % 128 == 0] buffer")
        A = A.view(-1).view(torch.uint8)[: m * k].view(m, k).view(torch.float8_e4m3fn)
    _validate_packed_wgrad_inputs(A, B, D, a_scales, b_scales, cu_seqlens_k, C, group)
    if add_to_output and C is not None:
        raise ValueError("add_to_output uses D as the accumulator and cannot also take C")
    if add_to_output and beta not in (None, 1, 1.0):
        raise ValueError("add_to_output is equivalent to beta=1 on D")
    if C is None and beta not in (None, 0, 0.0):
        raise ValueError("beta requires C")
    if C is not None and C.dtype != D.dtype:
        raise TypeError("wgrad C and D dtypes must match")
    debug_triton = kwargs.pop("debug_triton_fallback", False)
    if "force_cute" in kwargs:
        # Preserve call compatibility while making CuTe the production default.
        force_cute = kwargs.pop("force_cute")
        if not force_cute:
            raise ValueError("force_cute=False is not a production mode; use debug_triton_fallback")
    if debug_triton:
        if group != 32:
            raise ValueError("debug_triton_fallback supports the 1x32 wgrad ABI only")
        return _run_scaled_wgrad_triton(
            A, B, D, a_scales, b_scales, cu_seqlens_k,
            None if add_to_output else C, beta, add_to_output,
        )
    epi = dict(
        beta=float(beta) if C is not None and beta is not None else None,
        add_to_output=bool(add_to_output),
    )
    if group == 128:
        kwargs.setdefault("tile_shape", _SM90_WGRAD128_TILE)
        kwargs.setdefault("cluster_shape", _SM90_WGRAD128_CLUSTER)
    return _run_scaled(
        Sm90ScaledWgradVarlenKTileLocal if a_tile_local else Sm90ScaledWgradVarlenK,
        A, B, D, a_scales, b_scales,
        C=None if add_to_output else C, epi_fields=epi, cu_seqlens_k=cu_seqlens_k,
        activation_group=group, weight_block=group, **kwargs,
    )


def sm90_scaled_recompute_z(
    A, B, D, a_scales, b_scales, cu_seqlens_m, A_idx, *, z_scale_out=None, **kwargs
):
    """Gather + varlen-M recompute with BF16 or direct E4M3/raw-scale output."""
    kwargs.setdefault("activation_scale_order", "source")
    if D.dtype == torch.float8_e4m3fn:
        if z_scale_out is None:
            raise ValueError("E4M3 recompute output requires raw z_scale_out")
        dummy_postact = torch.empty(
            (D.shape[0], D.shape[1] // 2), dtype=torch.bfloat16, device=D.device
        )
        return sm90_scaled_gated(
            A, B, D, dummy_postact, a_scales, b_scales,
            z_scale_out=z_scale_out, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx, **kwargs,
        )
    if z_scale_out is not None:
        raise ValueError("BF16 recompute output forbids z_scale_out")
    return _run_scaled(
        Sm90ScaledRecomputeZ, A, B, D, a_scales, b_scales,
        cu_seqlens_m=cu_seqlens_m, A_idx=A_idx, **kwargs,
    )


__all__ = [
    "InstrumentedCompileCache", "Sm90ScaledGemm", "Sm90ScaledGated",
    "Sm90ScaledGatedZQuant", "Sm90ScaledGatedY1Quant", "Sm90ScaledGatedZY1Quant",
    "Sm90ScaledDGated", "Sm90ScaledDGatedFP8PreAct",
    "Sm90ScaledWgradVarlenK", "Sm90ScaledRecomputeZ",
    "sm90_scaled_gemm", "sm90_scaled_gated", "sm90_scaled_dgated",
    "sm90_scaled_wgrad_varlen_k", "sm90_scaled_recompute_z",
]
