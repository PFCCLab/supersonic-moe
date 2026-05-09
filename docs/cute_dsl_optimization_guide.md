# CuTe DSL FP8 Quantization — Architecture & Optimization Guide

> Session 43 accumulated knowledge for next agent. All claims NCU-verified on Target GPU.

## 1. Project Architecture Overview

### Data Flow: FP8 MoE Forward + Backward

```
Forward:
  x (T,H) bf16
    → iso32 weight quant: w1 (E,2I,H) bf16 → fp8 + ISA scales (cached)
    → x quant+gather: x (T,H) → x_fp8 (TK,H) + ISA scales
    → GemmGated (FP8 zero-mat): x_fp8 @ w1_fp8 → z (TK,2I) bf16 + y1 (TK,I) bf16
    → z blockscaled quant: z (TK,I) bf16 → z_fp8 + ISA scales (saved for bwd)
    → y1 ISA pack: y1 (TK,I) bf16 → y1_fp8 + ISA scales
    → DownProj (FP8 varlen): y1_fp8 @ w2_fp8 → output (TK,H) bf16
    → scatter back to (T,H)

Backward:
  dout (T,H) bf16
    → dout quant+gather → dout_fp8 (TK,H) + ISA scales
    → DGated (FP8 zero-mat + FP8CLoad): dout_fp8 @ w2^T → dz (TK,I) bf16
    → dw2 wgrad: dz^T @ y1 (BF16 GEMM, not FP8)
    → dz dual quant (split strategy):
        CuTe colwise: dz (TK,I) → col_fp8 + ISA scales (for wgrad)
        Triton rowwise: dz (TK,I) → row_fp8 + ISA scales (for actgrad)
    → dw1 wgrad (FP8): col_fp8(dz) @ col_fp8(x) via CUTLASS varlen_k
    → actgrad: row_fp8(dz) @ w1_fp8 via CUTLASS varlen
    → scatter dx back to (T,H)
```

### Key Tensor Shapes (Ernie)

| Tensor | Shape | Size (bf16) | Size (fp8) |
|--------|-------|:-----------:|:----------:|
| x | (8192, 3072) | 48 MiB | — |
| x gathered | (65536, 3072) | 384 MiB | 192 MiB |
| w1 (gated) | (8, 3072, 3072) | 144 MiB | 72 MiB |
| w2 | (8, 3072, 1536) | 72 MiB | 36 MiB |
| z | (65536, 1536) | 192 MiB | 96 MiB |
| y1 | (65536, 1536) | 192 MiB | 96 MiB |

### ISA Scale Packing (SM100 Blockscaled MX Format)

```
SF_TILE_M = 128, SF_TILE_K = 128, SF_TILE_STORAGE = 512 bytes
GROUP_SIZE = 32 (1×32 blockscaled)

For scale at logical position (row, group_idx):
  row_tiles = row // 128
  row_in_tile = row % 128
  k_tiles_idx = group_idx // 4    (4 groups per K-tile = 128/32)
  k_in_tile = group_idx % 4
  tile_base = (row_tiles * k_tiles + k_tiles_idx) * 512
  row_base = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
  isa_index = tile_base + row_base + k_in_tile
```

## 2. CuTe DSL Kernel Patterns (from QuACK 0.3.7)

### ReductionBase Pattern (rmsnorm.py, softmax.py)
```python
class MyOp(ReductionBase):
    def __init__(self, dtype, N):
        super().__init__(dtype, N, stage=1)

    def _threads_per_row(self): return 8   # 8 threads × vecsize elements = N
    def _num_threads(self): return 256

    @cute.jit
    def __call__(self, mIn, mOut, stream):
        tiled_copy, tiler_mn, tpr = self._get_tiled_copy(vecsize=4)
        self.kernel(mIn, mOut, tiler_mn, tiled_copy, tpr).launch(
            grid=[ceil_div(M, tiler_mn[0]), 1, 1],
            block=[tiled_copy.size, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mIn, mOut, tiler_mn, tiled_copy, tpr):
        # 1. Partition gmem + smem
        # 2. cp.async load
        # 3. autovec_copy smem → rmem
        # 4. .load().to(Float32) → TensorSSA
        # 5. row_reduce(TensorSSA, MAX/ADD/...)
        # 6. Scalar ops on result
        # 7. Store back
```

### TensorSSA Operations (what works on vector values)
```python
x = tSrX.load().to(cute.Float32)   # TensorSSA
# Supported:
x * x                    # element-wise multiply
x * scalar               # broadcast multiply
x - scalar               # broadcast subtract
cute.math.exp2(x)        # element-wise exp2
cute.math.rsqrt(x)       # element-wise rsqrt
x.to(cute.Float8E4M3FN)  # vector cast (MUST be 32-bit aligned, ≥4 elements for fp8)
x.reduce(ReductionOp.MAX, init_val=0.0)  # intra-thread reduce

# NOT supported on TensorSSA:
cute.arch.fmax(x, y)     # scalar only
cute.arch.fmin(x, y)     # scalar only
abs(x)                   # no built-in; use rmem loop
```

### FP8 Cast Requirement
```python
# fp8 cast requires 32-bit aligned vector (≥4 fp8 elements)
# WRONG: scalar cast
mFp8[i, j] = val.to(cute.Float8E4M3FN)  # ICE: cvt_fptrunc needs vector

# RIGHT: batch 4 elements via rmem
r4 = cute.make_rmem_tensor(4, cute.Float32)
r4[0] = val0; r4[1] = val1; r4[2] = val2; r4[3] = val3
rFp8 = cute.make_rmem_tensor(4, cute.Float8E4M3FN)
rFp8.store(r4.load().to(cute.Float8E4M3FN))
mFp8[i, j] = rFp8[0]  # then element-wise store is OK
```

### Custom PTX via dsl_user_op
```python
@dsl_user_op
def _abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "abs.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))

@dsl_user_op
def _rcp_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    # Bit-exact for E8M0: mantissa masked, so ≤1 ULP error vanishes
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "rcp.approx.f32 $0, $1;", "=f,f", ...))

@dsl_user_op
def _f32_as_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), ...))
```

### Compilation & Fake Tensors
```python
@jit_cache  # from quack.cache_utils
def _compile(dtype_width, TK, dim):
    dtype = cute.BFloat16 if dtype_width == 16 else cute.Float16
    # Symbolic dimensions
    TK_sym, dim_sym = cute.sym_int(), cute.sym_int()
    # divisibility=8 for bf16 enables 128-bit cp.async (8 × 16 = 128 bits)
    div = 128 // dtype.width
    mSrc = make_fake_tensor(dtype, (TK_sym, dim_sym), divisibility=div)
    # For ISA-packed scales: flat 1D
    mScale = cute.runtime.make_fake_tensor(cute.Uint8, (scale_sym,), stride=(1,), assumed_align=1)
    op = MyOp(dtype, TK)
    return cute.compile(op, mSrc, ...,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi")
```

## 3. NCU-Verified Optimization Techniques

### Bank Conflict Elimination
**Problem:** Triton colwise quant has 6.4-way smem bank conflicts (11.1M conflicts, 84% conflict rate) because column-wise reads from (TK, dim) smem have stride dim×2 bytes between rows → bank collision every 16 rows.

**Solution:** Row-major smem reads. Thread mapping: warp_id = TK group, lane = dim column. Access pattern: `sSrc[group*32+j, lane]` — 32 lanes read 32 consecutive bf16 in the same row = perfectly coalesced, zero bank conflicts.

**Result:** 11.1M → 110K conflicts (101× reduction). MIO stall: 39.8% → 2.5%.

### Register vs Occupancy Tradeoff
| Approach | Regs | Occupancy | GPU time | Winner? |
|----------|:---:|:---:|:---:|:---:|
| Single-pass, fully unrolled (rVals[32]) | 64 | 44% | 101µs | |
| Two-pass, outer unrolled, inner runtime | 48 | 55% | 105µs | |
| **Two-pass, both runtime loops** | **30** | **89%** | **91µs** | **YES** |

**Insight:** For compute-bound kernels on Target GPU, occupancy wins over ILP. 89% occupancy with more warps to hide ALU latency beats 44% occupancy with more ILP per warp.

### Coalesced Scale Store
**Problem:** (dim, num_groups) layout → 32 lanes write stride=num_groups apart = 97% sector waste.
**Solution:** (num_groups, dim) layout → 32 lanes write consecutive dim addresses = coalesced.
**Result:** L1 store traffic 201 MB → 104 MB (-48%).

### rcp.approx for E8M0
```python
# E8M0 only keeps exponent (mantissa = 0). rcp.approx has ≤1 ULP mantissa error.
# After extracting exponent, the mantissa error vanishes → bit-exact.
amax = cute.arch.fmax(amax, Float32(1e-12))
cand_bits = _f32_as_i32(Float32(448.0) * _rcp_approx(amax))
qexp = (cand_bits >> Int32(23)) & Int32(0xFF)
quant_scale = _i32_as_f32(qexp << Int32(23))
e8m0 = Int32(254) - qexp
```
Verified bit-exact across all float32 ranges including near-subnormal (1e-38).

## 4. Colwise Quant Kernel Architecture (cute_blockscaled_quant.py)

```
Tile: (TILE_TK=256, TILE_DIM=32), 256 threads (8 warps)
Grid: (ceil(dim/32), ceil(TK/256))

Phase 1: cp.async vectorized load (tiled_copy_2d, 128-bit)
          gmem (TK, dim) → smem (256, 32) row-major

Phase 2: Per-group amax (runtime loop, no unroll)
          8 warps, each processes 1 group of 32 TK rows
          Lane l reads sSrc[group*32+j, l] → row-major, bank-conflict-free
          amax = max(abs(val)) over 32 iterations

Phase 3: E8M0 via rcp.approx (5 ALU ops)

Phase 4: Scale + fp8 cast + gmem store (runtime loop × 4-unrolled inner)
          Re-read smem, multiply by quant_scale, batch cast 4 fp8 via rmem
          Direct gmem store (not smem-mediated — saves L1 traffic)

Phase 5: ISA scale store (1 byte per group, ISA tile layout)
```

NCU metrics: 90µs, 30 regs, 89% occ, 110K bank conflicts, SM 79%.

## 5. Dual Quant Design Exploration

### Fused Dual Quant (attempted, too slow)
**Goal:** Single HBM read → both row_fp8 and col_fp8.
**Approach:** In the colwise loop (Phase 2), also do per-row warp butterfly shuffle MAX across 32 lanes → row_amax → E8M0 → scale → fp8 store.
**Result:** 300µs (288M instructions, 3.6× bloat). Per-row shuffle (5×fmax) + E8M0 (10 ops) + rmem fp8 cast (4-elem alloc) per iteration = massive overhead.

### Split Strategy (current winner)
CuTe colwise (84µs NCU) + Triton rowwise (62µs) = 146µs.
Beats Triton fused (168µs) by 1.15×.
L2 cache keeps the 2nd read hot (200MB bf16 fits in Target GPU's 50MB L2 partially).

### Paddle Reference (gold standard for fused)
File: `/root/paddlejob/.../Paddle_B/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu`
Key: `quantize_1x128_kernel` — 128×128 tile, 32×16 2D block, all data in registers.
- `ComputeRowScale`: warp shuffle + smem cross-warp tree reduce
- `ComputeColumnScale`: smem tree reduce across 8 iterations
- Transpose output: `shm[128][129]` (+1 padding = bank-conflict-free transpose)
- `fp8x4_t` (32-bit) vector store for coalesced fp8 output
- **All data stays in registers** — zero re-read from smem

To replicate in CuTe DSL: need larger tile (128×128), 2D-like thread mapping via tidx decomposition, register-resident data with smem only for cross-warp reduction.

## 6. Dead Ends & Why

| Approach | Instructions | Regs | L1 store | GPU time | Why failed |
|----------|:---:|:---:|:---:|:---:|:---:|
| Smem fp8 vectorized store | 41.8M (-44%) | 25 | 204 MB (+96%) | 125µs | Double L1 write |
| 1D smem tensor view | 72.9M (-2.5%) | 48 (+60%) | same | 84µs | Tensor state in regs |
| Double-buffer pipeline | similar | 58 (+93%) | same | 101µs | SM-bound, not mem-bound |
| Full loop unroll | similar | 64 (+113%) | same | 101µs | Occupancy 44% |
| [32][33] padded smem | N/A | N/A | N/A | 2994µs | cp.async can't align stride-33 |
| Fused dual (butterfly shuffle) | 288M (3.6×) | 31 | 302 MB | 300µs | Per-row overhead |

## 7. Key File Reference

| File | Purpose |
|------|---------|
| `sonicmoe/quack_utils/cute_blockscaled_quant.py` | CuTe colwise quant (production) |
| `sonicmoe/quack_utils/cute_dual_quant.py` | CuTe dual quant (split entry + fused WIP) |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant kernels, CUTLASS GEMM wrappers |
| `sonicmoe/functional/__init__.py` | Forward/backward orchestration |
| `sonicmoe/moe.py` | MoE class, stash, weight refresh |
| `tools/fp8_frontier_breakdown.py` | Subprocess-isolated 3D analysis |
| `tools/ncu_profile_colwise.py` | NCU comparison script |
| `tests/test_cute_blockscaled.py` | Colwise quant correctness+perf |
| `tests/test_rcp_precision.py` | rcp.approx precision verification |
| `docs/FP8_ARCH_SPEC.md` | Full architecture spec |
| `docs/HANDOFF.md` | Project status + next steps |
| `reports/fp8_upgrade/engineering_log.md` | 31 lessons learned |

## 8. QuACK Library Reference

Location: `/root/paddlejob/.../envs/xfer/lib/python3.13/site-packages/quack/`

| File | What to learn |
|------|--------------|
| `rmsnorm.py` | TensorSSA `x * x` → `row_reduce(ADD)` pattern |
| `softmax.py` | `cute.math.exp2(TensorSSA)`, multi-stage reduce, TensorSSA × scalar |
| `topk.py` | rmem element-by-element loop for ops not vectorized on TensorSSA |
| `reduce.py` | `row_reduce` implementation: intra-thread → warp shuffle → cross-warp smem |
| `reduction_base.py` | `_get_tiled_copy`, `_allocate_reduction_buffer_and_mbar` |
| `copy_utils.py` | `tiled_copy_2d`, `copy`, `predicate_k`, swizzle utilities |
| `layout_utils.py` | `transpose_view` (zero-copy smem transpose) |
| `sm90_utils.py` | `make_smem_layout` with swizzle |
| `gemm_sm90.py` | Full GEMM: TMA, pipeline, producer/consumer, epilogue |
