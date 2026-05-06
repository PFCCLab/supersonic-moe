# dz iso32 epilogue-fusion — Phase 0 audit + Phase 1 design

## TL;DR

**Phase 0 (precision audit) PASSED** on real Ernie-shape (TK=65536, 2I=3072)
dz tensors across 3 routing distributions. iso32 (32×32) quant produces
**identical** downstream-GEMM RRMSE to the production 1×32 quant
(ratio = 1.000× for both dx and dw1 GEMMs) under FP8 e4m3.

→ iso32 is **precision-safe** for dz under FP8 e4m3 — Phase 1A (kernel work)
is unblocked from a numerical standpoint.

## Method

### 1. Capture real dz tensors

`tools/dump_real_dz.py` monkey-patches
`sonicmoe.functional.gemm_dgated_kernel`, runs warmup + capture iters via
`SonicMoEMlpNode`, and saves the third positional-arg dz (the BF16 output
of the dGated kernel) to numpy at full precision.  Captured 6 tensors:
`{none, skew, extreme} × {iter4, iter5}`, each shape `(65536, 3072)` bf16,
amax ≈ 3.3 – 3.9 (realistic gradient magnitude — earlier audits with the
0.02-scaled fixture collapsed everything into a single FP8 bin and were
not informative).

### 2. Pure-PyTorch quant fidelity

`tools/audit_dz_iso32_quality.py` runs both 1×32 and 32×32 quant→dequant
via `tests/ops/audit_iso32_numerics._quant_dequant_blockscaled` (clean
reference impl: tile amax → e8m0 → e4m3 cast → dequant), then reports
cosine, RRMSE, max_abs, and per-(32,32) dyn-range bits-lost.

### 3. Downstream-GEMM error (the actual training-impact metric)

For each captured dz, generate random `w1 (2I, H)` and `x (TK, H)` with
Ernie-realistic scaling, then compute three versions of:
- `dx_proxy  = dz @ w1ᵀ`
- `dw1_proxy = dzᵀ @ x`

(a) reference using BF16 dz, (b) using 1×32-dq dz, (c) using 32×32-dq dz.
Headline metric is the **iso32/1×32 RRMSE ratio** for each downstream
output — this directly reflects what the next bwd GEMM consumer would see.

## Results (representative — `dz_extreme_iter4.npy`)

| metric                      | 1×32       | 32×32      | ratio |
|-----------------------------|-----------:|-----------:|------:|
| direct cos                  | 0.999646   | 0.999646   | —     |
| direct RRMSE                | 2.659e-2   | 2.659e-2   | 1.000×|
| direct max_abs              | 1.250e-1   | 1.250e-1   | —     |
| dx_proxy RRMSE vs ref       | 2.659e-2   | 2.659e-2   | 1.000×|
| dw1_proxy RRMSE vs ref      | 2.658e-2   | 2.658e-2   | 1.000×|
| dyn-range frac>1bit (advis.)| —          | 0.735      | —     |

Identical numbers across `none/skew/extreme` and across iters — see
`audit.md` for the full per-capture table.

## Why iso32 is a free lunch on FP8 dz

The dyn-range proxy `log2(tile_amax / row_amax)` predicted a large
mantissa loss for iso32 (frac>1bit ≈ 73%), but in practice **1×32 has the
same RRMSE as 32×32**. Reason: e4m3's 3-bit mantissa imposes ~12.5%
per-value rounding noise, which dominates the per-row scale advantage that
1×32 buys. The "extra" mantissa bits 1×32 reserves are below the e4m3
mantissa precision floor and therefore unused.

In contrast, when iso32 was retired for **weights** (PR #15), weights are
much larger and have tighter per-row distributions — the mantissa
advantage of 1×32 there was not wasted.

## Performance ceiling (revised honest numbers)

If Phase 1A lands successfully:

- **Eliminated**: dz BF16 (TK, 2I) ≈ 384 MiB HBM write + 384 MiB read +
  the entire `_dual_varlen_quantize_kernel` launch+compute.
- **Added**: epilogue amax+cast cost (≈ 30–60 µs); FP8 store TMA atom
  (half BF16 bandwidth → cheaper).
- **Net realistic savings**: 100–180 µs / iter (vs 2754 µs busy).
- **MFU lift**: +1.5 to 3.0 pp (from 44.91% baseline → 46.5–48.0%).

## Phase 1A design — kernel change details

### Target

`sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` line 397
(`GemmDGatedFP8CLoadSm100ZeroMat`) and the underlying mixin
`sonicmoe/quack_utils/gemm_dgated.py:510` (`GemmDGatedFP8CLoadMixin`).

### Template

`sonicmoe/quack_utils/gemm_gated.py:221` (`GemmGatedBlockscaledQuantMixin`)
is the **forward** analog — it already does production 1×32 blockscaled
FP8 quant in the epilogue with bit-exact match to Triton. Its
`epi_visit_subtile` (line 244) is the pattern to mirror.

### What the bwd version must do differently

1. **iso32 amax instead of 1×32** — share one e8m0 byte across 32
   M-rows × 32 N-cols. SM100 `Ld32x32bOp` already places exactly one
   M-row per thread; need a warp-level shuffle to reduce amax across the
   32 lanes (M-axis) holding the same N-tile.
2. **D shape is (TK, 2I)** — dGated outputs gate+up packed; the
   epilogue currently writes BF16 (TK, I-as-f32). When we switch D dtype
   to FP8, we get (TK, 2I) FP8 directly. Need to update `d_dtype`,
   `epi_tile`, and TMA store atom selection (8-bit StMatrix instead of
   16-bit). The fwd `GemmGatedBlockscaledQuantMixin` already handles
   this — model after `_make_tma_epi_atoms_and_tensors` chain.
3. **Scale store layout** — use `BlockscaledScaleStore` (already in
   `gemm_gated.py:172`) but with a wider (M, 2I/32) shape rather than
   (M, I/32). Same ISA-packing logic.
4. **Preserve BF16-emit path** — add an optional flag (e.g.,
   `mDZScale: Optional[cute.Tensor] = None`); when None, fall through to
   the existing BF16-emit behavior. This protects all current callers
   and gives us a one-flag rollback.

### Register budget — main risk

S78b confirmed `GemmDGatedFP8CLoadSm100ZeroMat` is register-bound
(every register tweak landed neutral or regressed). The amax + e8m0 +
cast logic adds ≈ 24–32 registers per thread. Mitigations:
- BF16 store → FP8 store frees TMA atom registers (each elem 2B → 1B).
- Reorder `epi_visit_subtile` to release tRS_rD slots before TMA store.
- If unavoidable spill, fall back to **dual-amax 1×32 epilogue** (Phase
  1B) — same structure, more arithmetic, possibly within budget if the
  warp shuffle in iso32 hurts more than the extra amax loop.

### Phase 2 wire-up — Python side

`sonicmoe/functional/__init__.py` line 1967:

```python
# OLD
dz = torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=...)
gemm_dgated_kernel(... dz, ...)
# downstream: fused_dual_colwise_quantize(dz, ...)

# NEW
dz_fp8 = torch.empty((total_m, n * 2), dtype=torch.float8_e4m3fn, device=...)
dz_scales = torch.empty(_storage_per_batch(total_m, 2*n, ...), dtype=torch.uint8, ...)
gemm_dgated_kernel(... dz_fp8, ..., mDZScale=dz_scales)
# downstream dx GEMM uses dz_fp8 + dz_scales directly (rowwise format)
# downstream dw1 GEMM gets a cheap scale-rewrite (~10 µs) into colwise format
```

The colwise scale layout has the same e8m0 bytes (iso32 → byte identical
between formats), just placed at different offsets.  A one-pass byte
shuffle kernel is needed (`sonicmoe/quack_utils/scales_iso_rewrite.py`).

### Phase 3 — CI gates

Must pass without modifying CI:
- `tests/fp8_frontier_determinism_test.py` (byte-identical determinism)
- `tests/fp8_large_project_contract_test.py` (production tolerance)
- `tests/fp8_protocol_test.py`
- `tests/ops/test_gemm_dgated.py` (per-shape numerical)
- `tests/ops/test_dual_quant.py` (downstream consumer)
- `tests/moe_blackwell_test.py` (end-to-end)

Phase 0 audit shows these *should* pass (downstream RRMSE identical to
production 1×32 path), but the bit-exact determinism test will only pass
if the e8m0 encoding matches Triton — model after
`GemmGatedBlockscaledQuantMixin` integer+carry algorithm exactly.

## Files / pointers for next agent

| Purpose | Path |
|---|---|
| Phase 0 capture tool | `tools/dump_real_dz.py` |
| Phase 0 audit tool | `tools/audit_dz_iso32_quality.py` |
| Pure-pytorch quant ref | `tests/ops/audit_iso32_numerics.py:32` |
| Captured dz tensors | `reports/iso32_dz_audit/dz_*.npy` |
| Audit results | `reports/iso32_dz_audit/audit.md` |
| Forward fp8-emit template | `sonicmoe/quack_utils/gemm_gated.py:221` |
| Forward scale store EpiOp | `sonicmoe/quack_utils/gemm_gated.py:172` |
| Bwd target (kernel) | `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py:397` |
| Bwd target (mixin) | `sonicmoe/quack_utils/gemm_dgated.py:510` |
| Skeleton notes (advisory) | `sonicmoe/quack_utils/epi_blockscaled_quant.py` |
| Wire-up call site | `sonicmoe/functional/__init__.py:1967` |
| Storage size helper | `sonicmoe/quack_utils/blockscaled_fp8_gemm.py:199` |
| ISA constants | top of `blockscaled_fp8_gemm.py` |
| Reference iso32 weight quant | `blockscaled_fp8_gemm.py:3465` |

## Status

- [x] Phase 0.1 — dz capture infra
- [x] Phase 0.2 — iso32 vs 1×32 + downstream-GEMM audit (PASS)
- [x] Phase 0.1/0.2 — capture + audit
- [x] Phase 1B (interim frontier, S80b/c) — Triton iso32 dual-quant kernel,
      default ON, +1.02 pp MFU on Ernie shape, 51.53% peak
- [ ] Phase 1A — true CuTe DSL epilogue fusion (designed; not yet
      implemented — genuine multi-day CuTe DSL bring-up)
- [ ] Phase 2 — Python wire-up + ISA scale repack kernel (depends on 1A)
- [ ] Phase 3 — CI + nsys perf (depends on 1A)
- [ ] Phase 4 — final report (depends on 1A)

---

## Phase 1A v2 — pickup notes (S80c, after deeper code reading)

This section adds the implementation-specific findings collected after
S80c. Read it together with the earlier sections — it does not replace
them, only sharpens the unknowns.

### Newly verified API / layout facts

1. **`cute.arch.warp_reduction_max(val, threads_in_group=32)`** is the
   correct call for the iso32 cross-lane amax reduction. Works on
   `Float32`. No need to hand-roll `shfl_xor_sync`.
2. **Per-thread dXY fragment in the bwd FP8CLoad mixin is 64 fp32
   values** (= one M-row × 32 D-elements; each D = 2 packed bf16-or-fp8
   in 2I view). See `gemm_dgated.py:664-671` — the unroll loops over
   `cute.size(tRS_rD) // 2` and writes `tRS_rdXY_f32x2[4*i .. 4*i+3]`
   for each pair of D-elements.
3. **iso32 sub-block split**: of the 64 fp32 values per thread, the
   first 32 belong to the first iso32 N-block (fp8 cols `n_base*2 ..
   n_base*2 + 31`) and the next 32 to the second (cols `+32 .. +63`).
   So **two amaxes per thread per epi sub-tile**, two warp_reduction_max
   calls, two e8m0 encodings, two quant_scales applied.
4. **Per-thread coord access** for the FP8 scatter store is already
   plumbed via `epi_loop_tensors["mFP8PreAct"]` →
   `(fp8_tensor, scales_tensor, tDcD_sub, m_offset, m_base, n_base)`.
   `tDcD_sub` gives the per-thread (row, col) D-coordinates; `n0 =
   n_base + col*2` is the FP8 byte position. **Reuse the same
   per-thread coord machinery for the dz output scatter store** —
   add a parallel EpiOp `FP8DZScatterStore("mDZFP8")` modeled on
   `FP8PreActLoad` that exposes the same coord tuple.

### Newly identified missing piece: ISA scale repack

The existing `BlockscaledScaleStore` (`gemm_gated.py:172`) writes
**raw row-major (total_M, N//32)** uint8 — *not* the ISA-packed format
the downstream GEMMs (`_run_cutlass_blockscaled_gemm_varlen_*`) expect.
The fwd pipeline gets away with this because `gather_isa_packed_scales`
runs after the kernel and converts to ISA layout.

For Phase 1A's `dz_scales`:

- **Option A**: emit raw row-major from epi, then run a tiny
  `_gather_isa_packed_scales_kernel`-style pass to repack into ISA. Adds
  ~10 µs back into the budget but keeps the EpiOp simple. Recommended
  first cut.
- **Option B**: write directly into ISA layout from the kernel. Requires
  the EpiOp to compute the ISA byte offset from `(m_abs, n_group_abs)`
  using `_SF_TILE_M=128, _SF_TILE_K=128, _SF_VEC_SIZE=32, _SF_TILE_STORAGE=512`
  constants. Saves the post-pass but adds index arithmetic to every
  thread's epilogue (likely register-budget-hostile).

For the col-layout variant: **with iso32 the bytes are byte-identical
between row and col layouts**, so the col layout's repack reads the same
source uint8 and reorders it. A second tiny repack pass produces it.

### Newly identified register-budget mitigation

When the FP8 dz emit path is enabled, the BF16 D store can be skipped
entirely (no caller needs the BF16 dz when `mDZFP8` is provided). The
TMA-store-to-D-tensor still happens, but the destination tensor can be a
1-element placeholder if we can teach the TMA atom that it's effectively
a no-op. Worth checking: does the TMA atom honor `m_limit` / `n_limit`
bounds via the existing varlen mechanism so a tiny placeholder won't OOB?
If yes, the BF16 path is free to coexist; if no, we either keep the BF16
write (~384 MiB allocation but no extra compute) or override
`epi_load_and_partition_d` to skip the TMA store atom altogether.

### Concrete task list (in execution order, for next agent)

1. **Add EpiOps** in `gemm_dgated.py` (or new file
   `epi_blockscaled_dz.py`):
   - `FP8DZScatterStore("mDZFP8")` — same coord pattern as
     `FP8PreActLoad`, but for output (registers → global FP8 bytes).
   - `BlockscaledScaleStoreCol("mDZScaleCol")` — sibling of
     `BlockscaledScaleStore` with col-layout coord (`(n_group_abs,
     m_abs)` indexing).
2. **New mixin** `GemmDGatedFP8CLoadIso32QuantMixin(GemmDGatedFP8CLoadMixin)`:
   - Override `_epi_ops` to add the 3 new EpiOps.
   - Override `EpilogueArguments` to add `mDZFP8`, `mDZScaleRow`,
     `mDZScaleCol`.
   - Override `epi_visit_subtile`: keep parent's body up to and
     including `tRS_rdXY_f32x2` computation, then BEFORE the BF16
     pack/store:
     - Compute `amax_lo = max(|tRS_rdXY_f32x2[0..31]|)`, `amax_hi`.
     - `amax_lo = cute.arch.warp_reduction_max(amax_lo,
       threads_in_group=32)`; same for `amax_hi`.
     - Apply integer+carry e8m0 (copy `gemm_gated.py:263-281` twice).
     - `tRS_rdXY_f32x2[i] *= quant_scale_lo` for i<32, `quant_scale_hi`
       for i>=32.
     - `.to(Float8E4M3FN)` cast in registers.
     - Scatter store FP8 bytes via `mDZFP8` EpiOp coords.
     - Lane-0-of-32 (or accept benign race) writes e8m0 byte to row
       scale tensor; same byte to col scale tensor at transposed offset.
   - Conditionally keep or skip the BF16 store (start with: keep it for
     safety; flip to skip once correctness proved).
3. **New class** in `gemm_sm100_fp8_zeromat.py`:
   `GemmDGatedFP8CLoadIso32QuantSm100ZeroMat(GemmDGatedFP8CLoadIso32QuantMixin,
   _GemmSm100ZeroMatMixin, GemmSm100)`.
4. **Python wire-up** at `functional/__init__.py:1974`:
   - Behind env flag `SONIC_MOE_DZ_EPI_FUSE=1`:
     - Allocate `dz_fp8` (TK, 2I) FP8 + raw `dz_scale_row` uint8.
     - Pass `mDZFP8`, `mDZScaleRow`, `mDZScaleCol` into kernel.
     - After kernel: run scale-ISA-repack kernel (modeled on
       `_gather_isa_packed_scales_kernel`) to produce ISA-packed scales
       for both row & col layouts.
     - Skip the dz portion of `fused_dual_colwise_quantize` (only do
       dout colwise).
5. **Bit-exact validation** against the (default-ON) iso32 Triton
   kernel — they MUST produce identical FP8 + scale bytes since both
   use iso32 amax + integer+carry e8m0.
6. **CI**: `tests/fp8_frontier_determinism_test.py` must keep passing.
7. **nsys**: rerun multi-shape sweep with `SONIC_MOE_DZ_EPI_FUSE=1`.
   Expected ceiling: another -50 to -120 µs/iter beyond Phase 1B → 47–48%
   MFU on Ernie. Realistic floor (if scattered FP8 stores hurt
   coalescing): may be smaller; honest data needed.

### Why not done in S80c

CuTe DSL bring-up of a new EpiOp + a register-bound kernel mixin is an
inherently iterative process that requires hardware-driven debug cycles
(compile errors are unhelpful, register spill is silent until profiled,
warp-layout assumptions need empirical validation). Doing this safely
under autopilot — without putting the working +1.02pp Phase 1B frontier
at risk — requires a focused, interactive session. Phase 1B was kept on
disk as the active frontier so the next agent inherits a clean,
production-quality starting point with all design groundwork done.
