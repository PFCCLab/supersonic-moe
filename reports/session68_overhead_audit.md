# Session 68 — FP8 Frontier Per-Kernel Overhead Audit + recompute_z Wrap-up

> **Branch:** `myrepo/session60-ds-fix`
> **Shape:** ERNIE — T=8192, K=8, E=8, H=3072, I=1536 (uniform routing)
> **Hardware:** Target GPU (SM100)
> **Method:** nsys 2026.2.1 GPU-projection (merged-overlap kernel intervals
> inside an NVTX `BENCH` range, divided by iteration count). 8 warmup + 20 timed
> iters. Raw `.nsys-rep` files committed alongside this report.

---

## 1. Headline numbers

| Mode | GPU-projection µs/iter | Δ vs baseline | Peak fwd MB |
|------|-----------------------:|--------------:|------------:|
| FP8 frontier baseline (Option A, `recompute_z=False`) | **2864.7** | — | 1751.5 |
| FP8 frontier + `SONIC_MOE_FP8_RECOMPUTE_Z=1` (Option A) | **3311.4** | **+446.7 (+15.6 %)** | **1725.6 (−26 MB)** |

Reference baseline (Session 53, pure-PyTorch FP8 path, no compat shim, no
main-grad accum): **2715 µs/iter**.  Our paddle-compat FP8 frontier is
**5.5 % slower** than that reference — mostly main-grad fused epilogue work.

Recompute_z trades **+447 µs/iter for −26 MB peak fwd**.  Worth it only
when fwd peak is the genuine bottleneck (long-context training, very large
expert counts).

---

## 2. Per-kernel breakdown (baseline, `recompute_z=False`)

Top kernels by total time inside the BENCH window (20 iters):

| Rank | Kernel (shortName) | Calls | µs/iter | Cumulative % |
|----:|---|---:|---:|---:|
| 1 | `quack/GemmDefaultSm100` (wgrad / dgrad-w) | 80 | 1234.1 | 43.2 % |
| 2 | `sonicmoe.../GemmGatedSm100ZeroMatBlockscaledQuant` (fwd up-proj) | 20 | 449.1 | 58.9 % |
| 3 | `sonicmoe.../GemmDGatedFP8CLoad` (bwd down-proj dgrad) | 20 | 399.8 | 72.9 % |
| 4 | `_colwise_quantize_and_pack_kernel` (FP8 transpose+quant) | 60 | 247.5 | 81.5 % |
| 5 | `_dual_varlen_quantize_kernel` (varlen activation FP8 quant) | 20 | 163.1 | 87.2 % |
| 6 | `token_gather_sum_kernel` (combine over topk) | 40 | 148.2 | 92.4 % |
| 7 | `_quantize_and_pack_kernel` (per-token FP8 quant+pack) | 60 | 79.7 | 95.2 % |
| 8 | `VectorizedBroadcastKernel` (paddle elementwise) | 220 | 33.6 | 96.4 % |
| 9 | `_build_score_src_idx_kernel` (routing prep) | 20 | 25.8 | 97.3 % |
| 10 | `prefix_sums_kernel` | 20 | 19.6 | 98.0 % |
| 11 | `index_elementwise_with_tensor_kernel` | 60 | 16.8 | 98.5 % |
| 12 | `_gather_isa_packed_scales_kernel` | 40 | 16.0 | 99.1 % |
| 13 | `VectorizedElementwiseKernel` | 240 | 15.8 | 99.6 % |

**Cutlass GEMMs dominate:** 73 % of GPU time in three kernels (wgrad +
fwd up-proj + bwd down-proj).  All FP8 frontier kernels (fused-epilogue
maingrad-add wgrad, blockscaled gated fwd, blockscaled dgrad with fused
SwiGLU') are confirmed in trace.

**Quant/pack/transpose tax:** `_colwise_quantize_and_pack_kernel` (247) +
`_dual_varlen_quantize_kernel` (163) + `_quantize_and_pack_kernel` (80) +
`_gather_isa_packed_scales_kernel` (16) = **506 µs/iter (17.7 %)**.  This
is the FP8 plumbing cost the user asked about.

---

## 3. Recompute_z delta (`recompute_z=True`)

| Kernel | Baseline µs/iter | recompute µs/iter | Δ |
|---|---:|---:|---:|
| `GemmGatedSm100ZeroMat...` (fwd up-proj) | 449.1 | 900.9 (×2 calls) | **+451.8** |
| `_quantize_and_pack_kernel` (extra x-pack per recompute) | 79.7 | 99.5 | +19.8 |
| `_gather_isa_packed_scales_kernel` | 16.0 | 24.0 | +8.0 |
| Everything else | ~2320 | ~2287 | ~−33 |
| **Total** | **2864.7** | **3311.4** | **+446.7** |

The recompute_z helper (`_recompute_z_fp8`, Option A) adds exactly **one
extra fwd up-proj GEMM call + one extra activation FP8 quant + one extra
isa-packed scales gather** per layer per backward.  Numbers match
mechanism: 100 % attributable.

---

## 4. recompute_z implementation status

### Default path: Option A (correct, slightly higher overhead)
- `_recompute_z_fp8` reruns `_fused_blockscaled_gated_forward` and pops
  `z_fp8` from the prequant cache; allocates a TK×I bf16 y1 that is
  discarded.
- Bit-exact identical to the no-recompute reference on `out`, `dx`,
  `ds`, `dw1`, `dw2` (cos = 1.000000, rrmse = 0.000000) — verified by
  `tests/ops/test_recompute_z.py` + `tests/ops/test_mlpnode_correctness_large.py`
  (9 cases up to T=16K, including `skew80`, `extreme_one`, `tpe0_holes`).
- Memory delta: −26 MB peak fwd.  Time cost: +447 µs/iter.

### Opt-in research path: Option B (KNOWN-BROKEN)
- Set `SONIC_MOE_FP8_RECOMPUTE_OPT_B=1` to enable.
- Implementation:
  - `BlockscaledQuantOnlyMixin(GemmDefaultEpiMixin)` in
    `sonicmoe/quack_utils/gemm_gated.py` — non-gated CUTLASS DSL mixin
    that emits ONLY z_fp8 + scales (no y1, no SwiGLU, no PostAct
    smem/TMA/R2S/S2G).
  - `GemmSm100ZeroMatBlockscaledQuant` concrete class +
    `blockscaled_fp8_gemm_zeromat_quant(...)` wrapper in
    `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py`.
- Layer-1 bit-exact test (`tests/ops/test_recompute_z_optionB.py`) PASSES
  at T = 1024 / 4096 / 8192 with **uniform round-robin** routing
  (0/200 MiB byte mismatch).
- **BUG**: Triggers `cudaErrorIllegalInstruction` in standalone repro
  the moment routing is non-uniform (random ERNIE-style + hot-expert
  skew).  In the live MlpNodeV2 backward path the kernel does not crash
  but produces all-different output bytes (~24 M / 25 M mismatch),
  causing all gradients to go to garbage (cos ≈ 0 / rrmse ≈ 1.5).
- Suspected cause (unproven, left for future debug):
  the SFA layout setup in `_GemmSm100ZeroMatMixin.__call__` interacts
  differently with a non-postact epilogue than with the gated postact
  epilogue; the gated path always allocates `mPostAct` smem +
  `epi_c_smem_layout_staged`, the non-gated mixin does not.
- Code is preserved verbatim for future debugging — DO NOT ENABLE in
  production.

---

## 5. Improvement opportunities (ranked by µs/iter return on effort)

1. **Fuse the FP8 transpose+quant into wgrad B-loader**
   (`_colwise_quantize_and_pack_kernel`, 247 µs/iter, 8.6 %).
   Today the bf16 dy is materialized, then re-read by a transpose+quant
   triton kernel before wgrad.  An in-kernel TMA load + on-chip transpose
   would eliminate the round-trip.  Effort: ~2-3 weeks of CUTLASS DSL.
   Projected savings: 150–200 µs/iter.
2. **Merge the 3 quant kernels into one varlen pass**
   (`_dual_varlen_quantize` + `_quantize_and_pack` + `_gather_isa_packed_scales`
   = 259 µs/iter).  All three traverse the same TK×H tensor;
   register-tile reuse should yield ~30 % saving.
   Effort: ~1 week of triton.  Projected savings: 70–100 µs/iter.
3. **Get Option B working for recompute_z** — would reclaim
   ~170 µs/iter of the +447 recompute overhead (no y1 alloc + no smem
   for PostAct + no R2S/S2G for y1).  Effort: ~3 days of CUTLASS DSL
   debugging once we understand the SFA-layout vs no-postact
   interaction.
4. **Token-gather-sum** (148 µs/iter) is currently a bf16 scatter-reduce.
   Routing it through a bf16 `__shfl_xor_sync`-based warp reduction +
   block-level prefix would cut to ~60 µs.  Effort: ~3 days.

Total realistic savings (1+2+3): ~390–470 µs/iter — could bring the
frontier to **2400–2475 µs/iter**, beating the Session 53 PyTorch
reference (2715) by ~10 %.

---

## 6. Files and artefacts

- `reports/session68_baseline.nsys-rep` — 2864.7 µs/iter baseline trace.
- `reports/session68_recompute.nsys-rep` — 3311.4 µs/iter recompute_z trace.
- `tests/ops/test_recompute_z.py` — end-to-end numerical equivalence + peak-mem.
- `tests/ops/test_recompute_z_optionB.py` — Layer-1 bit-exact test for
  the Option B kernel (uniform routing only).
- `tests/ops/test_mlpnode_correctness_large.py` — 9-case regression run
  green with `SONIC_MOE_FP8_RECOMPUTE_Z=1`.
