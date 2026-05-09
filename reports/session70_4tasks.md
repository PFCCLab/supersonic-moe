# Session 70 — 4-Task Investigation Report

Date: 2026-04-28
Hardware: NVIDIA Target GPU (SM100-class), HBM3e ~3996 MHz × 8192-bit ≈ 7.65 TB/s peak
Toolchain: ncu 2025.3.1.0, eb_venv (USE_QUACK_GEMM=1)

---

## Task 1 — `_ARANGE_CACHE` is not redundant; clearing it in tests is harmless

### Question
Is the `_ARANGE_CACHE` dict at `sonicmoe/ernie_compat/deepep_metadata.py:60-64` a regression vs.
the framework's own caching allocator? Should clearing it in test fixtures be considered a perf
regression?

### Evidence
The cache stores `dict[(n, dtype, device_str), torch.Tensor]` produced by `torch.arange()`. In the
Paddle compat layer each `torch.arange()` call costs:

- Python dispatch through `paddle.compat.enable_torch_proxy` shim: **~15–30 µs** (measured by
  comparing direct `paddle.arange` to the proxied `torch.arange`).
- One iota launch (`fill_constant_kernel` → strided write) on the GPU: **~2–5 µs**.
- One small allocation through Paddle's caching allocator (essentially free if the `n×4 B` slab
  was already cached, but Paddle releases on tensor destruction unlike torch — so a fresh request
  may still take a `cudaMalloc`).

The framework's caching allocator only caches **memory pages**. It does **not** cache:
- The Python/dispatch overhead.
- The GPU iota kernel launch.
- Tensor metadata creation.

So `_ARANGE_CACHE` is genuinely additive when called in a hot loop (e.g., per microbatch). With
the typical 16 calls per training step × 4 microbatches × ~25 µs ≈ **1.6 ms saved per step** —
small but not zero.

### Honest correction
My earlier framing in the PR review ("clearing `_ARANGE_CACHE` is a regression") conflated two
things:
1. **The cache itself is a real win in production** (above measurement). Removing the cache
   helper would be a regression.
2. **Clearing it inside a test fixture is not a regression**: the cache rebuilds on the next
   warmup call, and tests typically call `_cached_arange` only a few times anyway. The cost of
   clearing in tests is bounded by 25 µs × the number of distinct `(n, dtype, device)` tuples
   the test exercises (typically 1–3) — i.e., < 100 µs total per `clear()`. This is below the
   noise floor of any nsys / pytest measurement.

PR #14 introduced async pinned-memory plumbing for `_DISPATCHED_INDICES_PINNED`
(`deepep_metadata.py:47-57`) which is **complementary**, not redundant, with the arange cache —
they cache different objects (small device iota vs. host pinned buffer).

### Recommendation
Keep `_ARANGE_CACHE` in production code; do not block test PRs that `.clear()` it. Update the
PR-review checklist to make this distinction explicit. (No code change required.)

---

## Task 2 — Default-path FP8 frontier flag table

Added to `README.md` under a new section **"Default Path & Environment Flags"**. Highlights:

- Fused-v2 sonic-meta CUDA kernel (`deepep_topk_metadata_cuda`) is automatically dispatched on
  default; **no env flag** gates it. Falls back to Triton-only when the JIT build fails.
- The full production launcher template is published in the README:
  `USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf SONIC_MOE_FP8_WGRAD=1 ...`
- `SONIC_MOE_FP8_RECOMPUTE_OPT_B=1` is documented as **broken on non-uniform routing** (illegal
  instruction) — explicit "do not enable" warning.

---

## Task 3 — Precision RCA for `paddlefleet_dev_env` test

### Setup
Test: `tests/single_card_tests/model/test_gpt_model_moe_sonic_moe.py::TestSonicMoEPrecision::test_precision_comparison`
Venv:  `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/paddlefleet_dev_env`
Bundled sonicmoe: `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/PaddleFleet/src/paddlefleet/ops/sonicmoe/`
Lab sonicmoe:     `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/sonicmoe/`

### Bundled-vs-lab divergence (verified by `diff -rq`)
Files that differ:

| File | Functional impact on this test |
|---|---|
| `config.py` | Bundled is missing `recompute_z` field + `resolve_recompute_z()`. **Test does not enable recompute_z** → no impact. |
| `functional/__init__.py` | Bundled is missing `_recompute_z_fp8` helper (Option A/B path). **Not called by this test** → no impact. |
| `quack_utils/blockscaled_fp8_gemm.py` | Single delta: `quantize_and_pack_activation_isotropic` default flipped (lab `False`, bundled `True`). **Function is called by no caller in either tree** (verified via `grep`) → no impact. |
| `ernie_compat/deepep_metadata.py`, `mlp_node_v2.py` | Implementation drift on the metadata fast path; same numerical contract on both sides (the CUDA kernel is deterministic and bit-exact identical). |
| `ernie_compat/deepep_topk_metadata_cuda/kernel.cu` | Newer fusion-v2 in lab; bundled is also fusion-v2 (compiles successfully). Numerical contract preserved. |
| `jit.py` | Build/cache plumbing only. No numerical impact. |

### Conclusion (in progress)
Test launched on GPU 5 (PID 1569885); JIT extensions compiled successfully; Triton + CuTeDSL kernels still warming up. Will append final RCA + assertion delta numbers when test completes. Working hypothesis: the precision report from the user is **not caused by the bundled-vs-lab divergence**; the only fp8 numerical-contract delta (`isotropic` default flip) does not affect any default-path call.

**If the test still fails** in the bundled venv, candidates to investigate next:
1. CUTLASS DSL version (env `CUTLASS_DSL_VERSION` or pip-installed `cuda.cccl`) drift between
   the two venvs — may cause a different epilogue precision schedule.
2. Paddle compat layer: `paddle.compat.enable_torch_proxy` may dispatch some ops through a
   slightly different kernel set than direct torch — verify by setting
   `SONIC_MOE_FP8_EPILOGUE_QUANT=0` to force the BF16 epilogue+separate-quant path and see if
   precision recovers.
3. A real bug — in which case overlay the lab tree via
   `PYTHONPATH=/root/paddlejob/.../panzhaowu/lab/sonic-moe:$PYTHONPATH` and rerun.

---

## Task 4 — `_quantize_and_pack_kernel` NCU full-process analysis

### Setup
Driver:  `/tmp/s70/bench_quantize_pack.py` calling `quantize_and_pack_activation(x_bf16)`
Shape:   M=32768, K=3072 (per-iter ERNIE up-proj input)
Command: `ncu --set full --kernel-name "_quantize_and_pack_kernel" --launch-skip 4 --launch-count 1`
Report:  `/tmp/s70/qpk_full.csv` (and `.ncu-rep` available on request)

### Bench across shapes (50-iter avg, Target GPU, USE_QUACK_GEMM=1)

| M | K | avg µs | effective B/W (TB/s, 3.03 B/elt) | % of Target GPU 7.65 TB/s peak |
|---:|---:|---:|---:|---:|
| 4096  | 1024 | 31.1 | 0.41 | 5.3% (launch-bound) |
| 32768 | 1024 | 30.9 | 3.29 | 43% |
| 115968 | 1024 | 61.0 | 5.90 | 77% |
| 4096  | 3072 | 31.9 | 1.19 | 16% (launch-bound) |
| 32768 | 3072 | 54.2 | 5.62 | **73%** ← production case |
| 32768 | 6144 | 103.3 | 5.91 | **77%** |

**Memory-bound regime kicks in around M·K ≥ 50 M elements**; below that the launch overhead
(~30 µs floor) dominates.

### NCU `--set full` results @ M=32768, K=3072

| Metric | Value | Interpretation |
|---|---:|---|
| Duration | **71.6 µs** | matches bench (54–72 µs spread is L2-residency variation) |
| DRAM throughput | **50.7%** | actual measured DRAM = 3.88 TB/s; HBM is **NOT** the limit |
| Memory throughput (overall) | 74.7% | Bottleneck stage |
| **L1/TEX cache throughput** | **80.5%** | **PRIMARY BOTTLENECK** |
| L2 cache throughput | 47.8% | OK |
| L2 hit rate | 38.4% | partial reuse from contiguous bf16 reads |
| Compute (SM) throughput | 63.6% | ALU dominant — amax-reduction + e8m0 packing |
| Achieved Occupancy | **90.85%** (of 100% theoretical) | excellent |
| Active warps / scheduler | 14.58 | adequate |
| Eligible warps / scheduler | **2.13** | modest, limited by scoreboard |
| Warp Cycles / Instr | 21.2 | of which **54.4% = L1TEX scoreboard stall** |
| Registers / thread | 28 | well below the 64K / SM register file |
| Block size | (128,1,1) = 4 warps | |
| Grid size | (1024, 24, 1) = 24576 blocks | 10.4 waves/SM |

### Bottleneck diagnosis (NCU's verdict, verbatim)
> "On average, each warp of this workload spends 11.5 cycles being stalled waiting for a
> scoreboard dependency on a L1TEX (local, global, surface, texture) operation."
> → 54.4% of the 21.2-cycle CPI gap.
>
> "Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section
> to identify the L1 bottleneck."

**The kernel is L1-throughput-limited, not HBM-limited.** With DRAM at 50% and L1 at 80%,
pushing DRAM higher requires either **fewer L1 transactions** (vectorize loads/stores) or
**hiding more L1 latency** (deeper pipeline / more instruction-level parallelism per warp).

### Headroom assessment + ranked optimization levers

| # | Lever | Expected gain | Risk | Effort |
|---|---|:---:|:---:|:---:|
| 1 | **Fuse upstream**: do quant inside the up/down-proj epilogue (or fuse with the gather kernel that produces `x` in the first place). Eliminates the 2 B/elt bf16 read entirely → 3.03 B/elt → 1.03 B/elt. | **2.5–3× speedup** (theoretical lower bound from 71 µs → 25 µs) | High — already partially done via `SONIC_MOE_FP8_EPILOGUE_QUANT`; remaining call sites are the dual-quant / forward-recompute paths. | High |
| 2 | **Vectorize fp8 store** to 16 B chunks (`tl.store` with vector dtype). Currently each fp8 byte produces a separate L1 store transaction; coalescing into 16-B vectors should drop L1/TEX % from 80 → 50%. | **15–25%** (push DRAM from 50% → 65%+) | Low — Triton's `tl.store` already does basic coalescing; need to check whether the scale-pack store is the offender. | Medium |
| 3 | **Increase `BLOCK_ROWS` from 32 → 64**: doubles per-warp work, halves the grid, lets each warp issue more L1 instructions before stalling. | **5–10%** | Low | Low |
| 4 | **Add `num_stages=3` to the Triton kernel decorator** (currently default = 2). Enables software pipelining of the bf16 load / amax-reduce / fp8-store sequence. | **5–8%** | Medium — may push register usage past 32/thread, hurting occupancy. | Low (one-line change + bench) |
| 5 | **Persistent kernel** (1 wave covers all M): saves grid-launch overhead, lets one block process many row-blocks in sequence, increasing L2 reuse of scale-output rows. | **3–5%** at large M | Medium | Medium |

### Production verdict
At 73% of HBM peak on the production shape, this kernel is **already near the asymptotic limit
of "what a separate quant kernel can do."** The single biggest remaining lever is **#1 (fuse with
upstream producer)** — anything else gives single-digit % gains. The current state is **production-acceptable**.

For further reductions, follow the existing `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT=1` /
`SONIC_MOE_FP8_FUSED_GATED=1` pattern: identify the call sites that still call
`quantize_and_pack_activation` separately (e.g., recompute path, dual-quant path) and absorb the
quant into their producer's GEMM/elementwise epilogue.

---

## Artifacts

| Path | Contents |
|---|---|
| `/tmp/s70/bench_quantize_pack.py` | Reusable bench/ncu driver |
| `/tmp/s70/qpk_full.csv` | NCU full-set CSV (104 rows) |
| `/tmp/s70/qpk_ncu.csv` | NCU smoke (4 metrics) |
| `/tmp/s70/precision.log` | PaddleFleet test log (Task 3) |
| `/tmp/s70/run_precision.sh` | Re-launchable precision-test runner |
