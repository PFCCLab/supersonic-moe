# FP8 Engineering Log

> See [`HANDOFF.md`](HANDOFF.md) for current state. This log records milestones chronologically.

---

## Phase 1: Initial FP8 Integration (Sessions 1–21)

- Integrated blockscaled FP8 (1×32 UE8M0) into SonicMoE aligned training path
- Built quant/dequant Triton kernels for blockscaled FP8 on SM100a
- Implemented three FP8 weight caches (forward, dgated, actgrad)
- At this point: FP8 was **slower** than BF16 due to quant overhead > GEMM savings

## Phase 2: Selective FP8 (Sessions 22–23)

- Profiled every kernel with nsys to find where FP8 is net-positive
- Result: 7.0% faster GPU projection by using FP8 selectively

## Phase 3: Quant Kernel Optimization (Sessions 24–25)

- Rewrote quant kernel: 1D→2D grid, uint32 scale packing
- A_idx for backward dgated: quantize T rows (~8µs) instead of TK rows (~96µs)
- Forward A_idx attempted but failed (93.5% RRMSE — scales must be gathered with data)

## Phase 4: Multi-Shape + Adaptive FP8 (Sessions 25–26)

- Adaptive FP8 down-proj (net-positive at I≥2048 only)
- Wall-clock: 1.66× (I=1024), 2.15× (I=2048), 2.37× (I=4096)

## Phase 5: Zero-Materialization Kernels (Sessions 27–28)

- `GemmGatedSm100ZeroMat` / `GemmDGatedSm100ZeroMat` (CUTLASS, SM100)
- Three-step gather pipeline (T-quant→fp8_gather→scale_gather)
- Z FP8 save (saves ~171 MiB at Ernie shape)

## Phase 6: Triton Weight Quant + Cleanup (Sessions 29–31)

- Triton weight quant: 8-op eager → single kernel (eliminated 3136µs/iter overhead)
- Z quant grid tuning: BR=32, GPB=12 → 44µs faster
- Weight cache retention (auto-invalidation via `w._version`)
- Dead code removal, 26/26 tests

## Phase 7: Fused Quant + Stream Parallel + Official Baseline (Session 32)

- **Fused z+y1 2D-grid quant kernel**: single launch, packed int32 scale writes
  - Grid (2048, 20) = 40960 blocks (vs 1D 2048 which was slower)
  - ncu-guided: uncoalesced excess 24%→6%, DRAM throughput 49%→67%
- **Stream parallelism**: z-dequant on side stream || dout-quant+s.float()+scale_gather
- **s.float() pre-cast**: 28µs elementwise hidden in stream overlap window
- **Official BF16 baseline**: fixed `moe(x)` API + `z.backward(dout)` for proper profiling
- **TMA investigation**: tested for both loads and stores — 2.3-2.4× slower (descriptor overhead)
- **CUTLASS constraint discovered**: GemmDGated requires bf16 PreAct (cannot feed FP8 z directly)
- 5 new precision tests → 31/31 total

## Phase 8: Authoritative Benchmarks (Session 33)

- **Fully idle node 0342** (8/8 GPUs idle, 0% utilization) — first clean measurement
- **Corrected performance**: BF16 3932µs/iter → FP8 3690µs/iter = **1.066× faster**
  - Previous 0344 data (6609 vs 6290µs) had GPU contention artifacts
  - GEMM savings: 764µs (21.1%), FP8 overhead: 532µs, net: 232µs
- **Corrected memory**: FP8 peak 1913.8 MiB > BF16 peak 1411.8 MiB (+502 MiB)
  - Weight caches (~650 MiB) outweigh Z FP8 savings (~186 MiB)
  - Previous claim "FP8 peak ≤ BF16" was wrong at production shape
- **Profiling runner**: `tools/_profiling_runner.sh` with `nsys_official_bf16`, `nsys_fp8_frontier`, `mem_fp8`, `mem_bf16`, `test` modes
- nsys install: `dpkg -i .../NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb` on remote nodes
- 31/31 tests pass (verified on node 0342)

## Phase 9: Native FP8 Exploration (Session 34)

- Initial `compute_scales_from_fp8_and_pack` was fundamentally wrong — removed
- After fixing, simulated native path was functionally identical to frontier

## Phase 10: True Native FP8 + A/B Comparison (Session 35)

- `NativeFP8Params` dataclass: 4 FP8 weight views via `_quantize_weight_3d_triton` — zero cache pollution
- `MoE.prepare_native_fp8()` + `forward(native_fp8_params=..., x_fp8_data=..., x_fp8_scales=...)`
- x-quant eliminated: `quantize_and_pack` 3→2 calls/iter
- fused_z_save_y1_quant restored (separate z+y1 was 15µs slower)
- **A/B on node 0267 GPU0**: BF16 3969µs → FP8 3669µs = **1.082×** (GEMM -22.4%, overhead +521µs, net -296µs)
- Per-variable precision: all RRMSE <8%, cosine >0.997. MaxRelErr only at near-zero values.
- Memory: FP8 2130 MiB vs BF16 1658 MiB (+472 MiB)
- **Direction A investigated**: z-dequant (129µs) is hard CUTLASS limitation — 5 constraints (view-f32 packing, TMA layout, epilogue recast, no dequant hook, need extra TMA for scales). Needs new kernel class.
- 31/31 frontier + 5/5 true native tests pass

## Phase 11: Comprehensive Benchmark Audit (Sessions 36–37)

- **Discovered process contamination**: `_IS_FP8_ACTIVE` cached at import time from env var (`utils.py:38`). Same-process FP8-vs-BF16 comparison produces fake bit-identical results. **All precision tests must use separate subprocesses.**
- **Bug 2 found & fixed**: `_DownProjection.backward` line 1310 `out=dw2_base` → `out=dw2.permute(2, 0, 1)` (matching BF16 path). c_proj.weight grad had wrong strides; expert 0 worked by luck.
- **Bug 1 found (OPEN)**: `GemmDGatedFP8CLoadSm100` writes dz only for expert 0. Experts 1-7 = denormalized garbage (~1e-36). y1s output is fine. Suspected C-output pointer offset bug in CUTLASS varlen path.
- **Workaround**: `SONIC_MOE_FP8_FUSED_GATED=0` disables fused dgated kernel, uses separate `blockscaled_fp8_gemm_varlen` + SwiGLU backward.
- **nsys profiling (idle node 00041, 8/8 GPUs idle)**:
  - BF16: 3930 µs/iter
  - FP8 fused (buggy): 3273 µs/iter → **1.20× faster** but numerically wrong
  - FP8 non-fused (correct): 5261 µs/iter → **0.75× (34% slower)** due to de-fused SwiGLU kernels adding 2203 µs/iter
- **Precision (subprocess-isolated, FUSED_GATED=0)**: all RRMSE 6.4-7.9%, cosine >0.997, uniform across 8 experts
- **Memory**: FP8 peak 3470 MiB vs Official BF16 1604 MiB (+1866 MiB, mostly QuACK 0.3.7 base overhead)
- **Conclusion**: Bug 1 fix is the critical path. Fused path has proven 1.20× speedup; non-fused workaround eliminates all gains.

---

## Lessons

1. **CUTLASS PreAct constraint is the bottleneck** — `assert PreAct.element_size() == 2` blocks the highest-value optimization (eliminating z-dequant 130µs). Requires CUTLASS DSL epilogue work.
2. **2D grids beat 1D grids for SM utilization** — the fused quant 1D grid (2048 blocks) was slower than separate 2D kernels. 2D grid (40960 blocks) matches separate kernel parallelism.
3. **Packed int32 scale writes fix coalescing** — accumulate 4 group bytes into uint32 before writing. Reduces excess sectors by 4×.
4. **TMA has overhead for small/fine-grained access** — descriptor creation costs dominate when data volume < ~100MB. Only beneficial for large structured GEMM tiles.
5. **nsys GPU projection is the only trustworthy metric** — wall-clock includes 40-60% CPU overhead on contested nodes.
6. **Official BF16 needs `z.backward(dout)`** not `z.sum().backward()` — the latter produces non-contiguous gradients that fail quack assertions.
7. **Stream parallelism has diminishing returns** — z-dequant (130µs) overlaps with dout-quant+gather (83µs), saving ~47µs. But adding more work to the overlap window only hides, doesn't save.
8. **GPU contention invalidates profiling data** — node 0344 (4/8 idle) showed 6609µs BF16; same on fully idle 0342 was 3932µs (1.68× inflation). Always use 8/8 idle nodes.
9. **FP8 weight caches dominate memory overhead** — 3 caches × 3 weights × ~72MB each ≈ 650 MiB, outweighing Z FP8 savings (186 MiB).
10. **nsys needs manual install on remote nodes** — `dpkg -i .../NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb` required before profiling.
11. **E8M0 scales encode BF16 magnitude, NOT FP8 magnitude** — You cannot reconstruct blockscaled scales from FP8 data. Scales must be computed from the original BF16 source and stored alongside FP8 data. `compute_scales_from_fp8_and_pack` was fundamentally wrong.
12. **PostAct FP8 from GemmGated has no ISA-packed scales** — CUTLASS clamp-casts without producing blockscaled metadata. Must still call `quantize_and_pack_activation(y1)` separately.
13. **Test isolation matters** — `enable_native_fp8()` modifies process-global state. Native FP8 tests must run in a separate pytest invocation from frontier tests.
14. **CUTLASS GemmDGated FP8 PreAct is a hard architectural limitation** — 5 constraints make it impossible to feed FP8 z directly: (1) view(f32) packing requires 2-byte elements, (2) recast_tensor assumes bf16 width, (3) TMA smem is dtype-specific, (4) no epilogue dequant hook, (5) scales need separate TMA. Requires a new kernel class, not a config change.
15. **MaxRelErr at near-zero values is FP8-normal** — when |gold|<1e-8, FP8 quantization noise dominates → relative error >1e6. But absolute error stays <5e-5 and all |gold|≥1e-4 elements have MaxRelErr < 0.35. Not an anomaly.
16. **z recompute increases peak memory in single-layer MoE** — GemmGated recompute allocates z(384)+y1(192) MiB during backward when other backward tensors are live. Only effective with `torch.utils.checkpoint` (framework-level lifetime management).
17. **Lazy bwd weights don't reduce peak** — bwd weights must exist during backward regardless of when they're allocated. Lazy allocation just shifts the allocation point, not the peak.
18. **Per-element FP8 cast ≠ blockscaled FP8** — GEMM epilogue's `.to(fp8)` is a simple saturating cast without per-group scaling. Blockscaled quantization (per-32-element amax → E8M0 → scale-aware cast) is required for precision. The two are NOT interchangeable.
19. **Epilogue blockscaled quant requires warp-level group reduction** — CUTLASS MMA register fragments are scattered across threads. Computing per-32-element amax needs `__shfl_xor_sync` within the epilogue, which is possible but requires knowledge of the SM100 MMA register-to-thread mapping.

## Phase 12: CuTe DSL Colwise Quant + Wgrad Integration (Session 43)

- **CuTe DSL `colwise_quantize_cute`**: 90µs vs Triton 136µs = 1.51× (NCU, clock-control=none, 65536×1536)
  - Bank conflicts: 11.1M → 110K (101× reduction) via row-major smem reads
  - Registers: 48 → 30/thread, occupancy: 60% → 89%
  - rcp.approx E8M0: bit-exact vs integer bitops, verified across all float32 ranges
  - abs.f32 PTX: 1 instruction vs fmax(x,-x) = 2, saves 5.9% total instructions
  - Coalesced (num_groups, dim) scale store: L1 store traffic -48%
  - ISA-packed scale output: 100% bit-exact vs Triton
  - gather_idx support: verified for x operand in wgrad path
- **Integrated into `functional/__init__.py`**: wgrad path uses CuTe DSL for both dz fallback and x colwise quant
- **Dead ends discovered**: smem fp8 store (+96% L1), 1D smem view (+18 regs), double-buffer (SM-bound), full unroll (64 regs)
- NCU profile of Triton `dual_quantize_varlen`: same bank-conflict pattern (84%, 11.2M), confirming CuTe DSL approach generalizes

### Lessons (Session 43)

20. **Row-major smem reads are the key to bank-conflict-free column-wise quant** — 32 lanes reading `sSrc[tk, lane]` = 32 consecutive bf16 bytes = coalesced, zero bank conflict. Column-wise reads cause 6.4-way conflicts.
21. **rcp.approx is bit-exact for E8M0** — mantissa is masked away in E8M0 encoding, so ≤1 ULP mantissa error in rcp.approx has zero effect on the scale byte.
22. **Scale layout matters for coalescing** — `(num_groups, dim)` is coalesced for warp writes; `(dim, num_groups)` causes 97% sector waste (stride = num_groups between lanes).
23. **Smem-mediated vectorized store can be WORSE** — writing fp8 to smem then gmem = double the L1 traffic. Only useful if store coalescing is terrible (not our case — fp8 stores are already 32-byte aligned).
24. **Runtime loops beat full unroll when occupancy matters** — 30 regs (89% occ) > 64 regs (44% occ) for this workload. The extra warps hide ALU latency better than ILP from unrolling.
25. **CuTe DSL `cute.make_tensor` creates register overhead** — each tensor object stores pointer + layout metadata in registers. Avoid creating temporary tensors in hot loops.
26. **NCU `--clock-control=none` is essential on contested nodes** — fixed clocks produce inconsistent results when GPU boost is affected by thermal throttling from other workloads.
27. **Split dual quant > fused dual quant [OUTDATED — reversed by nw=1 fix in Session 52]** — At nw=4: CuTe col (84µs) + Triton row (62µs) = 146µs beat Triton fused (168µs). But at nw=1: dual_quantize_varlen=183µs < separate col(137µs)+row(130µs)=267µs. The nw=1 fix reduces per-block overhead enough that fusion's single-HBM-read advantage dominates.
28. **Fused dual quant instruction bloat 3.6×** — per-TK-row warp butterfly shuffle (5 stages × fmax) + per-row E8M0 (10 ALU ops) + per-element rmem fp8 cast (4-element alloc+cast) = 288M instructions vs 80M for separate. The fp8 vector cast requiring 4-element rmem round-trip is the core overhead.
29. **[32][33] smem padding works for bank conflicts but breaks cp.async tiled_copy** — stride 33 is not multiple of 8, so 128-bit vector loads can't align. Element-wise load (no vectorization) is 10× slower.
30. **nsys GPU busy time ≠ wall-clock** — on busy nodes, GPU busy/iter = 5691µs but wall = 8144µs (30% gaps from CPU launch delay). Use nsys `cuda_gpu_trace` → merge intervals → compute actual GPU busy time.
31. **Paddle's `quantize_1x128_kernel`** — gold reference for fused dual quant. Key technique: load full 128×128 tile into registers, `ComputeRowScale` via warp shuffle + smem cross-warp reduce, `ComputeColumnScale` via smem tree reduce, transpose output via `shm[128][129]` (+1 padding). All data stays in registers — zero re-read from smem.

## Phase 13: NCU-Guided CuTe Gather Optimization + Epilogue FP8 D Output (Sessions 47-48)

- **23-kernel NCU report** (`/tmp/ncu_quant2.ncu-rep`): Full metrics for all quant kernels — time, regs, occupancy, DRAM%, LD efficiency, effective bandwidth
- **CuTe colwise+gather root cause**: Original kernel had 2.1 bytes/sector LD efficiency (93% waste) because each thread loaded one row with 32 sequential element loads — no coalescing across threads in a warp
- **Warp-cooperative coalesced gather**: Rewrote gather path so 32 lanes in a warp cooperatively load 32 consecutive columns of the same row (perfectly coalesced). 8 warps × 32 iterations = 256 rows = TILE_TK. Result: **154µs → 58µs** (2.7× improvement), LD efficiency 2.1→~16 bytes/sector
- **Pre-gather NOT viable**: `torch.index_select` (24µs) + CuTe colwise (29µs) = 53µs > Triton fused (39µs). The extra HBM write+read of the 48 MiB gathered buffer costs more than in-register gather. Verified with NCU.
- **Triton still wins gather** (39µs vs 58µs): Triton compiler generates efficient multi-address scatter-gather from `tl.load(ptr + offsets)`. CuTe element-wise `mSrc[row, col]` generates scalar loads.
- **Epilogue FP8 D output**: GemmGated writes z directly as `float8_e4m3fn` (eliminates standalone z quant ~141µs + z.to(fp8) cast ~288µs). BF16 placeholder with `as_strided((0,0))` for autograd graph.
- **Eager weight cache release**: After dgated GEMM, clear `_FUSED_WEIGHT_CACHE` + `_VARLEN_WEIGHT_CACHE`, `resize_(0)` consumed/unused weight caches (estimated -74 to -148 MiB peak)
- **Single-stream wgrad pipeline**: Removed cross-stream overlap. Single-stream enables caching allocator block reuse. +50µs latency but better memory reuse.
- **JIT cache trap discovered**: QuACK compiles CuTe kernels to `/tmp/root/quack_cache/<fingerprint>/*.o`. Fingerprint is based on quack package source, NOT user kernel source. Must manually clear `.o` files AND call `_compile_colwise_quant.cache_clear()` after editing CuTe kernels.
- **34/34 tests pass** (verified)

### Lessons (Sessions 47-48)

32. **CuTe element-wise gather generates scalar loads** — `mSrc[row, col]` compiles to individual SASS LDG instructions. Even with warp-cooperative coalescing (all lanes access same row, different columns), CuTe can't vectorize across gather indices. Triton's `tl.load(ptr + offsets)` generates vectorized multi-address loads.
33. **QuACK JIT disk cache is source-fingerprint-based** — the fingerprint covers `quack/` package source, not user kernel source. Editing CuTe kernels does NOT change the fingerprint. Must clear `/tmp/root/quack_cache/<hash>/*.o` manually.
34. **Pre-gather is mathematically correct but performance-worse** — bit-exact match vs fused approach, but the extra 48 MiB HBM roundtrip (write gathered bf16 + read it back) exceeds the cost of in-kernel scattered loads.
35. **Autograd + fp8 tensors = illegal memory access** — at large shapes, fp8 dtype tensors in the autograd graph cause segfaults in backward. Solution: store a bf16 placeholder with `as_strided((0,0))` (2-byte storage), actual fp8 data in a side dict.
36. **Cross-stream memory reuse needs record_stream** — PyTorch caching allocator won't reuse blocks across streams without explicit `record_stream` calls. Single-stream is simpler and lets freed blocks be immediately reusable.

## Phase 14: Pythonic Config + Unaligned Padding + nsys Engine (Sessions 44–46)

- **nsys GPU-projection engine** (Session 44): Integrated into `tools/introspect.py --mode nsys`. Launches separate BF16/FP8 subprocesses under nsys, parses `CUPTI_ACTIVITY_KIND_KERNEL` from sqlite, merges kernel intervals → per-iteration GPU busy time. Gold standard for latency measurement on idle GPUs.
- **Pythonic Config API** (`SonicMoEConfig`, Sessions 45–46): Dataclass with 10 fields + thread-local context manager. Priority: config > context managers > env vars. Replaced env-var-only control.
- **wgrad FP8 default-ON** (Session 45): `_use_fp8_wgrad()` returns True at all shapes. Trades ~19% slower wgrad GEMM for memory savings (early bf16 dz freeing).
- **Unaligned FP8 padding** (Session 45): `_padded_blockscaled_gated_forward()` pads expert segments to 128 for FP8 GEMM alignment. Backward stays BF16 (QuACK BF16 handles unaligned natively).
- **CuTe DSL colwise quant** (Session 43, continued): 29µs vs Triton 39µs = 1.3× faster without gather. 30 regs, 89–93% occupancy.
- **Wall-clock unreliable** (Session 44+): Established that shared-node contention swamps kernel-time signal in wall-clock measurements.

## Phase 15: Epilogue FP8 + CuTe Gather + NCU Profiling (Sessions 47–48)

- **Epilogue FP8 D output**: GemmGated writes z as `float8_e4m3fn` directly. Eliminates standalone z quant (~141µs) + z.to(fp8) cast (~288µs). No bf16 z allocation (saves 384 MiB transient).
- **BF16 autograd placeholder**: z stored as `as_strided((0,0))` bf16 (2 bytes). Actual fp8 z in `_PREQUANTIZED_SCALES["z_fp8"]`. Avoids fp8 tensors in autograd graph (segfaults at large shapes).
- **CuTe colwise+gather optimization**: Warp-cooperative coalesced loads — 154µs → 58µs (2.7×). Still behind Triton's 39µs for gather case.
- **23-kernel NCU report**: `/tmp/ncu_quant2.ncu-rep` with clock-control=none. Authoritative per-kernel metrics (time, regs, occ%, DRAM%, LD efficiency, effective BW).
- **Pre-gather proven suboptimal**: index_select(24µs) + CuTe colwise(29µs) = 53µs > Triton fused(39µs).
- **row_quant at ceiling**: 97% occupancy, 4613 GB/s. No further optimization possible.
- **Single-stream wgrad**: Removed cross-stream overlap. +50µs latency but better memory reuse via caching allocator.
- 34/34 tests pass.

## Phase 16: Memory Analysis + Env-Var Fix + `_fp8_mode()` Fix (Sessions 49–51)

- **Session 49**: First memory comparison attempted. **CONTAMINATED** by `SONIC_MOE_FP8_MODE=perf` env var leaking into "BF16" runs. All Session 49 data is invalid.
- **Session 50**: Fixed contamination via explicit `os.environ.pop('SONIC_MOE_FP8_MODE', None)` before BF16 measurement. Clean memory data obtained: FP8 saves 4–5% peak (fwd), +118 MiB bwd (wgrad quant temps). FP8+Stash saves 21–23% overall.
- **Session 50**: Backward memory deep-dive — peak is at wgrad (not dgated), so early cache eviction before dgated doesn't help at I=1536. Cache structure: only 37 MiB freeable (w2_varlen; w1T held by ctx reference).
- **Session 50**: "Save x as fp8" attempted and reverted — dequant creates +24.8 MiB transient spike.
- **Session 51**: **`_fp8_mode()` priority fix** — the actual root cause of Session 49 contamination. When `is_fp8_active()` returned False, the function fell through to env var check and returned "perf". Fix: return "off" immediately when `is_fp8_active()` is False.
- **Session 51**: CUDA events 3-round benchmark (20-trial median × 3 rounds, same-process). I=1536: 1.08× (BF16 high-variance), I=2048: 1.22× (consistent). CUDA events proved more reliable than nsys under 100% GPU contention.
- **Session 51**: Kernel classifier fix — ZeroMat GEMM kernels excluded from "Blockscaled Quant" category.

### Lessons (Sessions 44–51)

37. **env vars are process-global and cached at import** — `_IS_FP8_ACTIVE` is set once from env var. Same-process BF16/FP8 comparison with env var set will produce incorrect BF16 baselines.
38. **Context manager must override env var** — the `_fp8_mode()` priority chain must be: config > context manager > env var. A "False" context manager must short-circuit before reaching env var.
39. **CUDA events same-process is most reliable under contention** — both modes experience identical contention. nsys runs separate processes at different times with different contention levels.
40. **FP8 is more contention-resilient** — under 100% GPU util, BF16 showed 57% timing variance while FP8 showed 1.2% at I=1536. Compute-bound FP8 GEMMs are less affected by memory bandwidth contention.
41. **Weight stash is the dominant memory optimization** — 21–23% peak savings vs 4–5% from FP8 alone. The stash strategy works because FP8 caches serve backward, making bf16 master weights redundant during forward+backward.
42. **FP8 backward peak > BF16 backward peak without stash** — wgrad quant creates ~118 MiB temporaries. This is fundamental to blockscaled FP8 wgrad and cannot be eliminated without fusing quant into the GEMM epilogue.

## Phase 17: NCU-Guided Quant Optimization + Wgrad Auto-Tune (Session 52)

- **num_warps=1 discovery** (key insight): NCU showed Triton colwise at nw=4 spends 63% of cycles with no eligible warp (stall_barrier dominated). Reducing to nw=1 (32 threads/block) allows more blocks in-flight per SM → 2.3x speedup on colwise, 2.0x on dual_quantize_varlen. Bitwise identical output.
- **CuTe-to-Triton migration**: All hot-path nogather colwise calls now use Triton nw=1 (137us) instead of CuTe (182us). UpProj fallback also switched.
- **Fused dual quant in DownProj wgrad**: `dual_quantize_varlen(dz)` replaces separate `colwise_quantize_cute(dz)` + `quantize_and_pack_activation(dz)`. Saves one HBM read: 183us vs 311us.
- **Wgrad FP8 shape auto-tuning**: `_FP8Config.resolve_wgrad(I)` — ON for I>=2048, OFF for I<2048. Based on measured crossover: I=1536 wgrad is 0.913x (net negative).
- **Total quant overhead reduction in DownProj wgrad**: ~676us -> ~388us = 43% savings.
- **introspect.py: quant-bench + wgrad-bench modes** — isolated CUDA-event kernel benchmarks with statistics, JSON output.
- **End-to-end results** (CUDA events, TK=65536, B30Z): I=1536: 0.983x, I=2048: 1.031x, I=3072: 1.136x. FP8 forward consistently ~19% slower (quant overhead), FP8 backward consistently faster (9.6-23.2%).
- 34/34 tests + 20 subtests PASS.

### Lessons (Session 52)

43. **num_warps=1 dramatically better for bandwidth-bound Triton kernels** — counter-intuitive; fewer warps/block -> more blocks in-flight per SM -> better utilization. Key NCU diagnostic: "% cycles with no eligible warp". Does NOT help row_quant (nw=1/2/4 all 108us — already well-balanced).
44. **Fused dual quant viability depends on per-block overhead** — at nw=4, dual kernel's 288M instructions caused 3.6x bloat. At nw=1, overhead per block drops enough that single-HBM-read advantage makes dual competitive (183us vs 267us).
45. **CuTe nogather is no longer faster than Triton** — the nw=1 fix inverted the CuTe advantage: Triton nw=1 at 137us beats CuTe at 182us for nogather colwise at dim=3072. CuTe DSL retains its advantage only in theory (93% occ, 48% DRAM) but cannot match Triton's nw=1 block-level parallelism.
46. **FP8 forward overhead is the dominant bottleneck** — at I=1536, FP8 forward is 19% slower (2.457 vs 2.062ms). This single factor prevents FP8 from winning at small I. The forward quant overhead (quantize_and_pack_activation ~130us + GEMM dispatch overhead) cannot be hidden because it's on the critical path.
47. **FP8 wgrad auto-tuning is essential** — shape-dependent ROI: I=1536 wgrad is net negative (0.913x), I=2048 is break-even (1.057x), I=3072 is positive (1.182x). Without auto-tuning, small-I workloads pay an unnecessary penalty.

## Phase 18: Weight Cache Fixes + Official Baseline + 27-Shape Grid (Session 53)

- **VARLEN weight cache fix**: `_VARLEN_WEIGHT_CACHE.clear()` at DownProj backward forced re-quantization every iter (~360µs). Cache is version-keyed, auto-invalidates at optimizer step. This single fix took FP8 from 1.03× to 1.14× speedup.
- **FUSED weight cache fix**: Same pattern in `_FUSED_WEIGHT_CACHE.clear()` between forward and backward.
- **Cache corruption fix**: `ctx._w2_dgated_fp8.untyped_storage().resize_(0)` freed tensor aliased in `_FUSED_WEIGHT_CACHE` → E>8 crash. Don't free ctx tensors that alias cache entries.
- **B.mT.contiguous() removal**: Our `gemm_interface.py` added `.contiguous()` after `B.mT` (~600µs elementwise copy per iter). Official has just `B.mT`. Fix: remove, BF16-only guard.
- **Official BF16 baseline**: Used `/lab/official/sonic-moe` (env `official_bf16`) as the ONLY valid BF16 baseline. Our branch BF16 had ~9% overhead from FP8 infrastructure.
- **Wgrad threshold=0**: FP8 wgrad profitable at all I values after cache fixes (threshold was I≥2048 in Session 52).
- **Token rounding for E>8**: Official `forward_token_choice_rounding(Mtile=128)` for 128-alignment. `moe_general_routing_inputs` for FP8 E>8 workloads.
- **INT32 pointer overflow** (user-discovered): Triton int32 pointer arithmetic wraps at `row × stride > 2³¹-1`. Dual-path dispatch with `SAFE_INT64`. Full kernel audit in `docs/HANDOFF.md`.
- **introspect.py grid mode**: 27-shape (3T×3E×3I) parallel nsys profiling on 8 GPUs with LPT load balancing.
- **GPU isolation fix**: `_subprocess_env_for_gpu()` respects shell-level `CUDA_VISIBLE_DEVICES`.
- **Final results**: 27-shape grid, **1.29× – 1.70×, mean 1.53×**. Memory +5-10%. Precision all PASS.

### Lessons (Session 53)

48. **Weight cache invalidation must be version-keyed, not eager.** Clearing caches every backward costs 300-980µs/iter. `weight._version` auto-invalidates at optimizer step.
49. **Never `resize_(0)` tensors aliased in caches.** Context tensors saved by autograd may share storage with cache entries.
50. **Official baseline is non-negotiable.** Our branch BF16 had 9% overhead from FP8 infrastructure (1280 kernels vs 520). Must use official SonicMoE as the BF16 comparison point.
51. **CUTLASS JIT cache contamination across shapes.** Different shapes compile incompatible kernels; same-process multi-shape → ILLEGAL_INSTRUCTION. Isolated subprocesses required.
52. **Weight view refcount prevents stash memory savings.** Python references to weight views keep bf16 storage alive. Create views AFTER stash.
53. **I scaling dominates FP8 ROI.** GEMM savings ∝ O(I²), quant overhead ∝ O(I). The quadratic-linear gap widens with I.

## Phase 19: MoE Module-Level Test Suite + 0-Size Audit + ernie-core Analysis (Session 54)

- **MoE module test suite** (`tests/ops/test_moe_module.py`): 59 parametrized tests validating the FULL MoE pipeline (permute → up-proj → SwiGLU → down-proj → unpermute). Gold reference uses split-half SwiGLU (ERNIE convention) in float32. BF16 vs Gold: RRMSE=0.0044, cosine=0.99999. FP8 vs Gold: RRMSE=0.065, cosine=0.998.
- **Weight convention conversion**: `split_to_interleaved()` / `interleaved_to_split()` with bit-exact round-trip verification. SonicMoE uses interleaved (gate=even, up=odd), ERNIE uses split-half (gate=[:I], up=[I:]).
- **FP8 env var fix**: FP8 is activated via `enable_fp8()` context manager or `SONIC_MOE_FP8_MODE=perf`, NOT `SONICMOE_FP8_ENABLED` (which doesn't exist). Subprocess templates use `SONIC_MOE_FP8_MODE=perf`.
- **Weight stride fix**: CUTLASS expects `w1` as a `.permute(1,2,0)` view of `(E, 2I, H)` contiguous parameter (non-contiguous strides). Passing a `.contiguous()` tensor with wrong strides causes `RuntimeError: Expected strides[leading_dim] == 1`.
- **0-size expert audit**: Full audit of all forward/backward paths with empty experts. All components handle `cu_seqlens[e]==cu_seqlens[e+1]` (0-token experts) correctly. 0 % 128 == 0, so alignment check passes. BF16+FP8 forward+backward produce correct output and zero gradients for unused experts.
- **Empty expert edge cases**: Single-expert routing (7 empty), only-two-active, half-empty — all tested in BF16 fwd+bwd and FP8 fwd+bwd.
- **ERNIE-core analysis**: Deep read of `ernie_core/models/moe/token_dispatcher/fp8_utils.py`. Key findings: split-half SwiGLU, prob scaling after SwiGLU/before down-proj, `kitchen_quant` for 1×128 blockscaled FP8, `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` for grouped GEMM.
- **Robustness tests added**: Deterministic (bit-exact repeated runs), large tensor (T=4096), routing metadata correctness, numerical stability at varied activation scales, weight conversion round-trip, gold all-same-expert.

### Lessons (Session 54)

54. **FP8 is activated by `enable_fp8()` or `SONIC_MOE_FP8_MODE=perf`.** The `_IS_FP8_ACTIVE` flag (in `sonicmoe/functional/utils.py`) is process-global. Use `enable_fp8()` context manager for in-process switching, subprocesses for isolation.
55. **Weight tensor strides matter for CUTLASS.** The functional layer expects `w1.permute(1,2,0)` as a non-contiguous view. Creating a contiguous `(2I,H,E)` tensor produces wrong strides. Always store as `(E,2I,H)` and pass `w.permute(1,2,0)`.
56. **0-size expert segments are safe in all paths.** Triton 0-grid launches are no-ops, CUTLASS varlen scheduler emits 0 rows, `torch.empty(0, K)` is a valid tensor. The only failure is FP8+non-aligned non-empty segments (by design).
57. **Module-level precision is tighter than op-level.** Op-level RRMSE was ≤7%; module-level (chained permute+GEMM+SwiGLU+GEMM+scatter) is only 6.5% FP8. Error doesn't compound as much as expected because the scatter (weighted sum) acts as a low-pass filter.
58. **ERNIE-core applies prob scaling BETWEEN SwiGLU and down-proj** (`o2 = swiglu(o1) * probs` then `o3 = o2 @ W2`). SonicMoE applies it AFTER down-proj in `_router_forward`. Mathematically equivalent for linear down-proj but produces different intermediate tensors — matters for numerical precision analysis.

## Phase 20: Doc Correction + Backward Gradient Integrity Test (Session 57)

- **dz[pad]=0 proof completed**: Audited all 6 backward paths (1 CUTLASS fused in `gemm_dgated.py`, 5 Triton in `swiglu_triton.py`). In every path, router score multiplies gradient BEFORE dSwiGLU. Since `score[pad]=0` (IEEE 754 exact zero), `dz[pad]=0` exactly. This means padding rows contribute zero to dw1 and dx.
- **Doc correction**: `docs/pad_audit_methodology.md` Section 2.2 previously claimed dz[pad] was non-zero but "consistent between BF16 and FP8 paths". This was wrong. Replaced with correct analysis citing exact code locations in both CUTLASS and Triton paths.
- **`test_pad_gradient_integrity.py`**: 8 axiomatic tests covering full backward chain:
  1. Token conservation (dst_idx injective)
  2. Score invariant (original preserved, padding == exact 0.0)
  3. Forward near-exact (< 1e-13 threshold for GPU matmul tiling differences)
  4. **dz[pad] == exact 0.0** (364 padding rows, all precisely zero)
  5. dw1 near-exact (< 1e-13, measured exact 0.0)
  6. dw2 near-exact (< 1e-13, measured exact 0.0)
  7. dx near-exact with dx[0] (gather target) verified
  8. No token misrouting
- **test_pad_routing.py fix**: Pre-existing failure (diff=2.78e-17) from GPU matmul tiling difference in float64. Changed from bit-exact to 1e-13 epsilon threshold. Root cause: different expert segment lengths (7 vs 128 rows) produce different matmul reduction order on GPU.
- **Coverage audit**: Mapped all FP8 frontier strategies against test files. 48 testable techniques identified, all covered except isolated FP8 dgated kernel test (pre-existing gap, end-to-end coverage exists).

### Lessons (Session 57)

59. **Score-gating guarantees dz[pad]=0 exactly across ALL 6 backward paths.** IEEE 754: `finite * 0.0 = 0.0`. Route-level padding is backward-safe by construction, not by coincidence.
60. **GPU matmul tiling depends on problem shape.** Different row counts produce ULP-level float64 differences. Assertions for matmul-derived values need epsilon tolerance; score-gating zeros are truly bit-exact because they depend only on multiplication by zero.

## Phase 21: Multi-Stream Elimination + ERNIE Integration Plan (Session 58)

- **Removed all side-stream logic from forward/backward.** Deleted `_WGRAD_STREAM` (dead code —
  declared in Session 32 but `_get_wgrad_stream()` was never called), `_DEQUANT_STREAM` (used in
  `_UpProjection.backward` for x_col quant and `_DownProjection.backward` for z dequant), and
  all `wait_stream()` calls that produced `cudaStreamSynchronize` events.
- **Clarification on Phase 15 inconsistency**: Phase 15 recorded "Single-stream wgrad pipeline:
  Removed cross-stream overlap." This was accurate for wgrad-specific overlap, but `_DEQUANT_STREAM`
  survived in two other backward paths (`_UpProjection.backward` x_col quant overlap and
  `_DownProjection.backward` z-dequant overlap) until Session 58 fully removed it.
- **nsys verification**: Profiled with `nsys profile --trace=cuda,nvtx --capture-range=none`,
  exported to sqlite, queried `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE syncType=3`:
  zero STREAM_SYNCHRONIZE events. Backward path has zero sync calls; forward has 6 framework-internal
  `stream_wait_event` (type=2, Paddle/CUDA runtime internals, unavoidable).
- **Correctness**: `test_moe_general_routing_fp8.py` passes unchanged after the fix.
- **ERNIE-core MlpNode study**: Read `MlpNode.__init__`, `.forward`, `.forward_auto_subbatch`,
  `FusionFP8Expert`, `ExpertsGroupGemmContiguousNode`. Documented integration plan in HANDOFF §9.1
  with 6 key compatibility/incompatibility points.

### Edits Made

| File | Change |
|:-----|:-------|
| `sonicmoe/functional/__init__.py` | Deleted lines 506-525 (`_WGRAD_STREAM`, `_DEQUANT_STREAM`, getters). Rewrote `_UpProjection.backward` FP8 wgrad section to run x_col quant on default stream (-15 lines, +4 lines). Rewrote `_DownProjection.backward` z-dequant section to run on default stream (-11 lines, +3 lines). All `wait_stream()` removed. |

### Lessons (Session 58)

61. **Side-stream overlap provides no net benefit when overlap window is small.** `_DEQUANT_STREAM`
    saved ~47µs theoretical but each `wait_stream()` produced a `cudaStreamSynchronize` costing
    more, and cross-stream blocks prevent caching allocator reuse. Prefer single-stream unless
    overlap window is >100µs.
62. **nsys `--capture-range=nvtx` does NOT interoperate with Paddle's `nvprof_nvtx_push`.** 
    Different NVTX domain/API. Use `--capture-range=none` when profiling Paddle tests.
63. **`_WGRAD_STREAM` was dead code.** Declared in Session 32, never called. Always grep for
    callers before assuming infrastructure is live.

## Phase 21: Session 60 — Gate↔MLP Gradient Chain (2026-04-22)

- Fixed ds silently dropped (`_topk_scores_needs_grad` always True)
- Fixed segfault from native Paddle autograd nodes on ds return path
- Removed `x.detach()` that severed dx flow, removed identity layout path
- Lessons 64-72 in `docs/session60_lessons.md`

## Phase 22: Session 62 — Production Hardening (2026-04-24)

- Eliminated CPU-GPU sync in `_differentiable_router_scores` (Triton kernel, 17µs)
- Fixed 3 compile_keys with dynamic token dims: "varlen" (a_scales_packed.size(1)),
  "weight_grad" + "weight_grad_fast" (capacity from _auto_capacity)
- Fixed Paddle/PyTorch compat: _inplace_version, stream API, _offset, dtype comparison,
  bf16 tensor conversion (only torch.from_dlpack works correctly)
- Fixed warmup_jit: unpack (y1, z), run under Paddle proxy, 2-iter for wgrad accumulate
- Added SonicMoEMlpNode.step() API (flush grads + invalidate caches)
- Added input validation to 18 operator wrappers (zero GPU sync)
- Added _GEMM_FAST_PATH 64-entry high-water-mark eviction
- Deleted 4 dead functions + 2 dead unfused quant kernels + legacy flags
- Defaulted ASSUME_ALIGNED=True (route-level padding guarantees 128-alignment)
- Added test_cold_start_e2e.py: cache clear → warmup → 6-shape × 5-tensor precision
- nsys GPU-projection verified: 2871µs (3-GPU mean, CV=0.6%, +5.7% vs S53 baseline)

Lessons:
73. `mark_layout_dynamic` has zero GPU kernel overhead (A/B nsys: Δ<1%).
74. CUDA events ≠ GPU kernel time (5100µs events vs 2871µs GPU-proj).
75. nsys `cuda_gpu_kern_sum` is wrong without NVTX filtering.
76. Paddle bf16 → numpy is silently broken (returns uint16).
77. compile_key must not contain ISA-packed scale shapes (_storage_per_batch is dynamic).
78. `_auto_capacity` makes tensor shapes dynamic — never include in compile_key.

---

## Phase 23: E!=topk Fix + QuACK Paddle Compat (Session 63, 2026-04-24)

Fixed 3 critical bugs blocking E=32 production deployment:

1. **_build_score_src_idx_kernel PTXASError**: Triton `tl.min(vector, axis=0)` generates PTX incompatible with SM103a's ptxas (CUDA 12.8 bundled with Triton 3.5.0). Rewrote with pure scalar selection sort using WORK_ptr scratch buffer.

2. **varlen_K_max=E instead of topk**: When E=32 topk=8, `token_gather_sum_kernel` received MAX_K=32 instead of 8. This was masked when E==topk==8. Fixed by passing topk through `_SonicMoEDeepEPFunc._topk` class variable.

3. **QuACK autotuner BrokenPipeError**: `_compile_worker.py` crashed on `'paddle.bfloat16'` dtype string (KeyError in `_dtype_map`). Root cause: `str(tensor.dtype)` returns `'paddle.bfloat16'` under Paddle proxy. Fixed with dtype normalization + paddle.* entries in worker dtype map + robustness hardening.

Performance (ERNIE-shape E=32 H=3072 I=1536 K=8 EP=8 SEQ=4096):
- Forward GPU-proj: 625µs (CV=0.3%)
- Backward GPU-proj: 1904µs (CV=0.1%)
- Total: 2530µs (CV=0.2%)

Lessons:
79. `str(dtype)` under Paddle proxy returns `'paddle.bfloat16'` — normalize to `'torch.*'`.
80. Hidden semantic coupling: `E==topk` assumption hid varlen_K_max bug across all sessions.
81. `tl.min(vector, axis=0)` generates PTX that old ptxas can't handle on SM103a.
82. QuACK `_precompile` subprocess crash → BrokenPipe → must be try/except wrapped.
83. bench_mlpnode_mem.py `make_inputs` extremely slow (~30s) — looks like hang.

## Phase 24: fp8_wgrad=True 验证 + Bench Config 修正 (Session 64, 2026-04-24)

**Key discovery**: `bench_mlpnode_mem.py` and `nsys_grid2.py` hardcoded `fp8_wgrad=False`,
while auto-detect defaults to `fp8_wgrad=True` (threshold=0 since Session 53 cache fix).
All previous bench data was from a non-production config.

**Fixes**:
- Removed `fp8_wgrad=False` from `bench_mlpnode_mem.py` — now uses auto-detect (production default)
- Re-collected nsys GPU-projection with `fp8_wgrad=True`

**Performance (nsys GPU-proj, fp8_wgrad=True, production config)**:

| Shape | S53 BF16 (µs) | S53 FP8 (µs) | Paddle FP8 (µs) | vs BF16 | vs S53 FP8 |
|---|:---:|:---:|:---:|:---:|:---:|
| T=8192 E=8 | 3644 | 2715 | 2887 | 1.26x | 1.06x slower |
| T=8192 E=32 | 3844 | 2922 | 3372 | 1.14x | 1.15x slower |
| T=16384 E=8 | 7953 | 5227 | 5548 | 1.43x | 1.06x slower |
| T=16384 E=32 | 8129 | 5432 | 5916 | 1.37x | 1.09x slower |

**Key takeaways**:
- vs BF16: **1.14x – 1.43x faster** (production-relevant comparison)
- vs S53 native PyTorch FP8: 6-15% slower
- **100% of overhead from fused wgrad accumulate epilogue** (regs=50→86, +30-38% per-call)
- S53 has no main_grad accumulation — fair functional difference, not a bug
- Non-GEMM categories are actually faster in Paddle (-24 to -65µs)
- Precision verified: 4 shapes × 5 tensors, all cosine > 0.98 with fp8_wgrad=True

### Root Cause: Fused Wgrad Epilogue Register Pressure

nsys sqlite analysis (3 trials × 4 shapes, GPUs 2-7) shows the same CUTLASS
`quackgemm_default_epiGemmDefaultSm100` kernel compiled with different register counts:

| GEMM variant | S53 regs | Paddle regs | S53 µs | Paddle µs | Delta |
|---|:---:|:---:|:---:|:---:|:---:|
| FP8 wgrad compute | 54 | 54 | ~196 | ~222 | +6% |
| BF16 accumulate (dw1) | **50** | **86** | 325 | 449 | **+38%** |
| BF16 accumulate (dw2) | **50** | **86** | 174 | 226 | **+30%** |

S53 epilogue: `D = A@B` (simple, 50 regs, ~5 blocks/SM).
Paddle epilogue: `D = A@B + 1.0*C` (fused fp32 accumulate, 86 regs, ~2-3 blocks/SM).

GemmGated ZeroMat (fwd) and GemmDGated ZeroMat (bwd) are unaffected: same grid,
same regs=168, same per-call timing (<4% variance). These don't involve wgrad.

Lessons:
84. **Bench scripts must match production config.** `fp8_wgrad=False` was hardcoded while
    production auto-detect uses threshold=0 → always True.
85. **Fused epilogue trades register pressure for fewer kernel launches.** The `D = A@B + C`
    epilogue increases regs from 50→86 (+72%), reducing occupancy and causing 30-38%
    per-call slowdown. In a training loop where accumulation is needed anyway, the
    trade-off may still be net positive vs separate add kernel.
86. **Non-GEMM categories can be faster in Paddle.** S53 has extra torch elementwise,
    cuBLAS, and reduce kernels that Paddle's fused path eliminates.

## Phase 25: TMA Reduce-Add Wgrad Epilogue (Session 65, 2026-04-24)

**Key insight**: The fused `D = A@B + 1.0*C` wgrad accumulate epilogue (86 regs/thread)
can be replaced by TMA hardware atomic add on store (`add_to_output=True`, 50 regs/thread)
with zero precision impact. Each output tile is written by exactly one CTA (no split-K),
so the TMA atomic add is deterministic.

**Implementation**:
- New function `_run_cutlass_blockscaled_gemm_varlen_k_tma_add()` in `blockscaled_fp8_gemm.py`
  - `C=GemmTensorInfo(None)` (no C tensor loaded)
  - `epi_args = GemmDefaultSm100.EpilogueArguments(add_to_output=True)`
  - Separate caches: `_COMPILE_CACHE_VK_TMA_ADD`, `_GEMM_FAST_PATH_VK_TMA_ADD`
- Replaced all 4 wgrad accumulate sites in `functional/__init__.py`:
  - FP8 paths: `_accumulate` → `_tma_add`
  - BF16 paths: `gemm(beta=1.0)` → `gemm_add(C=accum, out=accum, beta=1.0)`
- QuACK `gemm_add()` auto-detects: when `C is out` and `beta==1.0` and `cu_seqlens_m is None`,
  uses `add_to_output=True` with `C=None`
- Legacy fallback: `SONIC_MOE_FP8_WGRAD_BETA_ACCUM=1`

**Results (nsys GPU-projection, MLP-node only)**:
- E=8: 2886 → 2820 µs/iter (-2.3%)
- E=32: 3420 → 3283 µs/iter (-4.0%)
- BF16 wgrad GEMM (4 calls/iter): E=8 -16µs/call (-5.0%), E=32 -33µs/call (-7.7%)
- FP8 wgrad (isolated bench): 6-13% faster, bitwise identical output
- FP8 forward GEMMs: unchanged (delta < 5µs, noise)

**Precision**: All 6 shapes × 4 tensors PASS (cos > 0.99, RRMSE < 7.6%).
E=32 production shape included. ds gradient cos=0.9972 (test_cold_start_e2e.py).

Lessons:
87. **TMA reduce-add is free when no split-K.** `tile_count_semaphore=None` means each CTA
    exclusively owns its output tiles. The atomic add degenerates to a simple store-with-add,
    but the epilogue doesn't load C tensor, saving registers and smem stages.
88. **QuACK `gemm_add()` vs `gemm()` auto-detection.** `gemm()` passes C/beta through
    `gemm_out()` which does NOT auto-detect add_to_output. Only `gemm_add()` has the
    `C is out and beta==1.0` identity check. Use `gemm_add()` for wgrad accumulate.
89. **Paddle torch-proxy `torch.equal()` fails across tensor types.** Use `(a == b).all()`
    instead. Also `torch.randn` produces paddle dtypes; use `paddle.randn` for bf16.
90. **nsys MUST use `--resolve-symbols=false`** on this machine or it hangs attempting to
    download symbols from network. Template in `/panzhaowu/env.md`.
91. **`bench_mlpnode_topk_nsys.py` is the gold standard for clean profiling.** Pre-computes
    routing once, NVTX "BENCH" range brackets measurement. `bench_deepep_topk_nsys.py` is
    noisy with framework operators (token_gather_sum, Eigen, cub::RadixSort dominating).

---

## Phase 26: TopK Kernel Bug Audit + Correctness Hardening (Session 66, 2026-04-27)

**Context**: User fixed two production-impact bugs in `deepep_topk_metadata_cuda/kernel.cu`
prior to session (commits 5987418, 1eadaa8). This session audited the rest of the repo for
the same bug classes, added a regression test that catches them, and re-anchored perf data.

**Audit scope**: every `.cu`, Triton kernel, and CuTe DSL launch in `sonicmoe/` was inspected
for two bug classes:
- Class A — grid-wide spin-wait / atomic barrier without `cudaLaunchCooperativeKernel`
  (deadlocks when grid > device-resident SM cap)
- Class B — `dim3 grid(min(blocks, CAP))` paired with `blockIdx.x * STRIDE → row` mapping
  (silent corruption: high-index rows never get a CTA)

**Result**: no other instances found. `count_cumsum.cu` uses cooperative launch correctly;
`deepep_metadata.cu` uses 1-block-per-expert (no grid cap); Triton kernels use proper
`cdiv`-based grids; CuTe launches managed by CUTLASS scheduler.

**New regression test**: `tests/ops/test_mlpnode_correctness_large.py`
- 9 cases × 5 tensors (out/dx/ds/dw1/dw2) validated against BF16 gold
- Subprocess-per-case with 600s hard timeout (hang detection)
- Includes `seq16K_E8` and `seq16K_E32` (TK=131072) — exact regime where Class B bug surfaced
- Includes skew80, extreme_one (all tokens to E0..K-1), tpe0_holes (0-token experts)
- All 9 cases PASS (out cos ≥ 0.9979, dx cos ≥ 0.9975, ds cos ≥ 0.9972, dw1/dw2 cos ≥ 0.9971)

**Perf re-anchored** (nsys 2026.2.1.210, sqlite GPU-projection, GPU 7 idle):
- T=8192 E=8 K=8 I=1536 H=3072 mlpnode-only: **2823 µs/iter** (BENCH-range, `bench_mlpnode_topk_nsys.py`)
- T=8192 E=8 per-ITER median (no flush): **2463 µs**
- With per-iter flush (non-default, grad_acc=1): 3110 µs
- vs S53 pure-torch FP8 baseline 2715 µs
- At realistic `grad_acc_steps=8`: ~2519 µs/microbatch — **beats S53 by 7.2%**

**Fixed `bench_coldstart_nsys.py` semantics**: previous version called `flush_native_grads()`
inside the per-iter loop, which is non-default usage and inflated per-iter timeline by
~280-340 µs of `permute / TilingSwapDim / Eigen meta_assign / broadcast_add` kernels.
Production usage is "flush at optimizer.step() time" via `node.step()`, not per-microbatch.
Bench now mirrors production: per-iter NVTX `ITER{n}` ranges with no in-loop flush, then
a single `FLUSH` NVTX range after the accumulation window.

Lessons:
91. **Two CUDA kernel-launch bug patterns to grep when reviewing custom kernels.**
    Class A: `cudaLaunch[ \t]*(?!Cooperative)` near grid-wide atomics. Class B:
    `dim3 grid\(.*min\(.*\)` paired with `blockIdx.x[ \t]*\*` row mapping.
92. **Per-iter `flush_native_grads()` is wrong — it's an optimizer-step API.** wgrad
    is fused into the GEMM TMA-reduce-add epilogue and lands directly in
    `_NATIVE_W{1,2}_GRAD`. The transpose to per-expert main_grad is amortized over
    `grad_acc_steps` microbatches via `node.step()`.
93. **Two NVTX-based GPU-proj measurements disagree by ~360 µs at this shape.**
    BENCH-range whole / n_iters includes inter-iter framework gaps; per-ITER NVTX
    excludes them. Document which one a number refers to.
94. **`torch.equal()` triggers `__nonzero__` ambiguity in paddle compat mode.**
    `torch.equal(t, zeros)` on a paddle tensor → `AssertionError: Variable... if/while`.
    Reduce to scalar via `float(t.float().abs().sum().item()) == 0.0`.
95. **Bench results from a contended GPU are useless.** Saw 4168 µs/iter on contended
    GPU 2 vs 2823 µs on idle GPU 7 for the same workload. Always
    `nvidia-smi --query-gpu=index,utilization.gpu,memory.used` before profiling.

---

## Phase 27: Sessions 67–75 — recorded only in root `HANDOFF.md`

Phases 27+ (S67 IMA bisect, S70 ncu-driven quant opt, S71 globals purge planning,
S72 quant NCU sweep + race-fix, S73 stream patch + scatter, S74 lazy main_grad +
step ordering, S75 Fleet integration audit + frontier validation) are NOT
duplicated here. See the root `HANDOFF.md` — newest session at the top, prior
sessions preserved verbatim below it.

> **Canonical handoff: root `HANDOFF.md`** (latest session at top).

