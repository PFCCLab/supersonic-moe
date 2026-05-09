# HANDOFF — SonicMoE FP8 Frontier (clean state, 2026-05-09)

> **Branch**: `race-fix-paddle`
>
> **Repository**: A-nnonymous/supersonic-moe (remote: `myrepo`)
>
> **Frontier status**: GREEN for the documented FP8 frontier path. Precision, determinism, stress, and integration contracts are understood; no known blocker in the default path.
>
> **Authoritative handoff**: this root `HANDOFF.md`. Historical notes in `reports/fp8_upgrade/engineering_log.md` are useful chronology only and contain superseded intermediate claims.

---

## 1. Project state in one paragraph

SonicMoE is a SM100/Hopper Mixture-of-Experts expert-MLP engine. The active production path on SM100 uses DeepEP topk metadata, route-level padding, blockscaled FP8 E4M3 + UE8M0 scales, CuTe/CUTLASS/QuACK GEMMs, zero-materialization `A_idx` gather, fused gated up-projection (`GEMM + SwiGLU + z FP8 epilogue quant`), FP8 down-projection, FP8-C-load `GemmDGated` backward, iso32 dz dual quant, and TMA reduce-add wgrad directly into ERNIE/Paddle `main_grad`. The Paddle integration entrypoint is `sonicmoe.ernie_compat.SonicMoEMlpNode`; `node.step()` must run before `optimizer.step()` to flush native CUTLASS wgrad layout into ERNIE layout.

---

## 2. Current performance, memory, and precision

### 2.1 Latest single-GPU training performance

Hardware/method: Target GPU SM100 (`SM100`, 148 SMs, HBM3e), nsys GPU-projection over BENCH NVTX range, `USE_QUACK_GEMM=1`, `SONIC_MOE_FP8_MODE=perf`, `SONIC_MOE_FP8_WGRAD=1`.

| Shape | FP8 busy | MFU vs 4500 TFLOPS | TFLOPS | Source |
|---|---:|---:|---:|---|
| `T=1024,H=3072,I=1536,E=8,K=8` | 566.0 µs | 27.32% | 1229 | `reports/fresh_benchmark_ws1/` |
| `T=2048,H=3072,I=1536,E=8,K=8` | 870.1 µs | 35.54% | 1599 | same |
| `T=4096,H=3072,I=1536,E=8,K=8` | 1459.1 µs | 42.39% | 1907 | same |
| **`T=8192,H=3072,I=1536,E=8,K=8`** | **2659.8 µs** | **46.51%** | **2093** | same; canonical Ernie shape |
| `T=16384,H=3072,I=1536,E=8,K=8` | 5224.9 µs | 47.35% | 2131 | same |
| `T=8192,H=3072,I=1536,E=16,K=8` | 2800.7 µs | 44.17% | 1987 | same |
| `T=8192,H=3072,I=1536,E=32,K=8` | 3187.5 µs | 38.81% | 1746 | same |
| `T=8192,H=4096,I=4096,E=8,K=8` | 8521.7 µs | **51.61%** | 2322 | measured MFU peak |

Baseline caveat:

- **FP8 vs true BF16** (nsys GPU-projection, same codebase): Ernie shape `2660 µs` (FP8) vs `4346 µs` (BF16, CuTe DSL GemmGatedSm100/GemmDGatedSm100, zero FP8 kernels) = **1.63x speedup**.
- **BF16 vs S53 official PyTorch**: Current BF16 is `4346 µs` vs S53's `3644 µs` (ratio 1.19x). The 19% gap comes from Paddle proxy dispatch overhead (~10%) plus FP8-infrastructure-loaded branch cost (~9%, documented in S53 engineering log Lesson 50). This is expected and NOT a regression.
- **True BF16 mode**: Set `SONIC_MOE_FP8_MODE=""` before import. `SonicMoEMlpNode` now respects this setting and dispatches to CuTe DSL BF16 GEMMs (GemmGatedSm100 + GemmDGatedSm100). No quantization kernels, no FP8 C-load.
- **Small batches**: FP8 crossover is around `T=3000-4000`.

### 2.2 Kernel breakdown at Ernie shape

Latest nsys GPU-projection per iter:

```text
1185 µs  44.1%  QuACK wgrad GEMMs (4 calls)
 441 µs  16.4%  GemmGatedSm100ZeroMatBlockscaledQuant (fwd up + SwiGLU + z quant)
 400 µs  14.9%  GemmDGatedFP8CLoadSm100ZeroMat (bwd actgrad + dSwiGLU)
 242 µs   9.0%  _colwise_quantize_and_pack (num_warps=1)
 148 µs   5.5%  token_gather_sum_kernel / token reduce-combine
 103 µs   3.8%  _dual_varlen_iso32_quantize (num_warps=1)
  83 µs   3.1%  _quantize_and_pack
  37 µs   1.4%  other broadcast/index kernels
─────────────────
~2639-2660 µs total
```

NCU headline for the 6 GEMMs (`reports/ernie_shape_ncu_s78b/`):

| Role | Tensor pipe | DRAM | L2 hit | regs/thread | Readout |
|---|---:|---:|---:|---:|---|
| fwd1 `GemmGated(fp8 swiglu epi)` | 64% | 10.5% | 88.8% | 168 | epilogue/register bound |
| fwd2 `GemmDefault` | 70% | 19.5% | 77.6% | 54 | mostly healthy |
| dgrad1 `GemmDGated(fp8 C-load)` | 42% | 22.5% | 61.3% | 168 | worst per-FLOP kernel; C-load/epilogue/register cliff |
| dgrad2 `GemmDefault` | 81% | 25.9% | 68.6% | 56 | near optimum |
| wgrad1 `GemmDefault` | 84% | 24.8% | 74.0% | 56 | near peak |
| wgrad2 `GemmDefault + main_grad add` | 81% | 15.7% | 78.7% | 54 | TMA add is essentially free |

### 2.3 Memory state

The current FP8 perf path is speed-oriented and keeps FP8 weight caches hot.

| Item | Current understanding |
|---|---|
| FP8 weight caches | Multiple physical layouts are cached for fwd/down/dgated/actgrad/wgrad. Cache keys include `data_ptr + inplace_version + shape/stride`; optimizer in-place updates naturally invalidate. |
| `z` activation | Saved as `z_fp8 + UE8M0 scales`, avoiding a persistent `z_bf16(TK,2I)` activation. Ernie `z_bf16` would be ~384 MiB; FP8 data is ~192 MiB plus scales. |
| `y1` | `y1_fp8 + scales` is passed across `_UpProjection` → `_DownProjection` via `_PREQUANTIZED_SCALES["fwd"]`; the BF16 logical object exists for autograd/flow but hot path consumes FP8. |
| backward peak | Dominated by `dz`, `y1s`, colwise/rowwise quant products, `dx_expanded`, and wgrad accumulators. Previous report gives ~1.3 GiB/layer activation peak at Ernie shape; exact peak depends on cache retention and stagewise-memory mode. |
| `SONIC_MOE_STAGEWISE_MEMORY=1` | Memory-saving mode; expect ~1.0-1.5 GiB peak savings at Ernie-like shape with ~3-5% cost. |
| `SONIC_MOE_FP8_RECOMPUTE_Z=1` | Saves roughly one active `z_fp8` lifetime but costs extra up-proj recompute. Default remains off for perf. |

Memory claims older than Session 65 are often apples-to-oranges: some measured no FP8 wgrad, some included per-iter `node.step()` flush, some used old QuACK overhead. Treat them as historical unless reproduced with `bench_mlpnode_topk_nsys.py` / current benches.

### 2.4 Precision and determinism

Precision vs BF16 gold:

| Tensor | Current result |
|---|---|
| output | cos ~0.9979 |
| dx | cos ~0.9975 |
| ds/router score grad | cos ~0.9971-0.9973 |
| dw1 | cos ~0.9975 |
| dw2 | cos ~0.9971-0.9972 |
| RRMSE | < 7.6% in covered precision suite |

Determinism:

- `tests/fp8_frontier_determinism_test.py` is a hard gate and proves bit-exact repeated fwd/bwd on the frontier path.
- This matters because TMA reduce-add, async TMA, and global FP8 caches could otherwise introduce bit-level drift.

Numerical caveats:

- iso32 dz dual quant is validated on real Ernie-like `dz` captures with downstream GEMM RRMSE ratio `1.000x`, but it is **not** a mathematical identity. Monitor `log2(block_amax / row_amax)` and zero ratio if using new distributions; fallback to 1x32 dual quant for high-risk distributions.
- TMA reduce-add is not a precision improvement. It keeps fp32 `main_grad` but changes the accumulation implementation from epilogue C-load to TMA store-side ADD. It does not do Kahan/pairwise accumulation.
- FP8 small values near zero can show large relative error; judge by cosine/RRMSE/absolute error and training loss, not max relative error alone.

---

## 3. Important implementation contracts

### 3.1 Training loop contract

```python
for step in range(num_steps):
    for mb in microbatches:
        out = node(dispatched_hidden_states, tokens_per_expert,
                   dispatched_indices, dispatched_probs)
        out.backward(grad)
    node.step()          # must run BEFORE optimizer.step()
    optimizer.step()
    optimizer.clear_grad()
```

`node.step()` flushes native CUTLASS accumulators:

```text
w1 native [E,2I,H] -> ERNIE [E,H,2I]
w2 native [E,H,I]  -> ERNIE [E,I,H]
```

### 3.2 Routing/metadata contract

- Production uses `deepep_topk_to_sonic_metadata()`.
- `dispatched_indices` is `[N_recv, topk] int32`; each row must contain distinct local expert ids where valid. `-1` means masked.
- Route-level padding pads each expert segment to 128 rows. Padding rows gather `x[0]` but use score 0, so they contribute no output or gradient.
- `x_gather_idx` lives in expert-sorted TK space and points into original token rows. This is the zero-materialization key.

### 3.3 JIT/cache contract

- Always `source .runenv.sh` on this host; it fixes Python/quack/ptxas/Paddle compat environment.
- CuTe compile keys must contain only static model dimensions (`H/I/E/dtype/tile`) and never `TK`, `total_M`, capacity, or ISA-packed scale sizes.
- Runtime fast-path caches may include exact shape but must have high-watermark eviction.
- Multi-process GPFS JIT is protected by `FileLock` and direct `.so` import fallback; heterogeneous cold starts are covered by `tests/ops/test_jit_concurrent_heterogeneous.py`.

---

## 4. High-value information sources

Read these before changing the frontier:

| Priority | Source | Why it matters |
|---:|---|---|
| 1 | `reports/sonic_moe_fp8_frontier_newcomer_guide.md` | Standalone training guide: basics, dataflow, symbols, roofline, precision, expert Q&A |
| 2 | `reports/fresh_benchmark_ws1/README.md` + `sweep.json` + `mfu_model.json` | Latest 22-point performance sweep and fitted MFU model |
| 3 | `reports/ernie_shape_ncu_s78b/README.md` | NCU full report and bottleneck reasoning for 6 GEMMs |
| 4 | `docs/gemm_dgated_fp8cload.md` | GemmDGatedFP8CLoad deep-dive: Int16 trick, dSwiGLU 9-SASS decomposition, FP8 blockscaled dequant, zero-materialization gather |
| 5 | `docs/expert_interleave_weight_layout.md` | Expert Interleave 权重布局设计动机 — 5 层全栈收益分析 + 独创性论证 |
| 6 | `reports/sonic_moe_comprehensive_analysis.md` | Broad analysis report and newcomer summary |
| 7 | `sonicmoe/functional/__init__.py` | `_UpProjection` / `_DownProjection` orchestration and FP8 state transfer |
| 8 | `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | zero-materialization SM100 specialization |
| 9 | `sonicmoe/quack_utils/gemm_gated.py` | fused gated GEMM + epilogue blockscaled quant |
| 10 | `sonicmoe/quack_utils/gemm_dgated.py` | FP8 C-load dGated backward |
| 11 | `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | quant kernels, iso32, TMA reduce-add, FP8 GEMM wrappers |
| 12 | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md` | machine, proxy, ncu lock reset, profiling methodology |

Historical but non-authoritative:

- `reports/fp8_upgrade/engineering_log.md`: chronological lessons only (Sessions 1–66). It has a current-state correction block at the top.
- `reports/fp8_upgrade/HANDOFF.md`: stale historical reference.

---

## 5. Deep technical insights (compacted from analysis sessions)

### 5.1 GemmDGatedFP8CLoad — the structural bottleneck

`GemmDGatedFP8CLoadSm100ZeroMat` (file: `sonicmoe/quack_utils/gemm_dgated.py:510-708`) is the most complex kernel in the forward/backward pipeline. Its GEMM is only $2MHI$ (not $4MHI$) because the actual matmul is $\partial o \cdot W_2$ with output shape $(M, I)$. The remaining computation is fused in the epilogue:

1. **FP8 C-Load via Int16 view trick**: The saved $z_{\text{fp8}}$ of shape $(M, 2I)$ is `view`ed as $(M, I)$ Int16. Each Int16 = `[z_g_fp8, z_u_fp8]` — two fp8 elements packed. This aligns C and D tile shapes for the CUTLASS epilogue framework without any framework changes.

2. **Blockscaled FP8 dequant in-register**: UE8M0 scale bytes are left-shifted 23 bits and reinterpreted as f32 power-of-two values (`_i32_as_f32(Int32(scale) << 23)`). This avoids any int-to-float conversion — exploits IEEE 754 biased exponent encoding directly.

3. **dSwiGLU fused at 9 SASS instructions per element pair**: `1 MUFU.TANH + 4 FMUL + 4 FMA` — algebraically minimal. The sigmoid is computed as `0.5 * tanh(0.5*x) + 0.5` via MUFU.TANH (¼ throughput of FMUL).

4. **Dual TMA store**: Outputs $\partial z_g$ and $\partial z_u$ as two FP8 streams + a BF16 `swiglu_out` stream via separate TMA descriptors.

**Why 168 regs/thread**: C-load smem layout + Int16 recast registers + dequant scale registers + dSwiGLU intermediates + dual TMA store coordinates. Block Limit = 1 (only 1 CTA per SM). No headroom for epilogue dz quant loops.

**Key profiled metrics**: TC pipe busy ~42%, DRAM ~22.5%, L2 hit ~61%. This kernel is epilogue-bound, not compute-bound or memory-bound in isolation.

### 5.2 Expert Interleave — system-level weight layout decision

SonicMoE stores $W_1$ as `[gate_row_0, up_row_0, gate_row_1, up_row_1, ...]` (interleaved) instead of the standard split-half `[all gate rows, all up rows]`. This is NOT a single-kernel optimization — it's a full-stack architectural decision with 5 cascading benefits:

1. **GEMM epilogue SwiGLU fusion** — gate/up pair lands in adjacent accumulator registers → no cross-tile access needed
2. **FP8 Int16 view trick** — physically adjacent `z_g, z_u` bytes can be `view`ed as a single Int16 → enables the C-load mechanism
3. **SwiGLU Triton kernel coalesced access** — stride-2 pattern hits same cacheline for both gate and up
4. **Wgrad GEMM native output** — CUTLASS wgrad naturally outputs interleaved → zero mid-iteration transpose
5. **FP8 blockscaled quantization symmetry** — each 32-element scale group mixes gate+up → error averaging

The layout conversion to ERNIE's split-half happens **only at `node.step()` time**, amortized over `grad_acc_steps` microbatches. See `docs/expert_interleave_weight_layout.md` for full derivation.

This design is enabled by self-authored CUTLASS-DSL GEMM kernels (QuACK) with programmable epilogues — frameworks using black-box GEMM libraries cannot realize these benefits.

### 5.3 MFU gap decomposition (updated 2026-05-07)

The newcomer guide Section 7 now contains the precise 6-GEMM FLOPs table and DGated real compute breakdown. Key insight: DGated's epilogue is only ~0.2% of its GEMM FLOPs, but causes ~15% TC pipe utilization loss because MUFU.TANH is ¼ throughput and 168 regs means no latency hiding.

Full MFU gap (100% → 46.5%):
- GEMM shape efficiency + varlen: ~46%
- Epilogue pipeline stall (DGated): ~15%
- Non-GEMM kernels: ~23%
- Fixed overhead: ~8%
- L2/DRAM contention: ~8%

---

## 6. Most valuable formulas and mental models

### 6.1 MFU

```text
F = 18 * TK * H * I
MFU = F / (busy_seconds * peak_FLOPs_per_second)

Ernie:
  TK = 8192 * 8 = 65536
  F = 18 * 65536 * 3072 * 1536 = 5.566e12 FLOPs
  ideal @4500 TFLOPS = 1237 µs
  measured = 2659.8 µs
  MFU = 46.51%
```

### 6.2 Empirical performance model

Fitted from multiple fresh sweep points, not one point:

```text
busy_us =
  18*TK*H*I / (4500e6 * eta_max)
  + a_quant * TK * max(H,2I) * 1e-9
  + a_expert * E * TK * 1e-6
  + c_fixed

eta_max = 0.541569
a_quant = 50.0
a_expert = 328.939
c_fixed = 201.047 µs
R^2 = 0.99896
```

Interpretation:

- `eta_max` is effective GEMM family efficiency, including shape/varlen/epilogue effects.
- `a_quant` is an empirical data-scale term, not the literal time of quant kernels.
- `a_expert` captures fragmentation and per-expert overhead as E increases.
- Refit if any fusion/fission/communication overlap changes the system.

### 6.3 iso32 precision risk

```text
extra_bits_lost = log2(block_amax_32x32 / row_amax_1x32)
```

If p95 exceeds ~1.5 bits or max exceeds ~2.5 bits, consider fallback to 1x32 dual quant for that tensor/expert/step. Thresholds need training calibration; this is a guardrail, not a theorem.

### 6.4 Fusion vs fission

Fuse when:

```text
saved_HBM + saved_launch + saved_allocator/cache
  >
extra_register_cost + lost_occupancy + lost_parallelism + extra_control
```

Fission when:

```text
occupancy/tensor_pipe gain + reduced spill + better tile shape
  >
extra HBM store/load + extra launch/grid sync + lost locality
```

Current application: `GemmDGatedFP8CLoadSm100ZeroMat` is a fission candidate because it has 168 regs/thread and only ~42% tensor-pipe utilization; direct epilogue fusion of dz quant is not the right next step.

---

## 7. Lessons learned / pitfalls

1. **FP8 真实加速比是 1.63x（vs 真 BF16）。** 之前报告的 “1.11x” 是由于 BF16 baseline 被 FP8 kernel 污染（SonicMoEMlpNode 强制 enable_fp8(True)），已在本次 session 修复。正确数据：FP8 2660 µs vs 真 BF16 4346 µs = 1.63x。历史 S53 cuBLAS BF16 = 3644 µs（当前 BF16 比 S53 慢 19%，原因是 Paddle proxy overhead + FP8 infra branch cost）。
2. **Do not try to directly add dz quant loops to GemmDGated epilogue.** NCU shows 168 regs/thread × 384 threads = 64512/65536 regs, leaving effectively no register headroom.
3. **compute-sanitizer can mask register-limit crashes.** Use it for memory safety, not as proof that a high-register kernel is production-safe.
4. **TMA reduce-add is performance, not higher precision.** It avoids C-load/register pressure; determinism must still be tested.
5. **iso32 is measured-scope safe, not universally exact.** Guard with amax-ratio/zero-ratio monitoring when changing distributions.
6. **compile_key must stay static.** Dynamic token dimensions in compile keys cause recompile storms.
7. **Paddle proxy differs from PyTorch.** `torch.equal`, stream handles, dtype strings, `_inplace_version`, storage offsets, and bf16 conversions all need compatibility handling.
8. **nsys and ncu answer different questions.** End-to-end busy/MFU uses nsys GPU-projection; kernel resource bottlenecks use ncu SoL/register/L2/DRAM. Do not compare their durations directly unless clock/replay policy matches.
9. **`node.step()` order is non-negotiable.** It must precede `optimizer.step()`.
10. **Use whitelisted env for paddlejob launch.** Denylist cleanup is unsafe; cluster env vars can silently force multi-node rendezvous.
11. **Never use `.numel()` or `.element_size()` in hot-path Paddle proxy code.** These trigger implicit cudaMemcpy D2H (GPU stream sync). Use `.size` (int property) and `.itemsize` instead. Gated by `tests/test_no_memcpy_sync.py`. See PR#22.
12. **SonicMoEMlpNode 现在支持真 BF16 模式。** 设置 `SONIC_MOE_FP8_MODE=""` 即可。BF16 路径使用 GemmGatedSm100 + GemmDGatedSm100（CuTe DSL BF16 GEMM），wgrad 通过 gemm_add() 累加到 main_grad，与 FP8 路径共享 zero-materialization 和 varlen 基础设施。
13. **MXFP8 128 行对齐浪费是硅片硬约束，kernel 层面无解。** 详细分析见 `/panzhaowu/bkup/mxfp8_alignment_waste_analysis.pdf`。TC:CUDA core 吞吐比 = 58.5:1，CUDA core 并发 tail 不可行（tail 必须 < 0.67% 才能隐藏）。系统级解法：token rounding（已实现未上线）或跨 microbatch buffer（可行，改善 = grad_acc_steps 倍）。
14. **Token Rounding 已实现但未进入生产。** 代码在 `forward_token_choice_rounding(Mtile=128)`，但因路由扰动对训练收敛的影响未验证，生产环境仍用 route-level padding。

---

## 8. Next plan

### P0 — dgrad1 structural optimization (GemmDGatedFP8CLoad)

Target: `GemmDGatedFP8CLoadSm100ZeroMat` — the worst per-FLOP kernel (42% TC pipe).

Facts:

- 168 regs/thread, near register cliff. Block Limit = 1.
- Tensor pipe ~42%, L2 hit ~61%, DRAM ~22.5%.
- Direct dz quant epilogue fusion is NOT viable (zero register headroom).
- The MFU loss from this kernel’s epilogue stall is ~15% of total gap.
- MUFU.TANH (sigmoid) is the SFU bottleneck: ¼ throughput of FMUL.

Promising directions (ranked by expected ROI):

1. **Live-range shortening in FP8 C-load + dSwiGLU epilogue** — the Int16 recast path holds `tRS_rXY_f32x2` and dequant scales across the full epilogue iteration. If scale reuse or register aliasing can free 4-8 regs, occupancy may jump to Block Limit = 2.
2. **Fission: split GEMM mainloop from dSwiGLU+store** — GEMM alone would be ~56 regs, full occupancy. dSwiGLU+store as a separate small kernel reading accumulator from smem/L2. Trade: +1 kernel launch + accumulator spill to smem (~12 MiB per tile). Needs careful modeling.
3. **C-load scheduling: prefetch scales while GEMM tiles still running** — overlap scale TMA with mainloop; currently scales are loaded only in `epi_visit_subtile`.
4. **Sigmoid approximation** — replace MUFU.TANH with polynomial approximation using FMUL/FMA (4× throughput). Risk: precision regression in dSwiGLU chain.

### P1 — Training-side persistent pipeline research

Inference megakernels don’t transfer to training. A realistic path:

- Stage-wise persistent pipeline (fwd1 → fwd2 → dgrad1 → dgrad2) with producer-consumer via smem/L2 — not a single-kernel solution, but a multi-kernel persistent occupancy scheme.
- Pre-requisite: dgrad1 occupancy improvement (P0) so it can share SMs with other kernels.

### P2 — Monitoring and safeguards

- Add optional runtime audits for iso32 `log2(block_amax/row_amax)` and zero ratio.
- Keep frontier determinism hard-gated.
- Add tests around `node.step()` ordering / repeated accumulation / cache invalidation.

### P3 — Documentation maintenance

- Keep `reports/sonic_moe_fp8_frontier_newcomer_guide.md` as the newcomer source (updated Section 7: MFU).
- Keep `docs/gemm_dgated_fp8cload.md` and `docs/expert_interleave_weight_layout.md` for deep technical reference.
- Keep README aligned with this handoff.
- Treat `engineering_log.md` as history only.

---

## 9. Validation commands for the next agent

```bash
source .runenv.sh

# Hard determinism gate
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/fp8_frontier_determinism_test.py -v

# Stress / routing robustness
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/fp8_frontier_stress_test.py -v

# Full regression when changing kernels or metadata
bash tests/run_regression.sh
```

For profiling:

```bash
nsys profile --trace=cuda,nvtx --sample=none --backtrace=none \
  --resolve-symbols=false --export=sqlite --output=OUTPUT \
  python tests/ops/bench_mlpnode_topk_nsys.py --T 8192 --E 8 --I 1536 --topk 8
```

If ncu exits abnormally and locks clocks, run:

```bash
ncu --clock-control=reset
```
