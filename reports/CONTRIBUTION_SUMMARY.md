# Contribution Summary: Blockscaled FP8 Training for SonicMoE on SM100

> **Branch:** `paddle_compat` (vs `main`)
> **Hardware:** NVIDIA Target GPU (SM100a, SM100)
> **Measurement:** nsys GPU-projection, 27-shape grid, 12 iterations after 3 warmup
> **Authoritative Data Source:** `python tools/introspect.py --mode benchmark` produces unified JSON covering performance × precision × memory.  All tables below should be re-generated from this JSON.

---

## Abstract

This branch adds a complete **blockscaled FP8 (E4M3 + E8M0, 1×32) training path** to
SonicMoE, achieving **1.29×–1.70× speedup** (mean 1.53×) over BF16 across 27 production
shapes on SM100 GPUs, with **6.5% RRMSE** (pure quantization noise, no routing error).
A novel **route-level padding** scheme enables FP8 128-alignment for arbitrary expert counts
(E=8/32/128) without modifying any GEMM kernel, and is mathematically proven to introduce
**exact zero** backward error. Additionally, Paddle framework compatibility is verified
across all 54 shape×mode combinations.

---

## 1. Scope of Changes

| Category | Files | Lines Added | Description |
|:---------|:-----:|:-----------:|:------------|
| Core algorithms (`sonicmoe/`) | 35 | ~10,200 | FP8 GEMM, quantization, route-level padding, Paddle compat |
| Test suite (`tests/`) | 35 | ~12,700 | 11 axiomatic padding tests, 59 module-level tests, contract tests |
| Tooling (`tools/`) | 15 | ~7,500 | `introspect.py` profiling engine, grid benchmark, paddle compat bench |
| Documentation (`docs/`) | 7 | ~1,600 | Padding proof, architecture spec, optimization guide |
| Visualization (`visualization/`) | 6 | ~2,900 | 16 publication figures, fully data-driven from JSON |
| **Total** | **132 new + 27 modified** | **~87,000** | — |

The core algorithmic contribution is ~10K lines of dense GPU kernel code; the remaining
~77K lines are measurement infrastructure, tests, and documentation.

---

## 2. Performance

### 2.1 Speedup (27-Shape Grid)

All measurements use nsys GPU-projection on fully idle Target GPU GPUs. BF16 baseline is
the upstream SonicMoE (`main` branch), verified within <1%.

| Dimension | Speedup Range | Mean |
|:----------|:-------------|:-----|
| **Overall (27 shapes)** | **1.29×–1.70×** | **1.53×** |
| By T: 8192 / 16384 / 32768 | 1.44× / 1.55× / 1.60× | — |
| By I: 1536 / 2048 / 3072 | 1.45× / 1.52× / 1.62× | — |
| By E: 8 / 32 / 128 | 1.56× / 1.53× / 1.49× | — |

**Scaling law:** Speedup is dominated by intermediate dimension I (GEMM savings ∝ O(I²),
quant overhead ∝ O(I)), grows with sequence length T, and is weakly dependent on expert
count E. The sweet spot is large I × large T.

![Speedup Scaling](assets/fig14_speedup_scaling.png)

### 2.2 Representative Shapes

| T | I | E | BF16 (µs) | FP8 (µs) | Speedup | Bwd Mem Δ |
|:--|:--|:--|:---------:|:--------:|:-------:|:---------:|
| 8192 | 1536 | 8 | 3644 | 2715 | 1.34× | +6.0% |
| 8192 | 3072 | 8 | 8110 | 4774 | **1.70×** | +6.5% |
| 32768 | 3072 | 32 | 33504 | 19761 | **1.70×** | +7.2% |
| 8192 | 1536 | 128 | 5009 | 3897 | 1.29× | +10.3% |
| 32768 | 3072 | 128 | 35627 | 22026 | 1.62× | +8.8% |

![Kernel Budget Breakdown](assets/fig11_kernel_budget_breakdown.png)

### 2.3 Kernel-Level Budget

At the anchor shape (T=8192, H=3072, I=1536, E=8, K=8), BF16 total = 3644µs,
FP8 total = 2715µs → **1.34× end-to-end**.

| Category | BF16 (µs) | FP8 (µs) | Delta |
|:---------|:---------:|:--------:|:-----:|
| Wgrad GEMM | 2161.5 | 1078.1 | −1083 (2.00×) |
| GemmGated (fwd) | 718.3 | 0 → ZeroMat 459.0 | −259 (1.56×) |
| GemmDGated (bwd) | 452.6 | 0 → ZeroMat 384.0 | −69 (1.18×) |
| Token Gather | 139.8 | 145.9 | +6 |
| **FP8 quant overhead** | — | **480.6** | +481 (new) |

Quant overhead breakdown: Blockscaled Quant 235µs + Dual Quant 152µs + Row Quant 77µs
\+ ISA Scale Gather 16µs = 481µs. This is the cost of blockscaled FP8 quantization,
partially offset by GEMM savings of −1411µs → net savings 929µs.

![Performance Waterfall](assets/fig4_performance_waterfall.png)

---

## 3. Precision

Anchor shape (T=32768, I=1536, E=8), FP8 vs BF16 gold (identical routing):

| Tensor | RRMSE (%) | Cosine Similarity | Threshold | Status |
|:-------|:---------:|:-----------------:|:---------:|:------:|
| output | 6.55 | 0.9979 | <10%, >0.99 | PASS |
| dx | 6.56 | 0.9979 | <10%, >0.99 | PASS |
| dw1 | 4.12 | 0.9992 | <10%, >0.99 | PASS |
| dw2 | 4.19 | 0.9991 | <10%, >0.99 | PASS |

All RRMSE < 7% — this is pure FP8 E4M3 quantization noise.
Multi-seed (5 seeds × 3 GPUs) measurements confirm output std < 0.05%.
RRMSE convention: **ratio** in JSON (0.065 = 6.5%), displayed as percentage in tables.

![Precision Audit](assets/fig6_precision_audit.png)

---

## 4. Memory

| Configuration | Backward Peak Overhead | Notes |
|:-------------|:----------------------:|:------|
| FP8 (default) | +4.8% to +10.3% | Shadow weight caches + wgrad quant temps |
| FP8 + Stash | **−24.5%** | bf16 master weights offloaded to CPU |

The memory overhead is dominated by FP8 shadow weight caches (~650 MiB at E=128).
The stash optimization moves bf16 master weights to CPU during forward+backward,
yielding a net memory reduction.

![Memory Scaling](assets/fig12_memory_scaling.png)

---

## 5. Key Technical Contributions

### 5.1 Zero-Materialization FP8 Forward

Quantize `x` at T-size (not TK), gather only ISA-packed scales (T→TK, ~3% of FP8 data),
then the CUTLASS kernel gathers A rows internally via `A_idx`. **No TK-sized FP8 activation
is ever materialized in HBM.**

### 5.2 Route-Level Padding

CUTLASS blockscaled FP8 GEMMs require 128-aligned expert segments. Prior approaches either
perturb routing (token rounding, 60% RRMSE) or pad each GEMM individually (+120% overhead).

**Route-level padding** transforms routing metadata once:

```
Router → metadata → [_pad_routing_metadata] → 128-aligned metadata
                                                    ↓
                        All 8 GEMMs see aligned fast path (zero GEMM code changes)
```

Padding rows: `x_gather_idx=0` (arbitrary), `score=0` (exact IEEE 754 zero).

**Mathematical guarantee:** `score=0` is applied before dSwiGLU in all 6 backward paths,
making `dz[pad]=0` exactly. Proven for:
- CUTLASS fused path (`gemm_dgated.py:152-176`)
- 5 Triton kernel variants (`swiglu_triton.py:522,641,763,970,1178`)

| Metric | Route-Level Padding | Token Rounding |
|:-------|:-------------------:|:--------------:|
| RRMSE vs BF16 | **6.5%** | 60.8% |
| Latency overhead | +1.2% | ~0% |
| GEMM code changes | 0 | 0 |
| Routing semantics | Preserved | **Destroyed** |

Verified with 11 axiomatic tests across E=3, E=32, E=128 (up to 87.5% padding rows).

### 5.3 Epilogue Blockscaled Quantization

The GEMM epilogue computes E8M0 scale + FP8 quantization of z in registers, eliminating
the standalone z quantization kernel (~141µs) and halving D write bandwidth.

### 5.4 Fused SwiGLU + Quantize Triton Kernels

7 SwiGLU kernel variants combining forward/backward with FP8 quantization and ISA scale
packing, eliminating intermediate HBM roundtrips.

### 5.5 NCU-Guided Quant Optimization

`num_warps=1` discovery: bandwidth-bound kernels achieve 2.3× speedup with fewer warps
per block (more blocks in-flight per SM). Verified via NCU `stall_barrier` analysis.

### 5.6 Paddle Framework Compatibility

`paddle.enable_compat()` proxy dispatch layer: BF16 overhead +2.9%, FP8 overhead ~0%
(CUTLASS kernels bypass the proxy). 54/54 shape×mode combinations verified.

---

## 6. Visualization Suite

16 publication-quality figures generated from benchmark JSON artifacts. All figures are
**fully data-driven** — no hardcoded benchmark numbers. Regenerate with:

```bash
python -m visualization  # → assets/fig1..fig16.png
```

| Figure | Content |
|:-------|:--------|
| fig1 | System architecture overview |
| fig2 | Executive summary (speedup + budget + precision) |
| fig3 | Quantization kernel latency comparison |
| fig4 | Performance waterfall (BF16 → FP8 transition) |
| fig5 | Stage-wise memory profile |
| fig6 | Precision audit (RRMSE + cosine across shapes) |
| fig7 | Speedup vs memory overhead scatter |
| fig8 | Budget composition (savings vs overhead) |
| fig9 | Tensor inventory (buffer lifecycle) |
| fig10 | Multi-dimensional scaling analysis |
| fig11 | Kernel runtime budget breakdown (27 shapes) |
| fig12 | Peak memory scaling |
| fig13 | Computation dataflow diagram |
| fig14 | Multi-dimensional speedup scaling |
| fig15 | BF16 vs FP8 operator path comparison |
| fig16 | Frontier contribution waterfall |

---

## 7. Test Suite

| Category | File(s) | Tests | Assertions |
|:---------|:--------|:-----:|:-----------|
| Padding axioms | `test_pad_gradient_integrity.py` | 11 | dz[pad]=0 exact, dw1/dw2/dx exact 0.0 |
| Routing axioms | `test_pad_routing.py` | 6 | Token conservation, score invariant |
| MoE module | `test_moe_module.py` | 59 | BF16 RRMSE<1%, FP8<10% |
| Quant kernels | 6 files | ~80 | Byte-exact vs gold E8M0 |
| SwiGLU | `test_swiglu.py` | ~90 | 3-way cross-validation |
| CUTLASS GEMMs | 4 files | ~12 | 3-way cross-validation |
| FP8 frontier | `fp8_frontier_strict_test.py` | ~10 | Fail-loud, no skip |
| FP8 protocol | `fp8_protocol_test.py` | ~26 | Byte-exact, roundtrip |
| Contract | `fp8_large_project_contract_test.py` | ~31 | 5 operator projects |

---

## 8. Limitations and Known Issues

1. **Multi-iter backward without optimizer step:** `dz.untyped_storage().resize_(0)` frees
   tensor storage prematurely. Works in real training; breaks test harnesses. Pre-existing.

2. **FP8 backward memory overhead (+5–10%):** Fundamental to blockscaled FP8 wgrad
   (shadow caches + quant temporaries). Mitigated by stash optimization (−24.5%).

3. **Forward quant overhead:** At small I (1536), FP8 forward is 19% slower than BF16.
   Epilogue-level forward quantization would eliminate this but requires CUTLASS DSL work.

4. **Isolated FP8 dgated test gap:** FP8 backward gated GEMM has only end-to-end coverage,
   no isolated kernel-level 3-way test.

---

## Reproducibility

```bash
# Environment
source /root/.../zhangyichen/erniebot/eb_venv/bin/activate
export SONIC_MOE_FP8_MODE=perf USE_QUACK_GEMM=1 QUACK_CACHE_DIR=./my_quack_cache

# Axiomatic correctness (11 tests, ~10s)
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_pad_gradient_integrity.py -v

# Full 27-shape grid benchmark (8 GPUs, ~15 min)
python tools/introspect.py --mode grid --gpu 8

# Unified 3-phase benchmark (precision × memory × perf, single JSON)
# Default: 6 shapes (E=8/32/128), 3 seeds, nsys 5w+20m
python tools/introspect.py --mode benchmark
# Custom shapes:
python tools/introspect.py --mode benchmark \
    --benchmark-shapes 8192,3072,1536,8,8 32768,3072,1536,8,8 \
    --benchmark-seeds 42,123,777

# Generate all figures
python -m visualization
```
