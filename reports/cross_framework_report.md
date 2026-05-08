# Cross-Framework MoE Benchmark Report (4-Way)

Generated: 2026-04-16 10:47:46  |  GPU: NVIDIA Target GPU (SM100)  |  Seeds: [42, 123, 777]
nsys: 5 warmup + 20 measured × 3 repeats (median)

## 1. Experimental Setup

| Path | Framework | Python | Compute | API |
|------|-----------|--------|---------|-----|
| Paddle BF16 | PaddlePaddle | 3.10 | BF16 matmul | `ExpertsGroupGemmContiguousNode(fp8=None)` fwd+bwd |
| Paddle FP8 | PaddlePaddle | 3.10 | FP8 (kitchen+deep_gemm) | `ExpertsGroupGemmContiguousNode(fp8='e4m3')` fwd+bwd |
| SonicMoE BF16 | PyTorch | 3.13 | BF16 QuACK GEMM | `MoE(...)` full module (same as introspect.py) |
| SonicMoE FP8 | PyTorch | 3.13 | FP8 blockscaled CUTLASS | `MoE(..., use_fp8=True)` + `enable_fp8(True)` |

### Shapes

| Label | S | H | I | E | K | TK | FLOPs (G) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|--------:|
| T8192_H3072_I1536_E8_K8 | 8192 | 3072 | 1536 | 8 | 8 | 65536 | 1855.4 |

## 2. Precision Analysis

### 2.1 Forward Output `output (S, H)` vs Float64 Gold

**Tensor**: final MoE output after down-proj + `F.moe_unpermute` (Paddle) / router scatter (SonicMoE).

**Two gold references** (prob-scaling location differs):
- **Gold-ERNIE**: `output = (swiglu(x@w1) * prob) @ w2`, accumulate at scatter (prob BEFORE down-proj)
- **Gold-SonicMoE**: `output = swiglu(x@w1) @ w2 * prob` (prob AFTER down-proj at scatter)

Note: intermediate tensors (preact, postact, expert_out) are NOT comparable cross-framework due to
different padding/rounding strategies (`F.moe_permute` vs `TC_topk_router_metadata_triton`).
Only `output (S, H)` is permutation-invariant and comparable.

Metrics averaged over 3 seeds.

### T8192_H3072_I1536_E8_K8

| Path | RRMSE | Cosine | Max |err| | Mean |err| | PASS? |
|------|------:|-------:|--------:|--------:|:-----:|
| Paddle BF16 | 0.005500 | 0.999985 | 5.1638e-06 | 4.0111e-07 | PASS |
| Paddle FP8 | 0.065258 | 0.997869 | 4.7570e-05 | 4.7753e-06 | PASS |
| SonicMoE BF16 | 0.004691 | 0.999989 | 4.5670e-06 | 3.4128e-07 | PASS |
| SonicMoE FP8 | 0.065298 | 0.997867 | 4.7506e-05 | 4.7782e-06 | PASS |

**Pairwise (seed-averaged):**

| A vs B | RRMSE | Cosine |
|--------|------:|-------:|
| SonicMoE BF16→FP8 dx | 0.070398 | 0.997520 |
| Paddle FP8 vs Paddle BF16 | 0.065258 | 0.997869 |
| SonicMoE FP8 vs SonicMoE BF16 | 0.065236 | 0.997871 |
| Paddle FP8 vs SonicMoE FP8 | 0.037777 | 0.999287 |
| Paddle BF16 vs SonicMoE BF16 | 0.004948 | 0.999988 |
| Gold ERNIE vs Gold SonicMoE | 0.000000 | 1.000000 |

## 3. Peak Memory (MiB) — fwd + bwd

Staged measurement: baseline (weights+routing) → post-fwd (+ saved tensors) → peak (fwd+bwd max).

| Shape | Path | Baseline | Post-Fwd | Peak (fwd) | Peak (fwd+bwd) | Fwd Delta | Bwd Delta |
|-------|------|--------:|--------:|--------:|--------:|--------:|--------:|
| T8192_H3072_I1536_E8_K8 | Paddle BF16 | 696.8 | 1897.0 | 2425.3 | 4402.0 | 1200.2 | 432.3 |
| T8192_H3072_I1536_E8_K8 | Paddle FP8 | 728.8 | 1929.3 | 2014.8 | 2843.8 | 1200.5 | 48.0 |
| T8192_H3072_I1536_E8_K8 | SonicMoE BF16 | 697.3 | 1177.3 | 1753.3 | 2129.3 | 480.0 | -72.0 |
| T8192_H3072_I1536_E8_K8 | SonicMoE FP8 | 623.1 | 1028.5 | 1696.4 | 2178.5 | 405.4 | 114.0 |

## 4. Performance — nsys GPU-Projection

**Metric**: merged GPU kernel busy time (sweep-line interval merge over `CUPTI_ACTIVITY_KIND_KERNEL`).
Warmup: 5 iters (discarded).  Measured: 20 iters.  Repeats: 3 (report median).
All 4 paths: **forward + backward** (permute → expert fwd → unpermute → expert bwd).

*nsys profiling skipped or failed.*

## 5. Theoretical Analysis

### Roofline Model (Target GPU)

- BF16 tensor-core peak: 2250 TFLOPS
- FP8 tensor-core peak: 4500 TFLOPS
- HBM3e bandwidth: 8000 GB/s

| Shape | FLOPs (G) | AI (BF16) | AI (FP8) | Compute Bound BF16 (us) | Compute Bound FP8 (us) | Mem Bound BF16 (us) |
|-------|--------:|------:|-----:|----------:|----------:|----------:|
| T8192_H3072_I1536_E8_K8 | 1855.4 | 1134.3 | 2268.6 | 824.6 | 412.3 | 204.5 |

### Efficiency Analysis

### FP8 Quantization Error Model

E4M3 FP8: 3 mantissa bits → per-element relative error ε ≈ 2^{-4} ≈ 6.25%.
MoE pipeline has 2 quantized matmuls (up-proj, down-proj) plus SwiGLU.
Expected RRMSE ≈ √(ε_up² + ε_act² + ε_down²) ≈ 0.05–0.08.
Both ERNIE and SonicMoE use blockscaled E4M3 with E8M0 power-of-2 scales,
so their FP8 errors should be comparable (differences from block size and fusion).

## 6. Conclusions

### T8192_H3072_I1536_E8_K8

- **Paddle BF16**: RRMSE=0.0055
- **Paddle FP8**: RRMSE=0.0653
- **SonicMoE BF16**: RRMSE=0.0047
- **SonicMoE FP8**: RRMSE=0.0653

---
*Benchmark: `tools/cross_framework_bench.py`*
