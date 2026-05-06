# WS1-WS4 综合结果报告 (Fresh Data, 2026-05-06)

> 硬件: NVIDIA B30Z (sm_103, 148 SMs, 2032 MHz, 268 GiB HBM3e)
> 分支: race-fix-paddle | 方法: nsys GPU-projection (BENCH NVTX range)
> node.step() 在 BENCH 循环外 (匹配多 microbatch 稳态)

---

## 1. Fresh Benchmark 数据

### FP8 Frontier (SONIC_MOE_FP8_MODE=perf)

| Shape | busy µs/iter | MFU | TFLOPS |
|-------|:---:|:---:|:---:|
| T1024-H3072-I1536-E8-K8 | 566.0 | 27.32% | 1229 |
| T2048-H3072-I1536-E8-K8 | 870.1 | 35.54% | 1599 |
| T4096-H3072-I1536-E8-K8 | 1459.1 | 42.39% | 1907 |
| **T8192-H3072-I1536-E8-K8 (Ernie)** | **2659.8** | **46.51%** | **2093** |
| T16384-H3072-I1536-E8-K8 | 5224.9 | 47.35% | 2131 |
| T8192-H3072-I1536-E16-K8 | 2800.7 | 44.17% | 1987 |
| T8192-H3072-I1536-E32-K8 | 3187.5 | 38.81% | 1746 |
| T4096-H4096-I2048-E8-K8 | 2239.8 | 49.09% | 2209 |
| T8192-H4096-I2048-E8-K8 | 4267.3 | 51.53% | 2319 |
| T8192-H4096-I4096-E8-K8 | 8521.7 | 51.61% | 2322 |
| T8192-H6144-I2048-E8-K8 | 6568.4 | 50.22% | 2260 |

**Peak MFU: 51.61%** (T8192-H4096-I4096-E8)

### BF16 Baseline (CuTe DSL BF16 GEMM, 无 FP8)

| Shape | busy µs/iter | MFU (vs BF16 peak 2250T) | TFLOPS |
|-------|:---:|:---:|:---:|
| T1024-H3072-I1536-E8-K8 | 540.0 | 57.27% | 1288 |
| T2048-H3072-I1536-E8-K8 | 847.7 | 72.96% | 1642 |
| T4096-H3072-I1536-E8-K8 | 1533.4 | 80.67% | 1815 |
| **T8192-H3072-I1536-E8-K8 (Ernie)** | **2942.5** | **84.07%** | **1892** |
| T16384-H3072-I1536-E8-K8 | 6022.1 | 82.16% | 1849 |
| T8192-H3072-I1536-E16-K8 | 3074.1 | 80.48% | 1811 |
| T8192-H3072-I1536-E32-K8 | 3218.4 | 76.87% | 1730 |
| T4096-H4096-I2048-E8-K8 | 2488.0 | 88.39% | 1989 |
| T8192-H4096-I2048-E8-K8 | 5037.9 | 87.30% | 1964 |
| T8192-H4096-I4096-E8-K8 | 10894.7 | 80.74% | 1817 |
| T8192-H6144-I2048-E8-K8 | 7691.2 | 85.77% | 1930 |

### Speedup (FP8 vs BF16)

| Shape | BF16 µs | FP8 µs | Speedup |
|-------|:---:|:---:|:---:|
| T1024-H3072-I1536-E8-K8 | 540.0 | 566.0 | **0.95x** (FP8 slower!) |
| T2048-H3072-I1536-E8-K8 | 847.7 | 870.1 | **0.97x** (FP8 slower!) |
| T4096-H3072-I1536-E8-K8 | 1533.4 | 1459.1 | **1.05x** |
| **T8192-H3072-I1536-E8-K8** | **2942.5** | **2659.8** | **1.11x** |
| T16384-H3072-I1536-E8-K8 | 6022.1 | 5224.9 | **1.15x** |
| T8192-H4096-I2048-E8-K8 | 5037.9 | 4267.3 | **1.18x** |
| T8192-H4096-I4096-E8-K8 | 10894.7 | 8521.7 | **1.28x** |
| T8192-H6144-I2048-E8-K8 | 7691.2 | 6568.4 | **1.17x** |

**关键发现**: FP8 在 T < 4096 时**反而比 BF16 慢**！FP8 量化开销在小 batch 时无法被 2× tensor-core throughput 抵消。crossover 点在 T ≈ 3000-4000。

### 与历史数据的一致性验证

| 指标 | S80c 报告 | 本次测量 | 偏差 |
|------|:---:|:---:|:---:|
| FP8 Ernie busy_us | 2693 | 2659.8 | −1.2% ✓ |
| FP8 peak MFU | 51.53% | 51.61% | +0.08pp ✓ |
| BF16 Ernie (S53 参考) | 3644 | 2942.5 | −19.2% ← 见下注 |

**注**: BF16 历史数据 3644 µs 来自 Session 53 使用 **PyTorch native cuBLAS BF16**。本次 2942.5 µs 使用 **CuTe DSL BF16 GEMM (USE_QUACK_GEMM=1)**，效率更高。两者不是同一个 baseline。

---

## 2. MFU 理论模型

### FP8 模型 (R² = 0.999)

```
busy_us = 18·TK·H·I / (4500e6 × η_max)  +  a_q × TK × max(H,2I) × 1e-9  +  a_e × E × TK × 1e-6  +  c

参数:
  η_max  = 0.5416  (GEMM 效率天花板)
  a_q    = 50.0    (FP8 量化: 50 ns 处理 1 GB data)
  a_e    = 329     (per-expert 开销: 329 µs per million E·TK)
  c      = 201 µs  (固定 per-iter 开销: Python launch, metadata)
```

**MFU 渐进公式**:
```
MFU_∞(H, I, E) = 1 / (1/η_max + peak·a_q·max(H,2I)/(18·H·I) + peak·a_e·E/(18·H·I))

Ernie (H=3072, I=1536, E=8):  MFU_∞ = 50.1%
Wide  (H=6144, I=2048, E=8):  MFU_∞ = 52.5%
```

**物理含义**:
- η_max = 54.16% → 即使无任何 overhead，FP8 GEMM 也只能达到 peak 的 54%（non-square shape + varlen penalty）
- 每倍增 E: 额外 329 µs × E × TK/1e6 开销 → 解释了 E=8→16→32 的 MFU 下降
- crossover T: 当 `18·TK·H·I/(4500e6×η) ≈ a_q·TK·max(H,2I)·1e-9` 时 overhead=GEMM → T_cross ≈ 3000

### BF16 模型 (R² = 0.997)

```
η_max  = 0.8392  (BF16 GEMM 效率极高!)
a_q    = 50.0    (combine/routing overhead, 无 FP8 quant)
a_e    = 97.2    (per-expert 开销: BF16 比 FP8 低 3.4×)
c      = 0 µs    (BF16 路径几乎无固定开销)
```

**为什么 BF16 效率如此高**: BF16 path 无需:
- FP8 量化 kernel (7 个 Triton kernel)
- Weight cache 管理
- Scale gather/pack
- 只有纯 GEMM + SwiGLU + combine

---

## 3. 稀疏性审查结论

**推论判定**: 部分正确。详见 `reports/sparsity_audit_ws2.md`。

- SonicMoE **代码本身正确处理 K<topk** (通过 -1 masking + variable-K combine)
- Bench 使用 topk=E (全本地) → 是 kernel 效率上界，不代表生产
- **无正确性问题** (score=0 的 IEEE 754 保证)
- **显存损失极微** (route-level padding < 2%)

---

## 4. Score 加权位置分析结论

**当前实现 (combine-level) 严格优于 pre-SwiGLU 方案**:
- 精度: s 在 fp32 epilogue 中乘入，无额外量化误差
- 显存: 只需 TK×4B 的 score 存储（vs TK×I×2B 的 s×y1）
- 性能: 融合进 GEMM epilogue，零额外 kernel

详见 `reports/score_weighting_analysis_ws3.md`。

---

## 5. 下一步: Phase 1A (WS5)

基于本次 fresh data，Phase 1A (dz epilogue fusion) 的预期收益:
- 消除 `_dual_varlen_iso32_quantize_kernel`: 当前占 5.6% = ~149 µs
- 消除 bf16 dz 的 HBM 写入 (384 MiB) + 后续重读
- 预期净节省: 100-180 µs → MFU 从 46.5% → 48-50%

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `reports/fresh_benchmark_ws1/sweep.json` | 22 个数据点 (原始 JSON) |
| `reports/fresh_benchmark_ws1/mfu_model.json` | 模型参数 |
| `tools/bench_sweep_fresh.py` | 8-GPU 并发 sweep 脚本 |
| `tools/mfu_model.py` | MFU 理论模型拟合脚本 |
| `reports/sparsity_audit_ws2.md` | 稀疏性审查 |
| `reports/score_weighting_analysis_ws3.md` | Score 加权分析 |
