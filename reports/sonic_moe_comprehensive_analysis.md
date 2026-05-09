# SonicMoE 综合技术分析报告

> 撰写日期: 2026-05-06 | 分支: `race-fix-paddle` | 硬件: NVIDIA Target GPU (SM100, 148 SMs, 2032 MHz boost, 268 GiB HBM3e)
> 本文档面向新接手的管理者和开发者，系统梳理 FP8 frontier 的实现、性能分析和 MoE 架构设计。

---

## 目录

1. [FP8 Frontier vs BF16 Baseline Breakdown 对照表](#1-fp8-frontier-vs-bf16-baseline-breakdown-对照表)
2. [FP8 Frontier 创新点与优化点分析](#2-fp8-frontier-创新点与优化点分析)
3. [MoE 结构性能 / 显存 Roofline 分析](#3-moe-结构性能--显存-roofline-分析)
4. [MoE 实现与优化手段完整讲解（面向新人）](#4-moe-实现与优化手段完整讲解面向新人)
5. [CI 测试基础设施与质量门禁](#5-ci-测试基础设施与质量门禁)

---

## 1. FP8 Frontier vs BF16 Baseline Breakdown 对照表

### 1.1 端到端性能对比 (nsys GPU-projection, Ernie shape T=8192, H=3072, I=1536, K=8)

| 指标 | BF16 Baseline (S53) | FP8 Frontier (S80c) | 加速比 | 说明 |
|------|:---:|:---:|:---:|------|
| **E=8 总耗时** | 3644 µs | **2659.8 µs** | **1.37×** | fresh S81 data; vs historical cuBLAS/PyTorch BF16 |
| **E=32 总耗时** | 3844 µs | **3283 µs** | **1.17×** | 更多专家→更多routing开销 |
| **T=16384 E=8** | 7953 µs | **5324 µs** | **1.49×** | 大batch更有利 |
| **T=16384 E=32** | 8129 µs | **5916 µs** | **1.37×** | |
| **MFU (Ernie)** | ~30% est. | **46.51%** | +16pp | 4500 TFLOPS peak |
| **MFU (峰值)** | — | **51.61%** | — | H4096-I4096-E8 fresh sweep |

### 1.2 算子级别 Breakdown (T=8192, E=8, K=8, FP8 路径)

| 算子类别 | Kernel名称 | 调用次数/iter | 耗时 µs/iter | 占比 |
|---------|-----------|:---:|:---:|:---:|
| **FP8 GEMM (fwd up+down)** | `kernel_cutlass_quackgemm_default_epi*` | 4 | 1215.8 | 44.4% |
| **FP8 GEMM (bwd dGated)** | `zeromatGemmDGated*` | 1 | 449.8 | 16.4% |
| **FP8 GEMM (bwd wgrad)** | `zeromatGemmGated*` (TMA reduce-add) | 1 | 399.3 | 14.6% |
| **Colwise FP8 量化** | `_colwise_quantize_and_pack_kernel` | 3 | 213.9 | 7.8% |
| **Dual 量化 (dz)** | `_dual_varlen_iso32_quantize_kernel` | 1 | 152.4 | 5.6% |
| **Token gather/reduce** | `token_gather_sum_kernel` | 2 | 147.9 | 5.4% |
| **Row FP8 量化** | `_quantize_and_pack_kernel` | 3 | 80.9 | 3.0% |
| **Router/scatter** | misc (broadcast, index, scatter) | — | 62.1 | 2.3% |
| **Scale gather** | `_gather_isa_packed_scales_kernel` | 2 | 15.8 | 0.6% |
| **合计** | — | — | **~2639-2660** | 100% |

> 注：最新 fresh benchmark 中，FP8 vs 当前 QuACK BF16 baseline 的 Ernie shape 加速为 **1.11×**（2659.8 µs vs 2942.5 µs）。上表的 1.37× 是相对 Session 53 历史 cuBLAS/PyTorch BF16 baseline；两个 baseline 不能混用。

### 1.3 BF16 vs FP8 算子类别对照

| 功能 | BF16 实现 | FP8 实现 | 耗时变化 |
|------|----------|---------|:---:|
| **前向 Up-proj GEMM** | cuBLAS BF16 matmul (700µs) | CuTe DSL ZeroMat GemmGated + SwiGLU epilogue + blockscaled quant (451µs) | **−249µs** |
| **前向 Down-proj GEMM** | cuBLAS BF16 matmul | CuTe DSL blockscaled_fp8_gemm_varlen | ~同 |
| **反向 dGated GEMM** | cuBLAS BF16 + separate dSwiGLU | CuTe DSL GemmDGatedFP8CLoad + in-register dSwiGLU | **−48µs** |
| **反向 Wgrad GEMM** | cuBLAS BF16 (2078µs total) | CuTe DSL FP8 + TMA reduce-add epilogue (1148µs) | **−930µs** |
| **激活函数** | 独立 SwiGLU kernel | 融合进 GEMM epilogue (零额外开销) | **−100+µs** |
| **FP8 量化** | 不需要 (0µs) | Row/Col/Dual 量化 kernels | **+481µs** |
| **净节省** | — | — | **~900µs (−25%)** |

### 1.4 显存对比 (E=8, Ernie shape)

| 类别 | BF16 | FP8 | 变化 |
|------|:---:|:---:|:---:|
| 模型参数 | 216 MiB (bf16) | 216 MiB (bf16, 原始) + 216 MiB (FP8 cache) | +216 MiB |
| 前向激活 (peak) | ~960 MiB | ~480 MiB (z_fp8 替代 z_bf16) | **−480 MiB** |
| 反向激活 (peak) | ~1100 MiB | ~1314 MiB | +214 MiB |
| 总 delta | — | — | **+4.8%~10.3%** |

> FP8 的显存代价来自 4 套 FP8 weight cache (不同 GEMM 需要不同物理 layout)；但前向用 FP8 存 z 节省了一半激活空间。

---

## 2. FP8 Frontier 创新点与优化点分析

### 2.1 核心创新点

#### (1) Zero-Materialization GEMM (零物化 gather)

**问题**: MoE 路由后 activation 从 T 维展开为 TK 维 (T×topk)。传统做法需要 gather 产生 TK 大小的新 tensor。

**创新**: SonicMoE 的 CuTe DSL GEMM 接受 `A_idx` (gather index) 参数，在 GEMM 计算时直接按索引从原始 T 维 tensor 中 TMA-load 数据，无需物化 TK 大小的 gathered activation。

**数学表达**:
```
传统:  x_gathered[j] = x[gather_idx[j]]  for j∈[0,TK)   // 需要 TK×H bytes
       y = x_gathered @ W
SonicMoE: y[j] = x[gather_idx[j]] @ W  // TMA 在 kernel 内部按 idx 寻址
```

**节省**: 避免 TK×H 的 bf16 intermediate (~384 MiB for Ernie shape)，并省去 gather kernel 本身。

#### (2) Epilogue-Fused Blockscaled FP8 Quantization

**问题**: FP8 训练需要在 GEMM 输出后做 blockscaled 量化 (计算 amax→e8m0 scale→fp8 cast)。传统实现需单独的量化 kernel，额外读写 HBM。

**创新**: 将 blockscaled 量化逻辑直接嵌入 CUTLASS GEMM 的 epilogue 阶段。在 epilogue 寄存器中完成：
1. 计算每 32 元素的 amax
2. 转换为 e8m0 scale (power-of-2 量化)
3. 将 fp32 结果乘以 quant_scale 得到 FP8 值
4. STG 存储 scale 字节

**效果**: `BlockscaledScaleStore` EpiOp 的开销为 **0 standalone cost** — 完全在 GEMM epilogue 的寄存器中执行。

#### (3) Fused Gated Up-Projection (SwiGLU + GEMM + Quant 三合一)

**问题**: SwiGLU 激活需要 `y1 = SiLU(gate) × up`，传统实现需 GEMM→SwiGLU→Quant 三个 kernel。

**创新**: `GemmGatedSm100ZeroMat` 在一个 CUTLASS kernel 内完成：
- GEMM 计算 z = x @ W1 (产出 gate 和 up 交错排列)
- Epilogue 内完成 SwiGLU: `y1 = sigmoid(gate) × up`
- 同步完成 blockscaled FP8 量化 + ISA scale pack

**效果**: 3 个 kernel → 1 个 kernel，省去 z_bf16 (384 MiB) 的 HBM 往返。

#### (4) TMA Reduce-Add Wgrad Epilogue

**问题**: Wgrad 需要跨 expert 累加梯度到同一参数。传统方式用 `D = A@B + 1.0*C` (beta-accumulation)，需 86 regs/thread。

**创新**: 使用 TMA 硬件原子加法 (TMA reduce-add on store)，寄存器压力从 86 regs → 50 regs/thread。

**效果**:
- E=8: −65 µs/iter (−2.3%)
- E=32: −138 µs/iter (−4.0%)

#### (5) iso32 Dual Quantization (S80 创新)

**问题**: 反向 dz 需要同时产出 row-wise 和 column-wise FP8 布局。传统需两次独立量化 + 两个 FP8 buffer。

**创新**: 使用 32×32 isotropic blockscale — 一个 amax 覆盖 32 行×32 列。由于 e4m3 的 3-bit mantissa 精度地板 (~12.5% per-value rounding) 远大于 row-wise 到 block-wise 的额外动态范围损失，row 和 block FP8 cast 的字节完全相同。因此：
- **1 个 FP8 buffer** (byte-identical) 被 row 和 col consumer 共享
- **2 个 scale tensor** (不同 ISA layout)

**数学理论**: 设 tile 内某值为 x，row amax 为 α_r，block amax 为 α_b (α_b ≥ α_r)：
- 1×32 量化: x_q = round_fp8(x / α_r × 127) → 误差 ε₁ ~ O(α_r × 2^{-3})
- 32×32 量化: x_q = round_fp8(x / α_b × 127) → 误差 ε₂ ~ O(α_b × 2^{-3})
- bits_lost = log₂(α_b / α_r)

对于梯度分布 (heavy-tailed, amax ≈ 3-4)，虽然 bits_lost ≈ 0.7-1.0，但 e4m3 本身的量化噪声 δ ~ 12.5% 已经远超 bits_lost 带来的额外误差。实测下游 GEMM RRMSE ratio = 1.000× (完全一致)。

**效果**: −60 µs/iter on Ernie shape (消除 192 MiB FP8 写入 + 一半 amax 计算)。

#### (6) Pre-quantized Scale Transfer (跨函数 FP8 共享)

**创新**: `_PREQUANTIZED_SCALES` dict 在 `_UpProjection` 和 `_DownProjection` 之间传递已量化的 FP8 张量 + scale，避免重复量化。

### 2.2 关键优化点总结

| 优化 | 机制 | 节省 |
|------|------|------|
| Zero-materialization gather | GEMM内部按idx寻址 | −384 MiB alloc + gather kernel |
| Epilogue fused quant | CUTLASS epilogue内量化 | −1 kernel launch + HBM R/W |
| SwiGLU+GEMM fusion | Gated epilogue | −2 kernel + 384 MiB |
| TMA reduce-add | 硬件原子加 | −36 regs + 2-4% speedup |
| iso32 dual-quant | 单buffer双消费 | −192 MiB write + 20% quant加速 |
| Weight cache + version key | data_ptr + _version | 避免optimizer step后重新量化 |
| Route-level padding | metadata-only对齐 | 零GEMM改动, +1.2%开销, 9×精度 |
| Single-stream execution | 全部在paddle计算流 | 消除sync, 降低调度延迟 |
| Dynamic tile layout | mark_layout_dynamic | seqlen变化零重编译 |

### 2.3 进一步优化方向

| 优先级 | 方向 | 预期收益 | 难度 |
|:---:|------|---------|:---:|
| **P0** | dgrad1 C-load/L2 reuse 或 kernel fission（先解决 GemmDGated 42% tensor-pipe + 168 regs/thread） | 100µs级潜在收益 | 高 (CuTe DSL/调度) |
| P1 | dz FP8 epilogue fusion 的替代设计（不可直接塞入现有 GemmDGated epilogue） | 需重新证明 | 高 |
| P1 | Producer-side quant fusion: wgrad产出tile直接量化 | ~5 pp MFU | 中 |
| P2 | 32×32 isotropic weight quant (1份FP8 data + 2份scale) | −50% weight cache | 中 |
| P2 | Shared-memory layout overhaul (K-major→MN-major) | −8 regs/thread | 高 |
| P3 | Megakernel (见§3.4) | 消除kernel launch gap | 极高 |

### 2.4 数学理论深入分析

#### FP8 Blockscaled 量化误差模型

设输入张量 X ∈ ℝ^{M×K}，按 1×32 分组量化：

**量化过程**:
```
对每组 g (32 元素):
  α_g = max(|X[i, g*32:(g+1)*32]|)                    // amax
  s_g = 2^{⌊log₂(α_g)⌋+1} / 240                      // e8m0 power-of-2 scale
  X_fp8[i, j] = clamp(round(X[i,j] / s_g), -240, 240) // e4m3 cast
```

**误差分析**:
- 量化误差: ε = X - dequant(quant(X)) = X - X_fp8 × s_g
- 逐元素相对误差: |ε[i,j]| / |X[i,j]| ≤ 2^{-3} = 12.5% (e4m3 mantissa精度)
- RRMSE = √(E[ε²]) / √(E[X²]) ≈ 2^{-3.5} ≈ 8.8% (实测 6.5~7.6%)

**下游GEMM误差传播**:
```
Y_exact = X @ W
Y_fp8   = dequant(quant(X)) @ dequant(quant(W))
RRMSE(Y) ≈ √(RRMSE(X)² + RRMSE(W)²) ≈ √(2) × 8.8% ≈ 12.4%
实测: cos(Y_exact, Y_fp8) > 0.997, RRMSE < 7.6% (比理论好——因为误差非完全独立)
```

#### MFU 计算公式

```
matmul_FLOPs = 18 × TK × H × I
  分解:
  - fwd up-proj:   2·TK·H·(2I) = 4·TK·H·I
  - fwd down-proj: 2·TK·I·H    = 2·TK·H·I
  - bwd dgrad up:  4·TK·H·I
  - bwd dgrad down:2·TK·H·I
  - bwd wgrad up:  4·TK·H·I
  - bwd wgrad down:2·TK·H·I

MFU = matmul_FLOPs / (busy_time × Peak_TFLOPS)
    = 18 × TK × H × I / (busy_µs × 4.5 × 10⁹)
```

---

## 3. MoE 结构性能 / 显存 Roofline 分析

### 3.1 性能 Roofline 模型

#### 极限分析法

**理想情况** (纯GEMM, 无routing/quant开销):
```
ideal_time = 18 × TK × H × I / Peak_TFLOPS
           = 18 × 65536 × 3072 × 1536 / 4.5e15
           = 1230 µs   (Ernie shape, K=8)
```

**实际耗时**: 2693 µs (S80c frontier)

**效率分析**:
```
理论MFU上限 = 100% (纯GEMM)
实际MFU = 45.93%
Gap = 54.07%, 分解如下:
  - GEMM本身效率损失: ~20% (非方阵, varlen, 小tile)
  - 非GEMM计算开销:   ~34% (quant + routing + scatter + activation)
```

**各组件耗时的理论下界**:

| 组件 | 实际耗时 | 理论下界 | 效率 | 瓶颈类型 |
|------|:---:|:---:|:---:|------|
| FP8 GEMM (6个) | 2065 µs | ~1230 µs | ~60% | Compute-bound |
| FP8 量化 (all) | 447 µs | ~300 µs* | ~67% | Memory-BW bound |
| Token scatter/gather | 148 µs | ~80 µs* | ~54% | Memory-BW bound |
| Routing metadata | ~80 µs | ~20 µs | ~25% | Latency-bound |

*理论下界基于 HBM bandwidth = 8 TB/s (Target GPU HBM3e)

#### GEMM Arithmetic Intensity 分析

```
Up-proj:   AI = 2·TK·H·2I / (TK·H + H·2I + TK·2I) × sizeof(fp8)
         = 2·65536·3072·3072 / (65536·3072 + 3072·3072 + 65536·3072) × 1
         ≈ 2·3072 / (1 + 3072/65536 + 1) ≈ 3025 FLOPs/byte

对于 Target GPU: compute/bandwidth ridge = 4500e12 / 8e12 ≈ 562 FLOPs/byte
→ AI >> ridge → 所有 GEMM 都是 compute-bound ✓
```

### 3.2 显存 Roofline 模型

#### 极限分析法 — 单层 MoE 显存公式

设 E=num_experts, H=hidden, I=intermediate, TK=tokens×topk, B=bytes_per_element:

**模型参数** (静态):
```
W1: E × 2I × H × B_param = E × 2I × H × 2 (bf16)
W2: E × H × I × B_param  = E × H × I × 2
FP8 caches (4 layouts): ≈ E × H × I × 4 × 1 (fp8) ≈ W1+W2 的一半
总参数: 3 × E × H × I × 2 bytes + 2 × E × H × I × 1 bytes
      = 8 × E × H × I bytes

Ernie (E=8, H=3072, I=1536): 8 × 8 × 3072 × 1536 = 288 MiB
```

**激活 (动态, FP8 路径)**:
```
前向 peak (Step 3后):
  z(TK, 2I) bf16 = TK × 2I × 2           = 384 MiB  (freed quickly)
  y1(TK, I) bf16 = TK × I × 2            = 192 MiB
  x_fp8(T, H)    = T × H × 1             = 24 MiB
  scales(各种)   ≈ 20 MiB
  PEAK ≈ 620 MiB (freed → z_fp8 192 + y1_fp8 96 + misc)

反向 peak (dz产出后):
  dz(TK, 2I) bf16   = 384 MiB
  z_fp8(TK, 2I) fp8 = 192 MiB  (from ctx)
  y1s(TK, I) bf16   = 192 MiB
  dx_expanded(TK,H) = 384 MiB
  PEAK ≈ 1314 MiB

总激活 peak/layer = max(620, 1314) = 1314 MiB
```

**显存公式 (L 层 MoE, 每层独立)**:
```
Total_mem = L × (Param_per_layer + Activation_per_active_layer × pipeline_depth)
          + Optimizer_states + Misc

对于 gradient checkpointing:
  只需存 1 层的激活 → Activation = 1314 MiB (不随 L 增长)

对于无 checkpointing:
  存所有层 → Activation = L × 1314 MiB (线性增长)
```

### 3.3 Scaling Laws (实测数据)

#### Token count 对 MFU 的影响

```
T=1024  → MFU=26.7%  (launch overhead 主导)
T=2048  → MFU=34.9%  (+8.2pp)
T=4096  → MFU=41.9%  (+7.0pp)
T=8192  → MFU=44.9%  (+3.0pp)  ← Ernie 生产配置
T=16384 → MFU=45.5%  (+0.6pp)  ← 接近饱和

规律: MFU ∝ 1 - β/(α·T + β), 其中 α=0.4066 µs/token, β=250 µs
```

#### Expert count 对 MFU 的影响 (固定 TK)

```
E=8  → MFU=44.9%
E=16 → MFU=42.9%  (−2.0pp)
E=32 → MFU=38.3%  (−4.6pp from E=8)

每翻倍 E → −2.3 pp MFU (routing/dispatch tax 量化)
原因: 更多 expert = 更多 per-expert kernel launch + 更碎片化的 varlen
```

#### 模型宽度对 MFU 的影响

```
H=3072, I=1536 → MFU=44.9%
H=4096, I=2048 → MFU=50.0%  (+5.1pp)
H=6144, I=2048 → MFU=50.9%  (+6.0pp)  ← 峰值

规律: 更宽的 matmul → 更好的 tensor-core 饱和度 → 固定开销被更多有效 FLOP 摊平
```

### 3.4 Megakernel 可行性分析

#### 什么是 Megakernel

将多个独立 kernel (GEMM + quant + activation + routing) 合并为单个大 kernel，消除 kernel 间的 launch gap 和 HBM 中间缓存。

#### 可行性评估

| 维度 | 分析 | 结论 |
|------|------|:---:|
| **Launch gap** | 当前 kernel 间 gap 约 2-5 µs × 30 kernels ≈ 60-150 µs | 有价值 |
| **HBM 节省** | 中间 bf16 buffer (384 MiB dz) 可变为 smem-resident | 有价值 |
| **寄存器压力** | GEMM 本身已 register-bound (86 regs/thread); 融合 quant 增加 24-32 regs | **极高风险** |
| **SM occupancy** | GEMM 需 128+ regs/thread → occupancy=2-4 warps; 融合后可能=1 warp | **性能退化风险** |
| **代码复杂度** | CuTe DSL 目前是 per-kernel 粒度; megakernel 需自定义 scheduler | 极高 |
| **调试难度** | 单一 kernel 中多阶段无法 nsys 细粒度 profile | 高 |

#### 数学极限分析

**Megakernel 理论极限**:
```
假设所有 intermediate 走 shared memory (48KB/SM):
  - z(TK,2I) fp8 = 192 MiB >> 48KB × 148 SMs = 7 MiB smem total
  → 必须分 tile (每 tile ≤ 48KB → ~24K elements per tile)
  → 需要 producer-consumer scheduling across tiles

Tile-level analysis:
  GEMM output tile = 128×128 elements × 2B = 32 KB (fits in smem)
  Quant input = same 32 KB tile → 可以 smem-resident
  Quant output = 128×128 × 1B = 16 KB → fits
  SwiGLU operands: gate + up from same tile → fits (32 KB)

  结论: 单 tile 的 GEMM→SwiGLU→Quant chain 可以 megakernel 化
  但: cross-tile reduction (wgrad) 和 gather/scatter 无法融入
```

**务实评估**: 完整 megakernel 不可行 (寄存器和 smem 约束)。但 **部分融合** 是可行的：

| 融合范围 | 可行性 | 预期收益 |
|---------|:---:|:---:|
| GEMM + SwiGLU + Quant (已实现) | ✅ 已落地 | Baseline |
| dGated + dz-quant | ⚠️ 现有 GemmDGated 直接融合受 168 regs/thread 限制 | 需新设计证明 |
| Wgrad + reduce-add (已实现 TMA) | ✅ 已落地 | +2~4% |
| 全链路 Megakernel | ❌ 不可行 | — |
| Fission + smem handoff | ⚠️ 实验性 | +8~15% on dgrad1 |

### 3.5 下一步工作方向 (优先级排序)

1. **dgrad1 C-load/L2 reuse 或 kernel fission**: 最新 NCU 显示 `GemmDGatedFP8CLoadSm100ZeroMat` 只有约 42% tensor-pipe、L2 hit 约 61%、且 168 regs/thread 已接近寄存器上限。下一步应优先研究 persistent tile / C-load reuse / fission，而不是直接往 epilogue 追加 dz quant 循环。
2. **dz FP8 epilogue fusion 的替代设计**: 旧版 Phase 1A（把 dz 量化直接融入 GemmDGated epilogue）在当前实现上不可作为默认 P0；任何方案都必须先证明不会寄存器溢出、不会降低 occupancy，并通过 determinism + precision + nsys GPU-projection。
3. **多层 pipeline overlap**: 当前已是 single-stream，但可以探索 layer-level overlap (前一层 wgrad 与下一层 fwd 重叠)。
4. **Communication-computation overlap**: DeepEP dispatch/combine 与 expert computation 的重叠 (ErnieCore 已有框架支持)。
5. **Weight 32×32 isotropic**: 减少 50% FP8 weight cache 显存。

---

## 4. MoE 实现与优化手段完整讲解（面向新人）

> **单独新人教材**: 本节的完整展开版已单独整理为
> [`reports/sonic_moe_fp8_frontier_newcomer_guide.md`](./sonic_moe_fp8_frontier_newcomer_guide.md)。
> 那份文档从 MoE/FP8/CuTe 基础讲起，覆盖环境、符号表、端到端数据流、kernel breakdown、roofline 数学、overhead、数值精度、运维坑和专家问答。本节保留在综合报告中，作为新同学 30 分钟内建立心智模型的摘要。

### 4.0 新人先记住的最新 Frontier 事实

#### 当前路径一句话

SonicMoE FP8 frontier 是一条 **DeepEP topk metadata → route-level padding → zero-materialization FP8 up-proj → fused SwiGLU/FP8 epilogue quant → FP8 down-proj → FP8-C-load dGated backward → iso32 dual quant → TMA reduce-add wgrad → ERNIE main_grad layout flush** 的训练热路径。

#### 最新性能基线

| 项 | 当前值 | 解释 |
|---|---:|---|
| Production shape | `T=8192,H=3072,I=1536,E=8,K=8` | `TK=T*K=65536` |
| 有用 FLOPs | `18*TK*H*I=5.566e12` | MoE expert MLP fwd+bwd 主 matmul 口径 |
| FP8 frontier busy | `2659.8 µs/iter` | fresh S81 nsys GPU-projection |
| Ernie MFU | `46.51%` | 分母为 Target GPU FP8 peak `4500 TFLOPS` |
| measured peak MFU | `51.61%` | fresh sweep wide shape |
| vs 当前 QuACK BF16 | `1.11×` | `2659.8 µs` vs `2942.5 µs` |
| vs 历史 cuBLAS/PyTorch BF16 | `~1.37×` | 与 S53 baseline 口径不同，不可混报 |

#### 新人最容易误解的 6 件事

| 误解 | 正确说法 |
|---|---|
| “FP8 一定比 BF16 快 2×” | 2×只是 peak ratio；小 T 下 FP8 quant/scale/metadata overhead 反而会让 FP8 慢。fresh sweep 中 T=1024/2048 分别只有 `0.95×/0.97×`。 |
| “zero-materialization 只是把 gather 藏起来” | 它避免了 `x_gathered(TK,H)` 的 HBM 写入和再读取；Ernie BF16 gathered activation 约 `384 MiB`。 |
| “data 不 gather，scale 也不需要 gather” | SM100 blockscaled SFA layout 按 GEMM `M=TK` 坐标解释；data 可以 `A_idx` 间接读，scale 必须预 gather 到 TK layout，否则 expert offset 会错。 |
| “下一步就是把 dz quant 塞进 GemmDGated epilogue” | 最新 NCU 显示 GemmDGated 已 `168 regs/thread`，几乎无寄存器余量；直接融合不是当前 P0。 |
| “nsys/ncu duration 可以直接比较” | ncu 默认 base clock + replay；nsys 是真实 timeline/boost clock。端到端信 nsys GPU-projection，单 kernel 资源瓶颈看 ncu SoL/regs/L2。 |
| “`node.step()` 就是 optimizer step” | 不是。`node.step()` 是 wgrad native layout → ERNIE `main_grad` layout flush，必须在 `optimizer.step()` 前。 |

#### 必懂符号表

| 符号 | shape / 类型 | 语义 |
|---|---|---|
| `T` | scalar | 当前 rank 的 token rows / DeepEP dispatched rows |
| `K` | scalar | top-k slots；benchmark 常用 `K=8` |
| `TK` | scalar | 有效 token-expert assignments，近似 `T*K` |
| `TK_padded` | scalar | route-level padding 后按 expert segment 128 对齐的 rows |
| `expert_frequency_offset` | `[E+1] int32` | expert-sorted TK 空间的 prefix sum / `cu_seqlens_m` |
| `x_gather_idx` | `[TK_padded] int32` | GEMM M row → 原始 token row；zero-materialization 的关键 |
| `s_scatter_idx` | `[TK_padded] int32` | token-major score → expert-sorted row |
| `s_reverse_scatter_idx` | `[TK] int32` | expert-sorted row → token-major score，用于 combine/reduce |
| `router_scores` | `[TK_padded] fp32` | padding rows 为 0，保证 pad 不贡献输出/梯度 |
| `z` | `[TK,2I]` | up-proj preactivation，逻辑上存在；frontier 中通常保存为 `z_fp8 + scales` |
| `y1` | `[TK,I]` | `silu(gate)*up` 的 SwiGLU 输出 |
| `dz` | `[TK,2I]` | dSwiGLU 后传给 W1 wgrad 和 dx actgrad 的梯度 |

#### 当前 6 个 GEMM 的专业解读

| launch | 角色 | NCU 结论 | 新人该怎么解释 |
|---:|---|---|---|
| 0 | fwd1 `GemmGatedSm100ZeroMatBlockscaledQuant` | tensor pipe ~64%，regs/thread 168，L2 hit ~89% | 不是 HBM bottleneck；主要是 fused SwiGLU + epilogue quant 的寄存器/指令压力。 |
| 1 | fwd2 default down GEMM | tensor pipe ~70%，regs/thread 54 | 较干净，仍有少量 DRAM/L2 reuse 影响。 |
| 2 | dgrad1 `GemmDGatedFP8CLoadSm100ZeroMat` | tensor pipe ~42%，regs/thread 168，L2 hit ~61% | 当前最值得研究的单 kernel；C-load + dSwiGLU epilogue 重，不能盲目再融合。 |
| 3 | dgrad2 default dx GEMM | tensor pipe ~81%，regs/thread 56 | 已接近最佳，不是优先优化对象。 |
| 4 | wgrad1 default GEMM | tensor pipe ~84%，regs/thread 56 | near peak，优化空间小。 |
| 5 | wgrad2 default GEMM + main-grad add | tensor pipe ~81%，regs/thread 54 | TMA/main-grad add 基本免费，证明 reduce-add 方向正确。 |

#### 新人必须能手算的 roofline

```
F = 18 * TK * H * I
  = 18 * 65536 * 3072 * 1536
  = 5.566e12 FLOPs

ideal_time = F / 4.5e15 = 1237 µs
measured_time = 2659.8 µs
MFU = F / (measured_time * 4.5e15) = 46.51%
```

GEMM arithmetic intensity 远高于 Target GPU ridge point：

```
AI_up ≈ 3000 FLOPs/byte
ridge ≈ 4500e12 / 8e12 = 562 FLOPs/byte
```

所以主 GEMM 不是传统 HBM roofline 下的 bandwidth-bound；真实 gap 来自 shape efficiency、varlen fragmentation、epilogue/register pressure、quant/scatter/metadata 和 fixed launch/control overhead。

#### 当前“满分回答”的优化优先级

1. **先看 dgrad1，而不是 wgrad**：wgrad GEMM 在 NCU 里已有 80%+ tensor pipe；dgrad1 只有约 42%。
2. **先研究 C-load/L2 reuse 或 fission，而不是直接 epilogue fusion**：GemmDGated 168 regs/thread 已接近上限。
3. **保留 iso32 dual quant**：真实 Ernie `dz` 审计中 downstream GEMM RRMSE ratio `1.000×`，且省掉一份 FP8 write 和半数 amax。
4. **所有优化必须同时过 determinism、precision、nsys GPU-projection**：只看 isolated kernel 或 CUDA event 不够。

### 4.1 什么是 MoE

**Mixture of Experts (MoE)** 是一种条件计算架构：不是让所有 token 通过同一个 FFN，而是将 FFN 拆分为多个"专家" (expert)，每个 token 只被路由到少数几个专家处理。

```
传统 Transformer FFN:
  每个 token → 同一个 FFN(H→4H→H) → 输出

MoE FFN:
  每个 token → Router 选择 Top-K 个 Expert → 各 Expert(H→I→H) 分别处理 → 加权合并
```

**优势**: 用更多参数但相同的计算量，获得更强的模型能力。例如 8 个 expert 但每个 token 只选 2 个 → 参数量×4 但计算量仅×1。

### 4.2 MoE 基本组件

#### Router (路由器)

**功能**: 决定每个 token 应该被哪些 expert 处理。

```python
# Router 是一个简单的线性层
gate_logits = token @ W_gate   # [seq_len, num_experts]
probs = softmax(gate_logits)   # 归一化为概率
topk_probs, topk_indices = topk(probs, k=K)  # 选择 Top-K
```

**ERNIE 配置**: Top-2 routing, 即 K=2 (但 SonicMoE 测试中 K=8 以最大化计算密度)。

**负载均衡**: 通过 auxiliary loss 鼓励均匀分配：
```
aux_loss = Σ_e (fraction_of_tokens_to_expert_e × mean_prob_of_expert_e)
```

#### Expert (专家)

**功能**: 每个 expert 就是一个标准的 MLP (带 SwiGLU 激活)。

```python
class Expert(nn.Module):
    def __init__(self, H, I):
        self.w1 = nn.Linear(H, 2*I)  # gate + up projection
        self.w2 = nn.Linear(I, H)    # down projection

    def forward(self, x):
        z = self.w1(x)                    # (tokens, 2I)
        gate, up = z.chunk(2, dim=-1)     # 各 (tokens, I)
        y1 = F.silu(gate) * up            # SwiGLU
        return self.w2(y1)                # (tokens, H)
```

#### Dispatch & Combine (分发与合并)

```
Dispatch (分发):
  tokens: [seq_len, H]
  → 根据 topk_indices 重排 → [num_experts × capacity, H]  (expert-sorted)
  → All-to-All 通信 → 各 GPU 获得本地 expert 的 tokens

Combine (合并):
  expert_outputs: [num_experts × capacity, H]
  → All-to-All 通信回收
  → 反向重排 + 乘以 routing probability → [seq_len, H]
```

### 4.3 并行策略详解

#### (1) Expert Parallelism (EP，专家并行)

**原理**: 将 E 个 expert 均匀分布在 EP_size 个 GPU 上，每个 GPU 持有 E/EP_size 个 expert。

```
假设 E=32, EP_size=8:
  GPU 0: Expert 0-3
  GPU 1: Expert 4-7
  ...
  GPU 7: Expert 28-31
```

**通信**: All-to-All — 每个 GPU 将自己的 token 发送给持有目标 expert 的 GPU。

```
Before All-to-All: 每个 GPU 有 seq_len/DP 个 token (所有 expert 的)
After All-to-All:  每个 GPU 有来自所有 GPU 的、属于本地 expert 的 token
```

**优缺点**:
- 优: 参数不重复，显存效率最高
- 缺: All-to-All 通信量 = O(seq × H × K / EP) 双向

#### (2) Tensor Parallelism (TP，张量并行)

**原理**: 每个 expert 的权重矩阵按列/行切分到多个 GPU。

```
Expert FFN with TP=4:
  W1: [H, 2I] 按列切为 4 份 → 每 GPU 持有 [H, 2I/4]
  W2: [I, H]  按行切为 4 份 → 每 GPU 持有 [I/4, H]

Forward:
  w1_out = x @ W1_local              # 无通信
  y1 = SwiGLU(w1_out)                # 本地
  y2 = y1 @ W2_local                 # 本地
  output = AllReduce(y2)              # TP 同步
```

**ERNIE 中**: 当 EP 启用时，expert 内部的 TP 通常关闭 (`expert_tensor_parallel_size=1`)，避免通信嵌套。

#### (3) Pipeline Parallelism (PP，流水线并行)

**原理**: 将模型按层分割到不同 GPU。MoE layer 被当作普通层处理。

```
Stage 0: Layers 0-7   (含 MoE layers at interval=2: L2, L4, L6)
Stage 1: Layers 8-15  (含 MoE layers: L8, L10, L12, L14)
...
```

**SonicMoE 兼容性**: `SonicMoEMlpNode` 每个实例独立维护 `_pending_flush` 状态，天然支持 1F1B schedule 的乱序 F/B。

#### (4) DeepEP (高效 Expert 并行通信)

**原理**: 将传统的 "permute → All-to-All → unpermute" 三步融合为单个 fused kernel。

```
传统 EP:
  1. Permute tokens by expert   (GPU kernel)
  2. All-to-All communication   (NCCL)
  3. Unpermute on receive side  (GPU kernel)

DeepEP Fused:
  1. fused_dispatch(tokens, indices, probs)  → 一步完成 permute + A2A
  2. Expert computation (local)
  3. fused_combine(expert_out)               → 一步完成 A2A + unpermute + scale
```

**PaddleFleet 中的实现**:
```python
# token_dispatcher.py
class MoEFlexTokenDispatcher:
    def dispatch(self, hidden_states, topk_indices, topk_probs):
        # DeepEP fused dispatch: permute + all-to-all in one step
        dispatched, handle = fused_dispatch(hidden_states, ...)
        return dispatched, handle

    def combine(self, expert_output, handle):
        # DeepEP fused combine: all-to-all + unpermute + scale in one step
        return fused_combine(expert_output, handle, ...)
```

### 4.4 SonicMoE 在 PaddleFleet 中的集成

#### 入口与调用链

```
PaddleFleet MoE Layer
  └── moe_layer.py::MoELayer.forward()
        ├── Router: TopKRouter.routing() → topk_indices, topk_probs
        ├── Dispatcher: MoEFlexTokenDispatcher.dispatch() → dispatched_tokens
        ├── Expert Computation:
        │     ├── [标准路径] GroupedMLPExpert.forward() → batched matmul
        │     └── [SonicMoE路径] run_sonic_moe() →
        │           ├── deepep_topk_to_sonic_metadata() → 转换路由元数据
        │           ├── _UpProjection.apply(x, w1, ...) → FP8 fused GEMM
        │           └── _DownProjection.apply(y1, w2, ...) → FP8 fused GEMM
        └── Combiner: MoEFlexTokenDispatcher.combine() → output
```

#### SonicMoE 的核心接口

```python
# 高层接口 (SonicMoEMlpNode)
from sonicmoe.ernie_compat import SonicMoEMlpNode

node = SonicMoEMlpNode(
    experts=grouped_experts,
    n_experts=8,
    hidden_size=3072,
    intermediate_size=1536,
)

# 前向
output = node(x, tokens_per_expert, dispatched_indices, dispatched_probs)

# 梯度刷新 (必须在 optimizer.step() 之前!)
node.step()
```

#### 低层接口 (PaddleFleet 当前使用)

```python
from paddlefleet.ops.sonicmoe.functional import _UpProjection, _DownProjection

# Up-projection: x(T,H) → y1(TK,I), 内含 SwiGLU + FP8 quant
y1, z_fp8 = _UpProjection.apply(
    x_fp8, w1_fp8,
    x_gather_idx, cu_seqlens_m,
    scales, config
)

# Down-projection: y1(TK,I) → y2(TK,H)
y2 = _DownProjection.apply(
    y1_fp8, w2_fp8,
    cu_seqlens_m, config
)
```

### 4.5 ERNIE MoE 配置详解 (ernie-core)

```python
# ERNIE MoE Configuration (典型值)
class ErnieMoEConfig:
    n_routed_experts = 32        # 总 expert 数
    n_shared_experts = 2         # 共享 expert (所有 token 都经过)
    num_experts_per_tok = 2      # Top-K (每 token 选 2 个 expert)
    moe_intermediate_size = 1536 # Expert MLP 中间维度 (SwiGLU → 2×1536=3072)
    hidden_size = 3072           # 隐藏层维度
    moe_layer_interval = 2       # 每隔 2 层插入一个 MoE 层
    moe_group = "world"          # EP 通信组 (world = 所有 GPU)

    # 通信优化
    use_deepep = True            # 使用 DeepEP fused A2A
    use_fp8_dispatch_a2a = True  # A2A 通信使用 FP8 (节省带宽)

    # 计算优化
    fp8 = "e4m3"                 # FP8 计算精度
    moe_grouped_gemm = True      # 使用 grouped GEMM
    using_sonic_moe = True       # 使用 SonicMoE 加速

    # 负载均衡
    router_aux_loss_coef = 1e-2  # 辅助损失系数
    router_z_loss_coef = 1e-4    # z-loss 系数
```

### 4.6 关键数据结构与索引空间

MoE 中最容易混淆的是两个索引空间：

#### T×K (flat-topk) 空间
```
大小: T×K
布局: token-major, [t*K+k] = "第 t 个 token 的第 k 个 expert 选择"
用于: topk_scores, topk_indices, s_reverse_scatter_idx
```

#### TK (expert-sorted) 空间
```
大小: T×K (= TK)
布局: expert-major, expert 0 的 token 排在前面, 然后 expert 1, ...
用于: 所有 GEMM 的 activation 张量, x_gather_idx, s_scatter_idx
```

#### 桥接索引
```
s_scatter_idx:         T×K空间 → TK空间 的映射
s_reverse_scatter_idx: TK空间 → T×K空间 的逆映射
x_gather_idx:          TK空间 → 原始 token ID 的映射
```

关系: `s_reverse[s_scatter[j]] == j` (互逆)

### 4.7 端到端数据流图

```
                     ┌─────────────────────────────────────────────┐
                     │           SonicMoE FP8 Forward              │
                     └─────────────────────────────────────────────┘

 x(T,H) bf16 ─┬─→ [quantize_and_pack] ──→ x_fp8(T,H) + x_scales
               │
               │   [scale_gather T→TK]  ──→ x_scales_tk
               │
               ├─→ [GemmGatedSm100ZeroMat]  (A_idx=gather_idx, 零物化!)
               │     内部: x_fp8 × W1_fp8 → z(TK,2I)
               │     epilogue: SwiGLU → y1(TK,I)
               │     epilogue: blockscaled quant → z_fp8 + scales
               │
               └─→ [blockscaled_fp8_gemm_varlen]
                     y1_fp8 × W2_fp8 → y2(TK,H) bf16

                  → [router_scatter] → output(T,H) bf16


                     ┌─────────────────────────────────────────────┐
                     │           SonicMoE FP8 Backward             │
                     └─────────────────────────────────────────────┘

 dout(T,H) bf16 ──→ [quantize_and_pack] ──→ dout_fp8

 [GemmDGatedFP8CLoad] (dout_fp8 × W2_fp8, C-load z_fp8)
   epilogue: dSwiGLU(dout_scaled, z_dequant) → dz(TK,2I) bf16 + y1s(TK,I)

 [wgrad_dw2] = dout_col_fp8.T × y1s_col_fp8  (TMA reduce-add)

 [iso32_dual_quant] dz → dz_fp8_row + dz_fp8_col  (单buffer!)

 [wgrad_dw1] = dz_col_fp8.T × x_col_fp8  (TMA reduce-add)

 [actgrad_dx] = dz_fp8_row × W1T_fp8 → dx_expanded(TK,H)

 [token_reduce] → dx(T,H) bf16
```

### 4.8 优化手段速查表

| 优化手段 | 层级 | 描述 | 在哪里实现 |
|---------|------|------|-----------|
| Expert Parallel | 系统级 | 专家分布到多 GPU | PaddleFleet token_dispatcher |
| DeepEP Fused A2A | 通信级 | 融合 permute+A2A | ernie-core fused_a2a.py |
| FP8 A2A | 通信级 | 通信数据用 FP8 (省50%带宽) | ernie-core fp8_utils.py |
| Grouped GEMM | 计算级 | E 个 expert 合并为一次 batched GEMM | PaddleFleet moe_expert.py |
| Zero-materialization | 计算级 | 避免 gather 大 tensor | sonicmoe gemm_gated.py |
| GEMM+SwiGLU fusion | 计算级 | 省 kernel launch + HBM | sonicmoe gemm_gated.py |
| Epilogue FP8 quant | 计算级 | 量化在寄存器中完成 | sonicmoe BlockscaledScaleStore |
| TMA reduce-add | 计算级 | 硬件原子累加梯度 | sonicmoe blockscaled_fp8_gemm |
| iso32 dual-quant | 算子级 | 单 buffer 双 layout | sonicmoe fused_quant_kernels |
| Weight cache | 内存级 | 避免重复 FP8 转换 | sonicmoe functional/__init__.py |
| Route-level padding | 算法级 | metadata对齐替代token padding | sonicmoe deepep_metadata.py |
| Lazy main_grad | 内存级 | 推迟到首次 backward 分配 | sonicmoe mlp_node_v2.py |
| Single-stream | 调度级 | 避免跨 stream sync | sonicmoe _triton_stream_compat |
| Dynamic tile layout | 编译级 | seqlen变化无需重编译 | sonicmoe mark_layout_dynamic |

### 4.9 精度保证

SonicMoE FP8 path 与 BF16 baseline 的精度对比:

| 梯度 | 验证方式 | cos similarity | RRMSE |
|------|---------|:---:|:---:|
| output | 直接对比 | 0.9979 | <6.5% |
| dx (hidden_states grad) | Paddle autograd | 0.9975 | <7.6% |
| ds (router scores grad) | Custom Triton scatter | 0.9971-0.9973 | — |
| dw1 (up-proj weight grad) | CUTLASS native buffer | 0.9975 | <7.6% |
| dw2 (down-proj weight grad) | CUTLASS native buffer | 0.9971 | <7.6% |

**确定性**: frontier 路径在相同配置下产出 bit-exact 相同结果 (已 CI 验证)。

---

## 5. CI 测试基础设施与质量门禁

> SonicMoE 的质量保障由 `tools/ci/run_core_tests.sh` 统一编排，覆盖精度、确定性、性能、JIT 机制、显存安全和代码覆盖率等维度。所有预算阈值集中存储在 `tools/ci/baselines.json` 中，任何放宽必须附 HANDOFF 文档说明。

### 5.1 CI Pipeline 概览

```
tools/ci/run_core_tests.sh
├── [✓] precision          tests/ops/test_mlpnode_precision.py (FP8 vs BF16 gold, 6 shapes)
├── [✓] import-smoke       tests/ops/test_import_smoke.py (40+ 模块 import 无回归)
├── [✓] multilayer         tests/ops/test_mlpnode_multilayer.py (PP + multistep 梯度正确性)
├── [F] quant              6 个量化内核测试 (byte-exact 断言)
├── [F] jit-cold           tools/ci/jit_bench.py --phase cold (首次编译预算 90s)
├── [F] jit-warm           tools/ci/jit_bench.py --phase warm (sentinel 快速跳过 <3s)
├── [F] jit-reload         tools/ci/jit_bench.py --phase reload (磁盘缓存命中 <80s)
├── [F] jit-reuse          tools/ci/jit_bench.py --phase reuse (进程内 dispatch <18ms)
├── [F] jit-parallel       tools/ci/jit_bench.py --phase parallel-cold (多进程编译 <90s)
├── [✓] jit-key-stability  tests/ops/test_jit_key_stability.py (动态维度无重编译)
├── [✓] extreme-shapes     tests/ops/test_mlpnode_extreme_shapes.py (T=0~32k, imbalance)
├── [✓] jit-concurrent     tests/ops/test_jit_concurrent_heterogeneous.py (GPFS 缓存并发)
├── [✓] frontier-determinism  tests/fp8_frontier_determinism_test.py (HARD-fail, bit-exact)
├── [✓] frontier-stress    tests/fp8_frontier_stress_test.py (17 shapes × 3 routing)
├── [F] perf               tools/ci/perf_gate.py (nsys GPU-projection ≤ 2800 µs)
├── [F] multicard          tools/ci/multicard_smoke.py (多卡 smoke, ≥2 GPU 自动检测)
└── [✓] coverage           coverage report (sonicmoe/, ≥30% gate)

✓ = --fast 也运行    F = 仅 full sweep 运行
```

**运行模式**:
- `tools/ci/run_core_tests.sh` — full sweep (所有 phase)
- `tools/ci/run_core_tests.sh --fast` — precision + multilayer + jit-warm + frontier + coverage
- 支持 `--no-perf`, `--no-multicard`, `--no-jit`, `--soft-jit` 等精细控制

**Exit codes**: 0 = 全部通过; 1 = 任一 phase 失败; 2 = 参数错误

**并行执行**: 自动检测 GPU 数量，通过 `pytest-xdist` 按 `--dist=loadfile` 分发测试到多 GPU (每个 worker 绑定独立 `CUDA_VISIBLE_DEVICES`)。

---

### 5.2 性能预算体系 (baselines.json)

所有预算集中定义在 `tools/ci/baselines.json` 中，每项包含 `budget` (hard-fail), `warn` (告警但不失败), `_actual_s78b` (基准实测值)。

| 组件 | 指标含义 | Budget | Warn | Actual (S78b) | 单位 |
|------|---------|:------:|:----:|:-------------:|:----:|
| **JIT cold-warmup** | `rm -rf cache` → 全量首次编译 + ptxas + autotune | 90 | 70 | 46 | s |
| **JIT warm-skip** | sentinel 存在 → warmup_jit 直接返回 False | 3 | 2 | 0.05 | s |
| **JIT reload** | sentinel 清除但磁盘缓存在 → Triton/Quack 命中，CuTe 重编译 | 80 | 60 | 44 | s |
| **JIT reuse** | 进程内重复 fwd+bwd × 5 次平均 (paddle dispatch + Python + kernel) | 18,000 | 17,000 | 16,303 | µs |
| **JIT parallel-cold** | 2 worker × 4 shapes 并发冷编译 → 共享 GPFS 缓存 | 90 | 75 | 56 | s |
| **Perf (GPU-proj)** | nsys GPU-projection, FP8 fwd+bwd, Ernie shape | 2,800 | 2,750 | 2,740 | µs |
| **Coverage** | sonicmoe/ 包代码覆盖率 | 30% | — | 31% | pct |

**Headroom 策略**: Budget 设为 actual 的 ~1.5× 以上，避免环境抖动导致假阴性。每次 tighten 必须附对应 HANDOFF 版本号。

**覆盖率分析 (S78b)**:
- 主路径高覆盖: `cute_blockscaled_quant` 83%, `fused_quant_kernels` 90%, `gemm_gated` 75%, `jit` 73%
- 备选路径欠覆盖: `blockscaled_fp8_gemm` 22% (mxfp8), `grouped_gemm` 6%, `swiglu_triton` 11%
- 达到 50% 目标需为非默认路径补充 per-kernel 集成测试

---

### 5.3 FP8 Frontier 确定性测试 (HARD-fail)

> **文件**: `tests/fp8_frontier_determinism_test.py` | **失败 = 阻断 CI** (不可降级为 warning)

#### 核心逻辑

```python
# 1. 构建 Ernie production MoE (E=8, K=8, H=3072, I=1536)
# 2. Warmup 4 次 → 触发 _ALIGNMENT_ASSUMED = True (frontier 路径锁定)
# 3. 固定种子生成 x, dy
# 4. 执行 3 次 fwd+bwd → 每次产出 (y, dx, param_grads)
# 5. 断言: iter0 == iter1 == iter2 (bit-exact, 非近似)
```

#### 两个 Test Case

| Test | Shape | 意义 |
|------|-------|------|
| `test_ernie_production_deterministic` | T=8192, H=3072, I=1536, E=8, K=8 | 生产形状 — 最大化 tile 数量,暴露原子序竞争 |
| `test_small_aligned_deterministic` | T=1024, H=3072, I=1536, E=8, K=8 | 小 batch — 暴露 warp-level 边界条件 |

#### 关键机制

- **`_FrontierProbe`**: forward hook 验证 `_ALIGNMENT_ASSUMED=True` + `_FUSED_GATED=True`。若 frontier 路径未被实际执行，测试通过是空谈 → probe 断言失败。
- **`_tensors_bit_equal(a, b)`**: 使用 `(a == b).all().item()` 而非 `torch.equal()`，因为 Paddle proxy 下 `torch.equal` 返回 element-wise tensor 而非标量 bool。
- **`_reset_fp8_state()`**: 清除 `_PREQUANTIZED_SCALES`, `_FUSED_WEIGHT_CACHE`, `_VARLEN_WEIGHT_CACHE` 等全局状态，确保每次测试从干净状态开始。

#### 为什么 bit-exact 是必须的

非确定性通常来自：
1. **原子序竞争 (atomic-order race)**: TMA reduce-add 的硬件原子加法在不同调度顺序下可能产生不同 rounding
2. **async-TMA 调度泄漏**: Tensor Memory Accelerator 的异步加载如果跨迭代复用旧数据
3. **状态污染**: 全局 FP8 缓存未正确清除，上一次迭代的量化 scale 泄漏到下一次

上述任一情况 = 训练精度不可复现 = 生产 bug。

---

### 5.4 FP8 Frontier 压力测试 (17 tests)

> **文件**: `tests/fp8_frontier_stress_test.py` | **soft-fail** (允许 CI 继续)

#### Shape 配置矩阵 (test_shape_no_crash)

| T | E | K | H | I | 覆盖场景 |
|:---:|:---:|:---:|:---:|:---:|---------|
| 8192 | 8 | 8 | 3072 | 1536 | Ernie production |
| 128 | 8 | 8 | 3072 | 1536 | launch-overhead dominated |
| 256 | 4 | 4 | 3072 | 1536 | 小 T + 少 expert |
| 16384 | 8 | 8 | 3072 | 1536 | 大 batch (2× production) |
| 4096 | 8 | 8 | 4096 | 2048 | 宽模型 |
| 4096 | 16 | 8 | 3072 | 1536 | 多 expert (E=16) |
| 2048 | 32 | 8 | 3072 | 1536 | 极多 expert (E=32) |
| 1000 | 8 | 8 | 3072 | 1536 | 非 128 对齐 T |
| 4097 | 8 | 8 | 3072 | 1536 | 质数+1 T (最差对齐) |
| 4096 | 8 | 2 | 3072 | 1536 | 部分路由 (K=2) |
| 8192 | 8 | 4 | 3072 | 1536 | 半数路由 (K=4) |

#### Routing Imbalance 模式 (test_routing_robustness)

| 模式 | 行为 | 压力点 |
|------|------|--------|
| `none` | 均匀随机 top-K | 基线对照 |
| `skew` | 80% token → expert 0 | 极端负载不均 |
| `extreme` | 所有 token → 相同 K 个 expert | 其余 expert 完全空闲 |

#### 4 类断言

1. **No crash**: `torch.cuda.synchronize()` 后无 CUDA async error
2. **All finite**: `torch.isfinite(output).all()` — 无 inf/nan
3. **Determinism** (`test_determinism_ernie_shape`): 同状态两次 forward → byte-identical
4. **Gradient flow** (`test_gradient_flow`): `node.step()` 后所有 expert 参数梯度非零

#### 特殊测试

- **`test_multi_iter_stability`**: 连续 10 次 fwd+bwd 后输出仍 finite (捕获累积数值不稳定)

---

### 5.5 多层 / PP 正确性测试

> **文件**: `tests/ops/test_mlpnode_multilayer.py` | **精度门限**: cos ≥ 0.9999, RRMSE ≤ 1e-3

#### 验证目标

SonicMoE 的原生梯度 (`main_grad`) 系统必须正确处理：
- 多层独立 backward 不互相覆盖
- Pipeline parallel 乱序 F/B schedule 不丢梯度
- 多 micro-batch 跨 optimizer step 梯度累积

#### 4 种 PP Schedule 测试 (`test_pipeline_parallel_interleaved`)

```
Schedule A (1F1B):     F0 F1 F2 B2 B1 B0   ← 标准
Schedule B (FFB-FBB):  F0 F1 F2 B0 B1 B2   ← forward-first
Schedule C (交错):     F0 F1 B0 F2 B1 B2   ← 完全交错
Schedule D (乱序):     F2 F0 F1 B1 B2 B0   ← forward 顺序也打乱
```

每种 schedule 必须产出与独立单层 baseline 相同的 per-layer 梯度。

#### 多步累积 (`test_multistep_pp_accumulation`)

```
3 optimizer steps × 4 micro-batches × 3 layers
验证: main_grad[layer] == Σ(micro-batch 独立 single-layer reference grads)
```

捕获的历史 bug: S74 之前的全局 `_pending_flush` flag 会导致多层共享一个 flush 信号，第二层的梯度被第一层的 flush 覆盖。修复后改为 per-instance flag。

---

### 5.6 JIT 机制基线测试

> **文件**: `tools/ci/jit_bench.py` + `tests/ops/test_jit_key_stability.py` + `tests/ops/test_jit_concurrent_heterogeneous.py`

#### 5 个正交 Phase

```
┌─── cold ───────────────────────────────────────────────────────────┐
│ rm -rf $CACHE_DIR; warmup_jit(force=True)                          │
│ 测量: 全量编译 (Triton kernel gen + ptxas + CuTe cubin + autotune) │
│ Budget: 90s                                                        │
└────────────────────────────────────────────────────────────────────┘

┌─── warm ───────────────────────────────────────────────────────────┐
│ sentinel file 存在 → warmup_jit() 返回 False (不编译)              │
│ 测量: sentinel 检查延迟                                            │
│ Budget: 3s (actual: 50ms)                                          │
└────────────────────────────────────────────────────────────────────┘

┌─── reload ─────────────────────────────────────────────────────────┐
│ 删除 sentinel, 保留磁盘缓存 → Triton/Quack 命中,CuTe 重编译       │
│ 测量: 从磁盘缓存恢复的时间                                         │
│ Budget: 80s                                                        │
└────────────────────────────────────────────────────────────────────┘

┌─── reuse ──────────────────────────────────────────────────────────┐
│ 进程内: warmup_jit() 后执行 5 次 fwd+bwd                          │
│ 测量: 端到端 wall time (paddle dispatch + Python + GPU kernel)      │
│ Budget: 18,000 µs/iter (平均)                                      │
└────────────────────────────────────────────────────────────────────┘

┌─── parallel-cold ──────────────────────────────────────────────────┐
│ 2 worker × 4 shapes 并发 warmup_jit_parallel()                     │
│ 测量: 多进程在共享 GPFS 缓存上的安全性和总耗时                      │
│ Budget: 90s                                                        │
└────────────────────────────────────────────────────────────────────┘
```

#### 缓存层级

```
Layer 0: warmup_sentinel.json (< 1 KB)
  存在 → phase=warm, 0.05s 直接跳过

Layer 1: Triton + Quack disk cache (数百 MB)
  存在 → 跳过 ptxas/autotune, CuTe 仍需 in-proc 重编译
  命中 → phase=reload, ~44s

Layer 2: CuTe in-memory compile cache (per-process)
  命中 → phase=reuse, ~16.3ms/iter (纯 dispatch 路径)
```

#### Key Stability 测试 (`test_jit_key_stability.py`)

**问题**: 如果 kernel 的编译 key 包含动态维度 (如 `total_K`), 每次推理时 token 数变化就会触发重编译 → 流水线停顿。

**测试方法**:
1. 用固定 shape 初始 prime (触发编译)
2. 记录所有 `_COMPILE_CACHE*` 字典大小
3. Sweep 6 种不同的 total_K: [E×128, E×512, E×1024, E×2048, E×96, E×257]
4. 断言: 所有编译缓存大小未变 (= 无动态维度 key)

#### 并发异构测试 (`test_jit_concurrent_heterogeneous.py`)

**场景**: 8-rank 训练中多个 rank 同时冷启动，写入同一 GPFS 缓存目录。

**测试方法**:
1. 子进程 A: shape (total_K=4096) 冷编译
2. 子进程 B: shape (total_K=8192) 冷编译 (并发)
3. 子进程 C: 两种 shape 热启动 → 必须 <60s (命中 A 和 B 的缓存)

**验证**: 无 torn writes, 无 missing artifacts, 无 duplicated rebuilds。

---

### 5.7 量化内核测试矩阵

> **文件目录**: `tests/ops/test_*quant*.py` | **共享 conftest**: `tests/ops/conftest.py`

#### 测试文件汇总

| 测试文件 | 被测 Kernel | 断言类型 | 关键容差 |
|---------|------------|---------|---------|
| `test_dual_quant.py` | `dual_quantize_varlen` (row+col 融合) | byte-exact (row+col 各自 vs 独立实现) | 0 |
| `test_colwise_quant.py` | `colwise_quantize_and_pack` (Triton) + `colwise_quantize_cute` (CuTe) | byte-exact (fp8 + scales vs gold) | 0 |
| `test_rowwise_quant.py` | `quantize_and_pack_activation` (row-wise ISA-packed) | byte-exact + roundtrip RRMSE | fp8=0, roundtrip<5% |
| `test_weight_quant.py` | `quantize_and_pack_weight_iso32` (32×32 isotropic) | byte-exact (fp8 + scales vs gold) | 0 |
| `test_dequant.py` | `dequantize_blockscaled_fp8` | RRMSE vs gold | dequant<0.1%, roundtrip<5% |
| `test_fused_quant.py` | `fused_dual_colwise_quantize` (API 级融合) | byte-exact (6 output buffers) | 0 |
| `test_fused_zy1_quant.py` | `fused_z_save_y1_quant` (z+y1 单 kernel) | byte-exact (z/y1 各自 vs 独立实现) | 0 |

#### 共享测试基础设施 (conftest.py)

**Shape 参数化**:
```python
QUANT_SHAPES = [
    (128, 128,    id="smoke"),         # 最小, 快速烟雾
    (1024, 1536,  id="aligned"),       # 对齐, 标准
    (384, 1536,   id="unaligned-TK"),  # TK 非 128 对齐
    (65536, 1536, id="large-TK"),      # 大 TK (OOM 边界)
    (8192, 3072,  id="production"),    # Ernie 生产形状
]
```

**Gold Reference 实现** (纯 torch, 无 Triton):
- `gold_e8m0_row_quant(x)`: 按行 1×32 分组量化
- `gold_e8m0_col_quant(x)`: 按列 32×1 分组量化
- `gold_e8m0_iso32_quant(x)`: 32×32 isotropic 量化
- `gold_dequant(fp8, scales)`: FP8 → BF16 反量化

**精度辅助函数**:
- `assert_byte_exact(a, b)`: raw bytes 逐位比较
- `unpack_isa_scales(packed)`: 将 ISA layout 解包为原始 uint8 以供验证
- `rrmse(a, b)`: 相对均方根误差 = ‖a-b‖ / ‖b‖

#### 测试设计哲学

为什么是 **byte-exact** 而非近似容差？

量化内核是 **确定性映射**: 给定相同输入, blockscaled FP8 量化的每一个字节都是确定的 (amax → e8m0 floor → clamp → cast)。如果 kernel 实现与 gold reference 有任何字节差异，说明：
- 浮点 amax 减少顺序不一致 (parallelism bug)
- Scale 计算使用了错误的 rounding mode
- ISA packing 布局错误 (导致下游 GEMM 读取错误 scale)

这些都是 correctness bug, 不是精度损失。

---

### 5.8 边界 / 鲁棒性测试

#### 极端形状 (`test_mlpnode_extreme_shapes.py`)

| Test Case | 参数 | 验证目标 |
|-----------|------|---------|
| `test_zero_total_tokens` | T=0 | 空 expert bucket 不 crash |
| `test_tiny_token_count` | T=8, 16 | launch overhead 路径, padding 正确 |
| `test_per_expert_empty_bucket` | 全部 token → expert 0 | 空 expert 不产生 spurious 梯度 |
| `test_extreme_imbalance_99` | 99% → expert 3 | CuTe varlen 极端不均 |
| `test_strong_imbalance_85` | 85% → expert 3 | 中等不均 (production worst-case) |
| `test_large_T` | T=16384, 32768 | 大 batch OOM / shape overflow |

#### Import 烟雾测试 (`test_import_smoke.py`)

验证 40+ 模块可以无错 import。捕获:
- 循环 import (module A → B → A)
- 缺失 stub (Paddle proxy 未 patch 的 torch API)
- Dataclass 初始化错误 (字段默认值冲突)
- `TRITON_PTXAS_PATH` 未配置时的 graceful fallback

#### 精度审计 (`test_mlpnode_precision.py`)

6 种 shape 下的 FP8 MlpNode vs BF16 gold 逐元素对比:
- output cos similarity ≥ 0.9975
- dx (activation gradient) cos ≥ 0.9970
- dw1, dw2 (weight gradients) cos ≥ 0.9970

使用独立 gold 计算 (不通过 SonicMoE 代码路径), 验证端到端数值精度。

---

### 5.9 如何运行与扩展

#### 日常开发 (快速验证)

```bash
source .runenv.sh
tools/ci/run_core_tests.sh --fast       # ~3min: precision + multilayer + frontier + coverage
```

#### 提交前 (完整 CI)

```bash
tools/ci/run_core_tests.sh              # ~8min: 全部 phase 含 JIT + perf + multicard
```

#### 单独测试某一类

```bash
# Frontier 确定性 (HARD-fail)
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/fp8_frontier_determinism_test.py -v -s

# Frontier 压力 (17 shapes)
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/fp8_frontier_stress_test.py -v

# 量化内核 (byte-exact)
python -m pytest tests/ops/test_dual_quant.py tests/ops/test_rowwise_quant.py -x -v

# JIT key stability
python -m pytest tests/ops/test_jit_key_stability.py -v

# 多层/PP
python -m pytest tests/ops/test_mlpnode_multilayer.py -v
```

#### 如何新增测试

1. **新增量化 kernel 测试**: 在 `tests/ops/` 新建 `test_<kernel_name>.py`, 使用 `conftest.py` 提供的 `QUANT_SHAPES` + `assert_byte_exact` / `rrmse`
2. **新增 shape 到压力测试**: 编辑 `tests/fp8_frontier_stress_test.py` 的 `STRESS_SHAPES` 列表
3. **调整 JIT 预算**: 编辑 `tools/ci/baselines.json` 对应字段, 附 `_actual_sXX` 实测数据 + HANDOFF 说明

#### 如何更新 baselines.json

```bash
# 1. 获取最新 actual (需完整运行一次 JIT bench)
python tools/ci/jit_bench.py --phase cold --baselines tools/ci/baselines.json --json /tmp/jit.json
cat /tmp/jit.json  # 查看新 actual

# 2. 修改 baselines.json (budget = actual × 1.5~2.0)
vim tools/ci/baselines.json

# 3. 更新 _actual_sXX 字段和 HANDOFF 文档
# 规则: 任何 budget 放宽都必须在 HANDOFF 中记录原因和对应的性能分析
```

---

## 附录 A: 关键文件索引

| 文件 | 用途 |
|------|------|
| `sonicmoe/ernie_compat/mlp_node_v2.py` | 生产入口 `SonicMoEMlpNode` |
| `sonicmoe/functional/__init__.py` | `_UpProjection`, `_DownProjection` 核心逻辑 |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 GEMM wrapper, weight cache, quant kernels |
| `sonicmoe/quack_utils/gemm_gated.py` | CuTe DSL GemmGated + BlockscaledScaleStore |
| `sonicmoe/quack_utils/gemm_dgated.py` | CuTe DSL GemmDGated + FP8PreActLoad |
| `sonicmoe/quack_utils/swiglu_triton.py` | 5 个 SwiGLU Triton kernel 变体 |
| `sonicmoe/quack_utils/fused_quant_kernels.py` | iso32 dual-quant Triton kernel |
| `sonicmoe/ernie_compat/deepep_metadata.py` | DeepEP topk→SonicMoE metadata 转换 |
| `sonicmoe/_triton_stream_compat.py` | Triton stream monkey-patch |
| `tools/ci/run_core_tests.sh` | CI 主编排脚本 (17 phase, fast/full 模式) |
| `tools/ci/baselines.json` | 所有性能预算集中定义 (JIT/perf/coverage) |
| `tools/ci/jit_bench.py` | JIT 5-phase 基线测量工具 (子进程隔离) |
| `tools/ci/perf_gate.py` | nsys GPU-projection 性能门禁 |
| `tests/fp8_frontier_determinism_test.py` | HARD-fail 确定性测试 (bit-exact) |
| `tests/fp8_frontier_stress_test.py` | 17-shape 压力测试 |
| `tests/ops/test_mlpnode_multilayer.py` | 多层/PP 梯度正确性 (4 schedule) |
| `tests/ops/test_jit_key_stability.py` | 动态维度无重编译断言 |
| `tests/ops/conftest.py` | 共享 fixtures: gold quant, assert_byte_exact, QUANT_SHAPES |

## 附录 B: 环境启动

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source .runenv.sh   # 激活环境 (PYTHONPATH, CUDA, USE_QUACK_GEMM=1, etc.)
bash tests/run_regression.sh  # 验证 frontier 状态
```

## 附录 C: 术语表

| 术语 | 含义 |
|------|------|
| TK | T×K, routing 展开后的 token 数 |
| EP | Expert Parallelism |
| DeepEP | 高效 expert-parallel 通信框架 |
| CuTe DSL | NVIDIA CUTLASS Python DSL for custom GEMM |
| Blockscaled FP8 | 每 32 个元素共享一个 E8M0 scale 的 FP8 格式 |
| ISA-packed | 硬件 tensor core 要求的 scale 物理布局 |
| TMA | Tensor Memory Accelerator (Hopper/SM100 硬件单元) |
| ZeroMat | Zero-materialization, 避免中间 tensor 物化 |
| SwiGLU | `SiLU(gate) × up`, 门控线性单元变体 |
| MFU | Model FLOPS Utilization = 有效计算/峰值算力 |
| Roofline | 性能上限分析模型 (compute vs memory bound) |
| E8M0 | 8-bit exponent, 0-bit mantissa (纯 power-of-2 scale) |
| E4M3 | 4-bit exponent, 3-bit mantissa (FP8 数据格式) |
| HARD-fail | CI 中不可降级为 warning 的严格门禁,失败即阻断 |
| Sentinel | JIT 缓存系统中的标记文件,存在则跳过重编译 |
| GPFS | General Parallel File System, 多节点共享存储 |
| xdist | pytest-xdist 插件, 支持多进程/多 GPU 并行测试 |
| byte-exact | 逐字节完全相同的断言 (非近似容差) |
| RRMSE | Relative Root Mean Square Error = ‖a-b‖/‖b‖ |
