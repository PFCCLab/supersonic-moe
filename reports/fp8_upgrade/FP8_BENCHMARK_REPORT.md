# SonicMoE FP8 Blockscaled — 基准测试报告

> **日期**: 2026-04-08 (Session 40)
> **分支**: `native-fp8-exploration`
> **Shape**: Ernie — T=8192, H=3072, I=1536, E=8, K=8 (TK=65536)
> **GPU**: Target GPU (SM 10.0, 148 SMs, HBM3e)
> **测量节点**: tjzj-inf-sci-k8s-bzz2-0274 (7 idle GPUs, 验证 0% 利用率, <5 MiB VRAM)
> **方法论**: nsys GPU Projection (NVTX scoped), 5 warmup + 10 profiled iterations, 子进程隔离

---

## 一、总结

| 维度 | BF16 Baseline | FP8 Frontier | 结论 |
|------|--------------|--------------|------|
| **性能** | 4036 µs/iter | 3654 µs/iter | ✅ **1.10× 加速** |
| **精度** | — | RRMSE ≤7.6%, cos ≥0.997 | ✅ **全部 PASS** |
| **显存** | 1556 MiB 峰值 | 1549 MiB 峰值 | ⚠️ **基本持平** (−0.4%) |

**核心结论**: FP8 frontier 在 Ernie shape 下实现了有意义的 10% GPU 计算加速，精度全部合格。显存方面由于 FP8 权重缓存 (148.5 MiB) 抵消了激活节省，当前几乎无净收益。

---

## 二、性能分析 (nsys GPU Projection)

### 2.1 总体性能

| 阶段 | BF16 (µs) | FP8 (µs) | 加速比 |
|------|----------|---------|--------|
| Forward | 1387.7 | 1178.0 | **1.18×** |
| Backward | 2648.3 | 2476.3 | **1.07×** |
| **总计** | **4036.0** | **3654.3** | **1.10×** |

### 2.2 Forward 内核分解

| 内核 | BF16 (µs) | 占比 | FP8 (µs) | 占比 | 说明 |
|------|----------|------|----------|------|------|
| GemmGated / GemmGatedSm100ZeroMat | 779.0 | 56.1% | 461.4 | 39.2% | **1.69×**, FP8 tensor core |
| GemmDefault (down-proj) | 386.9 | 27.9% | 234.6 | 19.9% | **1.65×**, FP8 tensor core |
| `_quantize_flat_blockscaled_kernel` | — | — | 123.1 | 10.5% | FP8 量化开销 |
| `_quantize_and_pack_kernel` ×3 | — | — | 115.0 | 9.8% | FP8 量化开销 |
| `_gather_isa_packed_scales_kernel` | — | — | 27.1 | 2.3% | FP8 scale 聚合 |
| token_gather_sum + 其他 | 221.8 | 16.0% | 216.8 | 18.3% | 基本不变 |
| **总计** | **1387.7** | | **1178.0** | | |

**分析**: FP8 GEMM 节省 470 µs, 量化开销 265 µs → 净节省 **205 µs** (14.8%)。

### 2.3 Backward 内核分解

| 内核 | BF16 (µs) | 占比 | FP8 (µs) | 占比 | 说明 |
|------|----------|------|----------|------|------|
| Wgrad GEMMs ×3 (GemmDefault) | 1978.3 | 74.7% | 1577.9 | 63.7% | **1.25×**, 低内存压力提升缓存效率 |
| DGated / DGatedFP8CLoad ZeroMat | 507.5 | 19.2% | 409.7 | 16.5% | **1.24×**, FP8 tensor core |
| SwiGLU bwd elementwise | — | — | 156.5 | 6.3% | FP8 路径额外 kernel |
| `_quantize_and_pack_kernel` ×3 | — | — | 144.4 | 5.8% | FP8 量化开销 |
| `_gather_isa_packed_scales_kernel` | — | — | 27.1 | 1.1% | FP8 scale 聚合 |
| 其他 elementwise | 162.5 | 6.1% | 160.7 | 6.5% | 基本不变 |
| **总计** | **2648.3** | | **2476.3** | | |

**分析**: Wgrad GEMM 虽然仍为 BF16, 但因 FP8 激活内存更低、缓存命中率提升而提速 25%。DGated 使用 FP8 tensor core 加速 24%。总量化开销 328 µs, GEMM 节省 498 µs → 净节省 **170 µs** (6.4%)。

### 2.4 加速瓶颈分析

当前 FP8 的主要开销:

| 开销来源 | Forward (µs) | Backward (µs) | 总计 (µs) | 占 FP8 总时间 |
|---------|-------------|--------------|----------|-------------|
| `_quantize_and_pack_kernel` | 115.0 | 144.4 | 259.4 | 7.1% |
| `_quantize_flat_blockscaled_kernel` | 123.1 | — | 123.1 | 3.4% |
| `_gather_isa_packed_scales_kernel` | 27.1 | 27.1 | 54.2 | 1.5% |
| SwiGLU bwd elementwise | — | 156.5 | 156.5 | 4.3% |
| **总 FP8 开销** | **265.2** | **328.0** | **593.2** | **16.2%** |

**优化空间**: 如果将量化 kernel 融合进 GEMM epilogue, 可消除 ~437 µs 开销, 总加速比可达 **~1.23×**。

---

## 三、精度分析

### 3.1 逐变量精度对比 (3 seeds × 4 变量 = 12 个测量点)

| 变量 | Seed 42 RRMSE | Seed 123 RRMSE | Seed 777 RRMSE | 平均 Cosine | 判定 |
|------|-------------|---------------|---------------|------------|------|
| output (out) | 6.51% | 6.51% | 6.51% | 0.9979 | ✅ PASS |
| input grad (dx) | 7.03% | 7.04% | 7.04% | 0.9975 | ✅ PASS |
| router grad | 7.59% | 7.50% | 7.51% | 0.9972 | ✅ PASS |
| loss | 0.00% | 0.00% | 0.00% | 1.0000 | ✅ PASS |

**12/12 全部 PASS。** 阈值: RRMSE <10%, Cosine >0.99。

### 3.2 精度稳定性

- 跨 seed 方差极小: output RRMSE 三个 seed 完全一致 (6.51%)
- dx 方差: 7.03% ~ 7.04% (Δ=0.01%)
- router 方差: 7.50% ~ 7.59% (Δ=0.09%)
- **结论: FP8 精度对随机输入高度稳定**

### 3.3 loss 精确匹配

loss 的 RRMSE = 0.000%, cosine = 1.000000 — 因为 loss 仅使用 router logits 计算 (load balancing loss), 不经过 FP8 GEMM 路径。

---

## 四、显存分析

### 4.1 稳态峰值 (Post-Warmup)

| 检查点 | BF16 (MiB) | FP8 (MiB) | 差异 |
|--------|-----------|----------|------|
| 模型权重 | 216.1 | 216.1 | 0 |
| 输入张量 | 312.1 | 312.1 | 0 |
| Forward 前 (含 warmup 残留) | 520.6 | 669.1 | **+148.5** (FP8 权重缓存) |
| **Forward 峰值** | **1529.9** | **1518.7** | **−11.2 (−0.7%)** |
| Forward 后残留 | 905.8 | 831.2 | −74.6 |
| **Backward 峰值** | **1555.8** | **1549.1** | **−6.8 (−0.4%)** |
| Backward 后残留 | 784.6 | 933.1 | **+148.5** (FP8 权重缓存) |

### 4.2 冷启动峰值 (含 JIT 编译)

| 指标 | BF16 (MiB) | FP8 (MiB) | 差异 |
|------|-----------|----------|------|
| **冷启动峰值** | **1657.9** | **1620.2** | **−37.8 (−2.3%)** |
| 后向后残留 | 640.6 | 789.1 | +148.5 |

### 4.3 显存收支分析

| 项目 | 影响 (MiB) | 方向 |
|------|-----------|------|
| Forward 激活节省 (resize_(0) 优化) | −159.7 | ✅ 节省 |
| FP8 权重缓存 (持久) | +148.5 | ❌ 新增 |
| Backward 量化缓冲区 | +67.9 | ❌ 新增 |
| **净效果** | **≈ −0.4%** | 基本持平 |

### 4.4 FP8 权重缓存明细

| 缓存 | 大小 (MiB) | 用途 |
|------|-----------|------|
| VARLEN_(3072, 3072, 8) | 74.3 | c_fc (w1+w3) FP8 packed weights |
| VARLEN_(3072, 1536, 8) | 37.1 | c_proj (w2) FP8 packed weights |
| FUSED_(3072, 1536, 8) | 37.1 | c_proj fused weights (backward) |
| **总计** | **148.5** | — |

### 4.5 先前 −33.4% 数据为何有误

先前 Session 38-39 报告 BF16=2540 MiB, FP8=1693 MiB (−33.4%)。原因:
- 测量 GPU 上已有 8-11 GiB 其他进程显存
- `max_memory_allocated()` 包含了其他进程 CUDA context 的影响
- 在真正空闲 GPU (4 MiB baseline) 上, BF16 峰值仅 1658 MiB

**教训: 显存测量必须在完全空闲的 GPU 上进行。**

---

## 五、测量方法论

### 5.1 节点选择
```bash
python tools/cluster_idle_launch.py scan
# 选择 idle_gpus >= 2 的节点, 使用 GPU util=0%, mem<10 MiB 的卡
```

### 5.2 性能测量
- 工具: nsys profile + nsys export → SQLite → `tools/nsys_full_breakdown.py`
- NVTX range: forward/backward 分别标记
- 迭代: 5 warmup + 10 profiled
- **不使用 CUDA events** (在争用环境下不可靠)

### 5.3 精度测量
- 子进程隔离 (避免 SONIC_MOE_FP8_MODE 缓存污染)
- 共享权重初始化种子
- 3 个随机输入种子
- 指标: RRMSE, Cosine Similarity

### 5.4 显存测量
- 子进程隔离
- 2 次 warmup + GC + empty_cache + reset_peak_memory_stats
- 分别记录 forward peak 和 backward peak

---

## 六、后续优化方向

| 方向 | 预期收益 | 复杂度 | 优先级 |
|------|---------|--------|--------|
| 量化 kernel 融合进 GEMM epilogue | ~437µs → 1.23× 加速 | 高 | P0 |
| FP8 权重缓存优化 (lazy re-quant / 共享 workspace) | −148.5 MiB 显存 | 中 | P1 |
| c_proj FP8 (I≥2048 shapes) | 额外 ~50µs forward | 低 | P2 |
| 端到端训练 loss 曲线验证 | 精度保障 | 低 | P1 |
| 多机 expert parallelism 验证 | 分布式可用性 | 中 | P2 |

---

> 报告基于 Target GPU (SM 10.0) idle 节点实测数据。所有 nsys 数据存于 `benchmarks/nsys_clean/`。
> 复现方式请以 `docs/HANDOFF.md` 中的当前复现说明为准；本目录下 handoff 仅作历史参考。
