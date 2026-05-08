# FP8 MoE 三方架构对比：DeepEP · SonicMoE BF16 · SonicMoE FP8 Frontier

> 作者：Session 33-35 探索总结 · 2026-04-07
> 基线：`native-fp8-exploration` branch @ `b651e17`

---

## 1. 核心设计哲学对比

| 维度 | DeepEP (Ernie) | SonicMoE BF16 | SonicMoE FP8 Frontier |
|------|---------------|---------------|----------------------|
| **设计目标** | 工程完备性：支持DP+EP+TP所有并行维度 | 算法极致性：IO-aware kernel fusion | 存算协同：在SonicMoE基础上压缩数据通路 |
| **权重格式** | BF16 master → 动态FP8 quant（可cache） | BF16 | BF16 master → blockscaled FP8（cache） |
| **激活通路** | 通信前quant → FP8 All-to-All → local compute | BF16 全链路 | BF16 x → blockscaled FP8 at compute boundary |
| **路由** | TopK + Softmax (外部，不在kernel内) | 融合 Softmax+TopK CUDA kernel | 同BF16 |
| **通信** | DeepEP buffer（NVLink/RDMA自适应） | 无EP通信（单节点MoE） | 同BF16 |
| **后端** | CUTLASS + Paddle GroupGEMM | CUTLASS DSL (quack) varlen GEMM | CUTLASS DSL blockscaled varlen GEMM |

## 2. 数据流对比

### 2.1 DeepEP: 通信优先型

```
x(BF16) ─→ Router → TopK
         ├─→ fp8_quant_blockwise(x, "1x128") → x_fp8 + scale
         ├─→ DeepEP dispatch(x_fp8, scale) ──→ [All-to-All NVLink/RDMA]
         │                                      ↓
         │   recv_x_fp8 + recv_scale ─→ UnZip(per-expert reorder)
         │                              ↓
         │   fwd_gate_up_fp8(x_fp8, w1, scale) ─→ o1(BF16)
         │   SwiGLU(o1) ─→ y1(BF16)
         │   tilewise_quant(y1) ─→ y1_fp8 + scale
         │   fwd_down_fp8(y1_fp8, w2, scale) ─→ y2(BF16)
         │                              ↓
         │   DeepEP combine(y2) ──→ [All-to-All]
         └─→ out(BF16)
```

**关键特点：**
- FP8 quant 发生在通信前（`fp8_quant_blockwise`），减少 All-to-All 传输量 50%
- 通信+计算 overlap：`FusedDispatchAsync` 在 comm stream 上跑 dispatch，compute stream 跑 router loss
- 权重 FP8 cache：`fp8_quant_weight()` 预量化后挂在 weight 对象上，避免每步重新量化
- Backward overlap：wgrad 在 comm stream 上和 dispatch_backward 并行

### 2.2 SonicMoE BF16: 融合极致型

```
x(BF16, T×H) ─→ Router → TopK → gather metadata
              ├─→ GemmGated(x, w1, cu_seqlens, A_idx)
              │   ↓  kernel 内部：GEMM + SwiGLU + store z(BF16) + store y1(BF16)
              │   z(TK, 2I), y1(TK, I)
              │   ↓
              │   gemm(y1, w2, cu_seqlens) ─→ y2(TK, H)
              │   ↓
              │   router_score_weighted_scatter ─→ out(T, H)
              └─→ out(BF16)
```

**关键特点：**
- **零Expert循环**：varlen GEMM 用 `cu_seqlens_m` 编码所有 expert 段，单kernel处理
- **零Gather Materialization**：`A_idx` 在 kernel 内部 gather，不产生 TK-sized 中间 buffer
- **GemmGated 融合**：GEMM + SwiGLU 在 epilogue 中完成，一次 mainloop 输出 z+y1
- **GemmDGated 融合**：backward GEMM + dSwiGLU + 可选 colvec_reduce 在单 kernel 完成
- 内存热点：z(TK, 2I) 必须保存给 backward dSwiGLU（BF16 = 384MB @ Ernie shape）

### 2.3 SonicMoE FP8 Frontier: 存算压缩型

```
x(BF16, T×H) ─→ Router → TopK → gather metadata
              ├─→ quantize_and_pack(x) ─→ x_fp8(T, H) + x_scales(T, H/32, ISA-packed)
              │   _gather_isa_packed_scales(x_scales, A_idx) ─→ x_scales_tk (ISA-packed)
              │   ↓
              │   GemmGatedSm100ZeroMat(x_fp8, w1_fp8, A_idx, x_scales_tk, w1_scales)
              │   ↓  kernel 内部：blockscaled FP8 GEMM + SwiGLU + store z(BF16) + y1(BF16)
              │   z(TK, 2I), y1(TK, I)
              │   ↓
              │   fused_z_save_y1_quant(z, y1) ─→ z_fp8 + z_scales, y1_fp8 + y1_scales
              │   ctx.save(z_fp8, z_scales)          # ~50% z memory saving
              │   _PREQUANTIZED_SCALES["fwd"] = (y1, y1_fp8, y1_scales)
              │   ↓
              │   blockscaled_fp8_gemm_varlen(y1_fp8, w2_fp8, scales...) ─→ y2(BF16)
              │   ↓
              │   router_score_weighted_scatter ─→ out(T, H)
              └─→ out(BF16)
```

**关键特点：**
- **零物化 FP8 gather**（Zero-Materialization）：x_fp8 保持 T-sized，A_idx+ISA-packed scales 在 kernel 内 gather
- **Prequant cache**：y1 量化与 GemmGated L2-hot 输出同步完成，down-proj 直接用 fp8，0 quant 开销
- **z FP8 save**：`fused_z_save_y1_quant` 同时量化 z+y1，z_fp8 存入 ctx 节省 ~171MB
- **Backward z dequant**：`dequantize_blockscaled_fp8(z_fp8, z_scales)` 在 backward 开始时恢复 BF16
- **Stream overlap**：z-dequant ‖ dout-quant 在不同 stream 并行，wgrad ‖ actgrad 并行

## 3. 计算效率分析

### 3.1 GEMM Precision vs. Throughput

| 路径 | Mainloop Precision | Epilogue | TFLOPS (Target GPU理论) |
|------|-------------------|----------|-------------------|
| DeepEP | FP8 E4M3×E4M3→FP32 | BF16 store | ~4500 (blockscaled) |
| SonicMoE BF16 | BF16×BF16→FP32 | BF16 store + SwiGLU | ~2250 |
| SonicMoE FP8 | FP8 blockscaled→FP32 | BF16 store + SwiGLU | ~4500 |

### 3.2 Up-Projection Kernel 融合度

| 系统 | Kernel 次数 | 描述 |
|------|------------|------|
| DeepEP | 3 | quant → GroupGEMM → SwiGLU (分离) |
| SonicMoE BF16 | 1 | GemmGated = GEMM + SwiGLU (epilogue 融合) |
| SonicMoE FP8 | 2 | quant+pack → GemmGatedZeroMat = GEMM + SwiGLU |

### 3.3 Backward 融合度

| 系统 | 操作 | Kernel 次数 |
|------|------|------------|
| DeepEP | dout×w2^T + dSwiGLU + y1×dout^T | 3+ (per-expert GroupGEMM) |
| SonicMoE BF16 | GemmDGated(dout, w2, z) + gemm(x^T, dz) | 2 |
| SonicMoE FP8 | z_dequant + dout_quant + blockscaled_dgated + gemm(x^T, dz) | 4+ |

> SonicMoE FP8 backward 的 kernel 数比 BF16 多，因为 GemmDGated 要求 BF16 PreAct（z），
> 必须先 dequant z_fp8→z_bf16。这是当前的核心瓶颈。

## 4. 内存效率分析

### 4.1 Activation Memory（以 Ernie shape: T=8192, K=8, E=8, H=3072, I=1536 为参考）

| 张量 | 大小 (BF16) | SonicMoE BF16 | SonicMoE FP8 | DeepEP |
|------|-------------|---------------|--------------|--------|
| x(T, H) | 48MB | 48MB (saved) | 48MB (saved) | 48MB → 24MB FP8 |
| z(TK, 2I) | 384MB | 384MB (saved) | ~213MB (fp8+scales) | 384MB (saved) 或 recompute |
| y1(TK, I) | 192MB | 不存（forward-only） | 不存（prequant cache） | 192MB 或 recompute |
| w1_fp8 cache | — | 0 | ~74MB | ~148MB (both layouts) |
| w2_fp8 cache | — | 0 | ~37MB | ~74MB (both layouts) |
| **peak delta vs BF16** | — | baseline | **-171MB +111MB = -60MB** | depends on recompute |

### 4.2 Weight Memory

| 系统 | 存储 | 训练时内存 |
|------|------|-----------|
| DeepEP | BF16 master + FP8 cache (可选) | master + cache + optimizer state |
| SonicMoE BF16 | BF16 only | weights + optimizer state |
| SonicMoE FP8 | BF16 master + FP8 cache | master + cache + optimizer state |

## 5. 精度控制对比

| 维度 | DeepEP | SonicMoE FP8 Frontier |
|------|--------|----------------------|
| **Scale 算法** | `ComputeScaleImpl<Power2Scaling>` + ε=1e-4 | integer+carry E8M0 (与Paddle/Triton bit-exact) |
| **Scale 格式** | UE8M0 (pow2 scale) on SM100 | UE8M0 ISA-packed |
| **量化粒度** | 1×128 blockscaled | 1×32 blockscaled (hardware-native) |
| **Epilogue 融合** | 无（quant在kernel外） | 有但当前 disabled（研发中） |
| **精度验证** | 训练 loss 对齐 | per-tensor RRMSE < 2.4%, cosine > 0.999 |

## 6. 当前瓶颈与攻坚路线

### 6.1 GemmDGated BF16 PreAct 约束

**问题**：`gemm_dgated.py` L126: `assert args.implicit_dtype.width == 16`
- GemmDGated 将 PreAct(z) 作为 C tensor 通过 `recast_tensor(bf16→f32)` 实现 bf16x2 packing
- 这意味着 z 必须以 BF16 传入 → backward 需要 dequant z_fp8→z_bf16（~124µs）

**攻坚方向**（按优先级排序）：

#### 方案 A：消除 forward dequant（不改 GemmDGated）✅ 推荐

核心 insight：forward 不需要 z_bf16 存在！baseline 的 `_save_z_fp8` 已经设计为：
1. Forward: 存 z_fp8+scales 到 ctx
2. Backward: dequant z_fp8 → z_bf16 → GemmDGated

dequant 只在 backward 发生，且可以与 dout_quant 在不同 stream 上 overlap。

**关键修复**：`z_is_fp8` 条件目前检查 `z.dtype == bf16`，但 epilogue quant 输出 fp8 z 时
条件不满足。修复方案：
```python
z_is_fp8 = (... and (z.dtype == bf16 or "z_fp8" in _PREQUANTIZED_SCALES))
```

**预期收益**：消除 forward dequant（~129µs），z 全程 FP8 → 节省 ~171MB。

#### 方案 B：Epilogue Quant 消除 standalone z_save_quant

已在 Session 34 中验证（commits 52bd65c-294b98b）：
- GemmGated epilogue 内实现 blockscaled quant（integer+carry 算法）
- 与 Paddle/Triton 参考实现 0 byte 误差
- 直接输出 z_fp8+scales，无需 standalone quant kernel

**问题**：GemmDGated 仍需 bf16 PreAct → 即使 epilogue 输出 fp8，backward 仍需 dequant。
但 epilogue quant 节省了 `fused_z_save_y1_quant` 的 kernel launch + HBM round-trip。

#### 方案 C：修改 GemmDGated 支持 FP8 PreAct（长期）

需要在 GemmDGated 的 C tensor 加载后插入 dequant：
```
load C_fp8 → register dequant → bf16x2 packing → dSwiGLU math
```

这需要：
1. 修改 `epi_to_underlying_arguments` 接受 fp8 C tensor
2. 在 `epi_visit_subtile` 的 C 加载后插入 register-level dequant
3. Scale 通过 EpiOp 传入 epilogue

**预期收益**：消除 backward dequant kernel（~124µs），但实现复杂度高。

### 6.2 实施优先级

```
Phase 1 (当前): 方案 A — z_is_fp8 条件修复 + epilogue quant 集成
  ├─ 1-line fix: z_is_fp8 condition
  ├─ Re-integrate epilogue quant from git history
  ├─ Run 31 tests + precision benchmarks
  └─ Expected: fwd -129µs, mem -171MB

Phase 2 (后续): 方案 B — Epilogue Quant 替代 standalone quant
  ├─ 已验证可行（0 byte error）
  ├─ 减少 1 个 kernel launch + 1 次 HBM round-trip
  └─ Expected: fwd -20~40µs additional

Phase 3 (长期): 方案 C — GemmDGated FP8 PreAct
  ├─ 消除 backward dequant entirely
  ├─ 需要 CUTLASS DSL 深度修改
  └─ Expected: bwd -124µs
```

## 7. Plan 审计：当前方案是否最优？

### 7.1 与 DeepEP 方案的对比审计

| 设计选择 | DeepEP 方案 | SonicMoE FP8 方案 | 优劣评估 |
|----------|------------|-------------------|---------|
| FP8 quant 位置 | 通信前（减少传输量） | 计算边界（减少 compute IO） | SonicMoE 无 EP 通信，选择正确 |
| 权重 FP8 | 动态 quant + cache | 同上 | 等价 |
| z save | BF16 或 recompute | FP8 blockscaled | **SonicMoE 优**：50% 压缩无 recompute 开销 |
| SwiGLU 融合 | 独立 kernel | Epilogue 融合 | **SonicMoE 优**：少 1 kernel launch |
| Backward overlap | wgrad ‖ dispatch_bwd | wgrad ‖ actgrad + z_dequant ‖ dout_quant | 等价 |

### 7.2 是否有更优替代方案？

**替代方案 1：z recompute 代替 z save（参考 DeepEP `recompute_moe_gate_up`）**
- Forward: 不存 z，backward 重算 `z = GemmGated(x, w1)`
- 优点：0 activation memory
- 缺点：多一次完整 GEMM forward（~200µs at Ernie shape），且 x 也需要存
- **结论**：当 z_fp8 save 已节省 50% 时，recompute 的 200µs 开销不值得。只在 memory 极端受限时考虑。

**替代方案 2：混合精度 wgrad（参考 DeepEP `fp8_wgrad` 开关）**
- 当前 SonicMoE wgrad 用 BF16 `gemm(x^T, dz)`
- 可以用 FP8 blockscaled GEMM 做 wgrad（x 和 dz 都已有 fp8 版本）
- **结论**：潜在 ~2x wgrad 加速，但精度影响需验证。列为 Phase 3 优化。

**替代方案 3：全 FP8 参数（native FP8 params）**
- Session 33-34 的探索证明：当前路径等价于 frontier FP8
- 真正的 native FP8 需要：预量化 x + 持久 FP8 权重 buffer + FP8 optimizer
- **结论**：这是完全不同的训练范式（类似 MSAMP），不适合增量优化。长期考虑。

### 7.3 审计结论

**当前方案（Phase 1: z_is_fp8 fix + epilogue quant）是最优的增量路径**，原因：
1. 不改变任何 kernel 的 mainloop 逻辑，只改 epilogue 和 Python routing
2. 内存节省确定：-171MB（z save FP8）
3. 性能改善确定：消除 forward dequant（-129µs）
4. 精度已验证：0 byte scale mismatch with Triton/Paddle
5. 回退安全：env var 控制，可随时回退到 BF16 baseline

**没有发现更优的替代方案可以在相同的工程复杂度下获得更好的 ROI。**

## 8. 三方总结表

| 能力 | DeepEP | SonicMoE BF16 | SonicMoE FP8 Frontier |
|------|--------|---------------|----------------------|
| EP 通信优化 | ⭐⭐⭐ | — | — |
| Kernel 融合度 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 内存效率 | ⭐⭐ (recompute可选) | ⭐⭐ | ⭐⭐⭐ |
| 计算吞吐 | ⭐⭐⭐ (FP8 mainloop) | ⭐⭐ (BF16 mainloop) | ⭐⭐⭐ (FP8 mainloop) |
| 精度控制 | ⭐⭐ (训练验证) | ⭐⭐⭐ (精确BF16) | ⭐⭐⭐ (per-tensor RRMSE<2.4%) |
| 工程完备性 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ (成熟中) |
| 扩展性（多节点） | ⭐⭐⭐ | ⭐ | ⭐ |

### 核心结论

1. **SonicMoE FP8 的核心竞争力**是 kernel 融合 + 零物化设计。DeepEP 的核心竞争力是通信优化。两者互补而非竞争。

2. **当前 FP8 frontier 的主要瓶颈**不是 mainloop（已经是 FP8），而是 **epilogue/backward 的数据格式转换**。z_fp8→z_bf16 dequant 是最大的单点开销。

3. **最优路线**：先修 z_is_fp8 条件消除 forward dequant（Phase 1），再集成 epilogue quant 减少 kernel launch（Phase 2），最后攻坚 GemmDGated FP8 PreAct 消除 backward dequant（Phase 3）。

4. **DeepEP 值得借鉴的模式**：
   - `recompute_moe_gate_up` 作为 memory 极端受限时的选项
   - `fp8_wgrad` 开关控制 wgrad 精度/速度 tradeoff
   - `_clear_to_zero_allocation()` 的激进内存释放策略
   - 分层 API（gate_compute/dispatch_compute/mlp_compute）支持外部 pipeline overlap
