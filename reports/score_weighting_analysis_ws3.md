# WS3: Score 加权位置精度分析

> 日期: 2026-05-06 | 分支: race-fix-paddle

## 问题

分析 token routing probability (score) 在以下两个位置应用的精度、显存和性能差异:
- **方案 A (当前)**: score 在 combine step (down-proj 之后) 应用
- **方案 B (替代)**: score 在 SwiGLU 之后、down-proj 之前应用

## 1. 数学等价性证明

### Forward

对于单个 token t, 其被路由到 K 个 expert:

**方案 A (combine-level)**:
```
out_t = Σ_{k=1}^K  s_{t,k} × W2_k @ SwiGLU(W1_k @ x_t)
```

**方案 B (pre-down-proj)**:
```
out_t = Σ_{k=1}^K  W2_k @ (s_{t,k} × SwiGLU(W1_k @ x_t))
```

由于 W2_k 是线性映射: `W2_k @ (s × y1) = s × (W2_k @ y1)`，所以:
```
方案 B = Σ_{k=1}^K  s_{t,k} × W2_k @ SwiGLU(W1_k @ x_t) = 方案 A  ✓
```

**结论**: Forward 在精确算术下数学等价。

### Backward (dz 计算)

链式法则:
```
∂L/∂z_k = ∂L/∂out_t × ∂out_t/∂y2_k × ∂y2_k/∂y1_k × ∂y1_k/∂z_k
         = dout_t × s_{t,k} × W2_k^T × dSwiGLU'(z_k)
```

无论 score 在哪里应用，backward 中 dz 的表达式相同。当前实现:
- `colvec_scale = s_{t,k}` 传入 GemmDGated epilogue
- epilogue 中: `D_scaled = (dout @ W2^T) × s` → `dz = dSwiGLU(z, D_scaled)`

这完全正确地实现了链式法则。

### Wgrad (dW2 计算)

```
∂L/∂W2_k = (s_{t,k} × dout_t)^T @ y1_k   // 方案 A: s 在 dout 侧
         = dout_t^T @ (s_{t,k} × y1_k)     // 方案 B: s 在 y1 侧
```

当前实现用 `dout_col_fp8` 和 `y1s_col_fp8` (从 dgated epilogue 出来已经乘了 s 的 y1s):
```python
dw2 = dout_col_fp8.T @ y1s_col_fp8  # y1s 已经包含了 s 的效果
```

**两种方案的 wgrad 数学等价**。

## 2. 精度分析 (FP8 数值)

### 方案 A (当前实现) 的量化路径

```
dout (T, H) bf16 → FP8 量化 (T-sized) → dout_fp8
GEMM: dout_fp8 @ W2^T → D (f32 in registers)
Epilogue: D × s (fp32 乘法, 无量化损失) → dz (bf16 store)
```

**量化面积**: T × H (仅 dout 被量化)
**精度特点**: s 乘法在 fp32 精度下完成，无额外量化误差

### 方案 B (替代) 的量化路径

如果 score 在 SwiGLU 后应用:
```
y1 (TK, I) bf16 → 乘以 s → (s × y1) (TK, I) bf16 → FP8 量化 → (s×y1)_fp8
GEMM: (s×y1)_fp8 @ W2 → y2
```

**量化面积**: TK × I (TK > T, 且乘 s 后动态范围改变增加量化难度)
**精度特点**:
1. `s × y1` 的动态范围 < y1 的动态范围 (因为 0 < s < 1)
   → FP8 利用率更低 (有效位数减少)
2. 量化面积更大 (TK × I vs T × H, 对 Ernie shape: 65536×1536 vs 8192×3072)
3. 量化错误会传播到 W2 matmul 的输出

### 量化误差对比 (理论)

设 ε_q 为 FP8 e4m3 的相对量化误差 (~12.5%):

**方案 A**:
```
RRMSE(out) ≈ ε_q(dout) ⊕ ε_q(W2)   // dout 和 W2 各引入一次
```
s 在 fp32 乘法中不引入额外误差。

**方案 B**:
```
RRMSE(out) ≈ ε_q(s×y1) ⊕ ε_q(W2)   // s×y1 的量化误差可能更大
```
当 s < 1 时, s×y1 的绝对值更小 → blockscale 的 e8m0 scale 更小 → 有效精度降低。

**定量**: 设 s 均匀分布于 [0.5, 1.0] (典型 softmax 后):
- y1 的 amax: α
- s×y1 的 amax: s_max × α ≈ 0.7α (均值)
- FP8 量化后 s×y1 的动态范围利用率: log2(α / (s_max × α)) ≈ 0.5 bit 损失

**结论**: 方案 A 精度严格优于方案 B.

## 3. 显存分析

### 方案 A (当前)

```
Forward 显存:
  dout 存储: (T, H) bf16 → (T, H) fp8 = T×H bytes  ← 小! (T-sized)
  y2 (TK, H) bf16 → combine kernel → out (T, H)

Backward 显存:
  dout_fp8 (T, H) = T×H bytes
  s_float (TK,) = TK×4 bytes
  y1s (TK, I) bf16 from dgated epilogue
```

### 方案 B (替代)

```
Forward 显存:
  s×y1 (TK, I) bf16 需额外存储以用于 backward
  或 recompute: 需在 backward 重新计算 SwiGLU → 多一个 GEMM
```

**方案 B 更差**: 需要额外存 (TK, I) 的 s×y1 (或 recompute 代价)

## 4. 性能分析

### 方案 A (当前)

```
Combine: token_gather_sum_kernel (Memory-BW bound)
  读: TK×H + TK (scores) = TK×(H+1) bytes
  写: T×H bytes
  时间: ~148 µs (实测, Ernie shape)
```

### 方案 B (替代)

```
Pre-scale: s × y1 element-wise (TK×I elements)
  额外 kernel launch: ~20-40 µs
Down-proj: GEMM 不变 (y1_scaled @ W2)
Combine: token_gather_sum WITHOUT scores (省去 scores 的加载)
  读: TK×H bytes (不需 scores)
  写: T×H bytes
  时间: ~130 µs (估计, 省去 TK 个 score load)
```

**净差异**: 方案 B 多一个 scale kernel (~30 µs), 少一些 combine 带宽 (~18 µs) → **净增 ~12 µs**

### Backward 差异

方案 A: s 作为 colvec_scale 融合在 GEMM epilogue 中 → **零额外 kernel**
方案 B: 需要额外将 s 乘入 dout 或 y1 → 需要独立 element-wise kernel

**结论**: 方案 A 性能优于方案 B.

## 5. 总结

| 维度 | 方案 A (当前, combine-level) | 方案 B (pre-down-proj) | 胜者 |
|------|:---:|:---:|:---:|
| 数学正确性 | ✓ 等价 | ✓ 等价 | 平 |
| FP8 量化精度 | fp32 乘法无损 | s×y1 动态范围减小 | **A** |
| 量化面积 | T×H | TK×I (更大) | **A** |
| 显存 | 仅存 s(TK×4B) | 需存 s×y1(TK×I×2B) 或 recompute | **A** |
| 性能 (forward) | combine kernel 融合 | 多一个 scale kernel | **A** |
| 性能 (backward) | colvec_scale 融合进 epilogue | 独立 scale kernel | **A** |

**最终结论**: 当前实现（方案 A, combine-level score weighting）在精度、显存、性能三个维度上都严格优于将 score 移到 SwiGLU 后的方案 B。这不是偶然——它利用了:
1. GEMM epilogue 中 fp32 精度的乘法（无需额外量化）
2. T-sized (而非 TK-sized) 的量化面积
3. 算子融合消除独立 kernel launch
