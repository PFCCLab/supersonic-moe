# WS2: SonicMoE 稀疏性假设审查报告

> 日期: 2026-05-06 | 分支: race-fix-paddle

## 待审查的推论

> "sonic-moe 假设一个 token 发送到的专家全都在一张卡上，所以 TK 维中每一个 Token 都切实发到了 K 个专家里，但是诸如 ernie-core、paddlefleet 中使用 deepep 的 moe 模块不能建立在这个假设上，所以单个 Token 发送到的专家只能小于等于 K，实际上比 sonic 官方的假想更稀疏，而且可能在 frontier 下具有一定正确性问题，而且当前实现会因为静态 shape 损失一些显存、甚至性能"

## 结论摘要

| 论点 | 判定 | 解释 |
|------|:---:|------|
| SonicMoE 假设 K 个专家全在一张卡 | **部分正确** | bench 测试确实使用 topk=E(全本地)；但代码本身已支持 K<topk |
| 生产中单 token 本地专家 < K | **正确** | EP=8, E_global=64, topk=8 → 平均 K_local=1 per rank |
| 存在正确性问题 | **错误** | masked (-1) 正确过滤, score=0 的 IEEE 754 保证精确零贡献 |
| 静态 shape 损失显存 | **极微** | TK_padded = TK + padding(≤2%), 不是 T×K 全量分配 |
| 性能损失 | **微小但真实** | padding rows 消耗 GEMM 算力但贡献 0, 损失 = pad_rows/TK_padded |

## 详细代码证据

### 1. SonicMoE 已正确处理变长 K (K < topk)

**`deepep_topk_to_sonic_metadata` (deepep_metadata.py:127-202)**:
```python
dispatched_indices: [N_recv, topk] int32, -1 = masked
valid = dispatched_indices >= 0           # 过滤 masked entries
tok_flat = tok_ids[valid]                 # 只保留有效分配
TK = tok_flat.shape[0]                    # 实际 token-expert 对数 (≤ N_recv × topk)
```

**`num_activated_expert_per_token_offset: [N_recv+1]`** 跟踪每个 token 的实际本地 K:
- 第 t 个 token 的本地 expert 数 = `offset[t+1] - offset[t]`
- 值范围: [0, topk] (可以是 0 如果该 token 的所有 K 个专家都在远端)

**`token_gather_sum_kernel` (reduction_over_k_gather.py:82-85)**:
```python
if is_varlen_K:
    Ms = tl.load(M_offset_ptr + t_idx)
    Me = tl.load(M_offset_ptr + t_idx + 1)
    K_this_token = Me - Ms  # 变长 K, 逐 token 不同
```

**结论**: 代码本身完全支持 K_local < K_global, 没有"假设全部在本卡"的硬约束。

### 2. Bench vs Production 的差异

**Bench (bench_mlpnode_topk_nsys.py:145-148)**:
```python
raw_scores = torch.randn(N_recv, E, device=device)
_, top_experts = raw_scores.topk(topk, dim=-1)  # topk=8, E=8 → 每 token 选全部 8 个
```
当 topk=E=8 时，每个 token 恰好选中所有 E 个本地 expert → TK = N_recv × K = 65536。

**生产 (EP=8, E_global=64, topk=8, E_local=8)**:
- 每个 token 选 8 个 expert from 64 total
- 均匀分布: 平均每 rank 收到 K × (E_local/E_global) = 8 × (8/64) = 1 个本地 assignment
- 所以 TK_prod ≈ N_recv × 1 = N_recv (远小于 N_recv × 8)

**影响**:
- Bench 测量的是 "计算密度最高" 场景 (所有 expert 满载)
- 生产中每 rank 的计算量约为 bench 的 1/K (约 1/8)
- MFU 公式中的 TK 在生产中应使用实际值，不应简单用 T×K

### 3. 正确性审查

**无正确性问题**。证明:

(a) **Forward**: padding 行用 `x_gather_idx[pad]=0, score[pad]=0`
```
contribution = score × y2 = 0 × y2 = IEEE 754 精确零
```

(b) **Backward**: `colvec_scale=s` 中 padding 位的 s=0
```
dz[pad] = dout @ W2^T × s[pad] × dSwiGLU(z[pad])
        = anything × 0 × anything = 0 (IEEE 754 保证 finite × 0 = 0)
```

(c) **Wgrad**: padding 行的 dz=0 参与 GEMM 但贡献零梯度:
```
dW = dz^T @ x  → dz[pad]=0 行的贡献 = 0 × x[0] = 0
```

(d) **变长 K combine**: `token_gather_sum_kernel` 正确使用 `M_offset` 跟踪每 token 的 segment
- K=0 的 token → Me=Ms → 循环不执行 → output[t]=0 (正确: 该 rank 没有这个 token 的结果)

### 4. 显存分析

**TK_padded 计算** (`deepep_metadata.py:229-275`):
```
对每个 expert e:
  count = tokens_per_expert[e]           # 实际分配数
  padded = ceil(count / 128) × 128       # 对齐到 128
  total_pad_rows += padded - count
TK_padded = sum(padded for all e)
```

**量化** (Ernie shape bench: E=8, TK=65536, each expert=8192):
- 8192 % 128 = 0 → **零 padding** (完美对齐!)
- TK_padded = TK = 65536, waste = 0%

**非均匀路由** (worst case: extreme imbalance):
- 如 expert 0 = 65528, expert 1-7 = 1 each:
  - expert 0: ceil(65528/128)×128 = 65536 (+8)
  - experts 1-7: ceil(1/128)×128 = 128 each (+127 each)
  - total_pad = 8 + 7×127 = 897
  - TK_padded = 65536 + 7×128 = 66432
  - waste = 897/66432 = **1.35%**

**结论**: padding 浪费极小，且不是"静态分配 T×K 全量"。分配精确到实际需要。

### 5. 性能影响

Padding 行通过 GEMM 的计算浪费:
- Ernie 均匀分布: 0% 浪费 (完美对齐)
- 生产非均匀: ~1-2% 浪费 (route-level padding)
- 比"token-level padding" (给每个 token pad 到 K) 好 9× (Session 57 验证)

**另一个性能考量**: bench 中 E=8, topk=8 → 每 expert 8192 tokens (大 GEMM, 高效率)。
生产中 EP=8, 每 expert ≈ 1024 tokens → 更小的 per-expert segment → tile 利用率下降 → GEMM 效率低。

这不是 SonicMoE 的"假设错误"，而是"bench 与生产场景的自然差异"。

### 6. 建议

1. **文档化**: bench 测量的是"单卡全本地 expert"场景，MFU 数字反映的是 kernel 效率上界
2. **生产 MFU 预估**: 应该用 `TK_actual = N_recv × K_effective` 而非 `T × K` 计算 matmul_flops
3. **进一步优化**: 对于 EP>1 场景，per-expert token count 变小 → varlen GEMM 的 tile waste 增加 → 可考虑 multi-expert tile fusion
4. **无需修复正确性**: 代码已经正确处理了所有边界情况
