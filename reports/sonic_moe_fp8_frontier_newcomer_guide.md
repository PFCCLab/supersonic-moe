# SonicMoE FP8 Frontier 新人入门与专家问答指南

> 目标读者：刚接手 SonicMoE、CuTe/CUTLASS、FP8、MoE 预训练高性能优化的新同学。
>
> 目标水平：读完后能解释 SonicMoE FP8 frontier 的端到端数据流、关键符号、kernel breakdown、roofline 数学、overhead 来源、精度风险和运维坑，并能回答苛刻的大模型预训练高性能专家追问。
>
> 当前事实基线：`race-fix-paddle` 分支，NVIDIA B30Z / Blackwell `sm_103`，Ernie production shape `T=8192, H=3072, I=1536, E=8, K=8`。最新 fresh benchmark 为 FP8 frontier `2659.8 µs/iter`、`46.51% MFU`；峰值测得 `51.61% MFU`。这些数字来自 `reports/fresh_benchmark_ws1/` 和根 `HANDOFF.md`，优先级高于历史报告。

---

## 0. 先建立正确心智模型

SonicMoE 不是“把 MoE 里的 GEMM 换成 FP8”这么简单。它是一个面向 Blackwell 的 **MoE expert MLP 计算子系统**，把路由后的 token-expert assignments、FP8 blockscaled 量化、CuTe/CUTLASS GEMM、SwiGLU、wgrad 累加、Paddle ERNIE `main_grad` 布局转换串成一条尽量少物化、少同步、少 HBM 往返的路径。

如果只记住一句话：

> SonicMoE FP8 frontier 的核心是：**让 token 不真实 gather，让中间激活尽早 FP8 化，让 SwiGLU/量化进入 GEMM epilogue，让 wgrad 直接 TMA reduce-add 到 ERNIE 的梯度缓冲，并用严格的 cache/key/metadata 规则保持可复现和可训练。**

这句话里每个词都很重要：

| 词 | 含义 | 为什么重要 |
|---|---|---|
| 不真实 gather | GEMM 通过 `A_idx` 在 kernel 内按索引读取原始 `x(T,H)`，不 materialize `x_gathered(TK,H)` | Ernie shape 下避免约 `TK*H*2B = 384 MiB` BF16 临时张量 |
| 中间激活 FP8 | `z(TK,2I)` / `y1(TK,I)` 保存为 FP8 + UE8M0 scales | 降低保存激活显存，但引入量化 kernel 和 scale 管理 |
| epilogue fusion | `GemmGated` 在寄存器中做 SwiGLU 和 blockscaled FP8 quant | 省 kernel launch 和 HBM 读写 |
| TMA reduce-add | wgrad GEMM 的 D store 用 TMA hardware atomic add | 避免 legacy `D=A@B+1*C` epilogue 的 C-load 和高寄存器压力 |
| cache/key/metadata | CuTe compile key 不含动态 token 数；weight cache 用 data_ptr + inplace version；route-level padding 保证对齐 | 没有这些，性能、确定性、动态图训练都会崩 |

---

## 1. 环境与上手顺序

### 1.1 你应该先读哪些文件

按这个顺序读，不要一上来全局 grep：

| 顺序 | 文件 | 作用 |
|---:|---|---|
| 1 | `README.md` | 默认路径、环境变量、Paddle 集成、JIT cache、CI 入口 |
| 2 | `HANDOFF.md` | 最新事实源；当前性能、NCU 结论、pitfalls、下一步优先级 |
| 3 | `reports/fresh_benchmark_ws1/README.md` | 最新 22 点性能 sweep 和 MFU 拟合模型 |
| 4 | `reports/ernie_shape_ncu_s78b/README.md` | 6 个 GEMM 的 NCU headline 和瓶颈解释 |
| 5 | `sonicmoe/functional/__init__.py` | 主 forward/backward 编排 |
| 6 | `sonicmoe/quack_utils/gemm_gated.py` | 正向 gated GEMM + epilogue quant |
| 7 | `sonicmoe/quack_utils/gemm_dgated.py` | 反向 dSwiGLU / C-load / FP8 preact |
| 8 | `sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py` | zero-materialization SM100 特化 |
| 9 | `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | Triton quant、varlen FP8 GEMM、TMA reduce-add |
| 10 | `sonicmoe/ernie_compat/mlp_node_v2.py` | ERNIE/Paddle `SonicMoEMlpNode`、`main_grad` 合约 |

### 1.2 环境基本纪律

本机是共享 root + GPFS 环境。做 profiling 或跑测试前要谨慎：

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source .runenv.sh
export USE_QUACK_GEMM=1
export SONIC_MOE_FP8_MODE=perf
export SONIC_MOE_FP8_WGRAD=1
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

运维注意：

| 事项 | 原因 |
|---|---|
| 优先用 `.runenv.sh` | 基础 `/usr/local/bin/python` 可能缺 paddle/quack；`.runenv.sh` 修正 PYTHONPATH、ptxas、CUDA 等 |
| nsys 加 `--resolve-symbols=false` | 离线环境符号下载会卡住 |
| ncu 异常退出后检查锁频 | `ncu` 可能残留 clock lock；用 `ncu --clock-control=reset` 修复 |
| 分布式/单卡测试要清理 Paddle 环境变量 | `PADDLE_TRAINERS`、`DISTRIBUTED_TRAINER_ENDPOINTS` 等会让 launcher 误入多机 rendezvous |
| 不抢忙卡 | 共享 8 GPU 机器；先扫描/查看 GPU util，必要时换卡或 yield |

---

## 2. MoE 从零开始

### 2.1 Dense FFN vs MoE FFN

Transformer 的普通 FFN 对每个 token 都执行同一组参数：

```text
x(t,H) -> Linear(H,4H) -> activation -> Linear(4H,H)
```

MoE 把 FFN 拆成多个 expert，每个 token 只进少数几个 expert：

```text
x(t,H)
  -> router logits [E]
  -> topK expert ids + scores
  -> dispatch 到 expert
  -> expert MLP
  -> combine 回原 token 顺序
```

SonicMoE 只负责 **expert MLP 计算** 这一段；router、DeepEP dispatch/combine 在上游 ERNIE/PaddleFleet 中完成，但 SonicMoE 必须接收它们产出的 metadata 并保证梯度能回流到 router score。

### 2.2 典型 expert MLP

每个 expert 是 SwiGLU MLP：

```python
z = x @ W1                 # W1 shape: [H, 2I]
gate, up = split(z)        # each [I]
y1 = silu(gate) * up       # SwiGLU
y2 = y1 @ W2               # W2 shape: [I, H]
```

在 SonicMoE 的 stacked layout 中：

| 权重 | ERNIE 原始语义 | SonicMoE native / kernel 语义 |
|---|---|---|
| `w1` | 每 expert `up_gate_proj.weight [H,2I]` | stacked 为 `[2I,H,E]`，wgrad native view `[E,2I,H]` |
| `w2` | 每 expert `down_proj.weight [I,H]` | stacked 为 `[H,I,E]`，wgrad native view `[E,H,I]` |

---

## 3. 符号表：看到变量名要立刻知道它是什么

### 3.1 维度符号

| 符号 | 含义 | Ernie 示例 |
|---|---|---:|
| `T` | 当前 rank 收到的 token 数 / DeepEP dispatched token rows | `8192` |
| `H` | hidden size | `3072` |
| `I` | MoE intermediate size；SwiGLU 的一半宽度 | `1536` |
| `2I` | up-proj 输出宽度，包含 gate 和 up 两半 | `3072` |
| `E` | local expert 数 | `8` |
| `K` | top-k slots / 每 token 激活 expert 数（benchmark 常用满 K） | `8` |
| `TK` | 有效 token-expert assignment 数，近似 `T*K` | `65536` |
| `TK_padded` | 按 expert segment 128 对齐后的 assignment 数 | ≥ `TK` |
| `N_recv` | DeepEP 收到的原 token 行数；topk path 下等于 `T_down` | `8192` |

### 3.2 路由与索引张量

| 张量 | shape | 空间 | 语义 |
|---|---:|---|---|
| `dispatched_indices` | `[N_recv, topk] int32` | token-major topk | 每个 token 的 local expert id；`-1` 表示 masked |
| `dispatched_probs` | `[N_recv, topk] fp32` | token-major topk | router score，必须可微 |
| `tokens_per_expert` | `[E]` | expert | 每个 local expert 实际收到多少 assignment |
| `expert_frequency_offset` / `cu_seqlens_m` | `[E+1] int32` | expert prefix | 每个 expert 在 expert-sorted TK 空间里的起止 offset |
| `x_gather_idx` | `[TK_padded] int32` | expert-sorted -> token | GEMM 读 A 时从原 `x(T,H)` gather 哪一行 |
| `s_scatter_idx` | `[TK_padded] int32` | token-major -> expert-sorted | router score 从 token-major 映射到 expert-sorted |
| `s_reverse_scatter_idx` | `[TK] int32` | expert-sorted -> token-major | combine / backward reduce 的逆映射 |
| `num_activated_expert_per_token_offset` | `[N_recv+1] int32` | token prefix | 每个 token 有多少有效 local expert assignment |
| `topk_scores` / `router_scores` | `[TK_padded] fp32` | expert-sorted | padding row score 为 0，保证 pad 不贡献输出和梯度 |
| `score_src_idx` | `[TK] int32` or `None` | expert-sorted -> token-major flat | CUDA metadata path 返回，用于可微重建 score |

两个空间最容易混：

```text
token-major topk space:
  flat j = token_id * topk + k_slot
  适合表达 router 输出

expert-sorted TK space:
  expert0 的 rows, expert1 的 rows, ...
  适合 varlen GEMM
```

桥接关系：

```text
x_gather_idx[expert_sorted_row] = original_token_row
router_scores[expert_sorted_row] = dispatched_probs[token_row, k_slot]
s_reverse_scatter_idx[token_major_flat] = expert_sorted_row
```

### 3.3 FP8 / scale 符号

| 张量 / dtype | 含义 |
|---|---|
| `torch.float8_e4m3fn` | FP8 E4M3 数据字节；max finite 约 448，实际代码按 kernel/参考一致的 fp8 max 处理 |
| `float8_e8m0fnu` / `uint8` | UE8M0 power-of-two scale 字节 |
| `x_fp8` | `x(T,H)` row-wise blockscaled FP8 |
| `x_scales_t` | `x` 在 token space 的 ISA-packed scales |
| `x_scales_tk` | 按 `x_gather_idx` gather 后的 TK-sized scales |
| `z_fp8` | up-proj preactivation `z(TK,2I)` 的 FP8 保存 |
| `z_raw_scales` | `z_fp8` 对应 scale，通常 shape 类似 `[TK, 2I/32]` 或 ISA-packed |
| `y1_fp8` | SwiGLU postactivation `y1(TK,I)` 的 FP8 |
| `dz_fp8` | backward `dz(TK,2I)` rowwise FP8，供 actgrad `dx` |
| `dz_col_fp8` | backward `dz` colwise FP8，供 `dw1` |
| `iso32` | 用一个 `32x32` amax 同时服务 row/col consumer 的 dz dual quant |

### 3.4 关键全局/缓存符号

| 符号 | 文件 | 作用 |
|---|---|---|
| `_PREQUANTIZED_SCALES` | `functional/__init__.py` | 在 `_UpProjection` 和 `_DownProjection` 之间传 `fwd/z_fp8/bwd/bwd_col` 预量化结果 |
| `_STASHED_FP8_WEIGHTS` | `functional/__init__.py` | weight BF16 storage 被释放/挪走时保留 FP8 cache 引用 |
| `_ALIGNMENT_ASSUMED` | `functional/__init__.py` | route-level padding 后默认相信 128 对齐，避免 D2H sync |
| `_FUSED_WEIGHT_CACHE` / `_VARLEN_WEIGHT_CACHE` | `blockscaled_fp8_gemm.py` | weight FP8 layout cache，key 含 `data_ptr + inplace_version` |
| `_COMPILE_CACHE_*` | 多个 quack utils | CuTe compile cache，key 只含静态模型维度 |
| `_GEMM_FAST_PATH_*` | `blockscaled_fp8_gemm.py` | runtime fast path cache，key 可含具体 problem shape，有 64-entry 高水位 |
| `_TOPK_CACHE` | `deepep_metadata.py` | topk CUDA metadata path 的高水位输出张量缓存 |

---

## 4. 端到端数据流

### 4.1 ERNIE/Paddle 入口

生产入口是 `SonicMoEMlpNode`：

```python
node = SonicMoEMlpNode(experts, n_experts=E, hidden_size=H, intermediate_size=I)

for step in range(num_steps):
    for mb in microbatches:
        out = node(dispatched_hidden_states, tokens_per_expert,
                   dispatched_indices, dispatched_probs)
        out.backward(grad)
    node.step()        # 必须在 optimizer.step() 前
    optimizer.step()
    optimizer.clear_grad()
```

`node.step()` 的意义不是优化器 step，而是 **把 CUTLASS native layout 的 fp32 wgrad buffer 就地转换成 ERNIE `weight.main_grad` 期望的 layout**。如果忘记或放到 `optimizer.step()` 后，优化器读到的是错误布局。

### 4.2 Metadata conversion

`deepep_topk_to_sonic_metadata()` 将 DeepEP 的 topk 输出变成 SonicMoE GEMM 需要的 expert-sorted metadata：

```text
dispatched_indices [N_recv, topk]
dispatched_probs   [N_recv, topk]
tokens_per_expert  [E]
  -> expert_frequency_offset [E+1]
  -> x_gather_idx             [TK_padded]
  -> s_scatter_idx            [TK_padded]
  -> s_reverse_scatter_idx    [TK]
  -> num_activated...offset   [N_recv+1]
  -> router_scores            [TK_padded]
```

当前优先使用 CUDA fused topk metadata kernel：warp-ballot progressive cumsum + micro-kernels fused by stream ordering。fallback 是 Python argsort，典型 shape 可能约毫秒级，不是生产热路径。

route-level padding：

```text
每个 expert segment pad 到 128 行对齐
padding row:
  x_gather_idx = 0       # 任意合法 row
  router score = 0       # 输出和梯度贡献为零
  不修改 x，不 append sentinel token
```

这使 FP8 GEMM 可以走 aligned fast path，同时避免 token-level padding 的大 tensor copy。

### 4.3 Forward 数据流

```text
Input:
  x: bf16 [T,H]
  w1: bf16 stacked [2I,H,E]
  w2: bf16 stacked [H,I,E]
  routing metadata: expert_frequency_offset, x_gather_idx, scores...

Step F0: prequantize_weights()
  w1/w2 BF16 -> 4 套 FP8 layout + ISA-packed scales
  cache key = data_ptr + inplace_version + shape/stride

Step F1: UpProjection / fused gated
  x(T,H) -> quantize_and_pack_activation -> x_fp8(T,H), x_scales_t
  x_scales_t --gather--> x_scales_tk
  GemmGatedSm100ZeroMatBlockscaledQuant:
      A = x_fp8(T,H)
      A_idx = x_gather_idx(TK)
      B = w1_fp8
      mainloop: 按 A_idx 读 x，不 materialize x_gathered
      epilogue: z = x @ W1
      epilogue: y1 = silu(gate) * up
      epilogue: z -> FP8 + UE8M0 scale
  output:
      y1 bf16 [TK,I]
      z placeholder or z_fp8 saved through _PREQUANTIZED_SCALES["z_fp8"]
      y1_fp8 + scales through _PREQUANTIZED_SCALES["fwd"]

Step F2: DownProjection
  pop _PREQUANTIZED_SCALES["fwd"]
  blockscaled_fp8_gemm_varlen:
      y1_fp8 [TK,I] @ w2_fp8 -> y2 bf16 [TK,H]

Step F3: router combine
  y2 expert-sorted -> output token order [T,H]
  multiply router_scores in fp32/bf16-safe path
```

### 4.4 Backward 数据流

```text
Input:
  dout bf16 [T,H]
  saved z_fp8 + scales, routing metadata, FP8 weight caches

Step B1: DownProjection backward / dGated
  dout -> quantize_and_pack_activation -> dout_fp8(T,H)
  gather dout scales to TK scale layout
  GemmDGatedFP8CLoadSm100ZeroMat:
      A = dout_fp8(T,H), A_idx=x_gather_idx
      B = w2_fp8
      C = z_fp8 viewed as Int16 pairs (gate+up)
      epilogue: dequant z_fp8 in registers
      epilogue: dSwiGLU -> dz bf16 [TK,2I] + y1s bf16 [TK,I]
      epilogue: accumulate ds partial

Step B2: wgrad dw2
  y1s -> colwise FP8
  dout -> colwise FP8 with gather
  CUTLASS varlen_k FP8 GEMM:
      dw2 += dout_col_fp8.T @ y1s_col_fp8
      accumulator is node._w2_native_view [E,H,I] fp32
      default uses TMA reduce-add

Step B3: dz dual quant
  dz bf16 -> iso32_dual_quantize_varlen
      one FP8 buffer dz_fp8
      row scales for dx
      col scales for dw1
  stash:
      _PREQUANTIZED_SCALES["bwd"] = row path
      _PREQUANTIZED_SCALES["bwd_col"] = col path

Step B4: UpProjection backward / wgrad dw1 first
  x -> colwise FP8 with gather_idx
  dw1 += dz_col_fp8.T @ x_col_fp8
  accumulator is node._w1_native_view [E,2I,H] fp32

Step B5: UpProjection backward / actgrad dx
  dz_fp8 row path @ w1T_fp8 -> dx_expanded [TK,H]
  token_broadcast_backward / reduce -> dx [T,H]

Step B6: node.step()
  native [E,2I,H] -> ERNIE [E,H,2I]
  native [E,H,I]  -> ERNIE [E,I,H]
```

---

## 5. Kernel breakdown：当前热路径到底花在哪里

### 5.1 nsys GPU-projection per-iter breakdown

Ernie shape `T=8192,H=3072,I=1536,E=8,K=8`，最新权威 breakdown：

| 类别 | kernel / 操作 | 时间 | 占比 | 解读 |
|---|---|---:|---:|---|
| wgrad GEMMs | QuACK wgrad GEMMs，4 calls | `1185 µs` | `44.1%` | 最大块；多数已 80%+ tensor pipe，接近可接受上限 |
| fwd gated | `GemmGatedSm100ZeroMatBlockscaledQuant` | `441 µs` | `16.4%` | 正向 up-proj + SwiGLU + z quant |
| bwd actgrad | `GemmDGatedFP8CLoadSm100ZeroMat` | `400 µs` | `14.9%` | dSwiGLU + FP8 C-load，寄存器最紧 |
| colwise quant | `_colwise_quantize_and_pack` | `242 µs` | `9.0%` | 已 num_warps=1 优化 |
| token combine/reduce | `token_gather_sum_kernel` | `148 µs` | `5.5%` | memory/scatter-gather bound |
| dz dual quant | `_dual_varlen_iso32_quantize` | `103 µs` | `3.8%` | iso32 后已显著降低 |
| row quant | `_quantize_and_pack` | `83 µs` | `3.1%` | bandwidth-bound |
| other | broadcast/index/etc. | `37 µs` | `1.4%` | launch/control overhead |
| total | GPU projection | `~2639-2660 µs` | `100%` | fresh benchmark median `2659.8 µs` |

注意：nsys GPU-projection 是按 NVTX BENCH 区间合并重叠 kernel 的 wall-clock/busy view，比单 kernel event 更适合作为端到端性能 gold standard。

### 5.2 NCU 对 6 个 GEMM 的相对瓶颈

NCU 使用 base clock + replay，绝对 duration 不能和 nsys duration 直接比，但 SoL、tensor pipe、register、L2/DRAM 相对结论有效：

| Launch | 角色 | kernel | NCU duration | Tensor pipe | DRAM | L2 hit | regs/thread | 结论 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | fwd1 | `GemmGated(fp8 swiglu epi)` | `762.66 µs` | `64.43%` | `10.51%` | `88.79%` | `168` | epilogue heavy，非 DRAM bound |
| 1 | fwd2 | `GemmDefault` | `360.96 µs` | `69.72%` | `19.50%` | `77.61%` | `54` | 较干净 |
| 2 | dgrad1 | `GemmDGated(fp8 dswiglu+CLoad)` | `610.98 µs` | `41.78%` | `22.48%` | `61.27%` | `168` | 最差 per-FLOP，寄存器/epilogue/C-load 限制 |
| 3 | dgrad2 | `GemmDefault` | `314.82 µs` | `80.93%` | `25.88%` | `68.55%` | `56` | 已接近最佳 |
| 4 | wgrad1 | `GemmDefault` | `581.54 µs` | `83.81%` | `24.80%` | `74.00%` | `56` | near peak |
| 5 | wgrad2 | `GemmDefault` | `600.83 µs` | `81.29%` | `15.69%` | `78.66%` | `54` | main-grad add epilogue 基本免费 |

最重要的反直觉结论：

> 不要再把“把 dz quant 融进 GemmDGated epilogue”当成默认 P0。NCU 已显示 `GemmDGatedFP8CLoadSm100ZeroMat` 是 `168 regs/thread × 384 threads = 64512/65536 regs`，寄存器余量约 1.6%。再加 amax/FP8 cast/scale store 循环会溢出或显著降 occupancy，历史上会触发 segfault。当前更可信的优化方向是重构 dgrad1 的 C-load/L2 reuse 或 kernel fission，而不是盲目 epilogue fusion。

---

## 6. 关键高性能实现细节

### 6.1 Zero-materialization GEMM

传统 MoE 会先 gather：

```text
x_gathered[TK,H] = x[x_gather_idx, :]
y = x_gathered @ W
```

SonicMoE 的做法：

```text
A = x_fp8[T,H]
A_idx = x_gather_idx[TK]
GEMM mainloop 内部:
  real_row = A_idx[m]
  load A[real_row, k]
```

优势：

| 项 | 节省 |
|---|---:|
| `x_gathered(TK,H)` BF16 | `65536*3072*2B = 384 MiB` |
| `x_gathered` FP8 | `192 MiB` |
| gather kernel launch + HBM write/read | 数十到百微秒级 |

实现点：

| 文件 | 关键点 |
|---|---|
| `gemm_sm100_fp8_zeromat.py` | `_GemmSm100ZeroMatMixin` 修正 gather_A + blockscaled 的 SFA layout |
| `gemm_gated_zeromat()` | `A` 是 `T` 行，不是 `TK` 行；`A_idx` 负责映射 |
| `_gather_isa_packed_scales_kernel` | data 不 gather，但 scales 要 gather 成 TK-sized，因为 blockscaled scale layout 跟 GEMM rows 对齐 |

为什么 scale 还要 gather？因为 tensor core blockscaled mainloop 读的是 `(M,K)` tile 的 scale。`M` 维在 GEMM 中已经是 expert-sorted `TK` 空间；如果仍用 `T` 空间 scale，expert 1+ 的 `cu_seqlens_m` offset 会错位。

### 6.2 Epilogue-fused blockscaled FP8 quant

Blockscaled FP8 的基本流程：

```text
对每个 block/group:
  amax = max(abs(values))
  e8m0 = encode_power_of_two_scale(amax)
  qscale = 2^(254 - e8m0)
  fp8_value = cast_e4m3(value * qscale)
  scale_byte = e8m0
```

`GemmGatedBlockscaledQuantMixin` 把这个流程放进 CUTLASS epilogue 的 register tile：

1. `tRS_rD` 中已有 GEMM accumulator。
2. 在寄存器上 reduce `amax`。
3. 用 integer+carry 算法得到 UE8M0 scale byte。
4. 对 `tRS_rD` 乘 `quant_scale`。
5. D store 时写 FP8；另通过 `BlockscaledScaleStore` 写 scale byte。

为什么不单独 kernel？

| 单独 quant | epilogue quant |
|---|---|
| GEMM 写 BF16 到 HBM | GEMM accumulator 仍在寄存器 |
| quant kernel 再读 BF16 | 直接 register quant |
| 写 FP8 + scale | 写 FP8 + scale |
| 多一次 launch + 大 HBM 往返 | 主要增加 epilogue 指令/寄存器 |

代价是寄存器压力。fwd gated 能承受；dgrad1 已经 168 regs/thread，再塞 quant 不诚实。

### 6.3 Fused SwiGLU

`GemmGated` 的 D 是 `z(TK,2I)`，postact 是 `y1(TK,I)`：

```text
gate = z[..., 0::2 or first half]
up   = z[..., 1::2 or second half]
y1   = silu(gate) * up
```

SonicMoE 利用 QuACK/CuTe epilogue 中的 activation callback，在 GEMM 结束后直接得到 `y1`。这省掉：

1. `z_bf16` 全量写出后再读入 SwiGLU kernel。
2. 单独 SwiGLU kernel launch。
3. `y1` 再读入 quant kernel 的一部分开销。

### 6.4 FP8 C-load 的 GemmDGated

Backward 中需要 `z` 来算 dSwiGLU：

```text
d gate, d up = d_swiglu(dy1, gate, up)
dz = concat(dgate, dup)
```

如果 forward 保存 `z_bf16(TK,2I)`，显存很贵。当前路径保存 `z_fp8 + scales`，`GemmDGatedFP8CLoadMixin` 在 epilogue 中把 `z_fp8` 当作 C 读进来：

```text
z_fp8 [TK,2I] viewed as Int16 [TK,I]
每个 Int16 包两个 fp8: gate + up
epilogue:
  recast Int16 -> 2 fp8
  fp8 -> f32
  multiply UE8M0 scale
  dSwiGLU
  output dz bf16 + y1s bf16
```

关键收益：避免单独 dequant `z_fp8 -> z_bf16` 的大临时张量。

关键限制：这个 epilogue 已经复杂，NCU 显示寄存器打满。

#### GemmDGatedCLoad 的寄存器是不是“浪费”了？

不能简单说“浪费”。`GemmDGatedFP8CLoadSm100ZeroMat` 的 168 regs/thread 来自多件必须同时活着的东西：

| live state | 为什么需要 | 是否容易删 |
|---|---|---|
| MMA accumulator `tRS_rD` | FP8 GEMM 的 fp32 accumulator | 不容易；这是主计算结果 |
| `tRS_rC` / Int16 preact load | TMA C-load 读 `z_fp8`，每个 Int16 包 gate/up 两个 fp8 | 不容易；删掉就退回 standalone dequant |
| `tRS_rXY_f32x2` | `z_fp8` dequant 后的 gate/up fp32 pair | 可优化，但 dSwiGLU 需要 |
| `tRS_rdXY_f32x2` | dSwiGLU 产生的 `dz` gate/up 梯度 | 可尝试流式化，风险高 |
| `tRS_rOut` | `y1s` / score-gradient reduction 相关输出 | 可拆分或延迟 |
| `colvec_scale/reduce` | router score 缩放与 `ds` partial reduce | 可考虑拆出，但会增加 kernel/HBM |
| zero-mat / varlen metadata | `A_idx`、expert offsets、bounds/mask | 不能删；正确性需要 |

所以“寄存器多”主要是 **融合设计的真实成本**，不是明显低级浪费。但仍有精简空间：

1. **缩短 live range**：把 dequant 的 gate/up 用完即释放，不让 `tRS_rXY_f32x2`、`tRS_rdXY_f32x2`、`tRS_rOut` 同时跨过太多循环。
2. **recompute vs store tradeoff**：对 sigmoid/SwiGLU 的中间量少存一些、多算一次，可能换寄存器；是否划算要看 SFU/ALU 压力。
3. **拆出 `ds` reduction**：如果 `mColVecReduce` 是寄存器峰值的重要来源，可以把 router score gradient reduction 变成独立轻 kernel。代价是一次额外读写和 grid launch。
4. **减小 epilogue tile 或改变 per-thread fragment**：降低每 thread 持有元素数可降 regs，但可能降低 tensor-core feed 效率或增加 CTA 数。
5. **C-load pipeline 重排**：把 `z_fp8` 的 load/dequant 更贴近消费点，减少 C fragment 和 dequant fragment重叠。
6. **fission**：拆成 “GEMM mainloop 写较窄中间结果” + “dSwiGLU/quant/reduce epilogue kernel”。这不是免费午餐，但当寄存器导致 occupancy 从可接受值掉到 1 CTA/SM 或出现 spill 时，fission 可能整体更快。

判断有没有实际空间不能靠源码肉眼，必须用 NCU 看：

```text
smsp__sass_average_data_bytes_per_sector_mem_local  # local spill
launch__registers_per_thread
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__pipe_tensor_op_hmma_cycles_active.avg.pct
smsp__inst_executed_pipe_alu / pipe_fma / pipe_tensor
lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct
```

如果没有 local spill、但 tensor pipe 只有 42%，通常说明瓶颈是 epilogue dependency / C-load / issue mix，而不是“随便减几个寄存器就会线性变快”。如果存在 spill，先消 spill；如果没有 spill，则要通过 live-range、tile shape 或 fission 改变调度结构。

### 6.5 iso32 dual quant

`dz` 同时有两个消费者：

```text
dx:  dz rowwise_fp8 @ W1T_fp8
dw1: dz_colwise_fp8.T @ x_colwise_fp8
```

传统会对 `dz` 做 rowwise 和 colwise 两次量化，甚至产出两个 FP8 buffer。iso32 的设计是：

```text
每个 32 rows × 32 cols 子块共享一个 amax
同一批 fp8 bytes 既给 row consumer 也给 col consumer
只额外写两套 scale layout（row-ISA / col-ISA）
```

为什么精度可以接受？

FP8 E4M3 每个值只有 3-bit mantissa，单值 rounding 粒度本来就是约 `O(2^-3)`。将 1x32 的 amax 放大到 32x32 会损失一些动态范围，但在真实 Ernie `dz` 分布上，下游 GEMM RRMSE ratio 测得 `1.000x`。换句话说，这里的误差增量被 E4M3 自身量化噪声吞没。

实现点：`_dual_varlen_iso32_quantize_kernel` 一次 HBM read、一次 FP8 store、两套 scale store，`num_warps=1`。

#### 什么时候 32x32 会带来精度下降？

32x32 的风险本质是 **同一个 scale 覆盖的动态范围变大**。令某一行的 1x32 amax 为 `a_r`，32x32 block amax 为 `a_b`，`a_b >= a_r`。对这一行里的小值，额外损失大约可用

```text
extra_bits_lost = log2(a_b / a_r)
```

衡量。`extra_bits_lost` 越大，小值越容易被粗量化甚至 flush 到 0。

| 数值情况 | 为什么会降精度 | 常见性 | 怎么避免 |
|---|---|---|---|
| 32 行里只有一行/少数行有巨大 outlier | outlier 抬高整个 32x32 block 的 scale，其他行的小梯度少 1-3 bit 有效精度 | 训练中偶发；梯度爆炸、bad batch、loss scale 不稳时更常见 | gradient clipping、loss scaling 稳定性检查；触发阈值时 fallback 到 1x32 dual quant |
| 同一 32x32 block 内行间方差极大 | `a_b/a_r` 对多数行很大，row-wise 自适应能力丢失 | 正常 Ernie `dz` 中实测不严重；极端路由/长尾 expert 可能增加 | 按 expert/row 统计 `log2(a_b/a_r)`；对高风险 expert 使用 1x32 |
| 大量值接近 FP8 subnormal/zero 边界 | scale 变大后，小值量化到 0，影响小梯度累积 | 后训练期、小学习率、强正则下可能出现 | 对关键梯度保留 BF16 或 1x32；监控 zero ratio |
| 分布非平稳，某一步有异常 spike | 单步误差可能传入 wgrad/dx | 不应假设永不发生 | online amax-ratio guard + precision canary |
| 把 iso32 用到 weight 或 activation 而不是 `dz` | weight/activation 分布不同，误差对收敛更敏感 | 不能直接从 `dz` 结论外推 | 重新做下游 GEMM + training loss 审计 |

一个实用 guard：

```text
per 32x32 block:
  r_i = log2(block_amax / row_amax_i)
  if percentile95(r_i) > 1.5 bits or max(r_i) > 2.5 bits:
      use 1x32 dual quant for this tensor/expert/step
```

阈值不是理论定理，需要用训练曲线校准。当前 Ernie `dz` 审计结论是：这些高风险情况在捕获样本中不常见，且下游 GEMM RRMSE/cos 没有恶化；但生产系统应保留 fallback 和监控，而不是把 iso32 写成无条件数学恒等。

### 6.6 TMA reduce-add wgrad

Wgrad 在 ERNIE 中要累积到 fp32 `main_grad`：

```text
main_grad += A @ B
```

legacy 做法是 CUTLASS epilogue：

```text
D = A @ B + 1.0 * C
```

这需要 epilogue load C，寄存器和 smem staging 都更重。当前默认用 TMA store reduce-add：

```text
GEMM accumulator -> TMA store with ADD -> accumulator buffer
```

代码点：`_run_cutlass_blockscaled_gemm_varlen_k_tma_add()` 中 `EpilogueArguments(add_to_output=True)`，TMA store 变成 `CopyReduceBulkTensorTileS2GOp(ADD)`。

实测收益：

| shape | 改善 |
|---|---:|
| E=8 | `-65 µs/iter`，约 `-2.3%` |
| E=32 | `-138 µs/iter`，约 `-4.0%` |
| regs/thread | legacy 约 `86` -> TMA add 约 `50` |

#### TMA reduce-add 是提升精度吗？

不是。它首先是 **性能/寄存器优化**，不是精度增强算法。

| 维度 | legacy beta-accum | TMA reduce-add |
|---|---|---|
| 数学表达 | `D = A@B + C` | `C += A@B` |
| accumulator dtype | 当前为 fp32 `main_grad` | 当前为 fp32 `main_grad` |
| 主要差异 | epilogue 读 C、做 fma/add、写 D | TMA store 侧做 hardware ADD |
| 精度上限 | fp32 累加顺序决定 | fp32 累加顺序决定 |
| 主要收益 | 无 | 少 C-load/smem/register，regs/thread 降低 |

精度风险来自 **累加顺序和重复累加语义**，不是 TMA 本身：

1. fp32 加法不满足结合律，tile store 顺序变化会带来 bit-level 差异。
2. 多 microbatch 梯度累积本来就是 `main_grad += grad_mb`，累积顺序固定时可复现；顺序变化时可能 bit 不同但通常数值误差很小。
3. TMA hardware add 不会自动做 Kahan、pairwise、stochastic rounding 或更高精度累加。
4. 如果多个 CTAs 对同一 `main_grad` tile reduce-add，必须保证调度/分块语义与期望一致；当前 determinism test 覆盖了同配置 bit-exact。

`main_grad` 的工程风险更多是合约错误：

| 风险 | 结果 | 防线 |
|---|---|---|
| `node.step()` 顺序错 | optimizer 读 native layout 梯度 | training loop contract + multilayer tests |
| `main_grad` 未清零或重复 flush | 梯度重复累加 | optimizer clear_grad 合约 + per-instance `_pending_flush` |
| FP8 wgrad cache stale | 用旧权重或旧 scale 算梯度 | data_ptr + inplace version key |
| atomic order 非确定 | bit-level drift | frontier determinism hard gate |

如果未来要“提升精度”，方向不是 TMA reduce-add，而是：BF16/FP8 输入误差控制、wgrad accumulator 保持 fp32、必要时局部 pairwise accumulation、关键层禁用 FP8 wgrad 或引入 error feedback。但这些都会牺牲性能或显存。

---

## 7. Roofline 与 MFU 数学分析

### 7.1 MFU 计量约定

**Model FLOPs Utilization (MFU)** 的标准定义（Chowdhery et al., PaLM 2022）：

$$
\text{MFU} = \frac{F_{\text{model}}}{t_{\text{iter}} \times \Pi_{\text{peak}}}
$$

其中 $F_{\text{model}}$ **仅计入 Tensor Core matmul FLOPs**（`2MNK` per GEMM），不包含：
- Epilogue ALU/SFU 操作（SwiGLU、dSwiGLU、bias、activation）
- Quantization / dequantization 计算
- Routing、gather/scatter、metadata
- 通信

这是业界惯例：MFU 衡量的是"硬件 matmul 管线利用率"，非 "总有用计算 / 总可用计算"。

### 7.2 精确 6-GEMM FLOPs 分解

一个 SwiGLU MoE layer 的 forward + backward 包含 **6 次 grouped GEMM**：

| # | 阶段 | GEMM shape (per expert) | FLOPs 公式 | 系数 |
|---|------|------------------------|------------|------|
| 1 | Fwd GemmGated: $\hat{x} \cdot W_1^\top$ | $(M_e, 2I) = (M_e, H) \times (H, 2I)$ | $2 \times M \times H \times 2I$ | $4 \cdot MHI$ |
| 2 | Fwd GemmDefault: $y \cdot W_2^\top$ | $(M_e, H) = (M_e, I) \times (I, H)$ | $2 \times M \times I \times H$ | $2 \cdot MHI$ |
| 3 | Bwd GemmDGated: $\partial o \cdot W_2$ | $(M_e, I) = (M_e, H) \times (H, I)$ | $2 \times M \times H \times I$ | $2 \cdot MHI$ |
| 4 | Bwd actgrad W1: $\partial z \cdot W_1$ | $(M_e, H) = (M_e, 2I) \times (2I, H)$ | $2 \times M \times 2I \times H$ | $4 \cdot MHI$ |
| 5 | Wgrad W1: $\hat{x}^\top \cdot \partial z$ | $(H, 2I) = (H, M_e) \times (M_e, 2I)$ | $2 \times H \times M \times 2I$ | $4 \cdot MHI$ |
| 6 | Wgrad W2: $y^\top \cdot \partial o$ | $(I, H) = (I, M_e) \times (M_e, H)$ | $2 \times I \times M \times H$ | $2 \cdot MHI$ |

**合计**：$(4+2+2+4+4+2) = 18 \cdot MHI$ 其中 $M = TK = T \times K$。

**关键澄清 — GemmDGated (#3) 为什么是 $2MHI$ 而非 $4MHI$**：

GemmDGated 的 matmul 部分仅是 $\partial o \cdot W_2$，输出 shape 为 $(M, I)$。虽然 kernel 在 epilogue 中融合了 dSwiGLU（产出 $(M, 2I)$ 的 $\partial z$），但 dSwiGLU 是 **ALU/SFU 逐元素操作**，不走 Tensor Core，故不计入 MFU 分子。

实际输出两路 TMA store 的形状之所以是 $(M, F)$ 和 $(M, F)$ 而非 $(M, 2F)$，是因为 dSwiGLU 在 accumulator 寄存器中原地完成，两路 $\partial z_g$ 和 $\partial z_u$ 分别 store。

### 7.3 GemmDGated 真实计算量分解

GemmDGated 是 FP8 frontier 中**最复杂的单体 kernel**（168 regs/thread，Block Limit=1），它在 GEMM mainloop 之上叠加了大量 epilogue 计算。精确拆分：

#### 7.3.1 Matmul 部分（Tensor Core）

```text
F_gemm = 2 × M × H × I
       = 2 × 65536 × 3072 × 1536
       = 6.176e11 FLOPs
```

这是标准 GEMM `2MNK`，计入 MFU。

#### 7.3.2 Epilogue 部分（ALU/SFU，不计入 MFU）

Epilogue 对每个 $(m, f)$ 元素对执行以下操作：

| 步骤 | 操作 | FLOPs/element | SASS 类型 |
|------|------|--------------|-----------|
| FP8 C-Load | `z_fp8 → f32 + scale_mul` | 2 (cast + mul) | MUFU + FMA |
| dSwiGLU 前半 | `σ(z_g)` | 1 | MUFU.TANH (SFU) |
| | `silu = z_g × σ(z_g)` | 1 | FMUL |
| | `silu × dout` | 1 | FMUL |
| | `σ - silu×σ` | 1 | FMA |
| | `(σ-silu×σ) × dout + silu×dout` | 1 | FMA |
| | `d_gate = above × z_u` | 1 | FMUL |
| | `d_up = silu × dout` | 0 (reuse) | — |
| | `swiglu_out = silu × z_u` | 1 | FMUL |
| Dual TMA store | 2路 fp8 量化 + store | 2 (scale+round) | FMUL×2 |
| BF16 store (swiglu) | dtype convert | 1 | F2F |

**每元素对 ~12 ALU/SFU ops**。总 epilogue FLOPs：

```text
F_epi = 12 × M × I
      = 12 × 65536 × 1536
      = 1.208e9 FLOPs
```

#### 7.3.3 为什么 epilogue 显著拖慢 DGated

虽然 $F_{\text{epi}} / F_{\text{gemm}} \approx 0.2\%$ 看似微不足道，但瓶颈不在 FLOPs 总量，而在 **pipeline 占用**：

1. **MUFU.TANH 吞吐**：SM90/SM100 上 MUFU 指令 throughput = 16 results/SM/cycle（¼ of FMA 的 64/cycle）。dSwiGLU 中的 sigmoid 必须走 MUFU，形成 SFU pipeline stall。

2. **寄存器压力**：168 regs/thread 意味着 Block Limit = 1（每 SM 只能驻留 1 个 CTA）。一旦 epilogue stall，没有其他 warp group 可以掩盖延迟。

3. **双路 TMA store**：正常 GEMM 只有 1 路 D 输出，DGated 需要写 $\partial z_g$、$\partial z_u$、`swiglu_out` 三路（其中 $\partial z$ 两路合并为双 TMA descriptor）。store 带宽需求 ~3× 正常 GEMM。

**Tensor Core pipe utilization 估算**：

```text
正常 GEMM（无重 epilogue）: TC pipe busy ~87%（profiled）
GemmDGated: TC pipe busy ~72%（profiled, SM100）

有效 MFU_DGated = 72% × (F_gemm / F_total_matmul) × nominal_MFU
```

### 7.4 全迭代 MFU 计算

Ernie shape：$T{=}8192, K{=}8, M{=}65536, H{=}3072, I{=}1536$

```text
F_total = 18 × M × H × I
        = 18 × 65536 × 3072 × 1536
        = 5.566e12 FLOPs
```

B30Z FP8 peak: $\Pi = 4500 \text{ TFLOPS} = 4.5 \times 10^{15} \text{ FLOPs/s}$

```text
ideal_time_at_100% = 5.566e12 / 4.5e15 = 1237 µs
```

实测：

```text
busy = 2659.8 µs
MFU  = 5.566e12 / (2659.8e-6 × 4.5e15) = 46.51%
```

#### 7.4.1 MFU Gap 分解（100% → 46.5%）

| 损失来源 | 估计占比 | 说明 |
|----------|---------|------|
| GEMM shape efficiency + varlen | ~46% | 非方阵、per-expert 碎片化、tile 边界浪费 |
| Epilogue pipeline stall (DGated 为主) | ~15% | SFU stall + 寄存器满 + 双路 store |
| 非 GEMM kernels (quant/gather/route) | ~23% | ~613 µs / 2660 µs |
| Fixed overhead (launch/control/cache) | ~8% | ~201 µs / 2660 µs |
| L2/DRAM bandwidth contention | ~8% | scale load、activation reuse miss |

### 7.5 Arithmetic Intensity 与 Roofline

以 Fwd GemmGated 为例：

```text
F_up  = 2 × M × H × 2I = 2 × 65536 × 3072 × 3072 = 1.236e12
Bytes = M×H + H×2I + M×2I  (FP8 = 1 byte/element)
      = 65536×3072 + 3072×3072 + 65536×3072
      = 2.01e8 + 9.44e6 + 2.01e8 ≈ 4.12e8 bytes

AI = 1.236e12 / 4.12e8 ≈ 3000 FLOPs/byte
```

B30Z ridge point：

```text
Ridge = Peak_compute / HBM_BW = 4500e12 / 8e12 = 562.5 FLOPs/byte
```

$\text{AI} \gg \text{Ridge}$，所有主 GEMM 深处 compute-bound 区域。**MFU gap 的主因不是带宽瓶颈，而是**：

1. **Tensor-core shape efficiency** — 非理想方阵 + varlen 导致 tile 填充率 < 100%
2. **Expert fragmentation** — E=8 个 expert 各自独立 tile scheduling，短 segment 边界浪费
3. **Epilogue 指令密度** — DGated 的 SFU stall 使 TC pipe 空转 ~15% 时间
4. **非 matmul kernels** — quant / metadata / scatter-reduce 占 ~23% wall time
5. **Fixed overhead** — launch / control / compile-cache lookup ~8%

### 7.6 经验 MFU 拟合模型

`tools/mfu_model.py` 对 `reports/fresh_benchmark_ws1/sweep.json` 里的 **多点 sweep** 拟合，不是拿 Ernie 一个点反推常数。当前 FP8 拟合使用 11 个 FP8 数据点，覆盖 token 数、expert 数和模型宽度变化；BF16 也有对应 11 个数据点用于交叉对照。

代表性 FP8 拟合点：

| shape | measured busy | MFU | 覆盖的变化 |
|---|---:|---:|---|
| `T1024-H3072-I1536-E8-K8` | `566.0 µs` | `27.32%` | 小 batch，固定开销主导 |
| `T2048-H3072-I1536-E8-K8` | `870.1 µs` | `35.54%` | crossover 前 |
| `T4096-H3072-I1536-E8-K8` | `1459.1 µs` | `42.39%` | crossover 附近 |
| `T8192-H3072-I1536-E8-K8` | `2659.8 µs` | `46.51%` | Ernie production |
| `T16384-H3072-I1536-E8-K8` | `5224.9 µs` | `47.35%` | 大 batch 渐近 |
| `T8192-H3072-I1536-E16-K8` | `2800.7 µs` | `44.17%` | expert 数增加 |
| `T8192-H3072-I1536-E32-K8` | `3187.5 µs` | `38.81%` | expert fragmentation 更强 |
| `T8192-H4096-I2048-E8-K8` | `4267.3 µs` | `51.53%` | 更宽 GEMM |
| `T8192-H4096-I4096-E8-K8` | `8521.7 µs` | `51.61%` | 更大 I，峰值 MFU |

拟合形式：

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

变量和常量意义：

| 符号 | 类型 | 意义 |
|---|---|---|
| `busy_us` | 因变量 | nsys GPU-projection 的 per-iter busy time，单位 µs |
| `T` | 自变量 | token rows |
| `K` | 自变量 | top-k slots；当前单机 benchmark 中 `TK=T*K` |
| `TK` | 自变量 | token-expert assignments；EP 生产中可小于 naive `T*K` 的本地满路由 |
| `H` | 自变量 | hidden size |
| `I` | 自变量 | intermediate size；up-proj 输出为 `2I` |
| `E` | 自变量 | local expert 数 |
| `4500e6` | 常量 | `4500 TFLOPS` 转成 `FLOPs/µs` 的 FP8 peak |
| `18*TK*H*I` | 物理项 | fwd+bwd 主 matmul FLOPs |
| `eta_max` | 拟合常量 | 该 kernel family 的有效 GEMM shape efficiency 天花板；吸收 varlen、tile、epilogue 影响 |
| `a_quant` | 拟合常量 | 与 `TK*max(H,2I)` 规模相关的 activation/scale/quant-ish overhead 系数；不是单独 quant kernel 计时 |
| `a_expert` | 拟合常量 | 与 `E*TK` 相关的 expert fragmentation / routing / per-expert overhead 系数 |
| `c_fixed` | 拟合常量 | per-iter 固定项，如 launch/control/cache lookup/metadata 常数部分 |

为什么要这样设计模型？

1. `18*TK*H*I` 是主计算量，决定大 shape 渐近斜率。
2. `TK*max(H,2I)` 近似 activation/scale 相关线性数据规模：有的 kernel 沿 H，有的沿 2I/I，取最大值是低维经验参数化。
3. `E*TK` 捕获 expert 数增加后的碎片化：同样 TK 下，E 越多，每 expert segment 越短，varlen scheduling 和 per-expert overhead 越明显。
4. `c_fixed` 捕获小 batch 时特别显著的固定开销。

对 Ernie 的代入只是 sanity check，不是拟合来源：

```text
t_gemm_model   ≈ 18*65536*3072*1536 / (4500e6*0.5416) = 2285 µs
t_quant_term   ≈ 50*65536*3072*1e-9 = 10 µs
t_expert_term  ≈ 328.939*8*65536*1e-6 = 172 µs
c_fixed        ≈ 201 µs
predicted      ≈ 2668 µs
measured       ≈ 2660 µs
```

由拟合 $\hat{t}_{\mu s}$ 反推 MFU：

$$
\widehat{\text{MFU}} = \frac{F_{\text{model}}}{\hat{t}_{\mu s} \times 10^{-6} \times \Pi_{\text{peak}}} = \frac{18 \cdot TK \cdot H \cdot I}{\hat{t}_{\mu s} \cdot \Pi_{\text{TFLOPS}} \cdot 10^{6}}
$$

其中 $\hat{t}_{\mu s}$ 为上式预测的 `busy_us`（µs），$\Pi_{\text{TFLOPS}}$ 为 FP8 峰值算力（TFLOPS，当前取 4500）。代入 Ernie 验证：

$$
\widehat{\text{MFU}} = \frac{5.566 \times 10^{12}}{2668 \times 4500 \times 10^{6}} = 46.3\% \quad (\text{实测 } 46.5\%)
$$

渐近 MFU（$TK \to \infty$，固定项可忽略）：

$$
\text{MFU}_{\infty} = \frac{1}{\frac{1}{\eta_{\max}} + \frac{\Pi \cdot a_q \cdot \max(H,2I)}{18 \cdot H \cdot I} + \frac{\Pi \cdot a_e \cdot E}{18 \cdot H \cdot I}} \approx 48.2\%
$$

解释限制：

1. 不要把 `t_quant_term=10µs` 误解成“量化 kernel 只花 10µs”。nsys breakdown 中 quant kernels 合计数百微秒。
2. 拟合项之间有相关性，`eta_max` 吸收了大量“非理想 GEMM + epilogue + pipeline”效应；`a_quant/a_expert/c_fixed` 是经验分解，不是硬件计数器。
3. 这个模型适合在相近路径、相近硬件、相近 routing 机制下预测 shape scaling；不适合替代 per-kernel attribution，也不应外推到通信占主导的多机 EP。
4. 如果引入新的 fusion/fission 或通信 overlap，常数必须重新拟合；不能继续沿用旧 `eta_max/a_*` 解释新系统。

### 7.7 Crossover：什么时候 FP8 反而慢

fresh sweep 显示：

| shape | BF16 busy | FP8 busy | FP8 speedup |
|---|---:|---:|---:|
| T=1024 | `540.0 µs` | `566.0 µs` | `0.95x` |
| T=2048 | `847.7 µs` | `870.1 µs` | `0.97x` |
| T=4096 | `1533.4 µs` | `1459.1 µs` | `1.05x` |
| T=8192 | `2942.5 µs` | `2659.8 µs` | `1.11x` |
| T=16384 | `6022.1 µs` | `5224.9 µs` | `1.15x` |

结论：

> FP8 不是无条件更快。小 token batch 时，FP8 quant/cache/scale/metadata 固定成本不能被 2x tensor core 吞吐摊平；crossover 大约在 `T=3000-4000`。

另一个容易误解的点：历史 `~1.37x` 是相对旧 cuBLAS/PyTorch BF16 baseline；当前 fresh `1.11x` 是相对 QuACK BF16 path。两个 baseline 不一样，不能混报。

---

## 8. Overhead 组成：专家会追问“慢在哪里”

### 8.1 显式 kernel overhead

| overhead | 当前量级 | 状态 |
|---|---:|---|
| colwise quant | `~242 µs` | 已 `num_warps=1`，继续优化空间有限 |
| row quant | `~83 µs` | bandwidth-bound，单 warp 无收益 |
| iso32 dual dz quant | `~103 µs` | 已从双 buffer/双 amax 降低 |
| token gather/reduce | `~148 µs` | scatter-gather/memory bound |
| routing metadata / misc | `~37-80 µs` | CUDA fused topk metadata 已避免 argsort 毫秒级开销 |

### 8.2 隐性 overhead

| overhead | 解释 | 对策 |
|---|---|---|
| JIT cold start | 首次 import CUDA extension + 首次 fwd/bwd CuTe/Triton 编译约 46s | shared GPFS cache + warmup |
| shape compile key 污染 | 如果 compile_key 含 `TK`，seqlen 变化会重编译 | compile key 只含静态模型维度；动态维用 `mark_layout_dynamic` |
| weight re-quant | optimizer step 后 weight 版本变，FP8 cache miss | cache key 自动含 inplace version；`prequantize_weights()` 单 pass 预热 |
| D2H sync | 对齐检查若每步 `.tolist()` 会同步 | route-level padding + `_ALIGNMENT_ASSUMED` |
| Paddle proxy API gap | `torch.equal`、stream、dtype、storage offset 行为不同 | 所有跨框架 API 用兼容分支 |
| nsys/ncu 工具偏差 | ncu base clock/replay；nsys boost clock/timeline | 不直接比较 duration，只比较同工具同 clock policy |

### 8.3 为什么不做全链路 megakernel

全链路 megakernel 听起来诱人。推理端如果 routing 已稳定、没有反向图、没有 wgrad、没有 optimizer/main_grad 合约，通信和 expert compute 可以更激进地融合；训练端难很多，因为 forward、backward、wgrad、router score grad、activation save/recompute、通信 overlap 和 optimizer layout 都耦合在一起。

| 约束 | 现实 |
|---|---|
| 中间张量太大 | `z_fp8(TK,2I)` Ernie 下 `~192 MiB`，不可能常驻 smem |
| smem 太小 | `48KB/SM` 量级，只能 tile 内 producer-consumer |
| 寄存器已满 | fwd1/dgrad1 都 `168 regs/thread` |
| wgrad 是 reduction | 跨 tile/expert 累加天然复杂 |
| profile/debug 变困难 | 单 kernel 内难归因 |

可行的是局部融合：已落地的 GEMM+SwiGLU+quant、FP8 C-load、TMA reduce-add；未来更现实的是 dgrad1 的 C-load reuse/fission，而不是“一把梭 megakernel”。

#### 人力无限时，训练端 megakernel 可行吗？

可行的不是“一个 kernel 吃完整 MoE layer fwd+bwd”，而是 **阶段化 megakernel / persistent pipeline**：

| 范围 | 训练端可行性 | 关键条件 |
|---|---|---|
| dispatch unpack + local expert fwd GEMM | 中等 | 通信库暴露 device-side queue / barrier；routing metadata 稳定；不破坏 DeepEP 语义 |
| fwd up-proj + SwiGLU + down-proj | 局部可行 | tile 内 `z/y1` 可在 smem/register 消费；但 `y1` 还要为 backward 保存或重算 |
| backward dGated + dz quant + dx | 有条件可行 | 解决 GemmDGated register cliff；可能需要 fission/pipeline 而非单 epilogue |
| wgrad + main_grad reduce-add | 已局部融合 | TMA reduce-add 是正确方向；跨 microbatch/optimizer 合约仍要在 kernel 外可控 |
| A2A communication + expert compute + combine | 研究型 | 需要跨 GPU progress、credit、ordering、fault isolation；训练还要保存 autograd context |
| 整层 fwd+bwd+optimizer megakernel | 不现实 | 生命周期跨 forward/backward，依赖 autograd schedule 和 optimizer step |

训练端真正可能的形态是：

```text
persistent scheduler CTA/cluster:
  while work_queue has expert tiles:
      receive/unwrap communication tile
      run up/down GEMM micro-pipeline
      emit saved activation or recompute token
      later consume dout tile
      run dgrad/wgrad micro-pipeline
      reduce-add to main_grad
```

这需要通信 runtime、autograd runtime、kernel scheduler 和 optimizer 都重构。人力无限可以做，但风险不是“写不写得出 kernel”，而是 **能否证明可复现、可调试、可恢复、可维护，并在真实 routing 分布下比局部融合更快**。

#### Fusion 与 fission 的充要判据

工程上没有单一数学定理，但可以用近似的充要条件判断。

设两个相邻阶段 `A -> B`，中间 tensor 大小为 `S` bytes，单独执行时间：

```text
T_sep = T_A + T_launch_gap + T_store(S) + T_load(S) + T_B
T_fuse = T_A' + T_B' + T_extra_regs + T_extra_sync + T_extra_control
```

**fusion 值得做的必要条件：**

1. `S` 足够大，或 launch gap 足够多：省掉的 HBM 往返/launch 是可见瓶颈。
2. producer/consumer tile shape 相容：`A` 产出的 tile 能被 `B` 立即消费，不需要全局 reorder/reduction。
3. 额外 live state 不会造成 spill 或 occupancy cliff：`regs_fused` 不能跨过关键阈值。
4. fused 后不破坏并行度：不能为了省一次 store，让 grid 从全 GPU 并行变成少数 CTA 串行。
5. 数值语义可保持：尤其是 reduction order、atomic order、router score grad。

**fusion 充分条件（实践版）：**

```text
saved_HBM_time + saved_launch_time + saved_allocator/cache_time
  >
extra_register_cost + lost_occupancy_cost + extra_recompute_cost
  + lost_parallelism_cost + debugging/maintenance risk premium
```

反过来，**fission 值得做的必要条件：**

1. 当前 fused kernel 被寄存器、smem、dependency chain 或 instruction mix 卡住，而不是被 HBM 纯带宽卡住。
2. 拆开后每个子 kernel 的 occupancy/tensor-pipe 利用显著提高。
3. 中间写回的 `S` 不太大，或能用 FP8/packed/smem/L2 locality 降低写回成本。
4. grid-level sync 正好是算法需要的边界：例如全局 amax、prefix-sum、跨 tile reduction、producer/consumer 负载均衡。

**fission 充分条件（实践版）：**

```text
(occupancy_gain + tensor_pipe_gain + reduced_spill + simpler_schedule)
  >
(extra HBM store/load + extra launch/grid sync + lost locality)
```

#### 什么时候 fission 的 grid sync 全局更优？

grid sync 看似昂贵，但在这些情况下可能全局更优：

| 场景 | 为什么 fission 更优 |
|---|---|
| fused kernel 触发 register cliff | 多一个 kernel launch 换来 occupancy 从 1 CTA/SM 到 2-4 CTA/SM，tensor pipe 利用大幅上升 |
| fused epilogue 长 dependency chain | 拆开后 GEMM mainloop 专心喂 tensor core，后处理 kernel 用不同 tile/warp layout |
| 需要全局信息 | quant amax、routing prefix、load balancing 需要全局边界；硬塞进单 kernel 会复杂且慢 |
| producer/consumer 最优 tile 不同 | GEMM 喜欢 128x128，quant/reduce 可能喜欢 32x128 或 1 warp；融合迫使一方用坏 tile |
| 中间 tensor 可压缩 | 如果中间写 FP8/packed scale 而不是 BF16，HBM 代价可低于 occupancy 损失 |
| 可利用 L2 locality | back-to-back launch 在同 stream 上读刚写的中间结果，L2 hit 高，grid sync 成本被摊薄 |
| 调试/确定性价值高 | 清晰边界让 determinism 和精度审计可控，减少 production 风险 |

对当前 SonicMoE，`GemmDGatedCLoad` 是 fission 候选，不是因为“fusion 不好”，而是因为它已经把 GEMM、FP8 C-load、dSwiGLU、score scale/reduce、dz/y1s 输出压进一个 epilogue，NCU 显示 tensor pipe 只有约 42%。如果 fission 能让 GEMM 部分接近 70-80% tensor pipe，而中间写回可控制在 FP8/packed 或 L2-friendly BF16，那么多一次 grid sync 可能全局更优。

---

## 9. 数值精度分析

### 9.1 FP8 blockscaled 误差模型

对一组 32 个或 32x32 个值：

```text
amax = max(abs(x))
scale = power_of_two(amax / fp8_max)
x_fp8 = round_e4m3(x / scale)
x_hat = x_fp8 * scale
```

E4M3 的主要误差来自 mantissa：

```text
相对 rounding 粗略上界 ~ 2^-3 = 12.5%
RRMSE 通常低于最坏情况，实测主路径 < 7.6%
```

GEMM 误差传播近似：

```text
Y = XW
X_hat = X + eX
W_hat = W + eW
Y_hat - Y = eX W + X eW + eX eW
```

如果误差近似零均值且不完全相关，下游 cosine 会远好于逐元素 worst-case。当前测试：

| 输出/梯度 | cos |
|---|---:|
| output | `~0.9979` |
| dx | `~0.9975` |
| router ds | `~0.9971-0.9973` |
| dw1 | `~0.9975` |
| dw2 | `~0.9971-0.9972` |

门限：所有 cosine > `0.99`，RRMSE < `7.6%`。

### 9.2 为什么 scale 用 UE8M0

UE8M0 是 power-of-two scale。好处：

1. 硬件/bit 操作简单，乘 scale 可变成指数位构造。
2. scale 存储 1 byte，overhead 低。
3. 与 Blackwell blockscaled FP8 tensor core layout 匹配。

代价：

1. scale 自身只能 power-of-two，不能精确表示任意 amax/fp8_max。
2. 需要 ISA-packed layout，不同 row/col consumer 的 scale 存储布局不同。
3. 对极端 outlier 分布，block 内小值相对误差会变大。

### 9.3 iso32 的风险边界

iso32 的经验结论来自真实 dz tensors，不应无限外推。

可信范围：

| 条件 | 说明 |
|---|---|
| Ernie-like `dz` 分布 | heavy-tailed 但非病态 |
| E4M3 主计算 | mantissa 噪声主导 |
| 下游指标是 GEMM output RRMSE/cos | 不是逐 byte 完全等价 |

需要重新验证的情况：

1. 换激活函数或 loss scaling，`dz` 分布显著变。
2. 使用不同 FP8 格式，比如 E5M2。
3. topk/EP 变化导致极端 sparse/empty expert 分布。
4. 专家想把 iso32 推到 weight quant；weight 分布和 activation gradient 分布不同，不能直接套。

### 9.4 确定性为什么重要

训练系统里“平均精度不错”不够。frontier 路径要求 bit-exact determinism，原因：

| 非确定性来源 | 风险 |
|---|---|
| TMA reduce-add atomic order | fp32 加法非结合，不同顺序可能 bit 不同 |
| async TMA 泄漏 | 上一 iter 数据污染下一 iter |
| global cache 污染 | scale/weight cache 过期但仍命中 |
| route padding 错误 | padding row 贡献非零输出/梯度 |

因此 `tests/fp8_frontier_determinism_test.py` 是 hard-fail：同状态多次 fwd/bwd 必须 byte-identical。

---

## 10. 当前优化状态与不要踩的坑

### 10.1 已经验证有效的优化

| 优化 | 状态 | 备注 |
|---|---|---|
| zero-materialization `A_idx` | 已落地 | 避免 TK-sized activation materialization |
| fused gated up-proj | 已落地 | GEMM + SwiGLU + z quant |
| FP8 save z | 已落地 | forward 激活显存下降 |
| FP8 C-load dGated | 已落地 | 避免 z_bf16 dequant temp |
| iso32 dual quant | 已落地 | `-61 µs/iter` 级别，precision-free lunch in measured scope |
| TMA reduce-add wgrad | 已落地 | 2-4% end-to-end |
| CUDA topk metadata | 已落地 | 避免 Python argsort hot path |
| single-stream execution | 已落地 | 避免跨 stream sync |
| dynamic CuTe layout | 已落地 | seqlen 变化不重编译 |
| fused weight prequant warmup | 已落地 | w1/w2 单 pass 产出四套 layout |

### 10.2 已经证明 ROI 低或不可行的方向

| 方向 | 结论 |
|---|---|
| GemmDGated epilogue 再融合 dz quant | 当前不可作为直接 P0：168 regs/thread，几乎无寄存器余量 |
| `_dual_varlen_iso32_quantize` 继续大改 | 已 74% BW efficiency，理论最多几十微秒，ROI 低 |
| row quant 改单 warp | 已验证无提升 |
| 全链路 megakernel | 受 smem/reg/reduction/debug 约束，不现实 |
| `SONIC_MOE_FP8_RECOMPUTE_OPT_B` | 非均匀 routing 会 illegal-instruction，默认禁止 |

### 10.3 仍可能有价值的方向

| 方向 | 为什么可能有价值 | 风险 |
|---|---|---|
| dgrad1 C-load/L2 reuse | NCU 显示 dgrad1 L2 hit 仅 61%、tensor pipe 42% | 需要重构 kernel/schedule |
| dgrad1 kernel fission | mainloop 与 heavy epilogue 拆开，让各自寄存器更合理 | 多一次 HBM 或 smem handoff，可能抵消收益 |
| layer-level overlap | wgrad 与下一层 forward overlap | 与 Paddle/DeepEP schedule 交互复杂 |
| weight iso32 | 减少 weight FP8 cache | weight 精度需重新审计 |
| 更好的 metadata/cache allocation | 降低 fixed overhead | 容易引入 stale cache / determinism bug |

---

## 11. 专家问答：刁钻问题与满分答复

### Q1：你说 FP8 frontier 46.5% MFU，分母是什么？FLOPs 有没有虚高？

**答：** 分母使用 B30Z FP8 peak `4500 TFLOPS`。FLOPs 采用 MoE expert MLP forward+backward 主 matmul 口径 `18*TK*H*I`。Ernie shape `TK=8192*8=65536,H=3072,I=1536`，FLOPs 为 `5.566e12`。实测 busy `2659.8µs`，所以 MFU=`5.566e12/(2659.8e-6*4.5e15)=46.51%`。这个口径不把 quant/scatter/metadata 算作有用 FLOPs，所以不会因为额外 kernel 虚高；反而这些 overhead 会降低 MFU。

### Q2：为什么不是 100%？GEMM arithmetic intensity 明明远高于 ridge point。

**答：** AI 高只能说明主 GEMM 不该被 HBM bandwidth 主导，不代表能达到 100% peak。实际限制包括 non-square/varlen shape efficiency、expert segment fragmentation、CuTe/CUTLASS tile scheduler、epilogue register pressure、SwiGLU/dSwiGLU 指令、FP8 scale load/store、以及非 GEMM kernel。fresh MFU 模型拟合的 `eta_max=0.5416` 表明即使忽略大量固定 overhead，该 shape 家族的 GEMM 有效上限也约 54% peak。

### Q3：FP8 为什么只比当前 BF16 快 1.11x？不是 tensor core 2x 吗？

**答：** 2x 是理论 peak 比。当前 BF16 baseline 是 QuACK/CuTe BF16 path，不是旧 cuBLAS BF16；它本身非常强，Ernie shape `2942.5µs`、BF16 MFU `84%` vs BF16 peak。FP8 虽有更高 peak，但要支付 activation/weight quant、scale gather/pack、FP8 cache 管理、epilogue 和 metadata overhead。小 T 下 FP8 甚至慢；T=8192 才到 `1.11x`。相对历史 cuBLAS BF16 baseline 才有 `~1.37x` 量级。

### Q4：zero-materialization 是不是只是把 gather 藏到 GEMM 里，访存还是一样？

**答：** 不是。读取 `x` 的总数据量类似，但避免了 `x_gathered(TK,H)` 的 HBM 写入和后续 GEMM 再读。也就是说传统路径是 read `x` -> write gathered -> read gathered；zero-mat 是 GEMM mainloop 按 `A_idx` 直接 read `x`。Ernie shape 对 BF16 gathered activation 可省 `384 MiB` materialization，且减少一个 kernel launch 和 allocator 压力。

### Q5：`A_idx` 不连续会不会破坏 coalescing，导致得不偿失？

**答：** 有代价，但 MoE expert-sorted 后同一 expert segment 内 token id 不一定连续。SonicMoE 用 FP8 + TMA/CuTe tile 化读 A，并只对 data zero-materialize；scale 预先 gather 成 TK layout 来保证 SFA TMA 正确。实测 fwd gated NCU L2 hit `88.8%`，DRAM `10.5%`，说明 fwd1 不是被 A_idx 随机访存主导，而是 epilogue/register bound。

### Q6：为什么 scale 必须 gather？data 都能 `A_idx`，scale 不能也 `A_idx`？

**答：** Blackwell blockscaled GEMM 的 SFA scale layout 和 GEMM M/K tile 坐标绑定。data 的 A rows 可以在 mainloop 用 `A_idx` 间接寻址，但 SFA layout 原生按 GEMM M rows (`TK`) 解释；如果直接用 T-space scale，`cu_seqlens_m` offset 对 expert 1+ 会越界/错位。当前做法只 gather scales，体积约 data 的几个百分点，比 gather FP8/BF16 activation 便宜得多。

### Q7：TMA reduce-add 的 atomic order 会不会导致非确定性？

**答：** 这是必须测试的风险。当前 frontier determinism test 对 Ernie shape 和小 aligned shape 做多次 fwd/bwd bit-exact 断言，覆盖 TMA reduce-add、cache 和 route padding。实测当前路径 bit-exact。原因包括 tile schedule 确定、同一输入和同一 launch 顺序下 TMA add 顺序稳定；但这不是数学必然，所以测试必须 hard-fail，不能只看 cosine。

### Q8：为什么 `node.step()` 必须在 `optimizer.step()` 前？

**答：** backward 中 CUTLASS wgrad 直接累积进 per-instance fp32 native view：`w1 [E,2I,H]`、`w2 [E,H,I]`。ERNIE optimizer 读取的是每个 expert 的 `weight.main_grad`，期望 layout 分别是 `[E,H,2I]`、`[E,I,H]`。`node.step()` 做 in-place native -> ERNIE layout flush。如果先 optimizer.step()，优化器读到 layout 错误的梯度。

### Q9：为什么不每次 backward 直接输出 ERNIE layout，省掉 `node.step()`？

**答：** 因为 GEMM native output layout 与 CUTLASS 高效 store / TMA reduce-add 更匹配。强行让 GEMM 直接写 ERNIE layout 会引入非连续 stride、额外 permute 或复杂 epilogue store，破坏 hot path。把 layout conversion 延迟到 microbatch accumulation 后、optimizer 前，只付一次成本，且不会影响多 microbatch 累加。

### Q10：FP8 weight cache 为什么安全？optimizer 更新后不会用旧权重吗？

**答：** cache key 包含 `data_ptr` 和 tensor inplace version（Paddle/PyTorch 兼容处理）。optimizer in-place update 会 bump version，下一步 cache miss 并重建 FP8 weight。`node.step()` 不需要手动清 cache；只有内存压力或参数 swap-out 时才调用 `invalidate_caches()`。

### Q11：为什么 compile key 不能含 `TK`？

**答：** 生产训练 seqlen 和 routing distribution 会变化。如果 CuTe compile key 含 `TK/total_M/capacity`，每个新 token count 都会触发几十秒级 JIT 编译。当前通过 `mark_layout_dynamic` 让动态维 runtime 处理，compile key 只含 `H/I/E/dtype/tile` 等静态模型维度；runtime fast path cache 可以含 exact shape，但有 64-entry 上限。

### Q12：`GemmDGated` 为什么不能继续融合 dz quant？省掉 103µs 不香吗？

**答：** 不应只看 103µs standalone quant。NCU 显示 `GemmDGatedFP8CLoadSm100ZeroMat` 已 `168 regs/thread`，384 threads/block 下占 `64512/65536` regs，几乎没有寄存器余量。dz quant 需要 amax reduction、E8M0 encode、FP8 cast、scale store，会增加循环和 live values，可能 spill、降 occupancy 或 crash。历史 compute-sanitizer 还会 mask 这类 crash，不能作为可行性依据。更合理方向是 dgrad1 fission 或 C-load reuse。

### Q13：iso32 为什么不是偷精度？一个 32x32 amax 明显比 1x32 粗。

**答：** 理论上确实更粗，所以它不是无条件正确。当前依据是针对真实 Ernie `dz` 捕获数据做下游 GEMM RRMSE 审计，iso32 vs 1x32 ratio=`1.000x`。原因是 E4M3 mantissa 噪声和 dz 分布下，额外动态范围损失没有反映到绑定指标。满分回答必须强调“measured scope 内 precision-free”，不能声称对所有分布严格等价。

### Q14：为什么 FP8 小 batch 会慢？

**答：** 有固定/近固定成本：JIT 之外的 launch/control、metadata、scale gather、weight cache lookup、quant kernels。T 小时 GEMM FLOPs 变少，但这些 overhead 不能等比例下降。fresh sweep 中 T=1024/2048 FP8 分别是 `0.95x/0.97x`，到 T=4096 才超过 BF16。

### Q15：nsys 和 ncu 数字冲突时信谁？

**答：** 看问题。端到端 busy time / MFU 信 nsys GPU-projection + NVTX BENCH 区间；单 kernel 资源瓶颈信 ncu SoL、tensor pipe、regs、L2/DRAM。不要直接比较 ncu duration 和 nsys duration，因为 ncu 默认 base clock 且 replay 多轮，nsys 通常 boost clock 和真实 timeline。

### Q16：`compute-sanitizer` 跑过就能证明 kernel 没问题吗？

**答：** 不能。HANDOFF 明确记录 GemmDGated epilogue 寄存器溢出/segfault 可能被 compute-sanitizer mask，因为 sanitizer 重编译/插桩改变 flags 和资源分配。sanitizer 适合查越界/非法访存，但性能资源边界和 race 仍要靠 ncu/nsys/确定性压力测试。

### Q17：route-level padding 会不会污染输出？

**答：** padding row 的 `x_gather_idx=0` 只是保证合法地址，关键是 `router score=0`，combine 时贡献为零；反向 score/activation 梯度也不应对真实 token 产生贡献。它避免了修改 `x` 或 append sentinel rows，开销只是额外 padded GEMM rows，通常远小于 full gather/pad tensor。

### Q18：Paddle proxy 最大的坑是什么？

**答：** 不要假设 PyTorch API 语义完全一致。例如 `torch.equal` 在 Paddle proxy 下返回 element-wise tensor，不是 bool；`stream.stream_base.raw_stream` 是 Paddle 独有而 PyTorch 用 `cuda_stream`；`tensor._inplace_version` 可能是方法；`storage_offset`/`_offset` 不同；dtype 比较要用字符串兼容。生产代码必须用 `hasattr/getattr/callable` 分支，而不是硬编码一边。

### Q19：如果专家问“下一步最高 ROI 是什么”，怎么答？

**答：** 基于最新事实，不再说“直接把 dz quant 融进 GemmDGated epilogue”。更稳妥回答是：dgrad1 是最低 tensor-pipe 利用（约 42%）且 L2 hit 只有约 61% 的单 kernel 瓶颈，优先研究 C-load reuse、persistent tile 或 fission；同时保持 iso32 quant 和 colwise quant 现状，因为它们已接近 bandwidth/ROI 边界。任何方案必须先过 determinism、precision、nsys GPU-projection，而不是只看 isolated kernel。

### Q20：如果要上线到多机预训练，最怕什么？

**答：** 最怕不是单 kernel 慢，而是系统性不可复现或集成契约错：`node.step()` 顺序错误、JIT cache 多 rank 竞态、Paddle launcher 误入多机 rendezvous、FP8 cache stale、route metadata contract 被自定义 dispatcher 破坏、ncu 锁频导致性能假回退。上线前必须跑 precision、determinism、stress、多层 PP、JIT concurrent、perf gate，并用 shared GPFS cache 预热。

### Q21：32x32 iso32 在什么数值情况下会真掉精度？常见吗？

**答：** 当 32 行之间动态范围差异很大时会掉精度。形式化说，若某行 row-wise amax 为 `a_r`，32x32 block amax 为 `a_b`，额外损失约 `log2(a_b/a_r)` bit。`a_b/a_r` 如果达到 4，就相当于小行少约 2 bit scale 精度，很多小梯度会被量化到更粗 bin 甚至 0。常见触发是单行 outlier、梯度爆炸、loss scale 不稳、极端 routing 让某些 expert 的 `dz` 分布长尾更重。当前 Ernie `dz` 捕获中这不常见，下游 GEMM RRMSE ratio=`1.000x`；但这只是 measured scope。生产上应监控 `log2(block_amax/row_amax)` 分位数和 zero ratio，超过阈值就 fallback 到 1x32 dual quant。

### Q22：TMA reduce-add 是不是提高了 main_grad 精度？

**答：** 不是。TMA reduce-add 是性能优化：把 `D=A@B+C` 的 C-load/beta epilogue 换成 TMA store-side ADD，降低寄存器和 smem staging。`main_grad` 仍是 fp32，精度上限仍由 FP8 输入误差、GEMM accumulator、fp32 加法顺序决定。它不会做 Kahan、pairwise 或更高精度累加。风险是 atomic/reduction order 造成 bit-level 非确定性，当前用 frontier determinism hard gate 兜住；另一个更实际的风险是 `node.step()`/clear_grad 合约错导致重复累加或 layout 错。

### Q23：MFU 模型是不是过拟合 Ernie 一个点？

**答：** 不是。`tools/mfu_model.py` 拟合 `fresh_benchmark_ws1/sweep.json` 的多点数据，FP8 有 11 个点，覆盖 `T=1024..16384`、`E=8/16/32`、`H/I` 变宽等维度，`R^2=0.99896`。Ernie 点只是代入 sanity check。模型里的 `eta_max` 是有效 GEMM shape efficiency，`a_quant` 是随 `TK*max(H,2I)` 增长的数据规模项，`a_expert` 是随 `E*TK` 增长的 expert fragmentation/routing 项，`c_fixed` 是固定开销。它能解释 scaling，但不能替代 nsys per-kernel breakdown；尤其不能把 `a_quant` 解读成单独量化 kernel 的真实耗时。

### Q24：GemmDGatedCLoad 的寄存器有没有进一步优化空间？

**答：** 有空间，但不是“删几行无用代码”那么简单。168 regs/thread 主要来自 MMA accumulator、FP8 C-load fragment、dequant 后 gate/up、dSwiGLU 输出、`y1s`、score scale/reduce 和 zero-mat metadata 同时 live。可尝试方向是缩短 live range、把 `ds` reduction 拆出、调整 epilogue tile、减少中间 fp32 fragment、或 fission 成 GEMM mainloop + dSwiGLU/quant/reduce 后处理。是否有效要看 NCU：如果有 local spill，先消 spill；如果没有 spill但 tensor pipe 低，重点是 dependency chain/C-load/issue mix，可能 fission 比继续融合更有效。

### Q25：推理能通信+专家计算 megakernel，训练为什么不能照搬？fusion/fission 怎么判？

**答：** 推理没有 backward activation 生命周期、wgrad、router score grad、optimizer `main_grad` layout、microbatch accumulation 和确定性训练合约，所以更容易把 dispatch/compute/combine 做成 persistent megakernel。训练可以做阶段化 megakernel，但整层 fwd+bwd+optimizer 一体化不现实。判据是比较省掉的 HBM/launch/allocator 时间和新增的 register/occupancy/lost-parallelism/debug 风险。fusion 的必要条件是中间 tensor 大、tile 相容、不会 register cliff、不会破坏 reduction order；fission 的必要条件是当前 fused kernel 被 regs/smem/dependency 卡住，拆开后 occupancy/tensor pipe 明显上升，且额外 HBM/grid sync 可被 L2 locality、FP8 packed 中间或更好 tile shape 抵消。当前 `GemmDGatedCLoad` 就是典型 fission 候选。

---

## 12. 最小验证清单

快速 sanity：

```bash
source .runenv.sh
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/fp8_frontier_determinism_test.py -v
```

压力：

```bash
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
  python -m pytest tests/fp8_frontier_stress_test.py -v
```

完整回归：

```bash
bash tests/run_regression.sh
```

性能 gate 需要看 nsys GPU-projection，不要用 Python wall time 或普通 CUDA events 当 gold standard。

---

## 13. 新人学习路线

1. 先手算 Ernie shape 的 `TK`、FLOPs、ideal time、MFU，确保 roofline 口径清楚。
2. 画出 token-major 和 expert-sorted 两个索引空间，能解释 `x_gather_idx/s_reverse_scatter_idx/router_scores`。
3. 读 `_SonicMoEDeepEPFunc.forward/backward`，只跟踪 tensor shape，不纠结每个 helper。
4. 读 `gemm_sm100_fp8_zeromat.py` 顶部注释，理解为什么 data 不 gather 但 scale 要 gather。
5. 读 `GemmGatedBlockscaledQuantMixin`，理解 epilogue quant 的 register 位置。
6. 读 `GemmDGatedFP8CLoadMixin`，理解 z_fp8 viewed-as-Int16 的 trick。
7. 读 `iso32_dual_quantize_varlen`，理解单 FP8 buffer + 双 scale layout。
8. 读 `_run_cutlass_blockscaled_gemm_varlen_k_tma_add`，理解 wgrad 为什么能直接加到 fp32 accumulator。
9. 对照 `reports/ernie_shape_ncu_s78b/README.md`，解释每个 GEMM 的瓶颈。
10. 最后再读 CI 和 HANDOFF pitfalls，建立“性能优化不能破坏确定性/集成契约”的底线。

---

## 14. 一页速记

```text
Production shape:
  T=8192, H=3072, I=1536, E=8, K=8, TK=65536

Latest performance:
  FP8 frontier busy = 2659.8 µs
  MFU = 46.51%
  measured peak MFU = 51.61%

FLOPs:
  18*TK*H*I = 5.566e12
  ideal @4500 TFLOPS = 1237 µs

Big kernels:
  wgrad GEMMs       1185 µs
  fwd gated          441 µs
  dgated             400 µs
  colwise quant      242 µs
  token reduce       148 µs
  iso32 dual quant   103 µs

Core wins:
  A_idx zero-materialization
  GEMM+SwiGLU+FP8 quant epilogue
  z_fp8 save + FP8 C-load dGated
  iso32 dual quant
  TMA reduce-add wgrad
  dynamic CuTe compile key

Core risks:
  GemmDGated has 168 regs/thread: do not blindly add epilogue loops
  FP8 small T can be slower
  node.step() before optimizer.step()
  cache key must include inplace version
  nsys for end-to-end, ncu for kernel resources
  determinism test is hard gate
```
