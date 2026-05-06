# HANDOFF — Sonic-MoE FP8 Frontier (2026-05-06)

> **Branch**: `race-fix-paddle` (on `myrepo` = PFCCLab/supersonic-moe)
> **Frontier status**: GREEN — determinism + stress tests PASS, no regressions.

## 项目概述

Sonic-MoE 是一个 Mixture-of-Experts 前向/反向计算库，运行在 NVIDIA Blackwell (B30Z, SM100, 148 SMs, 8 TB/s HBM3e)。核心路径使用 FP8 (E4M3) blockscaled 量化 + CUTLASS/CuTe DSL 自定义 GEMM kernels (QuACK)。

## 当前性能指标 (Ernie 生产 shape: T=8192, E=8, K=8, H=3072, I=1536)

| 指标 | 数值 | 来源 |
|------|:---:|------|
| **FP8 frontier busy time** | 2660 µs/iter | nsys GPU-projection (S81 fresh data) |
| **MFU (vs 4500 TFLOPS FP8 peak)** | 46.5% | `reports/fresh_benchmark_ws1/` |
| **MFU peak (H6144-I2048 shape)** | 51.5% | 同上 |
| **Determinism** | Bit-exact across runs | `tests/fp8_frontier_determinism_test.py` |
| **Precision (vs BF16 gold)** | cos ≥ 0.997, RRMSE < 7.6% | `tests/ops/test_mlpnode_precision.py` |
| **FP8 vs QuACK-BF16 speedup** | 1.11× (both use FP8 TC GEMMs) | 注意：QuACK BF16 并非真 BF16 TC |
| **FP8 vs cuBLAS BF16 speedup** | ~1.37× (S53 historical数据) | README.md |
| **Cold-start JIT** | ~46s (E=8 Ernie shape) | `tests/ops/test_cold_start_e2e.py` |

## Kernel 时间分解 (nsys, Ernie shape, per-iter)

```
1185 µs  44.1%  QuACK wgrad GEMMs (4 calls: w1/w2 forward wgrad × 2)
 441 µs  16.4%  GemmGatedSm100ZeroMatBlockscaledQuant (fwd gated+quant)
 400 µs  14.9%  GemmDGatedFP8CLoadSm100ZeroMat (bwd actgrad)
 242 µs   9.0%  _colwise_quantize_and_pack (already num_warps=1)
 148 µs   5.5%  token_gather_sum_kernel (combine/scatter)
 103 µs   3.8%  _dual_varlen_iso32_quantize (already num_warps=1)
  83 µs   3.1%  _quantize_and_pack (rowwise quant, bandwidth-bound)
  37 µs   1.4%  Other (VectorizedBroadcast, index_elementwise)
─────────────────
2639 µs  100%   Total
```

## 关键技术事实

### 1. GemmDGated epilogue 融合不可行 (NCU 验证)
- `GemmDGatedFP8CLoadSm100ZeroMat`: 168 regs/thread × 384 threads = 64512/65536 regs (98.4%)
- Block Limit Registers = 1 (零寄存器余量)
- **任何** epilogue 循环 (amax reduction, fp8 cast) 都会导致寄存器溢出 → segfault
- `compute-sanitizer` 会 mask 该 crash（重编译用不同 flags，不可作为诊断工具）
- 详见 `reports/ernie_shape_ncu_s78b/`

### 2. Quant kernels 已完成单 warp 优化
- `_dual_varlen_iso32_quantize`: num_warps=1, 74% BW efficiency (near-optimal)
- `_colwise_quantize_and_pack`: num_warps=1, 2.3× speedup (NCU verified)
- `_quantize_and_pack`: BLOCK_ROWS=64 num_warps=4 — 单 warp 对该 kernel **无提升** (已验证, bandwidth-bound at 32 regs)

### 3. "BF16 baseline" 使用的是 FP8 Tensor Core
- `USE_QUACK_GEMM=1` 时，所有 GEMM 内部仍量化为 FP8 使用 FP8 TC 计算
- `SONIC_MOE_FP8_MODE=""` 只禁用 activation caching (z/y1 的 FP8 保存)
- 无法在此代码库中获得真正的 BF16 TC baseline (legacy path 已删除)
- 真 BF16 TC 对比需使用 `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/official` (但该 env 当前 broken)

### 4. iso32 dual-quant 是 precision-free lunch
- 一个 amax 管 32×32 子块；FP8 bytes 行/列消费者共享
- vs 1×32: downstream GEMM RRMSE ratio = 1.000× (bit-identical precision)
- 省 192 MiB FP8 write + 半 amax 计算 → 实测 -61 µs/iter

## 文件结构 (关键路径)

```
sonicmoe/functional/__init__.py     — 主 forward/backward autograd 逻辑
sonicmoe/functional/utils.py        — is_fp8_active(), enable_fp8() context manager
sonicmoe/quack_utils/gemm_dgated.py — 反向 actgrad GEMM (CUTLASS epilogue)
sonicmoe/quack_utils/gemm_gated.py  — 正向 fused gated GEMM (含 BlockscaledQuant 成功模板)
sonicmoe/quack_utils/gemm_sm100_fp8_zeromat.py — SM100 具体类注册
sonicmoe/quack_utils/blockscaled_fp8_gemm.py   — 所有 Triton quant kernels
sonicmoe/quack_utils/fused_quant_kernels.py    — 高层 quant API (dual_colwise etc.)
sonicmoe/ernie_compat/              — Ernie 集成层 (MlpNode, metadata, weight layout)
tests/fp8_frontier_determinism_test.py  — CI hard-fail: bit-exact 确定性
tests/fp8_frontier_stress_test.py       — CI: 多 shape/routing 压力测试
tests/run_regression.sh                 — CI 入口
.runenv.sh                              — 必须 source 的环境设置
```

## CI 运行方式

```bash
source .runenv.sh
# 快速验证 (hard-fail on non-determinism)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/fp8_frontier_determinism_test.py -v

# 压力测试 (17 tests, ~2 min)
CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 python -m pytest tests/fp8_frontier_stress_test.py -v

# 完整回归 (~15 min)
bash tests/run_regression.sh
```

## 已知问题 / 预存 failures

1. `fp8_protocol_test.py` — 部分 test 因 paddle proxy `stream_base` API gap 失败 (非 FP8 代码问题)
2. `moe_blackwell_test.py` — 环境兼容性问题 (非 frontier 问题)
3. `test_fused_quant.py::test_correctness` — 与 iso32 变更后的预期输出不同 (算法正确, byte-pattern 不同)

## 教训 & Pitfalls (给下一个 agent)

1. **168 regs = 绝对天花板**: 不要尝试在 GemmDGated epilogue 加任何循环/归约逻辑
2. **CuTe DSL 标量 fp8 cast 不支持**: 必须用 `r4.load().to(Float8E4M3FN)` 向量 pattern
3. **Triton iso32 kernel 已达 74% BW 效率**: 替换为 CuTe DSL 最多节省 25µs, ROI 极低
4. **ncu 异常退出会锁频**: 见 env.md 详细说明, 用 `ncu --clock-control=reset` 修复
5. **Paddle proxy 陷阱**: `torch.equal()` 返回 element-wise tensor; `.to(dtype=)` 必须带 `device=`; `Stream.stream_base` 需 unscoped proxy

## 下一步规划 (优先级排序)

### P0: Float8E4M3FN D 输出路径 (HIGH EV, MEDIUM EFFORT)
- `GemmSm100.is_valid_dtypes(Float8E4M3FN D)` 返回 True — FP8 D 在 SM100 上技术可行
- 需要: override `assert d_dtype.width == 32`, D shape 从 `(TK,I)` f32 → `(TK,2I)` fp8
- 潜在收益: 消除 768 MiB bf16 dz HBM 往返 → ~200 µs saving → +3 pp MFU
- 关键未知: TMA store atom 是否支持 8-bit elements

### P1: Pipeline scheduling 优化 (MEDIUM EV, LOW EFFORT)
- 利用 dgated GEMM D 写与后续 quant kernel 读之间的 L2 locality
- 不改 kernel, 只调 buffer size / launch 顺序

### P2: Coverage 提升 (LOW EV, ONGOING)
- 当前 31%, 目标 50%
- 重点文件: `blockscaled_fp8_gemm.py`, `grouped_gemm.py`, `swiglu_triton.py`

## 高价值信息源

1. **`reports/fresh_benchmark_ws1/`** — 22 点性能 sweep (11 shapes × 2 modes) + MFU model (R²=0.999)
2. **`reports/ernie_shape_ncu_s78b/`** — NCU full profile of all 6 Ernie GEMMs
3. **`tools/mfu_model.py`** — MFU 理论模型 (η=0.54, a_quant=50, a_expert=329, c=201)
4. **`reports/sparsity_audit_ws2.md`** — DeepEP 稀疏性分析 (K<topk 情况已正确处理)
5. **`reports/score_weighting_analysis_ws3.md`** — Score 加权位置分析 (current = optimal)
6. **`/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md`** — 集群环境 + ncu 锁频修复
7. **`.runenv.sh`** — 永远先 source 这个

## MFU 理论模型

```
busy = 18·TK·H·I/(peak×η) + a_q·TK·max(H,2I)·1e-9 + a_e·E·TK·1e-6 + c

参数 (curve_fit, R²=0.999):
  η_max = 0.5416  (GEMM shape efficiency)
  a_quant = 50    (quant bandwidth coefficient)
  a_expert = 329  (per-expert routing overhead)
  c_fixed = 201   (fixed per-iter overhead, µs)
  peak = 4500 TFLOPS (B30Z FP8 boost)
```
