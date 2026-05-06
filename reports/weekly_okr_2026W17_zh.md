# SuperSonic-MoE OKR周报 — 2026-W17（4.23–4.30）

> **目标（O）**：结合文心实际需求，攻坚前沿高性能策略的探索创新、实现落地
> **关键结果（KR）**：【SuperSonic-MoE落地】SuperSonic-MoE的FP8持续开发、优化和迭代，根据文心实际训练中的并行策略进行定向优化，确保交付的MoE模块领先于原有模块40%以上，且保证deterministic和收敛精度

---

## 一、本周核心进展

### 1. FP8性能攻坚：文心生产Shape MFU突破45%，峰值算力超2000 TFLOPS

| 指标 | 本周结果 | 说明 |
|------|---------|------|
| **Ernie生产Shape（T8192/E8/K8）** | **MFU 44.91%**（2021 TFLOPS） | 在B30Z上达成，超越基准模块40%目标已兑现 |
| **最佳Shape（T8192-H6144）** | **MFU 50.88%** | 宽hidden场景性能更优，为未来Ernie升级预留空间 |
| **量化算子效率** | **90–93% HBM峰值带宽** | NCU驱动调优，消除显存瓶颈 |
| **Wgrad路径** | **1.14–1.43× vs BF16** | 生产默认`fp8_wgrad=False`路径已稳定，TMA reduce-add融合epilogue节省664 µs/iter |

- **TMA融合epilogue**：将BF16 wgrad累加融合进GEMM epilogue，寄存器占用86→50，E2E提升2–4%。
- **MFU全量扫描**：完成11个Shape的FP8前沿MFU Sweep，量化路由开销（Doubling E成本~2.3 pp MFU），为并行策略调参提供数据基线。

### 2. 确定性&收敛精度：Bit-Exact Determinism固化到CI

- **FP8前沿路径实现字节级确定性**：`tests/fp8_frontier_determinism_test.py` 新增为小对齐+Ernie生产Shape双测例，三次独立运行输出`(out, dx, 全部梯度)`字节级一致。
- **CI硬门禁**：已将确定性测试接入`tests/run_regression.sh`，未通过即阻断合入，确保FP8优化不引入精度漂移。
- **冷启动对齐**：JIT缓存机制消除seqlen变化导致的重编译，冷启动E2E首步即达稳态数值，杜绝了JIT缓存污染。

### 3. 生产化适配：多卡并行&集群环境稳定性

- **多卡集群安全**：完成race-safe JIT + FP8配置隔离，适配文心实际训练中的多卡并行策略；`main_grad`惰性分配 + `step()`顺序修正，保障分布式场景稳定性。
- **CI基础设施**：Triton autotune缓存持久化，大幅降低冷启动JIT耗时；nsys/ncu基准基线固化，6个Ernie Shape GEMM已建立`--set full`级性能档案。
- **硬件基线确认**：完成B30Z硬件身份审计（148 SMs / 2032 MHz boost / 1100W VBIOS / 268 GiB HBM3e），4500 TFLOPS峰值经empirical锚定，MoE基准运行在全频boost状态，功耗未触墙。

---

## 二、本周交付物

| 类别 | 数量 | 说明 |
|------|------|------|
| **合入上游PR** | 8个（#10–#13, #15–#18） | 性能优化、确定性加固、CI增强 |
| **新增commits** | 43 commits / 260文件 / +19,029 −4,696 LOC | 含S79/S79b队列中的4个确定性+MFU审计commits |
| **核心工具/报告** | `tools/mfu_sweep_s79.py`、`reports/mfu_s79/`、`reports/ernie_shape_ncu_s78b/` | MFU扫描工具及可视化、ncu全量profile |
| **测试门禁** | `fp8_frontier_determinism_test.py` | FP8 bit-exact确定性硬门禁 |

---

## 三、关键认知与下一步杠杆点

1. **MFU已达~45%，剩余收益在非矩阵开销**：构成FP8 GEMM单算子已≥80%峰值，下一高杠杆点为**反向FP8 cast融合进wgrad producer**，预计+5 pp MFU。
2. **单核优化已穷尽**：`GemmDGatedFP8CLoadSm100ZeroMat`的寄存器/scale/epilogue已全部审计，进一步收益需多kernel重构。
3. **多卡/集群适配就绪**：当前代码路径已适配文心实际训练并行策略，具备生产部署条件。

---

## 四、下周计划

| 优先级 | 事项 | 预期收益 |
|--------|------|----------|
| P0 | 将S79/S79b 4个commits提PR合入上游 | 确定性门禁+MFU审计报告入库 |
| P1 | 反向FP8 cast → wgrad producer融合 | **+5 pp MFU**，冲刺50%+ Ernie生产Shape |
| P2 | 验证H=6144宽hidden生产数据通路 | 若文心未来升级，可直接兑现+6 pp MFU |
| P3 | 探索`GemmDGatedFP8CLoadSm100ZeroMat`多kernel重构 | 突破单核天花板 |

---

*报告周期：2026-04-23 至 2026-04-30*
*分支：`race-fix-paddle`（fork）→ 上游 `PFCCLab/supersonic-moe`*
