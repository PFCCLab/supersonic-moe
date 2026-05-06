# WS5: Phase 1A 实施状态 — 历史设计草案，已被后续 NCU 结论 supersede

> 日期: 2026-05-06 | 基于 fresh benchmark data
> 状态: **不要直接执行本草案**。此文保留为历史设计记录；当前权威状态见根目录 `HANDOFF.md` 和 `reports/sonic_moe_fp8_frontier_newcomer_guide.md`。

## 2026-05-06 结论修正

后续 NCU 读数证明 `GemmDGatedFP8CLoadSm100ZeroMat` 已经处在寄存器悬崖：

- 168 regs/thread × 384 threads/block = 64512/65536 regs，几乎没有余量。
- Tensor pipe 约 42%，L2 hit 约 61%，瓶颈来自 FP8 C-load + dSwiGLU epilogue live range / register pressure。
- 直接把 dz quant/scale store 塞进 GemmDGated epilogue 风险很高：spill、占用率下降、甚至 segfault；`compute-sanitizer` 可能 mask 这类 crash。
- 因此本文原先的“下一 session 直接开始 Phase 1A epilogue quant fusion”不是当前 P0。

当前 P0 是结构性 dgrad1 优化，而不是直接 epilogue 加环：

1. C-load/L2 reuse 与 live-range reduction；
2. 将 dz quant 与 dgrad1 做受控 fission，比较 saved HBM vs occupancy/tensor-pipe gain；
3. 或重新设计更小寄存器占用的 stage-wise persistent pipeline。

下面的 blueprint 只能作为“曾经考虑过的设计”阅读，不能作为下一任 agent 的执行计划。

## 预期收益 (基于本次 fresh data)

| 指标 | 当前 | Phase 1A 后 (预期) | Δ |
|------|:---:|:---:|:---:|
| busy_us (Ernie) | 2660 | ~2460 | −200 µs |
| MFU | 46.5% | ~50.3% | **+3.8 pp** |
| 显存 (dz) | 384 MiB bf16 | 192 MiB fp8 + 6 MiB scales | **−186 MiB** |

## 为什么本 session 不实现

1. CuTe DSL epilogue 修改是多日工作（HANDOFF S80c 明确记录）
2. 需要 `cuobjdump --dump-resource-usage` 验证寄存器压力
3. 需要多轮 kernel compile → test → profile 迭代
4. 错误实现会破坏 frontier（determinism test + 精度 test 全部回归）

## Phase 1A 实现 blueprint (ready to execute)

### 修改 1: `sonicmoe/quack_utils/gemm_dgated.py` (line 711 之后)

新增 mixin class:

```python
class GemmDGatedFP8CLoadBlockscaledQuantMixin(GemmDGatedFP8CLoadMixin):
    """GemmDGated + epilogue 1×32 blockscaled FP8 quant of dz.

    Instead of writing bf16 dz to D, quantizes in-register and writes fp8 bytes.
    Template: GemmGatedBlockscaledQuantMixin (gemm_gated.py:221-295).
    """
    _epi_ops = (
        *GemmDGatedFP8CLoadMixin._epi_ops,
        BlockscaledScaleStore("mDZScale"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Same as parent, plus mDZScale
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        implicit_dtype: cutlass.Constexpr[type] = cutlass.BFloat16
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = 0
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mFP8PreAct_fp8: Optional[cute.Tensor] = None
        mFP8PreAct_scales: Optional[cute.Tensor] = None
        mDZScale: Optional[cute.Tensor] = None  # NEW: (TK, 2I//32) uint8

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        # Run parent: loads C(fp8), dequants, dSwiGLU, writes dz to tRS_rD
        tRS_rOut = GemmDGatedFP8CLoadMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )

        # NEW: if mDZScale is active, quantize tRS_rD (which now holds dz in packed bf16/f32)
        _dz_scale_info = epi_loop_tensors["mDZScale"]
        if const_expr(_dz_scale_info is not None):
            # tRS_rD contains packed f16x2 as f32 → reinterpret as 2N bf16 values
            # Need to work with the f32x2 representation (tRS_rdXY_f32x2 from parent)
            # KEY CHALLENGE: parent already wrote to tRS_rD, so we need the f32 dz values
            # SOLUTION: capture tRS_rdXY_f32x2 BEFORE the pack step
            #   → This requires restructuring: call parent's logic inline, not as super()
            #   → Or: read back from tRS_rD (unpack bf16→f32, apply quant)

            num_d = cute.size(tRS_rD)  # = N/2 (packed pairs)

            # Amax over all dz elements (1×32 per subtile)
            amax = Float32(0.0)
            for i in cutlass.range(num_d, unroll_full=True):
                val = tRS_rD[i]  # packed f16x2 as f32 — NOT the raw f32!
                # Need to unpack... this is the challenge
                # Actually: tRS_rD at this point holds the PACKED bf16 result
                # We need the PRE-PACK f32 values
                pass

            # ALTERNATIVE APPROACH (cleaner):
            # Don't call super(). Instead, COPY the parent's logic inline up to
            # the pack step, then insert quant BEFORE pack, then either:
            #   (a) write fp8 to D (change d_dtype to fp8) — requires D tensor change
            #   (b) write fp8 to a SEPARATE output tensor via a new EpiOp
            # Option (b) is safer but needs a new EpiOp for fp8 scatter store.

        return tRS_rOut


class GemmDGatedFP8CLoadBlockscaledQuantSm100(GemmDGatedFP8CLoadBlockscaledQuantMixin, GemmSm100):
    pass
```

### 核心技术挑战 (需要在实现时解决)

1. **tRS_rD 的时序问题**: parent 的 `epi_visit_subtile` 将 dz 从 f32 pack 成 bf16 再写入 tRS_rD。
   quant 必须在 pack 之前拿到 f32 值。
   → 解决方案: **inline parent logic**，在 pack 步骤前插入 quant。

2. **D tensor dtype**: 当前 D 是 f32 (packed bf16)。如果改为 fp8，需要改变整个 D tensor 的 dtype 传递链。
   → 解决方案: 保持 D 为 bf16 (作为 "side output")，通过新的 EpiOp 将 fp8 数据写到一个独立的 output tensor。
   → 或者: 直接将 D dtype 改为 uint8，在 functional/__init__.py 分配 fp8 buffer 传入。

3. **Scale 存储**: `BlockscaledScaleStore` 已经处理了坐标计算和 bounds check。直接复用。

4. **寄存器压力**: 添加 amax(1) + e8m0 vars(~8) + quant_scale(1) ≈ 10 个 f32/i32 regs。
   相比 kernel 现有的 ~180 regs，增加 ~5.5%。可能触发 spill。
   → 验证: 编译后用 `cuobjdump --dump-resource-usage` 检查 regs_per_thread。

### 修改 2: `sonicmoe/functional/__init__.py` (line 1967)

```python
# When Phase 1A is active:
use_dz_epi_fuse = os.environ.get("SONIC_MOE_DZ_EPI_FUSE", "0") == "1"

if use_dz_epi_fuse and use_fp8_preact:
    # Allocate fp8 output instead of bf16
    dz_fp8 = torch.empty((total_m, n * 2), dtype=torch.float8_e4m3fn, device=dout.device)
    dz_scales = torch.empty((total_m, (n * 2 + 31) // 32), dtype=torch.uint8, device=dout.device)

    # Call the new kernel variant
    gemm_dgated_kernel_with_quant(
        dout_fp8, w2_fp8_enk,
        dz_fp8,  # fp8 output!
        y1s,
        None, "swiglu", ...,
        colvec_scale=s_float,
        colvec_reduce=colvec_reduce_partial,
        mDZScale=dz_scales,
        ...
    )

    # Skip external dual-quant for dz (only do dout colwise)
    dz_packed_scales = pack_blockscaled_1x32_scales(dz_scales, total_m, n*2)
    # ... wire dz_fp8 + dz_packed_scales to downstream consumers
else:
    # Current path: bf16 dz + external Triton quant
    dz = torch.empty((total_m, n * 2), dtype=torch.bfloat16, device=dout.device)
    ...
```

### 修改 3: Unit test

`tests/ops/test_gemm_dgated_epi_quant.py`:
- 调用新 kernel 变体
- 比较 fp8 输出与 reference (bf16 → Triton quant → fp8)
- 验证 scale bytes 与 reference 的 bit-exact 匹配
- 验证下游 GEMM 的 RRMSE ratio ≤ 1.001

### 时间估计

| 步骤 | 预估 | 依赖 |
|------|:---:|------|
| Inline parent logic + insert quant | 2-4h | 理解 tRS_rD layout |
| D dtype change 或 separate EpiOp | 2-4h | 选方案 |
| Compile + fix errors | 1-2h | Hardware |
| Register pressure verification | 1h | cuobjdump |
| Unit test | 1-2h | 编译通过 |
| E2E precision test | 1h | Unit test 通过 |
| nsys A/B comparison | 1h | 精度通过 |
| **Total** | **~10-16h** | |

## 历史建议（已 supersede）

原始建议是下一 session 直接开始 Phase 1A 实现。该建议已被上面的 NCU 结论推翻。若未来重新尝试，只能在先证明 regs/thread、occupancy、tensor-pipe 不退化的前提下推进。历史关键决策点:
1. 先选 D-dtype 方案（改 D 为 fp8 vs 独立 EpiOp）
2. 编译后立即检查 regs_per_thread（如果 > 200，考虑 fallback）
3. 如果寄存器超标，降级为"只消除 bf16 write，但仍需 external fp8 cast"（partial gain）
