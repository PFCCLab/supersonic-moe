# Directory Index: `/tests/ops/`

> Focused operator and module-level tests, including the newer MoE module suite.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package marker for test discovery. | — |
| `audit_iso32_numerics.py` | Iso32 vs 1×32 weight blockscaled-FP8 quant — rigorous numerics audit. | — |
| `bench_coldstart_nsys.py` | Cold-start nsys benchmark: clears all caches, simulates first-time execution. | — |
| `bench_deepep_topk_nsys.py` | nsys GPU-projection benchmark for SonicMoEMlpNode E2E. | — |
| `bench_frontier_perf.py` | Session 62 FP8 frontier comprehensive benchmark. | — |
| `bench_gemm_dynamic_ab.py` | A/B experiment: compare per-call GEMM latency with/without mark_layout_dynamic. | — |
| `bench_iso32_quant_nsys.py` | Perf microbench: iso32 vs 1×32 weight blockscaled-FP8 quant kernel. | — |
| `bench_mlpnode_mem.py` | SonicMoEMlpNode 单次前反向显存基准 用法: CUDA_VISIBLE_DEVICES=0 python tests/ops/bench_mlpnode_mem.py 配置（默认值对应 ERNIE 真实业务规格）: H=3072 I=1536 K=8 E_LOCAL=8 EP_SIZE=32 SEQ_LEN=16384 精度策略： - 前向/…. | — |
| `bench_mlpnode_topk_nsys.py` | nsys GPU-projection benchmark for SonicMoEMlpNode topk path. | — |
| `bench_static_vs_dynamic_gemm.py` | A/B experiment: dynamic vs static compile_key for CuTe GEMM. | — |
| `bench_user_shape_fwd_nsys.py` | Session 69 — reproduce user shape & profile sonic-meta routing region. | — |
| `bench_wgrad_epilogue.py` | A/B benchmark: TMA reduce-add vs fused beta=1.0 for wgrad GEMM epilogue. | — |
| `conftest.py` | Shared fixtures, precision helpers, gold references, and shape constants for FP8 op tests. | — |
| `mlpnode_nsys_worker.py` | Minimal MlpNode worker for nsys profiling. | — |
| `ncu_deepep_topk.py` | Isolated NCU profiling harness for deepep_topk_metadata CUDA kernels. | — |
| `test_argsort_sync.py` | Minimal reproducer: Paddle argsort 1D path triggers cudaStreamSynchronize. | — |
| `test_cold_start_e2e.py` | Production cold-start E2E: cache-clear → warmup → multi-shape precision + perf. | — |
| `test_colwise_quant.py` | Unit tests for colwise_quantize_and_pack and colwise_quantize_cute. | — |
| `test_deepep_metadata_perf.py` | Test and benchmark deepep_metadata: CUDA kernel vs Python fallback. | — |
| `test_deepep_topk_metadata.py` | Tests for deepep_topk_to_sonic_metadata: real DeepEP topk dispatch conversion. | — |
| `test_dequant.py` | Unit tests for dequantize_blockscaled_fp8. | — |
| `test_dual_quant.py` | Unit tests for dual_quantize_varlen (fused row+col quant). | — |
| `test_e2e_mlpnode.py` | End-to-end SonicMoEMlpNode benchmark simulating real DeepEP pre-training. | — |
| `test_fused_quant.py` | Correctness + performance test for fused_dual_colwise_quantize. | — |
| `test_fused_zy1_quant.py` | Unit tests for fused_z_save_y1_quant. | — |
| `test_gemm_dgated.py` | Unit tests for gemm_dgated (bwd): torch ↔ BF16 3-way cross-validation. | — |
| `test_gemm_gated.py` | Unit tests for gemm_gated (fwd): torch ↔ BF16 ↔ FP8 3-way cross-validation. | — |
| `test_import_smoke.py` | Coverage smoke test: import every public module under ``sonicmoe`` so that module-level decorators, dataclass declarations, and constant tables are exercised by the coverage gate. | — |
| `test_jit_concurrent_heterogeneous.py` | Pytest coverage for jit concurrent heterogeneous. | — |
| `test_jit_key_stability.py` | Pytest coverage for jit key stability. | — |
| `test_jit_optimization.py` | Comprehensive JIT optimization validation: correctness, JIT recompilation, GPU performance (nsys), and memory. | — |
| `test_mlpnode_audit.py` | Rigorous precision + performance + memory audit for SonicMoEMlpNode. | — |
| `test_mlpnode_breakdown.py` | Paranoid-level precision breakdown + GPU-projection performance audit. | — |
| `test_mlpnode_correctness_large.py` | Large-SEQ correctness audit for SonicMoEMlpNode FP8 frontier. | — |
| `test_mlpnode_extreme_shapes.py` | Extreme-shape stress tests for SonicMoEMlpNode (CI gating). | — |
| `test_mlpnode_multilayer.py` | Multi-layer correctness for SonicMoEMlpNode + flush_native_grads. | — |
| `test_mlpnode_precision.py` | Element-wise precision audit: FP8 MlpNode vs BF16 gold (output/dx/dw1/dw2). | — |
| `test_moe_general_routing_fp8.py` | FP8 frontier unit-test for moe_general_routing_inputs. | — |
| `test_moe_module.py` | MoE module-level regression suite against a pure-torch reference. | — |
| `test_pad_gradient_integrity.py` | Axiomatic backward-correctness test for route-level padding. | — |
| `test_pad_routing.py` | Axiomatic correctness test for route-level padding. | — |
| `test_precompute_weight_fp8_warmup.py` | Test precompute_weight_fp8_warmup is bit-exact vs. | — |
| `test_recompute_z.py` | Focused validation of the ``recompute_z`` mode for SonicMoEMlpNode. | — |
| `test_recompute_z_optionB.py` | Bit-exact validation for recompute_z Option B. | — |
| `test_rowwise_quant.py` | Unit tests for quantize_and_pack_activation (row-wise blockscaled FP8 quant). | — |
| `test_swiglu.py` | Unit tests for SwiGLU forward/backward: torch ↔ BF16 ↔ FP8 3-way cross-validation. | — |
| `test_varlen_gemm.py` | Unit tests for blockscaled_fp8_gemm_varlen (down-projection): torch ↔ BF16 ↔ FP8 3-way. | — |
| `test_weight_quant.py` | Unit tests for quantize_and_pack_weight_iso32 (32x32 isotropic blockscaled). | — |
| `test_wgrad_gemm.py` | Unit tests for blockscaled_fp8_weight_grad_gemm: torch ↔ BF16 ↔ FP8 3-way. | — |
