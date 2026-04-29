# Directory Index: `/tools/`

> Developer tooling for profiling, benchmarking, validation, orchestration, and audits.
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
| `ci/` | CI runners and gates (`run_core_tests.sh`, `perf_gate.py`, `multicard_smoke.py`). | S76 |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `_inline_audit.py` | Inline audit: monkey-patch backward to print tensor inventory at dgated completion. | — |
| `cluster_idle_launch.py` | Developer utility for cluster idle launch. | — |
| `fp8_frontier_breakdown.py` | FP8 Frontier rigorous breakdown: memory, precision, performance. | — |
| `generate_directory_indexes.py` | Generate human- and agent-friendly INDEX.md files for stable directories. | untracked in git |
| `introspect.py` | Main profiling harness for nsys, precision, grid, and memory experiments. | — |
| `ncu_profile_colwise.py` | NCU profiling script specifically for colwise_quantize_and_pack. | — |
| `ncu_profile_kernels.py` | ncu profiling for individual FP8 kernels. | — |
| `ncu_profile_quant.py` | NCU profiling script for CuTe DSL vs Triton blockscaled FP8 quant kernels. | — |
| `ncu_profile_wgrad_quant.py` | NCU profiling script for wgrad colwise/fused-transpose quant kernels. | — |
| `nsys_benchmark.py` | nsys-compatible benchmark: BF16 vs FP8 with NVTX markers. | — |
| `parse_nsys_per_iter.py` | Generic nsys-sqlite GPU-projection per-iter parser; reads NVTX-bracketed ranges from a `.sqlite` and reports per-iter wall + per-kernel breakdown. | Session 67 |
| `nsys_session42_v2.sh` | Shell helper for nsys session42 v2. | — |
| `precision_audit.py` | Quick end-to-end precision audit: FP8 vs BF16 per-tensor comparison. | — |
| `rigorous_benchmark_s42.py` | Rigorous benchmark: 3 modes × 3 seeds × 3 repeats, subprocess-isolated. | — |
| `rigorous_profiler.py` | Rigorous GPU-projection kernel profiling + fine-grained memory lifecycle. | — |
| `run_blackwell_test_shards.py` | Developer utility for run blackwell test shards. | — |
| `run_frontier_perf_sweep.py` | Multi-shape multi-GPU FP8 frontier perf sweep driver (subprocess-per-shape, sqlite GPU-projection extraction). | Session 66 |
| `run_regression.sh` | Shell helper for run regression. | — |
| `scoreboard.py` | SonicMoE Buffer Scoreboard — Operator-buffer dependency & phase-state analysis. | — |
| `session42_benchmark.py` | Session 42 Complete Benchmark: Performance + Memory + Precision. | — |
| `session45_bench.py` | Subprocess-isolated BF16/FP8 benchmark with CUDA events. | — |
| `verify_memory_optimization.py` | Verify memory optimization: compare FP8 vs BF16 peak memory. | — |
| `warp-control-plane.sh` | Shell helper for warp control plane. | — |
| `wgrad_fp8_benchmark.py` | FP8 wgrad full-replacement benchmark. | — |
