# Directory Index: `/tools/`

> Developer tooling for profiling, benchmarking, validation, orchestration, and audits.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Stable child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `ci/` | Directory for ci. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `_inline_audit.py` | Inline audit: monkey-patch backward to print tensor inventory at dgated completion. | — |
| `audit_dz_iso32_quality.py` | Phase 0.2: Audit iso32 vs 1×32 dz quantization on REAL gradient tensors. | — |
| `bench_sweep_fresh.py` | Fresh benchmark sweep: FP8 frontier + BF16 baseline (8-GPU parallel). | — |
| `cluster_idle_launch.py` | Developer utility for cluster idle launch. | — |
| `cross_framework_bench.py` | Cross-Framework MoE Benchmark — 4-way: Paddle BF16 / Paddle FP8 / SonicMoE BF16 / SonicMoE FP8. | — |
| `dump_real_dz.py` | Dump real dz tensors from the SonicMoE bwd dGated path for iso32 audit. | — |
| `fp8_frontier_breakdown.py` | FP8 Frontier rigorous breakdown: memory, precision, performance. | — |
| `generate_directory_indexes.py` | Generate human- and agent-friendly INDEX.md files for stable directories. | — |
| `introspect.py` | Main profiling harness for nsys, precision, grid, and memory experiments. | — |
| `mfu_model.py` | MFU theory model: fit a parametric model to real benchmark data. | — |
| `mfu_sweep_s79.py` | S79 MFU sweep — drive bench_mlpnode_topk_nsys for many shapes, compute MFU. | — |
| `ncu_profile_colwise.py` | NCU profiling script specifically for colwise_quantize_and_pack. | — |
| `ncu_profile_kernels.py` | ncu profiling for individual FP8 kernels. | — |
| `nsys_pad_compare.py` | 4-way nsys timeline: bf16_raw, bf16_rounding, fp8_rounding, fp8_padding. | — |
| `paddle_compat_bench.py` | Paddle Compat Benchmark — SonicMoE under Paddle enable_compat(). | — |
| `paddle_compat_parallel.py` | 27-Shape Grid Paddle Compat Benchmark — 8-GPU parallel. | — |
| `paddle_compat_smoke.sh` | Shell helper: ═══════════════════════════════════════════════════════════════════════════════. | — |
| `parse_nsys_per_iter.py` | Parse nsys sqlite, compute per-iter GPU-projection µs for each BENCH range. | — |
| `rigorous_benchmark_s42.py` | Rigorous benchmark: 3 modes × 3 seeds × 3 repeats, subprocess-isolated. | — |
| `rigorous_profiler.py` | Rigorous GPU-projection kernel profiling + fine-grained memory lifecycle. | — |
| `run_blackwell_test_shards.py` | Developer utility for run blackwell test shards. | — |
| `run_frontier_perf_sweep.py` | Sweep nsys profiles for SonicMoE FP8 frontier MlpNode (Paddle compat path). | — |
| `run_regression.sh` | Shell helper for run regression. | — |
| `scoreboard.py` | SonicMoE Buffer Scoreboard — Operator-buffer dependency & phase-state analysis. | — |
| `wgrad_fp8_benchmark.py` | FP8 wgrad full-replacement benchmark. | — |
