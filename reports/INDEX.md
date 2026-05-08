# Directory Index: `/reports/`

> Collected benchmark outputs, summaries, and historical experiment artifacts.
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
| `ernie_shape_ncu_s78b/` | Directory for ernie shape ncu s78b. | — |
| `ernie_shape_nsys_s78/` | Directory for ernie shape nsys s78. | — |
| `fp8_upgrade/` | Historical FP8-upgrade notes; partly superseded by newer docs and reports. | `reports/fp8_upgrade/HANDOFF.md` is stale and explicitly superseded by root `HANDOFF.md`. |
| `fresh_benchmark_ws1/` | Directory for fresh benchmark ws1. | — |
| `grid_session53/` | Session 53 grid benchmark shards and consolidated JSON output. | — |
| `iso32_dz_audit/` | Directory for iso32 dz audit. | — |
| `jit_opt_validation/` | Directory for jit opt validation. | — |
| `mfu_s79/` | Directory for mfu s79. | — |
| `mfu_s80b/` | Directory for mfu s80b. | — |
| `mlpnode_nsys/` | Directory for mlpnode nsys. | — |
| `nsys_final/` | Final consolidated nsys-derived breakdowns for Session 53. | — |
| `nsys_xfw/` | Directory for nsys xfw. | — |
| `paddle_compat/` | Directory for paddle compat. | — |
| `paddle_compat_bench/` | Directory for paddle compat bench. | — |
| `paddle_compat_smoke/` | Directory for paddle compat smoke. | — |
| `perf_session62/` | Directory for perf session62. | — |
| `wgrad_tma_add_nsys/` | Directory for wgrad tma add nsys. | — |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `CONTRIBUTION_SUMMARY.md` | Markdown note: Contribution Summary: Blockscaled FP8 Training for SonicMoE on SM100. | — |
| `cross_framework_report.json` | JSON artifact with top-level keys: `T8192_H3072_I1536_E8_K8`. | — |
| `cross_framework_report.md` | Markdown note: Cross-Framework MoE Benchmark Report (4-Way). | — |
| `fp8_frontier_path_analysis.json` | Compiled BF16-vs-FP8 path-comparison report consumed by the new visualization module. | — |
| `frontier_perf_sweep_smoke.md` | Markdown note: SonicMoE FP8 Frontier (MlpNode) Performance Sweep. | untracked in git |
| `jit_opt_validation.json` | JSON artifact with top-level keys: `correctness`, `jit_recompile`, `memory`. | untracked in git |
| `mlpnode_bench.json` | JSON list artifact with 1 entries. | untracked in git |
| `nsys_breakdown_session64.json` | JSON artifact with top-level keys: `T8192_E8`, `T8192_E32`, `T16384_E8`, `T16384_E32`. | — |
| `nsys_grid_session64.json` | JSON artifact with top-level keys: `T8192_E8`, `T8192_E32`, `T16384_E8`, `T16384_E32`. | — |
| `pad_audit_results.json` | JSON artifact with top-level keys: `shape`, `precision`, `performance`, `memory`. | — |
| `phase1a_ws5_status.md` | Historical WS5 Phase 1A epilogue-fusion design draft; superseded by later NCU register-pressure findings. | historical; do not execute as current P0 |
| `quant_bench_final.json` | Legacy flat quant benchmark snapshot still emitted by `tools/introspect.py`. | legacy snapshot, compare with `reports/quant_bench.json` before reusing |
| `race_fix_paddle_migration.md` | Markdown note: Race-fix-paddle migration report. | — |
| `README.md` | High-level map of report outputs and profiling artifacts. | keep aligned with root `HANDOFF.md` |
| `score_weighting_analysis_ws3.md` | Markdown note: WS3: Score 加权位置精度分析. | — |
| `session53_breakdown.md` | Markdown note: Session 53 — Performance, Memory & Precision Breakdown (Final). | — |
| `session53_nsys_consolidated.json` | JSON artifact with top-level keys: `session`, `device`, `method`, `common`. | — |
| `session68_baseline.nsys-rep` | nsys report artifact for session68 baseline. | untracked in git |
| `session68_overhead_audit.md` | Markdown note: Session 68 — FP8 Frontier Per-Kernel Overhead Audit + recompute_z Wrap-up. | — |
| `session68_recompute.nsys-rep` | nsys report artifact for session68 recompute. | untracked in git |
| `session70_4tasks.md` | Markdown note: Session 70 — 4-Task Investigation Report. | — |
| `session72_frontier.nsys-rep` | nsys report artifact for session72 frontier. | — |
| `session73_quant_opt.nsys-rep` | nsys report artifact for session73 quant opt. | — |
| `sonic_moe_comprehensive_analysis.md` | Markdown note: SonicMoE 综合技术分析报告. | — |
| `sonic_moe_fp8_frontier_newcomer_guide.md` | Markdown note: SonicMoE FP8 Frontier 新人入门与专家问答指南. | — |
| `sparsity_audit_ws2.md` | Markdown note: WS2: SonicMoE 稀疏性假设审查报告. | — |
| `test_fp8_routing.nsys-rep` | nsys report artifact for test fp8 routing. | untracked in git |
| `weekly_okr_2026W17_zh.md` | Markdown note: SuperSonic-MoE OKR周报 — 2026-W17（4.23–4.30）. | — |
| `weekly_summary_2026W17.md` | Markdown note: Weekly Work Summary — 2026-W17 (Apr 23 → Apr 30). | — |
| `wgrad_bench.json` | Structured wgrad benchmark report under `reports/`. | — |
| `wgrad_fp8_benchmark_legacy.json` | Legacy full-replacement wgrad benchmark snapshot kept for history. | legacy benchmark snapshot, kept for historical comparison with `reports/wgrad_bench.json` |
| `wgrad_overhead_analysis.md` | Markdown note: Session 64: Wgrad GEMM Overhead Root Cause Analysis. | — |

## Redundancy and cleanup watchlist
- `reports/README.md` should stay aligned with root `HANDOFF.md` whenever the authoritative handoff changes.
- `quant_bench.json` and `quant_bench_final.json` look like structured-vs-legacy variants of the same benchmark family; verify the intended canonical file before adding new results.
- `wgrad_fp8_benchmark_legacy.json` is historical only; new wgrad report outputs should stay structured and live under `reports/`.
