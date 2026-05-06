# Directory Index: `/docs/`

> Canonical architecture, handoff, and design documentation.
> Regenerate with `python tools/generate_directory_indexes.py` from the repository root.

## Maintenance rules
- Before opening many files under this directory, read this `INDEX.md` first to narrow the search space.
- Any create / delete / rename / move in this directory must update the summaries in this `INDEX.md`.
- Any behavior-changing edit that invalidates a file summary must refresh the affected summary text here.
- If a change crosses directory boundaries, update this `INDEX.md` and the nearest affected ancestor `INDEX.md` files together.
- Prefer regenerating indexes with `python tools/generate_directory_indexes.py` after structural changes, then review the generated summaries.

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `cute_dsl_optimization_guide.md` | Markdown note: CuTe DSL FP8 Quantization — Architecture & Optimization Guide. | — |
| `FP8_ARCH_SPEC.md` | Markdown note: SonicMoE FP8 Frontier — Architecture Specification. | — |
| `fp8_architecture_comparison.md` | Markdown note: FP8 MoE 三方架构对比：DeepEP · SonicMoE BF16 · SonicMoE FP8 Frontier. | — |
| `HANDOFF.md` | Redirect/compatibility note that points readers to root `HANDOFF.md`. | redirect only |
| `KNOWLEDGE_BASE.md` | Markdown note: SonicMoE FP8 Frontier — Knowledge Base. | local/untracked reference |
| `pad_audit_methodology.md` | Markdown note: Route-Level Padding：面向 FP8 MoE 的零精度损失对齐方案. | — |
| `PADDLEFLEET_MIGRATION_S74.md` | Markdown note: PaddleFleet Migration — Sonic-MoE Session 74. | — |
| `session60_lessons.md` | Markdown note: Session 60 Engineering Log — Gate↔MLP Gradient Chain & Precision. | — |
| `setup_dev_b.sh` | Shell helper: Setup script for dev_b environment to run SonicMoE with ERNIE integration.. | — |
| `wgrad_fp8_dual_quant_design.md` | Markdown note: Wgrad FP8 Optimization — Zero-Copy Dual Quantization Design. | — |
