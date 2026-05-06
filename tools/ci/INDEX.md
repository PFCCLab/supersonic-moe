# Directory Index: `/tools/ci/`

> Directory for ci.
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
| `baselines.json` | JSON artifact with top-level keys: `_doc`, `jit`, `coverage`, `perf`. | — |
| `jit_bench.py` | Strict JIT-mechanism baseline gate for sonic-moe CI. | — |
| `multicard_smoke.py` | 2-rank distributed smoke test for sonicmoe mlp_node_v2. | — |
| `perf_gate.py` | CI perf gate: drive bench_mlpnode_topk_nsys via nsys, assert µs/iter ≤ budget. | — |
| `run_core_tests.sh` | Shell helper: sonic-moe core regression runner.. | — |
