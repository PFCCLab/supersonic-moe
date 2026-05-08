# Directory Index: `/tests/`

> Repository-level regression, integration, and contract tests.
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
| `ops/` | Focused operator and module-level tests, including the newer MoE module suite. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package marker for test discovery. | — |
| `conftest.py` | Pytest fixtures for sonic-moe tests. | — |
| `count_cumsum_test.py` | Pytest coverage for count cumsum. | — |
| `fp8_frontier_determinism_test.py` | FP8 Frontier Determinism Test — bit-exact equality across repeated runs. | — |
| `fp8_frontier_stress_test.py` | FP8 frontier stress tests — multi-shape, multi-routing, multi-iter. | — |
| `fp8_frontier_strict_test.py` | FP8 Frontier Strict Test — no implicit fallback, no skip, fail-loud. | — |
| `fp8_large_project_contract_test.py` | Pytest coverage for fp8 large project contract. | — |
| `fp8_operator_options.py` | Pytest coverage for fp8 operator options. | — |
| `fp8_protocol_test.py` | Pytest coverage for fp8 protocol. | — |
| `moe_sm100_test.py` | Pytest coverage for moe sm100. | — |
| `moe_test.py` | Pytest coverage for moe. | — |
| `precision_compare_paths.py` | Precision: Path A (direct .apply(), is_varlen_K=False) vs Path B (SonicMoEMlpNode, is_varlen_K=True, route-level padding) Same weights, same routing, same input, same out_grad. | — |
| `report1.nsys-rep` | nsys report artifact for report1. | untracked in git |
| `report2.nsys-rep` | nsys report artifact for report2. | untracked in git |
| `report3.nsys-rep` | nsys report artifact for report3. | untracked in git |
| `report4.nsys-rep` | nsys report artifact for report4. | untracked in git |
| `report5.nsys-rep` | nsys report artifact for report5. | untracked in git |
| `run_regression.sh` | Shell helper for run regression. | — |
| `test_blockscaled_fp8_varlen.py` | Test blockscaled_fp8_gemm_varlen against bf16 gold reference. | — |
| `test_commons.py` | Pytest coverage for commons. | — |
| `test_moe_layer.py` | Unit test for MoELayer single-card implementation, adapted from PaddleFleet's moe_layer.py. | — |
| `test_moe_layer_e2e.py` | Unit test for MoELayer single-card implementation, adapted from PaddleFleet's moe_layer.py. | — |
