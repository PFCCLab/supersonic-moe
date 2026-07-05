# Directory Index: `/sonicmoe/ernie_compat/`

> Directory for ernie compat.
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
| `deepep_metadata_cuda/` | Directory for deepep metadata cuda. | — |
| `deepep_topk_metadata_cuda/` | Directory for deepep top-k metadata cuda. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package marker and re-export surface. | — |
| `deepep_metadata.py` | DeepEP → SonicMoE metadata conversion (zero argsort, zero sync). Topk-path output tensors are now allocated inside the CUDA launcher (see `deepep_topk_metadata_cuda/`) and returned, so the Python layer no longer pre-allocates them — cuts per-call dygraph dispatch. | — |
| `mlp_node_v2.py` | SonicMoE ↔ ERNIE integration: ``SonicMoEMlpNode`` (FP8 production path). | — |
| `weight_layout_fusion.py` | Triton kernels for SonicMoE expert weight layout conversion used by paddlefleet_ops integration. | — |
