# Directory Index: `/sonicmoe/functional/`

> Core forward and backward orchestration, routing helpers, and FP8 protocol flow.
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
| `triton_kernels/` | Imported Triton helper kernels and license material. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Core FP8/BF16 forward-backward orchestration entrypoints. The FP8 prequant activation payload may include an optional prepacked Sonic ISA scale tensor for forward-only GEMM consumption while backward still saves raw scales. | — |
| `backward.py` | Python module for backward. | — |
| `forward.py` | Python module for forward. | — |
| `fp8_cutely_fused.py` | Python module for fp8 cutely fused. | — |
| `fp8_protocol.py` | Python module for fp8 protocol. | — |
| `fp8_quant.py` | Python module for fp8 quant. | — |
| `fp8_reference.py` | Python module for fp8 reference. | — |
| `grouped_gemm.py` | Python module for grouped gemm. | — |
| `moe_config.py` | Python module for moe config. | — |
| `reduction_over_k_gather.py` | Python module for reduction over k gather. | — |
| `tile_scheduler.py` | Python module for tile scheduler. | — |
| `topk_softmax.py` | Python module for top-k softmax. | — |
| `utils.py` | Python module for utils. | — |
