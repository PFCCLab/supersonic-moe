# Directory Index: `/sonicmoe/ernie_compat/deepep_topk_metadata_cuda/`

> Directory for deepep top-k metadata cuda.
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
| `__init__.py` | `@torch.library.custom_op` + `@cpp_jit()` stub. Op takes (indices, probs, tokens_per_expert, dims, stream) and **returns** the 7 metadata tensors (`mutates_args=()`) instead of mutating pre-allocated buffers. | — |
| `kernel.cu` | CUDA source. Launcher allocates the 7 output tensors + scratch via the caching allocator and returns them (independent storage per call → PP/1F1B ctx-save safe); the 4 kernels (histogram / block_offset_scan / prefix_sums / scatter_and_fixup) are unchanged. | — |
