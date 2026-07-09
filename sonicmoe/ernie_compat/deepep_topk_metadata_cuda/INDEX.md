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
| `__init__.py` | `@torch.library.custom_op` + `@cpp_jit()` stubs. The base op returns the 7 metadata tensors; `deepep_topk_metadata_cuda_with_scales` additionally returns Sonic ISA-packed FP8 activation scales. Both use `mutates_args=()`. | — |
| `kernel.cu` | CUDA source. Launcher allocates metadata outputs + scratch via the caching allocator and returns independent storage per call (PP/1F1B ctx-save safe). The optional `with_scales` entry appends a same-stream raw-scale gather/ISA-pack kernel after scatter/fixup and returns the packed scale tensor. | — |
