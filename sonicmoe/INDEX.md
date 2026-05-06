# Directory Index: `/sonicmoe/`

> Primary Python package implementing SonicMoE kernels, configuration, and module entrypoints.
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
| `cli/` | Directory for cli. | — |
| `count_cumsum/` | CUDA extension for count / cumsum helpers used by routing code. | — |
| `ernie_compat/` | Directory for ernie compat. | — |
| `functional/` | Core forward and backward orchestration, routing helpers, and FP8 protocol flow. | — |
| `include/` | C/C++ headers shared by compiled extensions. | — |
| `quack_utils/` | QuACK / CUTLASS / Triton utilities for BF16 and FP8 GEMM paths. | — |

## Volatile / generated child directories
| Path | Summary | Notes |
| --- | --- | --- |
| `__pycache__/` | Volatile / generated subtree. | Python bytecode cache; disposable. |

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `__init__.py` | Package export surface for SonicMoE. | — |
| `_quack_compat.py` | Quack ↔ Paddle distributed compatibility shims. | — |
| `_triton_autotune_persist.py` | Persistent Triton autotune cache. | — |
| `_triton_stream_compat.py` | Triton ↔ Paddle stream compatibility shim. | — |
| `cache_manager.py` | Unified JIT cache management: Triton, CuTe, in-memory dicts. | — |
| `config.py` | Pythonic `SonicMoEConfig` context manager and configuration helpers. | — |
| `enums.py` | Shared enums used across module configuration and dispatch. | — |
| `jit.py` | JIT and compilation helpers. | — |
| `jit_warmup.py` | JIT warmup: pre-compile all CuTe + Triton kernels before training begins. | — |
| `moe.py` | Main MoE module implementation and FP8 stash / optimizer helpers. | — |
| `triton_utils.py` | Python module for triton utils. | — |
| `utils.py` | General package-level utility helpers. | — |
