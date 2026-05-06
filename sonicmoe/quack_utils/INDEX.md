# Directory Index: `/sonicmoe/quack_utils/`

> QuACK / CUTLASS / Triton utilities for BF16 and FP8 GEMM paths.
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
| `__init__.py` | Package marker and re-export surface. | — |
| `_validate.py` | Low-overhead input validation helpers for GPU kernel wrappers. | — |
| `blockscaled_fp8_gemm.py` | Hot-path Triton FP8 quantization, packing, and cache utilities. | — |
| `cute_blockscaled_quant.py` | CuTe DSL colwise blockscaled FP8 quantize — v5 with gather + ISA packing. | — |
| `cute_dual_quant.py` | CuTe DSL dual blockscaled FP8 quantize — [32][33] padded smem design. | — |
| `epi_blockscaled_quant.py` | Python module for epi blockscaled quant. | — |
| `fp8_quack_patch.py` | Python module for fp8 quack patch. | — |
| `fused_quant_kernels.py` | API-level fusion: dual_quantize_varlen(dz) + colwise_quantize_and_pack(dout, gather). | — |
| `gemm_dgated.py` | Python module for gemm dgated. | — |
| `gemm_dgated_fp8c_design.py` | Python module for gemm dgated fp8c design. | — |
| `gemm_gated.py` | Python module for gemm gated. | — |
| `gemm_interface.py` | Python module for gemm interface. | — |
| `gemm_sm100_fp8_zeromat.py` | Zero-materialization SM100 FP8 GEMM kernels. | — |
| `sgl_mxfp8_gemm.py` | sgl-kernel MXFP8 blockscaled grouped GEMM for varlen MoE. | — |
| `swiglu_triton.py` | Fused Triton kernels for interleaved SwiGLU forward and backward. | — |
| `triton_blockscaled_gemm.py` | Custom Triton blockscaled FP8 varlen GEMM for down-proj forward. | — |
