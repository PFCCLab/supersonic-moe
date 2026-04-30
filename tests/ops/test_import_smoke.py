"""Coverage smoke test: import every public module under ``sonicmoe`` so
that module-level decorators, dataclass declarations, and constant tables
are exercised by the coverage gate. This catches import-time regressions
(missing stubs, broken patches, circular imports) in production paths
that aren't wired into the headline FP8 fwd+bwd path.

If a module legitimately requires a runtime dependency that may be
missing on a vanilla CI host (e.g. quack pieces), import errors are
recorded and asserted on as a group at the end so we get one failure
listing all broken modules instead of stopping at the first.
"""
from __future__ import annotations

import importlib
import pytest

import sonicmoe  # noqa: F401  ensure package init (and autotune persist) ran

MODULES = [
    "sonicmoe._quack_compat",
    "sonicmoe._triton_autotune_persist",
    "sonicmoe._triton_stream_compat",
    "sonicmoe.cache_manager",
    "sonicmoe.cli.warmup",
    "sonicmoe.config",
    "sonicmoe.enums",
    "sonicmoe.ernie_compat.deepep_metadata",
    "sonicmoe.ernie_compat.mlp_node_v2",
    "sonicmoe.functional.backward",
    "sonicmoe.functional.forward",
    "sonicmoe.functional.fp8_cutely_fused",
    "sonicmoe.functional.fp8_protocol",
    "sonicmoe.functional.fp8_quant",
    "sonicmoe.functional.fp8_reference",
    "sonicmoe.functional.grouped_gemm",
    "sonicmoe.functional.moe_config",
    "sonicmoe.functional.reduction_over_k_gather",
    "sonicmoe.functional.tile_scheduler",
    "sonicmoe.functional.topk_softmax",
    "sonicmoe.functional.triton_kernels.bitmatrix",
    "sonicmoe.functional.utils",
    "sonicmoe.jit",
    "sonicmoe.jit_warmup",
    "sonicmoe.moe",
    "sonicmoe.quack_utils._validate",
    "sonicmoe.quack_utils.blockscaled_fp8_gemm",
    "sonicmoe.quack_utils.cute_blockscaled_quant",
    "sonicmoe.quack_utils.cute_dual_quant",
    "sonicmoe.quack_utils.epi_blockscaled_quant",
    "sonicmoe.quack_utils.fp8_quack_patch",
    "sonicmoe.quack_utils.fused_quant_kernels",
    "sonicmoe.quack_utils.gemm_dgated",
    "sonicmoe.quack_utils.gemm_dgated_fp8c_design",
    "sonicmoe.quack_utils.gemm_gated",
    "sonicmoe.quack_utils.gemm_interface",
    "sonicmoe.quack_utils.gemm_sm100_fp8_zeromat",
    "sonicmoe.quack_utils.sgl_mxfp8_gemm",
    "sonicmoe.quack_utils.swiglu_triton",
    "sonicmoe.quack_utils.triton_blockscaled_gemm",
    "sonicmoe.triton_utils",
    "sonicmoe.utils",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import_smoke(mod: str) -> None:
    importlib.import_module(mod)
