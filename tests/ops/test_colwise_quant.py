"""Unit tests for colwise_quantize_and_pack and colwise_quantize_cute."""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_byte_exact,
    gold_e8m0_col_quant,
    unpack_isa_scales,
    QUANT_SHAPES, GROUP_SIZE,
)

pytestmark = [requires_blackwell, requires_quack]

# Colwise requires TK % 32 == 0
_COL_SHAPES = [(TK, dim) for TK, dim in
               [(v.values[0], v.values[1]) if hasattr(v, 'values') else v
                for v in QUANT_SHAPES]
               if TK % 32 == 0]
COL_SHAPES = [pytest.param(TK, dim, id=f"{TK}x{dim}") for TK, dim in _COL_SHAPES]


@pytest.mark.parametrize("TK,dim", COL_SHAPES)
def test_colwise_fp8_vs_gold(TK, dim, seed):
    """Gold col-quant FP8 bytes match Triton colwise kernel."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import colwise_quantize_and_pack

    x = torch.randn(TK, dim, dtype=torch.bfloat16, device="cuda")
    # colwise_quantize_and_pack: logical_rows=dim, logical_cols=TK (for ISA packing)
    fp8_kernel, scales_kernel = colwise_quantize_and_pack(x, dim, TK)
    fp8_gold, scales_gold = gold_e8m0_col_quant(x)
    assert_byte_exact(fp8_kernel, fp8_gold)
    assert_byte_exact(unpack_isa_scales(scales_kernel, dim, TK), scales_gold)


@pytest.mark.parametrize("TK,dim", [
    pytest.param(128, 128, id="smoke"),
    pytest.param(384, 1536, id="ragged-tile"),
    pytest.param(8192, 3072, id="production"),
])
def test_colwise_gather_fp8_and_scales_vs_gold(TK, dim, seed):
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import colwise_quantize_and_pack

    src_rows = TK + 17
    x = torch.randn(src_rows, dim, dtype=torch.bfloat16, device="cuda")
    gather_idx = torch.randint(0, src_rows, (TK,), dtype=torch.int32, device="cuda")
    gathered = x[gather_idx.long()]

    fp8_kernel, scales_kernel = colwise_quantize_and_pack(x, dim, TK, gather_idx=gather_idx)
    fp8_gold, scales_gold = gold_e8m0_col_quant(gathered)
    assert_byte_exact(fp8_kernel, fp8_gold)
    assert_byte_exact(unpack_isa_scales(scales_kernel, dim, TK), scales_gold)
