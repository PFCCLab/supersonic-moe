"""Tests for the gemm_num_sms SM-occupancy cap (compute/comm overlap knob).

Verifies that capping SMs via ``SonicMoEConfig(gemm_num_sms=...)``:
  1. shrinks ``max_active_clusters`` (grid.z) as expected,
  2. leaves the gated GEMM output bit-identical to the uncapped run,
  3. does NOT trigger a recompilation (the cap is a runtime dynamic Int32,
     not part of the compile key).
"""
import importlib

import pytest
import torch

from tests.ops.conftest import requires_blackwell, requires_quack

pytestmark = [requires_blackwell, requires_quack]


def _setup(T=2048, H=1024, I=1024, E=4, K=4):
    TK = T * K // E
    total_M = TK * E
    cu = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.02
    return x, w1, cu


def test_resolve_priority():
    """config field > env SONIC_MOE_GEMM_NUM_SMS > None; <=0 treated as unset."""
    import os
    from sonicmoe.config import SonicMoEConfig

    assert SonicMoEConfig().resolve_gemm_num_sms() is None
    assert SonicMoEConfig(gemm_num_sms=112).resolve_gemm_num_sms() == 112
    assert SonicMoEConfig(gemm_num_sms=0).resolve_gemm_num_sms() is None
    assert SonicMoEConfig(gemm_num_sms=-5).resolve_gemm_num_sms() is None

    prev = os.environ.get("SONIC_MOE_GEMM_NUM_SMS")
    try:
        os.environ["SONIC_MOE_GEMM_NUM_SMS"] = "64"
        assert SonicMoEConfig().resolve_gemm_num_sms() == 64
        # explicit field overrides env
        assert SonicMoEConfig(gemm_num_sms=112).resolve_gemm_num_sms() == 112
        os.environ["SONIC_MOE_GEMM_NUM_SMS"] = "not-an-int"
        assert SonicMoEConfig().resolve_gemm_num_sms() is None
    finally:
        if prev is None:
            os.environ.pop("SONIC_MOE_GEMM_NUM_SMS", None)
        else:
            os.environ["SONIC_MOE_GEMM_NUM_SMS"] = prev


def test_capped_max_active_clusters_numeric():
    """SM budget -> cluster budget = min(hw, num_sms // cluster_size), >= 1."""
    from sonicmoe.config import SonicMoEConfig, set_active_config
    from sonicmoe.quack_utils.sm_limit import capped_max_active_clusters
    from quack.cute_dsl_utils import get_max_active_clusters

    hw1 = get_max_active_clusters(1)
    hw2 = get_max_active_clusters(2)

    set_active_config(None)
    try:
        assert capped_max_active_clusters(1) == hw1
        assert capped_max_active_clusters(1, persistent=False) == 0

        set_active_config(SonicMoEConfig(gemm_num_sms=min(64, hw1)))
        assert capped_max_active_clusters(1) == min(hw1, min(64, hw1))

        # cluster_size=2: budget halves
        set_active_config(SonicMoEConfig(gemm_num_sms=112))
        assert capped_max_active_clusters(2) == min(hw2, 112 // 2)

        # cap larger than hw -> hw
        set_active_config(SonicMoEConfig(gemm_num_sms=10_000_000))
        assert capped_max_active_clusters(1) == hw1

        # tiny cap still yields at least one cluster
        set_active_config(SonicMoEConfig(gemm_num_sms=1))
        assert capped_max_active_clusters(4) == 1
    finally:
        set_active_config(None)


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_sm_cap_bit_exact_and_no_recompile(fp8, seed):
    """Capping SMs is bit-exact vs uncapped and does not force a recompile."""
    from sonicmoe.config import SonicMoEConfig, set_active_config
    from sonicmoe.quack_utils.gemm_interface import gemm_gated

    _gated = importlib.import_module("sonicmoe.quack_utils.gemm_gated")
    _zeromat = importlib.import_module("sonicmoe.quack_utils.gemm_sm100_fp8_zeromat")

    def cache_len():
        return len(_gated.gemm_gated.compile_cache) + len(_zeromat._zeromat_compile_cache)

    x, w1, cu = _setup()

    def run():
        if fp8:
            from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
                quantize_and_pack_activation,
                precompute_weight_fp8_for_fused_gated,
            )
            x_fp8, a_scales = quantize_and_pack_activation(x)
            w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)
            _, postact = gemm_gated(
                x_fp8, w_fp8, activation="swiglu", cu_seqlens_m=cu,
                a_scales=a_scales, b_scales=b_scales,
            )
        else:
            w1_3d = w1.permute(2, 1, 0).contiguous()
            _, postact = gemm_gated(x, w1_3d, activation="swiglu", cu_seqlens_m=cu)
        return postact.clone()

    set_active_config(None)
    try:
        out_base = run()
        n_base = cache_len()

        set_active_config(SonicMoEConfig(gemm_num_sms=64))
        out_64 = run()
        n_64 = cache_len()

        set_active_config(SonicMoEConfig(gemm_num_sms=112))
        out_112 = run()
        n_112 = cache_len()
    finally:
        set_active_config(None)

    assert n_base == n_64 == n_112, (
        f"SM cap forced a recompile (cache grew {n_base}->{n_64}->{n_112})"
    )
    assert torch.equal(out_base, out_64), "cap=64 changed the output"
    assert torch.equal(out_base, out_112), "cap=112 changed the output"
