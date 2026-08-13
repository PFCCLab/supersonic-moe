"""Test: deterministic autotune pins one config, so output bits are reproducible.

``@triton.autotune`` selects by measured time, and for
``token_gather_sum_kernel`` the winning margin over a config with a different
BLOCK_K is under 1.5% — inside timing noise. Different tile shapes give a
different float32 accumulation order over K, hence different bf16 output bits,
which breaks bit-exact loss comparisons (sharding-reshard CI).
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sonicmoe import triton_utils
from sonicmoe.functional import reduction_over_k_gather as R

# (H, MAX_K) -> expected pinned (BLOCK_H, BLOCK_K); MAX_K = num_experts_per_tok.
PRODUCTION_SHAPES = {(1024, 10): (1024, 1), (2048, 10): (2048, 1)}


def _prune(H, MAX_K):
    return R._prune_triton_autotune_config(
        R.token_gather_sum_kernel.kernel.configs, {"H": H, "MAX_K": MAX_K}, H=H, MAX_K=MAX_K
    )


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("SONIC_MOE_DETERMINISTIC_AUTOTUNE", raising=False)
    monkeypatch.delenv("FLAGS_cudnn_deterministic", raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"SONIC_MOE_DETERMINISTIC_AUTOTUNE": "1"}, True),
        ({"SONIC_MOE_DETERMINISTIC_AUTOTUNE": "0"}, False),
        ({"FLAGS_cudnn_deterministic": "1"}, True),
        ({"FLAGS_cudnn_deterministic": "0"}, False),
        # An explicit sonic-moe opt-out wins over Paddle's global switch.
        ({"FLAGS_cudnn_deterministic": "1", "SONIC_MOE_DETERMINISTIC_AUTOTUNE": "0"}, False),
    ],
)
def test_gate_env_semantics(clean_env, env, expected):
    for key, value in env.items():
        clean_env.setenv(key, value)
    assert triton_utils.deterministic_autotune_enabled() is expected


@pytest.mark.parametrize("shape", sorted(PRODUCTION_SHAPES))
def test_autotune_off_keeps_multiple_configs(clean_env, shape):
    assert len(_prune(*shape)) > 1


@pytest.mark.parametrize("shape", sorted(PRODUCTION_SHAPES))
def test_deterministic_mode_pins_a_single_config(clean_env, shape):
    clean_env.setenv("SONIC_MOE_DETERMINISTIC_AUTOTUNE", "1")
    configs = _prune(*shape)
    assert len(configs) == 1, "a single config makes triton skip benchmarking entirely"

    expected_block_h, expected_block_k = PRODUCTION_SHAPES[shape]
    (config,) = configs
    assert (config.kwargs["BLOCK_H"], config.kwargs["BLOCK_K"]) == (expected_block_h, expected_block_k)
    assert config.num_warps == 4

    # The choice must be a pure function of the shape, not of any call history.
    assert repr(_prune(*shape)[0]) == repr(config)
