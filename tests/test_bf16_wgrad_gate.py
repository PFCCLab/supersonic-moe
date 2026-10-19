"""BF16 varlen-K wgrad default config: the 2CTA 256x256 tile is chosen for
every shape with M >= 2048 and more than one 128-wide N tile (N > 128), the
1CTA 128x128 tile otherwise."""

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import paddle  # noqa: E402

paddle.enable_compat()

import sonicmoe.quack_utils.bf16_wgrad_gemm as W  # noqa: E402


@pytest.mark.parametrize(
    "M, N, two_cta",
    [
        (2048, 256, True),
        (7168, 512, True),
        (4096, 136, True),  # just above one N tile
        (8192, 28672, True),
        (2048, 128, False),  # a single N tile
        (7168, 88, False),
        (2047, 4096, False),
        (None, 4096, False),
        (4096, None, False),
    ],
)
def test_wgrad_gate(monkeypatch, M, N, two_cta):
    monkeypatch.setattr(W, "get_device_capacity", lambda device: (10, 0))
    cfg = W._bf16_wgrad_default_config(None, M, N)
    assert (cfg.tile_m, cfg.tile_n, cfg.cluster_m) == ((256, 256, 2) if two_cta else (128, 128, 1))
    assert cfg.device_capacity == 10


def test_wgrad_gate_requires_sm100(monkeypatch):
    monkeypatch.setattr(W, "get_device_capacity", lambda device: (9, 0))
    with pytest.raises(RuntimeError):
        W._bf16_wgrad_default_config(None, 4096, 4096)
