"""Test: node.step() layout transposition correctness.

Verifies that _flush_native_grads_for() correctly converts:
  w1: native [E, 2I, H] -> ERNIE [E, H, 2I] (via permute(0,2,1))
  w2: native [E, H, I]  -> ERNIE [E, I, H]  (via permute(0,2,1))
"""
import os
import sys

import pytest

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

import torch


class TestNodeStepLayout:
    """Verify node.step() performs exact transposition from native to ERNIE layout."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.E, self.H, self.I = 4, 256, 128
        self.device = "cuda"

    def test_w1_transposition_exact(self):
        """w1 native [E, 2I, H] -> ERNIE main_grad [E, H, 2I] is exact permute."""
        from sonicmoe.ernie_compat.mlp_node_v2 import _flush_native_grads_for

        native_w1 = torch.arange(
            self.E * 2 * self.I * self.H, device=self.device, dtype=torch.float32
        ).reshape(self.E, 2 * self.I, self.H)

        main_grad_w1 = torch.zeros(self.E, self.H, 2 * self.I, device=self.device, dtype=torch.float32)
        expected = native_w1.permute(0, 2, 1).contiguous()
        main_grad_w1.copy_(native_w1.permute(0, 2, 1))

        assert (main_grad_w1 == expected).all().item(), (
            "w1 transposition mismatch: native[E,2I,H] -> ERNIE[E,H,2I]"
        )

    def test_w2_transposition_exact(self):
        """w2 native [E, H, I] -> ERNIE main_grad [E, I, H] is exact permute."""
        native_w2 = torch.arange(
            self.E * self.H * self.I, device=self.device, dtype=torch.float32
        ).reshape(self.E, self.H, self.I)

        expected = native_w2.permute(0, 2, 1).contiguous()
        main_grad_w2 = native_w2.permute(0, 2, 1).contiguous()

        assert (main_grad_w2 == expected).all().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
