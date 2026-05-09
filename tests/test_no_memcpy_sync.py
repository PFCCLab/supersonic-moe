"""CI gate: no synchronized memcpy in metadata / router hot path.

Catches regressions where tensor.numel(), tensor.element_size(), or similar
Python-side shape queries trigger implicit cudaMemcpy D2H (synchronizing the
GPU stream and stalling the pipeline).

Fixed locations (2026-05-08, PR#22 by lshpku):
  - sonicmoe/ernie_compat/deepep_metadata.py:341
    .numel() * .element_size() → .size * .itemsize
  - sonicmoe/ernie_compat/mlp_node_v2.py:220-223
    dispatched_probs.numel() → isinstance check + torch.to_tensor

Test method: launch a long GPU kernel (large matmul ~3ms), then immediately
call the function under test. If the function has an implicit sync, its
CPU wall time will include the matmul execution (>2ms). If no sync, wall
time will be <0.5ms (just launch overhead).

This test is a HARD gate in CI — do NOT weaken the thresholds without
profiling proof that the sync is eliminated.
"""

import os
import sys
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

import paddle
paddle.enable_compat()
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M_GEMM = 4096  # large enough GEMM to keep GPU busy ~3ms
K_GEMM = 4096
N_GEMM = 4096
N_WARMUP = 3
N_BENCH = 5
# Threshold: if function takes >1.5ms wall after a GEMM launch, it synced.
SYNC_THRESHOLD_MS = 1.5


def _make_gemm_tensors(device="cuda"):
    a = torch.randn(M_GEMM, K_GEMM, device=device, dtype=torch.float32)
    b = torch.randn(K_GEMM, N_GEMM, device=device, dtype=torch.float32)
    return a, b


def _launch_heavy_gemm(a, b):
    return torch.matmul(a, b)


def _measure_wall_after_gemm(a, b, fn, n_warmup=N_WARMUP, n_bench=N_BENCH):
    """Measure wall time of fn() launched immediately after a heavy GEMM."""
    # Warmup
    for _ in range(n_warmup):
        _ = _launch_heavy_gemm(a, b)
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_bench):
        torch.cuda.synchronize()
        _ = _launch_heavy_gemm(a, b)  # GPU busy for ~3ms
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        torch.cuda.synchronize()

    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Test: _copy_tpe_h2d_async nbytes calculation has no sync
# ---------------------------------------------------------------------------
class TestNoSyncMemcpy:
    """Verify that metadata hot-path functions do not synchronize the stream."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"
        self.a, self.b = _make_gemm_tensors(self.device)

    def test_copy_tpe_h2d_async_no_sync(self):
        """deepep_metadata._copy_tpe_h2d_async must not synchronize.

        Regression guard for PR#22 fix: .numel()*.element_size() → .size*.itemsize
        """
        from sonicmoe.ernie_compat.deepep_metadata import _copy_tpe_h2d_async

        tpe_list = [128] * 8  # typical 8-expert tokens_per_expert

        def fn():
            _copy_tpe_h2d_async(tpe_list, self.device)

        avg_ms = _measure_wall_after_gemm(self.a, self.b, fn)
        assert avg_ms < SYNC_THRESHOLD_MS, (
            f"_copy_tpe_h2d_async took {avg_ms:.2f}ms after GEMM launch — "
            f"likely has an implicit cudaStreamSynchronize. "
            f"Threshold: {SYNC_THRESHOLD_MS}ms. "
            f"Check that nbytes calculation uses .size*.itemsize, not .numel()*.element_size()."
        )

    def test_dispatched_probs_numel_no_sync(self):
        """_differentiable_router_scores numel path must not synchronize.

        Regression guard for PR#22 fix: dispatched_probs.numel() →
        isinstance(dispatched_probs.size, int) check + torch.to_tensor.
        """
        # Simulate what _differentiable_router_scores does with dispatched_probs
        N_recv, topk = 1024, 8
        dispatched_probs = torch.randn(
            N_recv, topk, device=self.device, dtype=torch.float32
        )

        def fn():
            # This is the exact pattern from mlp_node_v2.py:220-223
            if isinstance(dispatched_probs.size, int):
                _ = torch.to_tensor(dispatched_probs.size, place="cpu")
            else:
                _ = dispatched_probs.numel()

        avg_ms = _measure_wall_after_gemm(self.a, self.b, fn)
        assert avg_ms < SYNC_THRESHOLD_MS, (
            f"dispatched_probs numel path took {avg_ms:.2f}ms after GEMM launch — "
            f"likely has an implicit cudaStreamSynchronize. "
            f"Threshold: {SYNC_THRESHOLD_MS}ms. "
            f"Check mlp_node_v2.py:220-223 isinstance guard."
        )

    def test_control_relu_no_sync(self):
        """Control: relu after GEMM must be fast (validates test methodology)."""
        x = torch.randn(1024, device=self.device, dtype=torch.float32)

        def fn():
            _ = torch.nn.functional.relu(x)

        avg_ms = _measure_wall_after_gemm(self.a, self.b, fn)
        assert avg_ms < SYNC_THRESHOLD_MS, (
            f"Control (relu) took {avg_ms:.2f}ms — test methodology may be broken"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
