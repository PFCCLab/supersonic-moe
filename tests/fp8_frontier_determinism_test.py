"""FP8 Frontier Determinism Test — bit-exact equality across repeated runs.

The fused-gated FP8 frontier (alignment_assumed=True) must produce
*bit-identical* forward outputs and parameter gradients across repeated
fwd+bwd passes for the same MoE/state/inputs. Non-determinism here is a
correctness bug (atomic-order races, async-TMA scheduling leakage, state
pollution across iterations, etc.), not flakiness.

This test:
  1. Builds a MoE on the production Ernie shape (E=8, K=8 — frontier).
  2. Runs N fwd+bwd iterations with `_reset_fp8_state()` + reseed between
     each, after a warmup that latches `_ALIGNMENT_ASSUMED=True`.
  3. Asserts `torch.equal()` for outputs and every parameter gradient
     across all iterations.
  4. Verifies the frontier was actually entered (alignment_assumed +
     fused_gated) — otherwise the determinism claim is vacuous.

Run:
    source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
    cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
    CUDA_VISIBLE_DEVICES=0 USE_QUACK_GEMM=1 \
      python -m pytest tests/fp8_frontier_determinism_test.py -v -s --tb=short
"""

import unittest
from typing import List, Tuple

import paddle

# Activate Paddle's torch-compat proxy BEFORE any sonicmoe/torch import so that
# `torch.cuda.Stream` exposes `.stream_base` (used by `MoE.__init__` and the
# fused FP8 kernels). This mirrors the bench/test files in `tests/ops/`.
paddle.compat.enable_torch_proxy(silent=True)

import torch  # noqa: E402

from sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm  # noqa: E402
from sonicmoe.enums import ActivationType  # noqa: E402
import sonicmoe.functional as F_mod  # noqa: E402

from tests.fp8_frontier_strict_test import (  # noqa: E402
    _SEED,
    _FrontierProbe,
    _hook_forward_probe,
    _require_blackwell,
    _require_quack_gemm,
    _reset_fp8_state,
)


def _make_moe(H: int, I: int, E: int, K: int) -> MoE:
    moe = MoE(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(device="cuda", dtype=torch.bfloat16)
    return moe


def _warmup_alignment(moe: MoE, T: int, H: int, *, iters: int = 4) -> None:
    """Drive `_ALIGNMENT_ASSUMED` to True so the measured runs all enter
    the same (frontier) code path."""
    for _ in range(iters):
        x = 0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        with enable_quack_gemm(True):
            moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe, use_fp8=True)


def _fwd_bwd_snapshot(
    moe: MoE, x: torch.Tensor, dy: torch.Tensor,
) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]], torch.Tensor, _FrontierProbe]:
    """Run one fwd+bwd; return (y, [(name, grad)], dx, probe)."""
    probe = _FrontierProbe()
    handle = _hook_forward_probe(moe, probe)

    x_local = x.detach().clone().requires_grad_()
    with enable_quack_gemm(True):
        y, _ = moe(
            x_local,
            kernel_backend_moe=KernelBackendMoE.sonicmoe,
            use_fp8=True,
        )
    handle.remove()

    params = [(n, p) for n, p in moe.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        y, [x_local] + [p for _, p in params], grad_outputs=dy,
    )
    dx = grads[0].detach().clone()
    param_grads = [(n, g.detach().clone()) for (n, _), g in zip(params, grads[1:])]
    return y.detach().clone(), param_grads, dx, probe


def _tensors_bit_equal(a, b) -> bool:
    """Bit-exact tensor equality that works under both real-torch and Paddle's
    torch-proxy shim (where `torch.equal` is element-wise instead of reducing)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool((a == b).all().item())


class FP8FrontierDeterminismTest(unittest.TestCase):
    """Repeated fwd+bwd on the FP8 frontier must be bit-identical."""

    @classmethod
    def setUpClass(cls) -> None:
        _require_blackwell()
        _require_quack_gemm()

    def setUp(self) -> None:
        _reset_fp8_state()

    def tearDown(self) -> None:
        _reset_fp8_state()
        torch.cuda.empty_cache()

    def _run_determinism(
        self, T: int, H: int, I: int, E: int, K: int, label: str,
        *, iterations: int = 3,
    ) -> None:
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        moe = _make_moe(H, I, E, K)

        _warmup_alignment(moe, T, H, iters=4)
        self.assertTrue(
            F_mod._ALIGNMENT_ASSUMED,
            f"[{label}] _ALIGNMENT_ASSUMED did not latch after warmup "
            f"(streak={F_mod._ALIGNMENT_STREAK}) — determinism check would "
            f"not be on the frontier path.",
        )

        torch.manual_seed(_SEED + 1)
        torch.cuda.manual_seed(_SEED + 1)
        x = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach()
        dy = (0.02 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)).detach()

        y0, grads0, dx0, probe0 = _fwd_bwd_snapshot(moe, x, dy)
        probe0.assert_aligned_frontier(self, f"{label} / iter0")
        print(f"\n{probe0.report(f'{label} / iter0')}")

        for i in range(1, iterations):
            yi, gradsi, dxi, probei = _fwd_bwd_snapshot(moe, x, dy)
            probei.assert_aligned_frontier(self, f"{label} / iter{i}")

            self.assertTrue(
                _tensors_bit_equal(y0, yi),
                f"[{label}] forward output diverged at iter {i} "
                f"(max_abs={(y0.float()-yi.float()).abs().max().item():.3e})",
            )
            self.assertTrue(
                _tensors_bit_equal(dx0, dxi),
                f"[{label}] dx diverged at iter {i} "
                f"(max_abs={(dx0.float()-dxi.float()).abs().max().item():.3e})",
            )
            self.assertEqual(len(grads0), len(gradsi),
                             f"[{label}] grad count changed at iter {i}")
            for (n0, g0), (ni, gi) in zip(grads0, gradsi):
                self.assertEqual(n0, ni)
                if not _tensors_bit_equal(g0, gi):
                    diff = (g0.float() - gi.float()).abs().max().item()
                    self.fail(
                        f"[{label}] param grad '{n0}' diverged at iter {i} "
                        f"(max_abs={diff:.3e})"
                    )
            print(f"  iter{i}: bit-exact match ✓ (output + dx + {len(gradsi)} param grads)")

        print(f"  [{label}] {iterations} iterations bit-identical ✓")

    def test_ernie_production_deterministic(self) -> None:
        """Ernie-shape frontier (T=8192, H=3072, I=1536, E=8, K=8) is deterministic."""
        self._run_determinism(8192, 3072, 1536, 8, 8, "ernie-prod", iterations=3)

    def test_small_aligned_deterministic(self) -> None:
        """Small aligned (T=1024, H=3072, I=1536, E=8, K=8) is deterministic."""
        self._run_determinism(1024, 3072, 1536, 8, 8, "small-aligned", iterations=3)


if __name__ == "__main__":
    unittest.main()
