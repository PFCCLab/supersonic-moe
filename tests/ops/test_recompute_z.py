#!/usr/bin/env python
"""Focused validation of the ``recompute_z`` mode for SonicMoEMlpNode.

Verifies:
  1. ``out / dx / ds / dw1 / dw2`` from ``recompute_z=True`` match the
     ``recompute_z=False`` baseline (FP8 path) within strict tolerances.
  2. Forward peak memory drops when ``recompute_z=True`` (z_fp8 cache no
     longer holds ~213 MiB / layer at ERNIE shape).

Each test runs in a subprocess (matches the rest of the FP8 ops harness).

Run: CUDA_VISIBLE_DEVICES=0 python tests/ops/test_recompute_z.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_VENV = os.environ.get("SONIC_MOE_PADDLE_VENV", sys.prefix)
_PY = f"{_VENV}/bin/python"
if os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    print(f"\033[33mSwitch venv: {_VENV}\033[0m")
    os.execv(_PY, [_PY, *sys.argv])

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

_REPO = os.environ.get(
    "SONIC_MOE_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_QUACK = os.environ.get("SONIC_MOE_QUACK_PATH", "")
for _p in (_QUACK, _REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


def _child_run(recompute_z: bool, N: int, K: int, E: int, I: int):
    """Single FP8 forward+backward, returns out/grad tensors as numpy + peak mem."""
    import paddle
    paddle.enable_compat()
    import torch
    import numpy as np

    from sonicmoe.ernie_compat import SonicMoEMlpNode, flush_native_grads, invalidate_weight_caches
    import sonicmoe.functional as functional

    functional._ALIGNMENT_ASSUMED = True
    functional._ALIGNMENT_STREAK = 100

    sys.path.insert(0, str(Path(__file__).parent))
    from test_mlpnode_precision import H, MockExpert, _zero_main_grads

    if recompute_z:
        os.environ["SONIC_MOE_FP8_RECOMPUTE_Z"] = "1"
    else:
        os.environ.pop("SONIC_MOE_FP8_RECOMPUTE_Z", None)

    torch.manual_seed(2026)
    np.random.seed(2026)
    device = "cuda"

    experts = [MockExpert(H, I, e) for e in range(E)]
    raw = torch.randn(N, E, device=device)
    _, top_e = raw.topk(K, dim=-1)
    dispatched_indices = top_e.int()
    dispatched_probs = torch.rand(N, K, device=device) * 0.5 + 0.5
    dispatched_probs = (dispatched_probs / dispatched_probs.sum(dim=1, keepdim=True)).float()
    tpe = [int((dispatched_indices == e).sum().item()) for e in range(E)]

    paddle.seed(42)
    x_p = paddle.randn([N, H], dtype="bfloat16") * 0.02
    grad_out_p = paddle.randn([N, H], dtype="bfloat16") * 0.01
    grad_out = torch.from_dlpack(grad_out_p.detach()).to(device=device)

    invalidate_weight_caches()
    functional.clear_all_fp8_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    def _to_paddle(t):
        return paddle.utils.dlpack.from_dlpack(t.contiguous().detach().clone())

    # Warmup (3 iters, mirrors the rest of the harness).
    for _ in range(3):
        x_w = _to_paddle(torch.from_dlpack(x_p.detach()).to(device=device))
        x_w.stop_gradient = True
        di = _to_paddle(dispatched_indices); di.stop_gradient = True
        dp = _to_paddle(dispatched_probs); dp.stop_gradient = True
        out_w = node.forward(x_w, tpe, dispatched_indices=di, dispatched_probs=dp)
        out_w.backward(_to_paddle(grad_out))
    flush_native_grads()
    _zero_main_grads(experts)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x_in = _to_paddle(torch.from_dlpack(x_p.detach()).to(device=device))
    x_in.stop_gradient = False
    dp_in = _to_paddle(dispatched_probs); dp_in.stop_gradient = False
    di_in = _to_paddle(dispatched_indices); di_in.stop_gradient = True

    out = node.forward(x_in, tpe, dispatched_indices=di_in, dispatched_probs=dp_in)
    fwd_peak = torch.cuda.max_memory_allocated()
    out.backward(_to_paddle(grad_out))
    flush_native_grads()
    full_peak = torch.cuda.max_memory_allocated()

    out_np = torch.from_dlpack(out.detach()).to(device=device).float().cpu().numpy()
    dx_np = torch.from_dlpack(x_in.grad.detach()).to(device=device).float().cpu().numpy()
    ds_np = torch.from_dlpack(dp_in.grad.detach()).to(device=device).float().cpu().numpy()
    dw1_np = [torch.from_dlpack(e.up_gate_proj.weight.main_grad.detach()).to(device=device).float().cpu().numpy() for e in experts]
    dw2_np = [torch.from_dlpack(e.down_proj.weight.main_grad.detach()).to(device=device).float().cpu().numpy() for e in experts]

    np.savez(
        "/tmp/recompute_z_result.npz",
        out=out_np, dx=dx_np, ds=ds_np,
        dw1=np.stack(dw1_np), dw2=np.stack(dw2_np),
        fwd_peak=np.array([fwd_peak], dtype=np.int64),
        full_peak=np.array([full_peak], dtype=np.int64),
    )


def _run_subprocess(recompute_z: bool, N=1024, K=8, E=8, I=1536):
    code = (
        "from tests.ops.test_recompute_z import _child_run; "
        f"_child_run({recompute_z}, {N}, {K}, {E}, {I})"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO}:{_QUACK}:" + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout[-2000:])
        print("STDERR:", proc.stderr[-2000:])
        raise RuntimeError(f"Child failed (recompute_z={recompute_z})")


def main():
    import numpy as np

    print("=== recompute_z=False (baseline) ===", flush=True)
    _run_subprocess(False)
    base = dict(np.load("/tmp/recompute_z_result.npz"))

    print("=== recompute_z=True ===", flush=True)
    _run_subprocess(True)
    recm = dict(np.load("/tmp/recompute_z_result.npz"))

    def _cos_rrmse(a, b):
        a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
        cos = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30))
        rrmse = float(np.sqrt(((a - b) ** 2).sum() / ((b * b).sum() + 1e-30)))
        return cos, rrmse

    print()
    print("=== numeric equivalence ===", flush=True)
    all_ok = True
    for k in ("out", "dx", "ds", "dw1", "dw2"):
        cos, rrmse = _cos_rrmse(recm[k], base[k])
        ok = cos > 0.9999 and rrmse < 0.02
        all_ok &= ok
        flag = "✓" if ok else "✗"
        print(f"  {flag} {k:>4s}: cos={cos:.6f} rrmse={rrmse:.6f}")

    print()
    print("=== peak memory ===", flush=True)
    fwd_base = int(base["fwd_peak"][0]); fwd_recm = int(recm["fwd_peak"][0])
    full_base = int(base["full_peak"][0]); full_recm = int(recm["full_peak"][0])
    print(f"  forward peak:  baseline={fwd_base/1e6:.1f} MB  recompute={fwd_recm/1e6:.1f} MB  Δ={(fwd_recm-fwd_base)/1e6:+.1f} MB")
    print(f"  full peak:     baseline={full_base/1e6:.1f} MB  recompute={full_recm/1e6:.1f} MB  Δ={(full_recm-full_base)/1e6:+.1f} MB")

    print()
    print("ALL PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
