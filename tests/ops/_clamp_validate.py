"""Standalone clamp validation (PR41 re-add).

Validates the *installed* sonicmoe (paddlefleet_ops.sonicmoe) so it tests the
exact code path training uses. Run with the erniebot venv:

    python tests/ops/_clamp_validate.py

Checks:
  1. gemm_gated forward: kernel(clamp=cv) == torch clamp reference, and differs
     from unclamped output.
  2. gemm_dgated backward: kernel(clamp=cv) == torch clamp-bwd reference.
"""
import os
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")

import paddle  # noqa: E402

paddle.compat.enable_torch_proxy()
import paddlefleet_ops  # noqa: E402  registers proxy quack + native ops

# Alias the already-loaded staged tree so `import sonicmoe` is the SAME module
# object (avoids re-registering native ops).
for _name in list(sys.modules):
    if _name == "paddlefleet_ops.sonicmoe" or _name.startswith("paddlefleet_ops.sonicmoe."):
        sys.modules[_name.replace("paddlefleet_ops.sonicmoe", "sonicmoe", 1)] = sys.modules[_name]

from sonicmoe.quack_utils.gemm_interface import gemm_gated, gemm_dgated  # noqa: E402
# Use the SAME torch the package uses (real torch, not the paddle proxy that
# `import torch` would resolve to after enable_torch_proxy()).
from sonicmoe.quack_utils import gemm_interface as _gi  # noqa: E402

torch = _gi.torch


def rrmse(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def _torch_swiglu_clamp_fwd(z, cv):
    """interleaved-layout SwiGLU with clamp; mirrors PR41 reference."""
    gate = z[:, 0::2].float()
    up = z[:, 1::2].float()
    if cv > 0.0:
        gate = gate.clamp(max=cv)
        up = up.clamp(min=-cv, max=cv)
    return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        print("SKIP: requires SM100+")
        return
    torch.manual_seed(0)
    T, H, I, E, K = 256, 768, 384, 8, 8
    TK = T * K // E
    total_M = TK * E
    cu = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.5
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.5
    w1_3d = w1.permute(2, 1, 0).contiguous()

    cv = 0.35
    # unclamped
    z0, y0 = gemm_gated(x, w1_3d, activation="swiglu", cu_seqlens_m=cu)
    # clamped
    z1, y1 = gemm_gated(x, w1_3d, activation="swiglu", cu_seqlens_m=cu,
                        swiglu_clamp_value=cv)

    gold_clamped = _torch_swiglu_clamp_fwd(z1, cv)
    gold_unclamped = _torch_swiglu_clamp_fwd(z0, 0.0)

    r_clamped = rrmse(y1, gold_clamped)
    r_unclamped = rrmse(y0, gold_unclamped)
    diff = (y1.float() - y0.float()).abs().max().item()

    print(f"[fwd] unclamped kernel vs torch  RRMSE={r_unclamped:.6f}")
    print(f"[fwd] clamped   kernel vs torch  RRMSE={r_clamped:.6f}")
    print(f"[fwd] |clamped - unclamped| max  = {diff:.6f}")

    assert r_unclamped < 0.02, f"unclamped mismatch {r_unclamped}"
    assert r_clamped < 0.02, f"clamped mismatch {r_clamped}"
    assert diff > 1e-3, "clamp had no visible effect (signal too small or wiring broken)"
    print("CLAMP_FWD_PASS")


def _torch_dswiglu_clamp(dout_temp, z, cv):
    """interleaved dSwiGLU with clamp; mirrors PR41 backward reference.

    dout_temp: (M, I) upstream grad already projected by w2 (== `temp` in gold).
    z:         (M, 2I) pre-activation (gate/up interleaved).
    Returns dx (M, 2I) interleaved.
    """
    gate = z[:, 0::2].float()
    up = z[:, 1::2].float()
    if cv > 0.0:
        gate_c = gate.clamp(max=cv)
        up_c = up.clamp(min=-cv, max=cv)
    else:
        gate_c, up_c = gate, up
    sig = torch.sigmoid(gate_c)
    silu_gate = gate_c * sig
    d_gate = dout_temp * up_c * sig * (1.0 + gate_c * (1.0 - sig))
    d_up = dout_temp * silu_gate
    if cv > 0.0:
        # grad is zero where the input was clamped (out of range)
        d_gate = torch.where(gate <= cv, d_gate, torch.zeros_like(d_gate))
        up_in = (up >= -cv) & (up <= cv)
        d_up = torch.where(up_in, d_up, torch.zeros_like(d_up))
    M, I = gate.shape
    dx = torch.empty(M, 2 * I, dtype=torch.bfloat16, device=z.device)
    dx[:, 0::2] = d_gate.to(torch.bfloat16)
    dx[:, 1::2] = d_up.to(torch.bfloat16)
    return dx


def main_bwd():
    torch.manual_seed(1)
    T, H, I, E, K = 256, 768, 384, 8, 8
    TK = T * K // E
    total_M = TK * E
    cu = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    dout = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.5
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.5
    z = torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.5
    w2_3d = w2.permute(2, 0, 1).contiguous()

    # torch `temp` = per-expert dout @ w2
    temp = torch.zeros(total_M, I, dtype=torch.float32, device="cuda")
    for exp in range(E):
        s = cu[exp].item()
        e = cu[exp + 1].item()
        temp[s:e] = dout[s:e].float() @ w2[:, :, exp].float()

    cv = 0.35
    dx0, _ = gemm_dgated(dout, w2_3d, z, activation="swiglu", cu_seqlens_m=cu)
    dx1, _ = gemm_dgated(dout, w2_3d, z, activation="swiglu", cu_seqlens_m=cu,
                         swiglu_clamp_value=cv)

    gold0 = _torch_dswiglu_clamp(temp, z, 0.0)
    gold1 = _torch_dswiglu_clamp(temp, z, cv)

    r0 = rrmse(dx0, gold0)
    r1 = rrmse(dx1, gold1)
    diff = (dx1.float() - dx0.float()).abs().max().item()
    print(f"[bwd] unclamped kernel vs torch  RRMSE={r0:.6f}")
    print(f"[bwd] clamped   kernel vs torch  RRMSE={r1:.6f}")
    print(f"[bwd] |clamped - unclamped| max  = {diff:.6f}")
    assert r0 < 0.05, f"unclamped bwd mismatch {r0}"
    assert r1 < 0.05, f"clamped bwd mismatch {r1}"
    assert diff > 1e-3, "clamp had no visible effect on bwd"
    print("CLAMP_BWD_PASS")


def main_fp8_fwd():
    """FP8 blockscaled gated forward clamp: kernel(clamp) vs torch clamp ref."""
    from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
        quantize_and_pack_activation,
        precompute_weight_fp8_for_fused_gated,
    )

    torch.manual_seed(2)
    T, H, I, E, K = 256, 768, 384, 8, 8
    TK = T * K // E
    total_M = TK * E
    cu = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    x = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.5
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.5

    x_fp8, a_scales = quantize_and_pack_activation(x)
    w_fp8, b_scales = precompute_weight_fp8_for_fused_gated(w1)

    cv = 0.35
    z0, y0 = gemm_gated(x_fp8, w_fp8, activation="swiglu",
                        out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
                        cu_seqlens_m=cu, a_scales=a_scales, b_scales=b_scales)
    z1, y1 = gemm_gated(x_fp8, w_fp8, activation="swiglu",
                        out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
                        cu_seqlens_m=cu, a_scales=a_scales, b_scales=b_scales,
                        swiglu_clamp_value=cv)

    # torch clamp ref applied on the kernel's own preact z (isolates epilogue clamp)
    gold1 = _torch_swiglu_clamp_fwd(z1, cv)
    r1 = rrmse(y1, gold1)
    diff = (y1.float() - y0.float()).abs().max().item()
    print(f"[fp8-fwd] clamped kernel vs torch(on kernel-z) RRMSE={r1:.6f}")
    print(f"[fp8-fwd] |clamped - unclamped| max = {diff:.6f}")
    assert r1 < 0.05, f"fp8 clamped fwd mismatch {r1}"
    assert diff > 1e-3, "fp8 clamp had no visible effect"
    print("CLAMP_FP8_FWD_PASS")


if __name__ == "__main__":
    main()
    main_bwd()
    main_fp8_fwd()
