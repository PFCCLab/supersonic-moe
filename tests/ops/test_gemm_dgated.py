"""Unit tests for gemm_dgated (bwd): torch ↔ BF16 3-way cross-validation.

gemm_dgated performs: dz = dSwiGLU(A @ B, PreAct).
  A = dout (total_M, H) gradient from downstream
  B = w2   (E, H, I) grouped by expert
  PreAct = z (total_M, 2I) pre-activation saved from forward

The FP8 blockscaled dgated path uses a low-level direct kernel that requires
specific tile configs, A_idx gather, and preact_fp8/preact_scales from the
forward pass. It is tested end-to-end via fp8_large_project_contract_test.py.

Here we validate the BF16 CUTLASS path against torch gold with detailed metrics,
and cross-check consistency across seeds and shapes.

3 tests per shape × seed:
  (1) torch gold vs BF16 CUTLASS — validates the BF16 kernel
  (2) torch gold z (preact) check — validates the GEMM output before activation
  (3) BF16 consistency check — same input different view = same output
"""
import math

import pytest
import torch

from tests.ops.conftest import (
    requires_quack,
    assert_bf16_close,
    rrmse, cosine_sim,
    GEMM_SHAPES,
)

def _has_sm90_plus() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _has_exact_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


requires_sm90_plus = pytest.mark.skipif(not _has_sm90_plus(), reason="Requires SM90+ GPU")
requires_exact_sm90 = pytest.mark.skipif(not _has_exact_sm90(), reason="Requires exact SM90 GPU")
pytestmark = [requires_quack, requires_sm90_plus]


def test_sm90_setup_hook_uses_hopper_signature(monkeypatch):
    import cutlass
    from quack.gemm_sm90 import GemmSm90
    from sonicmoe.quack_utils.gemm_dgated import GemmDGatedSm90

    instance = GemmDGatedSm90(cutlass.Float32, cutlass.BFloat16, (64, 64), (1, 1, 1))
    assert instance.blockscaled is False

    sentinel = object()
    seen = {}

    def fake_setup(self, epilogue_args):
        seen["self"] = self
        seen["epilogue_args"] = epilogue_args
        return sentinel

    monkeypatch.setattr(GemmSm90, "_setup_attributes", fake_setup)

    assert instance._setup_attributes("epi") is sentinel
    assert seen == {"self": instance, "epilogue_args": "epi"}


_DGATED_SHAPES = []
for p in GEMM_SHAPES:
    T, H, I, E, K = p.values
    TK = T * K // E
    if TK % 128 == 0 and H % 128 == 0 and I % 128 == 0 and (2 * I) % 128 == 0:
        _DGATED_SHAPES.append(pytest.param(T, H, I, E, K, id=p.id))
DGATED_SHAPES = _DGATED_SHAPES if _DGATED_SHAPES else GEMM_SHAPES[:1]


def _setup(T, H, I, E, K):
    TK = T * K // E
    total_M = TK * E
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    return TK, total_M, cu_seqlens


def _torch_dgated_gold(dout, w2, z_preact, cu_seqlens, E):
    """Torch gold: per-expert GEMM + dSwiGLU (interleaved layout).

    For each expert: temp[s:e] = dout[s:e] @ w2[:,:,exp]  → (tokens, I)
    Then dSwiGLU expands with z_preact interleaved layout.
    """
    total_M = dout.shape[0]
    I_dim = w2.shape[1]
    temp = torch.zeros(total_M, I_dim, dtype=torch.float32, device=dout.device)
    for exp in range(E):
        s = cu_seqlens[exp].item()
        e = cu_seqlens[exp + 1].item()
        if s < e:
            temp[s:e] = dout[s:e].float() @ w2[:, :, exp].float()

    z_f32 = z_preact.float()
    gate = z_f32[:, 0::2]
    up = z_f32[:, 1::2]
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    d_gate = temp * up * sig * (1.0 + gate * (1.0 - sig))
    d_up = temp * silu_gate

    dx = torch.empty(total_M, 2 * I_dim, dtype=torch.bfloat16, device=dout.device)
    dx[:, 0::2] = d_gate.to(torch.bfloat16)
    dx[:, 1::2] = d_up.to(torch.bfloat16)
    return dx


def _report(actual, expected, label):
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    mean_abs = (actual.float() - expected.float()).abs().mean().item()
    print(f"  [{label}] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={max_abs:.6f}, mean_abs_err={mean_abs:.6f}")
    return r, c


def _torch_dgated_gold_gather(dout, w2, z_preact, cu_seqlens, A_idx, E):
    total_M = A_idx.numel()
    I_dim = w2.shape[1]
    temp = torch.zeros(total_M, I_dim, dtype=torch.float32, device=dout.device)
    gathered = dout[A_idx.long()].float()
    for exp in range(E):
        s = cu_seqlens[exp].item()
        e = cu_seqlens[exp + 1].item()
        if s < e:
            temp[s:e] = gathered[s:e] @ w2[:, :, exp].float()

    z_f32 = z_preact.float()
    gate = z_f32[:, 0::2]
    up = z_f32[:, 1::2]
    sig = torch.sigmoid(gate)
    dx = torch.empty(total_M, 2 * I_dim, dtype=torch.bfloat16, device=dout.device)
    dx[:, 0::2] = (temp * up * sig * (1.0 + gate * (1.0 - sig))).to(torch.bfloat16)
    dx[:, 1::2] = (temp * gate * sig).to(torch.bfloat16)
    return dx


@requires_exact_sm90
def test_sm90_bf16_gather_uses_t_sized_source_and_a_idx(seed):
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    source_tokens, H, I, E = 37, 128, 128, 4
    counts = [0, 5, 1, 10]
    cu_seqlens = torch.tensor([0, 0, 5, 6, 16], dtype=torch.int32, device="cuda")
    A_idx = torch.randint(source_tokens, (sum(counts),), dtype=torch.int32, device="cuda")
    dout = torch.randn(source_tokens, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.02
    z_preact = torch.randn(A_idx.numel(), 2 * I, dtype=torch.bfloat16, device="cuda") * 0.02
    gold_dx = _torch_dgated_gold_gather(dout, w2, z_preact, cu_seqlens, A_idx, E)

    dx_out, _ = gemm_dgated(
        dout,
        w2.permute(2, 0, 1).contiguous(),
        z_preact,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens,
        A_idx=A_idx,
        dynamic_scheduler=False,
        tuned=False,
    )
    torch.cuda.synchronize()

    assert dout.shape == (source_tokens, H)
    assert A_idx.numel() == sum(counts)
    assert dx_out.shape == (A_idx.numel(), 2 * I)
    assert_bf16_close(dx_out, gold_dx, atol=1.4e-2)


@requires_sm90_plus
@pytest.mark.parametrize("T,H,I,E,K", DGATED_SHAPES)
def test_torch_vs_bf16(T, H, I, E, K, seed):
    """(1/3) BF16 CUTLASS gemm_dgated vs torch gold with detailed metrics."""
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    TK, total_M, cu_seqlens = _setup(T, H, I, E, K)
    dout = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.02
    z_preact = torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.02

    gold_dx = _torch_dgated_gold(dout, w2, z_preact, cu_seqlens, E)

    w2_3d = w2.permute(2, 0, 1).contiguous()  # (E, H, I)
    dx_out, _ = gemm_dgated(
        dout, w2_3d, z_preact,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens,
    )

    r, c = _report(dx_out, gold_dx, "dx: BF16 vs torch")
    assert_bf16_close(dx_out, gold_dx, atol=1.4e-2)
    if r > 0.005:
        import warnings
        warnings.warn(f"BF16 gemm_dgated dx RRMSE={r:.6f} (>0.5% — investigate)")


@requires_sm90_plus
@pytest.mark.parametrize("T,H,I,E,K", DGATED_SHAPES)
def test_bf16_deterministic(T, H, I, E, K, seed):
    """(2/3) BF16 CUTLASS gemm_dgated is deterministic across two calls."""
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    TK, total_M, cu_seqlens = _setup(T, H, I, E, K)
    dout = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.02
    z_preact = torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.02

    w2_3d = w2.permute(2, 0, 1).contiguous()
    dx1, _ = gemm_dgated(dout, w2_3d, z_preact, activation="swiglu", cu_seqlens_m=cu_seqlens)
    dx2, _ = gemm_dgated(dout, w2_3d, z_preact, activation="swiglu", cu_seqlens_m=cu_seqlens)

    r, c = _report(dx1, dx2, "dx: run1 vs run2")
    # Should be numerically identical
    max_diff = (dx1.float() - dx2.float()).abs().max().item()
    assert max_diff < 1e-6, f"Non-deterministic: max diff={max_diff}"


@requires_sm90_plus
@pytest.mark.parametrize("T,H,I,E,K", DGATED_SHAPES)
def test_bf16_postact_vs_torch(T, H, I, E, K, seed):
    """(3/3) BF16 CUTLASS gemm_dgated postact (y1s) vs torch gold SwiGLU fwd."""
    from sonicmoe.quack_utils.gemm_interface import gemm_dgated

    TK, total_M, cu_seqlens = _setup(T, H, I, E, K)
    dout = torch.randn(total_M, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.02
    z_preact = torch.randn(total_M, 2 * I, dtype=torch.bfloat16, device="cuda") * 0.02

    # Gold SwiGLU fwd for y1s comparison
    z_f32 = z_preact.float()
    gate = z_f32[:, 0::2]
    up = z_f32[:, 1::2]
    gold_y1 = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)

    w2_3d = w2.permute(2, 0, 1).contiguous()
    _, postact = gemm_dgated(
        dout, w2_3d, z_preact,
        activation="swiglu",
        cu_seqlens_m=cu_seqlens,
    )

    r, c = _report(postact, gold_y1, "y1s: BF16 vs torch gold SwiGLU")
    assert_bf16_close(postact, gold_y1, atol=1e-2)
