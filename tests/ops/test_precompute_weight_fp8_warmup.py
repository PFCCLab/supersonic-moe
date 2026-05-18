"""Test precompute_weight_fp8_warmup is bit-exact vs. the sequential call
sequence and substantially faster (via stream overlap).

Guards the fused pair-quantize Triton kernel (baseline path) and the
dual-stream iso32 warmup (ISO32 path). Both paths share the same contract:
warmup populates caches identically to lazy evaluation, and is faster than
serial cold-start.
"""
import os
import time
import pytest
import torch

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    precompute_weight_fp8,
    precompute_weight_fp8_for_fused_gated,
    precompute_weight_fp8_for_direct_fused_dgated,
    precompute_weight_fp8_warmup,
    clear_blockscaled_fp8_weight_cache,
    _iso32_weight_enabled,
)


def _eq_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare two same-shape tensors as raw bytes (handles fp8/uint8 views).

    Note: avoids ``torch.equal`` because the paddle/torch interop venv
    monkey-patches it into an elementwise ``==`` returning a tensor.
    """
    av = a.contiguous().view(torch.uint8) if not a.is_contiguous() else a.view(torch.uint8)
    bv = b.contiguous().view(torch.uint8) if not b.is_contiguous() else b.view(torch.uint8)
    if av.shape != bv.shape:
        return False
    return bool((av == bv).all().item())


@pytest.mark.parametrize("shape", [
    (3072, 1536, 8),   # Production reference shape
    (1024, 512, 4),    # Smaller for fast CI
])
def test_fused_warmup_bit_exact(shape):
    """Verify fused warmup populates caches bit-identically to the sequential path.

    NOTE: This test only applies to the pair-kernel path. Under ISO32, the
    sequential consumer calls fall through to pair-kernel (no iso32 cache miss
    handler), while warmup uses iso32_dual_quantize — different algorithm,
    different bytes. ISO32 warmup correctness is validated in
    test_iso32_weight_cache.py::TestIso32WeightCache.
    """
    if _iso32_weight_enabled():
        pytest.skip("bit-exact test not applicable under ISO32 (different quant algorithm)")
    H, I, E = shape
    torch.manual_seed(0)
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.05

    # Sequential reference (cold start: each call populates on first access)
    clear_blockscaled_fp8_weight_cache()
    ref_w1f = precompute_weight_fp8_for_fused_gated(w1)
    ref_w2v = precompute_weight_fp8(w2)
    ref_w2d = precompute_weight_fp8_for_direct_fused_dgated(w2)
    ref_w1t = precompute_weight_fp8(w1.permute(1, 0, 2))

    # Fused warmup (clears everything, then single fused call)
    clear_blockscaled_fp8_weight_cache()
    precompute_weight_fp8_warmup(w1, w2)
    new_w1f = precompute_weight_fp8_for_fused_gated(w1)
    new_w2v = precompute_weight_fp8(w2)
    new_w2d = precompute_weight_fp8_for_direct_fused_dgated(w2)
    new_w1t = precompute_weight_fp8(w1.permute(1, 0, 2))

    for name, (rfp8, rsc), (nfp8, nsc) in [
        ("w1_fused", ref_w1f, new_w1f),
        ("w2_varlen", ref_w2v, new_w2v),
        ("w2_dgated", ref_w2d, new_w2d),
        ("w1T_varlen", ref_w1t, new_w1t),
    ]:
        assert _eq_bytes(rfp8, nfp8), f"{name}: fp8 data differs"
        assert _eq_bytes(rsc, nsc), f"{name}: ISA-packed scales differ"


def test_fused_warmup_speedup():
    """Sanity-check that fused warmup (stream overlap) is faster than serial cold-start.

    Baseline: fused pair-kernel reads BF16 once → 2 FP8 outputs per weight.
    ISO32: fused dual-quantize on 2 streams overlaps w1 and w2.

    Both should be faster than the sequential 4-call path (which is serial
    on the default stream). Target: >= 1.2x speedup.
    """
    H, I, E = 3072, 1536, 8
    torch.manual_seed(0)
    w1 = torch.randn(2 * I, H, E, dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.05

    def serial_cold():
        clear_blockscaled_fp8_weight_cache()
        precompute_weight_fp8_for_fused_gated(w1)
        precompute_weight_fp8(w2)
        precompute_weight_fp8_for_direct_fused_dgated(w2)
        precompute_weight_fp8(w1.permute(1, 0, 2))

    def fused():
        clear_blockscaled_fp8_weight_cache()
        precompute_weight_fp8_warmup(w1, w2)

    # warm JIT
    for _ in range(3):
        serial_cold(); fused()
    torch.cuda.synchronize()

    n = 20
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): serial_cold()
    torch.cuda.synchronize(); t_serial = (time.perf_counter() - t0) / n * 1e6

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fused()
    torch.cuda.synchronize(); t_fused = (time.perf_counter() - t0) / n * 1e6

    speedup = t_serial / t_fused
    mode = "ISO32" if _iso32_weight_enabled() else "pair-kernel"
    print(f"\n[{mode}] serial cold: {t_serial:.1f} µs/iter")
    print(f"[{mode}] fused warmup: {t_fused:.1f} µs/iter")
    print(f"[{mode}] speedup:      {speedup:.2f}x")
    assert speedup >= 1.2, f"expected >=1.2x, got {speedup:.2f}x ({mode} mode)"
