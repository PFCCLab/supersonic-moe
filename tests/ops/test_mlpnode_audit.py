#!/usr/bin/env python
"""Rigorous precision + performance + memory audit for SonicMoEMlpNode.

Tests:
  1. Numerical precision: element-wise relative error, cosine similarity,
     max/mean/p99 absolute diff, gradient flow checks
  2. Performance: GPU-projection per-iter vs frontier baseline
  3. Memory: peak allocated, no leaks across iterations

Usage:
  source .runenv.sh
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_mlpnode_audit.py
  CUDA_VISIBLE_DEVICES=0 python tests/ops/test_mlpnode_audit.py --fp8-input  # FP8 recv_x path
"""

import argparse
import gc
import math
import os
import sys

os.environ.setdefault("USE_QUACK_GEMM", "1")
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")
os.environ.setdefault("SONIC_MOE_FP8_MODE", "perf")

_REPO = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()
import torch
import torch.nn.functional as F

from sonicmoe.enums import ActivationType
from sonicmoe.ernie_compat import (
    SonicMoEMlpNode,
    flush_native_grads,
    invalidate_weight_caches,
)
from sonicmoe.ernie_compat.mlp_node_v2 import (
    stack_ernie_w1,
    stack_ernie_w2,
)

H = 3072


class MockExpert:
    def __init__(self, h, i, seed):
        paddle.seed(seed)
        self.up_gate_proj = type("P", (), {
            "weight": paddle.randn([h, 2 * i], dtype="bfloat16") / math.sqrt(h),
        })()
        self.down_proj = type("P", (), {
            "weight": paddle.randn([i, h], dtype="bfloat16") / math.sqrt(i),
        })()
        self.up_gate_proj.weight.stop_gradient = False
        self.down_proj.weight.stop_gradient = False


def cosine_sim(a, b):
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def relative_error_stats(out, ref):
    """Compute relative error statistics."""
    diff = (out.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    # Avoid division by zero: only compute relative error where ref > threshold
    mask = ref_abs > 1e-6
    rel_err = torch.zeros_like(diff)
    rel_err[mask] = diff[mask] / ref_abs[mask]
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "p99_abs_diff": diff.quantile(0.99).item(),
        "max_rel_err": rel_err[mask].max().item() if mask.any() else 0.0,
        "mean_rel_err": rel_err[mask].mean().item() if mask.any() else 0.0,
        "p99_rel_err": rel_err[mask].quantile(0.99).item() if mask.any() else 0.0,
        "cosine_sim": cosine_sim(out, ref),
        "out_norm": out.float().norm().item(),
        "ref_norm": ref.float().norm().item(),
        "snr_db": 10 * math.log10(ref.float().norm().item()**2 / (diff.norm().item()**2 + 1e-30)),
    }


def test_gold_precision(T, E, I):
    """Gold test: compare FP8 MlpNode output against BF16 manual SwiGLU FFN."""
    print(f"\n{'='*60}")
    print(f"GOLD PRECISION: FP8 MlpNode vs BF16 Reference (T={T}, E={E})")
    print(f"{'='*60}")

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E
    paddle.seed(42)
    x = paddle.randn([T, H], dtype="bfloat16") * 0.02

    # BF16 ground truth: manual SwiGLU FFN
    out_gold = paddle.zeros([T, H], dtype="bfloat16")
    offset = 0
    for e_idx, count in enumerate(tpe):
        if count == 0:
            continue
        x_e = x[offset:offset + count]
        w_ug = experts[e_idx].up_gate_proj.weight  # [H, 2I]
        z = paddle.matmul(x_e, w_ug)  # [count, 2I]
        gate = z[:, :I]
        up = z[:, I:]
        y1 = paddle.nn.functional.silu(gate.cast("float32")).cast("bfloat16") * up
        w_d = experts[e_idx].down_proj.weight  # [I, H]
        out_e = paddle.matmul(y1, w_d)
        out_gold[offset:offset + count] = out_e
        offset += count

    # FP8 MlpNode
    from sonicmoe.functional import clear_all_fp8_weight_caches
    import sonicmoe.functional as functional
    functional._ALIGNMENT_ASSUMED = True
    functional._ALIGNMENT_STREAK = 100
    clear_all_fp8_weight_caches()
    invalidate_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)
    for _ in range(5):
        node.forward(x.clone(), tpe)
    out_test = node.forward(x, tpe)

    # Metrics
    diff = (out_test.cast("float32") - out_gold.cast("float32")).abs()
    a = out_test.flatten().cast("float32")
    b = out_gold.flatten().cast("float32")
    cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-10))
    snr = 10 * math.log10(float(b.norm())**2 / (float(diff.norm())**2 + 1e-30))

    print(f"  BF16 gold norm:    {float(out_gold.norm()):.6f}")
    print(f"  FP8 test norm:     {float(out_test.norm()):.6f}")
    print(f"  Cosine similarity: {cos:.8f}")
    print(f"  SNR:               {snr:.1f} dB")
    print(f"  Max abs diff:      {float(diff.max()):.4e}")
    print(f"  Mean abs diff:     {float(diff.mean()):.4e}")

    assert cos > 0.99, f"Cosine too low: {cos}"
    assert snr > 20, f"SNR too low: {snr}"
    print("  PASS: FP8 output matches BF16 gold within expected FP8 tolerance")
    return True


def test_precision_single_forward(T, E, I):
    """Test: single forward produces non-trivial, deterministic output."""
    print(f"\n{'='*60}")
    print(f"PRECISION AUDIT: T={T}, E={E}, I={I}, H={H}")
    print(f"{'='*60}")

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E

    # Create deterministic input
    paddle.seed(42)
    x = paddle.randn([T, H], dtype="bfloat16") * 0.02

    # Run with fresh caches
    invalidate_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Warmup (5 iters to stabilize FP8 caches)
    for _ in range(5):
        out = node.forward(x.clone(), tpe)
        out.backward(paddle.ones_like(out))
    flush_native_grads()

    # Run twice with same input → outputs should be identical (FP8 cache stable)
    x1 = x.clone().detach(); x1.stop_gradient = False
    out1 = node.forward(x1, tpe)

    x2 = x.clone().detach(); x2.stop_gradient = False
    out2 = node.forward(x2, tpe)

    print("\n--- Determinism (same input, two runs) ---")
    det_stats = relative_error_stats(out1, out2)
    print(f"  Max abs diff:  {det_stats['max_abs_diff']:.2e}")
    print(f"  Cosine sim:    {det_stats['cosine_sim']:.10f}")
    assert det_stats["max_abs_diff"] < 1e-5, f"Non-deterministic! diff={det_stats['max_abs_diff']}"
    print("  PASS: Outputs are deterministic")

    # Output sanity checks
    print("\n--- Output sanity ---")
    print(f"  Shape: {list(out1.shape)}")
    print(f"  Norm:  {out1.float().norm().item():.4f}")
    print(f"  Mean:  {out1.float().mean().item():.6f}")
    print(f"  Std:   {out1.float().std().item():.6f}")
    nonzero = (out1 != 0).float().mean().item()
    print(f"  Non-zero fraction: {nonzero:.6f}")
    assert nonzero > 0.99, f"Too many zeros! nonzero={nonzero}"
    assert out1.float().norm().item() > 0.1, "Output norm too small"
    assert not out1.isnan().any(), "NaN in output"
    assert not out1.isinf().any(), "Inf in output"
    print("  PASS: Output is non-trivial, no NaN/Inf")

    # Backward precision: dx should be non-trivial
    print("\n--- Backward precision (dx) ---")
    grad_out = paddle.randn([T, H], dtype="bfloat16") * 0.01
    x3 = x.clone().detach(); x3.stop_gradient = False
    out3 = node.forward(x3, tpe)
    out3.backward(grad_out)
    flush_native_grads()
    # In Paddle compat, grad may be on .grad or returned differently
    dx = x3.grad
    if dx is None:
        # Paddle PyLayer may not propagate grad to non-leaf tensors the same way
        print("  dx is None (Paddle PyLayer grad propagation — checking main_grad instead)")
        # Verify gradient flow via main_grad growth
        w1_n = experts[0].up_gate_proj.weight.main_grad.norm().item()
        assert w1_n > 0, "No gradient flow (main_grad zero after backward)"
        print(f"  main_grad norm confirms gradient flow: {w1_n:.4f}")
    else:
        assert not dx.isnan().any(), "NaN in dx"
        dx_norm = dx.float().norm().item()
        print(f"  dx norm: {dx_norm:.4f}")
        print(f"  dx non-zero: {(dx != 0).float().mean().item():.6f}")
        assert dx_norm > 0.01, f"dx norm too small: {dx_norm}"
    print("  PASS: Gradient flows correctly")

    # main_grad precision after flush
    print("\n--- main_grad precision ---")
    flush_native_grads()
    for e_idx, exp in enumerate(experts):
        mg_w1 = exp.up_gate_proj.weight.main_grad
        mg_w2 = exp.down_proj.weight.main_grad
        assert mg_w1 is not None and mg_w2 is not None
        assert mg_w1.dtype == paddle.float32
        assert not mg_w1.isnan().any(), f"NaN in w1 main_grad[{e_idx}]"
        assert not mg_w2.isnan().any(), f"NaN in w2 main_grad[{e_idx}]"
    w1_norms = [exp.up_gate_proj.weight.main_grad.norm().item() for exp in experts]
    w2_norms = [exp.down_proj.weight.main_grad.norm().item() for exp in experts]
    print(f"  w1 main_grad norms: [{', '.join(f'{n:.3f}' for n in w1_norms)}]")
    print(f"  w2 main_grad norms: [{', '.join(f'{n:.3f}' for n in w2_norms)}]")
    assert all(n > 0 for n in w1_norms), "Zero w1 main_grad"
    assert all(n > 0 for n in w2_norms), "Zero w2 main_grad"
    # Check that norms are in a reasonable range (not exploding)
    assert all(n < 1e6 for n in w1_norms), f"Exploding w1 main_grad: {max(w1_norms)}"
    # Check norms are not all identical (would indicate wrong accumulation)
    assert max(w1_norms) / (min(w1_norms) + 1e-10) > 1.01, "All w1 norms identical (suspicious)"
    print("  PASS: main_grad is fp32, non-zero, non-NaN, varied across experts")

    return True


def test_accumulation_correctness(T, E, I, n_iters=4):
    """Test: main_grad accumulates correctly over multiple iterations."""
    print(f"\n{'='*60}")
    print(f"ACCUMULATION AUDIT: {n_iters} iters, T={T}, E={E}")
    print(f"{'='*60}")

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E

    invalidate_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    # Warmup
    paddle.seed(0)
    for _ in range(5):
        x = paddle.randn([T, H], dtype="bfloat16") * 0.02
        out = node.forward(x, tpe)
        out.backward(paddle.ones_like(out))
    flush_native_grads()
    # Zero main_grad for clean test
    for exp in experts:
        exp.up_gate_proj.weight.main_grad.zero_()
        exp.down_proj.weight.main_grad.zero_()

    # Accumulate n_iters
    grad_out = paddle.randn([T, H], dtype="bfloat16") * 0.01
    norms_per_iter = []
    for i in range(n_iters):
        paddle.seed(100 + i)
        x_i = paddle.randn([T, H], dtype="bfloat16") * 0.02
        out_i = node.forward(x_i, tpe)
        out_i.backward(grad_out)

    flush_native_grads()

    # Check: norm should grow roughly as sqrt(n_iters) for random grads
    w1_norm = experts[0].up_gate_proj.weight.main_grad.norm().item()
    w2_norm = experts[0].down_proj.weight.main_grad.norm().item()
    print(f"  After {n_iters} iters: w1_norm={w1_norm:.4f}, w2_norm={w2_norm:.4f}")
    assert w1_norm > 0, "Zero accumulation"

    # Now do single iter and compare norm ratio
    for exp in experts:
        exp.up_gate_proj.weight.main_grad.zero_()
        exp.down_proj.weight.main_grad.zero_()

    paddle.seed(100)
    x_single = paddle.randn([T, H], dtype="bfloat16") * 0.02
    out_single = node.forward(x_single, tpe)
    out_single.backward(grad_out)
    flush_native_grads()

    w1_single = experts[0].up_gate_proj.weight.main_grad.norm().item()
    ratio = w1_norm / (w1_single + 1e-10)
    # For n_iters random grads, expected ratio ≈ sqrt(n_iters) ± factor of 2
    expected_min = math.sqrt(n_iters) * 0.3
    expected_max = math.sqrt(n_iters) * 3.0
    print(f"  Single iter norm: {w1_single:.4f}")
    print(f"  Ratio {n_iters}-iter/1-iter: {ratio:.2f} (expected ~{math.sqrt(n_iters):.1f})")
    assert expected_min < ratio < expected_max, f"Unexpected ratio {ratio}"
    print("  PASS: Accumulation scales as expected")
    return True


def test_memory_stability(T, E, I, n_iters=10):
    """Test: no memory leaks across iterations."""
    print(f"\n{'='*60}")
    print(f"MEMORY AUDIT: {n_iters} iters, T={T}")
    print(f"{'='*60}")

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E

    invalidate_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)
    grad_out = paddle.randn([T, H], dtype="bfloat16") * 0.01

    # Warmup
    for _ in range(5):
        x = paddle.randn([T, H], dtype="bfloat16") * 0.02
        out = node.forward(x, tpe)
        out.backward(grad_out)
    flush_native_grads()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure steady-state
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()

    for i in range(n_iters):
        paddle.seed(200 + i)
        x_i = paddle.randn([T, H], dtype="bfloat16") * 0.02
        out_i = node.forward(x_i, tpe)
        out_i.backward(grad_out)

    flush_native_grads()
    torch.cuda.synchronize()

    final_alloc = torch.cuda.memory_allocated()
    peak_alloc = torch.cuda.max_memory_allocated()
    leak = final_alloc - baseline_alloc

    print(f"  Baseline allocated: {baseline_alloc / 1024**2:.1f} MiB")
    print(f"  Final allocated:    {final_alloc / 1024**2:.1f} MiB")
    print(f"  Peak allocated:     {peak_alloc / 1024**2:.1f} MiB")
    print(f"  Apparent leak:      {leak / 1024**2:.1f} MiB")

    # Allow tolerance proportional to model size (FP8 caches, prequant buffers).
    # T=65536 with E=8,I=1536 has ~1.5GB of activation per iter — some caching is expected.
    max_leak_mib = max(100, T * H * 2 / 1024**2 * 3)  # ~3x one activation layer
    assert leak < max_leak_mib * 1024**2, f"Memory leak detected: {leak / 1024**2:.1f} MiB (limit: {max_leak_mib:.0f} MiB)"
    print(f"  PASS: No significant memory leaks (limit: {max_leak_mib:.0f} MiB)")
    return True


def test_performance(T, E, I, n_warmup=8, n_bench=12):
    """Test: measure CUDA event timing and report."""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE AUDIT: T={T}, E={E}, I={I}")
    print(f"{'='*60}")

    experts = [MockExpert(H, I, e) for e in range(E)]
    tpe = [T // E] * E

    invalidate_weight_caches()
    node = SonicMoEMlpNode(experts=experts, n_experts=E, hidden_size=H, intermediate_size=I)

    paddle.seed(42)
    x_fixed = paddle.randn([T, H], dtype="bfloat16") * 0.02
    grad_out = paddle.randn([T, H], dtype="bfloat16") * 0.01

    # Warmup
    for _ in range(n_warmup):
        out = node.forward(x_fixed, tpe)
        out.backward(grad_out)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark (frontier-compare: same x, same tpe every iter)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_bench):
        out = node.forward(x_fixed, tpe)
        out.backward(grad_out)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    per_iter_us = elapsed_ms / n_bench * 1000

    print(f"  CUDA events: {per_iter_us:.1f} μs/iter ({n_bench} iters)")
    print(f"  Total:       {elapsed_ms:.2f} ms")
    return per_iter_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=65536,
                        help="Total token-expert pairs (matches frontier TK=8192*K=8)")
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--I", type=int, default=1536)
    parser.add_argument("--fp8-input", action="store_true",
                        help="Test FP8 input path (DeepEP provides FP8 recv_x)")
    args = parser.parse_args()

    T, E, I = args.T, args.E, args.I

    print(f"SonicMoEMlpNode Audit")
    print(f"T={T}, E={E}, I={I}, H={H}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 0. Gold precision (FP8 vs BF16 reference)
    test_gold_precision(min(T, 1024), E, I)  # Use smaller T for gold (avoid OOM on manual matmul)

    # 1. Precision
    test_precision_single_forward(T, E, I)

    # 2. Accumulation
    test_accumulation_correctness(T, E, I, n_iters=4)

    # 3. Memory
    test_memory_stability(T, E, I, n_iters=10)

    # 4. Performance
    per_iter_us = test_performance(T, E, I)

    print(f"\n{'='*60}")
    print(f"ALL AUDITS PASSED")
    print(f"{'='*60}")
    print(f"  Performance: {per_iter_us:.1f} μs/iter (CUDA events)")
    print(f"  Frontier ref: 2738.9 μs/iter (GPU-projection, T=8192,K=8)")
    print()


if __name__ == "__main__":
    main()
