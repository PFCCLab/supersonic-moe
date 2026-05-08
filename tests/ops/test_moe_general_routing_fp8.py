"""FP8 frontier unit-test for moe_general_routing_inputs.

Entry point: moe_general_routing_inputs  (user-controlled routing)
FP8 backend: GemmGatedSm100ZeroMat + quackgemm_default (SM100)

Usage
-----
  EBVENV=/path/to/venv
  $EBVENV/bin/python tests/ops/test_moe_general_routing_fp8.py           # correctness + bench
  $EBVENV/bin/python tests/ops/test_moe_general_routing_fp8.py --bench   # bench only

Performance notes (profiled on SM100, BF16→FP8, seq=16384, ep=32, E=8, H=3072, I=1536)
---------------------------------------------------------------------------
  sonic_fw wall  ≈ 26 ms   GPU kernel time ≈ 9 ms   GPU utilisation ≈ 35 %

  Remaining 17 ms is CPU-blocking overhead INSIDE moe_general_routing_inputs:

    cudaStreamSynchronize  8.76 ms   waits for GEMM chain in custom op
    cudaMalloc + cudaFree  2.46 ms   RadixSort temp buf for argsort() in routing metadata
    cudaStreamSynchronize  4.55 ms   post-GEMM metadata sync  (× 2 calls)
    cudaMemcpy D2H         4.06 ms   _get_cu_seqlens_cpu (alignment check × 2)

  Of these, only the alignment-check D2H is caller-fixable:
    • Set SONIC_MOE_FP8_ASSUME_ALIGNED=1 → skipped after first call (saves ~2.3 ms/iter).
    • Alternatively, after 3 consecutive 128-aligned iterations _ALIGNMENT_ASSUMED is set
      automatically (streak-based fast-path).
  The remaining ~15 ms is internal to the operator and cannot be avoided from the outside.

Caller responsibilities (why the original test was slow)
---------------------------------------------------------
  The original convert_sonic_moe_inputs() called .tolist() on GPU tensors and used nested
  Python loops over ~500k entries — this is fine for one-time setup but adds seconds in a
  training loop.  This test replaces it with a fully vectorised GPU-only prepare_sonic_inputs()
  that has a single .item() call (for dynamic shape) during one-time setup.
"""

import math
import os
import sys
import argparse

# ── Cluster paths (hardcoded for this environment) ──────────────────────────
os.environ.setdefault("USE_QUACK_GEMM", "1")
# Eliminate the _get_cu_seqlens_cpu D2H sync for the alignment check.
# This is safe when padding always guarantees 128-alignment (which prepare_sonic_inputs does).
os.environ.setdefault("SONIC_MOE_FP8_ASSUME_ALIGNED", "1")

_QUACK = "/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack"
_REPO  = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
for _p in (_QUACK, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paddle
paddle.enable_compat()

from sonicmoe.functional import (
    moe_general_routing_inputs,
    clear_all_fp8_weight_caches,
)
from sonicmoe.functional.utils import enable_fp8
from sonicmoe.enums import ActivationType

# ── Shape config ─────────────────────────────────────────────────────────────
T_SEQ   = 16384   # local sequence length (tokens per rank before dispatch)
H       = 3072    # hidden size
I       = 1536    # expert intermediate size (per expert)
K_TOPK  = 8       # global top-K
EP_SIZE = 32      # expert-parallel size
E_LOCAL = 8       # local experts on this rank
BLOCK   = 128     # required alignment block size

N_WARMUP = 5      # FP8 weight-cache warm-up iterations (not profiled)
N_BENCH  = 10     # profiled benchmark iterations


# ── Input preparation ─────────────────────────────────────────────────────────

def make_dispatched_inputs(
    T_seq: int, E_local: int, K_topk: int, ep_size: int, hidden: int, seed: int = 42
):
    """Simulate EP-dispatched inputs on GPU without any Python loops.

    Returns
    -------
    x                  : [T_local, H]  bfloat16 — token features received on this rank
    dispatched_indices : [T_local, K]  int32    — local expert ID per slot; -1 = unused
    dispatched_probs   : [T_local, K]  float32  — routing weight per slot
    """
    paddle.seed(seed)
    T_total = T_seq * ep_size
    E_total = E_local * ep_size

    # Global softmax top-K routing (all on GPU, no Python loops)
    logits   = paddle.randn([T_total, E_total])
    scores   = paddle.nn.functional.softmax(logits.cast('float32'), axis=-1)   # [T, E_total]
    topk_scr, topk_idx = paddle.topk(scores, K_topk, axis=-1)                 # [T, K]

    # Keep only assignments to local experts (expert_id < E_local)
    local_mask = topk_idx < E_local
    disp_idx   = paddle.where(local_mask, topk_idx, paddle.full_like(topk_idx, -1)).cast('int32')
    disp_probs = paddle.where(local_mask, topk_scr, paddle.zeros_like(topk_scr))

    # Retain only tokens with ≥ 1 local expert assignment
    has_local = (disp_idx >= 0).any(axis=-1)
    disp_idx   = disp_idx[has_local]
    disp_probs = disp_probs[has_local]

    T_local = disp_idx.shape[0]
    x = paddle.randn([T_local, hidden], dtype='bfloat16') * 0.02
    return x, disp_idx, disp_probs


def prepare_sonic_inputs(
    x: paddle.Tensor,
    dispatched_indices: paddle.Tensor,
    dispatched_probs: paddle.Tensor,
    n_local_experts: int,
    block: int = BLOCK,
):
    """Convert EP-dispatched inputs to the format expected by moe_general_routing_inputs.

    Fully GPU-vectorised — no Python loops, no .tolist(), no D2H except one
    .item() call (for dynamic shape) that happens once during setup.

    Padding strategy
    ----------------
    Each expert with a non-block-aligned token count receives virtual (zero-weight)
    padding tokens.  Padding token i can be shared across multiple experts
    (all experts that need padding in iteration i), so only max(pad_needed_per_expert)
    extra rows are appended to x.  This matches the behaviour of the original
    convert_sonic_moe_inputs() but without the Python loop overhead.

    Returns
    -------
    x_padded       : [T + max_pad, H]               bfloat16
    token_indices  : [TK_valid + n_pad_pairs]        int32, sorted ascending ✓
    expert_indices : [TK_valid + n_pad_pairs]        int32
    router_scores  : [TK_valid + n_pad_pairs]        float32
    """
    T = x.shape[0]

    # ── 1. Flatten valid (token, expert, score) triples ──────────────────────
    # Expanding arange to a [T, K] grid and masking in row-major order
    # preserves the row index → token_indices is sorted ascending by construction.
    tok_ids  = paddle.arange(T, dtype='int32').unsqueeze(1).expand_as(dispatched_indices)
    valid    = dispatched_indices >= 0
    tok_flat = tok_ids[valid]                              # [TK_valid], sorted 0‥T-1
    exp_flat = dispatched_indices[valid].cast('int32')     # [TK_valid]
    scr_flat = dispatched_probs[valid].cast('float32')     # [TK_valid]

    # ── 2. Per-expert token counts and required padding ───────────────────────
    exp_counts = paddle.bincount(exp_flat, minlength=n_local_experts).cast('int32')  # [E]
    pad_counts = (block - exp_counts % block) % block                               # [E], in [0, block-1]
    max_pad    = int(pad_counts.max().item())   # one D2H call, done once at setup time

    if max_pad == 0:
        return x, tok_flat, exp_flat, scr_flat

    # ── 3. Build padding entries (fully vectorised) ───────────────────────────
    # Grid [max_pad, E]: cell (i, e) is active iff i < pad_counts[e].
    # All active experts in row i share x-padded row (T + i) — same as the
    # original loop that reused num_recv_tokens + i across experts.
    row_ids = paddle.arange(max_pad, dtype='int32').unsqueeze(1).expand([max_pad, n_local_experts])
    exp_ids = paddle.arange(n_local_experts, dtype='int32').unsqueeze(0).expand([max_pad, n_local_experts])
    active  = row_ids < pad_counts.unsqueeze(0)            # [max_pad, E], bool

    pad_tok = (T + row_ids[active]).cast('int32')          # token indices: T, T, .., T+1, ..
    pad_exp = exp_ids[active].cast('int32')                # expert indices
    pad_scr = paddle.zeros([pad_tok.shape[0]], dtype='float32')

    # ── 4. Concatenate: real pairs (0‥T-1) before padding pairs (T‥T+max_pad-1)
    # → combined token_indices is sorted ascending ✓
    token_indices  = paddle.concat([tok_flat, pad_tok])
    expert_indices = paddle.concat([exp_flat, pad_exp])
    router_scores  = paddle.concat([scr_flat, pad_scr])

    # ── 5. Append max_pad zero rows to x ─────────────────────────────────────
    x_padded = paddle.concat([x, paddle.zeros([max_pad, x.shape[1]], dtype=x.dtype)], axis=0)

    return x_padded, token_indices, expert_indices, router_scores


def make_weights(I: int, H: int, E_local: int, seed: int = 0, with_main_grad: bool = False):
    """Create w1 [2I, H, E] and w2 [H, I, E] in bfloat16 with gradients enabled.

    Layout: w1[0::2, :, e] = gate weights, w1[1::2, :, e] = up weights.
    The FP8 cache internally permutes to (E, H, 2I) via _make_fp8_weight("w1_ekh").

    Parameters
    ----------
    with_main_grad : bool
        If True, attach a float32 ``main_grad`` attribute to each weight tensor,
        mimicking the ERNIE training convention where weight gradients are
        accumulated in float32 precision (not the bf16 .grad from autograd).
    """
    paddle.seed(seed)
    w1 = paddle.randn([2 * I, H, E_local], dtype='bfloat16') / math.sqrt(H)
    w1.stop_gradient = False
    w2 = paddle.randn([H, I, E_local], dtype='bfloat16') / math.sqrt(I)
    w2.stop_gradient = False
    if with_main_grad:
        w1.main_grad = paddle.zeros(w1.shape, dtype='float32')
        w2.main_grad = paddle.zeros(w2.shape, dtype='float32')
    return w1, w2


def accumulate_main_grad(*weights):
    """Accumulate bf16 autograd .grad into float32 .main_grad, then release .grad.

    This mirrors the ERNIE training loop pattern where weight gradients are kept
    in float32 for optimizer precision.  See bf16_weight_grad() in
    ernie-core/src/ernie_core/models/moe/token_dispatcher/fp8_utils.py for reference.
    """
    for w in weights:
        if w.grad is not None and hasattr(w, 'main_grad') and w.main_grad is not None:
            w.main_grad.add_(w.grad.cast('float32'))
            w.grad = None


# ── Forward + backward pass ───────────────────────────────────────────────────

def moe_fp8_fwd_bwd(
    x_padded, router_scores, token_indices, expert_indices,
    w1, w2, out_grad=None,
):
    """One FP8 forward (+ optional backward) through moe_general_routing_inputs.

    Notes
    -----
    • enable_fp8 context wraps only the forward — consistent with how training
      frameworks use it.
    • _refresh_fp8_config() is called internally by moe_general_routing_inputs;
      no need to call it from the test.
    • The FP8 weight cache (keyed by data_ptr + inplace_version) stays warm
      across iterations as long as weights are not modified.
    """
    with enable_fp8(True):
        out, expert_freq = moe_general_routing_inputs(
            x_padded,
            router_scores,
            token_indices,
            expert_indices,
            w1,
            None,           # b1 — no bias
            w2,
            None,           # b2 — no bias
            E_LOCAL,
            0,              # stream_id
            ActivationType.SWIGLU,
        )
    if out_grad is not None:
        out.backward(out_grad)
    return out, expert_freq


# ── Correctness test ──────────────────────────────────────────────────────────

def test_correctness():
    """Forward + backward correctness checks: shape, no NaN/Inf, all grads present."""
    print("=== Correctness ===")

    x, disp_idx, disp_probs = make_dispatched_inputs(T_SEQ, E_LOCAL, K_TOPK, EP_SIZE, H)
    x_padded, token_indices, expert_indices, router_scores = prepare_sonic_inputs(
        x, disp_idx, disp_probs, E_LOCAL
    )

    T_local  = x.shape[0]
    T_padded = x_padded.shape[0]
    TK_pairs = token_indices.shape[0]
    print(f"  T_local={T_local}  T_padded={T_padded}  TK_pairs={TK_pairs}")
    print(f"  token_indices sorted: {bool((token_indices[1:] >= token_indices[:-1]).all().item())}")

    # Detach and re-enable grads so backward can populate .grad
    x_padded      = x_padded.detach();      x_padded.stop_gradient      = False
    router_scores = router_scores.detach(); router_scores.stop_gradient = False
    w1, w2 = make_weights(I, H, E_LOCAL)

    # Pass 1 (cold cache): forward only — determine output shape for out_grad
    clear_all_fp8_weight_caches()
    out, expert_freq = moe_fp8_fwd_bwd(
        x_padded, router_scores, token_indices, expert_indices, w1, w2,
    )
    out_grad = paddle.randn_like(out)

    # Pass 2 (warm cache): forward + backward with clean gradients
    x_padded.clear_gradient(False); router_scores.clear_gradient(False)
    w1.clear_gradient(False);       w2.clear_gradient(False)
    out, expert_freq = moe_fp8_fwd_bwd(
        x_padded, router_scores, token_indices, expert_indices, w1, w2,
    )
    out.backward(out_grad)

    # ── Shape checks ─────────────────────────────────────────────────────────
    assert list(out.shape) == [T_padded, H], \
        f"Output shape mismatch: {list(out.shape)} != [{T_padded}, {H}]"
    assert list(expert_freq.shape) == [E_LOCAL], \
        f"expert_freq shape mismatch: {list(expert_freq.shape)}"

    # ── Numerical checks ──────────────────────────────────────────────────────
    assert not out.isnan().any(),                   "NaN in output"
    assert not out.isinf().any(),                   "Inf in output"
    assert x_padded.grad      is not None,          "No gradient for x_padded"
    assert not x_padded.grad.isnan().any(),         "NaN in dx"
    assert router_scores.grad is not None,          "No gradient for router_scores"
    assert not router_scores.grad.isnan().any(),    "NaN in d_router_scores"
    assert w1.grad is not None,                     "No gradient for w1"
    assert not w1.grad.isnan().any(),               "NaN in dw1"
    assert w2.grad is not None,                     "No gradient for w2"
    assert not w2.grad.isnan().any(),               "NaN in dw2"

    # ── Padding sanity: real-token outputs should be non-zero ─────────────────
    real_out = out[:T_local]
    assert real_out.abs().sum().item() > 0, "All real-token outputs are zero"

    print(f"  out (real tokens): mean={real_out.cast('float32').mean().item():.4f}  "
          f"std={real_out.cast('float32').std().item():.4f}")
    print(f"  expert_freq: {expert_freq.tolist()}")
    print("  PASSED ✓\n")

    clear_all_fp8_weight_caches()
    return x_padded, token_indices, expert_indices, router_scores, w1, w2


# ── main_grad accumulation test ──────────────────────────────────────────────

def test_main_grad_accumulation():
    """Validate float32 main_grad accumulation across multiple fwd+bwd iterations.

    This mimics real training where weight gradients are accumulated in float32
    (main_grad) across micro-batches before the optimizer step.  SonicMoE backward
    produces bf16 dw1/dw2 via autograd; this test verifies the post-backward
    accumulation pattern: main_grad += w.grad.float(); w.grad = None.
    """
    N_ACCUM = 5
    print("=== main_grad Accumulation ===")
    print(f"  Accumulating over {N_ACCUM} forward+backward iterations")

    x, disp_idx, disp_probs = make_dispatched_inputs(T_SEQ, E_LOCAL, K_TOPK, EP_SIZE, H)
    x_padded, token_indices, expert_indices, router_scores = prepare_sonic_inputs(
        x, disp_idx, disp_probs, E_LOCAL
    )

    x_padded      = x_padded.detach();      x_padded.stop_gradient      = False
    router_scores = router_scores.detach(); router_scores.stop_gradient = False
    w1, w2 = make_weights(I, H, E_LOCAL, with_main_grad=True)

    # Warm FP8 cache
    clear_all_fp8_weight_caches()
    out_ref, _ = moe_fp8_fwd_bwd(
        x_padded, router_scores, token_indices, expert_indices, w1, w2
    )
    out_grad = paddle.randn_like(out_ref)
    del out_ref

    # Accumulate gradients over N_ACCUM iterations (no gradient clearing!)
    for i in range(N_ACCUM):
        x_padded.clear_gradient(False)
        router_scores.clear_gradient(False)
        # NOTE: do NOT clear w1/w2 gradients -- they are consumed by accumulate_main_grad
        out, _ = moe_fp8_fwd_bwd(
            x_padded, router_scores, token_indices, expert_indices, w1, w2
        )
        out.backward(out_grad)
        accumulate_main_grad(w1, w2)

    paddle.device.synchronize()

    # ── Validation ────────────────────────────────────────────────────────────
    assert w1.main_grad is not None, "w1.main_grad is None"
    assert w2.main_grad is not None, "w2.main_grad is None"
    assert not w1.main_grad.isnan().any(), "NaN in w1.main_grad"
    assert not w2.main_grad.isnan().any(), "NaN in w2.main_grad"
    assert not w1.main_grad.isinf().any(), "Inf in w1.main_grad"
    assert not w2.main_grad.isinf().any(), "Inf in w2.main_grad"

    w1_mg_norm = float(w1.main_grad.norm().item())
    w2_mg_norm = float(w2.main_grad.norm().item())
    assert w1_mg_norm > 0, "w1.main_grad is all zeros after accumulation"
    assert w2_mg_norm > 0, "w2.main_grad is all zeros after accumulation"

    # Verify that .grad has been consumed (released after accumulation)
    assert w1.grad is None, "w1.grad should be None after accumulation"
    assert w2.grad is None, "w2.grad should be None after accumulation"

    # Verify main_grad dtype is float32
    assert w1.main_grad.dtype == paddle.float32, f"w1.main_grad dtype={w1.main_grad.dtype}"
    assert w2.main_grad.dtype == paddle.float32, f"w2.main_grad dtype={w2.main_grad.dtype}"

    print(f"  w1.main_grad: dtype={w1.main_grad.dtype}, norm={w1_mg_norm:.4f}")
    print(f"  w2.main_grad: dtype={w2.main_grad.dtype}, norm={w2_mg_norm:.4f}")
    print(f"  Accumulated {N_ACCUM} iterations successfully.")
    print("  PASSED ✓\n")

    clear_all_fp8_weight_caches()


# ── Benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark(x_padded=None, token_indices=None, expert_indices=None,
                   router_scores=None, w1=None, w2=None):
    """Profiled benchmark with NVTX markers.

    FP8 weight cache management
    ---------------------------
    The cache is keyed by (data_ptr, inplace_version, tag).  As long as w1/w2
    are not modified (no optimizer step), the cache stays valid across iterations.
    We do NOT call clear_all_fp8_weight_caches() inside the loop.

    Warm-up strategy
    ----------------
    N_WARMUP non-profiled iterations bring the FP8 weight cache to a warm state
    and let _ALIGNMENT_ASSUMED reach True (needs ≥ 3 consecutive aligned iters;
    or use SONIC_MOE_FP8_ASSUME_ALIGNED=1 — already set at the top of this file
    for immediate effect).
    """
    print("=== Benchmark ===")
    assume_aligned = os.environ.get("SONIC_MOE_FP8_ASSUME_ALIGNED", "0")
    print(f"  SONIC_MOE_FP8_ASSUME_ALIGNED={assume_aligned}  "
          f"N_WARMUP={N_WARMUP}  N_BENCH={N_BENCH}")

    # Build fresh inputs if not reusing from correctness test
    if x_padded is None:
        x, disp_idx, disp_probs = make_dispatched_inputs(T_SEQ, E_LOCAL, K_TOPK, EP_SIZE, H)
        x_padded, token_indices, expert_indices, router_scores = prepare_sonic_inputs(
            x, disp_idx, disp_probs, E_LOCAL
        )
        x_padded      = x_padded.detach();      x_padded.stop_gradient      = False
        router_scores = router_scores.detach(); router_scores.stop_gradient = False
        w1, w2 = make_weights(I, H, E_LOCAL, with_main_grad=True)

    # Ensure main_grad is set up
    if not hasattr(w1, 'main_grad') or w1.main_grad is None:
        w1.main_grad = paddle.zeros(w1.shape, dtype='float32')
        w2.main_grad = paddle.zeros(w2.shape, dtype='float32')

    # Canonical out_grad (fixed shape, reused every iteration)
    out_ref, _ = moe_fp8_fwd_bwd(
        x_padded, router_scores, token_indices, expert_indices, w1, w2
    )
    out_grad = paddle.randn_like(out_ref)
    del out_ref

    # ── FP8 weight cache warm-up (not profiled) ───────────────────────────────
    clear_all_fp8_weight_caches()
    print(f"  Warming cache for {N_WARMUP} iterations …", end="", flush=True)
    for _ in range(N_WARMUP):
        x_padded.clear_gradient(False)
        router_scores.clear_gradient(False)
        out, _ = moe_fp8_fwd_bwd(
            x_padded, router_scores, token_indices, expert_indices, w1, w2
        )
        out.backward(out_grad)
        accumulate_main_grad(w1, w2)
    paddle.device.synchronize()
    print(" done.")

    # ── Profiled benchmark ────────────────────────────────────────────────────
    # FP8 weight cache is now warm — do NOT clear it inside the loop.
    paddle.base.core.nvprof_start()

    for i in range(N_BENCH):
        x_padded.clear_gradient(False)
        router_scores.clear_gradient(False)
        # Do NOT clear w1/w2 gradient — accumulate into main_grad instead.

        # ── Forward ──────────────────────────────────────────────────────────
        paddle.base.core.nvprof_nvtx_push("sonic_fw")
        out, expert_freq = moe_fp8_fwd_bwd(
            x_padded, router_scores, token_indices, expert_indices, w1, w2,
        )
        paddle.base.core.nvprof_nvtx_pop()

        # ── Backward ─────────────────────────────────────────────────────────
        paddle.base.core.nvprof_nvtx_push("sonic_bw")
        out.backward(out_grad)
        paddle.base.core.nvprof_nvtx_pop()

        # ── main_grad accumulation (real training pattern) ────────────────────
        accumulate_main_grad(w1, w2)

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()

    clear_all_fp8_weight_caches()
    print("  Benchmark complete.\n")
    _print_perf_notes()


def _print_perf_notes():
    print("Performance notes")
    print("-" * 60)
    print("  Bottlenecks inside moe_general_routing_inputs (not caller-fixable):")
    print("    cudaStreamSynchronize  ~8.76 ms  waits for GEMM chain")
    print("    cudaMalloc + cudaFree  ~2.46 ms  RadixSort temp buf for argsort()")
    print("    cudaStreamSynchronize  ~4.55 ms  post-GEMM metadata (×2 calls)")
    print("  Caller-fixable (already applied in this test):")
    print("    SONIC_MOE_FP8_ASSUME_ALIGNED=1  saves ~2.33 ms/iter  (alignment D2H)")
    print("    Warm FP8 weight cache           saves ~980 µs/iter vs cold cache")
    print("    GPU-vectorised input prep       no Python-loop overhead in hot path")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="Run benchmark only (skip correctness test)")
    parser.add_argument("--main-grad", action="store_true",
                        help="Run main_grad accumulation test only")
    args = parser.parse_args()

    if args.main_grad:
        test_main_grad_accumulation()
    elif args.bench:
        test_benchmark()
    else:
        saved = test_correctness()
        test_main_grad_accumulation()
        # Reuse the same inputs and weights for the benchmark
        test_benchmark(*saved)
