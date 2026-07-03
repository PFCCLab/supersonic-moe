"""Minimal reproducer: Paddle argsort 1D path triggers cudaStreamSynchronize.

Hypothesis
----------
Paddle's argsort on 1D tensors uses thrust::sort_by_key with the default
execution policy (no ThrustAllocator), causing thrust to call cudaMalloc
for workspace.  cudaMalloc is synchronous and drains the CUDA stream,
blocking the CPU until all pending GPU kernels finish.

Experiment design
-----------------
  Phase A: launch a heavy GPU kernel (large GEMM), then immediately call
           argsort on a 1D tensor.  If argsort has an implicit sync,
           the CPU wall time of argsort will include the GEMM execution time.

  Phase B: same, but call argsort on a 2D tensor (last-axis sort).
           The 2D path uses cub::DeviceSegmentedRadixSort with Paddle's
           caching allocator — no cudaMalloc, no sync expected.

  Phase C: same heavy GEMM, then a simple elementwise op (relu) instead of
           argsort.  This is the control — relu has no sync, so its wall
           time should be near-zero regardless of pending GPU work.

If the hypothesis is correct:
  - Phase A wall ≈ GEMM time + argsort kernel time  (sync drains GEMM)
  - Phase B wall ≈ argsort launch overhead only      (no sync)
  - Phase C wall ≈ relu launch overhead only          (no sync, control)

Usage
-----
  # Quick console check:
  $EBVENV/bin/python tests/ops/test_argsort_sync.py

  # Full nsys profile with NVTX:
  nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
       -o /tmp/argsort_sync \
       $EBVENV/bin/python tests/ops/test_argsort_sync.py --nsys
"""

import argparse
import os
import sys
import time

_REPO = os.environ.get("SONIC_MOE_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import paddle
paddle.enable_compat()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SORT   = 500_000   # 1D argsort size  (matches expert_indices in SonicMoE)
M_GEMM   = 4096      # GEMM dimensions  (large enough to keep GPU busy ~2ms)
K_GEMM   = 4096
N_GEMM   = 4096
N_WARMUP = 5
N_BENCH  = 10
ROWS_2D  = 100       # rows for the 2D argsort (Phase B)
COLS_2D  = N_SORT // ROWS_2D   # keep total elements similar


def heavy_gemm(a, b):
    """Launch a large matmul to keep the GPU busy."""
    return paddle.matmul(a, b)


def run_phases(use_nvtx: bool):
    """Run phases A/B/C and print wall times."""

    # ----- Allocate tensors ------------------------------------------------
    a = paddle.randn([M_GEMM, K_GEMM], dtype='float32')
    b = paddle.randn([K_GEMM, N_GEMM], dtype='float32')

    sort_1d = paddle.randint(0, 1000, [N_SORT]).cast('int32')        # 1D
    sort_2d = paddle.randint(0, 1000, [ROWS_2D, COLS_2D]).cast('int32')  # 2D
    relu_in = paddle.randn([N_SORT], dtype='float32')

    paddle.device.synchronize()

    def nvtx_push(name):
        if use_nvtx:
            paddle.base.core.nvprof_nvtx_push(name)

    def nvtx_pop():
        if use_nvtx:
            paddle.base.core.nvprof_nvtx_pop()

    if use_nvtx:
        paddle.base.core.nvprof_start()

    # ── Warmup (populate caching allocator + thrust workspace) ─────────────
    for _ in range(N_WARMUP):
        _ = heavy_gemm(a, b)
        paddle.device.synchronize()
        _ = sort_1d.argsort()
        _ = sort_2d.argsort(axis=-1)
        _ = paddle.nn.functional.relu(relu_in)
    paddle.device.synchronize()

    # ── Phase A: GEMM → argsort 1D (thrust path) ──────────────────────────
    times_a = []
    for i in range(N_BENCH):
        paddle.device.synchronize()

        nvtx_push("A_gemm_launch")
        _ = heavy_gemm(a, b)      # GPU busy ~2ms
        nvtx_pop()

        nvtx_push("A_argsort_1d")
        t0 = time.perf_counter()
        _ = sort_1d.argsort()     # hypothesis: will sync, draining GEMM
        t1 = time.perf_counter()
        nvtx_pop()

        times_a.append((t1 - t0) * 1000)

    paddle.device.synchronize()

    # ── Phase B: GEMM → argsort 2D last-axis (CUB path) ───────────────────
    times_b = []
    for i in range(N_BENCH):
        paddle.device.synchronize()

        nvtx_push("B_gemm_launch")
        _ = heavy_gemm(a, b)
        nvtx_pop()

        nvtx_push("B_argsort_2d")
        t0 = time.perf_counter()
        _ = sort_2d.argsort(axis=-1)  # hypothesis: no sync
        t1 = time.perf_counter()
        nvtx_pop()

        times_b.append((t1 - t0) * 1000)

    paddle.device.synchronize()

    # ── Phase C: GEMM → relu (control, no sync) ───────────────────────────
    times_c = []
    for i in range(N_BENCH):
        paddle.device.synchronize()

        nvtx_push("C_gemm_launch")
        _ = heavy_gemm(a, b)
        nvtx_pop()

        nvtx_push("C_relu")
        t0 = time.perf_counter()
        _ = paddle.nn.functional.relu(relu_in)
        t1 = time.perf_counter()
        nvtx_pop()

        times_c.append((t1 - t0) * 1000)

    paddle.device.synchronize()

    if use_nvtx:
        paddle.base.core.nvprof_stop()

    # ── Report ─────────────────────────────────────────────────────────────
    def stats(vals):
        avg = sum(vals) / len(vals)
        mn  = min(vals)
        mx  = max(vals)
        return f"avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}"

    print()
    print("=" * 70)
    print("Paddle argsort sync test")
    print("=" * 70)
    print(f"  GEMM shape: [{M_GEMM},{K_GEMM}]x[{K_GEMM},{N_GEMM}]  (keeps GPU busy)")
    print(f"  Sort 1D:    [{N_SORT}]  (thrust path)")
    print(f"  Sort 2D:    [{ROWS_2D},{COLS_2D}]  (CUB path)")
    print(f"  N_BENCH:    {N_BENCH}")
    print()
    print(f"Phase A  GEMM → argsort 1D (thrust): {stats(times_a)} ms")
    print(f"Phase B  GEMM → argsort 2D (CUB):    {stats(times_b)} ms")
    print(f"Phase C  GEMM → relu (control):       {stats(times_c)} ms")
    print()

    avg_a = sum(times_a) / len(times_a)
    avg_b = sum(times_b) / len(times_b)
    avg_c = sum(times_c) / len(times_c)

    if avg_a > avg_b * 3 and avg_a > avg_c * 3:
        print("CONCLUSION: argsort 1D is dramatically slower than 2D and relu,")
        print("  confirming implicit cudaStreamSynchronize in the 1D thrust path.")
        print(f"  Sync overhead ≈ {avg_a - avg_c:.3f} ms (Phase A - Phase C)")
    elif avg_a > avg_b * 1.5:
        print("CONCLUSION: argsort 1D is notably slower than 2D,")
        print("  suggesting partial sync in the thrust path.")
    else:
        print("INCONCLUSIVE: argsort 1D and 2D have similar wall times.")
        print("  The thrust allocator may be hitting the caching allocator after all,")
        print("  or the GEMM finishes before argsort launches.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys", action="store_true",
                        help="Enable NVTX markers for nsys profiling")
    args = parser.parse_args()
    run_phases(use_nvtx=args.nsys)
