# Race-fix-paddle migration report

Branch: `race-fix-paddle` (off `fork/paddle@108322c`)
Commit: `9f5f133` (`fix(deepep): correct cumsum_workspace size for session-71 kernel layout`)
PR: open against `fork/paddle` from `myrepo/race-fix-paddle`

## TL;DR

The reported "race" on `fork/paddle` was **not** a race. It was a silent
**workspace under-allocation** introduced when session-71 rewrote
`deepep_topk_metadata_cuda/kernel.cu` to a 4-kernel design (histogram →
block-offset-scan → prefix-sums → scatter-and-fixup) but did not bump the
caller's workspace size. The C++ launcher partitions the workspace into 4
regions, of which the new `block_naept_sum[B]` and `block_naept_base[B]` regions
overflow the old `2*B*E + 1` allocation by `2*B` int32 slots. Those writes
land on whatever torch's caching allocator placed next, corrupting downstream
consumers in non-deterministic ways. The autotune loop in
`token_gather_sum_kernel` was the most reliable victim because it issues many
back-to-back launches over varying configs that re-read the same indices.

## Fix

Single line in `sonicmoe/ernie_compat/deepep_metadata.py`:

```python
# was:
cumsum_workspace = torch.empty([2 * num_blocks * E + 1], dtype=torch.int32)
# now:
cumsum_workspace = torch.empty([2 * num_blocks * (E + 1)], dtype=torch.int32)
```

This matches the launcher's layout in `kernel.cu`:

```c++
int* block_hist        = workspace;
int* block_offset      = workspace + scatter_blocks * E;
int* block_naept_sum   = workspace + 2 * scatter_blocks * E;
int* block_naept_base  = workspace + 2 * scatter_blocks * E + scatter_blocks;
// total: 2*scatter_blocks*(E+1) int32 slots.
```

Also dropped a stale duplicate `_copy_tpe_h2d_async` left behind by an earlier
PR that referenced `_pin_memory_queue` (NameError — the surviving definition
uses `pin_memory_queue`).

**No `torch.cuda.synchronize()` calls were added. No new streams. No new
events.** The fix preserves full async execution on Paddle's
`current_stream(device)`.

## Why it presented as a race

- The bug manifested only after enough autotune launches to land an alloc on
  the OOB region. With small batch sizes / fewer autotune configs it would
  silently corrupt outputs without crashing.
- `compute-sanitizer` perturbed the allocator layout (its own bookkeeping
  allocations) so the OOB hit a benign region — masking the bug.
- A standalone probe (`/tmp/probe_naept.py`) confirmed the metadata kernel's
  outputs are bit-correct in isolation; only neighbouring allocations were
  being clobbered. This redirected the search from "stream ordering" to
  "buffer sizing".

## Validation

| Shape (T, H, I, E, K) | Imbalance | Result | µs/iter (CUDA events) |
|---|---|---|---|
| 8192, 3072, 1536, 8, 8 | none    | OK | 4360.7 |
| 8192, 3072, 1536, 8, 8 | skew    | OK | 4348.4 |
| 8192, 3072, 1536, 8, 8 | extreme | OK | 4380.8 |
| 16384, 3072, 1536, 8, 8 | none   | OK | 6759.3 |

All runs are clean: no IMA, no autotune compilation failures, no warnings.
Bench target script: `tests/ops/bench_mlpnode_topk_nsys.py`.

## PaddleFleet migration notes

For folks pulling this fix into PaddleFleet's bundled `sonicmoe`:

1. **Single source file** to update:
   `sonicmoe/ernie_compat/deepep_metadata.py`. The kernel.cu and bindings
   are unchanged.

2. **Watch for any other callers** of the deepep_topk_metadata_cuda C++
   entry. If PaddleFleet has its own caller, apply the same allocation rule:
   `2 * num_blocks * (E + 1)` int32, where `num_blocks = (N_recv + 31) // 32`
   and `E` is the **local** expert count (post-EP-shard).

3. **Stream-ordering invariant** (unchanged but worth restating):
   `_deepep_topk_to_sonic_metadata_cuda` issues all kernels on Paddle's
   `torch.cuda.current_stream(device)`. Output tensors are safe to consume by
   any subsequent op enqueued on the same stream — no manual events needed.

4. **Output tensor lifetime**: outputs are freshly-allocated per call (no
   global cache). PP / 1F1B schedules can interleave forward of layer L1
   with forward of layer L2 without metadata aliasing. Do not introduce a
   global cache for these — see the explanatory comment in the file.

5. **No new env vars or build flags** are needed for this fix.

## Out of scope (next session)

- Closing the perf gap to PyTorch native FP8 baseline (4406 vs 2715 µs).
  The session-71 rewrite is structurally cleaner but did not move the needle
  at this shape; needs a separate NCU sweep.
- The remaining session-72 work tracked in plan: quant-kernel NCU sweep,
  globals-purge in `mlp_node_v2`, PP-interleaved correctness regression.
